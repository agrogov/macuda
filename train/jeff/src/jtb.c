// src/jtb.c — mmap, validate and expose a .jtb file (design §9.1). The layout is written down in src/jtb.h.
//
// The reader is deliberately suspicious: a wrong offset in a batch file turns into a wrong gradient thousands of
// steps later, so every section is checked against the file size, every token id against the tokenizer's vocab and
// every key list against its row's n_opts before any of the file is handed to the trainer.
#include "jtb.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__)
#error "src/jtb.c reads JTB1 fields in place; it assumes a little-endian host"
#endif

#define JTB_HDR 28u                          // magic + 6 x u32
#define JTB_TABLE (JTB_NSEC * 8u)            // the section-offset table
#define JTB_ROW_BYTES 24u                    // len, n_opts, weight, teacher_top, id_off
#define JTB_TARGET_SUM_TOL 1e-3              // encode() in CLM-8B's train.py rejects a target further than 1e-4 off
#define JTB_PROB_SLACK 1e-6f                 // f32 rounding of the last option, not a tolerance for a bad target

struct Jtb {
  unsigned char *map;
  uint64_t size;
  uint64_t off[JTB_NSEC];                // section start in the file
  uint64_t len[JTB_NSEC];                // section length in bytes
  uint32_t version, n_rows, n_batches, max_tokens_per_batch, max_len, max_opts;
  JtbRow *rows;                          // parsed copy, host endian, n_rows entries
  const uint32_t *ids;                   // into the map
  const float *targets;                  // into the map
  const unsigned char *batches;          // into the map (walked by jtb_batch)
};

// ---- little-endian reads -------------------------------------------------------------------------------------------
// memcpy, not a cast: the row records start at file offset 68, which is not 8-byte aligned.
static uint32_t rd32(const unsigned char *p) {
  uint32_t v;
  memcpy(&v, p, sizeof v);
  return v;
}

static uint64_t rd64(const unsigned char *p) {
  uint64_t v;
  memcpy(&v, p, sizeof v);
  return v;
}

static float rd32f(const unsigned char *p) {
  float v;
  memcpy(&v, p, sizeof v);
  return v;
}

// ---- errors --------------------------------------------------------------------------------------------------------
// jtb_open() prints the message and exits(1); nothing here returns a half-validated file.
static int jtb_err(char *err, size_t errsz, const char *path, const char *fmt, ...) {
  int n = snprintf(err, errsz, "jtb: %s: ", path);
  if (n < 0 || (size_t)n >= errsz) return 0;
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(err + n, errsz - (size_t)n, fmt, ap);
  va_end(ap);
  return 0;
}

// In the section readers a failure ends the open: jtb_parse() munmaps and frees whatever was built.
#define JTB_FAIL(...) do { jtb_err(err, errsz, path, __VA_ARGS__); return 0; } while (0)

// ---- header and section layout -------------------------------------------------------------------------------------
static int jtb_read_header(Jtb *j, const char *path, char *err, size_t errsz) {
  const unsigned char *h = j->map;
  if (j->size < JTB_HDR + JTB_TABLE) {
    jtb_err(err, errsz, path, "file is %llu bytes, too small for the %u-byte header",
            (unsigned long long)j->size, JTB_HDR + JTB_TABLE);
    return 0;
  }
  if (memcmp(h, JTB_MAGIC, 4) != 0) {
    jtb_err(err, errsz, path, "bad magic %02x%02x%02x%02x, want \"%s\" (not a .jtb file?)",
            h[0], h[1], h[2], h[3], JTB_MAGIC);
    return 0;
  }
  j->version = rd32(h + 4);
  j->n_rows = rd32(h + 8);
  j->n_batches = rd32(h + 12);
  j->max_tokens_per_batch = rd32(h + 16);
  j->max_len = rd32(h + 20);
  j->max_opts = rd32(h + 24);
  if (j->version != JTB_VERSION)
    JTB_FAIL("version %u, this reader is version %u", j->version, JTB_VERSION);
  if (j->n_rows == 0) JTB_FAIL("n_rows is 0");
  if (j->n_batches == 0) JTB_FAIL("n_batches is 0");
  if (j->max_len == 0) JTB_FAIL("max_len is 0");
  if (j->max_opts == 0 || j->max_opts > JTB_MAX_OPTS)
    JTB_FAIL("max_opts %u, want 1..%u", j->max_opts, JTB_MAX_OPTS);
  for (int s = 0; s < JTB_NSEC; s++) j->off[s] = rd64(h + JTB_HDR + 8 * (size_t)s);
  if (j->off[JTB_SEC_ROWS] < JTB_HDR + JTB_TABLE)
    JTB_FAIL("the rows section starts at %llu, inside the %u-byte header+table",
             (unsigned long long)j->off[JTB_SEC_ROWS], JTB_HDR + JTB_TABLE);
  for (int s = 0; s < JTB_NSEC; s++) {
    if (j->off[s] > j->size)
      JTB_FAIL("section %d starts at %llu, past the end of the %llu-byte file",
               s, (unsigned long long)j->off[s], (unsigned long long)j->size);
    if (s > 0 && j->off[s] < j->off[s - 1])
      JTB_FAIL("section %d starts at %llu, before section %d at %llu",
               s, (unsigned long long)j->off[s], s - 1, (unsigned long long)j->off[s - 1]);
  }
  // No padding between sections (design §9.1), so a section's length is the next one's offset minus its own.
  for (int s = 0; s < JTB_NSEC; s++)
    j->len[s] = (s + 1 < JTB_NSEC ? j->off[s + 1] : j->size) - j->off[s];

  // The ids and targets are read through u32/float pointers into the map, so their sections must be 4-byte aligned
  // (the row records themselves are read with memcpy: 68 + 24i is not 8-byte aligned).
  for (int s = JTB_SEC_ROWS; s <= JTB_SEC_TARGETS; s++)
    if (j->off[s] % 4 != 0)
      JTB_FAIL("section %d starts at %llu, not 4-byte aligned", s, (unsigned long long)j->off[s]);

  if (j->len[JTB_SEC_ROWS] != (uint64_t)j->n_rows * JTB_ROW_BYTES)
    JTB_FAIL("rows section is %llu bytes, %u rows need %llu",
             (unsigned long long)j->len[JTB_SEC_ROWS], j->n_rows,
             (unsigned long long)j->n_rows * JTB_ROW_BYTES);
  if (j->len[JTB_SEC_IDS] % 4 != 0)
    JTB_FAIL("ids section is %llu bytes, not a multiple of 4", (unsigned long long)j->len[JTB_SEC_IDS]);
  if (j->len[JTB_SEC_TARGETS] % 4 != 0)
    JTB_FAIL("targets section is %llu bytes, not a multiple of 4", (unsigned long long)j->len[JTB_SEC_TARGETS]);
  if (j->len[JTB_SEC_STRINGS] < 2)
    JTB_FAIL("strings section is %llu bytes", (unsigned long long)j->len[JTB_SEC_STRINGS]);

  j->ids = (const uint32_t *)(const void *)(j->map + j->off[JTB_SEC_IDS]);
  j->targets = (const float *)(const void *)(j->map + j->off[JTB_SEC_TARGETS]);
  j->batches = j->map + j->off[JTB_SEC_BATCHES];
  return 1;
}

// ---- rows: ids and targets -----------------------------------------------------------------------------------------
static int jtb_read_rows(Jtb *j, const char *path, char *err, size_t errsz) {
  const unsigned char *p = j->map + j->off[JTB_SEC_ROWS];
  uint64_t id_at = 0, tgt_at = 0;
  j->rows = calloc(j->n_rows, sizeof *j->rows);
  if (j->rows == NULL) JTB_FAIL("out of memory for %u rows", j->n_rows);
  for (uint32_t i = 0; i < j->n_rows; i++, p += JTB_ROW_BYTES) {
    JtbRow *r = &j->rows[i];
    r->len = rd32(p);
    r->n_opts = rd32(p + 4);
    r->weight = rd32f(p + 8);
    r->teacher_top = rd32(p + 12);
    r->id_off = rd64(p + 16);
    r->ids_off = id_at;
    r->tgt_off = tgt_at;
    if (r->len == 0) JTB_FAIL("row %u: len is 0", i);
    if (r->len > j->max_len) JTB_FAIL("row %u: len %u is above the header's max_len %u", i, r->len, j->max_len);
    if (r->n_opts == 0) JTB_FAIL("row %u: n_opts is 0", i);
    if (r->n_opts > JTB_MAX_OPTS)
      JTB_FAIL("row %u: n_opts %u is above %u", i, r->n_opts, JTB_MAX_OPTS);
    if (r->n_opts > j->max_opts)
      JTB_FAIL("row %u: n_opts %u is above the header's max_opts %u", i, r->n_opts, j->max_opts);
    if (r->teacher_top >= r->n_opts)
      JTB_FAIL("row %u: teacher_top %u is not < n_opts %u", i, r->teacher_top, r->n_opts);
    if (!(r->weight >= 0.0f)) JTB_FAIL("row %u: weight %g is not a non-negative number", i, (double)r->weight);
    // A sum of len ids and of n_opts targets has to fit exactly in its section, or every later row is shifted.
    id_at += r->len;
    tgt_at += r->n_opts;
    if (id_at * 4 > j->len[JTB_SEC_IDS])
      JTB_FAIL("row %u: the rows' lens need %llu id bytes, the ids section has %llu",
               i, (unsigned long long)id_at * 4, (unsigned long long)j->len[JTB_SEC_IDS]);
    if (tgt_at * 4 > j->len[JTB_SEC_TARGETS])
      JTB_FAIL("row %u: the rows' n_opts need %llu target bytes, the targets section has %llu",
               i, (unsigned long long)tgt_at * 4, (unsigned long long)j->len[JTB_SEC_TARGETS]);
  }
  if (id_at * 4 != j->len[JTB_SEC_IDS])
    JTB_FAIL("the rows' lens sum to %llu ids, the ids section holds %llu",
             (unsigned long long)id_at, (unsigned long long)j->len[JTB_SEC_IDS] / 4);
  if (tgt_at * 4 != j->len[JTB_SEC_TARGETS])
    JTB_FAIL("the rows' n_opts sum to %llu targets, the targets section holds %llu",
             (unsigned long long)tgt_at, (unsigned long long)j->len[JTB_SEC_TARGETS] / 4);

  for (uint32_t i = 0; i < j->n_rows; i++) {
    const JtbRow *r = &j->rows[i];
    for (uint32_t t = 0; t < r->len; t++) {
      uint32_t id = j->ids[r->ids_off + t];
      if (id >= JTB_VOCAB_MAX)
        JTB_FAIL("row %u: the token at position %u has id %u, not below %u", i, t, id, JTB_VOCAB_MAX);
    }
    double sum = 0.0;
    for (uint32_t o = 0; o < r->n_opts; o++) {
      float v = j->targets[r->tgt_off + o];
      // A probability, with the slack that one float rounding of 1 - sum(others) needs (not a loosened gate).
      if (!(v >= -JTB_PROB_SLACK && v <= 1.0f + JTB_PROB_SLACK))
        JTB_FAIL("row %u: target %u is %g, not a probability in [0,1]", i, o, (double)v);
      sum += (double)v;
    }
    if (!(sum > 1.0 - JTB_TARGET_SUM_TOL && sum < 1.0 + JTB_TARGET_SUM_TOL))
      JTB_FAIL("row %u: targets sum to %.6f, want 1", i, sum);
  }
  return 1;
}

// ---- strings: the row id and its criteria keys ---------------------------------------------------------------------
static int jtb_read_strings(Jtb *j, const char *path, char *err, size_t errsz) {
  const char *base = (const char *)(const void *)(j->map + j->off[JTB_SEC_STRINGS]);
  const char *end = base + j->len[JTB_SEC_STRINGS];
  for (uint32_t i = 0; i < j->n_rows; i++) {
    const JtbRow *r = &j->rows[i];
    if (r->id_off >= j->len[JTB_SEC_STRINGS])
      JTB_FAIL("row %u: id_off %llu is outside the %llu-byte strings section",
               i, (unsigned long long)r->id_off, (unsigned long long)j->len[JTB_SEC_STRINGS]);
    const char *p = base + r->id_off;
    const char *z = memchr(p, '\0', (size_t)(end - p));
    if (z == NULL) JTB_FAIL("row %u: the id at strings+%llu is not NUL-terminated", i,
                            (unsigned long long)r->id_off);
    if (z == p) JTB_FAIL("row %u: the id at strings+%llu is empty", i, (unsigned long long)r->id_off);
    uint32_t keys = 0;
    p = z + 1;
    for (;;) {
      if (p >= end) JTB_FAIL("row %u: the key list runs past the end of the strings section", i);
      if (*p == '\0') break;  // the empty string that terminates the list (design §9.2)
      z = memchr(p, '\0', (size_t)(end - p));
      if (z == NULL) JTB_FAIL("row %u: criterion key %u is not NUL-terminated", i, keys);
      keys++;
      p = z + 1;
    }
    if (keys != r->n_opts)
      JTB_FAIL("row %u: %u criteria keys, but n_opts is %u (predictions are keyed by them, design §9.2)",
               i, keys, r->n_opts);
  }
  return 1;
}

// ---- batches: which rows make a step -------------------------------------------------------------------------------
static int jtb_read_batches(Jtb *j, const char *path, char *err, size_t errsz) {
  const unsigned char *p = j->batches;
  const unsigned char *end = p + j->len[JTB_SEC_BATCHES];
  unsigned char *seen = calloc(j->n_rows, 1);
  uint64_t covered = 0;
  if (seen == NULL) {
    jtb_err(err, errsz, path, "out of memory for a %u-row bitmap", j->n_rows);
    return 0;
  }
  for (uint32_t k = 0; k < j->n_batches; k++) {
    if ((size_t)(end - p) < 4) { jtb_err(err, errsz, path, "batch %u: no row count", k); goto fail; }
    uint32_t n = rd32(p);
    p += 4;
    if (n == 0) { jtb_err(err, errsz, path, "batch %u: 0 rows", k); goto fail; }
    if (n > j->n_rows) {
      jtb_err(err, errsz, path, "batch %u: %u rows, the file has %u", k, n, j->n_rows);
      goto fail;
    }
    if ((uint64_t)(end - p) < (uint64_t)n * 4) {
      jtb_err(err, errsz, path, "batch %u: %u row indices do not fit in the batches section", k, n);
      goto fail;
    }
    memset(seen, 0, j->n_rows);
    uint64_t tokens = 0;
    for (uint32_t t = 0; t < n; t++) {
      uint32_t row = rd32(p + 4u * (size_t)t);
      if (row >= j->n_rows) {
        jtb_err(err, errsz, path, "batch %u: row index %u, the file has %u rows", k, row, j->n_rows);
        goto fail;
      }
      if (seen[row]) {
        jtb_err(err, errsz, path, "batch %u: row %u appears twice", k, row);
        goto fail;
      }
      seen[row] = 1;
      tokens += j->rows[row].len;
    }
    if (tokens > j->max_tokens_per_batch) {
      jtb_err(err, errsz, path, "batch %u: %llu tokens, above the header's max_tokens_per_batch %u",
              k, (unsigned long long)tokens, j->max_tokens_per_batch);
      goto fail;
    }
    p += (uint64_t)n * 4;
    covered += n;
  }
  if (p != end) {
    jtb_err(err, errsz, path, "the batches section has %llu trailing byte(s)",
            (unsigned long long)(end - p));
    goto fail;
  }
  // One epoch at least: the trainer's step count is n_batches, so a file whose batches miss rows would silently
  // train on a subset.
  if (covered < j->n_rows) {
    jtb_err(err, errsz, path, "the batches hold %llu rows in total, fewer than n_rows %u",
            (unsigned long long)covered, j->n_rows);
    goto fail;
  }
  free(seen);
  return 1;
fail:
  free(seen);
  return 0;
}

// ---- open / close --------------------------------------------------------------------------------------------------
static int jtb_parse(Jtb *j, const char *path, char *err, size_t errsz) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    jtb_err(err, errsz, path, "cannot open: %s", strerror(errno));
    return 0;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    jtb_err(err, errsz, path, "stat failed: %s", strerror(errno));
    close(fd);
    return 0;
  }
  if (!S_ISREG(st.st_mode)) {
    jtb_err(err, errsz, path, "not a regular file");
    close(fd);
    return 0;
  }
  if ((uint64_t)st.st_size < JTB_HDR + JTB_TABLE) {
    jtb_err(err, errsz, path, "file is %llu bytes, too small for the %u-byte header",
            (unsigned long long)st.st_size, JTB_HDR + JTB_TABLE);
    close(fd);
    return 0;
  }
  void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (map == MAP_FAILED) {
    jtb_err(err, errsz, path, "mmap failed: %s", strerror(errno));
    return 0;
  }
  j->map = map;
  j->size = (uint64_t)st.st_size;
  if (!jtb_read_header(j, path, err, errsz)) goto cleanup;
  if (!jtb_read_rows(j, path, err, errsz)) goto cleanup;
  if (!jtb_read_strings(j, path, err, errsz)) goto cleanup;
  if (!jtb_read_batches(j, path, err, errsz)) goto cleanup;
  return 1;
cleanup:
  munmap(j->map, (size_t)j->size);
  j->map = NULL;
  free(j->rows);
  j->rows = NULL;
  return 0;
}

Jtb *jtb_open(const char *path) {
  char err[512];
  Jtb *j = calloc(1, sizeof *j);
  if (j == NULL) {
    fprintf(stderr, "jtb: out of memory\n");
    exit(1);
  }
  if (path == NULL || path[0] == '\0') {
    fprintf(stderr, "jtb: no path given\n");
    exit(1);
  }
  if (!jtb_parse(j, path, err, sizeof err)) {
    fprintf(stderr, "%s\n", err);
    free(j);
    exit(1);
  }
  return j;
}

void jtb_close(Jtb *j) {
  if (j == NULL) return;
  if (j->map != NULL) munmap(j->map, (size_t)j->size);
  free(j->rows);
  free(j);
}

// ---- accessors -----------------------------------------------------------------------------------------------------
uint32_t jtb_version(const Jtb *j) { return j->version; }
uint32_t jtb_n_rows(const Jtb *j) { return j->n_rows; }
uint32_t jtb_n_batches(const Jtb *j) { return j->n_batches; }
uint32_t jtb_max_tokens_per_batch(const Jtb *j) { return j->max_tokens_per_batch; }
uint32_t jtb_max_len(const Jtb *j) { return j->max_len; }
uint32_t jtb_max_opts(const Jtb *j) { return j->max_opts; }
uint64_t jtb_file_size(const Jtb *j) { return j->size; }

uint64_t jtb_section_offset(const Jtb *j, int sec) { return j->off[sec]; }
uint64_t jtb_section_size(const Jtb *j, int sec) { return j->len[sec]; }

const JtbRow *jtb_row(const Jtb *j, uint32_t i) {
  if (i >= j->n_rows) return NULL;
  return &j->rows[i];
}

const uint32_t *jtb_ids(const Jtb *j, uint32_t i) { return j->ids + j->rows[i].ids_off; }

const float *jtb_targets(const Jtb *j, uint32_t i) { return j->targets + j->rows[i].tgt_off; }

const char *jtb_id(const Jtb *j, uint32_t i) {
  return (const char *)(const void *)(j->map + j->off[JTB_SEC_STRINGS]) + j->rows[i].id_off;
}

int jtb_keys(const Jtb *j, uint32_t i, const char **out, int max) {
  const char *p = jtb_id(j, i);
  p += strlen(p) + 1;
  int n = 0;
  while (*p != '\0') {
    if (n >= max) return -1;
    out[n++] = p;
    p += strlen(p) + 1;
  }
  return n;
}

const uint32_t *jtb_batch(const Jtb *j, uint32_t k, uint32_t *n) {
  if (n != NULL) *n = 0;
  if (k >= j->n_batches) return NULL;
  const unsigned char *p = j->batches;
  uint32_t count = 0;
  for (uint32_t b = 0; b <= k; b++) {
    count = rd32(p);
    p += 4;
    if (b == k) break;
    p += (uint64_t)count * 4;  // validated at open, so this walk cannot leave the section
  }
  if (n != NULL) *n = count;
  return (const uint32_t *)(const void *)p;
}
