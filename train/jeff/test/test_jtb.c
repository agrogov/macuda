// test/test_jtb.c — M1.4's proof that the .jtb reader and py/make_fixture.py agree: open a fixture, check that the
// rows' lens and n_opts add up to the ids and targets sections' sizes, walk the batches and the strings, and check that
// a corrupted file stops with a clear message instead of a wrong gradient.
//
// It needs no GPU, so `make test-cpu` runs it on the fixture that Makefile builds (build/cpu/fx.jtb). With no
// argument it prints that it skipped and exits 0: `make test-null` runs every test binary that way (README.md).
//
//   build/cpu/test_jtb build/cpu/fx.jtb      the checks below
//   build/cpu/test_jtb --expect-error f.jtb   used by the corruption check: opens f.jtb and must fail
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "jtb.h"

static int failures;

#define CHECK(cond, ...)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      printf("FAIL test_jtb: " __VA_ARGS__);                          \
      printf("\n");                                                   \
      failures++;                                                     \
    }                                                                 \
  } while (0)

// ---- the fixture's own consistency ---------------------------------------------------------------------------------
static void check_header(const Jtb *j) {
  CHECK(jtb_version(j) == JTB_VERSION, "version %u, want %u", jtb_version(j), JTB_VERSION);
  CHECK(jtb_n_rows(j) > 0, "n_rows is 0");
  CHECK(jtb_n_batches(j) > 0, "n_batches is 0");
  CHECK(jtb_max_len(j) > 0, "max_len is 0");
  CHECK(jtb_max_opts(j) >= 1 && jtb_max_opts(j) <= JTB_MAX_OPTS, "max_opts %u", jtb_max_opts(j));
  CHECK(jtb_max_tokens_per_batch(j) >= jtb_max_len(j),
        "max_tokens_per_batch %u is below max_len %u (a batch holds at least one row)",
        jtb_max_tokens_per_batch(j), jtb_max_len(j));
  CHECK(jtb_section_offset(j, JTB_SEC_ROWS) >= 28u + 5u * 8u, "the rows section overlaps the header");
  for (int s = 0; s < JTB_NSEC; s++)
    CHECK(jtb_section_offset(j, s) + jtb_section_size(j, s) <= jtb_file_size(j),
          "section %d runs past the %llu-byte file", s, (unsigned long long)jtb_file_size(j));
}

// The plan's check: the rows' lens and n_opts must add up to the ids and targets sections exactly, or every row after
// the first mistake reads another row's tokens.
static void check_sums(const Jtb *j) {
  uint64_t len_sum = 0, opt_sum = 0, max_len = 0, max_opts = 0;
  for (uint32_t i = 0; i < jtb_n_rows(j); i++) {
    const JtbRow *r = jtb_row(j, i);
    CHECK(r != NULL, "row %u is missing", i);
    if (r == NULL) return;
    CHECK(r->len > 0, "row %u: len 0", i);
    CHECK(r->n_opts >= 1 && r->n_opts <= JTB_MAX_OPTS, "row %u: n_opts %u", i, r->n_opts);
    CHECK(r->teacher_top < r->n_opts, "row %u: teacher_top %u >= n_opts %u", i, r->teacher_top, r->n_opts);
    len_sum += r->len;
    opt_sum += r->n_opts;
    if (r->len > max_len) max_len = r->len;
    if (r->n_opts > max_opts) max_opts = r->n_opts;
  }
  CHECK(len_sum * 4 == jtb_section_size(j, JTB_SEC_IDS),
        "the rows' lens sum to %llu ids, the ids section holds %llu",
        (unsigned long long)len_sum, (unsigned long long)jtb_section_size(j, JTB_SEC_IDS) / 4);
  CHECK(opt_sum * 4 == jtb_section_size(j, JTB_SEC_TARGETS),
        "the rows' n_opts sum to %llu targets, the targets section holds %llu",
        (unsigned long long)opt_sum, (unsigned long long)jtb_section_size(j, JTB_SEC_TARGETS) / 4);
  CHECK((uint64_t)jtb_n_rows(j) * 24 == jtb_section_size(j, JTB_SEC_ROWS),
        "the rows section is %llu bytes, %u rows need %llu", (unsigned long long)jtb_section_size(j, JTB_SEC_ROWS),
        jtb_n_rows(j), (unsigned long long)jtb_n_rows(j) * 24);
  CHECK(max_len == jtb_max_len(j), "max_len %u, the longest row is %llu", jtb_max_len(j),
        (unsigned long long)max_len);
  CHECK(max_opts == jtb_max_opts(j), "max_opts %u, the widest row is %llu", jtb_max_opts(j),
        (unsigned long long)max_opts);

  for (uint32_t i = 0; i < jtb_n_rows(j); i++) {
    const JtbRow *r = jtb_row(j, i);
    const uint32_t *ids = jtb_ids(j, i);
    const float *targets = jtb_targets(j, i);
    for (uint32_t t = 0; t < r->len; t++)
      CHECK(ids[t] < JTB_VOCAB_MAX, "row %u: token %u has id %u, not below %u", i, t, ids[t], JTB_VOCAB_MAX);
    double sum = 0.0;
    for (uint32_t o = 0; o < r->n_opts; o++) {
      CHECK(targets[o] >= 0.0f && targets[o] <= 1.0f, "row %u: target %u is %g", i, o, (double)targets[o]);
      sum += (double)targets[o];
    }
    CHECK(fabs(sum - 1.0) < 1e-3, "row %u: targets sum to %.6f, want 1", i, sum);
    for (uint32_t o = 0; o < r->n_opts; o++)
      CHECK(targets[r->teacher_top] >= targets[o],
            "row %u: teacher_top %u is not an argmax (target %u is larger)", i, r->teacher_top, o);
  }
}

// The strings section: "fx-0000", then one key per option, then the empty string that ends the list (§9.2).
static void check_strings(const Jtb *j) {
  for (uint32_t i = 0; i < jtb_n_rows(j); i++) {
    const JtbRow *r = jtb_row(j, i);
    const char *id = jtb_id(j, i);
    char want[32];
    snprintf(want, sizeof want, "fx-%04u", i);
    CHECK(strlen(id) > 0, "row %u: empty id", i);
    CHECK(strcmp(id, want) == 0, "row %u: id \"%s\", want \"%s\"", i, id, want);

    const char *keys[256];
    int n = jtb_keys(j, i, keys, (int)(sizeof keys / sizeof keys[0]));
    CHECK(n == (int)r->n_opts, "row %u: %d criteria keys, n_opts is %u", i, n, r->n_opts);
    if (n < 0) continue;
    for (int k = 0; k < n; k++) {
      CHECK(keys[k][0] != '\0', "row %u: key %d is empty", i, k);
      CHECK(k == 0 || strcmp(keys[k - 1], keys[k]) != 0, "row %u: key %d repeats \"%s\"", i, k, keys[k]);
    }
    // One key only: the accessor must say it does not fit rather than write past out[].
    const char *one[1];
    int want_one = r->n_opts > 1 ? -1 : 1;
    CHECK(jtb_keys(j, i, one, 1) == want_one, "row %u: jtb_keys(max 1) on a %u-key row returned something else", i,
          r->n_opts);
  }
}

// The batches: consecutive chunks of the writer's --batch rows, each row at most once per batch, together covering
// every row at least once (prep.py's train file holds whole epochs, design §2.3).
static void check_batches(const Jtb *j) {
  unsigned char *seen = calloc(jtb_n_rows(j), 1);
  uint64_t covered = 0;
  CHECK(seen != NULL, "out of memory");
  if (seen == NULL) return;
  for (uint32_t k = 0; k < jtb_n_batches(j); k++) {
    uint32_t n = 0;
    const uint32_t *rows = jtb_batch(j, k, &n);
    CHECK(rows != NULL && n > 0, "batch %u: no rows", k);
    if (rows == NULL) break;
    memset(seen, 0, jtb_n_rows(j));
    uint64_t tokens = 0;
    for (uint32_t t = 0; t < n; t++) {
      CHECK(rows[t] < jtb_n_rows(j), "batch %u: row %u of %u", k, rows[t], jtb_n_rows(j));
      if (rows[t] >= jtb_n_rows(j)) continue;
      CHECK(!seen[rows[t]], "batch %u: row %u twice", k, rows[t]);
      seen[rows[t]] = 1;
      tokens += jtb_row(j, rows[t])->len;
    }
    CHECK(tokens <= jtb_max_tokens_per_batch(j), "batch %u: %llu tokens, header allows %u", k,
          (unsigned long long)tokens, jtb_max_tokens_per_batch(j));
    covered += n;
  }
  CHECK(covered >= jtb_n_rows(j), "the batches hold %llu rows, fewer than n_rows %u",
        (unsigned long long)covered, jtb_n_rows(j));
  uint32_t n = 12345;
  CHECK(jtb_batch(j, jtb_n_batches(j), &n) == NULL && n == 0, "a batch past the last one was accepted");
  CHECK(jtb_row(j, jtb_n_rows(j)) == NULL, "a row past the last one was accepted");
  free(seen);
}

// ---- the corruption check ------------------------------------------------------------------------------------------
// Copy the fixture, flip one byte, and run this binary again in --expect-error mode: jtb_open() must print a message
// and exit 1. A run that opens the corrupt file exits 2, so a reader that lost its validation cannot pass by luck.
static int copy_and_patch(const char *src, const char *dst, long off, unsigned char byte) {
  FILE *in = fopen(src, "rb");
  if (in == NULL) return -1;
  if (fseek(in, 0, SEEK_END) != 0) { fclose(in); return -1; }
  long size = ftell(in);
  if (size < 0 || off >= size) { fclose(in); return -1; }
  unsigned char *buf = malloc((size_t)size);
  if (buf == NULL || fseek(in, 0, SEEK_SET) != 0 || fread(buf, 1, (size_t)size, in) != (size_t)size) {
    free(buf);
    fclose(in);
    return -1;
  }
  fclose(in);
  buf[off] = byte;
  FILE *out = fopen(dst, "wb");
  int ok = out != NULL && fwrite(buf, 1, (size_t)size, out) == (size_t)size;
  if (out != NULL) fclose(out);
  free(buf);
  return ok ? 0 : -1;
}

// Runs `<self> --expect-error <path>` with its stderr captured; returns the exit code, and copies the message into
// msg (always NUL-terminated).
static int run_expecting_error(const char *self, const char *path, char *msg, size_t msg_size) {
  int fds[2];
  msg[0] = '\0';
  if (pipe(fds) != 0) return -1;
  pid_t pid = fork();
  if (pid < 0) {
    close(fds[0]);
    close(fds[1]);
    return -1;
  }
  if (pid == 0) {  // child: stderr into the pipe, then the same binary in --expect-error mode
    close(fds[0]);
    if (dup2(fds[1], 2) < 0) _exit(127);
    close(fds[1]);
    execl(self, self, "--expect-error", path, (char *)NULL);
    _exit(127);
  }
  close(fds[1]);
  ssize_t got = read(fds[0], msg, msg_size - 1);
  close(fds[0]);
  msg[got > 0 ? (size_t)got : 0] = '\0';
  int status = 0;
  if (waitpid(pid, &status, 0) != pid) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static void check_corrupt(const char *self, const char *fixture, const Jtb *j) {
  const char *dir = getenv("TMPDIR");
  if (dir == NULL || dir[0] == '\0') dir = "/tmp";
  char path[512];
  size_t dir_len = strlen(dir);
  snprintf(path, sizeof path, "%s%sjtb-corrupt-XXXXXX", dir, dir[dir_len - 1] == '/' ? "" : "/");
  int fd = mkstemp(path);
  if (fd < 0) {
    printf("FAIL test_jtb: mkstemp(%s): %s\n", path, strerror(errno));
    failures++;
    return;
  }
  close(fd);

  struct { const char *what; long off; unsigned char byte; const char *want; } cases[] = {
      // one byte of the magic: not a .jtb file at all
      {"magic", 0, 'X', "magic"},
      // the top byte of the first token id (4 bytes into the ids section): the low bytes stay below the vocab, this
      // one makes the id >= 2^24, so the reader must reject it
      {"ids", (long)(jtb_section_offset(j, JTB_SEC_IDS) + 3), 0xff, "token"},
      // the version field, second u32 of the header
      {"version", 4, 9, "version"},
  };
  for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
    char msg[1024];
    if (copy_and_patch(fixture, path, cases[c].off, cases[c].byte) != 0) {
      printf("FAIL test_jtb: cannot write the %s-corrupted copy %s\n", cases[c].what, path);
      failures++;
      continue;
    }
    int code = run_expecting_error(self, path, msg, sizeof msg);
    CHECK(code == 1, "%s-corrupted copy: exit %d, want 1 (message: \"%s\")", cases[c].what, code, msg);
    CHECK(strstr(msg, cases[c].want) != NULL, "%s-corrupted copy: message \"%s\" does not mention \"%s\"",
          cases[c].what, msg, cases[c].want);
    size_t mlen = strlen(msg);
    while (mlen > 0 && (msg[mlen - 1] == '\n' || msg[mlen - 1] == '\r')) msg[--mlen] = '\0';
    printf("corrupt %-8s off=%-6ld -> exit %d: %s\n", cases[c].what, cases[c].off, code, msg);
  }
  unlink(path);
}

// ---- main ----------------------------------------------------------------------------------------------------------
int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "--expect-error") == 0) {
    if (argc < 3) {
      fprintf(stderr, "usage: %s [--expect-error] <file.jtb>\n", argv[0]);
      return 2;
    }
    Jtb *j = jtb_open(argv[2]);  // exits 1, with a message, on a bad file
    jtb_close(j);
    fprintf(stderr, "test_jtb: --expect-error: %s opened with no error\n", argv[2]);
    return 2;
  }
  if (argc < 2) {
    printf("jtb skipped (no fixture given; usage: %s build/cpu/fx.jtb)\n", argv[0]);
    return 0;
  }

  Jtb *j = jtb_open(argv[1]);
  uint32_t n_rows = jtb_n_rows(j);
  check_header(j);
  check_sums(j);
  check_strings(j);
  check_batches(j);
  printf("fixture %s: %u rows, %u batches, max_len %u, max_opts %u, max_tokens_per_batch %u, %llu bytes\n",
         argv[1], n_rows, jtb_n_batches(j), jtb_max_len(j), jtb_max_opts(j),
         jtb_max_tokens_per_batch(j), (unsigned long long)jtb_file_size(j));
  check_corrupt(argv[0], argv[1], j);
  jtb_close(j);

  if (failures != 0) {
    printf("FAIL test_jtb: %d failure(s)\n", failures);
    return 1;
  }
  printf("jtb OK fixture=%s rows=%u\n", argv[1], n_rows);
  return 0;
}
