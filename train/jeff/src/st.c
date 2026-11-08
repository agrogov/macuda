// src/st.c — safetensors: mmap'ed reader (a tiny hand-written JSON scanner for the flat header), an F32 writer
// (design §3.2 and §9.2, plan M1.3). See st.h for the contract; the two rules this file is built around are
//   * nothing is copied on read: every tensor is a pointer into the mapping;
//   * every expectation the caller states (name, dtype, ndim, each dim) is checked here, once.
// The header JSON of this format is flat — one object of "name" -> {dtype, shape, data_offsets} plus an optional
// "__metadata__" — so the scanner needs exactly three productions: string, integer array, and "skip whatever this is". It
// is not a JSON library and does not pretend to be: a header it does not understand is a file it refuses, not a file it
// half-reads.
#include "st.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// A safetensors header is a few hundred KB at most (the model's is ~80 KB: 473 tensors). Anything past this is not one,
// and refusing early keeps the u64 length read from the file from steering an allocation.
#define ST_MAX_HEADER ((size_t)1 << 28)

// ---- messages ----------------------------------------------------------------------------------------------------------

// Every failure prints "st: <path>: <what>" and exits 1. path may be NULL when the message already says enough.
static void die(const char *path, const char *fmt, ...) {
  va_list ap;
  fputs("st: ", stderr);
  if (path != NULL) fprintf(stderr, "%s: ", path);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

static void *xmalloc(size_t n) {
  void *p = malloc(n ? n : 1);
  if (p == NULL) die(NULL, "out of memory (%zu bytes)", n);
  return p;
}

static void *xrealloc(void *p, size_t n) {
  void *q = realloc(p, n ? n : 1);
  if (q == NULL) die(NULL, "out of memory (%zu bytes)", n);
  return q;
}

static char *xstrdup(const char *s) {
  size_t n = strlen(s) + 1;
  char *p = xmalloc(n);
  memcpy(p, s, n);
  return p;
}

// dir + "/" + name, with the separator only when dir does not already end in one.
static char *join_path(const char *dir, const char *name) {
  size_t dl = strlen(dir), nl = strlen(name);
  size_t sep = dl > 0 && dir[dl - 1] != '/' ? 1 : 0;
  char *p = xmalloc(dl + sep + nl + 1);
  memcpy(p, dir, dl);
  if (sep) p[dl] = '/';
  memcpy(p + dl + sep, name, nl + 1);
  return p;
}

// ---- dtypes ------------------------------------------------------------------------------------------------------------

static const struct {
  const char *name;
  StDtype dtype;
} ST_DTYPES[] = {
    {"BOOL", ST_DTYPE_BOOL}, {"U8", ST_DTYPE_U8},   {"I8", ST_DTYPE_I8},     {"I16", ST_DTYPE_I16},
    {"U16", ST_DTYPE_U16},   {"F16", ST_DTYPE_F16}, {"BF16", ST_DTYPE_BF16}, {"I32", ST_DTYPE_I32},
    {"U32", ST_DTYPE_U32},   {"F32", ST_DTYPE_F32}, {"I64", ST_DTYPE_I64},   {"U64", ST_DTYPE_U64},
    {"F64", ST_DTYPE_F64},
};

size_t st_dtype_size(StDtype dtype) {
  switch (dtype) {
  case ST_DTYPE_BOOL:
  case ST_DTYPE_U8:
  case ST_DTYPE_I8:
    return 1;
  case ST_DTYPE_I16:
  case ST_DTYPE_U16:
  case ST_DTYPE_F16:
  case ST_DTYPE_BF16:
    return 2;
  case ST_DTYPE_I32:
  case ST_DTYPE_U32:
  case ST_DTYPE_F32:
    return 4;
  case ST_DTYPE_I64:
  case ST_DTYPE_U64:
  case ST_DTYPE_F64:
    return 8;
  default:
    return 0;
  }
}

const char *st_dtype_name(StDtype dtype) {
  for (size_t i = 0; i < sizeof ST_DTYPES / sizeof ST_DTYPES[0]; i++)
    if (ST_DTYPES[i].dtype == dtype) return ST_DTYPES[i].name;
  return "?";
}

StDtype st_dtype_parse(const char *name) {
  for (size_t i = 0; i < sizeof ST_DTYPES / sizeof ST_DTYPES[0]; i++)
    if (strcmp(ST_DTYPES[i].name, name) == 0) return ST_DTYPES[i].dtype;
  return ST_DTYPE_NONE;
}

// ---- the JSON scanner --------------------------------------------------------------------------------------------------

typedef struct {
  const unsigned char *p, *end;
  const char *path;
} Cur;

static void skip_ws(Cur *c) {
  while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r')) c->p++;
}

// The next non-space byte, or -1 at the end of the header. (Does not consume it.)
static int peek(Cur *c) {
  skip_ws(c);
  return c->p < c->end ? (int)*c->p : -1;
}

static void describe(int ch, char *out, size_t n) {
  if (ch < 0) snprintf(out, n, "the end of the header");
  else if (ch >= 0x20 && ch < 0x7f) snprintf(out, n, "'%c'", ch);
  else snprintf(out, n, "byte 0x%02x", (unsigned)ch);
}

// Consume exactly ch, or die saying what the header holds instead.
static void need(Cur *c, int ch, const char *what) {
  int got = peek(c);
  if (got != ch) {
    char d[32];
    describe(got, d, sizeof d);
    die(c->path, "header: %s: expected '%c', found %s", what, ch, d);
  }
  c->p++;
}

static void utf8_put(char *s, size_t *n, unsigned cp) {
  if (cp < 0x80) {
    s[(*n)++] = (char)cp;
  } else if (cp < 0x800) {
    s[(*n)++] = (char)(0xc0 | (cp >> 6));
    s[(*n)++] = (char)(0x80 | (cp & 0x3f));
  } else {
    s[(*n)++] = (char)(0xe0 | (cp >> 12));
    s[(*n)++] = (char)(0x80 | ((cp >> 6) & 0x3f));
    s[(*n)++] = (char)(0x80 | (cp & 0x3f));
  }
}

static unsigned hex4(Cur *c) {
  unsigned v = 0;
  for (int i = 0; i < 4; i++) {
    if (c->p >= c->end) die(c->path, "header: a string ends inside a \\u escape");
    unsigned char ch = *c->p++;
    v <<= 4;
    if (ch >= '0' && ch <= '9') v |= (unsigned)(ch - '0');
    else if (ch >= 'a' && ch <= 'f') v |= (unsigned)(ch - 'a' + 10);
    else if (ch >= 'A' && ch <= 'F') v |= (unsigned)(ch - 'A' + 10);
    else die(c->path, "header: bad hex digit in a \\u escape");
  }
  return v;
}

// A JSON string (opening quote not yet consumed), unescaped into a fresh buffer. Escapes are decoded because tensor names
// are matched byte for byte; "\u" is decoded for the BMP and a surrogate pair is refused, since no key this reader is
// used for contains one and a wrong match would be worse than a clear error.
static char *parse_str(Cur *c) {
  need(c, '"', "a string");
  size_t cap = 32, n = 0;
  char *s = xmalloc(cap);
  for (;;) {
    if (c->p >= c->end) die(c->path, "header: unterminated string");
    unsigned char ch = *c->p++;
    if (ch == '"') break;
    if (ch == '\\') {
      if (c->p >= c->end) die(c->path, "header: a string ends in a backslash");
      unsigned char e = *c->p++;
      switch (e) {
      case '"': ch = '"'; break;
      case '\\': ch = '\\'; break;
      case '/': ch = '/'; break;
      case 'b': ch = '\b'; break;
      case 'f': ch = '\f'; break;
      case 'n': ch = '\n'; break;
      case 'r': ch = '\r'; break;
      case 't': ch = '\t'; break;
      case 'u': {
        unsigned cp = hex4(c);
        if (cp >= 0xd800 && cp <= 0xdfff)
          die(c->path, "header: \\u surrogate escapes are not supported in tensor names");
        if (n + 4 > cap) {
          cap *= 2;
          s = xrealloc(s, cap);
        }
        utf8_put(s, &n, cp);
        continue;
      }
      default:
        die(c->path, "header: unknown escape \\%c in a string", (char)e);
      }
    }
    if (n + 2 > cap) {
      cap *= 2;
      s = xrealloc(s, cap);
    }
    s[n++] = (char)ch;
  }
  s[n] = '\0';
  return s;
}

static void skip_value(Cur *c);

// An object or array at the cursor, consumed whole. Used for everything this reader does not care about ("__metadata__",
// an unknown tensor field).
static void skip_container(Cur *c) {
  int close = peek(c) == '{' ? '}' : ']';
  c->p++; // the opening brace or bracket
  if (peek(c) == close) {
    c->p++;
    return;
  }
  for (;;) {
    if (close == '}') {
      char *key = parse_str(c);
      free(key);
      need(c, ':', "':' after an object key");
    }
    skip_value(c);
    int ch = peek(c);
    if (ch == ',') {
      c->p++;
      continue;
    }
    if (ch == close) {
      c->p++;
      return;
    }
    char d[32];
    describe(ch, d, sizeof d);
    die(c->path, "header: expected ',' or '%c', found %s", close, d);
  }
}

// A string, a container, or a bare token (number / true / false / null), which runs to the next delimiter.
static void skip_value(Cur *c) {
  int ch = peek(c);
  if (ch == '"') {
    char *s = parse_str(c);
    free(s);
    return;
  }
  if (ch == '{' || ch == '[') {
    skip_container(c);
    return;
  }
  const unsigned char *start = c->p;
  while (c->p < c->end && *c->p != ',' && *c->p != '}' && *c->p != ']' && *c->p != ' ' && *c->p != '\t' &&
         *c->p != '\n' && *c->p != '\r')
    c->p++;
  if (c->p == start) {
    char d[32];
    describe(ch, d, sizeof d);
    die(c->path, "header: expected a value, found %s", d);
  }
}

// An unsigned integer: shape dims and data offsets are never negative and never fractional.
static unsigned long long parse_uint(Cur *c, const char *what) {
  int ch = peek(c);
  if (ch < '0' || ch > '9') {
    char d[32];
    describe(ch, d, sizeof d);
    die(c->path, "header: %s: expected an unsigned integer, found %s", what, d);
  }
  unsigned long long v = 0;
  while (c->p < c->end && *c->p >= '0' && *c->p <= '9') {
    unsigned d = (unsigned)(*c->p++ - '0');
    if (v > (~0ull - d) / 10) die(c->path, "header: %s: integer overflows", what);
    v = v * 10 + d;
  }
  return v;
}

// "shape": [1024] -> ndim, dims. More than ST_MAX_DIMS is a header this reader refuses.
static int parse_shape(Cur *c, int shape[ST_MAX_DIMS]) {
  need(c, '[', "an array");
  int ndim = 0;
  if (peek(c) == ']') {
    c->p++;
    return 0;
  }
  for (;;) {
    if (ndim >= ST_MAX_DIMS) die(c->path, "header: shape has more than %d dims", ST_MAX_DIMS);
    unsigned long long v = parse_uint(c, "shape");
    if (v > 0x7fffffffULL) die(c->path, "header: shape dim %llu is out of range", v);
    shape[ndim++] = (int)v;
    int ch = peek(c);
    if (ch == ',') {
      c->p++;
      continue;
    }
    if (ch == ']') {
      c->p++;
      return ndim;
    }
    char d[32];
    describe(ch, d, sizeof d);
    die(c->path, "header: shape: expected ',' or ']', found %s", d);
  }
}

// "data_offsets": [a, b] -> the two offsets, relative to the start of the data section.
static void parse_offsets(Cur *c, unsigned long long off[2]) {
  need(c, '[', "an array");
  off[0] = parse_uint(c, "data_offsets");
  need(c, ',', "a second data offset");
  off[1] = parse_uint(c, "data_offsets");
  need(c, ']', "the end of data_offsets");
}

// ---- the St ------------------------------------------------------------------------------------------------------------

typedef struct {
  char *path;
  const unsigned char *base;
  size_t size;
} StFile;

struct St {
  char *path; // what st_open was called with, for messages
  size_t nfiles;
  StFile *files;
  size_t ntensors;
  StTensor *tensors;
};

typedef struct {
  char *dtype; // the header's dtype string, NULL until "dtype" was seen
  int ndim;
  int shape[ST_MAX_DIMS];
  unsigned long long off[2];
  int have_offsets;
} Pending;

static void parse_tensor_obj(Cur *c, const char *name, Pending *t) {
  need(c, '{', "a tensor description");
  if (peek(c) == '}') die(c->path, "tensor \"%s\": the description is empty", name);
  for (;;) {
    char *key = parse_str(c);
    need(c, ':', "':' after a tensor field name");
    if (strcmp(key, "dtype") == 0) {
      free(t->dtype); // a repeated field is not a reason to leak the first one
      t->dtype = parse_str(c);
    } else if (strcmp(key, "shape") == 0) {
      t->ndim = parse_shape(c, t->shape);
    } else if (strcmp(key, "data_offsets") == 0) {
      parse_offsets(c, t->off);
      t->have_offsets = 1;
    } else {
      skip_value(c);
    }
    free(key);
    int ch = peek(c);
    if (ch == ',') {
      c->p++;
      continue;
    }
    if (ch == '}') {
      c->p++;
      return;
    }
    char d[32];
    describe(ch, d, sizeof d);
    die(c->path, "tensor \"%s\": expected ',' or '}', found %s", name, d);
  }
}

// Register one parsed tensor: check dtype, shape and offsets against the file, then point at its bytes. Takes ownership of
// name (freed by st_close) and of t->dtype.
static void add_tensor(St *st, const StFile *f, size_t hdr_len, const char *name, Pending *t) {
  if (t->dtype == NULL) die(f->path, "tensor \"%s\": no dtype", name);
  StDtype dtype = st_dtype_parse(t->dtype);
  if (dtype == ST_DTYPE_NONE) die(f->path, "tensor \"%s\": unknown dtype \"%s\"", name, t->dtype);
  if (!t->have_offsets) die(f->path, "tensor \"%s\": no data_offsets", name);

  unsigned long long elems = 1;
  for (int i = 0; i < t->ndim; i++) {
    elems *= (unsigned long long)t->shape[i];
    if (elems > 0xffffffffffffULL) die(f->path, "tensor \"%s\": the shape is too large", name);
  }
  unsigned long long nbytes = elems * (unsigned long long)st_dtype_size(dtype);
  size_t data_size = f->size - 8 - hdr_len;
  if (t->off[0] > t->off[1] || t->off[1] > (unsigned long long)data_size)
    die(f->path, "tensor \"%s\": data_offsets [%llu,%llu] are outside the %zu data byte(s)", name, t->off[0], t->off[1],
        data_size);
  if (t->off[1] - t->off[0] != nbytes)
    die(f->path, "tensor \"%s\": data_offsets span %llu byte(s), the shape needs %llu", name, t->off[1] - t->off[0],
        nbytes);
  for (size_t i = 0; i < st->ntensors; i++)
    if (strcmp(st->tensors[i].name, name) == 0) die(f->path, "tensor \"%s\": appears twice", name);

  st->tensors = xrealloc(st->tensors, (st->ntensors + 1) * sizeof *st->tensors);
  StTensor *dst = &st->tensors[st->ntensors++];
  memset(dst, 0, sizeof *dst);
  dst->name = name;
  dst->dtype = dtype;
  dst->ndim = t->ndim;
  for (int i = 0; i < t->ndim; i++) dst->shape[i] = t->shape[i];
  dst->data = f->base + 8 + hdr_len + (size_t)t->off[0];
  dst->nbytes = (size_t)nbytes;
  dst->file = f->path;
  free(t->dtype);
  t->dtype = NULL;
}

static unsigned long long read_u64le(const unsigned char *p) {
  unsigned long long v = 0;
  for (int i = 7; i >= 0; i--) v = (v << 8) | (unsigned long long)p[i];
  return v;
}

static void parse_header(St *st, const StFile *f) {
  if (f->size < 8) die(f->path, "only %zu byte(s): too short for a header length", f->size);
  unsigned long long hlen = read_u64le(f->base);
  if (hlen < 2 || hlen > ST_MAX_HEADER) die(f->path, "header length %llu is not a safetensors header", hlen);
  if (hlen > (unsigned long long)f->size - 8)
    die(f->path, "header length %llu does not fit in %zu byte(s)", hlen, f->size);

  Cur c = {f->base + 8, f->base + 8 + (size_t)hlen, f->path};
  need(&c, '{', "the header object");
  for (;;) {
    char *name = parse_str(&c);
    need(&c, ':', "':' after a tensor name");
    if (strcmp(name, "__metadata__") == 0) {
      skip_value(&c);
      free(name);
    } else {
      Pending t;
      memset(&t, 0, sizeof t);
      parse_tensor_obj(&c, name, &t);
      add_tensor(st, f, (size_t)hlen, name, &t);
    }
    int ch = peek(&c);
    if (ch == ',') {
      c.p++;
      continue;
    }
    if (ch == '}') {
      c.p++;
      break;
    }
    char d[32];
    describe(ch, d, sizeof d);
    die(f->path, "header: expected ',' or '}' after a tensor, found %s", d);
  }
  // The header is padded with spaces (safetensors' convention) up to its stated length; only padding may follow.
  while (c.p < c.end && (*c.p == ' ' || *c.p == '\0')) c.p++;
  if (c.p != c.end) die(f->path, "header: %zu byte(s) after the JSON object", (size_t)(c.end - c.p));
}

static void open_file(St *st, const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) die(NULL, "%s: %s", path, strerror(errno));
  struct stat sb;
  if (fstat(fd, &sb) != 0) die(path, "stat: %s", strerror(errno));
  if (!S_ISREG(sb.st_mode)) die(path, "not a regular file");
  if (sb.st_size <= 0) die(path, "empty file");
  size_t size = (size_t)sb.st_size;
  void *base = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED) die(path, "mmap: %s", strerror(errno));
  close(fd);

  st->files = xrealloc(st->files, (st->nfiles + 1) * sizeof *st->files);
  StFile *f = &st->files[st->nfiles++];
  f->path = xstrdup(path);
  f->base = base;
  f->size = size;
  parse_header(st, f);
}

// ---- sharded checkpoints (model.safetensors.index.json) ----------------------------------------------------------------

static char *read_text(const char *path, size_t *n_out) {
  FILE *fp = fopen(path, "rb");
  if (fp == NULL) die(NULL, "%s: %s", path, strerror(errno));
  if (fseek(fp, 0, SEEK_END) != 0) die(path, "seek: %s", strerror(errno));
  long len = ftell(fp);
  if (len < 0) die(path, "tell: %s", strerror(errno));
  if (fseek(fp, 0, SEEK_SET) != 0) die(path, "seek: %s", strerror(errno));
  char *buf = xmalloc((size_t)len + 1);
  size_t got = fread(buf, 1, (size_t)len, fp);
  if (got != (size_t)len) die(path, "read %zu of %ld byte(s)", got, len);
  fclose(fp);
  buf[got] = '\0';
  *n_out = got;
  return buf;
}

// {"metadata": {...}, "weight_map": {"tensor": "shard.safetensors", ...}} -> one (tensor, shard) pair per tensor.
static size_t parse_index(const char *path, const char *buf, size_t len, char ***names, char ***shards) {
  Cur c = {(const unsigned char *)buf, (const unsigned char *)buf + len, path};
  size_t n = 0, cap = 0;
  *names = NULL;
  *shards = NULL;
  need(&c, '{', "the index object");
  for (;;) {
    char *key = parse_str(&c);
    need(&c, ':', "':' after an index key");
    int is_map = strcmp(key, "weight_map") == 0;
    free(key);
    if (!is_map) {
      skip_value(&c);
    } else {
      need(&c, '{', "the weight_map object");
      if (peek(&c) != '}') {
        for (;;) {
          char *name = parse_str(&c);
          need(&c, ':', "':' after a weight_map key");
          char *shard = parse_str(&c);
          if (n == cap) {
            cap = cap ? cap * 2 : 16;
            *names = xrealloc(*names, cap * sizeof **names);
            *shards = xrealloc(*shards, cap * sizeof **shards);
          }
          (*names)[n] = name;
          (*shards)[n++] = shard;
          if (peek(&c) != ',') break;
          c.p++;
        }
      }
      need(&c, '}', "the end of weight_map");
    }
    int ch = peek(&c);
    if (ch == ',') {
      c.p++;
      continue;
    }
    if (ch == '}') {
      c.p++;
      break;
    }
    char d[32];
    describe(ch, d, sizeof d);
    die(path, "index: expected ',' or '}', found %s", d);
  }
  if (n == 0) die(path, "weight_map names no tensors");
  return n;
}

static void open_sharded(St *st, const char *dir, const char *index_path) {
  size_t len = 0;
  char *buf = read_text(index_path, &len);
  char **names = NULL, **shards = NULL;
  size_t n = parse_index(index_path, buf, len, &names, &shards);
  free(buf);

  // Open each distinct shard once, in the order the weight_map first mentions it, so the tensor order is stable.
  for (size_t i = 0; i < n; i++) {
    size_t j = 0;
    for (; j < i; j++)
      if (strcmp(shards[i], shards[j]) == 0) break;
    if (j != i) continue;
    char *p = join_path(dir, shards[i]);
    open_file(st, p);
    free(p);
  }
  // Every name the index promises must be there: a shard that does not hold the tensor it was named for is a broken
  // checkpoint, and would otherwise show up much later as a missing tensor in the middle of a load.
  for (size_t i = 0; i < n; i++)
    if (st_find(st, names[i]) == NULL)
      die(index_path, "weight_map names \"%s\", which %s does not contain", names[i], shards[i]);
  for (size_t i = 0; i < n; i++) {
    free(names[i]);
    free(shards[i]);
  }
  free(names);
  free(shards);
}

// ---- open / close / lookup ----------------------------------------------------------------------------------------------

St *st_open(const char *path) {
  if (path == NULL || path[0] == '\0') die(NULL, "st_open: no path");
  St *st = xmalloc(sizeof *st);
  memset(st, 0, sizeof *st);
  st->path = xstrdup(path);

  struct stat sb;
  if (stat(path, &sb) != 0) die(NULL, "%s: %s", path, strerror(errno));
  if (S_ISDIR(sb.st_mode)) {
    char *idx = join_path(path, "model.safetensors.index.json");
    struct stat isb;
    if (stat(idx, &isb) == 0) {
      open_sharded(st, path, idx);
    } else {
      char *one = join_path(path, "model.safetensors");
      open_file(st, one);
      free(one);
    }
    free(idx);
  } else {
    open_file(st, path);
  }
  if (st->ntensors == 0) die(path, "no tensors");
  return st;
}

void st_close(St *st) {
  if (st == NULL) return;
  for (size_t i = 0; i < st->ntensors; i++) free((char *)st->tensors[i].name);
  free(st->tensors);
  for (size_t i = 0; i < st->nfiles; i++) {
    munmap((void *)st->files[i].base, st->files[i].size);
    free(st->files[i].path);
  }
  free(st->files);
  free(st->path);
  free(st);
}

size_t st_count(const St *st) { return st == NULL ? 0 : st->ntensors; }

const StTensor *st_tensor(const St *st, size_t i) { return (st == NULL || i >= st->ntensors) ? NULL : &st->tensors[i]; }

const StTensor *st_find(const St *st, const char *name) {
  if (st == NULL || name == NULL) return NULL;
  for (size_t i = 0; i < st->ntensors; i++)
    if (strcmp(st->tensors[i].name, name) == 0) return &st->tensors[i];
  return NULL;
}

// "BF16 [2,3]" — a dtype and a shape, for the messages that say what the file has and what the caller wanted.
static void describe_shape(char *out, size_t n, StDtype dtype, int ndim, const int *shape) {
  size_t used = (size_t)snprintf(out, n, "%s [", st_dtype_name(dtype));
  for (int i = 0; i < ndim && used < n; i++) used += (size_t)snprintf(out + used, n - used, i ? ",%d" : "%d", shape[i]);
  if (used < n) snprintf(out + used, n - used, "]");
}

// One error path for every mismatch, so the message always shows both sides.
static void mismatch(const StTensor *t, const char *name, StDtype dtype, int ndim, const int *shape) {
  char want[256], got[256];
  describe_shape(want, sizeof want, dtype, ndim, shape);
  describe_shape(got, sizeof got, t->dtype, t->ndim, t->shape);
  die(t->file, "tensor \"%s\" is %s, the caller expects %s", name, got, want);
}

const void *st_getv(const St *st, const char *name, StDtype dtype, int ndim, const int *shape) {
  if (ndim < 0 || ndim > ST_MAX_DIMS) die(st ? st->path : NULL, "st_get: ndim %d is not 0..%d", ndim, ST_MAX_DIMS);
  if (ndim > 0 && shape == NULL) die(st ? st->path : NULL, "st_get: \"%s\": ndim %d but no shape", name, ndim);
  const StTensor *t = st_find(st, name);
  if (t == NULL) die(st ? st->path : NULL, "st_get: no tensor named \"%s\"", name);
  if (t->dtype != dtype || t->ndim != ndim) mismatch(t, name, dtype, ndim, shape);
  for (int i = 0; i < ndim; i++)
    if (t->shape[i] != shape[i]) mismatch(t, name, dtype, ndim, shape);
  return t->data;
}

const void *st_get(const St *st, const char *name, StDtype dtype, int ndim, ...) {
  int shape[ST_MAX_DIMS];
  if (ndim < 0 || ndim > ST_MAX_DIMS) die(st ? st->path : NULL, "st_get: ndim %d is not 0..%d", ndim, ST_MAX_DIMS);
  va_list ap;
  va_start(ap, ndim);
  for (int i = 0; i < ndim; i++) shape[i] = va_arg(ap, int);
  va_end(ap);
  return st_getv(st, name, dtype, ndim, shape);
}

// ---- the writer (design §9.2) -------------------------------------------------------------------------------------------

typedef struct {
  char *p;
  size_t n, cap;
} Buf;

static void buf_add(Buf *b, const char *s, size_t len) {
  if (b->n + len > b->cap) {
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->n + len) cap *= 2;
    b->p = xrealloc(b->p, cap);
    b->cap = cap;
  }
  memcpy(b->p + b->n, s, len);
  b->n += len;
}

static void buf_puts(Buf *b, const char *s) { buf_add(b, s, strlen(s)); }

static void buf_add_ll(Buf *b, long long v) {
  char tmp[32];
  int k = snprintf(tmp, sizeof tmp, "%lld", v);
  buf_add(b, tmp, (size_t)k);
}

// An insertion sort over the index array: 150-300 keys, and qsort_r's signature differs between glibc and Darwin.
static void sort_indices(int *idx, int n, const char *const names[]) {
  for (int i = 1; i < n; i++) {
    int v = idx[i], j = i - 1;
    while (j >= 0 && strcmp(names[idx[j]], names[v]) > 0) {
      idx[j + 1] = idx[j];
      j--;
    }
    idx[j + 1] = v;
  }
}

static void check_writable_name(const char *path, const char *name) {
  if (name[0] == '\0') die(path, "st_write: a tensor has no name");
  for (const char *s = name; *s; s++)
    if ((unsigned char)*s < 0x20 || *s == '"' || *s == '\\')
      die(path, "st_write: tensor name \"%s\" needs JSON escaping", name);
}

void st_write(const char *path, int n, const char *const names[], const int shapes[][ST_MAX_DIMS], const int ndims[],
              const float *const data[]) {
  if (path == NULL || path[0] == '\0') die(NULL, "st_write: no path");
  if (n <= 0) die(path, "st_write: nothing to write");
  if (names == NULL || shapes == NULL || ndims == NULL || data == NULL) die(path, "st_write: a NULL argument");

  size_t *nbytes = xmalloc((size_t)n * sizeof *nbytes);
  int *idx = xmalloc((size_t)n * sizeof *idx);
  unsigned long long total = 0;
  for (int i = 0; i < n; i++) {
    if (names[i] == NULL) die(path, "st_write: tensor %d has no name", i);
    check_writable_name(path, names[i]);
    if (ndims[i] < 0 || ndims[i] > ST_MAX_DIMS) die(path, "st_write: tensor \"%s\" has %d dims", names[i], ndims[i]);
    unsigned long long elems = 1;
    for (int k = 0; k < ndims[i]; k++) {
      if (shapes[i][k] < 0) die(path, "st_write: tensor \"%s\" has dim %d", names[i], shapes[i][k]);
      elems *= (unsigned long long)shapes[i][k];
    }
    if (elems == 0)
      die(path, "st_write: tensor \"%s\" has no elements; an empty tensor cannot be written", names[i]);
    if (elems > (~0ull - total) / 4u) die(path, "st_write: the data section overflows");
    if (data[i] == NULL) die(path, "st_write: tensor \"%s\" has no data", names[i]);
    nbytes[i] = (size_t)(elems * 4u);
    total += elems * 4u;
    idx[i] = i;
  }
  sort_indices(idx, n, names);
  for (int k = 1; k < n; k++)
    if (strcmp(names[idx[k - 1]], names[idx[k]]) == 0) die(path, "st_write: tensor \"%s\" twice", names[idx[k]]);

  // The header: keys in sorted order, padded with spaces to a multiple of 8. Names are checked, not escaped, because a
  // safetensors key never needs an escape and a silently escaped key would not match on read.
  Buf b = {NULL, 0, 0};
  buf_puts(&b, "{");
  unsigned long long cur = 0;
  for (int k = 0; k < n; k++) {
    int i = idx[k];
    if (k) buf_puts(&b, ",");
    buf_puts(&b, "\"");
    buf_puts(&b, names[i]);
    buf_puts(&b, "\":{\"dtype\":\"F32\",\"shape\":[");
    for (int d = 0; d < ndims[i]; d++) {
      if (d) buf_puts(&b, ",");
      buf_add_ll(&b, shapes[i][d]);
    }
    buf_puts(&b, "],\"data_offsets\":[");
    buf_add_ll(&b, (long long)cur);
    buf_puts(&b, ",");
    cur += nbytes[i];
    buf_add_ll(&b, (long long)cur);
    buf_puts(&b, "]}");
  }
  buf_puts(&b, "}");
  while (b.n % 8 != 0) buf_puts(&b, " ");
  if (b.n > ST_MAX_HEADER) die(path, "st_write: the header would be %zu bytes", b.n);

  FILE *fp = fopen(path, "wb");
  if (fp == NULL) die(path, "%s", strerror(errno));
  unsigned char len8[8];
  for (int i = 0; i < 8; i++) len8[i] = (unsigned char)(((unsigned long long)b.n >> (8 * i)) & 0xff);
  int bad = fwrite(len8, 1, 8, fp) != 8 || fwrite(b.p, 1, b.n, fp) != b.n;
  for (int k = 0; k < n && !bad; k++) {
    int i = idx[k];
    if (fwrite(data[i], 1, nbytes[i], fp) != nbytes[i]) bad = 1;
  }
  if (fclose(fp) != 0) bad = 1;
  if (bad) die(path, "write failed: %s", strerror(errno));

  free(b.p);
  free(idx);
  free(nbytes);
}
