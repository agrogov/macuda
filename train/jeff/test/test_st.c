// test/test_st.c — M1.3: the safetensors writer and reader, end to end, plus the checkpoint listing the plan's Done-when
// compares with design §3.2.
//
// No arguments runs everything that needs no real weights: a three-tensor round trip (st_write -> st_open/st_get, bytes
// compared against the host buffers that were written), a sharded checkpoint built from two F32 shards plus a
// model.safetensors.index.json, and the shape of the header the writer produces (keys sorted, length a multiple of 8).
// It needs no GPU, so `make test-cpu` and `make test-null` both run it.
//   test_st                     round trip; prints "st OK" and exits 0
//   test_st --list PATH         every tensor's name, dtype and shape (PATH = a .safetensors file or a model directory)
//   test_st --write PATH        write the round-trip fixture to PATH and leave it there (read it back with Python's
//                               safetensors: that is what py/compare_layers.py will do with a jefftrain dump)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "st.h"

#define NAMES 3
#define PATHLEN 512
#define FIX_MAXELEMS 24

static int nfail;

static void check(int ok, const char *what, const char *detail) {
  if (ok) {
    printf("ok    %s%s%s\n", what, detail && detail[0] ? " " : "", detail ? detail : "");
  } else {
    printf("FAIL  %s%s%s\n", what, detail && detail[0] ? " " : "", detail ? detail : "");
    nfail++;
  }
}

static void make_path(char *out, size_t n, const char *dir, const char *name) {
  size_t dl = strlen(dir), nl = strlen(name);
  if (dl + nl + 2 > n) {
    printf("FAIL  the path %s/%s does not fit in %zu bytes\n", dir, name, n);
    exit(1);
  }
  memcpy(out, dir, dl);
  size_t i = dl;
  if (i > 0 && out[i - 1] != '/') out[i++] = '/';
  memcpy(out + i, name, nl + 1);
}

// ---- the fixture: three F32 tensors of different ranks, deliberately listed in an order that is not the sorted one ---

static const char *const FIX_NAMES[NAMES] = {"gamma", "alpha", "beta"};
static const int FIX_NDIM[NAMES] = {3, 1, 2};
static const int FIX_SHAPE[NAMES][ST_MAX_DIMS] = {
    {2, 3, 4, 0, 0, 0, 0, 0},
    {4, 0, 0, 0, 0, 0, 0, 0},
    {3, 5, 0, 0, 0, 0, 0, 0},
};
static float fix_buf[NAMES][FIX_MAXELEMS]; // what the writer is given, and what the reader is compared against

static size_t fix_elems(int k) {
  size_t n = 1;
  for (int d = 0; d < FIX_NDIM[k]; d++) n *= (size_t)FIX_SHAPE[k][d];
  return n;
}

// Deterministic, with negative values and a zero: a byte-order or offset mistake cannot pass by accident.
static void make_fixture(void) {
  for (int k = 0; k < NAMES; k++) {
    size_t n = fix_elems(k);
    for (size_t i = 0; i < n; i++) fix_buf[k][i] = ((float)(k - 1) * 0.5f) + ((float)(i % 7) - 3.0f) * 0.25f;
  }
}

static void write_fixture(const char *path) {
  const float *data[NAMES];
  for (int k = 0; k < NAMES; k++) data[k] = fix_buf[k];
  st_write(path, NAMES, FIX_NAMES, FIX_SHAPE, FIX_NDIM, data);
}

// ---- the header as it sits in the file --------------------------------------------------------------------------------

static void check_header(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (fp == NULL) {
    check(0, "header opens", path);
    return;
  }
  unsigned char len8[8];
  size_t got = fread(len8, 1, 8, fp);
  unsigned long long hlen = 0;
  for (int i = 7; i >= 0; i--) hlen = (hlen << 8) | (unsigned long long)len8[i];
  check(got == 8 && hlen >= 2 && hlen % 8 == 0, "header length is a multiple of 8", "");

  enum { HDRMAX = 4096 };
  char hdr[HDRMAX];
  int fits = got == 8 && hlen < HDRMAX;
  size_t hgot = fits ? fread(hdr, 1, (size_t)hlen, fp) : 0;
  hdr[hgot] = '\0';
  fclose(fp);
  check(fits && hgot == hlen, "header length matches the bytes written", "");

  // Design §9.2: keys sorted, one object per tensor with dtype, shape and data_offsets, data starting at 0.
  const char *a = strstr(hdr, "\"alpha\":");
  const char *b = strstr(hdr, "\"beta\":");
  const char *g = strstr(hdr, "\"gamma\":");
  check(strncmp(hdr, "{\"alpha\":", 9) == 0 && a && b && g && a < b && b < g, "keys are sorted", "");
  check(strstr(hdr, "\"dtype\":\"F32\"") != NULL, "dtype is written as F32", "");
  check(strstr(hdr, "\"data_offsets\":[0,") != NULL, "the first data offset is 0", "");
}

// ---- the round trip ----------------------------------------------------------------------------------------------------

static void check_fixture(const St *st) {
  static const char *const sorted[NAMES] = {"alpha", "beta", "gamma"};
  char detail[128];
  check(st_count(st) == NAMES, "three tensors", "");
  for (size_t i = 0; i < st_count(st) && i < NAMES; i++) {
    const StTensor *t = st_tensor(st, i);
    snprintf(detail, sizeof detail, "%s", t != NULL ? t->name : "(null)");
    check(t != NULL && strcmp(t->name, sorted[i]) == 0, "read order is name order", detail);
  }
  for (int k = 0; k < NAMES; k++) {
    size_t n = fix_elems(k) * sizeof(float);
    const void *got = st_getv(st, FIX_NAMES[k], ST_DTYPE_F32, FIX_NDIM[k], FIX_SHAPE[k]);
    snprintf(detail, sizeof detail, "%s %zu bytes", FIX_NAMES[k], n);
    check(got != NULL && memcmp(got, fix_buf[k], n) == 0, "bytes round trip", detail);
    const StTensor *t = st_find(st, FIX_NAMES[k]);
    check(t != NULL && t->nbytes == n && t->dtype == ST_DTYPE_F32 && t->ndim == FIX_NDIM[k], "dtype, rank and size",
          FIX_NAMES[k]);
    int same_shape = t != NULL;
    for (int d = 0; d < FIX_NDIM[k] && same_shape; d++) same_shape = t->shape[d] == FIX_SHAPE[k][d];
    check(same_shape, "shape", FIX_NAMES[k]);
  }
  // The variadic form the design §3.2 call sites use: dims as arguments.
  const float *alpha = st_get(st, "alpha", ST_DTYPE_F32, 1, FIX_SHAPE[1][0]);
  check(alpha != NULL && memcmp(alpha, fix_buf[1], fix_elems(1) * sizeof(float)) == 0, "st_get with variadic dims",
        "alpha");
  check(st_find(st, "nope") == NULL, "an absent name is not found", "");
  check(st_tensor(st, NAMES) == NULL, "an out-of-range index is NULL", "");
}

// ---- a sharded checkpoint (model.safetensors.index.json) ---------------------------------------------------------------

// Written by hand: st_write only writes tensors, and the shard list is the point of the test.
static void write_index(const char *path) {
  FILE *fp = fopen(path, "wb");
  if (fp == NULL) {
    check(0, "index file written", path);
    return;
  }
  const char *json = "{\"metadata\":{\"format\":\"pt\"},\"weight_map\":{"
                     "\"a.x\":\"s0.safetensors\",\"c.z\":\"s0.safetensors\",\"b.y\":\"s1.safetensors\"}}";
  size_t n = strlen(json);
  int ok = fwrite(json, 1, n, fp) == n;
  check(ok && fclose(fp) == 0, "index file written", path);
}

static void check_sharded(const char *sharddir) {
  static float va[2] = {1.0f, -2.0f}, vc[2] = {3.0f, 4.0f}, vb[2] = {7.0f, 8.5f};
  const char *n0[2] = {"c.z", "a.x"};
  const int s0[2][ST_MAX_DIMS] = {{1, 2, 0, 0, 0, 0, 0, 0}, {2, 0, 0, 0, 0, 0, 0, 0}};
  const int d0[2] = {2, 1};
  const float *p0[2] = {vc, va};
  const char *n1[1] = {"b.y"};
  const int s1[1][ST_MAX_DIMS] = {{1, 2, 0, 0, 0, 0, 0, 0}};
  const int d1[1] = {2};
  const float *p1[1] = {vb};
  char p[PATHLEN];

  make_path(p, sizeof p, sharddir, "s0.safetensors");
  st_write(p, 2, n0, s0, d0, p0);
  make_path(p, sizeof p, sharddir, "s1.safetensors");
  st_write(p, 1, n1, s1, d1, p1);
  make_path(p, sizeof p, sharddir, "model.safetensors.index.json");
  write_index(p);

  St *st = st_open(sharddir);
  const int a_shape[1] = {2};
  const int c_shape[2] = {1, 2};
  check(st_count(st) == 3, "a sharded checkpoint has every tensor", "");
  check(memcmp(st_getv(st, "a.x", ST_DTYPE_F32, 1, a_shape), va, sizeof va) == 0, "shard 0 tensor", "a.x");
  check(memcmp(st_getv(st, "c.z", ST_DTYPE_F32, 2, c_shape), vc, sizeof vc) == 0, "shard 0 tensor", "c.z");
  check(memcmp(st_getv(st, "b.y", ST_DTYPE_F32, 2, c_shape), vb, sizeof vb) == 0, "shard 1 tensor", "b.y");
  // Shards are opened in the order the weight_map first names them (s0 holds a.x, so s0 comes first), and inside a file
  // the tensors keep its header order — which st_write sorted, so a.x precedes c.z there.
  check(st_count(st) == 3 && strcmp(st_tensor(st, 0)->name, "a.x") == 0 &&
            strcmp(st_tensor(st, 1)->name, "c.z") == 0 && strcmp(st_tensor(st, 2)->name, "b.y") == 0,
        "shard order follows the weight_map", "");
  st_close(st);
}

// ---- the commands ------------------------------------------------------------------------------------------------------

static void print_tensors(const St *st) {
  for (size_t i = 0; i < st_count(st); i++) {
    const StTensor *t = st_tensor(st, i);
    printf("%s %s [", t->name, st_dtype_name(t->dtype));
    for (int d = 0; d < t->ndim; d++) printf(d ? ",%d" : "%d", t->shape[d]);
    printf("]\n");
  }
}

// One file: every tensor, in header order.
static void list_file(const char *path) {
  St *st = st_open(path);
  print_tensors(st);
  printf("# %s: %zu tensor(s)\n", path, st_count(st));
  st_close(st);
}

// A directory is a checkpoint: model.safetensors (or its shards, via the index) and, when it is there, readout.safetensors.
static int list_cmd(const char *path) {
  struct stat sb;
  if (stat(path, &sb) != 0) {
    fprintf(stderr, "test_st: %s: no such file or directory\n", path);
    return 1;
  }
  if (!S_ISDIR(sb.st_mode)) {
    list_file(path);
    return 0;
  }
  char p[PATHLEN];
  make_path(p, sizeof p, path, "readout.safetensors");
  struct stat rsb;
  int has_readout = stat(p, &rsb) == 0;

  printf("== %s\n", path);
  St *model = st_open(path);
  print_tensors(model);
  printf("# model: %zu tensor(s)\n", st_count(model));
  st_close(model);
  if (has_readout) {
    printf("== %s\n", p);
    list_file(p);
  }
  return 0;
}

// st_write -> st_open -> st_get, plus the header rules, in a fresh temporary directory.
static int roundtrip(void) {
  char tmpl[PATHLEN];
  const char *tmpdir = getenv("TMPDIR");
  snprintf(tmpl, sizeof tmpl, "%s/jt_st_XXXXXX", tmpdir != NULL && tmpdir[0] ? tmpdir : "/tmp");
  if (mkdtemp(tmpl) == NULL) {
    printf("FAIL  mkdtemp(%s)\n", tmpl);
    return 1;
  }
  char path[PATHLEN], sharddir[PATHLEN];
  make_path(path, sizeof path, tmpl, "rt.safetensors");
  make_path(sharddir, sizeof sharddir, tmpl, "shards");
  mkdir(sharddir, 0700);

  make_fixture();
  write_fixture(path);
  check_header(path);
  St *st = st_open(path);
  check_fixture(st);
  st_close(st);
  check_sharded(sharddir);

  // Clean up: the files, then the directory that held them.
  unlink(path);
  char p[PATHLEN];
  const char *const leftovers[] = {"s0.safetensors", "s1.safetensors", "model.safetensors.index.json"};
  for (size_t i = 0; i < sizeof leftovers / sizeof leftovers[0]; i++) {
    make_path(p, sizeof p, sharddir, leftovers[i]);
    unlink(p);
  }
  rmdir(sharddir);
  rmdir(tmpl);
  return nfail == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc >= 2 && strcmp(argv[1], "--list") == 0) {
    if (argc != 3) {
      fprintf(stderr, "usage: test_st --list <file-or-model-dir>\n");
      return 2;
    }
    return list_cmd(argv[2]);
  }
  if (argc >= 2 && strcmp(argv[1], "--write") == 0) {
    if (argc != 3) {
      fprintf(stderr, "usage: test_st --write <path.safetensors>\n");
      return 2;
    }
    make_fixture();
    write_fixture(argv[2]);
    printf("st: wrote %s\n", argv[2]);
    return 0;
  }
  if (argc >= 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    printf("usage: test_st [--list <file-or-model-dir> | --write <path>]\n");
    return 0;
  }
  if (argc >= 2) {
    fprintf(stderr, "test_st: unknown argument \"%s\" (try --help)\n", argv[1]);
    return 2;
  }
  if (roundtrip() != 0) {
    printf("FAIL test_st: %d check(s) failed\n", nfail);
    return 1;
  }
  printf("st OK\n");
  return 0;
}
