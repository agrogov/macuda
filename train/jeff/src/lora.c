// src/lora.c — the flat trainable-parameter buffer, its adapter I/O and the schedule (plan M2.6, design §2.1, §2.2, §8.1,
// §9.2). See src/lora.h for the layout and why it is one buffer.
//
// The table is built from src/model_table.h, so the 150 stems and shapes here are the model's, and the readout's entry is
// last so that the two learning-rate groups are one index (design §2.2). Reading and writing an adapter is the format
// design §9.2 defines, which is what CLM-8B's tools/jeff_lora/lora.py writes and reads; the factors are fp32 there, the
// checkpoint's readout is bf16, and both come out of here as fp32 in the flat buffer.
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "util.h"
#include "st.h"
#include "model_table.h"
#include "lora.h"

// ---- small helpers ---------------------------------------------------------------------------------------------------
static void die(const char *fmt, ...) {
  va_list ap;
  fflush(stdout);
  fputs("lora: ", stderr);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

// "<dir>/<name>" into buf; a truncated path is a bug, not something to carry on from.
static const char *path_join(char *buf, size_t cap, const char *dir, const char *name) {
  const size_t dl = dir != NULL ? strlen(dir) : 0;
  const char *slash = (dl > 0 && dir[dl - 1] == '/') ? "" : "/";
  if (snprintf(buf, cap, "%s%s%s", dir != NULL ? dir : "", slash, name) >= (int)cap) die("path too long: %s/%s", dir, name);
  return buf;
}

// 10484736 -> "10,484,736": train.py's f"{n:,}", so the log line a script greps for is the same.
static const char *fmt_count(char *buf, size_t cap, size_t v) {
  char tmp[32];
  snprintf(tmp, sizeof tmp, "%zu", v);
  const size_t len = strlen(tmp);
  size_t o = 0;
  if (cap == 0) return buf;
  for (size_t i = 0; i < len && o + 2 < cap; i++) {
    if (i > 0 && (len - i) % 3 == 0) buf[o++] = ',';
    buf[o++] = tmp[i];
  }
  buf[o] = '\0';
  return buf;
}

// mkdir -p for the directory a save goes into (train.py's `directory.mkdir(parents=True, exist_ok=True)`).
static void mkdirs(const char *path) {
  char buf[4096];
  const size_t n = path != NULL ? strlen(path) : 0;
  if (n == 0 || n >= sizeof buf) die("bad output directory '%s'", path != NULL ? path : "(null)");
  memcpy(buf, path, n + 1);
  for (size_t i = 1; i <= n; i++) {
    if (buf[i] != '/' && buf[i] != '\0') continue;
    const char saved = buf[i];
    buf[i] = '\0';
    if (mkdir(buf, 0777) != 0 && errno != EEXIST) die("mkdir %s: %s", buf, strerror(errno));
    buf[i] = saved;
  }
}

// ---- the table and the buffers ----------------------------------------------------------------------------------------
// The adapter stem of a checkpoint weight: its name with the ".weight" suffix removed (design §9.2's `{stem}.lora_A`).
static void stem_of(char *dst, size_t cap, const char *name, const char *what) {
  const size_t n = name != NULL ? strlen(name) : 0;
  if (n <= 7 || strcmp(name + n - 7, ".weight") != 0) die("%s: '%s' is not a .weight tensor", what, name ? name : "?");
  if (n - 7 >= cap) die("%s: stem of '%s' does not fit in %zu bytes", what, name, cap);
  memcpy(dst, name, n - 7);
  dst[n - 7] = '\0';
}

Lora *lora_alloc(void) {
  Lora *l = (Lora *)calloc(1, sizeof *l);
  if (l == NULL) die("out of memory (Lora)");
  l->n_ent = LORA_ENTRIES;
  l->ent = (LoraEntry *)calloc((size_t)LORA_ENTRIES, sizeof *l->ent);
  if (l->ent == NULL) die("out of memory (%d entries)", LORA_ENTRIES);

  // The 150 pairs, in the model table's order (mt_lora_pair): A [rank,in] then B [out,rank], contiguous.
  size_t off = 0;
  for (int k = 0; k < MT_LORA_PAIRS; k++) {
    const MtLinear *m = mt_lora_pair(k);
    if (m == NULL || m->name == NULL || m->lora != MT_LORA) die("model table: LoRA pair %d is not adapted", k);
    if (m->in <= 0 || m->out <= 0) die("model table: LoRA pair %d ('%s') has shape [%d,%d]", k, m->name, m->out, m->in);
    LoraEntry *e = &l->ent[k];
    stem_of(e->stem, sizeof e->stem, m->name, "LoRA pair");
    e->in = m->in;
    e->out = m->out;
    e->rank = LORA_RANK;
    e->off = off;
    e->b_off = off + (size_t)LORA_RANK * (size_t)m->in;
    e->count = (size_t)LORA_RANK * (size_t)m->in + (size_t)m->out * (size_t)LORA_RANK;
    off += e->count;
  }
  // The readout, last: [255,1024] fp32, no B, its own (smaller) learning rate (design §2.1, §2.2).
  LoraEntry *ro = &l->ent[LORA_READOUT_ENTRY];
  snprintf(ro->stem, sizeof ro->stem, "readout");
  ro->in = MT_READOUT_IN;
  ro->out = MT_READOUT_OUT;
  ro->rank = 0;
  ro->off = off;
  ro->b_off = 0;
  ro->count = (size_t)MT_READOUT_IN * (size_t)MT_READOUT_OUT;
  l->readout_off = off;
  off += ro->count;
  l->n_param = off;

  char cnt[32], ab[32], rd[32], split[32];
  fmt_count(cnt, sizeof cnt, l->n_param);
  fmt_count(ab, sizeof ab, l->readout_off);
  fmt_count(rd, sizeof rd, l->n_param - l->readout_off);
  fmt_count(split, sizeof split, l->readout_off);
  printf("lora: entries %d (%d LoRA pairs + readout)  rank %d  scale %.1f  trainable %s\n", l->n_ent, MT_LORA_PAIRS,
         LORA_RANK, (double)MT_LORA_SCALE, cnt);
  printf("lora: flat buffer %s floats = %s LoRA (lr group split at %s) + %s readout\n", cnt, ab, split, rd);
  fflush(stdout);

  const size_t bytes = l->n_param * sizeof(float);
  CK(cudaMalloc((void **)&l->param, bytes));
  CK(cudaMalloc((void **)&l->grad, bytes));
  CK(cudaMalloc((void **)&l->m, bytes));
  CK(cudaMalloc((void **)&l->v, bytes));
  CK(cudaMemset(l->param, 0, bytes)); // the caller initialises: an adapter load, or main.c's own init (design §2.1)
  CK(cudaMemset(l->grad, 0, bytes));
  CK(cudaMemset(l->m, 0, bytes));
  CK(cudaMemset(l->v, 0, bytes));
  return l;
}

void lora_free(Lora *l) {
  if (l == NULL) return;
  if (l->param != NULL) CK(cudaFree(l->param));
  if (l->grad != NULL) CK(cudaFree(l->grad));
  if (l->m != NULL) CK(cudaFree(l->m));
  if (l->v != NULL) CK(cudaFree(l->v));
  free(l->ent);
  free(l);
}

void lora_zero_grad(const Lora *l, cudaStream_t s) {
  if (l == NULL || l->grad == NULL) return;
  CK(cudaMemsetAsync(l->grad, 0, l->n_param * sizeof(float), s));
}

// ---- reading an adapter ----------------------------------------------------------------------------------------------
// A tensor as host fp32: the trainer's own files are fp32 and a checkpoint's readout is bf16 (design §9.2, plan M2.6).
static float *to_host_f32(const StTensor *t, const char *what) {
  size_t n = 1;
  for (int i = 0; i < t->ndim; i++) n *= (size_t)t->shape[i];
  float *out = (float *)malloc((n ? n : 1) * sizeof *out);
  if (out == NULL) die("%s: out of memory (%zu elements)", what, n);
  if (t->dtype == ST_DTYPE_F32) {
    if (t->nbytes != n * sizeof(float)) die("%s: %zu bytes for %zu F32 elements", what, t->nbytes, n);
    memcpy(out, t->data, n * sizeof(float));
  } else if (t->dtype == ST_DTYPE_BF16) {
    if (t->nbytes != n * sizeof(unsigned short)) die("%s: %zu bytes for %zu BF16 elements", what, t->nbytes, n);
    const unsigned short *b = (const unsigned short *)t->data;
    for (size_t i = 0; i < n; i++) { // bf16 -> fp32 is `bits << 16`, nothing else (design §8.1)
      const unsigned int u = (unsigned int)b[i] << 16;
      memcpy(&out[i], &u, sizeof u);
    }
  } else {
    die("%s: dtype %s, want F32 or BF16", what, st_dtype_name(t->dtype));
  }
  return out;
}

// One [rows,cols] matrix, by the first of `cands` that the file has, checked against the model table's shape and uploaded
// to `dev_dst`.
static void load_matrix(const St *st, const char *file, const char *const cands[], int ncand, const char *what, int rows,
                        int cols, float *dev_dst) {
  const StTensor *t = NULL;
  const char *used = NULL;
  for (int i = 0; i < ncand && t == NULL; i++) {
    t = st_find(st, cands[i]);
    if (t != NULL) used = cands[i];
  }
  if (t == NULL) die("%s: '%s' is not in %s", what, cands[0], file);
  if (t->ndim != 2 || t->shape[0] != rows || t->shape[1] != cols) {
    char got[48];
    if (t->ndim == 2)
      snprintf(got, sizeof got, "[%d,%d]", t->shape[0], t->shape[1]);
    else
      snprintf(got, sizeof got, "%d-dimensional", t->ndim);
    die("%s: '%s' is %s, want [%d,%d]", what, used, got, rows, cols);
  }
  float *host = to_host_f32(t, what);
  CK(cudaMemcpy(dev_dst, host, (size_t)rows * (size_t)cols * sizeof(float), cudaMemcpyHostToDevice));
  free(host);
}

void lora_load_adapter(Lora *l, const char *dir) {
  if (l == NULL || dir == NULL) die("lora_load_adapter: no buffer or directory");
  char path[4096], what[LORA_STEM_MAX + 96];
  // Three spellings of one key: the trainer's own (design §9.2), and the two a PEFT-written adapter in the wild uses
  // (`.lora_A.weight`, usually behind `base_model.model.`). Our own file matches the first, which is what lora_save_adapter
  // writes; the others cost one failed lookup each and let the same code read the adapters that already exist.
  char cand[3][LORA_STEM_MAX + 48];
  const char *cands[3] = {cand[0], cand[1], cand[2]};

  St *st = st_open(path_join(path, sizeof path, dir, "adapter_model.safetensors"));
  for (int k = 0; k < MT_LORA_PAIRS; k++) {
    const LoraEntry *e = &l->ent[k];
    snprintf(cand[0], sizeof cand[0], "%s.lora_A", e->stem);
    snprintf(cand[1], sizeof cand[1], "%s.lora_A.weight", e->stem);
    snprintf(cand[2], sizeof cand[2], "base_model.model.%s.lora_A.weight", e->stem);
    snprintf(what, sizeof what, "%s.lora_A", e->stem);
    load_matrix(st, path, cands, 3, what, e->rank, e->in, l->param + e->off);
    snprintf(cand[0], sizeof cand[0], "%s.lora_B", e->stem);
    snprintf(cand[1], sizeof cand[1], "%s.lora_B.weight", e->stem);
    snprintf(cand[2], sizeof cand[2], "base_model.model.%s.lora_B.weight", e->stem);
    snprintf(what, sizeof what, "%s.lora_B", e->stem);
    load_matrix(st, path, cands, 3, what, e->out, e->rank, l->param + e->b_off);
  }
  st_close(st);

  const LoraEntry *ro = &l->ent[LORA_READOUT_ENTRY];
  const char *rocands[1] = {mt_readout_name};
  st = st_open(path_join(path, sizeof path, dir, mt_readout_file));
  load_matrix(st, path, rocands, 1, "readout", ro->out, ro->in, l->param + ro->off);
  st_close(st);
}

// ---- writing an adapter ----------------------------------------------------------------------------------------------
// Just enough JSON for adapter_config.json (design §9.2). When src/json.c exists (design §13) this belongs there; it is
// here for now so that a saved adapter is complete and readable by CLM-8B's tools without waiting for another file.
static void json_string(FILE *f, const char *s) {
  fputc('"', f);
  for (const unsigned char *p = (const unsigned char *)s; *p != '\0'; p++) {
    if (*p == '"' || *p == '\\')
      fprintf(f, "\\%c", *p);
    else if (*p < 0x20)
      fprintf(f, "\\u%04x", (unsigned)*p);
    else
      fputc(*p, f);
  }
  fputc('"', f);
}

void lora_save_adapter(const Lora *l, const char *dir, int step, double dev_ce, const LoraMeta *meta) {
  if (l == NULL || dir == NULL) die("lora_save_adapter: no buffer or directory");
  mkdirs(dir);

  const size_t nfloat = l->n_param;
  float *host = (float *)malloc(nfloat * sizeof *host);
  if (host == NULL) die("out of memory (host copy, %zu elements)", nfloat);
  CK(cudaMemcpy(host, l->param, nfloat * sizeof *host, cudaMemcpyDeviceToHost));

  // The 300 factors: per pair, `{stem}.lora_A` [rank,in] then `{stem}.lora_B` [out,rank], the order the table keeps.
  const int nfac = 2 * MT_LORA_PAIRS;
  char(*names)[LORA_STEM_MAX + 16] = (char(*)[LORA_STEM_MAX + 16])calloc((size_t)nfac, sizeof *names);
  const char **namep = (const char **)calloc((size_t)nfac, sizeof *namep);
  int(*shapes)[ST_MAX_DIMS] = (int(*)[ST_MAX_DIMS])calloc((size_t)nfac, sizeof *shapes);
  int *ndims = (int *)calloc((size_t)nfac, sizeof *ndims);
  const float **data = (const float **)calloc((size_t)nfac, sizeof *data);
  if (names == NULL || namep == NULL || shapes == NULL || ndims == NULL || data == NULL) die("out of memory (names)");

  for (int k = 0; k < MT_LORA_PAIRS; k++) {
    const LoraEntry *e = &l->ent[k];
    for (int half = 0; half < 2; half++) {
      const int i = 2 * k + half;
      if (half == 0) {
        snprintf(names[i], sizeof names[i], "%s.lora_A", e->stem);
        shapes[i][0] = e->rank;
        shapes[i][1] = e->in;
        data[i] = host + e->off;
      } else {
        snprintf(names[i], sizeof names[i], "%s.lora_B", e->stem);
        shapes[i][0] = e->out;
        shapes[i][1] = e->rank;
        data[i] = host + e->b_off;
      }
      ndims[i] = 2;
      namep[i] = names[i];
    }
  }
  char path[4096];
  st_write(path_join(path, sizeof path, dir, "adapter_model.safetensors"), nfac, namep,
           (const int(*)[ST_MAX_DIMS])shapes, ndims, data);

  const LoraEntry *ro = &l->ent[LORA_READOUT_ENTRY];
  const int roshape[ST_MAX_DIMS] = {ro->out, ro->in};
  const float *rodata = host + ro->off;
  const char *roname = mt_readout_name;
  st_write(path_join(path, sizeof path, dir, mt_readout_file), 1, &roname, &roshape, (const int[]){2}, &rodata);

  FILE *f = fopen(path_join(path, sizeof path, dir, "adapter_config.json"), "w");
  if (f == NULL) die("cannot write %s: %s", path, strerror(errno));
  fprintf(f, "{\n  \"format\": \"anemll-jeff-lora-1\",\n  \"rank\": %d,\n  \"scale\": %.1f,\n  \"layers\": %d,\n",
          LORA_RANK, (double)MT_LORA_SCALE, MT_LORA_PAIRS);
  fprintf(f, "  \"step\": %d,\n  \"dev_soft_ce\": %.10g", step, dev_ce);
  if (meta != NULL && meta->base != NULL) {
    fputs(",\n  \"base\": ", f);
    json_string(f, meta->base);
  }
  if (meta != NULL && meta->targets != NULL) {
    fputs(",\n  \"targets\": ", f);
    json_string(f, meta->targets);
  }
  if (meta != NULL && meta->weights != NULL) {
    fputs(",\n  \"weights\": ", f);
    json_string(f, meta->weights);
  }
  fputs("\n}\n", f);
  if (fclose(f) != 0) die("cannot finish %s: %s", path, strerror(errno));

  free(host);
  free(names);
  free(namep);
  free(shapes);
  free(ndims);
  free((void *)data);
}

// ---- the schedule ----------------------------------------------------------------------------------------------------
double lr_at(int step, int total, double base) {
  static const double PI = 3.14159265358979323846;
  if (total < 1) total = 1; // train.py never runs a zero-step schedule; 1 makes the formula below well defined
  int warm = (int)((double)total * 0.03); // Python's int(total * .03), truncated
  if (warm < 1) warm = 1;
  double up = (double)(step + 1) / (double)warm;
  if (up > 1.0) up = 1.0;
  const int smin = step < total ? step : total;
  const double cyc = 0.1 + 0.9 * 0.5 * (1.0 + cos(PI * (double)smin / (double)total));
  return base * up * cyc;
}
