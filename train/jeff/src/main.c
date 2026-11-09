// src/main.c — jefftrain's driver: the flags, the model load, the adapter, and the whole-model forward and backward dumps
// (plan M4.3, M5.1 item 4, design §3.4, §6.2, §7, §10.2, §13).
//
// What exists here now is M4.3's and M5.1's pieces of the trainer: `--dump-forward OUT` runs one packed batch through the
// embedding gather, all 24 decoder layers and the decision head, and `--dump-backward OUT` runs the same batch through the
// head's backward and design §6.2's layer-recompute loop, then writes the gradients too. Both write OUT in
// py/dump_ref.py's format so that py/compare_layers.py can compare it with `ref.safetensors` key by key (plan M4.3's and
// M5.1's gates; the card checkpoints M4.4 and M5.4 use the same two files). `--model` and `--init-adapter` are plan M4.2's
// flags; `--train` is train.py's name for the .jtb split the run consumes (plan M5.2), and both dumps read its batch 0 —
// the same batch dump_ref.py dumps, so the files describe the same tokens.
//
// The training step itself (Adam, the schedule, the log lines) is plan M5.2 and is NOT here: --train without an action
// flag loads everything and then says so with exit 2 (a run that did no work must not look like a run).
//
// The keys the forward dump writes are dump_ref.py's *forward* keys:
//
//   in.ids [N]  in.off [B+1]                        the packed batch (see the note below: F32, not I32)
//   layer.{i}.in [N,1024]  layer.{i}.post_mixer [N,1024]  layer.{i}.out [N,1024]     i = 0..23
//   final [N,1024]  logits [B,n_opts]  loss []       the head (design §4.1)
//
// and the backward dump writes those plus every `grad.*` key dump_ref.py writes — a `grad.layer.{i}.out` per layer,
// `grad.final`, `grad.{stem}.lora_A/B` for all 150 adapted projections and `grad.readout`, i.e. all 403 keys of
// ref.safetensors, so compare_layers.py can compare the whole file (only the forward keys can also be compared on their
// own, which is why --dump-forward against the full ref reports the grad.* keys as missing: a forward cannot produce
// them).
//
// Two notes on being faithful to dump_ref.py:
//   * in.ids / in.off are integers there and F32 here, because src/st.h's writer is F32-only (design §9.2). Every id is
//     below MT_VOCAB = 248320 and every offset below the batch's token count, all of which are exact in fp32, and
//     compare_layers.py compares an integer key with a float one by value — so the check is still exact.
//   * the batch is the fixture's batch 0 (rows packed back to back, design §7), its targets are read from the same .jtb
//     rows, and the loss is train.py's `(per * wt).sum() / gw` with `gw` the batch's own summed row weight — the values
//     dump_ref.py feeds its own copy of train.py's loop.
//
// Memory: a dump sizes the activation arena for the batch it runs (cap_tokens = N, cap_rows = B, lmax = the longest
// row). The forward dump holds 25 activation slabs [N,1024] plus the 24 post_mixer slabs on the device because those are
// exactly the keys it writes; the backward dump holds the same plus one slab of layer-output gradients and the
// recompute's scratch — dump-time costs a training step does not pay (design §6.2).
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gemm.h"
#include "jtb.h"
#include "layer.h"
#include "lora.h"
#include "model.h"
#include "st.h"
#include "util.h"

typedef struct {
  const char *model;    // --model DIR: the checkpoint directory (or a model.safetensors file)
  const char *adapter;  // --init-adapter DIR: an adapter to load over the checkpoint's readout (may be absent)
  const char *train;    // --train FILE.jtb: train.py's flag for the split this run consumes
  const char *dump_fwd; // --dump-forward OUT: write the whole forward there (plan M4.3)
  const char *dump_bwd; // --dump-backward OUT: write the forward and every gradient there (plan M5.1 item 4)
} Args;

// src/model.c's design §6.2 backward driver. Declared here, not in src/model.h (another task's frozen file): this flag is
// its only caller in this build. `post_mixer` and `grad_out` are the optional dump buffers [MT_LAYERS][n][MT_HIDDEN].
int model_backward(JtCtx *c, const float *saved_in, int n, const JtBatch *b, float *post_mixer, float *grad_out,
                   float *dres);

static void die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void die(const char *fmt, ...) {
  va_list ap;
  fflush(stdout);
  fputs("jefftrain: ", stderr);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

static void usage(FILE *f) {
  fprintf(f,
          "usage: jefftrain --model DIR [--init-adapter DIR] [--train FILE.jtb] [--dump-forward OUT]\n"
          "                                                        [--dump-backward OUT]\n"
          "  --model DIR         the jeff-base checkpoint directory (model.safetensors + readout.safetensors)\n"
          "  --init-adapter DIR  load a Python-written adapter over the checkpoint's readout (train.py's flag)\n"
          "  --train FILE        the .jtb split (train.py's flag, plan M5.2); the dumps read its batch 0\n"
          "  --dump-forward OUT  run the embedding gather, 24 layers and the head on that batch and write OUT in\n"
          "                      py/dump_ref.py's format (plan M4.3; reads it with py/compare_layers.py)\n"
          "  --dump-backward OUT run the head's and the 24 layers' backward too (design §6.2's recompute loop) and\n"
          "                      write the forward keys plus every grad.* key of py/dump_ref.py (plan M5.1 item 4)\n"
          "  (no arguments)      print this and exit 0\n"
          "The training step (plan M5.2) is not part of this build: --train alone loads and exits 2.\n");
}

static int parse_args(int argc, char **argv, Args *a) {
  memset(a, 0, sizeof *a);
  for (int k = 1; k < argc; k++) {
    const char *opt = argv[k];
    const int has = k + 1 < argc;
    if (strcmp(opt, "-h") == 0 || strcmp(opt, "--help") == 0) {
      usage(stdout);
      exit(0);
    } else if (strcmp(opt, "--model") == 0 && has) {
      a->model = argv[++k];
    } else if (strcmp(opt, "--init-adapter") == 0 && has) {
      a->adapter = argv[++k];
    } else if (strcmp(opt, "--train") == 0 && has) {
      a->train = argv[++k];
    } else if (strcmp(opt, "--dump-forward") == 0 && has) {
      a->dump_fwd = argv[++k];
    } else if (strcmp(opt, "--dump-backward") == 0 && has) {
      a->dump_bwd = argv[++k];
    } else {
      die("unknown or incomplete option '%s' (--help for the usage)", opt);
    }
  }
  return 0;
}

// ---- device and host buffers -------------------------------------------------------------------------------------------
static float *dev_alloc(size_t n_floats) {
  float *p = NULL;
  CK(cudaMalloc((void **)&p, (n_floats ? n_floats : 1) * sizeof(float)));
  return p;
}

static void dev_free(float **p) {
  if (p != NULL && *p != NULL) CK(cudaFree(*p));
  if (p != NULL) *p = NULL;
}

static float *host_alloc(size_t n_floats) {
  float *p = (float *)malloc((n_floats ? n_floats : 1) * sizeof(float));
  if (p == NULL) die("out of host memory (%zu floats)", n_floats);
  return p;
}

// Copy `n` floats back to a fresh host array, ordered on the context's stream (a non-blocking one, design §12: a plain
// cudaMemcpy on the null stream would race the kernels).
static float *dev_pull(JtCtx *c, const float *dev, size_t n) {
  float *h = host_alloc(n);
  CK(cudaMemcpyAsync(h, dev, n * sizeof(float), cudaMemcpyDeviceToHost, c->stream));
  CK(cudaStreamSynchronize(c->stream));
  return h;
}

// ---- the .jtb batch the dump runs on (design §9.1, §7) ------------------------------------------------------------------
// dump_ref.py's batch 0: the rows' real tokens packed back to back, each row's live option count and weight, and the
// targets padded with zeros past the live count (train.py's collate does the same, so ce_head sees the same array).
typedef struct {
  int rows;                       // B
  int n;                          // sum of the rows' lengths
  int row[JT_MAX_ROWS];           // the file's row indices of batch 0 (for the log line)
  int off[JT_MAX_ROWS + 1];       // prefix sums of the lengths
  int live[JT_MAX_ROWS];          // each row's live option count
  float wrow[JT_MAX_ROWS];        // each row's weight
  float wgroup;                   // train.py's gw: the batch's summed weight
  int n_opts;                     // the widest row's live count = the logits width
  int *ids;                       // host [n]
  float *targets;                 // host [rows][n_opts], zero past each row's live count
} DumpBatch;

static void dump_batch_free(DumpBatch *d) {
  free(d->ids);
  free(d->targets);
  d->ids = NULL;
  d->targets = NULL;
}

static void read_batch0(const char *path, DumpBatch *d) {
  Jtb *j = jtb_open(path);
  if (jtb_n_batches(j) < 1) die("%s: the file has no batches", path);
  uint32_t nb = 0;
  const uint32_t *rows = jtb_batch(j, 0, &nb);
  if (nb < 1) die("%s: batch 0 is empty", path);
  if ((int)nb > JT_MAX_ROWS) die("%s: batch 0 has %u rows, at most %d are supported", path, nb, JT_MAX_ROWS);

  memset(d, 0, sizeof *d);
  d->rows = (int)nb;
  int n = 0, max_opts = 0;
  double wsum = 0.0;
  for (int r = 0; r < d->rows; r++) {
    const JtbRow *row = jtb_row(j, rows[r]);
    if (row == NULL) die("%s: batch 0 names row %u, which the file does not have", path, rows[r]);
    if (row->len < 1) die("%s: row %u is empty", path, rows[r]);
    if (row->n_opts < 1 || row->n_opts > JTB_MAX_OPTS) die("%s: row %u has %u options, want 1..%u", path, rows[r],
                                                           row->n_opts, JTB_MAX_OPTS);
    d->row[r] = (int)rows[r];
    d->off[r] = n;
    n += (int)row->len;
    d->live[r] = (int)row->n_opts;
    d->wrow[r] = row->weight;
    wsum += (double)row->weight;
    if ((int)row->n_opts > max_opts) max_opts = (int)row->n_opts;
  }
  d->off[d->rows] = n;
  d->n = n;
  d->n_opts = max_opts;
  d->wgroup = (float)wsum; // train.py's `gw` (its loss is `(per * wt).sum() / gw`)
  if (!(d->wgroup > 0.0f)) die("%s: batch 0's row weights sum to %g; the loss divides by that", path, (double)d->wgroup);

  d->ids = (int *)malloc((size_t)n * sizeof *d->ids);
  if (d->ids == NULL) die("out of host memory (%d token ids)", n);
  d->targets = (float *)calloc((size_t)d->rows * (size_t)max_opts, sizeof *d->targets);
  if (d->targets == NULL) die("out of host memory (%d x %d targets)", d->rows, max_opts);
  for (int r = 0; r < d->rows; r++) {
    const JtbRow *row = jtb_row(j, rows[r]);
    const uint32_t *ids = jtb_ids(j, rows[r]);
    const float *t = jtb_targets(j, rows[r]);
    for (int k = 0; k < (int)row->len; k++) d->ids[d->off[r] + k] = (int)ids[k];
    memcpy(d->targets + (size_t)r * (size_t)max_opts, t, (size_t)row->n_opts * sizeof(float));
  }
  jtb_close(j);
}

// ---- the forward dump (plan M4.3's gate) -------------------------------------------------------------------------------
// Every key dump_ref.py writes for the forward, in its own order, from the device buffers to one savetensors file. The
// in.ids/in.off cast to float is the note in the file's header comment.
static void write_dump(JtCtx *c, const char *path, const DumpBatch *d, const float *saved, const float *post,
                       const float *final, const float *logits, float loss) {
  const size_t stride = (size_t)d->n * MT_HIDDEN;
  const int nk = 2 + 3 * MT_LAYERS + 3;

  const float **data = (const float **)calloc((size_t)nk, sizeof *data);
  const char **names = (const char **)calloc((size_t)nk, sizeof *names);
  int (*shapes)[ST_MAX_DIMS] = (int(*)[ST_MAX_DIMS])calloc((size_t)nk, sizeof *shapes);
  int *ndims = (int *)calloc((size_t)nk, sizeof *ndims);
  char (*namebuf)[32] = (char(*)[32])calloc((size_t)nk, 32);
  const float *off_f = NULL, *ids_f = NULL, *loss_f = NULL;
  float *saved_h = NULL, *post_h = NULL, *final_h = NULL, *logits_h = NULL;
  if (data == NULL || names == NULL || shapes == NULL || ndims == NULL || namebuf == NULL)
    die("out of host memory (the dump's key table, %d keys)", nk);

  saved_h = dev_pull(c, saved, (size_t)(MT_LAYERS + 1) * stride);
  post_h = dev_pull(c, post, (size_t)MT_LAYERS * stride);
  final_h = dev_pull(c, final, stride);
  logits_h = dev_pull(c, logits, (size_t)d->rows * (size_t)d->n_opts);

  // in.ids / in.off as F32: the writer is F32-only, and these values (ids < 248320, offsets < N) are exact.
  float *ids = host_alloc((size_t)d->n);
  float *off = host_alloc((size_t)d->rows + 1);
  for (int k = 0; k < d->n; k++) ids[k] = (float)d->ids[k];
  for (int r = 0; r <= d->rows; r++) off[r] = (float)d->off[r];
  float *loss1 = host_alloc(1);
  loss1[0] = loss;
  ids_f = ids;
  off_f = off;
  loss_f = loss1;

  int k = 0;
  snprintf(namebuf[k], 32, "in.ids");
  shapes[k][0] = d->n, ndims[k] = 1, data[k] = ids_f, k++;
  snprintf(namebuf[k], 32, "in.off");
  shapes[k][0] = d->rows + 1, ndims[k] = 1, data[k] = off_f, k++;
  for (int i = 0; i < MT_LAYERS; i++) {
    snprintf(namebuf[k], 32, "layer.%d.in", i);
    shapes[k][0] = d->n, shapes[k][1] = MT_HIDDEN, ndims[k] = 2, data[k] = saved_h + (size_t)i * stride, k++;
    snprintf(namebuf[k], 32, "layer.%d.post_mixer", i);
    shapes[k][0] = d->n, shapes[k][1] = MT_HIDDEN, ndims[k] = 2, data[k] = post_h + (size_t)i * stride, k++;
    snprintf(namebuf[k], 32, "layer.%d.out", i);
    shapes[k][0] = d->n, shapes[k][1] = MT_HIDDEN, ndims[k] = 2, data[k] = saved_h + (size_t)(i + 1) * stride, k++;
  }
  snprintf(namebuf[k], 32, "final");
  shapes[k][0] = d->n, shapes[k][1] = MT_HIDDEN, ndims[k] = 2, data[k] = final_h, k++;
  snprintf(namebuf[k], 32, "logits");
  shapes[k][0] = d->rows, shapes[k][1] = d->n_opts, ndims[k] = 2, data[k] = logits_h, k++;
  snprintf(namebuf[k], 32, "loss");
  ndims[k] = 0, data[k] = loss_f, k++;

  for (int i = 0; i < k; i++) names[i] = namebuf[i];
  st_write(path, k, names, (const int(*)[ST_MAX_DIMS])shapes, ndims, data);
  // the data section: n ids + (rows+1) offsets + 49 activation slabs + the logits + the scalar loss
  const size_t nfloat = (size_t)d->n + (size_t)d->rows + 1 + (size_t)(2 * MT_LAYERS + 1) * stride +
                        (size_t)d->rows * (size_t)d->n_opts + 1;
  printf("jefftrain: wrote %s: %d keys (2 in + %d layers x 3 + final, logits, loss), %.1f MiB of data, loss %.8f\n",
         path, k, MT_LAYERS, (double)nfloat * sizeof(float) / 1e6, (double)loss);
  fflush(stdout);
  free(loss1);
  free(off);
  free(ids);
  free(logits_h);
  free(final_h);
  free(post_h);
  free(saved_h);
  free(namebuf);
  free(ndims);
  free(shapes);
  free(names);
  free(data);
}

static void run_dump_forward(Model *m, Lora *l, const char *jtb_path, const char *out) {
  DumpBatch d;
  read_batch0(jtb_path, &d);
  int lmax = 0;
  for (int r = 0; r < d.rows; r++)
    if (d.off[r + 1] - d.off[r] > lmax) lmax = d.off[r + 1] - d.off[r];
  printf("jefftrain: dump-forward: %s batch 0 = rows [", jtb_path);
  for (int r = 0; r < d.rows; r++) printf("%s%d", r != 0 ? ", " : "", d.row[r]);
  printf("], lengths [");
  for (int r = 0; r < d.rows; r++) printf("%s%d", r != 0 ? ", " : "", d.off[r + 1] - d.off[r]);
  printf("], n %d tokens, %d option(s), gw %g\n", d.n, d.n_opts, (double)d.wgroup);
  fflush(stdout);

  JtBatch b;
  jt_batch_set(&b, d.rows, d.n, d.off);
  JtCtx *c = jt_ctx_alloc(m, l, d.n, d.rows, lmax);

  const size_t stride = (size_t)d.n * MT_HIDDEN;
  int *ids = NULL;
  CK(cudaMalloc((void **)&ids, (size_t)d.n * sizeof(int)));
  float *saved = dev_alloc((size_t)(MT_LAYERS + 1) * stride);
  float *post = dev_alloc((size_t)MT_LAYERS * stride);
  float *final = dev_alloc(stride);
  float *logits = dev_alloc((size_t)d.rows * (size_t)d.n_opts);
  float *targets = dev_alloc((size_t)d.rows * (size_t)d.n_opts);
  float *wrow = dev_alloc((size_t)d.rows);
  CK(cudaMemcpyAsync(ids, d.ids, (size_t)d.n * sizeof(int), cudaMemcpyHostToDevice, c->stream));
  CK(cudaMemcpyAsync(targets, d.targets, (size_t)d.rows * (size_t)d.n_opts * sizeof(float), cudaMemcpyHostToDevice,
                     c->stream));
  CK(cudaMemcpyAsync(wrow, d.wrow, (size_t)d.rows * sizeof(float), cudaMemcpyHostToDevice, c->stream));
  CK(cudaStreamSynchronize(c->stream));

  int rc = model_forward(c, ids, d.n, &b, saved, post);
  if (rc == JT_LAYER_NOT_IMPL) die("the forward is incomplete: a layer is still a stub (see the line above)");
  if (rc != JT_LAYER_OK) die("model_forward returned %d", rc);

  HeadBatch hb;
  hb.n_opts = d.n_opts;
  hb.live = d.live;
  hb.targets = targets;
  hb.wrow = wrow;
  hb.wgroup = d.wgroup;
  float loss = NAN;
  jt_saves_reset(c); // the sweep keeps nothing (JT_FWD_ONLY); the head's own work comes from the arena's work half
  rc = head_fwd(c, saved + (size_t)MT_LAYERS * stride, d.n, &b, &hb, final, logits, &loss, JT_FWD_ONLY);
  if (rc == JT_LAYER_NOT_IMPL) die("the head is still a stub (see the line above)");
  if (rc != JT_LAYER_OK) die("head_fwd returned %d", rc);

  write_dump(c, out, &d, saved, post, final, logits, loss);

  dev_free(&wrow);
  dev_free(&targets);
  dev_free(&logits);
  dev_free(&final);
  dev_free(&post);
  dev_free(&saved);
  if (ids != NULL) CK(cudaFree(ids));
  jt_ctx_free(c);
  jt_batch_free(&b);
  dump_batch_free(&d);
}

// ---- the backward dump (plan M5.1 item 4's gate, and the flag plan M5.4 needs) -------------------------------------------
// `--dump-backward OUT` runs design §6.2's sweep on the same batch and writes the same 403 keys py/dump_ref.py writes, so
// compare_layers.py compares the whole file (--atol-rel 1e-3 --grad-rel 5e-3) rather than only the grad.* subset:
//
//   model_forward(ONLY)                  all 24 layers, keeping each layer's input only
//   head_fwd(SAVE), head_bwd(dres)        grad.readout, grad.final, dres = dL/d(layer 23's output)
//   model_backward                       for i = 23..0: the recompute of layer i from saved_in[i], then its backward,
//                                        dres in place (dL/d(layer i's out) -> dL/d(layer i's in))
//
// The 300 LoRA factors and the readout are slices of one flat gradient buffer (src/lora.h), so a single pull of it gives
// every `grad.{stem}.lora_A/B` and `grad.readout`. Every gradient is a contiguous device buffer here, so the file is
// written from host copies the way --dump-forward's is (the writer takes host pointers, src/st.h).
#define JT_DUMP_NAME_MAX 96

// The keys are collected in one table because st_write writes one file: names, shapes and data pointers only.
typedef struct {
  int n, cap;
  const float **data;
  const char **names;
  int (*shapes)[ST_MAX_DIMS];
  int *ndims;
  char (*namebuf)[JT_DUMP_NAME_MAX];
} KeyTable;

static void kt_init(KeyTable *t, int cap) {
  memset(t, 0, sizeof *t);
  t->cap = cap;
  t->data = (const float **)calloc((size_t)cap, sizeof *t->data);
  t->names = (const char **)calloc((size_t)cap, sizeof *t->names);
  t->shapes = (int(*)[ST_MAX_DIMS])calloc((size_t)cap, sizeof *t->shapes);
  t->ndims = (int *)calloc((size_t)cap, sizeof *t->ndims);
  t->namebuf = (char(*)[JT_DUMP_NAME_MAX])calloc((size_t)cap, JT_DUMP_NAME_MAX);
  if (t->data == NULL || t->names == NULL || t->shapes == NULL || t->ndims == NULL || t->namebuf == NULL)
    die("out of host memory (the dump's key table, %d keys)", cap);
}

static void kt_free(KeyTable *t) {
  free(t->namebuf);
  free(t->ndims);
  free(t->shapes);
  free(t->names);
  free(t->data);
  memset(t, 0, sizeof *t);
}

// One key: ndims is 0 (a scalar), 1 (d0) or 2 (d0 x d1). `data` must stay alive until the write.
static void kt_add(KeyTable *t, const char *name, int ndims, int d0, int d1, const float *data) {
  if (t->n >= t->cap) die("the dump's key table is full (%d keys)", t->cap);
  char *dst = t->namebuf[t->n];
  if (snprintf(dst, JT_DUMP_NAME_MAX, "%s", name) >= JT_DUMP_NAME_MAX)
    die("the dump key '%s' does not fit in %d bytes", name, JT_DUMP_NAME_MAX);
  t->names[t->n] = dst;
  t->shapes[t->n][0] = d0;
  t->shapes[t->n][1] = d1;
  t->ndims[t->n] = ndims;
  t->data[t->n] = data;
  t->n++;
}

// The host copies a --dump-backward holds: the forward keys (the same buffers --dump-forward writes), the layer-output
// gradients, grad.final and one pull of the flat parameter-gradient buffer.
typedef struct {
  float *saved;      // (MT_LAYERS + 1) slabs: layer.{i}.in and layer.{i}.out
  float *post;       // MT_LAYERS slabs: layer.{i}.post_mixer, from the recompute
  float *final;      // one slab
  float *logits;     // [rows][n_opts]
  float *loss;       // one scalar
  float *grad_layer; // MT_LAYERS slabs: grad.layer.{i}.out (the value on entry to layer i's backward)
  float *grad_final; // one slab
  float *flat;       // ctx->lora->grad, n_param floats (the LoRA factors and the readout are slices of it)
} GradDump;

static void grad_dump_free(GradDump *g) {
  free(g->flat);
  free(g->grad_final);
  free(g->grad_layer);
  free(g->loss);
  free(g->logits);
  free(g->final);
  free(g->post);
  free(g->saved);
  memset(g, 0, sizeof *g);
}

// dump_ref.py's keys, its order (in.* first, then the layers, then final/logits/loss, then the gradients), from the host
// copies above. st_write sorts the keys itself, so the order is for a human reading the table.
static void write_dump_backward(JtCtx *c, const char *path, const DumpBatch *d, const GradDump *g) {
  const size_t stride = (size_t)d->n * MT_HIDDEN;
  KeyTable t;
  kt_init(&t, 2 + 3 * MT_LAYERS + 3 + MT_LAYERS + 1 + 2 * MT_LORA_PAIRS + 1);

  float *ids = host_alloc((size_t)d->n);
  float *off = host_alloc((size_t)d->rows + 1);
  for (int k = 0; k < d->n; k++) ids[k] = (float)d->ids[k];
  for (int r = 0; r <= d->rows; r++) off[r] = (float)d->off[r];
  kt_add(&t, "in.ids", 1, d->n, 0, ids);
  kt_add(&t, "in.off", 1, d->rows + 1, 0, off);

  char name[JT_DUMP_NAME_MAX];
  for (int i = 0; i < MT_LAYERS; i++) {
    snprintf(name, sizeof name, "layer.%d.in", i);
    kt_add(&t, name, 2, d->n, MT_HIDDEN, g->saved + (size_t)i * stride);
    snprintf(name, sizeof name, "layer.%d.post_mixer", i);
    kt_add(&t, name, 2, d->n, MT_HIDDEN, g->post + (size_t)i * stride);
    snprintf(name, sizeof name, "layer.%d.out", i);
    kt_add(&t, name, 2, d->n, MT_HIDDEN, g->saved + (size_t)(i + 1) * stride);
  }
  kt_add(&t, "final", 2, d->n, MT_HIDDEN, g->final);
  kt_add(&t, "logits", 2, d->rows, d->n_opts, g->logits);
  kt_add(&t, "loss", 0, 0, 0, g->loss);

  for (int i = 0; i < MT_LAYERS; i++) { // dump_ref's keys for the gradient at each layer's output, 0..23
    snprintf(name, sizeof name, "grad.layer.%d.out", i);
    kt_add(&t, name, 2, d->n, MT_HIDDEN, g->grad_layer + (size_t)i * stride);
  }
  kt_add(&t, "grad.final", 2, d->n, MT_HIDDEN, g->grad_final);
  for (int k = 0; k < MT_LORA_PAIRS; k++) { // 150 adapted projections: A [rank,in] then B [out,rank]
    const LoraEntry *e = &c->lora->ent[k];
    snprintf(name, sizeof name, "grad.%s.lora_A", e->stem);
    kt_add(&t, name, 2, e->rank, e->in, g->flat + e->off);
    snprintf(name, sizeof name, "grad.%s.lora_B", e->stem);
    kt_add(&t, name, 2, e->out, e->rank, g->flat + e->b_off);
  }
  const LoraEntry *ro = &c->lora->ent[LORA_READOUT_ENTRY];
  kt_add(&t, "grad.readout", 2, ro->out, ro->in, g->flat + ro->off);

  st_write(path, t.n, t.names, (const int(*)[ST_MAX_DIMS])t.shapes, t.ndims, t.data);
  const size_t nfloat = (size_t)d->n + (size_t)d->rows + 1 + (size_t)(3 * MT_LAYERS + MT_LAYERS + 2) * stride +
                        (size_t)d->rows * (size_t)d->n_opts + 1 + c->lora->n_param;
  printf("jefftrain: wrote %s: %d keys (the %d forward keys + %d layer-gradient keys + grad.final + %d LoRA grads + "
         "grad.readout), %.1f MiB of data, loss %.8f\n",
         path, t.n, 2 + 3 * MT_LAYERS + 3, MT_LAYERS, 2 * MT_LORA_PAIRS, (double)nfloat * sizeof(float) / 1e6,
         (double)g->loss[0]);
  fflush(stdout);
  free(off);
  free(ids);
  kt_free(&t);
}

static void run_dump_backward(Model *m, Lora *l, const char *jtb_path, const char *out) {
  DumpBatch d;
  read_batch0(jtb_path, &d);
  int lmax = 0;
  for (int r = 0; r < d.rows; r++)
    if (d.off[r + 1] - d.off[r] > lmax) lmax = d.off[r + 1] - d.off[r];
  printf("jefftrain: dump-backward: %s batch 0 = rows [", jtb_path);
  for (int r = 0; r < d.rows; r++) printf("%s%d", r != 0 ? ", " : "", d.row[r]);
  printf("], lengths [");
  for (int r = 0; r < d.rows; r++) printf("%s%d", r != 0 ? ", " : "", d.off[r + 1] - d.off[r]);
  printf("], n %d tokens, %d option(s), gw %g\n", d.n, d.n_opts, (double)d.wgroup);
  fflush(stdout);

  JtBatch b;
  jt_batch_set(&b, d.rows, d.n, d.off);
  JtCtx *c = jt_ctx_alloc(m, l, d.n, d.rows, lmax);

  const size_t stride = (size_t)d.n * MT_HIDDEN;
  int *ids = NULL;
  CK(cudaMalloc((void **)&ids, (size_t)d.n * sizeof(int)));
  float *saved = dev_alloc((size_t)(MT_LAYERS + 1) * stride);
  float *post = dev_alloc((size_t)MT_LAYERS * stride);
  float *final = dev_alloc(stride);
  float *logits = dev_alloc((size_t)d.rows * (size_t)d.n_opts);
  float *targets = dev_alloc((size_t)d.rows * (size_t)d.n_opts);
  float *wrow = dev_alloc((size_t)d.rows);
  float *dres = dev_alloc(stride);                           // the sweep's one gradient buffer (design §6.2)
  float *grad_layer = dev_alloc((size_t)MT_LAYERS * stride); // grad.layer.{i}.out, copied out on entry to layer i's bwd
  CK(cudaMemcpyAsync(ids, d.ids, (size_t)d.n * sizeof(int), cudaMemcpyHostToDevice, c->stream));
  CK(cudaMemcpyAsync(targets, d.targets, (size_t)d.rows * (size_t)d.n_opts * sizeof(float), cudaMemcpyHostToDevice,
                     c->stream));
  CK(cudaMemcpyAsync(wrow, d.wrow, (size_t)d.rows * sizeof(float), cudaMemcpyHostToDevice, c->stream));
  CK(cudaStreamSynchronize(c->stream));

  int rc = model_forward(c, ids, d.n, &b, saved, post); // design §6.2: only each layer's input is kept
  if (rc == JT_LAYER_NOT_IMPL) die("the forward is incomplete: a layer is still a stub (see the line above)");
  if (rc != JT_LAYER_OK) die("model_forward returned %d", rc);

  HeadBatch hb;
  hb.n_opts = d.n_opts;
  hb.live = d.live;
  hb.targets = targets;
  hb.wrow = wrow;
  hb.wgroup = d.wgroup;
  float loss = NAN;
  jt_saves_reset(c); // the head's saves start at the arena's base (src/head.c); the sweep kept nothing
  rc = head_fwd(c, saved + (size_t)MT_LAYERS * stride, d.n, &b, &hb, final, logits, &loss, JT_FWD_SAVE);
  if (rc == JT_LAYER_NOT_IMPL) die("the head is still a stub (see the line above)");
  if (rc != JT_LAYER_OK) die("head_fwd returned %d", rc);

  lora_zero_grad(l, c->stream); // dR/dA/dB accumulate into the flat buffer (src/linear.h, src/head.c)
  const size_t hmark = jt_work_mark(c);
  rc = head_bwd(c, saved + (size_t)MT_LAYERS * stride, d.n, &b, dres);
  if (rc != JT_LAYER_OK) die("head_bwd returned %d", rc);
  // grad.final is head_bwd's dh, its first (and only) work-half allocation, so it sits at arena + mark - n * MT_HIDDEN
  // (src/head.c's header comment). Pulled here: the layer loop takes those very words for its own scratch.
  float *grad_final = dev_pull(c, c->arena + hmark - (size_t)d.n * MT_HIDDEN, stride);

  rc = model_backward(c, saved, d.n, &b, post, grad_layer, dres); // design §6.2's recompute loop, 23 down to 0
  if (rc != JT_LAYER_OK) die("model_backward returned %d", rc);
  printf("jefftrain: backward: %d layers recomputed from their saved inputs and differentiated\n", MT_LAYERS);
  fflush(stdout);

  GradDump g;
  memset(&g, 0, sizeof g);
  g.saved = dev_pull(c, saved, (size_t)(MT_LAYERS + 1) * stride);
  g.post = dev_pull(c, post, (size_t)MT_LAYERS * stride); // the recompute's post_mixer (what the backwards used)
  g.final = dev_pull(c, final, stride);
  g.logits = dev_pull(c, logits, (size_t)d.rows * (size_t)d.n_opts);
  g.loss = host_alloc(1);
  g.loss[0] = loss;
  g.grad_layer = dev_pull(c, grad_layer, (size_t)MT_LAYERS * stride);
  g.grad_final = grad_final;
  g.flat = dev_pull(c, l->grad, l->n_param); // the 300 LoRA gradients and the readout are slices of this
  write_dump_backward(c, out, &d, &g);

  grad_dump_free(&g);
  dev_free(&grad_layer);
  dev_free(&dres);
  dev_free(&wrow);
  dev_free(&targets);
  dev_free(&logits);
  dev_free(&final);
  dev_free(&post);
  dev_free(&saved);
  if (ids != NULL) CK(cudaFree(ids));
  jt_ctx_free(c);
  jt_batch_free(&b);
  dump_batch_free(&d);
}

int main(int argc, char **argv) {
  if (argc == 1) {
    usage(stdout);
    return 0;
  }
  Args a;
  parse_args(argc, argv, &a);
  if (a.model == NULL) die("--model is required (--help for the usage)");
  if (a.dump_fwd != NULL && a.dump_bwd != NULL) die("--dump-forward and --dump-backward are separate runs; pass one");
  if ((a.dump_fwd != NULL || a.dump_bwd != NULL) && a.train == NULL)
    die("--dump-forward/--dump-backward run a batch, so they need --train FILE.jtb");

  gemm_init(); // prints "gemm: exact fp32" or "gemm: tf32" once, before anything allocates (design §5)
  Lora *lora = lora_alloc();
  Model *m = model_load(a.model, lora);
  if (a.adapter != NULL) {
    lora_load_adapter(lora, a.adapter);
    printf("jefftrain: adapter %s loaded over the checkpoint's readout\n", a.adapter);
    fflush(stdout);
  }

  if (a.dump_fwd != NULL) run_dump_forward(m, lora, a.train, a.dump_fwd);
  if (a.dump_bwd != NULL) run_dump_backward(m, lora, a.train, a.dump_bwd);

  model_free(m);
  lora_free(lora);
  if (a.dump_fwd != NULL || a.dump_bwd != NULL) return 0;
  if (a.train != NULL) {
    fprintf(stderr, "jefftrain: %s is loaded; the training step (Adam, the schedule) is plan M5.2 and is not "
                    "in this build\n", a.train);
    return 2;
  }
  printf("jefftrain: %s loaded; nothing to do (--dump-forward/--dump-backward OUT need --train FILE.jtb)\n", a.model);
  return 0;
}
