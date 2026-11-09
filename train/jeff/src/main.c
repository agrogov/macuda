// src/main.c — jefftrain's driver: the flags, the model load, the adapter, and the whole-model forward dump
// (plan M4.3, design §3.4, §7, §10.2, §13).
//
// What exists here now is the M4.3 piece of the trainer: `--dump-forward OUT` runs one packed batch through the embedding
// gather, all 24 decoder layers and the decision head, and writes OUT in py/dump_ref.py's format so that
// py/compare_layers.py can compare it with `ref.safetensors` key by key (plan M4.3's gate; the card checkpoint M4.4 uses
// the same two files). `--model` and `--init-adapter` are plan M4.2's/M4.3's flags; `--train` is train.py's name for the
// .jtb split the run consumes (plan M5.2), and the forward dump reads its batch 0 — the same batch dump_ref.py dumps, so
// the two files describe the same tokens.
//
// The training step itself (backward, Adam, the schedule, the log lines) is plan M5.2 and is NOT here: --train without an
// action flag loads everything and then says so with exit 2 (a run that did no work must not look like a run).
//
// The keys the dump writes are dump_ref.py's *forward* keys and nothing else:
//
//   in.ids [N]  in.off [B+1]                        the packed batch (see the note below: F32, not I32)
//   layer.{i}.in [N,1024]  layer.{i}.post_mixer [N,1024]  layer.{i}.out [N,1024]     i = 0..23
//   final [N,1024]  logits [B,n_opts]  loss []       the head (design §4.1)
//
// dump_ref.py also writes `grad.*`, which a forward cannot produce (plan M5.1's --dump-backward is where those appear);
// so compare_layers.py, which counts a reference key missing from the other file as a failure, cannot exit 0 against the
// full ref for this flag alone. The forward keys are the ones plan M4.3 gates, and they are all here.
//
// Two notes on being faithful to dump_ref.py:
//   * in.ids / in.off are integers there and F32 here, because src/st.h's writer is F32-only (design §9.2). Every id is
//     below MT_VOCAB = 248320 and every offset below the batch's token count, all of which are exact in fp32, and
//     compare_layers.py compares an integer key with a float one by value — so the check is still exact.
//   * the batch is the fixture's batch 0 (rows packed back to back, design §7), its targets are read from the same .jtb
//     rows, and the loss is train.py's `(per * wt).sum() / gw` with `gw` the batch's own summed row weight — the values
//     dump_ref.py feeds its own copy of train.py's loop.
//
// Memory: the dump sizes the activation arena for the batch it runs (cap_tokens = N, cap_rows = B, lmax = the longest
// row), and holds 25 activation slabs [N,1024] plus the 24 post_mixer slabs on the device, because those are exactly the
// keys it writes. That is a dump-time cost a training step does not pay (design §6.2).
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
  const char *model;   // --model DIR: the checkpoint directory (or a model.safetensors file)
  const char *adapter; // --init-adapter DIR: an adapter to load over the checkpoint's readout (may be absent)
  const char *train;   // --train FILE.jtb: train.py's flag for the split this run consumes
  const char *dump_fwd; // --dump-forward OUT: write the whole forward there (plan M4.3)
} Args;

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
          "  --model DIR         the jeff-base checkpoint directory (model.safetensors + readout.safetensors)\n"
          "  --init-adapter DIR  load a Python-written adapter over the checkpoint's readout (train.py's flag)\n"
          "  --train FILE        the .jtb split (train.py's flag, plan M5.2); --dump-forward reads its batch 0\n"
          "  --dump-forward OUT  run the embedding gather, 24 layers and the head on that batch and write OUT in\n"
          "                      py/dump_ref.py's format (plan M4.3; reads it with py/compare_layers.py)\n"
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

int main(int argc, char **argv) {
  if (argc == 1) {
    usage(stdout);
    return 0;
  }
  Args a;
  parse_args(argc, argv, &a);
  if (a.model == NULL) die("--model is required (--help for the usage)");
  if (a.dump_fwd != NULL && a.train == NULL) die("--dump-forward runs a batch, so it needs --train FILE.jtb");

  gemm_init(); // prints "gemm: exact fp32" or "gemm: tf32" once, before anything allocates (design §5)
  Lora *lora = lora_alloc();
  Model *m = model_load(a.model, lora);
  if (a.adapter != NULL) {
    lora_load_adapter(lora, a.adapter);
    printf("jefftrain: adapter %s loaded over the checkpoint's readout\n", a.adapter);
    fflush(stdout);
  }

  if (a.dump_fwd != NULL) run_dump_forward(m, lora, a.train, a.dump_fwd);

  model_free(m);
  lora_free(lora);
  if (a.dump_fwd != NULL) return 0;
  if (a.train != NULL) {
    fprintf(stderr, "jefftrain: %s is loaded; the training step (backward, Adam, the schedule) is plan M5.2 and is not "
                    "in this build\n", a.train);
    return 2;
  }
  printf("jefftrain: %s loaded; nothing to do (--dump-forward OUT needs --train FILE.jtb)\n", a.model);
  return 0;
}
