// test/test_layer.c — one decoder layer (or the decision head) against dump_ref.py's reference, one tensor at a time
// (plan M4.3, design §10.2).
//
//   test_layer --ref R --model M --adapter D --layer i [--part mlp|mixer|all]
//   test_layer --ref R --model M --adapter D --head [--jtb fx.jtb]
//   test_layer --model M                      (load only: the shape checks and the model: line)
//   test_layer                                (nothing: exit 0 — what `make test-null` runs)
//
// The comparison is dump_ref.safetensors' own split of a layer (notes/J14.md, design §10.2):
//
//   --part mixer   layer.{i}.in         -> layer.{i}.post_mixer    the mixer + input norm + residual
//   --part mlp     layer.{i}.post_mixer -> layer.{i}.out           the post-attention norm + MLP + residual
//   --part all     layer.{i}.in         -> layer.{i}.out           the two in order (layer_fwd)
//   --head         layer.23.out         -> final, logits, loss     head_fwd
//
// The gate is design §10.2's activation gate, `max|a-b| / max(|b|) < 1e-3` with b the reference — the same number
// py/compare_layers.py uses, and not loosened here. It is only meaningful in an exact build (design §10.2: jefftrain
// should be built with JT_EXACT=1 for it); this program does not set the mode, gemm_init() reports the one it got.
//
// A stub (plan J18 wrote the four layer files as stubs) prints "not implemented" and this program exits 0: a missing
// piece is not a failure of the pieces that exist, exactly like test/harness.h's JT_NOT_IMPL. Any other non-zero return,
// a failed comparison and a missing ref key are failures (exit 1); bad arguments are exit 2.
//
// --jtb is optional and only the head needs it: ref.safetensors has the logits and the loss but not the targets they were
// computed from, so the loss can only be re-derived with the fixture's batch 0. Without it the head compares `final` and
// `logits` and says the loss was skipped. With it, the fixture's ids are also checked against `in.ids`, which catches
// passing the wrong fixture even for a mixer/mlp run.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jtb.h"
#include "layer.h"
#include "model.h"
#include "st.h"
#include "util.h"

// design §10.2's gate for activations, unchanged. The three exit codes are test/harness.h's (0 passed / nothing to do,
// 1 failed, 2 could not run).
#define TOL 1e-3

enum { PART_ALL = 0, PART_MIXER = 1, PART_MLP = 2 };

typedef struct {
  const char *model;   // --model: the checkpoint directory (or its model.safetensors)
  const char *adapter; // --adapter: an adapter directory, loaded over the checkpoint's readout (may be absent)
  const char *ref;     // --ref: dump_ref.py's ref.safetensors
  const char *jtb;     // --jtb: the fixture ref was made from (optional; the head needs it for the loss)
  int layer;           // --layer: 0..23, -1 when not given
  int part;            // --part
  int head;            // --head
} Args;

static void die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void die(const char *fmt, ...) {
  va_list ap;
  fflush(stdout);
  fputs("test_layer: ", stderr);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(1);
}

static void usage(FILE *f) {
  fprintf(f,
          "usage: test_layer --model DIR [--adapter DIR] [--ref FILE] [--layer i [--part mlp|mixer|all] | --head]\n"
          "                  [--jtb FILE]\n"
          "  --ref R --model M --adapter D --layer i   compare layer i with dump_ref.py's reference\n"
          "  --ref R --model M --adapter D --head      compare the final norm, the logits and (with --jtb) the loss\n"
          "  --model M                                 load the checkpoint only (the st_get shape checks)\n"
          "  (no arguments)                            exit 0\n"
          "  the gate is design §10.2's max|a-b| / max|b| < %.0e; build jefftrain with JT_EXACT=1 for it\n",
          (double)TOL);
}

// ---- small device helpers ---------------------------------------------------------------------------------------------
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

// A [rows][cols] F32 reference tensor, uploaded on the context's stream: a plain cudaMemcpy would use the null stream,
// which a non-blocking stream is not ordered against.
static float *ref_upload(JtCtx *c, const St *ref, const char *key, int rows, int cols) {
  const int shape[2] = {rows, cols};
  const float *src = (const float *)st_getv(ref, key, ST_DTYPE_F32, 2, shape);
  float *dst = dev_alloc((size_t)rows * (size_t)cols);
  CK(cudaMemcpyAsync(dst, src, (size_t)rows * (size_t)cols * sizeof(float), cudaMemcpyHostToDevice, c->stream));
  CK(cudaStreamSynchronize(c->stream));
  return dst;
}

// design §10.2's formula, printed the way py/compare_layers.py prints it: key, rel, ok/FAIL. The readback is ordered on
// ctx->stream and synchronised there: that stream is non-blocking (design §12's cudaStreamCreateWithFlags), so a copy on
// the null stream would not wait for the kernels the layer just launched.
static int check_key(JtCtx *c, const St *ref, const char *key, const float *dev, int rows, int cols) {
  const int shape[2] = {rows, cols};
  const float *want = (const float *)st_getv(ref, key, ST_DTYPE_F32, 2, shape);
  const size_t n = (size_t)rows * (size_t)cols;
  float *got = host_alloc(n);
  CK(cudaMemcpyAsync(got, dev, n * sizeof(float), cudaMemcpyDeviceToHost, c->stream));
  CK(cudaStreamSynchronize(c->stream));

  double maxdiff = 0.0, maxabs = 0.0;
  size_t nonfinite = 0;
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(got[i])) {
      nonfinite++;
      continue;
    }
    const double d = fabs((double)got[i] - (double)want[i]);
    const double a = fabs((double)want[i]);
    if (d > maxdiff) maxdiff = d;
    if (a > maxabs) maxabs = a;
  }
  const double rel = maxabs > 0.0 ? maxdiff / maxabs : maxdiff;
  const int ok = nonfinite == 0 && rel < TOL;
  if (nonfinite != 0)
    printf("%-22s %zu of %zu value(s) not finite  FAIL\n", key, nonfinite, n);
  else
    printf("%-22s rel %.3e  %s (want < %.0e, %zu values)\n", key, rel, ok ? "ok" : "FAIL", (double)TOL, n);
  free(got);
  return ok;
}

// A one-line summary of a scalar (the loss) against the reference's scalar.
static int check_scalar(const St *ref, const char *key, double got) {
  const float *want = (const float *)st_getv(ref, key, ST_DTYPE_F32, 0, NULL);
  const double w = (double)*want;
  const double rel = fabs(w) > 0.0 ? fabs(got - w) / fabs(w) : fabs(got - w);
  const int ok = isfinite(got) && rel < TOL;
  printf("%-22s rel %.3e  %s (got %.9g, ref %.9g)\n", key, rel, ok ? "ok" : "FAIL", got, w);
  return ok;
}

// ---- the reference's batch geometry -----------------------------------------------------------------------------------
static const int *ref_offsets(const St *ref, int *rows) {
  const StTensor *t = st_find(ref, "in.off");
  if (t == NULL || t->dtype != ST_DTYPE_I32 || t->ndim != 1 || t->shape[0] < 2)
    die("the reference has no usable 'in.off' (want I32 [B+1])");
  *rows = t->shape[0] - 1;
  const int shape[1] = {t->shape[0]};
  return (const int *)st_getv(ref, "in.off", ST_DTYPE_I32, 1, shape);
}

static const int *ref_ids(const St *ref, int *n) {
  const StTensor *t = st_find(ref, "in.ids");
  if (t == NULL || t->dtype != ST_DTYPE_I32 || t->ndim != 1) die("the reference has no usable 'in.ids' (want I32 [N])");
  *n = t->shape[0];
  const int shape[1] = {t->shape[0]};
  return (const int *)st_getv(ref, "in.ids", ST_DTYPE_I32, 1, shape);
}

// ---- the fixture's batch 0, when --jtb is given ------------------------------------------------------------------------
// ref.safetensors carries the logits and the loss but not the targets, so the head's loss can only be re-derived from the
// fixture the dump was made with. This also checks the packed ids against `in.ids`, which catches the wrong fixture.
typedef struct {
  int *live;      // host [B]
  float *targets; // device [B][n_opts]
  float *wrow;    // device [B]
  float wgroup;
  int n_opts;
} Fixture;

static void load_fixture(const char *path, const JtBatch *b, const int *ids, cudaStream_t s, Fixture *fx) {
  Jtb *j = jtb_open(path);
  uint32_t nb = 0;
  const uint32_t *rows = jtb_batch(j, 0, &nb);
  if ((int)nb != b->rows) die("%s: batch 0 has %u rows, the reference has %d", path, nb, b->rows);

  int max_opts = 0;
  double wsum = 0.0;
  fx->live = (int *)calloc((size_t)b->rows, sizeof *fx->live);
  float *targ = host_alloc((size_t)b->rows * JTB_MAX_OPTS);
  float *wrow = host_alloc((size_t)b->rows);
  if (fx->live == NULL) die("out of host memory (live)");
  for (int r = 0; r < b->rows; r++) {
    const JtbRow *row = jtb_row(j, rows[r]);
    if (row == NULL) die("%s: batch 0 has no row %u", path, rows[r]);
    if ((int)row->len != b->off[r + 1] - b->off[r]) die("%s: row %d is %u tokens, the reference has %d", path, r, row->len,
                                                       b->off[r + 1] - b->off[r]);
    if ((int)row->n_opts > max_opts) max_opts = (int)row->n_opts;
    fx->live[r] = (int)row->n_opts;
    wrow[r] = row->weight;
    wsum += (double)row->weight;
    if (row->n_opts > JTB_MAX_OPTS) die("%s: row %d has %u options", path, r, row->n_opts);
    const float *t = jtb_targets(j, rows[r]);
    memcpy(targ + (size_t)r * JTB_MAX_OPTS, t, (size_t)row->n_opts * sizeof(float));
    const uint32_t *rids = jtb_ids(j, rows[r]);
    for (int t2 = 0; t2 < (int)row->len; t2++)
      if ((int)rids[t2] != ids[b->off[r] + t2]) die("%s: row %d token %d is id %u, the reference has %d", path, r, t2,
                                                    rids[t2], ids[b->off[r] + t2]);
  }
  // train.py slices the logits to the widest row's live count, so that is the width of the targets too.
  float *compact = host_alloc((size_t)b->rows * (size_t)max_opts);
  for (int r = 0; r < b->rows; r++)
    for (int o = 0; o < max_opts; o++) compact[(size_t)r * max_opts + o] = o < fx->live[r] ? targ[(size_t)r * JTB_MAX_OPTS + o] : 0.0f;
  fx->n_opts = max_opts;
  fx->wgroup = (float)wsum; // gw: a test's batch is its own accumulation group (train.py:325)
  fx->targets = dev_alloc((size_t)b->rows * (size_t)max_opts);
  fx->wrow = dev_alloc((size_t)b->rows);
  CK(cudaMemcpyAsync(fx->targets, compact, (size_t)b->rows * (size_t)max_opts * sizeof(float), cudaMemcpyHostToDevice, s));
  CK(cudaMemcpyAsync(fx->wrow, wrow, (size_t)b->rows * sizeof(float), cudaMemcpyHostToDevice, s));
  CK(cudaStreamSynchronize(s));
  printf("test_layer: fixture %s: batch 0 = rows [", path);
  for (int r = 0; r < b->rows; r++) printf("%s%u", r != 0 ? ", " : "", rows[r]);
  printf("], lengths [");
  for (int r = 0; r < b->rows; r++) printf("%s%d", r != 0 ? ", " : "", b->off[r + 1] - b->off[r]);
  printf("], ids match in.ids, %d option(s), gw %g\n", max_opts, (double)fx->wgroup);
  fflush(stdout);
  free(compact);
  free(targ);
  free(wrow);
  jtb_close(j);
}

static void fixture_free(Fixture *fx) {
  free(fx->live);
  dev_free(&fx->targets);
  dev_free(&fx->wrow);
}

// ---- the three layer parts ----------------------------------------------------------------------------------------------
static const char *part_name(int part) { return part == PART_MIXER ? "mixer" : (part == PART_MLP ? "mlp" : "all"); }

static int run_layer(JtCtx *c, const St *ref, int i, int part, const JtBatch *b) {
  const int n = b->n;
  const MtLayer *ml = mt_layer(i);
  if (ml == NULL) die("layer %d, want 0..%d", i, MT_LAYERS - 1);
  char kin[64], kout[64];
  snprintf(kin, sizeof kin, part == PART_MLP ? "layer.%d.post_mixer" : "layer.%d.in", i);
  snprintf(kout, sizeof kout, part == PART_MIXER ? "layer.%d.post_mixer" : "layer.%d.out", i);
  printf("test_layer: layer %d part %s (%s): %s -> %s\n", i, part_name(part), ml->prefix, kin, kout);
  fflush(stdout); // the layer's own messages go to stderr; keep the log in the order things happened

  float *x_in = ref_upload(c, ref, kin, n, MT_HIDDEN);
  float *x_out = dev_alloc((size_t)n * MT_HIDDEN);
  float *post = dev_alloc((size_t)n * MT_HIDDEN);
  jt_saves_reset(c);
  const LayerWeights *lw = &c->model->layers[i];
  int rc;
  if (part == PART_MLP)
    rc = layer_mlp_fwd(c, lw, x_in, n, b, x_out, JT_FWD_SAVE);
  else if (part == PART_MIXER)
    rc = ml->full_attention ? layer_attn_mixer_fwd(c, lw, x_in, n, b, x_out, JT_FWD_SAVE)
                            : layer_gdn_mixer_fwd(c, lw, x_in, n, b, x_out, JT_FWD_SAVE);
  else
    rc = layer_fwd(c, i, x_in, n, b, x_out, post, JT_FWD_SAVE); // the layer's own split: mixer, then MLP

  int ret = 1;
  if (rc == JT_LAYER_NOT_IMPL) {
    printf("test_layer: layer %d %s: not implemented (stub)\n", i, part_name(part));
    ret = 0; // a missing piece is not a failure of the parts that exist (test/harness.h's JT_NOT_IMPL)
  } else if (rc != JT_LAYER_OK) {
    printf("test_layer: layer %d %s: FAIL (the layer returned %d)\n", i, part_name(part), rc);
  } else {
    ret = check_key(c, ref, kout, x_out, n, MT_HIDDEN) ? 0 : 1;
  }
  dev_free(&post);
  dev_free(&x_out);
  dev_free(&x_in);
  return ret;
}

// The head's comparison: final and logits always, the loss when the fixture gave targets.
static int run_head(JtCtx *c, const St *ref, const Fixture *fx, const JtBatch *b) {
  const int n = b->n;
  const float *rlogits = NULL;
  int n_opts = 0;
  {
    const StTensor *t = st_find(ref, "logits");
    if (t == NULL || t->dtype != ST_DTYPE_F32 || t->ndim != 2) die("the reference has no usable 'logits' (want F32 [B,n])");
    if (t->shape[0] != b->rows) die("the reference's logits are [%d,%d] but the batch has %d rows", t->shape[0], t->shape[1],
                                    b->rows);
    n_opts = t->shape[1];
    const int shape[2] = {b->rows, n_opts};
    rlogits = (const float *)st_getv(ref, "logits", ST_DTYPE_F32, 2, shape);
  }
  // 1e-9 is train.py's masked_fill value (option_logits): the rows' dead options are exactly that, so the live count of
  // each row is what is left (notes/J14.md's [2,5] logits with row 0 having 3 options).
  int *live = (int *)calloc((size_t)b->rows, sizeof *live);
  if (live == NULL) die("out of host memory (live)");
  for (int r = 0; r < b->rows; r++) {
    int k = 0;
    for (int o = 0; o < n_opts; o++)
      if (rlogits[(size_t)r * n_opts + o] > -1e8f) k++;
    live[r] = k;
  }
  if (fx != NULL && fx->n_opts != n_opts) die("the fixture's batch 0 has %d options, the reference's logits are %d wide",
                                              fx->n_opts, n_opts);
  if (fx != NULL)
    for (int r = 0; r < b->rows; r++)
      if (fx->live[r] != live[r]) die("fixture row %d has %d options, the reference's row has %d", r, fx->live[r], live[r]);

  HeadBatch hb;
  hb.n_opts = n_opts;
  hb.live = live;
  hb.targets = fx != NULL ? fx->targets : NULL;
  hb.wrow = fx != NULL ? fx->wrow : NULL;
  hb.wgroup = fx != NULL ? fx->wgroup : 1.0f;

  printf("test_layer: head: layer.23.out -> final, logits%s\n", fx != NULL ? ", loss" : "");
  fflush(stdout);
  float *x = ref_upload(c, ref, "layer.23.out", n, MT_HIDDEN);
  float *final = dev_alloc((size_t)n * MT_HIDDEN);
  float *logits = dev_alloc((size_t)b->rows * (size_t)n_opts);
  float loss = NAN;
  jt_saves_reset(c);
  const int rc = head_fwd(c, x, n, b, &hb, final, logits, fx != NULL ? &loss : NULL, fx != NULL ? JT_FWD_SAVE : JT_FWD_ONLY);
  int ok = 1;
  if (rc == JT_LAYER_NOT_IMPL) {
    printf("test_layer: layer 23 head: not implemented (stub)\n");
  } else if (rc != JT_LAYER_OK) {
    printf("test_layer: layer 23 head: FAIL (the head returned %d)\n", rc);
    ok = 0;
  } else {
    ok &= check_key(c, ref, "final", final, n, MT_HIDDEN);
    ok &= check_key(c, ref, "logits", logits, b->rows, n_opts);
    if (fx != NULL) ok &= check_scalar(ref, "loss", (double)loss);
    else printf("%-22s skipped (no --jtb: the reference has no targets)\n", "loss");
  }
  dev_free(&logits);
  dev_free(&final);
  dev_free(&x);
  free(live);
  return ok ? 0 : 1;
}

// ---- arguments ---------------------------------------------------------------------------------------------------------
static int parse_args(int argc, char **argv, Args *a) {
  memset(a, 0, sizeof *a);
  a->layer = -1;
  a->part = PART_ALL;
  for (int k = 1; k < argc; k++) {
    const char *opt = argv[k];
    const int has = k + 1 < argc;
    if (strcmp(opt, "-h") == 0 || strcmp(opt, "--help") == 0) {
      usage(stdout);
      exit(0);
    } else if (strcmp(opt, "--model") == 0 && has) {
      a->model = argv[++k];
    } else if (strcmp(opt, "--adapter") == 0 && has) {
      a->adapter = argv[++k];
    } else if (strcmp(opt, "--ref") == 0 && has) {
      a->ref = argv[++k];
    } else if (strcmp(opt, "--jtb") == 0 && has) {
      a->jtb = argv[++k];
    } else if (strcmp(opt, "--layer") == 0 && has) {
      char *end = NULL;
      const long v = strtol(argv[++k], &end, 10);
      if (end == NULL || *end != '\0' || v < 0 || v >= MT_LAYERS) die("--layer wants 0..%d, not '%s'", MT_LAYERS - 1, argv[k]);
      a->layer = (int)v;
    } else if (strcmp(opt, "--part") == 0 && has) {
      const char *p = argv[++k];
      if (strcmp(p, "mlp") == 0) a->part = PART_MLP;
      else if (strcmp(p, "mixer") == 0) a->part = PART_MIXER;
      else if (strcmp(p, "all") == 0) a->part = PART_ALL;
      else die("--part wants mlp, mixer or all, not '%s'", p);
    } else if (strcmp(opt, "--head") == 0) {
      a->head = 1;
    } else {
      die("unknown or incomplete option '%s' (--help for the usage)", opt);
    }
  }
  if (a->head && a->layer >= 0) die("--head and --layer are different runs; pass one of them");
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 1) {
    printf("test_layer: no arguments; nothing to do (--help for the usage)\n");
    return 0;
  }
  Args a;
  parse_args(argc, argv, &a);
  if (a.model == NULL) die("--model is required");

  Lora *lora = lora_alloc();
  Model *m = model_load(a.model, lora);
  if (a.adapter != NULL) {
    lora_load_adapter(lora, a.adapter);
    printf("test_layer: adapter %s loaded over the checkpoint's readout\n", a.adapter);
  }
  if (a.ref == NULL) {
    printf("test_layer: %s loaded and every tensor checked; no --ref, so nothing to compare\n", a.model);
    model_free(m);
    lora_free(lora);
    return 0;
  }
  if (!a.head && a.layer < 0) die("--layer i or --head is required with --ref (--help for the usage)");

  St *ref = st_open(a.ref);
  int rows = 0, n = 0;
  const int *off = ref_offsets(ref, &rows);
  const int *ids = ref_ids(ref, &n);
  if (off[rows] != n) die("the reference is inconsistent: in.off[%d] = %d but in.ids has %d", rows, off[rows], n);
  JtBatch b;
  jt_batch_set(&b, rows, n, off);

  int lmax = 0;
  for (int r = 0; r < rows; r++)
    if (b.off[r + 1] - b.off[r] > lmax) lmax = b.off[r + 1] - b.off[r];

  JtCtx *ctx = jt_ctx_alloc(m, lora, n, rows, lmax);

  Fixture fx;
  memset(&fx, 0, sizeof fx);
  if (a.jtb != NULL) load_fixture(a.jtb, &b, ids, ctx->stream, &fx);

  int rc = 0;
  if (a.head) {
    rc = run_head(ctx, ref, a.jtb != NULL ? &fx : NULL, &b);
  } else {
    rc = run_layer(ctx, ref, a.layer, a.part, &b);
  }

  if (a.jtb != NULL) fixture_free(&fx);
  jt_ctx_free(ctx);
  jt_batch_free(&b);
  st_close(ref);
  model_free(m);
  lora_free(lora);
  return rc;
}
