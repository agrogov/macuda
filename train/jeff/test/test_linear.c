// test/test_linear.c — `linear_bwd`'s gate (plan M5.1 item 2; design §4.2, §5's GEMM table).
//
//   test_linear --fd                                     the [37,64] -> [48] LoRA linear against a GPU central difference
//   test_linear --model M --adapter D --ref R [--layer i] layer i's MLP LoRA grads against the reference (default 23)
//   test_linear                                          nothing to do: exit 0 (what `make test-null` runs)
//
// Two independent gates, because the plan asks for both and they catch different mistakes:
//
//   1. `--fd` is the plan's "GPU finite-difference check on a [37, 64] -> [48] linear in exact mode". It builds the case
//      from synthetic data only (plan rule 5), runs linear_fwd on the device, and differentiates the scalar
//      L(x, A, B) = <dy, y> with a central difference over *every* element of x, of A and of B, one element at a time
//      (h = 1e-2, both signs, L summed in double on the host). The map is exactly linear in each single element — y is
//      affine in x and multi-affine through x A^T B^T — so the central difference has no truncation error at any h and
//      the only error left is the forward's own fp32 rounding. A wrong factor `s`, a transposed `u` in dB, or `e A` with
//      beta 0 instead of 1 each show up at rel ~ 0.5-1.
//   2. `--model/--adapter/--ref` is the plan's second half: `grad.{stem}.lora_A/B` of layer 23's MLP must match
//      dump_ref.py's reference. Layer 23 is a full-attention layer, so the gradient entering its MLP is the reference's
//      `grad.layer.23.out` (design §4.6: the residual adds the MLP's path, so dL/d(post_mixer) = dL/d(layer.out)) and the
//      three MLP linears' parameter gradients are exactly three linear_bwd calls strung together with design §4.5's
//      elementwise chain. The test walks the MLP by hand — rmsnorm_fwd, three linear_fwd with the u's kept,
//      k_silu_mul_fwd, then linear_bwd down/gate/up and k_silu_mul_bwd — instead of calling layer_mlp_bwd, which is plan
//      M5.1 item 3 and another task's file. As a bonus it checks the same forward path against `layer.23.out`, so the
//      activations the gradients were taken at are demonstrably the model's.
//
// Both gates are design §10.2 style relative gates, `max|got - want| / max|want|`, and both need the exact build: TF32's
// 10-bit mantissa moves a gradient by ~1e-3 of its own scale, which is the gate itself. Run them with JT_EXACT=1 (design
// §5/§11.1); the program refuses (exit 2) rather than print a number nobody can trust. Nothing here is loosened: the FD
// gate is the plan's gradient gate (M5.1's `--grad-rel 5e-3`), the reference gate is the same 5e-3, and the activation
// bonus is design §10.2's 1e-3. The measured numbers are in notes/J25.md.
//
// Exit codes are the tree's (test/harness.h): 0 passed, 1 a check failed, 2 bad arguments / wrong compute mode. On the
// shim's null device no value is computed, so the value checks print `(launch only)` and count nothing (harness.h's
// CHECK_REL rule), and the no-argument run exits 0.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "act.h"     // k_silu_mul_fwd / k_silu_mul_bwd (design §4.5)
#include "gemm.h"    // gemm_init / gemm_is_exact: the mode both gates need
#include "harness.h" // rnd_fill, rel_err, jt_report_rel, dev_alloc/dev_free, JT_OK/JT_FAIL
#include "layer.h"   // JtCtx, jt_ctx_alloc (the arena and the stream)
#include "linear.h"  // linear_fwd / linear_bwd, the op under test
#include "misc.h"    // jt_add_inplace (the residual)
#include "model.h"   // Model, LayerWeights, mt_layer
#include "norm.h"    // rmsnorm_fwd (the MLP's post-attention norm)
#include "st.h"      // dump_ref.py's reference

#define TOL_FD 5e-3   // plan M5.1's gradient gate (--grad-rel 5e-3) for the central difference
#define TOL_GRAD 5e-3 // the same number for the reference's grad.* keys
#define TOL_FWD 1e-3  // design §10.2's activation gate, for the bonus layer.23.out check

// The plan's FD shape: [37, 64] -> [48], rank 16, s = MT_LORA_SCALE. Odd everywhere a power of two would hide a stride
// mistake, and small enough that every element can have its own central difference.
#define FD_N 37
#define FD_IN 64
#define FD_OUT 48
#define FD_STEP 1e-2 // see the file comment: exact for a linear map, big enough to out-shout the forward's fp32 rounding

static void die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void die(const char *fmt, ...) {
  va_list ap;
  fflush(stdout);
  fputs("test_linear: ", stderr);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  exit(2);
}

static void usage(FILE *f) {
  fprintf(f,
          "usage: test_linear --fd\n"
          "       test_linear --model DIR --adapter DIR --ref FILE [--layer i]\n"
          "       test_linear\n"
          "  --fd        central-difference dx, dA, dB of a [%d,%d] -> [%d] LoRA linear (plan M5.1 item 2)\n"
          "  --model     the checkpoint (as test_layer takes it), with the reference and the init adapter\n"
          "  --layer     which layer's MLP to check against the reference (default 23)\n"
          "  (no arguments) nothing to do, exit 0\n"
          "  gates: FD and reference rel < %.0e, activations rel < %.0e; both need JT_EXACT=1\n",
          FD_N, FD_IN, FD_OUT, (double)TOL_FD, (double)TOL_FWD);
}

// ---- device buffers and stream-ordered copies -------------------------------------------------------------------------
// Everything goes on ctx->stream, which is non-blocking (design §12): a plain cudaMemcpy on the null stream is not ordered
// against the kernels, so the copy and the GEMM would race. Each helper synchronises, which is what a test wants.
static float *fbuf(size_t n_floats) { return (float *)dev_alloc(n_floats * sizeof(float)); }

static void up(JtCtx *c, float *dst, const float *src, size_t n_floats) {
  CK(cudaMemcpyAsync(dst, src, n_floats * sizeof(float), cudaMemcpyHostToDevice, c->stream));
  CK(cudaStreamSynchronize(c->stream));
}

static void down(JtCtx *c, float *dst, const float *src, size_t n_floats) {
  CK(cudaMemcpyAsync(dst, src, n_floats * sizeof(float), cudaMemcpyDeviceToHost, c->stream));
  CK(cudaStreamSynchronize(c->stream));
}

static float *host_floats(size_t n_floats) {
  float *p = (float *)malloc((n_floats ? n_floats : 1) * sizeof(float));
  if (p == NULL) die("out of host memory (%zu floats)", n_floats);
  return p;
}

// ---- part 1: the central difference on the plan's [37,64] -> [48] linear -----------------------------------------------
// The case is the string of numbers the plan names and nothing else: x [n,in], W [out,in], A [rank,in], B [out,rank] and a
// cotangent dy [n,out]. A then B live in one flat buffer at a_off = 0, b_off = rank*in, exactly as src/lora.c lays the
// model's out (src/lora.h). The distributions are py/make_adapter.py's: x ~ U(-1,1), W and A ~ U(-1/sqrt(in), +),
// B ~ U(-0.1, 0.1) — so the LoRA path is a small correction to W's, as in the model, and its gradients are still O(1).
typedef struct {
  JtCtx *c;
  Linear lin;                 // W, in, out, lora, a_off, b_off, stem
  float *x, *y, *u, *dy, *dx; // device
  float *param, *grad;        // device [A | B]: the flat buffer's trainable half and its gradient
  float *hx, *hparam, *hdy;   // host mirrors of the three that are perturbed / read
  float *hy;                  // host mirror of y
  int n, in, out, rank;
} Fd;

static size_t fd_param_floats(const Fd *f) { return (size_t)f->rank * (size_t)f->in + (size_t)f->out * (size_t)f->rank; }

// L = <dy, y> at the current host x and param: the one scalar the central difference differentiates. Both uploads are
// full buffers (2,368 + 1,792 floats at this shape), which is cheaper to read than an offset arithmetic would be.
static double fd_loss(Fd *f) {
  up(f->c, f->x, f->hx, (size_t)f->n * (size_t)f->in);
  if (f->lin.lora) up(f->c, f->param, f->hparam, fd_param_floats(f));
  if (linear_fwd(f->c, &f->lin, f->x, f->n, f->y, f->u) != JT_LAYER_OK) die("linear_fwd failed");
  down(f->c, f->hy, f->y, (size_t)f->n * (size_t)f->out);
  return jt_dot(f->hdy, f->hy, (size_t)f->n * (size_t)f->out); // jt_dot sums in double (harness.h)
}

// Put the device back where the host mirrors are and give the case its forward's u. Every fd_numeric() leaves the device
// holding a *perturbed* x or param (it restores the host copy, which is what the next numeric call reads) and a u that
// belongs to that perturbation, so anything that runs a backward afterwards must prime first.
static void fd_prime(Fd *f) { (void)fd_loss(f); }

// (L(p + h e_i) - L(p - h e_i)) / 2h for element `i` of x (which = FD_X) or of the flat parameter buffer (FD_PARAM),
// where i is counted from that buffer's own first element.
enum { FD_X = 0, FD_PARAM = 1 };
static double fd_numeric(Fd *f, int which, size_t i, double h) {
  float *p = which == FD_X ? f->hx + i : f->hparam + i;
  const float keep = *p;
  double lp, lm;
  *p = (float)((double)keep + h);
  lp = fd_loss(f);
  *p = (float)((double)keep - h);
  lm = fd_loss(f);
  *p = keep;
  return (lp - lm) / (2.0 * h);
}

// The analytic side: linear_fwd's saved u, then linear_bwd into a zeroed dx (when wanted) and a zeroed grad.
static void fd_analytic(Fd *f, float *dx, float *grad) {
  if (dx != NULL) CK(cudaMemsetAsync(dx, 0, (size_t)f->n * (size_t)f->in * sizeof(float), f->c->stream));
  CK(cudaMemsetAsync(grad, 0, fd_param_floats(f) * sizeof(float), f->c->stream));
  if (linear_bwd(f->c, &f->lin, f->x, f->lin.lora ? f->u : NULL, f->dy, f->n, dx, 0) != JT_LAYER_OK)
    die("linear_bwd failed");
}

// A whole array's FD: numeric[i] for every i from `base` on (A and B share one flat buffer, so dB's case starts at
// b_off), then the plan's metric against the analytic array.
static void fd_array(Fd *f, int which, size_t base, const char *what, const float *got_dev, size_t n, double tol) {
  double *num = (double *)malloc((n ? n : 1) * sizeof *num);
  double *got = (double *)malloc((n ? n : 1) * sizeof *got);
  float *flat = host_floats(n);
  if (num == NULL || got == NULL) die("out of host memory (fd %s)", what);
  down(f->c, flat, got_dev, n);
  if (!jt_has_card()) { // the null device computes nothing, so there is no number to gate (test/harness.h)
    jt_report_launch_only(what);
  } else {
    for (size_t i = 0; i < n; i++) {
      got[i] = (double)flat[i];
      num[i] = fd_numeric(f, which, base + i, FD_STEP);
    }
    jt_report_rel(what, rel_err_d(got, num, n), tol);
  }
  free(flat);
  free(got);
  free(num);
}

// The error contract (src/linear.h): a bad call says so and names the linear; a frozen linear with no dx is a no-op.
static void fd_contract(Fd *f, int lora) {
  printf("test_linear: the bad-call contract\n");
  struct {
    const char *what;
    int rc, want;
  } cases[5];
  const float *u = NULL; // a frozen linear ignores u; a LoRA one cannot work without it
  cases[0].what = lora ? "lora without the forward's u" : "frozen, u NULL is fine";
  cases[0].rc = linear_bwd(f->c, &f->lin, f->x, u, f->dy, f->n, f->dx, 0);
  cases[0].want = lora ? JT_LAYER_FAIL : JT_LAYER_OK;
  cases[1].what = "null dy";
  cases[1].rc = linear_bwd(f->c, &f->lin, f->x, NULL, NULL, f->n, f->dx, 0);
  cases[1].want = JT_LAYER_FAIL;
  cases[2].what = "dx aliasing dy";
  cases[2].rc = linear_bwd(f->c, &f->lin, f->x, f->u, f->dy, f->n, (float *)f->dy, 0);
  cases[2].want = JT_LAYER_FAIL;
  cases[3].what = "null x";
  cases[3].rc = linear_bwd(f->c, &f->lin, NULL, f->u, f->dy, f->n, f->dx, 0);
  cases[3].want = JT_LAYER_FAIL;
  cases[4].what = "the good call again";
  cases[4].rc = linear_bwd(f->c, &f->lin, f->x, f->u, f->dy, f->n, f->dx, 0);
  cases[4].want = JT_LAYER_OK;
  for (int k = 0; k < 5; k++) {
    const int ok = cases[k].rc == cases[k].want;
    printf("  %-40s rc %d (want %d)  %s\n", cases[k].what, cases[k].rc, cases[k].want, ok ? "ok" : "FAIL");
    if (!ok) jt_fail_inc();
  }
}

static void fd_check(const char *name, int lora, uint32_t seed) {
  const int rank = LORA_RANK;
  Fd f;
  memset(&f, 0, sizeof f);
  f.n = FD_N;
  f.in = FD_IN;
  f.out = FD_OUT;
  f.rank = rank;

  // A Lora with exactly one entry's worth of words: linear.c reads n_param, param and grad and nothing else, so the
  // model's 10.5M-float table is not needed for this case.
  Lora *lora_buf = (Lora *)calloc(1, sizeof *lora_buf);
  Model *scratch = (Model *)calloc(1, sizeof *scratch);
  if (lora_buf == NULL || scratch == NULL) die("out of host memory (Lora/Model)");
  lora_buf->n_param = (size_t)rank * (size_t)f.in + (size_t)f.out * (size_t)rank;
  lora_buf->param = fbuf(lora_buf->n_param);
  lora_buf->grad = fbuf(lora_buf->n_param);
  lora_buf->readout_off = 0;
  scratch->lora = lora_buf;
  // jt_ctx_alloc wants a Model, but this case never touches it: linear.c reads ctx->lora, ctx->cublas, ctx->arena and
  // ctx->stream. A zeroed one is honest about that (and keeps a 2.5 GB checkpoint out of a 37-row gradcheck).
  f.c = jt_ctx_alloc(scratch, lora_buf, FD_N, 1, FD_N);

  f.lin.stem = name;
  f.lin.in = f.in;
  f.lin.out = f.out;
  f.lin.lora = lora;
  f.lin.a_off = 0;
  f.lin.b_off = (size_t)rank * (size_t)f.in;

  const size_t xn = (size_t)f.n * (size_t)f.in, yn = (size_t)f.n * (size_t)f.out;
  const size_t pn = fd_param_floats(&f), un = (size_t)f.n * (size_t)rank;
  f.hx = host_floats(xn);
  f.hy = host_floats(yn);
  f.hdy = host_floats(yn);
  f.hparam = host_floats(pn);
  f.x = fbuf(xn);
  f.y = fbuf(yn);
  f.u = fbuf(un);
  f.dy = fbuf(yn);
  f.dx = fbuf(xn);
  f.param = lora_buf->param;
  f.grad = lora_buf->grad;
  float *W = fbuf((size_t)f.out * (size_t)f.in);

  // the plan's few hundred kilobytes of synthetic input (plan rule 5: no real data), all from one seed
  const double wscale = 1.0 / sqrt((double)f.in);
  rnd_fill(f.hx, xn, -1.0, 1.0, seed);
  rnd_fill(f.hdy, yn, -1.0, 1.0, seed + 1u);
  float *hW = host_floats((size_t)f.out * (size_t)f.in);
  rnd_fill(hW, (size_t)f.out * (size_t)f.in, -wscale, wscale, seed + 2u);
  rnd_fill(f.hparam, (size_t)rank * (size_t)f.in, -wscale, wscale, seed + 3u);
  rnd_fill(f.hparam + (size_t)rank * (size_t)f.in, (size_t)f.out * (size_t)rank, -0.1, 0.1, seed + 4u);
  up(f.c, W, hW, (size_t)f.out * (size_t)f.in);
  up(f.c, f.dy, f.hdy, yn);
  f.lin.W = W;

  printf("test_linear: %s: [%d,%d] -> [%d], rank %d, s %g, lora %d, FD h %g, %zu + %zu parameters x 2 forwards\n", name,
         f.n, f.in, f.out, rank, (double)MT_LORA_SCALE, lora, FD_STEP, xn, pn);
  fflush(stdout);

  float *adx = host_floats(xn), *adab = host_floats(pn); // the analytic gradients, on the host
  fd_prime(&f);
  fd_analytic(&f, f.dx, f.grad);
  down(f.c, adx, f.dx, xn);
  down(f.c, adab, f.grad, pn);

  up(f.c, f.dx, adx, xn);
  fd_array(&f, FD_X, 0, "dx", f.dx, xn, TOL_FD);
  if (lora) {
    up(f.c, f.grad, adab, pn);
    fd_array(&f, FD_PARAM, 0, "dA", f.grad, (size_t)rank * (size_t)f.in, TOL_FD);
    fd_array(&f, FD_PARAM, (size_t)rank * (size_t)f.in, "dB", f.grad + (size_t)rank * (size_t)f.in,
             (size_t)f.out * (size_t)rank, TOL_FD);
    fd_prime(&f); // the FD loops above left the device holding the last perturbation

    // The accumulate path (design §4.6): a bwd into a dx that already holds a residual gradient must add, not overwrite.
    float *pre = host_floats(xn), *want = host_floats(xn), *got = host_floats(xn);
    rnd_fill(pre, xn, -1.0, 1.0, seed + 5u);
    up(f.c, f.dx, pre, xn);
    CK(cudaMemsetAsync(f.grad, 0, pn * sizeof(float), f.c->stream));
    if (linear_bwd(f.c, &f.lin, f.x, f.u, f.dy, f.n, f.dx, 1) != JT_LAYER_OK) die("linear_bwd(accumulate) failed");
    down(f.c, got, f.dx, xn);
    for (size_t i = 0; i < xn; i++) want[i] = (float)((double)pre[i] + (double)adx[i]);
    if (jt_has_card()) jt_report_rel("dx accumulate_dx", rel_err_f(got, want, xn), 1e-5);
    else jt_report_launch_only("dx accumulate_dx");
    free(pre);
    free(want);
    free(got);

    // dx == NULL: only the parameter gradient is wanted, and it must be the same one.
    CK(cudaMemsetAsync(f.grad, 0, pn * sizeof(float), f.c->stream));
    if (linear_bwd(f.c, &f.lin, f.x, f.u, f.dy, f.n, NULL, 0) != JT_LAYER_OK) die("linear_bwd(dx NULL) failed");
    float *again = host_floats(pn);
    down(f.c, again, f.grad, pn);
    if (jt_has_card()) jt_report_rel("dA/dB with dx NULL", rel_err_f(again, adab, pn), 1e-6);
    else jt_report_launch_only("dA/dB with dx NULL");
    free(again);

    // How much of dx the LoRA path carries: with A = B = 0 the same backward can only give dy W, so the difference is the
    // e A term the design §4.2's second GEMM adds. Printed, not gated — it is the evidence that the FD's dx gate cannot
    // be satisfied by the W path alone (a missing dx += e A, or a beta of 0 there, would be this much of dx).
    float *zero = host_floats(pn), *dxw = host_floats(xn);
    memset(zero, 0, pn * sizeof(float));
    up(f.c, f.param, zero, pn);
    if (linear_fwd(f.c, &f.lin, f.x, f.n, f.y, f.u) != JT_LAYER_OK) die("linear_fwd (A=B=0) failed");
    CK(cudaMemsetAsync(f.dx, 0, xn * sizeof(float), f.c->stream));
    if (linear_bwd(f.c, &f.lin, f.x, f.u, f.dy, f.n, f.dx, 0) != JT_LAYER_OK) die("linear_bwd (A=B=0) failed");
    down(f.c, dxw, f.dx, xn);
    if (jt_has_card()) // on the null device nothing was computed, so there is no share to report
      printf("test_linear: info: max|dx - dyW| / max|dx| = %.1f%% (the LoRA path's share of dx)\n",
             100.0 * rel_err_f(dxw, adx, xn));
    up(f.c, f.param, f.hparam, pn);
    free(zero);
    free(dxw);
  }
  fd_prime(&f);
  fd_contract(&f, lora);

  free(adx);
  free(adab);
  free(hW);
  free(f.hx);
  free(f.hy);
  free(f.hdy);
  free(f.hparam);
  dev_free(f.x);
  dev_free(f.y);
  dev_free(f.u);
  dev_free(f.dy);
  dev_free(W);
  dev_free(lora_buf->param);
  dev_free(lora_buf->grad);
  jt_ctx_free(f.c);
  free(lora_buf);
  free(scratch);
}

// ---- part 2: layer 23's MLP LoRA gradients against dump_ref.py's reference --------------------------------------------
// A [rows][cols] F32 reference tensor, uploaded on the context's stream.
static float *ref_rows(JtCtx *c, const St *ref, const char *key, int rows, int cols) {
  const int shape[2] = {rows, cols};
  const float *src = (const float *)st_getv(ref, key, ST_DTYPE_F32, 2, shape);
  float *d = fbuf((size_t)rows * (size_t)cols);
  up(c, d, src, (size_t)rows * (size_t)cols);
  return d;
}

static void check_ref_key(JtCtx *c, const St *ref, const char *key, const float *dev, int rows, int cols, double tol) {
  const int shape[2] = {rows, cols};
  const float *want = (const float *)st_getv(ref, key, ST_DTYPE_F32, 2, shape);
  const size_t n = (size_t)rows * (size_t)cols;
  if (!jt_has_card()) { // the null device computes nothing (test/harness.h)
    jt_report_launch_only(key);
    return;
  }
  float *got = host_floats(n);
  down(c, got, dev, n);
  const double rel = rel_err_f(got, want, n);
  const int ok = rel < tol;
  printf("%-48s rel %.3e  %s (want < %.0e, %zu values)\n", key, rel, ok ? "ok" : "FAIL", tol, n);
  if (!ok) jt_fail_inc();
  fflush(stdout);
  free(got);
}

// The adapter key of a Linear, from the flat buffer's table (src/lora.h). It is not taken from `Linear.stem`: model.c
// builds those strings in a pool it frees at the end of model_load(`free(L.pool)`, src/model.c:301), so every stem it
// hands out is a dangling pointer once the model is loaded — a finding for that file's owner, reported in notes/J25.md.
// The table's own copy (LoraEntry.stem) is what lora_load_adapter/save_adapter key on, so it is the authoritative one.
static const char *adapter_stem(const JtCtx *c, const Linear *lin) {
  for (int k = 0; lin->lora && k < c->lora->n_ent; k++)
    if (c->lora->ent[k].rank > 0 && c->lora->ent[k].off == lin->a_off && c->lora->ent[k].b_off == lin->b_off)
      return c->lora->ent[k].stem;
  die("no flat-buffer entry at a_off %zu / b_off %zu", lin->a_off, lin->b_off);
}

// layer_bwd would call this path for layer i's MLP (design §4.5); the test spells it out so that layer_mlp_bwd — plan
// M5.1 item 3, another task — is not on the dependency path of this gate. Nothing is taken from the arena's save half:
// every buffer is the test's own, so a change to another file's save layout cannot move these numbers.
static void run_mlp(JtCtx *c, const St *ref, int layer) {
  const MtLayer *ml = mt_layer(layer);
  if (ml == NULL) die("layer %d, want 0..%d", layer, MT_LAYERS - 1);
  const LayerWeights *lw = &c->model->layers[layer];
  const int n = c->cap_tokens;
  const size_t hn = (size_t)n * MT_HIDDEN, an = (size_t)n * MT_MLP_INTER, un = (size_t)n * LORA_RANK;

  char kpm[48], kout[48], kdres[48];
  snprintf(kpm, sizeof kpm, "layer.%d.post_mixer", layer);
  snprintf(kout, sizeof kout, "layer.%d.out", layer);
  snprintf(kdres, sizeof kdres, "grad.layer.%d.out", layer);
  const char *keys[3] = {kpm, kdres, kout};
  for (int k = 0; k < 3; k++)
    if (st_find(ref, keys[k]) == NULL) die("the reference has no '%s'", keys[k]);

  float *x_in = ref_rows(c, ref, kpm, n, MT_HIDDEN);
  float *dres = ref_rows(c, ref, kdres, n, MT_HIDDEN);
  float *m = fbuf(hn), *r = fbuf((size_t)n), *h1 = fbuf(an), *h2 = fbuf(an), *act = fbuf(an);
  float *ug = fbuf(un), *uu = fbuf(un), *ud = fbuf(un);
  float *y = fbuf(hn), *out = fbuf(hn);
  float *dact = fbuf(an), *dh1 = fbuf(an), *dh2 = fbuf(an), *dm = fbuf(hn);

  printf("test_linear: layer %d MLP (%s): %s -> %s, then d/d(gate,up,down)\n", layer, ml->prefix, kpm, kout);
  fflush(stdout);
  // forward, design §3.4's last two lines (design §4.5's first three)
  rmsnorm_fwd(x_in, lw->post_attn_norm, m, r, n, MT_HIDDEN, 1, c->stream);
  if (linear_fwd(c, &lw->gate, m, n, h1, ug) != JT_LAYER_OK) die("linear_fwd(gate) failed");
  if (linear_fwd(c, &lw->up, m, n, h2, uu) != JT_LAYER_OK) die("linear_fwd(up) failed");
  k_silu_mul_fwd(h1, h2, act, (int)an, c->stream);
  if (linear_fwd(c, &lw->down, act, n, y, ud) != JT_LAYER_OK) die("linear_fwd(down) failed");
  // the residual: out = x_in + y. A reference key this test did not otherwise need, so the activations the gradients are
  // taken at are demonstrably the model's own.
  CK(cudaMemcpyAsync(out, x_in, hn * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));
  jt_add_inplace(out, y, hn, c->stream);
  check_ref_key(c, ref, kout, out, n, MT_HIDDEN, TOL_FWD);

  // backward, design §4.5: dact = dout through down; dh1/dh2 through silu_mul; dm = gate's + up's dx (design §4.6).
  lora_zero_grad(c->lora, c->stream);
  if (linear_bwd(c, &lw->down, act, ud, dres, n, dact, 0) != JT_LAYER_OK) die("linear_bwd(down) failed");
  k_silu_mul_bwd(dact, h1, h2, dh1, dh2, (int)an, c->stream);
  if (linear_bwd(c, &lw->gate, m, ug, dh1, n, dm, 0) != JT_LAYER_OK) die("linear_bwd(gate) failed");
  if (linear_bwd(c, &lw->up, m, uu, dh2, n, dm, 1) != JT_LAYER_OK) die("linear_bwd(up) failed");
  CK(cudaStreamSynchronize(c->stream));

  const Linear *lins[3] = {&lw->gate, &lw->up, &lw->down};
  for (int k = 0; k < 3; k++) {
    const char *stem = adapter_stem(c, lins[k]);
    char ka[LORA_STEM_MAX + 32], kb[LORA_STEM_MAX + 32];
    snprintf(ka, sizeof ka, "grad.%s.lora_A", stem);
    snprintf(kb, sizeof kb, "grad.%s.lora_B", stem);
    check_ref_key(c, ref, ka, c->lora->grad + lins[k]->a_off, LORA_RANK, lins[k]->in, TOL_GRAD);
    check_ref_key(c, ref, kb, c->lora->grad + lins[k]->b_off, lins[k]->out, LORA_RANK, TOL_GRAD);
  }
  // There is no reference key for the MLP's own input gradient, so only its shape is known: finite, and not all zeros
  // (which is what dropping `up`'s accumulate_dx would leave behind).
  {
    float *h = host_floats(hn);
    down(c, h, dm, hn);
    size_t zero = 0, bad = 0;
    for (size_t i = 0; i < hn; i++) {
      if (!isfinite(h[i])) bad++;
      if (h[i] == 0.0f) zero++;
    }
    const int ok = bad == 0 && zero < hn;
    printf("%-48s %s (%zu of %zu zero, %zu not finite)\n", "dL/dm = d(gate) + d(up)", ok ? "ok" : "FAIL", zero, hn, bad);
    if (!ok) jt_fail_inc();
    free(h);
  }
  dev_free(x_in);
  dev_free(dres);
  dev_free(m);
  dev_free(r);
  dev_free(h1);
  dev_free(h2);
  dev_free(act);
  dev_free(ug);
  dev_free(uu);
  dev_free(ud);
  dev_free(y);
  dev_free(out);
  dev_free(dact);
  dev_free(dh1);
  dev_free(dh2);
  dev_free(dm);
}

// ---- arguments -------------------------------------------------------------------------------------------------------
typedef struct {
  const char *model, *adapter, *ref;
  int layer, fd, have_layer;
} Args;

static void parse_args(int argc, char **argv, Args *a) {
  memset(a, 0, sizeof *a);
  a->layer = 23;
  for (int k = 1; k < argc; k++) {
    const char *opt = argv[k];
    const int has = k + 1 < argc;
    if (strcmp(opt, "-h") == 0 || strcmp(opt, "--help") == 0) {
      usage(stdout);
      exit(0);
    } else if (strcmp(opt, "--fd") == 0) {
      a->fd = 1;
    } else if (strcmp(opt, "--model") == 0 && has) {
      a->model = argv[++k];
    } else if (strcmp(opt, "--adapter") == 0 && has) {
      a->adapter = argv[++k];
    } else if (strcmp(opt, "--ref") == 0 && has) {
      a->ref = argv[++k];
    } else if (strcmp(opt, "--layer") == 0 && has) {
      char *end = NULL;
      const long v = strtol(argv[++k], &end, 10);
      if (end == NULL || *end != '\0' || v < 0 || v >= MT_LAYERS) die("--layer wants 0..%d, not '%s'", MT_LAYERS - 1, argv[k]);
      a->layer = (int)v;
      a->have_layer = 1;
    } else {
      die("unknown or incomplete option '%s' (--help for the usage)", opt);
    }
  }
  if (a->fd && (a->model != NULL || a->ref != NULL)) die("--fd and --model/--ref are different runs; pass one of them");
  if (a->fd && a->have_layer) die("--fd has its own shape; --layer is only for the reference run");
}

// The reference run's batch geometry: rows and tokens from the reference's own in.off / in.ids (dump_ref.py wrote them),
// the longest row for ctx->lmax. No .jtb is needed — this test reads forward and backward keys only, never the loss.
static void ref_geometry(const St *ref, int *rows, int *n, int *lmax) {
  const StTensor *t = st_find(ref, "in.off");
  if (t == NULL || t->dtype != ST_DTYPE_I32 || t->ndim != 1 || t->shape[0] < 2)
    die("the reference has no usable 'in.off' (want I32 [B+1])");
  const int shape[1] = {t->shape[0]};
  const int *off = (const int *)st_getv(ref, "in.off", ST_DTYPE_I32, 1, shape);
  const StTensor *ti = st_find(ref, "in.ids");
  if (ti == NULL || ti->dtype != ST_DTYPE_I32 || ti->ndim != 1) die("the reference has no usable 'in.ids' (want I32 [N])");
  *rows = t->shape[0] - 1;
  *n = ti->shape[0];
  if (off[*rows] != *n) die("the reference is inconsistent: in.off[%d] = %d but in.ids has %d", *rows, off[*rows], *n);
  *lmax = 0;
  for (int r = 0; r < *rows; r++)
    if (off[r + 1] - off[r] > *lmax) *lmax = off[r + 1] - off[r];
}

int main(int argc, char **argv) {
  if (argc == 1) {
    printf("test_linear: no arguments; nothing to do (--help for the usage)\n");
    return 0;
  }
  Args a;
  parse_args(argc, argv, &a);
  if (!a.fd && (a.model == NULL || a.ref == NULL)) die("--fd, or --model with --ref, is required (--help for the usage)");
  if (!a.fd && a.adapter == NULL)
    die("--adapter is required with --ref: the reference's LoRA gradients are at the adapter train.py started from");

  gemm_init();
  if (jt_has_card() && !gemm_is_exact())
    die("the gates are set for exact fp32; run with JT_EXACT=1 (a TF32 gradient moves by ~1e-3 of its own scale)");

  jt_fail_reset();
  if (a.fd) {
    fd_check("fd.lora [37,64]->[48]", 1, 1u);
    fd_check("fd.frozen [37,64]->[48]", 0, 7u);
  } else {
    Lora *lora = lora_alloc();
    Model *m = model_load(a.model, lora);
    lora_load_adapter(lora, a.adapter);
    printf("test_linear: adapter %s loaded over the checkpoint's readout\n", a.adapter);
    St *ref = st_open(a.ref);
    int rows = 0, n = 0, lmax = 0;
    ref_geometry(ref, &rows, &n, &lmax);
    JtCtx *ctx = jt_ctx_alloc(m, lora, n, rows, lmax);
    run_mlp(ctx, ref, a.layer);
    jt_ctx_free(ctx);
    st_close(ref);
    model_free(m);
    lora_free(lora);
  }

  const int fails = jt_fail_count();
  printf("test_linear: %s\n", fails == 0 ? "ok" : "FAILED");
  return fails == 0 ? 0 : 1;
}
