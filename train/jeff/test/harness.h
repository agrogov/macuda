// test/harness.h — everything a per-op test needs that is not the op itself (plan M2.0). Header-only; every function is
// static inline, so a group file has only to `#include "harness.h"` and nothing else from the harness.
//
// What is here
//   * the group contract and its three return codes,
//   * rnd_fill(x, n, lo, hi, seed): deterministic xorshift32 input, the float and the double form under one name,
//   * rel_err(got, want, n): the plan's single error metric, again for float and double arrays under one name,
//   * CHECK_REL / CHECK_ABS / CHECK_REL1 / CHECK_ABS1: print one line per check, count failures; a group ends with
//     `return jt_exit_code();`,
//   * dev_upload / dev_download / dev_upload_bytes / dev_download_bytes / dev_alloc / dev_free (no-ops' siblings under
//     -DJT_CPU, where a reference test needs none of them),
//   * jt_f32_to_bf16 / jt_bf16_to_f32 and the plan's two ragged batches,
//   * the gradcheck numeric core (jt_fd_grad, jt_grad_check) so a grad_<group>.c only writes the closure and the call.
//
// Group contract (plan M2.0). test/test_kernels.c calls one `int k_<group>(void)` from test/k_<group>.c and
// test/gradcheck.c one `int grad_<group>(void)` from test/grad_<group>.c. A group returns exactly JT_OK, JT_FAIL or
// JT_NOT_IMPL — never a failure count — because the dispatchers tell a stub from a real group by that value alone.
// Stubs print NOT IMPLEMENTED and return JT_NOT_IMPL; an implemented group checks its cases with CHECK_* and ends
//
//     return jt_exit_code();
//
// which is JT_OK when nothing failed and JT_FAIL otherwise. Each translation unit has its own counter, which is right:
// one group file, one counter, one return value.
//
// The null device. `make test-null` runs the shim binaries with no arguments and nothing computes: the launches happen,
// the memory is whatever was there. In a shim build (-DJT_SHIM) without both TINYNV_HW and TINYNV_SOCKET, CHECK_REL and
// its siblings therefore print `ok  <name> (launch only)` and count nothing, so a launch-and-link run exits 0
// (README.md). The gradcheck core is CPU-only arithmetic and is never skipped: use jt_grad_check, not CHECK_REL, for it.
#ifndef JT_HARNESS_H
#define JT_HARNESS_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "real.h" // real (float, or double with -DJT_REAL_DOUBLE)
#include "util.h" // CK(); declares no CUDA anything under -DJT_CPU

// ---- the group contract ---------------------------------------------------------------------------------------------
#define JT_OK 0        // every check in the group passed
#define JT_FAIL 1      // at least one check failed (each one printed FAIL)
#define JT_NOT_IMPL 2  // stub: nothing here yet (it printed NOT IMPLEMENTED)
// An unknown group or extra arguments exit 2 as well: "this could not be run" (the same convention card-check.sh uses).
#define JT_USAGE 2

// Test gates (plan M2.0 / design §10). Forward kernels are fp32 against fp32, so only the summation order differs;
// backward kernels carry a little more cancellation. Neither number is ever loosened to make a test pass.
#define JT_TOL_FWD 1e-5
#define JT_TOL_BWD 1e-4
#define JT_TOL_GRAD 1e-6

// ---- failures -------------------------------------------------------------------------------------------------------
// One counter per translation unit (a function-local static in a static inline function). Nothing else in the test tree
// is shared mutable state.
static inline int *jt_fail_slot(void) {
  static int n = 0;
  return &n;
}
static inline void jt_fail_inc(void) { (*jt_fail_slot())++; }
static inline int jt_fail_count(void) { return *jt_fail_slot(); }
static inline void jt_fail_reset(void) { *jt_fail_slot() = 0; }
// What a group returns: JT_OK if its checks all passed, JT_FAIL otherwise.
static inline int jt_exit_code(void) { return jt_fail_count() ? JT_FAIL : JT_OK; }

// ---- is there a device behind the shim? -----------------------------------------------------------------------------
// The hardware path needs both TINYNV_HW and TINYNV_SOCKET (design §11.2); with either missing the shim is the null
// device: launches are reported, values are not computed. Everything else (real nvcc, the CPU build) always computes.
static inline int jt_has_card(void) {
#if defined(JT_SHIM)
  const char *hw = getenv("TINYNV_HW");
  const char *sock = getenv("TINYNV_SOCKET");
  return hw != NULL && hw[0] != '\0' && sock != NULL && sock[0] != '\0';
#else
  return 1;
#endif
}
static inline int jt_is_launch_only(void) { return !jt_has_card(); }

// ---- deterministic random inputs (xorshift32, plan M2.0) ------------------------------------------------------------
static inline uint32_t jt_rng_u32(uint32_t *s) {
  uint32_t x = *s;
  if (x == 0) x = 0x9e3779b9u; // seed 0 must not stay at 0
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *s = x;
  return x;
}
// [0, 1) with 24 bits of resolution: the same stream for the float and the double filler, so one seed names one input.
static inline double jt_rng_unit(uint32_t *s) { return (double)(jt_rng_u32(s) >> 8) * (1.0 / 16777216.0); }
static inline double jt_rng_range(uint32_t *s, double lo, double hi) { return lo + (hi - lo) * jt_rng_unit(s); }
static inline uint32_t jt_rng_below(uint32_t *s, uint32_t n) { return n ? jt_rng_u32(s) % n : 0; }

static inline void rnd_fill_f(float *x, size_t n, double lo, double hi, uint32_t seed) {
  uint32_t s = seed;
  for (size_t i = 0; i < n; i++) x[i] = (float)jt_rng_range(&s, lo, hi);
}
static inline void rnd_fill_d(double *x, size_t n, double lo, double hi, uint32_t seed) {
  uint32_t s = seed;
  for (size_t i = 0; i < n; i++) x[i] = jt_rng_range(&s, lo, hi);
}
// rnd_fill(x, n, lo, hi, seed) fills n values in [lo, hi); x is float* or (with -DJT_REAL_DOUBLE) double*.
#define rnd_fill(x, n, lo, hi, seed)                                                                                   \
  _Generic((x), float *: rnd_fill_f, double *: rnd_fill_d)((x), (size_t)(n), (double)(lo), (double)(hi), (uint32_t)(seed))

// ---- the error metric (plan M1.2) -----------------------------------------------------------------------------------
// rel = max|got - want| / max(1e-30, max|want|). want is the truth (the bigger the reference values, the more absolute
// error is allowed). A NaN or infinity in got is infinitely wrong, never "close to the worst element".
static inline double rel_err_d(const double *got, const double *want, size_t n) {
  double scale = 0.0, worst = 0.0;
  int bad = 0;
  for (size_t i = 0; i < n; i++) {
    double w = fabs(want[i]);
    if (w > scale) scale = w;
    if (!isfinite(got[i])) { bad = 1; continue; }
    double e = fabs(got[i] - want[i]);
    if (e > worst) worst = e;
  }
  if (bad) return INFINITY;
  return worst / (scale > 1e-30 ? scale : 1e-30);
}
static inline double rel_err_f(const float *got, const float *want, size_t n) {
  double scale = 0.0, worst = 0.0;
  int bad = 0;
  for (size_t i = 0; i < n; i++) {
    double w = fabs((double)want[i]);
    if (w > scale) scale = w;
    double g = (double)got[i];
    if (!isfinite(g)) { bad = 1; continue; }
    double e = fabs(g - (double)want[i]);
    if (e > worst) worst = e;
  }
  if (bad) return INFINITY;
  return worst / (scale > 1e-30 ? scale : 1e-30);
}
// max|got - want|, for checks whose answer is exactly zero or exactly a constant (a masked row, a one-option loss).
static inline double abs_err_d(const double *got, const double *want, size_t n) {
  double worst = 0.0;
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(got[i])) return INFINITY;
    double e = fabs(got[i] - want[i]);
    if (e > worst) worst = e;
  }
  return worst;
}
static inline double abs_err_f(const float *got, const float *want, size_t n) {
  double worst = 0.0;
  for (size_t i = 0; i < n; i++) {
    double g = (double)got[i];
    if (!isfinite(g)) return INFINITY;
    double e = fabs(g - (double)want[i]);
    if (e > worst) worst = e;
  }
  return worst;
}
// rel_err(got, want, n) with float* or double* arrays.
#define rel_err(got, want, n)                                                                                          \
  _Generic((got), float *: rel_err_f, const float *: rel_err_f, double *: rel_err_d, const double *: rel_err_d)       \
      ((got), (want), (size_t)(n))
// abs_err(got, want, n): the same, as max|got - want|.
#define abs_err(got, want, n)                                                                                          \
  _Generic((got), float *: abs_err_f, const float *: abs_err_f, double *: abs_err_d, const double *: abs_err_d)       \
      ((got), (want), (size_t)(n))

// ---- one printed line per check --------------------------------------------------------------------------------------
static inline void jt_report_rel(const char *name, double rel, double tol) {
  if (rel < tol) {
    printf("ok  %s rel=%.3g\n", name, rel);
  } else {
    printf("FAIL %s rel=%.3g tol=%.3g\n", name, rel, tol);
    jt_fail_inc();
  }
  fflush(stdout);
}
static inline void jt_report_abs(const char *name, double err, double atol) {
  if (err < atol) {
    printf("ok  %s err=%.3g\n", name, err);
  } else {
    printf("FAIL %s err=%.3g tol=%.3g\n", name, err, atol);
    jt_fail_inc();
  }
  fflush(stdout);
}
static inline void jt_report_launch_only(const char *name) {
  printf("ok  %s (launch only)\n", name);
  fflush(stdout);
}
// A scalar through the same metric: rel = |got - want| / max(1e-30, |want|).
static inline void jt_check_rel1(const char *name, double got, double want, double tol) {
  double scale = fabs(want) > 1e-30 ? fabs(want) : 1e-30;
  jt_report_rel(name, fabs(got - want) / scale, tol);
}
static inline void jt_check_abs1(const char *name, double got, double want, double atol) {
  jt_report_abs(name, fabs(got - want), atol);
}

// CHECK_REL(name, got, want, n, tol): compare two arrays of n floats/doubles and count a failure.
#define CHECK_REL(name, got, want, n, tol)                                                                             \
  do {                                                                                                                 \
    if (jt_has_card())                                                                                                 \
      jt_report_rel((name), rel_err((got), (want), (size_t)(n)), (double)(tol));                                       \
    else                                                                                                               \
      jt_report_launch_only((name));                                                                                   \
  } while (0)
// CHECK_ABS(name, got, want, n, atol): max|got - want| < atol (an exactly-zero row, a one-option loss).
#define CHECK_ABS(name, got, want, n, atol)                                                                            \
  do {                                                                                                                 \
    if (jt_has_card())                                                                                                 \
      jt_report_abs((name), abs_err((got), (want), (size_t)(n)), (double)(atol));                                      \
    else                                                                                                               \
      jt_report_launch_only((name));                                                                                   \
  } while (0)
// The scalar forms (a loss, a norm, an inner product): CHECK_REL1(name, got, want, tol) is the plan's metric on one value.
#define CHECK_REL1(name, got, want, tol)                                                                               \
  do {                                                                                                                 \
    if (jt_has_card())                                                                                                 \
      jt_check_rel1((name), (double)(got), (double)(want), (double)(tol));                                             \
    else                                                                                                               \
      jt_report_launch_only((name));                                                                                   \
  } while (0)
#define CHECK_ABS1(name, got, want, atol)                                                                              \
  do {                                                                                                                 \
    if (jt_has_card())                                                                                                 \
      jt_check_abs1((name), (double)(got), (double)(want), (double)(atol));                                            \
    else                                                                                                               \
      jt_report_launch_only((name));                                                                                   \
  } while (0)

// ---- device buffers -------------------------------------------------------------------------------------------------
// Every allocation and copy goes through CK() (design §12), so a failure names the line. Sizes are elements for the
// float forms and bytes for the *_bytes forms. Nothing here is declared in the CPU build: a reference test has no device.
#ifndef JT_CPU
static inline void *dev_alloc(size_t nbytes) {
  void *p = NULL;
  CK(cudaMalloc(&p, nbytes ? nbytes : 1));
  return p;
}
static inline void dev_free(void *p) {
  if (p != NULL) CK(cudaFree(p));
}
static inline void *dev_upload_bytes(const void *host, size_t nbytes) {
  void *d = dev_alloc(nbytes);
  if (nbytes) CK(cudaMemcpy(d, host, nbytes, cudaMemcpyHostToDevice));
  return d;
}
static inline void dev_download_bytes(const void *dev, void *host, size_t nbytes) {
  if (nbytes) CK(cudaMemcpy(host, dev, nbytes, cudaMemcpyDeviceToHost));
}
// dev_upload(host, n) -> device copy of n floats; dev_download(dev, host, n) copies n floats back.
static inline float *dev_upload(const float *host, size_t n) {
  return (float *)dev_upload_bytes(host, n * sizeof *host);
}
static inline void dev_download(const float *dev, float *host, size_t n) {
  dev_download_bytes(dev, host, n * sizeof *host);
}
#endif // !JT_CPU

// ---- bf16, as misc/embed_gather and the weight loader use it (design §8.1) -------------------------------------------
// bf16 -> fp32 is the top 16 bits, nothing else. fp32 -> bf16 rounds to nearest, ties to even (what the file writer and
// a test fixture should do; the kernels only ever go the other way).
static inline float jt_bf16_to_f32(unsigned short b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, sizeof f);
  return f;
}
static inline unsigned short jt_f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof u);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0) return (unsigned short)((u >> 16) | 0x0040u); // NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}

// ---- the plan's ragged batches (M2.0: lengths {1,37,130} and {520,670,1026,300}) -------------------------------------
// 1026 is the longest real row: it crosses 1024, so it catches a buffer sized to a power of two.
static inline const int *jt_lens_ragged3(void) {
  static const int lens[3] = {1, 37, 130};
  return lens;
}
static inline const int *jt_lens_ragged4(void) {
  static const int lens[4] = {520, 670, 1026, 300};
  return lens;
}
#define JT_RAGGED3_ROWS 3
#define JT_RAGGED4_ROWS 4
// off[] (nrows + 1 entries, off[nrows] = token count) and row_of[] (one row index per token, may be NULL) for a packed
// batch: no padding, so row b is [off[b], off[b+1]). Returns the token count.
static inline int jt_ragged(const int *lens, int nrows, int *off, int *row_of) {
  int n = 0;
  off[0] = 0;
  for (int b = 0; b < nrows; b++) {
    if (row_of != NULL)
      for (int t = 0; t < lens[b]; t++) row_of[n + t] = b;
    n += lens[b];
    off[b + 1] = n;
  }
  return n;
}

// ---- the gradcheck numeric core (plan M2.0) --------------------------------------------------------------------------
// A group picks a random input x and a random cotangent w for the output, gets the analytic gradient from ref_*_bwd, and
// lets the core produce the numeric one:
//
//   L(x) = <w, f(x)>            (L is a scalar: f's output is nw values and w weights them)
//   num[i] = (L(x + h e_i) - L(x - h e_i)) / 2h      h = 1e-6, every element i
//
// then compares num against the analytic gx with rel_err < JT_TOL_GRAD. f is the reference forward of the same op with
// every other input fixed inside ctx: a grad_<group>.c defines one small wrapper function per differentiable input and a
// struct that holds the rest, e.g.
//
//   static void fx(void *v, const real *x, real *y) { G *g = v; ref_foo_fwd(x, g->w, y, g->rows, g->d); }
//   ...
//   ref_foo_bwd(w, ..., gx);                                    // analytic, all inputs at once
//   jt_grad_check("foo/d_x", fx, &g, x, nx, gx, w, ny);         // prints ok/FAIL and counts
//
// Sizes should be shrunk (the references take their dims as parameters) so that "every element" stays cheap; the gate and
// the meaning of the numbers do not change with the size.
typedef void (*jt_fd_fn)(void *ctx, const real *x, real *y);
#define JT_FD_H 1e-6
// Sum in double even in the float build: L(x + h) - L(x - h) is a difference of two nearly equal sums.
static inline double jt_dot(const real *a, const real *b, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; i++) s += (double)a[i] * (double)b[i];
  return s;
}
// num[i] = d/dx_i <w, f(x)>, central difference; both x and w are left unchanged.
static inline void jt_fd_grad(jt_fd_fn f, void *ctx, const real *x, size_t n, const real *w, size_t nw, real *num) {
  real *xs = (real *)malloc((n ? n : 1) * sizeof *xs);
  real *y = (real *)malloc((nw ? nw : 1) * sizeof *y);
  if (xs == NULL || y == NULL) {
    fprintf(stderr, "jt_fd_grad: out of memory (n=%zu, nw=%zu)\n", n, nw);
    exit(1);
  }
  for (size_t i = 0; i < n; i++) xs[i] = x[i];
  for (size_t i = 0; i < n; i++) {
    double lp, lm;
    xs[i] = (real)(x[i] + JT_FD_H);
    f(ctx, xs, y);
    lp = jt_dot(w, y, nw);
    xs[i] = (real)(x[i] - JT_FD_H);
    f(ctx, xs, y);
    lm = jt_dot(w, y, nw);
    xs[i] = x[i];
    num[i] = (real)((lp - lm) / (2.0 * JT_FD_H));
  }
  free(xs);
  free(y);
}
// Numeric vs analytic for one input: prints `ok  name rel=…` or `FAIL name rel=… tol=…`, counts a failure, and returns
// the relative error (num is the truth: want = numeric). Not affected by the null device — no device is involved.
static inline double jt_grad_check(const char *name, jt_fd_fn f, void *ctx, const real *x, size_t n, const real *gx,
                                   const real *w, size_t nw) {
  real *num = (real *)malloc((n ? n : 1) * sizeof *num);
  if (num == NULL) {
    fprintf(stderr, "jt_grad_check: out of memory (n=%zu)\n", n);
    exit(1);
  }
  jt_fd_grad(f, ctx, x, n, w, nw, num);
#if JT_REAL_IS_DOUBLE
  double rel = rel_err_d(gx, num, n);
#else
  double rel = rel_err_f(gx, num, n);
#endif
  jt_report_rel(name, rel, JT_TOL_GRAD);
  free(num);
  return rel;
}

#endif // JT_HARNESS_H
