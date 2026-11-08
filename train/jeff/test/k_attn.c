// test/k_attn.c — the `attn` group of test_kernels (plan M2.5): attn_softmax_fwd/bwd on the device against the CPU
// oracle in ref/attn.c, at the plan's L values {1, 2, 300, 1024} plus the model's head count (8) and small ones.
//
// Every case checks five things:
//   * the forward probabilities against ref_attn_softmax_fwd (plan gate JT_TOL_FWD),
//   * the entries above the diagonal are exactly 0 (the causal mask, not just small),
//   * every row sums to 1 over its live prefix,
//   * the backward against ref_attn_softmax_bwd, fed with the P the kernel itself produced (what the trainer does:
//     design §8.2 keeps P from the forward), gate JT_TOL_BWD,
//   * dS is exactly 0 above the diagonal.
//
// The gates are the plan's; nothing here is loosened. On the shim's null device the launches happen and the value checks
// print `(launch only)` and count nothing (test/harness.h), so `make test-null` stays green.
#include <stdio.h>
#include <stdlib.h>

#include "harness.h"
#include "attn.h"        // kernels/attn.h: the launch wrappers
#include "../ref/attn.h" // ref/attn.h: the CPU oracle

static float *f_alloc(size_t n) {
  float *p = (float *)malloc((n ? n : 1) * sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "k_attn: out of memory (%zu elements)\n", n);
    exit(1);
  }
  return p;
}

// max |x| over the strictly upper triangle of every (h, t) row: the masked half of P or dS.
static double masked_max(const float *x, int L, int heads) {
  double worst = 0.0;
  for (int h = 0; h < heads; h++) {
    for (int t = 0; t < L; t++) {
      const size_t base = ((size_t)h * (size_t)L + (size_t)t) * (size_t)L;
      for (int s = t + 1; s < L; s++) {
        const double v = fabs((double)x[base + (size_t)s]);
        if (v > worst) worst = v;
      }
    }
  }
  return worst;
}

// One (L, heads): forward and backward end to end.
static void case_attn(int L, int heads) {
  const float scale = 1.0f / 16.0f; // design §4.7: c = 1/16
  const size_t n = (size_t)heads * (size_t)L * (size_t)L;
  const size_t nrows = (size_t)heads * (size_t)L;
  char name[128];

  float *S = f_alloc(n), *P_ref = f_alloc(n), *P_got = f_alloc(n);
  float *dP = f_alloc(n), *dS_ref = f_alloc(n), *dS_got = f_alloc(n);
  float *sums = f_alloc(nrows), *ones = f_alloc(nrows);

  const uint32_t seed = 100u + (uint32_t)(L * 8 + heads);
  rnd_fill(S, n, -4.0, 4.0, seed);
  rnd_fill(dP, n, -2.0, 2.0, seed + 1u); // dP = do_t . v_s; only its s <= t half is used
  for (size_t i = 0; i < nrows; i++) ones[i] = 1.0f;

  ref_attn_softmax_fwd(S, P_ref, L, heads, scale);

  // The device's forward: S -> P, then the backward reads that same P back (design §8.2).
  float *S_d = dev_upload(S, n);
  float *P_d = dev_alloc(n * sizeof *P_got);
  float *dP_d = dev_upload(dP, n);
  float *dS_d = dev_alloc(n * sizeof *dS_got);

  attn_softmax_fwd(S_d, P_d, L, heads, scale, 0);
  JT_SYNC_DEBUG;
  dev_download(P_d, P_got, n);

  snprintf(name, sizeof name, "attn_fwd L=%d h=%d", L, heads);
  CHECK_REL(name, P_got, P_ref, n, JT_TOL_FWD);
  snprintf(name, sizeof name, "attn_fwd_masked_zero L=%d h=%d", L, heads);
  CHECK_ABS1(name, masked_max(P_got, L, heads), 0.0, 1e-12);
  for (int h = 0; h < heads; h++) {
    for (int t = 0; t < L; t++) {
      const size_t base = ((size_t)h * (size_t)L + (size_t)t) * (size_t)L;
      double sum = 0.0;
      for (int s = 0; s <= t; s++) sum += (double)P_got[base + (size_t)s];
      sums[(size_t)h * (size_t)L + (size_t)t] = (float)sum;
    }
  }
  snprintf(name, sizeof name, "attn_fwd_rowsum L=%d h=%d", L, heads);
  CHECK_REL(name, sums, ones, nrows, JT_TOL_FWD);

  ref_attn_softmax_bwd(P_ref, dP, dS_ref, L, heads, scale);
  attn_softmax_bwd(P_d, dP_d, dS_d, L, heads, scale, 0);
  JT_SYNC_DEBUG;
  dev_download(dS_d, dS_got, n);

  snprintf(name, sizeof name, "attn_bwd L=%d h=%d", L, heads);
  CHECK_REL(name, dS_got, dS_ref, n, JT_TOL_BWD);
  snprintf(name, sizeof name, "attn_bwd_masked_zero L=%d h=%d", L, heads);
  CHECK_ABS1(name, masked_max(dS_got, L, heads), 0.0, 1e-12);

  dev_free(S_d);
  dev_free(P_d);
  dev_free(dP_d);
  dev_free(dS_d);
  free(S);
  free(P_ref);
  free(P_got);
  free(dP);
  free(dS_ref);
  free(dS_got);
  free(sums);
  free(ones);
}

int k_attn(void) {
  case_attn(1, 1);    // one option: P = 1, dS = 0 — the degenerate edge
  case_attn(2, 2);
  case_attn(300, 8);  // the model's head count, a mid-length row
  case_attn(1024, 2); // the power-of-two boundary the plan calls out
  case_attn(1026, 1); // the longest real row (plan M2.0's ragged batch: 520/670/1026/300); no buffer may assume 1024
  return jt_exit_code();
}
