// test/k_head.c — the `head` group of test_kernels (plan M2.5): ce_head on the device against the CPU oracle in
// ref/head.c, with the plan's n_opts {1, 2, 255} and a batch that mixes live counts, including rows with a single live
// option (their loss and dlogits must be exactly 0).
//
// Each case checks:
//   * the per-row loss against ref_ce_head (gate JT_TOL_FWD),
//   * dlogits against ref_ce_head (gate JT_TOL_FWD — it is a forward result of this kernel, not a backward),
//   * the probabilities against ref_ce_head,
//   * every row's probabilities sum to 1 (dead options contribute exactly 0),
//   * the dead options' probabilities are exactly 0 (train.py's masked_fill(-1e9)),
//   * the single-live-option rows' loss and dlogits are exactly 0,
//   * a second launch with dlogits = probs_out = NULL leaves the loss unchanged (an evaluation only wants the loss).
//
// Dead options are -1e9 with target 0, exactly what CLM-8B's tools/jeff_lora/train.py hands to soft_ce. Nothing here is
// loosened. On the shim's null device the launches happen and the value checks print `(launch only)` and count nothing.
#include <stdio.h>
#include <stdlib.h>

#include "harness.h"
#include "head.h"        // kernels/head.h: the launch wrapper
#include "../ref/head.h" // ref/head.h: the CPU oracle

static float *f_alloc(size_t n) {
  float *p = (float *)malloc((n ? n : 1) * sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "k_head: out of memory (%zu elements)\n", n);
    exit(1);
  }
  return p;
}

// One batch: B rows of n options, row r has nlive[r] live options (1 <= nlive <= n), dead ones carry -1e9 / target 0.
static void case_head(const char *tag, int B, int n, const int *nlive, int normalise_targets, uint32_t seed) {
  const size_t nels = (size_t)B * (size_t)n;
  char name[160];

  float *logits = f_alloc(nels), *targets = f_alloc(nels), *wrow = f_alloc(B);
  float *loss_ref = f_alloc(B), *dl_ref = f_alloc(nels), *p_ref = f_alloc(nels);
  float *loss_got = f_alloc(B), *dl_got = f_alloc(nels), *p_got = f_alloc(nels);
  float *sums = f_alloc(B);

  rnd_fill(logits, nels, -5.0, 5.0, seed);
  rnd_fill(targets, nels, 0.0, 1.0, seed + 1u);
  rnd_fill(wrow, B, 0.5, 2.0, seed + 2u);
  for (int r = 0; r < B; r++) {
    double tsum = 0.0;
    for (int j = 0; j < n; j++) {
      tsum += nlive[r] > j ? (double)targets[(size_t)r * (size_t)n + (size_t)j] : 0.0;
    }
    for (int j = 0; j < n; j++) {
      const size_t i = (size_t)r * (size_t)n + (size_t)j;
      if (j >= nlive[r]) {
        logits[i] = (float)REF_HEAD_DEAD; // what train.py's masked_fill leaves behind
        targets[i] = 0.0f;
      } else if (normalise_targets) {
        targets[i] = (float)((double)targets[i] / tsum); // train.py's targets sum to 1 over the live options
      }
    }
    sums[r] = 1.0f; // the comparison target for the row-sum check
  }
  float wgroup = 0.0f;
  for (int r = 0; r < B; r++) wgroup += wrow[r];

  ref_ce_head(logits, n, targets, wrow, wgroup, B, loss_ref, dl_ref, p_ref);

  float *logits_d = dev_upload(logits, nels);
  float *targets_d = dev_upload(targets, nels);
  float *wrow_d = dev_upload(wrow, B);
  float *loss_d = dev_alloc(B * sizeof *loss_got);
  float *dl_d = dev_alloc(nels * sizeof *dl_got);
  float *p_d = dev_alloc(nels * sizeof *p_got);

  ce_head(logits_d, n, targets_d, wrow_d, wgroup, B, loss_d, dl_d, p_d, 0);
  JT_SYNC_DEBUG;
  dev_download(loss_d, loss_got, B);
  dev_download(dl_d, dl_got, nels);
  dev_download(p_d, p_got, nels);

  snprintf(name, sizeof name, "ce_head/%s loss", tag);
  CHECK_REL(name, loss_got, loss_ref, B, JT_TOL_FWD);
  snprintf(name, sizeof name, "ce_head/%s dlogits", tag);
  CHECK_REL(name, dl_got, dl_ref, nels, JT_TOL_FWD);
  snprintf(name, sizeof name, "ce_head/%s probs", tag);
  CHECK_REL(name, p_got, p_ref, nels, JT_TOL_FWD);

  // Every row's probabilities sum to 1, dead options included (they are 0).
  float *psum = f_alloc(B);
  for (int r = 0; r < B; r++) {
    double s = 0.0;
    for (int j = 0; j < n; j++) s += (double)p_got[(size_t)r * (size_t)n + (size_t)j];
    psum[r] = (float)s;
  }
  snprintf(name, sizeof name, "ce_head/%s rowsum", tag);
  CHECK_REL(name, psum, sums, B, JT_TOL_FWD);
  free(psum);

  // The dead options are exactly 0, and a single-live-option row has an exactly zero loss and dlogits.
  double dead_worst = 0.0, zero_loss_worst = 0.0, zero_dl_worst = 0.0;
  int have_single = 0;
  for (int r = 0; r < B; r++) {
    for (int j = 0; j < n; j++) {
      const size_t i = (size_t)r * (size_t)n + (size_t)j;
      if (j >= nlive[r] && fabs((double)p_got[i]) > dead_worst) dead_worst = fabs((double)p_got[i]);
    }
    if (nlive[r] == 1) {
      have_single = 1;
      if (fabs((double)loss_got[r]) > zero_loss_worst) zero_loss_worst = fabs((double)loss_got[r]);
      for (int j = 0; j < n; j++) {
        const double v = fabs((double)dl_got[(size_t)r * (size_t)n + (size_t)j]);
        if (v > zero_dl_worst) zero_dl_worst = v;
      }
    }
  }
  snprintf(name, sizeof name, "ce_head/%s dead_probs_zero", tag);
  CHECK_ABS1(name, dead_worst, 0.0, 1e-30);
  if (have_single) {
    snprintf(name, sizeof name, "ce_head/%s single_loss_zero", tag);
    CHECK_ABS1(name, zero_loss_worst, 0.0, 1e-30);
    snprintf(name, sizeof name, "ce_head/%s single_dlogits_zero", tag);
    CHECK_ABS1(name, zero_dl_worst, 0.0, 1e-30);
  }

  // A loss-only launch (an evaluation: no gradient, no probabilities) must give the same loss.
  for (int r = 0; r < B; r++) loss_got[r] = -1.0f;
  ce_head(logits_d, n, targets_d, wrow_d, wgroup, B, loss_d, NULL, NULL, 0);
  JT_SYNC_DEBUG;
  dev_download(loss_d, loss_got, B);
  snprintf(name, sizeof name, "ce_head/%s loss_only", tag);
  CHECK_REL(name, loss_got, loss_ref, B, JT_TOL_FWD);

  dev_free(logits_d);
  dev_free(targets_d);
  dev_free(wrow_d);
  dev_free(loss_d);
  dev_free(dl_d);
  dev_free(p_d);
  free(logits);
  free(targets);
  free(wrow);
  free(loss_ref);
  free(dl_ref);
  free(p_ref);
  free(loss_got);
  free(dl_got);
  free(p_got);
  free(sums);
}

int k_head(void) {
  static const int one[4] = {1, 1, 1, 1};                                  // n_opts = 1: every row is a single option
  static const int two[4] = {2, 1, 2, 2};                                  // n_opts = 2, one row trimmed to 1
  static const int wide[8] = {1, 2, 3, 255, 128, 7, 200, 50};              // n_opts = 255, mixed live counts
  case_head("n1", 4, 1, one, 0, 31u);
  case_head("n2", 4, 2, two, 0, 32u);           // unnormalised targets: the sum(t) factor in dlogits
  case_head("n255", 8, 255, wide, 1, 33u);      // a full-width row with train.py's normalised targets
  case_head("n255u", 8, 255, wide, 0, 34u);     // the same width with unnormalised targets
  return jt_exit_code();
}
