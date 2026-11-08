// test/k_conv.c — the `conv` group of test_kernels (plan M2.0/M2.4). GPU vs the CPU reference, fp32 both sides.
//
// Plan M2.4: conv_fwd (pre = the conv output, out = silu(pre)) and conv_bwd (design §4.8's transposed conv), with
// wconv[ch*4 + k] multiplying x[t - 3 + k], taps before a row's start reading zero, and the backward's taps before a
// row's end reading zero too.
//
// Checked here:
//   * forward (pre and out) against ref_conv_fwd and backward (dqkv) against ref_conv_bwd over a single token, a
//     non-multiple-of-the-block channel count, and the plan's two ragged batches at the model's 6144 channels — 168 and
//     2516 tokens, the second one 520 + 670 + 1026 + 300, whose 1026-token row is the longest real row;
//   * the row-boundary check the plan names: randomising row 1's last 3 tokens must leave row 2's first 3 outputs
//     exactly unchanged, in both directions (a perturbed dout at row 1's end must leave row 2's first 3 dqkv unchanged),
//     while row 1 itself does change — so the test would notice if the perturbation did nothing at all.
//
// Gates: JT_TOL_FWD (1e-5) forward, JT_TOL_BWD (1e-4) backward. Both sides are fp32 and each output element is the same
// four products in the same order, so the measured error is a few 1e-7. On the null device every CHECK_* prints
// "launch only" and the "did the perturbation change anything" checks do the same (there is nothing to compare).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h"
#include "conv.h"        // kernels/conv.h — conv_fwd, conv_bwd
#include "../ref/conv.h" // ref/conv.h — the CPU references (same basename, other directory)

enum { JT_MAXTOK = 8192, JT_MAXCH = 6144, JT_MAXROWS = 4 }; // the plan's 4-row batch is 2516 tokens

static float *fnew(size_t n) {
  float *p = (float *)calloc(n ? n : 1, sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "k_conv: out of memory (%zu floats)\n", n);
    exit(1);
  }
  return p;
}

// A boolean check for the "the perturbation really did change something" half of the row-boundary test. The value checks
// go through CHECK_REL/CHECK_ABS; those print launch-only on the null device, and so does this one, because an
// unexecuted kernel leaves nothing to compare.
static void check_true(const char *name, int cond) {
  if (jt_is_launch_only()) {
    jt_report_launch_only(name);
    return;
  }
  if (cond) {
    printf("ok  %s\n", name);
  } else {
    printf("FAIL %s\n", name);
    jt_fail_inc();
  }
  fflush(stdout);
}

static float max_diff(const float *a, const float *b, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; i++) {
    const float d = fabsf(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

// ---- forward and backward against the references ----------------------------------------------------------------
static void case_conv(const char *tag, const int *lens, int nrows, int channels, uint32_t seed) {
  int off[JT_MAXROWS + 1], row_of[JT_MAXTOK];
  int n = 0;
  for (int b = 0; b < nrows; b++) n += lens[b];
  if (nrows > JT_MAXROWS || n > JT_MAXTOK || channels > JT_MAXCH || channels < 1) {
    printf("k_conv: %s does not fit the fixture\n", tag);
    jt_fail_inc();
    return;
  }
  n = jt_ragged(lens, nrows, off, row_of);

  const size_t ne = (size_t)n * (size_t)channels;
  const size_t nw = (size_t)channels * JT_CONV_K;
  char nm[128];

  float *qkv = fnew(ne), *dout = fnew(ne), *wconv = fnew(nw);
  float *pre_ref = fnew(ne), *out_ref = fnew(ne), *dqkv_ref = fnew(ne);
  float *got_pre = fnew(ne), *got_out = fnew(ne), *got_dqkv = fnew(ne);
  rnd_fill(qkv, ne, -1.5, 1.5, seed);
  rnd_fill(dout, ne, -1.5, 1.5, seed ^ 0x9e3779b9u);
  rnd_fill(wconv, nw, -1.0, 1.0, seed ^ 0x5bd1e995u);

  ref_conv_fwd(qkv, wconv, row_of, off, pre_ref, out_ref, n, channels);
  ref_conv_bwd(dout, pre_ref, wconv, row_of, off, dqkv_ref, n, channels);

  float *qkvd = dev_upload(qkv, ne), *doutd = dev_upload(dout, ne), *wconvd = dev_upload(wconv, nw);
  float *pred = (float *)dev_alloc(ne * sizeof(float)), *outd = (float *)dev_alloc(ne * sizeof(float));
  float *dqkvd = (float *)dev_alloc(ne * sizeof(float));
  int *offd = (int *)dev_upload_bytes(off, (size_t)(nrows + 1) * sizeof *offd);
  int *rowd = (int *)dev_upload_bytes(row_of, (size_t)n * sizeof *rowd);

  conv_fwd(qkvd, wconvd, rowd, offd, pred, outd, n, channels, 0);
  conv_bwd(doutd, pred, wconvd, rowd, offd, dqkvd, n, channels, 0);

  dev_download(pred, got_pre, ne);
  dev_download(outd, got_out, ne);
  dev_download(dqkvd, got_dqkv, ne);

  snprintf(nm, sizeof nm, "conv_fwd %s n=%d ch=%d pre", tag, n, channels);
  CHECK_REL(nm, got_pre, pre_ref, ne, JT_TOL_FWD);
  snprintf(nm, sizeof nm, "conv_fwd %s n=%d ch=%d out", tag, n, channels);
  CHECK_REL(nm, got_out, out_ref, ne, JT_TOL_FWD);
  snprintf(nm, sizeof nm, "conv_bwd %s n=%d ch=%d dqkv", tag, n, channels);
  CHECK_REL(nm, got_dqkv, dqkv_ref, ne, JT_TOL_BWD);

  dev_free(qkvd), dev_free(doutd), dev_free(wconvd), dev_free(pred), dev_free(outd), dev_free(dqkvd);
  dev_free(offd), dev_free(rowd);
  free(qkv), free(dout), free(wconv), free(pre_ref), free(out_ref), free(dqkv_ref);
  free(got_pre), free(got_out), free(got_dqkv);
}

// ---- the ragged row-boundary check (plan M2.4) -----------------------------------------------------------------
// rows {5,7,6}: row 1 is tokens [5,12), so its last 3 tokens are 9,10,11 and row 2 is [12,18), its first 3 12,13,14.
// Both directions are checked, because the forward's `qkv[tl-3+k] < 0` and the backward's `dpre[tl+3-k] >= L` are
// different bounds and getting one right says nothing about the other.
static void case_row_boundary(void) {
  static const int lens[3] = {5, 7, 6};
  const int nrows = 3, channels = 17;
  int off[JT_MAXROWS + 1], row_of[JT_MAXTOK];
  const int n = jt_ragged(lens, nrows, off, row_of);
  const size_t ne = (size_t)n * channels;
  const size_t nw = (size_t)channels * JT_CONV_K;

  const int row1_end = off[2]; // row 2's first token (row 1 is off[1] .. off[2]-1)
  const int nl = 3; // the last 3 tokens of row 1 / the first 3 of row 2
  const float atol = 1e-9f;

  float *q1 = fnew(ne), *q2 = fnew(ne), *wconv = fnew(nw);
  float *d1 = fnew(ne), *d2 = fnew(ne);
  rnd_fill(q1, ne, -1.0, 1.0, 0x51ed270bu);
  memcpy(q2, q1, ne * sizeof *q2);
  rnd_fill(wconv, nw, -1.0, 1.0, 0x1b873593u);
  // Randomise row 1's last 3 tokens only. The values are far outside the original range so every pre/out there moves.
  rnd_fill(q2 + (size_t)(row1_end - nl) * channels, (size_t)nl * channels, -6.0, 6.0, 0x2545f491u);

  int *offd = (int *)dev_upload_bytes(off, (size_t)(nrows + 1) * sizeof *offd);
  int *rowd = (int *)dev_upload_bytes(row_of, (size_t)n * sizeof *rowd);
  float *wconvd = dev_upload(wconv, nw);
  float *q1d = dev_upload(q1, ne), *q2d = dev_upload(q2, ne);
  float *p1 = (float *)dev_alloc(ne * sizeof(float)), *o1 = (float *)dev_alloc(ne * sizeof(float));
  float *p2 = (float *)dev_alloc(ne * sizeof(float)), *o2 = (float *)dev_alloc(ne * sizeof(float));
  conv_fwd(q1d, wconvd, rowd, offd, p1, o1, n, channels, 0);
  conv_fwd(q2d, wconvd, rowd, offd, p2, o2, n, channels, 0);

  float *hp1 = fnew(ne), *ho1 = fnew(ne), *hp2 = fnew(ne), *ho2 = fnew(ne);
  dev_download(p1, hp1, ne), dev_download(o1, ho1, ne);
  dev_download(p2, hp2, ne), dev_download(o2, ho2, ne);

  const size_t row2_start = (size_t)row1_end * channels;                  // row 2's first 3 tokens, all channels
  const size_t row2_span = (size_t)nl * channels;
  const size_t row1_tail = (size_t)(row1_end - nl) * channels;            // the 3 tokens that were randomised

  CHECK_ABS("conv_fwd row2 first 3 pre unchanged", hp2 + row2_start, hp1 + row2_start, row2_span, atol);
  CHECK_ABS("conv_fwd row2 first 3 out unchanged", ho2 + row2_start, ho1 + row2_start, row2_span, atol);
  check_true("conv_fwd row1 last 3 changed",
             max_diff(hp2 + row1_tail, hp1 + row1_tail, row2_span) > 1e-2f);

  // Backward: perturb dout at row 1's last 3 tokens; row 2's first 3 dqkv must be untouched, row 1's own must not be.
  rnd_fill(d1, ne, -1.0, 1.0, 0x68e31da4u);
  memcpy(d2, d1, ne * sizeof *d2);
  rnd_fill(d2 + row1_tail, row2_span, -6.0, 6.0, 0x9e3779b1u);
  float *d1d = dev_upload(d1, ne), *d2d = dev_upload(d2, ne);
  float *g1 = (float *)dev_alloc(ne * sizeof(float)), *g2 = (float *)dev_alloc(ne * sizeof(float));
  conv_bwd(d1d, p1, wconvd, rowd, offd, g1, n, channels, 0);
  conv_bwd(d2d, p1, wconvd, rowd, offd, g2, n, channels, 0);
  float *hg1 = fnew(ne), *hg2 = fnew(ne);
  dev_download(g1, hg1, ne), dev_download(g2, hg2, ne);

  CHECK_ABS("conv_bwd row2 first 3 dqkv unchanged", hg2 + row2_start, hg1 + row2_start, row2_span, atol);
  check_true("conv_bwd row1 last 3 changed", max_diff(hg2 + row1_tail, hg1 + row1_tail, row2_span) > 1e-2f);

  // The row's own first token must still get a gradient from its own block (dqkv there sums dpre at tokens 0..3 of the
  // row); a backward that zeroed row starts to "be safe" would pass the row-boundary check and fail this one.
  float amax = 0.0f;
  for (size_t i = 0; i < (size_t)channels; i++) amax = fmaxf(amax, fabsf(hg1[row2_start + i]));
  check_true("conv_bwd row2 first token non-zero", amax > 1e-6f);

  dev_free(q1d), dev_free(q2d), dev_free(wconvd), dev_free(d1d), dev_free(d2d);
  dev_free(p1), dev_free(o1), dev_free(p2), dev_free(o2), dev_free(g1), dev_free(g2), dev_free(offd), dev_free(rowd);
  free(q1), free(q2), free(wconv), free(d1), free(d2);
  free(hp1), free(ho1), free(hp2), free(ho2), free(hg1), free(hg2);
}

int k_conv(void) {
  static const int one[1] = {1};
  static const int shorty[3] = {2, 3, 1};

  case_conv("single", one, 1, 8, 0x51ed270bu);
  case_conv("short", shorty, 3, 7, 0x1b873593u);
  case_conv("ragged3", jt_lens_ragged3(), JT_RAGGED3_ROWS, 6144, 0x2545f491u);
  case_conv("ragged4", jt_lens_ragged4(), JT_RAGGED4_ROWS, 6144, 0x68e31da4u);
  case_row_boundary();
  return jt_exit_code();
}
