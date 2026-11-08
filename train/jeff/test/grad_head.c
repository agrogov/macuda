// test/grad_head.c — the `head` group of gradcheck (plan M2.0/M2.5): the finite-difference check of the head's three
// gradients (design §4.1).
//
// What is differentiated, all against L = sum_r loss_out[r] (the batch loss train.py computes as
// `(per * wt).sum() / gw`):
//
//   1. dlogits   — x = the masked logits [B][n], y = loss_out [B] through ref_ce_head. Its numeric gradient is
//                  (w/W) * (p * sum(t) - t), dead options included (0 there).
//   2. dR        — x = the readout weight R [n][d], y = loss_out through ref_head_readout_fwd + ref_ce_head. Its
//                  numeric gradient is dR = dlogits^T z, as design §4.1 writes it.
//   3. dz        — x = the hidden states z [B][d], same composite. Its numeric gradient is dz = R^T dlogits.
//
// Sizes are shrunk (n = 4 options, d = 4 hidden) so every element is perturbed while staying cheap; the gate
// (JT_TOL_GRAD) does not depend on the size. One row has a single live option and another is fully live, so the dead
// half of the mask is exercised on both sides. The targets are deliberately not normalised: train.py only guarantees
// |sum(t) - 1| <= 1e-4, and the exact gradient carries the sum(t) factor.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h"
#include "../ref/head.h"

#define HEAD_B 3
#define HEAD_N 4
#define HEAD_D 4

typedef struct {
  int B, n, d;
  const real *z, *R;
  const unsigned char *live; // [B][n], nonzero = live
  const real *targets, *wrow;
  real wgroup;
  real *logits; // [B][n] scratch: the readout's forward output inside f_readout / f_hidden
} HeadCtx;

// x = the masked logits.
static void f_logits(void *v, const real *x, real *y) {
  const HeadCtx *c = (const HeadCtx *)v;
  ref_ce_head(x, c->n, c->targets, c->wrow, c->wgroup, c->B, y, NULL, NULL);
}

// x = the readout R: logits = z R^T (dead options masked), then the loss.
static void f_readout(void *v, const real *x, real *y) {
  const HeadCtx *c = (const HeadCtx *)v;
  ref_head_readout_fwd(c->z, x, c->live, c->B, c->n, c->d, c->logits);
  ref_ce_head(c->logits, c->n, c->targets, c->wrow, c->wgroup, c->B, y, NULL, NULL);
}

// x = the hidden states z: logits = z R^T (dead options masked), then the loss.
static void f_hidden(void *v, const real *x, real *y) {
  const HeadCtx *c = (const HeadCtx *)v;
  ref_head_readout_fwd(x, c->R, c->live, c->B, c->n, c->d, c->logits);
  ref_ce_head(c->logits, c->n, c->targets, c->wrow, c->wgroup, c->B, y, NULL, NULL);
}

static real *alloc(size_t n) {
  real *p = (real *)malloc((n ? n : 1) * sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "grad_head: out of memory (%zu elements)\n", n);
    exit(1);
  }
  return p;
}

int grad_head(void) {
  const size_t nlog = (size_t)HEAD_B * HEAD_N, nR = (size_t)HEAD_N * HEAD_D, nz = (size_t)HEAD_B * HEAD_D;
  real *logits = alloc(nlog), *targets = alloc(nlog), *ones = alloc(HEAD_B);
  real *dlogits = alloc(nlog), *loss = alloc(HEAD_B);
  real *z = alloc(nz), *R = alloc(nR), *dz = alloc(nz), *dR = alloc(nR);
  real *wrow = alloc(HEAD_B);
  unsigned char *live = (unsigned char *)malloc(nlog);
  if (live == NULL) {
    fprintf(stderr, "grad_head: out of memory (mask)\n");
    return JT_FAIL;
  }

  // Row 0: two live options. Row 1: all four. Row 2: a single live option (its dlogits are all 0: p = 1, t = 1).
  static const unsigned char live_rows[HEAD_B] = {2, 4, 1};
  rnd_fill(targets, nlog, 0.0, 1.0, 22u);
  rnd_fill(z, nz, -1.0, 1.0, 23u);
  rnd_fill(R, nR, -1.0, 1.0, 24u);
  rnd_fill(wrow, HEAD_B, 0.5, 2.0, 25u);
  for (int r = 0; r < HEAD_B; r++) {
    for (int j = 0; j < HEAD_N; j++) {
      const size_t i = (size_t)r * HEAD_N + j;
      live[i] = (unsigned char)(j < live_rows[r]);
      if (!live[i]) {
        targets[i] = REAL(0); // train.py's `target` is 0 past a row's option count; the logit is masked below
      }
    }
  }
  // The logits are the readout's own output (train.py's option_logits), masked where the option is dead. Every check
  // then differentiates the loss at the same point: dlogits, dR and dz must agree with the finite differences of
  // exactly this forward.
  ref_head_readout_fwd(z, R, live, HEAD_B, HEAD_N, HEAD_D, logits);
  real wgroup = REAL(0);
  for (int r = 0; r < HEAD_B; r++) {
    wgroup += wrow[r];
    ones[r] = REAL(1); // the cotangent that turns the B losses into L = sum_r loss_out[r]
  }

  HeadCtx c = {HEAD_B, HEAD_N, HEAD_D, z, R, live, targets, wrow, wgroup, logits};

  // 1. dlogits from the analytic reference, against the numeric gradient of the losses.
  ref_ce_head(logits, HEAD_N, targets, wrow, wgroup, HEAD_B, loss, dlogits, NULL);
  jt_grad_check("head/dlogits", f_logits, &c, logits, nlog, dlogits, ones, HEAD_B);

  // 2 + 3. dR and dz: one call of design §4.1's two readout lines, then perturb R and z in turn.
  memset(dR, 0, nR * sizeof *dR);
  ref_head_readout_bwd(z, R, dlogits, HEAD_B, HEAD_N, HEAD_D, dz, dR);
  jt_grad_check("head/readout_dR", f_readout, &c, R, nR, dR, ones, HEAD_B);
  jt_grad_check("head/readout_dz", f_hidden, &c, z, nz, dz, ones, HEAD_B);

  free(logits);
  free(targets);
  free(ones);
  free(dlogits);
  free(loss);
  free(z);
  free(R);
  free(dz);
  free(dR);
  free(wrow);
  free(live);
  return jt_exit_code();
}
