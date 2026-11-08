// ref/head.h — the CPU oracle for the decision head (plan M2.5, design §4.1).
//
// The head is `p = softmax(over the n live options of logits)`, per-row soft cross-entropy against the teacher's
// target, and the two gradients the loss sends back: dlogits, and the readout's dR / dz. In the trainer the logits
// come from a GEMM (`logits = z R^T`); plan M2.5 says the kernel does only the softmax part, so `ce_head` below takes
// already-masked logits, exactly as train.py's `option_logits` hands them over: dead options set to -1e9 and their
// target entry 0 (`logits.masked_fill(~live, -1e9)`, CLM-8B's tools/jeff_lora/train.py:143-156, which wins over the
// design where they disagree). `ref_head_readout_fwd/bwd` are the two GEMM steps around it — the oracle M5's readout
// and D§4.1's `dR += dlogits (x) z`, `dz = R^T dlogits` are checked against. They have no GPU counterpart of their own
// on purpose: they are cuBLAS calls in the trainer.
//
// The same source builds twice (ref/real.h): float for the GPU test, double for the gradchecks.
#ifndef JT_REF_HEAD_H
#define JT_REF_HEAD_H

#include "real.h"

#ifdef __cplusplus
extern "C" {
#endif

// What train.py's masked_fill uses for a dead option. Its p is exactly 0 in fp32 (exp underflows), and its dlogits entry
// is 0 because both p and the target are 0.
#define REF_HEAD_DEAD (-1e9)

// Per-row soft cross-entropy (design §4.1, train.py's `soft_ce`), for B rows of n_opts options:
//
//   p[r][j]      = softmax_j (logits[r][j])                    (dead entries hold -1e9, so p = 0 there)
//   ce[r]        = -sum_j targets[r][j] * log p[r][j]
//   loss_out[r]  = (wrow[r] / wgroup) * ce[r]                   the batch loss is sum_r loss_out[r]
//   dlogits[r][j]= (wrow[r] / wgroup) * (p[r][j] * sum_k targets[r][k] - targets[r][j])
//   probs_out[r][j] = p[r][j]                                   NULL: not written
//
// `dlogits` and `probs_out` may be NULL (an evaluation wants the loss and the probabilities, not a gradient; the
// forward half of a gradcheck wants the loss only). logits, targets, wrow and loss_out are always required.
//
// The `sum(t)` factor is the design's, not an assumption that the targets sum to 1: train.py only guarantees
// |sum(t) - 1| <= 1e-4, and the exact derivative of `-t . log_softmax` carries it. All four arrays are [B][n_opts],
// row-major. j >= n_opts does not exist here; the caller slices the readout to the batch's widest row (train.py's
// `[:, :live.shape[1]]`).
// The kernel of the same name (kernels/head.h) takes exactly the same arguments, so the GPU test compares them field by
// field and M5 calls one or the other with the same code.
void ref_ce_head(const real *logits, int n_opts, const real *targets, const real *wrow, real wgroup, int B,
                 real *loss_out, real *dlogits, real *probs_out);

// logits[r][j] = sum_k R[j][k] * z[r][k] for a live option, REF_HEAD_DEAD for a dead one. z is [B][d], R is [n][d]
// (the readout weight's torch layout [out, in], no bias), logits is [B][n]. `live` is [B][n] and nonzero means live;
// pass NULL to keep every option.
void ref_head_readout_fwd(const real *z, const real *R, const unsigned char *live, int B, int n, int d, real *logits);

// The backward of ref_head_readout_fwd plus ref_ce_head (design §4.1's last two lines):
//   dz[r][k]  = sum_j R[j][k] * dlogits[r][j]          [B][d], overwritten
//   dR[j][k] += sum_r dlogits[r][j] * z[r][k]          [n][d], accumulated (the readout's gradient buffer is shared)
void ref_head_readout_bwd(const real *z, const real *R, const real *dlogits, int B, int n, int d, real *dz,
                          real *dR);

#ifdef __cplusplus
}
#endif

#endif // JT_REF_HEAD_H
