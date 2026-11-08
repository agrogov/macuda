// kernels/head.h — the decision head's softmax and loss (plan M2.5, design §4.1). Host C includes this; the CPU oracle is
// ref/head.h and takes the same arguments.
#ifndef JT_HEAD_H
#define JT_HEAD_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Per-row soft cross-entropy over n_opts options for B rows (one block per row):
//
//   p[r][j]        = softmax_j(logits[r][j])            dead options hold -1e9 (train.py's masked_fill) so p = 0 there
//   loss_out[r]    = (wrow[r] / wgroup) * (-sum_j targets[r][j] * log p[r][j])
//   dlogits[r][j]  = (wrow[r] / wgroup) * (p[r][j] * sum_k targets[r][k] - targets[r][j])
//   probs_out[r][j]= p[r][j]
//
// so `sum_r loss_out[r]` is train.py's batch loss `(per * wt).sum() / gw` and dlogits is its exact gradient. All four
// arrays are [B][n_opts] fp32, row-major; logits and targets and wrow and loss_out are required, dlogits and probs_out
// may be NULL (an evaluation wants the loss and the probabilities; a loss-only probe wants neither).
//
// The logits come from the readout GEMM (design §4.1, `logits = z R^T`, sliced to the batch's widest row and masked);
// this kernel does only the softmax part, exactly as plan M2.5 says. 1e9 is assumed to be well below the live logits'
// magnitude, which holds because the readout is fp32 and the loss is only ever read at float precision.
void ce_head(const float *logits, int n_opts, const float *targets, const float *wrow, float wgroup, int B,
             float *loss_out, float *dlogits, float *probs_out, cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif // JT_HEAD_H
