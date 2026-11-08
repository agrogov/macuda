// ref/conv.h — CPU references of the GDN causal depthwise conv (plan M2.4; design §3.6 "causal depthwise conv", §4.8
// "the transposed causal conv", §7 row boundaries, §8.1 launch shape).
//
// Written before kernels/conv.cu (plan rule 2): test/k_conv.c compares the kernel against these, and test/grad_conv.c
// differentiates ref_conv_fwd in double (-DJT_REAL_DOUBLE) to check that ref_conv_bwd is its transpose. Every function
// takes `real *` (ref/real.h) and its dimensions as parameters so the gradcheck can shrink them.
//
// What it is. One channel at a time, a width-4 causal (left-padded) depthwise conv over time, then silu:
//
//   pre[t, ch] = sum_{k=0..3} wconv[ch*4 + k] * qkv[t - 3 + k, ch]        qkv[<0] = 0 at the row's start
//   out[t, ch] = pre[t, ch] * sigmoid(pre[t, ch])                          = silu(pre)
//
// The checkpoint's weight is [6144, 1, 4] (config: linear_conv_kernel_dim 4, in_proj_qkv output 6144 = 3 * 2048), so
// wconv[ch*4 + k] multiplies x[t - 3 + k] exactly as `F.conv1d(..., padding=3)` truncated to L does (torch's conv1d is
// a cross-correlation; padding 3 on the left is the causality).
//
// Two rules the whole file exists for:
//   * the taps never cross a row start (design §7): qkv[t_local - 3 + k] with t_local = t - off[row] and a negative
//     t_local - 3 + k is zero, never the previous row's token. The packed batch has no padding, so the token before a
//     row's first token belongs to another meeting.
//   * the backward is the transposed conv and it must not cross a row end either: dpre[t_local + 3 - k] past the
//     row's last token is zero.
//
// Frozen weight: wconv is a checkpoint tensor with no LoRA (design §2.1), so there is no dwconv — only dqkv.
#ifndef JT_REF_CONV_H
#define JT_REF_CONV_H

#include <stddef.h>

#include "real.h"

// The conv's width (config's linear_conv_kernel_dim, design §3.6: "sum_{k=0..3}").
#ifndef JT_CONV_K
#define JT_CONV_K 4
#endif

// ---- forward (design §3.6) ----------------------------------------------------------------------------------------------
// qkv: [n][channels] token-major; wconv: [channels][JT_CONV_K] (the checkpoint's [6144,1,4] flattened); row_of: [n] the
// row of each token; row_off: [nrows + 1] that row's first token. pre receives the conv output and out its silu — the
// backward needs pre, and the recurrence needs out, so both are written in one pass.
void ref_conv_fwd(const real *qkv, const real *wconv, const int *row_of, const int *row_off, real *pre, real *out,
                  int n, int channels);

// ---- backward (design §4.8) ----------------------------------------------------------------------------------------------
// dout is dLoss/dout ([n][channels]); pre is the forward's conv output (kept, so silu is not recomputed from qkv);
// dqkv is [n][channels]. Inside one row: dpre = dout * dsilu(pre), dqkv[t,ch] = sum_k wconv[ch*4+k] * dpre[t + 3 - k,ch],
// dpre past the row's last token taken as zero — the exact adjoint of the forward loop above.
void ref_conv_bwd(const real *dout, const real *pre, const real *wconv, const int *row_of, const int *row_off,
                  real *dqkv, int n, int channels);

#endif // JT_REF_CONV_H
