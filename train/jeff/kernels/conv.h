// kernels/conv.h — the GDN causal depthwise conv launch wrappers (plan M2.4; design §3.6, §4.8, §7, §8.1). Host C
// includes this and never sees CUDA syntax (design §11.1); every wrapper is extern "C". The CPU reference of the same op
// is ref/conv.h — same basename on purpose.
//
// Layout, in one place: qkv / pre / out / dout / dqkv are all token-major [n][channels] fp32 (channels = 6144 =
// 3 * 2048 for in_proj_qkv, design §3.6); wconv is the checkpoint's [6144, 1, 4] weight flattened to [channels][4], so
// wconv[ch*4 + k] multiplies qkv[t - 3 + k, ch]; row_of[t] is t's row and row_off[b] its first token, so the row-local
// index is t - row_off[row_of[t]] (design §7: taps never reach into the previous row, and the backward's never reach
// into the next one).
#ifndef JT_CONV_H
#define JT_CONV_H

#include <cuda_runtime.h>

#ifndef JT_CONV_K
#define JT_CONV_K 4
#endif

#ifdef __cplusplus
extern "C" {
#endif

// pre = conv(qkv), out = silu(pre), both [n][channels]. A no-op when n == 0. The forward keeps pre because the backward
// needs silu's derivative there (design §4.8).
void conv_fwd(const float *qkv, const float *wconv, const int *row_of, const int *row_off, float *pre, float *out,
              int n, int channels, cudaStream_t s);

// dqkv from dout (design §4.8): dpre = dout * dsilu(pre), then the transposed conv dqkv[t,ch] =
// sum_k wconv[ch*4+k] * dpre[t + 3 - k, ch] inside the token's own row.
void conv_bwd(const float *dout, const float *pre, const float *wconv, const int *row_of, const int *row_off,
              float *dqkv, int n, int channels, cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif // JT_CONV_H
