// kernels/conv.cu — the GDN causal depthwise conv (plan M2.4; design §3.6 the forward, §4.8 the transposed conv,
// §7 row boundaries, §8.1 "one thread per (token, channel)").
//
// Only the __global__ kernels and the extern "C" launch wrappers live here; the C side includes conv.h and never sees
// CUDA syntax (design §11.1). ref/conv.c is the same arithmetic and test/k_conv.c compares them, including the ragged
// row-boundary check (randomising row 1's last 3 tokens must leave row 2's first 3 outputs alone).
//
// Launch shape: grid.x = n, one 256-thread block per token, the block's threads stride over the channels (24 iterations
// at 6144). A block therefore reads one token's taps at a time and consecutive threads read consecutive channels, so
// every tap is a coalesced 256-float load; the pointer arithmetic per tap is one multiply, not a div/mod over a flat
// n*channels index. No shared memory, no atomics, no reduction: each output element is one thread's own sum in the
// reference's order (k = 0..3), so the result does not depend on the schedule.
#include <cuda_runtime.h>
#include <stddef.h>

#include "conv.h"
#include "util.h" // JT_SYNC_DEBUG (a no-op unless -DJT_DEBUG)

// silu and its derivative, as ref/conv.c has them (expf, not __expf: these kernels are not the bottleneck and are
// compared against the reference at 1e-4).
__device__ __forceinline__ float jt_sig(float u) { return 1.0f / (1.0f + expf(-u)); }

// dpre = dout * dsilu(pre) at token t, channel ch (design §4.8: "dpre = dc * dsilu(pre)"). The backward needs it at four
// neighbouring tokens, so it is computed per tap rather than materialised into a scratch buffer — no allocation, and no
// second array for a caller to size.
__device__ __forceinline__ float jt_dpre(const float *dout, const float *pre, int t, int ch, int channels) {
  const size_t i = (size_t)t * (size_t)channels + (size_t)ch;
  const float sig = jt_sig(pre[i]);
  return dout[i] * sig * (1.0f + pre[i] * (1.0f - sig));
}

__global__ void jt_conv_fwd_kernel(const float *qkv, const float *wconv, const int *row_of, const int *row_off,
                                   float *pre, float *out, int channels) {
  const int t = (int)blockIdx.x;
  const int start = row_off[row_of[t]];
  const int tl = t - start; // row-local index; the taps below are relative to it (design §7)

  for (int ch = (int)threadIdx.x; ch < channels; ch += (int)blockDim.x) {
    const float *w = wconv + (size_t)ch * JT_CONV_K;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < JT_CONV_K; k++) {
      const int tl2 = tl - (JT_CONV_K - 1) + k; // qkv[tl - 3 + k]
      if (tl2 >= 0) acc += w[k] * qkv[(size_t)(start + tl2) * (size_t)channels + (size_t)ch];
    }
    const size_t i = (size_t)t * (size_t)channels + (size_t)ch;
    pre[i] = acc;
    out[i] = acc * jt_sig(acc);
  }
}

__global__ void jt_conv_bwd_kernel(const float *dout, const float *pre, const float *wconv, const int *row_of,
                                   const int *row_off, float *dqkv, int channels) {
  const int t = (int)blockIdx.x;
  const int start = row_off[row_of[t]];
  const int len = row_off[row_of[t] + 1] - start; // this row's token count: the taps stop at its end
  const int tl = t - start;

  for (int ch = (int)threadIdx.x; ch < channels; ch += (int)blockDim.x) {
    const float *w = wconv + (size_t)ch * JT_CONV_K;
    float acc = 0.0f;
#pragma unroll
    for (int k = 0; k < JT_CONV_K; k++) {
      const int tl2 = tl + (JT_CONV_K - 1) - k; // dpre[tl + 3 - k]
      if (tl2 < len) acc += w[k] * jt_dpre(dout, pre, start + tl2, ch, channels);
    }
    dqkv[(size_t)t * (size_t)channels + (size_t)ch] = acc;
  }
}

void conv_fwd(const float *qkv, const float *wconv, const int *row_of, const int *row_off, float *pre, float *out,
              int n, int channels, cudaStream_t s) {
  if (n <= 0 || channels <= 0) return;
  jt_conv_fwd_kernel<<<n, 256, 0, s>>>(qkv, wconv, row_of, row_off, pre, out, channels);
  JT_SYNC_DEBUG;
}

void conv_bwd(const float *dout, const float *pre, const float *wconv, const int *row_of, const int *row_off,
              float *dqkv, int n, int channels, cudaStream_t s) {
  if (n <= 0 || channels <= 0) return;
  jt_conv_bwd_kernel<<<n, 256, 0, s>>>(dout, pre, wconv, row_of, row_off, dqkv, channels);
  JT_SYNC_DEBUG;
}
