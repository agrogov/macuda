// kernels/act.cu — activation and gate kernels (plan M2.3, design §3.3, §4.5, §4.7, §4.8, §8.1).
//
// The three ops are elementwise and grid-stride: one thread per element, 256 threads a block, so the launch shape never
// has to fit n and a packed full batch (2516 tokens x 3584) is the same code path as one token. Nothing here reduces and
// nothing is atomic, so the value on the card is fp32 arithmetic in the same order as ref/act.c: only the libm
// functions differ, by at most one ulp.
//
// softplus is torch's (beta = 1, threshold = 20): u > 20 ? u : log1p(exp(u)), and its derivative is 1 on that branch.
// sigmoid is 1 / (1 + exp(-u)): for u very negative exp overflows to +inf and the result is exactly 0, not a NaN.
#include <cuda_runtime.h>

#include "act.h"

// ---- the scalar pieces, device side (ref/act.h has the same five in `real`) ----------------------------------------
static __device__ __forceinline__ float act_sigmoid(float u) { return 1.0f / (1.0f + expf(-u)); }
static __device__ __forceinline__ float act_silu(float u) { return u * act_sigmoid(u); }
static __device__ __forceinline__ float act_dsilu(float u) {
  float s = act_sigmoid(u);
  return s * (1.0f + u * (1.0f - s));
}
static __device__ __forceinline__ float act_softplus(float u) { return u > 20.0f ? u : log1pf(expf(u)); }
static __device__ __forceinline__ float act_dsoftplus(float u) { return u > 20.0f ? 1.0f : act_sigmoid(u); }

// 256 threads a block, at most 65535 blocks: a grid-stride kernel needs no more than that, and the cap keeps the launch
// legal for any n.
static const int ACT_BLOCK = 256;
static inline int act_grid(int n) {
  long g = ((long)n + ACT_BLOCK - 1) / ACT_BLOCK;
  return g > 65535 ? 65535 : (int)g;
}

// ---- silu_mul: the MLP's silu(gate) * up (D§4.5) ---------------------------------------------------------------------
__global__ void silu_mul_fwd_kernel(const float *h1, const float *h2, float *act, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; i < n; i += stride) act[i] = act_silu(h1[i]) * h2[i];
}

__global__ void silu_mul_bwd_kernel(const float *dact, const float *h1, const float *h2, float *dh1, float *dh2,
                                    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    float silu = act_silu(h1[i]);
    float dsilu = act_dsilu(h1[i]);
    dh1[i] = dact[i] * h2[i] * dsilu;
    dh2[i] = dact[i] * silu;
  }
}

extern "C" void k_silu_mul_fwd(const float *h1, const float *h2, float *act, int n, cudaStream_t s) {
  if (n <= 0) return;
  silu_mul_fwd_kernel<<<act_grid(n), ACT_BLOCK, 0, s>>>(h1, h2, act, n);
}

extern "C" void k_silu_mul_bwd(const float *dact, const float *h1, const float *h2, float *dh1, float *dh2, int n,
                               cudaStream_t s) {
  if (n <= 0) return;
  silu_mul_bwd_kernel<<<act_grid(n), ACT_BLOCK, 0, s>>>(dact, h1, h2, dh1, dh2, n);
}

// ---- sigmoid_gate: the attention output gate (D§3.5, D§4.7) ----------------------------------------------------------
__global__ void sigmoid_gate_fwd_kernel(const float *o, const float *gate, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; i < n; i += stride) out[i] = o[i] * act_sigmoid(gate[i]);
}

__global__ void sigmoid_gate_bwd_kernel(const float *dout, const float *o, const float *gate, float *d_o, float *dgate,
                                        int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    float s = act_sigmoid(gate[i]);
    d_o[i] = dout[i] * s;
    dgate[i] = dout[i] * o[i] * s * (1.0f - s);
  }
}

extern "C" void k_sigmoid_gate_fwd(const float *o, const float *gate, float *out, int n, cudaStream_t s) {
  if (n <= 0) return;
  sigmoid_gate_fwd_kernel<<<act_grid(n), ACT_BLOCK, 0, s>>>(o, gate, out, n);
}

extern "C" void k_sigmoid_gate_bwd(const float *dout, const float *o, const float *gate, float *d_o, float *dgate,
                                   int n, cudaStream_t s) {
  if (n <= 0) return;
  sigmoid_gate_bwd_kernel<<<act_grid(n), ACT_BLOCK, 0, s>>>(dout, o, gate, d_o, dgate, n);
}

// ---- gdn_gates: beta and the log decay g (D§3.6, D§4.8) --------------------------------------------------------------
// One thread per (token, head) entry; entry i belongs to head i % heads, so the two [heads] tables are read at that
// index and every block touches all of them (16 floats, in L1 after the first token).
__global__ void gdn_gates_fwd_kernel(const float *bb, const float *aa, const float *A_log, const float *dt_bias,
                                     float *beta, float *g, int n, int heads) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    int h = i % heads;
    float decay = -expf(A_log[h]);
    beta[i] = act_sigmoid(bb[i]);
    g[i] = decay * act_softplus(aa[i] + dt_bias[h]);
  }
}

// dbeta/dg are the cotangents of beta and g. beta comes from the forward (torch's sigmoid backward uses the saved
// output), so dbb needs no exp; daa needs the softplus derivative at u = aa + dt_bias.
__global__ void gdn_gates_bwd_kernel(const float *dbeta, const float *dg, const float *beta, const float *aa,
                                     const float *A_log, const float *dt_bias, float *dbb, float *daa, int n, int heads) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (; i < n; i += stride) {
    int h = i % heads;
    float decay = -expf(A_log[h]);
    float b = beta[i];
    dbb[i] = dbeta[i] * b * (1.0f - b);
    daa[i] = dg[i] * decay * act_dsoftplus(aa[i] + dt_bias[h]);
  }
}

extern "C" void k_gdn_gates_fwd(const float *bb, const float *aa, const float *A_log, const float *dt_bias, float *beta,
                                float *g, int n, int heads, cudaStream_t s) {
  if (n <= 0) return;
  gdn_gates_fwd_kernel<<<act_grid(n), ACT_BLOCK, 0, s>>>(bb, aa, A_log, dt_bias, beta, g, n, heads);
}

extern "C" void k_gdn_gates_bwd(const float *dbeta, const float *dg, const float *beta, const float *aa,
                                const float *A_log, const float *dt_bias, float *dbb, float *daa, int n, int heads,
                                cudaStream_t s) {
  if (n <= 0) return;
  gdn_gates_bwd_kernel<<<act_grid(n), ACT_BLOCK, 0, s>>>(dbeta, dg, beta, aa, A_log, dt_bias, dbb, daa, n, heads);
}
