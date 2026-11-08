// kernels/rope.h — the RoPE launch wrappers and the table builder (plan M2.4; design §3.3, §8.1). Host C includes this
// and never sees CUDA syntax (design §11.1); every wrapper is extern "C". The CPU reference of the same op is
// ref/rope.h — same basename on purpose.
//
// Layout, in one place: x is token-major [n][heads * head_dim] (the packed batch, design §7); a head is head_dim floats
// of which the first JT_ROPE_DIM = 64 are rotated and the rest pass through; the position of token t is
// t - row_off[row_of[t]], never the packed index. cos and sin are the [lmax][64] tables rope_tables() builds, already on
// the device (design §3.3: precomputed once, passed as pointers — the shim has no cudaMemcpyToSymbol, design §12).
#ifndef JT_ROPE_H
#define JT_ROPE_H

#include <cuda_runtime.h>

#ifndef JT_ROPE_DIM
#define JT_ROPE_DIM 64
#endif
#ifndef JT_ROPE_HALF
#define JT_ROPE_HALF 32
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Build the [lmax][64] cos and sin tables on the host, in fp32, exactly as design §3.3 says: inv_freq[k] =
// 1e7^(-2k/64) in fp32 (torch's `1.0 / (theta ** (arange(0, 64, 2).float() / 64))`), ang = p * inv_freq in fp32, then
// cos/sin in fp32, with slot j and slot j + 32 carrying frequency j (torch's `torch.cat((f, f), -1)`). Upload them with
// cudaMemcpy before the first launch. lmax = the longest row's length (1026 for the real data).
void rope_tables(float *cos, float *sin, int lmax);

// In place on the first 64 dims of every head. A no-op when n == 0 or head_dim < 64.
void rope_fwd(float *x, const int *row_off, const int *row_of, int n, int heads, int head_dim, const float *cos,
              const float *sin, cudaStream_t s);

// The transpose of the forward (the same rotation with sin negated), in place on the cotangent dx.
void rope_bwd(float *dx, const int *row_off, const int *row_of, int n, int heads, int head_dim, const float *cos,
              const float *sin, cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif // JT_ROPE_H
