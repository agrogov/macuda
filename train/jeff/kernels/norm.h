// kernels/norm.h — the norm kernels' launch wrappers (plan M2.2; design §8.1). Host C includes this and never sees CUDA
// syntax (design §11.1); every wrapper here is extern "C". The CPU references of the same ops are ref/norm.h — the file
// has the same basename there on purpose, so a .cu includes the one beside it and a reference test includes its own.
//
// All of these are row kernels: [rows][d], one row reduced at a time, in fp32 (design §8.1 "Always fp32"). The
// reductions are warp shuffles plus shared memory, never atomicAdd, so the result does not depend on the block
// schedule. Launch shapes are the kernel's business, not the caller's.
#ifndef JT_NORM_H
#define JT_NORM_H

#include <cuda_runtime.h>

// rms_norm_eps as the device sees it: 1e-6 for every norm in the model (design §3.1). Not a parameter; the reference
// has the same constant (ref/norm.h).
#define JT_NORM_EPS_F 1e-6f

#ifdef __cplusplus
extern "C" {
#endif

// ---- RMSNorm (design §4.3) ------------------------------------------------------------------------------------------
// x, w, y: [rows][d]; w is the [d] frozen weight; r: [rows] out, the 1/sqrt(mean(x^2)+eps) the forward saves for the
// backward. y[b,i] = x[b,i] * r[b] * (zero_centred ? 1 + w[i] : w[i]).
// d = 1024 is the final norm (one 256-thread block per row); d = 256 is q_norm/k_norm (one warp per row).
void rmsnorm_fwd(const float *x, const float *w, float *y, float *r, int rows, int d, int zero_centred,
                 cudaStream_t s);

// dx[b,i] = r * (gy - x * r^2 * mean(gy * x)) with gy = dy * wp and mean over the row (design §4.3). `r` is the saved
// array from the forward. The weight is frozen, so there is no dw.
void rmsnorm_bwd(const float *x, const float *w, const float *dy, const float *r, float *dx, int rows, int d,
                 int zero_centred, cudaStream_t s);

// ---- Gated RMSNorm, per GDN head (design §4.4) ----------------------------------------------------------------------
// x, w, z, y: [rows][d] with rows = tokens * 16 (one row is one (token, head) pair) and d = 128; w is the [d] weight.
// y[b,i] = wp[i] * (x[b,i] * r[b]) * silu(z[b,i]), r[b] = 1/sqrt(mean(x^2)+eps), r saved into r[b] (same array as above).
void gated_rmsnorm_fwd(const float *x, const float *w, const float *z, float *y, float *r, int rows, int d,
                       int zero_centred, cudaStream_t s);

// dx and dz from dy (design §4.4): dz = dy * wp * n * dsilu(z), dx = r * (gn - n * mean(gn * n)) with n = x * r and
// gn = dy * wp * silu(z). Both outputs are written; `r` is the forward's saved array.
void gated_rmsnorm_bwd(const float *x, const float *w, const float *z, const float *dy, const float *r, float *dx,
                       float *dz, int rows, int d, int zero_centred, cudaStream_t s);

// ---- L2 norm, per GDN head (design §3.3 l.294, §4.8) ----------------------------------------------------------------
// r[b] = 1/sqrt(sum(x[b]^2) + eps)  (sum, not mean);  y[b,i] = x[b,i] * r[b] * post_scale. post_scale carries the
// model's 1/sqrt(128) on q and 1 on k (design §3.6), so the caller does not need a second pass over y.
void l2norm_fwd(const float *x, float post_scale, float *y, float *r, int rows, int d, cudaStream_t s);

// dqn = dy * post_scale, qn = x * r, dx = r * (dqn - qn * (qn . dqn))   (design §4.8).
void l2norm_bwd(const float *x, const float *dy, const float *r, float post_scale, float *dx, int rows, int d,
                cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif // JT_NORM_H
