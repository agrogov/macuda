// ref/norm.h — the CPU references of the norm kernels (plan M2.2; design §4.3, §4.4, §4.8, §8.1).
//
// Written before the kernels (plan rule 2): test/k_norm.c compares kernels/norm.cu against these, and test/grad_norm.c
// differentiates them in double (-DJT_REAL_DOUBLE). Every function takes `real *` (ref/real.h) and its dimensions as
// parameters — rows and d — so that the gradcheck can shrink them and still difference every element.
//
// Layout: row-major [rows][d]. A "row" is one token for the token-major norms (the final norm, d = 1024; q_norm/k_norm,
// d = 256) and one (token, head) pair for the GDN norms (l2norm and the gated norm, d = 128, rows = tokens * 16).
// The row length is therefore always a parameter, never implied by the row index (design §8.1).
//
// eps is the model's rms_norm_eps, 1e-6 for every norm in it (design §3.1), so it is a constant here and not an
// argument. The only difference between the two norms below is sum vs mean: RMSNorm averages x^2 over the row, l2norm
// sums it (design §3.3).
#ifndef JT_REF_NORM_H
#define JT_REF_NORM_H

#include <stddef.h>

#include "real.h"

#define JT_NORM_EPS 1e-6

// ---- RMSNorm, zero-centred (design §4.3; q_norm/k_norm l.749 with d = 256, final norm l.861 with d = 1024) ----------
// r[b] = 1 / sqrt(mean(x[b]^2) + eps),  y[b,i] = x[b,i] * r[b] * wp[i],  wp[i] = zero_centred ? 1 + w[i] : w[i].
// The forward saves r, one value per row: the backward needs it (design §4.3) and y alone would lose it.
// zero_centred is what the pitfall in plan M2.2 is about: the model's RMSNorm multiplies by (1 + w) (w is initialised
// to zeros, l.841), the GDN gated norm multiplies by w (initialised to ones, l.218). Both are tested.
void ref_rmsnorm_fwd(const real *x, const real *w, real *y, real *r, int rows, int d, int zero_centred);

// gy[b,i] = dy[b,i] * wp[i],  dx[b,i] = r * (gy - x * r^2 * mean(gy * x)),  with the r the forward saved and the mean
// taken over the row (length d). The weight is frozen (design §4.3), so there is no dw.
void ref_rmsnorm_bwd(const real *x, const real *w, const real *dy, const real *r, real *dx, int rows, int d,
                     int zero_centred);

// ---- Gated RMSNorm, per GDN head (design §4.4; l.218) ---------------------------------------------------------------
// n = x * r,  sg = silu(z),  y = wp * n * sg,  r = 1 / sqrt(mean(x^2) + eps). wp is (1 + w) or w, see above.
void ref_gated_rmsnorm_fwd(const real *x, const real *w, const real *z, real *y, real *r, int rows, int d,
                           int zero_centred);

// dz = dy * wp * n * dsilu(z),  gn = dy * wp * sg,  dx = r * (gn - n * mean(gn * n)),
// with dsilu(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z))).
void ref_gated_rmsnorm_bwd(const real *x, const real *w, const real *z, const real *dy, const real *r, real *dx,
                           real *dz, int rows, int d, int zero_centred);

// ---- L2 norm, per GDN head (design §3.3 l.294, backward §4.8) -------------------------------------------------------
// r[b] = 1 / sqrt(sum(x[b]^2) + eps)   (sum, not mean),  y = x * r * post_scale.
// post_scale is 1/sqrt(128) for q (design §3.6 passes the head-dim scale here, not through a separate multiply) and
// 1 for k; the backward must use the same value, so it is an argument of both directions.
void ref_l2norm_fwd(const real *x, real post_scale, real *y, real *r, int rows, int d);

// dqn = dy * post_scale,  qn = x * r,  dx = r * (dqn - qn * (qn . dqn))    (dot product over the row, no mean).
void ref_l2norm_bwd(const real *x, const real *dy, const real *r, real post_scale, real *dx, int rows, int d);

#endif // JT_REF_NORM_H
