// ref/rope.h — CPU references of the RoPE kernel (plan M2.4; design §3.3 "RoPE", §4.7, §7 positions, §8.1 launch).
//
// Written before kernels/rope.cu (plan rule 2): test/k_rope.c compares the kernel against these, and test/grad_rope.c
// differentiates ref_rope_fwd in double (-DJT_REAL_DOUBLE) to check that ref_rope_bwd is its transpose. Every function
// takes `real *` (ref/real.h) and its dimensions as parameters, so the gradcheck can shrink them.
//
// What it is (design §3.3). RoPE touches the first JT_ROPE_DIM = 64 dimensions of each 256-dim attention head; dims
// 64..255 are a pass-through. The table is [lmax][64] and built in fp32 exactly as torch builds it
// (`1.0 / (theta ** (arange(0, 64, 2).float() / 64))`, theta = 1e7, then ang = p * inv_freq, then cos/sin), so
// `cos[p][j] == cos[p][j + 32]` and `sin[p][j] == sin[p][j + 32]`; that duplication is torch's `torch.cat((f, f), -1)`
// and it is what makes the rotate_half convention below a plain rotation:
//
//   y[j]      = x[j]      * c[j] - x[j + 32] * s[j]
//   y[j + 32] = x[j + 32] * c[j] + x[j]      * s[j]        j = 0..31
//   y[j]      = x[j]                                        j = 64..head_dim-1
//
// The position is the token's index inside its own row — `t - off[row_of[t]]` (design §7: the packed batch has no
// padding, so a packed index is never a position) — which is why no function here takes a position array.
//
// The model's mRoPE sections [11, 11, 10] with mrope_interleaved = true collapse to this for text: all three axes carry
// the same p, so every frequency slot gets the same angle whichever axis it is assigned to (design §3.3).
#ifndef JT_REF_ROPE_H
#define JT_REF_ROPE_H

#include <stddef.h>

#include "real.h"

// The rotary width of a head (design §3.3, partial_rotary_factor 0.25 of head_dim 256) and half of it.
#ifndef JT_ROPE_DIM
#define JT_ROPE_DIM 64
#endif
#ifndef JT_ROPE_HALF
#define JT_ROPE_HALF 32
#endif

// ---- the tables (host, fp32) -------------------------------------------------------------------------------------------
// cos and sin: [lmax][JT_ROPE_DIM], row p the position p. Built in fp32 as design §3.3 spells it out (see the top of
// this file), so the float build of this function and kernels/rope.cu's rope_tables() agree bit for bit — which matters
// because a one-ulp difference in inv_freq becomes a p * 1e-7 difference in the angle and so ~1e-4 in cos at p = 1026.
void ref_rope_tables(real *cos, real *sin, int lmax);

// ---- in place on the first 64 dims of each head ------------------------------------------------------------------------
// x: [n][heads * head_dim], the token-major packed layout. row_of[t] is t's row, row_off[b] the row's first token, so the
// position is t - row_off[row_of[t]], and row_off, row_of and the tables are all the kernel needs.
// head_dim must be >= 64 (the model's is 256).
void ref_rope_fwd(real *x, const int *row_off, const int *row_of, int n, int heads, int head_dim, const real *cos,
                  const real *sin);

// The rotation's transpose, which is the same rotation with sin negated: y[j] = x[j]*c + x[j+32]*s,
// y[j+32] = x[j+32]*c - x[j]*s. In place on the cotangent.
void ref_rope_bwd(real *dx, const int *row_off, const int *row_of, int n, int heads, int head_dim, const real *cos,
                  const real *sin);

#endif // JT_REF_ROPE_H
