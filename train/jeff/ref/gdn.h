// ref/gdn.h — the CPU oracle of the GDN (gated delta net) recurrence (plan M3.1; design §3.6 forward, §4.8
// backward, §8.3 "gdn_ref ... the oracle"). Written before kernels/gdn.cu (plan rule 2): test_kernels' gdn group
// compares the kernels against these two functions, and test/grad_gdn.c differentiates ref_gdn_fwd in double
// (-DJT_REAL_DOUBLE) to check that ref_gdn_bwd is its adjoint.
//
// What it is, per row of a packed batch and per head (design §3.6), with S = 0 at the row's first token:
//
//   for t:
//     S        = S * exp(g[t])                      # the decay
//     kv[j]    = sum_i S[i,j] * k[t,i]
//     delta[j] = (v[t,j] - kv[j]) * beta[t]
//     S[i,j]  += k[t,i] * delta[j]                  # the delta-rule write
//     o[t,j]   = sum_i S[i,j] * q[t,i]
//
// Dimension conventions (the ones the design's kernels read): `i` indexes the key dimension (Dk, = 128 in the model),
// `j` the value dimension (Dv, = same). Each value column j of S evolves independently in the forward — kv[j],
// delta[j] and o[t,j] only touch column j — which is what the kernel's one-lane-per-column design (§8.3) exploits.
//
// Layouts. q, k are [N][H][Dk]; v, o are [N][H][Dv]; g and beta are [N][H]. N = off[B] is the token count of the packed
// batch and off[] holds the B + 1 row boundaries (off[b] .. off[b+1] is row b, design §7: no padding, no cross-row
// state). All of them are fp32 in the trainer; here they are `real` so the same file is the float build (what the GPU
// test compares against) and the double build (what gradcheck uses).
//
// Dimensions are parameters, not the constants the kernel will bake in, so the gradcheck can shrink them (design
// §8.3's last paragraph: L = 5, dims shrunk to 8). The function signatures are the plan's M3.1 ones; nothing is
// added to them.
#ifndef JT_REF_GDN_H
#define JT_REF_GDN_H

#include <stddef.h>

#include "real.h"

// ---- forward (design §3.6) ------------------------------------------------------------------------------------
// q, k: [N][H][Dk]; v: [N][H][Dv]; g, beta: [N][H]; off: [B + 1]. o is [N][H][Dv]. S is reset to 0 at every row start,
// so row b's output depends on row b's tokens only.
void ref_gdn_fwd(int B, const int *off, int H, int Dk, int Dv, const real *q, const real *k, const real *v,
                 const real *g, const real *beta, real *o);

// ---- backward (design §4.8) -----------------------------------------------------------------------------------
// dout is dLoss/do ([N][H][Dv]); the five outputs dq, dk ([N][H][Dk]), dv ([N][H][Dv]) and dg, dbeta ([N][H]) are
// dLoss/d(input) and must all be writable. dq_t and dbeta_t/dg_t are single-token terms (written, not accumulated);
// dS is carried from the last token of a row back to its first and is 0 again at the next row's end.
void ref_gdn_bwd(int B, const int *off, int H, int Dk, int Dv, const real *q, const real *k, const real *v,
                 const real *g, const real *beta, const real *dout, real *dq, real *dk, real *dv, real *dg,
                 real *dbeta);

#endif // JT_REF_GDN_H
