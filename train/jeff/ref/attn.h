// ref/attn.h — the CPU oracle for the attention softmax kernels (plan M2.5, design §4.7 and §8.2).
//
// Layout, the same as the kernels': scores and probabilities are [heads][L][L], row-major, the last index s is the key
// position and row (h, t) is query position t of head h. One packed row of a batch, so L is that row's length. The
// forward is causal: s > t is masked to exactly 0 and the softmax runs over s <= t only. The scale (1/16 in the model)
// is applied to the scores inside, as design §8.2 writes it — `P[h] = causal_softmax(Sc[h] / 16)` and
// `dSc = attn_softmax_bwd(P, dP) / 16` — so it is a parameter of both functions and the caller never pre-scales.
//
// The same source builds twice (ref/real.h): float, which the GPU test compares against, and double, which the
// gradchecks differentiate.
#ifndef JT_REF_ATTN_H
#define JT_REF_ATTN_H

#include "real.h"

#ifdef __cplusplus
extern "C" {
#endif

// P[h][t][s] = exp(scale*S - max_{s'<=t} scale*S) / sum_{s'<=t} exp(...) for s <= t, and exactly 0 for s > t (the
// max-subtracted form of design §8.1's `attn_softmax_fwd`).
void ref_attn_softmax_fwd(const real *S, real *P, int L, int heads, real scale);

// dS[h][t][s] = scale * P * (dP - sum_{s'<=t} P*dP) for s <= t, and exactly 0 for s > t. P is the forward's output;
// dP = do_t . v_s in design §4.7 (only its s <= t half matters; the masked half is ignored, not read).
void ref_attn_softmax_bwd(const real *P, const real *dP, real *dS, int L, int heads, real scale);

#ifdef __cplusplus
}
#endif

#endif // JT_REF_ATTN_H
