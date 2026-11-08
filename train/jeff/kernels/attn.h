// kernels/attn.h — the causal attention softmax (plan M2.5, design §4.7 and §8.1/§8.2). Host C includes this and never
// sees CUDA syntax; the CPU oracle of the same name is ref/attn.h.
//
// Scores S, probabilities P and cotangents are [heads][L][L] fp32, row-major, s fastest; row (h, t) is query position t
// of head h, and s > t is the causal mask. The scale (1/16 in the model) is applied to the scores inside, so the caller
// passes the raw GEMM output: D§8.2's `P[h] = causal_softmax(Sc[h] / 16)` and `dSc = attn_softmax_bwd(P, dP) / 16`.
#ifndef JT_ATTN_H
#define JT_ATTN_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// P[h][t][s] = softmax over s' <= t of scale*S, exactly 0 for s > t. One block per (head, t) row.
void attn_softmax_fwd(const float *S, float *P, int L, int heads, float scale, cudaStream_t s);

// dS[h][t][s] = scale * P * (dP - sum_{s'<=t} P*dP) for s <= t, exactly 0 for s > t (design §4.7). P is the forward's
// output; dP is do_t . v_s. One block per (head, t) row.
void attn_softmax_bwd(const float *P, const float *dP, float *dS, int L, int heads, float scale, cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif // JT_ATTN_H
