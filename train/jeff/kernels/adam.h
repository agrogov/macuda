// kernels/adam.h — the optimizer kernels (plan M2.6, design §2.2 and §8.1). Host C includes this; every wrapper is
// extern "C", takes plain pointers and a stream (0 = the default stream), so the C side never sees CUDA syntax
// (design §11.1). The CPU references it is compared against are in ref/adam.h.
//
//   sumsq_partial / sumsq_final   the global gradient norm: per-block partial sums, then one pass over them. A fixed
//                                 block count (K_ADAM_BLOCKS) and no atomics, so the value is identical on every run
//                                 (design §8.1: "Never atomicAdd on floats for anything compared with a reference").
//   clip_scale                    g *= coef, the clip train.py applies as clip_grad_norm_(params, 1.0)
//   adamw_step                    design §2.2's formula over a flat parameter buffer, one lr per index range
//
// The bias corrections 1 - b1^t and 1 - b2^t are the host's: computed in double and passed as floats (plan M2.6), which
// is why adamw_step takes bias1/bias2 rather than a step count.
#ifndef JT_ADAM_H
#define JT_ADAM_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// The fixed block count of the two-pass reduction (plan M2.6). partials must have this many floats; the reference's
// REF_ADAM_BLOCKS is the same number.
#define K_ADAM_BLOCKS 1024
// Threads per block for every kernel here: 256 like the rest of the small kernels (design §8.1).
#define K_ADAM_THREADS 256

// First pass. partials[b] = sum of g[i]^2 over the slice block b owns: i = b, b+nblocks, b+2*nblocks, ... (the same
// arithmetic progression ref_sumsq_partial sums sequentially, with a block of threads sharing the slice). nblocks is
// usually K_ADAM_BLOCKS; a test may pass a small ragged one. The result does not depend on scheduling.
void k_sumsq_partial(const float *g, int n, float *partials, int nblocks, cudaStream_t s);
// Second pass, one block: *out = partials[0] + partials[1] + ... (tree order inside the block).
void k_sumsq_final(const float *partials, int nblocks, float *out, cudaStream_t s);

// g[i] *= coef, i < n (design §2.2's `if coef < 1: g *= coef`, with coef = 1/(norm+1e-6) clamped to 1).
void k_clip_scale(float *g, int n, float coef, cudaStream_t s);

// One Adam step (design §2.2, weight_decay 0):
//   i < lr_group_split uses lr_lora (every LoRA A and B), the rest lr_readout (the readout, last in the flat buffer).
//   m = b1 m + (1-b1) g;  v = b2 v + (1-b2) g^2;  p -= lr * (m/bias1) / (sqrt(v/bias2) + eps)
// bias1 = 1-b1^t and bias2 = 1-b2^t come from the host (double, rounded to float). p, m and v are updated in place.
void k_adamw_step(float *p, const float *g, float *m, float *v, int n, int lr_group_split, float lr_lora,
                  float lr_readout, float b1, float b2, float eps, float bias1, float bias2, cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif // JT_ADAM_H
