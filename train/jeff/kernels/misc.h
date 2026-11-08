// kernels/misc.h — the plumbing kernels (design §8.1). Host C includes this; every wrapper here is extern "C" so the C
// side never sees CUDA syntax (design §11.1). The rest of the plumbing (zero, add_inplace, bf16_to_f32, embed_gather,
// qgate_split/merge, kv_expand/reduce, to_head_major/from_head_major) joins this header with its kernels/misc.cu and
// ref/misc.c, jt_fill below is the M0 hello kernel.
#ifndef JT_MISC_H
#define JT_MISC_H

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launch on stream s (0 = default): x[i] = v for every i < n. A no-op when n == 0.
void jt_fill(float *x, float v, size_t n, cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif // JT_MISC_H
