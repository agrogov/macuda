// kernels/gdn.h — the GDN (gated delta net) recurrence kernels (plan M3.2 forward, M3.3 backward; design §3.6 the forward,
// §4.8 the backward, §8.3 the kernel design). Host C includes this and never sees CUDA syntax; the CPU oracle of the same
// recurrence is ref/gdn.h, and test/test_gdn.c compares the two.
//
// Layouts, the ones the upstream kernels produce (design §8.3: "the kernel reads q, k, v, g and beta in the layout the
// upstream kernels produce"): q and k are [N][H][DIM] slices of the conv output after l2norm (q already scaled by
// 1/sqrt(128), design §3.6), v is [N][H][DIM], g and beta are [N][H], o is [N][H][DIM], all fp32. N = off[B] is the token
// count of a packed batch and off holds the B + 1 row boundaries (design §7: no padding, row b is [off[b], off[b+1])), and
// it lives on the device: the conv and the other token-major kernels already read their row table there. The state S is
// per (row, head) and starts at zero at each row's first token, so rows of one packed batch share nothing.
//
// The head dim is baked in: Dk = Dv = 128 is what the kernel's one-lane-per-value-column design needs (design §8.3: lane l
// keeps `float s[128]` in registers), so the two dimensions are one constant here and a runtime check would only hide a
// caller's mistake. H is a launch dimension and is free.
#ifndef JT_GDN_H
#define JT_GDN_H

#include <cuda_runtime.h>
#include <stddef.h>

#define JT_GDN_DIM 128                            // Dk = Dv (design §3.6: head_dim, 128 in the model)
#define JT_GDN_TILE 32                            // value columns a warp covers: block and tile width (design §8.3)
#define JT_GDN_TILES (JT_GDN_DIM / JT_GDN_TILE)   // 4 column tiles; gridDim.z
#define JT_GDN_SEG 64                             // tokens per checkpoint segment (design §8.3)
#define JT_GDN_CKPT_FLOATS ((size_t)JT_GDN_DIM * (size_t)JT_GDN_DIM) // one checkpoint: the whole 128x128 state

#ifdef __cplusplus
extern "C" {
#endif

// ---- forward (design §3.6, §8.3) ---------------------------------------------------------------------------------
// o[t,h,j] for every token t of every row, from S = 0 at the row's first token. Grid (B, H, 4), one warp per block.
//
// ckpt, when not NULL, is where the forward leaves the states the backward replays from (design §8.3): the state *before*
// the step of every token whose row-local index is a multiple of JT_GDN_SEG, at
//   ckpt[((b*H + h)*nseg + tl/64)][i][j] = S[i,j]  (i the key index, j the value index)
// so segment 0's slot is all zeros, and a row shorter than JT_GDN_SEG writes only that one. nseg must be at least
// gdn_ckpt_segs(off, B) (below); the buffer is gdn_ckpt_floats(B, H, nseg) floats. The forward never reads it, so a
// forward-only pass may pass NULL.
void gdn_fwd(const float *q, const float *k, const float *v, const float *g, const float *beta, const int *off, int B,
             int H, int nseg, float *ckpt, float *o, cudaStream_t s);

#ifdef __cplusplus
}
#endif

// ---- checkpoint geometry (host side; `off` here is the host copy of the row table) --------------------------------
// Segments a packed batch needs: ceil(longest row / JT_GDN_SEG), which is what sizes ckpt and what the kernel indexes
// with (design §8.3: "Checkpoints per (b,h) = ceil(max_len/64), 64 KB each: 17 at max_len 1026").
static inline int gdn_ckpt_segs(const int *off, int B) {
  int maxlen = 0;
  for (int b = 0; b < B; b++) {
    const int len = off[b + 1] - off[b];
    if (len > maxlen) maxlen = len;
  }
  return (maxlen + JT_GDN_SEG - 1) / JT_GDN_SEG;
}

// Floats in a ckpt buffer [B][H][nseg][JT_GDN_DIM][JT_GDN_DIM].
static inline size_t gdn_ckpt_floats(int B, int H, int nseg) {
  return (size_t)B * (size_t)H * (size_t)nseg * JT_GDN_CKPT_FLOATS;
}

#endif // JT_GDN_H
