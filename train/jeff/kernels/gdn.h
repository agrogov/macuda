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
// Backward geometry (design §8.3): block 64 = 2 warps, warp w owns key rows i in [64w, 64w+64), lane l owns value
// column j = tile*32 + l. So the two halves must cover the key dimension exactly.
#define JT_GDN_BWD_ROWS 64                                 // key rows one warp's threads hold
#define JT_GDN_BWD_WARPS (JT_GDN_DIM / JT_GDN_BWD_ROWS)     // 2
#define JT_GDN_BWD_THREADS (JT_GDN_BWD_WARPS * JT_GDN_TILE) // 64 threads per block

// The backward's supporting buffers, all fp32 device memory, allocated once by the caller (design §8.3) and sized by
// gdn_bwd_sp_floats() / gdn_bwd_part_floats() / gdn_bwd_gate_floats() below. Layouts, one per field:
//   sp                [B][H][4 tiles][JT_GDN_SEG][JT_GDN_DIM][JT_GDN_TILE]  one segment of S_prev per block (1 MB each)
//   dq_part, dk_part  [4 tiles][N][H][JT_GDN_DIM]                          per-tile partials of dq and dk
//   dbeta_part, dg_part [4 tiles][N][H]                                    per-tile partials of dbeta and dg
// dv needs no partial: value column j belongs to exactly one tile.
typedef struct {
  float *sp;
  float *dq_part;
  float *dk_part;
  float *dbeta_part;
  float *dg_part;
} gdn_bwd_work;

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

// ---- backward (design §4.8 the math, §8.3 phase A / phase B) ------------------------------------------------------
// The five input gradients of the recurrence from dout = dLoss/do, replayed from the forward's checkpoints (gdn_fwd's
// ckpt above, so the caller must run the forward with a ckpt buffer before this). Grid (B, H, 4), block 64.
//
// dq, dk, dv are [N][H][DIM] and dg, dbeta are [N][H], all written in full; q, k, v, g, beta and dout are the forward's
// inputs and its cotangent in the same layouts, and `off` is the device row table. N is the token count (off[B] on the
// host): the reduce pass needs it as a plain number, and the device copy of `off` is not readable from the host.
//
// `work` holds the caller's supporting buffers (design §8.3: "allocated once by the caller"), whose sizes the helpers
// below compute. This one call launches the reverse kernel and then the 4-tile reduce, so the five gradients are final
// when it returns.
void gdn_bwd(const float *q, const float *k, const float *v, const float *g, const float *beta, const float *dout,
             const int *off, int B, int H, int N, int nseg, const float *ckpt, const gdn_bwd_work *work, float *dv,
             float *dq, float *dk, float *dg, float *dbeta, cudaStream_t s);

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

// Floats in the backward's sp scratch: [B][H][4][JT_GDN_SEG][JT_GDN_DIM][JT_GDN_TILE], i.e. 1 MB per block (design
// §8.3: "1 MB per block", 512 MB at B = 8 / H = 16).
static inline size_t gdn_bwd_sp_floats(int B, int H) {
  return (size_t)B * (size_t)H * (size_t)JT_GDN_TILES * (size_t)JT_GDN_SEG * (size_t)JT_GDN_DIM *
         (size_t)JT_GDN_TILE;
}

// Floats in one of dq_part / dk_part: [4][n_tok][H][JT_GDN_DIM].
static inline size_t gdn_bwd_part_floats(size_t n_tok, int H) {
  return (size_t)JT_GDN_TILES * n_tok * (size_t)H * (size_t)JT_GDN_DIM;
}

// Floats in one of dbeta_part / dg_part: [4][n_tok][H].
static inline size_t gdn_bwd_gate_floats(size_t n_tok, int H) {
  return (size_t)JT_GDN_TILES * n_tok * (size_t)H;
}

#endif // JT_GDN_H
