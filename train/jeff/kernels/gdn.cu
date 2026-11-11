// kernels/gdn.cu — the GDN (gated delta net) recurrence, forward and backward (plan M3.2, M3.3; design §3.6 the forward
// loop, §4.8 the reverse loop, §8.3 the kernels).
// Only the __global__ kernels and the extern "C" launch wrappers live here; the C side includes gdn.h and never sees CUDA
// syntax (design §11.1). The oracle is ref/gdn.c (plan M3.1) and test/test_gdn.c compares the two.
//
// Two implementations of the same recurrence live here. The first, below, is the sequential one: one warp per (row, head,
// column tile), the state in registers, the token loop serial (design §8.3). The second is plan M8.2's chunked WY form,
// everything after the ===== banner: chunks of JT_GDN_SEG tokens turned into GEMMs. gdn_fwd/gdn_bwd dispatch on
// gdn_chunked_enabled() (JT_GDN_CHUNKED=1, off by default -- the measured reason is in gdn.h).
//
// Shape (design §8.3): grid (B, H, 4 column tiles), one 32-thread warp per block. Lane l owns value column
// j = tile*32 + l and keeps that column of the state, `float s[128]`, in registers for the whole row. A value column of S
// evolves independently in the forward (design §3.6: kv[j], delta[j] and o[t,j] only touch column j), so there is no
// cross-lane reduction anywhere: kv[j] and o[t,j] are this lane's own serial sums over the key index i.
//
// Registers. s[] stays in registers only while every index into it is a compile-time constant after unrolling, so the
// loops over i are `#pragma unroll` with JT_GDN_DIM = 128 a macro and no runtime index is ever applied to s[] (design
// §8.3, plan M3.2). Measured on gpubox (sm_120, `nvcc -O2 -arch=sm_120 -std=c++17 -Xptxas -v kernels/gdn.cu`):
//
//   gdn_fwd_kernel: 201 registers, 0 bytes spill stores, 0 bytes spill loads, 1024 bytes smem
//   gdn_bwd_kernel: 168 registers, 0 bytes spill stores, 0 bytes spill loads,  256 bytes smem
//   gdn_bwd_reduce_kernel: 28 registers, 0 bytes spill stores, 0 bytes spill loads
//
// One warp per block keeps that inside the register file: 201 * 32 = 6432 of an SM's 65536 registers, i.e. 10 blocks per
// SM, and 0 spill is the gate the plan sets. The split of each 128-iteration loop into two halves is what buys the zero:
// as one unrolled block ptxas prefetches all 128 shared loads of the sum and needs 255 registers plus 68 bytes of spill
// stores (52 loads). Two 64-iteration halves with a __syncwarp() between them — a scheduling barrier, and every lane of
// the warp reaches it, so it is free of divergence — leave 201 registers and no spill. The sums still run i = 0, 1, ...
// 127 ascending, so the arithmetic is the same as one loop.
//
// Checkpoints (design §8.3): before the step of every token whose row-local index is a multiple of JT_GDN_SEG the block
// writes its 128 values of S to ckpt[b][h][tl/64][i][j]. Because it is written *before* the step, segment 0's slot is all
// zeros; the backward (plan M3.3) starts each segment from it instead of from token 0.
//
// Backward (plan M3.3; design §4.8 the math, §8.3 phase A / phase B). gdn_bwd_kernel is the reverse pass and
// gdn_bwd_reduce_kernel the 4-tile reduction it hands its per-tile partials to. Grid (B, H, 4), block 64 = 2 warps: warp
// w owns key rows i in [64w, 64w+64), lane l owns value column j = tile*32 + l, so a thread's piece of every 128x128
// matrix is 64 floats, and its piece of dS -- `ds[64]` -- stays in registers for the whole row, carried from the row's
// last token back to its first across segment boundaries. Segments run last to first, each in two passes: phase A
// replays the forward from ckpt[b][h][seg] and writes every token's S_prev to the global scratch sp, phase B walks
// design §4.8's lines for the segment's tokens last to first. Registers: the plan's M3.3 budget is "at most two 64-float
// arrays live" -- `s[64]` in phase A (with `ds`) and `sd[64]` in phase B (with `ds`) -- and never `sp[64]` or `st[64]`;
// st_i is recomputed from sd_i inside the loop that uses it. 168 registers hold both phases: phase A's `s[64]` with
// `ds[64]` live, phase B's `sd[64]` with `ds[64]`, and never three arrays at once, so ptxas needs no spill and neither
// phase needed the split-half trick the forward uses (the loops here are 64 rows, not 128). Six blocks per SM.
//
// Cost (measured, gpubox sm_120, notes/J17.md): at B = 8 / N = 5094 a call is 11.6 ms warm, 1.39 ms of that the forward
// that produced the checkpoints. It is memory-bound, not compute-bound: the design's sp scratch is S_prev for every token
// -- 4096 floats per block per token, 5.3 GB written and 5.3 GB read at N = 5094 -- plus 3 GB for the dq/dk tile partials
// and their reduce, which is ~12 GB of traffic in 11.6 ms, i.e. ~1.1 TB/s of a 5090's 1.79 TB/s peak. Shrinking that
// traffic (a smaller sp window, or sp in shared memory) is a design change, not a kernel tweak: notes/J17.md has the
// breakdown and what J41 would have to decide.
//
// Reductions: kv_j and ddelta_j are sums over all 128 key rows, so each warp sums its own 64 and the block adds the two
// halves through __shared__ kvp[w][lane] in the fixed order 0 then 1; dq_i, dk_i, dbeta and dg are sums over the 32 lanes
// of one warp (warp_sum). No atomics anywhere, so no result depends on the schedule.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gdn.h"
#include "util.h" // JT_SYNC_DEBUG (a no-op unless -DJT_DEBUG)

__global__ void gdn_fwd_kernel(const float *q, const float *k, const float *v, const float *g, const float *beta,
                               const int *off, int H, int nseg, float *ckpt, float *o) {
  __shared__ float k_s[JT_GDN_DIM]; // the token's k[t,h,:], so every lane reads all 128 key entries
  __shared__ float q_s[JT_GDN_DIM]; // and q[t,h,:]
  const int b = (int)blockIdx.x;
  const int h = (int)blockIdx.y;
  const int lane = (int)threadIdx.x;
  const int j = (int)blockIdx.z * JT_GDN_TILE + lane; // this lane's value column
  const int t0 = off[b], t1 = off[b + 1];              // design §7: row b is [off[b], off[b+1])

  float s[JT_GDN_DIM]; // column j of S[i,j] for i = 0..127, live across the whole row
#pragma unroll
  for (int i = 0; i < JT_GDN_DIM; i++) s[i] = 0.0f; // S_0 = 0 at every row start, per head (design §3.6)

  for (int t = t0; t < t1; t++) {
    const size_t gi = (size_t)t * (size_t)H + (size_t)h; // token-major [N][H]
    const float *k_t = k + gi * JT_GDN_DIM;
    const float *q_t = q + gi * JT_GDN_DIM;
    // Each lane loads 4 of the 128 entries; the shared arrays are the warp's own, so __syncwarp() is the whole barrier.
#pragma unroll
    for (int c = 0; c < JT_GDN_DIM / JT_GDN_TILE; c++) {
      const int i = lane + c * JT_GDN_TILE;
      k_s[i] = k_t[i];
      q_s[i] = q_t[i];
    }
    __syncwarp(); // the stores are visible, and no lane is still reading the previous token's values

    const int tl = t - t0;
    if (ckpt != NULL && (tl & (JT_GDN_SEG - 1)) == 0) {
      float *c = ckpt + (((size_t)b * (size_t)H + (size_t)h) * (size_t)nseg + (size_t)(tl / JT_GDN_SEG)) * JT_GDN_CKPT_FLOATS;
#pragma unroll
      for (int i = 0; i < JT_GDN_DIM; i++) c[(size_t)i * JT_GDN_DIM + (size_t)j] = s[i]; // S_prev, i.e. before the step
    }

    const float e = expf(g[gi]); // decay of design §3.6: S = S * exp(g[t]), g <= 0
    const float bt = beta[gi];
    const size_t vi = gi * JT_GDN_DIM + (size_t)j;
    float kv = 0.0f;
#pragma unroll
    for (int i = 0; i < JT_GDN_DIM / 2; i++) {
      s[i] *= e;           // S = S * exp(g[t])
      kv += s[i] * k_s[i]; // kv[j] = sum_i S[i,j] * k[t,i]
    }
    __syncwarp(); // the register-pressure barrier explained above; the sum continues at i = 64
#pragma unroll
    for (int i = JT_GDN_DIM / 2; i < JT_GDN_DIM; i++) {
      s[i] *= e;
      kv += s[i] * k_s[i];
    }
    const float d = (v[vi] - kv) * bt; // delta[j] = (v[t,j] - kv[j]) * beta[t]
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < JT_GDN_DIM / 2; i++) {
      s[i] += k_s[i] * d;   // S[i,j] += k[t,i] * delta[j]
      acc += s[i] * q_s[i]; // o[t,j] = sum_i S[i,j] * q[t,i], with the S_t the line above just made
    }
    __syncwarp();
#pragma unroll
    for (int i = JT_GDN_DIM / 2; i < JT_GDN_DIM; i++) {
      s[i] += k_s[i] * d;
      acc += s[i] * q_s[i];
    }
    o[vi] = acc;
    __syncwarp(); // every lane is done with k_s/q_s before the next token overwrites them
  }
}

// ===================================================================================================================
// The chunked (WY-form) recurrence: plan M8.2, design §8.3's closing paragraph ("The fla chunked algorithm (WY form,
// GEMM-based) ... is the M8 optimisation").
//
// Why the same function can be computed with GEMMs. g[t,h] is a scalar, so unrolling design §3.6's recurrence over a
// chunk of C = JT_GDN_SEG tokens gives, with G_i = sum_{s<=i} g_s (G decreasing, so every factor below is <= 1),
//
//   S_i    = e^{G_i} S_0 + sum_{j<=i} e^{G_i - G_j} k_j delta_j^T                                   (state)
//   delta_i + beta_i sum_{j<i} e^{G_i-G_j} (k_i.k_j) delta_j = beta_i (v_i - e^{G_i} k_i^T S_0)      (the delta system)
//   o_i    = e^{G_i} (S_0^T q_i) + sum_{j<=i} e^{G_i-G_j} (q_i.k_j) delta_j                         (output)
//
// With W = K K^T, A[i][j] = W[i][j] e^{G_i-G_j} (j < i), L = diag(beta) A, the delta system is (I + L) Delta = X with
// X = diag(beta)(V - diag(e^{G_i}) K S_0), so with T = (I+L)^{-1} it is Delta = T X. P[i][j] = (QK^T)[i][j] e^{G_i-G_j}
// (j <= i) turns the output into O = diag(e^{G_i}) Q S_0 + P Delta, and the chunk's exit state into
// S_out = e^{G_{C-1}} S_0 + Kd^T Delta with Kd[j] = k_j e^{G_{C-1}-G_j}. Seven GEMMs and one 64x64 inverse per chunk.
//
// The backward is the exact derivative of those expressions (validated in double against the sequential oracle at 1e-15
// and against finite differences before a line of CUDA was written), with two identities that make it GEMM-shaped:
//   dT = dDelta X^T,  dX = T^T dDelta,  dL = -T^T dT T^T   (from (I+L)T = I, so dT = -T dL T)
//   dg comes from the decay factors themselves: every e^{G_i-G_j} and e^{G_i} is differentiated into a per-token
//   cotangent cG_i, and dg_s = sum_{i>=s} cG_i (a reverse cumulative sum inside the chunk).
//
// Padding. Rows have different lengths, so every chunk is gathered into a full C-token buffer and the tokens past the
// row's end are zeros (k = v = q = 0, g = 0, beta = 0). That makes the padded rows exactly inert: A, P, X and Delta are
// zero there, and the state carry e^{G_{C-1}} is e^{G_{last}} because g = 0 past the end. The one thing padding does not
// make free is a *cotangent*: the state carry's factor depends on the last *real* token's G, so the kernels that add to
// cG from it are given `last = cnt - 1` and not C - 1 (see gdn_chunk_dsadd_kernel / gdn_chunk_kdbwd_kernel).
//
// Each chunk is one step of a loop over the chunk index (the state is the only thing that crosses a chunk boundary),
// and every GEMM in a step is batched over all B*H sequences at once. The workspace is the caller's (gdn_chunk_plan).
// -------------------------------------------------------------------------------------------------------------------

// The compute type is CUBLAS_COMPUTE_32F here and not gemm.c's mode: gemm.h's wrapper follows the run's TF32/exact
// setting, and TF32's ~5e-4 mantissa would not meet this kernel's 1e-4 gate (plan M3.2, design §8.3). The recurrence is
// 18 small GEMMs per chunk, so the exact fp32 path costs little next to the projections' TF32 ones.
static void gdn_cgemm_b(cublasHandle_t h, int ta, int tb, int M, int N, int K, float alpha, const float *A, long sA,
                        const float *B, long sB, float beta, float *C, long sC, int batch) {
  CB(cublasGemmStridedBatchedEx(h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha, B,
                                CUDA_R_32F, tb ? K : N, (long long)sB, A, CUDA_R_32F, ta ? M : K, (long long)sA, &beta,
                                C, CUDA_R_32F, N, (long long)sC, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// Every buffer of the chunked path, per (row, head) sequence: `G` copies of each. The widths are the arena helper's
// (gdn.h: GDN_CHUNK_FLOATS_PER_SEQ counts 17 + 11 + 4 + 5 of these), and chunk_arena() is the only place the layout is
// written down -- the same discipline gdn.h's ckpt/sp helpers follow.
typedef struct {
  float *kb, *vb, *qb;                                                                            // [G][C][D]
  float *xb, *db, *kdb, *ksb, *qsb, *pdb;                                                          // [G][C][D]
  float *dob, *dqb, *dkb, *dvb;                                                                    // [G][C][D]
  float *ddb, *dxb, *dksb, *dkdb;                                                                  // [G][C][D]
  float *kk, *qq, *lm, *tm, *pm;                                                                   // [G][C][C]
  float *dl, *dt, *dqq, *dw, *yy, *dpg;                                                            // [G][C][C]
  float *st, *sn, *ds, *dsn;                                                                       // [G][D][D]
  float *gt, *bb, *cg, *dbe;                                                                       // [G][C]
} ChArena;

// Carve the arena and check its size. 1 on success, 0 when `n` floats are not enough (the caller then runs the
// sequential kernels and says so once, rather than computing something wrong).
static int chunk_arena(float *base, size_t n, int G, ChArena *a) {
  const size_t CD = (size_t)JT_GDN_SEG * (size_t)JT_GDN_DIM, CC = (size_t)JT_GDN_SEG * (size_t)JT_GDN_SEG;
  const size_t DD = (size_t)JT_GDN_DIM * (size_t)JT_GDN_DIM, C = (size_t)JT_GDN_SEG;
  struct {
    float **slot;
    size_t w;
  } f[] = {{&a->kb, CD},    {&a->vb, CD},   {&a->qb, CD},   {&a->xb, CD},   {&a->db, CD},   {&a->kdb, CD},
           {&a->ksb, CD},   {&a->qsb, CD},  {&a->pdb, CD},  {&a->dob, CD},  {&a->dqb, CD},  {&a->dkb, CD},
           {&a->dvb, CD},   {&a->ddb, CD},  {&a->dxb, CD},  {&a->dksb, CD}, {&a->dkdb, CD},
           {&a->kk, CC},    {&a->qq, CC},   {&a->lm, CC},   {&a->tm, CC},   {&a->pm, CC},   {&a->dl, CC},
           {&a->dt, CC},    {&a->dqq, CC},  {&a->dw, CC},   {&a->yy, CC},   {&a->dpg, CC},
           {&a->st, DD},    {&a->sn, DD},   {&a->ds, DD},   {&a->dsn, DD},
           {&a->gt, C},     {&a->bb, C},    {&a->cg, C},    {&a->dbe, C}};
  size_t off = 0;
  for (size_t i = 0; i < sizeof f / sizeof f[0]; i++) {
    if (base == NULL || off + f[i].w * (size_t)G > n) return 0;
    *f[i].slot = base + off;
    off += f[i].w * (size_t)G;
  }
  return off == (size_t)G * GDN_CHUNK_FLOATS_PER_SEQ;
}

// Real tokens in a chunk that starts at row-local token `start`: 0 past the row's end, the tail otherwise. Every kernel
// below derives it from the device row table the same way, which is why no buffer has to carry it.
static __device__ __forceinline__ int gdn_chunk_cnt(int len, int start) {
  const int r = len - start;
  return r <= 0 ? 0 : (r < JT_GDN_SEG ? r : JT_GDN_SEG);
}

// gt = the chunk's cumulative log-decay (constant past the row's end), bb its beta, and the two accumulators the backward
// adds into (cg, dbe) start at zero.
__global__ void gdn_chunk_gates_kernel(const float *g, const float *beta, const int *off, int H, int C, int ci, float *gt,
                                       float *bb, float *cg, float *dbe) {
  const int sq = (int)blockIdx.x;
  const int b = sq / H, h = sq - b * H;
  if (threadIdx.x != 0) return;
  const int len = off[b + 1] - off[b], start = ci * C, cnt = gdn_chunk_cnt(len, start);
  float G = 0.0f;
  for (int c = 0; c < C; c++) {
    const size_t gi = (size_t)(off[b] + start + c) * (size_t)H + (size_t)h;
    const float gv = c < cnt ? g[gi] : 0.0f, bv = c < cnt ? beta[gi] : 0.0f;
    G += gv;
    const size_t at = (size_t)sq * (size_t)C + (size_t)c;
    gt[at] = G;
    bb[at] = bv;
    cg[at] = 0.0f;
    dbe[at] = 0.0f;
  }
}

// One [N][H][D] array -> its chunk at [G][C][D], zero past the row's end (the padding the header comment describes).
__global__ void gdn_chunk_gather_kernel(const float *src, const int *off, int H, int C, int ci, float *dst) {
  const int sq = (int)blockIdx.x, d = (int)threadIdx.x;
  const int b = sq / H, h = sq - b * H;
  const int len = off[b + 1] - off[b], start = ci * C, cnt = gdn_chunk_cnt(len, start);
  for (int c = 0; c < C; c++) {
    const float v = c < cnt ? src[((size_t)(off[b] + start + c) * (size_t)H + (size_t)h) * JT_GDN_DIM + (size_t)d] : 0.0f;
    dst[((size_t)sq * (size_t)C + (size_t)c) * JT_GDN_DIM + (size_t)d] = v;
  }
}

// A = KK^T decayed (strictly lower, without beta), I + L = I + diag(beta) A, and P = QK^T decayed (lower incl. the
// diagonal) out of the two raw GEMMs. The decay e^{G_i-G_j} is only formed for j <= i: past that it would be e^{+...} and
// could overflow.
__global__ void gdn_chunk_mask_kernel(int C, float *kk, float *qq, float *lm, float *pm, const float *gt,
                                      const float *bb) {
  const size_t m = (size_t)blockIdx.x * (size_t)C * (size_t)C;
  const float *G = gt + (size_t)blockIdx.x * (size_t)C, *B = bb + (size_t)blockIdx.x * (size_t)C;
  for (int idx = (int)threadIdx.x; idx < C * C; idx += (int)blockDim.x) {
    const int i = idx / C, j = idx - i * C;
    const float a = kk[m + (size_t)idx], qv = qq[m + (size_t)idx];
    if (j > i) { // never used: A and P are lower triangular
      lm[m + (size_t)idx] = 0.0f;
      kk[m + (size_t)idx] = 0.0f;
      pm[m + (size_t)idx] = 0.0f;
      continue;
    }
    const float dec = expf(G[i] - G[j]);
    const int strict = j < i;
    const float av = strict ? a * dec : 0.0f;          // A (unscaled: the backward's dbeta and cG want it without beta)
    lm[m + (size_t)idx] = strict ? B[i] * av : 1.0f;   // I + L, the matrix the inverse is taken of
    kk[m + (size_t)idx] = av;                          // A, for gdn_chunk_abwd_kernel
    pm[m + (size_t)idx] = qv * dec;
  }
}

// T = (I+L)^{-1} by forward substitution, one column per thread: T[i][c] = -sum_{j=c}^{i-1} L[i][j] T[j][c]. The work is
// O(C^3/6) per chunk and stays inside a block; each thread reads its own column of the output, so no barrier is needed.
//
// T is lower triangular, and the GEMMs below take the *whole* matrix (Delta = T X, dX = T^T dDelta, ...), so the strictly
// upper part has to be written here and not left as the arena's leftovers: this kernel writes rows c..C-1 of column c,
// i.e. the lower triangle only, so a thread zeroes its own column's rows above the diagonal first (G-m8probe finding 1:
// on gpubox a fresh cudaMalloc happens to be zero and the read of the upper triangle was invisible there, while the
// shim's arena is recycled video memory and the leftover fired into the delta system, which amplified it to 0.14-0.42).
__global__ void gdn_chunk_trinv_kernel(int C, const float *lm, float *tm) {
  const int c = (int)threadIdx.x;
  if (c >= C) return;
  const float *L = lm + (size_t)blockIdx.x * (size_t)C * (size_t)C;
  float *T = tm + (size_t)blockIdx.x * (size_t)C * (size_t)C;
  for (int i = 0; i < c; i++) T[(size_t)i * C + c] = 0.0f; // the upper part, never entered by the loop below
  T[(size_t)c * C + c] = 1.0f;
  for (int i = c + 1; i < C; i++) {
    float acc = 0.0f;
    for (int j = c; j < i; j++) acc += L[(size_t)i * C + j] * T[(size_t)j * C + c];
    T[(size_t)i * C + c] = -acc;
  }
}

// X = diag(beta)(V - diag(e^{G_i}) K S_0) and Kd[j] = k_j e^{G_{C-1}-G_j}, the two elementwise steps of the chunk.
__global__ void gdn_chunk_xkd_kernel(int C, const float *vb, const float *ksb, const float *kb, const float *gt,
                                     const float *bb, float *xb, float *kdb) {
  const int sq = (int)blockIdx.x, d = (int)threadIdx.x;
  const float *G = gt + (size_t)sq * (size_t)C, *B = bb + (size_t)sq * (size_t)C;
  const float glast = G[C - 1];
  for (int c = 0; c < C; c++) {
    const size_t at = ((size_t)sq * (size_t)C + (size_t)c) * JT_GDN_DIM + (size_t)d;
    xb[at] = B[c] * (vb[at] - expf(G[c]) * ksb[at]);
    kdb[at] = kb[at] * expf(glast - G[c]);
  }
}

// o = diag(e^{G_i}) Q S_0 + P Delta, scattered back to the token-major [N][H][D] layout (only the chunk's real tokens).
__global__ void gdn_chunk_out_kernel(const float *qsb, const float *pdb, const float *gt, const int *off, int H, int C,
                                     int ci, float *o) {
  const int sq = (int)blockIdx.x, d = (int)threadIdx.x;
  const int b = sq / H, h = sq - b * H;
  const int len = off[b + 1] - off[b], start = ci * C, cnt = gdn_chunk_cnt(len, start);
  for (int c = 0; c < cnt; c++) {
    const size_t at = ((size_t)sq * (size_t)C + (size_t)c) * JT_GDN_DIM + (size_t)d;
    o[((size_t)(off[b] + start + c) * (size_t)H + (size_t)h) * JT_GDN_DIM + (size_t)d] =
        expf(gt[(size_t)sq * (size_t)C + (size_t)c]) * qsb[at] + pdb[at];
  }
}

// S_out += e^{G_{C-1}} S_0, the part of the chunk's exit state that is not a GEMM.
__global__ void gdn_chunk_state_kernel(int C, float *sn, const float *st, const float *gt) {
  const int sq = (int)blockIdx.x;
  const float e = expf(gt[(size_t)sq * (size_t)C + (size_t)C - 1]);
  const size_t base = (size_t)sq * (size_t)JT_GDN_DIM * (size_t)JT_GDN_DIM;
  for (int i = (int)threadIdx.x; i < JT_GDN_DIM * JT_GDN_DIM; i += (int)blockDim.x) sn[base + i] += e * st[base + i];
}

// The chunked forward's checkpoint = the state before each chunk's first token, the same buffer the sequential forward
// fills (gdn.h: ckpt[((b*H + h)*nseg + seg)][i][j]).
__global__ void gdn_chunk_ckpt_store_kernel(float *ckpt, const float *st, int nseg, int ci) {
  const int sq = (int)blockIdx.x;
  float *ck = ckpt + ((size_t)sq * (size_t)nseg + (size_t)ci) * JT_GDN_CKPT_FLOATS;
  const float *stb = st + (size_t)sq * (size_t)JT_GDN_CKPT_FLOATS;
  for (int i = (int)threadIdx.x; i < JT_GDN_CKPT_FLOATS; i += (int)blockDim.x) ck[i] = stb[i];
}

__global__ void gdn_chunk_ckpt_load_kernel(const float *ckpt, float *st, int nseg, int ci) {
  const int sq = (int)blockIdx.x;
  const float *ck = ckpt + ((size_t)sq * (size_t)nseg + (size_t)ci) * JT_GDN_CKPT_FLOATS;
  float *stb = st + (size_t)sq * (size_t)JT_GDN_CKPT_FLOATS;
  for (int i = (int)threadIdx.x; i < JT_GDN_CKPT_FLOATS; i += (int)blockDim.x) stb[i] = ck[i];
}

// dO <- diag(e^{G_i}) dO, and the output term's cG: cG_i += sum_j dO[i][j] e^{G_i} (Q S_0)[i][j]. One thread per row.
__global__ void gdn_chunk_doscale_kernel(int C, float *dob, const float *qsb, const float *gt, float *cg) {
  const int sq = (int)blockIdx.x, c = (int)threadIdx.x;
  if (c >= C) return;
  const float e = expf(gt[(size_t)sq * (size_t)C + (size_t)c]);
  float *row = dob + ((size_t)sq * (size_t)C + (size_t)c) * JT_GDN_DIM;
  const float *qs = qsb + ((size_t)sq * (size_t)C + (size_t)c) * JT_GDN_DIM;
  float acc = 0.0f;
  for (int d = 0; d < JT_GDN_DIM; d++) {
    row[d] *= e;
    acc += row[d] * qs[d];
  }
  cg[(size_t)sq * (size_t)C + (size_t)c] += acc;
}

// The A/P side of the backward, one thread per token i of the chunk: dA = diag(beta) dL and dW = dA decayed (the two
// matrices the dK GEMMs take), dQQ = dP decayed (the dQ/dK GEMMs' matrix), dbeta_i += sum_j dL[i][j] A[i][j], and cG
// from A and P -- the row terms (i as the first index) and the column terms (i as the second), in that fixed order.
__global__ void gdn_chunk_abwd_kernel(int C, const float *dl, const float *A, const float *pm, const float *dpg,
                                      const float *gt, const float *bb, float *dw, float *dqq, float *cg,
                                      float *dbe) {
  const int sq = (int)blockIdx.x, i = (int)threadIdx.x;
  if (i >= C) return;
  const size_t m = (size_t)sq * (size_t)C * (size_t)C;
  const float *dli = dl + m + (size_t)i * (size_t)C, *Ai = A + m + (size_t)i * (size_t)C;
  const float *Pi = pm + m + (size_t)i * (size_t)C, *dPi = dpg + m + (size_t)i * (size_t)C;
  float *dwi = dw + m + (size_t)i * (size_t)C, *dqqi = dqq + m + (size_t)i * (size_t)C;
  const float *G = gt + (size_t)sq * (size_t)C, bi = bb[(size_t)sq * (size_t)C + (size_t)i];
  float rA = 0.0f, rP = 0.0f, dbi = 0.0f;
  for (int j = 0; j < i; j++) { // the strict part: A's and P's cG row terms, dbeta, and both decayed matrices
    const float dec = expf(G[i] - G[j]);
    const float dA = bi * dli[j];
    dwi[j] = dA * dec;
    dqqi[j] = dPi[j] * dec;
    rA += dA * Ai[j];
    rP += dPi[j] * Pi[j];
    dbi += dli[j] * Ai[j];
  }
  dwi[i] = 0.0f;                     // A is strictly lower
  dqqi[i] = dPi[i];                  // P's diagonal factor is exactly 1, so it has no cG term
  for (int j = i + 1; j < C; j++) { // never read by a GEMM, never a cG term
    dwi[j] = 0.0f;
    dqqi[j] = 0.0f;
  }
  float cA = 0.0f, cP = 0.0f;
  for (int r = i + 1; r < C; r++) { // this token as the second index of A and P
    const size_t at = m + (size_t)r * (size_t)C + (size_t)i;
    cA -= bb[(size_t)sq * (size_t)C + (size_t)r] * dl[at] * A[at];
    cP -= dpg[at] * pm[at];
  }
  cg[(size_t)sq * (size_t)C + (size_t)i] += rA + rP + cA + cP;
  dbe[(size_t)sq * (size_t)C + (size_t)i] += dbi;
}

// dKS = -diag(beta) diag(e^{G_i}) dX, dV = diag(beta) dX, dbeta_i += sum_j dX[i][j] (V - e^{G_i} K S_0)[i][j], and cG
// from K S_0 (whose own decay is e^{G_i}). One thread per token.
__global__ void gdn_chunk_xbwd_kernel(int C, const float *dxb, const float *vb, const float *ksb, const float *gt,
                                      const float *bb, float *dksb, float *dvb, float *cg, float *dbe) {
  const int sq = (int)blockIdx.x, c = (int)threadIdx.x;
  if (c >= C) return;
  const float e = expf(gt[(size_t)sq * (size_t)C + (size_t)c]), b = bb[(size_t)sq * (size_t)C + (size_t)c];
  const size_t base = ((size_t)sq * (size_t)C + (size_t)c) * JT_GDN_DIM;
  float dbi = 0.0f, cgv = 0.0f;
  for (int d = 0; d < JT_GDN_DIM; d++) {
    const float dx = dxb[base + (size_t)d], ks = ksb[base + (size_t)d];
    dksb[base + (size_t)d] = -b * e * dx;
    dvb[base + (size_t)d] = b * dx;
    dbi += dx * (vb[base + (size_t)d] - e * ks);
    cgv += dksb[base + (size_t)d] * ks;
  }
  dbe[(size_t)sq * (size_t)C + (size_t)c] += dbi;
  cg[(size_t)sq * (size_t)C + (size_t)c] += cgv;
}

// dS_0 += e^{G_{C-1}} dS_out, and the state carry's cG: sum_{ij} dS_out[i][j] e^{G_{C-1}} S_0[i][j], which belongs to the
// last *real* token of the chunk, because that is the token G_{C-1} is built from (the padding's g is zero). Fixed-order
// block reduction.
__global__ void gdn_chunk_dsadd_kernel(const int *off, int H, int C, int ci, const float *ds, float *dsn,
                                       const float *st, const float *gt, float *cg) {
  const int sq = (int)blockIdx.x;
  const int b = sq / H;
  const int cnt = gdn_chunk_cnt(off[b + 1] - off[b], ci * C);
  const int last = cnt > 0 ? cnt - 1 : 0; // a chunk that does not exist has nothing to attribute
  const float e = expf(gt[(size_t)sq * (size_t)C + (size_t)C - 1]);
  const size_t base = (size_t)sq * (size_t)JT_GDN_DIM * (size_t)JT_GDN_DIM;
  float acc = 0.0f;
  for (int i = (int)threadIdx.x; i < JT_GDN_DIM * JT_GDN_DIM; i += (int)blockDim.x) {
    dsn[base + (size_t)i] += e * ds[base + (size_t)i];
    acc += ds[base + (size_t)i] * e * st[base + (size_t)i];
  }
  __shared__ float red[256];
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int stp = (int)blockDim.x / 2; stp > 0; stp >>= 1) {
    if ((int)threadIdx.x < stp) red[threadIdx.x] += red[threadIdx.x + stp];
    __syncthreads();
  }
  if (threadIdx.x == 0) cg[(size_t)sq * (size_t)C + (size_t)last] += red[0];
}

// dK += e^{G_{C-1}-G_j} dKd[j] (a row of the exit state is a row of k), and cG from Kd's decay factors: -S_j on token j and
// +sum_j S_j on the last real token. One thread per token plus a fixed-order block reduction.
// dK += e^{G_{C-1}-G_j} dKd[j] (a row of the exit state is a row of k), and cG from Kd's decay factors: -S_j on token j and
// +sum_j S_j on the last *real* token of the chunk (the padding's G is constant, so that is where the factor's derivative
// belongs). One thread per token, then a fixed-order block reduction.
__global__ void gdn_chunk_kdbwd_kernel(const int *off, int H, int C, int ci, const float *dkdb, const float *kdb,
                                       const float *gt, float *dkb, float *cg) {
  const int sq = (int)blockIdx.x, c = (int)threadIdx.x;
  if (c >= C) return;
  const int b = sq / H;
  const int cnt = gdn_chunk_cnt(off[b + 1] - off[b], ci * C);
  const int last = cnt > 0 ? cnt - 1 : 0; // a chunk that does not exist has nothing to attribute
  const float f = expf(gt[(size_t)sq * (size_t)C + (size_t)C - 1] - gt[(size_t)sq * (size_t)C + (size_t)c]);
  const size_t base = ((size_t)sq * (size_t)C + (size_t)c) * JT_GDN_DIM;
  float s = 0.0f;
  for (int d = 0; d < JT_GDN_DIM; d++) {
    dkb[base + (size_t)d] += f * dkdb[base + (size_t)d];
    s += dkdb[base + (size_t)d] * kdb[base + (size_t)d];
  }
  cg[(size_t)sq * (size_t)C + (size_t)c] -= s;
  __shared__ float red[64];
  red[c] = s;
  __syncthreads();
  for (int stp = C / 2; stp > 0; stp >>= 1) {
    if (c < stp) red[c] += red[c + stp];
    __syncthreads();
  }
  if (c == 0) cg[(size_t)sq * (size_t)C + (size_t)last] += red[0];
}

// The chunk's five gradients out to the token-major layout, plus dbeta and the reverse cumulative sum that turns cG into dg
// (dg_s = sum_{i>=s} cG_i, the header comment's identity). Thread 0 does the scan; every valid token is written once.
__global__ void gdn_chunk_scatter_kernel(const float *dqb, const float *dkb, const float *dvb, const float *cg,
                                         const float *dbe, const int *off, int H, int C, int ci, float *dq, float *dk,
                                         float *dv, float *dg, float *dbeta) {
  const int sq = (int)blockIdx.x, d = (int)threadIdx.x;
  const int b = sq / H, h = sq - b * H;
  const int len = off[b + 1] - off[b], start = ci * C, cnt = gdn_chunk_cnt(len, start);
  for (int c = 0; c < cnt; c++) {
    const size_t src = ((size_t)sq * (size_t)C + (size_t)c) * JT_GDN_DIM + (size_t)d;
    const size_t dst = ((size_t)(off[b] + start + c) * (size_t)H + (size_t)h) * JT_GDN_DIM + (size_t)d;
    dq[dst] = dqb[src];
    dk[dst] = dkb[src];
    dv[dst] = dvb[src];
  }
  if (d != 0) return;
  float acc = 0.0f;
  for (int c = C - 1; c >= 0; c--) {
    acc += cg[(size_t)sq * (size_t)C + (size_t)c];
    if (c < cnt) {
      const size_t gi = (size_t)(off[b] + start + c) * (size_t)H + (size_t)h;
      dg[gi] = acc;
      dbeta[gi] = dbe[(size_t)sq * (size_t)C + (size_t)c];
    }
  }
}

// ---- the chunked orchestration ------------------------------------------------------------------------------------
// One helper for everything a chunk needs that does not depend on the gradient side: the gathered inputs, the gates, the
// two raw GEMMs, the masked matrices, the inverse, and then the state-dependent K S_0 / Q S_0 and Delta. Both directions
// call it (the backward recomputes instead of storing a chunk's intermediates for every sequence).
static void chunk_prepare(ChArena *a, cublasHandle_t h, const float *k, const float *v, const float *q, const float *g,
                          const float *beta, const int *off, int H, int G, int ci, cudaStream_t s) {
  const size_t CD = (size_t)JT_GDN_SEG * (size_t)JT_GDN_DIM, CC = (size_t)JT_GDN_SEG * (size_t)JT_GDN_SEG;
  const size_t DD = (size_t)JT_GDN_DIM * (size_t)JT_GDN_DIM;
  const int C = JT_GDN_SEG, D = JT_GDN_DIM;
  gdn_chunk_gates_kernel<<<G, 1, 0, s>>>(g, beta, off, H, C, ci, a->gt, a->bb, a->cg, a->dbe);
  JT_SYNC_DEBUG;
  gdn_chunk_gather_kernel<<<G, JT_GDN_DIM, 0, s>>>(k, off, H, C, ci, a->kb);
  gdn_chunk_gather_kernel<<<G, JT_GDN_DIM, 0, s>>>(v, off, H, C, ci, a->vb);
  gdn_chunk_gather_kernel<<<G, JT_GDN_DIM, 0, s>>>(q, off, H, C, ci, a->qb);
  JT_SYNC_DEBUG;
  gdn_cgemm_b(h, 0, 1, C, C, D, 1.0f, a->kb, (long)CD, a->kb, (long)CD, 0.0f, a->kk, (long)CC, G); // K K^T
  gdn_cgemm_b(h, 0, 1, C, C, D, 1.0f, a->qb, (long)CD, a->kb, (long)CD, 0.0f, a->qq, (long)CC, G); // Q K^T
  gdn_chunk_mask_kernel<<<G, 256, 0, s>>>(C, a->kk, a->qq, a->lm, a->pm, a->gt, a->bb);
  gdn_chunk_trinv_kernel<<<G, JT_GDN_SEG, 0, s>>>(C, a->lm, a->tm);
  JT_SYNC_DEBUG;
  gdn_cgemm_b(h, 0, 0, C, D, D, 1.0f, a->kb, (long)CD, a->st, (long)DD, 0.0f, a->ksb, (long)CD, G); // K S_0
  gdn_cgemm_b(h, 0, 0, C, D, D, 1.0f, a->qb, (long)CD, a->st, (long)DD, 0.0f, a->qsb, (long)CD, G); // Q S_0
  JT_SYNC_DEBUG;
  gdn_chunk_xkd_kernel<<<G, JT_GDN_DIM, 0, s>>>(C, a->vb, a->ksb, a->kb, a->gt, a->bb, a->xb, a->kdb);
  JT_SYNC_DEBUG;
  gdn_cgemm_b(h, 0, 0, C, D, C, 1.0f, a->tm, (long)CC, a->xb, (long)CD, 0.0f, a->db, (long)CD, G); // Delta = T X
  JT_SYNC_DEBUG;
}

// The forward's per-chunk tail: P Delta, the exit state, o, and the state swap.
static void chunk_fwd_step(ChArena *a, cublasHandle_t h, const int *off, int H, int G, int ci, float *o,
                           cudaStream_t s) {
  const size_t CD = (size_t)JT_GDN_SEG * (size_t)JT_GDN_DIM, CC = (size_t)JT_GDN_SEG * (size_t)JT_GDN_SEG;
  const size_t DD = (size_t)JT_GDN_DIM * (size_t)JT_GDN_DIM;
  const int C = JT_GDN_SEG, D = JT_GDN_DIM;
  gdn_cgemm_b(h, 0, 0, C, D, C, 1.0f, a->pm, (long)CC, a->db, (long)CD, 0.0f, a->pdb, (long)CD, G); // P Delta
  gdn_cgemm_b(h, 1, 0, D, D, C, 1.0f, a->kdb, (long)CD, a->db, (long)CD, 0.0f, a->sn, (long)DD, G); // Kd^T Delta
  JT_SYNC_DEBUG;
  gdn_chunk_state_kernel<<<G, 256, 0, s>>>(C, a->sn, a->st, a->gt);
  gdn_chunk_out_kernel<<<G, JT_GDN_DIM, 0, s>>>(a->qsb, a->pdb, a->gt, off, H, C, ci, o);
  JT_SYNC_DEBUG;
  float *t = a->st; // S_0 of the next chunk
  a->st = a->sn;
  a->sn = t;
}

// The backward's per-chunk body, design §4.8's reverse loop expressed with the chunked forward's matrices. Every GEMM
// accumulates in the order the derivation fixes (dDelta before dT, dT before dL, dW before the dK GEMMs).
static void chunk_bwd_step(ChArena *a, cublasHandle_t h, const int *off, int H, int G, int ci, cudaStream_t s) {
  const size_t CD = (size_t)JT_GDN_SEG * (size_t)JT_GDN_DIM, CC = (size_t)JT_GDN_SEG * (size_t)JT_GDN_SEG;
  const size_t DD = (size_t)JT_GDN_DIM * (size_t)JT_GDN_DIM;
  const int C = JT_GDN_SEG, D = JT_GDN_DIM;
  // dP = dO Delta^T (the mask and the decay are applied in gdn_chunk_abwd_kernel, which also owns its cG terms)
  gdn_cgemm_b(h, 0, 1, C, C, D, 1.0f, a->dob, (long)CD, a->db, (long)CD, 0.0f, a->dpg, (long)CC, G);
  // dDelta = P^T dO + Kd dS_out
  gdn_cgemm_b(h, 1, 0, C, D, C, 1.0f, a->pm, (long)CC, a->dob, (long)CD, 0.0f, a->ddb, (long)CD, G);
  gdn_cgemm_b(h, 0, 0, C, D, D, 1.0f, a->kdb, (long)CD, a->ds, (long)DD, 1.0f, a->ddb, (long)CD, G);
  // dT = dDelta X^T, dX = T^T dDelta, dL = -T^T dT T^T
  gdn_cgemm_b(h, 0, 1, C, C, D, 1.0f, a->ddb, (long)CD, a->xb, (long)CD, 0.0f, a->dt, (long)CC, G);
  gdn_cgemm_b(h, 1, 0, C, D, C, 1.0f, a->tm, (long)CC, a->ddb, (long)CD, 0.0f, a->dxb, (long)CD, G);
  gdn_cgemm_b(h, 1, 0, C, C, C, 1.0f, a->tm, (long)CC, a->dt, (long)CC, 0.0f, a->yy, (long)CC, G);
  gdn_cgemm_b(h, 0, 1, C, C, C, -1.0f, a->yy, (long)CC, a->tm, (long)CC, 0.0f, a->dl, (long)CC, G);
  JT_SYNC_DEBUG;
  // dA/dW, dQQ, dbeta from A, and cG from both triangular matrices
  gdn_chunk_abwd_kernel<<<G, JT_GDN_SEG, 0, s>>>(C, a->dl, a->kk, a->pm, a->dpg, a->gt, a->bb, a->dw, a->dqq, a->cg,
                                                 a->dbe);
  gdn_chunk_doscale_kernel<<<G, JT_GDN_SEG, 0, s>>>(C, a->dob, a->qsb, a->gt, a->cg);
  JT_SYNC_DEBUG;
  // dQ = diag(e^{G}) dO S_0^T + dQQ K, dK = dQQ^T Q + dW K + dW^T K
  gdn_cgemm_b(h, 0, 1, C, D, D, 1.0f, a->dob, (long)CD, a->st, (long)DD, 0.0f, a->dqb, (long)CD, G);
  gdn_cgemm_b(h, 0, 0, C, D, C, 1.0f, a->dqq, (long)CC, a->kb, (long)CD, 1.0f, a->dqb, (long)CD, G);
  gdn_cgemm_b(h, 1, 0, C, D, C, 1.0f, a->dqq, (long)CC, a->qb, (long)CD, 0.0f, a->dkb, (long)CD, G);
  gdn_cgemm_b(h, 0, 0, C, D, C, 1.0f, a->dw, (long)CC, a->kb, (long)CD, 1.0f, a->dkb, (long)CD, G);
  gdn_cgemm_b(h, 1, 0, C, D, C, 1.0f, a->dw, (long)CC, a->kb, (long)CD, 1.0f, a->dkb, (long)CD, G);
  JT_SYNC_DEBUG;
  // dKS, dV, dbeta and the K S_0 cG term; then dK += dKS S_0^T
  gdn_chunk_xbwd_kernel<<<G, JT_GDN_SEG, 0, s>>>(C, a->dxb, a->vb, a->ksb, a->gt, a->bb, a->dksb, a->dvb, a->cg,
                                                a->dbe);
  gdn_cgemm_b(h, 0, 1, C, D, D, 1.0f, a->dksb, (long)CD, a->st, (long)DD, 1.0f, a->dkb, (long)CD, G);
  // dS_0 = Q^T dO + K^T dKS + e^{G_{C-1}} dS_out, and the carry's cG
  gdn_cgemm_b(h, 1, 0, D, D, C, 1.0f, a->qb, (long)CD, a->dob, (long)CD, 0.0f, a->dsn, (long)DD, G);
  gdn_cgemm_b(h, 1, 0, D, D, C, 1.0f, a->kb, (long)CD, a->dksb, (long)CD, 1.0f, a->dsn, (long)DD, G);
  JT_SYNC_DEBUG;
  // dK += e^{G_{C-1}-G_j} dKd[j], and the two cG terms that belong to the chunk's last real token
  gdn_cgemm_b(h, 0, 1, C, D, D, 1.0f, a->db, (long)CD, a->ds, (long)DD, 0.0f, a->dkdb, (long)CD, G);
  gdn_chunk_kdbwd_kernel<<<G, JT_GDN_SEG, 0, s>>>(off, H, C, ci, a->dkdb, a->kdb, a->gt, a->dkb, a->cg);
  gdn_chunk_dsadd_kernel<<<G, 256, 0, s>>>(off, H, C, ci, a->ds, a->dsn, a->st, a->gt, a->cg);
  JT_SYNC_DEBUG;
  float *t = a->ds; // dS_0 of the previous chunk
  a->ds = a->dsn;
  a->dsn = t;
}
extern "C" int gdn_chunked_enabled(void) {
  const char *e = getenv("JT_GDN_CHUNKED");
  return e != NULL && strcmp(e, "0") != 0;
}

// Is the chunked path usable for this call? A plan with a handle, a big-enough arena and at least one chunk. The warning
// is one-shot (a run makes this call 18 times a layer, so a misconfiguration would otherwise bury its own log).
static int chunk_ok(const gdn_chunk_plan *plan, int B, int H, int nseg, ChArena *a) {
  static int warned;
  if (plan == NULL || plan->blas == NULL || !gdn_chunked_enabled() || nseg <= 0) return 0;
  if (chunk_arena(plan->arena, plan->arena_n, B * H, a)) return 1;
  if (!warned) {
    fprintf(stderr, "gdn: the chunked path needs %zu floats of arena for B=%d H=%d (gdn_chunk_arena_floats), got %zu;\n"
                    "     running the sequential kernels instead (same numbers, different rounding)\n",
            gdn_chunk_arena_floats(B, H), B, H, plan->arena_n);
    warned = 1;
  }
  return 0;
}

extern "C" void gdn_fwd(const float *q, const float *k, const float *v, const float *g, const float *beta, const int *off,
                        int B, int H, int nseg, float *ckpt, float *o, const gdn_chunk_plan *plan, cudaStream_t s) {
  if (B <= 0 || H <= 0) return;
  if (q == NULL || k == NULL || v == NULL || g == NULL || beta == NULL || off == NULL || o == NULL) return;
  // A checkpoint buffer needs a segment count to be indexed with: without one, run the pass that computes o only rather
  // than write a buffer whose stride is not known. gdn_ckpt_segs() is the helper that produces a valid nseg.
  if (ckpt != NULL && nseg <= 0) ckpt = NULL;
  const int G = B * H;
  ChArena a;
  if (chunk_ok(plan, B, H, nseg, &a)) { // nseg is the chunk count: ceil(longest row / JT_GDN_SEG)
    CK(cudaMemsetAsync(a.st, 0, (size_t)G * JT_GDN_CKPT_FLOATS * sizeof(float), s)); // S_0 = 0 at every row start
    for (int ci = 0; ci < nseg; ci++) {
      chunk_prepare(&a, plan->blas, k, v, q, g, beta, off, H, G, ci, s);
      if (ckpt != NULL) {
        gdn_chunk_ckpt_store_kernel<<<G, 256, 0, s>>>(ckpt, a.st, nseg, ci);
        JT_SYNC_DEBUG;
      }
      chunk_fwd_step(&a, plan->blas, off, H, G, ci, o, s);
    }
    return;
  }
  const dim3 grid((unsigned)B, (unsigned)H, JT_GDN_TILES);
  gdn_fwd_kernel<<<grid, JT_GDN_TILE, 0, s>>>(q, k, v, g, beta, off, H, nseg, ckpt, o);
  JT_SYNC_DEBUG;
}

// ---- backward (plan M3.3; design §4.8, §8.3) -----------------------------------------------------------------------
// The warp butterfly the plan's M3.3 block names: "for 'per row i, sum over the 32 lanes', call warp_sum(st[i] * do_j)".
// Every lane gets the sum, so lane 0 is the one that stores it.
static __device__ float gdn_warp_sum(float v) {
  for (int o = JT_GDN_TILE / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Element (i) of tile `tile` in a [JT_GDN_TILES][N][H][JT_GDN_DIM] partial (dq_part, dk_part).
static __device__ __forceinline__ size_t gdn_part_ix(int tile, int t, int h, int i, int N, int H) {
  return ((((size_t)tile * (size_t)N + (size_t)t) * (size_t)H + (size_t)h) * (size_t)JT_GDN_DIM + (size_t)i);
}

// Element (t, h) of tile `tile` in a [JT_GDN_TILES][N][H] partial (dbeta_part, dg_part).
static __device__ __forceinline__ size_t gdn_gate_ix(int tile, int t, int h, int N, int H) {
  return (((size_t)tile * (size_t)N + (size_t)t) * (size_t)H + (size_t)h);
}

// The five pointers of gdn_bwd_work arrive unpacked: a kernel cannot dereference the host struct itself.
__global__ void gdn_bwd_kernel(const float *q, const float *k, const float *v, const float *g, const float *beta,
                               const float *dout, const int *off, int H, int N, int nseg, const float *ckpt, float *sp,
                               float *dq_part, float *dk_part, float *dbeta_part, float *dg_part, float *dv) {
  __shared__ float kvp[JT_GDN_BWD_WARPS][JT_GDN_TILE]; // this block's row-sums: kvp[w][lane] is warp w's half
  const int b = (int)blockIdx.x;
  const int h = (int)blockIdx.y;
  const int tile = (int)blockIdx.z;
  const int tid = (int)threadIdx.x;
  const int w = tid / JT_GDN_TILE, lane = tid % JT_GDN_TILE;
  const int j = tile * JT_GDN_TILE + lane; // this thread's value column
  const int i0 = w * JT_GDN_BWD_ROWS;      // this warp's key rows: [i0, i0 + 64)
  const int t0 = off[b], t1 = off[b + 1];
  if (t0 >= t1) return; // an empty row has nothing to reverse (a packed batch never makes one, but off[] is the caller's)

  sp += (((size_t)b * (size_t)H + (size_t)h) * (size_t)JT_GDN_TILES + (size_t)tile) *
        ((size_t)JT_GDN_SEG * (size_t)JT_GDN_DIM * (size_t)JT_GDN_TILE); // this block's 1 MB of S_prev
  const float *ckpt_bh = ckpt + ((size_t)b * (size_t)H + (size_t)h) * (size_t)nseg * JT_GDN_CKPT_FLOATS;

  float ds[JT_GDN_BWD_ROWS]; // this thread's piece of dS: live from the row's last token back to its first
#pragma unroll
  for (int r = 0; r < JT_GDN_BWD_ROWS; r++) ds[r] = 0.0f; // dS starts at 0 after the row's last token (design §4.8)
  float s[JT_GDN_BWD_ROWS];  // phase A's running S, reloaded from the segment's checkpoint
  float sd[JT_GDN_BWD_ROWS]; // phase B's decayed S_prev, i.e. design §4.8's Sd; both are dead outside their phase

  for (int seg = nseg - 1; seg >= 0; seg--) { // segments last to first (design §8.3)
    const int a0 = t0 + seg * JT_GDN_SEG;
    const int a1 = t1 < a0 + JT_GDN_SEG ? t1 : a0 + JT_GDN_SEG;
    if (a0 >= a1) continue; // this row is shorter than the batch's longest, so this segment of it does not exist
    const float *ck = ckpt_bh + (size_t)seg * JT_GDN_CKPT_FLOATS;

    // ---- phase A: recompute. Start from the checkpoint, run the forward step, keep every token's S_prev in sp.
#pragma unroll
    for (int r = 0; r < JT_GDN_BWD_ROWS; r++) s[r] = ck[(size_t)(i0 + r) * JT_GDN_DIM + (size_t)j];
    for (int t = a0; t < a1; t++) {
      float *sp_t = sp + (size_t)(t - a0) * (size_t)(JT_GDN_DIM * JT_GDN_TILE) + (size_t)i0 * JT_GDN_TILE + lane;
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) sp_t[(size_t)r * JT_GDN_TILE] = s[r]; // S_prev, before the step
      const size_t gi = (size_t)t * (size_t)H + (size_t)h;
      const float *k_t = k + gi * JT_GDN_DIM;
      const float e = expf(g[gi]);
      float acc = 0.0f;
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) {
        s[r] *= e; // Sd = S_prev * exp(g_t)
        acc += s[r] * k_t[i0 + r];
      }
      kvp[w][lane] = acc; // this warp's 64-row half of kv[j]
      __syncthreads();
      const float kv = kvp[0][lane] + kvp[1][lane]; // the whole 128-row sum, in the fixed order 0 then 1
      __syncthreads();                              // required before kvp is reused
      const float delta = (v[gi * JT_GDN_DIM + (size_t)j] - kv) * beta[gi];
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) s[r] += k_t[i0 + r] * delta; // S_t = Sd + k_t (x) delta
    }

    // ---- phase B: reverse steps within the segment, last token first (design §4.8, line for line).
    for (int t = a1 - 1; t >= a0; t--) {
      const float *sp_t = sp + (size_t)(t - a0) * (size_t)(JT_GDN_DIM * JT_GDN_TILE) + (size_t)i0 * JT_GDN_TILE + lane;
      const size_t gi = (size_t)t * (size_t)H + (size_t)h;
      const float *k_t = k + gi * JT_GDN_DIM;
      const float *q_t = q + gi * JT_GDN_DIM;
      const float e = expf(g[gi]); // "recomputing the forward step's values"
      const float bt = beta[gi];
      const float doj = dout[gi * JT_GDN_DIM + (size_t)j];
      const float vj = v[gi * JT_GDN_DIM + (size_t)j];
      float acc;

      // Sd = S_prev * exp(g_t), held for this token only (sp_i is read once per token: this is that read)
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) sd[r] = sp_t[(size_t)r * JT_GDN_TILE] * e;
      // kv[j] = sum_i Sd[i,j] k_t[i], over both halves of the key dimension
      acc = 0.0f;
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) acc += sd[r] * k_t[i0 + r];
      kvp[w][lane] = acc;
      __syncthreads();
      const float kv = kvp[0][lane] + kvp[1][lane];
      __syncthreads();
      const float delta = (vj - kv) * bt; // delta_j = (v_tj - kv_j) * beta_t
      // -- o_t = S_t^T q_t:  dq_i = sum_j S_t[i,j] do_tj. S_t = Sd + k_t (x) delta is formed inside the butterfly.
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) {
        const float dq = gdn_warp_sum((sd[r] + k_t[i0 + r] * delta) * doj);
        if (lane == 0) dq_part[gdn_part_ix(tile, t, h, i0 + r, N, H)] = dq;
      }
      // dS[i,j] += q_t_i do_tj
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) ds[r] += q_t[i0 + r] * doj;
      // ddelta_j = sum_i dS[i,j] k_t[i], which is S_t = Sd + k (x) delta's k side, over both halves of the rows
      acc = 0.0f;
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) acc += ds[r] * k_t[i0 + r];
      kvp[w][lane] = acc;
      __syncthreads();
      const float ddelta = kvp[0][lane] + kvp[1][lane];
      __syncthreads();
      // -- delta = (v - kv) * beta:  dv = ddelta * beta (one tile owns column j), dkv = -ddelta * beta,
      // dbeta_t = sum_j ddelta_j (v_tj - kv_j) over the tile's 32 columns
      const float dkv = -ddelta * bt;
      if (w == 0) dv[gi * JT_GDN_DIM + (size_t)j] = ddelta * bt;
      const float dbt = gdn_warp_sum(ddelta * (vj - kv));
      if (w == 0 && lane == 0) dbeta_part[gdn_gate_ix(tile, t, h, N, H)] = dbt;
      // dk_i = sum_j (dS[i,j] delta_j + Sd[i,j] dkv_j), with dS as it is *before* the k (x) dkv term below
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) {
        const float dk = gdn_warp_sum(ds[r] * delta + sd[r] * dkv);
        if (lane == 0) dk_part[gdn_part_ix(tile, t, h, i0 + r, N, H)] = dk;
      }
      // -- kv = Sd^T k_t:  dS[i,j] += k_t_i dkv_j
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) ds[r] += k_t[i0 + r] * dkv;
      // -- Sd = S_prev * exp(g_t):  dg_t = sum_{i,j} dS[i,j] Sd[i,j], dS *after* the k (x) dkv term, all 128 rows
      acc = 0.0f;
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) acc += ds[r] * sd[r];
      acc = gdn_warp_sum(acc);
      kvp[w][lane] = acc;
      __syncthreads();
      const float dgt = kvp[0][lane] + kvp[1][lane];
      __syncthreads();
      if (w == 0 && lane == 0) dg_part[gdn_gate_ix(tile, t, h, N, H)] = dgt;
      // dS = dS * exp(g_t)   -- now dS_{t-1}
#pragma unroll
      for (int r = 0; r < JT_GDN_BWD_ROWS; r++) ds[r] *= e;
    }
  }
}

// Sum the four tiles' partials in the order 0, 1, 2, 3 (design §8.3): a fixed order, so the result does not depend on the
// schedule -- the rule design §8.1 sets for every reduction compared with a reference. dv needs no pass of its own.
__global__ void gdn_bwd_reduce_kernel(const float *dq_part, const float *dk_part, const float *dbeta_part,
                                      const float *dg_part, size_t nq, size_t ng, float *dq, float *dk, float *dg,
                                      float *dbeta) {
  const size_t stride = (size_t)blockDim.x * (size_t)gridDim.x;
  const size_t start = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  for (size_t i = start; i < nq; i += stride) {
    float v = dq_part[i];
    v += dq_part[nq + i];
    v += dq_part[2 * nq + i];
    v += dq_part[3 * nq + i];
    dq[i] = v;
    v = dk_part[i];
    v += dk_part[nq + i];
    v += dk_part[2 * nq + i];
    v += dk_part[3 * nq + i];
    dk[i] = v;
  }
  for (size_t i = start; i < ng; i += stride) {
    float v = dbeta_part[i];
    v += dbeta_part[ng + i];
    v += dbeta_part[2 * ng + i];
    v += dbeta_part[3 * ng + i];
    dbeta[i] = v;
    v = dg_part[i];
    v += dg_part[ng + i];
    v += dg_part[2 * ng + i];
    v += dg_part[3 * ng + i];
    dg[i] = v;
  }
}

extern "C" void gdn_bwd(const float *q, const float *k, const float *v, const float *g, const float *beta,
                        const float *dout, const int *off, int B, int H, int N, int nseg, const float *ckpt,
                        const gdn_bwd_work *work, float *dv, float *dq, float *dk, float *dg, float *dbeta,
                        const gdn_chunk_plan *plan, cudaStream_t s) {
  if (B <= 0 || H <= 0 || N <= 0 || nseg <= 0) return;
  if (q == NULL || k == NULL || v == NULL || g == NULL || beta == NULL || dout == NULL || off == NULL || ckpt == NULL)
    return;
  if (dv == NULL || dq == NULL || dk == NULL || dg == NULL || dbeta == NULL) return;
  const int G = B * H;
  ChArena a;
  if (chunk_ok(plan, B, H, nseg, &a)) {
    CK(cudaMemsetAsync(a.ds, 0, (size_t)G * JT_GDN_CKPT_FLOATS * sizeof(float), s)); // dS = 0 after the last token
    for (int ci = nseg - 1; ci >= 0; ci--) {
      gdn_chunk_ckpt_load_kernel<<<G, 256, 0, s>>>(ckpt, a.st, nseg, ci); // this chunk's S_0, from the forward's ckpt
      JT_SYNC_DEBUG;
      chunk_prepare(&a, plan->blas, k, v, q, g, beta, off, H, G, ci, s);
      gdn_chunk_gather_kernel<<<G, JT_GDN_DIM, 0, s>>>(dout, off, H, JT_GDN_SEG, ci, a.dob);
      chunk_bwd_step(&a, plan->blas, off, H, G, ci, s);
      gdn_chunk_scatter_kernel<<<G, JT_GDN_DIM, 0, s>>>(a.dqb, a.dkb, a.dvb, a.cg, a.dbe, off, H, JT_GDN_SEG, ci, dq,
                                                       dk, dv, dg, dbeta);
      JT_SYNC_DEBUG;
    }
    return;
  }
  if (work == NULL || work->sp == NULL || work->dq_part == NULL || work->dk_part == NULL ||
      work->dbeta_part == NULL || work->dg_part == NULL)
    return;
  const dim3 grid((unsigned)B, (unsigned)H, JT_GDN_TILES);
  gdn_bwd_kernel<<<grid, JT_GDN_BWD_THREADS, 0, s>>>(q, k, v, g, beta, dout, off, H, N, nseg, ckpt, work->sp,
                                                    work->dq_part, work->dk_part, work->dbeta_part, work->dg_part, dv);
  JT_SYNC_DEBUG;
  // The partials are [4][N][H][DIM] and [4][N][H]; the grid below covers the first of those and the second is shorter, so
  // one grid-stride launch serves both.
  const size_t nq = (size_t)N * (size_t)H * (size_t)JT_GDN_DIM;
  const size_t ng = (size_t)N * (size_t)H;
  size_t nblk = (nq + 255) / 256;
  if (nblk > 8192) nblk = 8192;
  gdn_bwd_reduce_kernel<<<(unsigned)nblk, 256, 0, s>>>(work->dq_part, work->dk_part, work->dbeta_part, work->dg_part,
                                                       nq, ng, dq, dk, dg, dbeta);
  JT_SYNC_DEBUG;
}

