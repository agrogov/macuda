// kernels/gdn.cu — the GDN (gated delta net) recurrence, forward and backward (plan M3.2, M3.3; design §3.6 the forward
// loop, §4.8 the reverse loop, §8.3 the kernels).
// Only the __global__ kernels and the extern "C" launch wrappers live here; the C side includes gdn.h and never sees CUDA
// syntax (design §11.1). The oracle is ref/gdn.c (plan M3.1) and test/test_gdn.c compares the two.
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

extern "C" void gdn_fwd(const float *q, const float *k, const float *v, const float *g, const float *beta, const int *off,
                        int B, int H, int nseg, float *ckpt, float *o, cudaStream_t s) {
  if (B <= 0 || H <= 0) return;
  if (q == NULL || k == NULL || v == NULL || g == NULL || beta == NULL || off == NULL || o == NULL) return;
  // A checkpoint buffer needs a segment count to be indexed with: without one, run the pass that computes o only rather
  // than write a buffer whose stride is not known. gdn_ckpt_segs() is the helper that produces a valid nseg.
  if (ckpt != NULL && nseg <= 0) ckpt = NULL;
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
                        cudaStream_t s) {
  if (B <= 0 || H <= 0 || N <= 0 || nseg <= 0) return;
  if (q == NULL || k == NULL || v == NULL || g == NULL || beta == NULL || dout == NULL || off == NULL || ckpt == NULL)
    return;
  if (work == NULL || work->sp == NULL || work->dq_part == NULL || work->dk_part == NULL ||
      work->dbeta_part == NULL || work->dg_part == NULL)
    return;
  if (dv == NULL || dq == NULL || dk == NULL || dg == NULL || dbeta == NULL) return;
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
