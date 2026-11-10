// test/test_gdn.c — the GDN recurrence kernels against their CPU oracle (plan M3.2 the forward, M3.3 the backward;
// design §3.6, §4.8, §8.3). Its own binary, because one recurrence test spans a whole packed batch.
//
//   test_gdn fwd     the forward on the plan's five batches (card checkpoint m3-fwd)
//   test_gdn bwd     the backward, against ref_gdn_bwd (plan M3.3, card checkpoint m3)
//   test_gdn all     both (card checkpoint m3)
//   test_gdn         the implemented ones, stubs skipped, exit 0 (what `make test-null` runs)
//
// Exit codes are test_kernels' (plan M2.0): 0 passed, 1 a check failed, 2 a stub or an unknown argument. On the shim's null
// device the launches still happen, the value checks print `(launch only)` and count nothing (test/harness.h), and the
// no-argument run exits 0.
//
// Every case runs at the model's dims (H = 16, Dk = Dv = 128) on the plan's M3.2 batches {1}, {2}, {63,64,65},
// {200,7,130} and {1026,520,670,300,811,455,610,702}, and compares o against ref_gdn_fwd on the same host inputs. The gate
// is the plan's 1e-4 for this kernel (design §8.3: "max abs error < 1e-4 relative to the output's max" — a 1000-token
// recurrence is not a 128-term sum, so it does not get the plain forward's 1e-5). Beyond the numbers, three properties:
//
//   * the checkpoint buffer is prefilled with garbage before the launch, so o matching the reference proves the forward
//     never reads it, while segment 0 matching zero proves the kernel wrote it *before* the first token's step;
//   * a second launch into another output with ckpt == NULL must reproduce the first one bit for bit (the checkpoints are
//     stores; letting them change the arithmetic would be a bug);
//   * replaying the last segment on the host from the kernel's own 64-KB checkpoint must reproduce the o the kernel wrote
//     for the tokens of that segment — the checkpoint's contents and the moment it was written, both.
//
// `bwd` runs the same batches (plus a uniform B = 8 at the dataset's median row length, for the one-call time) through
// gdn_fwd for the checkpoints and then gdn_bwd, and checks dq, dk, dv, dg and dbeta against ref_gdn_bwd. Two more
// properties there: every output buffer is garbage-filled first, so a block that skipped a segment shows up as a failed
// check, and the kernel's sp scratch (phase A's S_prev) is replayed from the same checkpoint on the host in double —
// which is the only check on phase A's contents, since ref_gdn_bwd keeps its S_prev copies to itself.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h"    // CHECK_REL/CHECK_ABS1, rnd_fill, dev_*, jt_ragged, the exit codes
#include "gdn.h"        // kernels/gdn.h: gdn_fwd and the checkpoint geometry
#include "../ref/gdn.h" // ref/gdn.h: ref_gdn_fwd / ref_gdn_bwd, the oracle

#define H 16            // the model's head count (design §3.1)
#define D JT_GDN_DIM    // Dk = Dv = 128, the dims the kernel bakes in
#define MAXROWS 8       // the widest batch here is the plan's B = 8
#define TOL 1e-4        // plan M3.2/M3.3 gate for the recurrence

static void *xalloc(size_t n) {
  void *p = malloc(n != 0 ? n : 1);
  if (p == NULL) {
    fprintf(stderr, "test_gdn: out of host memory (%zu bytes)\n", n);
    exit(1);
  }
  return p;
}

// ---- the forward (plan M3.2) ------------------------------------------------------------------------------------
// One forward step of design §3.6 on the host, in double, from the state a checkpoint holds: this is not the oracle (that
// is ref_gdn.c) but the check that the kernel's checkpoint is the state *before* the step at that token, written at the
// right moment. o_rep receives the replay's o for the tokens seg*64 .. len-1 of the row, row-major [tokens][D].
static void replay_last_segment(const float *S0, const float *q, const float *k, const float *v, const float *g,
                                const float *beta, int H_, int t_row0, int seg, int len, int h, float *o_rep) {
  double *S = (double *)xalloc((size_t)D * (size_t)D * sizeof *S);
  double delta[D];
  for (int i = 0; i < D; i++)
    for (int j = 0; j < D; j++) S[(size_t)i * D + j] = (double)S0[(size_t)i * D + j];
  const int t_from = seg * JT_GDN_SEG, t_to = len < t_from + JT_GDN_SEG ? len : t_from + JT_GDN_SEG;
  for (int tl = t_from; tl < t_to; tl++) {
    const size_t gi = (size_t)(t_row0 + tl) * (size_t)H_ + (size_t)h;
    const double e = exp((double)g[gi]), bt = (double)beta[gi];
    for (int j = 0; j < D; j++) { // S = S * exp(g_t);  kv[j] = sum_i S[i,j] k_t[i];  delta[j] = (v - kv) * beta
      double acc = 0.0;
      for (int i = 0; i < D; i++) {
        S[(size_t)i * D + j] *= e;
        acc += S[(size_t)i * D + j] * (double)k[gi * (size_t)D + (size_t)i];
      }
      delta[j] = ((double)v[gi * (size_t)D + (size_t)j] - acc) * bt;
    }
    for (int j = 0; j < D; j++) { // S[i,j] += k_t[i] delta[j];  o_t[j] = sum_i S[i,j] q_t[i]
      double acc = 0.0;
      for (int i = 0; i < D; i++) {
        S[(size_t)i * D + j] += (double)k[gi * (size_t)D + (size_t)i] * delta[j];
        acc += S[(size_t)i * D + j] * (double)q[gi * (size_t)D + (size_t)i];
      }
      o_rep[(size_t)(tl - t_from) * (size_t)D + (size_t)j] = (float)acc;
    }
  }
  free(S);
}

// The largest |x| over a [D][D] checkpoint slice.
static double ckpt_max_abs(const float *x) {
  double worst = 0.0;
  for (size_t i = 0; i < JT_GDN_CKPT_FLOATS; i++) {
    const double a = fabs((double)x[i]);
    if (a > worst) worst = a;
  }
  return worst;
}

// The checkpoint properties that need numbers: segment 0 of every (row, head) is exactly zero, and (replay_h >= 0) the last
// row's last segment replayed from head replay_h's checkpoint reproduces the o the kernel wrote for its tokens.
static void check_ckpt(const float *ck_d, const float *o_got, const float *q, const float *k, const float *v,
                       const float *g, const float *beta, const int *off, const int *lens, int B, int nseg,
                       const char *tag, int replay_h) {
  char name[160];
  float *slice = (float *)xalloc(JT_GDN_CKPT_FLOATS * sizeof *slice);
  double worst0 = 0.0;
  for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++) {
      const size_t at = (((size_t)b * (size_t)H + (size_t)h) * (size_t)nseg) * JT_GDN_CKPT_FLOATS;
      dev_download(ck_d + at, slice, JT_GDN_CKPT_FLOATS);
      const double m = ckpt_max_abs(slice);
      if (m > worst0) worst0 = m;
    }
  snprintf(name, sizeof name, "gdn_fwd_ckpt0_zero %s", tag);
  CHECK_ABS1(name, worst0, 0.0, 1e-12);

  if (replay_h >= 0) {
    const int b = B - 1, len = lens[b], seg = (len - 1) / JT_GDN_SEG;
    const int t_from = seg * JT_GDN_SEG, cnt = len - t_from;
    const size_t at = (((size_t)b * (size_t)H + (size_t)replay_h) * (size_t)nseg + (size_t)seg) * JT_GDN_CKPT_FLOATS;
    dev_download(ck_d + at, slice, JT_GDN_CKPT_FLOATS);
    float *o_rep = (float *)xalloc((size_t)cnt * (size_t)D * sizeof *o_rep);
    float *o_kern = (float *)xalloc((size_t)cnt * (size_t)D * sizeof *o_kern);
    replay_last_segment(slice, q, k, v, g, beta, H, off[b], seg, len, replay_h, o_rep);
    const float *o_seg = o_got + ((size_t)(off[b] + t_from) * (size_t)H + (size_t)replay_h) * (size_t)D;
    for (int tl = 0; tl < cnt; tl++) // the kernel's o for those tokens, [cnt][D] without the H stride
      memcpy(o_kern + (size_t)tl * (size_t)D, o_seg + (size_t)tl * (size_t)H * (size_t)D, (size_t)D * sizeof *o_kern);
    snprintf(name, sizeof name, "gdn_fwd_ckpt_replay %s b=%d h=%d seg=%d", tag, b, replay_h, seg);
    CHECK_REL(name, o_rep, o_kern, (size_t)cnt * (size_t)D, TOL);
    free(o_kern);
    free(o_rep);
  }
  free(slice);
}

// One packed batch: the forward against ref_gdn_fwd and the three checkpoint properties. The same host inputs feed the
// oracle and the kernel.
static void case_fwd(const int *lens, int B, const char *tag, int replay_h) {
  if (B < 1 || B > MAXROWS) {
    fprintf(stderr, "test_gdn: batch of %d rows; MAXROWS is %d\n", B, MAXROWS);
    exit(1);
  }
  int off[MAXROWS + 1];
  const int N = jt_ragged(lens, B, off, NULL);
  const size_t nq = (size_t)N * (size_t)H * (size_t)D; // q, k, v, o: [N][H][D]
  const size_t ng = (size_t)N * (size_t)H;             // g, beta: [N][H]
  const uint32_t seed = 4242u + (uint32_t)(B * 1000 + N);
  char name[160];

  float *q = (float *)xalloc(nq * sizeof *q), *k = (float *)xalloc(nq * sizeof *k);
  float *v = (float *)xalloc(nq * sizeof *v), *o_ref = (float *)xalloc(nq * sizeof *o_ref);
  float *o_got = (float *)xalloc(nq * sizeof *o_got), *o_null = (float *)xalloc(nq * sizeof *o_null);
  float *o_seq = (float *)xalloc(nq * sizeof *o_seq); // the sequential kernels' o
  float *o_ck = (float *)xalloc(nq * sizeof *o_ck);   // and the chunked path's, for the cross-implementation check
  float *g = (float *)xalloc(ng * sizeof *g), *beta = (float *)xalloc(ng * sizeof *beta);

  // Inputs in the ranges the model produces: q and k are l2-normed (q additionally scaled by 1/sqrt(128), design §3.6),
  // v is a post-silu conv output, g is a decay so it is <= 0 (design §3.6: g = -exp(A_log)*softplus(...)), beta in [0,1].
  const float qk = (float)(1.0 / sqrt((double)D));
  rnd_fill(q, nq, -qk, qk, seed);
  rnd_fill(k, nq, -qk, qk, seed + 1u);
  rnd_fill(v, nq, -1.0, 1.0, seed + 2u);
  rnd_fill(g, ng, -1.0, 0.0, seed + 3u);
  rnd_fill(beta, ng, 0.0, 1.0, seed + 4u);
  ref_gdn_fwd(B, off, H, D, D, q, k, v, g, beta, o_ref);

  const int nseg = gdn_ckpt_segs(off, B);
  const size_t nck = gdn_ckpt_floats(B, H, nseg);
  int *off_d = (int *)dev_upload_bytes(off, (size_t)(B + 1) * sizeof *off);
  float *q_d = dev_upload(q, nq), *k_d = dev_upload(k, nq), *v_d = dev_upload(v, nq);
  float *g_d = dev_upload(g, ng), *b_d = dev_upload(beta, ng);
  float *o_d = dev_alloc(nq * sizeof *o_got);
  float *ck_d = dev_alloc(nck * sizeof *ck_d);

  // The chunked path (plan M8.2) needs a cuBLAS handle and an arena; the arena is the size layer_gdn.c passes -- the
  // backward's sp buffer -- so the test exercises the trainer's sizing (gdn.h: gdn_chunk_arena_floats).
  gdn_chunk_plan plan;
  const size_t narena = gdn_bwd_sp_floats((size_t)B, H);
  CB(cublasCreate(&plan.blas));
  plan.arena = (float *)dev_alloc(narena * sizeof *plan.arena);
  plan.arena_n = narena;
  printf("    (main run: %s kernels; the cross-check covers both)\n",
         gdn_chunked_enabled() ? "chunked" : "sequential");

  // 0x7f7f7f7f is a huge finite float: if the forward ever read the checkpoints the numbers would show it, and if it
  // failed to write segment 0 the zero check below would see this pattern.
  CK(cudaMemset(ck_d, 0x7f, nck * sizeof *ck_d));
  gdn_fwd(q_d, k_d, v_d, g_d, b_d, off_d, B, H, nseg, NULL, o_d, &plan, 0); // warm-up: the first launch pays the context's cost
  CK(cudaDeviceSynchronize());
  double t0 = jt_now();
  gdn_fwd(q_d, k_d, v_d, g_d, b_d, off_d, B, H, nseg, ck_d, o_d, &plan, 0);
  CK(cudaDeviceSynchronize());
  const double ms = 1e3 * (jt_now() - t0);
  JT_SYNC_DEBUG;
  dev_download(o_d, o_got, nq);

  snprintf(name, sizeof name, "gdn_fwd %s", tag);
  CHECK_REL(name, o_got, o_ref, nq, TOL);

  // The same forward with no checkpoint buffer: same numbers, bit for bit (the stores are not part of the arithmetic),
  // and the time the checkpoint stores cost (design §8.3 trades that memory for the backward's replay). nseg is passed
  // even without a buffer so the *same* implementation runs both times (the chunked path needs the chunk count).
  t0 = jt_now();
  gdn_fwd(q_d, k_d, v_d, g_d, b_d, off_d, B, H, nseg, NULL, o_d, &plan, 0);
  CK(cudaDeviceSynchronize());
  const double ms_nockpt = 1e3 * (jt_now() - t0);
  JT_SYNC_DEBUG;
  dev_download(o_d, o_null, nq);
  if (jt_has_card())
    printf("    gdn_fwd %-46s N=%d  %.2f ms with ckpt, %.2f ms without (one call each)\n", tag, N, ms, ms_nockpt);
  snprintf(name, sizeof name, "gdn_fwd_no_ckpt_identical %s", tag);
  CHECK_ABS1(name, abs_err(o_null, o_got, nq), 0.0, 1e-30);

  if (jt_has_card()) check_ckpt(ck_d, o_got, q, k, v, g, beta, off, lens, B, nseg, tag, replay_h);

  // The two implementations of the same function, in one build and whatever the ambient setting: JT_GDN_CHUNKED=1 runs
  // the chunked (plan M8.2) path, =0 the sequential kernels, and the two must agree to rounding (same function, different
  // summation order). The ambient value is restored afterwards, so the rest of the run follows what it asked for.
  const char *amb = getenv("JT_GDN_CHUNKED");
  setenv("JT_GDN_CHUNKED", "1", 1);
  gdn_fwd(q_d, k_d, v_d, g_d, b_d, off_d, B, H, nseg, NULL, o_d, &plan, 0);
  CK(cudaDeviceSynchronize());
  dev_download(o_d, o_ck, nq);
  setenv("JT_GDN_CHUNKED", "0", 1);
  gdn_fwd(q_d, k_d, v_d, g_d, b_d, off_d, B, H, nseg, NULL, o_d, &plan, 0);
  CK(cudaDeviceSynchronize());
  dev_download(o_d, o_seq, nq);
  if (amb != NULL) setenv("JT_GDN_CHUNKED", amb, 1); else unsetenv("JT_GDN_CHUNKED");
  snprintf(name, sizeof name, "gdn_fwd chunked_vs_sequential %s", tag);
  CHECK_REL(name, o_ck, o_seq, nq, TOL);

  dev_free(plan.arena);
  CB(cublasDestroy(plan.blas));
  dev_free(ck_d);
  dev_free(o_d);
  dev_free(b_d);
  dev_free(g_d);
  dev_free(v_d);
  dev_free(k_d);
  dev_free(q_d);
  dev_free(off_d);
  free(beta);
  free(g);
  free(o_ck);
  free(o_seq);
  free(o_null);
  free(o_got);
  free(o_ref);
  free(v);
  free(k);
  free(q);
}

// The plan's M3.2 batches, at the model's dims. replay_h picks the head whose last segment is replayed from the
// checkpoint (-1: no replay); the two shortest shapes with a segment boundary get one.
static const int LEN_1[] = {1};
static const int LEN_2[] = {2};
static const int LEN_3[] = {63, 64, 65};                                   // the 64-token segment boundary exactly
static const int LEN_4[] = {200, 7, 130};
static const int LEN_8[] = {1026, 520, 670, 300, 811, 455, 610, 702};      // the full ragged batch, 1026 > 1024

static int group_fwd(void) {
  case_fwd(LEN_1, 1, "B=1 {1}", -1);
  case_fwd(LEN_2, 1, "B=1 {2}", -1);
  case_fwd(LEN_3, 3, "B=3 {63,64,65}", 5);
  case_fwd(LEN_4, 3, "B=3 {200,7,130}", 0);
  case_fwd(LEN_8, 8, "B=8 {1026,520,670,300,811,455,610,702}", -1);
  return jt_exit_code();
}

// ---- the backward (plan M3.3) -----------------------------------------------------------------------------------
// Phase A's own check: design §3.6's step on the host in double, started from the same checkpoint the kernel started
// from, keeping S_prev for every token of one segment. Sp receives [cnt][D][D] with tl slowest. This is the check that
// the kernel's sp scratch holds S_prev -- the state *before* each step -- at the right token, which is what phase B
// reads; ref_gdn_bwd keeps its own S_prev copies internally and does not expose them (J13/J16 notes).
static void replay_segment_sp(const float *S0, const float *k, const float *v, const float *g, const float *beta,
                              int H_, int t_row0, int seg, int cnt, int h, double *Sp) {
  double *S = (double *)xalloc((size_t)D * (size_t)D * sizeof *S);
  double delta[D];
  for (int i = 0; i < D; i++)
    for (int j = 0; j < D; j++) S[(size_t)i * D + j] = (double)S0[(size_t)i * D + j];
  for (int tl = 0; tl < cnt; tl++) {
    memcpy(Sp + (size_t)tl * D * D, S, (size_t)D * D * sizeof *Sp); // S_prev, before the step
    const size_t gi = (size_t)(t_row0 + seg * JT_GDN_SEG + tl) * (size_t)H_ + (size_t)h;
    const double e = exp((double)g[gi]), bt = (double)beta[gi];
    for (int j = 0; j < D; j++) { // S = S * exp(g_t);  kv[j] = sum_i S[i,j] k_t[i];  delta[j] = (v - kv) * beta
      double acc = 0.0;
      for (int i = 0; i < D; i++) {
        S[(size_t)i * D + j] *= e;
        acc += S[(size_t)i * D + j] * (double)k[gi * (size_t)D + (size_t)i];
      }
      delta[j] = ((double)v[gi * (size_t)D + (size_t)j] - acc) * bt;
    }
    for (int j = 0; j < D; j++) // S[i,j] += k_t[i] delta[j]
      for (int i = 0; i < D; i++) S[(size_t)i * D + j] += (double)k[gi * (size_t)D + (size_t)i] * delta[j];
  }
  free(S);
}

// Compare the kernel's sp slice for one block (b, h, tile, segment 0 -- the last phase A a block runs) against the
// replay above. The buffer is garbage-filled before the launches, so a block that never wrote its segment is caught.
static void check_sp(const float *sp_d, const float *ck_d, const float *k, const float *v, const float *g,
                     const float *beta, const int *off, int B, int nseg, int h, int tile, const char *tag) {
  const int b = B - 1, len = off[b + 1] - off[b];
  const int cnt = len < JT_GDN_SEG ? len : JT_GDN_SEG; // segment 0's tokens
  const size_t blk = (size_t)JT_GDN_SEG * (size_t)D * (size_t)JT_GDN_TILE;
  const size_t at = (((size_t)b * (size_t)H + (size_t)h) * (size_t)JT_GDN_TILES + (size_t)tile) * blk;
  float *sp_h = (float *)xalloc(blk * sizeof *sp_h);
  float *ck0 = (float *)xalloc(JT_GDN_CKPT_FLOATS * sizeof *ck0);
  dev_download(sp_d + at, sp_h, blk);
  dev_download(ck_d + (((size_t)b * (size_t)H + (size_t)h) * (size_t)nseg) * JT_GDN_CKPT_FLOATS, ck0,
               JT_GDN_CKPT_FLOATS);
  // The replay's full [cnt][D][D] and the tile's columns of it, which is what one block holds in sp.
  double *Sp = (double *)xalloc((size_t)cnt * D * D * sizeof *Sp);
  float *want = (float *)xalloc((size_t)cnt * D * JT_GDN_TILE * sizeof *want);
  replay_segment_sp(ck0, k, v, g, beta, H, off[b], 0, cnt, h, Sp);
  for (int tl = 0; tl < cnt; tl++)
    for (int i = 0; i < D; i++)
      for (int l = 0; l < JT_GDN_TILE; l++)
        want[((size_t)tl * D + (size_t)i) * JT_GDN_TILE + (size_t)l] = (float)Sp[((size_t)tl * D + (size_t)i) * D +
                                                                            (size_t)tile * JT_GDN_TILE + (size_t)l];
  char name[160];
  snprintf(name, sizeof name, "gdn_bwd_sp %s b=%d h=%d tile=%d seg=0", tag, b, h, tile);
  CHECK_REL(name, sp_h, want, (size_t)cnt * D * JT_GDN_TILE, TOL);
  // The same tile read straight out of the segment's checkpoint, for tl = 0: exact, sp[0] is the state before the step.
  double worst = 0.0;
  for (int i = 0; i < D; i++)
    for (int l = 0; l < JT_GDN_TILE; l++) {
      const double a = fabs((double)sp_h[(size_t)i * JT_GDN_TILE + (size_t)l] -
                            (double)ck0[(size_t)i * D + (size_t)tile * JT_GDN_TILE + (size_t)l]);
      if (a > worst) worst = a;
    }
  snprintf(name, sizeof name, "gdn_bwd_sp_ckpt %s b=%d h=%d tile=%d", tag, b, h, tile);
  // sp[0] is the segment-0 checkpoint itself (all zeros), so this also proves the block wrote its slice: sp was
  // garbage-filled before the launch.
  CHECK_ABS1(name, worst, 0.0, 1e-30);
  free(want);
  free(Sp);
  free(ck0);
  free(sp_h);
}

// Every [D] row of x (n = rows * D, one row = one (token, head) GDN row) scaled to have exactly norm `target`: the shape
// the upstream l2norm kernel gives q and k (design §3.6), so a test can use the model's magnitudes instead of a box.
static void l2n_rows(float *x, size_t n, float target) {
  for (size_t r = 0; r < n / D; r++) {
    double s = 0.0;
    for (int i = 0; i < D; i++) s += (double)x[r * D + i] * (double)x[r * D + i];
    const float a = (float)((double)target / sqrt(s));
    for (int i = 0; i < D; i++) x[r * D + i] *= a;
  }
}

// One packed batch: the five gradients against ref_gdn_bwd, plus phase A's scratch. Inputs are case_fwd's (same
// generators, same ranges, different seeds), with a cotangent dout in [-1, 1]. norm_qk renormalizes every (token, head)
// row of q and k to the norms the model's recurrence sees -- |k| = 1 and |q| = 1/sqrt(128) after l2norm (design §3.6),
// which is where the delta rule's write is at its largest and the test is at its least forgiving.
static void case_bwd(const int *lens, int B, const char *tag, int sp_h, int norm_qk) {
  if (B < 1 || B > MAXROWS) {
    fprintf(stderr, "test_gdn: batch of %d rows; MAXROWS is %d\n", B, MAXROWS);
    exit(1);
  }
  int off[MAXROWS + 1];
  const int N = jt_ragged(lens, B, off, NULL);
  const size_t nq = (size_t)N * (size_t)H * (size_t)D, ng = (size_t)N * (size_t)H;
  const uint32_t seed = 9011u + (uint32_t)(B * 1000 + N);
  char name[160];

  float *q = (float *)xalloc(nq * sizeof *q), *k = (float *)xalloc(nq * sizeof *k);
  float *v = (float *)xalloc(nq * sizeof *v), *dout = (float *)xalloc(nq * sizeof *dout);
  float *g = (float *)xalloc(ng * sizeof *g), *beta = (float *)xalloc(ng * sizeof *beta);
  float *dq_r = (float *)xalloc(nq * sizeof *dq_r), *dk_r = (float *)xalloc(nq * sizeof *dk_r);
  float *dv_r = (float *)xalloc(nq * sizeof *dv_r), *dg_r = (float *)xalloc(ng * sizeof *dg_r);
  float *db_r = (float *)xalloc(ng * sizeof *db_r);
  float *dq = (float *)xalloc(nq * sizeof *dq), *dk = (float *)xalloc(nq * sizeof *dk);
  float *dv = (float *)xalloc(nq * sizeof *dv), *dg = (float *)xalloc(ng * sizeof *dg);
  float *db = (float *)xalloc(ng * sizeof *db);
  const float qk = (float)(1.0 / sqrt((double)D));
  rnd_fill(q, nq, -qk, qk, seed);
  rnd_fill(k, nq, -qk, qk, seed + 1u);
  rnd_fill(v, nq, -1.0, 1.0, seed + 2u);
  rnd_fill(dout, nq, -1.0, 1.0, seed + 3u);
  rnd_fill(g, ng, -1.0, 0.0, seed + 4u);
  rnd_fill(beta, ng, 0.0, 1.0, seed + 5u);
  if (norm_qk) { // every row to the model's own norm: |k| = 1, |q| = 1/sqrt(128)
    l2n_rows(q, nq, (float)(1.0 / sqrt((double)D)));
    l2n_rows(k, nq, 1.0f);
  }

  ref_gdn_bwd(B, off, H, D, D, q, k, v, g, beta, dout, dq_r, dk_r, dv_r, dg_r, db_r);

  const int nseg = gdn_ckpt_segs(off, B);
  const size_t nck = gdn_ckpt_floats(B, H, nseg);
  const size_t nsp = gdn_bwd_sp_floats(B, H);
  const size_t npart = gdn_bwd_part_floats((size_t)N, H), ngate = gdn_bwd_gate_floats((size_t)N, H);
  int *off_d = (int *)dev_upload_bytes(off, (size_t)(B + 1) * sizeof *off);
  float *q_d = dev_upload(q, nq), *k_d = dev_upload(k, nq), *v_d = dev_upload(v, nq);
  float *do_d = dev_upload(dout, nq), *g_d = dev_upload(g, ng), *b_d = dev_upload(beta, ng);
  float *o_d = dev_alloc(nq * sizeof *o_d), *ck_d = dev_alloc(nck * sizeof *ck_d);
  gdn_chunk_plan plan;
  const size_t narena = gdn_bwd_sp_floats((size_t)B, H); // the arena layer_gdn.c passes: the sp buffer itself
  CB(cublasCreate(&plan.blas));
  plan.arena = (float *)dev_alloc(narena * sizeof *plan.arena);
  plan.arena_n = narena;
  gdn_bwd_work wk;
  wk.sp = (float *)dev_alloc(nsp * sizeof *wk.sp);
  wk.dq_part = (float *)dev_alloc(npart * sizeof *wk.dq_part);
  wk.dk_part = (float *)dev_alloc(npart * sizeof *wk.dk_part);
  wk.dbeta_part = (float *)dev_alloc(ngate * sizeof *wk.dbeta_part);
  wk.dg_part = (float *)dev_alloc(ngate * sizeof *wk.dg_part);
  float *dq_d = dev_alloc(nq * sizeof *dq_d), *dk_d = dev_alloc(nq * sizeof *dk_d);
  float *dv_d = dev_alloc(nq * sizeof *dv_d), *dg_d = dev_alloc(ng * sizeof *dg_d);
  float *db_d = dev_alloc(ng * sizeof *db_d);
  // Garbage in every buffer the backward writes: a block that skips its segment shows up as a failed check rather than
  // as a stale value from the batch before.
  CK(cudaMemset(wk.sp, 0x7f, nsp * sizeof *wk.sp));
  CK(cudaMemset(dq_d, 0x7f, nq * sizeof *dq_d));
  CK(cudaMemset(dk_d, 0x7f, nq * sizeof *dk_d));
  CK(cudaMemset(dv_d, 0x7f, nq * sizeof *dv_d));
  CK(cudaMemset(dg_d, 0x7f, ng * sizeof *dg_d));
  CK(cudaMemset(db_d, 0x7f, ng * sizeof *db_d));

  // Warm-up: the first launch that writes a freshly allocated buffer is many times slower than the next one on gpubox
  // (WSL2's driver maps the pages lazily), so the timed calls follow one full pass over the same buffers.
  gdn_fwd(q_d, k_d, v_d, g_d, b_d, off_d, B, H, nseg, ck_d, o_d, &plan, 0);
  gdn_bwd(q_d, k_d, v_d, g_d, b_d, do_d, off_d, B, H, N, nseg, ck_d, &wk, dv_d, dq_d, dk_d, dg_d, db_d, &plan, 0);
  CK(cudaDeviceSynchronize());
  double t0 = jt_now();
  gdn_fwd(q_d, k_d, v_d, g_d, b_d, off_d, B, H, nseg, ck_d, o_d, &plan, 0);
  CK(cudaDeviceSynchronize());
  const double ms_fwd = 1e3 * (jt_now() - t0);
  t0 = jt_now();
  gdn_bwd(q_d, k_d, v_d, g_d, b_d, do_d, off_d, B, H, N, nseg, ck_d, &wk, dv_d, dq_d, dk_d, dg_d, db_d, &plan, 0);
  CK(cudaDeviceSynchronize());
  const double ms_bwd = 1e3 * (jt_now() - t0);
  double ms_bwd_best = ms_bwd;
  for (int rep = 0; rep < 3; rep++) {
    t0 = jt_now();
    gdn_bwd(q_d, k_d, v_d, g_d, b_d, do_d, off_d, B, H, N, nseg, ck_d, &wk, dv_d, dq_d, dk_d, dg_d, db_d, &plan, 0);
    CK(cudaDeviceSynchronize());
    const double ms = 1e3 * (jt_now() - t0);
    if (ms < ms_bwd_best) ms_bwd_best = ms;
  }
  JT_SYNC_DEBUG;
  if (jt_has_card())
    printf("    gdn_bwd %-46s N=%d  fwd %.2f ms, bwd %.2f ms (one call each; bwd best of 4 warm %.2f ms)\n", tag, N,
           ms_fwd, ms_bwd, ms_bwd_best);

  dev_download(dq_d, dq, nq);
  dev_download(dk_d, dk, nq);
  dev_download(dv_d, dv, nq);
  dev_download(dg_d, dg, ng);
  dev_download(db_d, db, ng);
  snprintf(name, sizeof name, "gdn_bwd dq %s", tag);
  CHECK_REL(name, dq, dq_r, nq, TOL);
  snprintf(name, sizeof name, "gdn_bwd dk %s", tag);
  CHECK_REL(name, dk, dk_r, nq, TOL);
  snprintf(name, sizeof name, "gdn_bwd dv %s", tag);
  CHECK_REL(name, dv, dv_r, nq, TOL);
  snprintf(name, sizeof name, "gdn_bwd dg %s", tag);
  CHECK_REL(name, dg, dg_r, ng, TOL);
  snprintf(name, sizeof name, "gdn_bwd dbeta %s", tag);
  CHECK_REL(name, db, db_r, ng, TOL);
  // sp is the chunked path's arena (plan M8.2 reuses the backward's scratch), so phase A's layout check belongs to the
  // sequential kernels only.
  if (jt_has_card() && sp_h >= 0 && !gdn_chunked_enabled())
    check_sp(wk.sp, ck_d, k, v, g, beta, off, B, nseg, sp_h, 1, tag);

  dev_free(db_d);
  dev_free(dg_d);
  dev_free(dv_d);
  dev_free(dk_d);
  dev_free(dq_d);
  dev_free(wk.dg_part);
  dev_free(wk.dbeta_part);
  dev_free(wk.dk_part);
  dev_free(wk.dq_part);
  dev_free(plan.arena);
  CB(cublasDestroy(plan.blas));
  dev_free(wk.sp);
  dev_free(ck_d);
  dev_free(o_d);
  dev_free(b_d);
  dev_free(g_d);
  dev_free(do_d);
  dev_free(v_d);
  dev_free(k_d);
  dev_free(q_d);
  dev_free(off_d);
  free(db);
  free(dg);
  free(dv);
  free(dk);
  free(dq);
  free(db_r);
  free(dg_r);
  free(dv_r);
  free(dk_r);
  free(dq_r);
  free(beta);
  free(g);
  free(dout);
  free(v);
  free(k);
  free(q);
}

// The same five batches as the forward, plus a uniform B = 8 at the dataset's median row length (about 520 tokens), which
// is the shape plan M3.3 asks the one-call time for.
static const int LEN_8P50[] = {520, 520, 520, 520, 520, 520, 520, 520};

static int group_bwd(void) {
  case_bwd(LEN_1, 1, "B=1 {1}", 7, 0);
  case_bwd(LEN_2, 1, "B=1 {2}", 7, 0);
  case_bwd(LEN_3, 3, "B=3 {63,64,65}", 5, 0);
  case_bwd(LEN_4, 3, "B=3 {200,7,130}", 0, 0);
  case_bwd(LEN_8, 8, "B=8 {1026,520,670,300,811,455,610,702}", 3, 0);
  case_bwd(LEN_8P50, 8, "B=8 {520 x 8}", 11, 0);
  case_bwd(LEN_4, 3, "B=3 {200,7,130} l2n q,k", 2, 1); // the model's own |k| = 1, |q| = 1/sqrt(128)
  return jt_exit_code();
}

// ---- dispatcher (test_kernels' contract, plan M2.0) ---------------------------------------------------------------
typedef struct {
  const char *name;
  int (*run)(void);
} jt_part;

static const jt_part PARTS[] = {{"fwd", group_fwd}, {"bwd", group_bwd}};
#define NPARTS ((int)(sizeof(PARTS) / sizeof(PARTS[0])))

static void usage(FILE *f) {
  fprintf(f, "usage: test_gdn [fwd|bwd|all]\n");
  fprintf(f, "  fwd    the forward at H=16, D=128 on the plan's M3.2 batches (card checkpoint m3-fwd)\n");
  fprintf(f, "  bwd    the backward, five gradients (plan M3.3)\n");
  fprintf(f, "  all    both (card checkpoint m3); a stub fails\n");
  fprintf(f, "  none   the implemented parts, stubs skipped, exit 0 (make test-null)\n");
}

// A part's return value decides the exit code, and only the three documented values do.
static int part_rc(int rc) { return rc == JT_OK ? JT_OK : (rc == JT_NOT_IMPL ? JT_NOT_IMPL : JT_FAIL); }

// strict (argv[1] == "all"): a stub is a failure. Otherwise (no argument) a stub is skipped: the null-device run only
// proves that what exists launches.
static int run_all(int strict) {
  int passed = 0, failed = 0, stubs = 0;
  for (int i = 0; i < NPARTS; i++) {
    printf("== %s\n", PARTS[i].name);
    fflush(stdout);
    const int rc = part_rc(PARTS[i].run());
    if (rc == JT_NOT_IMPL) {
      stubs++;
      if (!strict) printf("skip %s (not implemented)\n", PARTS[i].name);
    } else if (rc == JT_OK) {
      passed++;
    } else {
      failed++;
      printf("FAIL %s\n", PARTS[i].name);
    }
  }
  if (failed) {
    printf("gdn FAIL: %d part(s) failed\n", failed);
    return JT_FAIL;
  }
  if (strict && stubs) {
    printf("gdn FAIL all: %d part(s) not implemented\n", stubs);
    return JT_FAIL;
  }
  if (strict)
    printf("gdn OK all: %d part(s)\n", passed);
  else
    printf("gdn OK: %d implemented part(s) passed, %d stub(s) skipped\n", passed, stubs);
  return JT_OK;
}

int main(int argc, char **argv) {
  if (argc == 1) return run_all(0);
  if (argc > 2) {
    usage(stderr);
    return JT_USAGE;
  }
  if (strcmp(argv[1], "all") == 0) return run_all(1);
  if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
    usage(stdout);
    return JT_OK;
  }
  for (int i = 0; i < NPARTS; i++) {
    if (strcmp(argv[1], PARTS[i].name) != 0) continue;
    printf("== %s\n", PARTS[i].name);
    fflush(stdout);
    return part_rc(PARTS[i].run());
  }
  fprintf(stderr, "test_gdn: unknown part '%s'\n", argv[1]);
  usage(stderr);
  return JT_USAGE;
}
