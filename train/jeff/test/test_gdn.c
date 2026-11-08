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

  // 0x7f7f7f7f is a huge finite float: if the forward ever read the checkpoints the numbers would show it, and if it
  // failed to write segment 0 the zero check below would see this pattern.
  CK(cudaMemset(ck_d, 0x7f, nck * sizeof *ck_d));
  gdn_fwd(q_d, k_d, v_d, g_d, b_d, off_d, B, H, nseg, NULL, o_d, 0); // warm-up: the first launch pays the context's cost
  CK(cudaDeviceSynchronize());
  double t0 = jt_now();
  gdn_fwd(q_d, k_d, v_d, g_d, b_d, off_d, B, H, nseg, ck_d, o_d, 0);
  CK(cudaDeviceSynchronize());
  const double ms = 1e3 * (jt_now() - t0);
  JT_SYNC_DEBUG;
  dev_download(o_d, o_got, nq);

  snprintf(name, sizeof name, "gdn_fwd %s", tag);
  CHECK_REL(name, o_got, o_ref, nq, TOL);

  // The same forward with no checkpoint buffer: same numbers, bit for bit (the stores are not part of the arithmetic),
  // and the time the checkpoint stores cost (design §8.3 trades that memory for the backward's replay).
  t0 = jt_now();
  gdn_fwd(q_d, k_d, v_d, g_d, b_d, off_d, B, H, 0, NULL, o_d, 0);
  CK(cudaDeviceSynchronize());
  const double ms_nockpt = 1e3 * (jt_now() - t0);
  JT_SYNC_DEBUG;
  dev_download(o_d, o_null, nq);
  if (jt_has_card())
    printf("    gdn_fwd %-46s N=%d  %.2f ms with ckpt, %.2f ms without (one call each)\n", tag, N, ms, ms_nockpt);
  snprintf(name, sizeof name, "gdn_fwd_no_ckpt_identical %s", tag);
  CHECK_ABS1(name, abs_err(o_null, o_got, nq), 0.0, 1e-30);

  if (jt_has_card()) check_ckpt(ck_d, o_got, q, k, v, g, beta, off, lens, B, nseg, tag, replay_h);

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
// Not written yet: a later task fills this in (plan M3.3: five gradients against ref_gdn_bwd, the same batches). Until
// then the group is a stub, so `test_gdn all` fails on it (that is card checkpoint m3's line) and the no-argument run
// skips it.
static int group_bwd(void) {
  printf("NOT IMPLEMENTED\n");
  return JT_NOT_IMPL;
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
