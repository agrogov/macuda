// test/k_misc.c — the `misc` group of test_kernels (plan M2.1 on the plan M2.0 harness).
//
// The plumbing kernels of design §8.1, each against its CPU reference on the same random inputs: zero, add_inplace,
// scale, bf16_to_f32, embed_gather, qgate_split/qgate_merge, kv_expand/kv_reduce, to_head_major/from_head_major.
// Every op here is a copy, a sum or a widen, so the forward gate is JT_TOL_FWD and the expected error is 0 — a nonzero
// rel means a layout is wrong, not that rounding drifted.
//
// Two checks do not go through ref_* at all, because they are properties of the *pair* of kernels rather than of one:
//   * qgate_merge(qgate_split(x)) == x  (the split/merge interleave is an identity on [rows, 4096])
//   * <kv_expand(a), b> == <a, kv_reduce(b)>  (kv_reduce is the adjoint of kv_expand), rel < 1e-5
// Together with the ref comparison they pin the layout from two sides: a shared misunderstanding in ref/ shows up in
// one of the two.
//
// Shapes: the plan's two ragged batches (token counts 168 and 2516, the second one crossing 1024), plus rows = 1 for
// the row-structured ops. On the null device every CHECK_* prints "(launch only)" and counts nothing (harness.h), so
// `make test-null` only proves that these launches and their link line are right.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h" // CHECK_REL / CHECK_REL1 / rnd_fill / dev_* / the ragged batches / the bf16 helpers
#include "misc.h"    // kernels/misc.h, through -Ikernels (test/ has no misc.h); the jt_* wrappers under test

// ref/misc.h is reached by its path from test/, not as "misc.h": both headers are called misc.h and the kernels' one
// wins the -I search from this directory. Inside ref/misc.h, "real.h" resolves next to it.
#include "../ref/misc.h"

// The model's row layouts (kernels/misc.h); the test also uses smaller ones to exercise the stride loops.
#define ATTN_H JT_ATTN_HEADS
#define ATTN_D JT_ATTN_HEAD_DIM
#define KVH JT_ATTN_KV_HEADS
#define KVG JT_ATTN_KV_GROUP

static void *jtm_alloc(size_t nbytes) {
  void *p = malloc(nbytes ? nbytes : 1);
  if (p == NULL) {
    fprintf(stderr, "k_misc: out of memory (%zu bytes)\n", nbytes);
    exit(1);
  }
  return p;
}

// The plan's inner products are summed in double whatever `real` is: both sides of the adjoint check are sums of
// thousands of products, and the point of the check is the layout, not the summation order.
static double dot64(const float *a, const float *b, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; i++) s += (double)a[i] * (double)b[i];
  return s;
}

// ---- zero / add_inplace / scale / bf16_to_f32 -----------------------------------------------------------------------
// Each case returns early for an empty shape, which is the contract the kernels have as well (a zero dimension is a
// no-op). It also keeps the compiler from having to prove that a size is positive before it can see the buffers below
// being written.
static void t_elementwise(int n, uint32_t seed) {
  if (n <= 0) return;
  const size_t nel = (size_t)n;
  float *a = jtm_alloc(nel * sizeof *a);
  float *b = jtm_alloc(nel * sizeof *b);
  float *want = jtm_alloc(nel * sizeof *want);
  float *got = jtm_alloc(nel * sizeof *got);
  char name[80];

  rnd_fill(a, nel, -3.0, 3.0, seed);
  rnd_fill(b, nel, -3.0, 3.0, seed + 1);

  // zero: the wanted buffer is all +0.0, so rel_err is 0 only if every element really is 0 (scale = 0 -> the 1e-30
  // floor in rel_err turns any nonzero element into a huge rel, and a NaN into infinity).
  memcpy(want, a, nel * sizeof *want);
  ref_zero(want, nel);
  float *da = dev_upload(a, nel);
  jt_zero(da, nel, 0);
  dev_download(da, got, nel);
  snprintf(name, sizeof name, "zero n=%d", n);
  CHECK_REL(name, got, want, nel, JT_TOL_FWD);
  dev_free(da);

  // add_inplace: y += x, with y and x different buffers and then the same buffer (the alias the residuals use).
  memcpy(want, a, nel * sizeof *want);
  ref_add_inplace(want, b, nel);
  da = dev_upload(a, nel);
  float *db = dev_upload(b, nel);
  jt_add_inplace(da, db, nel, 0);
  dev_download(da, got, nel);
  snprintf(name, sizeof name, "add_inplace n=%d", n);
  CHECK_REL(name, got, want, nel, JT_TOL_FWD);
  jt_add_inplace(db, db, nel, 0);            // y == x: the double-add form
  dev_download(db, got, nel);
  memcpy(want, b, nel * sizeof *want);
  ref_add_inplace(want, b, nel);
  snprintf(name, sizeof name, "add_inplace self n=%d", n);
  CHECK_REL(name, got, want, nel, JT_TOL_FWD);
  dev_free(da);
  dev_free(db);

  // scale: x *= a, once with a = 2 and once with a = 0 (an in-place zeroing, which the buffers rely on elsewhere).
  memcpy(want, a, nel * sizeof *want);
  ref_scale(want, REAL(2.0), nel);
  da = dev_upload(a, nel);
  jt_scale(da, 2.0f, nel, 0);
  dev_download(da, got, nel);
  snprintf(name, sizeof name, "scale n=%d", n);
  CHECK_REL(name, got, want, nel, JT_TOL_FWD);
  jt_scale(da, 0.0f, nel, 0);
  dev_download(da, got, nel);
  memcpy(want, a, nel * sizeof *want);
  ref_scale(want, REAL(0.0), nel);
  snprintf(name, sizeof name, "scale 0 n=%d", n);
  CHECK_REL(name, got, want, nel, JT_TOL_FWD);
  dev_free(da);

  // bf16_to_f32: widen a bf16 buffer built from the same random floats; the reference widens it independently.
  unsigned short *bf = jtm_alloc(nel * sizeof *bf);
  for (size_t i = 0; i < nel; i++) bf[i] = jt_f32_to_bf16(a[i]);
  ref_bf16_to_f32(bf, want, nel);
  unsigned short *dbf = (unsigned short *)dev_upload_bytes(bf, nel * sizeof *bf);
  da = dev_alloc(nel * sizeof *got);
  jt_bf16_to_f32_n(dbf, da, nel, 0);
  dev_download(da, got, nel);
  snprintf(name, sizeof name, "bf16_to_f32 n=%d", n);
  CHECK_REL(name, got, want, nel, JT_TOL_FWD);
  dev_free(dbf);
  dev_free(da);

  free(bf);
  free(a);
  free(b);
  free(want);
  free(got);
}

// ---- embed_gather ---------------------------------------------------------------------------------------------------
static void t_embed(int n_tok, int vocab, int d, uint32_t seed) {
  if (n_tok <= 0 || vocab <= 0 || d <= 0) return;
  const size_t tab_el = (size_t)vocab * (size_t)d;
  const size_t out_el = (size_t)n_tok * (size_t)d;
  float *tfloat = jtm_alloc(tab_el * sizeof *tfloat);
  unsigned short *tab = jtm_alloc(tab_el * sizeof *tab);
  int *ids = jtm_alloc((size_t)n_tok * sizeof *ids);
  float *want = jtm_alloc(out_el * sizeof *want);
  float *got = jtm_alloc(out_el * sizeof *got);
  char name[80];

  rnd_fill(tfloat, tab_el, -1.0, 1.0, seed);
  for (size_t i = 0; i < tab_el; i++) tab[i] = jt_f32_to_bf16(tfloat[i]);
  uint32_t s = seed + 7u;
  for (int t = 0; t < n_tok; t++) ids[t] = (int)jt_rng_below(&s, (uint32_t)vocab);

  ref_embed_gather(ids, tab, want, n_tok, d);

  int *dids = (int *)dev_upload_bytes(ids, (size_t)n_tok * sizeof *ids);
  unsigned short *dtab = (unsigned short *)dev_upload_bytes(tab, tab_el * sizeof *tab);
  float *dout = dev_alloc(out_el * sizeof *got);
  jt_embed_gather(dids, dtab, dout, n_tok, d, 0);
  dev_download(dout, got, out_el);
  snprintf(name, sizeof name, "embed_gather n=%d vocab=%d d=%d", n_tok, vocab, d);
  CHECK_REL(name, got, want, out_el, JT_TOL_FWD);

  dev_free(dids);
  dev_free(dtab);
  dev_free(dout);
  free(tfloat);
  free(tab);
  free(ids);
  free(want);
  free(got);
}

// ---- qgate_split / qgate_merge --------------------------------------------------------------------------------------
static void t_qgate(int rows, uint32_t seed) {
  if (rows <= 0) return;
  const size_t nqg = (size_t)rows * (size_t)ATTN_H * 2u * (size_t)ATTN_D;
  const size_t nq = (size_t)rows * (size_t)ATTN_H * (size_t)ATTN_D;
  float *qg = jtm_alloc(nqg * sizeof *qg);
  float *q = jtm_alloc(nq * sizeof *q);
  float *gate = jtm_alloc(nq * sizeof *gate);
  float *want_a = jtm_alloc(nq * sizeof *want_a);
  float *want_b = jtm_alloc(nq * sizeof *want_b);
  float *got_a = jtm_alloc(nq * sizeof *got_a);
  float *got_b = jtm_alloc(nq * sizeof *got_b);
  float *want_qg = jtm_alloc(nqg * sizeof *want_qg);
  float *got_qg = jtm_alloc(nqg * sizeof *got_qg);
  char name[80];

  rnd_fill(qg, nqg, -2.0, 2.0, seed);
  rnd_fill(q, nq, -2.0, 2.0, seed + 1);
  rnd_fill(gate, nq, -2.0, 2.0, seed + 2);

  // split: one [rows, 4096] becomes two [rows, 2048] (per head, 256 query columns then 256 gate columns).
  ref_qgate_split(qg, want_a, want_b, rows, ATTN_H, ATTN_D);
  float *dqg = dev_upload(qg, nqg);
  float *dq = dev_alloc(nq * sizeof *q);
  float *dgate = dev_alloc(nq * sizeof *gate);
  jt_qgate_split(dqg, dq, dgate, rows, ATTN_H, ATTN_D, 0);
  dev_download(dq, got_a, nq);
  dev_download(dgate, got_b, nq);
  snprintf(name, sizeof name, "qgate_split q rows=%d", rows);
  CHECK_REL(name, got_a, want_a, nq, JT_TOL_FWD);
  snprintf(name, sizeof name, "qgate_split gate rows=%d", rows);
  CHECK_REL(name, got_b, want_b, nq, JT_TOL_FWD);

  // merge: two [rows, 2048] back into one [rows, 4096] (the interleave a q_proj output has).
  ref_qgate_merge(q, gate, want_qg, rows, ATTN_H, ATTN_D);
  float *dq_in = dev_upload(q, nq);
  float *dgate_in = dev_upload(gate, nq);
  float *dqg_out = dev_alloc(nqg * sizeof *qg);
  jt_qgate_merge(dq_in, dgate_in, dqg_out, rows, ATTN_H, ATTN_D, 0);
  dev_download(dqg_out, got_qg, nqg);
  snprintf(name, sizeof name, "qgate_merge rows=%d", rows);
  CHECK_REL(name, got_qg, want_qg, nqg, JT_TOL_FWD);

  // identity: merge(split(x)) == x on the device, compared against the untouched input rather than against a
  // reference, so a symmetric mistake in ref_qgate_split/merge cannot hide here.
  float *dqg2 = dev_alloc(nqg * sizeof *qg);
  jt_qgate_merge(dq, dgate, dqg2, rows, ATTN_H, ATTN_D, 0);
  dev_download(dqg2, got_qg, nqg);
  snprintf(name, sizeof name, "qgate_merge(split(x))==x rows=%d", rows);
  CHECK_REL(name, got_qg, qg, nqg, JT_TOL_FWD);

  dev_free(dqg);
  dev_free(dq);
  dev_free(dgate);
  dev_free(dq_in);
  dev_free(dgate_in);
  dev_free(dqg_out);
  dev_free(dqg2);
  free(qg);
  free(q);
  free(gate);
  free(want_a);
  free(want_b);
  free(got_a);
  free(got_b);
  free(want_qg);
  free(got_qg);
}

// ---- kv_expand / kv_reduce ------------------------------------------------------------------------------------------
static void t_kv(int rows, uint32_t seed) {
  if (rows <= 0) return;
  const size_t nkv = (size_t)rows * (size_t)KVH * (size_t)ATTN_D;             // [rows, 512]
  const size_t nexp = (size_t)rows * (size_t)KVH * (size_t)KVG * (size_t)ATTN_D; // [rows, 2048]
  float *a = jtm_alloc(nkv * sizeof *a);
  float *b = jtm_alloc(nexp * sizeof *b);
  float *want_e = jtm_alloc(nexp * sizeof *want_e);
  float *want_r = jtm_alloc(nkv * sizeof *want_r);
  float *got_e = jtm_alloc(nexp * sizeof *got_e);
  float *got_r = jtm_alloc(nkv * sizeof *got_r);
  char name[80];

  rnd_fill(a, nkv, -1.5, 1.5, seed);
  rnd_fill(b, nexp, -1.5, 1.5, seed + 1);

  ref_kv_expand(a, want_e, rows, KVH, KVG, ATTN_D);
  float *da = dev_upload(a, nkv);
  float *db = dev_upload(b, nexp);
  float *de = dev_alloc(nexp * sizeof *got_e);
  float *dr = dev_alloc(nkv * sizeof *got_r);
  jt_kv_expand(da, de, rows, KVH, KVG, ATTN_D, 0);
  dev_download(de, got_e, nexp);
  snprintf(name, sizeof name, "kv_expand rows=%d", rows);
  CHECK_REL(name, got_e, want_e, nexp, JT_TOL_FWD);

  ref_kv_reduce(b, want_r, rows, KVH, KVG, ATTN_D);
  jt_kv_reduce(db, dr, rows, KVH, KVG, ATTN_D, 0);
  dev_download(dr, got_r, nkv);
  snprintf(name, sizeof name, "kv_reduce rows=%d", rows);
  CHECK_REL(name, got_r, want_r, nkv, JT_TOL_FWD);

  // Adjoint: <kv_expand(a), b> == <a, kv_reduce(b)>, everything computed on the device, the inner products on the host
  // in double. Both sides are sums of the same products, so the only way to fail is a head that reads the wrong group
  // (or a reduce that drops one of the four).
  const double lhs = dot64(got_e, b, nexp);
  const double rhs = dot64(a, got_r, nkv);
  snprintf(name, sizeof name, "kv_expand/kv_reduce adjoint rows=%d", rows);
  CHECK_REL1(name, lhs, rhs, JT_TOL_FWD);

  dev_free(da);
  dev_free(db);
  dev_free(de);
  dev_free(dr);
  free(a);
  free(b);
  free(want_e);
  free(want_r);
  free(got_e);
  free(got_r);
}

// ---- to_head_major / from_head_major --------------------------------------------------------------------------------
static void t_head_major(int rows, uint32_t seed) {
  if (rows <= 0) return;
  const size_t n = (size_t)rows * (size_t)ATTN_H * (size_t)ATTN_D; // [rows, 2048] <-> [8][rows][256]
  float *x = jtm_alloc(n * sizeof *x);
  float *want_y = jtm_alloc(n * sizeof *want_y);
  float *want_x = jtm_alloc(n * sizeof *want_x);
  float *got = jtm_alloc(n * sizeof *got);
  char name[80];

  rnd_fill(x, n, -1.0, 1.0, seed);

  ref_to_head_major(x, want_y, rows, ATTN_H, ATTN_D);
  float *dx = dev_upload(x, n);
  float *dy = dev_alloc(n * sizeof *got);
  jt_to_head_major(dx, dy, rows, ATTN_H, ATTN_D, 0);
  dev_download(dy, got, n);
  snprintf(name, sizeof name, "to_head_major rows=%d", rows);
  CHECK_REL(name, got, want_y, n, JT_TOL_FWD);

  // from_head_major: the other direction, its own random input.
  float *y = jtm_alloc(n * sizeof *y);
  rnd_fill(y, n, -1.0, 1.0, seed + 1);
  ref_from_head_major(y, want_x, rows, ATTN_H, ATTN_D);
  float *dy_in = dev_upload(y, n);
  float *dx_out = dev_alloc(n * sizeof *got);
  jt_from_head_major(dy_in, dx_out, rows, ATTN_H, ATTN_D, 0);
  dev_download(dx_out, got, n);
  snprintf(name, sizeof name, "from_head_major rows=%d", rows);
  CHECK_REL(name, got, want_x, n, JT_TOL_FWD);

  // Round trip on the device: from_head_major(to_head_major(x)) == x, against the untouched input.
  jt_from_head_major(dy, dx_out, rows, ATTN_H, ATTN_D, 0);
  dev_download(dx_out, got, n);
  snprintf(name, sizeof name, "from(to(x))==x rows=%d", rows);
  CHECK_REL(name, got, x, n, JT_TOL_FWD);

  dev_free(dx);
  dev_free(dy);
  dev_free(dy_in);
  dev_free(dx_out);
  free(x);
  free(y);
  free(want_y);
  free(want_x);
  free(got);
}

int k_misc(void) {
  int off[JT_RAGGED4_ROWS + 1];
  const int n3 = jt_ragged(jt_lens_ragged3(), JT_RAGGED3_ROWS, off, NULL);   // 168
  const int n4 = jt_ragged(jt_lens_ragged4(), JT_RAGGED4_ROWS, off, NULL);   // 2516, crosses 1024

  t_elementwise(n3, 101u);
  t_elementwise(n4, 102u);

  t_embed(n3, 32, JT_HIDDEN, 103u);   // the model's width ...
  t_embed(n4, 64, 100, 104u);         // ... and a width that is not a multiple of the block's 256 threads

  t_qgate(1, 105u);
  t_qgate(n3, 106u);
  t_qgate(n4, 107u);

  t_kv(1, 108u);
  t_kv(n3, 109u);
  t_kv(n4, 110u);

  t_head_major(1, 111u);
  t_head_major(n3, 112u);
  t_head_major(n4, 113u);

  return jt_exit_code();
}
