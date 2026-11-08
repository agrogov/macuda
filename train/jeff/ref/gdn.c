// ref/gdn.c — the CPU oracle of the GDN recurrence (plan M3.1, design §3.6 forward and §4.8 backward).
//
// Both directions are written literally from the design's pseudo-code, in the layouts ref/gdn.h documents, with the
// D§4.8 lines in the order the design states them: dk_t's first term and ddelta read dS before the `k ⊗ dkv` term is
// added to it, and dg_t reads dS after that term (design §4.8, "The order matters"). Inner sums are accumulated in
// double even in the float build, then rounded once: over up to 128 terms in fp32 an order-dependent error of its own
// would compete with the kernel's 1e-4 gate, and a reference is the oracle (the rule ref/real.h's r_* wrappers follow).
//
// The backward needs S_prev = S_{t-1} at every token. Instead of keeping all of them it keeps design §8.3's
// checkpoints -- the state at every 64th token -- and replays a segment forward to rebuild the S_prev of the tokens
// inside it, exactly the way the kernel does, so the reference stays at (N/64) * H * Dk * Dv reals rather than
// N * H * Dk * Dv (84 MB instead of 5.3 GB for the plan's biggest batch at the model's dims). Sd, S_t, kv, delta and
// the rest of the forward step are then recomputed from S_prev in the reverse pass. The forward step lives in one
// function, gdn_step(), used by both directions, so a stored S_prev can never disagree with what the forward computed.
//
// The state is per (row, head), so S is one matrix per head and dS is zeroed for every (row, head) pair, not just once
// per row: the heads of one token share nothing.
#include "gdn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Allocations are small (Dk*Dv reals per head) and a NULL means the machine is out of memory, not a test outcome.
static void *xalloc(size_t n, int zero) {
  void *p = zero ? calloc(n ? n : 1, sizeof(real)) : malloc((n ? n : 1) * sizeof(real));
  if (p == NULL) {
    fprintf(stderr, "ref_gdn: out of memory (%zu reals)\n", n);
    exit(1);
  }
  return p;
}

// ---- forward (design §3.6) ------------------------------------------------------------------------------------
// One token of the recurrence, with S the state *before* the step, updated in place. delta is a caller buffer of Dv;
// o_t may be NULL when the caller only needs the state (gdn_row_run's checkpoint and replay passes).
//
//   S        = S * exp(g[t])
//   kv[j]    = sum_i S[i,j] * k[t,i]
//   delta[j] = (v[t,j] - kv[j]) * beta[t]
//   S[i,j]  += k[t,i] * delta[j]
//   o[t,j]   = sum_i S[i,j] * q[t,i]
static void gdn_step(real *S, const real *k_t, const real *q_t, const real *v_t, real g_t, real beta_t, int Dk, int Dv,
                     real *delta, real *o_t) {
  const double e = exp((double)g_t); // decays are <= 1 (plan M3.1: g in [-1, 0])
  for (int i = 0; i < Dk; i++)
    for (int j = 0; j < Dv; j++) S[(size_t)i * Dv + j] = REAL((double)S[(size_t)i * Dv + j] * e);
  for (int j = 0; j < Dv; j++) {
    double kv = 0.0;
    for (int i = 0; i < Dk; i++) kv += (double)S[(size_t)i * Dv + j] * (double)k_t[i];
    delta[j] = REAL(((double)v_t[j] - kv) * (double)beta_t);
  }
  for (int i = 0; i < Dk; i++)
    for (int j = 0; j < Dv; j++) S[(size_t)i * Dv + j] += k_t[i] * delta[j];
  if (o_t != NULL) // o uses the updated S, i.e. S_t
    for (int j = 0; j < Dv; j++) {
      double o = 0.0;
      for (int i = 0; i < Dk; i++) o += (double)S[(size_t)i * Dv + j] * (double)q_t[i];
      o_t[j] = REAL(o);
    }
}

// Valid dims and pointers, or the whole call is a no-op (the same tolerance ref/attn.c shows its callers).
static int gdn_bad_args(int B, const int *off, int H, int Dk, int Dv, const real *q, const real *k, const real *v,
                        const real *g, const real *beta) {
  return B <= 0 || H <= 0 || Dk <= 0 || Dv <= 0 || off == NULL || q == NULL || k == NULL || v == NULL || g == NULL ||
         beta == NULL;
}

void ref_gdn_fwd(int B, const int *off, int H, int Dk, int Dv, const real *q, const real *k, const real *v,
                 const real *g, const real *beta, real *o) {
  if (gdn_bad_args(B, off, H, Dk, Dv, q, k, v, g, beta) || o == NULL) return;
  const size_t ssz = (size_t)Dk * (size_t)Dv;
  // One state per head (design §3.6: "per head, S is [128 (key i), 128 (value j)]").
  real *S = (real *)xalloc(ssz * (size_t)H, 1);
  real *delta = (real *)xalloc((size_t)Dv, 0);
  for (int b = 0; b < B; b++) {
    // S_0 = 0 at every row start, for every head: rows of a packed batch share nothing (design §3.6 and §7).
    memset(S, 0, ssz * (size_t)H * sizeof *S);
    for (int t = off[b]; t < off[b + 1]; t++)
      for (int h = 0; h < H; h++) {
        const size_t gi = (size_t)t * (size_t)H + (size_t)h;
        gdn_step(S + (size_t)h * ssz, k + gi * (size_t)Dk, q + gi * (size_t)Dk, v + gi * (size_t)Dv, g[gi], beta[gi],
                 Dk, Dv, delta, o + gi * (size_t)Dv);
      }
  }
  free(delta);
  free(S);
}

// Tokens per checkpoint segment: design §8.3's 64, so a segment is replayed the way the kernel replays one.
#define JT_GDN_CKPT 64

// The forward over one (row, head) from token t0 to t1 (exclusive), starting from *S, which is advanced in place.
//   ck != NULL: store the state at every segment start, before that token's step (segment 0's checkpoint is all zeros),
//               at ck[((tl/64)*H + h) * Dk*Dv] with tl = t - t0 the row-local index.
//   sp != NULL: store every token's S_prev densely at sp[(t - t0) * Dk*Dv] instead, which is how a segment is replayed.
// Both are the same forward: the caller chooses what to keep.
static void gdn_row_run(real *S, const real *q, const real *k, const real *v, const real *g, const real *beta, int H,
                        int h, int t0, int t1, int Dk, int Dv, real *delta, real *ck, real *sp) {
  const size_t ssz = (size_t)Dk * (size_t)Dv;
  for (int t = t0; t < t1; t++) {
    const size_t gi = (size_t)t * (size_t)H + (size_t)h;
    const int tl = t - t0;
    if (ck != NULL && tl % JT_GDN_CKPT == 0)
      memcpy(ck + (((size_t)(tl / JT_GDN_CKPT) * (size_t)H + (size_t)h) * ssz), S, ssz * sizeof *S);
    if (sp != NULL) memcpy(sp + (size_t)tl * ssz, S, ssz * sizeof *S);
    gdn_step(S, k + gi * (size_t)Dk, q + gi * (size_t)Dk, v + gi * (size_t)Dv, g[gi], beta[gi], Dk, Dv, delta, NULL);
  }
}

// ---- backward (design §4.8) -----------------------------------------------------------------------------------
// The per-token buffers of the reverse pass: one Dk*Dv array each for the forward step's recomputed values and the
// carried dS, and Dv-vectors for the row vectors the design names.
typedef struct {
  int Dk, Dv;
  size_t ssz;
  real *Sd, *St, *dS, *kv, *delta, *ddelta, *dkv;
} gdn_bwd_work;

// One token of the reverse pass, design §4.8's block written line for line, in its order. sp is S_prev for this token,
// dS is carried in and out (dS_{t} in, dS_{t-1} out), and the five per-token gradients are written to dq_t/dk_t/dv_t/
// dg_t/dbeta_t.
static void gdn_bwd_step(gdn_bwd_work *w, const real *sp, const real *q_t, const real *k_t, const real *v_t,
                         const real *do_t, real g_t, real beta_t, real *dq_t, real *dk_t, real *dv_t, real *dg_t,
                         real *dbeta_t) {
  const int Dk = w->Dk, Dv = w->Dv;
  const size_t ssz = w->ssz;
  real *Sd = w->Sd, *St = w->St, *dS = w->dS, *kv = w->kv, *delta = w->delta, *ddelta = w->ddelta, *dkv = w->dkv;
  const double bt = (double)beta_t;
  const double e = exp((double)g_t); // "recomputing the forward step's values"

  // Sd = S_prev * exp(g_t)
  for (size_t x = 0; x < ssz; x++) Sd[x] = REAL((double)sp[x] * e);
  // kv[j] = sum_i Sd[i,j] k_t[i];  delta[j] = (v_t[j] - kv[j]) * beta_t
  for (int j = 0; j < Dv; j++) {
    double s = 0.0;
    for (int i = 0; i < Dk; i++) s += (double)Sd[(size_t)i * Dv + j] * (double)k_t[i];
    kv[j] = REAL(s);
    delta[j] = REAL(((double)v_t[j] - (double)kv[j]) * bt); // literal: delta uses the kv it just defined
  }
  // S_t = Sd + k_t ⊗ delta
  for (int i = 0; i < Dk; i++)
    for (int j = 0; j < Dv; j++)
      St[(size_t)i * Dv + j] = REAL((double)Sd[(size_t)i * Dv + j] + (double)k_t[i] * (double)delta[j]);
  // -- o_t = S_t^T q_t:   dq_t[i] = sum_j S_t[i,j] do_t[j]
  for (int i = 0; i < Dk; i++) {
    double s = 0.0;
    for (int j = 0; j < Dv; j++) s += (double)St[(size_t)i * Dv + j] * (double)do_t[j];
    dq_t[i] = REAL(s);
  }
  // dS[i,j] += q_t[i] do_t[j]
  for (int i = 0; i < Dk; i++)
    for (int j = 0; j < Dv; j++)
      dS[(size_t)i * Dv + j] = REAL((double)dS[(size_t)i * Dv + j] + (double)q_t[i] * (double)do_t[j]);
  // -- S_t = Sd + k_t ⊗ delta:   dk_t[i] = sum_j dS[i,j] delta[j];  ddelta[j] = sum_i dS[i,j] k_t[i]
  // dk_t's first term is written now and completed after the k ⊗ dkv term below: design §4.8 wants it read from dS
  // *before* that term is added.
  for (int i = 0; i < Dk; i++) {
    double s = 0.0;
    for (int j = 0; j < Dv; j++) s += (double)dS[(size_t)i * Dv + j] * (double)delta[j];
    dk_t[i] = REAL(s);
  }
  for (int j = 0; j < Dv; j++) {
    double s = 0.0;
    for (int i = 0; i < Dk; i++) s += (double)dS[(size_t)i * Dv + j] * (double)k_t[i];
    ddelta[j] = REAL(s);
  }
  // -- delta = (v - kv) * beta:   dv_t[j] = ddelta[j] * beta_t;  dkv[j] = -ddelta[j] * beta_t;
  //                               dbeta_t = sum_j ddelta[j] (v_t[j] - kv[j])
  double dbt = 0.0;
  for (int j = 0; j < Dv; j++) {
    dv_t[j] = REAL((double)ddelta[j] * bt);
    dkv[j] = REAL(-(double)ddelta[j] * bt);
    dbt += (double)ddelta[j] * ((double)v_t[j] - (double)kv[j]);
  }
  *dbeta_t = REAL(dbt);
  // -- kv = Sd^T k_t:   dS[i,j] += k_t[i] dkv[j];  dk_t[i] += sum_j Sd[i,j] dkv[j]
  for (int i = 0; i < Dk; i++)
    for (int j = 0; j < Dv; j++)
      dS[(size_t)i * Dv + j] = REAL((double)dS[(size_t)i * Dv + j] + (double)k_t[i] * (double)dkv[j]);
  for (int i = 0; i < Dk; i++) {
    double s = (double)dk_t[i];
    for (int j = 0; j < Dv; j++) s += (double)Sd[(size_t)i * Dv + j] * (double)dkv[j];
    dk_t[i] = REAL(s);
  }
  // -- Sd = S_prev * exp(g_t):   dg_t = sum_{i,j} dS[i,j] * Sd[i,j]      (dS *after* the k ⊗ dkv term)
  double dgt = 0.0;
  for (size_t x = 0; x < ssz; x++) dgt += (double)dS[x] * (double)Sd[x];
  *dg_t = REAL(dgt);
  // dS = dS * exp(g_t)                            # now dS_{t-1}
  for (size_t x = 0; x < ssz; x++) dS[x] = REAL((double)dS[x] * e);
}

void ref_gdn_bwd(int B, const int *off, int H, int Dk, int Dv, const real *q, const real *k, const real *v,
                 const real *g, const real *beta, const real *dout, real *dq, real *dk, real *dv, real *dg,
                 real *dbeta) {
  if (gdn_bad_args(B, off, H, Dk, Dv, q, k, v, g, beta) || dout == NULL) return;
  if (dq == NULL || dk == NULL || dv == NULL || dg == NULL || dbeta == NULL) return;

  const size_t ssz = (size_t)Dk * (size_t)Dv;
  gdn_bwd_work w;
  w.Dk = Dk;
  w.Dv = Dv;
  w.ssz = ssz;
  w.Sd = (real *)xalloc(ssz, 0);
  w.St = (real *)xalloc(ssz, 0);
  w.dS = (real *)xalloc(ssz, 0);
  w.kv = (real *)xalloc((size_t)Dv, 0);
  w.delta = (real *)xalloc((size_t)Dv, 0);
  w.ddelta = (real *)xalloc((size_t)Dv, 0);
  w.dkv = (real *)xalloc((size_t)Dv, 0);

  // One set of checkpoints per row: nseg[b] states per head (design §8.3's [B][16][max_segs][128][128] layout, one row
  // of it at a time here).
  int *nseg = (int *)calloc((size_t)B, sizeof *nseg);
  size_t *ckoff = (size_t *)calloc((size_t)B + 1, sizeof *ckoff);
  if (nseg == NULL || ckoff == NULL) {
    fprintf(stderr, "ref_gdn_bwd: out of memory\n");
    exit(1);
  }
  for (int b = 0; b < B; b++) {
    const int len = off[b + 1] - off[b];
    nseg[b] = (len + JT_GDN_CKPT - 1) / JT_GDN_CKPT;
    ckoff[b + 1] = ckoff[b] + (size_t)nseg[b] * (size_t)H * ssz;
  }
  real *ck = (real *)xalloc(ckoff[B], 1);
  real *S = (real *)xalloc(ssz, 0);                                     // the replay's running state
  real *sp = (real *)xalloc((size_t)JT_GDN_CKPT * ssz, 0);               // one segment's S_prev
  real *delta = (real *)xalloc((size_t)Dv, 0);                          // gdn_row_run's scratch

  // The checkpoint pass: the same forward, keeping the state at every segment start.
  for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++) {
      memset(S, 0, ssz * sizeof *S); // S_0 = 0 at every row start, per head (design §3.6)
      gdn_row_run(S, q, k, v, g, beta, H, h, off[b], off[b + 1], Dk, Dv, delta, ck + ckoff[b], NULL);
    }

  // The reverse pass: segments last to first, and the tokens inside a segment last to first. dS is carried from the
  // row's last token back to its first, so it is zeroed once per (row, head) and not per segment.
  for (int b = 0; b < B; b++)
    for (int h = 0; h < H; h++) {
      memset(w.dS, 0, ssz * sizeof *w.dS);
      for (int seg = nseg[b] - 1; seg >= 0; seg--) {
        const int t0 = off[b] + seg * JT_GDN_CKPT;
        const int t1 = off[b + 1] < t0 + JT_GDN_CKPT ? off[b + 1] : t0 + JT_GDN_CKPT;
        memcpy(S, ck + ckoff[b] + (((size_t)seg * (size_t)H + (size_t)h) * ssz), ssz * sizeof *S);
        gdn_row_run(S, q, k, v, g, beta, H, h, t0, t1, Dk, Dv, delta, NULL, sp);
        for (int t = t1 - 1; t >= t0; t--) {
          const size_t gi = (size_t)t * (size_t)H + (size_t)h;
          gdn_bwd_step(&w, sp + (size_t)(t - t0) * ssz, q + gi * (size_t)Dk, k + gi * (size_t)Dk,
                       v + gi * (size_t)Dv, dout + gi * (size_t)Dv, g[gi], beta[gi], dq + gi * (size_t)Dk,
                       dk + gi * (size_t)Dk, dv + gi * (size_t)Dv, dg + gi, dbeta + gi);
        }
      }
    }

  free(delta), free(sp), free(S), free(ck), free(ckoff), free(nseg);
  free(w.dkv), free(w.ddelta), free(w.delta), free(w.kv), free(w.dS), free(w.St), free(w.Sd);
}
