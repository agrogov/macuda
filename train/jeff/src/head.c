// src/head.c — the decision head: the final norm, the last token of each row, the readout GEMM, the soft CE and their
// backward (design §4.1, §13; plan M4.3 item 5 and M5.1 item 1).
//
// The forward is design §4.1 in order, with the call each step uses:
//
//   final = RMSNorm(x; norm)                                  rmsnorm_fwd, d = 1024, zero-centred, r saved
//   z     = final[off[b+1]-1] for each row b                  a gather of B rows (one cudaMemcpyAsync per row, on
//                                                             ctx->stream): design §7's "the head reads token
//                                                             off[b+1]-1 for row b"
//   logits= z R^T, first n_opts readout rows                  gemm(0,1) B x n_opts x 1024 against
//                                                             ctx->lora->param + ctx->lora->readout_off
//   logits[b][j >= live[b]] = -1e9                            train.py's masked_fill, exactly (one jt_fill per row)
//   ce_head(logits, n_opts, targets, wrow, wgroup, B, ...)    k_head.h: per-row loss, dlogits and probs
//
// head_fwd reads the loss back to the host once (*loss): the value is train.py's `(per * wt).sum() / gw`, i.e. the sum of
// ce_head's per-row losses, and a training step wants it on the host for its log/schedule. That makes head_fwd the one
// layer call that synchronises; pass loss == NULL to keep the step asynchronous (the logits stay device-side).
//
// The backward is design §4.1's second half (plan M5.1 item 1), in order, with the call each step uses:
//
//   dR    += dlogits^T z                                      gemm(1,0) 255 x 1024 x B, beta = 1, into
//                                                             ctx->lora->grad + readout_off (accumulated: a parameter
//                                                             gradient belongs to the step, src/linear.h's rule)
//   dz     = dlogits R                                        gemm(0,0) B x 1024 x 255, the transpose problem of the
//                                                             forward's readout GEMM, over all 255 readout rows
//   dhN[off[b+1]-1] = dz[b]                                   a zeroed dh [n][MT_HIDDEN] plus B cudaMemcpyAsync (D2D)
//                                                             per row: the forward gathered the same rows
//   dres   = RMSNormBackward(dhN)                             rmsnorm_bwd over all n tokens, h = x, r from the saves
//
// In JT_FWD_SAVE mode the forward's saves are taken from the arena's save half in this order, and nothing else is saved,
// so a backward only has to index them from the arena's base:
//
//   r        [n]                     the final norm's 1/sqrt(mean(x^2)+eps), one per token
//   z        [B][MT_HIDDEN]          the gathered last-token states (dR += dlogits^T z)
//   dlogits  [B][MT_READOUT_OUT]     ce_head's per-row logit gradient, ZERO-PADDED to the readout's 255 columns
//
// The padding is what makes dlogits' width independent of the batch (head_bwd's frozen signature carries no HeadBatch, so
// it cannot be told n_opts): the dead columns' dlogits is exactly 0 — their p and their target are both 0 — so the
// padded array is the same gradient as the [B][n_opts] one, and `dz = dlogits R` over all 255 readout rows adds
// nothing from them. A backward therefore needs the JT_FWD_SAVE of the same x WITH targets (the "logits only" pass saves
// no dlogits and has no gradient to propagate).
//
// dhN is dump_ref.py's `grad.final` — the gradient at the final norm's output, which design §10.2 lists and plan M5.1
// item 1 tests. The frozen signature has no room for it and does not need one: dhN is the first (and only) temporary
// head_bwd takes from the arena's work half, so for a mark read before the call it sits at
// `ctx->arena + mark - n * MT_HIDDEN`, and jt_work_restore moves the cursor back, not the data. A test — or M5.2's
// --dump-backward — reads those words after head_bwd returns; nothing else in the work half is that size at that
// address, because every later allocation in a bwd starts below it.
#include "gemm.h"
#include "head.h" // kernels/head.h: ce_head
#include "layer.h"
#include "misc.h" // kernels/misc.h: jt_fill, jt_zero
#include "norm.h" // kernels/norm.h: rmsnorm_fwd, rmsnorm_bwd
#include "util.h"

#include <stdio.h>
#include <stdlib.h>

// train.py's masked_fill value for an option a row does not have (train.py's `option_logits`, the file's spec):
// `logits.masked_fill(~live, -1e9)`. Dead entries then have p = 0 and dlogits = 0 (src/layer.h, kernels/head.h).
#define HEAD_DEAD (-1e9f)

static int head_bad(const char *why) {
  fprintf(stderr, "head: %s\n", why);
  return JT_LAYER_FAIL;
}

// z[r] = final's last token of row r: the readout reads position L-1 of a right-padded row (design §7). B copies of
// MT_HIDDEN floats, ordered on the context's stream, so the head needs no gather kernel and no extra buffer.
static void gather_last(const float *final_state, float *z, const JtBatch *b, cudaStream_t s) {
  const size_t row_bytes = (size_t)MT_HIDDEN * sizeof(float);
  for (int r = 0; r < b->rows; r++) {
    const size_t last = (size_t)(b->off[r + 1] - 1);
    CK(cudaMemcpyAsync(z + (size_t)r * MT_HIDDEN, final_state + last * MT_HIDDEN, row_bytes, cudaMemcpyDeviceToDevice,
                       s));
  }
}

// logits[r][live[r] .. n_opts) = -1e9, one launch per row (jt_fill is a no-op for an empty range).
static void mask_dead(float *logits, const JtBatch *b, const HeadBatch *hb, cudaStream_t s) {
  for (int r = 0; r < b->rows; r++) {
    const size_t dead = (size_t)(hb->n_opts - hb->live[r]);
    if (dead > 0) jt_fill(logits + (size_t)r * (size_t)hb->n_opts + (size_t)hb->live[r], HEAD_DEAD, dead, s);
  }
}

// ce_head over the (masked) logits, plus the padded copy of dlogits the backward will read. `dlogits_pad` is NULL in
// JT_FWD_ONLY mode (no bwd follows, so ce_head is asked for the losses only).
static void head_ce(JtCtx *c, const JtBatch *b, const HeadBatch *hb, const float *logits, float *dlogits_pad,
                    float *loss) {
  const int B = b->rows, nop = hb->n_opts;
  float *dlogits = jt_work_alloc(c, (size_t)B * (size_t)nop);
  float *row_loss = jt_work_alloc(c, (size_t)B);
  ce_head(logits, nop, hb->targets, hb->wrow, hb->wgroup, B, row_loss, dlogits, NULL, c->stream);
  if (dlogits_pad != NULL) { // jt_zero first, then the live prefix of every row: the padding stays exactly 0
    jt_zero(dlogits_pad, (size_t)B * (size_t)MT_READOUT_OUT, c->stream);
    for (int r = 0; r < B; r++) {
      CK(cudaMemcpyAsync(dlogits_pad + (size_t)r * (size_t)MT_READOUT_OUT, dlogits + (size_t)r * (size_t)nop,
                         (size_t)nop * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));
    }
  }
  JT_SYNC_DEBUG;
  if (loss == NULL) return;
  // train.py's `loss = (per * wt).sum() / gw` is ce_head's per-row loss summed: the kernel already folded wrow/wgroup in.
  // Summed in double, so the only rounding is the kernel's own.
  float *row = (float *)malloc((size_t)B * sizeof(float));
  if (row == NULL) {
    fprintf(stderr, "head: out of host memory (%d row losses)\n", B);
    exit(1);
  }
  CK(cudaMemcpyAsync(row, row_loss, (size_t)B * sizeof(float), cudaMemcpyDeviceToHost, c->stream));
  CK(cudaStreamSynchronize(c->stream));
  double sum = 0.0;
  for (int r = 0; r < B; r++) sum += (double)row[r];
  *loss = (float)sum;
  free(row);
}

int head_fwd(JtCtx *c, const float *x, int n, const JtBatch *b, const HeadBatch *hb, float *final, float *logits,
             float *loss, int mode) {
  if (c == NULL || c->model == NULL || c->lora == NULL) return head_bad("no context, model or parameter buffer");
  if (x == NULL || b == NULL || hb == NULL || logits == NULL) return head_bad("null argument");
  if (b->rows < 0 || b->rows > JT_MAX_ROWS) return head_bad("b->rows outside 0..JT_MAX_ROWS");
  if (n < 0 || b->off[b->rows] != n) return head_bad("b->off[rows] != n");
  if (hb->n_opts <= 0 || hb->n_opts > MT_READOUT_OUT) return head_bad("hb->n_opts outside 1..MT_READOUT_OUT");
  if (hb->live == NULL) return head_bad("hb->live is required");
  if (mode != JT_FWD_ONLY && mode != JT_FWD_SAVE) return head_bad("mode must be JT_FWD_ONLY or JT_FWD_SAVE");
  for (int r = 0; r < b->rows; r++) {
    if (hb->live[r] < 0 || hb->live[r] > hb->n_opts) return head_bad("hb->live[r] outside 0..n_opts");
  }
  // targets and wrow arrive together (both are train.py's collate output) or not at all ("logits only", src/layer.h).
  if ((hb->targets != NULL) != (hb->wrow != NULL)) return head_bad("hb->targets and hb->wrow are given together");
  if (loss != NULL && hb->targets == NULL) return head_bad("a loss needs hb->targets");
  if (hb->targets != NULL && !(hb->wgroup > 0.0f)) return head_bad("hb->wgroup must be positive with targets");
  const size_t readout_need = (size_t)MT_READOUT_OUT * (size_t)MT_HIDDEN;
  if (c->lora->readout_off + readout_need > c->lora->n_param) return head_bad("the readout is outside the buffer");
  if (b->rows == 0 || n == 0) return JT_LAYER_OK; // ce_head and the norms are no-ops on an empty batch

  const int B = b->rows;
  const int save = (mode == JT_FWD_SAVE);
  const size_t mark = jt_work_mark(c);

  // 1. final = RMSNorm(x; norm). The caller's buffer is the output when it wants it (dump_ref's `final` key); otherwise
  //    the norm's own output is a work temporary. r is always needed by rmsnorm_fwd, saved when a bwd will follow.
  float *h = (final != NULL) ? final : jt_work_alloc(c, (size_t)n * (size_t)MT_HIDDEN);
  float *r = save ? jt_save_alloc(c, (size_t)n) : jt_work_alloc(c, (size_t)n);
  rmsnorm_fwd(x, c->model->final_norm, h, r, n, MT_HIDDEN, 1, c->stream);
  JT_SYNC_DEBUG;

  // 2. the gather, and 3. the readout GEMM: logits = z R^T over the first n_opts rows of the [255,1024] readout.
  float *z = save ? jt_save_alloc(c, (size_t)B * MT_HIDDEN) : jt_work_alloc(c, (size_t)B * MT_HIDDEN);
  gather_last(h, z, b, c->stream);
  const float *R = c->lora->param + c->lora->readout_off; // [MT_READOUT_OUT][MT_HIDDEN], torch layout [out, in]
  gemm(c->cublas, 0, 1, B, hb->n_opts, MT_HIDDEN, 1.0f, z, R, 0.0f, logits);

  // 4. train.py's masked_fill, then 5. the weighted soft CE.
  mask_dead(logits, b, hb, c->stream);
  JT_SYNC_DEBUG;
  if (hb->targets != NULL) {
    float *dlogits_pad = save ? jt_save_alloc(c, (size_t)B * MT_READOUT_OUT) : NULL;
    head_ce(c, b, hb, logits, dlogits_pad, loss);
  }
  jt_work_restore(c, mark); // the temporaries are gone; the launches above are ordered before anything reuses the words
  return JT_LAYER_OK;
}

int head_bwd(JtCtx *c, const float *x, int n, const JtBatch *b, float *dres) {
  if (c == NULL || c->model == NULL || c->lora == NULL) return head_bad("no context, model or parameter buffer");
  if (x == NULL || b == NULL || dres == NULL) return head_bad("null argument");
  if (b->rows < 0 || b->rows > JT_MAX_ROWS) return head_bad("b->rows outside 0..JT_MAX_ROWS");
  if (n < 0 || b->off[b->rows] != n) return head_bad("b->off[rows] != n");
  for (int r = 0; r < b->rows; r++) { // the head reads off[r+1]-1, so a row needs at least one token (design §7)
    if (b->off[r + 1] <= b->off[r]) return head_bad("a row is empty (the head's last token is off[r+1]-1)");
  }
  if (c->lora->grad == NULL) return head_bad("the parameter buffer has no gradient array");
  const size_t readout_need = (size_t)MT_READOUT_OUT * (size_t)MT_HIDDEN;
  if (c->lora->readout_off + readout_need > c->lora->n_param) return head_bad("the readout is outside the buffer");
  if (b->rows == 0 || n == 0) return JT_LAYER_OK; // nothing to propagate, and rmsnorm_bwd would launch nothing

  const int B = b->rows;
  const size_t saves_need = (size_t)n + (size_t)B * ((size_t)MT_HIDDEN + (size_t)MT_READOUT_OUT);
  if (saves_need > c->arena_n) return head_bad("the head's saves are bigger than the arena (was head_fwd run SAVE?)");
  // head_fwd(SAVE)'s saves, in its order from the arena's base (the header comment).
  const float *r = c->arena;
  const float *z = r + (size_t)n;
  const float *dlogits = z + (size_t)B * (size_t)MT_HIDDEN;
  const float *R = c->lora->param + c->lora->readout_off; // [MT_READOUT_OUT][MT_HIDDEN], torch layout [out, in]

  const size_t mark = jt_work_mark(c);
  float *dh = jt_work_alloc(c, (size_t)n * (size_t)MT_HIDDEN); // dL/d(final) == grad.final; taken FIRST (see above)
  float *dz = jt_work_alloc(c, (size_t)B * (size_t)MT_HIDDEN);

  // 1. dR += dlogits^T z: the readout's gradient, accumulated into the flat buffer's readout slot (design §4.1).
  gemm(c->cublas, 1, 0, MT_READOUT_OUT, MT_HIDDEN, B, 1.0f, dlogits, z, 1.0f, c->lora->grad + c->lora->readout_off);

  // 2. dz = dlogits R over all 255 readout rows (the transpose problem of the forward's gemm(0,1)).
  gemm(c->cublas, 0, 0, B, MT_HIDDEN, MT_READOUT_OUT, 1.0f, dlogits, R, 0.0f, dz);

  // 3. dhN: zero over every token, then dz row b at that row's last token — the rows head_fwd gathered from.
  jt_zero(dh, (size_t)n * (size_t)MT_HIDDEN, c->stream);
  const size_t row_bytes = (size_t)MT_HIDDEN * sizeof(float);
  for (int r2 = 0; r2 < B; r2++) {
    const size_t last = (size_t)(b->off[r2 + 1] - 1);
    CK(cudaMemcpyAsync(dh + last * MT_HIDDEN, dz + (size_t)r2 * MT_HIDDEN, row_bytes, cudaMemcpyDeviceToDevice,
                       c->stream));
  }

  // 4. the final norm's backward over all n tokens, in full: this is the start of the backward sweep (src/layer.h).
  rmsnorm_bwd(x, c->model->final_norm, dh, r, dres, n, MT_HIDDEN, 1, c->stream);
  JT_SYNC_DEBUG;
  jt_work_restore(c, mark); // dh and dz are gone as allocations; their words stay readable (the grad.final note above)
  return JT_LAYER_OK;
}
