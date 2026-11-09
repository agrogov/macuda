// src/head.c — the decision head: the final norm, the last token of each row, the readout GEMM and the soft CE (design
// §4.1, §13; plan M4.3 item 5).
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
// The backward (design §4.1's second half) is NOT written yet — head_bwd is still plan J18's stub, and plan M5.1 item 1
// is its task. The save layout head_bwd will read is fixed here so that task does not have to change head_fwd: in
// JT_FWD_SAVE mode the saves are taken from the arena's save half in this order, and nothing else is saved —
//
//   r        [n]                     the final norm's 1/sqrt(mean(x^2)+eps), one per token
//   z        [B][MT_HIDDEN]          the gathered last-token states (dR += dlogits^T z)
//   dlogits  [B][MT_READOUT_OUT]     ce_head's per-row logit gradient, ZERO-PADDED to the readout's 255 columns
//
// The padding is what makes dlogits' width independent of the batch (head_bwd's frozen signature carries no HeadBatch, so
// it cannot be told n_opts): the dead columns' dlogits is exactly 0 — their p and their target are both 0 — so the
// padded array is the same gradient as the [B][n_opts] one, and `dz = R^T dlogits` over all 255 readout rows adds
// nothing from them. Everything else head_bwd needs it has: x (the head's input) and r.
#include "gemm.h"
#include "head.h" // kernels/head.h: ce_head
#include "layer.h"
#include "misc.h" // kernels/misc.h: jt_fill, jt_zero
#include "norm.h" // kernels/norm.h: rmsnorm_fwd
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

// Still plan J18's stub: the head's backward is plan M5.1 item 1. It reads the saves head_fwd's header comment lists
// (r [n], z [B][MT_HIDDEN], dlogits [B][MT_READOUT_OUT], in that order) and implements design §4.1's last three lines:
// dR += dlogits^T z, dz = R^T dlogits, then dhN's last-token rows = dz and rmsnorm_bwd over all n tokens into dres.
int head_bwd(JtCtx *c, const float *x, int n, const JtBatch *b, float *dres) {
  (void)c, (void)x, (void)n, (void)b, (void)dres;
  fprintf(stderr, "head: head_bwd: not implemented (plan M5.1)\n");
  return JT_LAYER_NOT_IMPL;
}
