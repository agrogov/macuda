// src/layer.h — the decoder layer's frozen interfaces: the context every call takes, the activation arena, the forward
// modes, and the fwd/bwd signatures of the four pieces that are written in parallel (design §3.4, §4, §6.2, §13).
//
// The model is 24 decoder layers, each
//
//     a     = RMSNorm(x; input_layernorm)          zero-centred (design §3.3)
//     x     = x + Mixer(a)                         Mixer = GDN (18 layers) or full attention (i % 4 == 3, 6 layers)
//     m     = RMSNorm(x; post_attention_layernorm)
//     x     = x + down(silu(gate(m)) * up(m))      the MLP
//
// so a layer splits at the post-mixer residual, which is also the boundary dump_ref.py writes (design §10.2) and the one
// test_layer uses:
//
//     layer_gdn_mixer_fwd  / layer_attn_mixer_fwd   layer.{i}.in  ->  layer.{i}.post_mixer
//     layer_mlp_fwd                                layer.{i}.post_mixer -> layer.{i}.out
//     layer_fwd                                    layer.{i}.in  ->  layer.{i}.out   (the two, in order)
//
// Files and owners (design §13). Each of the ten functions below is a stub in plan J18 and is replaced by its own task;
// *these declarations and struct JtCtx do not change* when that happens — a later change would stop every parallel task,
// which is why they are frozen here, before the code exists:
//
//     src/linear.c      linear_fwd, linear_bwd                       (design §4.2, declared in src/linear.h)
//     src/layer_mlp.c   layer_mlp_fwd, layer_mlp_bwd                 (design §3.4 last two lines, §4.5)
//     src/layer_gdn.c   layer_gdn_mixer_fwd, layer_gdn_mixer_bwd     (design §3.6, §4.8)
//     src/layer_attn.c  layer_attn_mixer_fwd, layer_attn_mixer_bwd   (design §3.5, §4.7)
//     src/head.c        head_fwd, head_bwd                           (design §4.1)
//     src/model.c       the load, the arena, layer_fwd/layer_bwd, model_forward
//
// ---- the contract every fwd/bwd obeys ---------------------------------------------------------------------------------
//
// * The layer's input and output are fp32, token-major and packed: [n][MT_HIDDEN], n = the batch's off[B] (design §7); a
//   piece's own buffers are however wide the design makes them (the GDN projections are [n][6144], the attention qg is
//   [n][4096], and so on). Nothing per token ever crosses a row boundary unless design §7 says it does (the conv and the
//   recurrence restart per row; RoPE positions are row-local; attention is per row).
// * `x_out` is a plain output: the fwd writes `x_out = x_in + <its path>` and reads x_in, never x_out. x_out MAY alias
//   x_in (layer_fwd uses that to keep a layer in one buffer's worth of copies), in which case the function must have
//   finished reading x_in before it accumulates into x_out — the input norm is always computed first, so a
//   straightforward implementation already has that property. Nothing else aliases.
// * `dres` in a bwd is the running residual gradient (design §4.6): on entry it holds dL/d(x_out), on return dL/d(x_in),
//   the same buffer, with this path's contribution added — "accumulate into the residual gradient buffer; never
//   overwrite it". head_bwd is the one exception: it starts the sweep and therefore writes dres in full.
// * A bwd needs its own fwd's saves. The trainer's flow is design §6.2's layer recompute:
//
//       forward sweep:   layer_fwd(..., JT_FWD_ONLY) for every layer   (saves nothing)
//       backward sweep:  layer_fwd(..., JT_FWD_SAVE) from the saved input, then layer_bwd(...)
//
//   JT_FWD_SAVE keeps whatever the bwd reads in the save half of the arena (below); JT_FWD_ONLY keeps nothing and
//   returns the same numbers. layer_fwd() clears the save half before a SAVE pass, so a layer's saves never leak into the
//   next. A caller that invokes one piece directly (test_layer does) must call jt_saves_reset() itself first.
// * No cudaMalloc, ever, inside these calls (design §6.2): the arena below is allocated once, sized for the largest
//   batch, and reused. A piece that needs more than the arena has dies with a message naming
//   JT_ARENA_FLOATS_PER_TOKEN rather than reading past a buffer.
// * Work goes on ctx->stream and ctx->cublas, wrapped in CK()/CB() (design §12). No function synchronises the stream
//   unless its comment says so, and the stream is a non-blocking one (design §12: no cudaStreamCreate), so a caller that
//   brings its own copies must order them on ctx->stream too: a plain cudaMemcpy on the null stream does not wait for
//   these kernels.
// * Return JT_LAYER_OK, or JT_LAYER_NOT_IMPL from a stub (which prints "not implemented"). Nothing else returns
//   JT_LAYER_NOT_IMPL, so a caller can tell the difference between a missing piece and a failure.
#ifndef JT_LAYER_H
#define JT_LAYER_H

#include <stdio.h>
#include <stdlib.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "gdn.h"   // kernels/gdn.h: gdn_bwd_work and the checkpoint/scratch size helpers
#include "model.h" // Model, LayerWeights, JtBatch, Linear, JT_LAYER_*

// ---- the activation arena (design §6.2) -------------------------------------------------------------------------------
// One layer's forward activations are about 40 K floats per packed token and its backward about as many, and both exist at
// once (the bwd runs off the fwd's saves), so the per-token budget is 96 K floats = 384 KB. Attention's P (and dP) are
// per row and per head instead of per token — design §6.2 budgets 2 x 8 x L^2 floats for one row at a time, because the
// attention layer works row by row — and jt_ctx_alloc adds that term separately.
#define JT_ARENA_FLOATS_PER_TOKEN ((size_t)98304)

// What a piece sees: the model, the trainable buffer, the GEMM handle, the stream, and the scratch it may use.
typedef struct JtCtx {
  Model *model;             // borrowed: the frozen weights and the tensor layout
  Lora *lora;               // borrowed: the flat buffer the linear gradients accumulate into (and the readout)
  cublasHandle_t cublas;    // the one handle every GEMM goes through (gemm.h), set to `stream`
  cudaStream_t stream;      // the one stream everything is ordered on
  int lmax;                 // the longest row the context was sized for; rope tables cover positions 0..lmax-1
  float *rope_cos;          // device [lmax][64]: design §3.3's cos table (kernels/rope.h's rope_tables)
  float *rope_sin;          // device [lmax][64]

  // The activation arena, fp32, `arena_n` floats. Saves grow up from arena[0] (save_at is the next free word) and work
  // grows down from arena[arena_n-1] (work_at is the lowest word work has taken); the two halves must never meet. Only
  // jt_save_alloc / jt_work_alloc / jt_saves_reset / jt_work_restore touch the cursors.
  float *arena;
  size_t arena_n;
  size_t save_at;
  size_t work_at;

  // GDN scratch, shared by the 18 GDN layers because only one layer is live at a time (design §6.2, §8.3). ckpt is the
  // forward's state checkpoints (gdn_fwd writes it when the bwd will follow), work the backward's supporting buffers.
  float *gdn_ckpt;          // gdn_ckpt_floats(cap_rows, MT_GDN_HEADS, gdn_nseg)
  int gdn_nseg;             // ceil(lmax / JT_GDN_SEG)
  gdn_bwd_work gdn_work;    // sp / dq_part / dk_part / dbeta_part / dg_part
  int cap_tokens, cap_rows; // the sizes the arena and the GDN scratch were allocated for
} JtCtx;

// The arena's one rule: a fwd(SAVE) takes its saves from the front, a bwd takes its temporaries from the back and gives
// them back before it returns (so calls do not accumulate). Both die on overflow with the numbers in the message.
static inline size_t jt_arena_free(const JtCtx *c) { return c->work_at - c->save_at; }

static inline float *jt_save_alloc(JtCtx *c, size_t n) {
  if (n > jt_arena_free(c)) {
    fprintf(stderr, "layer: save arena overflow: %zu floats wanted, %zu free of %zu (raise JT_ARENA_FLOATS_PER_TOKEN)\n", n,
            jt_arena_free(c), c->arena_n);
    exit(1);
  }
  float *p = c->arena + c->save_at;
  c->save_at += n;
  return p;
}

static inline float *jt_work_alloc(JtCtx *c, size_t n) {
  if (n > jt_arena_free(c)) {
    fprintf(stderr, "layer: work arena overflow: %zu floats wanted, %zu free of %zu (raise JT_ARENA_FLOATS_PER_TOKEN)\n", n,
            jt_arena_free(c), c->arena_n);
    exit(1);
  }
  c->work_at -= n;
  return c->arena + c->work_at;
}

// A bwd brackets its temporaries with these two so repeated calls reuse the same words.
static inline size_t jt_work_mark(const JtCtx *c) { return c->work_at; }
static inline void jt_work_restore(JtCtx *c, size_t mark) { c->work_at = mark; }
// layer_fwd() does this before a JT_FWD_SAVE pass; a caller invoking a piece directly does it itself.
static inline void jt_saves_reset(JtCtx *c) { c->save_at = 0; }

// Allocate everything a layer call needs, once (design §6.2): the handle and stream, the rope tables, the arena sized for
// the largest batch the trainer will see, and the GDN checkpoints and backward scratch. cap_tokens is the largest packed
// token count (a .jtb's max_tokens_per_batch), cap_rows the largest batch (its row count), lmax the longest row.
JtCtx *jt_ctx_alloc(Model *m, Lora *l, int cap_tokens, int cap_rows, int lmax);
void jt_ctx_free(JtCtx *c);

// ---- forward modes (design §6.2's layer recompute) --------------------------------------------------------------------
#define JT_FWD_ONLY 0 // forward only: keep nothing, use work temporaries only (the forward sweep)
#define JT_FWD_SAVE 1 // keep what the bwd reads in the arena's save half (the recompute pass in front of a bwd)

// ---- the MLP half, src/layer_mlp.c (design §3.4, §4.5) ---------------------------------------------------------------
// x_out = x_in + down(silu(gate(m)) * up(m)), m = RMSNorm(x_in; post_attention_layernorm). x_in is the post-mixer
// residual, i.e. the horizontal split above. Saves for the bwd: m, the three LoRA u's, h1 = gate(m), h2 = up(m),
// act = silu(h1)*h2 and RMSNorm's r.
int layer_mlp_fwd(JtCtx *c, const LayerWeights *lw, const float *x_in, int n, const JtBatch *b, float *x_out, int mode);
// dres += the MLP's path; needs the fwd(SAVE) of the same x_in.
int layer_mlp_bwd(JtCtx *c, const LayerWeights *lw, const float *x_in, float *dres, int n, const JtBatch *b);

// ---- the GDN mixer, src/layer_gdn.c (design §3.6, §4.8) --------------------------------------------------------------
// x_out = x_in + out_proj(gated_rmsnorm(gdn(...), z)), with a = RMSNorm(x_in; input_layernorm) as the mixer's input:
// rmsnorm -> in_proj_qkv / in_proj_z / in_proj_a / in_proj_b -> conv -> silu -> q/k/v views -> l2norm (+1/sqrt(128) on q)
// -> gates -> gdn_fwd -> gated_rmsnorm(gate z) -> out_proj. c->gdn_ckpt receives the recurrence's checkpoints in SAVE
// mode (gdn_fwd's ckpt argument; NULL in ONLY mode) and c->gdn_work is the backward's scratch.
int layer_gdn_mixer_fwd(JtCtx *c, const LayerWeights *lw, const float *x_in, int n, const JtBatch *b, float *x_out,
                        int mode);
int layer_gdn_mixer_bwd(JtCtx *c, const LayerWeights *lw, const float *x_in, float *dres, int n, const JtBatch *b);

// ---- the full-attention mixer, src/layer_attn.c (design §3.5, §4.7) ---------------------------------------------------
// a = RMSNorm(x_in; input_layernorm) -> q_proj -> q/gate split (per head, interleaved!) -> k_proj / v_proj -> q_norm and
// k_norm (d = 256, zero-centred) -> rope (row-local positions) -> per row: to_head_major, kv_expand, scores, causal
// softmax, PV, from_head_major -> sigmoid gate -> o_proj. P and dP are one row's at a time: c->lmax bounds their size
// and the arena's attention term covers 2 * MT_FULL_HEADS * lmax^2 floats.
int layer_attn_mixer_fwd(JtCtx *c, const LayerWeights *lw, const float *x_in, int n, const JtBatch *b, float *x_out,
                         int mode);
int layer_attn_mixer_bwd(JtCtx *c, const LayerWeights *lw, const float *x_in, float *dres, int n, const JtBatch *b);

// ---- one whole decoder layer, src/model.c (design §3.4) --------------------------------------------------------------
// i is 0..23; x_out = x_in + mixer(...) then += mlp(...). `post_mixer`, when not NULL, receives the residual right after
// the mixer — dump_ref's `layer.{i}.post_mixer` key. It is REQUIRED when mode == JT_FWD_SAVE (layer_bwd takes it as an
// argument, because the MLP's fwd overwrites its own input) and must not alias x_out; in JT_FWD_ONLY it may be NULL.
int layer_fwd(JtCtx *c, int i, const float *x_in, int n, const JtBatch *b, float *x_out, float *post_mixer, int mode);
// The backward of layer_fwd: the MLP's path first, then the mixer's, both into `dres` (design §4.6). x_in is the layer's
// input (the mixer's) and post_mixer the value the fwd(SAVE) wrote. The caller has already run layer_fwd(SAVE).
int layer_bwd(JtCtx *c, int i, const float *x_in, const float *post_mixer, float *dres, int n, const JtBatch *b);

// ---- the decision head, src/head.c (design §4.1, §13) ----------------------------------------------------------------
// The head's per-row inputs, host side: everything the loss is computed from besides the logits. `targets` is the batch's
// target distribution [B][n_opts] fp32 on the device, zero past the row's live options (train.py's collate pads the same
// way); `wrow` is the row's weight [B] and `wgroup` is the accumulation group's summed weight (train.py's `gw`, = the
// batch size when the weights are all 1). `live` (host [B]) is each row's live option count — the logits past it are
// exactly -1e9, as train.py's masked_fill writes them.
//
// targets and wrow may be NULL, meaning "logits only": no loss and no dlogits (an evaluation that only wants the logits,
// and test_layer's path when it has no .jtb fixture for the targets).
typedef struct {
  int n_opts;            // the widest row's live count = the logits width (train.py's option_logits slice)
  const int *live;       // host [B], each <= n_opts
  const float *targets;  // device [B][n_opts], or NULL
  const float *wrow;     // device [B], or NULL
  float wgroup;          // train.py's gw: the group's summed weight
} HeadBatch;

// final = RMSNorm(x; norm) -> gather each row's last token (off[b+1]-1) -> logits = z R^T over the first n_opts readout
// rows, with the dead options set to -1e9 -> ce_head. `final` and `logits` are the caller's device buffers: final is
// [n][MT_HIDDEN] (dump_ref's `final` key, plan M4.3 item 5) and may be NULL when only the logits are wanted; logits is
// [B][n_opts], which a step dumps and an evaluation keeps its probabilities from. *loss receives train.py's batch loss —
// sum_r loss_out[r], the decomposition of (per * wt).sum() / gw — and this is the one function that synchronises the
// stream to read it; pass NULL when the value is not wanted. In JT_FWD_SAVE mode it also saves what head_bwd needs
// (r, the gathered z, dlogits).
int head_fwd(JtCtx *c, const float *x, int n, const JtBatch *b, const HeadBatch *hb, float *final, float *logits,
             float *loss, int mode);
// dres = dL/dx for the head's input (layer 23's output): scatter dlogits' dz to each row's last token, then the final
// RMSNorm's backward over all n tokens. dres is written in full (this is the start of the backward sweep) and its rows
// other than off[b+1]-1 hold no dz. Needs the head_fwd(SAVE) of the same x.
int head_bwd(JtCtx *c, const float *x, int n, const JtBatch *b, float *dres);

#endif // JT_LAYER_H
