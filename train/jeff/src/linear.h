// src/linear.h — the frozen linear layer and the two calls that touch a weight (design §13, §4.2, §5), plus the three
// return codes every layer-level function in this tree uses.
//
// Design §13: "`struct Linear { const float *W; int in, out; int lora; size_t a_off, b_off; const char *stem; }`: `lora`
// set means `a_off`/`b_off` index the flat parameter buffer. `linear_fwd(ctx, &lin, x, N, y, u_save)` and
// `linear_bwd(ctx, &lin, x, u, dy, N, dx, accumulate_dx)` are the only functions that touch W, A or B. Every layer calls
// them." That is the contract this header freezes; src/linear.c is the only place that may talk to a weight's memory.
//
// The maths is design §4.2, in the order it is written there (ref/linear.c is the CPU oracle of the same order, and
// test/gradcheck's `lora` group is its gate):
//
//   forward   y  = x W^T + s * (x A^T) B^T,   s = alpha/rank = MT_LORA_SCALE = 2.0, u = x A^T saved [n, rank]
//   backward  e  = s * (dy B)                 [n, rank]      (built once, reused by dx and dA)
//             dx = dy W + e A                 [n, in]        (or += with accumulate_dx)
//             dB += s * dy^T u                [out, rank]    into the flat buffer's gradient
//             dA += e^T x                     [rank, in]     into the flat buffer's gradient
//
// Layouts, all row-major and all fp32 on the device: x [n,in], W [out,in], A [rank,in], B [out,rank], u [n,rank],
// y [n,out], dy [n,out], dx [n,in]. The frozen weights are the checkpoint's bf16 widened once at load (design §6.1), so
// the backward uses the same W pointer the forward did and no W^T is ever stored (design §5).
//
// Two frozen-weight storages (plan M8.1b). By default a Linear carries `W`, fp32 [out][in], exactly as above. With
// JT_BF16_WEIGHTS=1 (and only outside exact mode — model.c refuses it under JT_EXACT) the model instead keeps the
// weight bf16 in `Wb`, the checkpoint's own bits, and allocates no fp32 copy: 1.99 GB of frozen weights become 1.00 GB.
// `Wb != NULL` is then the whole test, and a linear that carries `Wb` also has `W` set to a pointer that is **not a
// weight**: eight zeroed fp32 words shared by every such linear (src/model.h's `w_mark`), so a caller that only asks
// "is this linear loaded?" (src/layer_mlp.c's `W == NULL` argument check) still sees a loaded one. Read a weight from
// `Wb` when it is set, and never dereference `W` then. The LoRA factors, the readout and everything trainable stay
// fp32 in both storages (design §6.1).
//
// dA and dB are *accumulated* into ctx->lora->grad (design §4.2, §8.1): a gradient belongs to the step, not to the
// layer, and an accumulation group sums several batches into it. A frozen linear (lora == 0, e.g. the GDN in_proj_a /
// in_proj_b, which design §2.1 does not adapt) writes no parameter gradient at all.
//
// Why the return type is int: src/linear.c and the four layer files arrive as stubs (plan J18) and a stub has to say
// "not implemented" without aborting a test run — so every layer-level entry point returns one of the three codes
// below. They are the same numbers as test/harness.h's JT_OK / JT_FAIL / JT_NOT_IMPL, so a test can map one to the other
// without a table, and they are deliberately *different names* so a test may include both headers.
#ifndef JT_LINEAR_H
#define JT_LINEAR_H

#include <stddef.h>

// The context every layer call takes. src/layer.h defines it (it needs the model, the LoRA buffer and the activation
// arena); linear.c only needs to know that it exists.
typedef struct JtCtx JtCtx;

// The layer-level return codes (see the header comment). JT_LAYER_NOT_IMPL is what a stub returns; nothing else ever
// does, so a caller can tell "not written yet" from "failed" by the value alone.
enum {
  JT_LAYER_OK = 0,       // did what it says
  JT_LAYER_FAIL = 1,     // a check inside failed (the function printed why)
  JT_LAYER_NOT_IMPL = 2, // stub: nothing here yet (the function printed "not implemented")
};

// One linear of the model. Model.c fills these from src/model_table.h at load; a layer never builds one by hand.
typedef struct Linear {
  const float *W;           // [out][in] fp32 device, frozen (widened from the checkpoint's bf16 at load)
  const unsigned short *Wb; // [out][in] bf16 device: the frozen weight when the model keeps bf16 (see below)
  int in;                   // columns of x / of W's rows
  int out;                  // columns of y
  int lora;                 // 1 = adapted (A and B live in the flat parameter buffer), 0 = frozen
  size_t a_off;             // element offset of A [rank][in] in ctx->lora->param; 0 when lora == 0
  size_t b_off;             // element offset of B [out][rank]; 0 when lora == 0
  const char *stem;         // the adapter key prefix, `{stem}.lora_A` / `{stem}.lora_B` (design §9.2); for messages and tests
} Linear;

// y = x W^T (+ s (x A^T) B^T). x [n,in], y [n,out]; both are written in full and may not alias. u_save, when not
// NULL, receives the saved u = x A^T [n,rank] the backward needs (design §4.2); pass NULL when no bwd will follow.
// A frozen linear writes only y.
int linear_fwd(JtCtx *ctx, const Linear *lin, const float *x, int n, float *y, float *u_save);

// The backward of linear_fwd. `u` is the forward's saved u [n,rank] (required when lora != 0). dx is written when
// accumulate_dx == 0 and accumulated (`+=`) when it is 1, so a caller may build a residual's gradient from several
// paths (design §4.6). dA/dB always accumulate into ctx->lora->grad. dx may be NULL when only the parameter gradient is
// wanted; a frozen linear then has nothing to do.
int linear_bwd(JtCtx *ctx, const Linear *lin, const float *x, const float *u, const float *dy, int n, float *dx,
               int accumulate_dx);

#endif // JT_LINEAR_H
