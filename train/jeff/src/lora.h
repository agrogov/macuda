// src/lora.h — the flat trainable-parameter buffer: LoRA A/B for the model's 150 adapted linears plus the readout, all in
// one fp32 array (design §8.1's "Keep all LoRA A, B and the readout in ONE flat fp32 buffer … with a host-side table of
// offsets", §2.1 for the counts, §2.2 for the optimizer and the schedule).
//
// Why one buffer: zeroing, the global gradient norm, clipping and the Adam step are each one launch over the flat buffer,
// the two learning-rate groups are one index (`readout_off`: everything below it is LoRA, everything at or above it is
// the readout), and saving is one copy. The table of entries below is built from src/model_table.h — the one description
// of the model's tensors — so a name or a shape cannot disagree with what model.c loads.
//
// Ordering is the model table's: layer 0..23, and within a layer its attention projections then the MLP's gate/up/down
// (mt_lora_pair()). Each LoRA entry owns A [rank, in] then B [out, rank], contiguous, in that order; the readout is the
// last entry (entry 150) and owns the [MT_READOUT_OUT, MT_READOUT_IN] matrix. 150 pairs and 1 readout = 151 entries,
// 10,223,616 + 261,120 = 10,484,736 floats (design §2.1, the number train.py prints).
#ifndef JT_LORA_H
#define JT_LORA_H

#include <stddef.h>

#include <cuda_runtime.h>

#include "model_table.h"

#define LORA_RANK MT_LORA_RANK
#define LORA_PAIRS MT_LORA_PAIRS
#define LORA_ENTRIES (MT_LORA_PAIRS + 1)
// The readout's entry, i.e. the first entry of the second lr group (design §2.2).
#define LORA_READOUT_ENTRY MT_LORA_PAIRS
// "language_model.layers.23.linear_attn.in_proj_qkv" is 48 bytes; the pool leaves room and is checked at build time.
#define LORA_STEM_MAX 72

// One entry of the flat buffer: a LoRA pair (rank > 0) or the readout (rank == 0).
typedef struct {
  char stem[LORA_STEM_MAX]; // the adapter key prefix: "language_model.layers.0.mlp.gate_proj", or "readout"
  int in, out;              // the frozen weight's shape [out, in]
  int rank;                 // LORA_RANK, or 0 for the readout
  size_t off;               // element offset of A [rank,in], or of the readout [out,in]
  size_t b_off;             // element offset of B [out,rank]; 0 for the readout
  size_t count;             // elements owned here: rank*in + out*rank, or out*in
} LoraEntry;

// The buffer and its table. param is the master copy of the weights (fp32, what Adam updates), grad the accumulated
// gradient, m and v Adam's two moments — four device arrays of n_param floats, same layout (design §8.1).
typedef struct {
  size_t n_param;     // 10,484,736
  size_t readout_off; // == ent[LORA_READOUT_ENTRY].off: the lr group split for adamw_step
  int n_ent;          // 151
  LoraEntry *ent;     // [n_ent]
  float *param, *grad, *m, *v;
} Lora;

// Build the table from the model table and allocate the four flat buffers (zeroed). Prints the trainable count in
// train.py's format ("trainable 10,484,736") and a one-line summary of the layout. Dies on failure, like CK().
Lora *lora_alloc(void);
void lora_free(Lora *l);

// grad = 0 (torch's opt.zero_grad(), before a new accumulation group).
void lora_zero_grad(const Lora *l, cudaStream_t s);

// Load a Python-written adapter directory (design §9.2): adapter_model.safetensors with `{stem}.lora_A` / `{stem}.lora_B`
// for all 150 pairs, and readout.safetensors with `weight`. The factors are fp32 in the trainer's own format, but the
// readout is bf16 in a checkpoint and fp32 when train.py saved it, so both are converted here (plan M2.6). Dies with a
// message naming the missing tensor, the wrong shape or the wrong dtype: a half-read adapter is never silently accepted.
void lora_load_adapter(Lora *l, const char *dir);

// Save what lora_load_adapter reads: `<dir>/adapter_model.safetensors` (300 factors, fp32), `<dir>/readout.safetensors`
// (the readout as fp32, as design §9.2 asks), and `<dir>/adapter_config.json`. `step` and `dev_ce` are train.py's
// selection metadata (`saved` at the best dev soft CE); meta (may be NULL) supplies the rest of its payload: the base
// checkpoint path and the --targets/--weights names.
typedef struct {
  const char *base;    // the checkpoint train.py would call "base"
  const char *targets; // "soft" or "hard"
  const char *weights; // "none" or "balanced"
} LoraMeta;
void lora_save_adapter(const Lora *l, const char *dir, int step, double dev_ce, const LoraMeta *meta);

// Design §2.2's schedule, as a multiplier on a group's base lr, at s = the number of optimizer steps already taken (0 for
// the first step), in double exactly as train.py's LambdaLR lambda computes it:
//   f(s) = min(1, (s+1)/max(1, int(total*0.03))) * (0.1 + 0.9*0.5*(1 + cos(pi*min(s,total)/total)))
// (train.py:307-308.) `total` < 1 is treated as 1 (train.py never runs a zero-step schedule); `base` is the group's lr,
// so lr_at(s, total, 2e-4) is the LoRA group's lr for step s and lr_at(s, total, 5e-6) the readout's.
double lr_at(int step, int total, double base);

#endif // JT_LORA_H
