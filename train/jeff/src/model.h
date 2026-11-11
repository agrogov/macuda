// src/model.h — the model as weights: one fp32 arena of frozen tensors, the bf16 embedding table, the readout in the
// flat parameter buffer, a batch's row geometry, and the forward driver (design §3.2, §6.1, §6.2, §7, §13).
//
// Design §13 puts `struct Model` in model.c: "weight pointers per layer, LoRA offsets; load from safetensors;
// forward/backward drivers". The tensor *names and shapes* are not here — src/model_table.h (J12) is the one description
// and model.c loads from it, so a shape cannot be written down twice. What is here is the runtime view: which device
// pointer holds which weight, per layer.
//
// Memory (design §6.1, §6.2): every frozen text weight is widened bf16 -> fp32 once at load into a single arena (about
// 2.0 GB, printed at load), except `embed_tokens`, which stays bf16 on the device (508 MB) because only the rows a batch
// uses are ever wanted and design §8.1's jt_embed_gather widens those. The trainable parameters are not here at all: the
// LoRA factors and the readout live in src/lora.c's flat buffer, and a Linear only carries their offsets.
//
// plan M8.1b's `JT_BF16_WEIGHTS=1` adds a second storage: the frozen *Linear* weights (the 0.5 B parameters of the two
// weight GEMMs) are then kept as bf16 in `wb`, the checkpoint's own bits, and no fp32 copy of them is allocated — the
// fp32 arena shrinks to the norms, the conv taps and the gate vectors, about 2 MB. The embedding table is bf16 in both
// storages, so the whole frozen set goes 1.99 + 0.51 GB -> 1.00 + 0.51 GB (the load prints `weights: bf16 (x GB)`).
// Since task J51 it is the *default* in TF32 mode (src/main.c's m8_defaults: worth it on gpubox and on the card, in
// G-m8arm's arm); JT_BF16_WEIGHTS=0 asks for the fp32 storage. Exact mode never takes it (src/main.c does not set it
// there, and model.c refuses it): M6.2's trace is bit-identical only because the exact path never changes.
//
// Nothing is random and nothing is a guess: st_get checks the dtype and every dimension of every tensor it is asked for
// (src/st.h), so a checkpoint with a different layout fails at load, naming the tensor and both shapes.
#ifndef JT_MODEL_H
#define JT_MODEL_H

#include <stddef.h>

#include "linear.h"       // Linear, JT_LAYER_*
#include "lora.h"         // Lora (the flat parameter buffer, which owns the readout)
#include "model_table.h"  // MT_* and the 24 layers' tensor descriptions

// A batch has at most this many rows (design §7's `off` search note, "B <= 64"); JtBatch keeps the host row table in the
// struct itself, so the limit is the struct's, not a pointer's.
#define JT_MAX_ROWS 64

// One decoder layer's weights, fp32 device pointers. The fields that do not apply to the layer's kind stay NULL
// (mt_layer(i)->full_attention says which kind it is; do not test a field for NULL).
typedef struct {
  const float *input_norm;      // [MT_HIDDEN], zero-centred (design §3.3: y = x r (1+w))
  const float *post_attn_norm;  // [MT_HIDDEN], zero-centred
  Linear gate, up, down;        // MLP, all three adapted [3584,1024] / [3584,1024] / [1024,3584]

  // Full attention only (design §3.5). q_proj's rows are interleaved per head: [h*512, h*512+256) is head h's query
  // block and [h*512+256, h*512+512) its gate block — NOT "first 2048 queries, last 2048 gates".
  Linear q_proj, k_proj, v_proj, o_proj; // [4096,1024], [512,1024], [512,1024], [1024,2048]
  const float *q_norm, *k_norm;          // [MT_HEAD_DIM], zero-centred

  // GDN only (design §3.6). in_proj_a / in_proj_b are frozen (design §2.1), so their Linear has lora == 0 and W set.
  Linear in_proj_qkv, in_proj_z, out_proj; // [6144,1024], [2048,1024], [1024,2048]
  Linear in_proj_a, in_proj_b;             // [16,1024] frozen
  const float *conv1d;                     // [6144][4], the checkpoint's [6144,1,4] flattened: conv1d[ch*4 + k] multiplies
                                           // x[t-3+k, ch] (design §3.6's tap order; kernels/conv.h)
  const float *a_log;                      // [16] fp32, the checkpoint's A_log (converted on the host)
  const float *neg_a;                      // [16] fp32, -exp(A_log) (plan M4.2)
  const float *dt_bias;                    // [16] fp32
  const float *gdn_norm;                   // [128] per head, NOT zero-centred (plain w, not 1+w)
} LayerWeights;

// The model. Every pointer is device memory owned by the Model (or, for the readout, by the borrowed Lora).
typedef struct Model {
  float *w;                 // the fp32 arena of frozen weights (design §6.1): everything but the bf16 linears' weights
  size_t w_floats;          // its size, in floats (the load prints it as GB)
  unsigned short *wb;       // the bf16 arena of frozen Linear weights (plan M8.1b); NULL in the fp32 storage
  size_t wb_el;             // its size, in bf16 elements (the load prints it as GB)
  const float *w_mark;      // 8 zeroed fp32 words: what every bf16 linear's Linear.W points at (src/linear.h)
  unsigned short *embed;    // [MT_VOCAB][MT_HIDDEN] bf16, never widened (design §8.1's jt_embed_gather)
  size_t embed_el;
  const float *final_norm;  // [MT_HIDDEN], zero-centred
  LayerWeights layers[MT_LAYERS];
  int n_gdn, n_full;        // 18 and 6, from the model table
  Lora *lora;               // borrowed: the flat parameter buffer whose readout slot holds the [255,1024] readout
} Model;

// Load a checkpoint directory (or a model.safetensors file): every tensor of src/model_table.h is checked with st_get
// and widened into one fp32 arena on the stream-0 default, `embed_tokens` is copied as bf16, A_log/dt_bias are converted
// on the host together with neg_a = -exp(A_log), and the readout is read from `<dir>/readout.safetensors` (bf16) into
// `lora->param + lora->readout_off` as fp32 (plan M4.2). A later lora_load_adapter() overwrites the readout, which is
// what train.py's --init-adapter does. Prints, in this order:
//   model: 24 layers (18 gdn, 6 attn), weights 1.99 GB fp32 + embed 0.51 GB bf16
//   model: 321 frozen tensors checked by st_get (dtype and every dimension), readout 1.04 MB fp32 in the flat buffer
// with JT_BF16_WEIGHTS=1 (and not in exact mode) the first line splits the two storages and one more line follows:
//   model: 24 layers (18 gdn, 6 attn), weights 1.00 GB bf16 + 0.00 GB fp32 + embed 0.51 GB bf16
//   weights: bf16 (1.00 GB)
// and in exact mode with the toggle set, where the fp32 storage is kept, `weights: bf16 ignored (...)` says so.
// Dies with a message naming the tensor, its shape and the expected one if anything disagrees.
Model *model_load(const char *path, Lora *lora);
void model_free(Model *m);

// A packed batch's row geometry (design §7): rows = B, n = off[B] tokens, off[b] is row b's first token. The host table
// lives in the struct; jt_batch_set copies it into permanent device arrays (`off_dev`, which the kernels see as their
// row_off, and `row_of`, token -> row) and dies if the table is not a prefix sum or B > JT_MAX_ROWS. jt_batch_free
// releases the two device arrays; the struct itself belongs to the caller.
//
// The uploads use the null stream, so call jt_batch_set before the first call on a context's (non-blocking) stream, or
// synchronise the device afterwards: jt_ctx_alloc() ends with one device-wide barrier, which orders everything a load
// did before the context existed.
typedef struct {
  int rows;                     // B
  int n;                        // off[rows]
  int off[JT_MAX_ROWS + 1];     // host prefix sums, off[rows] == n
  int *off_dev;                 // device [rows+1] (== the kernels' row_off)
  int *row_of;                  // device [n], token index -> row
} JtBatch;

void jt_batch_set(JtBatch *b, int rows, int n, const int *off);
void jt_batch_free(JtBatch *b);

// The whole-model forward (design §3.4, §6.2, §7): embedding gather, then the 24 decoder layers, each layer's output
// feeding the next. Ends at layer 23's output — the final norm and the head are head_fwd's (design §13, plan M4.3 item
// 5). `saved_in` holds 25 * n * MT_HIDDEN floats: saved_in[0] is the embedding's output, saved_in[i] is layer i's input
// and saved_in[24] is layer 23's output, i.e. exactly design §6.2's "the forward keeps only each layer's input" (820 MB
// at N = 8 x 1000 worst case). `ids` is a device int array of n token ids. `post_mixer`, when not NULL, holds
// MT_LAYERS * n * MT_HIDDEN floats and receives dump_ref's `layer.{i}.post_mixer` key for every layer (plan
// M4.3/M4.4); a training step passes NULL and saves nothing.
//
// This is a forward-only sweep: no layer keeps its activations, because the backward re-runs each layer from its saved
// input (design §6.2, the layer-recompute plan). A layer that is still a stub makes model_forward return
// JT_LAYER_NOT_IMPL after that layer's own message.
int model_forward(JtCtx *ctx, const int *ids, int n, const JtBatch *b, float *saved_in, float *post_mixer);

#endif // JT_MODEL_H
