// src/model_table.h — the model described as data (design §13, §3.1, §3.2): 24 decoder layers, the tensor names each one
// loads from model.safetensors, their shapes, and which of them carry a LoRA pair.
//
// Design §13 puts this table in model.c. It lives here instead because src/lora.c has to lay out the flat parameter buffer
// from the same table before model.c exists (plan J12 vs J18): src/model_table.c is the one description, and model.c
// (J18) includes this header and loads from it — it does not grow a second table of names.
//
// The design's struct Linear (D§13):
//   struct Linear { const float *W; int in, out; int lora; size_t a_off, b_off; const char *stem; }
// is split in two on purpose. MtLinear below is the *description* (a name, its shape, whether it is adapted, all
// compile-time constants); the pointer to W and the a_off/b_off offsets into the flat buffer are *runtime* fields that
// model.c and lora.c fill in (design §13's `Linear` is that runtime view; lora.h's LoraEntry is the same offsets seen flat).
//
// Every name here is the full safetensors tensor name (all frozen weights are bf16 in the checkpoint, design §3.2). The
// adapter key of an adapted linear is that name with ".weight" removed plus ".lora_A"/".lora_B" (design §9.2), which is
// exactly what a Python-written adapter file contains — checked against `tools/jeff_lora/lora.py`'s attach_lora() on the
// real config: 150 wrapped linears named `language_model.layers.{i}.…` (see notes/J12.md for the command).
#ifndef JT_MODEL_TABLE_H
#define JT_MODEL_TABLE_H

#include <stddef.h>

// ---- the config (design §3.1), as numbers the rest of the trainer can use ---------------------------------------------
#define MT_LAYERS 24
#define MT_HIDDEN 1024
#define MT_MLP_INTER 3584
#define MT_VOCAB 248320 // only the embedding table is indexed; the LM head is never computed
#define MT_OPTIONS 255  // readout rows: the largest n_opts a row can have (design §2, §2.1)
#define MT_RMS_EPS 1e-6f // every RMSNorm, including q_norm/k_norm and the GDN gated norm

// LoRA (design §2.1): rank 16, alpha 32, so scale = alpha/rank = 2.0.
#define MT_LORA_RANK 16
#define MT_LORA_ALPHA 32.0f
#define MT_LORA_SCALE (MT_LORA_ALPHA / (float)MT_LORA_RANK)

// Full attention (design §3.1, §3.2, §3.5): layer i has full attention iff i % 4 == 3.
#define MT_FULL_HEADS 8
#define MT_KV_HEADS 2
#define MT_HEAD_DIM 256
#define MT_ROPE_DIM 64  // partial rotary: the first 64 of each 256-dim head
#define MT_Q_OUT 4096   // 8 heads x (256 query + 256 gate), interleaved per head
#define MT_KV_OUT 512   // 2 kv heads x 256
#define MT_O_IN 2048    // 8 heads x 256

// GDN / linear attention (design §3.1, §3.6).
#define MT_GDN_HEADS 16
#define MT_GDN_HEAD_DIM 128
#define MT_GDN_QKV_OUT 6144 // [q 0..2047 | k 2048..4095 | v 4096..6143]
#define MT_GDN_QKV_IN 1024
#define MT_GDN_Z_OUT 2048
#define MT_GDN_CONV_K 4

// 150 adapted linears (design §2.1): 6 full-attention layers x 4 + 18 GDN layers x 3 + 24 layers x 3 MLP.
#define MT_LORA_PAIRS 150

// ---- the descriptions ------------------------------------------------------------------------------------------------
// A linear layer's checkpoint weight, [out, in] row-major. `lora` is MT_LORA when the layer is adapted, in which case the
// flat buffer holds A [rank, in] and B [out, rank] for it (fp32, design §2.1).
typedef enum { MT_FROZEN = 0, MT_LORA = 1 } MtAdapt;

typedef struct {
  const char *name; // full safetensors name, e.g. "language_model.layers.3.self_attn.q_proj.weight"
  int in;           // weight shape [out, in] as stored
  int out;
  int lora; // MtAdapt
} MtLinear;

// One decoder layer. The fields that do not apply to its kind are NULL / zero, so a full-attention layer has no
// conv1d/a_log/dt_bias/in_proj_a/in_proj_b/gdn_norm and a GDN layer has no q_proj/k_proj/v_proj/o_proj/q_norm/k_norm.
// Use mt_layer_full(i) rather than testing which field is NULL.
typedef struct {
  int index;           // 0..23
  int full_attention;  // 1 for i % 4 == 3, else 0
  const char *prefix;  // "language_model.layers.{i}" — for messages and dump_ref.py keys
  const char *input_norm;        // [1024], zero-centred (design §3.3)
  const char *post_attention_norm; // [1024], zero-centred
  // MLP (every layer, all three adapted): gate/up [3584,1024], down [1024,3584].
  MtLinear gate, up, down;
  // Full attention only (all four adapted): q has 256 query + 256 gate rows per head.
  MtLinear q_proj, k_proj, v_proj, o_proj;
  const char *q_norm, *k_norm; // [256] per head, zero-centred
  // GDN only: in_proj_qkv [6144,1024], in_proj_z [2048,1024] and out_proj [1024,2048] are adapted;
  // in_proj_a / in_proj_b [16,1024] are not (design §2.1).
  MtLinear in_proj_qkv, in_proj_z, out_proj;
  const char *in_proj_a, *in_proj_b; // [16,1024], frozen
  const char *conv1d;                // [6144,1,4] depthwise, causal, tap order design §3.6
  const char *a_log, *dt_bias;       // [16], frozen
  const char *gdn_norm;              // [128] per head, NOT zero-centred (design §3.3)
} MtLayer;

// The 24 layers, in checkpoint order.
extern const MtLayer mt_layers[MT_LAYERS];

// The model-level tensors (design §3.2).
extern const char *const mt_embed_name;      // "language_model.embed_tokens.weight", [248320,1024] bf16
extern const char *const mt_final_norm_name; // "language_model.norm.weight", [1024], zero-centred
// The readout lives in its own file next to the checkpoint (design §2.1, §3.2): a separate safetensors with one tensor.
extern const char *const mt_readout_file;   // "readout.safetensors"
extern const char *const mt_readout_name;   // "weight", [255,1024] bf16 in the checkpoint, fp32 in the trainer
#define MT_READOUT_IN MT_HIDDEN
#define MT_READOUT_OUT MT_OPTIONS

// mt_layer(i) is &mt_layers[i] for 0 <= i < MT_LAYERS, else NULL. mt_layer_full(i) is its full_attention flag.
const MtLayer *mt_layer(int i);
int mt_layer_full(int i);

// The adapted linears in flat-buffer order (what src/lora.c lays out): layer by layer, and within a layer its attention
// projections then the MLP's gate/up/down. mt_lora_pair_count() is MT_LORA_PAIRS (150); mt_lora_pair(k) returns the k-th
// one, or NULL past the end. Each returned MtLinear has lora == MT_LORA.
int mt_lora_pair_count(void);
const MtLinear *mt_lora_pair(int k);

#endif // JT_MODEL_TABLE_H
