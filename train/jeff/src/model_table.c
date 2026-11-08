// src/model_table.c — the model as data (design §13, §3.1, §3.2): the one place the checkpoint's tensor names and shapes
// are written down. See src/model_table.h for why this lives beside lora.c instead of in model.c (design §13 names
// model.c; src/lora.c needs the table before model.c exists, and J18's model.c includes this header rather than
// repeating a name).
//
// Every entry below is a line of design §3.2, in the checkpoint's own names and shapes. The list was checked twice
// against the real checkpoint (read-only, on this machine): against the header of model.safetensors (473 tensors, of
// which the table names the 24 layers' own tensors and the two model-level ones) and against
// tools/jeff_lora/lora.py's attach_lora() on the real config, which wraps exactly 150 linears named
// `language_model.layers.{i}.…` — the stems of design §9.2's adapter keys. The shapes are the header's constants (MT_*),
// so a number appears once in the tree.
//
// Which linears carry LoRA (design §2.1): all 24 x gate/up/down, the 6 full-attention layers' q/k/v/o, the 18 GDN
// layers' in_proj_qkv/in_proj_z/out_proj. Not adapted: in_proj_a, in_proj_b, conv1d, every norm, A_log, dt_bias, the
// embedding and the final norm. The readout is trained but is not a LoRA pair, so it is described separately (and it is
// not a tensor of model.safetensors: it comes from readout.safetensors, design §2.1).
#include "model_table.h"

const MtLayer mt_layers[MT_LAYERS] = {
    {.index = 0,
     .full_attention = 0,
     .prefix = "language_model.layers.0",
     .input_norm = "language_model.layers.0.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.0.post_attention_layernorm.weight",
     .gate = {"language_model.layers.0.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.0.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.0.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.0.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.0.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.0.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.0.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.0.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.0.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.0.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.0.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.0.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 1,
     .full_attention = 0,
     .prefix = "language_model.layers.1",
     .input_norm = "language_model.layers.1.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.1.post_attention_layernorm.weight",
     .gate = {"language_model.layers.1.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.1.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.1.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.1.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.1.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.1.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.1.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.1.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.1.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.1.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.1.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.1.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 2,
     .full_attention = 0,
     .prefix = "language_model.layers.2",
     .input_norm = "language_model.layers.2.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.2.post_attention_layernorm.weight",
     .gate = {"language_model.layers.2.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.2.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.2.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.2.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.2.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.2.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.2.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.2.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.2.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.2.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.2.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.2.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 3,
     .full_attention = 1,
     .prefix = "language_model.layers.3",
     .input_norm = "language_model.layers.3.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.3.post_attention_layernorm.weight",
     .gate = {"language_model.layers.3.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.3.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.3.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .q_proj = {"language_model.layers.3.self_attn.q_proj.weight", MT_HIDDEN, MT_Q_OUT, MT_LORA},
     .k_proj = {"language_model.layers.3.self_attn.k_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .v_proj = {"language_model.layers.3.self_attn.v_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .o_proj = {"language_model.layers.3.self_attn.o_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .q_norm = "language_model.layers.3.self_attn.q_norm.weight",  // [MT_HEAD_DIM]
     .k_norm = "language_model.layers.3.self_attn.k_norm.weight",
    },
    {.index = 4,
     .full_attention = 0,
     .prefix = "language_model.layers.4",
     .input_norm = "language_model.layers.4.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.4.post_attention_layernorm.weight",
     .gate = {"language_model.layers.4.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.4.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.4.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.4.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.4.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.4.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.4.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.4.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.4.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.4.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.4.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.4.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 5,
     .full_attention = 0,
     .prefix = "language_model.layers.5",
     .input_norm = "language_model.layers.5.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.5.post_attention_layernorm.weight",
     .gate = {"language_model.layers.5.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.5.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.5.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.5.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.5.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.5.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.5.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.5.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.5.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.5.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.5.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.5.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 6,
     .full_attention = 0,
     .prefix = "language_model.layers.6",
     .input_norm = "language_model.layers.6.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.6.post_attention_layernorm.weight",
     .gate = {"language_model.layers.6.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.6.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.6.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.6.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.6.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.6.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.6.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.6.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.6.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.6.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.6.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.6.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 7,
     .full_attention = 1,
     .prefix = "language_model.layers.7",
     .input_norm = "language_model.layers.7.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.7.post_attention_layernorm.weight",
     .gate = {"language_model.layers.7.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.7.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.7.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .q_proj = {"language_model.layers.7.self_attn.q_proj.weight", MT_HIDDEN, MT_Q_OUT, MT_LORA},
     .k_proj = {"language_model.layers.7.self_attn.k_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .v_proj = {"language_model.layers.7.self_attn.v_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .o_proj = {"language_model.layers.7.self_attn.o_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .q_norm = "language_model.layers.7.self_attn.q_norm.weight",  // [MT_HEAD_DIM]
     .k_norm = "language_model.layers.7.self_attn.k_norm.weight",
    },
    {.index = 8,
     .full_attention = 0,
     .prefix = "language_model.layers.8",
     .input_norm = "language_model.layers.8.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.8.post_attention_layernorm.weight",
     .gate = {"language_model.layers.8.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.8.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.8.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.8.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.8.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.8.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.8.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.8.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.8.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.8.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.8.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.8.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 9,
     .full_attention = 0,
     .prefix = "language_model.layers.9",
     .input_norm = "language_model.layers.9.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.9.post_attention_layernorm.weight",
     .gate = {"language_model.layers.9.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.9.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.9.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.9.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.9.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.9.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.9.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.9.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.9.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.9.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.9.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.9.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 10,
     .full_attention = 0,
     .prefix = "language_model.layers.10",
     .input_norm = "language_model.layers.10.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.10.post_attention_layernorm.weight",
     .gate = {"language_model.layers.10.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.10.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.10.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.10.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.10.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.10.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.10.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.10.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.10.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.10.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.10.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.10.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 11,
     .full_attention = 1,
     .prefix = "language_model.layers.11",
     .input_norm = "language_model.layers.11.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.11.post_attention_layernorm.weight",
     .gate = {"language_model.layers.11.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.11.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.11.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .q_proj = {"language_model.layers.11.self_attn.q_proj.weight", MT_HIDDEN, MT_Q_OUT, MT_LORA},
     .k_proj = {"language_model.layers.11.self_attn.k_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .v_proj = {"language_model.layers.11.self_attn.v_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .o_proj = {"language_model.layers.11.self_attn.o_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .q_norm = "language_model.layers.11.self_attn.q_norm.weight",  // [MT_HEAD_DIM]
     .k_norm = "language_model.layers.11.self_attn.k_norm.weight",
    },
    {.index = 12,
     .full_attention = 0,
     .prefix = "language_model.layers.12",
     .input_norm = "language_model.layers.12.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.12.post_attention_layernorm.weight",
     .gate = {"language_model.layers.12.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.12.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.12.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.12.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.12.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.12.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.12.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.12.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.12.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.12.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.12.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.12.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 13,
     .full_attention = 0,
     .prefix = "language_model.layers.13",
     .input_norm = "language_model.layers.13.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.13.post_attention_layernorm.weight",
     .gate = {"language_model.layers.13.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.13.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.13.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.13.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.13.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.13.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.13.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.13.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.13.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.13.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.13.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.13.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 14,
     .full_attention = 0,
     .prefix = "language_model.layers.14",
     .input_norm = "language_model.layers.14.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.14.post_attention_layernorm.weight",
     .gate = {"language_model.layers.14.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.14.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.14.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.14.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.14.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.14.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.14.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.14.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.14.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.14.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.14.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.14.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 15,
     .full_attention = 1,
     .prefix = "language_model.layers.15",
     .input_norm = "language_model.layers.15.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.15.post_attention_layernorm.weight",
     .gate = {"language_model.layers.15.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.15.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.15.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .q_proj = {"language_model.layers.15.self_attn.q_proj.weight", MT_HIDDEN, MT_Q_OUT, MT_LORA},
     .k_proj = {"language_model.layers.15.self_attn.k_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .v_proj = {"language_model.layers.15.self_attn.v_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .o_proj = {"language_model.layers.15.self_attn.o_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .q_norm = "language_model.layers.15.self_attn.q_norm.weight",  // [MT_HEAD_DIM]
     .k_norm = "language_model.layers.15.self_attn.k_norm.weight",
    },
    {.index = 16,
     .full_attention = 0,
     .prefix = "language_model.layers.16",
     .input_norm = "language_model.layers.16.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.16.post_attention_layernorm.weight",
     .gate = {"language_model.layers.16.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.16.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.16.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.16.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.16.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.16.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.16.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.16.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.16.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.16.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.16.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.16.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 17,
     .full_attention = 0,
     .prefix = "language_model.layers.17",
     .input_norm = "language_model.layers.17.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.17.post_attention_layernorm.weight",
     .gate = {"language_model.layers.17.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.17.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.17.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.17.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.17.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.17.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.17.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.17.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.17.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.17.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.17.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.17.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 18,
     .full_attention = 0,
     .prefix = "language_model.layers.18",
     .input_norm = "language_model.layers.18.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.18.post_attention_layernorm.weight",
     .gate = {"language_model.layers.18.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.18.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.18.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.18.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.18.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.18.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.18.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.18.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.18.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.18.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.18.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.18.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 19,
     .full_attention = 1,
     .prefix = "language_model.layers.19",
     .input_norm = "language_model.layers.19.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.19.post_attention_layernorm.weight",
     .gate = {"language_model.layers.19.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.19.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.19.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .q_proj = {"language_model.layers.19.self_attn.q_proj.weight", MT_HIDDEN, MT_Q_OUT, MT_LORA},
     .k_proj = {"language_model.layers.19.self_attn.k_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .v_proj = {"language_model.layers.19.self_attn.v_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .o_proj = {"language_model.layers.19.self_attn.o_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .q_norm = "language_model.layers.19.self_attn.q_norm.weight",  // [MT_HEAD_DIM]
     .k_norm = "language_model.layers.19.self_attn.k_norm.weight",
    },
    {.index = 20,
     .full_attention = 0,
     .prefix = "language_model.layers.20",
     .input_norm = "language_model.layers.20.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.20.post_attention_layernorm.weight",
     .gate = {"language_model.layers.20.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.20.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.20.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.20.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.20.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.20.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.20.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.20.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.20.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.20.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.20.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.20.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 21,
     .full_attention = 0,
     .prefix = "language_model.layers.21",
     .input_norm = "language_model.layers.21.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.21.post_attention_layernorm.weight",
     .gate = {"language_model.layers.21.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.21.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.21.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.21.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.21.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.21.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.21.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.21.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.21.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.21.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.21.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.21.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 22,
     .full_attention = 0,
     .prefix = "language_model.layers.22",
     .input_norm = "language_model.layers.22.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.22.post_attention_layernorm.weight",
     .gate = {"language_model.layers.22.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.22.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.22.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .in_proj_qkv = {"language_model.layers.22.linear_attn.in_proj_qkv.weight", MT_GDN_QKV_IN, MT_GDN_QKV_OUT, MT_LORA},
     .in_proj_z = {"language_model.layers.22.linear_attn.in_proj_z.weight", MT_HIDDEN, MT_GDN_Z_OUT, MT_LORA},
     .out_proj = {"language_model.layers.22.linear_attn.out_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .in_proj_a = "language_model.layers.22.linear_attn.in_proj_a.weight",  // [16,1024], frozen
     .in_proj_b = "language_model.layers.22.linear_attn.in_proj_b.weight",
     .conv1d = "language_model.layers.22.linear_attn.conv1d.weight",      // [6144,1,4]
     .a_log = "language_model.layers.22.linear_attn.A_log",               // [16]
     .dt_bias = "language_model.layers.22.linear_attn.dt_bias",            // [16]
     .gdn_norm = "language_model.layers.22.linear_attn.norm.weight",       // [128], not zero-centred
    },
    {.index = 23,
     .full_attention = 1,
     .prefix = "language_model.layers.23",
     .input_norm = "language_model.layers.23.input_layernorm.weight",
     .post_attention_norm = "language_model.layers.23.post_attention_layernorm.weight",
     .gate = {"language_model.layers.23.mlp.gate_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .up = {"language_model.layers.23.mlp.up_proj.weight", MT_HIDDEN, MT_MLP_INTER, MT_LORA},
     .down = {"language_model.layers.23.mlp.down_proj.weight", MT_MLP_INTER, MT_HIDDEN, MT_LORA},
     .q_proj = {"language_model.layers.23.self_attn.q_proj.weight", MT_HIDDEN, MT_Q_OUT, MT_LORA},
     .k_proj = {"language_model.layers.23.self_attn.k_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .v_proj = {"language_model.layers.23.self_attn.v_proj.weight", MT_HIDDEN, MT_KV_OUT, MT_LORA},
     .o_proj = {"language_model.layers.23.self_attn.o_proj.weight", MT_O_IN, MT_HIDDEN, MT_LORA},
     .q_norm = "language_model.layers.23.self_attn.q_norm.weight",  // [MT_HEAD_DIM]
     .k_norm = "language_model.layers.23.self_attn.k_norm.weight",
    },
};

// The adapted linears in flat-buffer order: layer by layer, the attention projections first, then gate/up/down. src/lora.c
// walks this list, so it fixes both the offset of every A/B in the flat buffer and the order of the 300 adapter keys.
// The array is unsized so that the typedef below is a real compile-time check of its length (a short initializer would
// otherwise just leave NULLs behind).
static const MtLinear *const mt_lora_pairs[] = {
  &mt_layers[ 0].in_proj_qkv, &mt_layers[ 0].in_proj_z, &mt_layers[ 0].out_proj, &mt_layers[ 0].gate, &mt_layers[ 0].up, &mt_layers[ 0].down,
  &mt_layers[ 1].in_proj_qkv, &mt_layers[ 1].in_proj_z, &mt_layers[ 1].out_proj, &mt_layers[ 1].gate, &mt_layers[ 1].up, &mt_layers[ 1].down,
  &mt_layers[ 2].in_proj_qkv, &mt_layers[ 2].in_proj_z, &mt_layers[ 2].out_proj, &mt_layers[ 2].gate, &mt_layers[ 2].up, &mt_layers[ 2].down,
  &mt_layers[ 3].q_proj, &mt_layers[ 3].k_proj, &mt_layers[ 3].v_proj, &mt_layers[ 3].o_proj, &mt_layers[ 3].gate, &mt_layers[ 3].up, &mt_layers[ 3].down,
  &mt_layers[ 4].in_proj_qkv, &mt_layers[ 4].in_proj_z, &mt_layers[ 4].out_proj, &mt_layers[ 4].gate, &mt_layers[ 4].up, &mt_layers[ 4].down,
  &mt_layers[ 5].in_proj_qkv, &mt_layers[ 5].in_proj_z, &mt_layers[ 5].out_proj, &mt_layers[ 5].gate, &mt_layers[ 5].up, &mt_layers[ 5].down,
  &mt_layers[ 6].in_proj_qkv, &mt_layers[ 6].in_proj_z, &mt_layers[ 6].out_proj, &mt_layers[ 6].gate, &mt_layers[ 6].up, &mt_layers[ 6].down,
  &mt_layers[ 7].q_proj, &mt_layers[ 7].k_proj, &mt_layers[ 7].v_proj, &mt_layers[ 7].o_proj, &mt_layers[ 7].gate, &mt_layers[ 7].up, &mt_layers[ 7].down,
  &mt_layers[ 8].in_proj_qkv, &mt_layers[ 8].in_proj_z, &mt_layers[ 8].out_proj, &mt_layers[ 8].gate, &mt_layers[ 8].up, &mt_layers[ 8].down,
  &mt_layers[ 9].in_proj_qkv, &mt_layers[ 9].in_proj_z, &mt_layers[ 9].out_proj, &mt_layers[ 9].gate, &mt_layers[ 9].up, &mt_layers[ 9].down,
  &mt_layers[10].in_proj_qkv, &mt_layers[10].in_proj_z, &mt_layers[10].out_proj, &mt_layers[10].gate, &mt_layers[10].up, &mt_layers[10].down,
  &mt_layers[11].q_proj, &mt_layers[11].k_proj, &mt_layers[11].v_proj, &mt_layers[11].o_proj, &mt_layers[11].gate, &mt_layers[11].up, &mt_layers[11].down,
  &mt_layers[12].in_proj_qkv, &mt_layers[12].in_proj_z, &mt_layers[12].out_proj, &mt_layers[12].gate, &mt_layers[12].up, &mt_layers[12].down,
  &mt_layers[13].in_proj_qkv, &mt_layers[13].in_proj_z, &mt_layers[13].out_proj, &mt_layers[13].gate, &mt_layers[13].up, &mt_layers[13].down,
  &mt_layers[14].in_proj_qkv, &mt_layers[14].in_proj_z, &mt_layers[14].out_proj, &mt_layers[14].gate, &mt_layers[14].up, &mt_layers[14].down,
  &mt_layers[15].q_proj, &mt_layers[15].k_proj, &mt_layers[15].v_proj, &mt_layers[15].o_proj, &mt_layers[15].gate, &mt_layers[15].up, &mt_layers[15].down,
  &mt_layers[16].in_proj_qkv, &mt_layers[16].in_proj_z, &mt_layers[16].out_proj, &mt_layers[16].gate, &mt_layers[16].up, &mt_layers[16].down,
  &mt_layers[17].in_proj_qkv, &mt_layers[17].in_proj_z, &mt_layers[17].out_proj, &mt_layers[17].gate, &mt_layers[17].up, &mt_layers[17].down,
  &mt_layers[18].in_proj_qkv, &mt_layers[18].in_proj_z, &mt_layers[18].out_proj, &mt_layers[18].gate, &mt_layers[18].up, &mt_layers[18].down,
  &mt_layers[19].q_proj, &mt_layers[19].k_proj, &mt_layers[19].v_proj, &mt_layers[19].o_proj, &mt_layers[19].gate, &mt_layers[19].up, &mt_layers[19].down,
  &mt_layers[20].in_proj_qkv, &mt_layers[20].in_proj_z, &mt_layers[20].out_proj, &mt_layers[20].gate, &mt_layers[20].up, &mt_layers[20].down,
  &mt_layers[21].in_proj_qkv, &mt_layers[21].in_proj_z, &mt_layers[21].out_proj, &mt_layers[21].gate, &mt_layers[21].up, &mt_layers[21].down,
  &mt_layers[22].in_proj_qkv, &mt_layers[22].in_proj_z, &mt_layers[22].out_proj, &mt_layers[22].gate, &mt_layers[22].up, &mt_layers[22].down,
  &mt_layers[23].q_proj, &mt_layers[23].k_proj, &mt_layers[23].v_proj, &mt_layers[23].o_proj, &mt_layers[23].gate, &mt_layers[23].up, &mt_layers[23].down
};
typedef char mt_lora_pairs_must_be_MT_LORA_PAIRS
    [(sizeof mt_lora_pairs / sizeof mt_lora_pairs[0]) == (size_t)MT_LORA_PAIRS ? 1 : -1];

const char *const mt_embed_name = "language_model.embed_tokens.weight";    // [248320,1024] bf16
const char *const mt_final_norm_name = "language_model.norm.weight";       // [1024], zero-centred
const char *const mt_readout_file = "readout.safetensors";                // design §2.1, §3.2
const char *const mt_readout_name = "weight";                             // [255,1024]

const MtLayer *mt_layer(int i) { return (i >= 0 && i < MT_LAYERS) ? &mt_layers[i] : NULL; }

int mt_layer_full(int i) { return (i >= 0 && i < MT_LAYERS) ? mt_layers[i].full_attention : 0; }

int mt_lora_pair_count(void) { return MT_LORA_PAIRS; }

const MtLinear *mt_lora_pair(int k) { return (k >= 0 && k < MT_LORA_PAIRS) ? mt_lora_pairs[k] : NULL; }
