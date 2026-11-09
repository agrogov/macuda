# jefftrain: a C/CUDA LoRA trainer for Jeff on the macuda shim. Design

Status: implemented and corrected against the built tree. Companion: `jeff-trainer-plan.md` (the step-by-step coding
plan with test gates; its Appendix B carries the measured numbers). Written 2026-11-08; the sections marked
*as built* below were reconciled with `train/jeff/` and its `notes/` afterwards, because several of the original
estimates (the GDN backward's cost, the `.jtb` header's contents, the TF32 gate) did not survive contact.

## 1. What this is and why

Jeff (`mstrasser/jeff-base`, a Qwen3.5-0.8B text decoder plus a 255-way option readout) is fine-tuned with LoRA in
CLM-8B by `tools/jeff_lora/train.py` (PyTorch). That trainer runs on CUDA (RTX 5090 box `gpubox`, about 0.15 s per
batch of 8) and on MPS (about 10 h per epoch, since no Metal kernels exist for the Gated DeltaNet).

`jefftrain` is the same training run, written in the style of llm.c: one C host program and a handful of hand-written
CUDA kernels, with GEMMs through cuBLAS. Because it only needs the cudart and cuBLAS calls the shim already provides,
it runs on the Mac through the macuda shim (libtinycudart + libtinycublas + libtinynv) on the eGPU card. The same
source also builds with real nvcc and real cuBLAS on gpubox, which is how its numbers are checked.

Non-goals:

* No ggml, no llama.cpp, no GGUF. Weights are read straight from `model.safetensors`.
* No generation, no KV cache, no recurrent-state cache. Each row is one forward and one backward over a full prompt.
* No vision tower and no MTP head. Only `language_model.*` tensors are loaded.
* No training of anything but the LoRA factors and the readout. The backbone is frozen.
* No tokenizer in C. Python tokenizes once and writes a binary file (section 9).

### 1.1 The reference implementation is the spec

Where this document and `CLM-8B/tools/jeff_lora/{train.py,lora.py,prompt.py}` disagree, the Python is right and this
document is a bug. Every number jefftrain produces is compared with that trainer (section 10). The Python run on
gpubox (`tools/jeff_lora/cuda/`) is already validated against the Mac (fp32 max |dp| 0.0015, 160/160 argmax), so it is
a trustworthy oracle.

## 2. The training problem, exactly

One row = token ids `x[0..L-1]` (p50 about 520, p95 about 670, max 1026 in the full train split; never hard-code a maximum: size every
length-dependent buffer, RoPE table and checkpoint count from the .jtb header's `max_len`), an option count
`n` (1 to 255), a target distribution `t[0..n-1]` (sums to 1), and a row weight `w` (1.0 unless `--weights balanced`).

Forward for one row:

```
h0     = embed[x]                              # [L,1024], bf16 table -> fp32 activations (see section 6)
h      = 24 decoder layers (section 4)
hN     = final RMSNorm(h)                       # [L,1024]
z      = hN[L-1]                                # only the LAST position is read
logits = R z                                    # R: readout [255,1024], trained, fp32 master
logits[j] = -1e9 for j >= n                      # dead options
loss   = -sum_j t[j] * log_softmax(logits)[j]   # soft cross-entropy
```

Batch loss = `sum_i w_i * loss_i / sum_i w_i` over the batch's rows (with `--accum`, the sum of weights runs over the
whole accumulation group). Rows are right-padded in Python, but causal attention and the causal GDN mean a pad
never influences a real position. **jefftrain does not pad.** It processes rows packed back to back (section 7) and
gets the same numbers.

Gradient flows only from position L-1 back. Positions after the last one that influences L-1 do not exist, but every
position < L-1 does influence it (through attention and the recurrence), so the backward still covers all L positions.

### 2.1 Trainable parameters

LoRA on 150 linear layers, rank r=16, alpha=32, scale = alpha/r = 2.0:

| layers | module | W shape [out,in] | count |
|---|---|---|---|
| full attention (3,7,11,15,19,23) | q_proj | 4096 x 1024 | 6 |
| | k_proj | 512 x 1024 | 6 |
| | v_proj | 512 x 1024 | 6 |
| | o_proj | 1024 x 2048 | 6 |
| linear attention (the other 18) | in_proj_qkv | 6144 x 1024 | 18 |
| | in_proj_z | 2048 x 1024 | 18 |
| | out_proj | 1024 x 2048 | 18 |
| all 24 | mlp.gate_proj | 3584 x 1024 | 24 |
| | mlp.up_proj | 3584 x 1024 | 24 |
| | mlp.down_proj | 1024 x 3584 | 24 |

Not adapted: `in_proj_a`, `in_proj_b` (16 x 1024), the conv, norms, `A_log`, `dt_bias`, embeddings.

For each adapted layer: `y = W x + scale * B (A x)`, A is [r,in] fp32, B is [out,r] fp32. Init: A ~ kaiming_uniform with
a=sqrt(5), which is U(-1/sqrt(in), 1/sqrt(in)); B = 0. jefftrain never needs to reproduce torch's RNG: parity runs
start from an adapter file the Python wrote (`--init-adapter`, section 10).

Total trainable: 10,223,616 LoRA + 261,120 readout = 10,484,736 (matches the Python log line).

Readout R: [255,1024], stored bf16 in `readout.safetensors`, trained in fp32, no bias.

### 2.2 Optimizer, word for word

* AdamW, betas (0.9, 0.999), eps 1e-8, weight_decay 0 (so it is Adam), torch's default non-amsgrad formula:
  `m = b1 m + (1-b1) g; v = b2 v + (1-b2) g^2; mhat = m/(1-b1^t); vhat = v/(1-b2^t); p -= lr * mhat / (sqrt(vhat) + eps)`
  with t starting at 1.
* Two groups: all LoRA A and B at `--lr` (2e-4), the readout at `--readout-lr` (5e-6).
* Gradient clipping before the step: global L2 norm over ALL trainable tensors (LoRA and readout together), clip to 1.0:
  `coef = 1.0 / (norm + 1e-6); if coef < 1: g *= coef`.
* Schedule, applied as a multiplier on both groups' base lr; s is the number of optimizer steps already taken (0 for
  the first step):
  `f(s) = min(1, (s+1)/max(1, int(total*0.03))) * (0.1 + 0.9*0.5*(1 + cos(pi*min(s,total)/total)))`.
  `total = ceil(rows/batch/accum) * epochs`, or `min(max_steps, that)`.
* The dev check (soft CE on the dev split, at T=1) runs at step 0 and every `--eval-every` steps. The best dev CE is
  saved. At the end the best adapter is reloaded and the `--eval` splits are scored.

### 2.3 Batch order

`train.py`'s `batches()` uses Python's `random.Random(seed)`, so its exact order can't be reproduced cheaply in C. The
order is therefore made in Python and handed over: the prep script (section 9) writes the epoch's batch list. jefftrain
consumes batches in file order. That gives bit-for-bit the same rows per step as the Python run.

## 3. The model, equation by equation

Source of truth: `transformers/models/qwen3_5/modeling_qwen3_5.py` (transformers 5.17.0, in CLM-8B's `.venv`). Line
references below are to that file. Notation: one row of length L; `x` is [L,1024] row-major (token-major), fp32.

### 3.1 Config (from `config.json` → `text_config`)

| name | value |
|---|---|
| hidden D | 1024 |
| layers | 24; layer i is full attention if i % 4 == 3 (3,7,11,15,19,23), else linear attention (GDN) |
| MLP intermediate F | 3584, act silu |
| full attention | 8 query heads, 2 KV heads (GQA group 4), head_dim 256, output gate, q_norm/k_norm on head_dim |
| RoPE | theta 1e7, partial 0.25 → rotary dim 64 (first 64 of 256 dims); rotate_half convention |
| GDN | 16 key heads, 16 value heads, head dims 128/128, conv kernel 4 (depthwise, causal, no bias) |
| rms_norm_eps | 1e-6 (every RMSNorm, including q_norm, k_norm and the gated norm) |
| vocab | 248320 (only the embedding table is used; the LM head is never computed) |

### 3.2 Tensors loaded (prefix `language_model.`; all BF16 in `model.safetensors`)

```
embed_tokens.weight                 [248320,1024]
norm.weight                         [1024]                 final norm, zero-centred (see 3.3)
layers.{i}.input_layernorm.weight   [1024]                 zero-centred
layers.{i}.post_attention_layernorm.weight [1024]          zero-centred
layers.{i}.mlp.{gate,up}_proj.weight [3584,1024]; down_proj.weight [1024,3584]
-- full attention layers i in {3,7,...,23}:
layers.{i}.self_attn.q_proj.weight  [4096,1024]            per head h: rows [h*512, h*512+256) = query, [h*512+256, h*512+512) = gate
layers.{i}.self_attn.k_proj.weight  [512,1024]   v_proj.weight [512,1024]   o_proj.weight [1024,2048]
layers.{i}.self_attn.q_norm.weight  [256]   k_norm.weight [256]             zero-centred
-- GDN layers:
layers.{i}.linear_attn.in_proj_qkv.weight [6144,1024]      output channels: [q 0..2047 | k 2048..4095 | v 4096..6143], head h of q = q[h*128..]
layers.{i}.linear_attn.in_proj_z.weight   [2048,1024]
layers.{i}.linear_attn.in_proj_a.weight   [16,1024]   in_proj_b.weight [16,1024]
layers.{i}.linear_attn.conv1d.weight      [6144,1,4]       tap order: weight[c,0,0] multiplies x[t-3], ..., weight[c,0,3] multiplies x[t]
layers.{i}.linear_attn.A_log  [16]   dt_bias [16]
layers.{i}.linear_attn.norm.weight        [128]            NOT zero-centred (plain multiply, init ones)
```

Plus `readout.safetensors`: `weight` [255,1024] BF16. Everything else in the file (`visual.*`) is skipped — and note
that the checkpoint this was checked against has **no `mtp.*` tensor at all** (473 tensors = 320 text + 153 visual +
the readout); the skip rule is right, the mention of `mtp.*` is not backed by the file. The safetensors reader is
small in code but not naive: an 8-byte little-endian header length, a JSON header (`dtype`, `shape`, `data_offsets`
relative to the end of the header), then raw little-endian data. mmap the file, parse the JSON with a tiny hand-written
scanner (the header is flat: name → {dtype, shape, data_offsets}), and look up by name. *As built* `src/st.c` is ~800
lines, not "about 80": it validates rather than trusts (dtype, `data_offsets` against the shape, duplicate names, junk
after the JSON, shard completeness, one message per failure naming both sides), which is what the loader needs. Its
shape array is `ST_MAX_DIMS 8`, not 4, because `visual.patch_embed.proj.weight` is `[768,3,2,16,16]` and the reader
registers every tensor in the header (that is what makes `--list` show `visual.*`).

### 3.3 Building blocks

RMSNorm, zero-centred (all `Qwen3_5RMSNorm`, l.841): over the last dim of size d,
```
r = 1 / sqrt(mean(x^2) + 1e-6);   y = x * r * (1 + w)
```
Gated RMSNorm (GDN `norm`, l.218), per GDN head, d=128, with gate z of the same shape:
```
r = 1 / sqrt(mean(x^2) + 1e-6);   y = w * (x * r) * silu(z)
```
L2 norm (l2norm, used on GDN q and k per head, d=128): `y = x / sqrt(sum(x^2) + 1e-6)`. Note: sum, not mean.

silu(u) = u * sigmoid(u). softplus(u) = log(1 + exp(u)) (torch uses `u` itself when u > 20; do the same).

RoPE on the first 64 dims of a 256-dim head, position p (0-based within the row):
```
inv_freq[k] = 1e7^(-2k/64), k = 0..31
ang[k] = p * inv_freq[k];  c[j] = cos(ang[j mod 32]),  s[j] = sin(ang[j mod 32]) for j = 0..63
y[j]    = x[j]*c[j] - x[j+32]*s[j]        j = 0..31
y[j+32] = x[j+32]*c[j] + x[j]*s[j]        j = 0..31
y[j] = x[j]                               j = 64..255  (pass-through)
```
The model's mRoPE sections [11,11,10] with `mrope_interleaved` collapse to this for text: all three position axes
carry the same p, so each frequency slot gets the same angle whichever axis it is assigned to. Compute `inv_freq` in
fp32 as torch does (`1.0 / (theta ** (arange(0,64,2).float() / 64))`), and ang = p * inv_freq in fp32, then cos/sin in
fp32. Precompute a [Lmax,64] cos table and sin table once.

### 3.4 Decoder layer (l.861)

```
a  = RMSNorm(x; input_layernorm)
x  = x + Mixer(a)                 # Mixer = FullAttention or GDN
m  = RMSNorm(x; post_attention_layernorm)
x  = x + down( silu(gate(m)) * up(m) )
```
After 24 layers: `hN = RMSNorm(x; norm)`.

### 3.5 Full attention mixer (l.749)

```
qg = q_proj(a)                         # [L,4096]; per head h: q_h = qg[:, h*512 : h*512+256], gate_h = qg[:, h*512+256 : h*512+512]
q_h = RMSNorm(q_h; q_norm)             # per head, d=256, zero-centred
k_g = RMSNorm(k_proj(a)[:, g*256:(g+1)*256]; k_norm)   g = 0,1
v_g = v_proj(a)[:, g*256:(g+1)*256]
q_h, k_g = RoPE(q_h, p), RoPE(k_g, p)
o_h = softmax_causal(q_h k_{h/4}^T / 16) v_{h/4}        # scaling 256^-0.5 = 1/16; query head h uses KV group h/4 (integer division)
o   = concat_h(o_h)                    # [L,2048], head-major: o[:, h*256 + j]
o   = o * sigmoid(concat_h(gate_h))    # gate laid out the same way, [L,2048]
out = o_proj(o)
```
The q/gate split is per head and interleaved: torch views q_proj's output as [L, 8, 512] and chunks the last dim. It is
NOT "first 2048 = queries, last 2048 = gates". Getting this wrong still trains, and parity is the only thing that
catches it.

### 3.6 GDN mixer (l.~560-640)

```
qkv = in_proj_qkv(a)                   # [L,6144]
z   = in_proj_z(a)                     # [L,2048] → [L,16,128]
bb  = in_proj_b(a)                     # [L,16]
aa  = in_proj_a(a)                     # [L,16]
c   = causal depthwise conv over time, per channel ch of 6144:
      c[t,ch] = sum_{k=0..3} wconv[ch,k] * qkv[t-3+k, ch]     (qkv[<0] = 0)
c   = silu(c)
q = c[:, 0:2048], k = c[:, 2048:4096], v = c[:, 4096:6144]   each reshaped to [L,16,128]
beta = sigmoid(bb)                                          [L,16]
g    = -exp(A_log) * softplus(aa + dt_bias)                 [L,16], fp32 (A_log and dt_bias read as fp32 from bf16)
q = l2norm(q) / sqrt(128);  k = l2norm(k)                   per head, fp32
-- recurrence, per head, S is [128 (key i), 128 (value j)], S_0 = 0:
for t in 0..L-1:
    S        = S * exp(g[t])
    kv[j]    = sum_i S[i,j] * k[t,i]
    delta[j] = (v[t,j] - kv[j]) * beta[t]
    S[i,j]  += k[t,i] * delta[j]
    o[t,j]   = sum_i S[i,j] * q[t,i]
-- output:
y   = GatedRMSNorm(o; norm.weight, gate = z)    per head, d=128
out = out_proj(y.reshape(L,2048))
```
`num_v_heads == num_k_heads` (16 == 16), so there is no repeat_interleave. The recurrence is exactly
`torch_recurrent_gated_delta_rule`. The fla chunked kernel Python uses on CUDA computes the same function in a
different order; differences are rounding only.

Each value column j of S evolves independently in the forward: kv[j], delta[j] and o[t,j] only touch column j. The
kernel design in section 8 is built on that.

## 4. Backward, equation by equation

Notation: `dX` is dLoss/dX. Every backward below is checked by a finite-difference gradcheck of its CPU reference in double precision (plan, M2), so
an error here is caught early. Only the gradients that are needed are computed: activations' gradients flow all the
way to layer 0's input; frozen weights get no gradient. LoRA A/B and the readout get one.

### 4.1 Head (loss, readout, final norm)

```
p = softmax(logits) over the n live options (dead ones are -1e9, so their p is exactly 0 in fp32)
dlogits[j] = (w_row / W_group) * (p[j] * sum(t) - t[j])     with sum(t) = 1 → (w/W) * (p[j] - t[j]);   j >= n → 0
dR   += dlogits ⊗ z                       # [255,1024], fp32
dz    = R^T dlogits                       # [1024]
dhN[L-1] = dz;  dhN[t] = 0 for t < L-1
```
`W_group` = sum of row weights over the accumulation group (`gw` in train.py). With weights none and accum 1 it is
the batch size.

### 4.2 Linear with LoRA

Forward `y = x W^T + s * (x A^T) B^T`, x [N,in], W [out,in], A [r,in], B [out,r], s = 2.0. Keep `u = x A^T` [N,r].
```
dx  = dy W + s * (dy B) A          # [N,in]
dB += s * dy^T u                   # [out,r]
dA += s * (dy B)^T x               # [r,in]
```
Write `e = s * (dy B)` [N,r] once and reuse it: `dx = dy W + e A`, `dA += e^T x`.

### 4.3 RMSNorm (zero-centred, weight frozen)

`y = x r (1+w)`, `r = (mean(x^2)+eps)^-1/2`, d = row length. With `gy = dy * (1+w)`:
```
dx = r * (gy - x * r^2 * mean(gy * x))
```
(the same formula serves q_norm/k_norm with d=256 and the final norm with d=1024). Save r per row in the forward.

### 4.4 Gated RMSNorm (GDN), per head, d=128

`n = x r`, `y = w * n * silu(z)`. Let `sg = silu(z)`, `dsilu(z) = sig(z) * (1 + z (1 - sig(z)))`:
```
dz = dy * w * n * dsilu(z)
gn = dy * w * sg
dx = r * (gn - n * mean(gn * n))
```

### 4.5 MLP

`h1 = gate(m), h2 = up(m), act = silu(h1) * h2, out = down(act)`:
```
dact = dout (through down, 4.2)
dh1  = dact * h2 * dsilu(h1)
dh2  = dact * silu(h1)
dm   = (dh1 through gate) + (dh2 through up)        both via 4.2, summed
```

### 4.6 Residuals

`x_out = x_in + f(norm(x_in))` → `dx_in = dx_out + dnorm_backward(df_input)`. Accumulate into the residual gradient
buffer; never overwrite it.

### 4.7 Full attention

Forward per query head h (group g = h/4), causal over positions s ≤ t, scale c = 1/16:
```
P[t,s] = softmax_s(c * q_t · k_s);  o_t = sum_s P[t,s] v_s
```
Backward (standard):
```
dP[t,s] = do_t · v_s
dS[t,s] = P[t,s] * (dP[t,s] - sum_s' P[t,s'] dP[t,s'])      (= P * (dP - do_t·o_t))
dq_t   = c * sum_s dS[t,s] k_s
dk_s  += c * sum_t dS[t,s] q_t     (summed over the 4 query heads of the group)
dv_s  += sum_t P[t,s] do_t         (summed over the 4 query heads of the group)
```
Then RoPE backward (the rotation transposed: same formula with s → -s), then q_norm/k_norm backward (4.3), then the
output gate: `o_gated = o * sig(gate)` → `do = dog * sig(gate)`, `dgate = dog * o * sig(gate) * (1 - sig(gate))`,
then scatter dq and dgate back into the interleaved [L,4096] layout and run 4.2 for q_proj; k_proj, v_proj likewise.

With L about 1000 (max 1026) and 6 attention layers the simple route is fine: materialise P per (row, head) as [L,L] fp32 (4 MB at
L=1024) with GEMMs (section 5) and do the softmax and dS in small kernels. No flash attention is needed.

### 4.8 GDN

Reverse the forward of 3.6 step by step.

Output side: `y = GatedRMSNorm(o, z)` → do, dz (4.4); `out_proj` via 4.2.

Recurrence backward, per head, with dS (128x128, fp32) starting at 0 after the last token. For t = L-1 down to 0,
with S_prev = S_{t-1} (S_{-1} = 0), and recomputing the forward step's values:
```
Sd       = S_prev * exp(g_t)                       # decayed
kv[j]    = sum_i Sd[i,j] k_t[i]
delta[j] = (v_t[j] - kv[j]) * beta_t
S_t      = Sd + k_t ⊗ delta
-- o_t = S_t^T q_t:
dq_t[i]  = sum_j S_t[i,j] do_t[j]
dS[i,j] += q_t[i] do_t[j]
-- S_t = Sd + k_t ⊗ delta:
dk_t[i]  = sum_j dS[i,j] delta[j]
ddelta[j]= sum_i dS[i,j] k_t[i]
-- delta = (v - kv) * beta:
dv_t[j]  = ddelta[j] * beta_t
dkv[j]   = -ddelta[j] * beta_t
dbeta_t  = sum_j ddelta[j] (v_t[j] - kv[j])
-- kv = Sd^T k_t:   (dS here is dSd, which equals dS_t at this point plus this term)
dS[i,j] += k_t[i] dkv[j]
dk_t[i] += sum_j Sd[i,j] dkv[j]
-- Sd = S_prev * exp(g_t):
dg_t     = sum_{i,j} dS[i,j] * Sd[i,j]
dS       = dS * exp(g_t)                            # now dS_{t-1}
```
The order matters: dk_t's first term and ddelta use dS before the `k ⊗ dkv` term is added, and dg_t uses dS after it.

Then upstream of the recurrence:
```
q = l2n(qc) * c128, c128 = 1/sqrt(128):  dqn = dq * c128;  dqc = rq * (dqn - qn * (qn · dqn))   (rq = 1/sqrt(sum qc^2 + eps), qn = qc*rq)
k = l2n(kc):                              dkc = rk * (dk - kn * (kn · dk))
dc = concat(dqc, dkc, dv)                 [L,6144]  (gradient of the post-silu conv output)
dpre = dc * dsilu(pre)                    pre = conv output before silu
dqkv[t,ch] = sum_{k=0..3} wconv[ch,k] * dpre[t+3-k, ch]     (dpre[>=L] = 0; the transposed causal conv)
beta = sig(bb):   dbb = dbeta * beta * (1-beta)
g = -exp(A_log) * softplus(aa + dt_bias):   daa = dg * (-exp(A_log)) * sig(aa + dt_bias)
da = (dqkv through in_proj_qkv, 4.2) + (dz through in_proj_z, 4.2) + dbb W_b + daa W_a     (W_a, W_b frozen, no LoRA)
```

## 5. GEMMs

All dense math goes through `cublasGemmStridedBatchedEx` / `cublasGemmEx` (both are in libtinycublas, and in real
cuBLAS). Row-major C convention: cuBLAS is column-major, so a row-major `C[M,N] = A[M,K] B[K,N]` is issued as the
column-major `C^T = B^T A^T`. Wrap that once, in `gemm.c`, as:

```c
// Row-major. C[M,N] = alpha * op(A) op(B) + beta * C.  ta/tb: 0 = as stored, 1 = transposed.
// A is stored [M,K] (ta=0) or [K,M] (ta=1); B is stored [K,N] (tb=0) or [N,K] (tb=1). All fp32, leading dim = stored row length.
void gemm(cublasHandle_t h, int ta, int tb, int M, int N, int K, float alpha,
          const float *A, const float *B, float beta, float *C);
// the same, batched with strides (elements) sA, sB, sC; used for attention heads
void gemm_batched(cublasHandle_t h, int ta, int tb, int M, int N, int K, float alpha,
          const float *A, long sA, const float *B, long sB, float beta, float *C, long sC, int batch);
```
Implementation: `cublasGemmEx(h, tb?T:N, ta?T:N, N, M, K, &alpha, B, CUDA_R_32F, tb?K:N, A, CUDA_R_32F, ta?M:K,
&beta, C, CUDA_R_32F, N, COMPUTE, CUBLAS_GEMM_DEFAULT)`. Unit-test it against a CPU triple loop for all four
(ta,tb) combinations with odd sizes before anything else uses it (plan, M1).

COMPUTE is `CUBLAS_COMPUTE_32F_FAST_TF32` by default and `CUBLAS_COMPUTE_32F` when built with `-DJT_EXACT=1` or run with `JT_EXACT=1` in the environment, as the shim checkpoints do (for
gradchecks). libtinycublas accepts both for fp32 A/B (cublas.c's type contract; `test/cublas_contract.c` checks it).
On the shim, `TINYCUBLAS_TC=0` in the environment additionally forces the non-tensor-core path; use it for gradchecks.

Per layer, with N = total tokens in the batch (about 4200 for 8 rows):

| use | op | M x N x K | fwd/bwd |
|---|---|---|---|
| linear y = x W^T | gemm(0,1) | N x out x in | fwd |
| LoRA u = x A^T | gemm(0,1) | N x 16 x in | fwd |
| LoRA y += s u B^T | gemm(0,1), alpha=s, beta=1 | N x out x 16 | fwd |
| dx = dy W | gemm(0,0) | N x in x out | bwd |
| e = s dy B | gemm(0,0), alpha=s | N x 16 x out | bwd |
| dx += e A | gemm(0,0), beta=1 | N x in x 16 | bwd |
| dB += s dy^T u | gemm(1,0), alpha=s, beta=1 | out x 16 x N | bwd |
| dA += e^T x | gemm(1,0), beta=1 | 16 x in x N | bwd |
| attention scores | batched gemm(0,1), per (row, head) | L x L x 256 | fwd+bwd |
| attention P V, dP, dq, dk, dv | batched, per (row, head) | L x 256 x L etc. | fwd+bwd |

Frozen W is used for both y = x W^T and dx = dy W, so the backward never needs W^T stored separately.

Attention GEMMs run per row (rows have different L) with `gemm_batched` over the 8 query heads. For K and V, which
have 2 groups shared by 4 heads each, either expand K/V to 8 heads first (simple: an [L,2048] copy; do this in v1),
or call the batched GEMM twice with stride 0 tricks (not needed).

## 6. Precision and memory

### 6.1 v1 precision (the one to build first)

* Frozen weights: converted from bf16 to fp32 once at load, kept on the card in fp32. Non-embedding text weights
  are about 0.5 B parameters = about 2.0 GB fp32. The embedding table stays bf16 (508 MB) and is gathered with a
  bf16→fp32 conversion kernel.
* Activations, gradients, LoRA factors, readout, Adam state: fp32.
* GEMMs: TF32 tensor cores (6.1's COMPUTE), fp32 accumulate.
* Reductions inside kernels (norms, softmax, recurrence): fp32.

Why fp32 and not bf16 activations: the bf16 fast path in libtinycublas needs contiguous extents, leading dims and
strides that are multiples of 8 and 16-byte-aligned pointers (cublas.c:145). Most shapes here qualify, but the LoRA
rank-16 GEMMs and per-row attention GEMMs with arbitrary L do not always, and cublasGemmEx needs A and B of the same
type. fp32+TF32 removes all of those cases. It also makes the Python fp32 run the parity reference, which agrees
with the Mac to 0.0015 in probability. bf16 activations are a later optimisation (M8), not v1.

How it relates to the Python CUDA run: Python runs under bf16 autocast, so its GEMMs (LoRA ones included) round inputs
to bf16. jefftrain v1 is closer to the fp32 reference than the Python bf16 run is. Parity is measured against both
(section 10), each with its own threshold.

**As built, TF32 is the speed mode, not the parity mode.** Every activation and gradient gate (10.2, M5.1, M6.2, M7.3)
runs with `JT_EXACT=1 TINYCUBLAS_TC=0`, because this model's 24-layer backward amplifies a relative perturbation by
about 1e21: a 1e-6 nudge to one entry of the last layer's incoming gradient reaches layer 0 at O(1e20) while the true
gradient rides a stable direction, so in TF32 the early layers' gradients are noise by construction (the same 403-key
dump gives `87 ok, 316 failed`). A 1e-4-level difference between jefftrain and a reference is therefore only meaningful
in exact mode; TF32 is what M6.3's timing, M6.4's arm and the card checkpoints' m1/m2/m3 steps use, and gradient
clipping and Adam inherit the noise if a training run is done in it.

### 6.2 Memory plan (target: fits a 12 GB card)

Layer-level recompute (the same thing as `--grad-checkpoint`): the forward keeps only each layer's input x (fp32
[N,1024]) and runs the layer with scratch buffers. The backward, for layer i from 23 down to 0, re-runs layer i's
forward from its saved input into scratch, then runs its backward. Cost: one extra forward, about +33% time. Memory:

| item | size |
|---|---|
| weights fp32 + embedding bf16 | about 2.5 GB |
| LoRA A,B + grads + Adam m,v (fp32) | 10.5 M x 4 x 4 B = 168 MB |
| saved layer inputs: 25 x N x 1024 x 4 B, N = 8 rows x 1000 worst case | 820 MB |
| one layer's scratch: about 40 K floats/token of forward activations (a GDN layer plus its MLP) and about as many gradients, reused across layers | N = 8000: about 2.5 GB |
| attention P and dP per row per head, done row by row: 2 x 8 x L^2 x 4 B, L = 1024 | 64 MB |
| GDN checkpoints (section 8.3) | B x 16 heads x 16 segments x 64 KB = 128 MB at B = 8 |
| GDN backward segment scratch `sp` (section 8.3) | B x 16 x 4 tiles x 1 MB = 512 MB at B = 8 |

Total about 7 GB at batch 8 in the worst case (8 rows of 1000 tokens), about 5 GB at typical lengths. If memory is short, lower the batch or the segment length. Never call cudaMalloc inside
the step loop: allocate everything once at start-up, sized for the largest batch in the batch file (`max_tokens` from
its header, section 9), and reuse.

## 7. Packed batches (no padding)

A batch of B rows with lengths L_0..L_{B-1} is stored as one token-major [N,1024] matrix, N = sum L_b, rows back to
back. A small host array `off[b]` (prefix sums, off[B] = N) is copied to the device every step.

* Every per-token op (linears, norms, MLP, GDN projections, embeddings) runs on all N tokens at once, ignoring row
  boundaries. That is where nearly all the FLOPs are.
* Ops that mix positions must respect boundaries:
  * The GDN conv: tap t-3+k must not cross into the previous row. Use `t_local = t - off[b]`, and treat
    `t_local - 3 + k < 0` as zero. The kernel finds b from t by a binary search over `off` (B ≤ 64) or from a
    precomputed `row_of[N]` int array (simpler; do that).
  * The GDN recurrence: S restarts at 0 at each row's first token. The kernel grid has a row index b and runs t over
    [off[b], off[b+1]).
  * Attention: done per row (`gemm_batched` on the row's slice), so the mask is just causal within the row.
  * RoPE positions are `t_local` (0-based within the row), never the packed index.
  * The q/k/v slices of the conv output are **copied**, not viewed: `kernels/norm.h`'s l2norm takes `rows, d` and
    `kernels/gdn.h`'s `gdn_fwd` indexes `((t*H) + h)*D`, and neither takes a row pitch, so `layer_gdn.c` gets its three
    contiguous `[n][2048]` slices from `jt_to_head_major(…, heads = 3, head_dim = 2048)` — one pass over the same
    `3·N·2048` floats the l2norm and the recurrence read anyway. A strided view of `[N][6144]` would need a pitch
    parameter in both kernels.
* The head reads token `off[b+1]-1` for row b.

Python's right-padding gives identical numbers: pads sit after the last real token, and causality means no real
token sees them.

## 8. Kernels

All kernels live in `train/jeff/kernels/*.cu`, are `extern "C"`, take plain pointers and sizes, and are launched
with `<<<grid, block, smem, stream>>>` from small host wrappers in the same file (`void k_rmsnorm_fwd(...)` etc.).
Every kernel has a CPU reference in `train/jeff/ref/` (plain C, fp32) and a unit test that compares them on random
inputs (plan, M1/M2). Block sizes: 256 threads unless stated. One CUDA stream. No `__constant__` memory (there is no
cudaMemcpyToSymbol in the shim; pass tables as pointers).

### 8.1 Elementwise and row kernels (simple)

| kernel | what | launch |
|---|---|---|
| embed_gather | out[t,:] = float(bf16 table[id[t],:]) | grid N, block 256 |
| bf16_to_f32 | weight conversion at load (as built: `jt_bf16_to_f32_n`, `_n` because the test harness owns a scalar `jt_bf16_to_f32`) | 1-D grid-stride |
| rmsnorm_fwd(d, zero_centred) | y and r per row (r saved) | one block per row for d > 256; one warp per row for d ≤ 256 (q_norm/k_norm) |
| rmsnorm_bwd | 4.3 (**one** row reduction, `mean(gy*x)`: `r` is taken as an input, not recomputed) | one block per row |
| gated_rmsnorm_fwd/bwd | 4.4, rows = N x 16 heads, d = 128 | one warp per row, 8 warps (256 threads) per block, grid = ceil(rows/8) |
| l2norm_fwd/bwd | rows = N x 16, d = 128; saves r | same shape as gated_rmsnorm |
| silu_mul_fwd/bwd | MLP gating | grid-stride |
| sigmoid_gate_fwd/bwd | attention output gate, interleaved layout (3.5) | grid-stride |
| rope_fwd/bwd | first 64 dims of each 256-dim head; positions from row_of/off | one thread per (token, head, j<32) |
| qgate_split / qgate_merge | [N,4096] interleaved ↔ q [N,2048] + gate [N,2048] | grid-stride |
| kv_expand / kv_reduce | [N,512] → [N,2048] (each group copied to 4 heads) and the sum back | grid-stride |
| conv_fwd/bwd | 3.6 / 4.8, causal depthwise k=4, per row; fwd also saves pre-silu | one thread per (token, channel) |
| gdn_gates_fwd/bwd | beta, g from bb, aa; their gradients | one thread per (token, head) |
| attn_softmax_fwd/bwd | causal, per (head, query t) row of length L, scale 1/16 applied inside; writes 0 above the diagonal | one block per (head, t) |
| ce_head | 4.1 for B rows: softmax over n live options, loss, dlogits | one block per row |
| add_inplace, scale, zero | residuals and bookkeeping | grid-stride |
| adamw_step | 2.2, one launch over a flat parameter buffer with a per-element lr group | grid-stride |
| sumsq | for the global grad norm: `sumsq_partial` + `sumsq_final` over a fixed 1024 blocks (deterministic); returns the **sum of squares**, the caller takes the root | |

The row-structured ops (`qgate_*`, `kv_*`, `to/from_head_major`) launch one block per row (or per head) *strided over
rows* rather than as a flat grid-stride: inside a row the block walks heads and columns with `for` loops, so no thread
divides an index and a large packed batch cannot overrun the grid's x limit. Every wrapper is a no-op for a zero
dimension, so no caller special-cases an empty slice.

Keep all LoRA A, B and the readout in ONE flat fp32 buffer (and grads, m, v in three more of the same layout), with a
host-side table of offsets. That makes zeroing, the grad norm, clipping and Adam each one launch, and saving a flat
copy.

Reductions use warp shuffles (`__shfl_xor_sync(0xffffffff, v, o)`, o = 16,8,4,2,1) then shared memory across warps.
Always fp32. Never `atomicAdd` on floats for anything compared with a reference: the order changes the rounding.
Deterministic two-pass reductions only.

### 8.2 Attention, per row

For row b (slice [off_b, off_b+L)), after q (8 heads), k, v (expanded to 8 heads) are in head-major scratch
[8][L][256] (a transpose kernel from token-major, or write them that way in rope_fwd):
```
Sc[h] = Q[h] K[h]^T                         gemm_batched(0,1) L x L x 256, batch 8
P[h]  = causal_softmax(Sc[h] / 16)           attn_softmax_fwd (masked entries = 0)
O[h]  = P[h] V[h]                            gemm_batched(0,0) L x 256 x L
-- backward
dP[h] = dO[h] V[h]^T                         gemm_batched(0,1)
dV[h] = P[h]^T dO[h]                         gemm_batched(1,0)
dSc   = attn_softmax_bwd(P, dP) / 16         dS = P*(dP - rowsum(P*dP)), times 1/16
dQ[h] = dSc[h] K[h]                          gemm_batched(0,0)
dK[h] = dSc[h]^T Q[h]                        gemm_batched(1,0)
```
P is kept from the forward of the same (recomputed) layer. Buffers are sized for the longest row and reused row by row.
Then dK, dV go through kv_reduce (sum of 4 heads per group) into [L,512].

### 8.3 GDN recurrence: the one hard kernel

Everything else is textbook. This one needs care, so it gets two implementations, both kept forever:

1. `gdn_ref` in `ref/` (CPU, fp32, the loops of 3.6 and 4.8 written literally). This is the oracle.
2. `gdn_fwd` / `gdn_bwd` CUDA kernels.

Inputs in [N,16,128] (q already l2-normed and scaled, k l2-normed, v) and [N,16] (g, beta), all fp32.

**Forward kernel `gdn_fwd`.** Grid (B rows, 16 heads, 4 column tiles); block 32 threads (one warp). Lane l owns value
column j = tile*32 + l and keeps that column of S, `float s[128]`, in registers. Loop with `#pragma unroll` over i
so the array stays in registers; check `-Xptxas -v` / cuobjdump for zero spill. *As built* one unrolled 128-iteration
loop does **not** stay in registers (255 registers, 68 B spill stores / 52 B loads: ptxas prefetches all 128 shared
loads of the sum); each 128-iteration loop is split into **two 64-iteration halves with a `__syncwarp()` between**,
which is a scheduling barrier ptxas cannot move loads across and costs only the instruction. That is **201 registers,
0 bytes spill stores, 0 bytes spill loads, 1024 B smem**, with the sums still running i = 0…127 ascending, so the
arithmetic is unchanged. Per token t of the row:
```
smem k_s[128], q_s[128] ← k[t,h,:], q[t,h,:]   (each lane loads 4 of each; __syncwarp())
e = expf(g[t,h]);  kv = 0
for i: s[i] *= e; kv += s[i]*k_s[i]
d = (v[t,h,j] - kv) * beta[t,h]
o = 0
for i: s[i] += k_s[i]*d; o += s[i]*q_s[i]
out[t,h,j] = o
```
No cross-lane reduction at all, because columns are independent (3.6). Checkpoint: when t_local % 64 == 0, before
the step, write s[] into `ckpt[b][h][t_local/64][i][j]`. Checkpoints per (b,h) = ceil(max_len/64), 64 KB each: 17 at max_len 1026.
*As built* the buffer is the **caller's** — `gdn_fwd(…, int nseg, float *ckpt, float *o, cudaStream_t s)` — because
the backward is a separate call that reads it and there is no global scratch; `kernels/gdn.h`'s `gdn_ckpt_segs()` /
`gdn_ckpt_floats()` are the only place its size is written down, and `ckpt == NULL` (or `nseg <= 0`) means
"forward only, no stores". At B = 8 / N = 5094 it is 142.6 MB and costs nothing measurable (1.26 ms with, 1.25–1.33 ms
without). The forward's measured cost is **1.26 ms** for that batch, so the "about 2 ms per layer for fwd + bwd"
budget below leaves the backward 0.7 ms, which it does not meet — see the correction at the end of this section.

**Backward kernel `gdn_bwd`.** Grid (B, 16, 4 tiles); block 64 threads = 2 warps. Warp w owns rows
i in [64w, 64w+64); lane l owns column j = tile*32 + l. Each thread holds `float ds[64]` (its piece of dS) across
the whole loop. Segments are processed last to first; inside each:

Phase A, recompute: start from `ckpt[b][h][seg]`, run the forward step for the segment's tokens, and before each
step write the thread's 64 values of S_prev into a per-block scratch `sp[tl][64 rows][32 cols]` (global memory, 1 MB
per block). kv needs a sum over all 128 rows, so each warp computes its 64-row partial, writes it to shared
`kvp[w][32]`, `__syncthreads()`, and both add `kvp[0][l] + kvp[1][l]` (in that fixed order).

Phase B, reverse steps within the segment, t from last to first:
```
load sp_i = S_prev[i,j] for my 64 rows;   e = expf(g_t)
sd_i = sp_i * e
kv_j = cross-warp sum over all rows of sd_i * k_i            (kvp exchange, as in phase A)
delta_j = (v_tj - kv_j) * beta_t
st_i = sd_i + k_i * delta_j
dq_i  = sum over the 32 lanes of st_i * do_tj                  (per row i: warp butterfly over lanes)
ds_i += q_i * do_tj
ddelta_j = cross-warp sum of sum_i ds_i * k_i
dv_tj   = ddelta_j * beta_t                                     (warp 0 writes)
dkv_j   = -ddelta_j * beta_t
dbeta contribution = ddelta_j * (v_tj - kv_j)                   (summed over the 32 lanes; warp 0 writes the tile's partial)
dk_i  = sum over lanes of (ds_i * delta_j + sd_i * dkv_j)       (one butterfly; ds_i BEFORE the next line)
ds_i += k_i * dkv_j
dg contribution = sum_i ds_i * sd_i  → sum over lanes and both warps → the tile's partial
ds_i *= e
```
The "sum over lanes" for a vector of 64 rows: for each of the 64 values, 5 `__shfl_xor_sync` steps; lane 0 then
holds all 64 sums (every lane does, in fact). Lane 0 of warp w writes dq rows [64w, 64w+64). That is 640 shuffles per
token per thread, well within budget (the estimate here was "about 2 ms per layer on a 5090 at batch 8"; the measured
cost is 11.6 ms, and the reason is this section's `sp` traffic — see the correction at the end of 8.3).

Output partials: dq_part, dk_part [4 tiles][N][16][128]; dbeta_part, dg_part [4][N][16]. A reduce kernel adds the 4
tiles in order 0,1,2,3, so the result is deterministic. dv needs no reduction (each column belongs to one tile).

Correctness gates for the kernel (plan, M3): fwd vs gdn_ref on random inputs, L = 1, 2, 63, 64, 65, 200, B = 3 with
different lengths, max abs error < 1e-4 relative to the output's max; bwd vs gdn_ref the same; gdn_ref's backward vs
finite differences in double precision on L = 5, dims shrunk to 8 (the ref takes dims as parameters for exactly this).

**As built, and the cost correction.** `gdn_bwd_kernel` is grid `(B, 16, 4)`, block 64 = 2 warps, **168 registers,
0 bytes spill stores, 0 bytes spill loads, 256 B smem**, plus a 4-tile reduce kernel (28 registers) that sums tiles
0,1,2,3 in that order. Neither fallback above was needed: `sd[64]` fits beside `ds[64]`, so `sp` is read **once** per
token, and 64 rows per warp fits, so **the 4-warp variant is not needed** either. Worst gradient error against
the reference is 3.77e-07 (gate 1e-4).

The budget line "about 2 ms per layer for fwd + bwd" is **wrong for this structure**, and the reason is this section's
own `sp` scratch: phase A writes S_prev for every token (4096 floats per block per token = 16 KB, and a token is
covered by 4 tiles x 16 heads = 64 blocks, i.e. **1 MB per token**), so at N = 5094 that is 5.3 GB written and 5.3 GB
read, plus the 4-tile `dq_part`/`dk_part` partials (668 MB each: 1.3 GB written, 1.3 GB read, 334 MB final) — about
**13 GB of traffic per call**. On a 5090 (1.79 TB/s) the floor is 7.3 ms and the kernel measures **11.6 ms** (63% of
peak; a plain copy reaches ~85%); a 4090 gives a 13.0 ms floor and 15.99 ms (81%). A whole GDN layer is therefore
**12.9 ms on gpubox and 17.3 ms on the card** at B = 8 / N = 5094, not 2 ms. No kernel that stores S_prev for every
token can meet 2 ms — 10.7 GB in 0.7 ms would need 15 TB/s — so the next 4x is either keeping a segment's `sp`
resident (8 tokens is 128 KB per block, 64 MB across 512 blocks in L2) or a rolling shared-memory window replayed from
the checkpoint per chunk (~25% more phase A work). Both change this section's `sp[tl][64][128][32]` layout, which is
why v1 did not do it; it is M8's chunked GDN, with `gdn_bwd_sp_floats()`/`gdn_bwd_part_floats()` in
`kernels/gdn.h` the only place a buffer size is written down. Memory for it: 1.99 GB of backward scratch at B = 8 /
N = 5094 (ckpt 142.6 MB + `sp` 512 MB + two 668 MB partials + gates 1.3 MB each).

The fla chunked algorithm (WY form, GEMM-based) is faster and is the M8 optimisation. It does not belong in v1.

## 9. Data in, results out

### 9.1 `prep.py` (Python, runs where CLM-8B's venv is: the Mac or gpubox)

`train/jeff/py/prep.py` imports CLM-8B's `tools.jeff_lora.prompt` and `train.py` helpers (`encode`, `balance`,
`batches`, `read_jsonl`) by path (`--clm /path/to/CLM-8B`), so the token ids and the batch order are the Python
trainer's own. It never copies their code. It writes one binary file per split:

```
magic "JTB1" | u32 version=1 | u32 n_rows | u32 n_batches | u32 max_tokens_per_batch | u32 max_len | u32 max_opts
rows:    n_rows x { u32 len, u32 n_opts, f32 weight, u32 teacher_top, u64 id_offset_in_strings }
ids:     concatenated u32 token ids (sum of len)
targets: concatenated f32 (sum of n_opts)
batches: n_batches x { u32 n, u32 row_index[n] }     (train: the epoch order from batches(rng, shuffle=True),
                                                        epoch by epoch, all epochs written; eval: shuffle=False order)
strings: row ids as NUL-terminated UTF-8 (for writing predictions keyed by id)
```
Little-endian, no padding between sections, a section-offset table right after the header (u64 x 5). The C reader
validates the magic, version, every offset against the file size, and that every token id < 248320 and every n_opts
≤ 255 — and more: the sums against the section sizes, `max_len`/`max_opts`/`max_tokens_per_batch` against the rows'
actual maxima, `teacher_top < n_opts`, targets in [0,1] summing to 1 (±1e-3), every id NUL-terminated, and every batch
non-empty, in range, duplicate-free and covering the rows.

Two things the header cannot carry, settled as built: `id_offset_in_strings` is **relative to the start of the strings
section**, and the header's maxima are the file's **actual** maxima. And the header has **no room for
`--batch`/`--accum`/`--epochs`** — six u32 words and a five-entry offsets table is all there is — so prep.py does not
write them into it. Instead it writes `<split>.prep.json` beside each `.jtb`, with the arguments that shaped the file
(`batch`, `accum`, `epochs`, `eval_batch`, `targets`, `weights`, `weight_cap`, `source`, `row_weights`, `tokens`, a
header echo, and `rows_per_batch`: one row count per batch, every epoch, in file order). That sidecar is what jefftrain
checks its own flags against (plus `n_batches == ceil_div(n_rows, batch) * epochs`), and it is why a shuffled short
batch in the middle of an epoch is not a mismatch.

The flags mirror train.py's: `--targets`, `--weights`, `--weight-cap`, `--batch`, `--accum` (recorded in the sidecar
above, not in the header), `--epochs`, `--train-rows`, `--dev-rows`, `--seed`. The same seed
gives the same rows, weights and order as `train.py`. prep.py calls `random.Random(seed)` in the same sequence
train.py does: train subset, dev subset, then per-epoch `batches(train, batch, rng, shuffle=True)` with the rng
seeded once. Read train.py `main()` and reproduce the call order exactly. A unit test compares prep.py's epoch-1
batch list with one captured from an instrumented train.py run (plan, M5).

**Real data never leaves CLM-8B's gitignored `data/` directory.** The .jtb files are written there (or to /tmp), not
into macuda. Tests in macuda use synthetic fixtures only: random token ids, random targets, made by
`py/make_fixture.py` from a seed.

### 9.2 Outputs (same formats as train.py, so CLM-8B's scoring tools read them unchanged)

* `adapter/adapter_model.safetensors`: `{stem}.lora_A` [16,in] and `{stem}.lora_B` [out,16] fp32, where stem is
  the base weight's name without `.weight`, e.g. `language_model.layers.0.linear_attn.in_proj_qkv`. Check the exact
  stems against a Python-written adapter (the key set must match exactly).
* `adapter/readout.safetensors`: `weight` [255,1024]. Python saves it fp32 (readout is fp32 there); do the same.
* `adapter/adapter_config.json`: `{"format": "anemll-jeff-lora-1", "rank": 16, "scale": 2.0, "layers": 150,
  "step": ..., "dev_soft_ce": ..., "base": ..., "targets": ..., "weights": ...}`. As built this one file is still
  written with a few lines of `fprintf` in `src/lora.c`, not by `src/json.c` — `json.c` owns `report.json`,
  `pred-*.jsonl` and `--trace-loss`, and moving the adapter config there is a cosmetic follow-up, not a behaviour
  difference.
* `pred-<name>.jsonl`: `{"id": ..., "probs": {"A": p, ...}}` per row, in input order, probabilities rounded to 6
  decimals, keyed by the question's criteria keys. prep.py stores each row's keys in the strings section after the
  id, NUL-separated and terminated by an empty string.
* `report.json`: the same keys as train.py writes (args, history, train_seconds, rows_per_second, steps,
  best_dev_soft_ce, eval_<name>, peak_gb).
* Log lines in train.py's format (`  step N/T  X.XXs  loss ...`), so `tools/jeff_lora/cuda/timing.sh` can parse
  jefftrain's log as is.
* `--trace-loss FILE`: one JSON line per optimizer step, `{"step": k, "row": [row indices], "ce": [per-row soft CE]}`,
  in batch order (the group's batches in file order, each batch's rows in file order), with `--accum k` covering the
  whole group. The per-row CE is computed on the host from the step's fp32 logits (double `-Σ t·log softmax`), because
  `head_fwd` only hands back the weighted batch sum. This is what section 10.3.2's comparison reads.

The safetensors writer: build the JSON header (keys sorted, `{"dtype":"F32","shape":[..],"data_offsets":[a,b]}`),
pad it with spaces to a multiple of 8 bytes, write the u64 length, the header, then the data.

## 10. Parity: how we know it is right

Three references, each with its own job:

| ref | where | gives | gate |
|---|---|---|---|
| CPU ref (`ref/*.c`) | anywhere | per-kernel truth | per kernel, 1e-4 relative |
| `py/dump_ref.py` | CLM-8B venv, CPU or CUDA, fp32 | per-layer activations and gradients for one tiny batch | per layer, section 10.2 |
| train.py runs on gpubox | gpubox | end-to-end probabilities, losses per step | section 10.3 |

### 10.1 Weights for tests

Kernel and gradcheck tests use random weights from a seed, at full model dims or shrunk. Whole-model parity uses the
real `jeff-base` checkpoint, which lives in the HF cache on the Mac and in `~/models/jeff-base` on gpubox. It is a
public model; the weights are fine to read in tests but never get copied into macuda.

### 10.2 `dump_ref.py`: layer-by-layer

Loads jeff-base in fp32 on CPU with `attn_implementation="eager"`, and replaces the GDN rule with a literal torch
loop of section 3.6's recurrence (the same algorithm jefftrain implements, so the tightest oracle; and on gpubox, where
fla is installed, transformers' hub-kernel decorators could otherwise route even a CPU run to fla). A call counter
asserts the replacement ran 18 times per forward. *As built it also replaces `modeling_qwen3_5.causal_conv1d_fn` with
a literal conv* (section 3.6's taps), 18 calls asserted as well: `causal_conv1d` is installed on gpubox and its
hub-kernel decorator routes it to a CUDA-only custom op, so a CPU dump otherwise dies with
`Expected x.is_cuda() to be true`. It attaches LoRA through CLM-8B's `lora.py`, loads an
adapter (`--init-adapter`; for tests, one written by `py/make_adapter.py` with A random and B random * 1e-2, so the
LoRA path is not zero), and runs one forward+backward on a batch of 2 synthetic rows (lengths 37 and 70, random ids,
5 and 12 live options, random targets). Forward hooks save each decoder layer's output; `retain_grad()` on them
gives the gradients. It writes `ref.safetensors` with: `in.ids` (packed, i32), `in.off`, `layer.{i}.out` [N,1024],
`layer.{i}.in` [N,1024], `layer.{i}.post_mixer` [N,1024] (the residual right after the mixer, i.e. the input of
`post_attention_layernorm`, so a mixer and an MLP can be gated separately), `final` [N,1024], `logits` [B, live options],
`loss`, `grad.layer.{i}.out`, `grad.final`, `grad.{stem}.lora_A/B` for all 150 layers, and `grad.readout` — 403 keys.

Two shape facts any comparison has to respect. `logits` is `[B, widest row's live option count]`, not `[B,255]`:
train.py's `option_logits` slices with `[:, :live.shape[1]]`, and the dead columns are `-1e9`, exactly. That also makes
`grad.readout`'s dead rows exactly zero, and it weakens `compare_layers`' logits metric (the `-1e9` sets `max|b|`), so
read the live columns separately if the number has to mean something. `ref.safetensors` has **no `grad.layer.0.in`** —
layer 0's input-side gradient needs either a patched dump (one more `retain_grad` on `text.layers[0]`, which reproduces
the other 403 keys bit for bit) or the directional finite difference used for the attention layers.

One more format detail on jefftrain's side of the comparison: `st_write` is F32-only, so `--dump-forward`'s `in.ids`
and `in.off` are written as F32 even though the reference stores them i32. Every id is below 248320 and every offset
below the token count, all exact in fp32, and `compare_layers.py` compares by value (`to(int64)` on both), so the check
is still exact — but a reader that types the tensors will see the difference.

With the literal GDN loop, the reference runs the same algorithm as jefftrain in fp32 on CPU; only the summation order
inside GEMMs and reductions differs. Gate: per tensor, `max|a-b| / max(|b|) < 1e-3` for activations and `< 5e-3` for
gradients, with jefftrain built `-DJT_EXACT=1` and `TINYCUBLAS_TC=0`. With TF32 on, loosen to 2e-2 and 5e-2. That
is a smoke check only; the exact build is the gate (and for gradients it is the only meaningful mode at all — see 6.1).
One key-set consequence: a forward-only dump has 77 keys, so a comparison against the full 403-key file reports the
326 `grad.*` keys as missing. Use a forward-only extract for the forward checkpoint (`checkpoint m4`), and
`--dump-backward` (all 403 keys) for the backward one.

### 10.3 End to end, on gpubox (real nvcc, real cuBLAS)

1. **Eval parity.** `jefftrain --eval-only` on the 160 test rows against `cuda-parity-fp32-*/pred-test.jsonl` from
   the kit run: `compare_preds.py --max-dp 0.05 --min-agree 158` (the same script and gate the PyTorch port passed).
2. **Step parity.** Same `--init-adapter`, same .jtb batches, 20 steps, fp32 Python (`--dtype fp32`) vs jefftrain.
   Gate: |loss_py - loss_c| / loss_py < 1e-3 for each of steps 1-5, < 1e-2 through step 20, and the dev soft CE at
   step 20 within 2e-3.
3. **Run parity.** One full soft arm, jefftrain vs the Python bf16 arm: dev soft CE at the chosen step within 0.01,
   test soft CE within 0.02, `score.py` metrics within noise (same arm, two Python seeds give the noise band; run
   seed 1 too). *As built the band is taken over all four runs of the arm* — both Python seeds and both jefftrain seeds —
   which is the definition the user chose after seeing the Python-only band reject seed 0 at 6 of 15 metrics. Read that
   table knowing a member-vs-member difference cannot exceed a band that contains both members: the informative parts
   are which implementation contributes more spread (jefftrain on 4 of 15) and whether the means' gap is inside it.
   `scripts/arm.sh` still builds its band from the Python runs only, so the four-run table is not reproducible from the
   script alone; that is an open follow-up. The other known difference in the trainer: `arm.sh` starts from
   `make_adapter.py`'s fresh adapter (A random, B ~ N(0,1)·1e-2) where train.py starts at B = 0.

Measured (section 10.3, all three): eval max |dp| 0.0014 TF32 / 0.0015 exact with argmax 160/160; 20 steps worst
step-mean difference 1.18e-03 at step 7 and dev soft CE within 2.70e-04; run parity PASS, 15/15 metrics inside the
four-run band with both CE gates at 0.00072/0.00130 (seed 0) and 0.00607/0.01170 (seed 1).

### 10.4 On the shim (Mac + eGPU)

Same binary source, linked against the shim. Each piece goes to the card soon after it passes on gpubox: the GEMMs, the
kernels, then the forward and backward against `dump_ref.py`, at the same gates as on gpubox (Plan M1–M5 checkpoints).
Whole-trainer gates, at the end: 10.3.1 (eval parity, 160 rows) and 10.3.2 (20 steps), comparing
with the gpubox jefftrain numbers. Expect near-identical (same kernels; cuBLAS vs tinycublas GEMM order is the only
difference). Then one full arm, timed.

Measured on card B (RTX 4090, `ARCH=sm_89`, shim): eval parity **max |dp| 0.0011** against gpubox jefftrain (9x inside
the 0.01 gate), argmax 160/160; 20-step parity worst **1.17e-03** (gate 1e-2) and **5.44e-05** per row against the
gpubox binary; the full arm was not run — at 8.94 s/step it is ~9 h on the card. The layer-level checkpoint numbers are
2-7x the gpubox ones (m4 `final` 4.572e-06 vs 2.104e-06, m5 worst 2.345e-05 vs 1.205e-05), i.e. a different
summation order, not a layout error.

## 11. Build and run

Everything lives in `macuda/train/jeff/` with its own Makefile and its own `build/` directory. It never writes into
`cuda-shim/build/` and never modifies the shim. It only links the shim's libraries.

### 11.1 Two builds of one source

| build | where | device code | host code | libraries |
|---|---|---|---|---|
| `make nv` | gpubox (WSL, `/usr/local/cuda-12.9/bin/nvcc`) | nvcc `-arch=sm_120` | nvcc / gcc | real cudart + `-lcublas` |
| `make shim` | the Mac | `cuda-shim/build/tinycc` (docker backend, or ssh to gpubox) | Homebrew clang | `cuda-shim/build/shim*/libtinycudart.a libtinycublas.a nv/libtinynv.a` + LLVMDemangle + `-lc++` |

Source rules that keep both builds working:
* Host program in C (`.c`, C11), kernels in `.cu`, each `.cu` exposing `extern "C"` launch wrappers. The C code never
  sees CUDA syntax. It includes `<cuda_runtime.h>` and `<cublas_v2.h>` only for types and API calls.
* Every `.cu` starts with `#include <cuda_runtime.h>`. tinycc adds `spike/tinycudart_compat.h` itself.
* Only the cudart calls listed in section 12. A `make check-symbols` target runs `nm -u` on the **unlinked objects** `build/shim/*.o` (the linked binary has
  nothing undefined left to show) and fails on any `_cuda*`/`_cublas*` symbol the shim archives do not define (same
  recipe as cuda-shim's `coverage`, which does it on `libggml-cuda.a`).
* No `__constant__`, no `cudaMemcpyToSymbol`, no `cudaMallocAsync`, no events for timing, no explicit `cudaLaunchKernel` call (use
  `<<<>>>`, which `spike/tinycudart_compat.h` lowers to the legacy `cudaConfigureCall`/`cudaSetupArgument`/`cudaLaunch`
  ABI the shim exports; libtinycudart has no `cudaLaunchKernel` symbol, despite the old comment in `spike/vecadd.cu`), no `cudaStreamCreate` (use `cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)`), no thrust/CUB.
* Dynamic shared memory over 48 KB needs `cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, n)`
  (in the shim). The v1 kernels above stay under 48 KB, so they don't need it.

`make shim` details, all from `cuda-shim/Makefile` and `build/tinycc`, *as built*:
```
CC       = /opt/homebrew/opt/llvm/bin/clang
MACUDA   ?= $(HOME)/Downloads/vfio/macuda          # the main checkout: a worktree has no git-ignored shim build or headers
SDK      = $(shell sh $(MACUDA)/cuda-shim/build/sdk.sh)
CUDA_INC = $(MACUDA)/cuda-shim/cuda-13/include
SHIM     ?= $(MACUDA)/cuda-shim/build/shim            # or build/shim-sm_89 for the Ada card; must match TINYCC_ARCH
CFLAGS   = -isysroot $(SDK) -std=c11 -O2 -g -Wall -Wextra -I$(CUDA_INC) -Isrc
kernels: cp kernels/X.cu kernels/.tc_<cksum-of-$PWD>_X.cu
         TINYCC_GGML_MAC=$(abspath .) TINYCC_ARCH=$(ARCH) TINYCC_SYSROOT=$(SDK) /usr/bin/lockf /tmp/jefftrain-tinycc.lock \
             sh $(MACUDA)/cuda-shim/build/tinycc kernels/.tc_<cksum>_X.cu -o build/shim/X.o \
             -I$(abspath {src,kernels,ref,test}) -DJT_SHIM
         rm -f kernels/.tc_<cksum>_X.cu
link:    $(CC) $(CFLAGS) build/shim/*.o src/*.c $(SHIM)/libtinycudart.a $(SHIM)/libtinycublas.a $(SHIM)/nv/libtinynv.a \
             /opt/homebrew/opt/llvm/lib/libLLVMDemangle.a -lc++ -lm -o build/shim/jefftrain
```
The staging copy is what makes parallel worktrees safe: the `.cu` gets a name unique to the checkout (the `cksum` of its
absolute path), so neither the container's `/ggml` mount nor the `/tmp/tinycc_*.fatbin` file tinycc derives from it can
collide with another task's build, and the copy is removed again even on failure (`.gitignore` covers `kernels/.tc_*`
anyway). The `lockf` serialises concurrent kernel compiles from `make -j8`, which would otherwise race on that mount.
`TINYCC_GGML_MAC` must be a directory that contains the `.cu` source: tinycc mounts it read-only at `/ggml` in the
container, or expects it synced at `$TINYCC_GGML_LINUX` on the ssh host. Point it at `train/jeff`. `ARCH` is the
card's: `sm_120a` for the 5090, `sm_89` for the Ada card (with `SHIM=$(MACUDA)/cuda-shim/build/shim-sm_89`). Ask which
card the run is for; don't guess. Each build directory records the arch (`build/shim/.arch`, `build/nv/.arch`) and
drops its objects when it changes — without that, switching cards silently ran the previous card's cubin.

The device half of a shim build is a container's nvcc (CUDA 13.0.3, one `--fatbin` per TU) and the host half Homebrew
clang; `cuda-shim/build/container-runtime.sh` prefers Apple's `container` CLI when it is installed, so a stopped Docker
daemon does not block the build. The shim build carries no `-DJT_EXACT`: exact mode comes from the environment
(`JT_EXACT=1 TINYCUBLAS_TC=0`), so one build serves both modes.

### 11.2 Running on the card

The null device (the hardware path needs both `TINYNV_HW` and `TINYNV_SOCKET`; run with `env -u TINYNV_HW -u TINYNV_SOCKET -u TINYNV_PCI` to be sure) links and runs the API but computes nothing, so it can only check that start-up,
allocation and launch sequencing work, not numbers. Every run on real hardware goes through the repository's
hardware protocol:
```
sh $MACUDA/tools/nv_shim_step.sh <A|B> probe $PWD/build/shim/jefftrain <args...>
```
which does preflight, the run as its own process with the log captured, then quiesce. The card is shared with other
sessions: ask the user before taking it, every time. `scripts/card-check.sh` (Plan M0.4) wraps a milestone's tests into
one such run per checkpoint, so one OK covers the checkpoint.

### 11.3 gpubox

`train/jeff/scripts/gpubox-sync.sh` rsyncs `train/jeff/` to `gpubox:~/$JT_REMOTE/` (no data, no weights);
`gpubox-build.sh` runs `make nv` there; the parity scripts run `~/$JT_REMOTE/build/nv/jefftrain` against
`~/models/jeff-base` and CLM-8B's `.jtb` files under `~/CLM-8B/data/`. `JT_REMOTE` defaults to `macuda-train/jeff`;
**export it before every call** (`export JT_REMOTE=macuda-train/wt-J34`) so parallel tasks never rsync `--delete` or
`make nv` over one tree. `~/macuda-train/ref/` is read-only. The PyTorch reference venv is `~/CLM-8B/.venv-cuda`
(torch 2.8.0+cu128, fla 0.5.2), already set up and checked (VERDICT OK, PARITY OK on 2026-11-08).

One trap: `rsync -a` restores the *source's* mtime, so a file re-synced after a build can look older than its object
and `make` will happily keep the stale one. `rm -rf build` (or `make -B`) after every sync that changed a file — a
re-synced experiment once left a mutated binary in place and reported the mutation's own numbers.

## 12. The shim's API surface (what jefftrain may call)

From `nm` on `build/shim-sm_89/libtinycudart.a` and `libtinycublas.a`, 2026-11-08:

* Memory: cudaMalloc, cudaFree, cudaMallocHost, cudaFreeHost, cudaHostAlloc, cudaMemcpy, cudaMemcpyAsync, cudaMemset,
  cudaMemsetAsync, cudaMemGetInfo.
* Device: cudaGetDeviceCount, cudaGetDevice, cudaSetDevice, cudaGetDeviceProperties(_v2), cudaDeviceGetAttribute,
  cudaDeviceSynchronize.
* Streams: cudaStreamCreateWithFlags, cudaStreamDestroy, cudaStreamSynchronize, cudaStreamWaitEvent,
  cudaEventCreateWithFlags, cudaEventRecord, cudaEventSynchronize, cudaEventDestroy.
* Kernels: the `<<<>>>` ABI (cudaConfigureCall / cudaSetupArgument / cudaLaunch, __cudaRegister*), cudaFuncSetAttribute,
  cudaFuncGetAttributes, cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags.
* Errors: cudaGetLastError, cudaPeekAtLastError, cudaGetErrorString.
* cuBLAS (all of libtinycublas's exports): cublasCreate_v2, cublasDestroy_v2, cublasSetStream_v2,
  cublasSetMathMode, cublasSetWorkspace_v2, cublasGetStatusString, cublasGemmEx, cublasGemmBatchedEx,
  cublasGemmStridedBatchedEx, cublasSgemm_v2, cublasSgemmBatched, cublasSgemmStridedBatched, cublasStrsmBatched.
  jefftrain uses only cublasGemmEx and cublasGemmStridedBatchedEx for math. Plan M1.2a checks that the strided-batched
  entry point accepts COMPUTE_32F_FAST_TF32 with fp32 A/B/C on the shim, as GemmEx does — and it does: card B's m1
  checkpoint printed `GEMM OK 112/112` in both modes with no `NOT_SUPPORTED`. As built the whole tree needs **24**
  cuda*/cublas* symbols (`make check-symbols`), all defined by the shim archives: the count grew from 10 at M0 to 15 at
  M1 (the five cuBLAS entries), 17 at M2.6 (`_cudaMemsetAsync`, `_cudaMemGetInfo`), and 23 by M4.3–M5, with 24 after
  `--trace-loss`'s host pulls. Nothing in `src/` or `kernels/` had to move to a different API to get there.

**Missing, do not use:** cudaLaunchKernel, cudaStreamCreate, cudaEventCreate, cudaEventElapsedTime (time on the host
with `clock_gettime(CLOCK_MONOTONIC)` after `cudaDeviceSynchronize`), cudaMemcpyToSymbol, cudaMallocAsync,
cudaMallocPitch.

Wrap every call in `CK(x)` (cudart) and `CB(x)` (cuBLAS) macros that print file:line and the error string and exit.
After every kernel launch in debug builds (`-DJT_DEBUG=1`), `CK(cudaGetLastError()); CK(cudaDeviceSynchronize())`.

## 13. File layout

```
train/jeff/
  Makefile                 targets: nv, shim, test-cpu, test-null, check-symbols, clean
  README.md                how to build and run, pointing here
  src/
    main.c                 flags (train.py's names), the training loop, the dev check, eval, report.json,
                           --dump-forward/--dump-backward, --trace-loss
    model.c / model.h      the loader (every tensor through st_get into one fp32 arena), jt_batch_set,
                           jt_ctx_alloc/free, layer_fwd/layer_bwd, model_forward, model_backward (section 6.2's sweep)
    model_table.c / .h     the model as data: per-layer names/shapes/LoRA flags, the 150 adapted linears in
                           flat-buffer order, config constants (section 3.2's list, as data)
    linear.c / linear.h    the frozen `struct Linear` and linear_fwd/linear_bwd (section 4.2/5)
    layer.h                the frozen layer API: JtCtx, the two-half arena, JT_FWD_ONLY/JT_FWD_SAVE, HeadBatch,
                           the ten fwd/bwd signatures, the three JT_LAYER_* return codes
    layer_gdn.c            one GDN layer: fwd (save/recompute mode), bwd; calls gemm + kernel wrappers
    layer_attn.c           one full-attention layer, same shape
    layer_mlp.c            the MLP + its norm, shared by both layer types
    head.c                 final norm, readout, CE, dlogits
    lora.c                 the flat parameter buffer, offsets, init/load/save adapter, Adam, clip, schedule
    gemm.c / gemm.h        row-major wrappers (section 5)
    st.c / st.h            safetensors read (mmap, validating) and write
    jtb.c / jtb.h          .jtb reader (section 9.1)
    json.c / json.h        just enough JSON: report.json, pred-*.jsonl, the --trace-loss record, plus the field
                           scanners for prep.py's .prep.json sidecar and decision_config.json's temperature
    util.h                 CK/CB macros, timers, arena allocator
  kernels/
    norm.cu  act.cu  rope.cu  conv.cu  gdn.cu  attn.cu  head.cu  adam.cu  misc.cu
  ref/                     CPU fp32 references, one per kernels/*.cu file, same function names with a ref_ prefix;
                           built twice (float, and double under -DJT_REAL_DOUBLE for the gradchecks)
  test/
    harness.h              gates (JT_TOL_*), rnd_fill/rel_err/CHECK_*, device helpers, the FD gradcheck core
    test_kernels.c  gradcheck.c        the two frozen dispatchers (a group returns JT_OK/JT_FAIL/JT_NOT_IMPL)
    k_*.c                  one GPU-vs-reference group each: misc norm act rope conv attn head adam
    grad_*.c               one CPU gradcheck group each: norm act rope conv attn head gdn lora
    test_gdn.c  test_gemm.c  test_st.c  test_jtb.c  test_hello.c  test_layer.c  test_linear.c
  py/
    prep.py  make_fixture.py  make_adapter.py  dump_ref.py  compare_layers.py
    trace_train_order.py   the prep.py vs train.py batch-order evidence
    trace_train_loss.py    train.py's per-step per-row CE, for M6.2
    compare_steps.py       trace_train_loss.py's output vs --trace-loss, with the M6.2 gate
  scripts/
    gpubox-sync.sh  gpubox-build.sh  gpubox-run.sh  prereq-check.sh  card-check.sh
    parity-eval.sh  parity-steps.sh  timing.sh  arm.sh
  notes/                   one <task>.md per step, first line STATUS: PASS or STATUS: BLOCKED — … (plan rule 9)
```

The model is described by one table, `src/model_table.c`, not by string building scattered through the code: for each
layer, the safetensors names it loads and whether each linear has LoRA. Section 3.2's list, as data. It is its own
translation unit rather than a part of `model.c` because `src/lora.c` needs it before `model.c` exists; `model.c` is the
loader that consumes it and verifies every name and dimension against the checkpoint.

`struct Linear { const float *W; int in, out; int lora; size_t a_off, b_off; const char *stem; }`: `lora` set means
`a_off`/`b_off` index the flat parameter buffer. `linear_fwd(ctx, &lin, x, N, y, u_save)` and
`linear_bwd(ctx, &lin, x, u, dy, N, dx, accumulate_dx)` are the only functions that touch W, A or B. Every layer
calls them. Two caveats found in use: `linear_fwd`/`linear_bwd` return `JT_LAYER_FAIL` (never `exit()`) on a bad call,
and `stem` must outlive the load — it once pointed into a pool `model_load` freed, which prints `?` in messages and
breaks any lookup keyed on it.

## 14. Risks and how each is handled

| risk | handling |
|---|---|
| a backward formula is wrong | gradcheck per kernel (double precision CPU ref) and per layer (fp32 GPU vs finite differences) before end-to-end |
| layout mistake (q/gate interleave, conv taps, head order) | dump_ref.py layer-by-layer parity on real weights pinpoints the first layer and tensor that diverges |
| GDN kernel too slow on the shim | v1's real cost is 12.9 ms/layer on gpubox and 17.3 ms on the card, memory-bound on section 8.3's `sp` scratch (not the estimate this row originally carried); the chunked WY form and a resident/rolling `sp` window are M8 |
| shim lacks an API | section 12 plus `make check-symbols` before anything runs |
| strided-batched TF32 rejected, or a GDN kernel misbehaves on the shim | found by the card checkpoints at Plan M1.2a and M3.2a, before the code that depends on them is written |
| tinycublas TF32 numerics differ from cuBLAS | the gate is against the exact build; TF32 runs are compared run-to-run only |
| the batch order differs from Python | prep.py reuses train.py's own functions; the M5 test compares batch lists |
| memory on a 12 GB card | layer recompute plus a single-batch arena; `--batch 4` fallback; the arena size is printed at start |
| the card is shared | every hardware run goes through nv_shim_step.sh, after asking |

## 15. Expected performance

Reference: PyTorch with fla on the 5090 runs batch 8 in 0.143 s/step (56 rows/s, about 9 min per epoch of 28,645
rows, 16.75 GB peak). Batch 16 goes over 32 GB and drops to 3.6 s/step (WSL pages to host memory), so batch 8 it is.

jefftrain v1 on the same card: per step about 3 x forward FLOPs (fwd + recompute + bwd x 2 / 2) of about
2 x 0.5 G params x 4200 tokens ≈ 4.2 TFLOP fwd, so about 17 TFLOP per step; TF32 at about 50% of the 5090's peak
(about 100 TFLOPS dense) gives about 0.17 s plus the recurrence (18 layers x 2 passes x about 2 ms) and small kernels.
Target: under 0.4 s/step on gpubox (under 25 min per epoch). On the shim, the eGPU link and tinycublas's GEMM set
the pace; measure it in M7 and write the number down.

**Measured, which supersedes every estimate above.** TF32, full train split, batch 8: **1.061 s/step** (7.5 rows/s),
i.e. **7.42x** PyTorch's 0.143 and **0.661 s/step over** the 0.4 target; peak 11.33 GB. One epoch of 28,645 rows is
**~63 min** of training, so a full arm is ~75 min (plan M6.4). The estimate that was clearly wrong is the recurrence
(18 layers x 12.9 ms = 232 ms per fwd+bwd, not 18 x 2 ms = 36 ms — section 8.3's `sp` traffic), and the remaining gap
is the many small kernels and their launch cost rather than the projections; v1 made no attempt to attribute the last
0.6 s, which is what M8's item 5 measures first. A full arm measures 1.202 / 1.164 s/step (10-13% over the 100-step
figure, because the dev checks leave the card cold) with peak 12.29 GB — inside gpubox's 32 GB and well over design
6.2's 12 GB target, which is what a batch-8 run of this size costs in fp32 weights plus arena. On the shim, exact mode
is **8.94 s/step** (7.6x gpubox) for 20 steps.
