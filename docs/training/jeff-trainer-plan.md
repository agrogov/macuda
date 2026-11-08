# jefftrain: implementation plan

Companion to `jeff-trainer-design.md` (called **the design** below, sections as `D§n`). The design says *what* and
*why*. This file says *in which order*, *which file*, *which function*, *which command proves it*. It is written so a
coding agent can follow it step by step without re-deriving anything. When this plan and the design disagree, the
design wins; when the design and CLM-8B's `tools/jeff_lora/train.py` disagree, train.py wins (D§1.1). Fix the doc
in the same change.

## 0. Rules for the agent

1. **One step at a time.** Each step below ends with a *Done when* line. Don't start the next step until that
   command passes. If it doesn't pass after honest attempts, stop and write down what you see (the command, the
   output, your best guess). Don't loosen a tolerance to make a test pass. Tolerances are set in this plan;
   changing one is a question for the user.
2. **Never write a kernel without its CPU reference first.** Every `kernels/X.cu` function `foo` has `ref/X.c`
   `ref_foo` written and tested first (gradcheck for backward ones). The GPU test compares against `ref_foo`.
3. **Read the examples before writing the first file of a kind:**
   * a `.cu` the shim compiles: `cuda-shim/spike/vecadd.cu`, `cuda-shim/spike/smem_probe.cu`
   * a C program using cudart + cuBLAS through the shim: `cuda-shim/test/cublas_contract.c`,
     `cuda-shim/test/splitk_card.c` (a real GEMM checked on the card)
   * the link line and the null-device test pattern: `cuda-shim/Makefile` (targets `cublas_contract`, `test`,
     `coverage`)
   * how `.cu` files are compiled for the shim today: `cuda-shim/build/build-ggml-cuda.sh` (the env it exports:
     `TINYCC_ARCH`, `TINYCC_SYSROOT`) and `cuda-shim/build/cc-one.sh` (one TU). Copy their recipe into the Makefile
     rather than inventing one, and in M0.2 confirm with `nm build/shim/misc.o` that tinycc's output is a host object
     with the kernel's registration code and the fatbin embedded (`__cuda_fatbin`/`__cudaRegisterFunction`-style
     symbols).
   * the PyTorch code you're reproducing: CLM-8B `tools/jeff_lora/train.py`, `lora.py`, `prompt.py`, and HF
     `transformers/models/qwen3_5/modeling_qwen3_5.py` (in CLM-8B's venv; the design cites its line numbers)
4. **Don't touch** `cuda-shim/` (link against it only), `vendor/`, anything under CLM-8B's `vendor/CLM`, or other
   people's uncommitted files in macuda (`git status` shows them: leave them alone, don't stage them).
5. **No real data in macuda.** Meeting transcripts and anything made from them (`.jtb` from real splits,
   predictions, reports) stay in CLM-8B's `data/` or on gpubox. Tests in macuda use synthetic fixtures (D§10.1).
6. **The card is shared.** Every run on the Mac's eGPU goes through `tools/nv_shim_step.sh` and needs the user's
   OK each time. gpubox runs (192.168.31.162, `ssh gpubox`) need an OK for anything long (over about 10 min) or anything
   that takes the whole GPU.
7. **Commit only when the user asks.** macuda's main checkout has other people's *staged* files, and a plain
   `git commit` on a new branch would sweep them in. Work in a separate worktree, which has its own index:
   `git -C macuda worktree add ../macuda-train-jeff -b train-jeff main`, and do all of this plan there. If you must
   commit from the main checkout, use `git commit -- <your paths>` only. Never add AI attribution to commits.
8. Code style: C11, `-Wall -Wextra` clean, no global mutable state except the context struct, every CUDA/cuBLAS
   call wrapped in `CK()`/`CB()` (D§12), every function under about 80 lines. Names follow the design's tables.

## Milestones at a glance

| M | what | proves | where it runs |
|---|---|---|---|
| M0 | scaffold, both builds of a trivial kernel | toolchains and link line | Mac (null device), gpubox |
| M1 | `gemm.c` row-major wrappers + `st.c` + `jtb.c` | the GEMM contract, file readers | gpubox; Mac null device |
| M2 | elementwise/row kernels + CPU refs + gradchecks | each small op, fwd and bwd | gpubox |
| M3 | GDN recurrence fwd/bwd | the hard kernel | gpubox |
| M4 | full forward on real weights vs `dump_ref.py` | layouts, model assembly | gpubox |
| M5 | backward, LoRA, Adam, schedule, prep.py batch order | one training step bit-close to PyTorch | gpubox |
| M6 | eval/step/run parity, timing | the whole trainer | gpubox |
| M7 | shim build, card run | macuda can train Jeff | Mac + eGPU |
| M8 | speed (bf16 weights, chunked GDN, fused kernels) | optional | both |

gpubox is the development box for M1–M6 because it has real nvcc and cuBLAS, so a failure there is jefftrain's
own fault. The shim enters at M7 with known-good code, which means any new failure there is a shim question.
A Mac null-device build is kept green from M0 onward (`make shim test-null`), so shim incompatibilities show up
early as link or launch errors, even though the numbers come later.

---

## M0. Scaffold and both builds

### M0.1 Directory and Makefile
Create the tree from D§13 with empty files where needed. `Makefile` targets:

| target | does |
|---|---|
| `nv` | `nvcc -O2 -arch=$(ARCH) -std=c++17 -Xcompiler -Wall` each `kernels/*.cu` → `build/nv/*.o`; `gcc -std=c11 -O2 -Wall -Wextra` each `src/*.c` → `build/nv/*.o`; link with `nvcc` + `-lcublas -lm` → `build/nv/jefftrain`; same for each `test/*.c` → `build/nv/test_*` (tests link every `src/*.o` except `main.o`) |
| `shim` | the D§11.1 recipe → `build/shim/jefftrain` and `build/shim/test_*` |
| `test-cpu` | builds `ref/*.c` + `test/test_ref*.c` with the host compiler only, runs them (no GPU) |
| `test-null` | runs `build/shim/test_*` with `env -u TINYNV_HW -u TINYNV_SOCKET -u TINYNV_PCI` (null device; the hardware path needs both `TINYNV_HW` and `TINYNV_SOCKET`): must exit 0 and print no CUDA error |
| `check-symbols` | D§11.1: `nm -u build/shim/*.o` (the unlinked objects: the linked binary shows nothing undefined), filter `_cuda`/`_cublas`, each must be defined in the shim archives |
| `clean` | `rm -rf build` |

`ARCH ?= sm_120` for `nv`. For `shim`, `ARCH` has no default: `make shim` without `ARCH=` prints
"set ARCH=sm_120a (5090) or sm_89 (Ada, with SHIM=.../shim-sm_89); ask which card" and fails.

### M0.2 Hello kernel
`kernels/misc.cu`: `extern "C" void jt_fill(float *x, float v, size_t n, cudaStream_t s)` launching a grid-stride
fill kernel. `src/util.h`: `CK(x)`, `CB(x)` macros printing file:line and the error string, then `exit(1)`;
`JT_SYNC_DEBUG` (D§12). `test/test_hello.c`: allocate 1M floats, `jt_fill(…, 3.0f)`, copy back, check all == 3,
print `hello OK`.

**Done when:**
* gpubox: `bash scripts/gpubox-sync.sh && ssh gpubox 'cd ~/macuda-train/jeff && make nv && build/nv/test_hello'` →
  `hello OK`.
* Mac: `make shim ARCH=sm_120a && make test-null && make check-symbols` → exit 0, and `nm build/shim/misc.o` shows
  the embedded fatbin and registration symbols (the null device can't check the
  value, so in the shim build (`-DJT_SHIM`) `test_hello` skips the value check unless both `TINYNV_HW` and `TINYNV_SOCKET` are set, printing
  `hello OK (null device: launch only)`).

### M0.3 gpubox scripts
* `scripts/gpubox-sync.sh`: `rsync -a --delete --exclude build/ train/jeff/ gpubox:~/macuda-train/jeff/`.
* `scripts/gpubox-build.sh`: sync, then `ssh gpubox 'cd ~/macuda-train/jeff && make nv ARCH=sm_120'`.
* `scripts/gpubox-run.sh <binary> [args]`: build, then run `build/nv/<binary>` there, streaming the output.

**Done when:** `bash scripts/gpubox-run.sh test_hello` prints `hello OK`.

---

## M1. GEMM wrappers and file readers

### M1.1 `src/gemm.c`
Write `gemm()` and `gemm_batched()` exactly as D§5 gives them. `gemm_batched` calls
`cublasGemmStridedBatchedEx` with the same argument mapping (strides follow their matrices: the cuBLAS "A"
argument is our B, so it gets `sB`). A file-static `cublasComputeType_t jt_compute` is set once in `gemm_init()`:
`CUBLAS_COMPUTE_32F` if `JT_EXACT` is defined at compile time or `getenv("JT_EXACT")` is "1", else
`CUBLAS_COMPUTE_32F_FAST_TF32`. Print the choice once at start-up (`gemm: exact fp32` / `gemm: tf32`).

Common mistakes to avoid:
* Mixing up leading dimensions. The leading dim is the *stored* row length of each matrix: A is `ta ? M : K`, B is
  `tb ? K : N`, and C is always `N`.
* Passing `alpha`/`beta` by value. They are pointers to host floats (the default pointer mode).

### M1.2 `test/test_gemm.c`
A CPU reference `ref_gemm` (triple loop, double accumulator), random fp32 inputs in [-1,1] from a fixed xorshift
seed, and a check of:
* all four (ta,tb) pairs × sizes {(1,1,1), (7,13,5), (64,64,64), (130,17,1024), (4200,16,1024), (16,1024,4200)};
* alpha ∈ {1, 2}, beta ∈ {0, 1} (beta=1 with C prefilled);
* gemm_batched with batch 8, (ta,tb) = (0,1), (0,0), (1,0), sizes (L=37, 256) and (L=300, 256), strides = M*K etc.

Error metric everywhere in this plan: `rel = max|got-want| / max(1e-30, max|want|)`.
Gate: `rel < 1e-5` with JT_EXACT=1; `rel < 5e-3` with TF32. Print one line per case plus `GEMM OK n/n`.

**Done when:** on gpubox, `JT_EXACT=1 build/nv/test_gemm` and `build/nv/test_gemm` both print `GEMM OK`. Mac:
`make shim test-null` is still green.

### M1.3 `src/st.c`: safetensors
* `st_open(path) → St*`: `mmap` the file; parse the u64 header length and the JSON header. A tiny hand-written
  JSON scanner is enough: the header is one flat object of `"name": {"dtype": "...", "shape": [...],
  "data_offsets": [a, b]}` plus an optional `"__metadata__"` object, which you skip. Store an array of
  `{name, dtype, ndim, shape[4], ptr, nbytes}`.
* `st_get(st, name, dtype, ndim, shape...) → const void*`: find by exact name. Exit with a clear message if the
  name is missing, or the dtype or shape differs from what the caller expects. Callers always state the
  expected shape, which catches layout mistakes at load time.
* `st_write(path, n, names[], shapes[][], ndims[], const float *data[])`: the D§9.2 recipe, F32 only, keys sorted.
* Jeff's weights may be sharded: if `model.safetensors.index.json` exists, read its `weight_map` and open each
  shard. Check what `~/models/jeff-base` on gpubox actually has (`ls`) before writing this. Handle only what you find.

`test/test_st.c` writes a 3-tensor file with `st_write`, reads it back with `st_open`/`st_get`, and compares the
bytes. Then gpubox: `test_st --list ~/models/jeff-base` prints every tensor's name, dtype and shape. Compare that by
eye with D§3.2 and fix the design if it's wrong.

**Done when:** the round trip passes, and the listing matches D§3.2 tensor for tensor (paste the listing into the
change description).

### M1.4 `src/jtb.c` and `py/make_fixture.py`
* `py/make_fixture.py --seed S --rows R --out f.jtb [--max-len 120] [--batch 4]` writes a valid JTB1 file
  (D§9.1) with random ids < 248320, random lengths in [8, max-len], n_opts in [2, 6], softmax-normalised random
  targets, weights 1.0, ids `fx-0000`…, criteria keys `A`,`B`,…, and batches = consecutive chunks of `--batch`
  rows (no shuffle; the real order comes from prep.py in M5). This is the only JTB writer until M5.
* `jtb_open(path) → Jtb*` (mmap plus validation, D§9.1), plus accessors `jtb_row(j, i)`, `jtb_ids(j, i)`,
  `jtb_targets(j, i)`, `jtb_batch(j, k, &n) → const u32*`, `jtb_id(j, i)`, `jtb_keys(j, i, out[], max)`.
* `test/test_jtb.c`: for a fixture, check that the sums of len and n_opts match the section sizes. Then corrupt one
  byte of the magic and expect a clean error.

**Done when:** `make test-cpu` (where test_jtb runs; it needs no GPU) is green on the Mac.

---

## M2. Small kernels, each with a CPU reference and a gradcheck

### M2.0 The test harness (write this first)
* `ref/real.h`: `typedef float real;` by default and `typedef double real;` under `-DJT_REAL_DOUBLE`. Every
  `ref_*` function takes `real*`. `make test-cpu` builds `ref/` twice: once as float (to compare with the GPU)
  and once as double (for gradchecks).
* `test/harness.h` (header-only):
  * `rnd_fill(float *x, n, lo, hi, seed)` (xorshift32, deterministic)
  * `rel_err(got, want, n)` (the M1.2 metric)
  * `CHECK_REL(name, got, want, n, tol)`, which prints `ok  name rel=…` or `FAIL name rel=… tol=…`, counts
    failures, and makes `main` return 1 if any failed
  * `dev_upload(host, n) → float*`, `dev_download(dev, host, n)`
* `test/gradcheck.c` (CPU only, double): for each backward ref, pick a random input x and a random cotangent w for
  the output. Analytic: `ref_*_bwd(w) → gx`. Numeric: for each element i of x (all of them; the sizes are small),
  `(L(x+h e_i) − L(x−h e_i)) / 2h`, with `L(x) = Σ w·f(x)` and h = 1e-6. Gate: `rel_err < 1e-6`. Also check the
  gradients of every other differentiable input (z in the gated norm, g and beta in GDN, A and B in LoRA, the
  readout R in the head).

Every GPU test below follows the same pattern. Run `ref_foo` (float) on the host with random inputs, run
`k_foo` on the device with the same inputs, and compare with `CHECK_REL`. Default gates: forward `1e-5`, backward
`1e-4` (these kernels use no tensor cores, so they are fp32 vs fp32 and only the summation order differs). Shapes
must include ragged ones: a packed batch of 3 rows with lengths {1, 37, 130}, and a full one with lengths
{520, 670, 1026, 300} (1026 is the longest real row: it crosses 1024, so it catches buffers sized to a power of two).

### M2.1 Plumbing kernels → `kernels/misc.cu`, `ref/misc.c`
`zero`, `add_inplace(y += x)`, `scale`, `bf16_to_f32`, `embed_gather(ids, table_bf16, out, N, 1024)`,
`qgate_split` / `qgate_merge` (D§3.5: head h, j<256 → q[t, h*256+j] = qg[t, h*512+j];
gate[t, h*256+j] = qg[t, h*512+256+j]), `kv_expand` / `kv_reduce` (head h reads group h/4; the reduce sums
heads 4g..4g+3 in that order), `to_head_major` / `from_head_major` ([L, H*D] ↔ [H][L][D] for one row slice).
bf16 → fp32 is `bits << 16`, nothing else.

**Done when:** `test_kernels misc` is green on gpubox. Also make sure `qgate_split` followed by `qgate_merge` is the
identity, and that `kv_reduce` is the adjoint of `kv_expand`: `<expand(a), b> == <a, reduce(b)>` to 1e-5.

### M2.2 Norms → `kernels/norm.cu`, `ref/norm.c`
`rmsnorm_fwd(x, w, y, r, rows, d, zero_centred)` and `rmsnorm_bwd` (D§4.3); `gated_rmsnorm_fwd/bwd` (D§4.4);
`l2norm_fwd/bwd` (D§4.8's l2 lines, with the 1/√128 scale as a parameter `post_scale`).
Launch shapes from D§8.1. One block per row for d=1024: 256 threads, 4 elements each, warp-shuffle then
shared-memory reduce. One warp per row for d=128 and d=256.

Pitfall: `zero_centred` multiplies by `(1+w)`. The GDN gated norm multiplies by `w`, not `(1+w)` (D§3.3). Pass the
right flag in each place and test both.

**Done when:** gradcheck has passed for all three backwards, and `test_kernels norm` is green on gpubox.

### M2.3 Activations and gates → `kernels/act.cu`, `ref/act.c`
`silu_mul_fwd(h1, h2, act)` / `silu_mul_bwd(dact, h1, h2, dh1, dh2)`; `sigmoid_gate_fwd/bwd` (D§4.7's last
lines); `gdn_gates_fwd(bb, aa, A_log, dt_bias, beta, g, N)` / `gdn_gates_bwd` (D§4.8's last lines);
`softplus` as torch does it: `u > 20 ? u : log1pf(expf(u))`.

**Done when:** gradcheck (pick inputs away from the softplus threshold) passes, and `test_kernels act` is green.

### M2.4 RoPE and conv → `kernels/rope.cu`, `kernels/conv.cu`
* `rope_tables(cos, sin, Lmax)`: built on the host in fp32 exactly as D§3.3 says, then uploaded. Tables are
  [Lmax][64].
* `rope_fwd/bwd(x, row_off, row_of, heads, head_dim=256, cos, sin)`: in place on [N, heads*256]; position =
  `t - off[row_of[t]]`. The backward is the same rotation with sin negated.
* `conv_fwd(qkv, wconv, row_of, off, pre, out, N, C=6144)`: `pre` = conv output, `out` = silu(pre). Taps
  across a row start read zero. Its backward, `conv_bwd(dout, pre, wconv, …) → dqkv`, follows D§4.8 (the
  transposed conv, which also must not cross row ends).
* The conv weight in the checkpoint is [6144,1,4]. `wconv[ch*4+k]` multiplies `x[t-3+k]` (D§3.6, verified against
  `F.conv1d` with padding 3 and the output truncated to L).

**Done when:** gradcheck passes for both backwards. Run a ragged test where row 2's first 3 outputs must not change
if you randomise row 1's last 3 tokens. That check exists because forgetting the row boundary is the classic bug
here.

### M2.5 Attention softmax and head → `kernels/attn.cu`, `kernels/head.cu`
* `attn_softmax_fwd(S, P, L, heads, scale)`: rows (h, t), causal (s > t → P = 0), max-subtracted, fp32.
* `attn_softmax_bwd(P, dP, dS, L, heads, scale)`: `dS = P*(dP − Σ P·dP) * scale`.
* `ce_head(logits, n_opts, targets, wrow, Wgroup, B, loss_out, dlogits, probs_out)`: D§4.1 (dead options set to
  −1e9 before the softmax). It writes the per-row loss, the dlogits, and the probabilities for predictions.
  Logits come from a gemm (readout); this kernel does only the softmax part.
* Test `attn_softmax` with L ∈ {1, 2, 300, 1024}, and test `ce_head` with n ∈ {1, 2, 255}, including a row with n=1
  (its loss must be 0 and its dlogits 0).

**Done when:** gradcheck and the GPU tests are green.

### M2.6 Optimizer kernels → `kernels/adam.cu`, `src/lora.c`
* `sumsq_partial(g, n, partials)` + `sumsq_final(partials, nb, out)`: two passes, a fixed block count (e.g. 1024),
  and no atomics.
* `clip_scale(g, n, coef)`.
* `adamw_step(p, g, m, v, n, lr_group_split, lr_lora, lr_readout, b1, b2, eps, step_t)`: elements with index <
  `lr_group_split` use `lr_lora`; the rest (the readout sits at the end of the flat buffer) use `lr_readout`. The
  formula is D§2.2's word for word. Compute `bias1 = 1 − b1^t` and `bias2 = 1 − b2^t` on the host in double and
  pass them as floats.
* `src/lora.c`: the flat buffer layout. It is a table of 151 entries (150 LoRA pairs plus the readout) with offsets,
  shapes and stems, generated from the D§13 model table. Functions: `lora_alloc`, `lora_load_adapter(dir)`
  (reads a Python-written `adapter_model.safetensors` + `readout.safetensors`; the readout may be bf16 or fp32, so
  convert), `lora_save_adapter(dir, step, dev_ce)`, `lr_at(step, total, base)` (D§2.2's schedule, in double).
* Test: compare one Adam step against a C ref of torch's formula, then 10 steps with random grads (rel 1e-6). Check
  that `lr_at` matches a table printed by Python (put a 10-line Python snippet in the test's comment that prints
  `f(s)` for total=100 and s=0..100). The clip must match the ref's norm to 1e-6.

**Done when:** `test_kernels adam` is green. Also check the total count: printing the size of `lora_alloc` must give
`trainable 10,484,736`.

---

## M3. The GDN recurrence

Read D§3.6, D§4.8 and D§8.3 completely before starting. This milestone is the only real algorithmic risk.

### M3.1 `ref/gdn.c`: the oracle
```c
// dims as parameters so gradcheck can shrink them. q,k,v,o: [N][H][Dk or Dv]; g,beta: [N][H]; off: [B+1]
void ref_gdn_fwd(int B, const int *off, int H, int Dk, int Dv,
                 const real *q, const real *k, const real *v, const real *g, const real *beta, real *o);
void ref_gdn_bwd(int B, const int *off, int H, int Dk, int Dv,
                 const real *q, const real *k, const real *v, const real *g, const real *beta, const real *dout,
                 real *dq, real *dk, real *dv, real *dg, real *dbeta);
```
The forward is D§3.6's loop written literally, with S as a local `real S[Dk*Dv]` (malloc'd) reset at each row
start. The backward first runs the forward, storing S_prev for every t (it's a reference, so memory doesn't matter),
then follows D§4.8's reverse loop line for line, **in the stated order**. Keep comments that quote each D§4.8 line
next to its code.

**Done when:** gradcheck passes in double with H=2, Dk=Dv=8, B=2 rows of lengths {5, 3}, for all five input
gradients. Use g in [-1, 0] (decays are ≤ 1) and beta in [0, 1].

### M3.2 `kernels/gdn.cu`: forward
Build it as D§8.3 describes: grid (B, 16, 4), 32 threads, `float s[128]` per lane. Two details matter:
* **Registers:** `s[]` stays in registers only if every index into it is a compile-time constant after unrolling.
  Write the two inner loops as `#pragma unroll` over `i = 0..127` with `Dk=128` as a template or `#define`
  constant. Never index `s[]` with a runtime variable. Verify with `nvcc -Xptxas -v kernels/gdn.cu` on gpubox: the
  line for `gdn_fwd_kernel` must say `0 bytes spill stores, 0 bytes spill loads`. Record the register count in a
  comment.
* **Checkpoints:** `ckpt` is a device buffer `[B][16][max_segs][128][128]` (`max_segs = ceil(max_len / 64)`).
  Lane j of tile `tile` writes column `tile*32 + j`. Write it *before* processing a token whose
  `t_local % 64 == 0` (so segment 0's checkpoint is all zeros).

The kernel reads q, k, v, g and beta in the layout the upstream kernels produce: q, k and v as [N][16][128] slices
of the conv output (after l2norm), and g and beta as [N][16].

**Done when:** `test_gdn fwd` on gpubox, with the full dims (H=16, D=128) and ragged batches {1}, {2}, {63, 64, 65},
{200, 7, 130}, {1026, 520, 670, 300, 811, 455, 610, 702}, gives `rel < 1e-4` against `ref_gdn_fwd` (float build),
and the spill check is clean.

### M3.3 `kernels/gdn.cu`: backward
Follow D§8.3's phase A / phase B text exactly. Supporting buffers (allocated once by the caller):
* `sp`: `[B*16*4][64][128][32]` floats, the per-block S_prev scratch for one segment (1 MB per block).
* `dq_part`, `dk_part`: `[4][N][16][128]`; `dbeta_part`, `dg_part`: `[4][N][16]`.
* `gdn_bwd_reduce` sums the 4 tile partials in order 0, 1, 2, 3 into dq, dk, dbeta and dg.

Write it in this order, testing at each stage:
1. Phase A only (recompute) writing S_prev to `sp`. Test: for one (b, h), compare `sp` for segment 0 with the
   ref's stored S_prev.
2. Phase B for dv and dq only (the do terms). Test dq and dv against the ref with dk, dg and dbeta ignored.
3. Add ddelta, dk, dkv, dbeta and dg. Test everything.

**Register budget (read before writing phase B).** The zero-spill gate is only reachable if each thread keeps
at most two 64-float arrays live: `ds[64]` (carried across the whole loop) and, if the count allows, `sd[64]`.
* Don't declare `sp[64]` or `st[64]`. Re-read `sp_i` from the `sp` scratch (global, L1/L2-cached) in each pass
  that needs it, and compute `sd_i = sp_i * e` and `st_i = sd_i + k_i * delta_j` on the fly inside the loop that
  uses them.
* Structure phase B for one token as separate `#pragma unroll` loops over the warp's 64 rows, one per D§8.3 line
  that needs a row loop. Each loop reads what it needs and writes its result immediately.
* Write each `warp_sum` result as soon as it's computed: lane 0 stores `dq_part[...][64w+i]` inside the loop. Never
  collect 64 sums in a local array first.
* If `-Xptxas -v` still shows spills with `ds` + `sd`, drop `sd` too (recompute it from `sp`). If it spills with
  `ds` alone, split each warp's 64 rows into two passes of 32 (4 warps per block instead of 2) and report it, since
  that changes D§8.3.

Warp-level helpers (put them in `kernels/gdn.cu`, `static __device__`):
```c
__device__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
```
For "per row i, sum over the 32 lanes", call `warp_sum(st[i] * do_j)` for each i of the warp's 64 rows, and
let lane 0 keep the result: lane 0 writes `dq_part[tile][t][h][64w + i]`. Compute dk's sum the same way.

Cross-warp column sums (kv_j and ddelta_j): each warp computes its 64-row partial per lane, then
`kvp[w][lane] = partial; __syncthreads(); total = kvp[0][lane] + kvp[1][lane]; __syncthreads();`. The second
`__syncthreads()` is required before kvp is reused.

**Done when:** `test_gdn bwd` gives `rel < 1e-4` for all five gradients on every M3.2 batch, and the spill check is
clean for `gdn_bwd_kernel`. Also time it: one call at B=8, lengths p50-ish (about 520), on gpubox. The design
budget is about 2 ms per layer for fwd + bwd. Write the number down. If it's over 10 ms, look for spills first.

---

## M4. The full forward on real weights

### M4.1 Python references → `py/make_adapter.py`, `py/dump_ref.py`, `py/compare_layers.py`
These run in CLM-8B's venv (Mac: `~/Downloads/CLM-8B/.venv`; gpubox: `~/CLM-8B/.venv-cuda`) and take
`--clm <CLM-8B dir>` to import `tools.jeff_lora.{lora,train,prompt}` from there.
* `make_adapter.py --model M --seed S --out DIR`: attach LoRA with `lora.attach_lora(model, 16, 32)`, set every A
  to U(-1/√in, 1/√in) and every B to N(0, 1)·1e-2 from `torch.Generator().manual_seed(S)`, set the readout to the
  model's own, and save with CLM-8B's `save_lora` (so the file format is the real one by construction).
* `dump_ref.py --model M --adapter DIR --fixture f.jtb --out ref.safetensors [--device cpu|cuda]`: D§10.2. Before
  the forward, replace the GDN rule: set `modeling_qwen3_5.torch_chunk_gated_delta_rule` **and** the same name on
  every `Qwen3_5GatedDeltaNet` instance (the `use_kernelized_func` decorator may attach it there) to `literal_gdn`,
  a plain torch loop of D§3.6 (q/k l2norm inside it when `use_qk_l2norm_in_kernel=True`, as the original does; read
  the original's signature and return value and match them). Count its calls and assert 18 per forward. Read the
  2-row fixture (`make_fixture.py --rows 2` with lengths forced to 37 and 70: add `--lengths 37,70`). Build the
  right-padded batch with train.py's `collate`. Run `option_logits` + `soft_ce` + the weighted loss exactly as
  train.py's loop does, in fp32, with `attn_implementation="eager"`. Register forward hooks on
  `model.layers[i]` that save the output **with the padding removed** (concatenate the rows' real tokens, so the
  layout is the packed [N,1024] jefftrain uses) and call `retain_grad()`. After backward, save everything in D§10.2's
  list. Also save `layer.{i}.in` (the input hidden state) so a single layer can be tested in isolation.
* `compare_layers.py a.safetensors b.safetensors [--atol-rel 1e-3 --grad-rel 5e-3]`: for every key in both files,
  print `key  rel  ok/FAIL` in file order. The **first** failing key is what matters, so also print it on its own
  line at the end.

### M4.2 `src/model.c`: load
* The model table (D§13), loaded with `st_get` and an explicit expected shape for every tensor. Convert to fp32 on
  the card (`bf16_to_f32` into one big weight arena), except `embed_tokens`, which stays bf16.
* `A_log` and `dt_bias` are converted on the host to fp32 and precomputed as `negA = -exp(A_log)` (fp32).
* Readout: from the base model's `readout.safetensors` (bf16) into the flat parameter buffer's readout slot as fp32,
  unless `--init-adapter` gives one.
* Print one line at load: `model: 24 layers (18 gdn, 6 attn), weights 2.0x GB fp32 + embed 0.51 GB bf16`.

### M4.3 Forward pieces
Write them in this order, each as `*_fwd(ctx, layer, x_in, N, batch, x_out)` that adds into the residual. Test each
one against `layer.{i}.in` → `layer.{i}.out` from `ref.safetensors` before writing the next.
1. `linear_fwd` (D§4.2), saving `u` when a save pointer is given.
2. `layer_mlp.c`: the post-attention norm and the MLP, D§3.4's last two lines.
3. `layer_gdn.c`: D§3.6, kernel by kernel: `rmsnorm` → 4 linears (qkv and z with LoRA, a and b without) → `conv_fwd`
   → split into q/k/v views (no copy; [N][6144] with q at column offset 0, k at 2048 and v at 4096; the l2norm kernel
   takes a row stride) → `l2norm` (q with post_scale 1/√128, k with 1) → `gdn_gates_fwd` → `gdn_fwd` →
   `gated_rmsnorm_fwd` (gate = z) → `out_proj` → residual add. Then the MLP.
4. `layer_attn.c`: D§3.5: `rmsnorm` → `q_proj` → `qgate_split` → `k_proj`, `v_proj` → q_norm and k_norm (row
   length 256, rows = N·heads) → `rope_fwd` → per row: `to_head_major`, `kv_expand`, scores gemm, softmax, PV gemm,
   `from_head_major` → `sigmoid_gate_fwd` → `o_proj` → residual add. Then the MLP.
5. `head.c`: final norm → gather the last token of each row (`off[b+1]-1`) → readout gemm → `ce_head`.

The per-layer test is `test_layer --ref ref.safetensors --layer i --model M --adapter DIR`. It loads only what it
needs and compares the layer's output, and for the head, logits and loss.

**Done when:** on gpubox, built with `JT_EXACT=1`, the whole forward (`jefftrain --dump-forward out.safetensors`,
writing the same keys as dump_ref) passes `compare_layers.py` with `--atol-rel 1e-3` for every `layer.{i}.out`,
`final`, `logits` and `loss`. Make `ref.safetensors` with `--device cpu` (fp32, eager attention, the literal GDN loop from M4.1).

If layer 3 (the first attention layer) is the first failure, suspect, in this order: the q/gate interleave, the RoPE
positions (local vs packed), the head-to-group mapping (h/4, not h%2), and the q_norm offset (1+w).
If layer 0 fails: the conv tap order, the l2norm being sum and not mean, the gated norm being w and not (1+w),
and the A_log sign.

---

## M5. Backward, optimizer, one training step

### M5.1 Backward pieces
Write a `*_bwd` for each M4.3 piece, in reverse order (head first), each tested against `ref.safetensors`'s
`grad.*` keys (or a finite-difference check of the GPU forward if a key isn't there):
1. `head_bwd`: `ce_head`'s dlogits → `dR += dlogits^T z` (gemm(1,0), beta=1) → `dz = dlogits R` → scatter dz
   into a zeroed `dh[N,1024]` at each row's last token → final rmsnorm backward. Test: `grad.readout` and
   `grad.final` (add `final` to dump_ref's retain_grad list).
2. `linear_bwd` (D§4.2) with all six GEMMs from the D§5 table. Test in isolation first: a GPU finite-difference
   check on a [37, 64] → [48] linear in exact mode, then `grad.{stem}.lora_A/B` of layer 23's MLP.
3. `layer_mlp` bwd; then `layer_gdn` bwd (D§4.8: gated norm bwd → `gdn_bwd` → gates bwd → l2norm bwd → conv bwd →
   four linear bwds summed into da → rmsnorm bwd → residual); then `layer_attn` bwd (D§4.7 and D§8.2).
4. The layer loop with recompute (D§6.2): forward saves only `x_in[i]` for i = 0..24. Backward for i = 23..0:
   `layer_fwd(i, save=1)` into scratch, then `layer_bwd(i)`. The dx of layer i is the dy of layer i−1.

**Done when:** `jefftrain --dump-backward out.safetensors` (exact build) passes `compare_layers.py --grad-rel 5e-3`
for every `grad.layer.{i}.out`, all 300 LoRA grads, and `grad.readout`. If one LoRA grad fails while its layer's
output grad passes, the bug is in `linear_bwd`'s dA/dB lines (usually the `s` factor or a transposed `u`).

### M5.2 The training step → `src/main.c`
```
zero grads (one launch on the flat buffer)
for each batch in the accumulation group (Wgroup = sum of row weights over the group):
    upload ids, off, row_of, targets, n_opts, weights for this batch
    forward (saving x_in), head (loss, dlogits scaled by w_row/Wgroup), backward
norm = sqrt(sumsq(grads)); coef = 1/(norm+1e-6); if coef < 1: clip_scale
lr multiplier f(step) (D§2.2), adamw_step with t = step+1
step += 1
```
Log exactly like train.py (the loop in CLM-8B `train.py` around its `if step <= 3 or step % 25 == 0` line): the
running mean loss `run_loss/run_n` is the unweighted mean of per-row CE since the last dev check. `peak` is
`cudaMemGetInfo`'s total − free, the maximum seen. Add `--log-every N` (default: train.py's rule). `--log-every 1`
prints every step, which M6 needs.

The dev check, adapter saving, best-adapter reload and `--eval` scoring follow train.py's `main()` and `evaluate()`
line by line:
* the eval batch size is `batch*2` for dev and `max(2, batch*2)` for `--eval`, with no shuffle (prep.py writes
  those batch lists too);
* the agreement is `argmax(logits) == teacher_top`;
* predictions use `softmax(logits / T)` over the row's own n options, rounded to 6 decimals, in input order.

`jefftrain`'s flags: `--model --train --dev --eval NAME=PATH --output --lr --readout-lr --epochs --batch --accum
--eval-every --max-steps --eval-temperature --eval-only --init-adapter --log-every --dump-forward --dump-backward`.
`--train/--dev/--eval` take `.jtb` files. Reject a flag combination that disagrees with the .jtb header (batch,
accum, epochs).

### M5.3 `py/prep.py`
D§9.1. The RNG facts from CLM-8B `train.py` `main()`, which prep.py reproduces:
* `train_rows` subset: `random.Random(seed).sample(rows, k)` (its own Random, not the shared rng);
* `dev_rows` subset: `random.Random(seed + 1).sample(...)`;
* then `encode()` and, with `--weights balanced`, `balance()`;
* the shared `rng = random.Random(seed)` is consumed **only** by `batches(train, batch, rng, shuffle=True)`, called
  once per epoch, in epoch order;
* dev and eval use `batches(rows, size, random.Random(0), shuffle=False)`.

Tokenisation must be train.py's: call its `encode(rows, decision, tok, hard)` with the same tokenizer and
`decision_config` loading (`load_decision_config(model_path)`). Don't reimplement it.

**Test** (`py/trace_train_order.py`, on gpubox, a few minutes): import `tools.jeff_lora.train` as `T`; wrap
`T.batches` so each call appends its result's row ids to a list; run `T.main([... --train-rows 64 --dev-rows 16
--epochs 2 --batch 8 --max-steps 16 --output /tmp/x ...])`; dump the list as JSON. Then compare it with
`prep.py`'s batch lists for the same arguments: they must be identical, call by call.

**Done when:** the trace matches for seeds 0 and 1, with `--weights none` and `balanced`.

---

## M6. End-to-end parity and timing on gpubox

Everything here reads real data, so it runs on gpubox inside `~/CLM-8B` (paths from CLM-8B
`tools/jeff_lora/cuda/env.sh`: `$DATA`, `$VENV`, `$REMOTE_MODEL`). Outputs go to `~/CLM-8B/data/ovril_detect/runs/`.
macuda's scripts (`scripts/parity-eval.sh`, `parity-steps.sh`, `arm.sh`) only orchestrate: they `cd ~/CLM-8B`, call
prep.py and `~/macuda-train/jeff/build/nv/jefftrain`, and print a verdict line. Ask the user before each of M6.3
and M6.4 (they hold the GPU for a while).

### M6.1 Eval parity (minutes)
`prep.py --eval test=$DATA/test.jsonl --batch 4` → `test.jtb`. Then `jefftrain --eval-only --eval test=test.jtb
--eval-temperature <T>`, where T = `decision_config.json`'s temperature (read it with a one-line python in the
script; CLM-8B's `load_decision_config` knows where it lives). Then
`compare_preds.py out/pred-test.jsonl <the kit's runs/cuda-parity-fp32-*/pred-test.jsonl> --max-dp 0.05
--min-agree 158`. Run it with the TF32 build and the exact build.

**Done when:** both builds pass with the same gate the PyTorch port passed. Write `max |dp|` and the agreement into
the change description.

### M6.2 Step parity (minutes)
* Python: `py/trace_train_loss.py` (the M5.3 trick: wrap `T.soft_ce`, and when `torch.is_grad_enabled()`, append the
  per-row values to the current step's list) running `T.main([--dtype fp32 --init-adapter A --train-rows 256
  --dev-rows 64 --batch 8 --max-steps 20 --eval-every 20 ...])` with A made by `make_adapter.py`.
* C: the same rows through prep.py, the same A, `jefftrain --log-every 1 --trace-loss steps.jsonl`, where
  `--trace-loss` writes each step's per-row CE in batch order.
* `py/compare_steps.py py.json c.jsonl`: per step, the mean of per-row CE.

**Gate** (D§10.3.2): relative difference < 1e-3 for steps 1–5 and < 1e-2 through step 20, and the dev soft CE at
step 20 within 2e-3. Step 1 depends only on the forward. If step 1 matches and step 2 doesn't, the bug is in the
gradients or Adam, so go back to M5.1's dump.

### M6.3 Timing (10 minutes)
`jefftrain` on the full train split, `--max-steps 100 --batch 8`, printing the same TIMING line as the kit's
`timing.sh`. Compare with PyTorch's 0.143 s/step and the D§15 target (< 0.4 s/step). Write the number down.
Being slower than PyTorch is expected for v1 and is not a failure. M8 is where speed gets work.

### M6.4 Run parity (one arm, about 30 min; ask first)
`scripts/arm.sh soft`: the kit's `arms.sh` soft arm settings (1 epoch, batch 8, dev every 500 steps on all of
development, eval calibration and test at T=1), with jefftrain doing the training. Then CLM-8B's
`fetch_and_score.sh` path scores it next to the Python arm.

**Done when:** D§10.3.3's gates hold (dev soft CE at the chosen step within 0.01 of the Python soft arm, test soft CE
within 0.02, score.py metrics inside the two-seed noise band). Report the table to the user.

---

## M7. The shim: macuda trains Jeff

Up to now, any shim-specific failure was impossible by construction. Now the binary changes and nothing else does.

### M7.1 Build and null-device run
`make shim ARCH=<ask>` and `make check-symbols` (every `cuda*`/`cublas*` symbol defined). `make test-null`: every
test reaches its end with no CUDA error (on the null device the numbers are meaningless; tests skip their value
checks unless `TINYNV_HW` and `TINYNV_SOCKET` are both set, as M0.2 set up). Then `jefftrain --eval-only` on a fixture with a null device runs
start-up, allocation and the launch sequence end to end.

**Done when:** all three are green. If a symbol is missing, don't add it to the shim: change jefftrain to use an API
from D§12's list, or ask the user.

### M7.2 Kernel and GEMM tests on the card (ask the user first, every time)
```
sh tools/nv_shim_step.sh <A|B> probe train/jeff/build/shim/test_gemm
sh tools/nv_shim_step.sh <A|B> probe train/jeff/build/shim/test_kernels all
sh tools/nv_shim_step.sh <A|B> probe train/jeff/build/shim/test_gdn all
```
Run `test_gemm` with `JT_EXACT=1 TINYCUBLAS_TC=0` and again in the default mode, and specifically check
`gemm_batched` with TF32 compute (D§12's open question about strided batched + TF32 on the shim). If tinycublas
returns NOT_SUPPORTED there, report it to the user. Don't work around it silently.

### M7.3 Model parity on the card
The Mac has the model in the HF cache and the fixtures. Copy `ref.safetensors` from gpubox (it's made from synthetic
rows and a public model, so it may sit in `/tmp` on the Mac but still not in the repo). Run `--dump-forward` and
`--dump-backward` through `nv_shim_step.sh` and compare with `compare_layers.py`, using the same gates as M4/M5.
Then eval parity: the 160 test rows. The .jtb is real data, so it lives under CLM-8B's `data/`, and the binary reads
it from there. Compare against gpubox jefftrain's `pred-test.jsonl` (expect max |dp| < 0.01).

### M7.4 A training run on the card (ask; it holds the card for hours)
20 steps first (step parity against gpubox jefftrain's `steps.jsonl`), then, if the user wants it, one full arm, timed.
Report s/step, peak memory and the D§10.3.3 metrics.

---

## M8. Speed (only after M6 passes; each item is its own change, re-gated by M6.1 + M6.2)

In expected order of payoff:
1. **bf16 frozen weights + bf16 GEMM inputs** where libtinycublas's bf16 fast path qualifies (D§6.1's alignment
   rules), keeping fp32 activations elsewhere. Halves weight memory and bandwidth.
2. **Chunked GDN** (fla's WY form): the recurrence becomes GEMMs over chunks of 64. Big win for the shim, where
   tinycublas GEMMs are the strong path.
3. **Fused kernels**: rmsnorm + the following linear's input cast; silu_mul into the down_proj prologue; the
   attention softmax fused with the scores (flash-style) for L > 512.
4. **No recompute** when memory allows (`--no-recompute`, which saves every layer's scratch): −25% time on the 5090.
5. **CUDA-graph-free launch batching** on the shim: fewer, larger launches. Measure launch overhead first
   (`cuda-shim/test/launch_copy_bench.c`).

---

## Appendix A. When stuck

| symptom | look at |
|---|---|
| NaN in the forward | `expf` of a positive g (g must be ≤ 0: check the `-exp(A_log)` sign); a softmax without the max subtraction; rsqrt of 0 (eps missing) |
| NaN only in the backward | the attention dS on fully masked entries (P must be exactly 0 there); dividing by Wgroup = 0 |
| layer 0 off by a few % | conv tap order; l2norm sum vs mean; the gated norm w vs (1+w) |
| layer 3 off, layers 0–2 fine | q/gate interleave; RoPE positions; head→group h/4 |
| fine at batch 1, wrong at batch 3 | row boundaries in conv/GDN/RoPE; `row_of`/`off` upload; head gather index `off[b+1]-1` |
| gradients right, loss diverges over steps | Adam bias correction t (starts at 1); lr schedule s (starts at 0); clip applied after the norm |
| run-to-run differences in exact mode | a float `atomicAdd` somewhere; a reduction whose order depends on the block count |
| works on gpubox, fails on the shim | `make check-symbols`; `JT_DEBUG=1` to find the first failing launch; dynamic smem > 48 KB without `cudaFuncSetAttribute` |

## Appendix B. Done-ness checklist (copy into the final change description)

- [ ] M0 both builds, check-symbols
- [ ] M1 GEMM 4 modes × exact/TF32; safetensors listing matches D§3.2; JTB reader
- [ ] M2 every kernel: gradcheck + GPU vs ref
- [ ] M3 GDN fwd/bwd vs ref, zero spills, time per layer: ___ ms
- [ ] M4 forward vs dump_ref, all layers < 1e-3
- [ ] M5 backward vs dump_ref, all grads < 5e-3; prep.py order == train.py order
- [ ] M6 eval parity ___/160, max dp ___; step parity; s/step ___; run parity table
- [ ] M7 shim: tests on card; eval parity vs gpubox; 20-step parity; (optional) full arm s/step ___
