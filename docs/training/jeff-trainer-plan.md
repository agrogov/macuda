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
   OK each time. A card checkpoint (`scripts/card-check.sh`, M0.4) is one run for this purpose: one OK covers all its
   steps. Ask which card in the same message, because `make shim` needs `ARCH=`. gpubox runs (192.168.31.162, `ssh gpubox`) need an OK for anything long (over about 10 min) or anything
   that takes the whole GPU.
7. **Commit only when the user asks.** macuda's main checkout has other people's *staged* files, and a plain
   `git commit` on a new branch would sweep them in. Work in a separate worktree, which has its own index:
   `git -C macuda worktree add ../macuda-train-jeff -b train-jeff main`, and do all of this plan there. If you must
   commit from the main checkout, use `git commit -- <your paths>` only. Never add AI attribution to commits.
8. Code style: C11, `-Wall -Wextra` clean, no global mutable state except the context struct, every CUDA/cuBLAS
   call wrapped in `CK()`/`CB()` (D§12), every function under about 80 lines. Names follow the design's tables.
9. **Notes are the hand-off, and they are checked.** A task's last act is `train/jeff/notes/<task>.md`, whose first
   line is exactly `STATUS: PASS` — followed by the commands it ran, a short summary of their output and every number
   the plan asks to record — or `STATUS: BLOCKED — <reason>`, followed by the command, the output and a best guess;
   parked work is saved as `notes/<task>.wip.patch` and every other change reverted, so only the notes land. Before
   anything else a task runs `sh train/jeff/scripts/prereq-check.sh <task ids…>`, which greps those first lines and
   prints `prereq: J34 J42 PASS` (exit 1 naming the note that is missing or not `PASS` otherwise). That is how a
   *Done when* is carried forward, and Appendix B is the same information in one place.

## Milestones at a glance

| M | what | proves | where it runs |
|---|---|---|---|
| M0 | scaffold, both builds of a trivial kernel | toolchains and link line | Mac (null device), gpubox |
| M1 | `gemm.c` row-major wrappers + `st.c` + `jtb.c` | the GEMM contract, file readers | gpubox, then a card check of the GEMMs; Mac null device |
| M2 | elementwise/row kernels + CPU refs + gradchecks | each small op, fwd and bwd | gpubox, then one card check |
| M3 | GDN recurrence fwd/bwd | the hard kernel | gpubox, then card checks after fwd and after bwd |
| M4 | full forward on real weights vs `dump_ref.py` | layouts, model assembly | gpubox, then one card check |
| M5 | backward, LoRA, Adam, schedule, prep.py batch order | one training step bit-close to PyTorch | gpubox, then one card check |
| M6 | eval/step/run parity, timing | the whole trainer | gpubox |
| M7 | eval parity, 20-step parity, timed arm on the shim | macuda can train Jeff | Mac + eGPU |
| M8 | speed (bf16 weights, chunked GDN, fused kernels) | optional | both |

gpubox is the development box for M1–M6 because it has real nvcc and cuBLAS, so a failure there is jefftrain's
own fault. Each piece passes on gpubox first and then goes to the card at the next checkpoint, so a failure that
appears only on the card is a shim question. A Mac null-device build is kept green from M0 onward
(`make shim test-null`), which catches link and launch errors between checkpoints.

Every gpubox script is told which remote tree it works in: `JT_REMOTE` (default `macuda-train/jeff`) is the path under
`~` on gpubox, e.g. `export JT_REMOTE=macuda-train/wt-J34`, so parallel tasks never rsync over each other's build.
`~/macuda-train/ref/` (D§10.1's reference set) is read-only. M6 and M7's scripts use `~/$JT_REMOTE/build/nv/jefftrain`
for the same reason.

**Card checkpoints** come early because two things could force a kernel redesign: tinycublas rejecting strided-batched
TF32, and tinycc's build of the GDN kernels behaving differently. The rules for every checkpoint:
1. Run it only after that piece's gpubox gate has passed. Then `make shim ARCH=<card> && make test-null &&
   make check-symbols`, ask the user (rule 6), and run the `card-check.sh` line the milestone gives. As built, every
   checkpoint ran on card **B**: RTX 4090, PCI `0x2684`, `ARCH=sm_89`,
   `SHIM=$MACUDA/cuda-shim/build/shim-sm_89` — never `sm_120a`, which is the 5090. `build/shim/.arch` records which
   card an object was built for (M0.1), and the step leaves the card quiesced.
2. The gates are the same numbers as on gpubox, and the shim build has no `-DJT_EXACT` (exact mode comes from the
   environment, so one build serves both modes).
3. While waiting for the OK, you may carry on with gpubox work, except for what the checkpoint says has to wait for
   its result. Don't skip the checkpoint; leave its Appendix B box empty.
4. If a checkpoint fails, stop and report it to the user with the log path before starting the next milestone. Say
   which step failed and that the same test passes on gpubox. Don't change the shim (rule 4), and don't change or
   weaken a kernel, gate or compute mode to make the card pass unless the user agrees.
5. M6 has no checkpoint: its reference runs are on gpubox.
6. The two dump checkpoints disagree about key sets, and that is not a failure: `--dump-forward` writes the 77
   forward keys while `JT_REF` is the 403-key `ref.safetensors`, so m4's `compare_layers.py` line prints
   `77 ok, 326 failed` and `checkpoint m4: 1 steps, 1 failed` until it is pointed at a forward-only extract of the
   same file. m5 is self-consistent (`--dump-backward` writes all 403). See M4.4.

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

`MACUDA ?= $(HOME)/Downloads/vfio/macuda` is the main checkout. The shim's libraries, `cuda-shim/cuda-13/include` and
the tools come from there, since the worktree of rule 7 has no git-ignored build output (D§11.1).
`ARCH ?= sm_120` for `nv`. For `shim`, `ARCH` has no default: `make shim` without `ARCH=` prints
"set ARCH=sm_120a (5090) or sm_89 (Ada, with SHIM=.../shim-sm_89); ask which card" and fails.

As built, each build directory records the `ARCH` it was compiled for (`build/shim/.arch`, `build/nv/.arch`) and drops
its objects when that changes — without it the second card silently ran the first card's cubin. Object names are flat
and distinct, which is what lets `check-symbols` look at every unlinked object: `misc.o` for `kernels/misc.cu`,
`host_misc.o` for `src/misc.c`, `ref_x.o`/`refd_x.o` for the float and double builds of `ref/x.c`, `test_x.o`,
`k_x.o` and `grad_x.o`. Host builds get `-D_GNU_SOURCE` (glibc hides `clock_gettime`/`mmap` under `-std=c11`), the
shim build `-DJT_SHIM`, the CPU build `-DJT_CPU`; `test-cpu` links a `test/test_x.c` with the float references plus
`src/x.c` when it exists, and `gradcheck` is the only binary built with the double references. `test_kernels` links
only `k_*.c` and `gradcheck` only `grad_*.c` (M2.0).

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
  `hello OK (null device: launch only)`). The object is 8688 bytes, its `__nv_fatbin` section 6096, and it carries
  `T __Z29__device_stub__jt_fill_kernelPffm`, `U ___cudaRegisterFatBinary`/`___cudaRegisterFunction` and `T _jt_fill`.

How `shim` compiles one `.cu`, as built: the Makefile copies it to `kernels/.tc_<cksum>_<name>.cu`, where `<cksum>` is
the `cksum` of this worktree's absolute path (so two worktrees' copies, and their `/tmp/tinycc_*.fatbin` files, cannot
collide), and runs
`TINYCC_GGML_MAC=$PWD TINYCC_ARCH=$ARCH TINYCC_SYSROOT=$(sh $MACUDA/cuda-shim/build/sdk.sh) /usr/bin/lockf
/tmp/jefftrain-tinycc.lock sh $MACUDA/cuda-shim/build/tinycc kernels/.tc_…_<name>.cu -o build/shim/<name>.o
-I$PWD/{src,kernels,ref,test} -DJT_SHIM`. The `lockf` is what keeps parallel (`make -j8`) kernel compiles apart; the copy
is deleted again even when tinycc fails, so `kernels/.tc_*` never reaches `git status` (`train/jeff/.gitignore`). The
device half is the container's nvcc (CUDA 13.0.3, one `--fatbin` per TU), the host half Homebrew clang;
`cuda-shim/build/container-runtime.sh` prefers Apple's `container` CLI, so a stopped Docker daemon is not a blocker.

### M0.3 gpubox scripts
* `scripts/gpubox-sync.sh`: `rsync -a --delete --exclude build/ train/jeff/ gpubox:~/$JT_REMOTE/`.
* `scripts/gpubox-build.sh`: sync, then `ssh gpubox 'cd ~/$JT_REMOTE && make nv ARCH=sm_120'`.
* `scripts/gpubox-run.sh <binary> [args]`: build, then run `build/nv/<binary>` there, streaming the output.
* `scripts/prereq-check.sh <task ids…>`: rule 9's check — `prereq: J34 J42 PASS`, or exit 1 naming the note that is
  missing or whose first line is not `STATUS: PASS`.

`JT_REMOTE` defaults to `macuda-train/jeff` and is the path under `~` on gpubox; **export it before every call when
other tasks may be running** (`export JT_REMOTE=macuda-train/wt-J01`), or two tasks' `rsync --delete` and `make nv`
fight over one tree. nvcc is not on a bare ssh PATH there, so the Makefile finds `/usr/local/cuda/bin/nvcc` itself
(13.2, `-arch=sm_120` accepted) and `gpubox-build.sh` also exports the PATH.

**Done when:** `bash scripts/gpubox-run.sh test_hello` prints `hello OK`, and `sh scripts/prereq-check.sh J01` prints
`prereq: J01 PASS`.

### M0.4 `scripts/card-check.sh`
One script runs each card checkpoint, so the user approves one run and not one per test. Write it as given:
```sh
#!/bin/sh
# One card checkpoint; the user's OK covers all its steps. DRY=1 uses the null device: no preflight, no card.
# usage: sh scripts/card-check.sh <A|B> <m0|m1|m2|m3-fwd|m3|m4|m5>
# m4 and m5 also need JT_REF (ref.safetensors) and JT_DUMP_ARGS (--model/--init-adapter/--train flags, Mac paths).
set -u
WHO=${1:?A|B}; CP=${2:?checkpoint}
JEFF=$(cd "$(dirname "$0")/.." && pwd); B=$JEFF/build/shim
MACUDA=${MACUDA:-$HOME/Downloads/vfio/macuda}   # the main checkout: the built shim, tools/ and logs/ are there
PY=${JT_PY:-$HOME/Downloads/CLM-8B/.venv/bin/python}
EXACT="JT_EXACT=1 TINYCUBLAS_TC=0"
n=0; failed=0
finish() { echo "checkpoint $CP: $n steps, $failed failed"; [ "$failed" -eq 0 ] && exit 0; exit 1; }
stop() { echo "STOP: $1"; failed=$((failed+1)); finish; }
# step <env words, or -> <binary> [args...]. A step that never ran, crashed or was void ends the checkpoint.
step() {
  envs=$1; bin=$2; shift 2; [ "$envs" = - ] && envs=
  n=$((n+1)); echo "=== step $n: $envs $(basename "$bin") $*"
  tmp=$(mktemp); env $envs sh "$MACUDA/tools/nv_shim_step.sh" "$WHO" probe "$bin" "$@" > "$tmp" 2>&1; rc=$?; cat "$tmp"
  ran=0; grep -q '   exit [0-9]* after' "$tmp" && ran=1; rm -f "$tmp"
  [ "$ran" = 1 ] || stop "step $n never ran (rc $rc)"
  [ "$rc" -gt 128 ] && stop "step $n crashed (rc $rc): the card's state is unknown"
  [ "$rc" = 97 ] && stop "step $n was void: it did not have the card to itself"
  [ "$rc" = 0 ] || { echo "FAILED: step $n (rc $rc)"; failed=$((failed+1)); }
}
case $CP in
  m0) step - "$B/test_hello" ;;
  m1) step "$EXACT" "$B/test_gemm"; step - "$B/test_gemm" ;;   # exact, then TF32
  m2) step - "$B/test_kernels" all ;;
  m3-fwd) step - "$B/test_gdn" fwd ;;
  m3) step - "$B/test_gdn" all ;;
  m4|m5)
    : "${JT_REF:?path to ref.safetensors}"; : "${JT_DUMP_ARGS:?flags for the dump run}"
    out=$(mktemp -d)/$CP.safetensors; flag=--dump-forward; [ "$CP" = m5 ] && flag=--dump-backward
    step "$EXACT" "$B/jefftrain" $JT_DUMP_ARGS $flag "$out"
    [ "${DRY:-0}" = 1 ] || "$PY" "$JEFF/py/compare_layers.py" "$JT_REF" "$out" --atol-rel 1e-3 --grad-rel 5e-3 \
      || { echo "FAILED: compare_layers"; failed=$((failed+1)); } ;;
  *) echo "unknown checkpoint $CP"; exit 2 ;;
esac
finish
```
`nv_shim_step.sh` comes from the main checkout because the worktree of rule 7 has no built shim (those files are
git-ignored). Its line `WARNING: binary is not a clean build…` and `build id unknown` are expected: it only knows
macuda's own binaries. A real run is `sh scripts/card-check.sh <A|B> <checkpoint>`, from `train/jeff`.

As built (J02), three things to know about it:
* `DRY=1` needs `build/shim/<test>` to exist, so `make shim ARCH=sm_120a` first; it unsets `TINYNV_SOCKET` inside
  `nv_shim_step.sh`, so there is no preflight, no quiesce and no card. `TINYNV_HW=1` without `TINYNV_SOCKET` is the
  null device (M0.1).
* A step that never reached the card (missing binary, missing `JT_REF`, a stub that exits 2) ends the checkpoint with
  `STOP: step n never ran` and exit 1; a step that ran and failed is counted (`FAILED: step n (rc k)`) and the
  checkpoint ends `checkpoint <cp>: n steps, k failed`, exit 1. An unknown checkpoint exits 2.
* **m4's key set.** `--dump-forward` writes the 77 forward keys while `JT_REF` is the full 403-key `ref.safetensors`, so
  the m4 branch prints `77 ok, 326 failed (not in …)` and `checkpoint m4: 1 steps, 1 failed` even though every forward
  key passed. Point `JT_REF` at a forward-only extract of the same file for m4; m5 uses the full file, because
  `--dump-backward` writes all 403 keys. See M4.4.

**Done when:** `DRY=1 sh scripts/card-check.sh A m0` prints `checkpoint m0: 1 steps, 0 failed`, and
`DRY=1 sh scripts/card-check.sh A nope` exits 2. Don't run m0 for real: the M1 checkpoint is the first one on the card.

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
Gate: `rel < 1e-5` with JT_EXACT=1; `rel < 5e-3` with TF32. Print one line per case plus `GEMM OK n/n`, and exit 1
if any case failed.

**Done when:** on gpubox, `JT_EXACT=1 build/nv/test_gemm` and `build/nv/test_gemm` both print `GEMM OK`. Mac:
`make shim test-null` is still green.

### M1.2a Card checkpoint: GEMMs (the first run on the card; ask for it now)
`sh scripts/card-check.sh <A|B> m1`, under the rules in "Milestones at a glance". It runs `test_gemm` in exact mode,
then in TF32, including `gemm_batched`.

This is the check for D§12's open question: does tinycublas accept `cublasGemmStridedBatchedEx` with
`CUBLAS_COMPUTE_32F_FAST_TF32`? If it returns NOT_SUPPORTED, report that and wait. The attention code
relies on this call, so the user decides what happens next.

**Answered (card B, G-m1): yes.** `GEMM OK 112/112` in both modes, 0 `NOT_SUPPORTED`, so `layer_attn.c` (M4.3) is not
blocked. Worst case in exact mode 3.07e-06 (1e-5 gate), in TF32 4.85e-04 (5e-3 gate).

M1.3 through M3 can go ahead while this waits. `layer_attn.c` (M4.3) can't start until the result is in.

**Done when:** both steps print `GEMM OK`, and `checkpoint m1: 2 steps, 0 failed`.

### M1.3 `src/st.c`: safetensors
* `st_open(path) → St*`: `mmap` the file; parse the u64 header length and the JSON header. A tiny hand-written
  JSON scanner is enough: the header is one flat object of `"name": {"dtype": "...", "shape": [...],
  "data_offsets": [a, b]}` plus an optional `"__metadata__"` object, which you skip. Store an array of
  `{name, dtype, ndim, shape[ST_MAX_DIMS], ptr, nbytes}`, with **`ST_MAX_DIMS 8`**, not 4:
  `visual.patch_embed.proj.weight` is `[768,3,2,16,16]` in the same header, and a 4-dim array would reject a valid
  file, since the reader registers every tensor it finds (that is what makes `--list` show `visual.*`).
* Validate more than the list above, as `src/st.c` does: dtype, `data_offsets` against the shape, duplicate names,
  junk after the JSON, a header length that does not fit the file, and shard completeness — every failure `exit(1)`
  naming the tensor and both sides. As built the file is ~800 lines for that reason; don't simplify it back to 80.
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
* `test/test_kernels.c` takes a group name (`misc`, `norm`, `act`, `rope`, `conv`, `attn`, `head`, `adam`) or `all`, and
  `test/test_gdn.c` takes `fwd`, `bwd` or `all`. The card checkpoints call them with those arguments.
* The group files are one per owner, so no two tasks share one: `test/k_<group>.c` holds a group's GPU-vs-reference
  checks and `test/grad_<group>.c` its CPU gradchecks, while `test/test_kernels.c` and `test/gradcheck.c` are the
  frozen dispatchers that link them (`k_*.c` only into `test_kernels`, `grad_*.c` only into `gradcheck`). A group
  function returns exactly `JT_OK`, `JT_FAIL` or `JT_NOT_IMPL` — never a failure count — and ends
  `return jt_exit_code();`. That is how a dispatcher tells a stub from a real group: `test_kernels <group>` and
  `gradcheck <group>` exit 2 on a stub; `test_kernels all` fails (1) while any group is a stub, which is exactly what
  card-check `m2` runs; `test_kernels` with no arguments runs the implemented groups, skips the stubs and exits 0 (that
  is what `make test-null` runs); `gradcheck all` prints `SKIP <group> (not implemented)` and exits 0, but a real
  failing group still fails it.
* Gates live in `test/harness.h` as `JT_TOL_FWD 1e-5`, `JT_TOL_BWD 1e-4` and `JT_TOL_GRAD 1e-6`, beside `rnd_fill`,
  `rel_err`, `CHECK_REL`/`CHECK_ABS`, `dev_upload`/`dev_download`, the bf16 helpers, `jt_ragged` (the two ragged
  batches below) and the gradcheck core `jt_fd_grad`/`jt_grad_check` (central difference `h = 1e-6` over every
  element, `rel < 1e-6`). On a shim build without both `TINYNV_HW` and `TINYNV_SOCKET`, `CHECK_REL` and its siblings
  print `ok <name> (launch only)` and count nothing.
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
bf16 → fp32 is `bits << 16`, nothing else. The n-element widening is `jt_bf16_to_f32_n` (the `_n` because
`test/harness.h` owns the scalar `jt_bf16_to_f32`), and it reads a bf16 table in place — never convert the whole
248320x1024 embedding to fp32.

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
lines); `gdn_gates_fwd(bb, aa, A_log, dt_bias, beta, g, N, heads)` / `gdn_gates_bwd` (D§4.8's last lines; `heads` is
a parameter because beta and g live per (token, head) and `A_log`/`dt_bias` are `[heads]`, so gradcheck can shrink it);
`softplus` as torch does it: `u > 20 ? u : log1pf(expf(u))`. Every wrapper takes a `cudaStream_t s` last (0 = the
default stream), and the naming follows the file that owns it: `jt_fill` came first, so `kernels/misc.cu`'s copy,
sum and gather ops are `jt_*` and the per-op wrappers elsewhere are `k_*`.

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
* `adamw_step(p, g, m, v, n, lr_group_split, lr_lora, lr_readout, b1, b2, eps, bias1, bias2)`: elements with index <
  `lr_group_split` use `lr_lora`; the rest (the readout sits at the end of the flat buffer) use `lr_readout`. The
  formula is D§2.2's word for word. `bias1 = 1 − b1^t` and `bias2 = 1 − b2^t` are pre-computed on the host in double
  and passed as floats (the step count is not). `sumsq_partial`/`sumsq_final` return the **sum of squares**, so the
  caller takes the `sqrt` before `ref_clip_coef` — the first version of M5.2's step forgot it and clipped 21x too
  hard.
* `src/lora.c`: the flat buffer layout. It is a table of 151 entries (150 LoRA pairs plus the readout) with offsets,
  shapes and stems, generated from the D§13 model table. Functions: `lora_alloc`, `lora_load_adapter(dir)`
  (reads a Python-written `adapter_model.safetensors` + `readout.safetensors`; the readout may be bf16 or fp32, so
  convert), `lora_save_adapter(dir, step, dev_ce, meta)` (`meta` is a nullable `LoraMeta *` that supplies the
  base/targets/weights that `adapter_config.json` carries), `lora_zero_grad()` (one `cudaMemsetAsync` on the flat
  gradient buffer, once per accumulation group) and `lr_at(step, total, base)` (D§2.2's schedule, in double).
* Test: compare one Adam step against a C ref of torch's formula, then 10 steps with random grads (rel 1e-6). Check
  that `lr_at` matches a table printed by Python (put a 10-line Python snippet in the test's comment that prints
  `f(s)` for total=100 and s=0..100). The clip must match the ref's norm to 1e-6.

**Done when:** `test_kernels adam` is green. Also check the total count: printing the size of `lora_alloc` must give
`trainable 10,484,736`.

### M2.7 Card checkpoint: small kernels
`sh scripts/card-check.sh <A|B> m2`, once every M2 test is green on gpubox.

**Done when:** `checkpoint m2: 1 steps, 0 failed`, with the same gates as M2.1–M2.6 (the host side here is the same
CPU reference, so the check is the kernel's value on the card).

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
start. The backward keeps the same 64-token checkpoints and segment replay as the kernel — the state at every 64th
token, written *before* that token's step, so segment 0's checkpoint is all zeros, and one segment replayed into a
`[64][Dk*Dv]` scratch — so the oracle is usable at the model's dims (`(N/64)·H·Dk·Dv` reals instead of
`N·H·Dk·Dv`: 84 MB instead of 5.3 GB at the biggest M3.2 batch) and a stored S_prev cannot disagree with the forward.
One `gdn_step` serves both directions. Then it follows D§4.8's reverse loop line for line, **in the stated order**.
Keep comments that quote each D§4.8 line next to its code. Inner sums accumulate in `double` and round once (the
rule `ref/real.h`'s `r_*` wrappers follow): the 16,384-term `dg` would otherwise carry an order-dependent error of
its own into the 1e-4 kernel gate. The backward does not expose its checkpoints, so M3.3's "compare `sp` with the
ref's stored S_prev" is a kernel-vs-kernel check; the honest independent form is a double-precision host replay of
the reference's own formula (which J17's `gdn_bwd_sp` does).

**Done when:** gradcheck passes in double with H=2, Dk=Dv=8, B=2 rows of lengths {5, 3}, for all five input
gradients. Use g in [-1, 0] (decays are ≤ 1) and beta in [0, 1].

### M3.2 `kernels/gdn.cu`: forward
Build it as D§8.3 describes: grid (B, 16, 4), 32 threads, `float s[128]` per lane. Two details matter:
* **Registers:** `s[]` stays in registers only if every index into it is a compile-time constant after unrolling.
  Write the two inner loops as `#pragma unroll` over `i = 0..127` with `Dk=128` as a template or `#define`
  constant. Never index `s[]` with a runtime variable. Verify with `nvcc -Xptxas -v kernels/gdn.cu` on gpubox: the
  line for `gdn_fwd_kernel` must say `0 bytes spill stores, 0 bytes spill loads`. Record the register count in a
  comment. As built, one unrolled 128-iteration loop spills (255 registers, 68 B stores / 52 B loads) because ptxas
  prefetches all 128 shared loads of the sum; **split each 128-iteration loop into two 64-iteration halves with a
  `__syncwarp()` between** and it is 201 registers, 0 spills, 1024 B of smem. The barrier is scheduling-only (it sits
  in the loop body with warp-uniform predicates) and the sums still run i = 0…127 ascending, so the numbers do not
  change — `test_gdn`'s `no_ckpt_identical` check ties the two halves together numerically.
* **Checkpoints:** `ckpt` is a device buffer `[B][16][max_segs][128][128]` (`max_segs = ceil(max_len / 64)`).
  Lane j of tile `tile` writes column `tile*32 + j`. Write it *before* processing a token whose
  `t_local % 64 == 0` (so segment 0's checkpoint is all zeros). The buffer belongs to the caller
  (`gdn_fwd(..., int nseg, float *ckpt, float *o, cudaStream_t s)`), because the backward is a separate call that
  reads it and there is no global scratch (rule 8); `kernels/gdn.h`'s `gdn_ckpt_segs()`/`gdn_ckpt_floats()` are the
  only place its size is written down, and `nseg <= 0` means "forward only, no stores". At B = 8 / N = 5094 it is
  142.6 MB and costs nothing measurable (1.26 ms with, 1.25–1.33 ms without).

The kernel reads q, k, v, g and beta in the layout the upstream kernels produce: q, k and v as [N][16][128] slices
of the conv output (after l2norm), and g and beta as [N][16].

**Done when:** `test_gdn fwd` on gpubox, with the full dims (H=16, D=128) and ragged batches {1}, {2}, {63, 64, 65},
{200, 7, 130}, {1026, 520, 670, 300, 811, 455, 610, 702}, gives `rel < 1e-4` against `ref_gdn_fwd` (float build),
and the spill check is clean.

### M3.2a Card checkpoint: GDN forward (before M3.3)
`sh scripts/card-check.sh <A|B> m3-fwd`. This is the riskiest kernel and the backward is built on the same structure,
so learn how it behaves on the shim before writing the backward. Report the failing batch and the log path if it
fails. M3.3 doesn't start until this result is in.

**Done when:** `checkpoint m3-fwd: 1 steps, 0 failed`.
As built: `test_gdn fwd` exit 0 after 7 s, 17 `ok`, worst rel **5.35e-07** (the same figure as gpubox), and one call at
B=8 / N=5094 is **1.28 ms with checkpoints, 1.20 without**. The shim build prints no ptxas report, so the card records
the null-device launch line's register count (`regs=168` for sm_89, against gpubox's 201 for sm_120 — the same kernel,
counted by two nvcc versions) and the spill figure stays gpubox's `0 bytes`; the local cubin is 18112 bytes.

### M3.3 `kernels/gdn.cu`: backward
Follow D§8.3's phase A / phase B text exactly. Supporting buffers (allocated once by the caller):
* `sp`: `[B*16*4][64][128][32]` floats, the per-block S_prev scratch for one segment (1 MB per block).
* `dq_part`, `dk_part`: `[4][N][16][128]`; `dbeta_part`, `dg_part`: `[4][N][16]`.
* `gdn_bwd_reduce` sums the 4 tile partials in order 0, 1, 2, 3 into dq, dk, dbeta and dg.

Write it in this order, testing at each stage:
1. Phase A only (recompute) writing S_prev to `sp`. Test: compare `sp` for segment 0 with a double-precision host
   replay of the reference's own forward formula started from the kernel's checkpoint (the reference no longer stores
   an S_prev array to compare with, M3.1), plus `sp`'s `tl = 0` slice against the all-zero segment-0 checkpoint.
2. Phase B for dv and dq only (the do terms). Test dq and dv against the ref with dk, dg and dbeta ignored.
3. Add ddelta, dk, dkv, dbeta and dg. Test everything.

As built (J17) the whole backward went in one pass, because the three stages share `ds[64]`'s lifetime and the same
register budget; each stage is then *isolated* by the checks instead of by the order of authorship (and by the four
mutations in J17's note). The result: **168 registers, `0 bytes spill stores, 0 bytes spill loads`, 256 B of smem**,
grid `(B, 16, 4)`, block 64 = 2 warps, plus `gdn_bwd_reduce_kernel` (28 registers, block 256). **Neither fallback below
was needed**: `sd[64]` fits beside `ds[64]`, so `sp` is read once per token, and 64 rows per warp fits, so D§8.3 does
not need the 4-warp variant.

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
clean for `gdn_bwd_kernel`. Also time it: one call at B=8, lengths p50-ish (about 520), on gpubox. The design's
original budget (about 2 ms per layer for fwd + bwd) is what the number goes next to; it is since corrected — see
D§8.3. Write the number down. If it's over 10 ms, look for spills first.

As measured: worst gradient `rel` **3.77e-07** (dv on {200,7,130}) over seven batches; one call at B = 8 / p50 lengths
(520 x 8, N = 4160) is **7.2 ms** and the ragged B = 8 batch (N = 5094) is **11.6 ms**, against the forward's 1.3 ms —
i.e. **12.9 ms** for a GDN layer's fwd + bwd, 6x over the design's budget and not a spill problem: the kernel is
memory-bound on D§8.3's own `sp` scratch (~13 GB of traffic per call at N = 5094, a 7.3 ms floor on a 5090). See D§8.3's
correction; shrinking it is M8's chunked GDN, not a v1 fix.

### M3.4 Card checkpoint: GDN forward and backward
`sh scripts/card-check.sh <A|B> m3`, with the time per layer from the same test, next to the gpubox number.

**Done when:** `checkpoint m3: 1 steps, 0 failed`, and the card's time per layer is written down in Appendix B.
On card B (4090, shim): `gdn_fwd` 1.28 ms + `gdn_bwd` 15.99 ms = **17.3 ms per layer** against gpubox's 13.1 ms — 1.32x,
all of it in the backward, and 81% of the 4090's bandwidth where the 5090 reached 63%.

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
  the original's signature and return value and match them). **Replace `modeling_qwen3_5.causal_conv1d_fn` with a
  literal conv as well**: on gpubox `causal_conv1d` is installed and the hub-kernel decorator routes it to a
  CUDA-only custom op, so a CPU dump dies with `Expected x.is_cuda() to be true`. Count and assert 18 calls of each.
  Read the 2-row fixture (`make_fixture.py --rows 2 --lengths 37,70 --opts 5,12`, the two flags the plan's own text
  needs). Build the
  right-padded batch with train.py's `collate`. Run `option_logits` + `soft_ce` + the weighted loss exactly as
  train.py's loop does, in fp32, with `attn_implementation="eager"`. Register forward hooks on
  `model.layers[i]` that save the output **with the padding removed** (concatenate the rows' real tokens, so the
  layout is the packed [N,1024] jefftrain uses) and call `retain_grad()`. After backward, save everything in D§10.2's
  list. Also save `layer.{i}.in` (the input hidden state) and `layer.{i}.post_mixer` (the residual right after the
  mixer, i.e. the input of `post_attention_layernorm`), so a mixer and an MLP can be tested separately, and add
  `final` to the `retain_grad` list so `grad.final` exists. 403 keys in all.
* `compare_layers.py a.safetensors b.safetensors [--atol-rel 1e-3 --grad-rel 5e-3]`: for every key in both files,
  print `key  rel  ok/FAIL` in file order. The **first** failing key is what matters, so also print it on its own
  line at the end. Exit 1 if any key failed or the files share no keys.

### M4.2 `src/model.c`: load
* The model table (D§13), loaded with `st_get` and an explicit expected shape for every tensor. As built the table
  is its own translation unit, `src/model_table.c/.h` — 24 layers x their names/shapes/LoRA flag, the 150 adapted
  linears in flat-buffer order, and the config constants — because `src/lora.c` needs it before `model.c` exists;
  `model.c` is the loader that consumes it (D§13). Verify it against the real checkpoint once, as J12 did: all 320
  text tensors present with the shapes it claims, and no `language_model.layers.*` tensor of the file missing.
* Convert to fp32 on the card (`bf16_to_f32` into one big weight arena), except `embed_tokens`, which stays bf16.
* `A_log` and `dt_bias` are converted on the host to fp32 and `A_log` is also precomputed as `negA = -exp(A_log)`.
  Note that `k_gdn_gates_fwd` applies `-exp` itself, so `layer_gdn.c` passes `a_log`, not `neg_a`.
* Readout: from the base model's `readout.safetensors` (bf16) into the flat parameter buffer's readout slot as fp32,
  unless `--init-adapter` gives one.
* Print one line at load: `model: 24 layers (18 gdn, 6 attn), weights 2.0x GB fp32 + embed 0.51 GB bf16`.

### M4.3 Forward pieces
Write them in this order, each as `*_fwd(ctx, layer, x_in, N, batch, x_out)` that adds into the residual. Test each
one against `ref.safetensors` before writing the next.
1. `linear_fwd` (D§4.2), saving `u` when a save pointer is given. Its own gate is `test/test_linear.c --fd`, a
   finite-difference check of a `[37,64] -> [48]` linear in exact mode (plus the frozen `dx = dy W` path, the
   `accumulate_dx` path and the bad-call contract).
2. `layer_mlp.c`: the post-attention norm and the MLP, D§3.4's last two lines; tested as
   `layer.{i}.post_mixer -> layer.{i}.out`.
3. `layer_gdn.c`: D§3.6, kernel by kernel: `rmsnorm` → 4 linears (qkv and z with LoRA, a and b without) → `conv_fwd`
   → split into q/k/v (`jt_to_head_major` with `heads = 3`, i.e. a copy into three `[n][2048]` slices — the l2norm
   and `gdn_fwd` kernels take no row stride, so a strided view of `[N][6144]` would be read as garbage) →
   `l2norm` (q with post_scale 1/√128, k with 1) → `gdn_gates_fwd` → `gdn_fwd` →
   `gated_rmsnorm_fwd` (gate = z) → `out_proj` → residual add; tested as `layer.{i}.in -> layer.{i}.post_mixer`.
4. `layer_attn.c`: D§3.5: `rmsnorm` → `q_proj` → `qgate_split` → `k_proj`, `v_proj` → q_norm and k_norm (row
   length 256, rows = N·heads) → `rope_fwd` → per row: `to_head_major`, `kv_expand`, scores gemm, softmax, PV gemm,
   `from_head_major` → `sigmoid_gate_fwd` → `o_proj` → residual add; same split point as the GDN mixer.
5. `head.c`: final norm → gather the last token of each row (`off[b+1]-1`) → readout gemm → `ce_head`.

The per-layer test is `test_layer --ref ref.safetensors --layer i --part mixer|mlp|all --model M --adapter DIR` (plus
`--head [--jtb F]` for the head; `--jtb` is needed for the loss, because the reference stores the logits but not the
targets). A stub prints `not implemented` and exits 0, so a partially landed tree still runs. The split point is the
one `dump_ref.py` writes: a **mixer** is `layer.{i}.in -> layer.{i}.post_mixer` (input norm + mixer + residual) and the
**MLP** is `layer.{i}.post_mixer -> layer.{i}.out` (post-attention norm + MLP + residual), so `layer_bwd` can hand
the MLP's accumulated `dres` to the mixer (M5.1).

**Done when:** on gpubox, built with `JT_EXACT=1`, the whole forward (`jefftrain --dump-forward out.safetensors`,
writing the same keys as dump_ref) passes `compare_layers.py` with `--atol-rel 1e-3` for every `layer.{i}.out`,
`final`, `logits` and `loss`. Make `ref.safetensors` with `--device cpu` (fp32, eager attention, the literal GDN loop from M4.1).

If layer 3 (the first attention layer) is the first failure, suspect, in this order: the q/gate interleave, the RoPE
positions (local vs packed), the head-to-group mapping (h/4, not h%2), and the q_norm offset (1+w).
If layer 0 fails: the conv tap order, the l2norm being sum and not mean, the gated norm being w and not (1+w),
and the A_log sign.

### M4.4 Card checkpoint: the full forward
Copy `ref.safetensors`, the adapter and the 2-row fixture from gpubox to `/tmp` on the Mac (all made from synthetic
rows and a public model; they stay out of the repo). The model is in the Mac's HF cache. Before the first run,
confirm it's byte-for-byte the model `ref.safetensors` was made from: compare `shasum -a 256 *.safetensors` in that
snapshot folder with `ssh gpubox 'cd ~/models/jeff-base && sha256sum *.safetensors'`, and stop and tell the user if
any shard differs (otherwise a wrong model gets reported as a shim failure). `test_st --list` isn't enough, since it
only shows names and shapes. Then, with the flags of the gpubox run and Mac paths:
```
JT_REF=/tmp/jeff-ref/ref-fwd.safetensors JT_DUMP_ARGS="--model <M> --init-adapter /tmp/jeff-ref/adapter --train /tmp/jeff-ref/fx.jtb" \
  sh scripts/card-check.sh <A|B> m4
```
Exact mode, because it's the same 1e-3 gate as gpubox. If `jefftrain` runs out of card memory, report the card and
the arena size it printed. Don't shrink the fixture.

**The key-set trap, which the first m4 run hit.** `--dump-forward` writes the 77 forward keys; `ref.safetensors` has
403 (the other 326 are `grad.*`). Against the full file `compare_layers.py` prints `77 ok, 326 failed` and the
checkpoint ends `1 steps, 1 failed` while every forward key passed. So m4's `JT_REF` is a forward-only extract of the
same file (`{k: v for k, v in load_file(ref).items() if not k.startswith("grad.")}`, 77 keys, byte-identical
tensors). Only `--dump-backward` writes all 403, so m5 uses the full file unchanged.

**Done when:** `checkpoint m4: 1 steps, 0 failed`, with `compare_layers.py` clean for every key (the forward subset).
As built: worst error **5.693e-06** (`layer.15.post_mixer`, 176x inside the 1e-3 gate), `final` 4.572e-06, `loss`
exact, live logits 1.642e-06; the gpubox run of the same test was 2.104e-06 on `final`, so the shim adds 2-7x of
summation-order noise and no layout error.

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

**As built, and why the loop is not the obvious one.** Each bwd indexes its saves from the arena's *base*, while
`layer_fwd(SAVE)` runs the mixer first and the MLP second — so **one** `fwd(SAVE)` can serve **one** of the two
backwards, not both. `layer_bwd` therefore recomputes each half itself (2 forwards); `model_backward` (and
`--dump-backward`) is the hot path: per layer it runs the mixer's `fwd(SAVE)` from `saved_in[i]`, *parks* its save
block (`save_at` words copied into the work half), runs the MLP's `fwd(SAVE)` + `bwd` at the base, copies the park back,
then the mixer's `bwd` — one recompute forward per layer, D§6.2's "+33%" rather than "+66%". The park's size is read
from the save cursor, so no piece's save layout is written down twice. The MLP's path runs **first**: the mixer's `dy`
is `dL/d(post_mixer) = grad.layer.{i}.out + the MLP's path`, and feeding it the entry value is an O(1) error.

**Exact mode is not optional at these gates.** In TF32 the same 403-key dump gives `87 ok, 316 failed`: the sweep is an
exact linearisation and this fixture's backward amplifies a relative perturbation by ~1e21 over 24 layers, so a 1e-4
nudge to the last layer's gradient arrives at layer 0 at O(1e20) while the true gradient rides a stable direction. A
1e-6 nudge to one entry of the entry gradient, run in exact mode, reproduces the TF32 numbers almost digit for digit.

**Done when:** `jefftrain --dump-backward out.safetensors` (exact build) passes `compare_layers.py --grad-rel 5e-3`
for every `grad.layer.{i}.out`, all 300 LoRA grads, and `grad.readout`. If one LoRA grad fails while its layer's
output grad passes, the bug is in `linear_bwd`'s dA/dB lines (usually the `s` factor or a transposed `u`).

`--dump-backward` writes the 77 forward keys too, so the whole 403-key reference passes at once: as built, worst
`grad.layer.{i}.out` **3.624e-06**, worst LoRA key **1.205e-05**, `grad.final` 2.047e-06, `grad.readout` 1.716e-06
(all 1380x–2900x inside 5e-3). `grad.final` is `head_bwd`'s `dh`, its first and only work-half allocation, readable at
`ctx->arena + jt_work_mark(ctx) - n*MT_HIDDEN` for a mark taken before the call; `grad.layer.{i}.out` must be copied
out **before** the MLP's bwd writes `dres` in place. `ref.safetensors` has **no `grad.layer.0.in`**, so layer 0's
input-side gate needs either the patched dump J28 describes (one more `retain_grad` on `text.layers[0]`, 40 s on
CPU, bit-identical on the other 403 keys) or the directional finite difference J29 uses for the attention layers.

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
--eval-every --max-steps --eval-temperature --eval-only --init-adapter --log-every --dump-forward --dump-backward`,
plus `--targets`, `--weights` and `--weight-cap` (recorded in `report.json`/`adapter_config.json`; only prep.py acts on
them) and `--trace-loss FILE` (M6.2), which writes one JSON line per optimizer step,
`{"step":k,"row":[…],"ce":[…]}` — the step's row indices and per-row unweighted soft CE, in batch order, with
`--accum k` covering the whole group. The per-row CE is computed on the host from the step's fp32 logits (double
`-Σ t·log softmax`), because `head_fwd` hands back only the weighted batch sum. `--log-every`'s default is train.py's
rule (steps 1–3, then every 25th).
`--train/--dev/--eval` take `.jtb` files. **Reject a flag combination that disagrees with the .jtb** — the header has no
room for batch/accum/epochs (D§9.1), so the check is two-part and as built it is `src/main.c`'s
`split_check_structure()`: (a) `n_batches == ceil_div(n_rows, batch) * epochs`, (b) with prep.py's
`<split>.prep.json` sidecar, every batch's row count against its `rows_per_batch[k]` (plus `accum`, `eval_batch`,
`targets`, `weights`, `weight_cap`); without a sidecar (the synthetic fixtures), every batch must hold 1..`--batch`
rows. The first version assumed the short batch was last and rejected prep.py's real, shuffled order (J42): that is what
aborted both M6 cards before step 1.

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

As built, prep.py writes next to each `<name>.jtb` a `<name>.prep.json` sidecar carrying the arguments that shaped it
(`batch`, `accum`, `epochs`, `eval_batch`, `targets`, `weights`, `weight_cap`, `source`, `row_weights`, `tokens`, the
header echo, and `rows_per_batch` — one row count per batch, epoch by epoch). That is M5.2's check, because D§9.1's
"`--accum` is written into the header" was never possible. Four trace runs passed (`seed 0/1` x `none/balanced`), 16
batches in 2 `main()` calls, the dev list identical in all three `evaluate()` calls, `max |dweight|` 1e-7 had the weight
been float-rounded (0.0 for `none`), and three deliberate mutations (epoch rng `seed+1`, `shuffle=False`, `balance()`
skipped) each failed with a message naming the first differing row.

### M5.4 Card checkpoint: the backward
Same files and flags as M4.4, once M5.1's gate has passed on gpubox: `sh scripts/card-check.sh <A|B> m5` (again with
`JT_REF` and `JT_DUMP_ARGS` set — here the **full** `ref.safetensors`, 403 vs 403 keys). It needs the `--dump-backward`
flag from M5.2 and nothing from M5.3.

**Done when:** `checkpoint m5: 1 steps, 0 failed`, with every `grad.layer.{i}.out`, all 300 LoRA grads and
`grad.readout` inside the 5e-3 gate. As built: **403 ok, 0 failed**, worst key 2.345e-05
(`layers.11.self_attn.k_proj.lora_A`, 213x inside), `grad.layer.*` 7.884e-06, `grad.readout` 7.184e-06, `grad.final`
4.884e-06, forward keys 5.693e-06 — 2.0-2.2x the gpubox numbers, same order, no layer an outlier.

---

## M6. End-to-end parity and timing on gpubox

Everything here reads real data, so it runs on gpubox inside `~/CLM-8B` (paths from CLM-8B
`tools/jeff_lora/cuda/env.sh`: `$DATA`, `$VENV`, `$REMOTE_MODEL`). Outputs go to `~/CLM-8B/data/ovril_detect/runs/`.
macuda's scripts (`scripts/parity-eval.sh`, `parity-steps.sh`, `timing.sh`, `arm.sh`) only orchestrate: they
`cd ~/CLM-8B`, call prep.py and `~/$JT_REMOTE/build/nv/jefftrain` (export `JT_REMOTE` first — one tree per task, M0.3),
and print a verdict line. Ask the user before each of M6.3 and M6.4 (they hold the GPU for a while).

### M6.1 Eval parity (minutes)
`prep.py --eval test=$DATA/test.jsonl --batch 4` → `test.jtb`. Then `jefftrain --eval-only --eval test=test.jtb
--eval-temperature <T>`, where T = `decision_config.json`'s temperature (read it with a one-line python in the
script; CLM-8B's `load_decision_config` knows where it lives). Then
`compare_preds.py out/pred-test.jsonl <the kit's runs/cuda-parity-fp32-*/pred-test.jsonl> --max-dp 0.05
--min-agree 158`. Run it with the TF32 build and the exact build.

**Done when:** both builds pass with the same gate the PyTorch port passed. Write `max |dp|` and the agreement into
the change description. As built: TF32 **max |dp| 0.0014**, exact **0.0015**, mean 1.4e-04, argmax **160/160** in both
modes (gate 0.05 / 158), eval soft CE 0.96363 / 0.96357 against the port's 0.96333, ~6x faster than the port on this
pass (5.9 s / 6.8 s vs 37.8 s), peak 7.57 GB. The script asserts each run's `gemm:` line, so TF32 and exact cannot be
silently swapped by a leftover `JT_EXACT`.

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

As built, `scripts/parity-steps.sh` runs the whole sequence (`make nv`, `make_adapter.py`, `prep.py`,
`py/trace_train_loss.py`, `jefftrain --trace-loss`, `py/compare_steps.py`) into one run directory and defaults to
**exact mode**: in TF32 the same 20 steps miss step 5's 1e-3 gate (1.31e-03), which is the same gradient amplification
M5.1 records. Result: 20 steps, 0 failed, worst step-mean difference **1.18e-03 at step 7** (gate 1e-2), worst early step
6.03e-04 at step 1, dev soft CE at 20 python 0.77117 vs jefftrain 0.77090 (**2.70e-04**, gate 2e-3); the exact-mode
trace is bit-reproducible, so G-m73 can compare against it unchanged.

### M6.3 Timing (10 minutes)
`jefftrain` on the full train split, `--max-steps 100 --batch 8`, printing the same TIMING line as the kit's
`timing.sh` (steps 1–3 dropped as warm-up, extrapolated to 28,645 rows). Compare with PyTorch's 0.143 s/step and the
D§15 target (< 0.4 s/step). Write the number down.
Being slower than PyTorch is expected for v1 and is not a failure. M8 is where speed gets work.

As measured (G-m63, full split 28,645 rows / 3,581 batches, TF32, peak 11.33 GB): **1.061 s/step** (mean of steps
4–100; median 1.050, min 0.64, max 1.57), i.e. 7.5 rows/s, **7.42x PyTorch's 0.143** and 0.661 s/step over the 0.4
target — expected for v1. The TIMING line already says what that costs an arm: `one epoch of 28645 rows ~63 min +
dev checks`. (The card itself is cheap — ~4 min of wall clock for sync, build, prep and 111 s of training — because
it measures 100 steps, not an epoch.)

### M6.4 Run parity (one arm, **about 75 minutes**; ask first)
`scripts/arm.sh soft`: the kit's `arms.sh` soft arm settings (1 epoch, batch 8, dev every 500 steps on all of
development, eval calibration and test at T=1), with jefftrain doing the training. Then CLM-8B's
`fetch_and_score.sh` path scores it next to the Python arm.

**Cost: budget 75 minutes, and get a cap that fits.** The earlier "about 30 minutes" here was an estimate made before
M6.3 existed. From the measured numbers, with nothing trimmed (any trim changes the checkpoint):

| piece | count | unit | total |
|---|---:|---|---:|
| optimizer steps | 3,581 | 1.061 s/step (G-m63) | 3,799 s = **63.3 min** |
| dev checks (step 0, every 500, and the epoch end), all 1,722 dev rows | 9 | 61 s each | 549 s = 9.2 min |
| calibration (1,679 rows) + test (160 rows) evals at T=1 | 2 | ~60 s + ~6 s | 66 s = 1.1 min |
| rebuild + adapter | 1 | ~41 s | 0.7 min |
| | | | **4,455 s = 74.3 min** |

A full arm is also 10–13% slower per step than the 100-step timing run (G-m64: 1.202 and 1.164 s/step) because the dev
checks leave the card cold, so the honest window is **75–85 min**; the measured runs were **75.5 and 76 min**. The
first attempt was refused a 60-minute card on exactly this arithmetic (`logs/jefftrain/G-m64-rerun.md`) and the
succeeded ones ran under a raised **90-minute** cap. Even the pure training loop alone (63.3 min) is over an hour, so a
one-hour cap is not enough for this checkpoint.

**Done when:** D§10.3.3's gates hold (dev soft CE at the chosen step within 0.01 of the Python soft arm, test soft CE
within 0.02, score.py metrics inside the two-seed noise band). Report the table to the user. As built (G-m64, band taken
over all four runs of the arm — both jefftrain seeds and both Python seeds, which is the definition the user chose):
**PASS**, 15/15 `score.py` metrics inside the band for both seeds; dev CE |Δ| 0.00072 (seed 0) / 0.00607 (seed 1)
against 0.01; calibrated test CE |Δ| 0.00130 / 0.01170 against 0.02; peak 12.29 GB. Two caveats for whoever reads
that table: the band now contains our own runs (so a member-vs-member difference cannot exceed it), and under the
Python-runs-only band the same arm is 6/15 (seed 0) and 1/15 (seed 1) over — both numbers are in the logs. The one
known difference left in the trainer is the start: `arm.sh` uses `make_adapter.py`'s fresh adapter (A random,
B ~ N(0,1)·1e-2) where train.py starts at B = 0, a ~2e-2 LoRA update at step 0 and the likeliest cause of the residual
1–3 argmax lines.

`scripts/arm.sh` still takes its band from the Python runs only, so this gate is not reproducible from the script alone;
accepting the jefftrain runs' directories the way `JT_REF_RUNS` accepts the Python ones is an open follow-up.

---

## M7. The shim: macuda trains Jeff

By now the GEMMs, every kernel, and the forward and backward at layer level have passed on the card (M1–M5). M7 tests
what those couldn't: the whole trainer on real rows, over many steps, and its speed. It needs the gpubox outputs from
M6. If any kernel or model code changed after its checkpoint, run that checkpoint again first.

### M7.1 Build and null-device run
`make shim ARCH=<ask>`, `make check-symbols` and `make test-null`. Then `jefftrain --eval-only` on a fixture with a
null device runs start-up, allocation and the launch sequence end to end.

**Done when:** all three are green. If a symbol is missing, don't add it to the shim: change jefftrain to use an API
from D§12's list, or ask the user. As built: `make shim ARCH=sm_120a` from clean 15.2 s, 0 warnings, 9 kernels +
every host/ref/test object; `check-symbols: all 24 cuda*/cublas* symbol(s)`; `test-null: 8 test(s) passed`; and
`--eval-only` on a 4-row fixture ran to `rc 0` with 1273 `would launch …` lines, 0 CUDA errors, both output files
written. The null device computes nothing, so its CE/agreement numbers are meaningless by construction — this proves
start-up, allocation, the launch sequence and the writers, not numbers.

### M7.2 Eval parity on the card (ask first)
The 160 test rows. The .jtb is real data, so it lives under CLM-8B's `data/`, and the binary reads it from there.
Run `jefftrain --eval-only` through `sh $MACUDA/tools/nv_shim_step.sh <A|B> probe $PWD/build/shim/jefftrain <flags>`.
Compare against gpubox jefftrain's `pred-test.jsonl` (M6.1's run directory; byte-identical `test.jtb` required).

**Done when:** max |dp| < 0.01 over the 160 rows. As built: **0.0011** against gpubox's TF32 predictions and 0.0010
against its exact ones, mean |dp| 0.00009, argmax **160/160**, eval soft CE 0.96353 (gpubox TF32 0.96363) — 9x inside
the gate, and the same comparison at `--max-dp 0.0001` fails, so it measures rather than floors.

### M7.3 Step parity on the card (ask first)
20 steps with the M6.2 arguments, in the same way. Compare against gpubox jefftrain's `steps.jsonl`.

**Done when:** M6.2's gate holds against that file. As built: all 20 steps inside it — worst 1.17e-03 at step 7
(gate 1e-2), worst early step 6.03e-04 at step 1 (gate 1e-3), dev soft CE |d| 2.70e-04 (gate 2e-3), same rows/order
8/8 every step. Against the gpubox binary line by line the card is **5.44e-05** per row over 20 real steps. Cost:
8.94 s/step (7.6x gpubox's exact 1.17), 278 s wall, `gemm: exact fp32` — the shim's dispatch cost, not correctness;
M8 is where speed gets work.

### M7.4 A timed arm on the card (ask; it holds the card for hours)
One full arm, if the user wants it. Report s/step, peak memory and the D§10.3.3 metrics. Not run: at M7.3's 8.94 s/step
a full arm is ~9 hours on the card, so this needs its own decision and a card held for that long.

**Done when:** the numbers are in Appendix B and the D§10.3.3 metrics are reported to the user, or the user has said
to skip the arm.

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
| works on gpubox, fails on the shim | stop and report (card checkpoint rule 4); `make check-symbols`; `JT_DEBUG=1` to find the first failing launch; dynamic smem > 48 KB without `cudaFuncSetAttribute` |
| gpubox builds an old object after a sync | `scripts/gpubox-sync.sh` uses `rsync -a`, so it restores the *source's* mtime and a re-synced file can look older than its object; `rm -rf build` (or `make -B`) after every sync that changed a file |
| a linear's name prints as `?` or empty, or a `grad.{stem}.lora_A` lookup finds nothing | `Linear.stem` points into a pool `model_load` freed (J25's finding): keep the pool alive in the `Model` or use `src/model_table.h`'s static names |
| preflight says `a new dext instance since the record` on one worktree only | each worktree and the main checkout keep their own `.dext-seen`; the file that gates `nv_shim_step.sh` is `$MACUDA`'s, so run `sh $MACUDA/tools/dart-stale.sh --clear` there |
| gradient parity fails only in TF32 | the sweep amplifies a 1e-6 relative perturbation by ~1e21 over 24 layers (M5.1); every gradient gate and M6.2/M7.3 runs in exact mode |
| a `.jtb` is right but jefftrain rejects its own flags | the batch/short-batch shape is prep.py's shuffled order plus `<split>.prep.json`'s `rows_per_batch`, not "full batches first" (M5.2) |
| a weight buffer reads back as zeros right after the load | `ctx->stream` is `cudaStreamNonBlocking` (D§12 forbids `cudaStreamCreate`) and this driver does not order the null stream against it; `jt_ctx_alloc` ends with one device-wide barrier, and anything uploaded or read back after that must use `ctx->stream` (J18) |

## Appendix B. Done-ness checklist (copy into the final change description)

Every number below is measured; the note in `train/jeff/notes/` that carries it is named in brackets.

- [x] M0 both builds, check-symbols — `make shim ARCH=sm_120a` 15.2 s / 0 warnings, `test-null: 8 test(s)`,
      `check-symbols: all 24 cuda*/cublas* symbol(s)`; gpubox `make nv` `hello OK` [J01, J35]
- [x] M1 GEMM 4 modes × exact/TF32; safetensors listing matches D§3.2; JTB reader; card m1 (strided-batched TF32
      accepted) — worst rel exact **7.74e-07** (1e-5 gate), TF32 **3.45e-04** (5e-3 gate); listing: 473 tensors, D§3.2's
      320 all present and shaped; card B: `GEMM OK 112/112` in both modes, **strided-batched TF32 accepted** [J03, J04,
      J05, G-m1]
- [x] M2 every kernel: gradcheck + GPU vs ref; card m2 — `test_kernels all` 275 checks, 0 FAIL, 8 groups; worst card
      rel **6.21e-06** (a 10,484,736-element fp32 reduction's partials), every gate unchanged; `trainable 10,484,736` [J06–J12,
      G-m2]
- [x] M3 GDN fwd/bwd vs ref, zero spills, time per layer: **gpubox fwd 1.26 ms + bwd 11.6 ms = 12.9 ms** at B=8/N=5094
      (7.2 ms at p50 lengths); fwd 201 regs, bwd 168 regs, both `0 bytes spill stores, 0 bytes spill loads`; worst rel
      fwd **5.35e-07**, bwd **3.77e-07** (1e-4 gate); card m3-fwd and m3 pass, **time per layer on the card 17.3 ms**
      (1.28 + 15.99) [J13, J16, J17, G-m3fwd, G-m3]
- [x] M4 forward vs dump_ref, all layers < 1e-3; card m4 — gpubox worst **2.104e-06** (`final`) over 77 keys, per-layer
      6.1e-08…1.3e-06, `loss` 3.356e-07, live logits 2.593e-07; card B worst **5.693e-06** (`layer.15.post_mixer`), `loss`
      exact, forward subset 77 ok / 0 failed (the full 403-key file reports 326 `grad.*` missing by construction) [J14–J24,
      G-m4]
- [x] M5 backward vs dump_ref, all grads < 5e-3; prep.py order == train.py order; card m5 — gpubox 403/403 ok, worst
      `grad.layer.*` **3.624e-06**, worst LoRA **1.205e-05**, `grad.final` 2.047e-06, `grad.readout` 1.716e-06; card B
      403/403 ok, worst **2.345e-05**; trace: 16 batches in 2 calls, seeds 0/1 x none/balanced all PASS [J15, J25–J30,
      G-m5]
- [x] M6 eval parity **160/160**, max dp **0.0015** (exact) / 0.0014 (TF32) vs the 0.05 gate; step parity: 20 steps,
      worst **1.18e-03** at step 7 (1e-2 gate), worst early **6.03e-04** (1e-3), dev soft CE |d| **2.70e-04** (2e-3); s/step
      **1.061** at batch 8 (7.42x PyTorch's 0.143, 0.661 over the 0.4 target), full-arm 1.202 / 1.164 s/step, peak
      11.33 / 12.29 GB; run parity **PASS**, 15/15 metrics inside the four-run band, dev CE |Δ| 0.00072 / 0.00607,
      calibrated test CE |Δ| 0.00130 / 0.01170, arm wall 75.5 / 76 min [J32, J33, G-m63, G-m64]
- [x] M7 shim: eval parity vs gpubox, max dp **0.0011** (argmax 160/160); 20-step parity worst **1.17e-03**, dev |d|
      2.70e-04, and **5.44e-05** per row against the gpubox binary; (optional) full arm s/step **not run** — at 8.94 s/step
      it is ~9 h on the card [J35, G-m72, G-m73]

One open follow-up the notes name and this plan does not own: `scripts/arm.sh`'s metric band still contains only the
Python runs, so M6.4's four-run table is not reproducible from the script alone (G-m64).
