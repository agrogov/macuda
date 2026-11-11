# jefftrain

A C/CUDA LoRA trainer for Jeff (Qwen3.5-0.8B + 255-way readout), written against the macuda shim so the same source
builds with real `nvcc` + cuBLAS on `gpubox` and with `tinycc`/`clang` + `libtinycudart`/`libtinycublas`/`libtinynv` on
the Mac's eGPU.

Read the two documents first: [`docs/training/jeff-trainer-design.md`](../../docs/training/jeff-trainer-design.md) (what
and why) and [`docs/training/jeff-trainer-plan.md`](../../docs/training/jeff-trainer-plan.md) (which step, which file,
which command). Where they disagree the design wins; where the design and CLM-8B's `tools/jeff_lora/train.py` disagree,
`train.py` wins.

## Layout

Everything is under `train/jeff/`; it never writes into `cuda-shim/` and never modifies it, it only links its libraries.

| path | what |
|---|---|
| `kernels/*.cu` | device code, each file exposing `extern "C"` launch wrappers |
| `src/*.c` | the C11 host program (`main.c` is the trainer itself) |
| `ref/*.c` | the CPU fp32 references, one `ref_*` per kernel |
| `test/test_*.c` | one binary each; `test/k_*.c` and `test/grad_*.c` are extra TUs of `test_kernels` and `gradcheck` |
| `py/*.py` | fixtures, tokenisation (`prep.py`), reference dumps and comparisons |
| `scripts/*.sh` | `gpubox-*` (Linux box), `card-check.sh` (one approved run on the eGPU), parity runs |
| `notes/*.md` | one per task; the first line is `STATUS: PASS` or `STATUS: BLOCKED — <reason>` |

## Building

`make nv` needs real CUDA (`gpubox`, or `nvcc` on `PATH`):

```sh
make nv                     # ARCH defaults to sm_120; build/nv/jefftrain and build/nv/test_*
build/nv/test_hello
```

`make shim` needs the shim's libraries and CUDA headers from the **main** macuda checkout (`cuda-shim/build/shim`,
`cuda-shim/cuda-13/include`); they are git-ignored, so a worktree never has them. `ARCH` is the card's and has no
default:

```sh
make shim ARCH=sm_120a                    # 5090, the default shim libraries
make shim ARCH=sm_89 SHIM=~/Downloads/vfio/macuda/cuda-shim/build/shim-sm_89   # the Ada card
make test-null                            # every build/shim/test_* with no arguments, null device
make check-symbols                        # no cuda*/cublas* symbol the shim does not define
```

`MACUDA ?= $(HOME)/Downloads/vfio/macuda` names the main checkout, `SHIM`, `NVCC`, `HOSTCC`, `SHIMCC`, `PYTHON` and
`CUDA_INC` are overridable. The shim kernel build compiles each `.cu` from a uniquely named sibling copy
(`kernels/.tc_<worktree hash>_<name>.cu`, deleted afterwards) under `/usr/bin/lockf /tmp/jefftrain-tinycc.lock`:
tinycc keys its fatbin on the source basename, so parallel worktrees would otherwise overwrite each other's device code.

Host-only work needs neither a GPU nor the shim:

```sh
make test-cpu               # ref/ float and double, gradcheck all, test_ref*, test_st, test_jtb (on build/cpu/fx.jtb)
```

One GPU build at a time is enough: `nv` and `shim` share `src/`, but their objects live in `build/nv` and `build/shim`.

## Running on gpubox

```sh
bash scripts/gpubox-run.sh test_hello            # sync, make nv there, run build/nv/test_hello
bash scripts/gpubox-build.sh                     # sync + make nv
bash scripts/gpubox-sync.sh                      # sync only
JT_REMOTE=macuda-train/wt-J01 bash scripts/gpubox-run.sh test_hello   # parallel tasks: separate remote dirs
```

`JT_REMOTE` defaults to `macuda-train/jeff` and is relative to `~` on `gpubox`. The rsync uses `--delete`, so two tasks
must not share it.

## Running jefftrain

`jefftrain` with no arguments (or `--help`) prints the full usage. The three run shapes:

```sh
jefftrain --model DIR --train FILE.jtb --dev FILE.jtb --output DIR [options]   # train
jefftrain --model DIR --eval-only --eval NAME=FILE.jtb --output DIR [options]  # score only
jefftrain --model DIR --train FILE.jtb --dump-forward|--dump-backward OUT      # parity dumps
```

The options (defaults in brackets): `--lr` [2e-4], `--readout-lr` [5e-6], `--epochs` [1], `--batch` [8], `--accum` [1]
(the last three must match the `.jtb`'s batches), `--eval-every N` [epoch end], `--max-steps N` [whole schedule],
`--log-every N` [train.py's rule: steps 1–3 then every 25th], `--eval NAME=FILE.jtb` (repeatable, scored at the end),
`--eval-temperature T|checkpoint`, `--init-adapter DIR` (a Python-written adapter), `--targets soft|hard` [soft],
`--weights none|balanced` [none], `--weight-cap F` [4.0] (the last three are recorded, not applied),
`--trace-loss FILE` (per-step per-row CE, plan M6.2), and `--no-recompute` (keep every layer's forward saves instead
of recomputing them in the backward — see `JT_NO_RECOMPUTE` below; automatic when memory allows since J51).

## Environment toggles

Every mode is an environment variable so the parity scripts can switch it without a flag. A toggle that changes
behaviour prints one startup line saying so; the scripts check that line through `JT_EXPECT`.

| variable | default | what |
|---|---|---|
| `JT_EXACT=1` | off (TF32) | exact fp32 GEMMs (`CUBLAS_COMPUTE_32F`) instead of TF32. The parity oracle: every gradient gate runs here, and no M8 default touches it. On the shim add `TINYCUBLAS_TC=0` (the recorded recipe); since J51's cublas fix `JT_EXACT=1` alone is also exact there. Prints `gemm: exact fp32` / `gemm: tf32`. |
| `JT_BF16_WEIGHTS` | **1 in TF32 mode** (J51), unset in exact | frozen Linear weights stay in the checkpoint's bf16 bits (1.99 → 1.00 GB) and their two GEMMs take bf16 operands, activations cast per call. `=0` restores the fp32 path byte for byte; under `JT_EXACT=1` it is ignored with a `weights: bf16 ignored …` line. Startup: `jefftrain: default JT_BF16_WEIGHTS=1 …` and `weights: bf16 (1.00 GB)`. |
| `JT_NO_RECOMPUTE` | **auto** (J51) | unset: the no-recompute store engages when the run's whole device footprint plus a 1 GB margin fits (`ctx: no-recompute (auto): …` / `ctx: recompute (auto): …` with both numbers). `=1` forces the store even where memory says no, `=0` forces recompute everywhere. Bit-identical either way; the store costs ~3.2 MB per packed token. `--no-recompute` is the flag spelling of `=1`. |
| `JT_NO_RECOMPUTE_TOKENS` | set by `main.c` | internal: the train split's widest batch, which the store is sized for; wider batches (dev checks) recompute. Unset in `--eval-only`/`--dump-forward`, which never allocate a store. |
| `JT_NO_RECOMPUTE_GUARD=<KB>` | off | diagnostic: poison `<KB>` KB above the store's work region and check it after every piece; a write past the top buffer is named (layer, piece, span) instead of faulting the card. Costs a host copy per piece — leave off. |
| `JT_GDN_CHUNKED=1` | off | the chunked (WY-form) GDN recurrence instead of the sequential kernels. Correct (~1e-6) but slower on both backends (+5.2% gpubox, +18.5% card), so opt-in only (J37/J51). |
| `TINYCUBLAS_TC=0` | unset | shim only: force the non-tensor-core cuBLAS path (part of the exact recipe). |

Removed: `JT_ATTN_BATCH` (J45's launch batching — bit-identical but `WORTH: no` on both backends; removed by J51,
restorable from `notes/J51-JT_ATTN_BATCH.patch` in reverse). There is no flash-attention toggle: J48 measured its
ceiling at 2.71% of a TF32 step and J49/J50 skipped it.

Build-time flags (not environment): `-DJT_DEBUG=1` (check + sync after every launch), `-DJT_CPU`, `-DJT_SHIM`,
`-DJT_REAL_DOUBLE` (the double references for gradcheck). The shim build carries no `-DJT_EXACT`; one build serves
both modes.

## Measuring and gating (the M8 tooling, J43)

```sh
export JT_REMOTE=macuda-train/wt-<task>          # before every gpubox script
sh scripts/parity-eval.sh                        # M6.1: 160 test rows, gate max |dp| <= 0.05, argmax >= 158/160
sh scripts/parity-steps.sh                       # M6.2: 20 steps vs train.py (EXACT=1 by default)
sh scripts/timing.sh                             # M6.3: full-split s/step (the kit's TIMING line, steps 1-3 warm-up)
sh scripts/ab.sh [--exact] [--pairs N]           # A = HEAD, B = this tree, interleaved: WORTH yes|no
sh scripts/ab.sh --same                          # exact-mode byte-identity: BITEXACT yes|no
sh scripts/arm.sh soft                           # M6.4: the full arm + score.py gates (ask first: ~25 min)
```

`WORTH: yes` means B is at least 1.5% faster than A and the gap is more than 3x the larger side's run-to-run spread;
`BITEXACT: yes` means the 20-step `steps.jsonl` and both prediction files are byte-identical to the baseline. The
scripts take `JT_ENV='JT_X=1 …'` (validated `JT_NAME=value` words, appended to jefftrain's environment) and
`JT_EXPECT='<regex>'` (the run fails if the log has no matching line — a toggle that silently did nothing cannot
pass); `JT_SKIP_BUILD=1` reuses the remote build. `ab.sh` also takes `--env-a/--env-b`, `--expect-a/--expect-b`,
`--steps`, `--batch`, `--accum`. All GPU work runs under `flock ~/macuda-train/.gpu.lock` on gpubox.

## Conventions every test follows

* **A `test_*` binary run with no arguments must exit 0.** On the null device it is a launch-only run: it does the
  allocations and the launches, skips the value check and says so — `hello OK (null device: launch only)`. This is what
  `make test-null` (and a card checkpoint's `DRY=1`) relies on. The card itself is never touched without
  `scripts/card-check.sh` and the user's OK.
* Tests with groups take their group as `argv[1]` (`all`, `misc`, `norm`, …, `fwd`, `bwd`); `all` always exists.
* `CHECK_REL` gates (design §10): float forward `1e-5`, backward `1e-4`, gradcheck `1e-6`, GEMM exact `1e-5` / TF32
  `5e-3`, always `rel = max|got-want| / max(1e-30, max|want|)`. Tolerances are not loosened to make a test pass.
* Every cudart call goes through `CK()`, every cuBLAS call through `CB()` (`src/util.h`); after every launch in a
  `-DJT_DEBUG=1` build, `JT_SYNC_DEBUG;` checks the launch and synchronises.
* No real data here (plan rule 5): tests use synthetic fixtures (`py/make_fixture.py`), never `.jtb` files or weights
  made from real rows. Those stay in CLM-8B's `data/` or on `gpubox`.
* `scripts/prereq-check.sh <ID>...` fails unless `notes/<ID>.md` starts with exactly `STATUS: PASS`.
