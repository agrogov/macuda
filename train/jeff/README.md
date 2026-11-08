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
