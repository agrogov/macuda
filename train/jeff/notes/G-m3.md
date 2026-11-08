STATUS: PASS

# G-m3 — card checkpoint m3: GDN forward and backward

Plan M3.4. Binary `train/jeff/build/shim/test_gdn all`, run as `sh scripts/card-check.sh B m3` (the step is `test_gdn all`:
the forward of M3.2a/J16 and the backward of M3.3/J17 in one process).

## Checkpoint

```
=== step 1:  test_gdn all
checkpoint m3: 1 steps, 0 failed
=== card-check rc 0
```

Step 1 `test_gdn all`: exit 0 after 20 s, `gdn OK all: 2 part(s)`. Counts from the probe log: 66 `ok` lines, 0 `FAIL`,
0 `NOT_SUPPORTED`, no CUDA error. No `JT_EXACT` (step `-`): one build, TF32-default host path, unchanged.

## Card

**B**, RTX 4090 (`0x2684`), `ARCH=sm_89`, `SHIM=$MACUDA/cuda-shim/build/shim-sm_89`. Link up x4 at 2.5 GT/s (gen1,
target gen3). Build id `ba4e38d (main~6)`, sdk 26.5, driver shim `libtinynv build ba4e38d`. Every number below is that
card; nothing here was measured with `sm_120a`.

## Worst error seen, against its gate

Gate `TOL = 1e-4` (the plan's M3.2/M3.3 number, `test_gdn.c`; not touched).

| | worst rel | case | against 1e-4 |
|---|---|---|---|
| whole run | **5.35e-07** | `gdn_fwd B=8 {1026,520,670,300,811,455,610,702}` | 187x inside |
| bwd gradients | **3.77e-07** | `gdn_bwd dv B=3 {200,7,130}` | 265x inside |

Both are the gpubox figures to the digit (J16: 5.35e-07; J17: 3.77e-07): the shim reproduces the CUDA build's arithmetic,
it does not merely stay inside the gate.

## Card's time per layer, next to the gpubox number

One GDN layer = one `gdn_fwd` call + one `gdn_bwd` call, B = 8, N = 5094 (the ragged median-length batch):

| | fwd | bwd | fwd + bwd (per layer) |
|---|---|---|---|
| **card B (4090, shim)** | 1.28 ms | 15.99 ms (best of 4 warm 15.96) | **17.3 ms** |
| gpubox (5090, nvcc) — J17 | 1.39 ms | 11.69 ms (best of 4 warm 11.64) | **13.1 ms** |

Card / gpubox = **1.32x** on the layer, and all of it is in the backward. That is the shape J17 predicted: the backward
is bandwidth-bound on D§8.3's `sp` scratch plus the 4-tile partials (~13 GB of traffic per call at N = 5094 by J17's own
breakdown). 13 GB against the 4090's 1008 GB/s peak is a 13.0 ms floor, and 15.96 ms is 81% of that peak, where the same
traffic on the 5090 (1.79 TB/s, 7.3 ms floor) ran at 63%. The forward is not bandwidth-bound and comes out slightly
faster than on gpubox (1.28 vs 1.39 ms). **No kernel or compute mode was changed to move any of these numbers.**

## Every number the test prints (one call each; warm best in brackets)

`fwd` is the `gdn_fwd` line (with ckpt). The `gdn_bwd` line prints its own fwd figure for the same batch — 0.07 / 0.06 /
0.14 / 0.28 / 1.22 / 0.70 / 0.29 ms — i.e. the same call to within the resolution the test quotes.

| batch | N | fwd ms | bwd ms | worst grad rel |
|---|---|---|---|---|
| B=1 {1} | 1 | 0.06 | 0.08 (0.07) | dk 1.83e-07 |
| B=1 {2} | 2 | 0.06 | 0.09 (0.08) | dv 2.6e-07 |
| B=3 {63,64,65} | 192 | 0.12 | 0.87 (0.87) | dv 2.76e-07 |
| B=3 {200,7,130} | 337 | 0.22 | 2.20 (2.20) | dv 3.77e-07 |
| B=8 {1026,520,670,300,811,455,610,702} | 5094 | 1.28 | 15.99 (15.96) | dv 2.97e-07 |
| B=8 {520 x 8} | 4160 | 0.70 | 11.53 (11.48) | dv 1.92e-07 |
| B=3 {200,7,130}, q/k at the model's norms | 337 | 0.22 | 2.20 (2.20) | dv 2.52e-07 |

Against J17 on gpubox: bwd 2.20 vs 2.01 ms (N=337), 0.87 vs 0.71 ms (N=192), 11.48 vs 7.25 ms warm ({520 x 8}:
1.58x, and the {520 x 8} traffic scaled to 4160 tokens is a 10.5 ms floor, so 11.48 ms is 92% of the 4090's peak — the
uniform batch is where the asymmetry shows least). The forward of the ragged B=8 batch reads `1.28 ms with ckpt, 1.25 ms
without`.

Also `ok` on the card, not just inside the gate: `gdn_fwd_no_ckpt_identical` (err=0, all five batches),
`gdn_fwd_ckpt0_zero` (err=0, all five), `gdn_fwd_ckpt_replay` (rel 2.72e-07 at `b=2 h=5 seg=1`, 2.9e-07 at
`b=2 h=0 seg=2`), and the backward's phase-A scratch checks `gdn_bwd_sp` / `gdn_bwd_sp_ckpt` on all seven batches
(worst `sp` rel 1.7e-07, every `sp_ckpt` err=0).

## Shipped sm_89 cubin (null-device launch report, `make test-null`)

| kernel | grid | block | regs | smem |
|---|---|---|---|---|
| `gdn_fwd_kernel` | (1,16,4) | (32,1,1) | 168 | 0 |
| `gdn_bwd_kernel` | (1,16,4) | (64,1,1) | 168 | 0 |
| `gdn_bwd_reduce_kernel` | (8,1,1) | (256,1,1) | 25 | 0 |

On the card the forward's module was placed as `[0x10205a0000, 0x10205b1a80)`, cubin 81536 bytes,
`fnv1a64 a37853336dd576d3`; the container's nvcc 13.0.3 agrees with gpubox's 13.2 on 168 registers for both kernels, so
the register budget J16/J17 recorded is what the card actually ran (the shim prints no ptxas report; the spill figure
stays gpubox's `0 bytes spill stores, 0 bytes spill loads`). tinycc's build behaves like nvcc's here: no launch failure,
no wrong-arch image, no difference in any `rel`.

## Pre-steps

| step | verdict |
|---|---|
| `sh scripts/prereq-check.sh J02 J17` | `prereq: J02 J17 PASS` |
| `sh tools/preflight.sh` (00:35:19) | **`VERDICT: OK to proceed`** — DART error data absent, no snapshot: clean; dext instances 2, newest pid 78368 started 2026-11-08 23:15:36, recorded as first sighting (not a replug: nothing cleared, `dart-stale.sh --clear` not needed) |
| `make shim ARCH=sm_89 SHIM=$MACUDA/cuda-shim/build/shim-sm_89` | rc 0, 0 warnings, 0 errors (full rebuild after `make clean`) |
| `make test-null` | `test-null: 6 test(s) passed with no arguments on the null device`, rc 0 |
| `make check-symbols` | `all 18 cuda*/cublas* symbol(s) the objects need are defined by the shim`, rc 0 |

After the step the probe quiesced as the protocol does: `gsp-rm acknowledged the unload notice and halted, 0.04 s` /
`gsp-rm unloaded and its region torn down in 0.24 s: the card is cold, and the next open needs no reset`. The eGPU is
left quiesced.

## Logs

| log | contents |
|---|---|
| `logs/jefftrain/G-m3.md` | this run's full record: commands, preflight verdict, both `rel` tables, timings, card notes |
| `logs/jefftrain/G-m3-preflight.log` | `sh tools/preflight.sh`, `VERDICT: OK to proceed` |
| `logs/jefftrain/G-m3-build.log` | `make shim ARCH=sm_89` (301 lines, rc 0), `make test-null` (6 tests), `make check-symbols` (18 symbols), the sm_89 launch reports |
| `logs/jefftrain/G-m3-cardcheck.log` | `sh scripts/card-check.sh B m3`: step 1, exit 0 after 20 s, `checkpoint m3: 1 steps, 0 failed`, rc 0 |
| `logs/shim-probe-20261109-003624.log` | the full card run: all 66 `ok` lines, per-batch `rel` and ms, module placement, submission stats, quiesce |

## Tracker

Every other tracked file is untouched by this card: `git status --short` shows this one new file and nothing else. Plan
rule 1/standing rule 3: no tolerance, kernel, gate or compute mode was changed.
