STATUS: PASS

# G-m3fwd — card checkpoint m3-fwd: GDN forward

Plan M3.2a. Binary `train/jeff/build/shim/test_gdn fwd`, run as `sh scripts/card-check.sh B m3-fwd` (`bwd` is still
J17's stub, so `all` was deliberately not run).

Card: **B**, RTX 4090, `ARCH=sm_89`, `SHIM=$MACUDA/cuda-shim/build/shim-sm_89`. No `JT_EXACT` (`step -`).

## Checkpoint

```
checkpoint m3-fwd: 1 steps, 0 failed
=== card-check rc 0
```

Step 1 `test_gdn fwd`: exit 0 after 7 s.

Counts: 17 `ok` lines, 0 `FAIL`, 0 `NOT_SUPPORTED`, no CUDA error.

Gate: five ragged batches at H=16, D=128, `rel < 1e-4` against `ref_gdn_fwd` (float build). Worst on the card:
**5.35e-07** at `B=8 {1026,520,670,300,811,455,610,702}`.

| batch | rel | card ms (ckpt / none) |
|---|---|---|
| B=1 {1} | 3.21e-07 | 0.07 / 0.06 |
| B=1 {2} | 4.35e-07 | 0.07 / 0.06 |
| B=3 {63,64,65} | 4.35e-07 | 0.11 / 0.11 |
| B=3 {200,7,130} | 3.63e-07 | 0.22 / 0.21 |
| B=8 {1026,520,670,300,811,455,610,702} | 5.35e-07 | 1.28 / 1.20 |

Beyond the numbers, all `ok` on the card: `gdn_fwd_no_ckpt_identical` (`err=0` on all five batches),
`gdn_fwd_ckpt0_zero` (`err=0` on all five), `gdn_fwd_ckpt_replay` (`rel=2.72e-07` at `b=2 h=5 seg=1`,
`rel=2.9e-07` at `b=2 h=0 seg=2`).

## Numbers the plan asks to record

| item | value |
|---|---|
| time per layer, forward (M3.2a / Appendix B) | **1.28 ms** at B=8 / N=5094, one `gdn_fwd` call |
| same call without checkpoints | 1.20 ms (the 142.6 MB of checkpoints cost nothing measurable) |
| spill check | `0 bytes spill stores, 0 bytes spill loads`, 201 registers — the gpubox `nvcc -Xptxas -v` figure from M3.2 (J16), recorded in `kernels/gdn.cu` |
| spill check on the card | **not recorded**: the shim build prints no ptxas report |
| null-device launch report, sm_89 object | `regs=168 smem=0 grid=(1,16,4) block=(32,1,1)`, cubin 18112 bytes, `fnv1a64 c97704fd6d7cd56c` |

No tolerance was changed (`TOL` is the plan's 1e-4).

## Logs

| log | contents |
|---|---|
| `logs/jefftrain/G-m3fwd.md` | run summary: preflight, pre-steps, per-batch table, numbers above |
| `logs/jefftrain/G-m3fwd-cardcheck.log` | `sh scripts/card-check.sh B m3-fwd`: step 1, exit 0 after 7 s, `checkpoint m3-fwd: 1 steps, 0 failed` |
| `logs/shim-probe-20261109-000452.log` | the full card run: all 17 `ok` lines, per-batch `rel` and ms, submission stats |
| `logs/jefftrain/G-m3fwd-build.log` | `make shim ARCH=sm_89`, `test-null` (6 tests), `check-symbols` (18 symbols), and the `regs=168` launch report |
