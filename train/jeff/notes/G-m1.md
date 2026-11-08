STATUS: PASS

# G-m1 — card checkpoint m1: GEMMs

Plan M1.2a, under "Milestones at a glance" → "Card checkpoints". Binary `train/jeff/build/shim/test_gemm` (no
arguments), run as `sh scripts/card-check.sh B m1`.

Card: **B**, RTX 4090, `ARCH=sm_89`, `SHIM=$MACUDA/cuda-shim/build/shim-sm_89`. Step 1 runs in exact mode
(`JT_EXACT=1 TINYCUBLAS_TC=0`), step 2 in TF32; one build serves both.

## Checkpoint

```
checkpoint m1: 2 steps, 0 failed
card-check rc=0
```

| step | mode | gate | result |
|---|---|---|---|
| 1 | exact fp32 (`JT_EXACT=1 TINYCUBLAS_TC=0`) | `rel < 1e-5` | exit 0 after 7 s, `GEMM OK 112/112` |
| 2 | TF32 | `rel < 5e-3` | exit 0 after 6 s, `GEMM OK 112/112` |

Worst error against the gate, with the case it came from:

| mode | gate | worst rel | case |
|---|---|---|---|
| exact fp32 | 1e-5 | 3.07e-06 | `gemm ta=1 tb=0 M=16 N=1024 K=4200 alpha=1 beta=0` |
| TF32 | 5e-3 | 4.85e-04 | `gemm ta=1 tb=0 M=1 N=1 K=1 alpha=2 beta=0` |
| exact, `gemm_batched` cases only | 1e-5 | 7.74e-07 | `batched ta=0 tb=0 M=300 N=256 K=256 batch=8 alpha=2 beta=1` |
| TF32, `gemm_batched` cases only | 5e-3 | 2.91e-04 | `batched ta=0 tb=1 M=37 N=256 K=256 batch=8 alpha=1 beta=0` |

Counts: 112 `ok` lines, 0 `FAIL`, 0 `NOT_SUPPORTED`, no cuBLAS error, in each mode.

Plan M1.2a's open question (D§12): **answered yes** — tinycublas accepts `cublasGemmStridedBatchedEx` with
`CUBLAS_COMPUTE_32F_FAST_TF32` on the card (no `NOT_SUPPORTED` in either mode), so `layer_attn.c` (M4.3) is not
blocked.

Two earlier attempts never reached the card (preflight `VERDICT: ABORT`, `checkpoint m1: 1 steps, 1 failed`); after
`tools/dart-stale.sh --clear`, run 2 above passed. No gate, kernel, compute mode or shim was changed.

## Logs

| log | contents |
|---|---|
| `logs/jefftrain/G-m1.md` | run summary: setup, the two aborted attempts, numbers above |
| `logs/jefftrain/G-m1-card-check-2.log` | `sh scripts/card-check.sh B m1`: both steps, exit 0, `checkpoint m1: 2 steps, 0 failed` |
| `logs/shim-probe-20261108-231815.log` | exact mode: all 112 case lines (`gemm: exact fp32`) |
| `logs/shim-probe-20261108-231823.log` | TF32: all 112 case lines (`gemm: tf32`) |
