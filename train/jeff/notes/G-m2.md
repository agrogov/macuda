STATUS: PASS

# G-m2 — card checkpoint m2: small kernels

Plan M2.7. Binary `train/jeff/build/shim/test_kernels all`, run as `sh scripts/card-check.sh B m2`.

Card: **B**, RTX 4090, `ARCH=sm_89`, `SHIM=$MACUDA/cuda-shim/build/shim-sm_89`. No `JT_EXACT` for m2 (`step -`): one
build, TF32-default host path, unchanged.

## Checkpoint

```
checkpoint m2: 1 steps, 0 failed
```

Step 1 `test_kernels all`: exit 0 after 9 s, `kernels OK all: 8 group(s)`.

Counts: 275 `ok` lines, 0 `FAIL`, 0 `NOT_SUPPORTED`, no CUDA error. Groups (8): `misc norm act rope conv attn head adam`.

Gates: the M2.1–M2.6 ones, unchanged — forward `1e-5`, backward `1e-4`, gradcheck `1e-6`, Adam step/`lr_at` `1e-6`.
Worst `rel` per group on the card, with the case:

| group | worst rel | case |
|---|---|---|
| misc | 1.69e-08 | `kv_expand/kv_reduce adjoint rows=2516` |
| norm | 3.09e-07 | `gated_rmsnorm rows=40256 d=128 zc=0 dz` |
| act | 2.92e-07 | `silu_mul_bwd/dh1 n=602112` |
| rope | 1.15e-07 | `rope_bwd ragged3 n=168 heads=8 d=256` |
| conv | 1.78e-07 | `conv_bwd single n=1 ch=8 dqkv` |
| attn | 1.67e-07 | `attn_bwd L=1026 h=1` |
| head | 4.77e-07 | `ce_head/n255 rowsum` |
| adam | 6.21e-06 | `sumsq/n10484736 partials` |

Worst overall: 6.21e-06, the partials of a 10,484,736-element fp32 reduction; the Adam step values the plan's `1e-6`
gate is on are `p rel=1.19e-07`, `m rel=1.89e-07`, `v rel=2.59e-07` at that size. No tolerance was changed.

## Numbers the plan asks to record (M2.6)

| item | value |
|---|---|
| `lora` entries | 151 (150 LoRA pairs + readout), rank 16, scale 2.0 |
| `trainable` | **10,484,736** |
| flat buffer | 10,484,736 floats = 10,223,616 LoRA (`lr_group_split` at 10,223,616) + 261,120 readout |
| readout | 255 rows × 1024 cols, rank 0, 261,120 floats |
| `lr_at` | 101 values for total=100, s=0..100, `rel<1e-12` |
| Adam n=257 split 200 | step1/step10 `p rel=0`, `m rel=9.41e-08`, `v rel=1.62e-07` |
| Adam n=10,484,736 split 10,223,616 | step1/step10 `p rel=1.19e-07`, `m rel=1.89e-07`, `v rel=2.59e-07` |
| clip, largest n | `coef rel=0`, `clipped_norm rel=2.4e-07` |
| conv ragged row test | in the `conv` group, no `FAIL` |

Pre-steps in the same log: `make test-null` `5 test(s) passed`, `make check-symbols` `all 17 cuda*/cublas* symbol(s)`.

## Logs

| log | contents |
|---|---|
| `logs/jefftrain/G-m2.md` | run summary: setup, per-group table, the numbers above |
| `logs/jefftrain/G-m2-cardcheck.log` | `sh scripts/card-check.sh B m2`: step 1, exit 0 after 9 s, `checkpoint m2: 1 steps, 0 failed` |
| `logs/shim-probe-20261108-233324.log` | the full card run: all 275 `ok` lines, group headers, `lora:`/`trainable` lines |
