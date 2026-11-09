STATUS: PASS

# G-m4 — card checkpoint m4: the full forward

Plan M4.4. `sh scripts/card-check.sh B m4` with `JT_REF=/tmp/jeff-ref/ref.safetensors` and the gpubox flags
`JT_DUMP_ARGS="--model <HF snapshot> --init-adapter /tmp/jeff-ref/adapter --train /tmp/jeff-ref/fx.jtb"`, exact mode
(`JT_EXACT=1 TINYCUBLAS_TC=0` from `card-check.sh`; the shim build has no `-DJT_EXACT`, per the checkpoint rules).

Card: **B**, RTX 4090 (PCI 0x2684, 24564 MB), `ARCH=sm_89`, `SHIM=$MACUDA/cuda-shim/build/shim-sm_89` — never `sm_120a`.
Card left quiesced cold ("the card is cold, and the next open needs no reset"). One run.

## Checkpoint

```
=== step 1: JT_EXACT=1 TINYCUBLAS_TC=0 jefftrain ... --dump-forward .../m4.safetensors
   exit 0 after 6 s
compare_layers: 77 ok, 326 failed, 403 key(s) compared
first failing key: grad.final
checkpoint m4: 1 steps, 1 failed
```

Read those two lines together or the card looks failed. **The card step passed and every forward key passed.** The 326
"failures" are all `FAIL (not in m4.safetensors)` on `grad.*` keys: `card-check.sh`'s m4 branch compares the 77-key
forward dump against the **full 403-key** `ref.safetensors`, whose other 326 keys are backward outputs that
`--dump-forward` cannot write and that only M5.1's `--dump-backward` will. J24's note predicted this command's output
verbatim ("it will count 326 failures and `checkpoint m4: 1 steps, 1 failed` **until** the forward-only key set is
accounted for") and left the choice to this card's owner. Taken here, with nothing loosened and no second card run: the
run stands exactly as the plan prescribes, and the gate is evaluated on the same dump against the forward subset of the
same reference file — the identical tensors, byte for byte, minus the keys no forward produces.

```
$ python py/compare_layers.py /tmp/jeff-ref/ref-fwd.safetensors logs/.../G-m4-card-dump.safetensors --atol-rel 1e-3
compare_layers: 77 ok, 0 failed, 77 key(s) compared                                    rc 0
```

`ref-fwd.safetensors` is `{k: v for k, v in load_file(ref) if not k.startswith("grad.")}` — 77 keys, 7,998,585 values,
made from the same file, and kept under `/tmp`. Against it the Done-when's second half is literally true: **clean for
every key**. Tolerance unmoved at the plan's `--atol-rel 1e-3`. `card-check.sh` and `compare_layers.py` are other tasks'
files and were not edited.

## Numbers

**Worst error of the whole forward: 5.693e-06 relative (`layer.15.post_mixer`) — 176x inside the 1e-3 gate.**
`final` 4.572e-06; `loss` **exact** (card 1.42077208 vs ref 1.42077208); `logits` live columns 1.642e-06 (`9.060e-15` as
printed, because the two dead `-1e9` columns set `max|b|` = 1e9 and both are exactly `-1e9` in each file);
`in.ids`/`in.off` exact.

```
$ env JT_EXACT=1 TINYCUBLAS_TC=0 <card> jefftrain --model ... --init-adapter ... --train .../fx.jtb --dump-forward ...
gemm: exact fp32
model: 24 layers (18 gdn, 6 attn), weights 1.99 GB fp32 + embed 0.51 GB bf16
model: 321 frozen tensors checked by st_get (dtype and every dimension), readout 1.04 MB fp32
jefftrain: dump-forward: fx.jtb batch 0 = rows [0, 1], lengths [37, 70], n 107 tokens, 5 option(s), gw 2
ctx: lmax 70, arena 0.04 GB fp32 (107 tokens x 98304 floats + 2 x 8 x 70^2), gdn nseg 2, ckpt 0.00 GB, bwd scratch 0.14 GB
jefftrain: wrote ...: 77 keys (2 in + 24 layers x 3 + final, logits, loss), 21.5 MiB of data, loss 1.42077208
```

Card memory was never short: **arena 0.04 GB fp32**, `ckpt 0.00 GB`, `bwd scratch 0.14 GB` on a 24 GB card — no OOM, and
the fixture was not touched. Per-layer relative error (`max|a-b|/max|b|`, 24 layers):

| layer | `.out` | `.post_mixer` | | layer | `.out` | `.post_mixer` |
|---|---|---|---|---|---|---|
| 0 | 5.151e-06 | 1.559e-06 | | 12 | 2.718e-06 | 2.741e-06 |
| 1 | 3.203e-06 | 2.040e-06 | | 13 | 2.270e-06 | 2.258e-06 |
| 2 | 3.959e-06 | 1.618e-06 | | 14 | 2.442e-06 | 2.270e-06 |
| 3 | 4.904e-06 | 3.568e-06 | | 15 | 3.953e-06 | **5.693e-06** |
| 4 | 5.623e-06 | 3.008e-06 | | 16 | 2.207e-06 | 2.408e-06 |
| 5 | 4.853e-06 | 3.694e-06 | | 17 | 1.880e-06 | 1.677e-06 |
| 6 | 1.452e-06 | 1.524e-06 | | 18 | 1.444e-06 | 1.512e-06 |
| 7 | 2.625e-06 | 2.413e-06 | | 19 | 1.265e-06 | 1.157e-06 |
| 8 | 2.417e-06 | 2.551e-06 | | 20 | 1.149e-06 | 1.140e-06 |
| 9 | 2.272e-06 | 2.346e-06 | | 21 | 1.002e-06 | 9.296e-07 |
| 10 | 1.848e-06 | 1.969e-06 | | 22 | 1.514e-06 | 1.024e-06 |
| 11 | 3.675e-06 | 2.925e-06 | | 23 | 2.335e-06 | 1.605e-06 |

Against the gpubox run of the same test (J24: worst `final` 2.104e-06, live logits 2.593e-07): 2–7x larger on the shim,
same order of magnitude — a different GEMM/Norm summation order, not a layout error (those are O(1)). Nothing near a gate.

## Prerequisites and environment

* `sh scripts/prereq-check.sh J02 J24` → `prereq: J02 J24 PASS`.
* `tools/preflight.sh` → `VERDICT: OK to proceed`: `NVIDIA 0x2684, link up, x4 at 2.5 GT/s`, `IOPCIResourced = Yes`,
  `DART error data absent, no snapshot: clean`, `dext instances: 2, newest pid 78368 started 2026-11-08 23:15:36 -
  unchanged since the record`, `lowpowermode 0`. **No stale record and no new dext instance**, so the replug branch
  (`dart-stale.sh --clear`) was correctly not taken and there is only the one clean verdict to record.
* `make shim ARCH=sm_89 SHIM=.../shim-sm_89` rc 0, 0 warnings (every kernel recompiled: `build/shim/.arch = sm_89`);
  `make test-null` → `7 test(s) passed`; `make check-symbols` → `all 23 cuda*/cublas* symbol(s) … defined by the shim`.
* Model identity, before any card step (plan M4.4): `shasum -a 256 *.safetensors` in the Mac HF snapshot
  `snapshots/da320e510c6eaa15a36a2eeaa2aad9d2370321fe` vs `ssh gpubox 'cd ~/models/jeff-base && sha256sum *.safetensors'`
  — **both shards identical** (`model.safetensors 22fd9a1e…2f9b`, `readout.safetensors fb175582…c486`), so the model is
  the one `ref.safetensors` was made from.
* Reference set copied with `rsync -a gpubox:~/macuda-train/ref/ /tmp/jeff-ref/` and kept under `/tmp` (synthetic rows,
  public model — plan rule 5). Driver stack: `libtinynv build ba4e38d`, macOS SDK 26.5, PCIe x4 at gen1 (2.5 GT/s,
  target gen3) — a trained-down link costs bandwidth, not correctness. 1180 batches for the whole dump.

## Logs and files

`logs/jefftrain/` (git-ignored): `G-m4-card-check.log` (the run), `G-m4-shim-build.log`, `G-m4-null-symbols.log`,
`G-m4-compare-fwd.log` (the forward gate), `G-m4.md` (full record with all excerpts); the driver's own step log is
`logs/shim-probe-20261109-013632.log`; the dump under test is kept at `logs/jefftrain/G-m4-card-dump.safetensors`.

This note is this card's only tracked change. No shim, kernel, host source, test, `Makefile`, `compare_layers.py`,
`card-check.sh`, tolerance, gate, compute mode or fixture was changed, and the card was run once. For M5.1: adding
`grad.layer.{i}.out`, `grad.final`, `grad.readout` and the 300 LoRA factors makes the full-ref comparison self-consistent
again, and the forward half of that file is already exactly this dump's.
