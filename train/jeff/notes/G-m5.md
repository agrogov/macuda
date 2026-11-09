STATUS: PASS

# G-m5 — card checkpoint m5: the backward

Plan M5.4. One run, `sh scripts/card-check.sh B m5` from `train/jeff`, with `JT_REF=/tmp/jeff-ref/ref.safetensors` and
M4.4's gpubox flags `JT_DUMP_ARGS="--model <HF snapshot> --init-adapter /tmp/jeff-ref/adapter --train
/tmp/jeff-ref/fx.jtb"`, exact mode (`JT_EXACT=1 TINYCUBLAS_TC=0`, from `card-check.sh`; the shim build carries no
`-DJT_EXACT`, per the checkpoint rules).

Card: **B**, RTX 4090 (PCI `0x2684`, 24564 MB), `ARCH=sm_89`, `SHIM=$MACUDA/cuda-shim/build/shim-sm_89` — not `sm_120a`.
Left quiesced by the step itself ("the card is cold, and the next open needs no reset"). One run, not retried.

## Checkpoint

```
=== step 1: JT_EXACT=1 TINYCUBLAS_TC=0 jefftrain --model <snapshot> --init-adapter /tmp/jeff-ref/adapter \
    --train /tmp/jeff-ref/fx.jtb --dump-backward <tmp>/m5.safetensors
   exit 0 after 6 s
compare_layers: /tmp/jeff-ref/ref.safetensors (reference) vs <tmp>/m5.safetensors: 403 vs 403 keys,
                atol-rel 0.001, grad-rel 0.005
compare_layers: 403 ok, 0 failed, 403 key(s) compared
checkpoint m5: 1 steps, 0 failed
```

`403 vs 403 keys` is the whole point of this checkpoint versus m4: `--dump-backward` writes the same 403 keys
`ref.safetensors` has, so unlike m4's forward-only dump there is no key in the reference left unaccounted for, and the
gate is the plan's own command with nothing subset out and nothing loosened.

## Numbers

**Worst error in the gate: 2.345e-05 relative (`grad.language_model.layers.11.self_attn.k_proj.lora_A`) — 213x inside
the 5e-3 gate.** Every group the Done-when names:

| group | keys | worst rel | gate | margin | worst key |
|---|---|---|---|---|---|
| `grad.layer.{i}.out`, i = 0..23 | 24 | **7.884e-06** | 5e-3 | 634x | `grad.layer.16.out` |
| `grad.{stem}.lora_A/B` | **300** | **2.345e-05** | 5e-3 | 213x | `layers.11.self_attn.k_proj.lora_A` |
| `grad.readout` | 1 | 7.184e-06 | 5e-3 | 696x | — |
| `grad.final` | 1 | 4.884e-06 | 5e-3 | 1024x | — |
| forward keys (`in.*`, `layer.*`, `final`, `logits`, `loss`) | 77 | 5.693e-06 | 1e-3 | 176x | `layer.15.post_mixer` |

All 300 LoRA grads are present and inside the gate, and they are exactly 150 LoRA pairs (7 stems x 2 factors on each of
the 6 attention layers = 84, 6 stems x 2 on each of the 18 GDN layers = 216), not a subset: 18x12 + 6x14 = 300.
`in.ids`/`in.off` exact; `loss` exact (`1.42077208` against the reference's `1.42077208`).

| layer | `grad.layer.{i}.out` | LoRA keys | worst LoRA | worst key of the layer |
|---|---|---|---|---|
| 0 | 6.549e-06 | 12 | 1.686e-05 | `mlp.down_proj.lora_A` |
| 1 | 6.491e-06 | 12 | 1.868e-05 | `linear_attn.in_proj_qkv.lora_A` |
| 2 | 6.620e-06 | 12 | 1.281e-05 | `mlp.gate_proj.lora_A` |
| 3 | 6.180e-06 | 14 | 2.004e-05 | `self_attn.q_proj.lora_B` |
| 4 | 6.348e-06 | 12 | 1.540e-05 | `mlp.gate_proj.lora_A` |
| 5 | 6.436e-06 | 12 | 1.558e-05 | `linear_attn.in_proj_qkv.lora_A` |
| 6 | 6.549e-06 | 12 | 1.926e-05 | `linear_attn.in_proj_z.lora_A` |
| 7 | 6.580e-06 | 14 | 1.998e-05 | `self_attn.v_proj.lora_A` |
| 8 | 6.284e-06 | 12 | 1.597e-05 | `linear_attn.in_proj_qkv.lora_B` |
| 9 | 6.215e-06 | 12 | 1.603e-05 | `mlp.up_proj.lora_A` |
| 10 | 6.091e-06 | 12 | 1.488e-05 | `mlp.up_proj.lora_B` |
| 11 | 6.291e-06 | 14 | **2.345e-05** | `self_attn.k_proj.lora_A` |
| 12 | 6.460e-06 | 12 | 1.223e-05 | `mlp.up_proj.lora_A` |
| 13 | 6.690e-06 | 12 | 1.449e-05 | `linear_attn.in_proj_qkv.lora_B` |
| 14 | 7.471e-06 | 12 | 1.335e-05 | `mlp.up_proj.lora_A` |
| 15 | 7.497e-06 | 14 | 1.402e-05 | `self_attn.v_proj.lora_A` |
| 16 | **7.884e-06** | 12 | 1.424e-05 | `mlp.up_proj.lora_B` |
| 17 | 6.667e-06 | 12 | 1.646e-05 | `linear_attn.in_proj_z.lora_B` |
| 18 | 6.581e-06 | 12 | 1.314e-05 | `linear_attn.in_proj_qkv.lora_B` |
| 19 | 5.839e-06 | 14 | 1.155e-05 | `self_attn.k_proj.lora_B` |
| 20 | 6.037e-06 | 12 | 1.524e-05 | `mlp.gate_proj.lora_B` |
| 21 | 5.815e-06 | 12 | 1.714e-05 | `mlp.up_proj.lora_B` |
| 22 | 6.249e-06 | 12 | 1.476e-05 | `linear_attn.in_proj_z.lora_B` |
| 23 | 6.156e-06 | 14 | 1.394e-05 | `mlp.gate_proj.lora_A` |

Against the gpubox run of the same test (J30: `grad.layer.*` worst 3.624e-06, LoRA worst 1.205e-05, `grad.final`
2.047e-06, `grad.readout` 1.716e-06): 2.0-2.2x larger on the shim — same order of magnitude, a different GEMM/Norm
summation order, not a layout error (those are O(1)). No layer is an outlier: `grad.layer.*` is flat at 5.8-7.9e-06
across all 24 layers, which is what the sweep should look like when the mixer's parked saves are correctly replayed
(J30's second deliberate break put layer 23's `self_attn.*` at 1.0-3.5 while its `mlp.*` stayed at 2.6e-06; nothing has
that shape here).

```
gemm: exact fp32
lora: entries 151 (150 LoRA pairs + readout)  rank 16  scale 2.0  trainable 10,484,736
model: 24 layers (18 gdn, 6 attn), weights 1.99 GB fp32 + embed 0.51 GB bf16
model: 321 frozen tensors checked by st_get (dtype and every dimension), readout 1.04 MB fp32 in the flat buffer
jefftrain: dump-backward: fx.jtb batch 0 = rows [0, 1], lengths [37, 70], n 107 tokens, 5 option(s), gw 2
ctx: lmax 70, arena 0.04 GB fp32 (107 tokens x 98304 floats + 2 x 8 x 70^2), gdn nseg 2, ckpt 0.00 GB, bwd scratch 0.14 GB
jefftrain: backward: 24 layers recomputed from their saved inputs and differentiated
jefftrain: wrote m5.safetensors: 403 keys (the 77 forward keys + 24 layer-gradient keys + grad.final + 300 LoRA grads
           + grad.readout), 84.9 MiB of data
```

Card memory was never short: **arena 0.04 GB fp32**, `ckpt 0.00 GB`, `bwd scratch 0.14 GB` on a 24 GB card — no OOM, and
the fixture was not touched (`n = 107`, the plan's 2-row batch). The dump under test is byte-for-byte the file kept as
`G-m5-card-dump.safetensors` (84,936,284 bytes, `sha256 dba23ec06721f07370384e30e7edcc02c3f66ee2c9d34e596bec087d39617a7f`).
Wall cost of the step: 6 s including the 2.5 GB model load and the 0.04 GB arena.

## Prerequisites and environment

* `sh scripts/prereq-check.sh J02 J30` → `prereq: J02 J30 PASS`.
* `tools/preflight.sh`, before the step → `VERDICT: OK to proceed`: `NVIDIA 0x2684, link up, x4 at 2.5 GT/s`,
  `IOPCIResourced = Yes`, `DART error data absent, no snapshot: clean`, `dext instances: 2, newest pid 78368 started
  2026-11-08 23:15:36`, `lowpowermode 0`. The dext line read `- recorded (first sighting)` on that run: the worktree's
  `.dext-seen` record did not exist yet (m4's record is in the other tree), so preflight seeded it from the same pid
  and start time m4 recorded. That is neither the documented replug case (`a new dext instance since the record`) nor a
  stale DART record, so `sh tools/dart-stale.sh --clear` was correctly **not** run and there is no second verdict to
  reconcile. The same gate re-run after the step (the log kept below) reads `newest pid 78368 … unchanged since the
  record` plus `VERDICT: OK to proceed`, i.e. the card was not re-enumerated by the run.
* `make shim ARCH=sm_89 SHIM=$MACUDA/cuda-shim/build/shim-sm_89` → rc 0, 0 warnings (`build/shim/.arch` = `sm_89`);
  `make test-null` → `test-null: 8 test(s) passed with no arguments on the null device`; `make check-symbols` → `all 23
  cuda*/cublas* symbol(s) the objects need are defined by the shim` (same 23 as m4: `--dump-backward` needed no new one).
* Model identity, before the card step (plan M4.4): `shasum -a 256 *.safetensors` in the Mac HF snapshot
  `snapshots/da320e510c6eaa15a36a2eeaa2aad9d2370321fe` vs `ssh gpubox 'cd ~/models/jeff-base && sha256sum
  *.safetensors'` — **both shards identical** (`model.safetensors 22fd9a1e…2f9b`, `readout.safetensors
  fb175582…c486`), so the model is the one `ref.safetensors` was made from.
* Reference staging unchanged from m4: `rsync -a gpubox:~/macuda-train/ref/ /tmp/jeff-ref/` (`ref.safetensors` 84.9 MB,
  `adapter/`, `fx.jtb`, plus m4's `ref-fwd.safetensors`); synthetic rows and a public model, so it stays out of the repo
  (plan rule 5). Driver stack `libtinynv build ba4e38d`, macOS SDK 26.5, `tinygpu.sock` already listening (pid 12676);
  the driver's own step log printed it unloaded the firmware afterwards: `gsp-rm unloaded and its region torn down in
  0.26 s: the card is cold, and the next open needs no reset`. PCIe x4 at gen1 (2.5 GT/s, target gen3) — a trained-down
  link costs bandwidth, not correctness: 540 upload claims, 0.278 s total spent (0.089 s host memcpy, 0.049 s
  build+submit, 0.140 s waiting for the card).

## Logs and files

`logs/jefftrain/` (git-ignored): `G-m5-card-check.log` (the whole run, all 403 key lines), `G-m5-shim-build.log`,
`G-m5-preflight.log` (the post-run verdict; the pre-run one is quoted above), `G-m5-card-dump.safetensors` (the file
gated above), `G-m5.md` (full record with all excerpts); the driver's own step log is
`logs/shim-probe-20261109-022958.log`.

This note is this card's only tracked change. No shim, kernel, host source, test, `Makefile`, `compare_layers.py`,
`card-check.sh`, tolerance, gate, compute mode or fixture was changed, and the card was run once (one `card-check.sh`
invocation, one step). `/tmp/jeff-ref/` was left as it was.
