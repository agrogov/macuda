# RTX 4090 tensor unlock — specification

Date: 2026-11-11. Status: proposed; no implementation or macuda hardware validation.

Scope: personal use on the owner's Mac with an RTX 4090 over Thunderbolt. This is not a distribution project. Companion documents: [design](design.md), [implementation and validation plan](plan.md).

## Objective and evidence

Increase BF16 matrix multiplication performance by enabling the AD102 tensor speed override before macuda starts its normal GPU firmware. Preserve BF16 inputs, FP32 accumulation and existing CUDA/cuBLAS contracts.

The published RTX 4090 experiment reports approximately twice the BF16/FP32 instruction throughput on one card. It explicitly measures instruction issue rather than an application and leaves independent cold-boot reproduction and representative numerical validation pending. This is evidence that the mechanism can work, not a macuda performance guarantee. See [the primary experiment](https://github.com/elias-gaeros/gpu-unlock/blob/rtx4090/rtx4090-register-report.md).

Locally, `cuda-shim/libtinycublas/gemm.cu` already has BF16 WMMA tensor kernels with float accumulators. The feature should initially reuse those kernels unchanged. Whether their execution is limited by tensor issue, memory access or occupancy depends on shape and must be measured.

## Scope and exclusions

- Support only the desktop RTX 4090, PCI vendor/device `10de:2684`, with AD102 chip identity independently checked. Reject other devices when the option is requested.
- Run through macuda's existing TinyGPU hardware interface on this Mac.
- Optimize BF16 with FP32 accumulation. FP16/FP32 may benefit incidentally but is not the primary acceptance workload.
- Do not add Ampere, other Ada cards, 4090 D, laptop GPUs or Blackwell support.
- Do not change clocks, power limits, model precision, VBIOS or normal GEMM arithmetic.
- Do not promise faster memory-bound decode or integer tensor operations.
- Initial implementation excludes automatic sleep/resume recovery, concurrent GPU clients and hot-unplug recovery. Reopen and verify after a power transition.

## Proposed interface

`TINYNV_TENSOR_UNLOCK=0|1`, default `0`. Reject other nonempty values. All names in these documents are proposals, not existing options.

With `0`, preserve the current initialization path, without unlock assets, allocations or additional unlock MMIO accesses. Disabling the option does not undo an override already retained by the powered GPU.

With `1`, require the eligible device, validated assets, exclusive access and successful post-boot verification. Fail the open if the requested unlock cannot be established. Do not silently benchmark a stock card as unlocked.

Report stage, device identity, asset revision, raw selector/permission values, elapsed initialization time and final status. Distinguish `verified-existing`, `applied-and-verified`, `unsupported`, `failed-preflight` and `failed-after-start`. A successful function return alone is not proof.

## Functional requirements

1. Validate chip, PCI identity, BAR accessibility, power/link state, reset capability and exclusive ownership before launching firmware.
2. Load a separately pinned AD102 unlock bundle. Do not substitute macuda's ordinary runtime booter or an Ampere payload. Validate hashes, sizes and every parsed range before device writes.
3. Inspect the existing override state. A retained override may skip reapplication only after validating effective readout and a usable cold handoff state. An inaccessible register is an error, never evidence of success.
4. Execute the selected upstream AD102 transaction, verify its completion and required permissions, set selectors, and verify the effective override.
5. Reset and restore PCI configuration using a verified FLR-equivalent operation; demonstrate that firmware DMA has stopped before releasing its buffers.
6. Start macuda's existing runtime firmware and verify the override again before creating application work queues or accepting compute launches.
7. Recheck after any initialization retry/reset; never reuse a stale software success flag.
8. On failure, preserve the first diagnostic and stop normal initialization. Contain DMA before unmapping host memory. If containment cannot be proven, retain the backing allocations through a transport-owned mechanism or refuse firmware launch until such a mechanism exists.

## Acceptance criteria

| Area | Required evidence |
|---|---|
| Default path | Existing applicable offline checks pass; disabled-path recorded MMIO behavior remains unchanged |
| Identification | Wrong device/chip and inaccessible reads are refused without launching unlock firmware |
| Transaction | Required completion, permission, selector and effective-readout checks pass |
| Handoff | Override survives transaction reset, runtime boot and an actual BF16 workload |
| Correctness | Existing GEMM tolerances pass with explicit non-finite rejection; representative application results pass the existing correctness protocol |
| Lifecycle | At least three independent enclosure power cycles and ten sequential open/workload/close cycles succeed |
| Performance | Interleaved matched stock/unlocked BF16 GEMMs and prefill show an improvement outside observed run variation; publish all shapes, including neutral/regressing cases |
| Cleanup | Failure injection demonstrates ordering of DMA containment and release; live runs show no new DART fault, MMU fault or unexplained hang |

No fixed application speedup is required to establish a correct unlock. If hardware state is verified but representative BF16 workloads do not improve, report that outcome and keep the feature experimental and off by default.

## Sources and provenance

Research reviewed 2026-11-11. Online `main` and branch links are moving references; implementation must record immutable commit IDs and asset hashes before reuse.

- [gpu-unlock overview](https://github.com/Zagitta/gpu-unlock): supported generations and operating model.
- [AD102 recipes](https://github.com/Zagitta/gpu-unlock/blob/main/kernel/gpu_unlock/recipes.c): chip-specific payload and recipe.
- [Transaction implementation](https://github.com/Zagitta/gpu-unlock/blob/main/kernel/gpu_unlock/unlock_610.c): lifecycle and hardware operations.
- Local baseline: `cuda-shim/libtinynv/src/{gpu,dev,flcn,tinynv,pci_tinygpu}.c`, `cuda-shim/libtinycublas/{gemm.cu,cublas.c}`.

The owner explicitly scoped this to personal use and accepted licensing for that scope. Preserve upstream notices and provenance when reusing source. No publication or redistribution work is required by this plan.
