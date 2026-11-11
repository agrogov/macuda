# RTX 4090 tensor unlock — technical design

Date: 2026-11-11. Status: proposed architecture, subject to transport and hardware validation.

Requirements: [specification](spec.md). Execution gates: [plan](plan.md).

## Architecture decision

Implement an isolated pre-boot transaction in `libtinynv`, using the existing TinyGPU transport. Reuse the upstream AD102 recipe and asset preparation for personal use; adapt Linux resource management to macuda explicitly. Do not port Linux module registration, DKMS, initramfs hooks or Linux suspend notifiers.

No initial changes to `libtinycudart` or `libtinycublas` are expected. The normal BF16 kernels remain the performance probe. Kernel tuning is a separate later decision.

## Existing integration constraints

`tinynv.c:device_boot` calls `tinynv_gpu_open` before firmware boot. `gpu.c:tinynv_gpu_open` currently combines device early initialization, device MMU initialization and allocator creation. `dev.c` can reset the device during early initialization. `gpu.c:unmake_boot` can reset again during firmware retries.

Consequently, inserting a firmware transaction after `tinynv_gpu_open` and then resetting would invalidate assumptions about initialized device state. The implementation must establish an earlier transaction boundary or fully tear down and recreate affected state. Do not preserve MMU or queue state across the transaction FLR.

`flcn.c` already handles AD102 firmware, a SEC2 scrubber and normal boot/unload. Those facilities are useful references, but their current helpers are not assumed to accept the unlock transaction's signature data or DMA layout.

`pci_tinygpu.c` exposes config access, reset and quiesce. Audit their actual semantics before reuse: a method named reset is not sufficient proof that it matches the upstream transaction's FLR and retention properties.

## Proposed components

| Component | Responsibility |
|---|---|
| `tensor_unlock.c/.h` | Eligibility, transaction state machine, checked register snapshots and result |
| Unlock asset preparation script and manifest | Pinned source revision, extraction, payload generation, asset hashes and format validation |
| Internal transaction transport adapter | BAR/config access, bounded waits, temporary device-visible memory, reset and containment |
| `gpu.c` / `dev.c` integration | Split minimal access from normal initialization; verify state after resets and boot |
| Existing boot diagnostics | Publish unlock state beside build/firmware provenance |

These are proposed files and boundaries, not implemented APIs. Keep transaction-specific temporary memory separate from the normal application allocator.

## State machine

`OFF -> normal boot`

`REQUESTED -> ASSETS_VALIDATED -> DEVICE_CLAIMED -> COLD_ACCESS -> INSPECTED`

From `INSPECTED`:

- Valid retained override and proven cold handoff: proceed to normal initialization.
- Stock eligible cold card: prepare and execute the unlock transaction.
- Live firmware, inaccessible hardware or ambiguous state: stop, or take only an explicitly validated cold-reset path before reinspection.

Transaction path:

`PREPARED -> FWSEC_READY -> PRIVILEGED_STAGE_COMPLETE -> SELECTORS_VERIFIED -> RESET_CONTAINED -> NORMAL_BOOT -> POST_BOOT_VERIFIED -> READY`

Any failure enters `FAILED`, records the first error, and follows containment cleanup appropriate to the last reached state. Do not automatically rerun a partially executed unlock in the same open. Normal firmware retries may proceed only after reset retention and resource ownership are verified.

## Hardware transaction and checks

The upstream AD102 recipe uses a signed booter with crafted signature data to obtain privileged writes. It then allows host-side selector updates. A direct host write on an otherwise stock card is insufficient. Use an immutable reviewed recipe; never derive offsets from the running firmware version or combine payloads across chip families. [Recipe source](https://github.com/Zagitta/gpu-unlock/blob/main/kernel/gpu_unlock/recipes.c)

The register manifest includes the following AD102 BAR0 fields, taken from the [upstream register definitions](https://github.com/Zagitta/gpu-unlock/blob/main/kernel/gpu_unlock/registers.h):

| Field | Offset | Purpose |
|---|---:|---|
| FEAT permission | `0x00823804` | Check access gate |
| Effective speed readout | `0x00823818` | Validate effective state under mask `0x07ffffff` |
| SS0 | `0x0082381c` | Requested override `0x88888888` |
| SS1 | `0x00823820` | Requested override `0x8` |

These values are documentation, not an instruction to write registers manually. The transaction must use the full recipe and its completion/permission gates, including WPR-related handling. Reject all-ones and hardware error-pattern reads before interpreting fields.

For each stage record register values and validity independently. A mailbox marker without engine completion and permission readbacks is insufficient. Likewise selector readback alone is insufficient without effective state and post-boot checks. Exact marker and masks must come from the pinned current recipe, not an older experiment's payload.

## Memory ownership and reset

The adapter must distinguish CPU pointers, GPU physical addresses and host IOVAs; Linux coherent DMA assumptions cannot be transplanted directly to Apple DART mappings. Prove placement, alignment, address-width constraints, synchronization and lifetime for each buffer.

Allocate and validate transaction memory before publishing addresses to firmware. Mark it device-owned at publication. Release only after verified engine termination and DMA containment. Quiescing bus mastering is necessary on failure but must also account for pending transactions and firmware completion before unmapping. Reuse the project's established containment ordering without assuming it covers a new SEC2 transaction automatically.

A process-local allocation leak does not survive process exit. If failed firmware can retain access, a real transport-owned retained mapping or equivalent containment guarantee is required before live execution. This is an implementation gate, not deferred cleanup.

After successful reset, restore PCI state and rebuild normal device/MMU/allocator state. Verify that no transaction mapping remains referenced. Retention must be read again after all normal initialization resets and after runtime firmware is running.

## Firmware bundle separation

The upstream build currently pins NVIDIA 610.43.03 inputs; macuda's documented normal runtime images are 570.144. Keep the unlock bundle separate from the runtime bundle. Do not update normal runtime firmware as an incidental dependency. Their sequential compatibility is an open hardware question.

Manifest fields: upstream repository and commit; extraction/generator revision; original input hashes; output hashes and sizes; chip/device allowlist; recipe identity; expected binary layout; required notices. Asset discovery must be deterministic and must not download during device open. [Upstream firmware provenance](https://github.com/Zagitta/gpu-unlock/blob/main/firmware/NOTICE.md)

## Lifecycle and observability

Software state is per open; GPU override state may outlive a process. `off` means do not apply the transaction, not force stock state. A stock benchmark requires a verified power-cycle/reset method that actually clears the override; do not assume FLR clears it.

Record snapshots before the transaction, after selector application, after reset, after runtime boot and after workload. On subsequent opens inspect hardware again. On lost power, disconnect or unsupported resume, invalidate software status and require a fresh open. Automatic resume handling is out of scope.

Failure logs identify the last successful stage and whether replug/power-cycle is required. Do not attempt to restore stock selectors through an unvalidated reverse transaction.

## Open decisions before live implementation

1. Which immutable upstream AD102 recipe is selected, and has that revision itself been reproduced?
2. Does TinyGPU provide the exact reset and temporary-memory guarantees required?
3. Can transaction allocations be retained safely across failure and process exit?
4. Does the normal AD102 scrubber/runtime boot preserve the override?
5. Can normal boot retries retain the override without repeating the privileged stage?
6. Which real BF16 model fits this 24 GB card and provides stable prefill measurements?

The design is feasible in principle; none of these questions is answered by stock NVIDIA driver success alone.
