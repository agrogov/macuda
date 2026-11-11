# RTX 4090 tensor unlock — implementation and validation plan

Date: 2026-11-11. Status: planning only. This document does not authorize or perform hardware experiments by itself.

Scope: personal RTX 4090 use on this Mac. Read [spec](spec.md) and [design](design.md) first. Work sequentially through the gates below; keep the feature off by default throughout.

## Phase 1 — freeze inputs and establish the baseline

- Record macuda commit and working-tree state, Mac/macOS, TinyGPU version, enclosure, PCI identity, AD102 identity, VBIOS, link state and normal firmware hashes.
- Pin a gpu-unlock commit and inspect its full AD102 call graph, including asset extraction, layout, Falcon execution, reset, success gates and failure unwind. Compare the pinned recipe with the published experiment; do not assume moving branches are equivalent.
- Preserve upstream notices with reused source/assets for personal use. Public packaging is outside scope, per the owner.
- Confirm the unchanged macuda AD102 path works on this exact setup before attributing any boot issue to the unlock.
- Establish stock measurements after a genuine enclosure power cycle. Capture selector/effective-readout state; disabling a software flag does not prove stock state.

Deliverable: immutable input manifest, source audit notes and baseline logs. Gate: normal AD102 boot and representative BF16 correctness pass.

## Phase 2 — transport and resource feasibility

Audit TinyGPU config/BAR access, FLR semantics, mapping lifetime, device-visible memory synchronization and failure cleanup. Trace reset behavior through `pci_tinygpu.c`, `dev.c`, `gpu.c` and the TinyGPU server implementation.

Determine how to execute before the normal MMU/allocator initialization and how to contain or retain transaction buffers if firmware never stops. Document explicit CPU/IOVA/physical-address conversions.

Deliverable: supported-operation matrix and ownership diagram. Gate: all mandatory transaction and cleanup operations are implementable; otherwise stop before launching firmware and identify the smallest transport extension needed.

## Phase 3 — offline implementation

1. Add strict opt-in parsing, desktop RTX 4090/AD102 eligibility and structured status reporting.
2. Add deterministic manifest-driven asset preparation and checked parsing. Keep normal runtime assets unchanged.
3. Implement the isolated transaction adapter and state machine.
4. Refactor initialization only as needed to execute before persistent normal mappings exist.
5. Add post-reset and post-runtime-boot verification, including normal firmware retry paths.
6. Keep CUDA runtime entry points and BF16 kernel sources/cubins unchanged.

Meaningful offline checks:

| Test | What it must detect |
|---|---|
| Disabled mode | No unlock-specific allocations, asset reads or MMIO |
| Eligibility and parsing | Wrong chip, missing/corrupt assets, bad ranges, unsupported option values |
| Stage gates | Marker-only false success, bad permissions, wrong selectors, bad effective readout |
| Accessibility | All-ones/error-pattern registers and lost PCI identity |
| Lifecycle failures | Timeout at every firmware stage, reset failure, lost retention, normal boot retry |
| Ownership | Device-visible memory cannot be released before containment; first error survives cleanup |
| Existing suites | Applicable libtinynv replay/unit tests and shim contract checks remain green |

Use synthetic transaction traces for new operations; do not rewrite an old recording merely to conceal a changed default path. Mock tests prove control flow, not hardware completion or DMA safety.

Deliverable: reviewable implementation with passing offline checks. Gate: no unresolved resource-lifetime or reset assumptions.

## Phase 4 — first live transaction

Use the repository's operating protocol: read-only preflight, exclusive GPU lock, one GPU process, controlled release. Ensure no server or other client owns the card. Use external laptop power as described in the project README.

Start from a power-cycled enclosure. Capture stock state, execute one transaction and save each stage's diagnostics. Stop before inference until reset containment and retained override are verified. Then boot normal macuda firmware, verify again, run a small known-correct BF16 GEMM and close normally.

Do not mask failures by looping the privileged transaction. Preserve diagnostics before recovery. Follow existing quiesce/replug procedures; do not experiment with arbitrary resets after an ambiguous failure.

Deliverable: complete transaction/handoff log with asset/build provenance. Gate: valid hardware evidence at every stage and no new DART/MMU fault.

## Phase 5 — correctness and lifecycle

- Exercise aligned and partial BF16 GEMMs, supported transposes, float/BF16 outputs, alpha/beta, strided and pointer-batched APIs, and split-K on/off where applicable.
- Use the existing numerical reference and tolerances with explicit NaN/Inf rejection. Compare stock and unlocked runs using identical cubins and inputs; investigate any output difference rather than relaxing tolerance.
- Run existing relevant ggml operation checks and deterministic application probes with a recorded model hash.
- Complete at least three independent enclosure power-cycle trials and ten sequential open/workload/close trials. Record state after boot, workload and close/reopen.
- Exercise normal boot retry through a controlled test mechanism where possible. Do not create an unsafe live firmware fault solely to force a retry.
- Verify that off-mode remains usable on a fresh stock power cycle and on a card retaining a previously applied override.

Deliverable: pass/fail matrix and all failure logs. Gate: numerical and lifecycle criteria in the spec pass.

## Phase 6 — performance study

Separate instruction throughput, real GEMMs and complete BF16 prefill. An instruction-only gain does not establish inference speedup.

Use an A/B/A/B sequence of verified stock/unlocked sessions. Because the override may survive FLR, stock sessions require a verified clearing power cycle. Match warmup, temperature range, clock/power policy, model, prompt, graph configuration, build, cubin and GEMM dispatch. Log clocks, power, temperature and link state throughout; avoid comparing a cold baseline with a warm optimized run.

Measure:

- A register-resident BF16/FP32 tensor probe to establish instruction-rate change.
- Actual model GEMM shapes, including small N, large prefill, partial tiles and split-K cases. Report time and dense operation counts consistently.
- End-to-end BF16 prefill for a model fitting 24 GB, with fixed prompt and batch configuration.
- Decode as a secondary result, without assuming it improves.
- Initialization time separately from steady-state computation.

For every session retain raw timings, use at least five measured repeats after warmup and report median plus range or spread. Repeat matched sessions across power cycles. Report neutral/regressing cases and thermal/power throttling; do not select only the best run. Require improvements to exceed the variation in stock repetitions.

Deliverable: reproducible benchmark report with correctness results and qualified conclusions. Gate: decide whether the feature is useful for this owner's actual workload; no preset 2x application target.

## Phase 7 — personal-use completion

Document setup, opt-in invocation, verified hardware/assets, known lifecycle limitations, recovery and how to return to a verified stock state. Retain source notices. Keep off by default and mark RTX 4090-only experimental support even if local validation succeeds.

Only after the unchanged-kernel study should a separate proposal consider GEMM tiling or split-K tuning. Existing dispatch includes constants derived from a 170-SM target; a 4090-specific tuning study may be useful, but combining it with the unlock would obscure attribution.

## Stop conditions

Stop live testing on an unexplained DMA/DART fault, lost hardware access, uncontained transaction memory, failed reset handoff, incorrect numerical output or lost override after normal boot. Record the last verified state and the required recovery. Resume only after the cause or containment gap is resolved.

## Research references

- [Upstream repository](https://github.com/Zagitta/gpu-unlock)
- [Current transaction source](https://github.com/Zagitta/gpu-unlock/blob/main/kernel/gpu_unlock/unlock_610.c)
- [Current AD102 recipe](https://github.com/Zagitta/gpu-unlock/blob/main/kernel/gpu_unlock/recipes.c)
- [Published RTX 4090 experiment](https://github.com/elias-gaeros/gpu-unlock/blob/rtx4090/rtx4090-register-report.md)

These are research entry points. Replace moving-reference dependencies with pinned revisions during Phase 1.
