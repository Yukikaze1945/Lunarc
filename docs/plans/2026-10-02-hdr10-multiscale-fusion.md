# HDR10 independent dual/triple DLSSNR and multiscale fusion

Date: 2026-10-02. Owner: Codex (decomposition and review). Since 2026-10-07,
implementation continues with Luna subagents at the user's explicit request;
Qoder is no longer used. Earlier Qoder evidence remains historical evidence.

## Goal and scope

Extend the existing experimental HDR10 GPU Transfer path to independent dual and
triple Feature 18 observations and the existing data-consistency fusion. Preserve
the verified single HDR path, SDR path, and the latest uncommitted FFmpeg remux
repairs. The user authorized automatic repair/progression and commit/push after
all final gates pass. Performance remains `NOT_CHARACTERIZED`.

Required final state:

- Full unit suite passes, including the nine motion-wrapper failures.
- One short latest-tree single HDR10 hardware E2E passes through real remux.
- Dual 50/75 HDR smoke passes with fusion actually executed.
- Triple 40/67/85 HDR smoke passes with fusion actually executed.
- Latest remux mapping, static HDR SEI and existing SDR regressions pass.

## Starting evidence and boundaries

- Starting HEAD: `7195f8979583fc48b4f47654f09841e82025cec4`, branch `main`.
- `gpu_hdr_transfer_export.negotiate_hdr` explicitly refuses multiscale today.
- `dlss_gpu_engine.Live` already opens independent format-aware layer sessions
  through `shared_with` and `dlss_gpu_host_open_layer_ex`; use `rgba16f` explicitly.
- `GuidanceServiceFlowProvider.new_independent_layer` provides independent NVOF
  engines/history. The existing SDR chain demonstrates ownership and diagnostics.
- `gpu_multiscale_fusion_bridge` has genuine forward/adjoint/CG machinery, but its
  full-source transport and final composite currently assume RGBA8. Those entries
  cannot serve HDR without a typed extension.
- The HDR bridge already computes a signed half residual against a pristine
  linear working master. Expose/reuse that observation and its producer lifetime;
  do not subtract proxy RGB codes and call them HDR residuals.
- The nine motion-wrapper failures use `Live.__new__` without the current color
  format / bytes-per-pixel initialization. Fix the fake fixture contract; preserve
  production validation, float32 motion, assertions and test coverage.
- Existing remux repairs and `HDR_REMUX_INTEGRATION.md` remain uncommitted. Preserve
  and include their relevant changes at final commit. Do not add `AGENTS.md`,
  `_selftest.txt`, `native/db_exports.bat`, or
  `native/nvenc_hdr_options_probe2.txt`, which are unrelated local files.

## Color and mathematical contract

For each distinct integer-percent scale in 25..100, use the existing
`resolve_working_dims` and serialized float32 Lanczos plans:

```
P010/PQ -> pristine linear scRGB FP16 full master x
each layer independently:
    x_i = D_i x
    fixed-job bounded FP16 proxy(x_i)
    independent Feature 18 / temporal and motion state
    existing local_linear_stops HDR reconstruction against pristine x_i -> z_i
    signed linear HDR residual r_i = z_i - x_i
```

NVOF reads a separate RGBA8 guidance plane only. Picture color and HDR residuals
remain FP16 or wider floating point, never RGBA8. Paper white / proxy policy is
fixed for the entire job and identical in policy across layers, never frame-adapted.

`catmull_baseline` executes all observations but outputs only the existing layer-0
Catmull baseline. `data_consistency_experimental` reuses the existing objective:

```
min_r sum_i w_i ||D_i r - r_i||^2 + beta ||r - r0||^2
H(v) = sum_i w_i D_i^T D_i v + beta v
b    = sum_i w_i D_i^T r_i + beta r0
final = x + residual_strength * r
```

Use all taps, sampling centers and duplicated clamped edge indices. The adjoint is
the transpose of those same linear float32 filters; FP16 storage rounding is not
itself a linear operator. Do not substitute resize for the adjoint. CG work vectors
may remain FP32; HDR master and final delivery remain FP16. Retain the current HDR
final clamp/encode rules; never apply the SDR [0,1]/[0,255] composite to HDR.

Weights, beta, tolerance and maximum iterations stay explicit. Nonconvergence or
invalid resources must fail with a reason, not select another algorithm silently.
No direct sum of layer residuals, repeated-session passes or residual pyramid.

## Ordered implementation jobs

### A. Format-safe native foundation and motion-wrapper repair

Progress audit (2026-10-02): initial Qoder job `qbg_20261002_8ca407f3`
crashed after 55m17s without a final result. No Qoder worker/CLI remained at
inspection. Only the motion-wrapper fixture repair is verified: Codex ran
`python -m pytest tests/unit/test_gpu_host_motion_wrapper.py -q`, 14 passed.
The native HDR DLL was rebuilt but its source has no new multiscale exports;
that binary change is not evidence that the foundation is implemented.

Continue A in two shorter jobs: A1 exposes the existing HDR bridge's GPU-only
master import, signed low residual observation and FP16 composite import,
with backward-compatible Python bindings; A2 adds typed FP16 transport and
composite to the existing fusion solver and runs its short synthetic gate.
Review A1 before starting A2; do not start B on incomplete native capability.

A1 review: job `qbg_20261002_0671c3f8` completed. Native GPU-only master import,
signed half observation, fenced full FP16 composite import, optional capability
group and Python bindings are present. Qoder reports 110 wrapper tests and 174
adjacent tests passed. Codex independently loaded the current DLL: optional group
supported/version 1, 72-byte counter struct; changed-file diff check passes.
Two review corrections belong at the start of A2: proxy provenance must follow
the working master it actually read (proxy-before-downscale must not become valid
after a later downscale), and scalar dispatch accounting must distinguish the
currently observe-only count from all native shader counter readbacks. Preserve
the actual existing composite rule: signed and unclamped until PQ encode; its
unrepresentable-half fallback to pristine master is counted. No SDR clamp in HDR.

Audit ownership and add the smallest backward-compatible FP16 extensions needed
to share a pristine master, obtain signed HDR layer observations, run existing
fusion in scRGB units, and feed the fused FP16 composite into existing PQ/P010
encode. Preserve old ABI entries and old SDR and single HDR behavior. Build only
changed native components and run relevant wrapper, HDR bridge and fusion tests.
No production export yet and no full unit-suite rerun at this intermediate stage.
Record the actual API/resource/fence contract and test/build evidence for review.

### B. Independent HDR layers, fusion and existing UI configuration

A2 review (2026-10-03): job `qbg_20261002_6f9937e2` completed. Typed
FP16 master/observation transport, shared FP32 D/Dt/CG, fenced FP16 composite
delivery, and both A1 corrections are implemented. The recorded targeted suite
has 533 passing tests. Codex verified the current DLL SHA256 values against the
raw gate, independently probed typed ABI 1, and checked the existing raw output:
HDR solver-to-bridge handoff and archived/current SDR composites are byte-identical.
Offline unweighted consistency on the same stored output decreases from
0.0045380174993597 to 0.0004620317312618251. The native solver converged in 7
iterations (relative residual 7.3425260779913515e-05); GPU output/reference error
includes the FP16 store grid. The adjoint pairing gate is CPU-side and native
agreement is transitive through the unchanged shared solver, not a direct HDR
GPU adjoint dispatch proof. Do not repeat the successful synthetic hardware gate.

One small A2 compatibility correction starts B1: `hdr_fusion_engine` currently
uses a strictly-positive validator for `residual_strength`, refusing zero while
the native and existing paths permit it. Accept finite nonnegative strength in
the typed wrapper; retain the existing production HDR range/semantics. Enforce
actual convergence and valid completion in the production handoff. B1 wires
the backend/lifecycle/diagnostics and packaging with targeted tests. B2 updates
existing UI controls and then performs C's final gates on the completed tree.

After A is reviewed, wire an HDR multiscale chain using `multi_scale_config`, the
new typed native interfaces, independent `Live(..., color_format='rgba16f')`
sessions and per-layer guidance providers. Reset each layer on job start and every
discontinuity/seek; recreate affected resources on scale/extent changes, and close
all owned resources on cancellation/failure/success. Keep single on its old path.

Reuse existing single/dual/triple, scale/weight/fusion controls. Remove only obsolete
HDR single-only blockers/preset overrides and update readiness/preview text. Keep
the experimental label, current audio/subtitle/container choices and SDR semantics.
Do not hardcode scale whitelist or assert that any scale is best.

Report layer scales, resolved dimensions, distinct opaque session IDs, reset flags,
DLSS and flow call counts, fixed paper white, actual fusion mode and per-frame
solver convergence. Extend transfer accounting over every layer and fusion stage.
Scalar CG/counter readback must be separately reported; do not claim total host
traffic is zero because picture readback is zero.

### C. Final short gates on the completed tree

Reuse the already valid rich source fixture
`bench_out/hdr_remux_export/hdr_remux_src.mkv`, preserving prior evidence in place.
Use a fresh ignored evidence directory and record source/module/DLL hashes.

1. Run the full current unit suite once; fix and rerun actual failures, then obtain
   a final green suite when repairs require it. No skip/xfail/tolerance relaxation.
2. Run one 4-6-frame single HDR10 production E2E on the completed tree through real
   audio/subtitle/chapter/metadata/attachment remux. Check Main10/10-bit, full extent,
   bt2020nc/smpte2084/bt2020, packet/frame counts and presentation PTS, measured host
   picture boundaries, source/intermediate/final mastering137 and CLL144 payloads,
   ffprobe side-data, track mapping/language/disposition and copied packet/attachment
   bytes. No silent fallback. Do not mistake B-frame packet order for display order.
3. Dual 50/75: one 4-6-frame HDR smoke with `data_consistency_experimental`, distinct
   sessions, correct per-layer calls/resets and converged solver. Same HDR and remux
   contract checks on its deliverable. Confirm independent close/reset/reopen with
   the smallest meaningful lifecycle probe if not covered by existing evidence.
4. Only after dual passes, triple 40/67/85: one 2-3-frame HDR fusion smoke, three
   distinct sessions, full HDR output and no host picture fallback. Reuse checkers;
   do not duplicate a whole benchmark matrix.
5. Short synthetic fusion evidence: true adjoint and existing reference agreement,
   objective and data-consistency before/after on signed HDR-range samples including
   values above 1 and negative components. Report actual numerical/FP16 rounding
   error without describing the result as visual quality improvement.
6. Existing SDR multiscale/remux regressions and relevant frontend tests/build pass.
   A compact SDR output regression is warranted if typed fusion changes its shared
   shader path. Do not rerun successful HDR A/B, roundtrip or performance gates.

Successful hardware gates stay recorded and are not repeated for checker-only
repairs. A production/native fix invalidating a result requires only the affected
short gate to be rerun against the corrected tree.

### D. Review, commit and push

Codex reviews final diff and raw gate evidence, verifies latest build hashes and
green results, and stages only this round's HDR multiscale changes plus the pending
related remux changes/docs/tests. Include tracked release DLLs when their sources
change; do not force-add ignored build/benchmark artifacts. Commit and push to the
verified remote branch, then verify remote HEAD. No force push or cleanup of local
unrelated files. Stop/delete the follow-up automation after verified completion.

## Orchestration

2026-10-07 transition: the final Qoder job `qbg_20261003_892ffb82` completed B1;
its report records 641 targeted passing tests but no production hardware export.
B2 UI integration and C final gates remain pending. Do not call Qoder again.
Codex owns task decomposition, backend audit, integration, gate review and final
commit/push. Use one Luna worker for bounded implementation and repair, with
disjoint write scopes during concurrent review. Preserve all existing evidence
and local changes. Ordinary build/test/partial failures receive a minimal
automatic continuation; only a demonstrated external SDK/API/hardware/Feature18
blocker warrants stopping. The existing 20-minute follow-up now uses this workflow
and remains quiet when nothing actionable changes.

## Final acceptance (2026-10-08)

B2 UI integration, independent-layer cleanup and per-layer diagnostic readback
projection are complete. Final current-tree unit suite: 1388 passed, 0 failed,
1 existing Windows GBK reader warning from the intentional double-free harness.
Single six-frame HDR10/remux: 38/38 checks; dual 50/75 six-frame actual fusion:
42/42; triple 40/67/85 three-frame actual fusion: 42/42. The final triple fixture
preserves original HDR video, audio, attachments, chapters and authored metadata,
but moves the first caption to 0-40 ms while retaining the second at 200-240 ms,
so the unchanged shared `-shortest` behavior preserves all three video packets
and an actual copied subtitle packet. Initial fixture failures are retained.

The final report `HDR_MULTISCALE_FINAL_ACCEPTANCE.md` records raw evidence paths,
distinct sessions, resolved extents, convergence, byte-identical HDR SEI and
remux payload checks, scalar/image traffic distinctions, prior accepted signed
synthetic/SDR evidence and remaining limits. No successful A/B, roundtrip or
benchmark gate was repeated; performance remains `NOT_CHARACTERIZED`.
