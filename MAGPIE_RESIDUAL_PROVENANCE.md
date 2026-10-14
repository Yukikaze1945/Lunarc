# Magpie Residual Reconstruction Provenance

Scope: the CPU reference (`pipeline.py`) and the native hot path
(`native/residual_bridge.cpp`) in this repository, against the Magpie checkout under
`Magpie/`. Each "Observed behavior" cell was read out of the HLSL/C++ in that checkout
on 2026-09-19, not out of a previous version of this document. Each "Proven by" cell
names a test that runs and passes at the current HEAD; anything without one is listed
under *Not equivalent / not proven* instead of being claimed.

## Stage table

| Stage | Magpie source | Observed behavior | Standalone implementation | Proven by |
|---|---|---|---|---|
| Working color | `DLSSNRFilter.cpp` `DownsampleColorVertical` then `DownsampleColorHorizontal` | `scale = target/source`; `position = (d+0.5)/scale - 0.5`; `support = 2.0/scale`; taps `ceil(position-support) .. floor(position+support)`; weight `Lanczos2((i-position)*scale)`; accumulator divided by its own weight sum | Python: `ResamplerPlan`/`AxisPlan` + `_apply_axis_plan` pair, vertical axis first. Native: same order in `rr_downsample_bgr8` | `test_lanczos_8x8_to_4x4_matches_scalar_oracle`, `test_lanczos_non_integer_scale_matches_scalar_oracle`, `test_native_lanczos_8x8_to_4x4_matches_scalar_oracle`, `test_native_lanczos_non_integer_scale_matches_scalar_oracle`, `test_lanczos_reproduces_linear_ramp` |
| Anti-aliased tap count | same | Support is in source units, so 50% reads 8 samples and 25% reads 16 | `AxisPlan.tap_counts`, padded to `max_taps` with weight 0.0; native mirrors `tap_counts` | `test_lanczos_tap_count_widens_with_downscale_ratio`, `test_native_plan_reports_four_axis_plans` |
| Edge handling | `clamp(x, 0, SourceExtent-1)` inside the tap loop | Out-of-range taps repeat the border sample; no tap is dropped, no window is re-centred | Plan indices are clamp-to-edge at build time; both backends index with them | `test_lanczos_8x8_to_3x3_matches_scalar_oracle` (impulse and edge-impulse hit both borders) |
| Normalization | `total / (abs(totalWeight) > 1e-6 ? totalWeight : 1.0)` | Guard on the absolute value, so a negative weight sum still divides | Same rule per axis row at plan build; native builds in double then stores float32, like the reference | `test_lanczos_tap_count_widens_with_downscale_ratio` |
| Signed residual | `PrepareResidual` | `denoised - original` kept signed, never clamped | FP32 in both backends; native computes it inside `rr_reconstruct_bgr8` | `test_signed_residual_preserves_negative_values`, `test_native_signed_residual_positive_negative_mixed` |
| Residual upsample | `UpsampleResidualHorizontal` then `CompositeResidualVertical` | `position = (d+0.5)*src/ dst - 0.5`; `center = floor(position)`; exactly 4 taps `center-1..center+2`; Keys B=0,C=0.5; negative lobes kept; clamp-to-edge; divided by weight sum | `catmull_x`/`catmull_y` plans, horizontal axis first, vertical fused with the composite in the native path | `test_catmull_rom_4x4_to_8x8_matches_scalar_oracle`, `test_catmull_rom_non_integer_scale_matches_scalar_oracle`, `test_catmull_rom_preserves_negative_lobes`, `test_native_catmull_rom_4x4_to_8x8_matches_scalar_oracle`, `test_native_catmull_rom_non_integer_scale_matches_scalar_oracle` |
| Residual strength | `ApplyResidualControls` at low resolution, then `saturate(original + residual)` | Multiplier applied in the **low-resolution domain**, plus saturation and lightness controls that this project does not have | Multiplier applied to the **upsampled** residual (`np.rint` composite), no saturation/lightness. See *residual_strength semantics* below | `test_residual_strength_zero`, `test_residual_strength_one`, `test_residual_strength_one_point_five`, `test_native_strength_zero`, `test_native_strength_one`, `test_native_strength_one_point_five` |
| Final clamp | `saturate()` on an unorm8 store | The only clamp of the chain; the store rounds | `np.rint(np.clip(...))` / `std::rint` after clamping; `astype()` truncation is deliberately avoided | `test_catmull_rom_residual_reconstruction_matches_scalar_reference`, `test_native_parallel_output_is_deterministic_and_matches_reference` |
| Guidance inputs | `DownsampleGuidance` | Motion and depth arrive at the working extent | `FlowProvider` (zero / guidance-service adapter / cached `.flo`); depth guidance was removed and the GPU host uses a persistent zero texture | `test_flow_service_is_used_once_per_frame`, `test_removed_depth_modes_are_rejected` |
| Preview / export parity | n/a (Magpie has one path) | n/a | Both call the same `ResidualBackend` for working color and residual | `test_preview_export_working_color_parity`, `test_preview_export_residual_parity` |

## residual_strength semantics — deliberate difference

`residual_strength` in this project multiplies the **full-resolution** residual after
Catmull-Rom reconstruction. Magpie's `ApplyResidualControls` multiplies in the
**working-resolution** domain before either upsample pass, and also applies
`ResidualSaturation` and `ResidualLightness`, which are not implemented here.

Multiplying before or after a linear filter gives the same exact mathematics, so the
difference is rounding and the two missing controls. This is user-visible behaviour and
was **not** changed while porting the hot path: the native backend reproduces the
Python reference bit for bit (`test_native_matches_python_reference_random_cases`),
which is *not* the same statement as matching Magpie's control semantics. Do not
describe this as "exact Magpie residual controls" unless saturation and lightness are
implemented and the multiply is moved to low resolution with its own tests.

## Backend contract and selection

`ResidualBackend` is the single seam: `prepare_working_color`, `reconstruct`, `close`,
`backend_info`, `diagnostics`. Preview, export and the benchmarks all get their backend
from `pipeline.create_residual_backend(requested, ...)`, which records
`residual_backend_requested`, `residual_backend_actual` and a fallback reason.
`requested=native` with an unloadable bridge raises instead of quietly running the
reference; `auto` may fall back, but reports `cpu_reference_python`.

`cpu_reference_native` owns its filter plans, its FP32 scratch and a fixed row pool
inside one handle: threads are created in `rr_create_plan` and joined in
`rr_destroy_plan`, and `diagnostics()` exposes them in two separate scopes —
`this_plan_create_count`, `this_plan_scratch_alloc_count` and
`this_plan_worker_create_count` describe this handle only, `process_*_total` are the
whole-library counts — alongside `downsample_call_count` and `reconstruct_call_count`, so
reuse is demonstrated at runtime rather than argued from the source (see
`perf_results/<sha>/native_residual_counters.json`). It does not
allocate a full-resolution FP32 residual buffer: the composite is fused into the
vertical pass, so the largest per-frame scratch is the `working_height x source_width`
intermediate.

The reference backend is the numerical authority: it is what the scalar oracle is
compared against and what the native path must match. It converts and allocates per
frame in Python and is not a performance path.

## Not equivalent / not proven

- `ResidualSaturation`, `ResidualLightness`, `ShadowStructureMultiplier` and
  `ReflectionGlowMultiplier` are absent; only the multiplier exists.
- FP16 vs FP32: Magpie's intermediate is a shared FP16 texture. Both standalone paths
  use FP32, which is why tolerances are "max 1 code value, mean < 0.05" and never
  "exact match".
- The final resize to the *output* resolution still uses `cv2.resize` with
  `scale_algorithm_to_cv2`, as does the branch that upscales DLSS output when the
  residual pyramid is off. Only the working-resolution stages above follow the
  reference kernels.
- Magpie has real depth guidance; this project now supplies zero depth only, so
  its depth downsampling behaviour is outside the standalone feature set.
- Hand-written SIMD is not used; kernels rely on compiler auto-vectorisation, and
  `rr_cpu_has_avx2` is informational only.
- One DLSS pass. The native host ABI has no per-pass session handle, so
  `ProductionExportCore` rejects `len(passes) != 1` rather than pretending otherwise.
- `gui.py` still contains a legacy offline PNG-rendering loop that computes its own
  guidance tuples. It is not the streaming export path and is not covered by the
  parity tests above.

## Test inventory

Counted by the runner, never typed by hand:

    python -m pytest -q tests/unit

The transcript of the run that backs this round's evidence — including the
hardware-gated NVOF cases and the real mid-flight cancellation E2E — is
`perf_results/<code-under-test-sha>/pytest_output.txt`, and the gate table derived from
it is `perf_results/<code-under-test-sha>/final_status.json`. A number written into
this document would go stale the moment a test is added; the transcript will not.
