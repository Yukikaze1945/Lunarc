# HDR10 multiscale foundation - stage A2

Date: 2026-10-03. Owner of review: Codex. Implementation: Qoder (this document).
Previous stage: `HDR_MULTISCALE_STAGE_A1.md`. Plan:
`docs/plans/2026-10-02-hdr10-multiscale-fusion.md`.

A2 is the typed HDR path through the multiscale fusion solver, a compact typed
Python wrapper, the two A1 corrections Codex asked for, and one small synthetic GPU
gate that runs the freshly compiled entries. It does **not** wire fusion into the HDR
production export or the UI, does not touch Feature 18 / NVDEC / NVOF / NVENC maths,
metadata, audio, subtitle or remux code, and makes **no** numerical-quality or
performance claim. `performance = NOT_CHARACTERIZED` in every artefact below.

What was proven, in one line: a two-layer scRGB residual solve runs on the GPU
against a pristine FP16 master through `gpu_fusion_process_hdr`, hands back a
full-resolution `R16G16B16A16_FLOAT` texture in `COPY_SOURCE` behind the solver's own
producer fence, is consumed by a real `gpu_hdr_set_composite` bit for bit, agrees with
the CPU reference inside the binary16 grid, and fails outright where it must.

---

## 1. Files actually changed

| File | Change |
|---|---|
| `native/gpu_multiscale_fusion_bridge.h` | purely additive (+167/-0): the `--- Typed HDR (scRGB) path` section - `GPU_FUSION_HDR_ABI_VERSION`, `GpuFusionHdrOptions`, `GpuFusionHdrObservation`, `GpuFusionHdrReport` (72 B), `GpuFusionHdrCounters` (48 B), `gpu_fusion_hdr_abi_version`, `gpu_fusion_open_hdr`, `gpu_fusion_process_hdr`, `gpu_fusion_get_hdr_counters`, the stage-code list, the sample-width / units / lifetime / completion contracts. Nothing above the new section was edited. |
| `native/gpu_multiscale_fusion_bridge.cpp` | (+542/-103) `bridge->hdr`, `pitch16_for`, per-layer `observed` FP32 texture, `is_final_format` + `make_texture` initial state, `stores_final` covering `kCompositeHdr`, `unpack_half`, `reduce_partial`, the typed open/process/counters entries. **No** SDR entry, struct, layout, default or kernel numeric was removed: the deleted lines are shared-internal helpers (`make_texture`'s format test, the open body factored for both contracts). |
| `native/gpu_multiscale_fusion_shader.h` | (+83/-0) the typed section only: `unpack_half2`, `unrepresentable`, `load_half4`, `UnpackHalf`, `CompositeHdr` with the counted fallback and the 1e30 poison. Every kernel above it is byte-for-byte the SDR code. |
| `hdr_fusion_engine.py` | new, 553 lines: ctypes mirror of the typed group + `HdrFusionBridge`, `layer_options`, `probe_hdr_fusion`, `load`, `bind_shared`, `last_error_text`. Reuses `gpu_transfer_export._FusionLayer/_FusionOptions/_fusion_axis` and `hdr_gpu_engine._require_scrgb16f_plane` rather than defining a second plan set. |
| `tests/unit/test_hdr_fusion_engine.py` | new, 54 tests: field-for-field struct mirrors, ABI refusal, typed-plane refusal, one-residual-per-layer, stage-code messages, counters, close/partial-open paths, and the plan-reuse identity (the marshalled tap pointers are the pipeline's own). |
| `native/gpu_multiscale_fusion_a2_test.cpp` + `native/build_gpu_multiscale_fusion_a2_test.bat` | new, 1609 lines: one probe that loads *either* build of either bridge by path and runs the typed solve, the handoff, the A1 ordering, the counted fallback, the three failure gates and the legacy SDR control on one 64x48 fixture. |
| `tests/unit/hdr_multiscale_fusion_a2_gate.py` | new, 711 lines: builds the fixture from the pipeline's own plans, runs the probe, compares the dumps against a CPU reference, and writes `verdict.json`. |
| `native/gpu_hdr_scrgb_bridge.{h,cpp}` + `hdr_gpu_engine.py` | the two A1 corrections only (section 5). |
| both DLLs | rebuilt from the above (section 6). |

Untouched, deliberately: `multiscale_fusion.py` (the CPU reference), the objective and
CG maths, `GpuFusionOptions` / `GpuFusionObservation` / `GpuFusionSolveReport` /
`GpuFusionTransferCounters`, `gpu_fusion_open` / `gpu_fusion_process` behaviour,
`nvof_d3d12_interop` (used as-is: `cuda_d3d12_color_create2(..., pixel_bytes=8)`,
`row_capabilities`, `get_layout`), `gpu_transfer_export.py`'s SDR path,
`gpu_hdr_transfer_export.py`, the UI, and `AGENTS.md`. Nothing was committed, pushed,
stashed or cleaned; the pending related changes are left as they were.

## 2. New native entries

```c
#define GPU_FUSION_HDR_ABI_VERSION 1u
GPU_FUSION_API uint32_t gpu_fusion_hdr_abi_version(void);
GPU_FUSION_API int32_t gpu_fusion_open_hdr(void* d3d_device,
    const GpuFusionOptions* options, const GpuFusionHdrOptions* hdr_options,
    GpuFusionBridge** out);
GPU_FUSION_API int32_t gpu_fusion_process_hdr(GpuFusionBridge* bridge,
    void* cuda_context, uint64_t full_scrgb_ptr, size_t full_scrgb_pitch,
    const GpuFusionHdrObservation* observations, uint32_t observation_count,
    void** out_texture, void** out_fence, uint64_t* out_fence_value,
    GpuFusionHdrReport* report);
GPU_FUSION_API int32_t gpu_fusion_get_hdr_counters(
    GpuFusionBridge* bridge, GpuFusionHdrCounters* counters);
```

`GPU_FUSION_ABI_VERSION` is still `1` and `GpuFusionSolveReport` /
`GpuFusionTransferCounters` keep the layouts SDR callers read: the group is probed by
symbol name plus `gpu_fusion_hdr_abi_version()`, never inferred from the SDR ABI
number. An archived pre-A2 build answers `fusion_hdr_abi = 0` / `typed group absent`
and still runs the whole RGBA8 chain - that is a capability answer, not a load
failure, and the gate asserts it as one.

Stage codes of `gpu_fusion_process_hdr` are stable and are what the wrapper prints:
1 arguments or contract, 2 an observation that does not describe its layer's plane,
3 a transport this build cannot make, 4 a CUDA context that changed, 5 a source copy,
6 an observation copy, 7 a producer wait, 8..12 the solver's own stages exactly as
`gpu_fusion_process` numbers them (dispatch, initial norm, CG step, lost positive
definiteness, iteration budget), 13 the store dispatch, 14 an unreadable counted
fallback, 15 a store that could only hold a value the half grid cannot, 16 a failed
flush. Any non-zero return hands back **no** texture, and there is no baseline
fallback and no retry-with-something-else path. *Corrected in B1 against the compiled
source:* the mark-failed half of that sentence is true from stage 5 onwards - every
stage that writes device state - and not for stages 1 to 4, which refuse the call
before anything is written and leave the bridge usable. A later frame on a
non-poisoned bridge is therefore answered rather than refused at stage 1, which is a
fact about the handle and not a licence to continue a job: either way this frame has
no composite. `hdr_fusion_engine.POISONING_STAGES` states the accurate set and the
wrapper's message says which state a returned code left the bridge in.

## 3. Format, units, synchronisation, lifetime

* **Source**: the pristine full-resolution linear scRGB master as an
  `rgba16f_cuda` plane, 8 bytes a sample, imported at `width * 8` with a 256-byte
  row pitch. Never `width * 4`; `pixel_bytes` is checked against what the transport
  can actually move and 4 is refused on an HDR bridge.
* **Observations**: one signed working residual per layer, exactly what A1's
  `gpu_hdr_observe` handed back - a borrowed CUDA `rgba16f_cuda` plane at that
  layer's working extent, RGB in linear scRGB, alpha 0. Not an R32 plane, not a
  proxy difference, and **not** reinterpreted: each half plane is *widened* into the
  solver's own FP32 texture by one `UnpackHalf` dispatch per layer, so every kernel
  after that is the kernel the SDR path already runs on the same float32 plans.
  Each layer belongs to its own bridge, because the observing bridge rewrites that
  plane on its next `observe()`.
* **CUDA-plane completion contract** (in the header, not implied): `gpu_hdr_observe`
  waits its own pass fence before returning the plane, so the samples are
  device-visible when the descriptor is filled; the solver's copy completes inside
  the call, so the pointer only has to survive the call. A producer that has not
  retired its own work must be made complete by its own caller first - this entry
  takes no fence because a CUDA device pointer has no fence to name, and it will not
  pretend to wait one.
* **Units**: scRGB throughout, 1.0 == 80 cd/m2. A component may be negative (out of
  gamut) or above 1.0 (headroom) and neither is scaled, normalised, clamped or
  rounded. The SDR observation is in 8-bit pixel units and the HDR one is not, which
  is why the entries are separate rather than one entry with a factor in it.
* **CG work vectors** stay FP32 with the same float32 taps, centres and clamp edges;
  only the transport and the store are FP16.
* **Output**: the bridge's own full-extent `DXGI_FORMAT_R16G16B16A16_FLOAT` texture,
  left in `D3D12_RESOURCE_STATE_COPY_SOURCE`, plus `out_fence` / `out_fence_value`
  naming the fence the solver's queue actually signalled. `report.output_format`,
  `output_width`, `output_height` are read off the texture's `GetDesc()` rather than
  asserted; `output_state` is the state `record()`'s two transitions leave it in, and
  `stores_final` now covers `kCompositeHdr` so an R16 final is transitioned as the
  store it is instead of as a work texture. The texture belongs to the bridge and its
  contents are overwritten by the next call.
* **Old entries stay `R8G8B8A8_UNORM`**, and an HDR bridge refuses
  `gpu_fusion_process` with a message naming the entry to use instead. There is no
  fallback between the two contracts in either direction.
* **Counted fallback preserved, not redefined**: `final = source + strength *
  residual`, unclamped; a component outside +-65504 or non-finite falls back to the
  pristine master sample and adds 3 to `unrepresentable_components`, which the kernel
  reduces per group into the same scalar buffer the CG already uses and the host sums
  as a scalar. No picture is ever downloaded by a production entry. If even the
  fallback sample is outside the grid the group writes a 1e30 poison, the host reads
  it as > 1e20 and the call fails at stage 15: a GPU half never stores an Inf and
  then reports success.

## 4. What the solver did **not** change

`min_r sum w_i ‖D_i r - r_i‖² + beta ‖r - r0‖²`, `H(v) = sum w_i D_iᵀD_i v + beta v`,
`b = sum w_i D_iᵀr_i + beta r0`, `r0` = the layer-0 Catmull-Rom reconstruction, the
matrix-free CG, the tap tables, the edge clamp, the descriptor layout, the
`FrameConstants` shape and every SDR kernel are the code that was already there. A2
added a colour contract and a store, not a second solver.

## 5. The two A1 corrections

1. **Proxy provenance follows the working plane it actually read.**
   `gpu_hdr_downscale` now sets `working_epoch = master_epoch` and bumps
   `working_generation`; `gpu_hdr_proxy` records `proxy_generation =
   working_generation`; `gpu_hdr_observe` refuses when `working_epoch !=
   master_epoch` and then when `proxy_generation != working_generation`, naming
   `gpu_hdr_proxy` as the entry to run again. So `import -> proxy -> downscale ->
   observe` is refused on live state instead of returning a signed residual of a
   superseded plane. No new refusal and no maths change reached the single-layer
   entries: `gpu_hdr_reconstruct` and the old chain behave as before.
   The proof is the real-state sequence in `native/gpu_multiscale_fusion_a2_test.cpp`
   (`gate_ordering`), not a regex over one assignment line: it drives import,
   downscale, proxy, a second import that XORes one sample of the master, proxy,
   downscale, and asserts the *pair* of refusals - stale proxy -> the message names
   `gpu_hdr_proxy`; after a fresh proxy -> the message is the missing Feature 18
   output, i.e. the call got past provenance. The archived HEAD build of the HDR
   bridge carries no multiscale group at all (`import_master` / `observe` /
   `pass_dispatches` are absent from it), so a pre/post comparison of this guard is
   impossible by construction; the live-state pair is the evidence.
2. **`multiscale.pass_dispatches` is centralised in `run_pass`.** The counter
   increments in exactly one place now (the duplicate in `gpu_hdr_observe` is gone),
   so any future pass is counted without editing it. The 72-byte
   `GpuHdrMultiscaleCounters` layout is preserved, the ×256 relationship
   (`pass_dispatches * 256` = the counter bytes that crossed the bus, `kCounterSlots
   * sizeof(uint32_t)`) is documented in the header and at the increment, every
   counter readback is still reported by its own field, and nothing here claims total
   host traffic is zero. Measured: the ordering sequence dispatches 7 passes (2
   downscale + proxy + 2 downscale + proxy + proxy) and reports
   `pass_dispatches = 7`, `scalar_counter_readback_bytes = 1792`.

## 6. Build and targeted tests

Commands, both exit 0, logs and hashes in
`bench_out/hdr_a2_evidence/build/build_record.txt`:

```
cd native && cmd //c build_gpu_multiscale_fusion_bridge.bat   -> exit 0
cd native && cmd //c build_gpu_hdr_scrgb_bridge.bat           -> exit 0
cd native && cmd //c build_gpu_multiscale_fusion_a2_test.bat  -> exit 0   (the probe)
```

| Artefact | bytes | sha256 |
|---|---|---|
| `native/gpu_multiscale_fusion_bridge.dll` | 185 856 | `76a81702f7536cc93d3162e8d85340e956941855871f280a2bc49e32931f2007` |
| `native/gpu_hdr_scrgb_bridge.dll` | 251 392 | `6d0cb889d3fa6f4685ca6e2cbbb4845967eb7beaa18375d3110e1d47b356937f` |
| `build/hdr_a2/gpu_multiscale_fusion_a2_test.exe` | 305 152 | `52a1a1879f9adaace2cee9c1dd81ec7c37960dc6ae0ded12115f1cc3f14327a0` |

Capability-ABI answers recorded by the gate itself: `fusion_abi = 1`,
`fusion_hdr_abi = 1`, `hdr_abi = 1`, `hdr_multiscale_abi = 1`; the archived fusion DLL
(`4e19c8119a552bfdedb78958599e00da257ce1c76854ac53651ff72a8a040183`, 173 568 B)
answers `fusion_abi = 1`, `fusion_hdr_abi = 0`, typed group `absent`.

No full roundtrip, no Feature 18 A/B gate and no benchmark was rerun - they are
outside A2 and the instruction said not to repeat them.

Targeted unit files, all green, no skip, no xfail, no threshold touched
(`bench_out/hdr_a2_evidence/build/targeted_unit_tests.log`, exit 0):

```
tests/unit/test_hdr_gpu_engine.py                  \
tests/unit/test_hdr_fusion_engine.py               \
tests/unit/test_multiscale_fusion_reference.py     \
tests/unit/test_gpu_transfer_export_backend.py     \
tests/unit/test_hdr_transfer_export_branch.py      \
tests/unit/test_packaged_hdr_imports.py            \
tests/unit/test_gpu_host_motion_wrapper.py         ->  533 passed
```

## 7. The synthetic GPU gate, and its raw numbers

`python tests/unit/hdr_multiscale_fusion_a2_gate.py` - verdict **PASS**, evidence in
`bench_out/hdr_a2_evidence/a2-20261003-015253/` (`verdict.json`, the fixture,
`typed/{summary.txt,notes.txt,stdout.txt,fused.bin,composited.bin,fallback.bin,sdr_composite.bin}`,
`sdr_new/`, `sdr_archive/`). A first passing run is kept beside it at
`a2-20261003-014852`; nothing earlier was overwritten.

Everything below is one 64x48 frame on the real device, through the freshly compiled
entries. CPU-only: the reference and all comparisons. Every download in the gate is
harness-side correctness instrumentation on synthetic data; the production entries
download no picture.

**Fixture.** full 64x48; layers 32x24 (w 1.0) and 16x12 (w 0.5); `beta = 0.05`;
`residual_strength = 1.0`; `relative_tolerance = 1e-4`; `max_iterations = 64`;
declared support ceiling 12.5 scRGB; paper white resolved to 8.8704 scRGB
(1.409 relative proxy actionable band). The master is three ramps carrying headroom
above 1.0 and out-of-gamut negatives, peak |component| 2.3848 as stored; the
observations are `D_i(r_true)`; the baseline `r0` is the layer-0 Catmull upscale
(max abs 0.2486 against `r_true`). The tap tables are the pipeline's own Lanczos-2
and Catmull-Rom plans padded to the bridge's tap layout - checked array-for-array
against `pipeline._build_*_axis_plan`, and against `gpu_transfer_export.TapPlans`
pointers, so there is no second plan set - and they carry duplicated clamp-to-edge
taps (144 / 192 per axis, 168 in the baseline vertical) that the transpose has to sum
together rather than drop.

**Thresholds, declared before the run from the format budgets** (in the driver's
docstring and echoed into `verdict.json.thresholds`): a stored component gets one
binary16 ulp at the reference value + 1e-3 (the 1e-3 is the float32 CG, whose
residual is allowed 1e-4 of an initial norm on a field of magnitude <= 0.4 at
strength 1.0); the correction read out of the store gets one binary16 ulp of the
*stored* sample + 2e-3; the adjoint pairing gets 1e-3 relative against float64.

**Typed solve (device).** `gpu_fusion_open_hdr` 0, upload of the master and both
residual planes 0, `gpu_fusion_process_hdr` 0, `gpu_fusion_get_hdr_counters` 0.
iterations 7; initial residual norm 0.030660467222332954; final 2.2512526811624411e-06;
relative 7.34253e-05; converged 1; `unrepresentable_components` 0.
Store as the texture reports it: `output_format` 10 (`DXGI_R16G16B16A16_FLOAT`),
64x48, `output_state` 2048 (`COPY_SOURCE`), fence and value both non-null.
This call's traffic: `source_device_copy_bytes` 24576 (64*48*8),
`observation_device_copy_bytes` 7680 (32*24*8 + 16*12*8),
`scalar_readback_bytes` 3072; lifetime counters the same for one frame
(`process_count` 1).
The harvested picture: no Inf, no NaN; 2464 components above 1.0 and 2307 below 0.0
survived the solve (preserved, not clipped - and not called lossless, it is a half
grid). Harness-side readback, metered as such: 24576 bytes at row pitch 512.

**Same problem, CPU reference.** iterations 7, relative residual 7.34266e-05 against
the device's 7.34253e-05. Device correction vs the reference solution: max abs
0.0019246935844421387 against a budget of one ulp of the stored sample (0.001953125
at the brightest) plus 2e-3 - worst ratio 0.4869. Device picture vs the reference
picture `master + strength * r`: max abs 0.001924753189086914 against
ulp + 1e-3 - worst ratio 0.6518. FP16 quantisation range at the peak sample: one ulp
is 0.001953125.

**Objective and data term (both actually dropping).** Catmull baseline 0.0043665172 ->
device 0.0011379809; data term sum w‖D r - r_i‖² 0.0043665172 -> 0.0004218930. The CPU
reference at its own solution: objective 0.0008550574, data term 0.0001785870 - the
gap between the device's 0.0011380 and the reference's 0.0008551 is the store's
rounding, which is why the correction budget moves with the sample.

**Adjoint pairing** (`<Dv,y>` against `<v,Dᵗy>`, float64 accumulation): layer 0
3.4362038836676803 against 3.4362013213894844, relative error 7.4567e-07; layer 1
-1.396812308393389 against -1.3968122767688353, relative 2.2641e-08.

**Handoff to a real HDR bridge.** `gpu_hdr_set_composite` accepted the fused texture
with the solver's fence and value (code 0), and the bridge's composited plane is the
fused samples **bit for bit** at pitch 512. The wait was performed by a second queue
of the same device - the harness's own - which is the claim that the fence is usable,
not just present.

**Counted out-of-range fallback.** Same fixture, same solve, `residual_strength`
raised to 1e9 so `source + strength * residual` leaves the grid: the call still
returns 0, `unrepresentable_components = 9216` = 3072 pixels x 3 components, all 3072
pixels hold the pristine master sample bit for bit, and the store holds no non-finite
component and no component outside +-65504. Iterations 7 (the store is after the
solve, so the count is the store's own).

**Failure gates.** `max_iterations = 1` with `relative_tolerance = 1e-12`: stage 12,
no texture handed back, and the next frame on that bridge is refused at stage 1.
A master with one component set to a half NaN: stage 15 with the message naming the
source plane, no texture, next frame refused. Neither case produced a picture, and
neither produced a silent baseline in place of one. A budget of 4 or 8 would not have
proven convergence either way - these gates are about refusal, not about a guarantee.

**Legacy SDR control, same fixture, both builds.** New build and archived pre-A2
build, through `gpu_fusion_open` / `gpu_fusion_process` on an RGBA8 master and two
`R32G32B32A32_FLOAT` signed residuals: code 0, iterations 7, initial norm
7.8184189796447754, final 0.00057409325381740928, relative 7.3428302130196244e-05,
converged 1, `sdr_output_format` 28, 64x48 at row pitch 256,
`output_readback_bytes` 0, `scalar_readback_bytes` 2880,
`full_device_copy_bytes` 12288, `process_count` 1 - and the two harvested composites
are byte-identical, with every report key equal. The SDR entry's observation contract
is unchanged and still refuses a picture where it wants a signed FP32 residual
(that refusal is what the fixture had to satisfy, not a new rule).

## 8. Two D3D12 findings from the validator, both kept as evidence

The probe has an opt-in `--debug-layer` (no gate turns it on; `debug_layer = off` in
every recorded run) because a failing `Close()` reports an HRESULT and not a reason.
Two things it said are worth carrying into B:

* **A texture cannot be copied into an `UPLOAD` heap.** `0x80070057` at
  `Close()`, and the layer's sentence is "the destination resource cannot be on a
  D3D12_HEAP_TYPE_UPLOAD heap". The harness's readback buffer moved to
  `D3D12_HEAP_TYPE_READBACK` in `COPY_DEST`. Any host-side harvest in B has to do the
  same; the bridge's own mapped planes are upload because the GPU writes them with
  *buffer*-to-texture copies, which is the other direction.
* **`CreateCommittedResource: Ignoring InitialState UNORDERED_ACCESS` for buffers** -
  pre-existing in the shared `make_buffer` path (present in HEAD, SDR included),
  harmless because buffers are effectively `COMMON`, and left alone rather than
  touched outside A2's scope.

One measurement the fixture records deliberately: the correction is read back as
`composite - master`, both of which are binary16. Where the master is bright that
subtraction carries the master's own rounding - at 9.4 scRGB one ulp is 7.8e-3, four
percent of the 0.25 correction, and the objective computed from the readback then
measures the store rather than the solve. The fixture's master peak is capped at 2.2
so the drop the solver produced is visible through the picture, and the correction
budget is per component instead of flat. The production path never makes that
subtraction: the encode consumes the composite directly.

## 9. A2 -> B hook example

This is the shape B wires; it is not wired anywhere yet. The names are the ones the
two wrappers actually publish.

```python
import gpu_transfer_export as gtx      # the pipeline's own tap plans
import hdr_gpu_engine as hdr           # the A1 bridge: master, observe, set_composite
import hdr_fusion_engine as fusion      # A2: the typed solver

layers = [{"plan": gtx.TapPlans(w, h, w2, h2), "dims": (w2, h2), "weight": 1.0},
          {"plan": gtx.TapPlans(w, h, w4, h4), "dims": (w4, h4), "weight": 0.5}]
with hdr.GpuHdrBridge(...) as master_bridge, \
     fusion.HdrFusionBridge(
         d3d12_device=device, full_width=w, full_height=h, layers=layers,
         beta=0.05, residual_strength=1.0, relative_tolerance=1e-4,
         max_iterations=64) as solver:
    master = master_bridge.decode(...)                    # rgba16f_cuda plane
    observers = [...]                                     # one bridge per layer
    for b in observers:
        b.import_master(master)                           # each layer owns its bridge
    fused = solver.process(cuda_context=context, master=master,
                           observations=[b.observe(dlss_output, fence, value)
                                         for b in observers])
    master_bridge.set_composite(fused["d3d12_resource"],
                                fence=fused["d3d12_fence"],
                                fence_value=fused["fence_value"])
    # ... and the existing PQ / P010 encode takes the bridge's output unchanged.
```

`layer_options` (which `HdrFusionBridge.__init__` calls for you) takes
`{"plan": TapPlans, "dims": (w, h), "weight": float}` per layer and marshals the
plans through the SDR chain's own `_fusion_axis`, so layer `i`'s `D_i` is
`plan.working_color` and `r0` is layer 0's `plan.upscale` - the same arrays the CPU
reference walks, and the caller must keep them alive for the whole open session
because the views carry raw pointers into them. `process()` returns
`d3d12_resource` / `d3d12_fence` / `fence_value` in the shape
`GpuHdrBridge.set_composite` takes, plus the report dict.
`unrepresentable_components > 0` is a success with a number attached, and the wrapper
says so rather than quietly re-solving.

## 10. Packaging dependency to carry into B

Both DLLs are already shipped: `bundle_paths.NATIVE_BINARIES` names
`gpu_multiscale_fusion_bridge.dll` and `gpu_hdr_scrgb_bridge.dll`, so no new payload
entry is needed. The new **module** is not reachable yet: nothing in
`pipeline.py` / `dlss_service.py` imports `hdr_fusion_engine`, so
`DLSS5Tool_standalone.spec`'s `HIDDENIMPORTS` (today
`gpu_hdr_transfer_export`, `hdr_gpu_engine`) must gain `hdr_fusion_engine` when B
calls it through a lazy `__import__`, exactly like the two names already there, and
`tests/unit/test_packaged_hdr_imports.py`'s `HDR_CHAIN` / `LITERAL_IMPORT_SITES`
contract has to be extended in the same change. If B uses the CPU reference for a
fallback check it also needs `multiscale_fusion`. `hdr_fusion_engine` imports
`bundle_paths`, `gpu_transfer_export` and `hdr_gpu_engine`, and opens no native
library at import time - the same lazy-loader property that test exists to protect.

## 11. Not proven here (the limits, plainly)

* **Performance: NOT_CHARACTERIZED.** No timing, no throughput, no frame-rate and no
  bandwidth comparison was measured or claimed. `scalar_readback_bytes` and the
  device-copy counters are byte *counts*, not a claim that host traffic is zero.
* **No image-quality claim.** The synthetic fixture has one analytic correction and
  a deterministic ramp master. Nothing here says a real frame looks better.
* The device's transpose is verified **transitively**: the device solution matches a
  CPU reference built from the explicit transpose, and the tables are proven to be
  the pipeline's own padded arrays. There is still no direct adjoint dispatch on the
  GPU against a CPU `Dᵗ` for the *HDR* path specifically.
* Three layers, one CUDA context, one device, one frame per bridge. `layer_count` up
  to 3 is what the solver's arrays allow and 2 was exercised; 3 is untested on
  hardware. Backpressure, real DLSS Feature 18 outputs, real decode timing and the
  encode side are B and C.
* `residual_strength` beyond the grid was probed at 1e9 to force the counted
  fallback; that is a store test, not a recommendation.
* Convergence is a *reported* property, never a guarantee: the fixture converged in
  7 iterations at reltol 1e-4 and the budget gates prove refusal, not success.
* The counted fallback's per-pixel granularity (3 components a pixel, whole-pixel
  fallback) is inherited from the single-layer HDR composite and was preserved, not
  re-validated as an aesthetic choice.
* Nothing was measured about two bridges sharing one CUDA context concurrently, and
  the header's completion contract is a rule for the caller, not a guard the solver
  can enforce.
