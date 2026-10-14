# HDR10 multiscale foundation - stage B1

Date: 2026-10-03. Owner of review: Codex. Implementation: Qoder (this document).
Previous stage: `HDR_MULTISCALE_STAGE_A2.md` (typed fusion), before it
`HDR_MULTISCALE_STAGE_A1.md` (observation foundation). Plan:
`docs/plans/2026-10-02-hdr10-multiscale-fusion.md`.

B1 puts the A1/A2 native foundation onto the **production HDR10 backend**: a dual or
triple HDR job now decodes once, drives one independent Feature 18 session and one
colour bridge per layer, takes one signed linear-scRGB residual per layer, and meets
either in the typed CG data-consistency solver or in the unchanged Catmull baseline -
then encodes and remuxes exactly as a single-layer job does. It does **not** touch the
UI, `dlss_service.py`'s queue validation for HDR, any native source, or the SDR chain;
it runs no hardware export and makes **no** numerical-quality or performance claim.
`performance = NOT_CHARACTERIZED`.

What was proven, in one line: on fake typed planes and fake fenced resources, a
production-shaped dual job runs the per-frame order the borrow contract requires,
hands every layer's residual to `gpu_fusion_process_hdr`, imports the fused texture
through `gpu_hdr_set_composite` over a real producer fence, encodes and map-and-copies
the result, resets all layers on one discontinuity, unwinds partially-built layers in
build order, refuses a stale capability and an unconverged solve without falling back,
and leaves the `single` path calling exactly the five entries it called before.

---

## 1. Files changed by B1

| File | B1 change |
|---|---|
| `gpu_hdr_transfer_export.py` | `hdr_multi_layer_refusals()` (815-874); the multi half of `negotiate_hdr()` (1002-1016) and its `multi_scale`/`layers`/`reconstruction_mode`/`fusion_mode` negotiation fields (1107-1133); `HdrMultiScaleChain` (1604-2020) - `_require_multiscale_group`, `_open_stages`, `_open_fusion`, `_close_layers`, `process`, `_layer_record`, `layer_diagnostics`, `counters`; the run's aggregation of the layer blocks into the report (2218-2342); `_open_extra_layers()` (2466-2500) and the chain selection at 2524-2543. 2799 lines now. |
| `pipeline.py` | `export_gpu_hdr_transfer_stream()` asks `gtx.load_bridges(requires_flow=..., requires_multiscale=True)` **only** when the job names `dual`/`triple` (3294-3302), so a single-layer HDR job loads the identical bridge set it always loaded. The `probe_source(include_chapters=..., include_video_metadata=...)` lines in the same hunk are the pre-existing uncommitted remux fix, not B1. |
| `hdr_fusion_engine.py` | the B1-0 wrapper-boundary fixes: `layer_options()` and the `_require_open` / typed-plane / report-validation edges the production chain leans on. 660 lines (553 at A2 + 107). |
| `dlss_gpu_engine.py` | `Live.layer_capability()` (+24): the three entry-by-entry answers the gate needs (`shared_layer_sessions`, `format_aware_layer_open`, `rgba16f_sessions`, plus `reportable_session_id`) instead of an ABI number. |
| `DLSS5Tool_standalone.spec` | `hdr_fusion_engine` added to `HIDDENIMPORTS` beside the two existing HDR chain entries, with the reason stated in the spec. |
| `tests/unit/test_packaged_hdr_imports.py` | the third chain member in `HDR_CHAIN`, in the no-native-load-at-import subprocess, and in the `_library is None` assertions. |
| `tests/unit/test_hdr_transfer_export_branch.py` | 3495 lines, 195 collected tests. B1-1's negotiation section adds 12 test functions (463-594); B1-2/3/5's orchestration section adds 25 test functions / 27 collected tests (2860-3495). The file went 168 -> 195 collected. |

No native file, shader, header or DLL was modified. The two binaries under test are the
ones A2 hashed:

```
76a81702f7536cc93d3162e8d85340e956941855871f280a2bc49e32931f2007  native/gpu_multiscale_fusion_bridge.dll
6d0cb889d3fa6f4685ca6e2cbbb4845967eb7beaa18375d3110e1d47b356937f  native/gpu_hdr_scrgb_bridge.dll
```

Untouched, deliberately: every SDR path in `gpu_transfer_export.py` (its
`multi_scale_config` is *called*, never edited), `native/*`, `multiscale_fusion.py`,
the objective/CG maths, the Feature 18 / NVDEC / NVOF / NVENC wrappers, the paper-white
policy (one fixed value, read once, handed to every bridge), the metadata and remux
rules, `AGENTS.md`, `_selftest.txt`, `native/db_exports.bat`,
`native/nvenc_hdr_options_probe2.txt`, and all A1/A2 evidence and `bench_out/`
directories. Nothing was committed, pushed, stashed or cleaned.

## 2. What a dual job now runs, per frame

`single` is not re-expressed as a special case of the new chain: `HdrMultiScaleChain`
subclasses `HdrChain`, layer 0 *is* the single-layer chain (the decoder's P010, the
primary bridge's own downscale and proxy, the anchor session), and the encode tail is
inherited verbatim. The frame order, as the tests pin it
(`DUAL_FRAME`/`DUAL_FUSED_FRAME` at 2867-2869):

```
decode -> downscale -> proxy            (primary bridge: the one source measurement)
import_master -> downscale -> proxy     (each extra bridge, on the pristine FP16 master)
observe                                 (layer 0: its own signed working residual)
observe                                 (each extra layer, at its own resolved extent)
  data_consistency_experimental:
      gpu_fusion_process_hdr(master, [r_0, r_1, ...])   -> fenced texture
      primary.set_composite(fused)                       (real producer fence, value != 0)
  catmull_baseline:
      primary.reconstruct(layer_0_output, residual_strength)
encode                                  (once; PQ/P010 -> hevc_nvenc Main10)
```

Then the unchanged tail: intermediate -> `run_copy_pass` map-and-copy of the source's
audio/subtitle/chapters -> uncached read-back of the deliverable. A dual job with the
source's streams still reports `hdr_copy_pass=True` and
`hdr_deliverable_written_by="ffmpeg_map_and_copy_pass"` (test at 3119).

An extra layer's main colour is a device-to-device copy of the **pristine**
full-resolution linear scRGB master (`import_master`, provenance 2), never a resample of
the proxy, never an 8-bit round trip, never another decode. The tests assert the imported
plane is byte-for-byte the master descriptor - same device pointer, pitch 15360,
1920x1080 - for every extra layer (2910).

## 3. Lifetime, ownership and reset

Built before the first frame, in this order: one `Live` per extra layer through
`open_layer_ex` on the anchor's core with `color_format="rgba16f"` and that layer's own
resolved extent, plus `new_independent_layer()` motion history per layer when
`guidance_mode=1`; then, on frame 0, the primary chain's stages, then the multiscale
group re-check, then one colour bridge per layer (opened on *that session's* D3D12
device, the decoder's CUDA context, the job's one paper white), then the typed solver
on the anchor's device with the pipeline's own `TapPlans` per layer.

Teardown is reverse ownership, and it is asserted as a ledger
(`TEARDOWN`, test at 3224):

```
solver -> layer bridge -> layer session -> layer flow history -> ... ->
encoder -> primary colour bridge -> anchor session -> CUDA context -> decoder
```

The anchor session, the primary bridge, the context and the decoder are the parent's to
release in the parent's order, and the **root flow provider is never closed by the
chain** - only the layers it handed out (`flow.closed_layers == 0`, the layer's engine
`== 1`). Three partial-open doors are parametrized (bridge / solver / session) and each
hands back exactly what it built: a session whose open raised is not a handle to close,
so `Solver.opened == []` and `Solver.closed == 0` in all three, and the session door
never reaches the decoder or encoder. A cancel between frames closes the same list
(3289). A solve that raises ends the job with no `reconstruct` call at all (3302): a
data-consistency job is never answered with the baseline picture.

Reset is all-layers-at-once: one PTS gap sets `reset=True` on every layer's session,
every layer's NVOF history and every layer's recorded flag on the same frame - the
report's `hdr_layer_reset_flags` is the same 6-element vector for both layers (3139). A
decoder that yields a different frame extent mid-job is refused with "rebuild" in the
message before that frame is drawn (3192), because reusing another extent's history is
the failure this guards.

## 4. The gate, and what a job may name

The layer vector is validated by `gpu_transfer_export.multi_scale_config` - the same
helper, the same sealed contract, no HDR-side copy of the rules: `single|dual|triple`;
`layer_scales` exactly N distinct integer percentages 25..100; `layer_scales[0]` must
equal `working_scale`; `layer_weights` N finite positives; `fusion_mode`
`catmull_baseline|data_consistency_experimental`; `fusion_beta > 0`,
`0 < fusion_relative_tolerance < 1`, `fusion_max_iterations` 1..1000; no layer may
resolve beyond the source. A triple may name a 100% layer - the identity plan is
accepted (`TapPlans(1920,1080,1920,1080)`) and nothing is added for it (test at 473).

Then `hdr_multi_layer_refusals` asks four separately-versioned things, each on its own
evidence, and never from `GPU_HDR_ABI_VERSION`:

1. `hdr.probe_multiscale(colour bridge)` - are `gpu_hdr_import_master`,
   `gpu_hdr_observe`, `gpu_hdr_set_composite`, `gpu_hdr_get_multiscale_counters`
   exported (A1's optional group, by symbol name).
2. layer 0's resolved extent equals the extent the anchor session and primary bridge are
   opened at.
3. `live.layer_capability()` - can this host open an extra **RGBA16F** session on this
   core (`shared_layer_sessions` + `format_aware_layer_open` + `rgba16f_sessions`).
4. `new_independent_layer` / `close_independent_layer` on the flow provider, when
   `guidance_mode=1` - a layer sharing one motion history is a refusal, not a feature.
5. and only for `data_consistency_experimental`: `hfe.probe_hdr_fusion(...)` - are
   `gpu_fusion_hdr_abi_version`, `gpu_fusion_open_hdr`, `gpu_fusion_process_hdr`,
   `gpu_fusion_get_hdr_counters` there. A `catmull_baseline` dual needs no fusion read
   at all (test at 492), and a single job needs none of the four (test at 581).

`_require_multiscale_group()` re-asks question 1 from the loaded library inside the run:
a negotiation made on one answer and a frame driven on a handle that no longer gives it
refuses with "no HDR multiscale capability read", after one layer session was built and
before any layer bridge existed, and the session plus its history are handed back (3325).
Refusals are never softened into a fallback: a job that asked for dual and was refused
reads as a refused dual job.

## 5. Diagnostics and counters

The single-layer record is unchanged and every multi fact is audited on top of it (test
at 3372): `layers` becomes a list of per-layer records (layer, scale, `working_dims`,
weight, `session_id`, `master_provenance` - `decoded_pristine_master` for layer 0,
`imported_pristine_master` for the rest, `paper_white_scrgb`, bridge extents, its own
proxy report, its own planes, its own Feature 18 output, its own stage timings), and
only layer 0 carries a `source_report`, because an importing bridge has no decode pass
to report.

Job-level keys added: `hdr_layers_measured`, `hdr_reconstruction_mode`,
`hdr_fusion_mode_requested`, `hdr_layer_scales`, `hdr_layer_working_dims`,
`hdr_layer_weights`, `hdr_layer_session_ids`, `hdr_layer_sessions_distinct`,
`hdr_layer_paper_white_scrgb`, `hdr_layer_calls_measured`,
`hdr_layer_flow_calls_measured`, `hdr_layer_zero_motion_calls_measured`,
`hdr_layer_reset_flags`, `hdr_layer_masters` (with the `why` string),
`hdr_layer_transfers_measured` (one block per extra bridge), `hdr_fusion_counters_measured`,
`hdr_solver_frames_measured`, `hdr_solver_iterations_max_measured`,
`hdr_solver_initial_residual_norm_max_measured`,
`hdr_solver_final_residual_norm_max_measured`, `hdr_solver_relative_residual_max_measured`,
`hdr_solver_converged_all_frames`, `hdr_solver_unrepresentable_components_total_measured`,
`hdr_solver_threshold`, `hdr_solver_max_iterations`, and `hdr_multi_scale`. `hdr_layers` stays the negotiated
count - 1 for a single job - while every `hdr_*_measured` key above is written only when
a multi chain actually ran, so a single-layer report carries none of them rather than a
zero that would read as a vector of length one.

Traffic is attributed to the bridge that moved it, never merged into one figure the
primary bridge never saw. For a 3-frame dual run at scales (0.5, 0.25) on 1920x1080 -
all of this pinned by the test at 3074:

```
layer_1 (480x272):  master_import_count 3, master_import_device_bytes 3*1920*1080*8
                    observe_count 3, observe_copy_bytes = observe_residual_bytes
                    = 3*480*272*8, composite_import_count 0,
                    composited_device_write_bytes 0, encode_device_write_bytes 0,
                    master_provenance 2, pass_dispatches 9
primary:            process_count 15, composited_device_write_bytes 3*1920*1080*8,
                    master_provenance 1, pass_dispatches 15
scalar boundary:    pass_dispatches_measured {"primary": 15, "layer_1": 9},
                    total 24, range_bytes_from_completed_dispatch_count 24*256 = 6144
                    fusion_scalar_readback_bytes_measured 9216
                    hdr_scalar_report_readback_bytes_measured  None   (still a boundary)
                    hdr_diagnostic_image_readback_bytes_measured 15*96 (never folded in)
solver:             process_count 3, source_device_copy_bytes 3*full,
                    observation_device_copy_bytes 3*(960*544*8 + 480*272*8)
```

The dispatch arithmetic is causal, not asserted: an import is a full-resolution FP16
write that dispatches no shader, an observe copies in and writes out at the working
extent and dispatches one, and `pass_dispatches` counts dispatched passes only - so 15 =
5 dispatched stages x 3 frames on the primary (decode, downscale, proxy, observe,
encode) and 9 = 3 x 3 on the layer (downscale, proxy, observe). `256` is read off the
bridge's own `kCounterSlots * sizeof(uint32_t)`, so the scalar figure is stated as a
**range derived from a counted number**, reported beside the image counters and never
instead of them. `hdr_scalar_report_readback_bytes_measured` stays `None`.

Counted `unrepresentable_components` remain a success with a number attached (test at
3419): those samples fell back to the pristine master because half float cannot hold the
fused value, which is what the single-layer composite already does, and the frame is
still a frame.

## 6. The `single` path, and packaging

`test_a_single_layer_job_builds_no_session_no_extra_bridge_and_no_solver` (3350) runs a
6-frame single job with **every** multi seam patched in and asserts the ledger
`["decode","downscale","proxy","reconstruct","encode"] * 6 + ["close"]`, one bridge
constructed, `LayerSession.created == []`, `Solver.opened == []`, `flow.layers == []`,
`hdr_layers == 1`. `test_the_multiscale_libraries_are_loaded_only_for_a_job_that_named_layers`
(3442) asserts the route's two `load_bridges` calls as `[(requires_flow,
requires_multiscale)] == [(True, False), (True, True)]`. The chain imports
`hdr_fusion_engine` lazily inside `_open_fusion`, so a single job, a baseline job and an
import of `gpu_hdr_transfer_export` load no solver DLL - `test_packaged_hdr_imports.py`
now checks that for all three modules and that the spec names all three.

Known and unchanged: `gtx.load_bridges(requires_multiscale=True)` also loads
`gpu_low_residual_bridge.dll`, which is the SDR multiscale helper and is not used by the
HDR chain. It is the shared loader's existing behaviour, so a dual HDR job pays one
extra unused DLL load. Not worth an HDR-specific loader in this stage; noted for review.

## 7. Targeted tests actually run

Only the Python files this stage can affect. Commands, exit codes, summaries:

| # | Command | Exit | Summary |
|---|---|---|---|
| 1 | `python -m pytest tests/unit/test_hdr_transfer_export_branch.py -q` | 0 | `195 passed in 2.95s` |
| 2 | `python -m pytest tests/unit/test_hdr_transfer_export_branch.py tests/unit/test_hdr_fusion_engine.py tests/unit/test_hdr_gpu_engine.py tests/unit/test_packaged_hdr_imports.py tests/unit/test_export_pipeline.py tests/unit/test_gpu_transfer_export_backend.py tests/unit/test_transfer_counters.py tests/unit/test_transfer_modes.py tests/unit/test_hdr_proxy_slope_oracle.py tests/unit/test_hdr_production_gate_readback_reread.py -q` | 0 | `641 passed in 27.68s` |
| 3 | `python -m pytest tests/unit/test_hdr_ui_optin_settings.py -q` | 0 | `16 passed in 4.92s` |

Run 2 covers the SDR backend and the remux-side files that had to keep passing
(`test_gpu_transfer_export_backend.py`, `test_export_pipeline.py`,
`test_transfer_counters.py`, `test_transfer_modes.py` - the only two files in the tree
that mention the copy pass are the two HDR/SDR transfer files, both in run 2).

The intermediate state is worth recording because it found a real bug: the first
complete run of the new section was **7 failed, 188 passed**. One of the seven was a
production defect on a path that had never executed before -
`_guidance_and_inference(counted=(self.layer_flow_calls, self.layer_zero_motion_calls))`
followed by `counted[0] += 1`, i.e. `TypeError: 'tuple' object does not support item
assignment` on the first dual frame. It is fixed by one per-layer tally dict
(`layer_motion_calls[number]`), bumped by the call that decides flow-vs-zero-motion
rather than restating that decision at the call site. The other six were this file's own
wrong expectations (two bridge closes instead of one, the float32-marshalled solver
tolerance, and which object records a layer close) and were corrected to the behaviour
the production code actually implements, not the reverse.

Not re-run, per the brief: the A2 synthetic gate, the A-B CI gate, the roundtrip gate,
the SDR byte-identity check and Codex's offline consistency result
(`0.0045380174993597` -> `0.0004620317312618251`) are prior evidence and unchanged by B1
- B1 edited no native code and no DLL was rebuilt. No hardware gate was run. No new
`bench_out` directory was created and none was overwritten.

## 8. What B1 does not establish

- **No pixel was ever moved by this stage.** Every plane, texture, fence, session and
  counter in the new tests is a fake that borrows the product's own validators
  (`hdr._require_scrgb16f_plane`, `hdr.GpuHdrBridge._producer_texture`,
  `dge.Live._check_color/_check_flow`, `hfe.layer_options`). The evidence is
  call order, argument vectors, extents, formats, fence presence, lifetimes and
  attribution - never image quality.
- **Triple is code-complete, not hardware-verified.** A 100% third layer, its identity
  plan, its session, its bridge, its observation and its residual are all driven and
  asserted in the fake harness (3039); three layers have never run on a card.
- No production HDR export of any layer count was run; nothing here shows a file that
  plays.
- `performance = NOT_CHARACTERIZED` for every new stage
  (`hdr_layers_colour`, `hdr_fusion`, `hdr_composite_import`, `layer_nvof_flow_ms`,
  `layer_dlss_inference_ms` are measured timings on the fakes, not characterised costs).
- Numerical equivalence of the production chain to `multiscale_fusion.py` is A2's result,
  not a B1 claim: B1 proves the chain *calls* the solver with the master and one residual
  per layer, not that the picture is right.

## 9. Remaining work

**B2 - open the UI (blocked in exactly three places):**

1. `dlss_service.py:913-917` refuses any `experimental_hdr_scrgb` job naming
   `dual`/`triple` at the queue boundary ("HDR 色彩桥按一个工作分辨率开一个合成，多层与
   多趟尚未实现"). This is now false and must go, with the refusal kept for the real
   limits (a build without the A1 group, a host that cannot open an RGBA16F layer, a
   flow provider without independent layers, a fusion-less data-consistency job) -
   which are the gate's own answers, not a UI guess.
2. `frontend/src/services/hdrExport.ts:183-186` mirrors refusal 1 client-side, and
   `:360` / `:387` pin `reconstruction_mode: 'single'` into the HDR preview request and
   its summary line. Both need the layer vector surfaced, and the preview has to show
   what the chain will actually run.
3. The tests that pin those: `tests/unit/test_hdr_ui_optin_settings.py:165-178`,
   `tests/node/hdr_export_ui.test.ts`, `tests/e2e/17_hdr_ui_optin_mock.spec.ts`. They
   assert the restriction today, so they must be rewritten to assert the capability-based
   acceptance rather than deleted.

**C - the plan's final short gates, on the completed tree** (`docs/plans/`
`2026-10-02-hdr10-multiscale-fusion.md`, section C): the full unit suite once; one
4-6-frame **single** HDR10 production E2E through the real audio/subtitle/chapter/metadata
remux; a **dual** 50/75 fusion smoke with distinct sessions and a converged solver; only
after that a **triple** 40/67/85 2-3-frame smoke; the compact synthetic fusion evidence;
and the SDR multiscale/remux regressions plus the frontend tests. Reuse
`bench_out/hdr_remux_export/hdr_remux_src.mkv` and a fresh ignored evidence directory,
recording source/module/DLL hashes; nothing successful is re-run for a checker-only
repair. Also still open and out of B1's scope: `multi_pass > 1` for HDR, the 25..100
percent scale sweep at HDR extents on real buffers, and one unused-DLL load
(`gpu_low_residual_bridge`) in a dual HDR job's bridge set.
