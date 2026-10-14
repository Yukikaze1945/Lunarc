# Lunarc comprehensive optimization acceptance — 2026-10-09

## Scope and baseline

Baseline: `ed1ffa480c809c855483add1c6a66af127cb4db7` on main. This round keeps
independent Feature 18 sessions, the existing CG objective/true adjoint, HDR
P010/PQ/scRGB transport and shared FFmpeg container mapping. Implementation and
review used Luna subagents; Qoder was not used. Unrelated user files are retained.
The original product identity is **Lunarc**: three light arcs representing
independent reconstruction layers joining a single image, with an SVG mark and
PNG/ICO application assets. It does not use another video application's branding.

## Completed stages and source map

| Stage | Actual change | Main files |
| --- | --- | --- |
| 1 — remux/lifecycle | GPU remux derives the exact encoded packet span, omits `-shortest`, bounds nonvideo packet starts with `-t`; legacy helper defaults remain compatible. Owned intermediates are removed on failure/cancellation as well as success. | `gpu_transfer_export.py`, `gpu_hdr_transfer_export.py`, shared regression tests |
| 2 — post-fusion controls | Independent GPU module, strict neutral bypass; total/lightness/chroma/low/high gains, hue/dark/highlight protection, soft compression, feathered rectangular/elliptical source-preserving ROI, truthful diagnostic views. | `post_control.py`, `native/gpu_post_control_*`, `dlss_service.py`, `pipeline.py` |
| 3 — scenes/temporal | Sampled GPU structural scene detector; continuous-PTS cut resets every independent layer. Default-off static low-frequency history rejects changed details, signs, HUD and exposure; current high frequencies remain responsive. | `temporal_control.py`, `native/gpu_temporal_*`, both GPU export chains |
| 4 — reduction/resources | Multilevel GPU partial-sum reduction replaces CPU array reduction with a 4-byte dot result. CG, adjoint and convergence rules remain; resource reuse remains bound to identity and reset. | `native/gpu_multiscale_fusion_bridge.cpp`, shader, DLL, reduction probe |
| 5 — flow contracts | Actual NVOF direction/units/grid/history/HDR guidance documented and exposed. FidelityFX has no usable compiled adapter here and is explicitly unavailable. | `flow_contracts.py`, `server.py`, diagnostics UI |
| 6 — core acceptance | Full unit/Node/build plus real SDR/HDR single/dual/triple exports, signed HDR/native probes, media payload/SEI checks and baseline comparisons. | `tests/hardware/optimization_export_gate.py`, focused unit tests, native smoke/probe sources |
| 7 — original UI | Seven original pages, layered workstation, searchable controls, real backend save/reset, preflight, persistent queue status/history, validated drafts/presets, responsive layout and original application assets. | `frontend/src/App*`, `pages/*`, `components/*`, `services/workspace.ts`, `public/lunarc.*`, Electron shell/packaging |

Processing order is **source → independent layer observations → existing fusion
→ optional temporal low-frequency stabilization → final post controls → encoder**.
Placing final post controls last prevents history from defeating protected ROI or
zero residual strength. HDR processing remains signed linear scRGB FP16 or wider;
its master is not clamped to SDR. The original PQ conversion limits are retained.
The objective remains `sum_i w_i ||D_i r - r_i||² + beta ||r - r0||²`.

## Verification

Hardware: NVIDIA GeForce RTX 5060 Ti, driver 617.14. No long benchmark was run.
Raw local results are retained under `bench_out/optimization_20261009/` (ignored,
not distributed as source). Each hardware gate captures source/module/DLL hashes,
settings, intermediate/final media, ffprobe output and measured diagnostics.

- Final Python 3.11: **1456 passed**, no warnings/skips; `full_unit_final.txt`.
- Final Node: **52 passed**, no skipped tests; `node_final.txt`.
- TypeScript/Vite production build passed; Electron Windows directory packaging
  uses the existing backend layout and the new real ICO asset.
- Neutral SDR/HDR single/dual/triple exports passed. Neutral controls return the
  same candidate object without loading the new control DLLs.
- Enabled controls exports all passed with six frames each:

| Material | Single | Dual 50/75 | Triple 40/67/85 |
| --- | --- | --- | --- |
| HDR10 | 17/17 checks | 27/27 checks | 27/27 checks |
| SDR | 15/15 checks | 23/23 checks | 23/23 checks |

Evidence directories are `{hdr,sdr}_{single,dual,triple}_controls_final`.
HDR cut observations are `first, duplicate_static, duplicate_static, cut,
ambiguous, duplicate_static`; reset flags are `[true,false,false,true,false,false]`
for all independent layers. Distinct session IDs, resolved extents, call counts,
per-frame solver convergence and full output extent/count/display PTS are checked.

Native post-control smoke covers signed negative/>1 HDR values, highlight limits,
frequency controls, source-preserving ROI/debug views, malformed inputs, CUDA and
fenced D3D12 transport, close/reopen and debug-layer errors. The temporal probe
also covers achromatic cuts, exposure/translation guards and detector-history
preservation after process reset. Static synthetic flicker delta changes from
0.0124183 to 0.003267974 (RGBA8) and 0.0126956 to 0.0026006 (FP16); measured high-band
error is 1.69e-9 and 4.10e-5 respectively. Detail steps, sign reversals and changed
HUD regions reject stale history. These are bounded synthetic results, not a
long-term perceptual quality claim.

### Fusion before/after

Captured old DLL and sources are in `fusion_baseline/`. Fixed input A2 probes
retain output values/iterations with reference error 0.00192475 and adjoint error
7.46e-7 / 2.26e-8. Reduction sizes 1/255/256/257/65537, large finite values and
NaN/Inf refusal are tested. Per-dot final CPU readback is 4 bytes.

Actual isolated three-frame HDR triple exports with fixed zero-motion guidance
(`hdr_triple_isolated_new` / `hdr_triple_isolated_old`) have **bit-identical video
packet payloads and decoded YUV10**, max/mean/RMS difference 0, eight iterations
and convergence on every frame. This isolates the reduction change from varying
temporal guidance. Real independent NVOF before/after exports are not bit-identical:
max absolute 10-bit code difference 266, mean 0.2286414, RMS 1.3555873. The first
frame matches, subsequent temporal runs differ. Broad diagnostic bounds on those
runs are not evidence of tight equivalence; their raw differences are preserved.
Floating-point summation order changed, so general bit identity is not promised.

### HDR/remux integrity

Final HDR files contain HEVC Main10 10-bit video, AAC audio, SRT subtitles,
chapters, global/stream metadata and a copied attachment according to the fixture
and bounded packet-start rules. Audio/subtitle payload hashes and attachment
bytes are checked, with default FFmpeg source-start timestamp rebasing handled
explicitly. Source → video intermediate → final actual HEVC mastering-display
SEI 137 and content-light SEI 144 are byte-identical; ffprobe side-data agree.
Colors remain `bt2020nc`, `smpte2084`, `bt2020`. Fixture MaxCLL/MaxFALL are 900/250.
No silent fallback or additional video encode is used during remux.

Measured host **image** boundaries are zero for tested decode/layer/fusion/post/
temporal paths. Scalar readback still exists: post controls 32 bytes per process,
scene observation 32 bytes, temporal process 16 bytes, plus solver scalars.
This is not a claim of zero total host traffic. CUDA/D3D device transport and
synchronous fence waits also remain.

## UI and compatibility

Pages: home/import, reconstruction/compare/layers, export/queue, diagnostics/logs,
presets, software preferences, about. The new controls save validated backend
parameters; preview clearly labels its 8-bit path and export-only controls.
Requested fusion/backend is distinguished from backend-reported actual execution;
unexecuted jobs show unknown, not fictional measurements. GPU/HDR previews show
the GPU chain and do not present the ordinary raw-BGRA FFmpeg command as actual.

The existing Electron user-data location remains unchanged to preserve Chromium
storage. Version-1 workspace, export drafts and preset backups merge new neutral
controls explicitly; the backend rejects invalid parameters before application.
Malformed draft enum/numeric fields fall back individually. Density and reduced
motion preferences persist. Legacy user files are not deleted. The ESM Windows
backend process-tree cleanup is corrected, and transient startup initialization
failures retry without replacing real settings with defaults.

Actual Electron screenshots and interaction records are in `electron_ui_final/`
and the final packaged/scaling/queue captures referenced below. They are real
running application captures, not concept art.

## Limitations and honest support states

- HDR10, data-consistency fusion, scene detection and static low-frequency
  stabilization remain experimental; the short fixture is not a broad quality
  or compatibility certification. Extreme FP16 values can lose precision.
- The sampled ±4-cell detector can mistake sufficiently large motion or unusual
  flashes for cuts. Temporal stabilization supports similar static regions,
  **not motion-reprojected history**. It does not claim improved final quality
  on every moving scene.
- NVOF uses current→previous pixel motion with independent layer histories;
  no usable confidence output is exposed in this project. Official FidelityFX
  Optical Flow uses a different contract (8×8 signed motion, scene-change output,
  transfer/luminance/SM6.2/DX12 requirements). No AMD adapter is integrated:
  **NOT_TESTED / NOT_CHARACTERIZED**; no provider comparison winner is claimed.
- Copied subtitle event tails/audio padding can extend container duration beyond
  video. FFmpeg may retain a zero-duration chapter at the exact cutoff. Payloads
  are preserved, rather than rewritten to conceal this behavior.
- GUI preview is uncalibrated 8-bit display, not an accurate HDR reference.
- The Windows directory package still requires a compatible Python environment,
  FFmpeg, NVIDIA driver/runtime and the original GPU SDK dependencies; it is not
  a fully self-contained Python installer or a signed distribution claim.
- Command-list submission and fence waits remain conservative. FPS, latency,
  memory savings and long-duration stability: **Performance=NOT_CHARACTERIZED**.

Reference audits: Veyra 2.0.6 `e3aa842a00f693a4a17d1fc4c454d0ebc59428f6` (GPLv3),
Magpie experimental `27c5df91177a29b33be612e98274169f3d2fca49` (GPLv3), OpenDLSS-NR
`9d08f4184bbcb9d858e2fb7a7834ec0837a9d2f1` (MIT). No incompatible GPL code was copied.
Official flow contract: https://gpuopen.com/manuals/fidelityfx_sdk/techniques/optical-flow/

## Final combined regression and delivery

- Canonical Python 3.11.15 final suite: **1456 passed**, 140.79 seconds.
  Supplemental actual Electron Python 3.10.18 runtime: **1454 passed, 2 skipped**;
  the two packaging tests need PyInstaller, which is absent in that interpreter.
  Both tests pass in the canonical environment. Evidence:
  `full_unit_final.txt`, `full_unit_python310_final.txt`.
- Node **52 passed**, TypeScript/Vite build passed and Windows directory package
  passed: `node_final.txt`, `frontend_build_final.txt`, `package_final.txt`.
  The actual executable's ProductName/FileDescription and original embedded
  icon are Lunarc; `packaged_release/win-unpacked/Lunarc.exe` is the tested app.
- Actual packaged Electron interaction and layout gates passed at Chromium
  device scale **1.25 and 1.5**, recording matching `devicePixelRatio`, with
  1000×680, 1920×1080 and 3840×2160 window bounds. Captures:
  `electron_scale125_netstat/`, `electron_scale150_final/`. This simulates display
  scaling in the actual application; it does not change Windows display settings.
  Import/real preview, backend-validated sliders/reset, all seven pages, preset
  creation/deletion/persistence, native-session backup download, partial-v1 import,
  display preference/draft restoration and one-shot startup HTTP failure recovery
  passed without renderer errors. `cleanup.json` records each owned Python PID
  exiting after the app closes. Earlier checker-only failures are retained.
- Real queue preflight/export passed in `electron_queue_ui_run16/`: unsupported
  SDR GPU NV12 output selection blocks Add with actionable Auto/YUV420P guidance;
  the valid Auto request completes six HEVC frames with actual GPU Transfer.
- Actual UI **start → cancel while running → cancelled → row retry → pending →
  start → completed** passed in `electron_cancel_retry_final/`. Backend snapshots
  confirm each state, and final ffprobe confirms HEVC 1280×720, six frames. Actual
  backend diagnostics confirm `gpu_transfer`, with no renderer errors. Cancellation
  occurs during startup at frame 0; native failure/reset/close/reopen probes and
  unit gates separately cover GPU resource lifecycle and mid-pipeline failures.

Relevant accepted sources/tests/assets/native DLLs are committed together,
excluding unrelated AGENTS/build/release/workbuddy files and ignored evidence.
The final delivery verifies remote main against the pushed commit and reports
that immutable commit ID in the chat. No measured performance improvement or
long-duration stability claim is made: **Performance=NOT_CHARACTERIZED**.
