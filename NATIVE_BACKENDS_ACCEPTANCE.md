# Native backends and preview acceptance — 2026-10-10

## Implementation

- Native preview consumes the same Single/Dual/Triple processing chains as GPU
  Transfer export. It keeps decoder, independent Feature18 sessions, guidance
  histories and solver resources on a persistent execution thread. Seek, repeated
  frames, source/settings changes and scene discontinuities reset history.
- SDR10 and HDR10 retain their P010 and FP16 processing paths. Preview performs
  its display conversion and readback after processing; those display bytes are
  counted separately. Existing signed observations and typed data-consistency
  solver are reused, not replaced with serial NR passes or residual pyramids.
- DLAA creates/evaluates the actual NGX SuperSampling feature with quality DLAA
  and equal input/output dimensions. The NR→DLAA combination shares the D3D12
  device and uses a separate feature. A second auxiliary DLAA feature on that
  same NR core is rejected. Signed DLSS runtime stays external.
- RTX VideoSuperRes calls the licensed NvVFX VideoSuperRes model. Its D3D12/CUDA
  adapter publishes CUDA completion before the D3D12 output copy, including the
  first frame. Standalone VSR owns interoperability resources without creating
  a fake NGX session. Runtime path changes while effects are live are refused.
- AMD FidelityFX Optical Flow uses the pinned MIT SDK and an independent context
  per processing layer. Provider discovery checks its adapter ABI without GPU
  allocation; device admission remains a per-job check. No DIS substitution is
  used for a FidelityFX request. See `docs/AMD_FLOW_DEPENDENCIES.md`.
- Original/final comparison decodes actual files, never recomputes AI. It uses
  actual PTS, common media time, nearest available frames, one audio player,
  side/wipe/A-B views, frame stepping and fit/100%/200% zoom. Completed jobs,
  saved history and manual file selection open the page.

## Current-worktree gates

All picture tests are short fixtures; no performance benchmark was run.
Final current-worktree Python unit suite: 1505 passed. Node tests: 57 passed.
TypeScript/Vite build passed. Application-owned staged whitespace checks passed;
the pinned upstream SDK's original whitespace is retained.

| Case | Frames | Maximum display MAE / p95, 8-bit values | Result |
|---|---:|---:|---|
| SDR8 Single | 6 | 2.68 / 7 | PASS, offline recheck of retained pictures |
| SDR8 Dual 50/75 | 6 | 2.71 / 8 | PASS, offline recheck of retained pictures |
| SDR8 Triple 40/67/85 | 6 | 2.62 / 7 | PASS, offline recheck of retained pictures |
| SDR10 Single | 6 | 1.81 / 7 | PASS |
| SDR10 Dual 50/75 | 6 | 1.87 / 8 | PASS |
| SDR10 Triple 40/67/85 | 6 | 1.86 / 8 | PASS |
| HDR10 Single | 6 | 0.90 / 2 | PASS |
| HDR10 Dual 50/75 | 6 | 0.87 / 2 | PASS |
| HDR10 Triple 40/67/85 | 6 | 0.98 / 2 | PASS |

The tolerance is MAE ≤8, p95 ≤24 and correlation ≥0.95 for visible detail;
near-black frames have an error ≤1 diagnostic limit and cannot establish picture
quality. The user's first six SDR10 frames were near-black, so visible SDR10
detail was tested with a separate six-frame BT.709/Main10 fixture with PCM audio.
SDR8's interrupted original checker did not persist its live diagnostics or mux
argv. Its pictures and packet evidence were checked offline, with that missing
evidence explicitly recorded; the separate nine-frame FidelityFX pipeline gate
proves its independent Single/Dual/Triple histories, dispatch and solver calls.

Current SDR10/HDR10 Dual/Triple evidence contains 2/3 distinct session IDs,
resolved layer dimensions, six per-frame converged typed solver reports,
source/preview/export PTS, intermediate/final compressed packet equality,
full extent and final mapping checks. The six-frame FidelityFX cases are entirely
inside the SDK's startup suppression period. Nonzero vector direction was
separately verified after startup on the nine-frame shifted-input gate.

Actual DLAA create/evaluate/release, textured moving inputs and reset were tested.
VSR quality 1–4 passed 720→1080 and 1080→4K model/extent gates. Standalone DLAA,
VSR, NR→DLAA and NR→VSR passed meaningful first-frame and all-frame pixel checks,
preview and six-frame exports. NR→DLAA was additionally retested after the final
host ownership fix. No quality-improvement claim follows from these smoke tests.

## HDR remux

The latest rich HDR fixture yields one HEVC Main10/yuv420p10le video stream with
bt2020nc/smpte2084/bt2020, one Japanese AAC track, one English SubRip track, one
attachment and two chapters, with global title/comment retained. Audio/subtitle
packet payloads and attachment bytes match the source under existing video-span
cutoff semantics. All six video packets, display PTS and extent survive remux.
Mastering and CLL/FALL side data match source/intermediate/final. Actual HEVC SEI
137 and 144 payloads match byte-for-byte (source 2 instances per type,
intermediate/final 6 each). This reuses the existing FFmpeg copy/mapping and
uncached final verification, with no custom muxer or extra video encode.

Measured export decoder/Feature18/output/encoder host-image counters are zero in
these SDR10/HDR10 gates. Colour-counter scalar readback still exists and its
bytes are unmeasured; solver and FidelityFX scalar counters are reported
separately. Preview intentionally downloads display pixels. This is not a claim
of zero total host traffic, lossless FP16 conversion or lossless HEVC encoding.

## UI and remaining limits

The Swiss workbench and comparison page use the same palette, flat grid and fine
dividers. Basic controls remain Default/Movie/Natural, four strength sliders and
automatic mask; Advanced is closed. Both launch scripts build the current
frontend and open Lunarc. Classic Tkinter is an explicit optional entry only.

SDR comparison keeps BT.709 code values through its display adapter; actual
midgray tests cover 8/10-bit sources. HDR comparison uses the same fixed,
uncalibrated preview display mapping; it is not calibrated HDR monitor output.
DLAA/VSR currently admit SDR8 only and refuse SDR10/HDR instead of downcasting.
Prerecorded DLAA uses zero synthetic depth/jitter, not engine geometry.
The FidelityFX adapter currently requires this application's NVIDIA CUDA/D3D12
interop; its AMD algorithm name does not establish AMD-GPU end-to-end support.
FidelityFX has fixed 8×8 integer vectors and a six-frame startup period, with no
invented quality selector. VFR comparison uses actual PTS; preview's requested
frame index still follows nominal frame rate. MKV is the verified HDR/SDR10
container; compressed audio padding and subtitle cue duration may extend it.
External NVIDIA runtimes/models are not redistributed. Performance=NOT_CHARACTERIZED.

## Raw evidence (ignored local artifacts)

- `bench_out/preview_parity_20261010/current_complete/report.json`
- `bench_out/preview_parity_20261010/sdr8_offline_recheck.json`
- `bench_out/amd_flow_20261010/pipeline_acceptance_640x480/report.json`
- `bench_out/amd_flow_20261010/native_gate_startup_corrected/fidelityfx_flow_native_gate.json`
- `bench_out/native_backends_20261010/dlaa/textured_final/report.json`
- `bench_out/native_backends_20261010/vsr_fixed/rtx_vsr_native_gate.json`
- `bench_out/native_backends_20261010/native_quality_full/report.json`
- `bench_out/native_backends_20261010/nr_dlaa_current_host/report.json`
- `bench_out/native_backends_20261010/electron_swiss_final_run4/report.json`
- `bench_out/native_backends_20261010/package_final_manifest.json`
- `bench_out/native_backends_20261010/unit_commit_gate.txt`
- `bench_out/native_backends_20261010/node_final.txt`
- `bench_out/native_backends_20261010/frontend_release_build.txt`

Final package: `bench_out/swiss_current_20261010/win-unpacked/Lunarc.exe`.
Its current frontend, Python modules and native bridges match source SHA256s.
Earlier package paths are historical evidence and must not be launched as current UI.
