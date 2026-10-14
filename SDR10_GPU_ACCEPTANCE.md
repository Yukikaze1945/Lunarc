# SDR 10-bit GPU chain — 2026-10-10

## Implemented

Limited-range, explicitly tagged BT.709 `yuv420p10le` input on GPU Transfer now
uses a genuine high precision path:

`CUDA P010 → relative linear Rec.709 FP16 master → fixed bounded FP16 proxy →
Feature 18 → signed linear reconstruction / multiscale fusion → BT.709 OETF →
P010 → HEVC Main10 → existing FFmpeg remux → MKV`.

The reference white is relative linear 1 for the whole job. SDR never uses PQ,
absolute scRGB luminance, BT.2020 retagging or mastering/content-light SEI. Native
SDR decode and encode independently enforce BT.709; unexpected HDR metadata is
refused. A separate 8-bit guidance plane is used only for NVOF. The original SDR8
and experimental HDR10 branches remain separate.

`GpuSdr10Bridge` reuses existing FP16 resources, downsampling, reconstruction and
typed fusion. Each multiscale layer has its own Feature 18 session. The existing
decoder/encoder lifecycle, mapping/copy pass, output verification, cancellation,
overwrite and temporary-file behavior are reused; no new muxer was introduced.

The service selects `rgba16f` before session creation. The export page's explicit
**准备 SDR 10-bit 导出** button selects GPU Transfer / HEVC / MKV / P010 / Main10 /
preserve color and leaves NR settings intact. Overrides are shown as a custom
export preset. Preview rendering remains an 8-bit view, independent of export.

## Verification

The user's AV1 source was stream-copied to a six-frame fixture without re-encoding
video: `bench_out/sdr10_20261010/source6.mkv`. It is 3840×2160, 120 fps,
limited-range BT.709 10-bit 4:2:0, with PCM float audio.

| Gate | Result | Retained evidence under `bench_out/sdr10_20261010/` |
| --- | --- | --- |
| Single, NVOF, 6 frames | PASS | `single_v2/result.json`, diagnostics, ffprobe, output |
| Dual 50/75, NVOF, 4 frames | PASS | `dual/result.json`, diagnostics, output |
| Triple 40/67/85, NVOF, 3 frames | PASS | `triple/result.json`, diagnostics, output |
| Actual service job, zero motion, 6 frames | PASS | `service/accepted_summary.json`, job, probe, audio hashes |
| Existing HDR10, NVOF, 6 frames | PASS | `hdr_regression/result.json`, diagnostics, source/final HEVC |
| Existing SDR8, NVOF, 6 frames | PASS | `sdr8_regression/result.json`, diagnostics, output |
| Full unit suite | 1486 passed, 2 skipped | `full_unit.txt` |
| Subsequent export UI contract checks | 41 passed | Targeted run after custom-preset label correction |
| Node preview/precision tests | 31 passed | `tests/node/export_preview.test.ts`, `source_precision.test.ts` |
| TypeScript/Vite and Windows package | PASS | `package_final/win-unpacked/` |
| Final packaged UI, prepared settings and queued job | PASS | `ui_final_verified/result.json`, settled light/dark screenshots |

The SDR10 outputs are HEVC Main 10 / `yuv420p10le` / limited range / BT.709
matrix, transfer and primaries, with original display dimensions and frame counts.
Single display PTS are `[0, 8, 17, 25, 33, 42]` at 1/1000 seconds. Both source PCM
packet payload hashes match the single/service outputs. No mastering (137) or
content-light (144) HEVC SEI is added.

Dual session IDs are `[1, 2]`, resolved dimensions `[1920×1080, 2880×1616]`, and
each session executed four flow calls. Triple IDs are `[1, 2, 3]`, dimensions
`[1536×864, 2576×1448, 3264×1840]`, with three flow calls per session. Every
frame executed the typed data-consistency solver. Maximum relative solver
residuals were 0.000553 and 0.000851, below the requested 0.001 threshold.

The retained HDR regression contains video, AAC audio, SubRip subtitle and an
attachment; its actual mastering and CLL/FALL prefix SEI payloads match the source
byte for byte, and ffprobe still reports PQ/BT.2020 and mastering side data.

The independent 64×32 native SDR color probe exercised runtime shader compilation,
BT.709 bars and a grayscale ramp, decode, guidance, zero-strength reconstruction
and P010 encode. Observed error against its FP16-aware reference was 5.96e-8 for
the FP16 master, 0 guidance codes and 0/0 P010 Y/UV codes. That probe explicitly
counted 36,864 diagnostic image readback bytes, including row padding. Its source
and executable remain under `bench_out/sdr10_colour_gate/`; these synthetic
numbers are not a bound for all video content.

Actual export counters report zero at the measured decoder, Feature 18 color
upload/output readback, encoder host-pixel and diagnostic image-readback boundaries.
Color-bridge scalar counter readback still occurs and its total is unmeasured;
fusion scalar readback was 288 bytes for dual and 240 bytes for triple. Compressed
bitstream/remux traffic also crosses host memory. No total-zero-host-traffic claim
is made.

## Limits and delivery

- Current SDR10 admission is limited-range BT.709 10-bit 4:2:0, original output
  dimensions, HEVC Main10 and MKV. Other domains are not guessed or converted by
  this route. Auto/Legacy retain their existing 8-bit processing behavior.
- FP16, output quantization/clipping to nominal SDR range and lossy video encoding
  are not lossless. No general quality improvement is claimed.
- This revision used short fixtures, not a full-length render or benchmark.
  `Performance=NOT_CHARACTERIZED`.
- Native codec/color DLL hashes in the package match the rebuilt canonical DLLs.
- No commit, push, reset, stash or unrelated cleanup was performed.

Run `bench_out/sdr10_20261010/package_final/win-unpacked/Lunarc.exe` with its
containing folder intact; old running processes must be restarted to load the new
native libraries.
