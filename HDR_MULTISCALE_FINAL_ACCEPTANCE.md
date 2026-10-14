# Experimental HDR10 multiscale: final acceptance

Date: 2026-10-08. Branch: `main`. Starting HEAD:
`7195f8979583fc48b4f47654f09841e82025cec4`.

This report supersedes the outstanding hardware/UI limitations in the historical
`HDR_REMUX_INTEGRATION.md`, A1/A2/B1/B2 reports. Earlier evidence stays intact;
it is not represented as a run of the final tree. Qoder is no longer used;
Luna implemented bounded UI and lifecycle/diagnostic repairs, and Codex reviewed
the integration and ran the final gates.

## Final gate ledger

Final evidence root (ignored, not part of the source commit):
`bench_out/hdr_multiscale_final_20261008/`. Every hardware gate records source,
module, native DLL, inspector and harness hashes. Raw JSON and decoded/packet
evidence accompany each gate; exit zero alone is not acceptance.

| Gate | Evidence | Result |
|---|---|---|
| Full current unit suite | `unit_suite_final_v2.log` | 1388 passed, 0 failed, 1 warning; 145.56 s |
| Single HDR10, 6 frames, 50% | `single/gate_log.json` | 38/38 checks passed |
| Dual HDR10, 6 frames, 50/75 | `dual_final/gate_log.json` | 42/42 checks passed |
| Triple HDR10, 3 frames, 40/67/85 | `triple_final_v2/gate_log.json` | 42/42 checks passed |
| UI opt-in/settings | `HDR_MULTISCALE_STAGE_B2.md` | 17 Python tests, 45 Node tests, TypeScript/Vite build and 3 mock Playwright tests passed |
| Signed HDR solver/reference and SDR composite | `HDR_MULTISCALE_STAGE_A2.md`, `bench_out/hdr_a2_evidence/a2-20261003-015253/verdict.json` | Previously accepted; binaries unchanged |

The nine `Live.bytes_per_pixel` failures were repaired by initializing fake
sessions with the production color-format/byte-width mapping. The production
validation and float32 motion contract were retained; the motion-wrapper tests
are included in the full suite.

The remaining warning is the existing intentionally crashing double-free harness's
Windows GBK subprocess-reader `UnicodeDecodeError`, surfaced as
`PytestUnhandledThreadExceptionWarning`. Its test passed. This is not a
warning-free run and the original warning is retained in the log.

Single was executed before the final extra-layer-only telemetry projection and
construction-failure cleanup repairs. These do not change its execution path;
the successful single gate was not rerun. Dual/triple final gates include each
extra color bridge's diagnostic image-readback measurement, including preservation
of unknown (`None`) values in the unit tests. Native binaries were not rebuilt
during the UI/final integration stage.

## Data flow and actual fusion

```
P010 limited-range BT.2020/PQ
 -> pristine full linear scRGB FP16 master x (1.0 = 80 nits)
 -> independent layer D_i x, fixed-job bounded FP16 proxy
 -> independent RGBA16F Feature 18 session and independent NVOF history
 -> existing HDR reconstruction against each pristine working master
 -> signed linear HDR residual r_i
 -> existing float32 D / D^T / CG data-consistency solver
 -> fenced full FP16 composite x + strength * r
 -> existing PQ / P010 / NVENC HEVC Main10
 -> existing FFmpeg video-copy remux and uncached final verification
```

The solver minimizes

```
sum_i w_i ||D_i r - r_i||^2 + beta ||r - r0||^2
```

`r0` is the existing layer-0 Catmull baseline. Serialized float32 Lanczos plans,
all taps, clamped duplicate edge samples and the genuine transpose are shared with
SDR. The solver uses float32 work vectors and stores a full FP16 composite. It
does not sum a residual pyramid or relabel repeated calls on one session as layers.
The selected gate uses weights 1 each, beta 0.1, tolerance 0.001 (marshalled float32
`0.0010000000474974513`) and maximum 128 iterations. Solver nonconvergence fails
the job without substituting the baseline.

NVOF alone reads separate 8-bit guidance, with float32 motion output; picture color
and residuals remain FP16 or wider. Paper white is fixed at scRGB 8.0 / 640 nits
throughout all layers and frames, with no dynamic exposure/paper-white choice.
The existing final PQ encoding clamp and counted unrepresentable-half fallback
are preserved. No HDR mathematics, Feature 18 model, new muxer or GPL-derived
code was introduced in the final integration work.

Dual resolves to `960x544` and `1440x808`, with distinct session IDs `[1,2]`,
6 inference and 6 NVOF calls per layer, and reset `[true,false,false,false,false,false]`
per layer. All six frames execute the typed solver, take 7 iterations, and have
relative residual at most `0.0008318258333019912`; unrepresentable components are 0.
Triple resolves to `768x432`, `1288x720` and `1632x920`, with distinct session IDs
`[1,2,3]`, 3 inference and 3 NVOF calls per layer, and reset `[true,false,false]`
per layer. Its frames take 8, 8 and 7 solver iterations, with relative residuals
`0.0007234618533402681`, `0.0005769955459982157` and
`0.0009855427779257298`; all satisfy the actual float32 tolerance and have zero
unrepresentable components.

## Existing remux reused

The HDR route uses `gpu_transfer_export.needs_copy_pass`, `build_mux_command` and
`run_copy_pass`, the existing pipeline mapping rules, and its uncached final output
verification. The encoder finishes/releases a uniquely owned video-only temporary
MKV before FFmpeg reads it as input 0 alongside the original as input 1. Video is
always `-c:v copy` in that pass; its packet payload hashes equal the intermediate.

Existing audio-mode/`with_audio` priority, audio copy or requested codec settings,
subtitle copy/convert/drop, chapters, global/stream descriptive metadata,
attachments and MKV container rules are reused. SDR defaults keep their existing
mapping semantics. The shared copy-pass execution now also terminates an expired
FFmpeg process rather than leaving a writer running. Cancellation, overwrite,
source/output collision, original-error preservation, independent-session cleanup,
and temporary-file cleanup are covered by the unit suite.

The final lifecycle repair guards the interval after extra sessions/providers
have opened but before a chain adopts them: TapPlans or chain-construction failure
releases every extra resource even when one close fails, preserving the original
exception with cleanup notes and leaving the borrowed anchor/root provider alone.
An injected two-extra-layer construction failure tests all four release attempts.
This failure-only change did not invalidate the successful dual output; the triple
gate and final full suite were executed after it.

The tested rich final MKV contains:

- 1 full-resolution HEVC Main10 video stream: `yuv420p10le`, limited (`tv`),
  `bt2020nc / smpte2084 / bt2020`;
- 1 AAC LC audio stream, `jpn`, title `Commentary`, default disposition;
- 1 SubRip subtitle stream, `eng`, title `Captions`, original disposition;
- 1 text/plain attachment, original `note.txt` content (36 bytes);
- 2 chapters with original time bases, bounds and titles, and original authored
  global title/comment.

Audio and subtitle packet payloads are checked as exact copied source prefixes;
attachment bytes are checked directly. Packet order is decoded/display order when
checking timestamps, so B-frame decode order is not mislabeled a PTS failure.
Single/dual have 6 packets and 6 decoded frames at display PTS
`[0,40,80,120,160,200]` ms. Triple has 3 packets and 3 decoded frames at display
PTS `[0,40,80]` ms, 6 copied audio packets and 1 copied subtitle packet.

For the three-frame smoke, the original six-frame fixture's first subtitle begins
at 40 ms; shared `-shortest` emitted no subtitle packet on the first run. An initial
fixture correction ended all captions at 40 ms, which instead truncated video to
one packet/frame. Both failures are preserved under `triple/` and `triple_final/`.
The final variant changes the first cue to 0–40 ms and retains the second cue at
200–240 ms, so the subtitle track's total duration remains 240 ms. It retains all
six source HEVC packet hashes/SEIs, original audio/attachment/chapters and authored
metadata, and both original caption payloads. The exact configured FFmpeg and
shared copy helper were checked offline against the captured three-frame encode
before running `triple_final_v2`. See
`triple_fixture_probe_v2/probe_evidence.json`. No remux rule or assertion was
removed to make the gate pass.

## Static HDR metadata

Source, encoded intermediate and remuxed final retain actual HEVC mastering-display
SEI type 137 and content-light SEI type 144 payloads byte for byte. Occurrence counts
may differ between source and encode; every payload has to match.

- Mastering 137: `21349b46199608fc8a4839083d1340420098968000000032`
  (BT.2020/D65, mastering 1000 / 0.005 nits).
- Content light 144: `038400fa` (MaxCLL 900, MaxFALL 250).

ffprobe frame side-data independently confirms mastering and content light, and
Main10/10-bit/color fields are checked in the final container. There is no second
video encode or silent SDR fallback in the tested remux path.

## Traffic, numerical limits and remaining container limits

The measured primary decode/DLSS/HDR/encoder image-host boundaries are 0; each
extra DLSS session's staging/output readback and color-bridge diagnostic image
readback are also measured as 0. Extra-layer NVOF device-input traffic is D2D,
including the first frame copied twice to initialize history, not host traffic.
The fusion ABI counts D2D source/observation transport and scalar readback; it has
no separate fusion host-image counter. The latter claim is structural/source/API
evidence, not an invented zero measurement.

For dual, fusion scalar readback is measured as 12,441,600 bytes; for triple it is
6,739,200 bytes. Color-bridge
counter readback additionally exists; completed dispatch counts give arithmetic
byte ranges, but the ABI does not measure total scalar report traffic. The total
scalar field remains `None`. This report does **not** claim total host traffic 0.

Accepted A2 synthetic evidence retains signed negative and above-one HDR values,
real FP16 storage/reference error (maximum 0.001924753, half-grid ULP 0.001953125),
and objective reduction from 0.004366517 to 0.001137981. Weighted consistency drops
from 0.004366517 to 0.000421893; offline unweighted consistency also decreases.
Adjoint pairing is CPU-side on the shared plans (relative errors 7.4567e-7 and
2.2641e-8), not a direct HDR GPU adjoint-dispatch experiment. Existing/new SDR
composite dumps are byte identical. Extreme unrepresentable FP16 samples have
counted losses/fallback, so neither HDR roundtrip nor fusion is described as lossless.

Remaining limits:

- Experimental HDR export is limited to the accepted limited-range BT.2020/PQ
  inputs and MKV/Main10 output; HLG, full-range/constant-luminance input, resized
  output, multiple passes and nonempty caller FFmpeg `extra_args` are refused.
- Shared remux `-shortest` semantics are preserved. Sparse subtitle timing can
  omit subtitle packets on a very short clip, and a subtitle track ending before
  the video can truncate the video. Fixture-specific investigations and failed
  gates remain recorded; these are not repaired by changing SDR/remux behavior.
- Audio/subtitle codec conversion choices are covered by unit/offline command
  checks, not an exhaustive hardware codec/container matrix. Existing metadata-off
  semantics can retain chapter markers without their descriptive titles.
- Only the specified short single/dual/triple fixture and one GPU configuration
  are accepted here. Arbitrary scales, long-duration temporal behavior, visual
  quality and 4K memory use are not comprehensively characterized.
- Performance is `NOT_CHARACTERIZED`; no benchmark was run.

## Native artifacts

Final native DLLs match the accepted A2 builds:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `native/gpu_hdr_scrgb_bridge.dll` | 251392 | `6d0cb889d3fa6f4685ca6e2cbbb4845967eb7beaa18375d3110e1d47b356937f` |
| `native/gpu_multiscale_fusion_bridge.dll` | 185856 | `76a81702f7536cc93d3162e8d85340e956941855871f280a2bc49e32931f2007` |

The final source commit includes related HDR multiscale/remux source, tests,
documentation and these tracked release DLLs. Ignored gate/build output and the
unrelated local `AGENTS.md`, `_selftest.txt`, `native/db_exports.bat` and
`native/nvenc_hdr_options_probe2.txt` remain excluded and preserved.

Final MKV file hashes (container hashes are not bitstream determinism claims):

| Gate | SHA256 |
|---|---|
| Single | `d0fba25f83f8801a3d545da2503bb68412f2aa31585faca7abf29f6780e2080c` |
| Dual | `06193c963fc4e219f399c58516c3b637f3323da09a55fb944a95882056687280` |
| Triple | `bd0f316b059aacb5299a23d36333af08f97c33acf851a99a4fde0a4e1b8cb3a2` |
