# HDR Stage B — P010 → HEVC Main10 NVENC encoder boundary with HDR10 metadata handoff

Date: 2026-09-30. Owner: Qoder (implementation), Codex (audit).
Evidence: `bench_out/hdr_p010_nvenc/` — `p010_nvenc_gate.json` (106 checks, all passing),
`raw_p010_nvenc_gate.txt`, `raw_nvenc_hdr_probe.txt`, `nvenc_hdr_options.txt`,
`raw_ffmpeg_encoder_surface.txt`, `raw_build_and_option_evidence.txt`,
`raw_git_diff_nvenc_sdr_isolation.txt`, the three `raw_build_*.txt` logs, the four media
outputs and six `ffprobe_*.json` dumps.
Scope of this job: additive native API + one 2-frame encoder-boundary gate. **No UI, no full
export orchestration, no DLSS in this stage.**

## Verdict

**Exit 0, 106 checks, 0 failures.** What is proven, and nothing more:

* `nvenc_hdr_open` + `nvenc_encode_p010` take **borrowed CUDA P010 planes straight from
  `nvdec_bridge_open_hdr`** and produce a real HEVC **Main10** stream: the coded bytes, demuxed
  with `-c copy` and probed, read `hevc / Main 10 / yuv420p10le / 1920x1080`, so the 10-bit depth
  and the extents come from the parameter sets, not from a container tag.
* The HDR10 colour contract is declared and confirmed by the encoder itself:
  `actual_pix_fmt = 117 (cuda)`, `actual_sw_pix_fmt = 158 (p010le)`, `actual_profile = 2`,
  `bits_per_raw_sample = 10`, `range/primaries/trc/space = 1/9/16/9` (limited, BT.2020,
  SMPTE ST 2084, BT.2020nc), `hw_frames pool 16 @ sw 158`.
* **No host pixel memory and no conversion is involved on the processing path**: encoder
  `host_pixel_bytes = 0`, `rgb_conversion_bytes = 0`, ABI 2 `staging_gpu_copy_bytes = 0` and
  `conversion_output_bytes = 0`; decoder `host_frame_copy_count = 0`, `host_frame_copy_bytes = 0`,
  `scale_device_output_bytes = 0`. The only frame traffic is
  `p010_device_copy_bytes = 12 441 600 = 2 × 3 × 1920 × 1080`.
* **The supplied rational metadata survives the handoff exactly.** Every ST 2086 primary, the
  white point and both luminance limits, plus MaxCLL/MaxFALL, come back equal **in value**; the
  per-frame mastering and content-light side-data counters read 2 and 2 for two frames, and the
  decoded frames of the output carry both blocks. Absence stays absent: a source with no metadata
  gains none (`metadata_written_to = 8` — colour tags only, no metadata bits).
* Unsupported combinations are **refused, not dropped or downgraded**. Seven open attempts return
  −1 with the contract named in the message and **no file written**, and three encode attempts
  (undersized pitch, null plane, null bridge) return −1 with nothing submitted; there is no NV12
  fallback and no silent SDR path. A fourth refusal is measured on the other side of the boundary —
  `nvenc_encode_rgba` on an HDR bridge returns −1, but with an **empty** error string, so it is
  recorded as a diagnostic gap in §6 rather than as a clean contract refusal.
* The adapter ownership question is answered with a real number: `cuDeviceGetLuid` of the borrowed
  CUcontext gives LUID `[173,113,1,0,0,0,0,0]`, node mask 1, captured on the encoder bridge that
  opened on the decoder's context.

**Not proven here.** No DLSS, no resampler, no FP16→P010 conversion and no full production frame
were involved: the two frames are a decode of an existing HDR10 fixture, so this is an **encoder
boundary** gate only. The **in-bitstream SEI is NOT emitted** — see §5, which is the failing piece
of requirement 3 and is recorded with runtime proof rather than glossed over. No HDR10 export has
passed yet; the formal Feature 18 HDR contract remains unproven and performance is
**NOT_CHARACTERIZED** by this job.

## 1. Additive API surface (ABI 2 untouched)

`native/nvenc_bridge.h` keeps every ABI 2 declaration byte-for-byte and adds a separate section
from `#define NVENC_HDR_ABI_VERSION 1u`:

| entry | purpose |
| --- | --- |
| `nvenc_hdr_abi_version()` | 1 |
| `nvenc_hdr_get_defaults(NvencHdrDefaults*)` | the entry's own HDR10 defaults, so a caller never hardcodes them |
| `nvenc_hdr_capability(cuda_context, w, h, NvencHdrCapability*)` | what the *installed* encoder can do, queried at runtime |
| `nvenc_hdr_open(cuda_context, w, h, fps_n, fps_d, tb_n, tb_d, utf8_path, NvencHdrOptions*, NvencHdrMetadata*, NvencBridge**)` | HDR10 open; **takes no D3D12 device** |
| `nvenc_encode_p010(bridge, y_ptr, y_pitch, uv_ptr, uv_pitch, pts)` | two borrowed device planes + pitches + source PTS |
| `nvenc_hdr_get_counters` / `nvenc_hdr_get_contract` | additive diagnostics and the accepted-format readout |

`nvenc_finish` and `nvenc_close` are reused unchanged, so flush/drain/cancel/cleanup semantics are
the SDR pipeline's, not a second implementation.

Sizes are pinned twice: `native/nvenc_hdr_probe.cpp` (new, 210 lines) asserts them **and every ABI 2
offset** at compile time, and the Python gate re-pins the ctypes mirrors against the numbers the
compiler printed:

```
abi nvenc=2 nvenc_hdr=1 nvdec=3
sizes options=112 counters=64 defaults=56 capability=128 contract=268 metadata=108
```

`NvencHdrMetadata` is asserted layout-identical to the decoder's `NvdecHdrMetadata`
(`sizeof` equal and `offsetof(max_fall)` equal), so a caller copies fields rather than re-spelling
them, and the ABI 2 `NvencBridgeOptions` offsets (0/8/16/24/32/40/48/56/64/68/72, size 80) and
`NvencTransferCounters` (size 48) are still asserted — a future edit that moves them fails the
build instead of silently misreading a member.

## 2. Capacity actually queried, and refusals actually taken

`nvenc_hdr_capability` on the decoder's own CUcontext (`p010_nvenc_gate.json`):

```
encoder_present=1 p010_supported=1 cuda_supported=1 main10_profile_supported=1
cuda_device_created=1 hdr_metadata_options=0 supported=1
encoder=hevc_nvenc profile_name="Main 10"
```

P010 and CUDA are read from `avcodec_get_supported_config(AV_CODEC_CONFIG_PIX_FORMAT)`; the encoder
also advertises them in its own banner (`Supported pixel formats: … p010le … cuda`,
`raw_ffmpeg_encoder_surface.txt`). `codec->profiles` is NULL for `hevc_nvenc`, so the profile is
resolved by asking the encoder to set the name and checking the number it lands on. The probe
records the trap that cost the first attempt:

```
codec->profiles=0000000000000000
profile children rc=0 generic_value=-99 name=(null)
profile plain    rc=0 generic_value=2   name=Main 10
```

`AV_OPT_SEARCH_CHILDREN` returns 0 but writes NVENC's **child** copy, leaving
`AVCodecContext::profile == AV_PROFILE_UNKNOWN (-99)`; the plain search is what lands the 2 that
NVENC switches on. `hdr_apply_profile` (`nvenc_bridge.cpp:921`) retries plain, corrects the generic
field, and refuses if the build cannot name the profile at all. It does **not** compare the string
`"main10"` to `avcodec_profile_name()`, because this library spells profile 2 `"Main 10"`.

Defaults, from the entry itself rather than typed in: `sw_pix_fmt=158 range=1 primaries=9 trc=16
space=9 profile=main10`.

Refusals actually taken at runtime. The first seven are **open** attempts — each `rc=-1`, a null
bridge out-parameter and **no file created**; the last three go through `nvenc_encode_p010` (two on
a live HDR bridge, one with a null handle) — each `rc=-1`, and the bridge's `frames_submitted` is
still 0 afterwards:

| request | error returned |
| --- | --- |
| `codec = h264_nvenc` | `codec h264_nvenc is not the HDR10 entry's hevc_nvenc (this bridge does not fall back to another codec or to NV12)` |
| `profile = main` | `profile main is not the HDR10 entry's main10` |
| `sw_pixel_format = nv12(23)` | `requested software pixel format 23 is not the HDR10 value 158: this entry encodes P010 HEVC Main10 with BT.2020/PQ/limited only` |
| `color_primaries = bt709(1)` | `requested color primaries 1 is not the HDR10 value 9: …` |
| `extra_options = pix_fmt=nv12` | `extra option pix_fmt is fixed by the HDR10 entry; refusing rather than overriding the request` |
| width 1921 (odd) | `invalid nvenc_hdr_open arguments` |
| `gop = 2` | `open NVENC HDR10 encoder: Invalid argument (-22)` — NVENC's own rule, see §9.4 |
| `y_pitch = 1920` (< width×2) | refused, `frames_submitted` stays 0 |
| null `y_ptr` | refused (`encode_p010 was given a null device plane`) |
| null `bridge` | refused by the argument guard, `invalid encode_p010 arguments` (`nvenc_bridge.cpp:1701-1705`) |

**Inspected, not tested:** the `!b->hdr_mode` branch two guards later (`nvenc_bridge.cpp:1706-1710`,
"this one is an SDR NV12 bridge") is read in the source but the gate never feeds it a real SDR
bridge handle — it has no reason to open one, so that refusal is asserted from code, not measured.
The opposite direction *is* measured: `nvenc_encode_rgba` on an HDR bridge (see §6).

Locked-option refusal covers `profile`, `pix_fmt`, `sw_pix_fmt`, `rc`, `colorspace`,
`color_range`, `alpha_bits`. Extent validation requires both dimensions positive and even, checks
`pitch & 1`, `pitch >= width*2` for **both** planes, and rejects a `pitch + row*(rows-1)` span that
would overflow (`nvenc_bridge.cpp:1717-1730`). A `NvencHdrOptions`/`NvencHdrCounters`/
`NvencHdrContract` whose `struct_size` is not the compiled size is refused at runtime.

## 3. The 2-frame boundary smoke (`native/nvenc_hdr_p010_smoke.py`, 1412 lines)

`build/hdr_probe/hdr_src.mkv` (1920×1080 HEVC Main10, 25 fps, `tbn 1/1000`, tv/bt2020nc/bt2020/
smpte2084, in-bitstream mastering + CLL) is opened with `nvdec_bridge_open_hdr`, and **exactly two
frames** go through `nvenc_hdr_open` → `nvenc_encode_p010` → `nvenc_finish` → `nvenc_close`.

The frames are **streamed, not batched**: borrow → encode → borrow → encode. This is load-bearing
and the gate records why — the second borrow reported the *same* device address as the first
(`0xd0b600000`, `0xd0b600000`) because `nvdec_bridge_next` unrefs the surface it returned last time
and NVDEC's pool hands it straight back. Decoding both frames first would have submitted the second
frame's pixels twice. `nvenc_encode_p010` copies both planes with pitched `cuMemcpy2D_v2` and then
`cuCtxSynchronize` before returning (`nvenc_bridge.cpp:1204-1223`), so the source is consumed
before the next borrow runs. Pixel-level content equality is **not** asserted, because proving it
would need a host readback.

The copy writes `row_bytes = width*2` per row for `height` luma rows plus `height/2` chroma rows
into an encoder-owned `av_hwframe_get_buffer` frame, checking that the *encoder's* own linesizes are
not under the row payload either; row padding is never touched.

| counter (additive `NvencHdrCounters`) | case A | case B (no-metadata source) |
| --- | --- | --- |
| `frames_submitted` | 2 | 2 |
| `packets_written` | 2 | 2 |
| `p010_device_copy_bytes` | 12 441 600 = 2×3×1920×1080 | 614 400 = 2×3×320×320 |
| `host_pixel_bytes` | 0 | 0 |
| `rgb_conversion_bytes` | 0 | 0 |
| `mastering_side_data` / `content_light_side_data` | 2 / 2 | 0 / 0 |
| ABI 2 counters on the same bridge | 0 across the board | 0 across the board |
| decoder host copies / scale output | 0 / 0 | 0 / 0 |

PTS are the source's own (`[0, 40]` at `tbn 1/1000`), the output holds exactly two packets and two
decoded frames in the same order, and `nvenc_finish` drains both.

## 4. ffprobe and decoder read-back of the output

```
codec_name=hevc profile=Main 10 pix_fmt=yuv420p10le width=1920 height=1080
color_range=tv color_space=bt2020nc color_transfer=smpte2084 color_primaries=bt2020 r_frame_rate=25/1
```

Two expectations were corrected against reality rather than against hope:

* a CPU decode of P010 4:2:0 is named **`yuv420p10le`**, not `p010le` — `p010le` is what the
  encoder-side contract reports for the CUDA surface; both are gated, on their own side;
* the Matroska demuxer emits **no `bits_per_raw_sample`**, so 10-bit is gated on the decoded sample
  format here and on the parameter sets in the Annex-B probe, with the encoder's own
  `actual_bits_per_raw_sample = 10` gated separately. `bits_per_raw_sample` is assigned to the
  codec context **after** `avcodec_open2` (asking NVENC before the open makes the wrapper compare 10
  against the CUDA descriptor's depth of 0 and log *"Specified bit depth 10 not possible with the
  specified pixel formats depth 0"*); it still reaches `avcodec_parameters_from_context`.

Re-decoding `a_hdr10_out.mkv` through the same HDR borrow reports a **P010 CUDA surface** and the
metadata block, so the file is readable as HDR by this project's own decoder, not just by ffprobe.

## 5. Metadata: the mechanism, the exact comparison, and the SEI that is not there

Mechanism (`nvenc_bridge.cpp:891`, `1026`, `1106`, reported as `metadata_written_to = 15`):

1. `av_frame_new_side_data(AV_FRAME_DATA_MASTERING_DISPLAY_METADATA)` and
   `AV_FRAME_DATA_CONTENT_LIGHT_LEVEL` on **every submitted frame**, filled straight from the
   caller's `NvencHdrMetadata` rationals — `fill_mastering` copies `AVRational` in/out with no
   intermediate float, and partial `has_*` flags are honoured (`mastering_has_primaries`,
   `mastering_has_luminance`).
2. `av_packet_side_data_add` of `AV_PKT_DATA_MASTERING_DISPLAY_METADATA` /
   `AV_PKT_DATA_CONTENT_LIGHT_LEVEL` into `AVCodecContext::coded_side_data` (bit 4) and into
   `AVStream::codecpar->coded_side_data` (bit 2). `add_coded_side_data` leaves an existing entry of
   the same type alone: the source declared it once, not twice.
3. Container colour/bit-depth declaration (bit 8); the muxer derives the Matroska mastering and
   content-light elements from the coded side data.
4. Per-frame attach counted (bit 1). **Requirement 1's "no fallback": if a metadata write fails the
   open fails**, it is not skipped.

Exact comparison, source spelling vs read-back (`metadata_handoff.exact_comparison`,
`equal_in_value: true`, `value_differences: {}`):

| quantity | as supplied | read back from the output | value |
| --- | --- | --- | --- |
| red primary | 34000/50000 | 17/25 | 0.68 |
| green primary | 13250/50000 · 34500/50000 | 53/200 · 69/100 | 0.265 · 0.69 |
| blue primary | 26500/50000 · 10000/50000 | 53/100 · 1/5 | 0.53 · 0.20 |
| white point | 15635/50000 · 16450/50000 | 3127/10000 · 329/1000 | 0.3127 · 0.329 |
| min / max luminance | 1000/10000 · 10000000/10000 | 1/10 · 1000/1 | 0.1 / 1000 |
| MaxCLL / MaxFALL | 1000 / 400 | 1000 / 400 | exact integers |

Matroska stores ST 2086 in its own fixed-point denominators, so the round trip **reduces the
spelling**; the values are identical. The gate therefore compares `fractions.Fraction` values, never
the numerator/denominator pair and never a float — comparing spellings would have reported a loss
that did not happen. Independently of the bridge, ffprobe's decoded-frame side data on
`a_hdr10_out.mkv` shows `Mastering display metadata` and `Content light level metadata` on **both**
frames, with `red_x "17/25" … max_luminance "1000/1"`, and the gate `A: the mastering values on the
decoded frames equal the source's` parses those strings back into Fractions and compares. Container
tags alone were deliberately not accepted as proof.

**The bitstream does not carry the SEI.** Both files below are demuxed `-c copy`, so they are the
bytes the encoder produced — no re-encode:

| stream | mastering blocks | content-light blocks |
| --- | --- | --- |
| authored fixture → `a_src_copy.hevc` (positive control) | 6 | 6 |
| this entry's output → `a_hdr10_out.hevc` | **0** | **0** |

The probe demonstrably detects an in-bitstream SEI (it found the fixture's), so the zero is the
encoder's, not the measurement's. Corroborating API evidence: `hdr_metadata_options = 0` — the entry
asked the encoder about eleven plausible spellings with `AV_OPT_SEARCH_CHILDREN` and found none —
and the encoder's own option surface has no metadata authoring entry at all; the only SEI-adjacent
options are `-extra_sei` ("Pass on extra SEI data (e.g. a53 cc)", default true) and `-udu_sei`
(user-data-unregistered pass-on), neither of which forwards frame mastering side data, and the
generic `av_opt_serialize` dump (`nvenc_hdr_options.txt`, 77 entries) contains only
`color_primaries/color_trc/colorspace/color_range` as colour *declarations*. This is a name-and-
behaviour probe, not a proof that no option exists under some other name; the behavioural zero above
is the load-bearing evidence. **Requirement 3's "HEVC bitstream SEI" therefore is not met on this
build/driver (RTX 5060 Ti, 617.14) and is reported as such: the container and the decoded frames
carry the metadata, the elementary stream does not, and a player that reads only the raw `.hevc`
would see none.** An earlier draft gated extents on the encoder's *direct* raw-ES output; that file
is not standalone-decodable (global-header parameter sets live in extradata and the raw muxer has no
slot for them), so the evidence now comes from demuxing the container instead.

## 6. SDR isolation: tested vs inspected

* `git diff` of `native/nvenc_bridge.cpp` against HEAD is **1035 insertions, 1 deletion**, and the
  deletion is a single blank separator line (`raw_git_diff_nvenc_sdr_isolation.txt`). The SDR
  entries — `nvenc_open`, `nvenc_encode_rgba`, `nvenc_encode_texture`, `nvenc_get_last_nv12_planes`,
  `encode_frame`, `convert`, `configure_encoder`, `nvenc_finish`, `nvenc_close` — are textually
  untouched; the HDR open has its own `avcodec_open2` call site (SDR `:603`, HDR `:1043`) and the
  only shared additions are new struct members and the `cuMemcpy2D_v2` driver symbol, which the SDR
  path does not call.
* **Tested:** the ABI 2 layout asserts compile; `nvenc_get_transfer_counters` on an HDR bridge
  reports all-zero ABI 2 counters; and a child process opens a real HDR bridge, calls
  `nvenc_encode_rgba` on it and reports `GRACEFUL rc=-1 frames_submitted=0 rgb_conversion_bytes=0`
  — the two entries cannot be confused at runtime.
* **Not tested, and not run because the brief makes it conditional:** no SDR encode regression was
  re-run. Nothing in `configure_encoder` changed, so the "concretely risks SDR" trigger did not
  fire, and the passing Feature 18 SDR host test was left alone as instructed. The residual risk is
  honest: the DLL was rebuilt, so a SDR caller now loads new bytes; the interface is gated, a full
  SDR encode is not. **Inspected, not measured.**
* One diagnostic gap found while testing it: `nvenc_encode_rgba` on an HDR bridge refuses with an
  **empty** `nvenc_last_error()` — the refusal comes from the SDR `convert` path, which returns −1
  without setting a message. Behaviour is correct, the message is missing; left unfixed because
  patching it means editing shared SDR code, which this stage does not touch.
* Two artefacts record the old cleanup semantics rather than a new one: `rejection_bridge.mkv` and
  `child_rgba_on_hdr_bridge.mkv` are 652-byte header-only Matroska files, because a bridge closed
  without `nvenc_finish` leaves whatever the muxer wrote — the same shared `nvenc_close` path the SDR
  pipeline has always had.

## 7. Build and ABI gates

| target | command shape | result |
| --- | --- | --- |
| `nvenc_bridge.dll` | `build_nvenc_bridge.bat` → `cl /LD /O2 /EHsc /std:c++17 /W4 /WX /wd4244 /utf-8` | `raw_build_nvenc_bridge.txt`: 0 `warning C`, 0 `error C`, 0 `error LNK`, library + DLL produced |
| `nvdec_bridge.dll` | `build_nvdec_bridge.bat`, same flags | same, 0 diagnostics |
| `nvenc_hdr_probe.exe` | `build_nvenc_hdr_probe.bat`, same flags, links `nvenc_bridge.lib` + `nvdec_bridge.lib` | same, 0 diagnostics |
| ctypes ↔ compiled layouts | `binding_size_gate` | 6/6 sizes equal, `av_get_pix_fmt("p010le") = 158` |

The exit codes of the three builds were not captured into an artefact (they were run by hand before
the gate); the log scan above is what is recorded. The probe run, both demuxes, all eleven ffprobe
calls and the child probe **are** captured with their exit codes in `p010_nvenc_gate.json` under
`commands` (11 entries, every `rc = 0`).

`nvdec_bridge.dll` in the tree did **not** export `nvdec_bridge_open_hdr` /
`nvdec_bridge_get_hdr_metadata` — proved by a byte scan of the checked-in file. It was preserved as
`bench_out/hdr_p010_nvenc/nvdec_bridge.prev.dll` (`77bac28f…`, 154 624 bytes) and rebuilt from the
ABI 3 source (`263c4397…`, 155 648 bytes). Nothing was committed, reset or cleaned.

## 8. Files

SHA-256 as recorded by the gate itself, which hashes the DLLs, the sources and its own script
(`binaries`, 9 entries), the fixtures (`fixtures`, 2) and every file in its output directory
(`files`, 26 entries) — so the record names the bytes that produced it rather than a transcription
of them. The inventory is not recursive: `bench_out/hdr_p010_nvenc/scratch/` holds two throwaway
probe scripts written while building this gate and is deliberately absent from it.

| file | bytes | sha256 |
| --- | --- | --- |
| `native/nvenc_bridge.cpp` | 75 373 | `a2dbe7f5956c0981308c9d63b5bc1966b8f7d06437cf2cd454ffeb75af49f147` |
| `native/nvenc_bridge.h` | 15 343 | `68d39446768f92e4a15429c4fc3d84fad982c60f66acd51d1df45db4b4d3471c` |
| `native/nvenc_hdr_probe.cpp` | 10 409 | `c29c7422d9056b12445afa03cceafa3e21b97ca4dfe127dafb109228694c7ca0` |
| `native/nvenc_hdr_p010_smoke.py` | 74 681 | `8c3069796ecb3ce54c34ea7c2b57b6f99f3a9f3fb9e98722556d32f0e546fccd` |
| `native/build_nvenc_hdr_probe.bat` | 505 | `89573cbfe3ef6cee7c52f79fd8d45bccffd87e36999d39d86e73cbacaa143dff` |
| `native/nvenc_bridge.dll` | 199 168 | `4e347275ef1274e59bceb665074c2593ee29a788b6ee0cdd2c760ec0215b4911` |
| `native/nvdec_bridge.dll` (rebuilt) | 155 648 | `263c4397876e729fd964e1457c2bc5365de3032c77ba481961af69b85f801ea0` |
| `…/nvdec_bridge.prev.dll` (checked-in) | 154 624 | `77bac28ff6d290eac514449e2e2fb32b3a30f68c4d7a4680fca4cae4d80a7803` |
| `native/nvenc_hdr_probe.exe` | 154 624 | `54f115d4c7b93cb7beae8dddd4950b9afc90637184a43ebfe0707df18db8e84a` |
| `build/hdr_probe/hdr_src.mkv` (fixture) | 148 007 | `48697ed8deb0f10475151dbc0b9312d3408b513087f2c3f76be6e7d12ee83b8b` |
| `build/hdr_probe/ctrl.mkv` (no-metadata control) | 23 525 | `8b09d067fd962489adc211eb14960aeb6be4b0d271b48f0e3075ebe709aca5fb` |
| `a_hdr10_out.mkv` | 74 527 | `b820d3a2661883ac4a98617706b67ee7c7146bf6191185bf3595d4be5a7d10af` |
| `a_hdr10_out.hevc` (demuxed `-c copy`) | 73 798 | `750990764f378eea6c40919f36577c96ac3b19444fa98a52c94fc1f201dea1d4` |
| `a_src_copy.hevc` (positive control) | 149 848 | `4e3402cf45188338fcd661e1f9ec0389599090b83e51c401bc855132cd4adbb6` |
| `b_no_metadata_out.mkv` | 9 031 | `19af58d2091cfda6dfb99ab77b44caf7183692ab32132b5500892bf907f43ca8` |
| `nvenc_hdr_options.txt` | 1 053 | `e7a3b865dfa8cba9e11babd27cd58bbb518a9daef0a0609ee1b85cea88a121bb` |
| `p010_nvenc_gate.json` | 66 789 | `4687317d536bab7f219ab24e16a8d8772c9efef67b0bf9b93e09854f142337db` |

Two things worth knowing about those hashes. The two `.mkv` outputs change on every run while their
sizes do not: Matroska writes its own random segment/track UIDs and a date, and NVENC's rate
allocation is not bit-exact run to run — the table above is the fourth run, and both MKV digests
moved while nothing else did. The **Annex-B** file is identical in every run of this gate
(`75099076…`, including both demuxes inside a single run), so the reproducible artefact is the coded
video, not the container — which is also why the bitstream evidence in §5 is taken from it.

The fixture and the CPU decode that produced it are **authoring and verification, outside the
measured path**: the fixture already existed (`build/hdr_probe`, from the Stage A/A2 work), this job
only read it. Reported separately for that reason: no CPU pixel work happened between the borrowed
P010 planes and the encoder.

## 9. Open items

1. **No in-bitstream ST 2086/CTA-861.3 SEI from NVENC on this build** (§5). Decide whether the
   target player needs it. If it does, the only installed routes are a post-mux SEI writer or a
   different encoder — a separate decision, not a quiet fallback.
2. Nothing is wired into production: Python export, the FP16→P010 conversion stage and the DLSS
   chain are untouched by this job, so the metadata here came from a fixture, not from the measured
   chain.
3. `nvenc_encode_rgba` on an HDR bridge refuses with an empty error string (§6).
4. `gop = 2` is refused by NVENC itself (*"Gop Length should be greater than number of B frames +
   1"*, `avcodec_open2` → −22). The gate leaves `gop` unset and records this as an accepted/rejected
   option data point; a production HDR encode with a short GOP needs the wrapper's B-frame count
   addressed too, which is a real pipeline question, not this boundary.
5. The name-based metadata-option probe can only disprove the eleven spellings it asks about (§5).

## 10. Next step

1. **Python HDR export integration**: feed `nvenc_hdr_open`/`nvenc_encode_p010` from the production
   job, with the metadata taken from the measured FP16 chain's own reference values (and the real
   loss record in `HDR_STAGE_B_ROUNDTRIP_FIX.md` carried along, not hidden), behind an explicit
   opt-in that cannot silently become SDR.
2. **One 4–6 frame full production gate**: DLSS → resample → fuse → convert → P010 encode, keeping
   every host-pixel and RGB-conversion counter at zero and the existing colour reference gates
   unchanged. This stage deliberately stopped at two frames and one boundary.
3. **Experimental UI toggle** only after gate 2 passes; the formal Feature 18 HDR contract stays
   unproven until then, and no quality or performance claim is made from this job.

Reruns, when needed: `python native/nvenc_hdr_p010_smoke.py` (gate),
`native/build_nvenc_hdr_probe.bat` + `nvenc_hdr_probe.exe <dump-path>` (ABI/capacity probe),
`native/build_nvenc_bridge.bat` / `native/build_nvdec_bridge.bat` (DLLs).
