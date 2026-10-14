# HDR Stage B — in-bitstream HDR10 static SEI repair (ST 2086 / CTA-861.3 in the elementary HEVC stream)

Bounded follow-up to `qbg_20260930_c133ab69` and to `HDR_STAGE_B_P010_NVENC.md`.
Scope: close the required-but-missing in-bitstream HDR10 static metadata SEI at the
encoder packet boundary. No Python HDR export, no UI, no benchmark, no
NVENC/NVDEC/NVOF redesign.

Evidence directory: `bench_out/hdr_sei_repair/` (fresh). Machine-readable companion:
`bench_out/hdr_sei_repair/hdr_sei_repair_evidence.json`.

## Verdict

**The gap is closed for the two-frame boundary, and the gate that used to be green on
its own absence is now red on absence.**

| Question | Measured answer |
|---|---|
| Does the demuxed elementary HEVC stream carry ST 2086 payloadType 137? | Yes — 2 NALs, one per access unit, values equal to the source as exact `Fraction`s (`mastering_value_agreement.exact_equal = true`) |
| Does it carry CTA-861.3 payloadType 144? | Yes — MaxCLL 1000, MaxFALL 400, on both frames |
| Read without any Matroska metadata? | Yes — `bench_out/hdr_sei_repair/a_hdr10_out.hevc` (stream copy out of the mkv) is probed as a standalone Annex B elementary stream and an independent decoder derives both payloads from the bitstream |
| Did the VCL change? | No — `vcl_hash` in the demuxed file `17542399068109509777` equals the bridge's own session fold, `vcl_bytes` 73658, and the coded slices are byte-equal to the previous stage's baseline |
| Was any pixel downloaded? | No — `host_pixel_bytes = 0`, `rgb_conversion_bytes = 0`, `staging_gpu_copy_bytes = 0`, `conversion_output_bytes = 0`; the writer reads compressed host bytes only |
| Does absence stay absence? | Yes — the no-metadata source produces 0 mastering and 0 content-light SEI blocks, `metadata_written_to = 8` (colour tags only), writer status `packets_seen = 0` |
| Is there a public route we skipped? | No — measured, see §1 |
| Gate status | `native/nvenc_hdr_p010_smoke.py`: `PASS`, 136 checks, 0 failed, exit 0. `native/nvenc_hdr_sei_cpu_gate.py`: `PASS`, 58 checks, 0 failed, exit 0. Build `/W4 /WX`: exit 0, zero diagnostics |

**Goal completeness is a different statement from the checks above, and is kept
separate in §9.** This stage proves the *mechanism and the contract at the packet
boundary*. It does not prove the formal Feature18 HDR contract, does not characterise
performance (`NOT_CHARACTERIZED` — no benchmark was run by design), and is not a
completed production export.

## 1. Why the writer is in-project: the public routes, measured

The brief said not to assume `hevc_metadata` can do this. It cannot, and that is now a
measurement rather than a guess (`bench_out/hdr_sei_repair/public_bsf_capability_probe.txt`):

* FFmpeg `7.1.1`, `libavcodec 61.19.101 / 61.19.101`.
* `ffmpeg -bsfs` lists no filter whose name mentions SEI.
* `hevc_metadata` (`h265_metadata_bsf AVOptions`) exposes exactly: `aud`,
  `sample_aspect_ratio`, `video_format`, `video_full_range_flag`, `colour_primaries`,
  `transfer_characteristics`, `matrix_coefficients`, `chroma_sample_loc_type`,
  `tick_rate`, `num_ticks_poc_diff_one`, `crop_left/right/top/bottom`, `width`,
  `height`, `level`. Every one of those edits a **parameter set / VUI field**; none
  takes an SEI payload, a payloadType, a chromaticity or a luminance.
* `hevc_nvenc`'s own dumped option set (`bench_out/hdr_sei_repair/nvenc_hdr_options.txt`,
  previously captured) contains no mastering or content-light option:
  `encoder_hdr_metadata_option_count = 0`.

That is the whole public surface that could have carried the values: the frame side
data and `coded_side_data` reach the muxer (and therefore the Matroska elements and the
decoded-frame side data), and nothing forwards them into the elementary stream. So the
repair is a **compressed-packet metadata authorer** at the receive-packet-before-mux
boundary — no pixel download, no re-encode, no container-only workaround.

## 2. Standards basis (consulted, not copied)

Written from the format specifications. The reference files kept for reading are
`scratch/cbs_sei_syntax_template.c`, `scratch/sei.h`, `scratch/h2645_sei_ref.txt`
(FFmpeg/LGPL syntax templates and the H.264/H.265 SEI tables). Nothing was copied from
them into `native/nvenc_bridge.cpp`, and nothing was taken from any GPL Magpie source.

The numbers the writer is built on, and the numbers it publishes through
`nvenc_hdr_sei_capability` so they can be checked rather than trusted:

| Fact | Value | Where it is enforced |
|---|---|---|
| `mastering_display_colour_volume` payloadType | 137 | `sei_mastering_wire`, capability |
| its payloadSize | 24 bytes | same |
| field order | 8 × u(16) chromaticities, then u(32) **max** then u(32) **min** luminance | `sei_mastering_wire` (ten integers, not twelve) |
| chromaticity wire order | **green, blue, red**, then white point | `kSeiWireToStructIndex = {1, 2, 0}` (`native/nvenc_bridge.cpp:963`) |
| chromaticity grid | 1/50000 | `sei_to_wire` |
| luminance grid | 1/10000 cd/m² | `sei_to_wire` |
| accepted ranges | chromaticity 0..50000 mapped value, luminance 50000..100000000 (i.e. 5..10000 cd/m²) | `sei_to_wire` bounds |
| `content_light_level_info` payloadType / size | 144 / 4 bytes (MaxCLL, MaxFALL u(16)) | `sei_content_light_wire` |
| prefix SEI `nal_unit_type` | 39 | NAL header `4e 01` |
| `nuh_layer_id`, `nuh_temporal_id_plus1` | 0, 1 | NAL header `4e 01` |
| VCL range | types 0..31 (insertion point = first VCL) | `sei_parse_annexb`, `sei_apply_bytes` |
| emulation prevention | `00 00 03` after two zero bytes | escape pass |
| `rbsp_trailing_bits()` | single 1 bit then zeros (0x80 for a byte-aligned payload) | payload tail |

`rbsp_trailing_bits` cannot be recognised by testing the high bit of the last RBSP byte
alone — 144 is `0x90` and has it set — which is why both the writer and the independent
Python reader treat it as "a lone `0x80` with an all-zero remainder".

## 3. The writer, precisely

`native/nvenc_bridge.cpp`, SEI section (anonymous namespace from line 803;
`kSeiWireToStructIndex` 963, `sei_to_wire` 988, `sei_mastering_wire` 1098,
`sei_parse_annexb` 1215, `sei_apply_bytes` 1362, `hdr_sei_prepare` 1571,
`hdr_sei_apply` 1636). The single hook is in the shared `write_packet`:

```
native/nvenc_bridge.cpp:413
    if (b->hdr_mode && b->hdr_sei_wanted && hdr_sei_apply(b) < 0) return -1;
```

* **When it runs.** After the encoder hands back a packet and before the muxer sees it.
  An SDR bridge has `hdr_mode` false; an HDR bridge whose source declared nothing has
  `hdr_sei_wanted` false. Neither enters the branch, and both take exactly the bytes
  they took before.
* **What it builds.** One prefix SEI NAL per access unit, prepared once at the open
  (`hdr_sei_prepare`): payloadType 137 (`sei_message` of 26 bytes: payloadType,
  payloadSize 24, then the 24 payload bytes) plus payloadType 144 (6 bytes), one
  `rbsp_trailing_bits()` for the whole unit — RBSP 33 bytes, escaped to 34, in a type-39
  NAL of **36 bytes**; with a 4-byte start code that is
  **40 bytes per access unit**, **80 bytes** for the two-AU stream.
* **Where it goes.** Immediately before the *first VCL NAL* of that access unit. The
  AU's own framing is detected per packet, not assumed: `framing = 1`
  (`NVENC_HDR_SEI_FRAMING_ANNEXB4`) for the real NVENC packets, and the inserted NAL is
  emitted in that framing. A length-prefixed packet is rewritten as a length-prefixed
  packet (`framing = 3`, LP4) rather than being handed Annex B bytes, and a packet that
  is neither is refused with `framing = 6` and the note "refusing to guess a framing". A
  4-byte start code's extra zero lives at `code - 1` and the NAL body always begins at
  `code + 3`, which is what `sei_parse_annexb` relies on.
* **What is preserved.** Every NAL the packet already had — an AUD if one is present,
  parameter sets, other SEI NALs, all VCL bytes — stays in place and in order; only the
  authored NAL is added. The encoder's own prefix SEI survives
  untouched — the first real packet begins `00000001 4e01010300000a80 00000001 2601…`
  (`sei_status.packet_sample`), i.e. an existing type-39 NAL carrying payloadType 1
  (8 bytes), then the IDR slice; the writer adds a second type-39 NAL with 137/144 and
  leaves that one alone.
* **How the packet is swapped.** A new `AVBufferRef` from `av_buffer_alloc(size +
  AV_INPUT_BUFFER_PADDING_SIZE)`, memcpy of the authored bytes, padding zeroed, then the
  ref is installed on the `AVPacket`. `pts`, `dts`, `duration`, `flags` and all side
  data are untouched; the packet count is unchanged. A packet whose buffer is not
  reference-counted is **refused** rather than rewritten in place.
* **No extradata change.** The codec's `extradata` / container configuration record is
  not modified. In this stream the VPS/SPS/PPS come from the container configuration
  record and are re-inserted by the demuxer, which is why the writer's session fold of
  "received" NAL bytes excludes them and the gate asserts their equality separately.

**Placement rule, stated once because the brief asked for it to be documented:** an SEI
NAL unit with `nal_unit_type` 39 must lie in the same access unit as the picture it
refers to and must precede the **first VCL NAL unit** of that picture. A *static*
declaration therefore repeats per picture rather than being sent once. That is the
standard's rule, not a workaround, and it is why `mastering_inserted = 2` and
`content_light_inserted = 2` for two coded frames.

## 4. Values, conflicts, refusals

* **Exact rational arithmetic, no floats.** `sei_to_wire` scales the declared numerator
  onto the grid with checked integer arithmetic and refuses when the result is not an
  exact multiple of the grid, so no declared value is ever rounded into the bitstream.
* **Has-flags respected.** No mastering declaration and no content-light declaration →
  nothing authored. Both → both. Content light alone → content light alone.
* **Partial mastering is not invented around.** If primaries are declared without
  luminance (or the reverse), the ST 2086 payload has no way to carry half a
  declaration, so the mastering SEI is **absent**, the content-light SEI is still
  authored, `nvenc_hdr_open` accepts the stream, and the writer status reports
  `error = 2` (`NVENC_HDR_SEI_PARTIAL_MASTERING`) with `mastering_serialized = 0`.
  The container values are retained; the required-SEI contract is **not** marked met.
  `nvenc_hdr_sei_capability.supports_partial_mastering = 0` says this before any run.
  *This was a real defect found by the new test*: `hdr_sei_prepare` previously
  overwrote `PARTIAL_MASTERING` with `OK` as soon as the content-light half succeeded.
* **Non-representable complete declarations refuse the open, and nothing is written.**
  Measured (`sei_metadata_cases` in the gate artifact):

  | Case | Result |
  |---|---|
  | white point `15635/3` — not on the 1/50000 grid | `nvenc_hdr_open` rc `-1`, no bridge, **no file written**, message names the grid |
  | max luminance `600000000/10000` — beyond the accepted range | rc `-1`, no file, names `outside 50000..100000000` |
  | min luminance denominator 0 | rc `-1`, no file, names the zero denominator |
  | primaries declared, luminance not | rc `0`, file written, `payload_types_in_stream = [1, 144]` (1 = the encoder's own prefix SEI, which is preserved), `error = 2`, `metadata_written_to = 31` |

* **Duplicate and conflict policy.** A pre-existing equal payload in the packet is
  preserved and counted (`mastering_already_present` / `content_light_already_present`);
  a pre-existing *differing* payload is a hard `CONFLICT` refusal (error 5).
  Contradictory declarations are never merged. Both conflict branches are measured by
  the CPU gate — *"a conflicting content-light declaration already in the AU is refused,
  never overwritten or duplicated"* and *"a conflicting mastering declaration already in
  the AU is refused"* — and in the live run both already-present counters are 0, because
  NVENC emits only payloadType 1.

## 5. Byte-exactness, proved by two independent implementations

`native/nvenc_hdr_sei_cpu_gate.py` (58 checks, PASS, exit 0) is CPU-only and needs no
encoder: it computes the expected bytes independently from the Annex D tables and
compares them with the implemented entries, then runs the same entries over the real
NVENC bytes of the preserved baseline stream.

```
nal_hex  = 4e01 | 89 18 | 33c2 86c4 6784 2710 84d0 3e80 3d13 4042 | 00989680 | 0000 03 03e8 | 90 04 | 03e8 0190 | 80
             hdr   137/24   G.x   G.y   B.x   B.y   R.x   R.y   Wx    Wy        max           min(escaped)  144/4  CLL FALL  trail

nal_bytes = 36, rbsp_bytes_unescaped = 33, rbsp_bytes_escaped = 34, insertions = 1
wire_integers_expected = [13250, 34500, 26500, 10000, 34000, 16000, 15635, 16450, 10000000, 1000]
values_read_from_wire  = green 53/200,69/100 · blue 53/100,1/5 · red 17/25,8/25 ·
                         white 3127/10000,329/1000 · max 1000 · min 1/10 · CLL 1000 · FALL 400
```

Note the layout itself: ten integers, not twelve — eight u(16) chromaticities, then two
u(32) luminances in **max-then-min** order, and the `rbsp_trailing_bits()` belongs to the
NAL unit, not to each `sei_message()`.
The `00 00 03 03` in the middle of that hex is one emulation-prevention insertion: the
min luminance 1/10 cd/m² is the wire integer 1000 = `0x000003E8`, and its leading
`00 00 03` is exactly the pattern Annex B.1.3 forbids inside a NAL, so a `03` is
inserted after the two zeros. The
escape path is therefore exercised by the real declared numbers — 33 RBSP bytes in, 34
out, one insertion — not by a synthetic one. The reader refuses to be fooled in the
other direction too: `00 00 03` inside escaped data is legal, so the forbidden set
checked is `00 00 00/01/02`.

Framing case (`packet_annexb`, a synthetic AU in Annex B with the encoder's own SEI and
parameter sets present): `types_before = [39, 32, 33, 34, 19]` → `types_after =
[39, 32, 33, 34, 39, 19]`, `vcl_index = 4`, 40 bytes inserted, framing 1 preserved.
Real-packet case: 73798 in → 73878 out (+80), `vcl_hash_in == vcl_hash_out ==
17542399068109509777`. The CPU-authored file `sei_cpu_authored.hevc`
(`691cba4ac72ddacf…`) and the native-gate output `a_hdr10_out.hevc`
(`615a34cd882a92a3…`) are the same 73878 bytes containing the **same nine NAL bodies**:
compared as multisets they are equal, and the two authored 36-byte SEI NALs are
byte-identical between the two paths. They differ only in *order*, and the order is
inherited from their different inputs, not produced by the writer: the CPU path authors
into the previous stage's demuxed file, where the demuxer had already placed
VPS/SPS/PPS right after the encoder's own prefix SEI, so the new NAL lands after the
parameter sets (`39, 32, 33, 34, 39, 19, 39, 39, 1`); the native path authors into the
live encoder packet, which carried only the encoder's prefix SEI plus the slice, and the
demuxer re-inserts the parameter sets afterwards (`39, 39, 32, 33, 34, 19, 39, 39, 1`).
Both put the authored SEI before the first VCL NAL of its access unit, which is the rule
the standard actually states, and both files are read back by the installed decoder with
the same mastering and content-light blocks — measured, not argued.

The two gates were run separately and each wrote its own JSON. The CPU gate artifact was
regenerated once more after correcting a misnomer in two of its check names
("picture-timing SEI" → "payloadType 1 SEI": payload type 1 is buffering period, 2 is
picture timing). The authored bytes and all 58 results are unchanged by that edit, which
is itself the determinism check.

## 6. The native two-frame boundary, and the required gate

`native/nvenc_hdr_p010_smoke.py` (136 checks, PASS, exit 0) writes the fresh directory.
The `REQUIRED bitstream` group is the point of this stage:

```
PASS REQUIRED bitstream: the encoded elementary stream carries a mastering display SEI on every coded frame
PASS REQUIRED bitstream: it carries a content light level SEI on every coded frame
PASS REQUIRED bitstream: the in-stream mastering values equal the source's, as exact rationals, with no rounding
PASS REQUIRED bitstream: the in-stream MaxCLL and MaxFALL equal the source's
PASS REQUIRED bitstream: the prefix SEI NAL units the decoder read are type 39 and carry payload types 137 and 144
PASS REQUIRED bitstream: absence stayed absence - a source that declared nothing gains no mastering or content-light SEI in the elementary stream
PASS REQUIRED bitstream: the no-metadata stream still decodes as Main 10 to its two coded frames
```

Before this stage the same probe read **0 mastering / 0 content-light blocks** from the
encoded elementary stream while the positive control (a source copied straight to
`.hevc`) read **6 / 6**. The previous stage's artifact is preserved unchanged at
`bench_out/hdr_p010_nvenc/` and still shows `status: PASS` with 106 checks including a
check named *"the encoded elementary stream carries no mastering or content-light SEI"*
— that is the mis-specification the brief pointed at, kept as a factual artifact and
**not** accepted as stage completion. The new gate turns that observation into a
requirement. The negative control is recorded durably rather than only remembered: the
CPU gate's own `baseline_probe` runs the same ffprobe form over the SEI-less baseline and
reports `frames: 2, kinds: []` — no metadata block of any kind — so the `REQUIRED
bitstream` group has been measured red on a stream without the SEI and green with it.

Carriage, now measured at five levels rather than four: frame side data on every
submitted frame, `AVCodecContext::coded_side_data` before the open,
`AVStream::codecpar->coded_side_data` after `avcodec_parameters_from_context`, the
Matroska elements the muxer derives, **and the elementary stream itself** — one
bridge-authored prefix SEI NAL per access unit.

Counters and boundaries from the run (`case_a.encoder_counters`):
`host_pixel_bytes 0`, `rgb_conversion_bytes 0`, `conversion_output_bytes 0`,
`staging_gpu_copy_bytes 0`, `p010_device_copy_bytes 12441600` (device-only),
`packets_written 2`, `frames_submitted 2`, `metadata_written_to 31` — which decodes as
**8** container colour tags declared | **1** mastering/content-light attached per frame |
**2** encoder `coded_side_data` | **4** stream `codecpar->coded_side_data` | **16**
in-bitstream SEI authored. The no-metadata case reports **8**, and its check asserts
`metadata_written_to & 23 == 0`, i.e. colour tags yes, metadata carriage nowhere.
Decoder `host_frame_copy_bytes 0`.
Writer status: `packets_seen 2`, `packets_without_vcl 0`, `nals_inserted 2`,
`bytes_inserted 80`, `framing 1`, `error 0`, `vcl_bytes 73658`,
`existing_nal_bytes 73674`, `existing_nal_hash 366683077645557683` — all recomputable
from the demuxed stream, which the gate does.

Timing and shape are unchanged: 2 packets / 2 frames, container packet pts `[0, 40]`,
decoded frame pts `[0, 40]`, submitted source pts `[0, 40]` with time_base `1/1000`.
The *elementary* stream reports frame pts `[None, None]` — that is a property of a raw
HEVC stream with no container timing, and the previous stage's artifact reports the same
`[None, None]` for the same reason, so it is not evidence of a lost timestamp. The
independent decoder still sees `I` then `P` in presentation order.

SDR path: one shared line changed (§3), so the existing SDR encoder regression was
re-run against this DLL. `native/nvenc_gpu_smoke.py` printed `nvenc_gpu_smoke=PASS`
with 15 frames at 3840×2160 av1 and `bridge_host_pixel_bytes=0`, exit code **0**,
verbatim in `bench_out/hdr_sei_repair/raw_sdr_regression_nvenc_gpu_smoke.txt`. It runs
as its own command, so the gate reports it as a separately measured run.

## 7. Build, ABI, and what was *not* moved

```
native/build_nvenc_bridge.bat   cl /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /wd4244 /utf-8 …   BUILD_EXIT=0
native/build_nvenc_hdr_probe.bat                                                   PROBE_EXIT=0
```

Both are recorded with their exit codes in `build_nvenc_bridge_verify.txt` and
`build_nvenc_hdr_probe_verify.txt`. `/W4 /WX` means any warning is an error, and the
logs contain no diagnostics. Honest detail: the two native gates ran against the link
made at 00:19 (`nvenc_bridge.dll` sha256 `f2a482653cf30669…`); the builds above were
relaunched to record exit codes durably, and because a PE link embeds its timestamp the
rebuilt DLL hashes differently (`00787148984890af…`, same 218624 bytes, same sources —
`native/nvenc_bridge.cpp` last modified 00:16 — same flags). Re-running the CPU gate
after the relink reproduces `sei_cpu_authored.hevc` byte-for-byte
(`691cba4ac72ddacf…`) and changes only the DLL-hash field it records, which confirms the
authored bytes do not depend on which link was loaded.

Struct sizes stay pinned, and no established struct moved:
`capability 128`, `contract 268`, `counters 64`, `defaults 56`, `metadata 108`,
`options 112`; the three new structs are appended: `NvencHdrSeiCapability 56`,
`NvencHdrSeiInfo 244`, `NvencHdrSeiStatus 384`. ABIs: `nvenc 2`, `nvenc_hdr 1`,
`nvdec 3`, and the new writer answers `nvenc_hdr_sei_abi_version() == 1`. Pinned twice —
compile-time asserts in `native/nvenc_hdr_probe.cpp` and a ctypes re-pin in the smoke
gate (`binding_sizes`).

Capability is now **split**, which was its own item: the encoder's hardware Main10
availability and the writer's availability are reported separately, and
`encoder_offers_metadata_option = 0` no longer implies metadata is impossible — it is
precisely the fact that made the writer necessary.

Diff against HEAD, measured now: `native/nvenc_bridge.cpp` **+1761 / −3**,
`native/nvenc_bridge.h` **+370 / −0**, all of it in the HDR10/SEI sections or the new
struct declarations appended after them. (The narrative string recorded inside
`p010_nvenc_gate.json` says 1758/3; it was written three lines before the
session-fold/partial-status fix, so the measured number above supersedes it.)

## 8. Artifacts

`bench_out/hdr_sei_repair/` — `a_hdr10_out.mkv` / `a_hdr10_out.hevc` (73878 B, sha
`615a34cd882a92a3…`), `b_no_metadata_out.mkv` / `.hevc`, `sei_partial_mastering.hevc`,
`sei_primaries_without_luminance.mkv`, `a_src_copy.hevc` (positive control, 6+6 blocks),
`sei_cpu_authored.hevc`, `p010_nvenc_gate.json` (136 checks), `sei_cpu_gate.json`
(58 checks), the ffprobe JSONs, `nvenc_hdr_options.txt`,
`public_bsf_capability_probe.txt`, the two build logs and the SDR regression log.
Full sizes and sha256 values in `hdr_sei_repair_evidence.json`.

Prior artifacts preserved, not rewritten: `bench_out/hdr_p010_nvenc/` (baseline
`a_hdr10_out.hevc`, 73798 B, sha `750990764f378eea…`) and the two
`preserved_20260924_*.mkv`.

The strongest single statement available about preservation, recomputed by
`scratch/_hdr_sei_repair_evidence_json.py` from the files on disk: the two inserted spans
in `a_hdr10_out.hevc` sit at offsets **12..52** and **48859..48899** (40 bytes each: a
4-byte start code plus the 36-byte NAL). Deleting exactly those two spans leaves 73798
bytes that are **identical to the previous stage's elementary stream** — same length,
same bytes. Nothing else in the stream changed, not one byte.

## 9. Measured vs claimed — goal completeness

**Measured and passing:** serializer byte-exactness against an independent
implementation; framing detection on real NVENC packets; per-AU prefix SEI placement
before the first VCL with all other NALs retained; both payloads present in a standalone
demuxed elementary stream and equal to the source as exact rationals; VCL hash and byte
equality against the previous stage's stream; unchanged packet/PTS/flags behaviour;
absence behaviour; four refusal classes with the reason named and no file written on
the hard ones; partial-mastering honesty; SDR regression; ABI pins; `/W4 /WX` builds.

**Not measured, and therefore not claimed:** a full multi-hundred-frame HDR export;
integration with the Python HDR export path or any UI (explicitly out of scope here);
the formal Feature18 HDR contract on a real DLSS-generated HDR frame chain; player or
TV behaviour; any performance characteristic — **Performance:
`NOT_CHARACTERIZED`, no benchmark was run**; HDR10+ dynamic metadata, which is a
different payload family and was not attempted; verification of what a downstream
consumer does with per-picture repetition beyond FFmpeg's own decoder.

**Remaining limits.** The writer handles the two static HDR10 payloads and nothing
else; other SEI payloadTypes are preserved but never authored. Four framings are
distinguished in code — ANNEXB4, ANNEXB3, LP4 and "neither" — and the CPU gate measures
ANNEXB4 on real encoder bytes, LP4 on a synthetic length-prefixed packet, and the
"neither" refusal; **ANNEXB3 is implemented and reported but has no dedicated case**, and
this encoder session only ever produced ANNEXB4 packets. The `00 00 03` case is covered
by the real declared numbers here; a payload needing two escapes would use the same loop
but was not exercised by a second stream. Colour-range/primaries tags still travel by the
codec context, unchanged by this stage.

## 10. Next step

One step, and it is the one the brief named: wire the proven boundary into the real
product — the Python HDR export path plus the experimental UI switch — and then do
**one** final real 4–6 frame HDR export through that path, demuxed and probed with the
same required checks. Nothing about the mechanism needs another experiment first: the
writer is byte-exact, refusal-complete and VCL-preserving at the packet boundary, and
the required gate is now red when the bitstream SEI is missing.
