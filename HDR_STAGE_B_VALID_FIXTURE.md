# HDR Stage B — a physically valid HDR10 fixture, and the native bridge admitted at 250 scRGB

Workspace `F:\dlss5standaloneV2`, continuing from `HDR_STAGE_B_REAL_ADMISSION.md`.

That document closed with the bridge admitting one real decoder frame and refusing
BT.2020 constant-luminance on the same frame — but the clip it proved that on,
`build/hdr_probe/hdr_src.mkv`, was admitted inside a declared domain of **10000 scRGB**.
Its own recorded numbers say why that is not a colour proof: at a declared 250 scRGB the
native report measured `min_component −614.1239013671875`, `max_component 9420.1640625`
and **629,189** components above the ceiling (`evidence_attempt2_ceiling250_failed.json`),
and `HDR_STAGE_B_REAL_ADMISSION.md` works that peak back to a coded PQ input of
**1.39762** — past BT.2100-2's top of 1.0, on the asymptotic branch that runs away toward
the pole at 1.9921. ITU-R BT.2100-2 defines the PQ component E′ as living in [0, 1] with
normalised linear in [0, 1], where 1.0 is 10000 cd/m² — 125 scRGB at this bridge's
80 cd/m² reference white. So `hdr_src.mkv` is *tagged* HDR10 without being an HDR10
colour-domain sample, and widening its ceiling to 10000 scRGB proved only that the
bridge does not crash on a source that lies about itself.

**The old fixture's 10000 scRGB admission is not HDR colour verification, and this task
does not re-open it.** Its evidence stays exactly as recorded, including both failed
ceiling-250 attempts.

This task delivers the missing sample instead: a clip authored in cd/m², verified in the
physical domain against a decoder that is not the product's, and admitted by the native
bridge at a conservative **250 scRGB** in one attempt, with no ceiling widening.

## Result

| Question | Answer |
| --- | --- |
| Is there a fixture whose decoded samples are actually inside the BT.2100 PQ domain? | **YES.** An independent software decode of frame 0 reconstructs PQ components in **[0.313927, 0.740868]** — inside [0, 1] with room on both sides, 0 components outside, 0 non-finite. Normalised linear [0.001195, 0.090432], i.e. **11.95 to 904.32 cd/m²**, and scRGB **[−4.5197, 13.9930]**. |
| Did the lossless coder preserve the authored samples? | **EXACTLY.** 0 mismatching samples out of 2,073,600 luma + 518,400 Cb + 518,400 Cr; max abs code error 0 in all three planes. The decoded frame's bytes hash to the authored frame's hash — `ddd225cf30e75557…`, identical. |
| Does the native bridge admit that clip at the conservative domain? | **YES, first attempt.** `gpu_hdr_decode` returned success on the decoder's own `range 1 primaries 9 transfer 16 space 9` with `support_ceiling_scrgb 250.0`, and its source report measured **min −4.5198, max 13.9928 scRGB**, `above_ceiling_components == 0`, `nonfinite_samples == 0`. |
| Does it still refuse constant-luminance on the new clip? | **YES.** Same live frame, only `color_space` set to 10: refused, rc −1, *"HDR decode wants limited-range BT.2020 NCL with ST 2084, got range 1 primaries 9 transfer 16 space 10"*. |
| Do the device and the CPU reference agree per component? | **YES, exactly.** Both count **221,184** negative scRGB components on the same 2,073,600 pixels; min/max agree to 1.3e-4 / 1.6e-4, which is FP16 output resolution, not a domain difference. |
| Verdicts | offline physical check **PASS (30 checks, 0 failed)**; consolidated physical + native **PASS (34 checks, 0 failed)**; native admission gate **PASS (42 checks, 0 failed)**. |

No pixel image reached the host in the native run: decoder
`host_frame_copy_count == 0` / `host_frame_copy_bytes == 0`, bridge
`report_readback_bytes == 0`, and no `gpu_hdr_readback_*` export is bound in that process.
No Feature 18 call, no NVOF, no NVENC, no export, no UI, no orchestrator. Nothing in the
native algorithm, the FP16 path or the Stage A loss record changed.

## The scene, and why 250 scRGB is a bound rather than a guess

`native/make_hdr10_valid_fixture.py` authored 1920×1080, 6 frames, 25 fps, libx265
**lossless Main 10**, `yuv420p10le`, limited range, bt2020 / smpte2084 / bt2020nc. The
scene is known linear BT.2020 RGB in cd/m² before any code exists: 203 cd/m² surround, an
8-step grey stair 12 → 900 cd/m², a 256-px logarithmic luma ramp 12 → 900 cd/m² that
shifts 8 px (four chroma boxes) per frame, and six 192×192 colour patches whose dominant
channel steps 700 → 660 → 620 → 700 → 660 → 620 cd/m² against 60 cd/m² minors. Every
patch edge is 2×2 aligned, so no chroma box ever straddles two patches and the sample the
product reads is the sample that was authored.

The ceiling comes from the contract, not from the clip's metadata:

| Quantity | Value | Where it comes from |
| --- | --- | --- |
| PQ top in this bridge's units | **125 scRGB** | 10000 cd/m² ÷ 80 cd/m² reference white |
| Derived Rec.2020 → Rec.709 positive row sums | 1.66020, 1.13286, 1.11685 | primaries 0.708/0.292, 0.170/0.795, 0.131/0.046 at D65, re-derived inside the verifier and agreed with the manifest to < 1e-12 |
| Legal peak after the primaries conversion | **207.53 scRGB** | 1.66020 × 125 |
| Declared support ceiling | **250 scRGB** | conservative rounding above the derived legal bound |
| This clip's measured peak | 13.99 scRGB | native report *and* CPU reference, independently |
| MaxCLL equivalent (deliberately **not** used) | 11.25 scRGB | 900 cd/m² ÷ 80; using it would make the above-ceiling check circular |

So the gate's ceiling check and the "not the MaxCLL equivalent" check both hold with the
margin measured: 250 > 207.53 (legal content can never reach it) > 13.99 (what this clip
actually carries). Negative components are legal signed wide-gamut values; 221,184 of
them are reported and treated as data, never as a failure.

## Commands actually run

| Command | Exit | Note |
| --- | --- | --- |
| `python native/hdr_real_admission_gate.py --source build/hdr_probe/hdr_valid_src.mkv --support-ceiling 250.0 --output-dir bench_out/hdr_valid_fixture/admission` | **0** | the only native run; PASS 42/42, console kept in `…/admission/gate_stdout.txt` |
| `python native/verify_hdr10_valid_fixture.py` | **0** | offline physical check → `physical_check.json`, console in `verify_stdout.txt` |
| `python native/verify_hdr10_valid_fixture.py --native-evidence bench_out/hdr_valid_fixture/admission/evidence.json --output bench_out/hdr_valid_fixture/evidence.json` | **0** | consolidated evidence → `evidence.json`, console in `evidence_stdout.txt` |
| `python native/hdr_real_admission_gate.py --help` | 0 | confirms the three flags and their original defaults |

The gate was **reused, not cloned**: it gained `--source`, `--support-ceiling` and
`--output-dir`, each defaulting to the original run (the legacy clip, 10000 scRGB,
`bench_out/hdr_real_admission`), so a bare invocation still reproduces the evidence
already on disk. Its two failed ceiling-250 attempts against `hdr_src.mkv` moved from
`declared_domain.history` to `declared_domain.prior_attempts_on_the_legacy_fixture`,
labelled as that clip's history rather than a domain declaration for this one. The old
evidence files were not touched. No generator rerun, no re-encode, no old A/B, math or
unit gate was re-run, and no GPU work was repeated to restate a comment.

The verifier loads no DLL and creates no device. Its only external commands are
`ffmpeg` (software `hevc` decoder, `-hwaccel none`, one frame, `-pix_fmt yuv420p10le`)
and `ffprobe`.

## What the offline check measured

Frame 0, read from the clip as it sits on disk. Format stated explicitly: **planar
yuv420p10le** — Y plane, Cb plane, Cr plane, each 10-bit code in the low 10 bits of a
little-endian uint16, strides 3840 and 1920 bytes with no padding.

| Item | Measured |
| --- | --- |
| Frame byte count | 6,220,800 = 4,147,200 (Y 1920×1080) + 1,036,800 (Cb 960×540) + 1,036,800 (Cr) |
| Decoded code ranges | Y [339, 713], Cb [398, 626], Cr [398, 626] — inside limited range, nothing above bit 9 |
| Code error vs `authored_f0.yuv` | 0 mismatching samples per plane, max abs code error 0, mean 0 |
| Decoded frame sha256 | `ddd225cf30e755577f81eab9f1782fc968db04c78ef84edd4c6ade2048bf7b34` — equal to the authored reference |
| PQ components, product's own chroma phase | [0.313927, 0.740868]; 0 below 0, 0 above 1, 0 non-finite |
| PQ components, all 9 box phases | [0.313927, 0.820450]; still inside [0, 1] |
| Normalised linear | [0.001195, 0.090432] own phase, [0.001195, 0.187598] worst phase — inside [0, 1] |
| cd/m² | 11.95 → 904.32 own phase; 1875.98 worst phase (see the disclosure below) |
| scRGB | [−4.519660, 13.992961] own phase, [−4.519660, 25.932445] worst phase; 221,184 negative components |
| vs the generator's pre-encode record | max abs delta **exactly 0.0** in all four domains, own phase and worst phase |
| Frame mean luminance | 227.04 cd/m² (declared MaxFALL 250) |
| Frame peak pixel | 904.32 cd/m² at luma code 713; one code step at that level is 9.53 cd/m², so the declared MaxCLL of 900 differs from the decoded peak by 4.32 cd/m² — under half a code step of 10-bit rounding, not an unbounded claim |
| Container | hevc **Main 10**, `yuv420p10le`, 1920×1080, `tv` / `bt2020` / `smpte2084` / `bt2020nc`, `-count_frames` → exactly **6** frames at `25/1`. ffprobe reports `level 255` (unspecified in the VUI) while the encoder's own log recorded "Main 10 profile, Level-8.5 (Main tier)"; neither the gate nor this check depends on the level, and the discrepancy is reported rather than reconciled |
| Side data | Mastering display G(8500,39750) B(6550,2300) R(35400,14600) WP(15635,16450) L(10000000,50)/10000 — equal to BT.2020/D65, 1000 cd/m² max, 0.005 min; Content light max_content 900 / max_average 250 |

Disclosure, stated rather than smoothed: the worst-case number above comes from pairing
every luma sample with the chroma of a *neighbouring* box, a geometry the product does
not read — the bridge's decode shader replicates each 4:2:0 sample to its own box
(`native/gpu_hdr_scrgb_bridge.cpp:653-684`), which is the own-phase column. Under that
hypothetical reading a bright surround luma sample can be paired with a chroma-excited
box and reach 1876 cd/m², above the 1000 cd/m² mastering maximum, while PQ itself stays
at 0.8204 — inside [0, 1]. The fixture is in range under either reading; only the
own-phase one is what the pipeline sees.

## What the native admission measured

One borrowed frame, the decoder's own fields, the production wrapper, the accepted DLL.

| Item | Reported by the decoder / bridge |
| --- | --- |
| Colour fields sent, as the decoder said them | `color_range 1`, `color_primaries 9`, `color_transfer 16`, `color_space 9` → **accepted** |
| Frame | 1920×1080, `software_format 2` (P010), y/uv device pointers `0xd0b600000` / `0xd0ba38000`, both pitches 4096 bytes, `frames_decoded 1`, `device_frame_count 1` |
| Source HDR metadata as the decoder parsed it | `has_mastering 1`, `has_content_light 1`, max_cll 900, max_fall 250, max_luminance 10000000/10000, min 50/10000 |
| Output descriptor | `rgba16f_cuda`, **full resolution** 1920×1080, 8 bytes/px, pitch 15360, device ptr `0x205200000`, plane 2, CUDA context == the decoder's own context |
| Bridge contract units | scrgb_per_nit 0.0125, nits_per_scrgb 80.0, pq_c1/c2/c3 0.8359375 / 18.8515625 / 18.6875, kr/kg/kb 0.2626544 / 0.6781436 / 0.0592021, paper white configured and effective 8.0 |
| Working extent | 960×544 via `pipeline.resolve_working_dims(…, 0.5)`; the borrowed `Live` session was given **0 frames** |
| Source report | ceiling 250.0, min **−4.519786834716797**, max **13.992795944213867**, `negative_components 221184`, `above_ceiling_components 0`, `nonfinite_samples 0`, `pixels 2073600` |
| PQ geometry | pole at coded input 1.9921; measured max sits at coded input 0.7641, **38.4% of the way to the pole** — nowhere near the runaway region the legacy clip lived in |
| Decoder counters | `host_frame_copy_count 0`, `host_frame_copy_bytes 0`, `scale_device_output_bytes 0`, `bitstream_bytes_submitted 33401` |
| Bridge counters | `process_count 1`, `source_device_copy_bytes 6220800` (= one full P010 frame), `source_device_write_bytes 16588800`, `report_readback_bytes 0`, all downstream write counters 0 |
| Negative case | same frame, `color_space` → 10: `HdrBridgeError`, refusal text names NCL/ST 2084 and reports `space 10`; bridge still open, no fallback |
| Teardown order | `gpu_hdr_bridge` → `dlss_gpu_engine.Live` → `nvdec_bridge` (CUcontext popped in between) |

## Binaries, hashed after the FFmpeg libraries were loaded

Loader order is part of the contract: the FFmpeg dependency directory and
`nvdec_bridge.dll` are registered before anything that can bind `libcrypto`, or the
avformat chain resolves to the interpreter's own copy and fails with WinError 127.

| Module | sha256 |
| --- | --- |
| `native/gpu_hdr_scrgb_bridge.dll` (the accepted build; match asserted, not assumed) | `e016fce6e1b3cbc01b4acd751db71144c1546dcd57e362cdb0486a6a3a269e47` |
| `native/nvdec_bridge.dll` | `263c4397876e729fd964e1457c2bc5365de3032c77ba481961af69b85f801ea0` |
| `native/dlss_gpu_host.dll` | `5640d4fe0fcb03f5a3e7b50ffe287acb18bd7b1c5cfd6f4342af318e1b799cfd` |
| `avutil-59.dll` | `1e6ff943de8f95ecd19df06f8a5f2ee24ac653b99c34ccb122f790a64fd34342` |
| `avcodec-61.dll` | `cbcfa732690ff0de92c3fc3d34185991ac4f8582d631f0c99bf0c9873367a69d` |
| `avformat-61.dll` | `e338eee769b66cf4bffc368b7c4dc3abf7fadaa4d9c17c67869081bc9fafd379` |
| `libcrypto-3-x64.dll` (resolved to `F:\WhisperJAV\Library\bin`, which is the point of the ordering) | `e5a134bc7562c8f279b5f88612c7d9140ee52ee2f6368d92c383ec587366d972` |
| `nvcuda.dll` | `340e161989d0bf3b1731b1e4d707aa744311c5fdff166023df8b07376772e76f` |

## Artifacts

| Path | Bytes | sha256 |
| --- | --- | --- |
| `build/hdr_probe/hdr_valid_src.mkv` (6 frames, final production-export source) | 37,561 | `facd372c2e160c4aa3ff9301612a8994ddb7226c8b59d3feb5ff9421bdce68e3` |
| `bench_out/hdr_valid_fixture/evidence.json` (physical check + native admission + cross-check) | 48,699 | `8fe0752c768eab329bca692069e4272fb05560409baa63c01c7157f38da075bd` |
| `bench_out/hdr_valid_fixture/evidence_stdout.txt` (its console) | 3,247 | `890e6401444fa8d658bee21f17394bb7e8b6d42701d614c47dcbce5a61d3735e` |
| `bench_out/hdr_valid_fixture/physical_check.json` (offline only, no native section) | 19,748 | `bfb67c09cd7a25dddf3ef5b7e5340ef9b105a88883f07b3989ba784e573342e6` |
| `bench_out/hdr_valid_fixture/verify_stdout.txt` (its console) | 2,720 | `e37a48c90163b68690185892b78bc35daa4b37c9e1a614c47dcbce5a61d3735e` |
| `bench_out/hdr_valid_fixture/admission/evidence.json` | 24,149 | `f94ba3c8769281e10bc1beb5fad25eb6be4dd0d6387097fd86ad4edc025a2921` |
| `bench_out/hdr_valid_fixture/admission/gate_stdout.txt` | 2,326 | `e8f2e060c93eb8bcd4406832c9d1e575928fa40b3b2ffab11014a8da1541ac3a` |
| `bench_out/hdr_valid_fixture/decoded_f0.yuv420p10le.raw` (diagnostic bytes) | 6,220,800 | `ddd225cf30e755577f81eab9f1782fc968db04c78ef84edd4c6ade2048bf7b34` |
| `bench_out/hdr_valid_fixture/authored_f0.yuv` (generator reference, identical bytes) | 6,220,800 | `ddd225cf30e755577f81eab9f1782fc968db04c78ef84edd4c6ade2048bf7b34` |
| `bench_out/hdr_valid_fixture/authored_scene.json` | 31,341 | `500a7746b5687ee594d45065e9db59c106650f4d76d250ab0c3c3051c230bbc6` |
| `native/verify_hdr10_valid_fixture.py` | 36,281 | `f873781ff7b69354b2663ecfa3ff473f20c1ee9b0a32674e314f737c64a7e485` |
| `native/hdr_real_admission_gate.py` (after adding the three flags) | 42,486 | `544c7d84faf4ee799036cdd018ae4bd7fb3cf232afe9ef678fd4f4d0d6b8ae9e` |
| `native/make_hdr10_valid_fixture.py` (unchanged since it authored the clip) | 26,627 | `0208922f70be207f0e7de59e695b4622555dc04c71da42a8b71a9000d6b5cca6` |
| `build/hdr_probe/hdr_src.mkv` (legacy clip, untouched) | 148,007 | `48697ed8deb0f10475151dbc0b9312d3408b513087f2c3f76be6e7d12ee83b8b` |

## What this is not

- **Not an export.** The six-frame clip is the source for the next production export; no
  export ran.
- **Not Feature 18 acceptance.** No evaluate call was made. The `Live` session exists only
  to lend its D3D12 device handle and processed 0 frames. Formal Feature 18 HDR acceptance
  remains **unproven**.
- **Not a performance claim.** `NOT_CHARACTERIZED` by both runs.
- **Not a numeric change.** Native algorithms and the Stage A FP16 loss record are neither
  restated nor superseded here.
- **Not product wiring.** Orchestrator, UI and the final 4–6 frame gate are still pending.
- The legacy clip's 10000 scRGB admission is left as recorded; it stays a crash-freedom
  proof, not a colour proof.

## Next

1. The final 4–6 frame gate: every frame of `hdr_valid_src.mkv` through native admission
   and the production transfer path at the 250 scRGB domain, still without readback.
2. The first production export from this fixture, then the orchestrator/UI surface.
3. Feature 18 HDR acceptance proper.
