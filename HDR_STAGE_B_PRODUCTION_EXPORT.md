# HDR Stage B — the experimental HDR10 chain inside the production export, and the one six-frame gate that proves it

Workspace `F:\dlss5standaloneV2`, continuing from `HDR_STAGE_B_VALID_FIXTURE.md`.

That record ended with a physically valid HDR10 sample and the native colour bridge
admitting it at a conservative **250 scRGB**. What it did not have was the product: the
chain lived in `hdr_gpu_engine.py` bindings and native smoke scripts, and nothing in the
queue could ask for it. This task connects it. One file was added to the product, three
were edited, and **one** 6-frame export was driven through the entry point the queue
calls. It delivered `bench_out/hdr_production_export/hdr_valid_src_hdr10_gpu.mkv`, HEVC
Main 10, yuv420p10le, tv / bt2020nc / smpte2084 / bt2020, with the fixture's own
mastering-display and content-light numbers present as byte-identical prefix SEIs, and
**no image byte crossing the host boundary in either direction**.

No UI work is in here. The front end cannot ask for this chain yet, so the whole feature
is off by default and reachable only through `job_settings`. Nothing is committed; the
dirty baseline is the one this task inherited.

## What changed

| File | Role in this task |
| --- | --- |
| `gpu_hdr_transfer_export.py` | **new.** The whole HDR chain: capability probe, negotiation gate, the per-frame bridge sequence, the report, the run. ~1650 lines, no algorithm of its own — every pixel decision is the accepted native bridge's. |
| `pipeline.py` | `export_gpu_transfer_stream` grew one delegation: a job whose settings opted into HDR leaves for `export_gpu_hdr_transfer_stream` **before** the SDR gate is consulted. New entry `export_gpu_hdr_transfer_stream` (refusal → diagnostics + `RuntimeError`, success → the chain's report printed as JSON). `transfer_module` seam added so a CPU test can stub the bridge set without touching the SDR entry's `module` seam. |
| `dlss_service.py` | `_hdr_request()` reads the chain's own predicate; `add_job` refuses a bad flag, a bad paper-white value, HDR without `export_backend=gpu_transfer`, and HDR with dual/triple reconstruction; `_execute_job` opens `Live(..., color_format="rgba16f")` **only** on the HDR branch — the SDR call is byte-identical to what it was. `experimental_hdr_scrgb` and `dlss_session_color_format` are written onto the job row only for an HDR job. |
| `tests/unit/test_hdr_transfer_export_branch.py` | **new.** 81 CPU tests: opt-in/opt-out, every refusal, the plan and extent maths, the NCL-9 colour enums, FP16 session selection, the per-frame stage order, teardown order, cancel, a failed frame, metadata hand-off, counter honesty, and the two pipeline routing seams. |
| `tests/unit/hdr_production_export_gate.py` | **new.** The one gate: drives the export through `pipeline.export_gpu_transfer_stream`, then inspects the finished file offline with ffprobe and an Annex-B reader. `--recheck` re-reads the same deliverable without driving a second export. |
| `bench_out/hdr_production_export/` | **new evidence.** `settings_and_inputs.json` (source + nine library + six module sha256, the settings), `diagnostics.json` (the run and the chain's whole report), `gate_log.json` (38-check verdict), the deliverable, both elementary streams, and `run1_inspector_false_negative/`. |

`hdr_gpu_engine.py`, `dlss_gpu_engine.py`, `gpu_transfer_export.py`, every native source and
every `.dll` are **reused as accepted**. The colour bridge loaded by the run hashes to
`e016fce6e1b3cbc01b4acd751db71144c1546dcd57e362cdb0486a6a3a269e47` — the file the Stage A
and admission records already signed off. No native math, structure or kernel was touched,
and no earlier gate was re-run.

## The data flow of one frame

Nothing in this list can be reordered, overlapped, or run off-thread; each stage hands
back borrowed device memory that the next call into the same bridge invalidates.

```
NVDEC borrow (CUDA context, P010, 1920x1080, its own colour enums + HDR metadata)
  └─ nvdec_bridge_get_hdr_metadata          valid only on the next thread, frame current
      └─ hdr.decode(P010 device, range 1 primaries 9 transfer 16 space 9, ceiling 250)
          └─ hdr.downscale 1920x1080 → 960x544        signed, unclamped FP16 scRGB
              └─ hdr.proxy → FP16 colour + RGBA8 motion guidance (4 bytes, not the picture)
                  ├─ flow_provider.get_flow_device_rgba(FP16 colour, RGBA8 guidance)   NVOF
                  └─ Live.process_device_color_flow_resource(FP16 proxy, float2 flow, reset)
                      └─ hdr.reconstruct(FP16 texture + fence, residual_strength)
                          └─ hdr.encode(clip ceiling 125) → P010 Y + UV on the device
                              └─ nvenc_encode_p010(device Y, pitch, UV, pitch, source PTS)
```

Opens are lazy and in this order — decoder, CUcontext push, colour bridge (on the
**Feature 18 session's** D3D12 device), encoder — and `close()` tears down in the exact
reverse: encoder, colour bridge, Feature 18, context pop, decoder. The tests assert that
order, including on a frame that fails and on a cancel. One Feature 18 session is opened
per job and reset on the first frame only; the run measured `hdr_feature_resets_measured
= 1` for six frames, so no session is rebuilt mid-job. The borrowed frame is fully
consumed before the decoder advances; a discontinuity, a seek, a job start or a cancel is
handled by resetting NVOF history and issuing an independent Feature reset, never by
guessing. Guidance is the only thing that leaves the picture path: `guidance_mode=0` may
use the resident zero-motion FP16 entry, and a failed NVOF call is a failure, not a
fallback — there is no path in this file that would draw a frame anywhere else.

`assert_output_not_input` runs before either entry writes a byte.

## Fixed parameters, and what is never derived

| Quantity | Value | Where it comes from |
| --- | --- | --- |
| support ceiling | **250.0 scRGB** | a per-job constant, the conservative bound the bridge was admitted at |
| encode clip ceiling | **125.0 scRGB** | PQ's top (10000 cd/m²) at this bridge's 80 cd/m² reference |
| paper white | **8.0 scRGB** = 640 cd/m² | `hdr_paper_white_scrgb`, validated finite and ≥ 1, fixed per session |
| nits per scRGB | 80.0 | bridge contract |
| derived legal BT.2020 peak | 207.53 scRGB | reported as arithmetic, **not used** as a ceiling |
| layers | 1 | dual/triple refused |
| passes | 1 | `multi_pass > 1` refused |
| output resize | none | output extent == source extent |
| tone mapping | none | `hdr_to_sdr` and every mapping preset refused, because the bridge holds a signed unclamped master precisely so nothing is mapped twice |
| mux | none | the chain runs no mux pass |

**Neither ceiling is derived from MaxCLL, and neither adapts per frame.** The fixture's
`max_cll 900 / max_fall 250` are handed to the encoder as metadata and appear nowhere in
the ceiling arithmetic; the ceiling is the same string in the report on frame 1 and
frame 6: `"fixed per job: never derived from MaxCLL, never adapted per frame"`.

## The refusals, all before a pixel moves

`negotiate_hdr` refuses, with the reason written into diagnostics: a session that is not
`rgba16f` (an already-open RGBA8 session is **refused, never silently used**); a backend
that is not `gpu_transfer`; a flow engine that is not NVOF, or a provider without a device
RGBA input entry point; a source that is not 10-bit PQ limited-range BT.2020 NCL (HLG,
full-range, constant-luminance, 8-bit, other primaries); dual/triple reconstruction;
`multi_pass` above 1; output resizing; any colour preset other than `preserve`, including
`hdr_to_sdr`; a custom colour clause a muxer would have written; a container that is not
`.mkv`; **any job that would need a rewrap** — audio, subtitles, chapters, attachments,
format metadata, an mp4 tag or `extra_args` — which is refused with the source inventory
printed rather than quietly dropped; locked or unknown encoder options; a non-positive
frame rate; and any encoder, SEI-writer or ABI capability the loaded libraries do not
answer. Unsupported or partial metadata is reported as refused, never claimed as preserved.

## Scope A — CPU contract tests

```
python -m pytest tests/unit/test_hdr_transfer_export_branch.py -q
81 passed in 0.88s
```

No GPU, no bridge, no DLL: both pipeline seams are stubbed, so `load_bridges` never runs.
The cases that carry the contract are the RGBA8 refusal, the "one FP16 session, reset
once", the five-stage order repeated six times, the reverse teardown order on both a
failing frame and a cancel, a decode that stops early (5 processed against 6 requested,
reported as two separate numbers), the metadata rationals handed to the encoder compared
field by field, and the counter-honesty rule — an unanswered counter reads `None`, and a
counter that has no entry point at all reads absent. `python -m py_compile` passes on all
six touched modules.

## Scope B — one export, 38 checks

One run of `python tests/unit/hdr_production_export_gate.py`. Fixture
`build/hdr_probe/hdr_valid_src.mkv` (sha256 `facd372c2e160c4a…68e3`, 1920x1080, 6 frames
@ 25/1), settings `guidance_mode=1 flow_engine=nvof flow_quality=4M working_scale=0.5
multi_pass=1 dlss_backend=gpu_flow_d3d12 export_backend=gpu_transfer
experimental_hdr_scrgb=true hdr_paper_white_scrgb=8.0 reconstruction_mode=single`, encoder
`hevc_nvenc p7 cq 24 mkv`, no audio in the fixture.

| Question | What the finished file and the chain's own report say |
| --- | --- |
| Did the product entry run the HDR chain? | `export_backend_actual=gpu_transfer`, `transfer_mode_actual=gpu_hdr_transfer`, `export_backend_fallback_reason=null`, `dlss_backend_actual=gpu_flow_d3d12`. A refusal raises instead of falling through; none was raised. |
| Six frames? | requested 6, processed 6, `decode_frames_decoded_measured=6`, encoder `frames_submitted 6 / packets_written 6`, ffprobe counts **6 packets** and decodes **6 frames**, progress reported 1..6. |
| Presentation order? | frame `pts` strictly increasing `0 40 80 120 160 200`; 6 distinct packet `pts`, the same set; packets are stored in decode order because the encoder used B-frames, which is why 2 of them carry no `dts` ahead of the first random-access point. |
| The HDR10 declaration? | `hevc / Main 10 / yuv420p10le / tv / bt2020nc / smpte2084 / bt2020 / 25/1` — identical to the source's own tags. |
| The source's mastering and CLL numbers? | container side data **identical**: `red_x 35400/50000 red_y 14600/50000 green_x 8500/50000 green_y 39750/50000 blue_x 6550/50000 blue_y 2300/50000 white 15635/50000 16450/50000 min 50/10000 max 10000000/10000`, `max_content 900 max_average 250`. |
| And in the elementary stream, not just the container? | demuxed to Annex-B and parsed: **137** prefix SEI (nal_unit_type 39) payload `21349b46199608fc8a4839083d1340420098968000000032` (24 bytes) and **144** payload `038400fa` (4 bytes), **byte-identical to the source's**. The writer inserted them into every access unit: `packets_seen 6, nals_inserted 6, bytes_inserted 240, mastering_inserted 6, content_light_inserted 6, already_present 0/0, error "ok"`, and `metadata_written_to = 31` — frame side data, stream coded side data, encoder coded side data, container colour tags **and** in-bitstream prefix SEI. No SEI was written in Python. |
| Stable metadata? | 6 snapshots, one per frame, `hdr_metadata_stable_across_frames=true`, `divergence=null`. |
| Feature 18 on the right session? | `hdr_color_format=rgba16f`, `bytes_per_pixel=8`, host `dlss_gpu_host.dll` abi 3, working 960x544, full and output 1920x1080, `hdr_frame_rate="25/1"`, `hdr_feature_resets_measured=1`. |
| Real NVOF guidance? | `hdr_uses_flow=true`, `hdr_dlss_flow_calls_measured=6`, `hdr_dlss_zero_motion_calls_measured=0`, `nvof_device_input_copy_count_total=7`, `nvof_device_input_bytes_total=14622720`, and NVOF itself reported `device=NVIDIA GeForce RTX 5060 Ti`, `supports_device_rgba_input() True`. |
| Any image byte on the host? | **All seven host counters are the bridge's own zero:** decoder `host_frame_copy_bytes 0` / `host_frame_copy_count 0`, Feature 18 `host_color_stage_bytes 0` / `host_color_stage_count 0` / `output_readback_bytes 0`, encoder `host_pixel_bytes 0` / `rgb_conversion_bytes 0`. Decoder `scale_device_output_bytes 0` — the HDR open cannot scale. Structurally: `hdr_pixel_readback_exports_bound=[]` (no readback export exists in the bound set) and `decoder_pixel_readback_entry_bound=false` (`nvdec_bridge_readback_last_frame` is not bound), so `hdr_pixel_readback_bytes_measured` is reported as **None, not zero** — there is nothing to measure. |
| Where the pixels did go | device-to-device, counted: borrowed P010 copy 37,324,800 (6 × 6,220,800), source plane writes 99,532,800, working plane writes 100,270,080, guidance plane writes 12,533,760 (6 × 2,088,960 RGBA8), composited plane writes 99,532,800, encoded P010 writes 37,324,800, Feature 18 device colour copy 25,067,520, encoder P010 device copy 37,324,800. The planes are described apart from the traffic: 8 B/px FP16 for the picture, 4 B/px 8-bit for the guidance. |
| The two readback boundaries, read apart (corrected) | `bench_out/hdr_production_export/diagnostics.json` still carries `hdr_scalar_report_readback_bytes_measured=0`, and that zero is a real bridge number — but it is `report_readback_bytes`, the counter for **image planes** the diagnostic `gpu_hdr_scrgb_bridge` readback entries copy out, which this chain never binds. It was filed under a name that claimed the per-pass scalar counter readback, and that claim was false: every executed pass copies `kCounterSlots * sizeof(uint32_t)` = 256 bytes of counters into a CPU-visible (`D3D12_HEAP_TYPE_READBACK`) buffer and the host memcpy's them out, because that host snapshot is what the report structs are answered from. So the honest reading of the same artifact is: **no picture byte reached the host** — seven image host-boundary counters at 0, `hdr_pixel_readback_exports_bound=[]`, `decoder_pixel_readback_entry_bound=false`, `hdr_pixel_readback_bytes_measured=null` — while **scalar report readback is present on every dispatch and its byte count is unmeasured**, not zero. `total_frames_reported=6` and `hdr_process_count_measured=6` are the two counters that say the passes ran and their reports arrived. Corrected in code by `HDR_STAGE_B_BACKEND_AUDIT_REPAIR.md`; the field renamed there to `hdr_diagnostic_image_readback_bytes_measured`, the scalar slot answered `None`. Per-stage extremes remain envelopes, not means. |
| The encoder's own contract | `hevc_nvenc / main10`, `actual_pix_fmt 117` (yuv420p10le), `actual_sw_pix_fmt 158` (p010le), `bits_per_raw_sample 10`, `color_range 1 primaries 9 trc 16 space 9` requested **and** actual, hw pool 16, `spatial-aq 1 temporal-aq 0 aq-strength 8`. |
| Verdict | **38/38 checks passed.** Deliverable 26,149 bytes, sha256 `4727d40adfef6b93b07ca51a15a9f9b70158eafc70229723964bb4b9363ea5d6`. |

### What the picture actually did at the ceilings

All counts are the bridge's own per-stage reports summed over six frames; the negatives
and rejections are preserved against the pristine HDR master rather than clamped away.

| Stage | Counts read |
| --- | --- |
| source | 12,441,600 pixels, **1,216,512 negative scRGB components preserved**, 0 above the 250 ceiling, 0 non-finite, envelope **−4.519786834716797 … 13.992795944213867** — the same min/max the valid-fixture record measured offline on frame 0, which is the cross-check that the decode domain is the one that was admitted. |
| proxy | 3,133,440 pixels, 303,310 negative, 288 pinned, 74,139 dead-zone, 162,782 above actionable, 288 above saturation, code range 0.0 … 1.0. |
| reconstruct | 8,665,374 applied components, 357,209 gain-capped, **288 pinned / 74,139 dead / 303,310 negative rejected** (the proxy's own rejections carried through, not re-decided), 0 non-finite, 1,223,278 composite-out-of-domain, composite envelope −4.586250305175781 … 14.494584083557129, max abs residual 7.19140625 scRGB, max abs stop envelope 1.0. |
| encode | **0 clamped high against 125 scRGB**, 567,848 clamped low, 567,848 out-of-gamut components, 0 non-finite; code envelopes Y 64 … 710, Cb/Cr 390 … 628, max linear component 14.4921875. Nothing reached the top of PQ; the clamps are all at black, and the composite's 14.49 scRGB peak is 1,159 cd/m² — inside a 125 scRGB transfer function with room to spare. `total_encode_pixels_total` reads 0, which is the field the bridge answers, not a number this file invented. |

### The five false negatives, and why no second export was driven

The first pass of the gate reported FAIL on five lines. **Every one of them was a bug in
the gate script's own read-back, not in the chain:**

1. frames were filtered on `codec_type`, and this ffprobe build emits `media_type` for
   frames, so all six pictures were dropped before anything else was read;
2. and 3. mastering and content light then read `null` for both files, for the same reason
   — the side data lives on the frames. Read directly, they are identical;
4. the PTS check read *packets*, which are stored in decode order in a B-frame stream, so
   `0 160 80 40 120 200` looked non-monotonic while `0 40 80 120 160 200` in presentation
   order is exactly right;
5. the report-readback check demanded a positive number from `report_readback_bytes`,
   which the bridge answers 0 for, and which the two accepted Stage B records read as 0.
   (A later audit found the *label* on that line wrong rather than its number: 0 is the
   image-plane diagnostic counter, and the per-pass scalar counter readback it was named
   for is real traffic that no counter measures. See the corrected row above and
   `HDR_STAGE_B_BACKEND_AUDIT_REPAIR.md`; the gate's own checklist now reads the two
   boundaries separately.)

The chain and its deliverable were untouched by that discovery. Run 1's artifacts are
preserved whole in `bench_out/hdr_production_export/run1_inspector_false_negative/` —
including the deliverable itself and its elementary stream — with
`WHY_THIS_RUN_DID_NOT_COUNT.txt` beside them. The corrected checklist was then re-read
over **the same** files with `--recheck`, which drives no export. There is exactly one
HDR export in this record.

## Scope C — the SDR path is off-path, and it shows

The branch is entered only through `htx.hdr_requested(js)`, and that predicate is
`bool(job_settings.get("experimental_hdr_scrgb"))`, which defaults to `False` in
`dlss_service`'s own settings dict. Everything else — negotiation, bridges, session
format, report keys — is the SDR chain unchanged.

One real regression was found and fixed here, and it was found by the SDR tests, not by
the HDR ones: writing `color_format=` unconditionally into the `Live(...)` call in
`dlss_service._execute_job` broke 10 tests whose host is a stub factory that does not take
the keyword. The call is now branched, so an SDR job issues exactly the call it always
did, and the two job-row fields an HDR run adds are written only inside the HDR branch.

```
python -m pytest tests/unit/test_hdr_transfer_export_branch.py -q                        81 passed
python -m pytest tests/unit/{test_gpu_transfer_export_backend,test_export_pipeline,
  test_cancellation_contract,test_cancellation_e2e,test_export_backend_ui_contract,
  test_output_overwrite_guard,test_no_misleading_claims,test_export_zero_motion_pipe,
  test_export_cpu_flow_gpu_host}.py -q                                                   286 passed
python -m pytest tests/unit/{test_hdr_gpu_engine,test_guidance_contracts,
  test_nvof_independence,test_preview_export_parity,test_gui_stream_export_gpu_host,
  test_gui_preview_gpu_host}.py -q                                                       175 passed
```

No native host or encoder regression was re-run: the FP16 host connection and the P010
NVENC admission are already recorded in `HDR_STAGE_B_FP16_HOST_CONNECT.md` and
`HDR_STAGE_B_P010_NVENC.md`, and this task changed neither. No short SDR production
comparison was driven, because after the branch fix the SDR code path is textually the one
those 461 tests exercise and the risk that remained was the session call, which is one of
them.

## What this does not claim

* **The UI is not done.** Nothing in `frontend/` can set `experimental_hdr_scrgb` or
  `hdr_paper_white_scrgb`, no control warns that the chain is experimental, and the queue
  can only be reached by editing settings. **The HDR task is not finished.**
* **Formal HDR conformance is not claimed.** The output carries an HDR10 declaration;
  carrying one is not certifying one.
* **No losslessness.** FP16 quantisation of the working master is the Stage A record's, and
  the 8-bit guidance proxy discards colour by design.
* **No quality improvement.** Nothing here measures perceptual quality.
* **Performance is NOT_CHARACTERIZED.** One 6-frame export in one process took 5.13 s of
  wall clock with per-stage timings in `diagnostics.json`; that is a record of one process,
  not a throughput or latency claim, and no benchmark or A/B repeat was run.
* **Domains refused, not served:** HLG, full-range, non-BT.2020 primaries,
  constant-luminance matrix, 8-bit sources, resized output, dual/triple layer, multi-pass,
  tone mapping, any container but Matroska.
* **No mux.** One video stream and nothing else: the encoder's own Matroska writer produced
  this file, and a job that needs audio, subtitles, chapters or attachments copied across
  an HDR10 bitstream is refused with its inventory printed, because that capability has
  not been validated against the SEIs and container colour tags this chain exists to
  preserve.
* **Metadata coverage is two payloads, not the whole bitstream.** The source's elementary
  stream also carries two unregistered user-data SEIs (payload type 5, 2,319 bytes, whose
  text is `x265 (build 199)` plus its encode settings). Those are **not** reproduced: the
  chain hands off mastering-display and content-light only. That is a tool tag rather than
  a colour declaration, but it is a difference between the source bitstream and the
  delivered one and is stated here rather than glossed.
* **Declaration density differs too, on the output's side.** The source put 137/144 in the
  prefix SEI of 2 of its 6 access units; the writer inserts them into all 6, and the
  encoder adds its own picture-timing SEI (payload type 1) per AU. Every payload byte is
  identical; how often it is repeated is not.
* **Paper white is fixed.** No scene-adaptive, per-frame or MaxCLL-derived anything.
* 250 scRGB is a conservative bound, not a measured ceiling of this pipeline; the ceiling
  is not raised on the strength of one clip whose peak landed at 14.49 scRGB.
