# Experimental HDR10 × the existing FFmpeg remux path

Date: 2026-10-02. Baseline: `main` @ `7195f8979583fc48b4f47654f09841e82025cec4`. Plan:
`docs/plans/2026-10-02-hdr10-remux-integration.md`. Audit repair: same day, read back in
§8 (`bench_out/hdr_remux_export/audit_repair/`).

This round changed one thing about the HDR chain's shape: a job whose request carries
anything besides its video stream used to be refused by the HDR gate, and now it runs. It
runs on the project's *existing* map-and-copy pass, the same two helpers the SDR GPU-transfer
chain and the pipe chain have always used, and the deliverable is that pass's output file.
No muxer was written, no audio, subtitle, chapter, metadata or attachment handling was
reimplemented, and no HDR mathematics, Feature 18 session, NVOF path, multiscale layer,
native bridge or DLL was touched.

The audit that followed that work left four things open, and they are what §8 is about: the
intermediate's cleanup had to cover the whole run rather than the remux, the chapter count had
to come from the file instead of a field that file never writes, `extra_args` had to stop being
sorted into safe and unsafe, and the source video's own track description had to reach the
deliverable without becoming a setting the operator must additionally enable. None of the four
changed what the copy pass maps, and none of them is a new GPU result: the repair round ran no
GPU, encoded nothing and re-rendered no picture.

Performance: **NOT_CHARACTERIZED**. The elapsed seconds of one process are recorded in the
evidence and no throughput or latency claim is made from them. There is no quality claim
here either: this document is about which bytes landed in which stream.

---

## 1. What is reused, by name

| Need | Reused, unchanged | Where it is called from now |
| --- | --- | --- |
| Read the source's facts | `gtx.probe_source(video, ffprobe=None, include_chapters=False, include_video_metadata=False)` | HDR: `pipeline.py:3306` with both opt-ins True. SDR: `pipeline.py:3208`, still `probe_source(video)` — the defaults are what the SDR chain has always received |
| Decide whether the request needs a rewrap | `gtx.needs_copy_pass(source, ts, with_audio, out_path, video_descriptive_metadata=None)` | `gpu_hdr_transfer_export.py:1001` (inside `negotiate_hdr`) |
| Assemble the map-and-copy command | `gtx.build_mux_command(ffmpeg, encoded_video, source, ts, out_path, with_audio)` | `gpu_transfer_export.py:1174`, reached from `gpu_hdr_transfer_export.py:1991` |
| Execute it with hooks, timeout and a named failure | `gtx.run_copy_pass(..., video_descriptive_metadata=None)` | same |
| The video track's own description, narrowed and then expressed as clauses | `gtx.source_video_descriptive`, `gtx.video_descriptive_clauses`, `gtx.extra_args_request` | derived by `gpu_hdr_transfer_export.py:783`, appended by the shared builder |
| The audio/subtitle mode vocabulary the pass expresses | `gtx.MUX_AUDIO_MODES`, `gtx.MUX_SUBTITLE_MODES` | `hdr_mux_refusals`, mirrored in the client |
| Refuse a destination that is the source | `pipeline.assert_output_not_input` | already called by `export_gpu_hdr_transfer`, and again inside `build_mux_command` for the remux's own input/output pair |
| Uncached final verification | `pipeline.verify_export_output(out_path, frames, probe=video_info_uncached)` | after the copy pass, on the caller's name |
| Ordinary mapping policy | `pipeline.build_transcode_command` | untouched; the HDR chain adds no mapping rule of its own |

`gtx.run_copy_pass` is the one extraction this round made in the SDR file: the execution half
of the copy pass (`subprocess.Popen` → hooks → `communicate(timeout=MUX_TIMEOUT_SEC)` → kill
on timeout → `BridgeError` naming FFmpeg's stderr tail → return the command) moved out of
`export_gpu_transfer`'s body into a function beside `build_mux_command`, and the SDR
caller now calls it. Clause-for-clause the command is what it always was, and so is the
mapping, the hook order, the non-zero-exit failure and the success path. One difference is
worth naming rather than glossing: the inlined SDR version passed `timeout=1800` but caught
nothing, so a remux that stopped responding raised a bare `subprocess.TimeoutExpired` out of
the job and left the FFmpeg process running; the shared helper kills it and raises the chain's
own `BridgeError`. That is the timeout the SDR pass always asked for, now honoured — no SDR job
that used to succeed behaves differently. The existing SDR tests still pin the command, the
hooks and the non-zero-exit failure; the timeout rule is pinned where it now lives, by
`test_a_rewrap_that_stops_responding_is_terminated_and_reported`, which drives
`run_copy_pass` itself rather than the HDR branch around it.

The audit repair added four more things to that same file, all of them defaults-preserving:
`probe_source`'s two opt-ins (§8.1, §8.4), `extra_args_request` (§8.3), and
`source_video_descriptive` + `video_descriptive_clauses` with the one new keyword on
`needs_copy_pass`/`run_copy_pass` (§8.4). Every one of them is off unless a caller asks: the SDR
entry passes no opt-ins, reads `format.nb_chapters` exactly as it always did, sees an empty
`video_descriptive_metadata`, and builds the identical argv — pinned by
`test_the_descriptive_clause_names_the_encoded_stream_and_the_sdr_command_is_unchanged` and by
`tests/unit/test_gpu_transfer_export_backend.py` as a whole.

The evidence that no parallel muxer exists: the copy-pass command the production run actually
executed is `build_mux_command`'s output verbatim, recorded in
`bench_out/hdr_remux_export/gate_log.json`:

```
ffmpeg -y -loglevel warning
  -i C:\Users\18392\AppData\Local\Temp\dlss_gpu_hdr_pgfk1zs2.mkv   # input 0: the encode
  -i F:\dlss5standaloneV2\bench_out\hdr_remux_export\hdr_remux_src.mkv   # input 1: the source
  -map 0:v:0 -c:v copy
  -map 1:a? -map 1:s? -map 1:t?
  -map_metadata 1 -map_chapters 1
  -c:a copy -shortest -c:s copy -c:t copy
  F:\dlss5standaloneV2\bench_out\hdr_remux_export\hdr_remux_out.mkv
```

That is the command the hardware run issued, and its fixture's *video* track carries no title,
no language and no disposition flag, so it has no descriptive clause. When a source does carry
one, the same builder puts `-metadata:s:v:0 title=…` / `language=…` / `-disposition:v:0 …` on the
end of that same argv — §8.4 read both forms back offline, with the video packet payload and the
in-band SEIs unchanged.

## 2. What the HDR gate accepts now, and what it still refuses

`negotiate_hdr` no longer refuses `copy_pass`. In its place it adds `hdr_mux_refusals(ts)`
(`gpu_hdr_transfer_export.py:724`) to the same reason list, evaluated **before** any frame is
decoded, so a request the pass cannot honour is refused while it is still free.

Refused at the gate:

* `audio_mode` outside `gtx.MUX_AUDIO_MODES`, `subtitle_mode` outside
  `gtx.MUX_SUBTITLE_MODES`. An unrecognised subtitle value would leave the muxer to pick a
  codec itself — a silent conversion, so it is named as a refusal instead.
* **Any non-empty `extra_args`.** The first version of this gate sorted the caller's arguments
  into safe and unsafe (`HDR_MUX_PROTECTED_OPTIONS`, plus a `shlex`-style tokeniser and a
  whole-token name match). The audit found that classifier was the wrong shape: `-bsf:v
  filter_units=remove_types=39|40` deletes the ST 2086 and CTA-861.3 SEIs this chain exists to
  preserve and names nothing on that list; `-filter_complex` rewrites the picture; a bare
  trailing token is a second output file; a quoting form can be made to slip past any denylist
  that has to be right every time. The stage writes one video stream it did not encode, so it
  carries no caller-written FFmpeg argument at all. `hdr_mux_refusals`
  (`gpu_hdr_transfer_export.py:725`) now refuses the field whole whenever
  `gtx.extra_args_request` reads anything from it, and the refusal message names the fields that
  *are* honoured (`audio_mode`, `subtitle_mode`, `copy_attachments`, `copy_metadata`,
  `copy_chapters`) so the operator is not left guessing.
* Absent, empty, whitespace-only and JSON `null` are all the same non-request and are not
  refused. Treating `extra_args=None` as the literal argument `None` was a live defect, not a
  hypothetical one: `needs_copy_pass` rewrapped a file for nothing and `build_mux_command`
  appended `shlex.split("None")` to the FFmpeg line. `extra_args_request`
  (`gpu_transfer_export.py:1048`) is the one normaliser, used by the admission test, the builder
  and the gate, so the three cannot disagree about what "no arguments" means.
* The SDR chain keeps its behaviour exactly: `build_mux_command` still appends a caller's
  `extra_args` last, and `needs_copy_pass` still treats a non-empty one as a reason to rewrap.
  The refusal is HDR-only, and both the Python branch tests and
  `tests/node/hdr_export_ui.test.ts` assert the SDR side is untouched.

Nothing about the mapping rules changed: the pass still maps `0:v:0` from the encode and `1:a?`
/ `1:s?` / `1:t?` from the source, so a caller cannot pull the original video into the
deliverable by any route the gate accepts.

Still refused, exactly as before this round: a container other than `.mkv`; HLG;
non-`smpte2084`, non-limited-range, non-BT.2020 or constant-luminance sources; 8-bit sources;
`hdr_to_sdr` and every tone-mapping preset; `custom_color_*` that names a value; a resized
output; a working extent above the source or not resolving through
`pipeline.resolve_working_dims`; a non-`rgba16f` Feature 18 session (no RGBA8 main colour);
`guidance_mode` outside {0, 1}; a flow provider without the device RGBA path; `multi_pass≠1`;
`reconstruction_mode` other than `single`; `export_backend` other than `gpu_transfer`;
`dlss_backend` other than `gpu_flow_d3d12`; more than one video stream in the source. There is
no fall-through to the pipe in any of these cases.

## 3. The run's shape

`export_gpu_hdr_transfer` (`gpu_hdr_transfer_export.py:1633`):

1. `assert_output_not_input(video, out_path)` — before the decoder opens anything.
2. If `negotiation["copy_pass"]`: `tempfile.mkstemp(suffix=".mkv", prefix="dlss_gpu_hdr_")`,
   close the handle, `os.unlink` it (the existing rule: mkstemp reserves the name, the encoder
   bridge wants to create the file), and encode into that path. Otherwise the encoder writes
   `out_path` directly, as before. **The caller's name is never the encoder's target when a
   remux is coming**, so a failed remux cannot leave a video-only file under it. The reserve is
   `_reserve_intermediate` (`gpu_hdr_transfer_export.py:1677`), it happens before the session is
   opened, and the wrapper's `finally` is what owns the returned name — see 5.
3. The frame loop, unchanged: same lock, same watchdog, same cancellation between frames, same
   shortfall rule, and the chain closes (encoder flushed and file released) in the `finally`
   that releases the loader cookies. The copy pass therefore reads a complete bitstream.
4. `cancel()` is checked again, then the shortfall rule, then `gtx.run_copy_pass(ffmpeg,
   intermediate, video, transcode_settings, out_path, with_audio, on_proc_started=…,
   on_proc_exited=…, video_descriptive_metadata=negotiation["hdr_video_stream_metadata"])` —
   the same hooks the queue passed into the export, so 终止任务 kills the remux instead of only
   setting a flag. `cancel()` is checked once more *after* the remux returns: a job cancelled
   while FFmpeg was rewriting the container is a cancellation, not a deliverable.
5. `finally`: unlink the intermediate (`_remove_intermediate`,
   `gpu_hdr_transfer_export.py:1694`). It goes whether the run succeeded, failed, timed out,
   was cancelled, was refused after the name was minted, threw out of the encoder bridge, or got
   all the way through verification; the unlink targets only the path this run
   minted, and a failure to remove it is printed, not turned into a false error. Nothing else is
   deleted, and the source and `out_path` are never that path. This is the shape the audit
   asked for — ownership of the temporary spans *construction through verification*, not the
   remux alone — and the outer wrapper is where it lives, so an early `return`, a `raise` or a
   `JobCancelled` between the reserve and the pass cannot escape it.
6. `report["hdr_copy_pass"]`, `hdr_mux_command` = the command that ran, then
   `pipeline.verify_export_output(out_path, frames, probe=video_info_uncached)` — on the
   deliverable, after the remux, never on the intermediate.

A non-zero FFmpeg exit is a `BridgeError` from `run_copy_pass` naming FFmpeg's stderr tail. The
exception propagates: there is no path in which a failed remux reports success, and no fallback
to delivering the encode alone (that file is missing the streams the job asked for, and it is
deleted anyway).

New report fields, so a run can say which shape it ran instead of leaving it to be guessed:
`hdr_copy_pass` (bool), `hdr_deliverable_written_by`
(`"ffmpeg_map_and_copy_pass"` | `"nvenc_hdr_bridge"`), `hdr_mux_command`,
`hdr_video_stream_metadata` (the fields this run put on the video track, or `null`), and the
`negotiation["hdr_encoder_contract_declared"]` / `hdr_fixed_policy["mux"]` wording.
`hdr_not_claimed.mux` and `.host_traffic` were rewritten to describe the two-stage truth; the
scalar counter readback is still reported as `hdr_scalar_report_readback_bytes_measured: null`
with its 256-byte-per-dispatch provenance read off the native constants, and
`hdr_pixel_readback_bytes_measured: null`. No counter was moved, re-defined or inflated.

## 4. Production dispatch and the client

`pipeline.export_gpu_transfer_stream` already handed an opted-in job to
`export_gpu_hdr_transfer_stream` and threaded `on_proc_started` / `on_proc_exited` / `lock` /
`cancel` through; nothing in `pipeline.py` or `dlss_service.py` changed this round, and the
production run below proves the queue's path reaches the remux. `skip_existing` and
`overwrite_output` are decided in `dlss_service.py:1225-1238` before dispatch, unchanged.

`frontend/src/services/hdrExport.ts`: the mapping refusals are gone; the container refusal
wording describes the reused pass; `copy_attachments` is no longer forced off. `hdrExtraArgsReason`
now mirrors the gate exactly — read the field, refuse it whole if it carries anything — and the
classifier it replaced is deleted, not left alongside: `HDR_MUX_PROTECTED_OPTIONS`, `shellTokens`
and `claimsProtectedOption` no longer exist in the TS, which
`test_the_mirror_refuses_extra_args_the_way_the_gate_does_and_for_the_same_reason` in
`tests/unit/test_hdr_ui_optin_settings.py` asserts by reading the service source. A mirror needs
one answer per input, and two parsers that can disagree is how the `=`-form hole was written the
first time. The blocklists that *are* the pass's own vocabulary — `HDR_MUX_AUDIO_MODES` and
`HDR_MUX_SUBTITLE_MODES` — stay mirrored against `gtx`, because the shared builder is the only
thing that reads them. `buildHdrExportPreset` keeps the user's `audio_mode`, `subtitle_mode`,
`copy_attachments`, `copy_metadata` and `copy_chapters`, clears any non-empty `extra_args` and
*says it cleared it* in the summary, and only patches a mapping value back when the *backend*
would refuse it. `exportPreview.ts` gained the seventh stage line (收尾封装: `-c:v copy`, no
second encode, the description written on the encoder's own stream without a `-map` of the source
video), states that the chapter count comes from `-show_chapters`, and that the temporary
intermediate is owned by the run for its whole life. The video track's title, language and flags
follow `copy_metadata`; there is no new control, and `HDR_LIMITATIONS` says so. The experimental
HDR10 label and the HLG / resize / multi-layer / `.mkv`-only restrictions are retained.
`DeliverPage.tsx` needed no change: it renders what those two services say.

## 5. The fixture, and what it carries

`bench_out/hdr_remux_export/hdr_remux_src.mkv`, built by one logged FFmpeg command
(`fixture_build.json`: argv, `exit_code: 0`) from the already validated HDR10 master
`build/hdr_probe/hdr_valid_src.mkv` with **`-c:v copy`** — so the pictures, the colour
declaration and the in-band SEIs under test are the validated master's own. Audio is a 2-second
sine against a 0.24-second video, deliberately longer, so the mux's `-shortest` ends on the
audio and cannot cut a picture.

| Stream | What the fixture says |
| --- | --- |
| 0 | hevc Main 10, 1920x1080, yuv420p10le, tv, bt2020nc / smpte2084 / bt2020, 25/1, 6 packets, 6 frames — and **no** stream title, no language and an all-zero disposition, which is why §8.4 works from a named temporary copy rather than from this row |
| 1 | aac LC, `language=jpn`, `title=Commentary`, disposition default, 95 packets, 2.021 s |
| 2 | subrip, `language=eng`, `title=Captions`, disposition original, 2 packets |
| 3 | attachment, `filename=…\fixture\note.txt`, `mimetype=text/plain`, 36 bytes |
| chapters | 2: `chapter one` 0→120000000, `chapter two` 120000000→240000000 at `1/1000000000` |
| format tags | `title=HDR remux fixture`, `COMMENT=global tag written by the fixture builder` |
| frame side data | Mastering display (35400/50000, 14600/50000, 8500/50000, 39750/50000, 6550/50000, 2300/50000, white 15635/16450, min 50/10000, max 10000000/10000) and Content light level (max_content 900, max_average 250) |

The mastering and CLL/FALL numbers were read out of the fixture rather than assumed, and they
are what the brief's expected payloads correspond to: SEI 137 `21349b46199608fc8a4839083d13404
20098968000000032` (24 bytes) and SEI 144 `038400fa` (MaxCLL 900, MaxFALL 250).

## 6. The one production export

`bench_out/hdr_remux_export/hdr_remux_gate.py`, one run, called the way `dlss_service` calls it:
`pipeline.export_gpu_transfer_stream(video=fixture, live=Live(960, 544, …, color_format="rgba16f"),
lock=threading.Lock(), frames=6, fps, cq=24, with_audio=True, out_path, transcode_settings,
job_settings, flow_provider=GuidanceServiceFlowProvider(nvof/4M, guidance_mode=1), progress,
on_proc_started, on_proc_exited, diagnostics_out)`.

Settings: `experimental_hdr_scrgb=True`, `export_backend=gpu_transfer`,
`dlss_backend=gpu_flow_d3d12`, `hdr_paper_white_scrgb=8.0` (fixed), `working_scale=0.5`,
`reconstruction_mode=single`, `multi_pass=1`, `guidance_mode=1`, NVOF 4M; transcode side
`hevc_nvenc` / `main10` / `p010le` / `container=mkv` / `color_preset=preserve` /
`audio_mode=copy` / `subtitle_mode=copy` / `copy_attachments=true` / `copy_metadata=true` /
`copy_chapters=true` / `extra_args=""`.

One export was driven. It returned, produced the deliverable, and the checklist that now reads
**35/35 PASS** was produced against that same output by `--recheck`, offline: the first pass
crashed inside the inspector (`TypeError` on a field the hook had logged as a one-element list)
after the export had already written its files. No second GPU run was needed and none was made.

Files under `bench_out/hdr_remux_export/`: `hdr_remux_gate.py` (reproduction script),
`fixture/` (the three source inputs), `fixture_build.json` (command + exit code),
`commands_and_inputs.json` (settings, hashes, library hashes, module hashes),
`diagnostics.json` (the chain's whole report + the hook's record),
`intermediate_video_only.mkv` (copied out by the hook while it still existed),
`hdr_remux_src.mkv`, `hdr_remux_out.mkv`, `demuxed_{master,fixture,intermediate,final}.265`,
`gate_log.json` (the checklist with every value it read), and `audit_repair/` — §8's script,
`audit_repair_log.json` and its `work/` directory. The last three are the only files in this
directory that the repair round wrote; everything above them is the hardware run's own record and
is byte-identical to what §7 describes.

Commands:

```
python bench_out/hdr_remux_export/hdr_remux_gate.py            # the one export, then the read-back
python bench_out/hdr_remux_export/hdr_remux_gate.py --recheck  # offline read-back only
```

## 7. What was measured

Hashes and sizes:

| File | sha256 (first 24) | bytes |
| --- | --- | --- |
| master `build/hdr_probe/hdr_valid_src.mkv` | `facd372c2e160c4a…` | 37 561 |
| fixture `hdr_remux_src.mkv` | `2ab2d5d8aa802f6f…` | 70 179 |
| encoder intermediate (video-only) | `24f4650101d3fd01…` | 27 978 |
| deliverable `hdr_remux_out.mkv` | `cd5089a16f34b58a…` | 32 878 |

The deliverable, read offline and uncached:

* one video, one audio, one subtitle, one attachment stream; **no second video stream**, and
  the video is the GPU encode, not the source (`fixture v:0 concat 99c0de370e189eb5` ≠
  `final 6035b69fa47e866c`).
* **6 video packets, 6 decoded frames**, presentation-order PTS
  `[0, 40, 80, 120, 160, 200]` strictly monotonic, 25/1, last frame ending at 0.240 s. Packets
  arrive in decode order and the list is sorted before the claim is made.
* video stream declares `hevc / Main 10 / yuv420p10le / tv / bt2020nc / smpte2084 / bt2020` at
  1920x1080 — the same declaration the encode wrote, read off the finished container.
* `-c:v copy` proved at the payload level: the intermediate's concatenated video packet payload
  hash `6035b69fa47e866c` **equals** the final's, and the per-packet hashes are the six frames'
  own (`36143561 023db348 2b79df49 bb8f1b6a b2fcfdae 2e7832c7`). The demuxed elementary streams
  are byte-equal in size (intermediate 27 224 = final 27 224) and both differ from the source's
  36 904. Every packet's `-show_data` dump was complete (`packets_read=117 ragged=[]`), so no
  hash comparison rests on a truncated read.
* audio is a prefix of the fixture's own compressed packets: 12 of the fixture's 95 AAC packets,
  payload-hash prefix exact — `-shortest` ended the sound at the picture, as intended, and did
  not touch a frame. Subtitle: both packets, payload identical (`3b71b714c63c712e` on both sides).
* language / title / disposition survived as the user set them: `audio=jpn/Commentary/default=1`,
  `subtitle=eng/Captions/original=1`.
* chapters: both, with titles and spans, at `1/1000000000`. Global tags: `title` and `COMMENT`
  are the fixture's (plus the muxer's own `ENCODER`). Attachment: the same `filename` string the
  fixture carries, `mimetype=text/plain`, and the 36 content bytes found in the finished
  container at offset 1067 (in the fixture at 3312).
* SEI equality in the **demuxed elementary stream**, not just the container: type 137 payload
  `21349b46199608fc8a4839083d1340420098968000000032` and type 144 payload `038400fa` are
  identical fixture → intermediate → final, and the fixture's are identical to the master's.
  The encode authored its own copies in-band: `encoder_in_bitstream_sei_written=true`,
  `sei_writer.nals_inserted=6`, `mastering_inserted=6`, `content_light_inserted=6`,
  `bytes_inserted=240`, `written_to=31`. The count of SEI occurrences differs (2 in the
  libx265 master and fixture, 6 in the NVENC encode: one per access unit) while the payload
  bytes do not — a difference in insertion frequency, not in declared values.
* container side data: Mastering display and Content light level dicts in the final are
  byte-equal to the fixture's.
* no fallback: `export_backend_actual=gpu_transfer`, `transfer_mode_actual=gpu_hdr_transfer`,
  `export_backend_fallback_reason=null`, `hdr_copy_pass=true`,
  `hdr_deliverable_written_by=ffmpeg_map_and_copy_pass`, `progress` fired 1…6,
  `hdr_working_dims=[960, 544]` at scale 0.5 on one `rgba16f` pass.
* host boundaries as claimed: all eight image-plane counters `0`,
  `hdr_pixel_readback_bytes_measured=null`,
  `hdr_scalar_report_readback_bytes_measured=null` (present-and-unmeasured, with the
  256-bytes-per-dispatch provenance and its source citation), and the decoder's scale counter
  `0` because the HDR open makes that stage impossible. The device-side counters answered with
  their real non-zero figures (e.g. 99 532 800 source-plane writes) — the checklist asks that
  they answered and stayed device-side, not that they were zero.
* mux traffic is stated, not hidden: `decode_bitstream_submitted_bytes_measured=34 432`, and the
  finished 32 878-byte file was read and rewritten through the host by FFmpeg's own buffers.
* the master and the fixture were not modified: their hashes after the run equal the hashes
  recorded before it. The queue's process ledger was empty at the end, so both hooks fired on a
  real handle and the intermediate was gone from `%TEMP%`.

One probe observation recorded in the first version of this document, because it is the origin of
§8.1: `probe_source` asked ffprobe's *format* section for `nb_chapters`, and this probe build does
not put that key there unless chapters are requested — so the chain's source facts said
`nb_chapters: 0` for a file with two chapters. It changed nothing in that run (the pass was
already admitted on `copy_attachments` and on the audio and subtitle streams, and
`build_mux_command` emits `-map_chapters 1` from the *setting*, not from the probe), but it was a
wrong number in a report field, and a chapter-only request would have skipped the pass entirely.
It is now closed: the HDR route opts into `-show_chapters` and reports the real count of 2, the
SDR route still reads the format field it always read, and neither behaviour is a guess — see
§8.1.

## 8. The audit repair, read back offline

The audit that followed the hardware run left four things open. Three were closed in code the
same day and the fourth was found to be a limit rather than a defect; all four are answered here
by running the production helpers over files that already exist, never by another GPU job.

`bench_out/hdr_remux_export/audit_repair/hdr_remux_audit_repair.py` drives no GPU, encodes
nothing and re-renders no picture. Its inputs are `hdr_remux_src.mkv` and
`intermediate_video_only.mkv` — the fixture and the video-only file the hardware gate's hook
copied out of `%TEMP%` — plus temporary `-c copy` copies of that same fixture. Its verdict is
**35 checks passed, 0 failed**, with one recorded limit, in
`audit_repair/audit_repair_log.json`: every FFmpeg argv with its exit code, every value read
back, the nine files it wrote (all under `audit_repair/work/`), and the sha256 of the three
modules that produced the answers — `gpu_transfer_export.py 13f76efb…`,
`gpu_hdr_transfer_export.py f8c397ff…`, `pipeline.py c1b751c5…`, so a reader can tell which
revision of the code a claim belongs to.

**This is not a hardware verification.** The 35-item checklist in `gate_log.json` was produced
by code that predates these four repairs and has not been rewritten or re-run; the repair round
changed no byte under `bench_out/hdr_remux_export/` (§8.5).

### 8.1 Chapters: the count the file states, not the field it omitted

| Read | Answer on the real fixture |
| --- | --- |
| `ffprobe -show_chapters` | 2 chapters, `chapter one` and `chapter two` |
| `ffprobe -show_format` → `format.nb_chapters` | **absent** — the key is not in this file's format section |
| `probe_source(video)` (SDR's own call, unchanged) | `nb_chapters: 0` |
| `probe_source(video, include_chapters=True, …)` (HDR's call) | `nb_chapters: 2` |

Asked of the production admission test with a chapters-only request
(`audio_mode=remove`, `subtitle_mode=drop`, no attachments, no metadata, `copy_chapters=true`):
**True** on the opted-in probe and **False** on the default one. That second answer is the defect
recorded, not a passing state — a job that only asked for chapters used to be handed straight to
the encoder and delivered without them, silently. The opted-in route then ran for real:
`work/chapters_only.mkv` carries both chapter markers with `-an -sn`, no attachment stream and no
format title, which is the request honoured and nothing else.

*Limit, recorded rather than fixed.* With `copy_metadata=false` the shared builder writes
`-map_metadata -1`, and FFmpeg takes the chapter **names** with the global metadata while leaving
the **markers**: 2 chapters, titles `None`. The script compares the argv for that request against
`gtx.build_mux_command` called the SDR chain's way and the two are identical up to the output
path, so this is the shared pass's long-standing semantics that the HDR route inherits by reusing
it — changing it would change what SDR deliverables have always contained, which this round may
not do. `copy_metadata=true` (the default) keeps `chapter one` / `chapter two`, also verified.

### 8.2 The intermediate's lifecycle, across the whole run

Not measurable here: this script cannot allocate a GPU encoder, and a mocked `Popen` would prove
nothing about a file the real chain created. The claim is carried by the unit tests, which can
inject each way out of the run and see the name gone in every one:
`test_a_cancel_after_the_first_frame_removes_the_intermediate`,
`test_a_cancel_between_the_encode_and_the_rewrap_leaves_no_intermediate`,
`test_a_finish_that_raises_still_removes_the_intermediate`,
`test_a_close_that_raises_still_removes_the_intermediate`,
`test_a_decoder_open_that_raises_still_removes_the_intermediate`,
`test_a_chain_that_cannot_be_built_still_removes_the_intermediate`,
`test_a_report_that_cannot_be_aggregated_still_removes_the_intermediate`,
`test_a_verification_failure_still_removes_the_intermediate`,
`test_an_encode_that_left_no_intermediate_refuses_before_ffmpeg_is_named`,
`test_a_rewrap_that_exits_non_zero_is_a_failed_job_and_leaves_nothing_behind`. Two more pin the
other half of the shape: `test_a_copy_pass_job_encodes_into_a_unique_intermediate_and_delivers_the_rewrap`
asserts the encoder's target is never the caller's name and that the intermediate does not
outlive the run, and `test_the_rewrap_starts_only_after_the_encoder_has_released_the_intermediate`
asserts FFmpeg is not even named until the bridge has given the file back.

### 8.3 `extra_args`: the stage carries none

19 argument lists, all refused: the two bitstream-filter forms
(`-bsf:v filter_units=remove_types=39|40`, which deletes the SEIs this chain exists to preserve),
`-filter_complex`, a re-encode in both option forms, a mapping clause in both forms, a colour
clause, a pixel format, `-an`, `-y`, `-f mp4` in both forms, a bare second output filename, a
quoted protected clause, a `#`-commented payload, a padded form, a tab-wrapped form, and a
metadata clause the pass owns nothing about. 5 non-values are not refused: `""`, whitespace,
newline, tab and `None` — all of them the same request, which is none.

The SDR side, checked in the same run: `build_mux_command` still appends a caller's
`extra_args`, `-bsf:v` lands on its line and the output path is still last.

### 8.4 The source video's own description, on the deliverable

The hardware fixture's **video** stream carries no title, no language and no disposition flag —
measured, not assumed — so the claim could not be made from the hardware evidence and a second
input was built: the same fixture, `-c copy`, with `-metadata:s:v:0 title=Master shot`,
`language=jpn` and a disposition, written into `audit_repair/work/`. The pictures, the colour
declaration and the in-band SEIs therefore stay the encoder bridge's own bytes throughout.

| Source video track | Derived request | Clause the pass ran | Delivered track |
| --- | --- | --- | --- |
| title + `jpn`, disposition all-zero | `{title, language, disposition: []}` | `-metadata:s:v:0 title=Master shot -metadata:s:v:0 language=jpn -disposition:v:0 -default` | `title=Master shot`, `language=jpn`, flags `[]` |
| title + `jpn`, `default`+`original` | `{title, language, disposition: [default, original]}` | `… -disposition:v:0 +default+original` | same title/language, flags `[default, original]` |
| same source, clause withheld (control) | — | — | tags `[DURATION]` only — anonymous, which is the gap |

* `ENCODER` is **not** restated: the source says `Lavc61.19.101 libx265`, the deliverable's video
  stream says nothing. `DURATION` does appear on the deliverable, but the Matroska muxer writes it
  for the stream it just wrote; it is not a copied source tag, which is why it also shows up in
  the control.
* The all-zero row is the one that needed thought. A source that states no flags is a statement,
  and `default` is what a muxer assumes about a track nobody described — so the pass says
  `-default` outright (FFmpeg's disposition option takes `+flag`/`-flag` and rejects
  `default=0`, which this round discovered by being refused). Without that clause the deliverable
  happens to be correct today because this encoder's intermediate carries no flags; with it, it is
  correct because the chain said so.
* Nothing here is a new control. `copy_video_stream_metadata` is an internal default derived from
  the job's existing `copy_metadata` (on unless the operator turned it off) and can be named
  directly if a caller wants it while leaving format metadata alone; it is in
  `HDR_MUX_ONLY_KEYS`, so it never reaches the encoder plan as an unknown key and never becomes a
  refusal the operator has to understand. `report["hdr_video_stream_metadata"]` says which fields
  went out.
* **The bitstream did not move.** For both rows: the video packet payload hash of the deliverable
  equals the encoder intermediate's (`6035b69fa47e866c…5d950b`), SEI type 137 payload
  `21349b46199608fc8a4839083d1340420098968000000032` and type 144 payload `038400fa` are present
  and byte-equal after the rewrap, and the audio, subtitle, attachment and two chapters all still
  arrive with the titles and languages the fixture carries.

### 8.5 What did not move

The nine raw hardware artifacts hashed before the script starts and after it finishes are
byte-identical, the file listing of `bench_out/hdr_remux_export/` is identical (nothing added,
nothing removed), and all nine files this script wrote are inside `audit_repair/work/`. The
fixture, the intermediate and the deliverable of the hardware run are read-only inputs to this
document as they are to that check.

```
python bench_out/hdr_remux_export/audit_repair/hdr_remux_audit_repair.py
```

## 9. Cancellation, overwrite and cleanup semantics

* Cancel before or during the frame loop: `pipeline.JobCancelled`, as before; no remux runs, and
  the `finally` removes the intermediate, so nothing appears under the caller's name.
* Cancel between the encode and the remux: checked explicitly before `run_copy_pass`.
* Cancel during the remux: the mux process is registered with the queue's own hooks, so
  `_kill_active_procs` terminates it; the job then fails and the intermediate is removed. The
  partial `out_path` is not verified and is not reported as a success.
* Timeout: `gtx.MUX_TIMEOUT_SEC` (1800 s) → the process is killed and a `BridgeError` naming the
  timeout is raised; the intermediate still goes.
* `skip_existing` / `overwrite_output=false`: decided in `dlss_service.add_job`/the queue before
  dispatch (unchanged); an HDR job is no different from an SDR one here.
* `out_path == source`: refused twice over — by `export_gpu_hdr_transfer` before anything opens,
  and by `build_mux_command`/`assert_output_not_input` for the remux's own input/output pair.
* A pre-existing `out_path` is overwritten by FFmpeg's `-y` only after the encode succeeded,
  which is the SDR chain's existing behaviour for a job the queue allowed to run.
* Cleanup deletes exactly one path: the `dlss_gpu_hdr_*` name this run minted. An
  `OSError` there is printed and does not change the job's outcome.

## 10. Test results

| Run | Result |
| --- | --- |
| **Repair round**, targeted Python: `test_hdr_transfer_export_branch.py`, `test_gpu_transfer_export_backend.py`, `test_hdr_ui_optin_settings.py`, `test_render_queue_ui_contract.py`, `test_gui_gpu_transfer_selftest.py`, `test_output_overwrite_guard.py` | **439 passed, 0 failed** (12.7 s) |
| **Repair round**, `node --test tests/node/hdr_export_ui.test.ts tests/node/export_preview.test.ts tests/node/working_scale.test.ts` | **46 passed, 0 failed** |
| **Repair round**, `npm --prefix frontend run build` (`tsc -b && vite build`) / `run lint` | built clean / 0 errors, the same 4 pre-existing `set-state-in-effect` and `refs` warnings in `DLSSPage.tsx` and `DeliverPage.tsx` (the round's TS edits are in `services/`, and none fire there) |
| **Repair round**, `audit_repair/hdr_remux_audit_repair.py` | **35 checks passed, 0 failed, 1 recorded limit** |
| *Earlier round, not re-run here:* `tests/unit` (full) | 1157 passed, 9 failed — all 9 in `test_gpu_host_motion_wrapper.py`, `AttributeError: 'Live' object has no attribute 'bytes_per_pixel'` at `dlss_gpu_engine.py:422`. Pre-existing on the baseline and unrelated: neither that test file nor `dlss_gpu_engine.py` appears in `git diff --name-only`. The brief for the repair round forbade a full unit run, so this figure is the earlier round's. |
| *Earlier round, not re-run here:* `npx playwright test` specs 17/07/08 | 5 passed against the built bundle. Spec 17's only `extra_args` assertion is `toBe('')`, which the new rule does not change; the repair round verified the build but did not start a browser. |

Tests changed by the repair round, all in files that already existed:

* the whole-lifecycle cleanup family in `test_hdr_transfer_export_branch.py` (§8.2), plus
  `test_the_run_hands_the_video_track_description_to_the_pass_that_writes_it`;
* the chapter facts: `test_the_hdr_entry_asks_the_probe_for_the_facts_the_encode_cannot_carry`,
  `test_the_sdr_entry_still_asks_the_probe_for_exactly_what_it_always_asked` (the injected fake
  now takes the two opt-ins and asserts the HDR call passes both, and that SDR passes neither),
  `test_a_chapter_only_request_is_a_copy_pass_request_and_not_a_silent_drop`;
* `extra_args`: `test_any_extra_arg_at_all_is_refused_before_the_gpu_work` replaces the
  safe/unsafe parametrization, and the None-as-a-request defect is pinned;
* the descriptive-metadata set in the same file plus
  `test_a_source_that_states_no_video_flags_does_not_inherit_the_muxers_own`,
  `test_the_source_encoders_technical_tags_are_never_repeated_as_this_runs`,
  `test_the_descriptive_clause_names_the_encoded_stream_and_the_sdr_command_is_unchanged`
  (the last of these is what holds the SDR argv still);
* the mirror: `test_the_mirror_refuses_extra_args_the_way_the_gate_does_and_for_the_same_reason`
  asserts the deleted parser stays deleted — no `HDR_MUX_PROTECTED_OPTIONS`, no `shellTokens`, no
  `claimsProtectedOption`, exactly one `return null` in `hdrExtraArgsReason`, and no second
  tokeniser on either side of the seam;
* Node: `this stage carries no caller-written FFmpeg arguments at all`, covering 20 argument lists
  and 6 empty values.

One unrelated stale pin was repaired earlier in this work and is still in place:
`tests/unit/test_render_queue_ui_contract.py::test_the_preview_switches_on_the_selected_chain`
compared the `useMemo` dependency text against a list `DeliverPage.tsx` had already outgrown at
the baseline (verified with `git show HEAD:frontend/src/pages/DeliverPage.tsx`); the pin now
matches the real multi-line dependency list.

## 11. Limits of what this proves

* **One** HDR job with a copy pass ran on real hardware, six frames, one 50 % pass, cq 24. The
  audio-conversion modes (`aac`/`opus`/`flac`/`pcm`), `subtitle_mode=convert` and the
  `copy_metadata=false` / `copy_chapters=false` off-switches are covered by mocked
  command-building tests plus §8's offline FFmpeg reads, not by another GPU run. `extra_args` is
  no longer in that list at all: nothing is accepted, so nothing accepted is unproven. The brief
  forbade per-setting GPU runs; they remain unproven on hardware.
* **No hardware evidence was produced after these four repairs.** The repair round ran no GPU, no
  A/B, no roundtrip and no benchmark, so §6 and §7 describe the pre-repair code. The repairs that
  could change a container (chapter probing, the descriptive clauses) were read back offline
  against the pre-repair encoder output, which is the strongest thing available without a GPU run
  and is weaker than a fresh end-to-end one. Nothing here claims the post-repair chain was
  verified on hardware, and the 35-item `gate_log.json` is left as the record of the code that
  produced it.
* The video-track description is proved on a **temporary named copy** of the fixture, because the
  hardware fixture's video stream carries no title, language or flag to preserve. It is not proved
  on a clip somebody actually exported, and `wants_video_descriptive_metadata` follows
  `copy_metadata`, so a job with `copy_metadata=false` and no explicit naming gets no description
  either — which is what turning metadata off has always meant.
* Chapter **names** are lost by `copy_metadata=false` (§8.1). The markers survive, the count is
  right, and the behaviour is the shared builder's in SDR too; naming it here is the whole
  response, since fixing it would change SDR deliverables.
* Performance: **NOT_CHARACTERIZED**, unchanged from the top of this document. No elapsed time of
  any FFmpeg or GPU process in §8 is offered as a throughput or latency figure.
* `.mkv` only. No other container is offered, and the refusal is a refusal, not a rename.
* HDR10/PQ only. HLG, full-range, non-BT.2020 primaries, CL matrix, 8-bit sources, resized HDR
  output, multi-layer reconstruction and multi-pass remain unsupported and refused. The label
  stays experimental.
* Payload equality was read from FFmpeg's own `-show_data` dumps and demuxed elementary streams,
  which is software decoding of the *container* for inspection, not a host boundary inside the
  chain. Nothing here reads a picture back to make a claim about one.
* No claim that the deliverable is correct on a HDR display: no conformance claim, no quality
  claim, no benchmark. The mastering/CLL numbers are proven to survive as declared bytes, which
  is a different and weaker statement than a graded picture.
* Repeated-run behaviour of the temp-name allocation, disk-full and permission-denied cleanup
  paths, and a source whose attachment stream is unreadable are exercised only by the mocked
  suites.

## 12. Open items

Nothing in the chain is failing, and the `nb_chapters` gap this section used to list is closed for
the HDR route (§8.1) without moving the SDR route's answer. Three things remain, none of them
introduced by these repairs:

* The 9 pre-existing `test_gpu_host_motion_wrapper.py` failures, which need `Live.bytes_per_pixel`
  in `dlss_gpu_engine.py` — a Feature 18 host-engine change outside this round's stated scope,
  left untouched and unfixed rather than papered over.
* Chapter names under `copy_metadata=false` (§8.1). Closing it means changing what the *SDR* copy
  pass writes when it is told not to copy metadata, so it is named as a limit and left.
* One post-repair hardware run, when the brief allows one. Everything above §9 that concerns the
  container was checked offline; the encode itself was not re-run since the repairs.
