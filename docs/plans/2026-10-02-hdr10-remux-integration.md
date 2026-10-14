# Experimental HDR10: reuse the existing FFmpeg remux path

## Scope and baseline

Baseline: main 7195f8979583fc48b4f47654f09841e82025cec4. Codex owns task decomposition and review; Qoder owns implementation. The user authorizes ordinary repair and continuation without phase approvals.

Reconnect the already verified experimental single-layer HDR10 GPU video export to the project's existing FFmpeg mapping and final verification. Deliver a final MKV carrying the encoded Main10 video and requested source streams and metadata. Do not change HDR math, Feature18, NVOF, multiscale algorithms, native DLLs, or SDR remux semantics. No benchmark; performance remains NOT_CHARACTERIZED. No commit/push requested for this phase.

## Existing integration points

- gpu_hdr_transfer_export.py: negotiate_hdr currently refuses copy_pass jobs; export_gpu_hdr_transfer writes its deliverable directly through HdrChain.
- gpu_transfer_export.py: needs_copy_pass, build_mux_command and the existing intermediate/video-copy orchestration are the primary reuse targets.
- pipeline.py: build_transcode_command defines ordinary mapping policy; assert_output_not_input and verify_export_output protect and verify delivery.
- frontend/src/services/hdrExport.ts, exportPreview.ts and pages/DeliverPage.tsx: remove only obsolete mux restrictions and describe the real two-stage path.
- tests/unit/test_hdr_transfer_export_branch.py and test_gpu_transfer_export_backend.py: negotiation/orchestration and existing SDR regression tests.

## Implementation sequence

1. Audit existing audio/subtitle mapping, chapters, metadata, attachment and container rules. Reuse shared command construction, not a parallel audio/mux implementation.
2. Permit supported HDR copy_pass requests. Produce a unique video-only MKV intermediate, finish and close the GPU encoder before invoking existing FFmpeg remux with -c:v copy. Keep final MKV only as the supported HDR output container this phase.
3. Preserve source audio preferences/with_audio precedence, subtitles, chapters, global/stream metadata, language/disposition and MKV attachments. Preserve encoded HDR video colour declaration and mastering/content-light/SEI through the remux. Never map the original video into the output or introduce another video encode.
4. Implement correct output collision/overwrite handling, cancellation, subprocess hooks, failure reporting and temporary-file cleanup. Verify the final file using uncached probing; do not report intermediate creation as success. Do not silently drop requested components or fall back to SDR.
5. Update only UI restrictions and tests that described the previous video-only limitation. Retain experimental HDR label and other genuine HDR restrictions.

## Verification

- Targeted CPU/mock tests cover accepted copy_pass, command reuse, requested mappings, no mux when unnecessary, overwrite/source alias safeguards, cancel/failure/cleanup, and final verification. Existing SDR remux tests must keep passing.
- One six-frame valid HDR10 fixture, based on the validated PQ source, with one audio track, subtitle, chapter, global and stream tags, and a small attachment. One real GPU export through the normal production dispatch, followed by remux. Do not re-run prior HDR A/B/roundtrip gates.
- Offline inspect final MKV: six video frames/packets, expected PTS in presentation order, full dimensions, HEVC Main10/yuv420p10le/tv/bt2020nc/smpte2084/bt2020, correct stream mapping/count/codecs/tags/dispositions/chapter timestamps, attachment bytes, source audio/subtitle packet payload preservation in copy mode.
- Compare mastering and content-light metadata and actual demuxed HEVC SEI payloads before/after remux and against source. Container side data alone is insufficient.
- Preserve no-fallback and measured image-host-boundary evidence; compressed bitstream remux is allowed host traffic. Scalar report readbacks remain present and unmeasured, not zero.
- Save commands, exit codes, hashes, ffprobe output, diagnostics, checks and failures. Correct offline inspector errors using the same output; only re-export if implementation failure requires it.

## Completion

Codex reviews actual diff and artifacts rather than accepting Qoder exit 0. Ordinary build/test/incomplete failures receive a minimal Qoder continuation. Stop only for a demonstrated external capability blocker. Final report contains reused remux code, final streams, HDR metadata preservation, SDR regression results, and container limits.
