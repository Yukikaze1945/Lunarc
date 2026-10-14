# Preview, RAFT and Chinese terminology optimization

Source: user request 2026-10-10. Implement directly in the main agent; no Qoder or subagents.

## Scope
Preserve Swiss UI, independent neural layers, SDR10/HDR10 processing precision, remux and export semantics. Display optimization affects preview only. No unverified CUDA context sharing, silent fallback or GPL implementation copying.

## Sequence
1. Record current-tree short hardware baseline with matched fixtures and settings: cold setup separately; decode, completed GPU processing, display readback/conversion, JPEG encoding, WS serialization/send, Electron decode/paint and seek/setting response. Host wall scopes must not be mislabeled GPU kernel timestamps.
2. Bound preview work to one active request and latest pending request; revisions discard stale results without cancelling an in-flight native operation. Separate forward decode from temporal reset; avoid GOP rewinds during frame skipping. Reuse display buffers and skip unused original readback. Reuse sessions for runtime NR parameter changes with explicit history reset; retain rebuild for structural changes. Bound display caches and preserve actual PTS.
3. Binary JPEG transport and decode/paint acknowledgement, latest-frame UI presentation, viewport display size separate from full processing resolution, remove duplicate seek effects and reconnect correctly.
4. Inspect actual DLSS5Tool and pinned Magpie 0.6.9 language resources. Implement RAFT long-edge 512/720/1080, validated iterations, current-to-previous flow and separate-axis restoration, bounded normalized tensor reuse. Investigate device interop against actual CUDA/D3D ownership; ship only verified device behavior and disclose remaining copies.
5. Align Chinese terms, add accessible circle help with short original explanations, preserve parameter meaning and basic/Advanced hierarchy.
6. Same-fixture after measurements; direction/scale/cache/lifecycle/stale-response tests, full units, frontend build, real current Electron smoke. Recheck affected SDR/HDR preview precision and export boundaries.

## Evidence and limits
Raw data under ignored bench_out/preview_raft_20261010; acceptance document contains commands, settings, sample count, p50/p95 and limitations. Short measurements only; no claims about general quality or long-run throughput. Missing optional weights/runtime does not stop independent preview/localization work. Preserve unrelated working-tree files.
