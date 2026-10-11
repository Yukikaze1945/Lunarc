# -*- coding: utf-8 -*-
"""One real 4K export at working_scale=1.0 through the production entry point.

Not a pytest module: this needs the GPU, the NVOF hardware and the five real
bridges. It is the correctness gate the phase document asks for at the identity
chain - a short 4K, 8-bit SDR, NVOF 4M, NVENC export run once - and it answers
the question the 50% gate cannot: whether the chain really skips the two
exact-2x stages at 1.0 rather than running them and not saying so.

What it holds the run to, in order:

* the negotiation accepts 1.0 and the export reports the GPU backend it was asked
  for, so no part of the result can be read as the legacy pipe;
* the stage that is not on the path has no number: the working-colour and
  residual fields stay ``None``, because no bridge opened to count them, while
  the boundaries that did run answer with their own counters. The same split is
  checked on every frame's own ledger row, which is where a skipped stage has to
  be absent for all of them rather than merely uncounted in total;
* every host image figure the chain claims is a measured zero next to the frames
  it counted over - a decode that moved no pixels to host memory, a DLSS host
  that staged and read back nothing, an encoder that took no host pixels;
* the file itself: requested frame count, packet count, 3840x2160, 8-bit 4:2:0
  limited range, the matrix the conversion used, and presentation timestamps
  that step by one frame duration;
* the process is left clean: no ``dlss_gpu_transfer_*`` intermediate, and the
  DLSS session the chain borrowed is closed.

One export per invocation, and no timing conclusions: the wall-clock per-stage
columns the pipeline prints are not a benchmark, and nothing here repeats the
run to average them. The 50% chain, the cancellation pass and the legacy
comparability pass are ``gpu_transfer_export_gate.py``'s business.

    python tests/unit/gpu_transfer_export_fullscale_gate.py
    python tests/unit/gpu_transfer_export_fullscale_gate.py src.mkv out.mkv 8
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import dlss_gpu_engine  # noqa: E402
import gpu_transfer_export as gtx  # noqa: E402
import optical_flow  # noqa: E402
import pipeline  # noqa: E402
import transfer_profiler  # noqa: E402

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "bench_out/fixture_4k_gpu15.mp4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "bench_out/gpu15_gpu_transfer_ws10.mkv"
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 8

JOB_SETTINGS = {
    "preset": 1, "style": 0, "intensity": 1.0, "local_tone": 1.0,
    "local_struct": 1.0, "skin_struct": 1.0, "use_auto_mask": 1,
    "ui_correction": 0, "motion_scale_x": 1.0, "motion_scale_y": 1.0,
    "guidance_mode": 1, "flow_engine": "nvof", "flow_quality": "4M",
    "working_scale": 1.0, "residual_pyramid": True, "residual_strength": 1.0,
    "multi_pass": 1, "dlss_backend": "gpu_flow_d3d12",
    "export_backend": "gpu_transfer",
}
TRANSCODE = {
    "codec": "av1_nvenc", "preset": "p7", "container": "mkv",
    "audio_mode": "copy", "subtitle_mode": "copy", "b_frames": "2",
    "lookahead": "8", "decode_mode": "cuda",
}
# The two exact-2x bridges, and the only fields that may read None on this chain.
SKIPPED_FIELDS = (
    "working_color_device_bytes_measured", "working_color_process_count_measured",
    "working_color_source_device_copy_bytes_measured",
    "working_color_scratch_device_write_bytes_measured",
    "residual_full_device_copy_bytes_measured",
    "residual_working_device_copy_bytes_measured",
    "residual_output_readback_bytes_measured",
)
# Every field that claims pixels moved into or out of the process. On the
# identity chain each one has a counter behind it, so a zero is a measurement.
HOST_FIELDS = (
    "decode_host_copy_bytes_measured", "decode_host_copy_count_measured",
    "dlss_host_color_stage_bytes_measured", "dlss_host_color_stage_count_measured",
    "dlss_output_readback_bytes_measured", "encoder_host_pixel_bytes_measured",
)

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
    return bool(condition)


n, fps, w, h = pipeline.video_info(VIDEO)
print(f"[ws1.0] source {VIDEO}: {n} frames @ {fps} {w}x{h}")
source = gtx.probe_source(VIDEO)
probe_keys = ("pix_fmt", "color_range", "color_space", "color_transfer",
              "r_frame_rate", "audio_streams")
print("[ws1.0] probe: " + json.dumps({k: source.get(k) for k in probe_keys},
                                     ensure_ascii=False))
check(FRAMES <= n, f"{FRAMES} frames asked, the fixture holds {n}")
check(source.get("pix_fmt") == "yuv420p",
      f"pix_fmt={source.get('pix_fmt')}: the gate is for 8-bit 4:2:0 SDR")
check(source.get("color_range") == "tv",
      f"color_range={source.get('color_range')}: limited range is the contract")
check(str(source.get("color_transfer")) not in ("smpte2084", "arib-std-b67"),
      "HDR transfer: the chain carries 8-bit SDR pixels")
source_audio = [s.get("codec_name") for s in json.loads(subprocess.check_output([
    pipeline._bundle_ffprobe(), "-v", "error", "-print_format", "json",
    "-show_streams", "-select_streams", "a", VIDEO]).decode("utf-8", "replace"))
    .get("streams", [])]

live_w, live_h = pipeline.resolve_working_dims(w, h, JOB_SETTINGS["working_scale"])
print(f"[ws1.0] working {live_w}x{live_h} (working_scale=1.0)")
check((live_w, live_h) == (w, h),
      f"working_scale 1.0 resolved to {live_w}x{live_h}, not the source {w}x{h}")

service = optical_flow.get_frame_guidance_service("nvof", "4M")
check(bool(service.supports_device_rgba_input()),
      "the NVOF service has no device RGBA input, so the run could not stay on the GPU")
flow_provider = pipeline.GuidanceServiceFlowProvider(service, guidance_mode=1)

temps_before = set(glob.glob(os.path.join(tempfile.gettempdir(), "dlss_gpu_transfer_*")))
if os.path.exists(OUT):
    os.unlink(OUT)

live = dlss_gpu_engine.Live(live_w, live_h, JOB_SETTINGS)
diagnostics, seen, ledger = {}, [], transfer_profiler.TransferProfiler()


def progress(current, total, message):
    seen.append(current)


try:
    finished = pipeline.export_dlss_stream(
        video=VIDEO, live=live, frames=FRAMES, fps=fps, cq=24, with_audio=True,
        flow_provider=flow_provider, progress=progress, out_path=OUT,
        transcode_settings=TRANSCODE, job_settings=JOB_SETTINGS,
        diagnostics_out=diagnostics, transfer_collector=ledger)
finally:
    live.close()

print("[ws1.0] diagnostics:")
print(json.dumps(diagnostics, indent=2, default=str, ensure_ascii=False))

# --- what the run says it did -------------------------------------------------
check(finished == OUT and os.path.isfile(OUT), "the export returned no deliverable")
check(os.path.getsize(OUT) > 0, "the deliverable is empty")
check(len(seen) == FRAMES, f"progress reported {len(seen)} frames, {FRAMES} were asked")
check(diagnostics.get("export_backend_actual") == "gpu_transfer",
      f"export_backend_actual={diagnostics.get('export_backend_actual')!r}: the gate "
      "must not read as the legacy pipe")
check(diagnostics.get("export_backend_fallback_reason") is None,
      f"the run fell back: {diagnostics.get('export_backend_fallback_reason')!r}")
check(diagnostics.get("gpu_transfer_working_scale") == 1.0,
      f"the run reported scale {diagnostics.get('gpu_transfer_working_scale')!r}")
check(diagnostics.get("gpu_transfer_residual_backend") == "none",
      f"residual_backend={diagnostics.get('gpu_transfer_residual_backend')!r}, "
      "but no residual stage exists at 1.0")

# --- the skipped stages have no number, and the run stages have theirs --------
for field in SKIPPED_FIELDS:
    value = diagnostics.get(field)
    check(value is None,
          f"{field} is {value!r}: a stage that never opened cannot report a count, "
          "and a zero here would be a number nobody measured")

declared_matrix = diagnostics.get("gpu_transfer_color_matrix")
frame_rgba = w * h * 4
check(declared_matrix in ("bt709", "smpte170m"),
      f"gpu_transfer_color_matrix={declared_matrix!r}")
check(diagnostics.get("decode_frames_decoded_measured") == FRAMES,
      f"decode_frames_decoded_measured={diagnostics.get('decode_frames_decoded_measured')!r}")
check(diagnostics.get("decode_device_frame_count_measured") == FRAMES,
      "the decoder did not count every frame it handed out as a device surface")
check(diagnostics.get("nv12_to_rgba_convert_count_measured") == FRAMES,
      f"nv12_to_rgba_convert_count_measured="
      f"{diagnostics.get('nv12_to_rgba_convert_count_measured')!r}")
check(diagnostics.get("nv12_to_rgba_device_bytes_measured") == FRAMES * frame_rgba,
      f"nv12_to_rgba_device_bytes_measured="
      f"{diagnostics.get('nv12_to_rgba_device_bytes_measured')!r}, "
      f"expected {FRAMES} x {frame_rgba}")
check(diagnostics.get("dlss_device_color_copy_bytes_measured") == FRAMES * frame_rgba,
      "the DLSS host did not copy the full-size plane it was handed, per frame")
check(diagnostics.get("encoder_frames_submitted_measured") == FRAMES,
      f"encoder_frames_submitted_measured="
      f"{diagnostics.get('encoder_frames_submitted_measured')!r}")
check(diagnostics.get("encoder_staging_device_copy_bytes_measured")
      == FRAMES * frame_rgba,
      "the encoder's stage did not copy every full-size texture it was handed")
# The engine's own running total, read through the same accessor the chain reads:
# the report's figure has to be the engine's number, not a size the gate computed.
expected_flow = flow_provider.device_input_counters()[1]
check(diagnostics.get("nvof_device_input_bytes_total") == expected_flow,
      "the report's flow figure is not the engine's own total")
check((diagnostics.get("nvof_device_input_copy_count_total") or 0) >= FRAMES,
      f"nvof_device_input_copy_count_total="
      f"{diagnostics.get('nvof_device_input_copy_count_total')!r}: flow did not read "
      "the device plane")
check(bool(expected_flow),
      "the NVOF engine counted no device input bytes, so the full-size plane it was "
      "handed is not evidenced by this run")
check(diagnostics.get("working_color_bytes_logical") == frame_rgba,
      "the identity report no longer labels the working colour as the full plane")
check(diagnostics.get("decode_nv12_bytes_logical") == w * h * 3 // 2,
      "the logical NV12 size is derived from the source extent")

for field in HOST_FIELDS:
    value = diagnostics.get(field)
    check(value == 0, f"{field} is {value!r}, not a measured zero - "
                      "a frame crossed into or out of the process")

# --- what every frame's own ledger row says ----------------------------------
# The session report carries the run's figures; the per-frame row is what names
# the backends that ran, and it is the only place a skipped stage can be shown
# absent for every single frame rather than merely uncounted in total.
rows = ledger.records
check(len(rows) == FRAMES, f"{len(rows)} per-frame ledger rows for {FRAMES} frames")
for index, row in enumerate(rows):
    for field, wanted in (("export_backend_actual", "gpu_transfer"),
                          ("transfer_mode_actual", "gpu_transfer"),
                          ("dlss_backend_actual", "gpu_flow_d3d12"),
                          ("decode_actual", "cuda_nvdec_bridge"),
                          ("residual_backend", "none")):
        check(row.get(field) == wanted,
              f"frame {index}: {field}={row.get(field)!r}, expected {wanted!r}")
    for absent in ("working_color", "residual"):
        check(absent not in row,
              f"frame {index}: the ledger has a {absent!r} column, and the identity "
              "chain has no such stage to time")
    for ran in ("nv12_to_rgba", "nvof_flow", "dlss_inference", "nvenc_submit"):
        check(isinstance(row.get(ran), (int, float)),
              f"frame {index}: no timing for {ran}, a stage this chain does run")
    check(bool(row.get("guidance_backend")),
          f"frame {index}: the ledger does not name the flow backend that ran")

steady_bytes = [row.get("nvof_device_input_bytes") for row in rows[1:]]
check(all(v == frame_rgba for v in steady_bytes),
      f"a steady-state frame handed NVOF {steady_bytes}, not one full-size "
      f"{frame_rgba} B plane each")
# Frame 0's own delta is the one row the chain cannot measure: the engine reports
# nothing before its first call, so the row omits the field rather than guessing a
# value for it. Its share is still known, because the session total is the engine's
# own cumulative counter - which is what makes the residue below arithmetic rather
# than an assumption.
first_delta = rows[0].get("nvof_device_input_bytes")
check(first_delta in (None, 2 * frame_rgba),
      f"frame 0's flow delta is {first_delta}, which is neither the unreadable "
      f"first call nor one plane plus the reset ({2 * frame_rgba} B)")
total_copies = diagnostics.get("nvof_device_input_copy_count_total") or 0
check(total_copies - (FRAMES - 1) == 2,
      f"the engine counted {total_copies} copies over {FRAMES} frames; the identity "
      "chain owes one plane per frame plus one for the reset the first frame asks "
      f"for, so the first frame's share is {total_copies - (FRAMES - 1)}")
check(diagnostics.get("nvof_device_input_bytes_total") == total_copies * frame_rgba,
      f"{total_copies} copies x {frame_rgba} B is "
      f"{total_copies * frame_rgba}, not the {diagnostics.get('nvof_device_input_bytes_total')} "
      "B the report gave: a copy that was not a full-size plane")

ledger_pts = [row.get("flow_pts") for row in rows]
check(all(isinstance(p, int) for p in ledger_pts),
      f"the decoder PTS is not an integer grid: {ledger_pts}")
check(all(b > a for a, b in zip(ledger_pts, ledger_pts[1:])),
      f"presentation order reached the encoder out of order: {ledger_pts}")

# --- the file itself ----------------------------------------------------------
dump = json.loads(subprocess.check_output([
    pipeline._bundle_ffprobe(), "-v", "error", "-print_format", "json",
    "-show_streams", "-show_entries", "packet=pts_time", "-count_packets",
    "-select_streams", "v:0", OUT]).decode("utf-8", "replace"))
stream = dump["streams"][0]
pts = sorted({round(float(p["pts_time"]), 4) for p in dump.get("packets", [])})
audio = json.loads(subprocess.check_output([
    pipeline._bundle_ffprobe(), "-v", "error", "-print_format", "json",
    "-show_streams", "-select_streams", "a", OUT]).decode("utf-8", "replace"))
frame_duration = 1001.0 / 30000.0
facts = {
    "codec": stream.get("codec_name"),
    "size": f"{stream.get('width')}x{stream.get('height')}",
    "pix_fmt": stream.get("pix_fmt"), "range": stream.get("color_range"),
    "matrix": stream.get("color_space"), "transfer": stream.get("color_transfer"),
    "primaries": stream.get("color_primaries"),
    "packets": int(stream.get("nb_read_packets") or 0),
    "distinct_pts": len(pts), "first_pts": pts[0] if pts else None,
    "last_pts": pts[-1] if pts else None,
    "max_pts_gap_error": (max(abs(gap - frame_duration) for gap in
                           (b - a for a, b in zip(pts, pts[1:]))) if len(pts) > 1 else None),
    "audio": [s.get("codec_name") for s in audio.get("streams", [])],
}
print(f"[ws1.0] file facts: {json.dumps(facts, ensure_ascii=False)}")

check(facts["packets"] == FRAMES,
      f"{facts['packets']} packets in the file, {FRAMES} frames were asked")
check(facts["distinct_pts"] == FRAMES,
      f"{facts['distinct_pts']} distinct presentation timestamps for {FRAMES} frames")
check(facts["size"] == f"{w}x{h}",
      f"the file is {facts['size']}, the source is {w}x{h}")
check(facts["pix_fmt"] == "yuv420p" and facts["range"] == "tv",
      f"the file is {facts['pix_fmt']}/{facts['range']}, not 8-bit 4:2:0 limited")
expected_matrix = "bt709" if declared_matrix == "bt709" else ("smpte170m", "bt470bg")
check(facts["matrix"] in (expected_matrix if isinstance(expected_matrix, tuple)
                          else (expected_matrix,)),
      f"the file declares matrix {facts['matrix']!r} against the negotiated "
      f"{declared_matrix!r}")
check(facts["primaries"] == facts["transfer"] or facts["primaries"] is None,
      f"the file declares primaries {facts['primaries']!r} / transfer "
      f"{facts['transfer']!r}")
if facts["max_pts_gap_error"] is not None:
    check(facts["max_pts_gap_error"] <= 0.002,
          f"presentation timestamps step by {frame_duration:.6f} s at worst "
          f"{facts['max_pts_gap_error']:.6f} s off")
check(facts["audio"] == source_audio,
      f"the file carries {facts['audio']} audio, the source carries {source_audio}")

# --- the process is left clean ------------------------------------------------
temps_after = set(glob.glob(os.path.join(tempfile.gettempdir(), "dlss_gpu_transfer_*")))
left = sorted(temps_after - temps_before)
check(not left, f"intermediate temp files left behind: {left}")
check(not live.is_alive(),
      "the DLSS session the chain borrowed is still open after the export")

print(json.dumps({"ok": not failures, "frames": FRAMES, "out": OUT,
                  "diagnostics_subset": {k: diagnostics.get(k) for k in (
                      "export_backend_actual", "gpu_transfer_working_scale",
                      "gpu_transfer_residual_backend", "gpu_transfer_color_matrix",
                      "gpu_transfer_frame_rate", "gpu_transfer_copy_pass",
                      "decode_frames_decoded_measured",
                      "nv12_to_rgba_device_bytes_measured",
                      "encoder_packets_written_measured",
                      "nvof_device_input_copy_count_total")},
                  "ledger_rows": len(rows),
                  "ledger_frame0_nvof_bytes": first_delta,
                  "ledger_decoder_pts": ledger_pts,
                  "file_facts": facts}, indent=2, ensure_ascii=False))
if failures:
    print(f"[ws1.0] FAIL ({len(failures)}):")
    for failure in failures:
        print("  -", failure)
    raise SystemExit(1)
print(f"[ws1.0] PASS: {FRAMES} frames at {w}x{h} through the identity chain - the "
      f"working-colour and residual stages report no number because none opened, "
      f"every host image figure is a measured zero, and nothing was left behind.")
