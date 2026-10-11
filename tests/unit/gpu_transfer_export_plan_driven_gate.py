# -*- coding: utf-8 -*-
"""Run one short 4K export through the production gpu_transfer chain at a plan-driven scale.

Not a test: this needs the GPU, NVOF hardware and the real bridges. It asks the
questions the plan-driven branch exists for - whether the gate classifies the
scale as plan-driven, whether the chain then runs to completion, whether the
transfer report still says no image frame crossed a host copy, whether every
frame reached the encoder on its own presentation timestamp, and whether the run
left no device or temp file behind.

    python tests/unit/gpu_transfer_export_plan_driven_gate.py
    python tests/unit/gpu_transfer_export_plan_driven_gate.py src.mkv out.mkv 4 0.67

The scale defaults to 0.75, the ratio this gate was first written against, and is
checked against gpu_transfer_export.plan_driven_scale before the export starts so
that a typo is a refusal naming the range this backend serves rather than four frames of the
wrong chain. The frame count defaults to four: enough to show the two plan-driven
bridges hold the table they uploaded across calls, and no more device work than
that. A refusal is a valid outcome and is printed with the gate's own reasons.
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import dlss_gpu_engine
import gpu_transfer_export as gtx
import optical_flow
import pipeline
import transfer_profiler

# The temp intermediate the copy-pass chain creates and, on success, unlinks
# after the mux. Its prefix is the whole signature: a run that leaks one has
# left a device-produced file nobody asked to keep.
INTERMEDIATE_PREFIX = "dlss_gpu_transfer_"


def leaked_intermediates():
    return {name for name in os.listdir(tempfile.gettempdir())
            if name.startswith(INTERMEDIATE_PREFIX)}

TAG = "gate"
SCALE = float(sys.argv[4]) if len(sys.argv) > 4 else 0.75
VIDEO = sys.argv[1] if len(sys.argv) > 1 else "bench_out/fixture_4k_gpu15.mp4"
OUT = (sys.argv[2] if len(sys.argv) > 2 else
       f"bench_out/gpu15_gpu_transfer_{int(round(SCALE * 100)):03d}.mkv")
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 4

JOB_SETTINGS = {
    "preset": 1, "style": 0, "intensity": 1.0, "local_tone": 1.0,
    "local_struct": 1.0, "skin_struct": 1.0, "use_auto_mask": 1,
    "ui_correction": 0, "motion_scale_x": 1.0, "motion_scale_y": 1.0,
    "guidance_mode": 1, "flow_engine": "nvof", "flow_quality": "4M",
    "working_scale": SCALE, "residual_pyramid": True, "residual_strength": 1.0,
    "multi_pass": 1, "dlss_backend": "gpu_flow_d3d12",
    "export_backend": "gpu_transfer",
}
TRANSCODE = {
    "codec": "av1_nvenc", "preset": "p7", "container": "mkv",
    "audio_mode": "copy", "subtitle_mode": "copy", "b_frames": "2",
    "lookahead": "8", "decode_mode": "cuda",
}


def probe_color(path):
    """The container's own idea of this stream's extent and colour marking.

    nvenc_bridge stamps the range and the matrix-derived primaries/transfer on
    every frame it encodes, so what ffprobe reads back is the marker the
    deliverable actually carries rather than the one the report intended.
    """
    ffprobe = pipeline._bundle_ffprobe()
    if not ffprobe:
        return None
    res = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,codec_name,pix_fmt,color_range,color_space,"
        "color_primaries,color_transfer", "-of", "json", path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        timeout=30)
    streams = json.loads(res.stdout).get("streams") or []
    return streams[0] if streams else None


def log(message):
    print(f"[{TAG}] {message}", flush=True)


n, fps, w, h = pipeline.video_info(VIDEO)
live_w, live_h = pipeline.resolve_working_dims(w, h, SCALE)
served = gtx.plan_driven_scale(w, h, live_w, live_h, SCALE)
if served is None:
    raise SystemExit(f"{SCALE:g} at {live_w}x{live_h} is not a percentage the "
                     "plan-driven stages serve (a whole percentage from "
                     f"{gtx.WORKING_PERCENT_MIN} to {gtx.WORKING_PERCENT_MAX}, "
                     "resolved to a working size below the source on both axes); "
                     "the export below would not have been the chain this gate "
                     "exists to check")
taps = gtx.plan_tap_widths(w, h, live_w, live_h)
frames = min(FRAMES, n)
log(f"source {VIDEO}: {n} frames @ {fps} {w}x{h}")
log(f"working {live_w}x{live_h} at scale={SCALE}, exporting {frames}")
log(f"plan taps: working_color={taps[0]} (window {gtx.LANCZOS_MAX_TAPS}), "
    f"upscale={taps[1]} (window {gtx.RESIDUAL_MAX_TAPS})")

if os.path.exists(OUT):
    os.unlink(OUT)
temp_before = leaked_intermediates()
service = optical_flow.get_frame_guidance_service("nvof", "4M")
flow_provider = pipeline.GuidanceServiceFlowProvider(service, guidance_mode=1)
live = dlss_gpu_engine.Live(live_w, live_h, JOB_SETTINGS)
diagnostics, seen, collector = {}, [], transfer_profiler.TransferProfiler()
try:
    finished = pipeline.export_dlss_stream(
        video=VIDEO, live=live, frames=frames, fps=fps, cq=24, with_audio=True,
        flow_provider=flow_provider,
        progress=lambda current, total, message: seen.append(current),
        cancel=lambda: False, out_path=OUT, transcode_settings=TRANSCODE,
        job_settings=dict(JOB_SETTINGS), diagnostics_out=diagnostics,
        transfer_collector=collector)
finally:
    live.close()

log(f"finished: {finished} frames reported: {len(seen)}")
log(f"exists: {os.path.exists(OUT)} size: {os.path.getsize(OUT)}")
log(f"resampling: {diagnostics.get('gpu_transfer_resampling')} "
    f"working: {diagnostics.get('gpu_transfer_working_dims')} "
    f"scale: {diagnostics.get('gpu_transfer_working_scale')}")
log("report:")
print(json.dumps({key: value for key, value in sorted(diagnostics.items())},
                 indent=2, default=str, ensure_ascii=False))

# The claims this gate exists for. Every one is a bridge's own counter, so a
# missing answer reads None and fails here rather than passing as a zero.
HOST_FIELDS = ("decode_host_copy_count_measured", "decode_host_copy_bytes_measured",
               "dlss_host_color_stage_bytes_measured",
               "dlss_output_readback_bytes_measured",
               "residual_output_readback_bytes_measured",
               "encoder_host_pixel_bytes_measured")
log("host-side counters: " + json.dumps(
    {key: diagnostics.get(key) for key in HOST_FIELDS}))
log("device-side counters: " + json.dumps(
    {key: diagnostics.get(key) for key in (
        "nvof_device_input_copy_count_total", "nvof_device_input_bytes_total",
        "working_color_device_bytes_measured",
        "working_color_source_device_copy_bytes_measured",
        "working_color_scratch_device_write_bytes_measured",
        "residual_full_device_copy_bytes_measured",
        "residual_working_device_copy_bytes_measured",
        "encoder_staging_device_copy_bytes_measured")}, ensure_ascii=False))
assert diagnostics.get("export_backend_actual") == "gpu_transfer", diagnostics
assert diagnostics.get("export_backend_fallback_reason") is None
assert diagnostics.get("gpu_transfer_resampling") == "plan_driven"
assert diagnostics.get("gpu_transfer_working_scale") == served
assert diagnostics.get("gpu_transfer_working_dims") == [live_w, live_h]
assert diagnostics.get("working_color_bytes_logical") == live_w * live_h * 4
for key in HOST_FIELDS:
    assert diagnostics.get(key) == 0, (key, diagnostics.get(key))
packets = diagnostics.get("encoder_packets_written_measured")
submitted = diagnostics.get("encoder_frames_submitted_measured")
log(f"encoder: submitted={submitted} packets={packets}")
assert submitted == frames and packets == frames, (
    f"the deliverable is {frames} frames of work; {submitted} submitted and "
    f"{packets} written is not that")

# Presentation timestamps: every frame the encoder accepted must have arrived on
# its own clock tick, or the deliverable is a slideshow regardless of how clean
# the byte path was. flow_pts is the decode timestamp the chain hands straight
# to nvenc_encode, so this is the timeline the file is built on.
pts = [r.get("flow_pts") for r in collector.records]
log(f"ledger rows: {len(pts)} flow_pts: {pts}")
assert len(pts) == frames, f"{len(pts)} ledger rows for {frames} frames"
assert all(v is not None for v in pts), pts
step = {b - a for a, b in zip(pts, pts[1:])}
assert pts[0] == 0 and step and min(step) > 0 and len(step) == 1, (
    f"presentation timestamps must start at 0 and advance by one uniform "
    f"positive step; got {pts} (steps {sorted(step)})")

out_n, out_fps, out_w, out_h = pipeline.video_info(OUT)
marker = probe_color(OUT)
log(f"delivered {OUT}: {out_n} frames @ {out_fps} {out_w}x{out_h}")
log("delivered stream: " + json.dumps(marker, ensure_ascii=False))
assert (out_w, out_h) == (w, h), "the residual stage owns the output extent"
assert marker["color_space"] == diagnostics.get("gpu_transfer_color_matrix"), (
    f"the encoder wrote {marker['color_space']} while the report says "
    f"{diagnostics.get('gpu_transfer_color_matrix')}")
assert marker["color_range"] == "tv", marker["color_range"]

# Cleanup: the copy-pass chain creates a temp intermediate and unlinks it after
# the mux; Chain.close drops every bridge and the DLSS session in its own slot,
# and live.close in the finally above is idempotent over that. A leaked temp from
# this run is the observable half of that claim - a device-produced file left in
# the system temp dir that nobody asked to keep.
leaked = leaked_intermediates() - temp_before
log(f"temp intermediates leaked by this run: {sorted(leaked)}")
assert not leaked, f"the copy pass left {sorted(leaked)} behind"
log(f"gpu_transfer_plan_driven_{served:g}_export=COMPLETE "
    f"working={live_w}x{live_h} frames={frames} packets={packets} "
    f"pts={pts[0]}+{next(iter(step))}n cleanup=clean")
