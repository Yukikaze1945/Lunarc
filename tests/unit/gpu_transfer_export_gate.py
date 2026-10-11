# -*- coding: utf-8 -*-
"""Run the opt-in GPU-transfer export once, end to end, and print what it reports.

Not a test: this needs the GPU, NVOF hardware and the real bridges. It is the gate
the phase document asks for - one short 4K export at 50% working scale with NVOF 4M,
through the production entry point - plus the cancellation pass, which has to leave
no deliverable and no intermediate behind.

    python tests/unit/gpu_transfer_export_gate.py                       # export + cancel
    python tests/unit/gpu_transfer_export_gate.py src.mkv out.mkv 14 legacy_out.mkv

Passing the fourth argument adds the comparability pass: the same fixture through the
legacy pipe, then ffprobe facts for both files and per-frame PSNR against it.
"""
import glob
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import dlss_gpu_engine
import optical_flow
import pipeline

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "bench_out/fixture_4k_gpu15.mp4"
OUT = sys.argv[2] if len(sys.argv) > 2 else "bench_out/gpu15_gpu_transfer.mkv"
FRAMES = int(sys.argv[3]) if len(sys.argv) > 3 else 14
# Naming a second output asks the gate for the comparability pass too: the same
# fixture through the legacy pipe, then the two files measured against each other.
LEGACY_OUT = sys.argv[4] if len(sys.argv) > 4 else None

JOB_SETTINGS = {
    "preset": 1, "style": 0, "intensity": 1.0, "local_tone": 1.0,
    "local_struct": 1.0, "skin_struct": 1.0, "use_auto_mask": 1,
    "ui_correction": 0, "motion_scale_x": 1.0, "motion_scale_y": 1.0,
    "guidance_mode": 1, "flow_engine": "nvof", "flow_quality": "4M",
    "working_scale": 0.5, "residual_pyramid": True, "residual_strength": 1.0,
    "multi_pass": 1, "dlss_backend": "gpu_flow_d3d12",
    "export_backend": "gpu_transfer",
}
TRANSCODE = {
    "codec": "av1_nvenc", "preset": "p7", "container": "mkv",
    "audio_mode": "copy", "subtitle_mode": "copy", "b_frames": "2",
    "lookahead": "8", "decode_mode": "cuda",
}

n, fps, w, h = pipeline.video_info(VIDEO)
print(f"[gate] source {VIDEO}: {n} frames @ {fps} {w}x{h}")
live_w, live_h = pipeline.resolve_working_dims(w, h, JOB_SETTINGS["working_scale"])
print(f"[gate] working {live_w}x{live_h}")

service = optical_flow.get_frame_guidance_service("nvof", "4M")
print("[gate] NVOF device rgba input supported:", service.supports_device_rgba_input())
flow_provider = pipeline.GuidanceServiceFlowProvider(service, guidance_mode=1)


def run(out_path, frames, cancel_after=None):
    """One export through the production entry point; returns its diagnostics."""
    live = dlss_gpu_engine.Live(live_w, live_h, JOB_SETTINGS)
    diagnostics, seen = {}, []

    def progress(current, total, message):
        seen.append(current)

    def cancel():
        return cancel_after is not None and len(seen) >= cancel_after

    try:
        result = pipeline.export_dlss_stream(
            video=VIDEO, live=live, frames=frames, fps=fps, cq=24, with_audio=True,
            flow_provider=flow_provider, progress=progress, cancel=cancel,
            out_path=out_path, transcode_settings=TRANSCODE,
            job_settings=JOB_SETTINGS, diagnostics_out=diagnostics)
    finally:
        live.close()
    return result, diagnostics, seen


temps_before = set(glob.glob(os.path.join(tempfile.gettempdir(),
                                          "dlss_gpu_transfer_*")))
if os.path.exists(OUT):
    os.unlink(OUT)
finished, diagnostics, seen = run(OUT, min(FRAMES, n))
print("[gate] finished:", finished, "frames reported:", len(seen))
print("[gate] exists:", os.path.exists(OUT), "size:", os.path.getsize(OUT))
print("[gate] diagnostics:")
print(json.dumps(diagnostics, indent=2, default=str, ensure_ascii=False))

# Cancellation: 14 frames requested, stopped after 4. Nothing may be delivered.
CANCEL_OUT = "bench_out/gpu15_cancelled.mkv"
if os.path.exists(CANCEL_OUT):
    os.unlink(CANCEL_OUT)
try:
    run(CANCEL_OUT, min(FRAMES, n), cancel_after=4)
    print("[cancel] UNEXPECTED: the cancelled export returned successfully")
except pipeline.JobCancelled as cancelled:
    print("[cancel] JobCancelled:", cancelled)
print("[cancel] deliverable left behind:", os.path.exists(CANCEL_OUT))
temps_after = set(glob.glob(os.path.join(tempfile.gettempdir(), "dlss_gpu_transfer_*")))
print("[cancel] intermediate temp files left:", sorted(temps_after - temps_before))


if LEGACY_OUT:
    import subprocess

    import numpy as np

    import dlss_gpu_engine

    print("\n[compare] legacy pipe export ->", LEGACY_OUT)
    # The comparability arm is the other export chain, not another DLSS host: both
    # chains draw on native/dlss_gpu_host.dll now that dlssnr_host.dll is retired.
    legacy_settings = {k: v for k, v in JOB_SETTINGS.items() if k != "export_backend"}
    legacy_live = dlss_gpu_engine.Live(live_w, live_h, legacy_settings)
    legacy_provider = pipeline.GuidanceServiceFlowProvider(service, guidance_mode=1)
    legacy_diag = {}
    try:
        pipeline.export_dlss_stream(
            video=VIDEO, live=legacy_live, frames=min(FRAMES, n), fps=fps, cq=24,
            with_audio=True, flow_provider=legacy_provider, out_path=LEGACY_OUT,
            transcode_settings=TRANSCODE, job_settings=dict(legacy_settings),
            diagnostics_out=legacy_diag)
    finally:
        legacy_live.close()
    print("[compare] legacy backend:", json.dumps(
        {k: legacy_diag.get(k) for k in ("export_backend_actual",
                                         "export_backend_fallback_reason",
                                         "transfer_mode_actual",
                                         "dlss_backend_actual")}, ensure_ascii=False))

    def facts(path):
        """Container facts the two files have to agree on, read with ffprobe."""
        dump = json.loads(subprocess.check_output([
            pipeline._bundle_ffprobe(), "-v", "error", "-print_format", "json",
            "-show_streams", "-show_entries", "packet=pts_time", "-count_packets",
            "-select_streams", "v:0", path]).decode("utf-8", "replace"))
        stream = dump["streams"][0]
        pts = [round(float(p["pts_time"]), 4) for p in dump.get("packets", [])]
        audio = json.loads(subprocess.check_output([
            pipeline._bundle_ffprobe(), "-v", "error", "-print_format", "json",
            "-show_streams", "-select_streams", "a", path]).decode("utf-8", "replace"))
        return {
            "file": os.path.basename(path), "codec": stream.get("codec_name"),
            "size": f"{stream.get('width')}x{stream.get('height')}",
            "pix_fmt": stream.get("pix_fmt"), "range": stream.get("color_range"),
            "matrix": stream.get("color_space"), "transfer": stream.get("color_transfer"),
            "primaries": stream.get("color_primaries"),
            "packets": int(stream.get("nb_read_packets") or 0),
            "first_pts": pts[0] if pts else None, "last_pts": pts[-1] if pts else None,
            "pts_deltas": sorted({round(b - a, 3) for a, b in zip(pts, pts[1:])}) if pts else [],
            "audio": [a.get("codec_name") for a in audio.get("streams", [])],
        }

    for label, path in (("legacy", LEGACY_OUT), ("gpu", OUT)):
        print(f"[{label}] {json.dumps(facts(path), ensure_ascii=False)}")

    def rgb_frames(path, count):
        """Decode to rgb24 with the same settings for both files."""
        raw = subprocess.run([pipeline._bundle_ffmpeg(), "-v", "error", "-i", path,
                              "-frames:v", str(count), "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE).stdout
        _, _, fw, fh = pipeline.video_info(path)
        frame = fw * fh * 3
        return [np.frombuffer(raw[i * frame:(i + 1) * frame], dtype=np.uint8)
                .reshape(fh, fw, 3) for i in range(min(count, len(raw) // frame))]

    def psnr(a, b):
        diff = a.astype(np.int32) - b.astype(np.int32)
        mse = float(np.mean(diff.astype(np.float64) ** 2))
        return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)

    legacy_rgb = rgb_frames(LEGACY_OUT, FRAMES)
    gpu_rgb = rgb_frames(OUT, FRAMES)
    print(f"[picture] compared {min(len(legacy_rgb), len(gpu_rgb))} decoded frames. "
          "A wrong YUV matrix moves every channel mean by tens of code values, so the "
          "means are printed rather than folded into the PSNR.")
    print("[picture] frame | psnr(gpu,legacy) | mean R,G,B legacy | gpu")
    for i in range(min(len(legacy_rgb), len(gpu_rgb))):
        means = [" ".join(f"{c:6.2f}" for c in x[i].reshape(-1, 3).mean(axis=0))
                 for x in (legacy_rgb, gpu_rgb)]
        print(f"[picture] {i:2d}  | {psnr(legacy_rgb[i], gpu_rgb[i]):9.2f} | "
              f"{means[0]} | {means[1]}")
