"""One single-layer export against the pre-multiscale 14-frame baseline."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dlss_gpu_engine
import optical_flow
import pipeline

VIDEO = os.path.join(ROOT, "bench_out", "fixture_4k_gpu15.mp4")
GOLDEN = os.path.join(ROOT, "bench_out", "gpu15_gpu_transfer.mkv")
OUTPUT = os.path.join(tempfile.mkdtemp(prefix="dlss5_single_regression_"),
                      "single.mkv")
SETTINGS = {
    "preset": 1, "style": 0, "intensity": 1.0, "local_tone": 1.0,
    "local_struct": 1.0, "skin_struct": 1.0, "use_auto_mask": 1,
    "ui_correction": 0, "motion_scale_x": 1.0, "motion_scale_y": 1.0,
    "guidance_mode": 1, "flow_engine": "nvof", "flow_quality": "4M",
    "working_scale": 0.5, "residual_pyramid": True,
    "residual_strength": 1.0, "multi_pass": 1,
    "dlss_backend": "gpu_flow_d3d12", "export_backend": "gpu_transfer",
}
TRANSCODE = {"codec": "av1_nvenc", "preset": "p7", "container": "mkv",
             "audio_mode": "copy", "subtitle_mode": "copy", "b_frames": "2",
             "lookahead": "8", "decode_mode": "cuda"}


def video_hash(path):
    stream = subprocess.check_output([pipeline._bundle_ffmpeg(), "-v", "error",
        "-i", path, "-map", "0:v:0", "-c:v", "copy", "-f", "ivf", "-"],
        timeout=120)
    return hashlib.sha256(stream).hexdigest()


def main():
    frames, fps, width, height = pipeline.video_info(VIDEO)
    low_width, low_height = pipeline.resolve_working_dims(width, height, 0.5)
    service = optical_flow.get_frame_guidance_service("nvof", "4M")
    provider = pipeline.GuidanceServiceFlowProvider(service, guidance_mode=1)
    live = dlss_gpu_engine.Live(low_width, low_height, SETTINGS)
    report = {}
    try:
        pipeline.export_dlss_stream(
            video=VIDEO, live=live, frames=frames, fps=fps, cq=24,
            with_audio=True, flow_provider=provider, out_path=OUTPUT,
            transcode_settings=TRANSCODE, job_settings=SETTINGS,
            diagnostics_out=report)
    finally:
        live.close()
    old, new = video_hash(GOLDEN), video_hash(OUTPUT)
    facts = {"baseline": GOLDEN, "output": OUTPUT,
             "frames": frames, "baseline_video_sha256": old,
             "new_video_sha256": new, "identical": old == new,
             "export_backend_actual": report.get("export_backend_actual")}
    print(json.dumps(facts, indent=2))
    assert facts["identical"]
    assert report["export_backend_actual"] == "gpu_transfer"


if __name__ == "__main__":
    main()
