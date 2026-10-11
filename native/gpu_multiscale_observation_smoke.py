"""Short hardware gate for independent 50% and 75% observations or fusion."""

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
import transfer_profiler


VIDEO = os.path.join(ROOT, "tests", "fixtures", "fixture_gpu_transfer_short.mkv")
OUTPUT = os.path.join(tempfile.mkdtemp(prefix="dlss5_dual_observation_"),
                      "dual_observation.mkv")
SETTINGS = {
    "preset": 1, "style": 0, "intensity": 1.0, "local_tone": 1.0,
    "local_struct": 1.0, "skin_struct": 1.0, "use_auto_mask": 1,
    "ui_correction": 0, "motion_scale_x": 1.0, "motion_scale_y": 1.0,
    "guidance_mode": 1, "flow_engine": "nvof", "flow_quality": "4M",
    "working_scale": 0.50, "residual_pyramid": True,
    "residual_strength": 1.0, "multi_pass": 1,
    "dlss_backend": "gpu_flow_d3d12", "export_backend": "gpu_transfer",
    "reconstruction_mode": "dual", "layer_scales": [0.50, 0.75],
    "layer_weights": [1.0, 1.0], "fusion_mode": "catmull_baseline",
}
TRANSCODE = {"codec": "av1_nvenc", "preset": "p7", "container": "mkv",
             "audio_mode": "copy", "subtitle_mode": "copy", "b_frames": "2",
             "lookahead": "8", "decode_mode": "cuda"}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    fusion = mode in ("fusion", "triple", "full_layer", "zero_motion")
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    settings = dict(SETTINGS)
    if fusion:
        settings["fusion_mode"] = "data_consistency_experimental"
        if len(sys.argv) > 3:
            settings["fusion_max_iterations"] = int(sys.argv[3])
            settings["fusion_relative_tolerance"] = 1e-8
    if mode == "triple":
        settings["reconstruction_mode"] = "triple"
        settings["working_scale"] = 0.4
        settings["layer_scales"] = [0.4, 0.67, 0.85]
        settings["layer_weights"] = [1.0, 1.0, 1.0]
    if mode == "full_layer":
        settings["layer_scales"] = [0.5, 1.0]
    if mode == "zero_motion":
        settings["guidance_mode"] = 0
    _, fps, width, height = pipeline.video_info(VIDEO)
    low_width, low_height = pipeline.resolve_working_dims(
        width, height, settings["working_scale"])
    provider = None
    if settings["guidance_mode"]:
        service = optical_flow.get_frame_guidance_service("nvof", "4M")
        provider = pipeline.GuidanceServiceFlowProvider(service, guidance_mode=1)
    live = dlss_gpu_engine.Live(low_width, low_height, settings)
    report = {}
    try:
        pipeline.export_dlss_stream(
            video=VIDEO, live=live, frames=frames, fps=fps, cq=24,
            with_audio=False, flow_provider=provider, out_path=OUTPUT,
            transcode_settings=TRANSCODE, job_settings=settings,
            diagnostics_out=report,
            transfer_collector=transfer_profiler.TransferProfiler())
    finally:
        live.close()
    selected = {key: report.get(key) for key in (
        "export_backend_actual", "layer_count", "layer_scales",
        "layer_working_dims", "layer_session_ids", "layer_reset_flags",
        "layer_dlss_calls", "fusion_mode", "decode_host_copy_bytes_measured",
        "dlss_host_color_stage_bytes_measured",
        "dlss_output_readback_bytes_measured",
        "residual_output_readback_bytes_measured",
        "encoder_host_pixel_bytes_measured", "encoder_packets_written_measured",
        "fusion_scalar_readback_bytes_measured")}
    print(json.dumps({"output": OUTPUT, "result": selected}, indent=2))
    assert selected["export_backend_actual"] == "gpu_transfer"
    count = len(settings["layer_scales"])
    assert selected["layer_count"] == count
    assert len(set(selected["layer_session_ids"])) == count
    assert selected["layer_dlss_calls"] == [frames] * count
    assert selected["layer_reset_flags"] == [[True] + [False] * (frames-1)] * count
    assert selected["fusion_mode"] == settings["fusion_mode"]
    assert selected["encoder_packets_written_measured"] == frames
    for key in ("decode_host_copy_bytes_measured",
                "dlss_host_color_stage_bytes_measured",
                "dlss_output_readback_bytes_measured",
                "residual_output_readback_bytes_measured",
                "encoder_host_pixel_bytes_measured"):
        assert selected[key] == 0, (key, selected[key])
    if fusion:
        assert selected["fusion_scalar_readback_bytes_measured"] > 0
    ffprobe = pipeline._bundle_ffprobe()
    probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:packet=pts",
        "-show_packets", "-show_streams", "-of", "json", OUTPUT],
        check=True, capture_output=True, text=True, timeout=30)
    metadata = json.loads(probe.stdout)
    stream = metadata["streams"][0]
    pts = [int(packet["pts"]) for packet in metadata["packets"]]
    assert (stream["width"], stream["height"]) == (width, height)
    assert len(pts) == frames and all(a < b for a, b in zip(pts, pts[1:]))
    print(json.dumps({"full_resolution": [width, height],
                      "packet_pts": pts, "pts_monotonic": True}))


if __name__ == "__main__":
    main()
