"""Short real SDR10 GPU-transfer export through DLSSService._execute_job.

Requires the project's Windows CUDA/D3D12/NVENC runtime. This is an acceptance
gate, not a benchmark; output and JSON evidence are kept in a fresh directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _probe(path, *args):
    import pipeline
    return json.loads(subprocess.check_output(
        [pipeline._bundle_ffprobe(), "-v", "error", *args, "-of", "json", str(path)],
        text=True, encoding="utf-8"))


def _packets(path, selector):
    result = _probe(path, "-select_streams", selector, "-show_packets",
                    "-show_data_hash", "sha256",
                    "-show_entries", "packet=size,data_hash,pts,duration")
    return result.get("packets", [])


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_result(job, diagnostics, streams, video_frames, audio_source, audio_output,
                    expected_frames):
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    _require(job.get("precision_mode") == "sdr10_bt709",
             f"service precision_mode={job.get('precision_mode')!r}")
    _require(job.get("dlss_session_color_format") == "rgba16f",
             f"DLSS session format={job.get('dlss_session_color_format')!r}")
    _require(diagnostics.get("export_backend_actual") == "gpu_transfer",
             f"actual export backend={diagnostics.get('export_backend_actual')!r}")
    _require(diagnostics.get("transfer_mode_actual") == "gpu_sdr10_transfer",
             f"actual transfer mode={diagnostics.get('transfer_mode_actual')!r}")
    _require(diagnostics.get("sdr10_frames_processed_measured") == expected_frames,
             f"processed frames={diagnostics.get('sdr10_frames_processed_measured')!r}")
    _require(len(video_frames) == expected_frames,
             f"output frames={len(video_frames)}, expected {expected_frames}")
    _require(video.get("codec_name") == "hevc" and video.get("profile") == "Main 10",
             f"unexpected output codec/profile: {video.get('codec_name')}/{video.get('profile')}")
    _require(video.get("pix_fmt") == "yuv420p10le",
             f"unexpected output pixel format: {video.get('pix_fmt')}")
    _require((video.get("color_primaries"), video.get("color_transfer"),
              video.get("color_space")) == ("bt709", "bt709", "bt709"),
             "output color tags are not BT.709: " + repr({k: video.get(k) for k in
                 ("color_primaries", "color_transfer", "color_space")}))
    _require(bool(audio) and bool(audio_output), "copied audio stream/packets missing")
    source_hashes = [(p.get("size"), p.get("data_hash")) for p in audio_source]
    output_hashes = [(p.get("size"), p.get("data_hash")) for p in audio_output]
    _require(output_hashes == source_hashes[:len(output_hashes)] and
             len(output_hashes) <= len(source_hashes),
             "output audio packet payloads are not a copied source prefix")
    for key in ("decode_host_copy_bytes_measured",
                "dlss_host_color_stage_bytes_measured",
                "dlss_output_readback_bytes_measured",
                "encoder_host_pixel_bytes_measured",
                "sdr10_diagnostic_image_readback_bytes_measured"):
        _require(diagnostics.get(key) == 0,
                 f"measured host image boundary {key}={diagnostics.get(key)!r}")
    _require(diagnostics.get("sdr10_pixel_readback_bytes_measured") is None,
             "pixel readback total must remain explicitly unmeasured, not inferred")
    return {"precision_mode": job["precision_mode"],
            "session_color_format": job["dlss_session_color_format"],
            "export_backend_actual": diagnostics["export_backend_actual"],
            "transfer_mode_actual": diagnostics["transfer_mode_actual"],
            "frames": expected_frames, "video": video,
            "audio_packets_copied": len(output_hashes),
            "audio_source_payloads_sha256": [p.get("data_hash") for p in audio_source],
            "audio_output_payloads_sha256": [p.get("data_hash") for p in audio_output],
            "host_image_boundaries_zero": True}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument('--stages', nargs='+', choices=['nr', 'dlaa', 'vsr'])
    parser.add_argument('--vsr-scale', type=float, default=2.0)
    args = parser.parse_args()
    _require(4 <= args.frames <= 6, "gate is limited to 4–6 source frames")
    source = args.source.resolve()
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    output = evidence / "service_sdr10_hevc.mkv"
    _require(source.is_file(), f"source missing: {source}")

    import dlss_service
    import pipeline
    source_frames, fps, width, height = pipeline.video_info(str(source))
    _require(source_frames == args.frames,
             f"service exports the whole input; source has {source_frames} frames, "
             f"expected {args.frames}")
    service = dlss_service.DLSSService()
    service.load_video(str(source))
    settings = dict(service.settings)
    settings.update(export_backend="gpu_transfer", dlss_backend="gpu_flow_d3d12",
                    guidance_mode=0, reconstruction_mode="single",
                    working_scale=0.5, residual_pyramid=True, residual_strength=1.0,
                    multi_pass=1)
    if args.stages:
        settings.update(render_pipeline=args.stages, dlaa_preset=11,
                        vsr_output_mode='scale', vsr_scale=args.vsr_scale)
        if 'nr' not in args.stages:
            settings.update(working_scale=1.0, guidance_mode=0)
    transcode = dict(codec="hevc_nvenc", preset="p7", tune="hq", multipass="qres",
        pixel_format="auto", profile="main10", color_preset="preserve",
        container="mkv", rate_mode="cq", cq=24, aq_mode="spatial",
        gop_mode="auto", decode_mode="cuda", audio_mode="copy",
        subtitle_mode="copy", copy_attachments=True, copy_metadata=True,
        copy_chapters=True, overwrite_output=True)
    job = service.add_job({"name": "SDR10 service acceptance", "input_path": str(source),
        "output_path": str(output), "codec": "hevc_nvenc", "container": "mkv",
        "cq": 24, "with_audio": True, "settings": settings,
        "transcode_settings": transcode})
    diagnostics = {}
    status = "failed"
    error = None
    try:
        service._execute_job(job, None)
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        diagnostics = job.get("residual_backend_diagnostics") or {}
        cap = getattr(service, "_cap", None)
        if cap is not None:
            cap.release()
            service._cap = None
        (evidence / "job.json").write_text(json.dumps(job, indent=2, default=str),
                                             encoding="utf-8")
        (evidence / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2,
            ensure_ascii=False, default=str), encoding="utf-8")
        (evidence / "run.json").write_text(json.dumps({"status": status, "error": error,
            "source": str(source), "output": str(output), "expected_frames": args.frames,
            "dimensions": [width, height], "fps": fps}, indent=2), encoding="utf-8")

    output_probe = _probe(output, "-show_streams", "-show_format", "-show_chapters")
    streams = output_probe.get("streams", [])
    frames = _probe(output, "-select_streams", "v:0", "-show_frames",
                    "-show_entries", "frame=best_effort_timestamp,pkt_duration")
    video_frames = frames.get("frames", [])
    audio_source = _packets(source, "a:0")
    audio_output = _packets(output, "a:0")
    (evidence / "output_probe.json").write_text(json.dumps(output_probe, indent=2,
        ensure_ascii=False), encoding="utf-8")
    (evidence / "audio_packet_hashes.json").write_text(json.dumps({
        "source": audio_source, "output": audio_output}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    summary = validate_result(job, diagnostics, streams, video_frames,
                              audio_source, audio_output, args.frames)
    summary.update(status="PASS", source=str(source), output=str(output))
    (evidence / "accepted_summary.json").write_text(json.dumps(summary, indent=2,
        ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    run()
