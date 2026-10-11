"""Real FidelityFX preview and GPU Transfer export acceptance gate.

Run only after the standalone optical-flow gate passes on the same machine:
  python tests/helpers/fidelityfx_pipeline_gate.py --directory bench_out/amd_flow_20261010/pipeline_gate

The gate creates a fresh 9-frame SDR 8-bit source with a moving textured patch and
PCM audio, then exercises production preview and export for NR single, dual, and
triple reconstruction. It never substitutes a CPU flow provider. Outputs and JSON
evidence are kept under the required new directory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIDTH, HEIGHT, FPS, FRAME_COUNT = 640, 480, 24, 9
MODES = (
    ("single", "single", [1.0]),
    ("dual", "dual", [0.50, 0.75]),
    ("triple", "triple", [0.40, 0.67, 0.85]),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _textured_frame(index):
    """Mostly static scene plus a translated, detailed center patch after warmup."""
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    base = np.stack((
        (x * 13 + y * 29 + (x * y) % 97) & 255,
        (x * 37 + y * 11 + (x ^ y) * 3) & 255,
        (x * 7 + y * 43 + (x // 7) * (y // 9)) & 255,
    ), axis=-1).astype(np.uint8)
    x0, x1 = WIDTH // 5, 4 * WIDTH // 5
    y0, y1 = HEIGHT // 5, 4 * HEIGHT // 5
    dx, dy = (8 * max(0, index - 5), 4 * max(0, index - 5))
    sy, sx = y[y0:y1, x0:x1] - dy, x[y0:y1, x0:x1] - dx
    valid = (sx >= x0) & (sx < x1) & (sy >= y0) & (sy < y1)
    patch = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    patch[valid] = base[sy[valid], sx[valid]]
    base[y0:y1, x0:x1] = patch
    return np.ascontiguousarray(base[:, :, ::-1])  # OpenCV BGR raw input.


def _create_fixture(path, ffmpeg):
    frames = b"".join(_textured_frame(i).tobytes() for i in range(FRAME_COUNT))
    command = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
        "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
        f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={FRAME_COUNT/FPS}",
        "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(FRAME_COUNT),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
        "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
        "-color_trc", "bt709", "-color_primaries", "bt709",
        "-c:a", "pcm_s16le", "-shortest", str(path)]
    subprocess.run(command, input=frames, check=True, timeout=120)


def _probe(path, *arguments):
    import pipeline
    return json.loads(subprocess.check_output(
        [pipeline._bundle_ffprobe(), "-v", "error", *arguments, "-of", "json", str(path)],
        timeout=120, text=True, encoding="utf-8"))


def _packet_facts(path, selector):
    data = _probe(path, "-select_streams", selector, "-show_packets", "-show_data_hash",
                  "sha256", "-show_entries", "packet=codec_type,size,data_hash,pts,duration")
    return data.get("packets", [])


class _TrackedProvider:
    """Transparent observer around each actual native FidelityFX provider."""
    def __init__(self, inner, ledger):
        self.inner, self.ledger = inner, ledger
        self.samples = []
        self.engine_identity = None
        self.native_handle = None
        ledger.append(self)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def new_independent_layer(self):
        return _TrackedProvider(self.inner.new_independent_layer(), self.ledger)

    def get_flow_device_rgba(self, *args, **kwargs):
        result = self.inner.get_flow_device_rgba(*args, **kwargs)
        engine = self.inner.engine
        self.engine_identity = id(engine)
        self.native_handle = int(engine._handle.value or 0)
        counters = dict(engine.counters())
        self.samples.append(counters)
        _require(result["format"] == "float32x2" and result["direction"] == "current_to_previous",
                 f"FidelityFX returned an unexpected flow descriptor: {result}")
        return result

    def close_independent_layer(self):
        self.inner.close_independent_layer()


def _track_fidelityfx(ledger):
    import native_effects
    factory = native_effects.make_flow_provider

    def tracked(settings, live):
        return _TrackedProvider(factory(settings, live), ledger)

    native_effects.make_flow_provider = tracked
    return lambda: setattr(native_effects, "make_flow_provider", factory)


def _provider_facts(ledger, expected_layers, expected_calls, expected_resets):
    _require(len(ledger) == expected_layers,
             f"created {len(ledger)} FidelityFX histories; expected {expected_layers}")
    identities = [provider.engine_identity for provider in ledger]
    handles = [provider.native_handle for provider in ledger]
    _require(all(identities) and len(set(identities)) == expected_layers,
             f"FidelityFX providers shared an engine/history: {identities}")
    _require(all(handles) and len(set(handles)) == expected_layers,
             f"FidelityFX providers shared a native session handle: {handles}")
    reports = []
    for provider in ledger:
        _require(len(provider.samples) == expected_calls,
                 f"history had {len(provider.samples)} calls, expected {expected_calls}")
        final = provider.samples[-1]
        quality = provider.inner.quality
        analysis_size = [((final[key]+1)//2 if quality == 'Performance' else final[key])
                         for key in ('width', 'height')]
        _require([final['grid_width'], final['grid_height']] == [(v+7)//8 for v in analysis_size],
                 'layer did not use the requested AMD quality dimensions')
        _require(final["process_calls"] == expected_calls and
                 final["successful_dispatches"] == expected_calls,
                 f"FidelityFX SDK dispatch counters: {final}")
        _require(final["reset_dispatches"] == expected_resets,
                 f"FidelityFX reset dispatches={final['reset_dispatches']}, "
                 f"expected {expected_resets}")
        reports.append({"engine_identity": provider.engine_identity,
                        "quality": quality, "analysis_resolution": analysis_size,
                        "native_handle": provider.native_handle,
                        "calls": len(provider.samples), "sdk_counters": final})
    return reports


def _settings(service, mode, scales):
    settings = dict(service.settings)
    settings.update(neural_mode="nr", guidance_mode=1, flow_engine="fidelityfx",
        export_backend="gpu_transfer", reconstruction_mode=mode,
        working_scale=scales[0], layer_scales=list(scales),
        layer_weights=[1.0] * len(scales),
        fusion_mode=("catmull_baseline" if len(scales) == 1
                     else "data_consistency_experimental"),
        fusion_beta=0.1, fusion_relative_tolerance=0.001,
        fusion_max_iterations=128, residual_strength=1.0,
        experimental_hdr_scrgb=False)
    return settings


def _render(service, frame):
    blob, status = service.render_preview(frame, view="dlss", preview_mode=True)
    _require(status == "ok" and blob, f"preview frame {frame} failed: {status}")
    image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
    _require(image is not None and image.shape == (HEIGHT, WIDTH, 3),
             f"preview frame {frame} has invalid display image")
    diagnostics = service.preview_diagnostics()
    _require(diagnostics.get("frame") == frame and diagnostics.get("reset") == (frame == 0),
             f"preview frame/reset diagnostics mismatch: {diagnostics}")
    return blob, image, diagnostics


def _verify_layer_report(report, expected_layers, expected_frames, expected_resets):
    _require(report.get("layer_count") == expected_layers,
             f"reported layer_count={report.get('layer_count')}")
    session_ids = report.get("layer_session_ids") or []
    _require(len(session_ids) == expected_layers and len(set(session_ids)) == expected_layers,
             f"per-layer DLSS sessions are not independent: {session_ids}")
    calls = report.get("layer_dlss_calls") or []
    _require(calls == [expected_frames] * expected_layers,
             f"per-layer DLSS calls={calls}")
    reset_flags = report.get("layer_reset_flags") or []
    _require(len(reset_flags) == expected_layers,
             f"per-layer reset histories missing: {reset_flags}")
    expected = [True] + [False] * (expected_frames - expected_resets) + [True] * (expected_resets - 1)
    for index, flags in enumerate(reset_flags):
        _require(flags == expected, f"layer {index} reset history={flags}, expected {expected}")
    solver = report.get("fusion_solver_frames") or []
    if expected_layers > 1:
        _require(len(solver) == expected_frames and all(row.get("converged") is True for row in solver),
                 f"CG convergence evidence is incomplete: {solver}")


def _probe_export(source, output):
    in_streams = _probe(source, "-show_streams").get("streams", [])
    out_streams = _probe(output, "-show_streams").get("streams", [])
    input_video = [s for s in in_streams if s.get("codec_type") == "video"]
    output_video = [s for s in out_streams if s.get("codec_type") == "video"]
    input_audio = [s for s in in_streams if s.get("codec_type") == "audio"]
    output_audio = [s for s in out_streams if s.get("codec_type") == "audio"]
    _require(len(input_video) == len(output_video) == 1, "video stream missing from source or export")
    _require([output_video[0].get("width"), output_video[0].get("height")] == [WIDTH, HEIGHT],
             f"export video extent changed: {output_video[0]}")
    _require(len(input_audio) == len(output_audio) == 1, "copied PCM audio stream missing")
    _require(output_audio[0].get("codec_name") == "pcm_s16le",
             f"audio was not preserved as PCM: {output_audio[0].get('codec_name')}")
    input_video_packets = _packet_facts(source, "v:0")
    output_video_packets = _packet_facts(output, "v:0")
    input_audio_packets = _packet_facts(source, "a:0")
    output_audio_packets = _packet_facts(output, "a:0")
    _require(len(input_video_packets) == len(output_video_packets) == FRAME_COUNT,
             f"video packet counts source/export={len(input_video_packets)}/{len(output_video_packets)}")
    _require(len(input_audio_packets) == len(output_audio_packets) > 0,
             f"audio packet counts source/export={len(input_audio_packets)}/{len(output_audio_packets)}")
    in_hashes = [packet.get("data_hash") for packet in input_audio_packets]
    out_hashes = [packet.get("data_hash") for packet in output_audio_packets]
    _require(in_hashes == out_hashes, "copied audio packet payload hashes changed")
    return {"video_packet_count_source": len(input_video_packets),
            "video_packet_count_export": len(output_video_packets),
            "audio_packet_count_source": len(input_audio_packets),
            "audio_packet_count_export": len(output_audio_packets),
            "audio_packet_payload_sha256_equal": True,
            "video_codec_export": output_video[0].get("codec_name"),
            "audio_codec_export": output_audio[0].get("codec_name")}


def run(directory: Path, flow_gate: Path):
    out = directory.resolve()
    evidence = flow_gate.resolve()
    _require(evidence.is_file(), f"standalone native flow gate evidence missing: {evidence}")
    flow_report = json.loads(evidence.read_text(encoding="utf-8"))
    _require(flow_report.get("status") == "pass",
             f"standalone hardware motion gate has not passed: {flow_report.get('status')}")
    _require(flow_report.get("counters", {}).get("successful_dispatches", 0) >= 9,
             "standalone gate did not verify native SDK dispatches")
    observed = (flow_report.get("frames") or [{}, {}])[7].get("median_flow_xy")
    _require(observed and abs(observed[0]) > 1 and abs(observed[1]) > 1,
             f"standalone gate has no measured nonzero 2D hardware motion: {observed}")
    out.mkdir(parents=True, exist_ok=False)
    import pipeline
    from dlss_service import DLSSService

    source = out / "textured_motion_640x480_9f.mkv"
    _create_fixture(source, pipeline._bundle_ffmpeg())
    source_info = _probe(source, "-show_streams")
    source_video = next(s for s in source_info["streams"] if s.get("codec_type") == "video")
    _require((source_video.get("width"), source_video.get("height")) == (WIDTH, HEIGHT),
             f"fixture dimensions: {source_video}")
    _require(source_video.get("pix_fmt") == "yuv420p",
             f"fixture pixel format is not SDR 8-bit 4:2:0: {source_video.get('pix_fmt')}")
    _require(len(_packet_facts(source, "v:0")) == FRAME_COUNT,
             "fixture does not contain exactly nine video packets")

    rows = []
    try:
        for name, reconstruction, scales in MODES:
            service = DLSSService()
            preview_providers = []
            export_providers = []
            output = out / f"nr_{name}.mkv"
            if output.exists():
                raise FileExistsError(f"Refuse to overwrite export: {output}")
            try:
                service.load_video(str(source))
                settings = _settings(service, reconstruction, scales)
                settings['flow_quality'] = flow_report.get('quality', 'Quality')
                service.update_settings(settings)

                restore_factory = _track_fidelityfx(preview_providers)
                try:
                    preview_rows = []
                    for index in list(range(FRAME_COUNT)) + [0]:
                        blob, image, diagnostics = _render(service, index)
                        (out / f"preview_{name}_{len(preview_rows):02d}_f{index:02d}.jpg").write_bytes(blob)
                        preview_rows.append(diagnostics)
                finally:
                    restore_factory()
                _verify_layer_report(preview_rows[-1].get("layers") or {}, len(scales),
                                     FRAME_COUNT + 1, 2) if len(scales) > 1 else None
                preview_provider_rows = _provider_facts(preview_providers, len(scales),
                                                        FRAME_COUNT + 1, 2)
                _require(preview_rows[6].get("motion", {}).get("sdk_counters", {}).get(
                    "successful_dispatches", 0) >= 7,
                    "preview did not dispatch FidelityFX through SDK warmup and motion frames")
                service._gpu_preview.shutdown()
                service._gpu_preview = None

                # Track the real factory and every independent child history during export.
                restore_factory = _track_fidelityfx(export_providers)
                try:
                    job = service.add_job({
                        "name": f"fidelityfx NR {name}", "input_path": str(source),
                        "output_path": str(output), "codec": "hevc_nvenc", "container": "mkv",
                        "cq": 24, "with_audio": True, "settings": settings,
                        "transcode_settings": {"codec": "hevc_nvenc", "container": "mkv",
                            "preset": "p4", "tune": "hq", "audio_mode": "copy",
                            "subtitle_mode": "copy", "copy_chapters": True,
                            "copy_metadata": True, "copy_attachments": True,
                            "overwrite_output": False},
                    })
                    service._execute_job(job, None)
                finally:
                    restore_factory()
                _require(output.is_file() and output.stat().st_size > 0,
                         f"{name} export produced no output")
                export_providers_rows = _provider_facts(export_providers, len(scales),
                                                        FRAME_COUNT, 1)
                export_diagnostics = job.get("residual_backend_diagnostics") or {}
                _require(export_diagnostics.get("export_backend_actual") == "gpu_transfer",
                         f"export did not use GPU Transfer: {export_diagnostics}")
                if len(scales) > 1:
                    _verify_layer_report(export_diagnostics, len(scales), FRAME_COUNT, 1)
                else:
                    _require(export_diagnostics.get("gpu_transfer_dlss_flow_calls") == FRAME_COUNT,
                             "single export did not use guided DLSS for all frames")
                packets = _probe_export(source, output)
                row = {"mode": name, "scales": scales, "source": str(source),
                    "output": str(output), "preview_frames": preview_rows,
                    "preview_provider_histories": preview_provider_rows,
                    "export_provider_histories": export_providers_rows,
                    "export_diagnostics": export_diagnostics, "packets": packets,
                    "status": "PASS"}
                rows.append(row)
                (out / "report.json").write_text(json.dumps({"status": "running", "rows": rows,
                    "standalone_flow_gate": flow_report}, indent=2, ensure_ascii=False,
                    default=str), encoding="utf-8")
            except Exception as exc:
                rows.append({"mode": name, "scales": scales, "status": "FAIL",
                             "error": f"{type(exc).__name__}: {exc}"})
                (out / "report.json").write_text(json.dumps({"status": "fail", "rows": rows,
                    "standalone_flow_gate": flow_report}, indent=2, ensure_ascii=False,
                    default=str), encoding="utf-8")
                raise
            finally:
                engine = getattr(service, "_gpu_preview", None)
                if engine is not None:
                    engine.shutdown()
                    service._gpu_preview = None
                if getattr(service, "_cap", None):
                    service._cap.release()
                    service._cap = None
        final = {"status": "PASS", "source": str(source), "directory": str(out),
                 "standalone_flow_gate": flow_report, "rows": rows}
        (out / "report.json").write_text(json.dumps(final, indent=2, ensure_ascii=False,
            default=str), encoding="utf-8")
        return final
    except Exception:
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--flow-gate", type=Path, default=ROOT /
        "bench_out/amd_flow_20261010/pipeline_flow_prereq_640x480/fidelityfx_flow_native_gate.json")
    args = parser.parse_args()
    report = run(args.directory, args.flow_gate)
    print(json.dumps({"status": report["status"], "directory": report["directory"],
        "modes": [row["mode"] for row in report["rows"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
