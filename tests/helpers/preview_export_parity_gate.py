"""Real GPU preview/export parity and remux gate (not a performance test).

Example:
  python tests/helpers/preview_export_parity_gate.py --directory bench_out/preview_parity_20261010/run

Supply SDR8, SDR10, and HDR10 sources explicitly. Every source/mode pair uses
one settings dictionary for both preview and export. Output directories must be
new; exports and decoded-frame evidence are never overwritten.
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

MODES = {"single": [1.0], "dual": [0.50, 0.75], "triple": [0.40, 0.67, 0.85]}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _run(command):
    return subprocess.run([str(x) for x in command], check=True, capture_output=True,
                          timeout=240)


def _probe(ffprobe, path, *args):
    return json.loads(_run([ffprobe, "-v", "error", *args, "-of", "json", path]).stdout)


def _settings(service, mode, scales, case):
    settings = dict(service.settings)
    settings.update(neural_mode="nr", guidance_mode=1, flow_engine="fidelityfx",
        export_backend="gpu_transfer", reconstruction_mode=mode,
        working_scale=scales[0], layer_scales=list(scales),
        layer_weights=[1.0] * len(scales), residual_strength=1.0,
        fusion_mode=("catmull_baseline" if len(scales) == 1 else "data_consistency_experimental"),
        fusion_beta=0.1, fusion_relative_tolerance=0.001, fusion_max_iterations=128,
        experimental_hdr_scrgb=(case == "hdr"))
    return settings


def _frame_pts(ffprobe, path):
    frames = _probe(ffprobe, path, "-select_streams", "v:0", "-show_frames",
                    "-show_entries", "frame=best_effort_timestamp_time")
    return [float(x["best_effort_timestamp_time"]) for x in frames.get("frames", [])]


def _decode_display_at_pts(ffmpeg, ffprobe, path, pts_values, case):
    """Decode selected frames to linear Rec.709 and apply the app's display map."""
    if case == "sdr8":
        # SDR8 preview exposes the UNORM output directly; no linear-light
        # display transform is applied by GpuPreview for this processing path.
        # Decode the final stream with FFmpeg's normal playback conversion.
        raw = _run([ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-i", str(path),
            "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo",
            "-pix_fmt", "bgr24", "pipe:1"]).stdout
        metadata = _probe(ffprobe, path, "-select_streams", "v:0", "-show_streams")["streams"][0]
        width, height = int(metadata["width"]), int(metadata["height"])
        frame_bytes = width * height * 3
        _require(len(raw) == len(pts_values) * frame_bytes,
                 "SDR8 output decode frame count differs from its PTS list")
        return list(np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3))
    from comparison_media import MediaComparisonService, _Media, linear_rgb_to_display_bgr

    comparison = MediaComparisonService(ffmpeg=ffmpeg, ffprobe=ffprobe)
    metadata = comparison._probe_metadata(str(path))
    origin = comparison._origin(str(path), metadata)
    media = _Media(str(path), metadata, origin)
    display = []
    for pts in pts_values:
        linear = comparison._decode_linear(media, float(pts))
        display.append(linear_rgb_to_display_bgr(linear, hdr=bool(metadata["is_hdr"])))
    return display


def _compare(preview, exported):
    a, b = preview.astype(np.float32), exported.astype(np.float32)
    delta = np.abs(a - b)
    std_a, std_b = float(a.std()), float(b.std())
    flat = max(std_a, std_b) < 1.0
    corr = (None if flat or min(std_a, std_b) == 0 else
            float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1]))
    facts = {"mean_absolute_error": float(delta.mean()),
             "p95_absolute_error": float(np.percentile(delta, 95)),
             "preview_std": std_a, "export_std": std_b,
             "correlation": corr,
             "frame_classification": "flat_or_near_black_diagnostic_only" if flat else "visible_detail",
             "lossy_tolerance": ({"mean_absolute_error_max": 1.0,
                                  "p95_absolute_error_max": 1.0,
                                  "correlation": "not_applicable_flat_low_detail"} if flat else
                                 {"mean_absolute_error_max": 8.0,
                                  "p95_absolute_error_max": 24.0,
                                  "correlation_min": 0.95}),
             "bitexact_required": False}
    pixel_error_ok = (facts["mean_absolute_error"] <= (1.0 if flat else 8.0) and
                      facts["p95_absolute_error"] <= (1.0 if flat else 24.0))
    facts["within_lossy_tolerance"] = (pixel_error_ok if flat else
        pixel_error_ok and corr is not None and np.isfinite(corr) and corr >= .95)
    return facts


def _case_from_source(ffprobe, source):
    streams = _probe(ffprobe, source, "-show_streams").get("streams", [])
    video = next(x for x in streams if x.get("codec_type") == "video")
    transfer, pix = video.get("color_transfer"), video.get("pix_fmt", "")
    case = "hdr" if transfer == "smpte2084" else "sdr10" if "10" in pix else "sdr8"
    return case, video


def run(directory: Path, sources: dict[str, Path], modes, cases=None, resume=False):
    out = directory.resolve()
    _require(not out.exists() or resume, f"refusing to overwrite evidence directory: {out}")
    _require(set(sources) == {"sdr8", "sdr10", "hdr"}, "all SDR8, SDR10, and HDR10 sources are required")
    for key, path in sources.items():
        _require(path.is_file(), f"{key} source missing: {path}")
    out.mkdir(parents=True, exist_ok=resume)
    import pipeline
    import gpu_transfer_export as gtx
    from dlss_service import DLSSService
    from tests.hardware.optimization_export_gate import (
        evaluate_artifacts, probe_artifact, probe_hdr_sei,
    )

    ffmpeg, ffprobe = pipeline._bundle_ffmpeg(), pipeline._bundle_ffprobe()
    _require(ffmpeg and ffprobe, "bundled FFmpeg/ffprobe unavailable")
    rows = (json.loads((out / "report.json").read_text(encoding="utf-8"))["rows"]
            if resume and (out / "report.json").exists() else [])
    selected_cases = tuple(cases or ("sdr8", "sdr10", "hdr"))
    for case in selected_cases:
        source = sources[case].resolve()
        detected, video = _case_from_source(ffprobe, source)
        _require(detected == case, f"{source}: expected {case}, found {detected}/{video}")
        width, height = int(video["width"]), int(video["height"])
        source_pts = _frame_pts(ffprobe, str(source))
        for mode in modes:
            scales = MODES[mode]
            name = f"{case}_{mode}"
            if any(r["case"] == case and r["mode"] == mode and r["status"] == "PASS"
                   for r in rows):
                continue
            evidence = out / name
            evidence.mkdir(exist_ok=resume)
            output, intermediate = evidence / "final.mkv", evidence / "intermediate_video_only.mkv"
            _require(not output.exists(), f"existing export requires offline recheck: {output}")
            service = DLSSService()
            settings = _settings(service, mode, scales, case)
            tx = {"codec": "hevc_nvenc", "preset": "p4", "container": "mkv",
                  "audio_mode": "copy", "subtitle_mode": "copy", "copy_chapters": True,
                  "copy_metadata": True, "copy_attachments": True, "overwrite_output": False}
            service.load_video(str(source))
            service.update_settings(settings)
            frames = len(source_pts)
            _require(frames <= 6, f"{source} exceeds the six-frame acceptance limit")
            _require(frames >= 3, f"{source} needs at least three frames")
            previews, diagnostics = [], []
            try:
                cached = evidence / "preview_diagnostics.json"
                reuse_preview = resume and cached.exists()
                if reuse_preview:
                    diagnostics = json.loads(cached.read_text(encoding="utf-8"))
                    _require(len(diagnostics) == frames, "incomplete preview evidence cannot be reused")
                    previews = [cv2.imread(str(evidence / f"preview_{i:03d}.jpg"))
                                for i in range(frames)]
                for index in range(frames):
                    if reuse_preview:
                        _require(previews[index] is not None and
                            previews[index].shape == (height, width, 3) and
                            diagnostics[index].get("frame") == index,
                            "cached preview evidence is incomplete")
                        continue
                    blob, status = service.render_preview(index, view="dlss", preview_mode=True)
                    _require(status == "ok" and blob, f"{name} preview frame {index}: {status}")
                    image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                    _require(image is not None and image.shape[:2] == (height, width),
                             f"{name} preview frame {index} has wrong display dimensions")
                    diag = service.preview_diagnostics()
                    _require(diag.get("frame") == index, f"preview frame diagnostic mismatch: {diag}")
                    previews.append(image)
                    diagnostics.append(diag)
                    (evidence / f"preview_{index:03d}.jpg").write_bytes(blob)
                    (evidence / "preview_diagnostics.json").write_text(
                        json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
                if len(scales) > 1:
                    layers = diagnostics[-1].get("layers") or {}
                    ids = layers.get("layer_session_ids") or []
                    _require(diagnostics[-1].get("layer_count") == len(scales) and
                             len(ids) == len(scales) and len(set(ids)) == len(scales),
                             f"{name} preview sessions/layers invalid: {layers}")
                    solves = [d.get("solver") for d in diagnostics]
                    _require(all(s and s.get("converged") is True for s in solves) and
                             all(d.get("fusion_mode_actual") in (
                                 "typed_gpu_data_consistency_solve", "gpu_data_consistency_solve")
                                 for d in diagnostics),
                             f"{name} preview CG convergence missing")
                preview_pts = [float(d["source_relative_time"]) for d in diagnostics]
                preview_pts = [source_pts[0] + x for x in preview_pts]
                _require(all(abs(a-b) <= .0011 for a, b in zip(preview_pts, source_pts[:frames])),
                         f"{name} preview PTS do not identify requested source frames")
                if getattr(service, "_gpu_preview", None):
                    service._gpu_preview.shutdown()
                    service._gpu_preview = None

                original = gtx.run_copy_pass
                mux = {"argv": None}
                def capture(ffmpeg_arg, encoded, original_source, transcode, target,
                            with_audio=True, **kwargs):
                    import shutil
                    shutil.copy2(encoded, intermediate)
                    command = original(ffmpeg_arg, encoded, original_source, transcode,
                                       target, with_audio, **kwargs)
                    mux["argv"] = [str(x) for x in command]
                    return command
                gtx.run_copy_pass = capture
                try:
                    job = service.add_job({"name": name, "input_path": str(source),
                        "output_path": str(output), "codec": "hevc_nvenc", "container": "mkv",
                        "cq": 24, "with_audio": True, "settings": settings,
                        "transcode_settings": tx})
                    service._execute_job(job, None)
                finally:
                    gtx.run_copy_pass = original
                export_diag = job.get("residual_backend_diagnostics") or {}
                (evidence / "export_job.json").write_text(
                    json.dumps(job, indent=2, default=str), encoding="utf-8")
                _require(export_diag.get("export_backend_actual") == "gpu_transfer",
                         f"{name} did not export through GPU Transfer: {export_diag}")
                output_pts = _frame_pts(ffprobe, str(output))
                _require(len(output_pts) == frames and all(
                    abs((x-output_pts[0])-(y-source_pts[0])) <= .0011
                    for x, y in zip(output_pts, source_pts[:frames])),
                    f"{name} encoded PTS differ from source: {source_pts[:frames]} -> {output_pts}")
                decoded = _decode_display_at_pts(ffmpeg, ffprobe, output, output_pts, case)
                _require(len(decoded) == frames and all(
                    image.shape == (height, width, 3) for image in decoded),
                    f"{name} decoded export frame count/dimensions are incorrect")
                comparisons = [_compare(previews[i], decoded[i]) for i in range(frames)]
                parity_ok = all(x["within_lossy_tolerance"] for x in comparisons)

                src_facts, int_facts, final_facts = [probe_artifact(ffprobe, p) for p in
                    (source, intermediate, output)]
                seis = ({key: probe_hdr_sei(ffmpeg, path, evidence, key) for key, path in
                         (("source", source), ("intermediate", intermediate), ("final", output))}
                        if case == "hdr" else {})
                remux_checks = evaluate_artifacts(src_facts, int_facts, final_facts, frames,
                    "hdr" if case == "hdr" else "sdr", mux["argv"], seis.get("source"),
                    seis.get("intermediate"), seis.get("final"), tx, True)
                failed = [k for k, v in remux_checks.items() if not v[0]]
                row = {"case": case, "mode": mode, "scales": scales,
                    "settings_shared_by_preview_and_export": True, "source": str(source),
                    "output": str(output), "source_pts": source_pts[:frames],
                    "preview_pts": preview_pts, "export_pts": output_pts,
                    "comparisons": comparisons, "remux_checks": {
                        k: {"passed": bool(v[0]), "detail": v[1]} for k, v in remux_checks.items()},
                    "diagnostics": diagnostics, "export_diagnostics": export_diag,
                    "parity_passed": parity_ok, "remux_passed": not failed,
                    "status": "PASS" if parity_ok and not failed else "FAIL"}
                rows.append(row)
                (out / "report.json").write_text(json.dumps({"status": "running", "rows": rows,
                    "comparison_policy": "visible detail: MAE <= 8, p95 <= 24, corr >= .95; "
                    "flat low-detail: MAE/p95 <= 1, correlation N/A; no bit-exact claim"},
                    indent=2, default=str), encoding="utf-8")
                _require(parity_ok, f"{name} display/export differs beyond tolerance: {comparisons}")
                _require(not failed, f"{name} remux preservation failed: {failed}")
            except Exception as exc:
                (evidence / "failure.json").write_text(json.dumps({"error": str(exc),
                    "diagnostics": diagnostics}, indent=2, default=str), encoding="utf-8")
                raise
            finally:
                engine = getattr(service, "_gpu_preview", None)
                if engine:
                    engine.shutdown()
                cap = getattr(service, "_cap", None)
                if cap:
                    cap.release()
    report = {"status": "PASS", "directory": str(out), "rows": rows,
              "comparison_policy": "lossy HEVC display tolerance; bit-exact picture equality not required"}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sdr8", type=Path, required=True)
    parser.add_argument("--sdr10", type=Path, required=True)
    parser.add_argument("--hdr", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES), default=list(MODES))
    parser.add_argument("--cases", nargs="+", choices=("sdr8", "sdr10", "hdr"),
                        default=["sdr8", "sdr10", "hdr"])
    parser.add_argument("--resume", action="store_true",
                        help="reuse complete preview evidence and skip passed cases; never overwrite exports")
    args = parser.parse_args()
    result = run(args.directory, {"sdr8": args.sdr8, "sdr10": args.sdr10, "hdr": args.hdr},
                 args.modes, args.cases, args.resume)
    print(json.dumps({"status": result["status"], "directory": result["directory"],
                      "runs": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
