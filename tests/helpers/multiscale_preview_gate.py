"""Short real-GPU preview acceptance gate for the shared export frame core.

Example (Windows GPU runtime required):
  python tests/helpers/multiscale_preview_gate.py --source bench_out/fixture_4k_bt709_short.mkv --mode dual --directory bench_out/preview_20261010/sdr8_dual

This is a correctness gate, not a benchmark. It writes only display JPEGs and
scalar diagnostics; the preview consumer does not open an encoder or export.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _settings(service, mode):
    preset = {
        "dual": ("dual", [0.50, 0.75]),
        "triple": ("triple", [0.40, 0.67, 0.85]),
    }
    reconstruction, scales = preset[mode]
    settings = dict(service.settings)
    settings.update(guidance_mode=0, dlss_backend="gpu_flow_d3d12",
        export_backend="gpu_transfer", reconstruction_mode=reconstruction,
        working_scale=scales[0], layer_scales=scales,
        layer_weights=[1.0] * len(scales),
        fusion_mode="data_consistency_experimental", fusion_beta=0.1,
        fusion_relative_tolerance=0.001, fusion_max_iterations=128,
        residual_pyramid=True, residual_strength=1.0,
        experimental_hdr_scrgb=False, hdr_paper_white_scrgb=8.0)
    return settings


def _render(service, frame):
    data, status = service.render_preview(frame, view="dlss", preview_mode=True)
    _require(status == "ok" and data, f"preview frame {frame}: {status}")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    _require(image is not None and image.size, f"preview frame {frame} JPEG did not decode")
    diag = service.preview_diagnostics()
    _require(diag.get("status") != "failed", f"preview diagnostics: {diag}")
    return data, image, diag


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--mode", required=True, choices=("dual", "triple"))
    ap.add_argument("--directory", required=True, type=Path)
    args = ap.parse_args()
    source = args.source.resolve()
    out = args.directory.resolve()
    _require(source.is_file(), f"source missing: {source}")
    out.mkdir(parents=True, exist_ok=False)

    import dlss_service
    service = dlss_service.DLSSService()
    engine = None
    rows = []
    status = "failed"
    try:
        service.load_video(str(source))
        service.update_settings(_settings(service, args.mode))
        total = int(service.video_info["frames"])
        _require(total > 0, "source has no frames")
        first = 0
        jump = min(2, total - 1)
        requests = [first, jump, jump] if jump else [first, first, first]
        decoded = []
        for ordinal, index in enumerate(requests):
            blob, image, diag = _render(service, index)
            (out / f"frame_{ordinal:02d}_source_{index:06d}.jpg").write_bytes(blob)
            layers = diag.get("layers") or {}
            solver_frames = layers.get("fusion_solver_frames") or []
            layer_calls = (layers.get("layer_dlss_calls") or
                           layers.get("layer_calls") or [])
            session_ids = layers.get("layer_session_ids") or []
            rows.append({"request": ordinal, "diagnostics": diag,
                         "image_shape": list(image.shape),
                         "jpeg_bytes": len(blob)})
            expected_layers = 2 if args.mode == "dual" else 3
            _require(diag.get("layer_count") == expected_layers,
                     f"frame {index}: layer_count={diag.get('layer_count')}")
            _require(len(session_ids) == expected_layers and len(set(session_ids)) == expected_layers,
                     f"frame {index}: layer sessions are not distinct: {session_ids}")
            _require(len(layer_calls) == expected_layers and all(int(c) >= 1 for c in layer_calls),
                     f"frame {index}: per-layer calls={layer_calls}")
            _require(all(int(c) == ordinal + 1 for c in layer_calls),
                     f"frame {index}: cached/skipped layer work; call counters={layer_calls}")
            expected_fusion = ('typed_gpu_data_consistency_solve' if diag.get('processing_format') == 'rgba16f'
                               else 'gpu_data_consistency_solve')
            _require(diag.get("fusion_mode_actual") == expected_fusion,
                     f"frame {index}: actual fusion={diag.get('fusion_mode_actual')}")
            solver = diag.get("solver")
            if not solver and solver_frames:
                solver = solver_frames[-1]
            _require(isinstance(solver, dict) and solver.get("converged") is True,
                     f"frame {index}: solver did not report convergence: {solver}")
            decoded.append(image)

        # Same source frame is recomputed after an explicit discontinuity. A stable
        # decoded image is expected; the reset/call counters prove it was processed.
        _require(rows[-1]["diagnostics"].get("reset") is True,
                 "repeated frame did not request reset")
        _require(rows[-1]["diagnostics"].get("generation") ==
                 rows[-2]["diagnostics"].get("generation"),
                 "same settings unexpectedly rebuilt the preview generation")
        _require(np.array_equal(decoded[-1], decoded[-2]),
                 "repeat request for same source frame was not deterministic")
        _require(rows[-1]["diagnostics"].get("processing_ms", 0) > 0,
                 "repeat frame lacks fresh processing timing")

        # A fusion parameter patch must rebuild the session generation and is checked
        # by actual solver settings/report as well as display response when visible.
        settings = dict(service.settings)
        settings["fusion_beta"] = 0.35
        service.update_settings(settings)
        blob, changed, diag = _render(service, jump)
        (out / f"parameter_change_source_{jump:06d}.jpg").write_bytes(blob)
        _require(diag.get("generation", 0) > rows[-1]["diagnostics"].get("generation", 0),
                 "parameter update did not rebuild preview generation")
        _require(diag.get("fusion_mode_actual") == expected_fusion and
                 isinstance(diag.get("solver"), dict) and diag["solver"].get("converged") is True,
                 "parameter-changed frame did not run a converged solver")
        rows.append({"request": "parameter_change", "diagnostics": diag,
                     "image_shape": list(changed.shape), "jpeg_bytes": len(blob),
                     "fusion_beta": settings["fusion_beta"],
                     "image_changed": not np.array_equal(changed, decoded[-1]),
                     "visual_change_check": ("checked" if decoded[-1].std() >= 1.0
                                              else "inconclusive_flat_source_frame")})
        if decoded[-1].std() >= 1.0:
            _require(rows[-1]["image_changed"],
                     "fusion parameter change produced identical visible JPEG pixels")
        status = "PASS"
    finally:
        engine = getattr(service, "_gpu_preview", None)
        if engine is not None:
            engine.shutdown()
            service._gpu_preview = None
        cap = getattr(service, "_cap", None)
        if cap is not None:
            cap.release()
            service._cap = None
        (out / "diagnostics.json").write_text(json.dumps(rows, indent=2,
            ensure_ascii=False, default=str), encoding="utf-8")
        (out / "run.json").write_text(json.dumps({"status": status,
            "source": str(source), "mode": args.mode,
            "render_requests": [r.get("diagnostics", {}).get("frame") for r in rows],
            "note": "JPEG display readbacks are explicit; this gate performs no export."},
            indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": status, "mode": args.mode,
        "source": str(source), "directory": str(out),
        "frames": [r.get("diagnostics", {}).get("frame") for r in rows]},
        ensure_ascii=False))


if __name__ == "__main__":
    run()
