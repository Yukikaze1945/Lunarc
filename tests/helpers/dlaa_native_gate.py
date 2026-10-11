"""Short real-GPU contract and image-difference gate for signed NVIDIA DLAA.

Runs synthetic 640x360 SDR texture through the official NGX DLAA feature with
zero-motion and NVOF host-motion entry points. Diagnostic host copies/readbacks
are intentional and reported. This gate makes no image-quality improvement claim.
"""
import argparse
import ctypes as C
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIDTH, HEIGHT = 640, 360
FRAME_REQUESTS = (0, 1, 2, 2)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def make_scene_frames():
    """A small moving scene with slanted edges, thin lines, text and texture."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    frames = []
    for shift in (0, 2, 4):
        phase = ((xx + shift) // 5 + yy // 7) & 1
        img = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
        img[..., 0] = (35 + xx * 120 // WIDTH + phase * 38).astype(np.uint8)
        img[..., 1] = (48 + yy * 125 // HEIGHT + phase * 24).astype(np.uint8)
        img[..., 2] = (45 + ((xx + yy + shift) % 96) + phase * 42).astype(np.uint8)
        # Fine diagonal lines and a high-frequency checker patch.
        for x in range(-HEIGHT, WIDTH, 9):
            cv2.line(img, (x + shift, 0), (x + HEIGHT + shift, HEIGHT - 1),
                     (225, 230, 235), 1, cv2.LINE_AA)
        tile = ((np.indices((96, 128)).sum(axis=0) // 2) & 1).astype(np.uint8) * 185 + 35
        img[28:124, 24:152] = np.repeat(tile[..., None], 3, axis=2)
        cv2.rectangle(img, (190 + shift, 35), (329 + shift, 160), (20, 20, 230), 2,
                      cv2.LINE_AA)
        cv2.putText(img, "DLAA 1px / A-B 012", (190 + shift, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, .64, (248, 246, 236), 1, cv2.LINE_AA)
        cv2.putText(img, "thin-edge 123", (30, 300), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.circle(img, (430 + shift, 122), 24, (30, 230, 220), 2, cv2.LINE_AA)
        cv2.line(img, (400 + shift, 210), (575 + shift, 270), (240, 230, 30), 1,
                 cv2.LINE_AA)
        frames.append(np.ascontiguousarray(img))
    return frames


def _rgba(bgr):
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA))


def _resize_control(rgba):
    h, w = rgba.shape[:2]
    low = cv2.resize(rgba, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)


def _image_metrics(actual, reference):
    a = actual[..., :3].astype(np.float32)
    b = reference[..., :3].astype(np.float32)
    d = np.abs(a - b)
    return {"mae_rgb": float(d.mean()),
            "rmse_rgb": float(np.sqrt(np.mean((a - b) ** 2))),
            "p95_abs_rgb": float(np.percentile(d, 95)),
            "p99_abs_rgb": float(np.percentile(d, 99)),
            "changed_component_fraction": float(np.count_nonzero(d) / d.size),
            "changed_pixel_fraction": float(np.count_nonzero(np.any(d != 0, axis=2)) /
                                            (d.shape[0] * d.shape[1]))}


def _native_counters(lib):
    values = (C.c_uint64 * 3)()
    lib.dlss_gpu_host_dlaa_counters.argtypes = [C.POINTER(C.c_uint64)]
    lib.dlss_gpu_host_dlaa_counters.restype = C.c_int32
    code = lib.dlss_gpu_host_dlaa_counters(values)
    if code:
        raise RuntimeError(f"native DLAA counter read failed: {code}")
    return dict(zip(("creates", "successful_evaluates", "release_calls"), values))


def _counter_delta(after, before):
    return {key: int(after[key] - before[key]) for key in before}


def _run_path(name, dll, frames_bgr, out_dir, guidance=None):
    import dlaa_gpu_engine
    import dlss_gpu_engine

    lib = dlss_gpu_engine._load()
    native_before = _native_counters(lib)
    rgba_frames = [_rgba(frame) for frame in frames_bgr]
    settings = {"preset": 1, "style": 0, "intensity": 1.0,
                "local_tone": 1.0, "local_struct": 1.0, "skin_struct": 1.0,
                "use_auto_mask": 1, "ui_correction": 0,
                "motion_scale_x": 1.0, "motion_scale_y": 1.0}
    live = dlaa_gpu_engine.Live(WIDTH, HEIGHT, settings, dll_path=str(dll))
    _require(live.color_format == "rgba8" and live._w == WIDTH and live._h == HEIGHT,
             "DLAA input/output contract is not equal-resolution RGBA8")
    _require(live.dlss_dll == str(dll), "DLAA did not use the requested signed SDK DLL")
    before_transfer = live.transfer_counters()
    outputs, records = [], []
    prior_index = None
    try:
        if name == "nvof_host_motion":
            _require(live.supports_host_motion(), "native host-motion DLAA entry is unavailable")
        for ordinal, index in enumerate(FRAME_REQUESTS):
            current = frames_bgr[index]
            reset = prior_index is None or index != prior_index + 1
            previous = current if reset else frames_bgr[prior_index]
            motion_info = None
            start = time.perf_counter()
            if guidance is None:
                result = live.process_zero_motion(rgba_frames[index], reset=reset)
                method = "process_zero_motion"
            else:
                flow, motion_info = guidance.get_guidance(
                    frame_id=index, cur_bgr=current, prev_bgr=previous,
                    target_w=WIDTH, target_h=HEIGHT, guidance_mode=1,
                    is_discontinuous=reset, input_format="bgr24_host")
                _require(flow.shape == (HEIGHT, WIDTH, 2) and flow.dtype == np.float32 and
                         flow.flags.c_contiguous, "NVOF did not produce pixel-space float32 HxWx2")
                result = live.process_host_motion(rgba_frames[index], flow, reset=reset)
                method = "process_host_motion"
            elapsed = (time.perf_counter() - start) * 1000.0
            _require(result.shape == (HEIGHT, WIDTH, 4) and result.dtype == np.uint8,
                     f"{name} changed output dimensions or pixel format")
            outputs.append(result.copy())
            counts = live.counters()
            transfer = live.transfer_counters()
            records.append({"request": ordinal, "source_frame": index,
                "reset": reset, "method": method,
                "ng_x_process_ms": float(live.last_process_ms),
                "wall_process_and_guidance_ms": elapsed,
                "sdk_counters": counts,
                "sdk_counter_delta_from_path_open": _counter_delta(counts, native_before),
                "transfer_counters": transfer,
                "transfer_delta_from_path_open": _counter_delta(transfer, before_transfer),
                "guidance": motion_info,
                "input_rgba_bytes": int(rgba_frames[index].nbytes),
                "output_host_readback_bytes_expected": int(result.nbytes)})
            prior_index = index
    finally:
        transfer_after = live.transfer_counters()
        live.close()
        native_after = _native_counters(lib)

    native_delta = _counter_delta(native_after, native_before)
    _require(native_delta["creates"] == 1, f"{name}: expected one official NGX DLAA create, got {native_delta}")
    _require(native_delta["successful_evaluates"] == len(FRAME_REQUESTS),
             f"{name}: expected {len(FRAME_REQUESTS)} successful DLAA evaluations, got {native_delta}")
    _require(native_delta["release_calls"] == 1, f"{name}: expected one feature release, got {native_delta}")
    _require([row["reset"] for row in records] == [True, False, False, True],
             f"{name}: unexpected playback/seek/repeat reset flags")

    diffs = []
    for ordinal, (index, result) in enumerate(zip(FRAME_REQUESTS, outputs)):
        input_rgba = rgba_frames[index]
        upscaled_control = _resize_control(input_rgba)
        input_metrics = _image_metrics(result, input_rgba)
        resize_metrics = _image_metrics(result, upscaled_control)
        name_prefix = f"{name}_request_{ordinal:02d}_frame_{index:02d}"
        cv2.imwrite(str(out_dir / f"{name_prefix}_input.png"), cv2.cvtColor(input_rgba, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(str(out_dir / f"{name_prefix}_dlaa.png"), cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
        diff = cv2.absdiff(result[..., :3], input_rgba[..., :3])
        cv2.imwrite(str(out_dir / f"{name_prefix}_diff_x4.png"),
                    np.minimum(diff.astype(np.uint16) * 4, 255).astype(np.uint8))
        diffs.append({"request": ordinal, "source_frame": index,
            "reset": records[ordinal]["reset"],
            "output_shape": list(result.shape),
            "exactly_equal_input": bool(np.array_equal(result, input_rgba)),
            "exactly_equal_half_resolution_bilinear_control": bool(
                np.array_equal(result, upscaled_control)),
            "vs_input": input_metrics, "vs_half_resolution_bilinear_control": resize_metrics,
            "claim": "numeric difference only; this is not a perceptual quality verdict"})

    _require(any(not row["exactly_equal_input"] for row in diffs),
             f"{name}: every DLAA output exactly matches input; no visible output change")
    _require(any(not row["exactly_equal_half_resolution_bilinear_control"] for row in diffs),
             f"{name}: output unexpectedly matches resize-only control for every frame")
    expected_readback = WIDTH * HEIGHT * 4 * len(FRAME_REQUESTS)
    actual_readback = int(transfer_after["output_readback_bytes"] - before_transfer["output_readback_bytes"])
    _require(actual_readback == expected_readback,
             f"{name}: measured output readback {actual_readback}, expected {expected_readback}")
    return {"path": name, "sdk": {
                "requested_feature": "NVSDK_NGX_Feature_SuperSampling",
                "requested_feature_expected_code": 1,
                "requested_quality": "NVSDK_NGX_PerfQuality_Value_DLAA",
                "requested_quality_expected_code": 5,
                "equal_input_output_resolution": [WIDTH, HEIGHT],
                "signed_dll": str(dll), "native_lifecycle_counter_delta": native_delta},
            "frame_requests": records,
            "image_differences": diffs,
            "host_boundary": {
                "diagnostic_gate_only": True,
                "input_rgba_host_bytes": sum(r["input_rgba_bytes"] for r in records),
                "actual_native_output_readback_bytes": actual_readback,
                "native_transfer_counters_before": before_transfer,
                "native_transfer_counters_after": transfer_after,
                "native_transfer_counter_delta": _counter_delta(transfer_after, before_transfer),
                "note": "This correctness probe intentionally stages RGBA frames and reads each result to host for PNG/numeric diagnostics."}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=Path,
        default=Path(os.environ.get("LUNARC_DLSS_DLL", r"F:\dlss\DLSS-310.6-dll\nvngx_dlss.dll")))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    dll = args.dll.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = (args.out or (ROOT / "bench_out" / "native_backends_20261010" /
                            "dlaa" / f"textured_{stamp}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "FAILED", "dll": str(dll), "resolution": [WIDTH, HEIGHT],
              "frame_requests": list(FRAME_REQUESTS), "paths": {},
              "image_claim": "Quantitative differences only; no subjective DLAA quality-improvement claim.",
              "host_boundary_note": "Diagnostic host staging and readback are deliberate and counted."}
    flow = None
    try:
        _require(dll.is_file(), f"signed NGX DLSS DLL not found: {dll}")
        frames = make_scene_frames()
        # A representative synthetic input for visual inspection, not evidence of gain.
        cv2.imwrite(str(out_dir / "synthetic_scene_frame_02.png"), frames[2])
        report["scene"] = {"generator": "gradient, diagonal one-pixel lines, checker texture, text, small translation",
                            "source_frame_hashes_sha256": [__import__("hashlib").sha256(f.tobytes()).hexdigest() for f in frames]}
        report["paths"]["zero_motion"] = _run_path("zero_motion", dll, frames, out_dir)

        import optical_flow
        flow = optical_flow.FrameGuidanceService("nvof", "1M", dedicated=True)
        report["paths"]["nvof_host_motion"] = _run_path(
            "nvof_host_motion", dll, frames, out_dir, guidance=flow)
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if flow is not None:
            flow.close()
        (out_dir / "report.json").write_text(json.dumps(report, indent=2,
            ensure_ascii=False, default=str), encoding="utf-8")
        print(f"DLAA evidence: {out_dir}")
    print(json.dumps({"status": report["status"], "out": str(out_dir),
                      "paths": list(report["paths"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
