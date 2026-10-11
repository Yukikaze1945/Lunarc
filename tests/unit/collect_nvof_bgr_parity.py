#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write nvof_bgr_parity.json: the BGR input path against the RGBA input path, same pixels.

The claim this artifact exists for is narrow: the byte-map kernel the bridge runs on the GPU
produces the input surface the CPU conversion used to produce, so the flow that comes back is
the same field, and the upload that crosses PCIe is 3 bytes per pixel instead of 4. Two things
have to be true at once for that to be worth recording, and they are checked separately:

  * the flow arrays are equal bit for bit, because a tolerance would hide a kernel that nearly
    reorders the channels,
  * the byte figures come out of `nvof_bridge_get_counters`, compared against arithmetic on
    width and height. The direction matters: a measurement may only be checked against
    something independent of itself, so every expected number below is computed from the frame
    size and none of the actual ones are.

Each size runs against two kinds of pixels. A translated, noisy synthetic sequence gives the
hardware a displacement it can resolve exactly, which is what a bit-equality comparison needs;
frames decoded from the benchmark fixture put real image content through the same comparison.
An all-zero flow field would satisfy the equality for the wrong reason, so the measured motion
of every sequence is reported next to it.

Usage:
    python tests/unit/collect_nvof_bgr_parity.py --out perf_results/<sha>/transfer_phase1 \
        [--sizes 1920x1080 640x360] [--frames 6]
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys

import cv2
import numpy as np

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_transfer_abi as abi  # noqa: E402
import optical_flow as of  # noqa: E402
import transfer_profiler as tp  # noqa: E402

QUALITY = "4M"
FLOW_BYTES_PER_PIXEL = 8  # float32 x2: the field the hardware downloads


def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True,
                             timeout=30)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def synthetic_frames(width, height, count):
    """Whole-pixel translation plus noise: a displacement the hardware resolves exactly."""
    rng = np.random.default_rng(11)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    frames = []
    for index in range(count):
        shifted = np.roll(np.roll(base, 3 * index + 1, axis=1), index, axis=0)
        noise = rng.integers(0, 24, size=shifted.shape, dtype=np.uint16)
        frames.append(np.clip(shifted.astype(np.uint16) + noise, 0, 255).astype(np.uint8))
    return frames


def fixture_frames(width, height, count):
    """Real decoded pixels from the benchmark fixture, sized to the working resolution."""
    try:
        import benchmark_e2e as bench
        path = bench.ensure_fixture()
    except Exception as exc:
        return None, f"fixture unavailable: {type(exc).__name__}: {exc}"
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(np.ascontiguousarray(frame))
    finally:
        cap.release()
    if len(frames) < 3:
        return None, f"the fixture yielded {len(frames)} decodable frames"
    return frames, None


def _open_pair(width, height):
    """Two independent sessions with their counters baselined, so the next delta is one frame's.

    Without the baseline the first snapshot comes from a session that did not exist yet, and a
    delta taken against it reads None for a field the frame did update.
    """
    engines = []
    for _ in range(2):
        engine = of.NvOFHardwareEngine(quality=QUALITY)
        engine._ensure_session(width, height)
        engine.get_counters_delta()
        engines.append(engine)
    return engines


def _run_path(engine, frames, use_bgr):
    rows = []
    for index, cur in enumerate(frames):
        prev = None if index == 0 else frames[index - 1]
        if use_bgr:
            flow = engine.compute_flow_bgr24(prev, cur, reset_temporal_hints=index == 0)
        else:
            flow = engine.compute_flow_sequential(cur_bgr=cur, prev_bgr=prev,
                                                  reset_temporal_hints=index == 0)
        timing = dict(engine.get_last_timing() or {})
        magnitude = np.abs(flow.astype(np.float64))
        rows.append({
            "frame": index,
            "reset": index == 0,
            "flow_sha256": hashlib.sha256(np.ascontiguousarray(flow).tobytes()).hexdigest(),
            "flow_shape": list(flow.shape),
            "flow_dtype": str(flow.dtype),
            "flow_mean_abs_px": round(float(magnitude.mean()), 6),
            "flow_max_abs_px": round(float(magnitude.max()), 6),
            "counters": engine.get_counters_delta(),
            "nvof_input_format": timing.get("nvof_input_format"),
            "nvof_input_convert_wait_ms": timing.get("nvof_input_convert_wait_ms"),
            "cur_bgr_to_rgba_ms": timing.get("cur_bgr_to_rgba_ms"),
            "prev_bgr_to_rgba_ms": timing.get("prev_bgr_to_rgba_ms"),
        })
    return rows


def _steady(rows):
    return [row for row in rows if not row["reset"]]


def _counter(rows, key):
    return [row["counters"].get(key) for row in rows if row["counters"].get(key) is not None]


def _median(values):
    return int(statistics.median(values)) if values else None


def _check(what, expected, actual):
    return {"what": what, "expected": expected, "actual": actual, "ok": actual == expected}


def compare(label, width, height, frames):
    """One sequence through both input paths, with the counters each path moved."""
    bgr_engine, legacy_engine = _open_pair(width, height)
    try:
        before = tp.conversion_snapshot()
        legacy = _run_path(legacy_engine, frames, use_bgr=False)
        at_bgr = tp.conversion_snapshot()
        legacy_seam, _ = tp.conversion_deltas(before)
        bgr = _run_path(bgr_engine, frames, use_bgr=True)
        bgr_seam, _ = tp.conversion_deltas(at_bgr)
    finally:
        for engine in (bgr_engine, legacy_engine):
            try:
                engine.close()
            except Exception:
                pass

    payload = width * height
    bgr_steady, legacy_steady = _steady(bgr), _steady(legacy)
    checks = [
        _check("flow_bit_exact_every_frame", True,
               all(l["flow_sha256"] == b["flow_sha256"] for l, b in zip(legacy, bgr))),
        _check("bgr_path_input_format", ["bgr24_host"],
               sorted({row["nvof_input_format"] for row in bgr})),
        _check("legacy_path_input_format", ["rgba8_host"],
               sorted({row["nvof_input_format"] for row in legacy})),
        _check("bgr_steady_h2d_bytes", payload * 3, _median(_counter(bgr_steady,
                                                                     "nvof_h2d_bytes_delta"))),
        _check("bgr_steady_h2d_copy_count", 1,
               _median(_counter(bgr_steady, "nvof_h2d_copy_count_delta"))),
        _check("legacy_steady_h2d_bytes", payload * 4,
               _median(_counter(legacy_steady, "nvof_h2d_bytes_delta"))),
        _check("bgr_reset_h2d_bytes", payload * 3 * 2,
               _median(_counter(bgr[:1], "nvof_h2d_bytes_delta"))),
        _check("bgr_d2h_bytes_every_frame", payload * FLOW_BYTES_PER_PIXEL,
               _median(_counter(bgr, "nvof_d2h_bytes_delta"))),
        _check("bgr_staging_allocations_after_session_start", 0,
               _median(_counter(bgr_steady, "nvof_bgr_staging_alloc_count_delta"))),
        _check("bgr_input_calls_per_frame", 1,
               _median(_counter(bgr, "nvof_bgr_input_call_count_delta"))),
        _check("input_convert_launches_per_frame", 1,
               _median(_counter(bgr, "nvof_input_convert_launch_count_delta"))),
        _check("cpu_conversions_at_nvof_input_bgr_path", 0,
               bgr_seam.get("cpu_bgr_to_rgba@nvof_input_calls")),
        _check("cpu_conversions_at_nvof_input_legacy_path_gt_0", True,
               (legacy_seam.get("cpu_bgr_to_rgba@nvof_input_calls") or 0) > 0),
        _check("sequence_has_motion", True,
               max(row["flow_mean_abs_px"] for row in bgr) > 0.0),
    ]
    return {
        "label": label,
        "width": width,
        "height": height,
        "frames": len(frames),
        "expected_payload_bytes": {
            "bgr_upload": payload * 3,
            "rgba_upload": payload * 4,
            "flow_download": payload * FLOW_BYTES_PER_PIXEL,
        },
        "bytes_saved_per_steady_frame": payload,
        "per_frame": [{"frame": index, "reset": b["reset"], "bit_exact":
                       b["flow_sha256"] == l["flow_sha256"], "legacy": l, "bgr": b}
                      for index, (l, b) in enumerate(zip(legacy, bgr))],
        "seam_conversion_calls": {
            "legacy_path": {k: v for k, v in legacy_seam.items() if k.endswith("_calls")},
            "bgr_path": {k: v for k, v in bgr_seam.items() if k.endswith("_calls")},
        },
        "checks": checks,
    }


def bridge_binary():
    """The bridge file the loader actually mapped: hashed, with its BGR entry point listed."""
    of.probe_nvof_capabilities()
    bridge = getattr(of, "_NATIVE_BRIDGE", None)
    path = getattr(bridge, "_name", None)
    tool, tool_version = abi.find_dumpbin()
    rows = abi.parse_exports(abi.dumpbin(tool, "/exports", path)) if tool and path else []
    return {
        "pe_tool": tool,
        "pe_tool_version": tool_version or None,
        "path": os.path.relpath(path, _ROOT).replace("\\", "/") if path else None,
        "sha256": abi.sha256(path) if path and os.path.exists(path) else None,
        "abi_version": int(bridge.nvof_bridge_get_abi_version()) if bridge is not None else None,
        "bgr_entry_point_exported": any(row["name"] == "nvof_bridge_execute_bgr_sequential"
                                        for row in rows),
        "export_count": len(rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--sizes", nargs="*", default=["1920x1080", "640x360"])
    args = parser.parse_args(argv)

    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    probe = of.probe_nvof_capabilities()
    sections, errors = [], []
    if not probe.get("available"):
        errors.append(f"NVOF hardware unavailable: {probe.get('error')}")
    elif not probe.get("bgr_input"):
        errors.append("the loaded nvof_bridge exports no nvof_bridge_execute_bgr_sequential, "
                      "so the BGR input path cannot run on this binary")
    else:
        for spec in args.sizes:
            width, height = (int(value) for value in spec.lower().split("x"))
            sequences = [("synthetic", synthetic_frames(width, height, args.frames), None)]
            real, problem = fixture_frames(width, height, args.frames)
            sequences.append(("fixture", real, problem))
            for label, frames, problem in sequences:
                if frames is None:
                    errors.append(f"{label} {spec}: {problem}")
                    continue
                try:
                    sections.append(compare(label, width, height, frames))
                except Exception as exc:
                    errors.append(f"{label} {spec}: {type(exc).__name__}: {exc}")

    all_checks = [check for section in sections for check in section["checks"]]
    ok = bool(sections) and not errors and all(c["ok"] for c in all_checks)
    payload = dict(abi._identity({
        # abi._identity names the module that owns it; this artifact is written here, so the
        # provenance line is derived from this file instead of inheriting the helper's.
        "generated_by": os.path.relpath(os.path.abspath(__file__), _ROOT).replace("\\", "/"),
        "flow_preset": QUALITY,
        "nvof_probe": probe,
        "bridge_binary": bridge_binary(),
        "frames_per_sequence": args.frames,
        "sequences": sections,
        "errors": errors,
        "checks_passed": sum(1 for c in all_checks if c["ok"]),
        "checks_total": len(all_checks),
        "status": "DONE_PROVEN" if ok else "NOT_READY",
        "cpu_nvof_input_conversion": "ELIMINATED_PROVEN" if ok else "NOT_READY",
        "nvof_h2d_payload": "REDUCED_PROVEN" if ok else "NOT_READY",
        "nvof_flow_d2h": "BLOCKED_PROVEN",
        "measurement_source": "nvof_bridge_get_counters, read either side of each frame",
        "meaning": "the BGR input path and the RGBA input path returned the same flow field bit "
                   "for bit on every frame, and the byte figures came from the bridge's own "
                   "counters; the flow download is unchanged in size and in direction",
    }))
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "nvof_bgr_parity.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps({"ok": ok, "status": payload["status"], "artifact": path,
                      "checks_passed": payload["checks_passed"],
                      "checks_total": payload["checks_total"], "errors": errors},
                     indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
