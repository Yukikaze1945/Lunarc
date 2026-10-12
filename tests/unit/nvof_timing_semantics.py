"""Shared surface for the NVOF timing-semantics gate and its evidence report.

The rule these helpers measure is one-number-per-stage: every key the pipeline profiler
publishes for optical flow has to come from exactly one QueryPerformanceCounter span in
native/nvof_bridge.cpp, and a span the bridge does not measure has to be reported as
unavailable rather than as 0.0. A fabricated zero used to sit in the same dict next to real
measurements and averaged into a p50 like a real number.

test_nvof_timing_semantics.py asserts on this; the cleanup artifact is written from it too,
so the report and the gate cannot drift apart.
"""

import ctypes
import json
import os
import sys

import numpy as np

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import optical_flow as of  # noqa: E402

# Native struct field -> the one reported key that may carry it. Two keys reading one field
# is the alias this exists to prevent, so the mapping has to stay injective.
FIELD_TO_KEY = {
    "upload_ms": "nvof_upload_ms",
    "submit_cpu_ms": "nvof_submit_cpu_ms",
    "dependency_wait_ms": "nvof_dependency_wait_ms",
    "download_ms": "nvof_download_ms",
    "postprocess_wait_inclusive_ms": "nvof_postprocess_wait_inclusive_ms",
    "total_native_ms": "nvof_total_native_ms",
}

# Spans the bridge cannot time on its own. They stay in the report so the absence is
# visible, and they must always read as unavailable.
ALWAYS_UNMEASURED = ("nvof_convert_ms", "nvof_gpu_execute_ms")

# Spans that exist only on one entry point, and the paths allowed to report a number for
# them. The rule is the same one-number-per-span rule, just conditional: a byte-map span
# published by an RGBA execute was measured by something else and is being passed off as
# the kernel it names.
PATH_SPECIFIC_FIELDS = {
    "input_convert_wait_ms": ("nvof_input_convert_wait_ms", ("bgr",)),
}

# Keys the old report carried. Each one either aliased another number or was a fabricated
# zero, so none of them may come back.
RETIRED_KEYS = ("upload_ms", "download_ms", "convert_ms", "upscale_ms", "nvof_upload",
                "nvof_upscale", "nvof_download", "nvof_gpu_execute_ms_inclusive",
                "postprocess_gpu_ms", "d2h_ms", "native_total_ms", "nvof_to_host_total_ms",
                "nvof_execute_ms")


def hardware_present():
    probe = of.probe_nvof_capabilities()
    return bool(probe.get("available") and probe.get("hardware")), probe


def _frames(width, height):
    a = np.zeros((height, width, 3), dtype=np.uint8)
    b = np.zeros((height, width, 3), dtype=np.uint8)
    # A translated block: real motion for the hardware to find, deterministic to build.
    a[100:400, 100:600] = 200
    b[140:440, 180:680] = 200
    rng = np.random.default_rng(7)
    a = np.clip(a.astype(np.int16) + rng.integers(0, 24, a.shape), 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.int16) + rng.integers(0, 24, b.shape), 0, 255).astype(np.uint8)
    return a, b


def collect_rows(frames=5, width=1920, height=1080, quality="4M", grid=4,
                 sequential_frames=2, bgr_frames=2):
    """Run real executes and pair every reported key with the raw struct field behind it.

    The raw struct is read straight from the bridge after each call, so this cannot be
    satisfied by a Python-side dict that made its numbers up.
    """
    engine = of.OpticalFlowEngine("nvof", quality=quality, grid=grid)
    inner = engine._engine
    provider = engine.provider_info()
    bgr_capable = bool(of.probe_nvof_capabilities().get("bgr_input"))
    rows = []

    def raw_timing():
        raw = of.NvOFBridgeTiming()
        code = inner._bridge.nvof_bridge_get_last_timing(
            inner._session, ctypes.byref(raw))
        assert code == 0, "the bridge refused to hand back its own timing"
        return {name: getattr(raw, name) for name, _ in raw._fields_}

    try:
        a, b = _frames(width, height)
        prev, cur = a, b
        for index in range(frames):
            flow = engine.compute_flow(prev, cur)
            assert flow.shape[2] == 2 and np.all(np.isfinite(flow)), "flow is not finite"
            rows.append({
                "path": "timed",
                "frame": index,
                "provider": provider,
                "reported": dict(engine.get_last_timing()),
                "raw": raw_timing(),
            })
            prev, cur = cur, prev

        for index in range(sequential_frames):
            flow = engine.compute_flow_sequential(cur, prev)
            assert np.all(np.isfinite(flow)), "flow is not finite"
            rows.append({
                "path": "sequential",
                "frame": index,
                "provider": provider,
                "reported": dict(engine.get_last_timing()),
                "raw": raw_timing(),
            })

        if bgr_capable:
            # A fresh engine, because a session that has been fed RGBA frames already carries
            # history the BGR frames did not write, and the point of these rows is that the
            # byte-map span belongs to this entry point and to no other.
            engine.close()
            engine = of.OpticalFlowEngine("nvof", quality=quality, grid=grid)
            inner = engine._engine
            provider = engine.provider_info()
            for index in range(bgr_frames):
                flow = engine.compute_flow_bgr24(prev if index else None, cur,
                                                 reset_temporal_hints=index == 0)
                assert np.all(np.isfinite(flow)), "flow is not finite"
                rows.append({
                    "path": "bgr",
                    "frame": index,
                    "provider": provider,
                    "reported": dict(engine.get_last_timing()),
                    "raw": raw_timing(),
                })
    finally:
        engine.close()
    return rows


def analyze(rows):
    """Check the collected rows against the one-number-per-stage rule."""
    problems = []
    distinct_frames = 0
    for row in rows:
        reported, raw = row["reported"], row["raw"]
        for retired in RETIRED_KEYS:
            if retired in reported:
                problems.append(f"frame {row['frame']}: retired key {retired} is back")
        for field, key in FIELD_TO_KEY.items():
            value = reported.get(key, "<missing>")
            if value == "<missing>":
                problems.append(f"frame {row['frame']}: {key} is not reported at all")
                continue
            measured = raw[field]
            if measured < 0.0:
                if value != of.NVOF_TIMING_UNAVAILABLE:
                    problems.append(
                        f"frame {row['frame']}: {key} reports {value!r} for a span the "
                        f"bridge marked unmeasured ({measured})")
            elif value != measured:
                problems.append(
                    f"frame {row['frame']}: {key} reports {value} but the bridge measured "
                    f"{measured}")
        for key in ALWAYS_UNMEASURED:
            if key not in reported:
                problems.append(f"frame {row['frame']}: {key} vanished from the report")
            elif isinstance(reported[key], str):
                continue
            else:
                problems.append(f"frame {row['frame']}: {key} carries the number "
                                f"{reported[key]!r}, but this build cannot measure it")
        # Spans only some entry points produce. A number on a path that does not run the work
        # is the same alias as two stages sharing one measurement, just harder to see.
        for field, (key, paths) in PATH_SPECIFIC_FIELDS.items():
            if field not in raw:
                problems.append(
                    f"frame {row['frame']}: the bridge reports no {field} at all, so "
                    f"{key} cannot be its measurement")
                continue
            measured = raw[field]
            if measured < 0.0:
                if reported.get(key) != of.NVOF_TIMING_UNAVAILABLE:
                    problems.append(
                        f"frame {row['frame']}: {key} reports {reported.get(key)!r} for a span "
                        f"the bridge marked unmeasured ({measured})")
                if row["path"] in paths:
                    problems.append(
                        f"frame {row['frame']}: the {row['path']} path reports {field} "
                        f"unmeasured, so the work that span times never ran")
                continue
            if reported.get(key) != measured:
                problems.append(
                    f"frame {row['frame']}: {key} reports {reported.get(key)} but the bridge "
                    f"measured {measured}")
            whole = reported.get("nvof_total_native_ms")
            if isinstance(whole, float) and measured > whole + 0.05:
                problems.append(
                    f"frame {row['frame']}: {key} = {measured} ms does not fit inside "
                    f"nvof_total_native_ms = {whole} ms")
            if row["path"] not in paths:
                problems.append(
                    f"frame {row['frame']}: the {row['path']} path measured {field} "
                    f"({measured} ms), which no work on that path produces")
        numeric = {key: reported[key] for key in FIELD_TO_KEY.values()
                   if isinstance(reported.get(key), float)}
        seen = {}
        for key, value in numeric.items():
            if value == 0.0:
                problems.append(f"frame {row['frame']}: {key} measured exactly 0.0 ms")
            seen.setdefault(value, []).append(key)
        shared = {value: keys for value, keys in seen.items() if len(keys) > 1}
        # Equal rounded spans are possible. Provenance is checked against each
        # native field above; numerical uniqueness is not a timing contract.
        if not shared and len(numeric) == len(FIELD_TO_KEY):
            distinct_frames += 1
        # A part cannot be larger than the whole: total_native_ms is the wall clock around
        # the same execute that produced the individual spans, and python_total_ms wraps
        # that call. Both orderings are semantic facts, not performance claims.
        whole = numeric.get("nvof_total_native_ms")
        if whole is not None:
            for key, value in numeric.items():
                if key != "nvof_total_native_ms" and value > whole + 0.05:
                    problems.append(
                        f"frame {row['frame']}: {key} = {value} ms exceeds the whole native "
                        f"execute ({whole} ms), so one of the two is mislabeled")
            python_total = reported.get("python_total_ms")
            if isinstance(python_total, float) and python_total + 1.0 < whole:
                problems.append(
                    f"frame {row['frame']}: the Python call around the native execute took "
                    f"{python_total} ms, less than the {whole} ms it contains")

    sequential = [r for r in rows if r["path"] == "sequential"]
    for row in sequential:
        if row["reported"].get("nvof_dependency_wait_ms") != of.NVOF_TIMING_UNAVAILABLE:
            problems.append(
                "the sequential path does not time the hardware wait, so "
                f"nvof_dependency_wait_ms must be unavailable, got "
                f"{row['reported'].get('nvof_dependency_wait_ms')!r}")

    timed = [r for r in rows if r["path"] == "timed"]
    bgr = [r for r in rows if r["path"] == "bgr"]
    sample = {key: [r["reported"][key] for r in timed] for key in FIELD_TO_KEY.values()
              if all(isinstance(r["reported"].get(key), float) for r in timed)}
    return {
        "pass": not problems,
        "problems": problems,
        "frames_timed": len(timed),
        "frames_sequential": len(sequential),
        "frames_bgr": len(bgr),
        "frames_with_six_distinct_spans": distinct_frames,
        "frames_native_mapped": sum(all(r['reported'].get(key)==r['raw'][field]
                                       for field,key in FIELD_TO_KEY.items()) for r in timed),
        "field_to_key": dict(FIELD_TO_KEY),
        "path_specific_fields": {key: {"field": field, "paths": sorted(paths)}
                                 for field, (key, paths) in PATH_SPECIFIC_FIELDS.items()},
        "always_unmeasured": list(ALWAYS_UNMEASURED),
        "retired_keys_still_absent": not any(
            retired in r["reported"] for r in rows for retired in RETIRED_KEYS),
        "per_key_ms": {key: [round(v, 4) for v in values] for key, values in sample.items()},
        "native_struct_fields": [name for name, _ in of.NvOFBridgeTiming._fields_],
    }


# What each published key is, in one line, so a report reader can check the name against
# the span instead of against another number.
STAGE_MEANING = {
    "nvof_upload_ms": "the ABGR host to device copy that feeds this execute",
    "nvof_submit_cpu_ms": "CPU time inside nvOFExecute: the submit and its return, not the "
                          "hardware's own duration",
    "nvof_dependency_wait_ms": "waiting for the input frame's producer to finish, timed "
                              "separately from the execute",
    "nvof_download_ms": "the device to host copy of the flow grid",
    "nvof_postprocess_wait_inclusive_ms": "the S10.5 convert-and-upscale kernel plus the "
                                          "synchronize that waits for it, as one inclusive "
                                          "span",
    "nvof_total_native_ms": "wall clock around the whole native bridge call",
    "nvof_input_convert_wait_ms": "the BGR to RGBA byte-map kernel plus the synchronize that "
                                  "waits for it, measured on the BGR input entry point only",
    "nvof_convert_ms": "not separable: the conversion and the upscale are one fused kernel",
    "nvof_gpu_execute_ms": "not separable: the execute and the synchronize that waits for "
                           "it are one span on this hardware",
}


def write_report(path, rows=None, **kwargs):
    rows = rows if rows is not None else collect_rows(**kwargs)
    report = {"rows": rows, "analysis": analyze(rows), "stage_meaning": STAGE_MEANING,
              "retired_keys": list(RETIRED_KEYS)}
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True))
            handle.write(chr(10))
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="", help="write the report here")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args(argv)

    present, probe = hardware_present()
    if not present:
        print(json.dumps({"pass": False, "skipped_reason": probe}, indent=2,
                         sort_keys=True))
        return 2
    report = write_report(args.out, collect_rows(frames=args.frames, width=args.width,
                                                 height=args.height))
    analysis = report["analysis"]
    print(json.dumps({"pass": analysis["pass"], "problems": analysis["problems"],
                      "per_key_ms": analysis["per_key_ms"],
                      "frames_timed": analysis["frames_timed"],
                      "frames_sequential": analysis["frames_sequential"]},
                     indent=2, sort_keys=True))
    return 0 if analysis["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
