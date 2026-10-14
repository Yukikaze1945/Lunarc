"""Micro benchmark for the residual backends (sections 5 and 24 of the task).

Measures the two hot-path stages on their own at 4K source:

    prepare_working_color   Lanczos-2 source -> working
    reconstruct             signed residual working -> source, plus composite

This explains what a backend costs. It is not an end-to-end result and must never be
reported as one: no decoding, no NVOF, no DLSSNR inference and no encoder live here.

Usage:
    python tests/unit/benchmark_residual_backends.py --backends python --iters 10 \
        --out perf_results/reference_python_baseline.json
    python tests/unit/benchmark_residual_backends.py --backends python,native --iters 12 \
        --out perf_results/native_residual_microbenchmark.json
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline  # noqa: E402
import resource_metrics  # noqa: E402

SOURCE_W = 3840
SOURCE_H = 2160
DEFAULT_SCALES = (0.75, 0.63, 0.50, 0.25)

ALIASES = {"python": "cpu_reference_python", "native": "cpu_reference_native",
           "auto": "auto"}


def _stats(samples):
    ordered = sorted(samples)
    return {
        "iterations": len(samples),
        "mean_ms": round(statistics.fmean(samples), 3),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
    }


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def _bit_exact_against_reference(requested, src_w, src_h, live_w, live_h, rng_seed):
    """One 4K frame through both backends: the native path must be bit-identical.

    Timing alone cannot show that a port kept the mathematics, and the small-case tests
    cannot show it at the size where buffer reuse and row chunking could go wrong.
    """
    rng = np.random.default_rng(rng_seed)
    source = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)
    dlss_low = rng.integers(0, 256, size=(live_h, live_w, 3), dtype=np.uint8)

    reference = pipeline.create_residual_backend("python", src_w, src_h, live_w, live_h)
    candidate = pipeline.create_residual_backend(ALIASES.get(requested, requested),
                                                  src_w, src_h, live_w, live_h)
    try:
        ref_low = np.zeros((live_h, live_w, 3), dtype=np.uint8)
        can_low = np.zeros((live_h, live_w, 3), dtype=np.uint8)
        reference.prepare_working_color(source, ref_low)
        candidate.prepare_working_color(source, can_low)
        down_equal = bool(np.array_equal(ref_low, can_low))
        down_max = int(np.abs(ref_low.astype(np.int32) - can_low.astype(np.int32)).max())

        ref_out = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        can_out = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        reference.reconstruct(source, ref_low, dlss_low, 1.0, ref_out)
        candidate.reconstruct(source, can_low, dlss_low, 1.0, can_out)
        rec_equal = bool(np.array_equal(ref_out, can_out))
        rec_max = int(np.abs(ref_out.astype(np.int32) - can_out.astype(np.int32)).max())
    finally:
        reference.close()
        candidate.close()
    return {"source": [src_w, src_h], "working": [live_w, live_h],
            "bit_equal_downsample": down_equal, "downsample_max_abs_diff": down_max,
            "bit_equal_reconstruct": rec_equal, "reconstruct_max_abs_diff": rec_max,
            "max_abs_diff": max(down_max, rec_max)}


def run_one(requested, scale, warmup, iters, rng_seed=20260919):
    live_w, live_h = pipeline.resolve_working_dims(SOURCE_W, SOURCE_H, scale)
    rng = np.random.default_rng(rng_seed)
    source = rng.integers(0, 256, size=(SOURCE_H, SOURCE_W, 3), dtype=np.uint8)
    dlss_low = rng.integers(0, 256, size=(live_h, live_w, 3), dtype=np.uint8)
    working = np.empty((live_h, live_w, 3), dtype=np.uint8)
    recon = np.empty((SOURCE_H, SOURCE_W, 3), dtype=np.uint8)

    backend = pipeline.create_residual_backend(ALIASES.get(requested, requested),
                                               SOURCE_W, SOURCE_H, live_w, live_h)
    try:
        for _ in range(warmup):
            backend.prepare_working_color(source, working)
            backend.reconstruct(source, working, dlss_low, 1.0, recon)

        down_ms = []
        recon_ms = []
        for _ in range(iters):
            down_ms.append(_timed(lambda: backend.prepare_working_color(source, working)))
            recon_ms.append(_timed(lambda: backend.reconstruct(source, working, dlss_low, 1.0, recon)))
        diagnostics = backend.diagnostics()
    finally:
        backend.close()

    parity = None
    if requested == "native":
        parity = _bit_exact_against_reference(requested, SOURCE_W, SOURCE_H, live_w, live_h, rng_seed)

    return {
        "requested_backend": requested,
        "parity_with_reference_4k": parity,
        "actual_backend": diagnostics.get("residual_backend", backend.name),
        "backend_version": getattr(backend, "version", None),
        "working_scale": scale,
        "source": [SOURCE_W, SOURCE_H],
        "working": [live_w, live_h],
        "working_color": _stats(down_ms),
        "reconstruct": _stats(recon_ms),
        "backend_diagnostics": diagnostics,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="python", help="comma list: python,native,auto")
    ap.add_argument("--scales", default=",".join(str(s) for s in DEFAULT_SCALES))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]
    if args.warmup < 1 or args.iters < 10:
        raise SystemExit("--warmup must be >= 1 and --iters >= 10 (task section 5)")

    results = []
    for requested in backends:
        for scale in scales:
            row = run_one(requested, scale, args.warmup, args.iters)
            results.append(row)
            print(f"{requested:<8} {row['actual_backend']:<22} scale {scale:<5} "
                  f"working {row['working']} down p50={row['working_color']['p50_ms']}ms "
                  f"recon p50={row['reconstruct']['p50_ms']}ms", flush=True)

    speedups = {}
    by_scale = {}
    for row in results:
        by_scale.setdefault(row["working_scale"], {})[row["requested_backend"]] = row
    for scale, per_backend in sorted(by_scale.items()):
        if "python" in per_backend and "native" in per_backend:
            py, na = per_backend["python"], per_backend["native"]
            speedups[str(scale)] = {
                "downsample_speedup": round(py["working_color"]["p50_ms"] / max(1e-9, na["working_color"]["p50_ms"]), 3),
                "reconstruct_speedup": round(py["reconstruct"]["p50_ms"] / max(1e-9, na["reconstruct"]["p50_ms"]), 3),
            }

    payload = {
        "kind": "residual_backend_microbenchmark",
        "environment": resource_metrics.environment_info(),
        "config": {"backends": backends, "scales": scales, "warmup": args.warmup,
                   "iterations": args.iters, "source": [SOURCE_W, SOURCE_H]},
        "results": results,
        "speedup_native_vs_python": speedups,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
