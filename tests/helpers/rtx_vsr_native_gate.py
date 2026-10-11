"""Short opt-in real NVIDIA NvVFX VideoSuperRes CUDA smoke gate.

Example: python tests/helpers/rtx_vsr_native_gate.py --runtime-dir <licensed SDK runtime>
It executes the loaded VideoSuperRes effect; it does not mock the runtime.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rtx_vsr_engine import RtxVsr


def _cu(dll, symbol, args, result=ctypes.c_int):
    fn = getattr(dll, symbol)
    fn.argtypes = args
    fn.restype = result
    return fn


def _check(code, what):
    if code:
        raise RuntimeError(f"{what} returned CUDA error {code}")


def run(runtime_dir: Path, bridge: Path, evidence: Path) -> dict:
    os.add_dll_directory(str(runtime_dir.resolve()))
    cuda = ctypes.WinDLL("nvcuda.dll")
    cuinit = _cu(cuda, "cuInit", [ctypes.c_uint])
    devget = _cu(cuda, "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    ctxcreate = _cu(cuda, "cuCtxCreate_v2", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int])
    ctxdestroy = _cu(cuda, "cuCtxDestroy_v2", [ctypes.c_void_p])
    alloc = _cu(cuda, "cuMemAlloc_v2", [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t])
    free = _cu(cuda, "cuMemFree_v2", [ctypes.c_uint64])
    h2d = _cu(cuda, "cuMemcpyHtoD_v2", [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t])
    d2h = _cu(cuda, "cuMemcpyDtoH_v2", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t])
    _check(cuinit(0), "cuInit")
    dev = ctypes.c_int()
    _check(devget(ctypes.byref(dev), 0), "cuDeviceGet")
    context = ctypes.c_void_p()
    _check(ctxcreate(ctypes.byref(context), 0, dev.value), "cuCtxCreate")
    cases = [(1280, 720, 1920, 1080), (1920, 1080, 3840, 2160)]
    report = {"status": "running", "vendor_effect": "NvVFX_CreateEffect(VideoSuperRes)", "quality_levels": {}, "cases": []}
    all_output_hashes: dict[str, set[str]] = {str(case): set() for case in cases}
    try:
        for iw, ih, ow, oh in cases:
            pitch_i, pitch_o = iw * 4, ow * 4
            source = bytearray(pitch_i * ih)
            for y in range(ih):
                for x in range(iw):
                    o = y * pitch_i + x * 4
                    source[o:o+4] = bytes(((x * 3 + y) & 255, (y * 5 + x // 3) & 255,
                                           ((x ^ y) * 7) & 255, 255))
            srcptr, dstptr = ctypes.c_uint64(), ctypes.c_uint64()
            _check(alloc(ctypes.byref(srcptr), len(source)), "cuMemAlloc(source)")
            output_len = pitch_o * oh
            _check(alloc(ctypes.byref(dstptr), output_len), "cuMemAlloc(output)")
            _check(h2d(srcptr.value, (ctypes.c_ubyte * len(source)).from_buffer(source), len(source)), "cuMemcpyHtoD")
            try:
                for quality in range(1, 5):
                    with RtxVsr(bridge, runtime_dir, context.value, (iw, ih), (ow, oh), quality) as vsr:
                        vsr.process(srcptr.value, pitch_i, dstptr.value, pitch_o)
                        raw = bytearray(output_len)
                        _check(d2h((ctypes.c_ubyte * output_len).from_buffer(raw), dstptr.value, output_len), "cuMemcpyDtoH")
                        digest = hashlib.sha256(raw).hexdigest()
                        all_output_hashes[str((iw, ih, ow, oh))].add(digest)
                        row = {"input": [iw, ih], "output": [ow, oh], "quality": quality,
                               "output_sha256": digest, "output_bytes": output_len, "counters": vsr.counters()}
                        report["cases"].append(row)
                        report["quality_levels"][str(quality)] = "NvVFX_Run completed"
            finally:
                _check(free(dstptr.value), "cuMemFree(output)")
                _check(free(srcptr.value), "cuMemFree(source)")
        report["quality_output_variants"] = {k: len(v) for k, v in all_output_hashes.items()}
        report["status"] = "pass" if all(len(v) > 1 for v in all_output_hashes.values()) else "fail"
    finally:
        _check(ctxdestroy(context), "cuCtxDestroy")
    evidence.mkdir(parents=True, exist_ok=True)
    path = evidence / "rtx_vsr_native_gate.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "pass":
        raise RuntimeError(f"VSR quality levels failed to produce distinct output; evidence: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--bridge", type=Path, default=ROOT / "native" / "rtx_vsr_bridge.dll")
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "bench_out" / "optimization_20261010" / "rtx_vsr")
    args = parser.parse_args()
    result = run(args.runtime_dir, args.bridge, args.evidence_dir)
    print(json.dumps({"status": result["status"], "case_count": len(result["cases"]),
                      "quality_output_variants": result["quality_output_variants"]}, indent=2))


if __name__ == "__main__":
    main()
