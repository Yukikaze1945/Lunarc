"""Isolated Feature 18 FP16 headroom probe; never used by production export.

The diagnostic DLL reads the actual color texture just before NGX evaluation.
An external input/output scale is tested separately from any vendor exposure API.
"""
import ctypes as C
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
VALUES = (0.1, 0.5, 1.0, 1.5, 2.0, 4.0)


class Options(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32), ("preset", C.c_int32),
        ("style", C.c_int32), ("intensity", C.c_float),
        ("local_tone", C.c_float), ("local_structure", C.c_float),
        ("skin_structure", C.c_float), ("auto_mask", C.c_int32),
        ("ui_correction", C.c_int32), ("motion_scale_x", C.c_float),
        ("motion_scale_y", C.c_float),
    ]


def main():
    with os.add_dll_directory(str(ROOT / "native")):
        lib = C.CDLL(str(ROOT / "build" / "fp16_probe" /
                         "dlss_gpu_fp16_probe.dll"))
    lib.dlss_gpu_host_open.argtypes = [C.c_uint32, C.c_uint32, C.c_wchar_p,
                                       C.POINTER(Options), C.POINTER(C.c_void_p)]
    lib.dlss_gpu_host_open.restype = C.c_int32
    lib.dlss_gpu_host_process_zero_motion.argtypes = [
        C.c_void_p, C.c_void_p, C.c_size_t, C.c_int32, C.c_void_p, C.c_size_t]
    lib.dlss_gpu_host_process_zero_motion.restype = C.c_int32
    lib.dlss_gpu_host_probe_readback_input.argtypes = [
        C.c_void_p, C.c_void_p, C.c_size_t]
    lib.dlss_gpu_host_probe_readback_input.restype = C.c_int32
    lib.dlss_gpu_host_probe_set_generic_exposure.argtypes = [
        C.c_void_p, C.c_float, C.c_float]
    lib.dlss_gpu_host_probe_set_generic_exposure.restype = C.c_int32
    lib.dlss_gpu_host_close.argtypes = [C.c_void_p]
    lib.dlss_gpu_host_last_error.restype = C.c_char_p

    options = Options(C.sizeof(Options), 0, 0, 1, 1, 1, 0, 0, 0, 1, 1)
    width, height = 640, 360
    handle = C.c_void_p()

    def check(code, stage):
        if code:
            raise RuntimeError(f"{stage}: {code} {lib.dlss_gpu_host_last_error()!r}")

    def measure(value, input_scale, output_scale):
        src = np.full((height, width, 4), np.float16(value * input_scale),
                      dtype=np.float16)
        src[..., 3] = np.float16(1)
        dst = np.empty_like(src)
        readback = np.empty_like(src)
        check(lib.dlss_gpu_host_process_zero_motion(
            handle, src.ctypes.data, src.strides[0], 1,
            dst.ctypes.data, dst.strides[0]), "evaluate FP16")
        check(lib.dlss_gpu_host_probe_readback_input(
            handle, readback.ctypes.data, readback.strides[0]),
            "read back pre-NGX color texture")
        if not np.isfinite(dst).all() or not np.isfinite(readback).all():
            raise RuntimeError("non-finite FP16 value")
        center = dst[height // 4: 3 * height // 4,
                     width // 4: 3 * width // 4, :3].astype(np.float32)
        return {
            "source_half": float(src[0, 0, 0]),
            "pre_ngx_texture_min": float(np.min(readback[..., :3])),
            "pre_ngx_texture_max": float(np.max(readback[..., :3])),
            "pre_ngx_exact": bool(np.array_equal(src, readback)),
            "output_center_median": float(np.median(center) * output_scale),
            "output_center_min": float(np.min(center) * output_scale),
            "output_center_max": float(np.max(center) * output_scale),
            "output_full_max": float(np.max(dst[..., :3]) * output_scale),
        }

    try:
        check(lib.dlss_gpu_host_open(width, height,
              str(ROOT / "nvngx_dlssnr.dll"), C.byref(options),
              C.byref(handle)), "open Feature 18")
        raw = {str(v): measure(v, 1.0, 1.0) for v in VALUES}
        # Diagnostic external normalization, not an NGX exposure parameter.
        normalized = {str(v): measure(v, 0.25, 4.0) for v in VALUES}
        generic = {}
        for pre, exposure_scale in ((0.25, 1.0), (1.0, 0.25), (4.0, 1.0)):
            check(lib.dlss_gpu_host_probe_set_generic_exposure(
                handle, pre, exposure_scale), "set generic DLSS scalar keys")
            generic[f"pre={pre},scale={exposure_scale}"] = {
                str(v): measure(v, 1.0, 1.0) for v in (0.5, 2.0, 4.0)}
        print(json.dumps({"format": "DXGI_FORMAT_R16G16B16A16_FLOAT",
                          "feature": 18, "reset_each_evaluation": True,
                          "raw": raw, "external_input_quarter_output_times_four":
                          normalized, "generic_dlss_scalar_keys": generic},
                         indent=2))
    finally:
        if handle.value:
            lib.dlss_gpu_host_close(handle)


if __name__ == "__main__":
    main()
