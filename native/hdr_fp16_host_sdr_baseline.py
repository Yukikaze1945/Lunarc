"""Deterministic three-frame SDR baseline for the production GPU host.

The baseline exists so an additive colour-format change to dlss_gpu_host.dll can
be shown to leave the shipped RGBA8 behaviour alone: the same fixture, the same
extent, the same NVOF flow and the same baked options run through
``dlss_gpu_host_process_device_color`` and the output bytes are hashed per frame.
Run it before a rebuild and after it; the two JSON files must agree.

    python native/hdr_fp16_host_sdr_baseline.py bench_out/hdr_fp16_host_connect/sdr_pre.json

It is a device-colour run on purpose: that is the entry whose row width an FP16
session changes, so an RGBA8 session that still hashes identically is evidence
about the path that was touched, not about an untouched one.
"""
import ctypes as C
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "native"
sys.path.insert(0, str(ROOT))

WIDTH, HEIGHT = 256, 256
FRAMES = 3
OPTIONS = (0, 0, 1.0, 1.0, 1.0, 0.0, 0, 0, 1.0, 1.0)


class _Options(C.Structure):
    _fields_ = [("struct_size", C.c_uint32)] + [
        ("preset", C.c_int32), ("style", C.c_int32), ("intensity", C.c_float),
        ("local_tone", C.c_float), ("local_structure", C.c_float),
        ("skin_structure", C.c_float), ("auto_mask", C.c_int32),
        ("ui_correction", C.c_int32), ("motion_scale_x", C.c_float),
        ("motion_scale_y", C.c_float)]


class _Counters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32)] + [
        (name, C.c_uint64) for name in (
            "host_color_stage_count", "host_color_stage_bytes",
            "device_color_copy_count", "device_color_copy_bytes",
            "output_readback_count", "output_readback_bytes")]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fixture(index):
    """The one input generator this gate knows; identical on every run."""
    y, x = np.indices((HEIGHT, WIDTH))
    plane = np.empty((HEIGHT, WIDTH, 4), np.uint8)
    plane[..., 0] = (x * 7 + y + index * 3) % 256
    plane[..., 1] = (x + y * 5 + index * 11) % 256
    plane[..., 2] = (x * 3 + y * 2 + index * 29) % 256
    plane[..., 3] = 255
    return np.ascontiguousarray(plane)


def run_baseline():
    cuda = C.WinDLL("nvcuda.dll")
    nvof = C.WinDLL(str(NATIVE / "nvof_bridge.dll"))
    host = C.WinDLL(str(NATIVE / "dlss_gpu_host.dll"))

    def cuda_error(step, rc):
        raise RuntimeError(f"{step} failed (CUDA {int(rc)})")

    for name, argtypes in (
            ("cuCtxPushCurrent_v2", [C.c_void_p]),
            ("cuCtxPopCurrent_v2", [C.POINTER(C.c_void_p)]),
            ("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t]),
            ("cuMemFree_v2", [C.c_uint64]),
            ("cuMemcpyHtoD_v2", [C.c_uint64, C.c_void_p, C.c_size_t]),
            ("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t])):
        getattr(cuda, name).argtypes = argtypes
        getattr(cuda, name).restype = C.c_int

    nvof.nvof_bridge_create.argtypes = [C.c_uint32, C.c_uint32, C.c_uint32,
        C.c_uint32, C.POINTER(C.c_void_p)]
    nvof.nvof_bridge_create.restype = C.c_int32
    nvof.nvof_bridge_execute_rgba_device_sequential.argtypes = [C.c_void_p,
        C.c_uint64, C.c_uint32, C.c_void_p, C.c_int32, C.c_void_p]
    nvof.nvof_bridge_execute_rgba_device_sequential.restype = C.c_int32
    nvof.nvof_bridge_get_output_deviceptr.argtypes = [C.c_void_p,
        C.POINTER(C.c_uint64), C.POINTER(C.c_size_t)]
    nvof.nvof_bridge_get_output_deviceptr.restype = C.c_int32
    nvof.nvof_bridge_get_session_info.argtypes = [C.c_void_p, C.POINTER(C.c_uint32),
        C.POINTER(C.c_void_p), C.POINTER(C.c_uint32)]
    nvof.nvof_bridge_get_session_info.restype = C.c_int32
    nvof.nvof_bridge_get_last_error.restype = C.c_char_p

    host.dlss_gpu_host_abi_version.restype = C.c_uint32
    host.dlss_gpu_host_open.argtypes = [C.c_uint32, C.c_uint32, C.c_wchar_p,
        C.POINTER(_Options), C.POINTER(C.c_void_p)]
    host.dlss_gpu_host_open.restype = C.c_int32
    host.dlss_gpu_host_process_device_color.argtypes = [C.c_void_p, C.c_void_p,
        C.c_uint64, C.c_size_t, C.c_void_p, C.c_uint64, C.c_size_t, C.c_int32,
        C.c_void_p, C.c_size_t]
    host.dlss_gpu_host_process_device_color.restype = C.c_int32
    host.dlss_gpu_host_get_transfer_counters.argtypes = [C.c_void_p,
        C.POINTER(_Counters)]
    host.dlss_gpu_host_get_transfer_counters.restype = C.c_int32
    host.dlss_gpu_host_close.argtypes = [C.c_void_p]
    host.dlss_gpu_host_last_error.restype = C.c_char_p

    def host_error(step, code):
        detail = host.dlss_gpu_host_last_error()
        raise RuntimeError(f"{step} failed ({code}): "
                           f"{detail.decode('utf-8', 'replace') if detail else ''}")

    session = C.c_void_p()
    if nvof.nvof_bridge_create(WIDTH, HEIGHT, 4, 10, C.byref(session)):
        raise RuntimeError(f"nvof_bridge_create: "
                           f"{nvof.nvof_bridge_get_last_error()!r}")
    device_index = C.c_uint32()
    context = C.c_void_p()
    owner = C.c_uint32()
    if nvof.nvof_bridge_get_session_info(session, C.byref(device_index),
                                         C.byref(context), C.byref(owner)):
        raise RuntimeError(f"nvof_bridge_get_session_info: "
                           f"{nvof.nvof_bridge_get_last_error()!r}")

    pitch = WIDTH * 4
    plane_bytes = pitch * HEIGHT
    previous = C.c_void_p()
    if cuda.cuCtxPushCurrent_v2(context):
        raise RuntimeError("cuCtxPushCurrent failed")
    try:
        source = C.c_uint64()
        if cuda.cuMemAlloc_v2(C.byref(source), plane_bytes):
            raise RuntimeError("cuMemAlloc(source) failed")
        try:
            handle = C.c_void_p()
            options = _Options(C.sizeof(_Options), *OPTIONS)
            code = host.dlss_gpu_host_open(WIDTH, HEIGHT,
                str(ROOT / "nvngx_dlssnr.dll"), C.byref(options),
                C.byref(handle))
            if code:
                host_error("open", code)
            frames = []
            try:
                for index in range(FRAMES):
                    plane = fixture(index)
                    if (rc := cuda.cuMemcpyHtoD_v2(source, plane.ctypes.data,
                                                    plane_bytes)):
                        cuda_error("cuMemcpyHtoD", rc)
                    if nvof.nvof_bridge_execute_rgba_device_sequential(
                            session, source.value, pitch, context,
                            int(index == 0), None):
                        raise RuntimeError(
                            f"nvof execute: "
                            f"{nvof.nvof_bridge_get_last_error()!r}")
                    flow_ptr = C.c_uint64()
                    flow_pitch = C.c_size_t()
                    if nvof.nvof_bridge_get_output_deviceptr(session,
                            C.byref(flow_ptr), C.byref(flow_pitch)):
                        raise RuntimeError(
                            f"flow pointer: "
                            f"{nvof.nvof_bridge_get_last_error()!r}")
                    out = np.empty((HEIGHT, WIDTH, 4), np.uint8)
                    code = host.dlss_gpu_host_process_device_color(handle,
                        context, source.value, pitch, context, flow_ptr.value,
                        flow_pitch.value, int(index == 0), out.ctypes.data,
                        out.strides[0])
                    if code:
                        host_error(f"frame {index}", code)
                    frames.append({
                        "index": index,
                        "reset": int(index == 0),
                        "input_sha256": hashlib.sha256(plane.tobytes()).hexdigest(),
                        "output_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
                        "flow_pitch": int(flow_pitch.value),
                    })
                counters = _Counters(C.sizeof(_Counters))
                if host.dlss_gpu_host_get_transfer_counters(handle,
                                                            C.byref(counters)):
                    host_error("counters", 1)
            finally:
                host.dlss_gpu_host_close(handle)
        finally:
            cuda.cuMemFree_v2(source.value)
    finally:
        cuda.cuCtxPopCurrent_v2(C.byref(previous))
    nvof.nvof_bridge_destroy(session)

    return {
        "kind": "sdr_rgba8_baseline",
        "extent": [WIDTH, HEIGHT],
        "frames": FRAMES,
        "options": list(OPTIONS),
        "entry_point": "dlss_gpu_host_process_device_color",
        "host_abi": int(host.dlss_gpu_host_abi_version()),
        "host_dll_sha256": sha256_file(NATIVE / "dlss_gpu_host.dll"),
        "interop_dll_sha256": sha256_file(NATIVE / "nvof_d3d12_interop.dll"),
        "nvof_dll_sha256": sha256_file(NATIVE / "nvof_bridge.dll"),
        "counters": {name: getattr(counters, name) for name, _ in
                     _Counters._fields_ if name != "struct_size"},
        "frames_run": frames,
        "aggregate_output_sha256": hashlib.sha256(
            "".join(f["output_sha256"] for f in frames).encode()).hexdigest(),
    }


def main(argv):
    result = run_baseline()
    text = json.dumps(result, indent=2)
    if len(argv) > 1:
        Path(argv[1]).parent.mkdir(parents=True, exist_ok=True)
        Path(argv[1]).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
