"""Two-frame hardware correctness gate for NVOF CUDA -> D3D12 -> DLSSNR.

Run from the repository root with ``python native/dlss_gpu_host_smoke.py`` after
building nvof_bridge.dll, nvof_d3d12_interop.dll and dlss_gpu_host.dll. This
does not run during CPU-only unit tests.

The gate is the device path: two frames through the GPU host with the flow
descriptor left on the card, and zero device-to-host copies reported by NVOF.
It used to also compare those bytes against the closed dlssnr_host.dll session,
which is retired; that arm runs only when ``dlss_engine`` is still present, and
its absence is printed rather than passed off as a result.
"""
import ctypes as C
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from optical_flow import NvOFHardwareEngine  # noqa: E402

WIDTH, HEIGHT = 1920, 1080


class Options(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32), ("preset", C.c_int32),
        ("style", C.c_int32), ("intensity", C.c_float),
        ("local_tone", C.c_float), ("local_structure", C.c_float),
        ("skin_structure", C.c_float), ("auto_mask", C.c_int32),
        ("ui_correction", C.c_int32), ("motion_scale_x", C.c_float),
        ("motion_scale_y", C.c_float),
    ]


class TransferCounters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32)] + [
        (name, C.c_uint64) for name in (
            "host_color_stage_count", "host_color_stage_bytes",
            "device_color_copy_count", "device_color_copy_bytes",
            "output_readback_count", "output_readback_bytes")]


class DeviceOutput(C.Structure):
    _fields_ = [("struct_size", C.c_uint32),
        ("width", C.c_uint32), ("height", C.c_uint32),
        ("dxgi_format", C.c_uint32), ("d3d12_device", C.c_void_p),
        ("d3d12_resource", C.c_void_p), ("d3d12_fence", C.c_void_p),
        ("fence_value", C.c_uint64)]


def make_frames():
    y, x = np.indices((HEIGHT, WIDTH))
    first = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    first[..., 0] = (x * 7 + y) % 256
    first[..., 1] = (x + y * 5) % 256
    first[..., 2] = (x * 3 + y * 2) % 256
    frames = [first]
    for amount in (1, 2):
        shifted = np.empty_like(first)
        shifted[:, amount:] = first[:, :-amount]
        shifted[:, :amount] = first[:, :1]
        frames.append(shifted)
    return frames


def rgba_from_bgr(bgr):
    rgba = np.empty((*bgr.shape[:2], 4), np.uint8)
    rgba[..., 0] = bgr[..., 2]
    rgba[..., 1] = bgr[..., 1]
    rgba[..., 2] = bgr[..., 0]
    rgba[..., 3] = 255
    return rgba


def load_gpu_host():
    lib = C.CDLL(str(ROOT / "native" / "dlss_gpu_host.dll"))
    lib.dlss_gpu_host_abi_version.restype = C.c_uint32
    assert lib.dlss_gpu_host_abi_version() in (2, 3), (
        "this smoke drives the guided entries, which ABI 2 and 3 both have")
    lib.dlss_gpu_host_open.argtypes = [C.c_uint32, C.c_uint32, C.c_wchar_p,
        C.POINTER(Options), C.POINTER(C.c_void_p)]
    lib.dlss_gpu_host_open.restype = C.c_int32
    lib.dlss_gpu_host_process.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t,
        C.c_void_p, C.c_uint64, C.c_size_t, C.c_int32, C.c_void_p, C.c_size_t]
    lib.dlss_gpu_host_process.restype = C.c_int32
    lib.dlss_gpu_host_process_device_color.argtypes = [C.c_void_p,
        C.c_void_p, C.c_uint64, C.c_size_t, C.c_void_p, C.c_uint64,
        C.c_size_t, C.c_int32, C.c_void_p, C.c_size_t]
    lib.dlss_gpu_host_process_device_color.restype = C.c_int32
    lib.dlss_gpu_host_process_device_color_resource.argtypes = [C.c_void_p,
        C.c_void_p, C.c_uint64, C.c_size_t, C.c_void_p, C.c_uint64,
        C.c_size_t, C.c_int32, C.POINTER(DeviceOutput)]
    lib.dlss_gpu_host_process_device_color_resource.restype = C.c_int32
    lib.dlss_gpu_host_readback_output.argtypes = [C.c_void_p,
        C.c_void_p, C.c_size_t]
    lib.dlss_gpu_host_readback_output.restype = C.c_int32
    lib.dlss_gpu_host_get_transfer_counters.argtypes = [C.c_void_p,
        C.POINTER(TransferCounters)]
    lib.dlss_gpu_host_get_transfer_counters.restype = C.c_int32
    lib.dlss_gpu_host_close.argtypes = [C.c_void_p]
    lib.dlss_gpu_host_last_error.restype = C.c_char_p
    return lib


def retired_host_outputs(frames, colors):
    """What the closed dlssnr_host.dll answers for the same frames, or None.

    Returns None when the retired wrapper or its binary is gone: the module was
    deleted with the host it loaded, so an archival A/B run on a current tree has
    to say it compared nothing instead of comparing nothing quietly.
    """
    try:
        from dlss_engine import Live
    except ImportError as exc:
        print(f"reference_ab=unavailable {exc}")
        return None
    settings = {"preset": 0, "style": 0, "intensity": 1.0,
        "local_tone": 1.0, "local_struct": 1.0, "skin_struct": 0.0,
        "use_auto_mask": 0, "ui_correction": 0, "guidance_mode": 1,
        "motion_scale_x": 1.0,
        "motion_scale_y": 1.0}
    reference_engine = NvOFHardwareEngine("4M", grid=4, preset="medium")
    outputs = []
    live = None
    try:
        live = Live(WIDTH, HEIGHT, settings)
        for index, color in enumerate(colors):
            flow = reference_engine.compute_flow_bgr24(
                frames[index] if index == 0 else None, frames[index + 1],
                reset_temporal_hints=(index == 0))
            outputs.append(live.process(color, flow,
                reset=(index == 0), raise_on_error=True).copy())
    finally:
        if live is not None:
            live.close()
        reference_engine.close()
    return outputs


def main():
    frames = make_frames()
    colors = [rgba_from_bgr(frame) for frame in frames[1:]]
    depth = np.zeros((HEIGHT, WIDTH), np.float32)
    options = Options(C.sizeof(Options), 0, 0, 1, 1, 1, 0, 0, 0, 1, 1)
    lib = load_gpu_host()
    engine = NvOFHardwareEngine("4M", grid=4, preset="medium")
    handle = C.c_void_p()
    gpu_outputs = []
    try:
        code = lib.dlss_gpu_host_open(WIDTH, HEIGHT,
            str(ROOT / "nvngx_dlssnr.dll"), C.byref(options), C.byref(handle))
        if code:
            raise RuntimeError(f"GPU host open {code}: {lib.dlss_gpu_host_last_error()!r}")
        for index, color in enumerate(colors):
            resource = engine.compute_flow_bgr24_device(
                frames[index] if index == 0 else None, frames[index + 1],
                reset_temporal_hints=(index == 0))
            output = np.empty_like(color)
            code = lib.dlss_gpu_host_process(handle, color.ctypes.data,
                color.strides[0], resource["cuda_context"], resource["device_ptr"],
                resource["pitch_bytes"], int(index == 0),
                output.ctypes.data, output.strides[0])
            if code:
                raise RuntimeError(
                    f"GPU host frame {index} code {code}: {lib.dlss_gpu_host_last_error()!r}")
            gpu_outputs.append(output)
        counters = engine.get_counters()
        assert counters["execute_count"] == 2, counters
        assert counters["d2h_copy_count"] == counters["d2h_bytes"] == 0, counters
    finally:
        if handle.value:
            lib.dlss_gpu_host_close(handle)
        engine.close()

    host_outputs = retired_host_outputs(frames, colors)
    if host_outputs is None:
        print(f"gpu_frames={len(gpu_outputs)} ab_against_retired_host=skipped "
              f"nvof_device_d2h_copies=0 nvof_device_d2h_bytes=0")
        return
    for index, (gpu, host) in enumerate(zip(gpu_outputs, host_outputs)):
        if not np.array_equal(gpu, host):
            difference = np.abs(gpu.astype(np.int16) - host.astype(np.int16))
            raise AssertionError(f"frame {index}: mean={difference.mean()} "
                f"max={difference.max()} bytes={np.count_nonzero(difference)}")
        print(f"frame={index} rgba_bytes={gpu.nbytes} output=identical "
              f"byte_sum={int(gpu.sum())}")
    print("nvof_device_d2h_copies=0 nvof_device_d2h_bytes=0")


if __name__ == "__main__":
    main()
