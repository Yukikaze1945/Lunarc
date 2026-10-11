"""Short hardware gate for simultaneous, independent Feature 18 sessions.

Run after build_dlss_gpu_host.bat. This intentionally uses the host-color entry
to isolate NGX session lifetime from NVDEC/NVOF while checking two Feature 18
handles can coexist and closing the first does not tear down the second.
"""
import ctypes as C
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


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
    lib = C.CDLL(str(ROOT / "native" / "dlss_gpu_host.dll"))
    lib.dlss_gpu_host_open.argtypes = [C.c_uint32, C.c_uint32, C.c_wchar_p,
                                       C.POINTER(Options), C.POINTER(C.c_void_p)]
    lib.dlss_gpu_host_open.restype = C.c_int32
    lib.dlss_gpu_host_open_layer.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32,
                                             C.POINTER(Options), C.POINTER(C.c_void_p)]
    lib.dlss_gpu_host_open_layer.restype = C.c_int32
    lib.dlss_gpu_host_session_id.argtypes = [C.c_void_p]
    lib.dlss_gpu_host_session_id.restype = C.c_uint64
    lib.dlss_gpu_host_process_zero_motion.argtypes = [
        C.c_void_p, C.c_void_p, C.c_size_t, C.c_int32, C.c_void_p, C.c_size_t]
    lib.dlss_gpu_host_process_zero_motion.restype = C.c_int32
    lib.dlss_gpu_host_close.argtypes = [C.c_void_p]
    lib.dlss_gpu_host_last_error.restype = C.c_char_p
    options = Options(C.sizeof(Options), 0, 0, 1, 1, 1, 0, 0, 0, 1, 1)
    sizes = [(640, 360), (960, 544), (1088, 608)]
    handles = [C.c_void_p() for _ in sizes]
    ids = []

    def check(code, action):
        if code:
            raise RuntimeError(f"{action}: {code} {lib.dlss_gpu_host_last_error()!r}")

    def frame(index, reset):
        width, height = sizes[index]
        src = np.empty((height, width, 4), dtype=np.uint8)
        yy, xx = np.indices((height, width))
        src[..., 0] = (xx + 3 * yy) % 256
        src[..., 1] = (2 * xx + yy) % 256
        src[..., 2] = (xx + yy) % 256
        src[..., 3] = 255
        dst = np.empty_like(src)
        check(lib.dlss_gpu_host_process_zero_motion(
            handles[index], src.ctypes.data, src.strides[0], int(reset),
            dst.ctypes.data, dst.strides[0]), f"layer {index} process")
        assert dst.shape == src.shape
        return int(dst[..., :3].sum())

    try:
        check(lib.dlss_gpu_host_open(*sizes[0], str(ROOT / "nvngx_dlssnr.dll"),
                                     C.byref(options), C.byref(handles[0])), "open A")
        for index in (1, 2):
            check(lib.dlss_gpu_host_open_layer(handles[0], *sizes[index],
                                               C.byref(options), C.byref(handles[index])),
                  f"open {index}")
        ids = [int(lib.dlss_gpu_host_session_id(h)) for h in handles]
        assert len(set(ids)) == 3 and all(ids), ids
        sums = []
        for sequence in range(3):
            sums.append([frame(index, sequence == 0) for index in range(3)])
        lib.dlss_gpu_host_close(handles[0])
        handles[0] = C.c_void_p()
        survived = frame(1, False)
        check(lib.dlss_gpu_host_open_layer(handles[1], *sizes[0],
                                           C.byref(options), C.byref(handles[0])),
              "reopen A after close")
        reopened = int(lib.dlss_gpu_host_session_id(handles[0]))
        assert reopened not in ids and reopened > 0
        frame(0, True)
        print({"session_ids": ids, "reopened_id": reopened,
               "calls_per_layer": [3, 4, 3], "reset_flags": [[1, 0, 0]] * 3,
               "checksums": sums, "survived_checksum": survived,
               "result": "MULTI_SESSION_OK"})
    finally:
        for h in handles:
            if h.value:
                lib.dlss_gpu_host_close(h)


if __name__ == "__main__":
    main()
