"""Compare one scaled CUDA frame with the existing FFmpeg scale_cuda path."""
import ctypes as C
import os
from pathlib import Path
import subprocess

from nvdec_bridge_smoke import (FFMPEG_BIN, Frame, ROOT, bind_counters,
                                read_counters)


class Copy2D(C.Structure):
    _fields_ = [
        ("srcXInBytes", C.c_size_t), ("srcY", C.c_size_t),
        ("srcMemoryType", C.c_int), ("srcHost", C.c_void_p),
        ("srcDevice", C.c_uint64), ("srcArray", C.c_void_p),
        ("srcPitch", C.c_size_t), ("dstXInBytes", C.c_size_t),
        ("dstY", C.c_size_t), ("dstMemoryType", C.c_int),
        ("dstHost", C.c_void_p), ("dstDevice", C.c_uint64),
        ("dstArray", C.c_void_p), ("dstPitch", C.c_size_t),
        ("WidthInBytes", C.c_size_t), ("Height", C.c_size_t),
    ]


def main():
    source = ROOT / "bench_out" / "fixture_4k_300.mp4"
    width, height = 1920, 1080
    command = ["ffmpeg", "-v", "error", "-hwaccel", "cuda",
               "-hwaccel_output_format", "cuda", "-i", str(source),
               "-vf", f"scale_cuda=w={width}:h={height}:format=nv12,hwdownload,format=nv12",
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]
    reference = subprocess.check_output(command, timeout=30)
    assert len(reference) == width * height * 3 // 2

    cookie = os.add_dll_directory(str(FFMPEG_BIN))
    try:
        lib = C.CDLL(str(ROOT / "native" / "nvdec_bridge.dll"))
        cuda = C.WinDLL("nvcuda.dll")
        lib.nvdec_bridge_open_scaled.argtypes = [C.c_char_p, C.c_uint32,
                                                 C.c_uint32, C.POINTER(C.c_void_p)]
        lib.nvdec_bridge_open_scaled.restype = C.c_int32
        lib.nvdec_bridge_next.argtypes = [C.c_void_p, C.POINTER(Frame)]
        lib.nvdec_bridge_next.restype = C.c_int32
        lib.nvdec_bridge_close.argtypes = [C.c_void_p]
        lib.nvdec_bridge_last_error.restype = C.c_char_p
        bind_counters(lib)
        cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        cuda.cuCtxPushCurrent_v2.restype = C.c_int
        cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        cuda.cuCtxPopCurrent_v2.restype = C.c_int
        cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(Copy2D)]
        cuda.cuMemcpy2D_v2.restype = C.c_int

        handle = C.c_void_p()
        if lib.nvdec_bridge_open_scaled(os.fsencode(source), width, height,
                                        C.byref(handle)):
            raise RuntimeError(lib.nvdec_bridge_last_error())
        try:
            frame = Frame()
            frame.struct_size = C.sizeof(Frame)
            if lib.nvdec_bridge_next(handle, C.byref(frame)) != 1:
                raise RuntimeError(lib.nvdec_bridge_last_error())
            assert (frame.width, frame.height, frame.software_format) == (width, height, 1)
            assert cuda.cuCtxPushCurrent_v2(frame.cuda_context) == 0
            try:
                planes = []
                for pointer, pitch, rows in ((frame.y_device_ptr, frame.y_pitch_bytes, height),
                                             (frame.uv_device_ptr, frame.uv_pitch_bytes, height // 2)):
                    host = (C.c_ubyte * (width * rows))()
                    copy = Copy2D()
                    copy.srcMemoryType = 2  # CU_MEMORYTYPE_DEVICE
                    copy.srcDevice = pointer
                    copy.srcPitch = pitch
                    copy.dstMemoryType = 1  # CU_MEMORYTYPE_HOST
                    copy.dstHost = C.cast(host, C.c_void_p)
                    copy.dstPitch = width
                    copy.WidthInBytes = width
                    copy.Height = rows
                    assert cuda.cuMemcpy2D_v2(C.byref(copy)) == 0
                    planes.append(bytes(host))
            finally:
                previous = C.c_void_p()
                assert cuda.cuCtxPopCurrent_v2(C.byref(previous)) == 0
            assert b"".join(planes) == reference, "scaled NV12 bytes differ from FFmpeg scale_cuda"
            # The scaler's own device write is counted; the host copy the Python
            # code above made went around the bridge and must not appear in it.
            counters = read_counters(lib, handle)
            assert counters["frames_decoded"] == 1, counters
            assert counters["device_frame_count"] == 1, counters
            assert counters["scale_device_output_bytes"] == width * height * 3 // 2, counters
            assert counters["host_frame_copy_count"] == 0, counters
            assert counters["host_frame_copy_bytes"] == 0, counters
        finally:
            lib.nvdec_bridge_close(handle)
    finally:
        cookie.close()
    print("nvdec_scale_cuda=1920x1080 first_frame_nv12=BYTE_IDENTICAL "
          f"scale_device_output_bytes={counters['scale_device_output_bytes']} "
          "host_copy_bytes=0")


if __name__ == "__main__":
    main()
