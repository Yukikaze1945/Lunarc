"""Two-frame NVDEC -> CUDA RGBA -> NVOF device-input parity gate."""
import ctypes as C
import os
import subprocess

import cv2
import numpy as np

from nvdec_bridge_scale_smoke import Copy2D
from nvdec_bridge_smoke import FFMPEG_BIN, Frame, ROOT


class Counters(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "execute_count", "h2d_copy_count", "h2d_bytes", "d2h_copy_count",
        "d2h_bytes", "reset_count", "bgr_staging_alloc_count",
        "bgr_input_call_count", "input_convert_launch_count")]


def main():
    width, height = 1920, 1080
    source = ROOT / "bench_out" / "fixture_4k_300.mp4"
    command = ["ffmpeg", "-v", "error", "-hwaccel", "cuda",
               "-hwaccel_output_format", "cuda", "-i", str(source),
               "-vf", f"scale_cuda=w={width}:h={height}:format=nv12,hwdownload,format=nv12",
               "-frames:v", "2", "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]
    raw = subprocess.check_output(command, timeout=30)
    frame_bytes = width * height * 3 // 2
    assert len(raw) == 2 * frame_bytes
    references = [cv2.cvtColor(
        np.frombuffer(raw[i * frame_bytes:(i + 1) * frame_bytes], np.uint8)
        .reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
        for i in range(2)]

    cookie = os.add_dll_directory(str(FFMPEG_BIN))
    try:
        decoder = C.CDLL(str(ROOT / "native" / "nvdec_bridge.dll"))
        converter = C.CDLL(str(ROOT / "native" / "nv12_rgba_bridge.dll"))
        nvof = C.CDLL(str(ROOT / "native" / "nvof_bridge.dll"))
        cuda = C.WinDLL("nvcuda.dll")
        decoder.nvdec_bridge_open_scaled.argtypes = [C.c_char_p, C.c_uint32,
            C.c_uint32, C.POINTER(C.c_void_p)]
        decoder.nvdec_bridge_open_scaled.restype = C.c_int32
        decoder.nvdec_bridge_next.argtypes = [C.c_void_p, C.POINTER(Frame)]
        decoder.nvdec_bridge_next.restype = C.c_int32
        decoder.nvdec_bridge_close.argtypes = [C.c_void_p]
        decoder.nvdec_bridge_last_error.restype = C.c_char_p
        converter.nv12_rgba_open.argtypes = [C.c_void_p, C.c_uint32,
            C.c_uint32, C.POINTER(C.c_void_p)]
        converter.nv12_rgba_open.restype = C.c_int32
        converter.nv12_rgba_convert.argtypes = [C.c_void_p, C.c_uint64,
            C.c_uint32, C.c_uint64, C.c_uint32, C.POINTER(C.c_uint64),
            C.POINTER(C.c_uint32)]
        converter.nv12_rgba_convert.restype = C.c_int32
        converter.nv12_rgba_close.argtypes = [C.c_void_p]
        converter.nv12_rgba_last_error.restype = C.c_char_p
        nvof.nvof_bridge_create.argtypes = [C.c_uint32, C.c_uint32,
            C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p)]
        nvof.nvof_bridge_create.restype = C.c_int32
        nvof.nvof_bridge_execute_rgba_device_sequential.argtypes = [C.c_void_p,
            C.c_uint64, C.c_uint32, C.c_void_p, C.c_int32, C.c_void_p]
        nvof.nvof_bridge_execute_rgba_device_sequential.restype = C.c_int32
        nvof.nvof_bridge_execute_bgr_sequential.argtypes = [C.c_void_p,
            C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint32, C.c_void_p,
            C.c_uint32, C.c_int32, C.c_void_p]
        nvof.nvof_bridge_execute_bgr_sequential.restype = C.c_int32
        nvof.nvof_bridge_get_output_deviceptr.argtypes = [C.c_void_p,
            C.POINTER(C.c_uint64), C.POINTER(C.c_size_t)]
        nvof.nvof_bridge_get_output_deviceptr.restype = C.c_int32
        nvof.nvof_bridge_get_session_info.argtypes = [C.c_void_p,
            C.POINTER(C.c_uint32), C.POINTER(C.c_void_p), C.POINTER(C.c_uint32)]
        nvof.nvof_bridge_get_session_info.restype = C.c_int32
        nvof.nvof_bridge_get_counters.argtypes = [C.c_void_p, C.POINTER(Counters)]
        nvof.nvof_bridge_get_counters.restype = C.c_int32
        nvof.nvof_bridge_get_device_input_counters.argtypes = [C.c_void_p,
            C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
        nvof.nvof_bridge_get_device_input_counters.restype = C.c_int32
        nvof.nvof_bridge_destroy.argtypes = [C.c_void_p]
        nvof.nvof_bridge_get_last_error.restype = C.c_char_p
        cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        cuda.cuCtxPushCurrent_v2.restype = C.c_int
        cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        cuda.cuCtxPopCurrent_v2.restype = C.c_int
        cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(Copy2D)]
        cuda.cuMemcpy2D_v2.restype = C.c_int

        decode = C.c_void_p()
        assert decoder.nvdec_bridge_open_scaled(os.fsencode(source), width,
            height, C.byref(decode)) == 0, decoder.nvdec_bridge_last_error()
        convert = C.c_void_p()
        device_flow = C.c_void_p()
        host_flow = C.c_void_p()
        try:
            for session in (device_flow, host_flow):
                assert nvof.nvof_bridge_create(width, height, 4, 10,
                    C.byref(session)) == 0, nvof.nvof_bridge_get_last_error()
            for i, bgr in enumerate(references):
                frame = Frame()
                frame.struct_size = C.sizeof(Frame)
                assert decoder.nvdec_bridge_next(decode, C.byref(frame)) == 1
                if i == 0:
                    assert converter.nv12_rgba_open(frame.cuda_context, width,
                        height, C.byref(convert)) == 0, converter.nv12_rgba_last_error()
                rgba = C.c_uint64()
                pitch = C.c_uint32()
                assert converter.nv12_rgba_convert(convert,
                    frame.y_device_ptr, frame.y_pitch_bytes,
                    frame.uv_device_ptr, frame.uv_pitch_bytes,
                    C.byref(rgba), C.byref(pitch)) == 0, converter.nv12_rgba_last_error()
                assert nvof.nvof_bridge_execute_rgba_device_sequential(
                    device_flow, rgba.value, pitch.value, frame.cuda_context,
                    int(i == 0), None) == 0, nvof.nvof_bridge_get_last_error()
                expected = np.empty((height, width, 2), np.float32)
                assert nvof.nvof_bridge_execute_bgr_sequential(host_flow,
                    bgr.ctypes.data, bgr.strides[0], None, 0,
                    expected.ctypes.data, expected.strides[0], int(i == 0),
                    None) == 0, nvof.nvof_bridge_get_last_error()

                pointer = C.c_uint64()
                flow_pitch = C.c_size_t()
                context = C.c_void_p()
                device_idx = C.c_uint32()
                owner = C.c_uint32()
                assert nvof.nvof_bridge_get_output_deviceptr(device_flow,
                    C.byref(pointer), C.byref(flow_pitch)) == 0
                assert nvof.nvof_bridge_get_session_info(device_flow,
                    C.byref(device_idx), C.byref(context), C.byref(owner)) == 0
                actual = np.empty_like(expected)
                copy = Copy2D()
                copy.srcMemoryType = 2
                copy.srcDevice = pointer.value
                copy.srcPitch = flow_pitch.value
                copy.dstMemoryType = 1
                copy.dstHost = actual.ctypes.data
                copy.dstPitch = actual.strides[0]
                copy.WidthInBytes = width * 8
                copy.Height = height
                assert cuda.cuCtxPushCurrent_v2(context) == 0
                try:
                    assert cuda.cuMemcpy2D_v2(C.byref(copy)) == 0
                finally:
                    previous = C.c_void_p()
                    assert cuda.cuCtxPopCurrent_v2(C.byref(previous)) == 0
                assert np.array_equal(actual, expected), f"flow differs on frame {i}"
            counters = Counters()
            copies = C.c_uint64()
            bytes_copied = C.c_uint64()
            assert nvof.nvof_bridge_get_counters(device_flow, C.byref(counters)) == 0
            assert nvof.nvof_bridge_get_device_input_counters(device_flow,
                C.byref(copies), C.byref(bytes_copied)) == 0
            assert counters.execute_count == 2
            assert (counters.h2d_copy_count, counters.h2d_bytes,
                    counters.d2h_copy_count, counters.d2h_bytes) == (0, 0, 0, 0)
            assert copies.value == 3 and bytes_copied.value == 3 * width * height * 4
        finally:
            if convert.value:
                converter.nv12_rgba_close(convert)
            if device_flow.value:
                nvof.nvof_bridge_destroy(device_flow)
            if host_flow.value:
                nvof.nvof_bridge_destroy(host_flow)
            decoder.nvdec_bridge_close(decode)
    finally:
        cookie.close()
    print("nvdec_rgba_nvof_frames=2 flow=BYTE_IDENTICAL h2d=0 d2h=0")


if __name__ == "__main__":
    main()
