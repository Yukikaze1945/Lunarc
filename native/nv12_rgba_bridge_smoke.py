"""One-frame test-only readback against OpenCV's NV12 to RGBA conversion."""
import ctypes as C
import os
import subprocess

import cv2
import numpy as np

from nvdec_bridge_scale_smoke import Copy2D
from nvdec_bridge_smoke import FFMPEG_BIN, Frame, ROOT


class Nv12Counters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("convert_count", C.c_uint64),
                ("output_device_write_bytes", C.c_uint64)]


def read_counters(converter, handle):
    counters = Nv12Counters()
    counters.struct_size = C.sizeof(Nv12Counters)
    assert converter.nv12_rgba_get_transfer_counters(handle, C.byref(counters)) == 0
    return counters


def main():
    source = ROOT / "bench_out" / "fixture_4k_300.mp4"
    width, height = 1920, 1080
    command = ["ffmpeg", "-v", "error", "-hwaccel", "cuda",
               "-hwaccel_output_format", "cuda", "-i", str(source),
               "-vf", f"scale_cuda=w={width}:h={height}:format=nv12,hwdownload,format=nv12",
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]
    nv12 = subprocess.check_output(command, timeout=30)
    reference = cv2.cvtColor(
        np.frombuffer(nv12, dtype=np.uint8).reshape(height * 3 // 2, width),
        cv2.COLOR_YUV2RGBA_NV12)
    # Independent floating-point BT.709 limited-range reference. NV12's UV
    # samples cover 2x2 luma pixels; round once at the final RGB pixel.
    planes = np.frombuffer(nv12, dtype=np.uint8).reshape(height * 3 // 2, width)
    y = np.maximum(planes[:height].astype(np.float32) - 16, 0) * (255 / 219)
    uv = planes[height:].reshape(height // 2, width // 2, 2)
    u = np.repeat(np.repeat(uv[:, :, 0], 2, 0), 2, 1).astype(np.float32) - 128
    v = np.repeat(np.repeat(uv[:, :, 1], 2, 0), 2, 1).astype(np.float32) - 128
    reference_709 = np.empty((height, width, 4), dtype=np.uint8)
    reference_709[:, :, 0] = np.clip(np.floor(y + v * (255 * 1.5748 / 224) + 0.5), 0, 255)
    reference_709[:, :, 1] = np.clip(np.floor(y - u * (255 * 0.187324 / 224) - v * (255 * 0.468124 / 224) + 0.5), 0, 255)
    reference_709[:, :, 2] = np.clip(np.floor(y + u * (255 * 1.8556 / 224) + 0.5), 0, 255)
    reference_709[:, :, 3] = 255

    cookie = os.add_dll_directory(str(FFMPEG_BIN))
    try:
        decoder = C.CDLL(str(ROOT / "native" / "nvdec_bridge.dll"))
        converter = C.CDLL(str(ROOT / "native" / "nv12_rgba_bridge.dll"))
        cuda = C.WinDLL("nvcuda.dll")
        decoder.nvdec_bridge_open_scaled.argtypes = [C.c_char_p, C.c_uint32,
            C.c_uint32, C.POINTER(C.c_void_p)]
        decoder.nvdec_bridge_open_scaled.restype = C.c_int32
        decoder.nvdec_bridge_next.argtypes = [C.c_void_p, C.POINTER(Frame)]
        decoder.nvdec_bridge_next.restype = C.c_int32
        decoder.nvdec_bridge_close.argtypes = [C.c_void_p]
        decoder.nvdec_bridge_last_error.restype = C.c_char_p
        converter.nv12_rgba_abi_version.restype = C.c_uint32
        assert converter.nv12_rgba_abi_version() == 3
        converter.nv12_rgba_open.argtypes = [C.c_void_p, C.c_uint32,
            C.c_uint32, C.POINTER(C.c_void_p)]
        converter.nv12_rgba_open.restype = C.c_int32
        converter.nv12_rgba_open_matrix.argtypes = [C.c_void_p, C.c_uint32,
            C.c_uint32, C.c_int32, C.POINTER(C.c_void_p)]
        converter.nv12_rgba_open_matrix.restype = C.c_int32
        converter.nv12_rgba_convert.argtypes = [C.c_void_p, C.c_uint64,
            C.c_uint32, C.c_uint64, C.c_uint32, C.POINTER(C.c_uint64),
            C.POINTER(C.c_uint32)]
        converter.nv12_rgba_convert.restype = C.c_int32
        converter.nv12_rgba_get_transfer_counters.argtypes = [C.c_void_p,
            C.POINTER(Nv12Counters)]
        converter.nv12_rgba_get_transfer_counters.restype = C.c_int32
        converter.nv12_rgba_close.argtypes = [C.c_void_p]
        converter.nv12_rgba_last_error.restype = C.c_char_p
        cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        cuda.cuCtxPushCurrent_v2.restype = C.c_int
        cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        cuda.cuCtxPopCurrent_v2.restype = C.c_int
        cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(Copy2D)]
        cuda.cuMemcpy2D_v2.restype = C.c_int

        decode_handle = C.c_void_p()
        assert decoder.nvdec_bridge_open_scaled(os.fsencode(source), width,
            height, C.byref(decode_handle)) == 0, decoder.nvdec_bridge_last_error()
        try:
            frame = Frame()
            frame.struct_size = C.sizeof(Frame)
            assert decoder.nvdec_bridge_next(decode_handle, C.byref(frame)) == 1
            for matrix, expected in ((1, reference), (2, reference_709)):
                convert_handle = C.c_void_p()
                if matrix == 1:
                    opened = converter.nv12_rgba_open(frame.cuda_context, width,
                        height, C.byref(convert_handle))
                else:
                    opened = converter.nv12_rgba_open_matrix(frame.cuda_context,
                        width, height, matrix, C.byref(convert_handle))
                assert opened == 0, converter.nv12_rgba_last_error()
                try:
                    # A bridge that has converted nothing counts zero, and that
                    # zero is the measurement it claims to be.
                    fresh = read_counters(converter, convert_handle)
                    assert (fresh.convert_count, fresh.output_device_write_bytes) \
                        == (0, 0), (matrix, fresh.convert_count,
                                    fresh.output_device_write_bytes)
                    rgba_ptr = C.c_uint64()
                    rgba_pitch = C.c_uint32()
                    assert converter.nv12_rgba_convert(convert_handle,
                        frame.y_device_ptr, frame.y_pitch_bytes,
                        frame.uv_device_ptr, frame.uv_pitch_bytes,
                        C.byref(rgba_ptr), C.byref(rgba_pitch)) == 0, \
                        converter.nv12_rgba_last_error()
                    assert rgba_pitch.value == width * 4
                    counted = read_counters(converter, convert_handle)
                    assert counted.convert_count == 1, (matrix, counted.convert_count)
                    assert counted.output_device_write_bytes == width * height * 4, \
                        (matrix, counted.output_device_write_bytes)
                    # A refused call is not a conversion and must not be counted.
                    assert converter.nv12_rgba_convert(convert_handle,
                        frame.y_device_ptr, width - 2, frame.uv_device_ptr,
                        frame.uv_pitch_bytes, C.byref(rgba_ptr),
                        C.byref(rgba_pitch)) != 0
                    refused = read_counters(converter, convert_handle)
                    assert refused.convert_count == counted.convert_count, matrix
                    assert refused.output_device_write_bytes == \
                        counted.output_device_write_bytes, matrix
                    host = (C.c_ubyte * (width * height * 4))()
                    copy = Copy2D()
                    copy.srcMemoryType = 2
                    copy.srcDevice = rgba_ptr.value
                    copy.srcPitch = rgba_pitch.value
                    copy.dstMemoryType = 1
                    copy.dstHost = C.cast(host, C.c_void_p)
                    copy.dstPitch = width * 4
                    copy.WidthInBytes = width * 4
                    copy.Height = height
                    assert cuda.cuCtxPushCurrent_v2(frame.cuda_context) == 0
                    try:
                        assert cuda.cuMemcpy2D_v2(C.byref(copy)) == 0
                    finally:
                        previous = C.c_void_p()
                        assert cuda.cuCtxPopCurrent_v2(C.byref(previous)) == 0
                    actual = np.frombuffer(host, dtype=np.uint8).reshape(height, width, 4)
                    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
                    limit = 0 if matrix == 1 else 2
                    assert diff.max() <= limit, (f"matrix={matrix} RGBA mismatch: "
                        f"max={diff.max()} pixels={np.count_nonzero(diff)}")
                finally:
                    converter.nv12_rgba_close(convert_handle)
        finally:
            decoder.nvdec_bridge_close(decode_handle)
    finally:
        cookie.close()
    print("nv12_rgba_gpu=1920x1080 bt601_opencv=BYTE_IDENTICAL "
          "bt709_float_reference=max_delta_2 abi=3 "
          "counters=convert_count_1_output_write_bytes_8294400_per_call")


if __name__ == "__main__":
    main()
