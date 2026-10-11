"""Two-frame 4K plan-driven Lanczos parity gate at a non-half working scale.

The exact-2x stage is covered by gpu_lanczos_smoke.py. This one covers the other
shape: a scale whose taps no constant table describes, so the plan the bridge is
opened with comes from the same pipeline.AxisPlans the CPU path builds. A
byte-identical result then means the device filter and the reference filter are
the same filter, and a difference of one means they are not.

    python native/gpu_lanczos_scaled_smoke.py            # 0.75, the first served
    python native/gpu_lanczos_scaled_smoke.py 0.26       # any plan-driven percentage

The scale is an argument because the proof is per-extents: each percentage
resolves to a different 8-aligned working size, and each builds a different table.
What is not per-ratio is the filter, which is why this file holds one comparison
rather than one per served scale.

Every pixel comparison is against residual_native.NativeResidualBackend, which
is the working-color implementation the production CPU export actually calls.
The device-to-host copy below is this smoke's own readback; the chain it is
gating performs none.
"""
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nvdec_bridge_scale_smoke import Copy2D
from nvdec_bridge_smoke import FFMPEG_BIN, Frame, ROOT

import gpu_transfer_export as gtx
import pipeline
from residual_native import NativeResidualBackend

FULL_WIDTH, FULL_HEIGHT = 3840, 2160
WORKING_SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75


def _check_requested_scale():
    """Refuse to measure nothing: a scale this chain does not serve has no plan.

    Resolved through the same function the gate uses, so the smoke cannot be run
    against a size the production chain would refuse to open.
    """
    if gtx.plan_driven_scale(FULL_WIDTH, FULL_HEIGHT, *pipeline.resolve_working_dims(
            FULL_WIDTH, FULL_HEIGHT, WORKING_SCALE), WORKING_SCALE) is None:
        raise SystemExit(
            f"working_scale={WORKING_SCALE} is not a percentage the plan-driven "
            f"stages serve (a whole percentage from {gtx.WORKING_PERCENT_MIN} to "
            f"{gtx.WORKING_PERCENT_MAX}, resolved to a working size below the "
            "source on both axes); this gate exists to prove those, and an "
            "exact-2x or identity scale would compare a plan against itself")


class LanczosCounters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("process_count", C.c_uint64),
        ("source_device_copy_bytes", C.c_uint64),
        ("scratch_device_write_bytes", C.c_uint64),
        ("output_device_write_bytes", C.c_uint64)]


def read_lanczos_counters(lanczos, handle):
    counters = LanczosCounters()
    counters.struct_size = C.sizeof(LanczosCounters)
    assert lanczos.gpu_lanczos_get_transfer_counters(handle,
        C.byref(counters)) == 0
    return counters


def main():
    _check_requested_scale()
    width, height = FULL_WIDTH, FULL_HEIGHT
    low_width, low_height = pipeline.resolve_working_dims(width, height,
                                                          WORKING_SCALE)
    print(f"working_dims={low_width}x{low_height} at scale={WORKING_SCALE}",
          flush=True)
    source = ROOT / "bench_out" / "fixture_4k_300.mp4"
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-hwaccel",
        "cuda", "-hwaccel_output_format", "cuda", "-i", str(source), "-vf",
        "hwdownload,format=nv12", "-frames:v", "2", "-f", "rawvideo",
        "-pix_fmt", "nv12", "pipe:1"], timeout=30)
    frame_bytes = width * height * 3 // 2
    assert len(raw) == 2 * frame_bytes

    # The view's fields are raw pointers into the arrays keepalive holds. The
    # bridge reads them while it uploads the plan at open, so what matters is
    # that neither is dropped before that, and both are locals.
    plan, keepalive = gtx.working_color_plan(width, height, low_width,
                                             low_height)
    print(f"plan_taps={plan.taps} rows={plan.vertical_dst_len}x"
          f"{plan.horizontal_dst_len}", flush=True)

    cookie = os.add_dll_directory(str(FFMPEG_BIN))
    try:
        decoder = C.CDLL(str(ROOT / "native" / "nvdec_bridge.dll"))
        converter = C.CDLL(str(ROOT / "native" / "nv12_rgba_bridge.dll"))
        lanczos = C.CDLL(str(ROOT / "native" / "gpu_lanczos_bridge.dll"))
        cuda = C.WinDLL("nvcuda.dll")
        decoder.nvdec_bridge_open.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        decoder.nvdec_bridge_open.restype = C.c_int32
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
        lanczos.gpu_lanczos_abi_version.restype = C.c_uint32
        assert lanczos.gpu_lanczos_abi_version() == gtx.LANCZOS_ABI
        assert any(getattr(lanczos, name, None) is not None
                   for name in ("gpu_lanczos_open_scaled",)), (
            "this gate is the scaled contract; an exact-2x-only build cannot pass it")
        lanczos.gpu_lanczos_open_scaled.argtypes = [C.c_void_p, C.c_uint32,
            C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(gtx._TapTables),
            C.POINTER(C.c_void_p)]
        lanczos.gpu_lanczos_open_scaled.restype = C.c_int32
        lanczos.gpu_lanczos_process.argtypes = [C.c_void_p, C.c_uint64,
            C.c_size_t, C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)]
        lanczos.gpu_lanczos_process.restype = C.c_int32
        lanczos.gpu_lanczos_get_transfer_counters.argtypes = [C.c_void_p,
            C.POINTER(LanczosCounters)]
        lanczos.gpu_lanczos_get_transfer_counters.restype = C.c_int32
        lanczos.gpu_lanczos_close.argtypes = [C.c_void_p]
        lanczos.gpu_lanczos_last_error.restype = C.c_char_p
        cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        cuda.cuCtxPushCurrent_v2.restype = C.c_int
        cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        cuda.cuCtxPopCurrent_v2.restype = C.c_int
        cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(Copy2D)]
        cuda.cuMemcpy2D_v2.restype = C.c_int

        decode = C.c_void_p()
        convert = C.c_void_p()
        scale = C.c_void_p()
        assert decoder.nvdec_bridge_open(os.fsencode(source),
            C.byref(decode)) == 0, decoder.nvdec_bridge_last_error()
        cpu = NativeResidualBackend(width, height, low_width, low_height)
        try:
            for index in range(2):
                frame = Frame()
                frame.struct_size = C.sizeof(Frame)
                assert decoder.nvdec_bridge_next(decode, C.byref(frame)) == 1
                assert (frame.width, frame.height) == (width, height)
                if index == 0:
                    assert converter.nv12_rgba_open(frame.cuda_context,
                        width, height, C.byref(convert)) == 0, \
                        converter.nv12_rgba_last_error()
                    # The plan is sized for the extents the decoder reports, and
                    # a bridge opened for a different pair refuses to run rather
                    # than resampling against the wrong table.
                    assert plan.vertical_src_len == height
                    assert plan.horizontal_src_len == width
                    assert lanczos.gpu_lanczos_open_scaled(frame.cuda_context,
                        width, height, low_width, low_height,
                        C.byref(plan), C.byref(scale)) == 0, \
                        lanczos.gpu_lanczos_last_error()
                    # Nothing has been processed yet, so every count is a
                    # measured zero rather than a default.
                    fresh = read_lanczos_counters(lanczos, scale)
                    assert (fresh.process_count, fresh.source_device_copy_bytes,
                            fresh.scratch_device_write_bytes,
                            fresh.output_device_write_bytes) == (0, 0, 0, 0)
                rgba_ptr = C.c_uint64()
                rgba_pitch = C.c_uint32()
                assert converter.nv12_rgba_convert(convert,
                    frame.y_device_ptr, frame.y_pitch_bytes,
                    frame.uv_device_ptr, frame.uv_pitch_bytes,
                    C.byref(rgba_ptr), C.byref(rgba_pitch)) == 0, \
                    converter.nv12_rgba_last_error()
                low_ptr = C.c_uint64()
                low_pitch = C.c_uint32()
                assert lanczos.gpu_lanczos_process(scale,
                    rgba_ptr.value, rgba_pitch.value,
                    C.byref(low_ptr), C.byref(low_pitch)) == 0, \
                    lanczos.gpu_lanczos_last_error()
                assert (low_ptr.value, low_pitch.value) != (0, 0)
                host = np.zeros((low_height, low_width, 4), np.uint8)
                copy = Copy2D()
                copy.srcMemoryType = 2
                copy.srcDevice = low_ptr.value
                copy.srcPitch = low_pitch.value
                copy.dstMemoryType = 1
                copy.dstHost = host.ctypes.data
                copy.dstPitch = host.strides[0]
                copy.WidthInBytes = low_width * 4
                copy.Height = low_height
                assert cuda.cuCtxPushCurrent_v2(frame.cuda_context) == 0
                try:
                    assert cuda.cuMemcpy2D_v2(C.byref(copy)) == 0
                finally:
                    previous = C.c_void_p()
                    assert cuda.cuCtxPopCurrent_v2(C.byref(previous)) == 0
                nv12 = np.frombuffer(raw[index * frame_bytes:
                    (index + 1) * frame_bytes], np.uint8).reshape(
                        height * 3 // 2, width)
                full_bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
                expected_bgr = np.zeros((low_height, low_width, 3), np.uint8)
                expected_rgba = np.zeros_like(host)
                cpu.prepare_working_color_rgba(full_bgr, expected_bgr,
                    expected_rgba)
                difference = np.abs(host.astype(np.int16) -
                    expected_rgba.astype(np.int16))
                print(f"frame={index} max_difference={difference.max()} "
                      f"differing_channels={np.count_nonzero(difference)}",
                      flush=True)
                assert np.array_equal(host, expected_rgba), (
                    f"frame {index}: {int(np.count_nonzero(difference))} "
                    f"channels differ by at most {int(difference.max())}, so the "
                    "device filter is not the CPU filter")
                assert np.any(expected_bgr), (
                    "the reference wrote no working BGR, so the comparison above "
                    "cannot mean anything")

            # Two process calls: one full-frame device copy into the shared
            # source buffer, one scratch write of full_w*low_h float4 elements
            # and one RGBA8 working-frame write per call. Every figure is the
            # payload these extents require, not a buffer capacity.
            counted = read_lanczos_counters(lanczos, scale)
            assert counted.process_count == 2, counted.process_count
            assert counted.source_device_copy_bytes == \
                2 * width * height * 4, counted.source_device_copy_bytes
            assert counted.scratch_device_write_bytes == \
                2 * width * low_height * 16, counted.scratch_device_write_bytes
            assert counted.output_device_write_bytes == \
                2 * low_width * low_height * 4, counted.output_device_write_bytes
        finally:
            cpu.close()
            if scale.value:
                lanczos.gpu_lanczos_close(scale)
            if convert.value:
                converter.nv12_rgba_close(convert)
            decoder.nvdec_bridge_close(decode)
    finally:
        cookie.close()
    print(f"gpu_lanczos_scaled_4k_to_{low_width}x{low_height}=BYTE_IDENTICAL "
          f"frames=2 scale={WORKING_SCALE} taps={plan.taps} abi={gtx.LANCZOS_ABI}")


if __name__ == "__main__":
    main()
