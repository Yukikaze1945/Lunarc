"""Two-frame 4K plan-driven residual upscale and composite parity gate.

Phase one (gpu_lanczos_scaled_smoke.py) proves the working colour at a
plan-driven scale is the CPU reference's working colour. This proves the other
half: that a DLSS output at that working size is upsampled and composited back
onto the 3840x2160 full frame exactly as residual_native does it, with no host
frame copy anywhere between the decoder and the composited output.

    python native/gpu_residual_scaled_smoke.py           # 0.75, the first served
    python native/gpu_residual_scaled_smoke.py 0.26      # any plan-driven percentage

Both tap tables come from the same pipeline.AxisPlans the CPU path builds, so a
difference of one here is a different filter, not a different rounding of the
same one. The two readbacks below are this smoke's own correctness copies; the
chain they gate performs none.
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
from dlss_gpu_host_smoke import DeviceOutput, Options, TransferCounters, load_gpu_host

import gpu_transfer_export as gtx
import pipeline
from residual_native import NativeResidualBackend

FULL_WIDTH, FULL_HEIGHT = 3840, 2160
WORKING_SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75


def _check_requested_scale():
    """Refuse to measure nothing: a scale this chain does not serve has no plan.

    The same guard gpu_lanczos_scaled_smoke.py applies, asked of the same
    function the production gate applies it in, so a run cannot be aimed at a
    size the chain would never open.
    """
    live = pipeline.resolve_working_dims(FULL_WIDTH, FULL_HEIGHT, WORKING_SCALE)
    if gtx.plan_driven_scale(FULL_WIDTH, FULL_HEIGHT, live[0], live[1],
                             WORKING_SCALE) is None:
        raise SystemExit(
            f"working_scale={WORKING_SCALE} is not a percentage the plan-driven "
            f"stages serve (a whole percentage from {gtx.WORKING_PERCENT_MIN} to "
            f"{gtx.WORKING_PERCENT_MAX}, resolved to a working size below the "
            "source on both axes); this gate exists to prove those, and an "
            "exact-2x or identity scale would compare a plan against itself")



class NvofCounters(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "execute_count", "h2d_copy_count", "h2d_bytes",
        "d2h_copy_count", "d2h_bytes", "reset_count",
        "bgr_staging_alloc_count", "bgr_input_call_count",
        "input_convert_launch_count")]


class ResidualCounters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32),
        ("full_device_copy_bytes", C.c_uint64),
        ("working_device_copy_bytes", C.c_uint64),
        ("output_readback_bytes", C.c_uint64)]


class LanczosCounters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("process_count", C.c_uint64),
        ("source_device_copy_bytes", C.c_uint64),
        ("scratch_device_write_bytes", C.c_uint64),
        ("output_device_write_bytes", C.c_uint64)]


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

    # The views hold raw pointers into the arrays the second element owns; both
    # stay locals for as long as the bridges can still read them.
    color_plan, color_buffers = gtx.working_color_plan(width, height, low_width,
                                                       low_height)
    upscale_plan, upscale_buffers = gtx.residual_plan(width, height, low_width,
                                                      low_height)
    print(f"lanczos_taps={color_plan.taps} residual_taps={upscale_plan.taps}",
          flush=True)

    cookie = os.add_dll_directory(str(FFMPEG_BIN))
    try:
        decoder = C.CDLL(str(ROOT / "native" / "nvdec_bridge.dll"))
        converter = C.CDLL(str(ROOT / "native" / "nv12_rgba_bridge.dll"))
        lanczos = C.CDLL(str(ROOT / "native" / "gpu_lanczos_bridge.dll"))
        nvof = C.CDLL(str(ROOT / "native" / "nvof_bridge.dll"))
        residual = C.CDLL(str(ROOT / "native" / "gpu_residual_bridge.dll"))
        host_lib = load_gpu_host()
        cuda = C.WinDLL("nvcuda.dll")
        nvof.nvof_bridge_create.argtypes = [C.c_uint32, C.c_uint32,
            C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p)]
        nvof.nvof_bridge_create.restype = C.c_int32
        nvof.nvof_bridge_execute_rgba_device_sequential.argtypes = [C.c_void_p,
            C.c_uint64, C.c_uint32, C.c_void_p, C.c_int32, C.c_void_p]
        nvof.nvof_bridge_execute_rgba_device_sequential.restype = C.c_int32
        nvof.nvof_bridge_get_output_deviceptr.argtypes = [C.c_void_p,
            C.POINTER(C.c_uint64), C.POINTER(C.c_size_t)]
        nvof.nvof_bridge_get_output_deviceptr.restype = C.c_int32
        nvof.nvof_bridge_get_session_info.argtypes = [C.c_void_p,
            C.POINTER(C.c_uint32), C.POINTER(C.c_void_p), C.POINTER(C.c_uint32)]
        nvof.nvof_bridge_get_session_info.restype = C.c_int32
        nvof.nvof_bridge_destroy.argtypes = [C.c_void_p]
        nvof.nvof_bridge_get_last_error.restype = C.c_char_p
        nvof.nvof_bridge_get_counters.argtypes = [C.c_void_p,
            C.POINTER(NvofCounters)]
        nvof.nvof_bridge_get_counters.restype = C.c_int32
        residual.gpu_residual_abi_version.restype = C.c_uint32
        residual.gpu_residual_process.argtypes = [C.c_void_p, C.c_void_p,
            C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64, C.c_size_t,
            C.c_void_p, C.c_uint64, C.c_size_t, C.c_float,
            C.POINTER(C.c_void_p)]
        residual.gpu_residual_process.restype = C.c_int32
        residual.gpu_residual_readback.argtypes = [C.c_void_p,
            C.c_void_p, C.c_size_t]
        residual.gpu_residual_readback.restype = C.c_int32
        residual.gpu_residual_close.argtypes = [C.c_void_p]
        residual.gpu_residual_last_error.restype = C.c_char_p
        residual.gpu_residual_get_transfer_counters.argtypes = [C.c_void_p,
            C.POINTER(ResidualCounters)]
        residual.gpu_residual_get_transfer_counters.restype = C.c_int32
        decoder.nvdec_bridge_open.argtypes = [C.c_char_p,
            C.POINTER(C.c_void_p)]
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
        assert lanczos.gpu_lanczos_abi_version() == gtx.LANCZOS_ABI
        assert residual.gpu_residual_abi_version() == gtx.RESIDUAL_ABI, (
            "the scaled export is an added entry point, not a new ABI")
        # Reached through getattr so that a build without the scaled contract
        # fails here, naming the export, rather than at the first frame.
        assert getattr(residual, "gpu_residual_open_scaled", None) is not None
        residual.gpu_residual_open_scaled.argtypes = [C.c_void_p, C.c_uint32,
            C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(gtx._TapTables),
            C.POINTER(C.c_void_p)]
        residual.gpu_residual_open_scaled.restype = C.c_int32
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
        gpu_residual = C.c_void_p()
        flow_session = C.c_void_p()
        assert decoder.nvdec_bridge_open(os.fsencode(source),
            C.byref(decode)) == 0, decoder.nvdec_bridge_last_error()
        cpu = NativeResidualBackend(width, height, low_width, low_height)
        try:
            assert nvof.nvof_bridge_create(low_width, low_height, 4, 10,
                C.byref(flow_session)) == 0, nvof.nvof_bridge_get_last_error()
            for index in range(2):
                frame = Frame()
                frame.struct_size = C.sizeof(Frame)
                assert decoder.nvdec_bridge_next(decode, C.byref(frame)) == 1
                assert (frame.width, frame.height) == (width, height)
                if index == 0:
                    assert converter.nv12_rgba_open(frame.cuda_context,
                        width, height, C.byref(convert)) == 0, \
                        converter.nv12_rgba_last_error()
                    assert lanczos.gpu_lanczos_open_scaled(frame.cuda_context,
                        width, height, low_width, low_height,
                        C.byref(color_plan), C.byref(scale)) == 0, \
                        lanczos.gpu_lanczos_last_error()
                    fresh = read_lanczos(lanczos, scale)
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
                host = np.zeros((low_height, low_width, 4), np.uint8)
                download(host, low_ptr.value, low_pitch.value, low_width,
                    low_height, cuda, frame.cuda_context)
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
                print(f"frame={index} working_max_difference={difference.max()}",
                      flush=True)
                assert np.array_equal(host, expected_rgba), (
                    f"frame {index}: the {WORKING_SCALE:g} working colour is not "
                    "the CPU reference, so the residual compared below would be "
                    "composited onto the wrong base")
                assert nvof.nvof_bridge_execute_rgba_device_sequential(
                    flow_session, low_ptr.value, low_pitch.value,
                    frame.cuda_context, int(index == 0), None) == 0, \
                    nvof.nvof_bridge_get_last_error()

            flow_ptr = C.c_uint64()
            flow_pitch = C.c_size_t()
            flow_context = C.c_void_p()
            device = C.c_uint32()
            owner = C.c_uint32()
            assert nvof.nvof_bridge_get_output_deviceptr(flow_session,
                C.byref(flow_ptr), C.byref(flow_pitch)) == 0
            assert nvof.nvof_bridge_get_session_info(flow_session,
                C.byref(device), C.byref(flow_context), C.byref(owner)) == 0
            nvof_counters = NvofCounters()
            assert nvof.nvof_bridge_get_counters(flow_session,
                C.byref(nvof_counters)) == 0
            # Flow runs on the working CUDA surface the Lanczos stage left in
            # device memory and writes its own device output: no host copy on
            # either side of it.
            assert (nvof_counters.h2d_copy_count, nvof_counters.h2d_bytes,
                    nvof_counters.d2h_copy_count,
                    nvof_counters.d2h_bytes) == (0, 0, 0, 0)

            options = Options(C.sizeof(Options), 0, 0, 1, 1, 1, 0, 0, 0, 1, 1)
            dlss_handle = C.c_void_p()
            assert host_lib.dlss_gpu_host_open(low_width, low_height,
                str(ROOT / "nvngx_dlssnr.dll"), C.byref(options),
                C.byref(dlss_handle)) == 0, host_lib.dlss_gpu_host_last_error()
            try:
                # The last decoded frame is the one the DLSS session is holding,
                # and the residual is what this gate is about, so the composite
                # is checked on it rather than replayed for every frame.
                output = DeviceOutput()
                output.struct_size = C.sizeof(DeviceOutput)
                assert host_lib.dlss_gpu_host_process_device_color_resource(
                    dlss_handle, frame.cuda_context, low_ptr.value,
                    low_pitch.value, flow_context, flow_ptr.value,
                    flow_pitch.value, 1, C.byref(output)) == 0, \
                    host_lib.dlss_gpu_host_last_error()
                counters = TransferCounters()
                counters.struct_size = C.sizeof(TransferCounters)
                assert host_lib.dlss_gpu_host_get_transfer_counters(
                    dlss_handle, C.byref(counters)) == 0
                assert counters.host_color_stage_bytes == 0
                assert counters.output_readback_bytes == 0

                assert residual.gpu_residual_open_scaled(output.d3d12_device,
                    width, height, low_width, low_height,
                    C.byref(upscale_plan), C.byref(gpu_residual)) == 0, \
                    residual.gpu_residual_last_error()
                before = read_residual(residual, gpu_residual)
                assert (before.full_device_copy_bytes,
                        before.working_device_copy_bytes,
                        before.output_readback_bytes) == (0, 0, 0)
                full_output = C.c_void_p()
                assert residual.gpu_residual_process(gpu_residual,
                    output.d3d12_resource, output.d3d12_fence,
                    output.fence_value, frame.cuda_context,
                    rgba_ptr.value, rgba_pitch.value,
                    frame.cuda_context, low_ptr.value, low_pitch.value,
                    C.c_float(1.0), C.byref(full_output)) == 0, \
                    residual.gpu_residual_last_error()
                assert full_output.value
                counted = read_residual(residual, gpu_residual)
                # One process call: the full RGBA and the working RGBA each move
                # once, at the extents this bridge was opened with, and nothing
                # has come back to the host yet.
                assert counted.full_device_copy_bytes == width * height * 4, \
                    counted.full_device_copy_bytes
                assert counted.working_device_copy_bytes == \
                    low_width * low_height * 4, counted.working_device_copy_bytes
                assert counted.output_readback_bytes == 0
                actual_rgba = np.zeros((height, width, 4), np.uint8)
                assert residual.gpu_residual_readback(gpu_residual,
                    actual_rgba.ctypes.data, actual_rgba.strides[0]) == 0, \
                    residual.gpu_residual_last_error()
                assert read_residual(residual,
                    gpu_residual).output_readback_bytes == width * height * 4
                dlss_rgba = np.zeros((low_height, low_width, 4), np.uint8)
                assert host_lib.dlss_gpu_host_readback_output(dlss_handle,
                    dlss_rgba.ctypes.data, dlss_rgba.strides[0]) == 0, \
                    host_lib.dlss_gpu_host_last_error()
                expected_full = np.zeros((height, width, 3), np.uint8)
                cpu.reconstruct_from_rgba(full_bgr, expected_bgr,
                    dlss_rgba, 1.0, expected_full)
                actual_bgr = cv2.cvtColor(actual_rgba, cv2.COLOR_RGBA2BGR)
                diff = np.abs(actual_bgr.astype(np.int16) -
                    expected_full.astype(np.int16))
                print(f"residual_4k_composite_max_difference={diff.max()} "
                      f"differing_channels={np.count_nonzero(diff)}", flush=True)
                assert np.array_equal(actual_bgr, expected_full), (
                    f"{int(np.count_nonzero(diff))} channels of the composited "
                    f"4K frame differ by at most {int(diff.max())}, so the "
                    "device upscale is not the CPU upscale")
            finally:
                host_lib.dlss_gpu_host_close(dlss_handle)
                if gpu_residual.value:
                    residual.gpu_residual_close(gpu_residual)
        finally:
            cpu.close()
            if flow_session.value:
                nvof.nvof_bridge_destroy(flow_session)
            if scale.value:
                lanczos.gpu_lanczos_close(scale)
            if convert.value:
                converter.nv12_rgba_close(convert)
            decoder.nvdec_bridge_close(decode)
    finally:
        cookie.close()
    print(f"gpu_residual_scaled_4k_composite=BYTE_IDENTICAL "
          f"working={low_width}x{low_height} scale={WORKING_SCALE} "
          f"abi={gtx.RESIDUAL_ABI}")


def read_lanczos(lanczos, handle):
    counters = LanczosCounters()
    counters.struct_size = C.sizeof(LanczosCounters)
    assert lanczos.gpu_lanczos_get_transfer_counters(handle,
        C.byref(counters)) == 0
    return counters


def read_residual(residual, handle):
    counters = ResidualCounters()
    counters.struct_size = C.sizeof(ResidualCounters)
    assert residual.gpu_residual_get_transfer_counters(handle,
        C.byref(counters)) == 0
    return counters


def download(host, device_ptr, pitch, width, height, cuda, context):
    """A device-to-host copy for the parity check. The gated chain makes none."""
    copy = Copy2D()
    copy.srcMemoryType = 2
    copy.srcDevice = device_ptr
    copy.srcPitch = pitch
    copy.dstMemoryType = 1
    copy.dstHost = host.ctypes.data
    copy.dstPitch = host.strides[0]
    copy.WidthInBytes = width * 4
    copy.Height = height
    assert cuda.cuCtxPushCurrent_v2(context) == 0
    try:
        assert cuda.cuMemcpy2D_v2(C.byref(copy)) == 0
    finally:
        previous = C.c_void_p()
        assert cuda.cuCtxPopCurrent_v2(C.byref(previous)) == 0


if __name__ == "__main__":
    main()
