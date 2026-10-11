"""GPU surface -> NVENC proof of conduct.

Stage B is the production-shaped run: the 15-frame 4K chain
nvdec -> nv12_rgba -> gpu_lanczos -> NVOF -> DLSSNR -> gpu_residual feeds its
final D3D12 RGBA texture straight into an in-process av1_nvenc encoder. Only
D3D12 and CUDA command traffic crosses that boundary; no image byte is
downloaded to the host and no FFmpeg stdin pipe exists.

The RGB->NV12 matrix has to be the mate of the NV12->RGB matrix that produced
the pixels. nv12_rgba_bridge converts with BT.601, which is also what the
legacy cv2.COLOR_YUV2BGR_NV12 decode in pipeline.iter_frames does, so Stage B
encodes with BT.601 and declares SMPTE170M. A BT.709 declaration here would
describe pixels that were never produced with BT.709 coefficients.

Stage A is a small numeric gate for the new RGB->NV12 kernel: a synthetic
gradient is uploaded (test input, not pipeline traffic), converted, encoded,
and the NV12 planes are read back with an explicitly counted copy and compared
byte for byte against an identical integer reference. Its input never passes
through a decoder, so it stays on BT.709 and covers that coefficient set.

Run from the repository root:
    python native/nvenc_gpu_smoke.py
"""
import ctypes as C
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FFMPEG_BIN = Path(os.environ.get("FFMPEG_SDK_DIR", r"F:\WhisperJAV\Library")) / "bin"

from nvdec_bridge_scale_smoke import Copy2D  # noqa: E402
from nvdec_bridge_smoke import Frame  # noqa: E402
from dlss_gpu_host_smoke import (DeviceOutput, Options, TransferCounters,  # noqa: E402
                                 load_gpu_host)

WIDTH, HEIGHT = 3840, 2160
LOW_WIDTH, LOW_HEIGHT = WIDTH // 2, HEIGHT // 2
FRAMES = 15

# Limited-range 16-bit fixed-point coefficients, mirroring nvenc_bridge.cpp.
BT709 = ((11960, 40201, 4064), (-6596, -22188, 28784), (28784, -26148, -2639))
BT601 = ((16823, 33039, 6418), (-9714, -19070, 28784), (28784, -24104, -4681))
# Selector values from native/nvenc_bridge.h.
BT709_MATRIX, BT601_MATRIX = 0, 1


class NvencOptions(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("codec", C.c_char_p),
        ("preset", C.c_char_p), ("tune", C.c_char_p), ("multipass", C.c_char_p),
        ("rate_control", C.c_char_p), ("constant_quality", C.c_int32),
        ("bitrate", C.c_int64), ("gop", C.c_int32), ("color_matrix", C.c_int32),
        ("extra_options", C.c_char_p)]


class NvencCounters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32)] + [(name, C.c_uint64) for name in (
        "frames_submitted", "packets_written", "staging_gpu_copy_bytes",
        "conversion_output_bytes", "host_pixel_bytes")]


class ResidualCounters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32),
        ("full_device_copy_bytes", C.c_uint64),
        ("working_device_copy_bytes", C.c_uint64),
        ("output_readback_bytes", C.c_uint64)]


def open_nvenc(library, device, context, width, height, fps, time_base,
               output_path, color_matrix):
    options = NvencOptions(C.sizeof(NvencOptions), b"av1_nvenc", b"p7", b"hq",
        b"qres", b"vbr", 24, 0, fps[0] * 2, color_matrix, None)
    handle = C.c_void_p()
    code = library.nvenc_open(device, context, width, height, fps[0], fps[1],
        time_base[0], time_base[1], os.fsencode(str(output_path)),
        C.byref(options), C.byref(handle))
    if code:
        raise RuntimeError(f"nvenc_open {output_path.name}: "
                           f"{library.nvenc_last_error().decode()}")
    return handle


def counters_of(library, handle):
    counters = NvencCounters()
    counters.struct_size = C.sizeof(NvencCounters)
    assert library.nvenc_get_transfer_counters(handle, C.byref(counters)) == 0
    return counters


def read_plane(cuda, context, device_ptr, pitch, width, height):
    """Diagnostic device->host copy. Counted by the caller, never a pipeline step."""
    host = np.zeros((height, width), np.uint8)
    copy = Copy2D()
    copy.srcMemoryType = 2
    copy.srcDevice = device_ptr
    copy.srcPitch = pitch
    copy.dstMemoryType = 1
    copy.dstHost = host.ctypes.data
    copy.dstPitch = host.strides[0]
    copy.WidthInBytes = width
    copy.Height = height
    assert cuda.cuCtxPushCurrent_v2(context) == 0
    try:
        assert cuda.cuMemcpy2D_v2(C.byref(copy)) == 0
    finally:
        previous = C.c_void_p()
        assert cuda.cuCtxPopCurrent_v2(C.byref(previous)) == 0
    return host


def block_average(values):
    rows, columns = values.shape
    return (values.reshape(rows // 2, 2, columns // 2, 2).sum(axis=(1, 3)) + 2) >> 2


def reference_nv12(rgba, matrix):
    (cy_r, cy_g, cy_b), (cu_r, cu_g, cu_b), (cv_r, cv_g, cv_b) = matrix
    red = rgba[..., 0].astype(np.int32)
    green = rgba[..., 1].astype(np.int32)
    blue = rgba[..., 2].astype(np.int32)
    luma = ((cy_r * red + cy_g * green + cy_b * blue + 32768) >> 16) + 16
    luma = np.clip(luma, 0, 255).astype(np.uint8)
    average = (block_average(red), block_average(green), block_average(blue))
    chroma = []
    for coefficients, offset in (((cu_r, cu_g, cu_b), 128), ((cv_r, cv_g, cv_b), 128)):
        value = sum(factor * sample for factor, sample in zip(coefficients, average))
        value = (value + 32768) >> 16
        chroma.append(np.clip(value + offset, 0, 255).astype(np.uint8))
    return luma, chroma[0], chroma[1]


def probe(path):
    dump = subprocess.check_output(["ffprobe", "-v", "error", "-print_format",
        "json", "-show_streams", str(path)], timeout=60)
    streams = json.loads(dump)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    return video


def main():
    source = ROOT / "bench_out" / "fixture_4k_300.mp4"
    assert source.exists(), source
    out_dir = ROOT / "bench_out"
    texture_output = out_dir / "nvenc_gpu_texture_av1.mkv"
    synth_output = out_dir / "nvenc_gpu_synth_av1.mkv"
    synth_width, synth_height = 640, 384

    cookie = os.add_dll_directory(str(FFMPEG_BIN))
    readback_bytes = 0
    try:
        directory = ROOT / "native"
        decoder = C.CDLL(str(directory / "nvdec_bridge.dll"))
        converter = C.CDLL(str(directory / "nv12_rgba_bridge.dll"))
        lanczos = C.CDLL(str(directory / "gpu_lanczos_bridge.dll"))
        nvof = C.CDLL(str(directory / "nvof_bridge.dll"))
        residual = C.CDLL(str(directory / "gpu_residual_bridge.dll"))
        encoder = C.CDLL(str(directory / "nvenc_bridge.dll"))
        host_lib = load_gpu_host()
        cuda = C.WinDLL("nvcuda.dll")

        decoder.nvdec_bridge_open.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        decoder.nvdec_bridge_open.restype = C.c_int32
        decoder.nvdec_bridge_next.argtypes = [C.c_void_p, C.POINTER(Frame)]
        decoder.nvdec_bridge_next.restype = C.c_int32
        decoder.nvdec_bridge_close.argtypes = [C.c_void_p]
        decoder.nvdec_bridge_last_error.restype = C.c_char_p
        converter.nv12_rgba_open.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32,
            C.POINTER(C.c_void_p)]
        converter.nv12_rgba_open.restype = C.c_int32
        converter.nv12_rgba_convert.argtypes = [C.c_void_p, C.c_uint64,
            C.c_uint32, C.c_uint64, C.c_uint32, C.POINTER(C.c_uint64),
            C.POINTER(C.c_uint32)]
        converter.nv12_rgba_convert.restype = C.c_int32
        converter.nv12_rgba_close.argtypes = [C.c_void_p]
        converter.nv12_rgba_last_error.restype = C.c_char_p
        lanczos.gpu_lanczos_open.argtypes = [C.c_void_p, C.c_uint32,
            C.c_uint32, C.POINTER(C.c_void_p)]
        lanczos.gpu_lanczos_open.restype = C.c_int32
        lanczos.gpu_lanczos_process.argtypes = [C.c_void_p, C.c_uint64,
            C.c_size_t, C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)]
        lanczos.gpu_lanczos_process.restype = C.c_int32
        lanczos.gpu_lanczos_close.argtypes = [C.c_void_p]
        lanczos.gpu_lanczos_last_error.restype = C.c_char_p
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
        residual.gpu_residual_open.argtypes = [C.c_void_p, C.c_uint32,
            C.c_uint32, C.POINTER(C.c_void_p)]
        residual.gpu_residual_open.restype = C.c_int32
        residual.gpu_residual_process.argtypes = [C.c_void_p, C.c_void_p,
            C.c_void_p, C.c_uint64, C.c_void_p, C.c_uint64, C.c_size_t,
            C.c_void_p, C.c_uint64, C.c_size_t, C.c_float, C.POINTER(C.c_void_p)]
        residual.gpu_residual_process.restype = C.c_int32
        residual.gpu_residual_close.argtypes = [C.c_void_p]
        residual.gpu_residual_last_error.restype = C.c_char_p
        residual.gpu_residual_get_transfer_counters.argtypes = [C.c_void_p,
            C.POINTER(ResidualCounters)]
        residual.gpu_residual_get_transfer_counters.restype = C.c_int32
        cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        cuda.cuCtxPushCurrent_v2.restype = C.c_int
        cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        cuda.cuCtxPopCurrent_v2.restype = C.c_int
        cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(Copy2D)]
        cuda.cuMemcpy2D_v2.restype = C.c_int
        cuda.cuMemAllocPitch_v2.argtypes = [C.POINTER(C.c_uint64),
            C.POINTER(C.c_size_t), C.c_size_t, C.c_size_t, C.c_uint]
        cuda.cuMemAllocPitch_v2.restype = C.c_int
        cuda.cuMemFree_v2.argtypes = [C.c_uint64]
        cuda.cuMemFree_v2.restype = C.c_int

        assert encoder.nvenc_abi_version() == 2
        encoder.nvenc_open.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32,
            C.c_uint32, C.c_uint32, C.c_uint32, C.c_int32, C.c_int32,
            C.c_char_p, C.POINTER(NvencOptions), C.POINTER(C.c_void_p)]
        encoder.nvenc_open.restype = C.c_int32
        encoder.nvenc_last_error.restype = C.c_char_p
        encoder.nvenc_encode_rgba.argtypes = [C.c_void_p, C.c_uint64,
            C.c_size_t, C.c_int64]
        encoder.nvenc_encode_rgba.restype = C.c_int32
        encoder.nvenc_encode_texture.argtypes = [C.c_void_p, C.c_void_p,
            C.c_void_p, C.c_uint64, C.c_int64]
        encoder.nvenc_encode_texture.restype = C.c_int32
        encoder.nvenc_get_last_nv12_planes.argtypes = [C.c_void_p,
            C.POINTER(C.c_uint64), C.POINTER(C.c_uint32), C.POINTER(C.c_uint64),
            C.POINTER(C.c_uint32)]
        encoder.nvenc_get_last_nv12_planes.restype = C.c_int32
        encoder.nvenc_get_transfer_counters.argtypes = [C.c_void_p,
            C.POINTER(NvencCounters)]
        encoder.nvenc_get_transfer_counters.restype = C.c_int32
        encoder.nvenc_finish.argtypes = [C.c_void_p]
        encoder.nvenc_finish.restype = C.c_int32
        encoder.nvenc_close.argtypes = [C.c_void_p]

        decode = C.c_void_p()
        convert = C.c_void_p()
        scale = C.c_void_p()
        flow_session = C.c_void_p()
        gpu_residual = C.c_void_p()
        dlss_handle = C.c_void_p()
        texture_bridge = None
        source_pts = []
        assert decoder.nvdec_bridge_open(os.fsencode(source),
            C.byref(decode)) == 0, decoder.nvdec_bridge_last_error()
        try:
            assert nvof.nvof_bridge_create(LOW_WIDTH, LOW_HEIGHT, 4, 10,
                C.byref(flow_session)) == 0, nvof.nvof_bridge_get_last_error()
            options = Options(C.sizeof(Options), 0, 0, 1, 1, 1, 0, 0, 0, 1, 1)
            assert host_lib.dlss_gpu_host_open(LOW_WIDTH, LOW_HEIGHT,
                str(ROOT / "nvngx_dlssnr.dll"), C.byref(options),
                C.byref(dlss_handle)) == 0, host_lib.dlss_gpu_host_last_error()
            time_base = None
            d3d12_device = None
            for index in range(FRAMES):
                frame = Frame()
                frame.struct_size = C.sizeof(Frame)
                assert decoder.nvdec_bridge_next(decode, C.byref(frame)) == 1, \
                    decoder.nvdec_bridge_last_error()
                if index == 0:
                    time_base = (frame.time_base_num, frame.time_base_den)
                    context = frame.cuda_context
                    assert converter.nv12_rgba_open(frame.cuda_context,
                        WIDTH, HEIGHT, C.byref(convert)) == 0, \
                        converter.nv12_rgba_last_error()
                    assert lanczos.gpu_lanczos_open(frame.cuda_context,
                        WIDTH, HEIGHT, C.byref(scale)) == 0, \
                        lanczos.gpu_lanczos_last_error()
                rgba_ptr = C.c_uint64()
                rgba_pitch = C.c_uint32()
                assert converter.nv12_rgba_convert(convert,
                    frame.y_device_ptr, frame.y_pitch_bytes,
                    frame.uv_device_ptr, frame.uv_pitch_bytes,
                    C.byref(rgba_ptr), C.byref(rgba_pitch)) == 0, \
                    converter.nv12_rgba_last_error()
                low_ptr = C.c_uint64()
                low_pitch = C.c_uint32()
                assert lanczos.gpu_lanczos_process(scale, rgba_ptr.value,
                    rgba_pitch.value, C.byref(low_ptr),
                    C.byref(low_pitch)) == 0, lanczos.gpu_lanczos_last_error()
                assert nvof.nvof_bridge_execute_rgba_device_sequential(
                    flow_session, low_ptr.value, low_pitch.value,
                    frame.cuda_context, int(index == 0), None) == 0, \
                    nvof.nvof_bridge_get_last_error()
                flow_ptr = C.c_uint64()
                flow_pitch = C.c_size_t()
                flow_context = C.c_void_p()
                assert nvof.nvof_bridge_get_output_deviceptr(flow_session,
                    C.byref(flow_ptr), C.byref(flow_pitch)) == 0
                device_id = C.c_uint32()
                owner = C.c_uint32()
                assert nvof.nvof_bridge_get_session_info(flow_session,
                    C.byref(device_id), C.byref(flow_context),
                    C.byref(owner)) == 0
                output = DeviceOutput()
                output.struct_size = C.sizeof(DeviceOutput)
                assert host_lib.dlss_gpu_host_process_device_color_resource(
                    dlss_handle, frame.cuda_context, low_ptr.value,
                    low_pitch.value, flow_context, flow_ptr.value,
                    flow_pitch.value, 1, C.byref(output)) == 0, \
                    host_lib.dlss_gpu_host_last_error()
                if index == 0:
                    assert residual.gpu_residual_open(output.d3d12_device,
                        LOW_WIDTH, LOW_HEIGHT,
                        C.byref(gpu_residual)) == 0, \
                        residual.gpu_residual_last_error()
                    d3d12_device = output.d3d12_device
                    texture_bridge = open_nvenc(encoder, d3d12_device,
                        frame.cuda_context, WIDTH, HEIGHT, (60, 1), time_base,
                        texture_output, BT601_MATRIX)
                full_output = C.c_void_p()
                assert residual.gpu_residual_process(gpu_residual,
                    output.d3d12_resource, output.d3d12_fence,
                    output.fence_value, frame.cuda_context, rgba_ptr.value,
                    rgba_pitch.value, frame.cuda_context, low_ptr.value,
                    low_pitch.value, C.c_float(1.0),
                    C.byref(full_output)) == 0, residual.gpu_residual_last_error()
                assert full_output.value
                assert encoder.nvenc_encode_texture(texture_bridge,
                    full_output.value, None, 0, frame.pts) == 0, \
                    encoder.nvenc_last_error().decode()
                source_pts.append(frame.pts * frame.time_base_num /
                                  frame.time_base_den)
                print(f"frame={index} pts={frame.pts} "
                      f"texture={full_output.value:#x}", flush=True)

            # Stage A: synthetic gradient through the device-pointer entry point.
            synth = np.zeros((synth_height, synth_width, 4), np.uint8)
            y_axis, x_axis = np.indices((synth_height, synth_width))
            synth[..., 0] = (x_axis * 3 + y_axis * 5) % 256
            synth[..., 1] = (x_axis * 7 - y_axis * 2) % 256
            synth[..., 2] = (x_axis + y_axis * 11) % 256
            synth[..., 3] = 255
            pitch = C.c_size_t()
            pointer = C.c_uint64()
            context = frame.cuda_context
            assert cuda.cuCtxPushCurrent_v2(context) == 0
            try:
                assert cuda.cuMemAllocPitch_v2(C.byref(pointer),
                    C.byref(pitch), synth_width * 4, synth_height, 16) == 0
                upload = Copy2D()
                upload.srcMemoryType = 1
                upload.srcHost = synth.ctypes.data
                upload.srcPitch = synth.strides[0]
                upload.dstMemoryType = 2
                upload.dstDevice = pointer.value
                upload.dstPitch = pitch.value
                upload.WidthInBytes = synth_width * 4
                upload.Height = synth_height
                assert cuda.cuMemcpy2D_v2(C.byref(upload)) == 0
            finally:
                previous = C.c_void_p()
                assert cuda.cuCtxPopCurrent_v2(C.byref(previous)) == 0
            synth_bridge = open_nvenc(encoder, output.d3d12_device, context,
                synth_width, synth_height, (60, 1), (1, 60), synth_output,
                BT709_MATRIX)
            try:
                assert encoder.nvenc_encode_rgba(synth_bridge, pointer.value,
                    pitch.value, 0) == 0, encoder.nvenc_last_error().decode()
                planes_y = C.c_uint64()
                planes_y_pitch = C.c_uint32()
                planes_uv = C.c_uint64()
                planes_uv_pitch = C.c_uint32()
                assert encoder.nvenc_get_last_nv12_planes(synth_bridge,
                    C.byref(planes_y), C.byref(planes_y_pitch),
                    C.byref(planes_uv), C.byref(planes_uv_pitch)) == 0, \
                    encoder.nvenc_last_error().decode()
                actual_y = read_plane(cuda, context, planes_y.value,
                    planes_y_pitch.value, synth_width, synth_height)
                actual_uv = read_plane(cuda, context, planes_uv.value,
                    planes_uv_pitch.value, synth_width, synth_height // 2)
                readback_bytes += (synth_width * synth_height +
                                   synth_width * synth_height // 2)
                expected_y, expected_u, expected_v = reference_nv12(synth, BT709)
                expected_uv = np.empty((synth_height // 2, synth_width), np.uint8)
                expected_uv[:, 0::2] = expected_u
                expected_uv[:, 1::2] = expected_v
                assert np.array_equal(actual_y, expected_y), \
                    f"luma mismatch {np.abs(actual_y.astype(int) - expected_y.astype(int)).max()}"
                assert np.array_equal(actual_uv, expected_uv), \
                    f"chroma mismatch {np.abs(actual_uv.astype(int) - expected_uv.astype(int)).max()}"
                synth_counters = counters_of(encoder, synth_bridge)
            finally:
                encoder.nvenc_close(synth_bridge)
            assert encoder.nvenc_finish(texture_bridge) == 0, \
                encoder.nvenc_last_error().decode()
            texture_counters = counters_of(encoder, texture_bridge)
            encoder.nvenc_close(texture_bridge)
            texture_bridge = None
            residual_counters = ResidualCounters()
            residual_counters.struct_size = C.sizeof(ResidualCounters)
            assert residual.gpu_residual_get_transfer_counters(gpu_residual,
                C.byref(residual_counters)) == 0
            dlss_counters = TransferCounters()
            dlss_counters.struct_size = C.sizeof(TransferCounters)
            assert host_lib.dlss_gpu_host_get_transfer_counters(dlss_handle,
                C.byref(dlss_counters)) == 0
        finally:
            if texture_bridge:
                encoder.nvenc_close(texture_bridge)
            if gpu_residual.value:
                residual.gpu_residual_close(gpu_residual)
            if dlss_handle.value:
                host_lib.dlss_gpu_host_close(dlss_handle)
            if flow_session.value:
                nvof.nvof_bridge_destroy(flow_session)
            if scale.value:
                lanczos.gpu_lanczos_close(scale)
            if convert.value:
                converter.nv12_rgba_close(convert)
            decoder.nvdec_bridge_close(decode)
    finally:
        cookie.close()

    # The bridge only exposes device-side counters; the harness readback above
    # is the single host image copy in this script and it is reported, not hidden.
    assert texture_counters.host_pixel_bytes == 0
    assert synth_counters.host_pixel_bytes == 0
    assert texture_counters.frames_submitted == FRAMES
    assert texture_counters.packets_written == FRAMES
    assert texture_counters.staging_gpu_copy_bytes == FRAMES * WIDTH * HEIGHT * 4
    assert texture_counters.conversion_output_bytes == FRAMES * WIDTH * HEIGHT * 3 // 2
    assert residual_counters.output_readback_bytes == 0
    assert dlss_counters.host_color_stage_bytes == 0
    assert dlss_counters.output_readback_bytes == 0

    texture_stream = probe(texture_output)
    synth_stream = probe(synth_output)
    print("texture_stream=" + json.dumps({key: texture_stream.get(key)
        for key in ("codec_name", "profile", "width", "height", "pix_fmt",
                    "field_order", "color_range", "color_space",
                    "color_transfer", "color_primaries", "r_frame_rate",
                    "avg_frame_rate", "time_base", "nb_frames")}, sort_keys=True),
        flush=True)
    print("synth_stream=" + json.dumps({key: synth_stream.get(key)
        for key in ("codec_name", "width", "height", "pix_fmt", "color_range",
                    "color_space", "r_frame_rate", "nb_frames")}, sort_keys=True),
        flush=True)
    assert (texture_stream["codec_name"], texture_stream["width"],
            texture_stream["height"]) == ("av1", WIDTH, HEIGHT)
    assert texture_stream["r_frame_rate"] == "60/1"
    # The declared matrix must be the one the pixels were actually written with.
    assert texture_stream["color_space"] == "smpte170m", texture_stream
    assert texture_stream["color_primaries"] == "smpte170m", texture_stream
    assert texture_stream["color_transfer"] == "smpte170m", texture_stream
    assert (synth_stream["codec_name"], synth_stream["width"],
            synth_stream["height"]) == ("av1", synth_width, synth_height)
    assert synth_stream["color_space"] == "bt709", synth_stream

    packets = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams",
        "v:0", "-show_entries", "packet=pts_time,duration_time",
        "-of", "csv=p=0", str(texture_output)], timeout=60)
    pts = sorted(float(line.split(",")[0]) for line in
                 packets.decode().splitlines() if line.strip())
    assert len(pts) == FRAMES, pts
    deltas = np.diff(np.array(source_pts))
    encoded = np.diff(np.array(pts))
    # Matroska stores presentation time in milliseconds, so a source time base
    # of 1/15360 cannot survive the container exactly.
    worst = float(np.abs(deltas - encoded).max())
    assert worst < 2e-3, (worst, deltas[:5], encoded[:5])

    # Decode the synthetic encode on the CPU and compare against the same
    # reference: a wrong plane base, pitch or chroma order survives to here.
    decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i",
        str(synth_output), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt",
        "nv12", "-"], timeout=120)
    plane = np.frombuffer(decoded, np.uint8).reshape(synth_height * 3 // 2,
                                                     synth_width)
    decoded_y = plane[:synth_height]
    decoded_uv = plane[synth_height:]
    error = max(float(np.abs(decoded_y.astype(np.int32) -
                             expected_y.astype(np.int32)).mean()),
                float(np.abs(decoded_uv.astype(np.int32) -
                             expected_uv.astype(np.int32)).mean()))
    psnr = 20 * np.log10(255.0 / error) if error else float("inf")
    assert psnr > 25.0, psnr

    print(f"nvenc_gpu_smoke=PASS frames={FRAMES} size={WIDTH}x{HEIGHT} "
          f"codec=av1 pts_max_error_seconds={worst:.3e} "
          f"synth_decoded_psnr_db={psnr:.2f} "
          f"staging_device_copy_bytes={texture_counters.staging_gpu_copy_bytes} "
          f"conversion_device_write_bytes={texture_counters.conversion_output_bytes} "
          f"bridge_host_pixel_bytes={texture_counters.host_pixel_bytes} "
          f"harness_readback_bytes={readback_bytes}", flush=True)


if __name__ == "__main__":
    main()
