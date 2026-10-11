"""One-frame DLSS GPU-color input parity using decoded CUDA RGBA and NVOF flow."""
import ctypes as C
import os
import subprocess

import cv2
import numpy as np

from dlss_gpu_host_smoke import DeviceOutput, Options, TransferCounters, load_gpu_host
from dlss_gpu_engine import Live as GpuLive
from nvdec_bridge_smoke import FFMPEG_BIN, Frame, ROOT
from residual_native import NativeResidualBackend


class ResidualCounters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32),
        ("full_device_copy_bytes", C.c_uint64),
        ("working_device_copy_bytes", C.c_uint64),
        ("output_readback_bytes", C.c_uint64)]


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
    rgba_host = cv2.cvtColor(np.frombuffer(raw[frame_bytes:], np.uint8)
        .reshape(height * 3 // 2, width), cv2.COLOR_YUV2RGBA_NV12)
    full_width, full_height = width * 2, height * 2
    full_raw = subprocess.check_output(["ffmpeg", "-v", "error", "-hwaccel",
        "cuda", "-hwaccel_output_format", "cuda", "-i", str(source), "-vf",
        "hwdownload,format=nv12", "-frames:v", "2", "-f", "rawvideo",
        "-pix_fmt", "nv12", "pipe:1"], timeout=30)
    full_bytes = full_width * full_height * 3 // 2
    assert len(full_raw) == 2 * full_bytes
    full_bgr = cv2.cvtColor(np.frombuffer(full_raw[full_bytes:], np.uint8)
        .reshape(full_height * 3 // 2, full_width), cv2.COLOR_YUV2BGR_NV12)
    low_bgr = cv2.cvtColor(rgba_host, cv2.COLOR_RGBA2BGR)
    cpu_preparer = NativeResidualBackend(full_width, full_height, width, height)
    try:
        production_low = np.empty_like(low_bgr)
        cpu_preparer.prepare_working_color(full_bgr, production_low)
    finally:
        cpu_preparer.close()
    working_difference = np.abs(production_low.astype(np.int16)
        - low_bgr.astype(np.int16))
    print("scale_cuda_vs_production_lanczos_max_difference=" +
        str(int(working_difference.max())) + " differing_channels=" +
        str(int(np.count_nonzero(working_difference))), flush=True)

    cookie = os.add_dll_directory(str(FFMPEG_BIN))
    try:
        decoder = C.CDLL(str(ROOT / "native" / "nvdec_bridge.dll"))
        converter = C.CDLL(str(ROOT / "native" / "nv12_rgba_bridge.dll"))
        nvof = C.CDLL(str(ROOT / "native" / "nvof_bridge.dll"))
        residual = C.CDLL(str(ROOT / "native" / "gpu_residual_bridge.dll"))
        host = load_gpu_host()
        residual.gpu_residual_open.argtypes = [C.c_void_p, C.c_uint32,
            C.c_uint32, C.POINTER(C.c_void_p)]
        residual.gpu_residual_open.restype = C.c_int32
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
        nvof.nvof_bridge_get_output_deviceptr.argtypes = [C.c_void_p,
            C.POINTER(C.c_uint64), C.POINTER(C.c_size_t)]
        nvof.nvof_bridge_get_output_deviceptr.restype = C.c_int32
        nvof.nvof_bridge_get_session_info.argtypes = [C.c_void_p,
            C.POINTER(C.c_uint32), C.POINTER(C.c_void_p), C.POINTER(C.c_uint32)]
        nvof.nvof_bridge_get_session_info.restype = C.c_int32
        nvof.nvof_bridge_destroy.argtypes = [C.c_void_p]
        nvof.nvof_bridge_get_last_error.restype = C.c_char_p

        decode = C.c_void_p()
        full_decode = C.c_void_p()
        convert = C.c_void_p()
        full_convert = C.c_void_p()
        flow_session = C.c_void_p()
        assert decoder.nvdec_bridge_open_scaled(os.fsencode(source), width,
            height, C.byref(decode)) == 0, decoder.nvdec_bridge_last_error()
        try:
            assert decoder.nvdec_bridge_open(os.fsencode(source),
                C.byref(full_decode)) == 0, decoder.nvdec_bridge_last_error()
            for index in range(2):
                full_frame = Frame()
                full_frame.struct_size = C.sizeof(Frame)
                assert decoder.nvdec_bridge_next(full_decode,
                    C.byref(full_frame)) == 1
                if index == 0:
                    assert converter.nv12_rgba_open(full_frame.cuda_context,
                        full_width, full_height,
                        C.byref(full_convert)) == 0, converter.nv12_rgba_last_error()
                full_rgba_ptr = C.c_uint64()
                full_rgba_pitch = C.c_uint32()
                assert converter.nv12_rgba_convert(full_convert,
                    full_frame.y_device_ptr, full_frame.y_pitch_bytes,
                    full_frame.uv_device_ptr, full_frame.uv_pitch_bytes,
                    C.byref(full_rgba_ptr), C.byref(full_rgba_pitch)) == 0, \
                    converter.nv12_rgba_last_error()
            assert nvof.nvof_bridge_create(width, height, 4, 10,
                C.byref(flow_session)) == 0, nvof.nvof_bridge_get_last_error()
            for index in range(2):
                frame = Frame()
                frame.struct_size = C.sizeof(Frame)
                assert decoder.nvdec_bridge_next(decode, C.byref(frame)) == 1
                if index == 0:
                    assert converter.nv12_rgba_open(frame.cuda_context, width,
                        height, C.byref(convert)) == 0, converter.nv12_rgba_last_error()
                rgba_ptr = C.c_uint64()
                rgba_pitch = C.c_uint32()
                assert converter.nv12_rgba_convert(convert,
                    frame.y_device_ptr, frame.y_pitch_bytes,
                    frame.uv_device_ptr, frame.uv_pitch_bytes,
                    C.byref(rgba_ptr), C.byref(rgba_pitch)) == 0, \
                    converter.nv12_rgba_last_error()
                assert nvof.nvof_bridge_execute_rgba_device_sequential(
                    flow_session, rgba_ptr.value, rgba_pitch.value,
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
            assert (full_frame.pts * full_frame.time_base_num /
                full_frame.time_base_den) == (frame.pts * frame.time_base_num /
                frame.time_base_den)
            options = Options(C.sizeof(Options), 0, 0, 1, 1, 1, 0, 0, 0, 1, 1)
            outputs = {}
            for mode in ("host", "device", "resource"):
                handle = C.c_void_p()
                code = host.dlss_gpu_host_open(width, height,
                    str(ROOT / "nvngx_dlssnr.dll"), C.byref(options),
                    C.byref(handle))
                if code:
                    raise RuntimeError(f"host open {code}: {host.dlss_gpu_host_last_error()!r}")
                try:
                    results = []
                    for pass_index in range(2):
                        result = np.empty_like(rgba_host)
                        if mode == "host":
                            code = host.dlss_gpu_host_process(handle,
                                rgba_host.ctypes.data, rgba_host.strides[0],
                                flow_context, flow_ptr.value, flow_pitch.value,
                                int(pass_index == 0), result.ctypes.data,
                                result.strides[0])
                        elif mode == "device":
                            code = host.dlss_gpu_host_process_device_color(handle,
                                frame.cuda_context, rgba_ptr.value, rgba_pitch.value,
                                flow_context, flow_ptr.value, flow_pitch.value,
                                int(pass_index == 0), result.ctypes.data,
                                result.strides[0])
                        else:
                            resource = DeviceOutput()
                            resource.struct_size = C.sizeof(DeviceOutput)
                            code = host.dlss_gpu_host_process_device_color_resource(
                                handle, frame.cuda_context, rgba_ptr.value,
                                rgba_pitch.value, flow_context, flow_ptr.value,
                                flow_pitch.value, int(pass_index == 0),
                                C.byref(resource))
                            if code == 0:
                                assert (resource.width, resource.height,
                                    resource.dxgi_format) == (width, height, 28)
                                assert resource.d3d12_device and resource.d3d12_resource
                                assert resource.d3d12_fence and resource.fence_value
                                before = TransferCounters()
                                before.struct_size = C.sizeof(TransferCounters)
                                assert host.dlss_gpu_host_get_transfer_counters(
                                    handle, C.byref(before)) == 0
                                assert (before.output_readback_count,
                                    before.output_readback_bytes) == (pass_index,
                                    pass_index * width * height * 4)
                                code = host.dlss_gpu_host_readback_output(handle,
                                    result.ctypes.data, result.strides[0])
                                if pass_index == 0 and code == 0:
                                    gpu_recon = C.c_void_p()
                                    assert residual.gpu_residual_open(
                                        resource.d3d12_device, width, height,
                                        C.byref(gpu_recon)) == 0, \
                                        residual.gpu_residual_last_error()
                                    try:
                                        output_texture = C.c_void_p()
                                        code = residual.gpu_residual_process(
                                            gpu_recon, resource.d3d12_resource,
                                            resource.d3d12_fence,
                                            resource.fence_value,
                                            full_frame.cuda_context,
                                            full_rgba_ptr.value,
                                            full_rgba_pitch.value,
                                            frame.cuda_context, rgba_ptr.value,
                                            rgba_pitch.value, C.c_float(1.0),
                                            C.byref(output_texture))
                                        assert code == 0 and output_texture.value, \
                                            residual.gpu_residual_last_error()
                                        transfer = ResidualCounters()
                                        transfer.struct_size = C.sizeof(ResidualCounters)
                                        assert residual.gpu_residual_get_transfer_counters(
                                            gpu_recon, C.byref(transfer)) == 0
                                        assert transfer.output_readback_bytes == 0
                                        assert transfer.full_device_copy_bytes == \
                                            full_width * full_height * 4
                                        assert transfer.working_device_copy_bytes == \
                                            width * height * 4
                                        gpu_result = np.empty((full_height,
                                            full_width, 4), np.uint8)
                                        assert residual.gpu_residual_readback(
                                            gpu_recon, gpu_result.ctypes.data,
                                            gpu_result.strides[0]) == 0, \
                                            residual.gpu_residual_last_error()
                                        cpu = NativeResidualBackend(full_width,
                                            full_height, width, height)
                                        try:
                                            expected = np.empty_like(full_bgr)
                                            cpu.reconstruct_from_rgba(full_bgr,
                                                low_bgr, result, 1.0, expected)
                                        finally:
                                            cpu.close()
                                        actual = cv2.cvtColor(gpu_result,
                                            cv2.COLOR_RGBA2BGR)
                                        difference = np.abs(actual.astype(np.int16)
                                            - expected.astype(np.int16))
                                        print("gpu_residual_max_difference=" +
                                            str(int(difference.max())) +
                                            " differing_channels=" +
                                            str(int(np.count_nonzero(difference))),
                                            flush=True)
                                        assert np.array_equal(actual, expected)
                                        assert residual.gpu_residual_process(
                                            gpu_recon, resource.d3d12_resource,
                                            resource.d3d12_fence,
                                            resource.fence_value,
                                            full_frame.cuda_context,
                                            full_rgba_ptr.value,
                                            full_rgba_pitch.value,
                                            frame.cuda_context, rgba_ptr.value,
                                            rgba_pitch.value, C.c_float(0.5),
                                            C.byref(output_texture)) == 0, \
                                            residual.gpu_residual_last_error()
                                        assert residual.gpu_residual_readback(
                                            gpu_recon, gpu_result.ctypes.data,
                                            gpu_result.strides[0]) == 0
                                        cpu = NativeResidualBackend(full_width,
                                            full_height, width, height)
                                        try:
                                            cpu.reconstruct_from_rgba(full_bgr,
                                                low_bgr, result, 0.5, expected)
                                        finally:
                                            cpu.close()
                                        actual = cv2.cvtColor(gpu_result,
                                            cv2.COLOR_RGBA2BGR)
                                        assert np.array_equal(actual, expected), \
                                            "GPU residual strength=0.5 differs from CPU"
                                    finally:
                                        residual.gpu_residual_close(gpu_recon)
                        if code:
                            raise RuntimeError(f"{mode} frame {code}: {host.dlss_gpu_host_last_error()!r}")
                        results.append(result.copy())
                    counters = TransferCounters()
                    counters.struct_size = C.sizeof(TransferCounters)
                    assert host.dlss_gpu_host_get_transfer_counters(handle,
                        C.byref(counters)) == 0
                    assert counters.output_readback_count == 2
                    assert counters.output_readback_bytes == 2 * width * height * 4
                    if mode != "host":
                        assert (counters.host_color_stage_count,
                                counters.host_color_stage_bytes) == (0, 0)
                        assert (counters.device_color_copy_count,
                                counters.device_color_copy_bytes) == (2, 2 * width * height * 4)
                    outputs[mode] = results
                finally:
                    host.dlss_gpu_host_close(handle)
            for index, (host_result, device_result, resource_result) in enumerate(zip(
                    outputs["host"], outputs["device"], outputs["resource"])):
                assert np.array_equal(host_result, device_result), \
                    f"DLSS host/device color output bytes differ on pass {index}"
                assert np.array_equal(device_result, resource_result), \
                    f"DLSS device resource output bytes differ on pass {index}"
            settings = {"preset": 0, "style": 0, "intensity": 1.0,
                "local_tone": 1.0, "local_struct": 1.0, "skin_struct": 0.0,
                "use_auto_mask": 0, "ui_correction": 0,
                "motion_scale_x": 1.0, "motion_scale_y": 1.0}
            live = GpuLive(width, height, settings)
            try:
                color = {"format": "rgba8_cuda", "width": width, "height": height,
                    "cuda_context": frame.cuda_context,
                    "device_ptr": rgba_ptr.value, "pitch_bytes": rgba_pitch.value}
                flow = {"format": "float32x2", "width": width, "height": height,
                    "cuda_context": flow_context.value,
                    "device_ptr": flow_ptr.value, "pitch_bytes": flow_pitch.value}
                wrapped = live.process_device_color_flow(color, flow, reset=True)
                assert np.array_equal(wrapped, outputs["host"][0])
                assert live.transfer_counters()["host_color_stage_bytes"] == 0
            finally:
                live.close()
            live = GpuLive(width, height, settings)
            try:
                descriptor = live.process_device_color_flow_resource(
                    color, flow, reset=True)
                assert descriptor["d3d12_resource"] and descriptor["dxgi_format"] == 28
                assert live.transfer_counters()["output_readback_bytes"] == 0
                assert np.array_equal(live.readback_output(), outputs["host"][0])
            finally:
                live.close()
        finally:
            if convert.value:
                converter.nv12_rgba_close(convert)
            if full_convert.value:
                converter.nv12_rgba_close(full_convert)
            if flow_session.value:
                nvof.nvof_bridge_destroy(flow_session)
            decoder.nvdec_bridge_close(decode)
            if full_decode.value:
                decoder.nvdec_bridge_close(full_decode)
    finally:
        cookie.close()
    print("nvdec_to_dlss_device_output=BYTE_IDENTICAL passes=2 "
          "color_host_stage=0 color_device_copy_per_pass=8294400 "
          "output_readback_at_process=0 explicit_parity_readback_per_pass=8294400")


if __name__ == "__main__":
    main()
