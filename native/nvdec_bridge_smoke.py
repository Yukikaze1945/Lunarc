"""Short NVDEC device-surface, timestamp and transfer-counter contract gate."""

import ctypes as C
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
FFMPEG_BIN = Path(os.environ.get("FFMPEG_SDK_DIR", r"F:\WhisperJAV\Library")) / "bin"


class Frame(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32), ("width", C.c_uint32),
        ("height", C.c_uint32), ("software_format", C.c_uint32),
        ("y_device_ptr", C.c_uint64), ("uv_device_ptr", C.c_uint64),
        ("y_pitch_bytes", C.c_int32), ("uv_pitch_bytes", C.c_int32),
        ("cuda_context", C.c_void_p), ("pts", C.c_int64),
        ("time_base_num", C.c_int32), ("time_base_den", C.c_int32),
        ("color_range", C.c_int32), ("color_primaries", C.c_int32),
        ("color_transfer", C.c_int32), ("color_space", C.c_int32),
    ]


class Counters(C.Structure):
    """Mirror of NvdecTransferCounters in native/nvdec_bridge.h."""
    _fields_ = [
        ("struct_size", C.c_uint32), ("frames_decoded", C.c_uint64),
        ("device_frame_count", C.c_uint64), ("bitstream_bytes_submitted", C.c_uint64),
        ("host_frame_copy_count", C.c_uint64), ("host_frame_copy_bytes", C.c_uint64),
        ("scale_device_output_bytes", C.c_uint64),
    ]


COUNTER_FIELDS = [name for name, _ in Counters._fields_]


def bind_counters(lib):
    lib.nvdec_bridge_get_transfer_counters.argtypes = [C.c_void_p, C.POINTER(Counters)]
    lib.nvdec_bridge_get_transfer_counters.restype = C.c_int32


def read_counters(lib, handle):
    block = Counters()
    block.struct_size = C.sizeof(Counters)
    if lib.nvdec_bridge_get_transfer_counters(handle, C.byref(block)):
        raise RuntimeError(f"counters: {lib.nvdec_bridge_last_error()!r}")
    return {name: getattr(block, name) for name in COUNTER_FIELDS}


def hwdownload_reference(source, frames=1):
    """The same frames through FFmpeg's own CUDA download, as a byte reference."""
    command = ["ffmpeg", "-v", "error", "-hwaccel", "cuda",
               "-hwaccel_output_format", "cuda", "-i", str(source),
               "-vf", "hwdownload,format=nv12", "-frames:v", str(frames),
               "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"]
    return subprocess.check_output(command, timeout=60)


def main():
    source = ROOT / "bench_out" / "fixture_4k_300.mp4"
    if not source.exists():
        raise FileNotFoundError(source)
    cookie = os.add_dll_directory(str(FFMPEG_BIN))
    try:
        lib = C.CDLL(str(ROOT / "native" / "nvdec_bridge.dll"))
        lib.nvdec_bridge_abi_version.restype = C.c_uint32
        assert lib.nvdec_bridge_abi_version() == 3
        lib.nvdec_bridge_open.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        lib.nvdec_bridge_open.restype = C.c_int32
        lib.nvdec_bridge_open_scaled.argtypes = [C.c_char_p, C.c_uint32,
                                                 C.c_uint32, C.POINTER(C.c_void_p)]
        lib.nvdec_bridge_open_scaled.restype = C.c_int32
        lib.nvdec_bridge_next.argtypes = [C.c_void_p, C.POINTER(Frame)]
        lib.nvdec_bridge_next.restype = C.c_int32
        lib.nvdec_bridge_readback_last_frame.argtypes = [C.c_void_p, C.c_void_p,
                                                         C.c_size_t]
        lib.nvdec_bridge_readback_last_frame.restype = C.c_int32
        lib.nvdec_bridge_cancel.argtypes = [C.c_void_p]
        lib.nvdec_bridge_close.argtypes = [C.c_void_p]
        lib.nvdec_bridge_last_error.restype = C.c_char_p
        bind_counters(lib)

        def decode(open_scaled, width, height):
            handle = C.c_void_p()
            if open_scaled:
                code = lib.nvdec_bridge_open_scaled(
                    os.fsencode(source), width, height, C.byref(handle))
            else:
                code = lib.nvdec_bridge_open(os.fsencode(source), C.byref(handle))
            if code:
                raise RuntimeError(f"NVDEC open: {lib.nvdec_bridge_last_error()!r}")
            frames = []
            readback = None
            copied_bytes = 0
            try:
                for index in range(12):
                    frame = Frame()
                    frame.struct_size = C.sizeof(Frame)
                    code = lib.nvdec_bridge_next(handle, C.byref(frame))
                    if code != 1:
                        raise RuntimeError(f"NVDEC next {code}: {lib.nvdec_bridge_last_error()!r}")
                    assert (frame.width, frame.height, frame.software_format) == (width, height, 1)
                    assert frame.y_device_ptr and frame.uv_device_ptr and frame.cuda_context
                    assert frame.y_pitch_bytes >= frame.width
                    assert frame.uv_pitch_bytes >= frame.width
                    frames.append((frame.pts, frame.time_base_num, frame.time_base_den,
                                   frame.cuda_context))
                    # A decode call never moves a frame to host memory: the only
                    # copy this bridge can make is the readback asked for below,
                    # so these two fields may only reflect the readbacks made.
                    counters = read_counters(lib, handle)
                    assert counters["frames_decoded"] == index + 1, counters
                    assert counters["device_frame_count"] == index + 1, counters
                    assert counters["host_frame_copy_count"] == (1 if readback else 0), counters
                    assert counters["host_frame_copy_bytes"] == copied_bytes, counters
                    if index == 0 and not open_scaled:
                        # One counted download, against FFmpeg's own hwdownload.
                        payload = width * (height + height // 2)
                        host = (C.c_ubyte * payload)()
                        refused = lib.nvdec_bridge_readback_last_frame(
                            handle, host, width - 1)
                        assert refused != 0, "a stride narrower than the frame was accepted"
                        assert read_counters(lib, handle)["host_frame_copy_count"] == 0
                        code = lib.nvdec_bridge_readback_last_frame(handle, host, width)
                        if code:
                            raise RuntimeError(
                                f"readback {code}: {lib.nvdec_bridge_last_error()!r}")
                        moved = read_counters(lib, handle)
                        assert moved["host_frame_copy_count"] == 1, moved
                        assert moved["host_frame_copy_bytes"] == payload, moved
                        readback = bytes(host)
                        copied_bytes = payload
                counters = read_counters(lib, handle)
                assert counters["bitstream_bytes_submitted"] > 0, counters
                if open_scaled:
                    assert counters["scale_device_output_bytes"] == \
                        12 * (width * height * 3 // 2), counters
                else:
                    assert counters["scale_device_output_bytes"] == 0, counters
                    assert counters["host_frame_copy_count"] == 1, counters
                    assert counters["host_frame_copy_bytes"] == width * height * 3 // 2, counters
                lib.nvdec_bridge_cancel(handle)
                cancelled = Frame()
                cancelled.struct_size = C.sizeof(Frame)
                assert lib.nvdec_bridge_next(handle, C.byref(cancelled)) == -1
                assert b"cancelled" in lib.nvdec_bridge_last_error()
            finally:
                lib.nvdec_bridge_close(handle)
            assert len({item[3] for item in frames}) == 1
            return [item[0] * item[1] / item[2] for item in frames], counters, readback

        timestamps, full_counters, readback = decode(False, 3840, 2160)
        scaled_timestamps, scaled_counters, _ = decode(True, 1920, 1080)
    finally:
        cookie.close()
    assert timestamps == sorted(timestamps), timestamps
    assert scaled_timestamps == timestamps, (scaled_timestamps, timestamps)
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0",
               "-read_intervals", "%+#12", str(source)]
    reference = [float(value.split(",", 1)[0]) for value in subprocess.check_output(
        command, text=True, timeout=30).splitlines() if value.strip()][:12]
    assert len(reference) == 12
    assert all(abs(a - b) < 1e-5 for a, b in zip(timestamps, reference)), (timestamps, reference)
    assert readback == hwdownload_reference(source), \
        "the counted readback is not the frame FFmpeg downloads for the same input"
    print("nvdec_gpu_frames=12 source=3840x2160 scaled=1920x1080 format=NV12 "
          "cuda_context=stable pts=matched cancel=PASS device_surface_contract=PASS "
          f"host_copy_count={full_counters['host_frame_copy_count']} "
          f"host_copy_bytes={full_counters['host_frame_copy_bytes']} "
          f"readback=BYTE_IDENTICAL_TO_HWDOWNLOAD "
          f"scaled_device_output_bytes={scaled_counters['scale_device_output_bytes']}")


if __name__ == "__main__":
    main()
