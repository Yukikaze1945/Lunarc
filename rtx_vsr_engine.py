"""Minimal Python binding for the optional NVIDIA VideoSuperRes VFX runtime.

No NVIDIA runtime binaries are bundled. Point runtime_directory at a locally
installed/licensed Maxine VFX SDK Core plus VideoSuperRes feature runtime.
VFX 1.3 accepts packed RGB10A2 as well as SDR RGBA8 device buffers.
"""
from __future__ import annotations

import ctypes
from pathlib import Path


class RtxVsrError(RuntimeError):
    pass


class Counters(ctypes.Structure):
    _fields_ = [
        ("process_calls", ctypes.c_uint64), ("successful_runs", ctypes.c_uint64),
        ("input_width", ctypes.c_uint32), ("input_height", ctypes.c_uint32),
        ("output_width", ctypes.c_uint32), ("output_height", ctypes.c_uint32),
        ("quality_level", ctypes.c_uint32), ("model_loaded", ctypes.c_uint32),
    ]


class RtxVsr:
    def __init__(self, bridge_path: str | Path, runtime_directory: str | Path,
                 cuda_context: int, input_size: tuple[int, int], output_size: tuple[int, int],
                 quality_level: int, *, is_hdr: bool = False, packed10: bool = False):
        if is_hdr and not packed10:
            raise RtxVsrError("HDR VSR requires packed RGB10A2; no 8-bit conversion is allowed")
        if not 1 <= int(quality_level) <= 4:
            raise ValueError("quality_level must be 1, 2, 3, or 4")
        self._dll = ctypes.WinDLL(str(Path(bridge_path).resolve()))
        self._dll.rtx_vsr_abi_version.restype = ctypes.c_uint32
        if self._dll.rtx_vsr_abi_version() != 1:
            raise RtxVsrError("unsupported RTX VSR bridge ABI")
        self._dll.rtx_vsr_last_error.restype = ctypes.c_char_p
        self._dll.rtx_vsr_open.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 5 + [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
        self._dll.rtx_vsr_open.restype = ctypes.c_int32
        self._dll.rtx_vsr_process.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p]
        self._dll.rtx_vsr_process.restype = ctypes.c_int32
        self._dll.rtx_vsr_get_counters.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters)]
        self._dll.rtx_vsr_get_counters.restype = ctypes.c_int32
        self._dll.rtx_vsr_close.argtypes = [ctypes.c_void_p]
        self._handle = ctypes.c_void_p()
        iw, ih = map(int, input_size)
        ow, oh = map(int, output_size)
        if packed10:
            self._dll.rtx_vsr_open2.argtypes = [ctypes.c_void_p] + [ctypes.c_uint32] * 6 + [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p)]
            result = self._dll.rtx_vsr_open2(ctypes.c_void_p(cuda_context), iw, ih, ow, oh,
                int(quality_level), 11, str(Path(runtime_directory).resolve()), ctypes.byref(self._handle))
        else:
            result = self._dll.rtx_vsr_open(ctypes.c_void_p(cuda_context), iw, ih, ow, oh,
                int(quality_level), str(Path(runtime_directory).resolve()), ctypes.byref(self._handle))
        if result:
            raise self._error("open", result)

    def _error(self, operation: str, code: int) -> RtxVsrError:
        message = self._dll.rtx_vsr_last_error()
        text = message.decode("utf-8", "replace") if message else "unknown native error"
        return RtxVsrError(f"RTX VSR {operation} failed ({code}): {text}")

    def process(self, input_cuda_rgba: int, input_pitch_bytes: int,
                output_cuda_rgba: int, output_pitch_bytes: int, cuda_stream: int = 0) -> None:
        result = self._dll.rtx_vsr_process(self._handle, input_cuda_rgba, input_pitch_bytes,
                                          output_cuda_rgba, output_pitch_bytes, ctypes.c_void_p(cuda_stream))
        if result:
            raise self._error("process", result)

    def counters(self) -> dict[str, int]:
        value = Counters()
        result = self._dll.rtx_vsr_get_counters(self._handle, ctypes.byref(value))
        if result:
            raise self._error("get_counters", result)
        return {name: getattr(value, name) for name, *_ in Counters._fields_}

    def close(self) -> None:
        if self._handle:
            self._dll.rtx_vsr_close(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "RtxVsr":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
