"""ctypes adapter for the native FidelityFX Optical Flow bridge.

The returned descriptor is a borrowed CUDA float32x2 device allocation, full
input resolution, with pitch in bytes. It is valid through the next process
call or until close; no image pixels are copied to host memory here.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

_DLL_SEARCH_HANDLES = {}  # At most one search cookie per resolved runtime folder.
_PROBE_LIBRARIES = {}  # UI refreshes must not repeatedly LoadLibrary the same DLL.


def probe_bridge():
    """Check the installed adapter ABI without creating GPU sessions."""
    import bundle_paths

    try:
        dll_path = Path(bundle_paths.require_native('fidelityfx_flow_bridge.dll',
            'AMD FidelityFX Optical Flow')).resolve()
        for folder in (dll_path.parent, Path(__file__).resolve().parent / 'native'):
            if folder.exists() and str(folder) not in _DLL_SEARCH_HANDLES:
                _DLL_SEARCH_HANDLES[str(folder)] = os.add_dll_directory(str(folder))
        key = str(dll_path)
        library = _PROBE_LIBRARIES.get(key)
        if library is None:
            library = ctypes.WinDLL(key)
            _PROBE_LIBRARIES[key] = library
        library.ffx_flow_abi_version.restype = ctypes.c_uint32
        if library.ffx_flow_abi_version() != 1:
            raise RuntimeError('Unsupported FidelityFX flow bridge ABI')
        return {'available': True, 'error': None,
                'availability_scope': 'adapter ABI; actual device admission occurs per job'}
    except (OSError, RuntimeError, AttributeError) as exc:
        return {'available': False, 'error': str(exc),
                'availability_scope': 'adapter ABI; actual device admission occurs per job'}


class _Counters(ctypes.Structure):
    _fields_ = [
        ("process_calls", ctypes.c_uint64), ("successful_dispatches", ctypes.c_uint64),
        ("reset_dispatches", ctypes.c_uint64), ("scd_scene_change_bits", ctypes.c_uint64),
        ("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
        ("grid_width", ctypes.c_uint32), ("grid_height", ctypes.c_uint32),
        ("output_pitch_bytes", ctypes.c_uint32), ("flow_format_float32x2", ctypes.c_uint32),
        ("scd_history_bits", ctypes.c_uint32), ("scd_workgroups", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class FlowOutput:
    cuda_context: int
    deviceptr: int
    width: int
    height: int
    pitch_bytes: int
    format: str = "float32x2"
    units: str = "input_pixels"
    direction: str = "current_to_previous"


class FidelityFXFlow:
    def __init__(self, library: str | os.PathLike[str], d3d12_device: int,
                 cuda_context: int, width: int, height: int, quality: str = 'Quality'):
        dll_path = Path(library).resolve()
        # Keep Windows DLL search handles alive for the optical-flow runtime and
        # its existing CUDA/D3D interop import library dependency.
        for folder in (dll_path.parent, Path(__file__).resolve().parent / "native"):
            if folder.exists() and str(folder) not in _DLL_SEARCH_HANDLES:
                _DLL_SEARCH_HANDLES[str(folder)] = os.add_dll_directory(str(folder))
        self._dll = ctypes.WinDLL(str(dll_path))
        self._dll.ffx_flow_abi_version.restype = ctypes.c_uint32
        self._dll.ffx_flow_last_error.restype = ctypes.c_char_p
        self._dll.ffx_flow_open.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
        self._dll.ffx_flow_open.restype = ctypes.c_int32
        self._dll.ffx_flow_process.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint32)]
        self._dll.ffx_flow_process.restype = ctypes.c_int32
        self._dll.ffx_flow_get_counters.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters)]
        self._dll.ffx_flow_get_counters.restype = ctypes.c_int32
        self._dll.ffx_flow_close.argtypes = [ctypes.c_void_p]
        if self._dll.ffx_flow_abi_version() != 1:
            raise RuntimeError("Unsupported FidelityFX flow bridge ABI")
        self.cuda_context = int(cuda_context)
        self.width, self.height = int(width), int(height)
        self._handle = ctypes.c_void_p()
        self.quality = resolve_quality(quality)
        if self.quality == 'Performance':
            try:
                open_mode = self._dll.ffx_flow_open2
            except AttributeError:
                raise RuntimeError('FidelityFX performance mode requires the updated native bridge') from None
            open_mode.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
            open_mode.restype = ctypes.c_int32
            rc = open_mode(ctypes.c_void_p(d3d12_device), ctypes.c_void_p(cuda_context),
                self.width, self.height, 0, ctypes.byref(self._handle))
        else:
            rc = self._dll.ffx_flow_open(ctypes.c_void_p(d3d12_device),
                ctypes.c_void_p(cuda_context), self.width, self.height,
                ctypes.byref(self._handle))
        if rc:
            raise RuntimeError(self._error("open", rc))

    def _error(self, operation: str, rc: int) -> str:
        raw = self._dll.ffx_flow_last_error()
        return f"FidelityFX flow {operation} failed ({rc}): " + (raw or b"unknown").decode("utf-8", "replace")

    def process(self, cuda_rgba8_deviceptr: int, pitch_bytes: int, reset: bool = False) -> FlowOutput:
        ptr, pitch = ctypes.c_uint64(), ctypes.c_uint32()
        rc = self._dll.ffx_flow_process(self._handle, int(cuda_rgba8_deviceptr),
            int(pitch_bytes), int(bool(reset)), ctypes.byref(ptr), ctypes.byref(pitch))
        if rc:
            raise RuntimeError(self._error("process", rc))
        return FlowOutput(self.cuda_context, ptr.value, self.width, self.height, pitch.value)

    def counters(self) -> dict[str, int]:
        c = _Counters()
        rc = self._dll.ffx_flow_get_counters(self._handle, ctypes.byref(c))
        if rc:
            raise RuntimeError(self._error("counter query", rc))
        return {name: int(getattr(c, name)) for name, _ in c._fields_}

    def close(self) -> None:
        if self._handle:
            self._dll.ffx_flow_close(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "FidelityFXFlow":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def resolve_quality(value):
    # Existing saved AMD settings inherited the NVOF default, but used full size.
    if value in (None, '1F', 'Quality'):
        return 'Quality'
    if value == 'Performance':
        return value
    raise ValueError('AMD FidelityFX quality must be Performance or Quality')


class FlowProvider:
    """Independent SDK history per layer, borrowing the decode CUDA context."""
    name = 'AMD FidelityFX Optical Flow 1.1.4 (D3D12/CUDA)'
    borrows_decoder_context = True
    def __init__(self, device, quality='Quality'):
        self.device, self.engine = device, None
        self.quality = resolve_quality(quality)
        self._timing = {}
        self.frames_since_reset = 0

    def new_independent_layer(self):
        return FlowProvider(self.device, self.quality)

    def close_independent_layer(self):
        if self.engine:
            self.engine.close(); self.engine = None

    def supports_device_rgba_flow(self):
        return True

    def get_flow_device_rgba(self, frame_idx, color, cuda_context, is_discontinuous=False):
        import bundle_paths
        import time
        if color.get('format') != 'rgba8_cuda':
            raise ValueError('FidelityFX guidance requires independent RGBA8 CUDA input')
        size = (int(color['width']), int(color['height']))
        if self.engine is None:
            self.engine = FidelityFXFlow(bundle_paths.require_native('fidelityfx_flow_bridge.dll',
                'AMD FidelityFX Optical Flow'), self.device, cuda_context, *size, quality=self.quality)
        if size != (self.engine.width, self.engine.height) or cuda_context != self.engine.cuda_context:
            raise ValueError('FidelityFX history extent/context changed without reopening')
        started = time.perf_counter()
        result = self.engine.process(color['device_ptr'], color['pitch_bytes'], is_discontinuous)
        self.frames_since_reset = 1 if is_discontinuous else self.frames_since_reset + 1
        counters = self.engine.counters()
        self._timing = dict(guidance_backend='fidelityfx_device_rgba',
            flow_quality=self.quality,
            analysis_width=(size[0]+1)//2 if self.quality == 'Performance' else size[0],
            analysis_height=(size[1]+1)//2 if self.quality == 'Performance' else size[1],
            flow_ms=(time.perf_counter()-started)*1000, sdk_counters=counters,
            startup_suppression_frames=6, scene_history_suppression_frames=4,
            frames_since_reset=self.frames_since_reset,
            startup_suppressed=self.frames_since_reset <= 6,
            scene_history_suppressed=bool(counters['scd_history_bits'] & 15),
            scalar_readback_bytes_measured=counters['successful_dispatches'] * 12,
            host_image_bytes_measured=0)
        return dict(format=result.format, cuda_context=result.cuda_context,
            device_ptr=result.deviceptr, pitch_bytes=result.pitch_bytes,
            width=result.width, height=result.height, direction=result.direction, units=result.units)

    def timing(self):
        return self._timing

    def device_input_counters(self):
        if self.engine is None:
            return 0, 0
        counters = self.engine.counters()
        return (counters['successful_dispatches'],
                counters['successful_dispatches'] * self.engine.width * self.engine.height * 4)
