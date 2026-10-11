"""Default-off GPU scene observation and static low-frequency residual memory.

Scene observation is sampled and experimental. Static stability rejects changing
source patches; it does not claim motion compensation or universal cut detection.
The current high-frequency residual remains current, and no image is read to CPU.
"""
from __future__ import annotations
import ctypes as C
from dataclasses import dataclass, asdict
import math
import bundle_paths
from post_control import candidate_descriptor, _Candidate, PostControlError


@dataclass(frozen=True)
class TemporalControls:
    scene_detection: bool = False
    mode: str = "off"
    history_weight: float = .6

    @property
    def enabled(self):
        return self.scene_detection or self.mode != "off"

    def as_dict(self):
        return asdict(self)


def temporal_from_settings(settings=None):
    raw = (settings or {}).get("temporal_controls")
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or set(raw) - set(TemporalControls.__dataclass_fields__):
        raise ValueError("temporal_controls must contain only scene_detection, mode, history_weight")
    values = TemporalControls().as_dict()
    values.update(raw)
    if type(values["scene_detection"]) is not bool:
        raise ValueError("temporal_controls.scene_detection must be boolean")
    if values["mode"] not in ("off", "static_low_frequency"):
        raise ValueError("temporal mode supports off or static_low_frequency")
    weight = values["history_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= .9:
        raise ValueError("temporal history_weight must be finite in [0, 0.9]")
    values["history_weight"] = float(weight)
    return TemporalControls(**values)


CLASSIFICATIONS = ("first", "duplicate_static", "exposure_flash", "translation",
                   "cut", "ambiguous", "disabled")


class _Observe(C.Structure):
    _fields_ = [(key, C.c_uint32) for key in ("struct_size", "abi_version", "classification", "scene_cut")] + [
        (key, C.c_float) for key in ("similarity", "normalized_delta", "chroma_delta", "exposure_ratio", "shift_x", "shift_y")
    ] + [("scene_reset_count", C.c_uint32), ("reserved", C.c_uint32),
         ("source_transport_bytes", C.c_uint64), ("scalar_readback_bytes", C.c_uint64)]


class _Process(C.Structure):
    _fields_ = [(key, C.c_uint32) for key in ("struct_size", "abi_version", "history_used", "history_rejected", "unrepresentable_pixels", "reserved")] + [
        ("effective_history_weight", C.c_float), ("scene_reset_count", C.c_uint32)
    ] + [(key, C.c_uint64) for key in ("source_transport_bytes", "candidate_transport_bytes", "scalar_readback_bytes")]


class GpuTemporalControl:
    def __init__(self, controls=None, *, hdr=False):
        self.controls = controls or TemporalControls()
        self.hdr = bool(hdr)
        self._lib = None
        self._handle = None
        self._identity = None
        self._closed = False
        self.observations = []
        self.process_reports = []

    def _check(self, code, operation):
        if code:
            detail = self._lib.gpu_temporal_last_error()
            raise PostControlError(f"GPU temporal {operation} failed ({code}): {detail.decode('utf-8', 'replace')}")

    def _open(self, device, context, width, height):
        if self._closed:
            raise PostControlError("temporal owner is closed")
        identity = (int(device), int(context), int(width), int(height), 8 if self.hdr else 4)
        if self._identity is not None and self._identity != identity:
            raise PostControlError("temporal device/context/extent changed; reopen required")
        if self._handle is not None:
            return
        bundle_paths.add_dll_directory(bundle_paths.native_dir())
        self._lib = C.CDLL(bundle_paths.require_native("gpu_temporal_bridge.dll", "run GPU scene/temporal controls"))
        lib = self._lib
        lib.gpu_temporal_abi_version.restype = C.c_uint32
        if lib.gpu_temporal_abi_version() != 1:
            raise PostControlError("GPU temporal ABI incompatible")
        lib.gpu_temporal_last_error.restype = C.c_char_p
        lib.gpu_temporal_open.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p)]
        lib.gpu_temporal_open.restype = C.c_int32
        lib.gpu_temporal_observe.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.c_uint32, C.c_uint32, C.POINTER(_Observe)]
        lib.gpu_temporal_observe.restype = C.c_int32
        lib.gpu_temporal_process.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.POINTER(_Candidate), C.c_float, C.c_uint32,
            C.POINTER(C.c_void_p), C.POINTER(C.c_void_p), C.POINTER(C.c_uint64), C.POINTER(_Process)]
        lib.gpu_temporal_process.restype = C.c_int32
        lib.gpu_temporal_close.argtypes = [C.c_void_p]
        lib.gpu_temporal_close.restype = None
        handle = C.c_void_p()
        self._check(lib.gpu_temporal_open(device, context, width, height, identity[4], C.byref(handle)), "open")
        self._identity, self._handle = identity, handle

    def observe(self, *, device, cuda_context, source_ptr, source_pitch, width, height, reset):
        if self._closed:
            raise PostControlError("temporal owner is closed")
        if not self.controls.enabled:
            return {"classification": "disabled", "scene_cut": False}
        self._open(device, cuda_context, width, height)
        report = _Observe(C.sizeof(_Observe), 1)
        self._check(self._lib.gpu_temporal_observe(self._handle, source_ptr, source_pitch,
            int(reset), int(self.controls.scene_detection), C.byref(report)), "observe")
        if report.classification >= len(CLASSIFICATIONS):
            raise PostControlError("GPU temporal returned an unknown scene classification")
        result = {key: getattr(report, key) for key, _ in _Observe._fields_[2:] if key != "reserved"}
        result["classification"] = CLASSIFICATIONS[report.classification]
        result["scene_cut"] = bool(report.scene_cut)
        self.observations.append(result)
        return result

    def process(self, *, device, cuda_context, source_ptr, source_pitch, width, height, candidate, reset):
        if self._closed:
            raise PostControlError("temporal owner is closed")
        if self.controls.mode == "off":
            return candidate
        self._open(device, cuda_context, width, height)
        desc = candidate_descriptor(candidate)
        texture, fence, value = C.c_void_p(), C.c_void_p(), C.c_uint64()
        report = _Process(C.sizeof(_Process), 1)
        self._check(self._lib.gpu_temporal_process(self._handle, source_ptr, source_pitch,
            C.byref(desc), self.controls.history_weight, int(reset), C.byref(texture),
            C.byref(fence), C.byref(value), C.byref(report)), "process")
        self.process_reports.append({key: getattr(report, key) for key, _ in _Process._fields_[2:] if key != "reserved"})
        return {"d3d12_resource": texture.value, "d3d12_fence": fence.value, "fence_value": value.value,
                "d3d12_device": device, "width": width, "height": height, "state": 0x800,
                "dxgi_format": 10 if self.hdr else 28}

    def diagnostics(self):
        return {"controls": self.controls.as_dict(), "enabled": self.controls.enabled,
                "motion_reprojection": False, "host_image_bytes": 0,
                "observations": self.observations, "process_reports": self.process_reports}

    def close(self):
        if self._handle is not None:
            self._lib.gpu_temporal_close(self._handle)
            self._handle = None
        self._closed = True
