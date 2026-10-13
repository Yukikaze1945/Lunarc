"""Independent GPU controls applied after reconstruction/fusion, before encode.

HDR is signed linear scRGB (1 = 80 nits). SDR is normalized RGB code values;
this module does not pretend that the SDR code domain is linear light. Neutral
configuration does not load a DLL, allocate, copy, round or change a descriptor.
Lightness and chroma controls use Oklab; FP16 inputs are linear Rec.709.
The model and CG objective are outside this module.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass, asdict
import math
import os

import bundle_paths


@dataclass(frozen=True)
class PostControls:
    total_gain: float = 1.0
    lightness_gain: float = 1.0
    chroma_gain: float = 1.0
    low_frequency_gain: float = 1.0
    high_frequency_gain: float = 1.0
    hue_protection: float = 0.0
    dark_protection: float = 0.0
    highlight_protection: float = 0.0
    soft_compression: float = 0.0
    roi_kind: int = 0  # 0 disabled, 1 rectangle, 2 ellipse
    roi_bounds: tuple = (0.0, 0.0, 1.0, 1.0)
    roi_feather: float = 0.02  # fraction of min(width, height), feather inward
    debug_mode: int = 0  # 0 output, 1 raw residual, 2 controlled, 3 protection

    @property
    def neutral(self):
        return (all(getattr(self, key) == 1.0 for key in GAIN_FIELDS)
                and all(getattr(self, key) == 0.0 for key in PROTECTION_FIELDS)
                and self.roi_kind == 0 and self.debug_mode == 0)

    def as_dict(self):
        return asdict(self)


GAIN_FIELDS = ("total_gain", "lightness_gain", "chroma_gain",
               "low_frequency_gain", "high_frequency_gain")
PROTECTION_FIELDS = ("hue_protection", "dark_protection",
                     "highlight_protection", "soft_compression")


def controls_from_settings(settings=None):
    raw = (settings or {}).get("post_controls")
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError("post_controls must be an object")
    unknown = set(raw) - set(PostControls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown post controls: {', '.join(sorted(unknown))}")
    values = PostControls().as_dict()
    values.update(raw)
    for key in GAIN_FIELDS + PROTECTION_FIELDS + ("roi_feather",):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"post_controls.{key} must be a number")
        value = float(value)
        limit = 2.0 if key in GAIN_FIELDS else 1.0
        if not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError(f"post_controls.{key} must be finite in [0, {limit:g}]")
        values[key] = value
    for key in ("roi_kind", "debug_mode"):
        value = values[key]
        limit = 2 if key == "roi_kind" else 3
        if type(value) is not int or not 0 <= value <= limit:
            raise ValueError(f"post_controls.{key} must be an integer in [0, {limit}]")
    bounds = values["roi_bounds"]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        raise ValueError("post_controls.roi_bounds needs [left, top, right, bottom]")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 0 <= v <= 1 for v in bounds):
        raise ValueError("post_controls.roi_bounds must be finite normalized coordinates")
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("post_controls.roi_bounds must have positive area")
    values["roi_bounds"] = tuple(float(v) for v in bounds)
    return PostControls(**values)


class PostControlError(RuntimeError):
    pass


class _Options(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
                ("width", C.c_uint32), ("height", C.c_uint32),
                ("pixel_bytes", C.c_uint32)] + [
                    (key, C.c_float) for key in GAIN_FIELDS + PROTECTION_FIELDS
                ] + [("roi_kind", C.c_uint32), ("roi_bounds", C.c_float * 4),
                     ("roi_feather", C.c_float), ("debug_mode", C.c_uint32), ("color_mode", C.c_uint32)]


class _Candidate(C.Structure):
    _fields_ = [("kind", C.c_uint32), ("d3d_state", C.c_uint32),
                ("cuda_ptr", C.c_uint64), ("cuda_pitch", C.c_size_t),
                ("texture", C.c_void_p), ("fence", C.c_void_p),
                ("fence_value", C.c_uint64)]


class _Counters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32)] + [
        (key, C.c_uint64) for key in ("process_count", "source_transport_bytes",
            "candidate_transport_bytes", "scalar_readback_bytes", "unrepresentable_pixels")]


def candidate_descriptor(candidate):
    desc = _Candidate()
    if "device_ptr" in candidate:
        desc.kind, desc.cuda_ptr = 1, int(candidate["device_ptr"])
        desc.cuda_pitch = int(candidate["pitch_bytes"])
    else:
        desc.kind, desc.texture = 2, candidate["d3d12_resource"]
        desc.fence, desc.fence_value = candidate.get("d3d12_fence"), candidate.get("fence_value", 0)
        desc.d3d_state = 0x800
        if not desc.fence:
            if candidate.get("producer_completed") is not True:
                raise PostControlError("D3D candidate needs a producer fence or explicit completed contract")
            desc.kind = 3
    return desc


def native_options(controls, width, height, pixel_bytes, *, sdr10=False):
    options = _Options()
    options.struct_size, options.abi_version = C.sizeof(options), 2
    options.width, options.height, options.pixel_bytes = width, height, pixel_bytes
    options.color_mode = 2 if sdr10 else 1 if pixel_bytes == 8 else 0
    for key in GAIN_FIELDS + PROTECTION_FIELDS + ("roi_kind", "roi_feather", "debug_mode"):
        setattr(options, key, getattr(controls, key))
    options.roi_bounds[:] = controls.roi_bounds
    return options


class GpuPostControl:
    """Lazy, per-export owner. Output borrowed until next call; close before CUDA."""
    def __init__(self, controls=None, *, hdr=False, sdr10=False, library=None):
        self.controls = controls or PostControls()
        self.hdr = bool(hdr)
        self.sdr10 = bool(sdr10)
        self._lib = library
        self._handle = None
        self._identity = None
        self._closed = False
        self._report = {"enabled": not self.controls.neutral,
                        "lightness_chroma_space": "Oklab",
                        "domain": "relative_linear_rec709" if sdr10 else "linear_scrgb" if hdr else "normalized_sdr_code",
                        "neutral_bypass": self.controls.neutral,
                        "host_image_bytes": 0, "controls": self.controls.as_dict()}

    def _check(self, result, symbol):
        if result:
            error = self._lib.gpu_post_control_last_error()
            detail = error.decode("utf-8", "replace") if isinstance(error, bytes) else error
            raise PostControlError(f"{symbol} failed ({result}): {detail}")

    def _load(self):
        if self._lib is None:
            path = bundle_paths.require_native("gpu_post_control_bridge.dll", "apply GPU residual controls")
            if not os.path.isfile(path):
                raise PostControlError("GPU post controls requested but native bridge is missing")
            bundle_paths.add_dll_directory(bundle_paths.native_dir())
            self._lib = C.CDLL(path)
        lib = self._lib
        lib.gpu_post_control_abi_version.restype = C.c_uint32
        if lib.gpu_post_control_abi_version() != 2:
            raise PostControlError("GPU post control ABI is incompatible")
        lib.gpu_post_control_last_error.restype = C.c_char_p
        lib.gpu_post_control_open.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32,
            C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p)]
        lib.gpu_post_control_open.restype = C.c_int32
        lib.gpu_post_control_process.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t,
            C.POINTER(_Candidate), C.POINTER(_Options), C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p), C.POINTER(C.c_uint64), C.c_void_p]
        lib.gpu_post_control_process.restype = C.c_int32
        lib.gpu_post_control_close.argtypes = [C.c_void_p]
        lib.gpu_post_control_close.restype = None

    def process(self, *, device, cuda_context, source_ptr, source_pitch,
                width, height, candidate):
        if self._closed:
            raise PostControlError("post control owner is closed")
        if self.controls.neutral:
            return candidate  # strict identity, including producer ownership/fence
        identity = (int(device), int(cuda_context), int(width), int(height), 8 if self.hdr else 4)
        if self._identity is not None and identity != self._identity:
            raise PostControlError("post control device/context/extent changed; reopen required")
        if self._handle is None:
            self._load()
            handle = C.c_void_p()
            self._check(self._lib.gpu_post_control_open(device, cuda_context, width,
                height, identity[4], C.byref(handle)), "open")
            self._handle, self._identity = handle, identity
        desc = candidate_descriptor(candidate)
        options = native_options(self.controls, width, height, identity[4], sdr10=self.sdr10)
        texture, fence, value = C.c_void_p(), C.c_void_p(), C.c_uint64()
        counters = _Counters()
        counters.struct_size, counters.abi_version = C.sizeof(counters), 2
        self._check(self._lib.gpu_post_control_process(self._handle, source_ptr,
            source_pitch, C.byref(desc), C.byref(options), C.byref(texture),
            C.byref(fence), C.byref(value), C.byref(counters)), "process")
        self._report.update({key: getattr(counters, key) for key, _ in _Counters._fields_[2:]})
        return {"d3d12_resource": texture.value, "d3d12_fence": fence.value,
                "fence_value": value.value, "d3d12_device": device,
                "width": width, "height": height,
                "dxgi_format": 10 if self.hdr else 28, "state": 0x800}

    def diagnostics(self):
        return dict(self._report)

    def update_controls(self, controls):
        """Serialized preview-owner update; native options are sent per frame."""
        if self._closed:
            raise PostControlError("post control owner is closed")
        self.controls = controls
        self._report.update(enabled=not controls.neutral, neutral_bypass=controls.neutral,
                            controls=controls.as_dict())

    def close(self):
        if self._handle is not None:
            self._lib.gpu_post_control_close(self._handle)
            self._handle = None
        self._closed = True
