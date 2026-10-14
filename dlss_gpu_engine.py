"""The project DLSSNR host for a GPU-resident frame chain.

Motion arrives from a borrowed NVOF CUDA flow descriptor, from a CPU float2
plane the caller already holds in host memory (the shape a CPU optical-flow
engine hands over), or from a zero motion texture the host creates at
open and never writes again: the guidance-off entry points touch neither CUDA nor
the NVOF bridge, and upload no per-frame motion. Color is uploaded from host
memory or borrowed from a CUDA allocation; the DLSS output returns to host memory
or as a borrowed D3D12 texture. Depth is an immutable zero texture on the GPU
either way.

A session settles its colour once, at open: RGBA8 exactly as it always was, or -
only when the caller asks for it and the loaded host can transport an eight-byte
row - R16G16B16A16_FLOAT scRGB for an HDR chain. Nothing here converts between the
two, and a host that cannot open the wide session refuses rather than drawing an
8-bit picture nobody asked for.

This is the only DLSS host this build ships a loader for.
"""
import ctypes
import os
import time

import numpy as np

import bundle_paths
from dlss_errors import DLSSInferenceError


from dlssnr_runtime import resolve_runtime

# Legacy path hint for callers; sessions resolve the verified user-owned cache.
# DLSSNR is no longer collected into the application bundle.
DLSSNR_DLL = bundle_paths.collected("nvngx_dlssnr.dll")


# native/ as the running layout put it: the contents directory of a PyInstaller bundle
# is not the folder this file sits in once the code is compiled into the archive, so a
# packaged GUI that resolved it from __file__ was pointing at a directory the spec
# never collected into.
_LIBRARY = bundle_paths.native_dll("dlss_gpu_host.dll")
# ABI 3 adds the guidance-off entries; ABI 2 is the flow-only host, which this
# wrapper still drives so an unbuilt tree cannot break the guided backend.
_ABI_MAXIMUM = 3
_ABI_MINIMUM = 2


class _Options(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32), ("preset", ctypes.c_int32),
        ("style", ctypes.c_int32), ("intensity", ctypes.c_float),
        ("local_tone", ctypes.c_float), ("local_structure", ctypes.c_float),
        ("skin_structure", ctypes.c_float), ("auto_mask", ctypes.c_int32),
        ("ui_correction", ctypes.c_int32), ("motion_scale_x", ctypes.c_float),
        ("motion_scale_y", ctypes.c_float),
    ]


# (settings key, default, caster) in _Options field order after struct_size, so a
# caller that asks what a session baked reads the same list the struct is built from.
_BAKED_OPTIONS = (
    ("preset", 1, int), ("style", 0, int), ("intensity", 1.0, float),
    ("local_tone", 1.0, float), ("local_struct", 1.0, float),
    ("skin_struct", 1.0, float), ("use_auto_mask", 1, int),
    ("ui_correction", 0, int), ("motion_scale_x", 1.0, float),
    ("motion_scale_y", 1.0, float),
)


def baked_option_key(settings=None):
    """The native option tuple, usable as a hashable session settings key.

    Preset changes require recreation; supported NR options may be updated on
    the owner thread through update_runtime_options with a history reset.
    """
    s = settings or {}
    return tuple(cast(s.get(name, default)) for name, default, cast in _BAKED_OPTIONS)


class _TransferCounters(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32)] + [
        (name, ctypes.c_uint64) for name in (
            "host_color_stage_count", "host_color_stage_bytes",
            "device_color_copy_count", "device_color_copy_bytes",
            "output_readback_count", "output_readback_bytes")]


class _DeviceOutput(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32),
        ("width", ctypes.c_uint32), ("height", ctypes.c_uint32),
        ("dxgi_format", ctypes.c_uint32),
        ("d3d12_device", ctypes.c_void_p),
        ("d3d12_resource", ctypes.c_void_p),
        ("d3d12_fence", ctypes.c_void_p),
        ("fence_value", ctypes.c_uint64)]


# Per-session colour formats. "rgba8" is every entry point this wrapper has ever
# used; "rgba16f" opens a session whose Feature 18 colour and output are
# R16G16B16A16_FLOAT, which is what the HDR chain needs and what a caller opts a
# single session into - the shipped default never changes.
_COLOR_BYTES = {"rgba8": 4, "rgba16f": 8}
_COLOR_FORMAT = {"rgba8": 0, "rgba16f": 1}
_COLOR_DESCRIPTOR = {"rgba8": "rgba8_cuda", "rgba16f": "rgba16f_cuda"}
# The two new refusal codes of an FP16 open, and what they mean.
_OPEN_MESSAGES = {
    22: "the colour format or open info is not one this host knows",
    23: "the loaded nvof_d3d12_interop.dll does not transport an eight-byte row",
}


class _OpenInfo(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32),
        ("color_format", ctypes.c_uint32), ("flags", ctypes.c_uint32)]


class _Capabilities(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32),
        ("color_formats", ctypes.c_uint32),
        ("row_bridge_capabilities", ctypes.c_uint32),
        ("fp16_row_bridge_ready", ctypes.c_uint32)]


class _ColorInfo(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32)] + [
        (name, ctypes.c_uint32) for name in ("width", "height", "dxgi_format",
            "bytes_per_pixel")] + [
        ("row_bytes", ctypes.c_uint64)] + [
        (name, ctypes.c_uint32) for name in ("color_footprint_pitch",
            "output_footprint_pitch", "bridge_row_pitch",
            "bridge_bytes_per_row")] + [
        (name, ctypes.c_uint64) for name in ("last_copy_src_pitch",
            "last_copy_row_bytes", "last_copy_fence_value",
            "device_color_copy_count")]


def _host_dtype(bytes_per_pixel):
    return np.uint8 if bytes_per_pixel == 4 else np.float16


def _plane_description(bytes_per_pixel):
    return ("contiguous uint8 HxWx4" if bytes_per_pixel == 4 else
            "contiguous float16 HxWx4")


def _bind_zero_motion(library):
    """Prototype the guidance-off entries, if the loaded host has them."""
    try:
        library.dlss_gpu_host_process_zero_motion.argtypes = [ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int32, ctypes.c_void_p,
            ctypes.c_size_t]
        library.dlss_gpu_host_process_zero_motion.restype = ctypes.c_int32
        library.dlss_gpu_host_process_device_color_zero_motion_resource.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t,
            ctypes.c_int32, ctypes.POINTER(_DeviceOutput)]
        library.dlss_gpu_host_process_device_color_zero_motion_resource.restype = \
            ctypes.c_int32
    except AttributeError:
        return False
    return True


def _bind_host_motion(library):
    """Prototype the host-motion entry and its readback, if the host has them.

    An additive entry at the same ABI as the zero-motion host: a build that
    predates it simply lacks the symbol, so this probe - not a version check -
    is what tells a guided host-pointer caller that the frame would otherwise
    run behind its back on the legacy DLL.
    """
    try:
        library.dlss_gpu_host_process_host_motion.argtypes = [ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t]
        library.dlss_gpu_host_process_host_motion.restype = ctypes.c_int32
        library.dlss_gpu_host_readback_motion.argtypes = [ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_size_t]
        library.dlss_gpu_host_readback_motion.restype = ctypes.c_int32
    except AttributeError:
        return False
    return True


def _bind_fp16_color(library):
    """Prototype the opt-in per-session colour entries and ask the loaded host.

    The ABI number is deliberately unchanged - nothing an existing caller binds
    changed - so symbol presence plus the host's own answer is what tells a
    caller that an FP16 session is openable here. When the host says the row
    transport is not ready, an FP16 request has to be refused rather than run as
    an 8-bit picture nobody asked for.
    """
    try:
        library.dlss_gpu_host_open_ex.argtypes = [ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_wchar_p, ctypes.POINTER(_Options),
            ctypes.POINTER(_OpenInfo), ctypes.POINTER(ctypes.c_void_p)]
        library.dlss_gpu_host_open_ex.restype = ctypes.c_int32
        library.dlss_gpu_host_open_layer_ex.argtypes = [ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(_Options),
            ctypes.POINTER(_OpenInfo), ctypes.POINTER(ctypes.c_void_p)]
        library.dlss_gpu_host_open_layer_ex.restype = ctypes.c_int32
        library.dlss_gpu_host_query_capabilities.argtypes = [
            ctypes.POINTER(_Capabilities)]
        library.dlss_gpu_host_query_capabilities.restype = ctypes.c_int32
        library.dlss_gpu_host_get_color_info.argtypes = [ctypes.c_void_p,
            ctypes.POINTER(_ColorInfo)]
        library.dlss_gpu_host_get_color_info.restype = ctypes.c_int32
        library.dlss_gpu_host_d3d12_device.argtypes = [ctypes.c_void_p]
        library.dlss_gpu_host_d3d12_device.restype = ctypes.c_void_p
    except AttributeError:
        return False
    capabilities = _Capabilities(ctypes.sizeof(_Capabilities))
    if library.dlss_gpu_host_query_capabilities(ctypes.byref(capabilities)):
        return False
    library.color_capabilities = {
        name: getattr(capabilities, name) for name, _ in
        _Capabilities._fields_ if name != "struct_size"}
    return bool(capabilities.fp16_row_bridge_ready and
                capabilities.color_formats & (1 << _COLOR_FORMAT["rgba16f"]))


def hdr_fp16_supported():
    """Whether a session can be opened with R16G16B16A16_FLOAT colour here.

    False says the running build cannot, which a caller must treat as a missing
    capability rather than as permission to run the SDR path instead.
    """
    return bool(getattr(_load(), "fp16_color_supported", False))


def _load():
    # The host imports nvof_d3d12_interop.dll at load time. Its own directory resolves
    # that, and so does this one - kept for the case where a bundle flattened the
    # interop DLL to the contents root instead of beside its user.
    bundle_paths.add_dll_directory(bundle_paths.native_dir())
    if not os.path.isfile(_LIBRARY):
        raise bundle_paths.missing_binary(_LIBRARY, "run the GPU-resident DLSS host")
    library = ctypes.CDLL(_LIBRARY)
    library.dlss_gpu_host_abi_version.restype = ctypes.c_uint32
    abi = library.dlss_gpu_host_abi_version()
    if not _ABI_MINIMUM <= abi <= _ABI_MAXIMUM:
        raise RuntimeError(f"dlss_gpu_host.dll ABI mismatch: {abi}, this wrapper "
                           f"speaks {_ABI_MINIMUM} to {_ABI_MAXIMUM}")
    # Kept off the version check so a session can answer which host file and which
    # ABI it is running: what drew a frame is invisible in the frame, and the status
    # record is the only place an operator can read it back.
    library.abi_version = abi
    library.dlss_gpu_host_open.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_wchar_p, ctypes.POINTER(_Options), ctypes.POINTER(ctypes.c_void_p)]
    library.dlss_gpu_host_open.restype = ctypes.c_int32
    try:
        library.dlss_gpu_host_open_layer.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.POINTER(_Options), ctypes.POINTER(ctypes.c_void_p)]
        library.dlss_gpu_host_open_layer.restype = ctypes.c_int32
        library.dlss_gpu_host_session_id.argtypes = [ctypes.c_void_p]
        library.dlss_gpu_host_session_id.restype = ctypes.c_uint64
        library.multi_session_supported = True
    except AttributeError:
        library.multi_session_supported = False
    library.dlss_gpu_host_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t,
        ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t]
    library.dlss_gpu_host_process.restype = ctypes.c_int32
    library.dlss_gpu_host_process_device_color.argtypes = [ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int32, ctypes.c_void_p,
        ctypes.c_size_t]
    library.dlss_gpu_host_process_device_color.restype = ctypes.c_int32
    library.dlss_gpu_host_process_device_color_resource.argtypes = [ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int32,
        ctypes.POINTER(_DeviceOutput)]
    library.dlss_gpu_host_process_device_color_resource.restype = ctypes.c_int32
    library.dlss_gpu_host_readback_output.argtypes = [ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_size_t]
    library.dlss_gpu_host_readback_output.restype = ctypes.c_int32
    library.dlss_gpu_host_get_transfer_counters.argtypes = [ctypes.c_void_p,
        ctypes.POINTER(_TransferCounters)]
    library.dlss_gpu_host_get_transfer_counters.restype = ctypes.c_int32
    library.dlss_gpu_host_close.argtypes = [ctypes.c_void_p]
    library.dlss_gpu_host_last_error.restype = ctypes.c_char_p
    library.zero_motion_supported = (abi >= 3 and _bind_zero_motion(library))
    library.host_motion_supported = _bind_host_motion(library)
    library.fp16_color_supported = _bind_fp16_color(library)
    if not hasattr(library, "color_capabilities"):
        library.color_capabilities = None
    return library


class Live:
    """One Feature 18 session; layers may share an NGX core, never a Feature.

    ``color_format`` selects the session's own colour: "rgba8" is the frame the
    host has always taken, and "rgba16f" opens an R16G16B16A16_FLOAT session whose
    input is a float16 HxWx4 plane and whose borrowed output is an RGBA16F
    texture. It is settled once, at open, and every later descriptor, stride and
    readback on this object is checked against it - a session never converts,
    reinterprets or falls back.
    """

    def __init__(self, width, height, settings=None, *, shared_with=None,
                 color_format="rgba8"):
        if color_format not in _COLOR_BYTES:
            raise ValueError(f"unknown color_format {color_format!r}; this host "
                             f"speaks {sorted(_COLOR_BYTES)}")
        self._w, self._h = int(width), int(height)
        self.color_format = color_format
        self.bytes_per_pixel = _COLOR_BYTES[color_format]
        self.settings = dict(settings or {})
        self._lib = _load()
        self._handle = ctypes.c_void_p()
        self.last_error = None
        self.last_process_ms = 0.0
        # Which host file and ABI this session is driving, read off the library that
        # answered rather than off a setting: a run's diagnostics have to name the
        # thing that drew its frames.
        self.host_dll = os.path.basename(_LIBRARY)
        self.host_abi = getattr(self._lib, "abi_version", None)
        if self.bytes_per_pixel != 4 and \
                not getattr(self._lib, "fp16_color_supported", False):
            raise DLSSInferenceError(
                "HDR FP16 colour is not available on the loaded host: it exposes "
                "no format-aware open entry point or no eight-byte row transport "
                f"(capabilities {getattr(self._lib, 'color_capabilities', None)}); "
                "an SDR session is not a substitute")
        if shared_with is not None:
            if not getattr(self._lib, "multi_session_supported", False):
                raise DLSSInferenceError("GPU host has no independent layer sessions")
            if not shared_with.is_alive():
                raise DLSSInferenceError("anchor GPU host session is closed")
        result = self._open(shared_with)
        if result:
            raise DLSSInferenceError(self._open_error(result))
        self.session_id = (int(self._lib.dlss_gpu_host_session_id(self._handle))
                           if getattr(self._lib, "multi_session_supported", False) else 0)

    def _open(self, shared_with):
        """One open call.

        The SDR session keeps the entry point it has always used, so a build that
        predates the format-aware additions still opens exactly the way it did;
        only a session that asks for a colour the legacy entry cannot name goes
        through the _ex form.
        """
        self.runtime_dll = shared_with.runtime_dll if shared_with is not None else resolve_runtime()
        options = self._options()
        if self.bytes_per_pixel == 4:
            if shared_with is None:
                return self._lib.dlss_gpu_host_open(self._w, self._h, self.runtime_dll,
                    ctypes.byref(options), ctypes.byref(self._handle))
            return self._lib.dlss_gpu_host_open_layer(shared_with._handle,
                self._w, self._h, ctypes.byref(options), ctypes.byref(self._handle))
        info = _OpenInfo(ctypes.sizeof(_OpenInfo),
                         _COLOR_FORMAT[self.color_format], 0)
        if shared_with is None:
            return self._lib.dlss_gpu_host_open_ex(self._w, self._h, self.runtime_dll,
                ctypes.byref(options), ctypes.byref(info), ctypes.byref(self._handle))
        return self._lib.dlss_gpu_host_open_layer_ex(shared_with._handle,
            self._w, self._h, ctypes.byref(options), ctypes.byref(info),
            ctypes.byref(self._handle))

    def _options(self):
        return _Options(ctypes.sizeof(_Options), *baked_option_key(self.settings))

    def update_runtime_options(self, settings):
        """Owner-thread update; all fields except preset are read at Evaluate."""
        if int(settings.get('preset', 1)) != int(self.settings.get('preset', 1)):
            raise ValueError('Preset changes require a new Feature 18 session')
        fn = getattr(self._lib, 'dlss_gpu_host_update_options', None)
        if fn is None:
            return False  # An older DLL requires the ordinary rebuild path.
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Options)]
        fn.restype = ctypes.c_int32
        options = _Options(ctypes.sizeof(_Options), *baked_option_key(settings))
        code = fn(self._handle, ctypes.byref(options))
        if code:
            raise RuntimeError(self._error('Update runtime NR options', code))
        self.settings = dict(settings)
        return True

    def _error(self, operation, code):
        detail = self._lib.dlss_gpu_host_last_error()
        return f"{operation} failed ({code}): {detail.decode('utf-8', 'replace') if detail else ''}"

    def _open_error(self, code):
        base = self._error("GPU-flow host open", code)
        note = _OPEN_MESSAGES.get(code)
        return f"{base} ({note})" if note else base

    def is_alive(self):
        return bool(self._handle.value)

    def d3d12_device(self):
        """The session's ID3D12Device pointer, for a bridge that must share it.

        The GPU HDR bridge only accepts a Feature 18 output that belongs to its
        own device, so a chain that wants one opens the host first and hands this
        pointer to it rather than the other way round.
        """
        if not self.is_alive():
            raise DLSSInferenceError("GPU-flow DLSS session is closed")
        if not hasattr(self._lib, "dlss_gpu_host_d3d12_device"):
            raise DLSSInferenceError("loaded host exposes no d3d12_device entry")
        return self._lib.dlss_gpu_host_d3d12_device(self._handle)

    def color_info(self):
        """The session's colour geometry as the host itself reports it."""
        if not self.is_alive():
            raise DLSSInferenceError("GPU-flow DLSS session is closed")
        if not hasattr(self._lib, "dlss_gpu_host_get_color_info"):
            raise DLSSInferenceError("loaded host exposes no colour info entry")
        info = _ColorInfo(ctypes.sizeof(_ColorInfo))
        if self._lib.dlss_gpu_host_get_color_info(self._handle, ctypes.byref(info)):
            raise DLSSInferenceError(self._error("GPU host colour info", 1))
        return {name: getattr(info, name) for name, _ in
                _ColorInfo._fields_ if name != "struct_size"}

    def layer_capability(self):
        """What this session's host would answer to a second session sharing its
        core, stated entry by entry.

        A multi-layer chain asks this instead of reading the ABI number, because the
        two halves of what it needs arrived separately: an independent layer session
        is one export, an FP16 layer session is a second one that only the
        format-aware open carries. A host that has the first and not the second can
        run a dual-layer SDR job and cannot run a dual-layer HDR job, and the answer
        that says so is this dict, not a version.
        """
        library = self._lib
        return {
            "shared_layer_sessions": bool(getattr(library,
                                                  "multi_session_supported", False)),
            "format_aware_layer_open": callable(
                getattr(library, "dlss_gpu_host_open_layer_ex", None)),
            "rgba16f_sessions": bool(getattr(library, "fp16_color_supported", False)),
            "reportable_session_id": bool(
                getattr(library, "multi_session_supported", False)) and
                callable(getattr(library, "dlss_gpu_host_session_id", None)),
            "host_color_capabilities": getattr(library, "color_capabilities", None),
        }

    def _check_color(self, color):
        if not isinstance(color, dict) or \
                color.get("format") != _COLOR_DESCRIPTOR[self.color_format] or \
                (color.get("width"), color.get("height")) != (self._w, self._h) or \
                not color.get("cuda_context") or not color.get("device_ptr") or \
                int(color.get("pitch_bytes", 0)) < self._w * self.bytes_per_pixel:
            raise ValueError(f"color must be a matching "
                f"{self.color_format.upper()} CUDA descriptor")

    def _check_flow(self, flow):
        if not isinstance(flow, dict) or flow.get("format") != "float32x2" or \
                (flow.get("width"), flow.get("height")) != (self._w, self._h) or \
                not flow.get("cuda_context") or not flow.get("device_ptr") or \
                int(flow.get("pitch_bytes", 0)) < self._w * 8:
            raise ValueError("flow must be a matching float32x2 CUDA descriptor")

    def _new_plane(self):
        return np.empty((self._h, self._w, 4), dtype=_host_dtype(self.bytes_per_pixel))

    def _check_plane(self, plane, what):
        if not (isinstance(plane, np.ndarray)
                and plane.dtype == _host_dtype(self.bytes_per_pixel)
                and plane.shape == (self._h, self._w, 4)
                and plane.flags.c_contiguous):
            raise ValueError(f"{what} must be {_plane_description(self.bytes_per_pixel)}")

    def process_device_flow(self, rgba, flow, reset=False, out=None):
        if not self.is_alive():
            raise DLSSInferenceError("GPU-flow DLSS session is closed")
        if not isinstance(flow, dict) or flow.get("format") != "float32x2" or \
                (flow.get("width"), flow.get("height")) != (self._w, self._h):
            raise ValueError("NVOF flow must be a matching float32x2 CUDA descriptor")
        self._check_plane(rgba, "RGBA input")
        if out is None:
            out = np.empty_like(rgba)
        elif out.shape != rgba.shape or out.dtype != rgba.dtype or not out.flags.c_contiguous:
            raise ValueError(f"output must be {_plane_description(self.bytes_per_pixel)}")
        t0 = time.perf_counter()
        result = self._lib.dlss_gpu_host_process(self._handle,
            rgba.ctypes.data, rgba.strides[0], flow["cuda_context"],
            flow["device_ptr"], flow["pitch_bytes"],
            int(bool(reset)), out.ctypes.data, out.strides[0])
        self.last_process_ms = (time.perf_counter() - t0) * 1000.0
        if result:
            self.last_error = self._error("GPU-flow DLSS frame", result)
            raise DLSSInferenceError(self.last_error)
        self.last_error = None
        return out

    def process_device_color_flow(self, color, flow, reset=False, out=None):
        """Evaluate with borrowed CUDA colour and NVOF flow; output returns to host memory."""
        if not self.is_alive():
            raise DLSSInferenceError("GPU-flow DLSS session is closed")
        self._check_color(color)
        self._check_flow(flow)
        if out is None:
            out = self._new_plane()
        else:
            self._check_plane(out, "output")
        t0 = time.perf_counter()
        result = self._lib.dlss_gpu_host_process_device_color(self._handle,
            color["cuda_context"], color["device_ptr"], color["pitch_bytes"],
            flow["cuda_context"], flow["device_ptr"], flow["pitch_bytes"],
            int(bool(reset)), out.ctypes.data, out.strides[0])
        self.last_process_ms = (time.perf_counter() - t0) * 1000.0
        if result:
            self.last_error = self._error("GPU-color DLSS frame", result)
            raise DLSSInferenceError(self.last_error)
        self.last_error = None
        return out

    def transfer_counters(self):
        counters = _TransferCounters()
        counters.struct_size = ctypes.sizeof(_TransferCounters)
        if self._lib.dlss_gpu_host_get_transfer_counters(
                self._handle, ctypes.byref(counters)):
            raise DLSSInferenceError("GPU host transfer counters unavailable")
        return {name: getattr(counters, name) for name, _ in
                _TransferCounters._fields_ if name != "struct_size"}

    def process_device_color_flow_resource(self, color, flow, reset=False):
        """Return borrowed D3D12 pointers, valid until the next process or close."""
        if not self.is_alive():
            raise DLSSInferenceError("GPU-flow DLSS session is closed")
        self._check_color(color)
        self._check_flow(flow)
        output = _DeviceOutput()
        output.struct_size = ctypes.sizeof(_DeviceOutput)
        t0 = time.perf_counter()
        result = self._lib.dlss_gpu_host_process_device_color_resource(
            self._handle, color["cuda_context"], color["device_ptr"],
            color["pitch_bytes"], flow["cuda_context"], flow["device_ptr"],
            flow["pitch_bytes"], int(bool(reset)), ctypes.byref(output))
        self.last_process_ms = (time.perf_counter() - t0) * 1000.0
        if result:
            self.last_error = self._error("GPU-output DLSS frame", result)
            raise DLSSInferenceError(self.last_error)
        self.last_error = None
        return {name: getattr(output, name) for name, _ in
                _DeviceOutput._fields_ if name != "struct_size"}

    def supports_zero_motion(self) -> bool:
        """Whether the loaded host binds a resident zero motion texture.

        An ABI-2 host has no such entry point, and a caller that asked to run
        without flow has to hear that from the gate rather than get a frame drawn
        by some other route it never selected.
        """
        return bool(getattr(self._lib, "zero_motion_supported", False))

    def _require_zero_motion(self):
        if not self.supports_zero_motion():
            raise DLSSInferenceError(
                "dlss_gpu_host.dll exposes no zero-motion entry point (ABI 2); "
                "rebuild native/dlss_gpu_host.dll to run guidance_mode=0 on the "
                "GPU host")

    def process_zero_motion(self, rgba, reset=False, out=None):
        """Host colour, no flow of any kind; the output returns to host memory."""
        if not self.is_alive():
            raise DLSSInferenceError("GPU-flow DLSS session is closed")
        self._require_zero_motion()
        self._check_plane(rgba, "RGBA input")
        if out is None:
            out = np.empty_like(rgba)
        elif out.shape != rgba.shape or out.dtype != rgba.dtype or not out.flags.c_contiguous:
            raise ValueError(f"output must be {_plane_description(self.bytes_per_pixel)}")
        t0 = time.perf_counter()
        result = self._lib.dlss_gpu_host_process_zero_motion(self._handle,
            rgba.ctypes.data, rgba.strides[0], int(bool(reset)),
            out.ctypes.data, out.strides[0])
        self.last_process_ms = (time.perf_counter() - t0) * 1000.0
        if result:
            self.last_error = self._error("zero-motion DLSS frame", result)
            raise DLSSInferenceError(self.last_error)
        self.last_error = None
        return out

    def process_device_color_zero_motion_resource(self, color, reset=False):
        """Borrowed CUDA colour in, borrowed D3D12 output out, nothing for motion.

        The returned pointers follow the same rule as the guided entry: valid
        until the next process call or close.
        """
        if not self.is_alive():
            raise DLSSInferenceError("GPU-flow DLSS session is closed")
        self._require_zero_motion()
        self._check_color(color)
        output = _DeviceOutput()
        output.struct_size = ctypes.sizeof(_DeviceOutput)
        t0 = time.perf_counter()
        result = self._lib.dlss_gpu_host_process_device_color_zero_motion_resource(
            self._handle, color["cuda_context"], color["device_ptr"],
            color["pitch_bytes"], int(bool(reset)), ctypes.byref(output))
        self.last_process_ms = (time.perf_counter() - t0) * 1000.0
        if result:
            self.last_error = self._error("zero-motion DLSS frame", result)
            raise DLSSInferenceError(self.last_error)
        self.last_error = None
        return {name: getattr(output, name) for name, _ in
                _DeviceOutput._fields_ if name != "struct_size"}

    def supports_host_motion(self) -> bool:
        """Whether the loaded host can take a CPU motion plane for guidance.

        A build that predates the entry exposes no such symbol; a caller holding
        host motion has to hear that here rather than have the frame run behind
        its back on the legacy DLL.
        """
        return bool(getattr(self._lib, "host_motion_supported", False))

    def _require_host_motion(self):
        if not self.supports_host_motion():
            raise DLSSInferenceError(
                "dlss_gpu_host.dll exposes no host-motion entry point; rebuild "
                "native/dlss_gpu_host.dll to run a guided host-pointer frame on "
                "the GPU host")

    def process_host_motion(self, rgba, motion, reset=False, out=None):
        """A guided frame whose motion arrives from host memory, not from NVOF.

        ``motion`` is the pixel-space displacement plane in the shape the old
        dlssnr_host.dll ``Live.process`` took: a float32 HxWx2 plane at the DLSS
        input resolution, x fastest, top-left origin, ``reset`` meaningful as
        elsewhere. The card narrows float32 to the bound R16G16_FLOAT resource
        with the same single-rounded compute pass the guided NVOF conversion runs,
        so numpy never enters the conversion. The plane is never dropped or zeroed:
        ``readback_motion`` shows the exact half2 bytes DLSS binds.
        """
        if not self.is_alive():
            raise DLSSInferenceError("GPU-flow DLSS session is closed")
        self._require_host_motion()
        self._check_plane(rgba, "RGBA input")
        if not (isinstance(motion, np.ndarray) and
                motion.shape == (self._h, self._w, 2) and
                motion.dtype == np.float32 and motion.flags.c_contiguous):
            raise ValueError("motion must be a contiguous float32 HxWx2 plane at "
                             "the DLSS input resolution")
        if out is None:
            out = np.empty_like(rgba)
        elif out.shape != rgba.shape or out.dtype != rgba.dtype or not out.flags.c_contiguous:
            raise ValueError(f"output must be {_plane_description(self.bytes_per_pixel)}")
        t0 = time.perf_counter()
        result = self._lib.dlss_gpu_host_process_host_motion(self._handle,
            rgba.ctypes.data, rgba.strides[0], motion.ctypes.data,
            motion.strides[0], int(bool(reset)), out.ctypes.data, out.strides[0])
        self.last_process_ms = (time.perf_counter() - t0) * 1000.0
        if result:
            self.last_error = self._error("host-motion DLSS frame", result)
            raise DLSSInferenceError(self.last_error)
        self.last_error = None
        return out

    def readback_motion(self):
        """The half2 plane currently bound to DLSSNR.MVec, as an HxWx2 view.

        A D2H readback for a correctness gate that wants to prove a host-motion
        frame reached the resource rather than being replaced by zero; it is not
        on any production frame path.
        """
        if not self.is_alive():
            raise DLSSInferenceError("GPU-flow DLSS session is closed")
        self._require_host_motion()
        buffer = np.empty((self._h, self._w, 4), dtype=np.uint8)
        result = self._lib.dlss_gpu_host_readback_motion(
            self._handle, buffer.ctypes.data, buffer.strides[0])
        if result:
            self.last_error = self._error("GPU motion readback", result)
            raise DLSSInferenceError(self.last_error)
        return buffer.view(np.float16).reshape(self._h, self._w, 2)

    def readback_output(self, out=None):
        """Explicit D2H copy for a CPU consumer or a correctness check.

        The plane comes back in the session's own colour: uint8 HxWx4 for an
        RGBA8 session, float16 HxWx4 scRGB for an FP16 one.
        """
        if not self.is_alive():
            raise DLSSInferenceError("GPU-flow DLSS session is closed")
        if out is None:
            out = self._new_plane()
        else:
            self._check_plane(out, "output")
        result = self._lib.dlss_gpu_host_readback_output(
            self._handle, out.ctypes.data, out.strides[0])
        if result:
            self.last_error = self._error("GPU output readback", result)
            raise DLSSInferenceError(self.last_error)
        return out

    def close(self):
        if self._handle.value:
            handle = self._handle
            self._handle = ctypes.c_void_p()
            # A foreign close may fault after deleting the host. Never call it
            # again with that address during outer-chain cleanup.
            self._lib.dlss_gpu_host_close(handle)
