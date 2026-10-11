"""Python wrapper for the native residual hot path (native/residual_bridge.dll).

Nothing here falls back. If the bridge is missing, importing this module raises; if a
call fails, it raises with the message the DLL produced. The only decision about which
backend runs lives in pipeline.create_residual_backend(), and this module reports what
actually loaded so that decision can never be papered over.
"""

import ctypes
import os
import sys

import numpy as np

import bundle_paths

try:
    import pipeline
except ImportError:  # pragma: no cover - direct execution from another cwd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pipeline

_EXPECTED_API_MAJOR = 2
# The minor version that added rr_prepare_working_bgr_rgba8 and its three counters, and the
# one that added rr_reconstruct_from_rgba8 with its two. Older 2.x libraries stay usable for
# the BGR-only paths, so these gate a capability and never a load.
_MINOR_WITH_FUSED_BGR_RGBA = 2
_MINOR_WITH_RECONSTRUCT_RGBA = 3
_CHANNEL_COUNT = 3
_RGBA_CHANNEL_COUNT = 4

# (capability name, export it needs, first minor version that declares it). A capability is
# reported as three facts and one verdict so the reason it is absent survives into the
# diagnostics instead of collapsing into a bare False.
_CAPABILITY_SOURCES = (
    ("prepare_bgr_rgba", "rr_prepare_working_bgr_rgba8", _MINOR_WITH_FUSED_BGR_RGBA),
    ("reconstruct_rgba", "rr_reconstruct_from_rgba8", _MINOR_WITH_RECONSTRUCT_RGBA),
)

_ERROR_NAMES = {
    -1: "RR_ERR_NULL_POINTER",
    -2: "RR_ERR_INVALID_DIMS",
    -3: "RR_ERR_INVALID_STRIDE",
    -4: "RR_ERR_DIM_MISMATCH",
    -5: "RR_ERR_BAD_HANDLE",
    # -6 is not returned by API 2.1 - an allocation failure is reported as -9. It stays in
    # this table so a message from an older DLL still decodes.
    -6: "RR_ERR_NO_SCRATCH",
    -7: "RR_ERR_INTERNAL",
    -8: "RR_ERR_INVALID_STRENGTH",
    -9: "RR_ERR_OUT_OF_MEMORY",
}


class ResidualBridgeError(RuntimeError):
    """A failure reported by residual_bridge. Never swallowed."""


class _Counters(ctypes.Structure):
    """Mirrors rr_counters in native/residual_bridge.h.

    The first twelve fields describe the plan whose handle was passed in and nothing else;
    the five process_* fields are totals for the whole library instance; the last five are
    per-plan again and were appended in ABI 2.2 and 2.3, which is why they sit after the
    process totals rather than beside the other plan fields. Order and types must match the
    header exactly - _EXPECTED_API_MAJOR guards against a stale DLL whose layout differs, but
    a mismatch here is a silent misread, not an error.
    test_residual_bridge_failure_paths.py compares this list against the header text so the
    two cannot drift unnoticed.
    """

    _fields_ = [
        ("plan_create_count", ctypes.c_int64),
        ("plan_destroy_count", ctypes.c_int64),
        ("scratch_alloc_count", ctypes.c_int64),
        ("scratch_alloc_bytes", ctypes.c_int64),
        ("downsample_call_count", ctypes.c_int64),
        ("reconstruct_call_count", ctypes.c_int64),
        ("downsample_pixel_count", ctypes.c_int64),
        ("reconstruct_pixel_count", ctypes.c_int64),
        ("worker_create_count", ctypes.c_int64),
        ("worker_count", ctypes.c_int64),
        ("scratch_bytes", ctypes.c_int64),
        ("axis_plan_count", ctypes.c_int64),
        ("process_plan_create_total", ctypes.c_int64),
        ("process_plan_destroy_total", ctypes.c_int64),
        ("process_scratch_alloc_total", ctypes.c_int64),
        ("process_scratch_bytes_total", ctypes.c_int64),
        ("process_worker_create_total", ctypes.c_int64),
        ("prepare_bgr_only_call_count", ctypes.c_int64),
        ("prepare_bgr_rgba_call_count", ctypes.c_int64),
        ("fused_rgba_store_bytes", ctypes.c_int64),
        ("reconstruct_bgr_only_call_count", ctypes.c_int64),
        ("reconstruct_rgba_input_call_count", ctypes.c_int64),
    ]


def _library_path():
    return bundle_paths.native_dll(
        "residual_bridge.dll" if os.name == "nt" else "libresidual_bridge.so")


def _load_library():
    path = _library_path()
    if not os.path.exists(path):
        raise ResidualBridgeError(str(bundle_paths.missing_binary(
            path, "run the CPU residual backend")))
    if os.name == "nt":
        bundle_paths.add_dll_directory(os.path.dirname(path))
    lib = ctypes.CDLL(path)

    lib.rr_get_api_version.argtypes = []
    lib.rr_get_api_version.restype = ctypes.c_int32
    lib.rr_get_last_error.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.rr_get_last_error.restype = ctypes.c_int32
    lib.rr_cpu_has_avx2.argtypes = []
    lib.rr_cpu_has_avx2.restype = ctypes.c_int32
    lib.rr_create_plan.argtypes = [ctypes.POINTER(ctypes.c_int64),
                                   ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    lib.rr_create_plan.restype = ctypes.c_int32
    lib.rr_destroy_plan.argtypes = [ctypes.c_int64]
    lib.rr_destroy_plan.restype = ctypes.c_int32
    i32, i64, f32, pu8 = (ctypes.c_int32, ctypes.c_int64, ctypes.c_float,
                          ctypes.POINTER(ctypes.c_uint8))
    lib.rr_downsample_bgr8.argtypes = [i64, pu8, i32, i32, i64, pu8, i32, i32, i64]
    lib.rr_downsample_bgr8.restype = i32
    lib.rr_reconstruct_bgr8.argtypes = [i64,
                                        pu8, i32, i32, i64,
                                        pu8, i32, i32, i64,
                                        pu8, i32, i32, i64,
                                        f32,
                                        pu8, i32, i32, i64]
    lib.rr_reconstruct_bgr8.restype = i32
    lib.rr_get_counters.argtypes = [i64, ctypes.POINTER(_Counters)]
    lib.rr_get_counters.restype = i32
    # Conditionally, so a 2.1 library beside this module still loads and runs the BGR-only
    # paths. Setting argtypes on a symbol that is not there would raise here and take the
    # whole backend down for a capability the caller may not even want.
    if hasattr(lib, "rr_prepare_working_bgr_rgba8"):
        lib.rr_prepare_working_bgr_rgba8.argtypes = [i64, pu8, i32, i32, i64,
                                                     pu8, i32, i32, i64,
                                                     pu8, i64]
        lib.rr_prepare_working_bgr_rgba8.restype = i32
    if hasattr(lib, "rr_reconstruct_from_rgba8"):
        lib.rr_reconstruct_from_rgba8.argtypes = [i64,
                                                  pu8, i32, i32, i64,
                                                  pu8, i32, i32, i64,
                                                  pu8, i32, i32, i64,
                                                  f32,
                                                  pu8, i32, i32, i64]
        lib.rr_reconstruct_from_rgba8.restype = i32
    return lib


_LIB = None
_CAPABILITIES = None


def _probe_capabilities(lib, version):
    """What this library instance can do, read off its own exports.

    The version number is not enough on its own in either direction: it says what the build
    intended, the symbol says what the binary in front of us offers. Both are reported, and
    a capability needs both, so a mismatch shows up as an absent capability with a reason
    rather than as an AttributeError in the middle of a frame.
    """
    minor = version & 0xFFFF
    caps = {"api_version": version, "api_minor": minor}
    for capability, symbol, since in _CAPABILITY_SOURCES:
        caps[f"{capability}_export"] = hasattr(lib, symbol)
        caps[f"{capability}_minor"] = minor >= since
        caps[capability] = caps[f"{capability}_export"] and caps[f"{capability}_minor"]
    return caps


def capabilities():
    """The ABI facts of the loaded bridge, as the DLL itself reports them."""
    _library()
    return dict(_CAPABILITIES)


def _library():
    global _LIB, _CAPABILITIES
    if _LIB is None:
        lib = _load_library()
        version = lib.rr_get_api_version()
        if (version >> 16) != _EXPECTED_API_MAJOR:
            raise ResidualBridgeError(
                f"residual_bridge api version 0x{version:08x} is not supported "
                f"(expected major {_EXPECTED_API_MAJOR})")
        # Probed before the handle is published, so nothing can call a half-described
        # library: capabilities() and every call site read the same snapshot.
        _CAPABILITIES = _probe_capabilities(lib, version)
        _LIB = lib
    return _LIB


def api_version():
    return _library().rr_get_api_version()


def avx2_available():
    return bool(_library().rr_cpu_has_avx2())


def is_available():
    """True only when the DLL loads, the ABI matches and a plan really builds."""
    try:
        backend = NativeResidualBackend(16, 16, 8, 8)
    except Exception:
        return False
    backend.close()
    return True


def _describe(array, name, channels=_CHANNEL_COUNT):
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != channels:
        raise ResidualBridgeError(
            f"{name} must be uint8 HxW x{channels}, got {array.dtype} shape {array.shape}")
    if not array.flags["C_CONTIGUOUS"]:
        raise ResidualBridgeError(
            f"{name} must be C-contiguous; row stride {array.strides[0]} != "
            f"{array.shape[2] * array.itemsize * array.shape[1]}. Copy it instead of "
            "passing a view with an unexpected stride.")


class NativeResidualBackend(pipeline.ResidualBackend):
    """Same contract as the Python reference, executed by residual_bridge."""

    name = "cpu_reference_native"

    def __init__(self, src_w: int, src_h: int, live_w: int, live_h: int):
        self._lib = _library()
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.live_w = int(live_w)
        self.live_h = int(live_h)
        self._handle = ctypes.c_int64(0)
        self._closed = True
        code = self._lib.rr_create_plan(ctypes.byref(self._handle),
                                        self.src_w, self.src_h, self.live_w, self.live_h)
        if code != 0:
            raise ResidualBridgeError(f"rr_create_plan failed: {self._error(code)}")
        self._closed = False
        self.version = f"api-0x{api_version():08x}"
        self.downsample_call_count = 0
        self.reconstruct_call_count = 0
        self.fused_prepare_call_count = 0
        self.rgba_reconstruct_call_count = 0
        # Read once, from the loaded DLL, and carried on the backend: a caller deciding
        # which path to run asks this rather than re-deriving it from a version string.
        self.capabilities = dict(_CAPABILITIES)

    def _error(self, code):
        buf = ctypes.create_string_buffer(512)
        self._lib.rr_get_last_error(buf, ctypes.c_int32(len(buf)))
        message = buf.value.decode("utf-8", "replace")
        return f"{_ERROR_NAMES.get(code, code)}: {message}" if message else str(code)

    def _check(self, code, what):
        if code != 0:
            raise ResidualBridgeError(f"{what} failed: {self._error(code)}")

    def _require(self, capability, symbol):
        """Refuse a call the loaded bridge cannot serve, and say which of the two facts said
        no. The same three numbers are in the diagnostics, so a report can show why a frame
        took the older path without anyone re-deriving it."""
        if not _CAPABILITIES[capability]:
            raise ResidualBridgeError(
                f"{symbol} is not available: api 0x{_CAPABILITIES['api_version']:08x}, "
                f"export {_CAPABILITIES[f'{capability}_export']}, "
                f"minor {_CAPABILITIES[f'{capability}_minor']}")

    def _u8(self, array, name, channels=_CHANNEL_COUNT):
        _describe(array, name, channels)
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    def prepare_working_color(self, source_bgr, dst_bgr):
        if self._closed:
            raise ResidualBridgeError("backend already closed")
        src_ptr = self._u8(source_bgr, "source_bgr")
        dst_ptr = self._u8(dst_bgr, "dst_bgr")
        code = self._lib.rr_downsample_bgr8(
            self._handle, src_ptr, source_bgr.shape[1], source_bgr.shape[0],
            ctypes.c_int64(source_bgr.strides[0]),
            dst_ptr, dst_bgr.shape[1], dst_bgr.shape[0], ctypes.c_int64(dst_bgr.strides[0]))
        self._check(code, "rr_downsample_bgr8")
        self.downsample_call_count += 1
        return dst_bgr

    def prepare_working_color_rgba(self, source_bgr, dst_bgr, dst_rgba):
        """One Lanczos pass, both channel orders, no conversion pass afterwards.

        Raises when the bridge cannot do it. There is deliberately no fallback in here: a
        caller that asked for the fused store and silently got the two-pass path would keep
        the round trip it was trying to delete, and the counters would say so while the
        code looked reduced.
        """
        if self._closed:
            raise ResidualBridgeError("backend already closed")
        self._require("prepare_bgr_rgba", "rr_prepare_working_bgr_rgba8")
        src_ptr = self._u8(source_bgr, "source_bgr")
        bgr_ptr = self._u8(dst_bgr, "dst_bgr")
        rgba_ptr = self._u8(dst_rgba, "dst_rgba", _RGBA_CHANNEL_COUNT)
        if dst_rgba.shape[:2] != dst_bgr.shape[:2]:
            raise ResidualBridgeError(
                f"dst_rgba is {dst_rgba.shape[:2]} but dst_bgr is {dst_bgr.shape[:2]}; the "
                "fused store writes both at the working resolution")
        code = self._lib.rr_prepare_working_bgr_rgba8(
            self._handle, src_ptr, source_bgr.shape[1], source_bgr.shape[0],
            ctypes.c_int64(source_bgr.strides[0]),
            bgr_ptr, dst_bgr.shape[1], dst_bgr.shape[0], ctypes.c_int64(dst_bgr.strides[0]),
            rgba_ptr, ctypes.c_int64(dst_rgba.strides[0]))
        self._check(code, "rr_prepare_working_bgr_rgba8")
        self.downsample_call_count += 1
        self.fused_prepare_call_count += 1
        return dst_bgr, dst_rgba

    def reconstruct(self, source_full_bgr, source_low_bgr, dlss_low_bgr, strength, dst_bgr):
        if self._closed:
            raise ResidualBridgeError("backend already closed")
        sf = self._u8(source_full_bgr, "source_full_bgr")
        sl = self._u8(source_low_bgr, "source_low_bgr")
        dl = self._u8(dlss_low_bgr, "dlss_low_bgr")
        out = self._u8(dst_bgr, "dst_bgr")
        code = self._lib.rr_reconstruct_bgr8(
            self._handle,
            sf, source_full_bgr.shape[1], source_full_bgr.shape[0],
            ctypes.c_int64(source_full_bgr.strides[0]),
            sl, source_low_bgr.shape[1], source_low_bgr.shape[0],
            ctypes.c_int64(source_low_bgr.strides[0]),
            dl, dlss_low_bgr.shape[1], dlss_low_bgr.shape[0],
            ctypes.c_int64(dlss_low_bgr.strides[0]),
            ctypes.c_float(float(strength)),
            out, dst_bgr.shape[1], dst_bgr.shape[0], ctypes.c_int64(dst_bgr.strides[0]))
        self._check(code, "rr_reconstruct_bgr8")
        self.reconstruct_call_count += 1
        return dst_bgr

    def reconstruct_from_rgba(self, source_full_bgr, source_low_bgr, dlss_low_rgba,
                              strength, dst_bgr):
        """The composite with the DLSS output read in the RGBA8 the host wrote.

        Raises when the bridge cannot do it, for the same reason prepare_working_color_rgba
        does: a caller that asked to read RGBA directly and quietly got a converted copy kept
        the round trip this phase exists to delete, and it is worse to discover that in the
        output than in an exception.
        """
        if self._closed:
            raise ResidualBridgeError("backend already closed")
        self._require("reconstruct_rgba", "rr_reconstruct_from_rgba8")
        sf = self._u8(source_full_bgr, "source_full_bgr")
        sl = self._u8(source_low_bgr, "source_low_bgr")
        dl = self._u8(dlss_low_rgba, "dlss_low_rgba", _RGBA_CHANNEL_COUNT)
        out = self._u8(dst_bgr, "dst_bgr")
        code = self._lib.rr_reconstruct_from_rgba8(
            self._handle,
            sf, source_full_bgr.shape[1], source_full_bgr.shape[0],
            ctypes.c_int64(source_full_bgr.strides[0]),
            sl, source_low_bgr.shape[1], source_low_bgr.shape[0],
            ctypes.c_int64(source_low_bgr.strides[0]),
            dl, dlss_low_rgba.shape[1], dlss_low_rgba.shape[0],
            ctypes.c_int64(dlss_low_rgba.strides[0]),
            ctypes.c_float(float(strength)),
            out, dst_bgr.shape[1], dst_bgr.shape[0], ctypes.c_int64(dst_bgr.strides[0]))
        self._check(code, "rr_reconstruct_from_rgba8")
        self.reconstruct_call_count += 1
        self.rgba_reconstruct_call_count += 1
        return dst_bgr

    def close(self):
        if self._closed:
            return
        code = self._lib.rr_destroy_plan(self._handle)
        self._closed = True
        if code != 0:
            raise ResidualBridgeError(f"rr_destroy_plan failed: {self._error(code)}")
        self._handle = ctypes.c_int64(0)

    def backend_info(self):
        return {"residual_backend": self.name, "residual_backend_version": self.version,
                "source_width": self.src_w, "source_height": self.src_h,
                "working_width": self.live_w, "working_height": self.live_h,
                "bridge_library": _library_path(), "avx2": avx2_available(),
                "prepare_bgr_rgba_capable": self.capabilities["prepare_bgr_rgba"],
                "reconstruct_rgba_capable": self.capabilities["reconstruct_rgba"]}

    def diagnostics(self):
        """Lifecycle and work counters, in two strictly separated scopes.

        Every this_plan_* value is about this plan only, so another backend's plan or the
        probe in is_available() cannot move it. The process_* values are the library-wide
        totals, which is where those plans do belong.
        """
        counters = _Counters()
        code = self._lib.rr_get_counters(self._handle, ctypes.byref(counters))
        self._check(code, "rr_get_counters")
        return {"residual_backend": self.name,
                "residual_backend_version": self.version,
                "this_plan_create_count": counters.plan_create_count,
                "this_plan_destroy_count": counters.plan_destroy_count,
                "this_plan_scratch_alloc_count": counters.scratch_alloc_count,
                "this_plan_scratch_alloc_bytes": counters.scratch_alloc_bytes,
                "this_plan_worker_create_count": counters.worker_create_count,
                "this_plan_worker_count": counters.worker_count,
                "native_scratch_bytes": counters.scratch_bytes,
                "axis_plan_count": counters.axis_plan_count,
                "downsample_call_count": counters.downsample_call_count,
                "reconstruct_call_count": counters.reconstruct_call_count,
                "downsample_pixel_count": counters.downsample_pixel_count,
                "reconstruct_pixel_count": counters.reconstruct_pixel_count,
                # Which of the two preparation calls those downsamples were. Compared with
                # downsample_call_count these three say, from inside the library, whether a
                # frame got one fused store or a store plus a separate conversion.
                "prepare_bgr_only_call_count": counters.prepare_bgr_only_call_count,
                "prepare_bgr_rgba_call_count": counters.prepare_bgr_rgba_call_count,
                "fused_rgba_store_bytes": counters.fused_rgba_store_bytes,
                "py_fused_prepare_call_count": self.fused_prepare_call_count,
                "prepare_bgr_rgba_capable": self.capabilities["prepare_bgr_rgba"],
                # The same pair on the reconstruct side: which of the two entry points turned
                # the DLSS output into this frame, as the library counted it.
                "reconstruct_bgr_only_call_count": counters.reconstruct_bgr_only_call_count,
                "reconstruct_rgba_input_call_count": counters.reconstruct_rgba_input_call_count,
                "py_rgba_reconstruct_call_count": self.rgba_reconstruct_call_count,
                "reconstruct_rgba_capable": self.capabilities["reconstruct_rgba"],
                "process_plan_create_total": counters.process_plan_create_total,
                "process_plan_destroy_total": counters.process_plan_destroy_total,
                "process_scratch_alloc_total": counters.process_scratch_alloc_total,
                "process_scratch_bytes_total": counters.process_scratch_bytes_total,
                "process_worker_create_total": counters.process_worker_create_total,
                "process_plan_open_total": (counters.process_plan_create_total
                                            - counters.process_plan_destroy_total),
                "closed": self._closed}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
