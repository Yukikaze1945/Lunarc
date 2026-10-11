"""CPU-only gate for the GPU host's host-motion wrapper (dlss_gpu_engine.Live).

The real plane-reaching-the-texture proof is the hardware smoke
(native/dlss_gpu_host_motion_smoke.py). This file pins the parts of the contract
that do not need a card, so a machine with no GPU still catches a regression that
would otherwise show up only as a subtly wrong frame:

* the caller's float32 motion is handed to the native entry UNCAST. The whole point
  of the compute-pass design is that numpy's float32->float16 double-rounding never
  enters the loop; a wrapper that quietly casts to float16 first would pass the GPU
  gate on integer fixtures and drift on real sub-pixel motion, so this asserts the
  dtype and row stride that actually reach the DLL;
* a malformed plane (wrong extent, wrong channel count, wrong dtype, non-contiguous,
  wrong RGBA) is refused in Python before a single native call is made;
* a host that predates the entry point is refused with a reason instead of running
  the frame on zeroed motion behind the caller's back.
"""

import ctypes
import os
import sys
import types

import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UNIT_DIR)
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dlss_gpu_engine as engine  # noqa: E402

W, H = 8, 4


class FakeNative:
    """The two host-motion exports, recording the arguments a frame reaches them with."""

    def __init__(self, host_motion_supported=True, result=0):
        self.host_motion_supported = host_motion_supported
        self.result = result
        self.calls = []
        if host_motion_supported:
            self.dlss_gpu_host_process_host_motion = self._process
            self.dlss_gpu_host_readback_motion = self._readback

    def _process(self, handle, rgba, rgba_stride, motion, motion_stride,
                 reset, out, out_stride):
        self.calls.append({"rgba": rgba, "rgba_stride": rgba_stride,
                           "motion": motion, "motion_stride": motion_stride,
                           "reset": reset, "out": out, "out_stride": out_stride})
        return self.result

    def _readback(self, handle, dst, dst_stride):
        return self.result

    def dlss_gpu_host_last_error(self):
        return b"fake host error"


def _live(native, color_format="rgba8"):
    """A Live whose session is already open, wired to a fake DLL instead of a card.

    ``__new__`` skips ``__init__``, so the fake has to carry the two attributes the
    real constructor sets and the plane checks read: ``color_format`` and the
    ``bytes_per_pixel`` derived from it. They come from the production mapping
    (``_COLOR_BYTES``: rgba8 -> 4, rgba16f -> 8), not a hard-coded 4, so a session
    that binds an eight-byte colour row is checked against eight bytes here exactly
    as it is on a card. The fake must not be the reason production weakens its
    format validation.
    """
    live = engine.Live.__new__(engine.Live)
    live._w, live._h = W, H
    live.color_format = color_format
    live.bytes_per_pixel = engine._COLOR_BYTES[color_format]
    live._lib = native
    live._handle = ctypes.c_void_p(0xABCD)
    live.settings = {}
    live.last_error = None
    live.last_process_ms = 0.0
    return live


def _plane():
    rng = np.random.default_rng(1)
    return rng.uniform(-1.5, 1.5, size=(H, W, 2)).astype(np.float32)


def _rgba(dtype=np.uint8):
    return np.arange(H * W * 4, dtype=dtype).reshape(H, W, 4)


def test_float32_motion_reaches_the_native_entry_uncast():
    native = FakeNative()
    live = _live(native)
    plane = _plane()
    rgba = np.ascontiguousarray(_rgba())
    out = live.process_host_motion(rgba, plane, reset=True)

    assert out.shape == (H, W, 4) and out.dtype == np.uint8
    assert len(native.calls) == 1, native.calls
    call = native.calls[0]
    assert call["motion"] == plane.ctypes.data, "the wrapper handed a different buffer"
    assert call["motion_stride"] == plane.strides[0] == W * 8, (
        "a float32x2 row is width*8 bytes; a smaller pitch means the wrapper narrowed "
        "the plane before the card could run the single-rounded compute conversion")
    assert call["reset"] == 1
    assert native.host_motion_supported


def test_wrapper_does_not_widen_or_narrow_the_dtype_it_passes():
    # A float16 plane is the legacy caller's mistake, not the contract: the native
    # entry reads float32x2 rows, so a half plane must be refused rather than passed
    # at a stride the compute would misread.
    native = FakeNative()
    live = _live(native)
    rgba = np.ascontiguousarray(_rgba())
    with pytest.raises(ValueError):
        live.process_host_motion(rgba, _plane().astype(np.float16), reset=False)
    assert native.calls == [], "a float16 plane must not reach the native entry"


@pytest.mark.parametrize("color_format,main_dtype", [
    ("rgba8", np.uint8),
    ("rgba16f", np.float16),
])
def test_main_plane_stride_follows_session_format_and_motion_stays_float32x2(
        color_format, main_dtype):
    # The colour plane a session binds is HxWx4 in that session's own dtype, so the
    # row it hands the DLL is width*bytes_per_pixel: 4 bytes/px for RGBA8, 8 for
    # FP16. That stride is independent of the motion contract, which stays a
    # float32x2 plane (width*8 bytes) passed uncast in every format — the whole
    # reason an FP16 colour row and an FP32 motion row can both be width*8 without
    # one implying the other.
    native = FakeNative()
    live = _live(native, color_format)
    rgba = np.ascontiguousarray(_rgba(main_dtype))
    plane = _plane()
    out = live.process_host_motion(rgba, plane, reset=True)

    assert out.dtype == main_dtype and out.shape == (H, W, 4)
    assert len(native.calls) == 1, native.calls
    call = native.calls[0]
    assert call["rgba_stride"] == rgba.strides[0] == W * live.bytes_per_pixel
    assert call["motion"] == plane.ctypes.data
    assert call["motion_stride"] == plane.strides[0] == W * 8


def test_a_session_refuses_a_main_plane_of_the_other_colour_format():
    # Handing an FP16 session uint8 bytes, or an RGBA8 session a half plane, is a
    # dtype mismatch caught in Python: the wrong row width must never reach the DLL.
    native = FakeNative()
    with pytest.raises(ValueError):
        _live(native, "rgba16f").process_host_motion(
            np.ascontiguousarray(_rgba(np.uint8)), _plane())
    with pytest.raises(ValueError):
        _live(native, "rgba8").process_host_motion(
            np.ascontiguousarray(_rgba(np.float16)), _plane())
    assert native.calls == []


@pytest.mark.parametrize("bad", [
    np.zeros((H // 2, W, 2), np.float32),          # wrong extent
    np.zeros((H, W, 3), np.float32),               # not a 2-vector plane
    np.zeros((H, W), np.float32),                  # not 3-dimensional
    np.zeros((H, W, 2), np.float64),               # wrong dtype
    np.zeros((H, W, 4), np.float32)[:, :, :2],     # non-contiguous slice
])
def test_malformed_motion_is_refused_before_any_frame(bad):
    native = FakeNative()
    live = _live(native)
    with pytest.raises(ValueError):
        live.process_host_motion(np.ascontiguousarray(_rgba()), bad, reset=False)
    assert native.calls == [], "a malformed plane was handed to the host"


def test_non_contiguous_rgba_is_refused():
    native = FakeNative()
    live = _live(native)
    strided = np.zeros((H, W + 1, 4), np.uint8)[:, :W]
    with pytest.raises(ValueError):
        live.process_host_motion(strided, _plane(), reset=False)
    assert native.calls == []


def test_old_host_without_the_symbol_is_refused_with_a_reason():
    native = FakeNative(host_motion_supported=False)
    live = _live(native)
    assert live.supports_host_motion() is False
    with pytest.raises(engine.DLSSInferenceError) as raised:
        live.process_host_motion(np.ascontiguousarray(_rgba()), _plane(), reset=False)
    assert "host-motion" in str(raised.value).lower()
    with pytest.raises(engine.DLSSInferenceError):
        live.readback_motion()


def test_native_failure_is_surfaced_not_swallowed():
    native = FakeNative(result=7)
    live = _live(native)
    with pytest.raises(engine.DLSSInferenceError):
        live.process_host_motion(np.ascontiguousarray(_rgba()), _plane(), reset=True)
    assert live.last_error and "fake host error" in live.last_error


def test_bind_host_motion_reports_presence():
    missing = types.SimpleNamespace()  # a host that exposes neither new export
    assert engine._bind_host_motion(missing) is False
    present = types.SimpleNamespace(
        dlss_gpu_host_process_host_motion=types.SimpleNamespace(),
        dlss_gpu_host_readback_motion=types.SimpleNamespace())
    assert engine._bind_host_motion(present) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
