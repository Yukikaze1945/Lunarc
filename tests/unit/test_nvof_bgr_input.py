"""Gate: NVOF can be handed BGR24 host frames, and the channel reorder happens off the CPU.

The path this measures replaces `BGR on the CPU -> cv2.cvtColor(BGR2RGBA) -> upload 4 bytes
per pixel` with `BGR on the CPU -> upload 3 bytes per pixel -> byte-map kernel on the GPU`.
Three claims have to hold together, and none of them follows from the previous one:

  * the bytes that reach NVOF are the bytes the CPU conversion would have produced,
  * the flow that comes back is the flow the RGBA path produced, from the same session
    settings and the same frame history,
  * the transfer counters say 3 bytes per pixel crossed PCIe, one frame per steady frame, and
    nothing was allocated per frame - which is the point of the change.

Every byte figure asserted here is read out of `nvof_bridge_get_counters` and compared against
arithmetic on the frame size. The direction matters: a ledger may only publish what a counter
measured, and a test may only check a counter against something independent of the counter, so
the expected numbers below come from width and height and the actual ones never do.

Two skips are allowed, and only these: the probe says there is no NVOF hardware, or the C++
toolchain is not on this machine so the test build cannot exist. A present GPU whose bridge
lacks the BGR entry point is a failure, not a skip - that is the shipped binary's contract.
"""

import ctypes
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nvof_timing_semantics as timing  # noqa: E402
import optical_flow as of  # noqa: E402
import transfer_profiler as tp  # noqa: E402

PRESENT, PROBE = timing.hardware_present()
pytestmark = pytest.mark.skipif(
    not PRESENT,
    reason=f"no NVIDIA optical flow hardware to feed (probe said {PROBE})")

NATIVE_DIR = os.path.join(_ROOT, "native")
BUILD_SCRIPT = os.path.join(NATIVE_DIR, "build_nvof_test.bat")
SOURCE = os.path.join(NATIVE_DIR, "nvof_bridge.cpp")
PRODUCTION_DLL = os.path.join(NATIVE_DIR, "nvof_bridge.dll")
TEST_DLL = os.path.join(_ROOT, "scratch", "native_test", "nvof_bridge_test.dll")

# The working extent this round's byte budget is stated at: 50% of the 3840x2160 fixture.
GATE_W, GATE_H = 1920, 1080
BGR_BYTES_PER_PIXEL = 3
FLOW_BYTES_PER_PIXEL = 8  # float32 x2, unchanged by this round
RESET_H2D_BYTES = GATE_W * GATE_H * BGR_BYTES_PER_PIXEL * 2
STEADY_H2D_BYTES = GATE_W * GATE_H * BGR_BYTES_PER_PIXEL
FLOW_D2H_BYTES = GATE_W * GATE_H * FLOW_BYTES_PER_PIXEL

SMALL_W, SMALL_H = 640, 360
QUALITY = "4M"


def _require_bgr_input():
    probe = of.probe_nvof_capabilities()
    assert probe.get("available"), f"NVOF hardware went away mid-run: {probe.get('error')}"
    assert probe.get("bgr_input"), (
        "the loaded nvof_bridge exports no nvof_bridge_execute_bgr_sequential, so the BGR "
        "input path cannot run; rebuild native/nvof_bridge.dll from this checkout's source")
    return probe


def _frames(width, height, count=6):
    """A translated, noisy sequence: real motion for the hardware, deterministic to build."""
    rng = np.random.default_rng(11)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    frames = []
    for index in range(count):
        # Rolling by whole pixels gives NVOF a displacement it can resolve exactly instead of
        # a gradient it has to guess at, which is what a parity comparison needs.
        shifted = np.roll(np.roll(base, 3 * index + 1, axis=1), index, axis=0)
        noise = rng.integers(0, 24, size=shifted.shape, dtype=np.uint16)
        frames.append(np.clip(shifted.astype(np.uint16) + noise, 0, 255).astype(np.uint8))
    return frames


def _baseline(engine, width, height):
    """Open the session, then set the counter baseline, so the next delta is one frame's.

    Without this the first snapshot comes from a session that does not exist yet, where the
    staging counters have nothing to report. A delta taken against that reads None for a field
    the frame did update, which is the ambiguity this round is not allowed to create.
    """
    engine._ensure_session(width, height)
    engine.get_counters_delta()


@pytest.fixture()
def engines():
    """A factory handing out real NVOF engines on independent sessions, closed afterwards."""
    _require_bgr_input()
    made = []

    def make():
        engine = of.NvOFHardwareEngine(quality=QUALITY)
        made.append(engine)
        return engine

    yield make
    for engine in made:
        engine.close()


# ---------------------------------------------------------------------------
# Parity with the RGBA input path
# ---------------------------------------------------------------------------

def test_nvof_bgr_input_matches_legacy_rgba_input(engines):
    """Same frames, same session settings, same reset schedule: same flow, bit for bit.

    Two independent sessions, because a shared one would let whichever path ran first leave
    history behind for the other. Bit-exactness is the requirement rather than an agreement
    inside a tolerance: the byte map is a permutation of the same bytes, so anything other
    than equality would mean the two sessions were not shown the same pixels.
    """
    bgr_engine, legacy_engine = engines(), engines()
    frames = _frames(SMALL_W, SMALL_H, count=6)
    for index, cur in enumerate(frames):
        prev = None if index == 0 else frames[index - 1]
        reset = index == 0
        legacy = legacy_engine.compute_flow_sequential(
            cur_bgr=cur, prev_bgr=prev, reset_temporal_hints=reset).copy()
        bgr = bgr_engine.compute_flow_bgr24(prev, cur, reset_temporal_hints=reset).copy()
        assert bgr.dtype == np.float32 and bgr.shape == (SMALL_H, SMALL_W, 2)
        assert np.array_equal(legacy, bgr), (
            f"frame {index}: BGR input and RGBA input produced different flow fields (max "
            f"absolute difference {float(np.abs(legacy - bgr).max())} px), so the bytes "
            f"entering the hardware are not the same bytes")
    # An all-zero field would satisfy that equality for the wrong reason.
    assert float(np.abs(bgr).mean()) > 0.0, (
        "the sequence produced no motion at all, so the parity claim proved nothing")


def test_nvof_bgr_has_no_cpu_previous_frame_copy(engines):
    """The nvof_input conversion site reads a measured zero, not a deleted assertion.

    The counter lives in the seam that performs the conversion, so it stops moving only when
    the call goes away. The RGBA path runs in the same test as the control: without it, a
    broken counter would look exactly like a removed copy.
    """
    bgr_engine, legacy_engine = engines(), engines()
    frames = _frames(SMALL_W, SMALL_H, count=4)

    before = tp.conversion_snapshot()
    for index, cur in enumerate(frames):
        prev = None if index == 0 else frames[index - 1]
        bgr_engine.compute_flow_bgr24(prev, cur, reset_temporal_hints=index == 0)
    bgr_deltas, middle = tp.conversion_deltas(before)
    assert bgr_deltas["cpu_bgr_to_rgba@nvof_input_calls"] == 0
    assert bgr_deltas["cpu_bgr_to_rgba@nvof_input_output_bytes"] == 0
    assert bgr_deltas["cpu_bgr_to_rgba_calls"] == 0
    reported = bgr_engine.get_last_timing()
    assert reported["nvof_input_format"] == "bgr24_host"
    # The two zeros in the reported dict describe the same absence the seam just measured.
    assert reported["cur_bgr_to_rgba_ms"] == 0.0
    assert reported["prev_bgr_to_rgba_ms"] == 0.0
    assert bgr_engine.get_counters()["bgr_input_call_count"] == len(frames)

    for index, cur in enumerate(frames):
        prev = None if index == 0 else frames[index - 1]
        legacy_engine.compute_flow_sequential(cur_bgr=cur, prev_bgr=prev,
                                             reset_temporal_hints=index == 0)
    legacy_deltas, _ = tp.conversion_deltas(middle)
    assert legacy_deltas["cpu_bgr_to_rgba@nvof_input_calls"] > 0, (
        "the RGBA path made no conversions at the nvof_input site either, so the counter is "
        "not observing what it claims to observe")
    assert legacy_deltas["cpu_bgr_to_rgba@nvof_input_output_bytes"] > 0


# ---------------------------------------------------------------------------
# Transfer counters
# ---------------------------------------------------------------------------

def test_nvof_device_output_skips_flow_download(engines):
    """The new device entry point executes NVOF but makes no flow D2H copy."""
    assert of.probe_nvof_capabilities().get("device_output"), (
        "the native bridge must export nvof_bridge_execute_bgr_sequential_device")
    engine = engines()
    width, height = 64, 64
    prev, cur = _frames(width, height, count=2)
    _baseline(engine, width, height)

    resource = engine.compute_flow_bgr24_device(prev, cur, reset_temporal_hints=True)
    delta = engine.get_counters_delta()
    assert resource["device_ptr"] > 0
    assert resource["pitch_bytes"] >= width * FLOW_BYTES_PER_PIXEL
    assert resource["cuda_context"] > 0
    assert resource["width"] == width and resource["height"] == height
    assert delta["nvof_d2h_copy_count_delta"] == 0
    assert delta["nvof_d2h_bytes_delta"] == 0
    assert delta["nvof_h2d_copy_count_delta"] == 2
    assert engine.get_last_timing()["nvof_download_ms"] == "unavailable"

    # Independent test-only CUDA readback checks the device result's actual bytes.
    # It bypasses the bridge, so its copy does not change the production counters.
    cuda = ctypes.WinDLL("nvcuda.dll")
    cuda.cuCtxPushCurrent_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxPushCurrent_v2.restype = ctypes.c_int
    cuda.cuCtxPopCurrent_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cuda.cuCtxPopCurrent_v2.restype = ctypes.c_int
    cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    cuda.cuMemcpyDtoH_v2.restype = ctypes.c_int
    raw = ctypes.create_string_buffer(resource["pitch_bytes"] * height)
    assert cuda.cuCtxPushCurrent_v2(resource["cuda_context"]) == 0
    try:
        assert cuda.cuMemcpyDtoH_v2(
            raw, resource["device_ptr"], len(raw)) == 0
    finally:
        popped = ctypes.c_void_p()
        assert cuda.cuCtxPopCurrent_v2(ctypes.byref(popped)) == 0
    device_flow = np.frombuffer(raw, dtype=np.uint8).reshape(
        height, resource["pitch_bytes"])[:, :width * FLOW_BYTES_PER_PIXEL].copy()
    device_flow = device_flow.view(np.float32).reshape(height, width, 2)

    host_flow = engine.compute_flow_bgr24(prev, cur, reset_temporal_hints=True)
    host_delta = engine.get_counters_delta()
    assert host_flow.shape == (height, width, 2)
    assert host_delta["nvof_d2h_copy_count_delta"] == 1
    assert host_delta["nvof_d2h_bytes_delta"] == width * height * FLOW_BYTES_PER_PIXEL
    assert np.array_equal(device_flow, host_flow)


def test_nvof_bgr_reset_h2d_is_two_frames(engines):
    """A reset frame uploads both frames, measured: two copies, 3 bytes per pixel each.

    The same history behaviour the sequential RGBA path has, except for the bytes per pixel,
    which is the entire difference this round claims.
    """
    engine = engines()
    cur, prev = _frames(GATE_W, GATE_H, count=2)
    _baseline(engine, GATE_W, GATE_H)
    engine.compute_flow_bgr24(prev, cur, reset_temporal_hints=True)
    delta = engine.get_counters_delta()
    assert delta["nvof_h2d_copy_count_delta"] == 2, (
        f"a reset frame must upload both frames; the counters said {delta}")
    assert delta["nvof_h2d_bytes_delta"] == RESET_H2D_BYTES, (
        f"a reset upload must be 2 x {GATE_W}x{GATE_H}x3 = {RESET_H2D_BYTES} bytes; the "
        f"counters said {delta['nvof_h2d_bytes_delta']}")
    assert delta["reset_count_delta"] == 1
    assert delta["nvof_bgr_input_call_count_delta"] == 1
    # The flow tensor still comes back at float32x2: this round removed no download, and a
    # smaller number here would mean something else had quietly changed.
    assert delta["nvof_d2h_bytes_delta"] == FLOW_D2H_BYTES


def test_nvof_bgr_steady_state_h2d_is_3_bytes_per_pixel(engines):
    """The claimed saving, from the native counter: one copy of width*height*3 per frame."""
    engine = engines()
    frames = _frames(GATE_W, GATE_H, count=4)
    _baseline(engine, GATE_W, GATE_H)
    engine.compute_flow_bgr24(frames[1], frames[0], reset_temporal_hints=True)
    engine.get_counters_delta()
    for cur in frames[1:]:
        engine.compute_flow_bgr24(None, cur, reset_temporal_hints=False)
        delta = engine.get_counters_delta()
        assert delta["nvof_h2d_copy_count_delta"] == 1, (
            f"a steady frame must upload exactly one frame; the counters said {delta}")
        assert delta["nvof_h2d_bytes_delta"] == STEADY_H2D_BYTES, (
            f"a steady upload must be {STEADY_H2D_BYTES} bytes "
            f"({BGR_BYTES_PER_PIXEL} bytes per pixel); the counters said "
            f"{delta['nvof_h2d_bytes_delta']}")
        assert delta["nvof_d2h_bytes_delta"] == FLOW_D2H_BYTES
        assert delta["reset_count_delta"] == 0


def test_nvof_bgr_keeps_native_history(engines):
    """On a steady frame the caller's previous frame is not merely unused, it is unreachable.

    Both engines get the same reset frame and the same current frame; only one of them is
    handed a previous frame on the steady call, and that one is deliberately wrong. Identical
    flow can then only come from the frame the session is still holding on the GPU, and the
    upload counters show the wrong frame never crossed PCIe.
    """
    bgr_engine, legacy_engine = engines(), engines()
    frames = _frames(SMALL_W, SMALL_H, count=3)
    wrong_prev = np.full_like(frames[0], 37)

    bgr_engine.compute_flow_bgr24(frames[0], frames[0], reset_temporal_hints=True)
    _baseline(bgr_engine, SMALL_W, SMALL_H)
    bgr_flow = bgr_engine.compute_flow_bgr24(wrong_prev, frames[1],
                                             reset_temporal_hints=False).copy()
    bgr_delta = bgr_engine.get_counters_delta()

    legacy_engine.compute_flow_sequential(cur_bgr=frames[0], prev_bgr=frames[0],
                                         reset_temporal_hints=True)
    _baseline(legacy_engine, SMALL_W, SMALL_H)
    legacy_flow = legacy_engine.compute_flow_sequential(
        cur_bgr=frames[1], prev_bgr=wrong_prev, reset_temporal_hints=False).copy()

    assert bgr_delta["nvof_h2d_copy_count_delta"] == 1, (
        f"the steady frame uploaded more than the current frame: {bgr_delta}")
    assert np.array_equal(bgr_flow, legacy_flow), (
        "the BGR session's reference frame was not the one its own GPU history kept")


def test_nvof_bgr_staging_buffer_is_reused(engines):
    """100 frames, one pair of allocations: the staging and the kernel module are session-scope.

    A per-frame cuMemAlloc would show up as growth in exactly the counter that is supposed to
    stay where it is, and a per-frame free would show up as the session spending more time in
    the upload span than a copy costs.
    """
    engine = engines()
    frames = _frames(SMALL_W, SMALL_H, count=100)
    _baseline(engine, SMALL_W, SMALL_H)
    at_create = engine.get_counters()
    assert at_create["bgr_staging_alloc_count"] == 2, (
        f"create() allocates two staging rows and nothing else should; the session reports "
        f"{at_create['bgr_staging_alloc_count']}")
    assert at_create["reset_count"] == 0

    # The first frame of a stream is its reset: both surfaces filled, still no allocation.
    engine.compute_flow_bgr24(frames[0], frames[0], reset_temporal_hints=True)
    reset = engine.get_counters_delta()
    assert reset["nvof_bgr_staging_alloc_count_delta"] == 0
    assert reset["nvof_h2d_copy_count_delta"] == 2
    assert reset["nvof_input_convert_launch_count_delta"] == 2

    for index, cur in enumerate(frames[1:], start=1):
        engine.compute_flow_bgr24(None, cur, reset_temporal_hints=False)
        delta = engine.get_counters_delta()
        assert delta["nvof_bgr_staging_alloc_count_delta"] == 0, (
            f"frame {index} allocated device memory: {delta}")
        assert delta["nvof_bgr_input_call_count_delta"] == 1
        assert delta["nvof_input_convert_launch_count_delta"] == 1
        assert delta["nvof_h2d_copy_count_delta"] == 1

    total = engine.get_counters()
    assert total["bgr_staging_alloc_count"] == 2, "staging grew across 100 frames"
    assert total["bgr_input_call_count"] == len(frames)
    # One byte-map launch per steady frame and two for the reset that filled both surfaces.
    assert total["input_convert_launch_count"] == len(frames) + 1
    assert total["h2d_copy_count"] == len(frames) + 1
    assert total["reset_count"] == 1


def test_nvof_bgr_reports_only_its_own_convert_span(engines):
    """`nvof_input_convert_wait_ms` is the byte map plus its synchronize, and nothing else.

    The negative half of the claim is the part that matters: the same key has to read as
    unavailable on the two RGBA entry points, because a number that shows up on a path with
    no such kernel was borrowed from somewhere else.
    """
    engine = engines()
    cur, prev = _frames(SMALL_W, SMALL_H, count=2)

    engine.compute_flow_bgr24(prev, cur, reset_temporal_hints=True)
    reported = engine.get_last_timing()
    convert = reported["nvof_input_convert_wait_ms"]
    assert isinstance(convert, float), (
        f"the BGR path must time its own convert span, got {convert!r}")
    whole = reported["nvof_total_native_ms"]
    parts = {"upload": reported["nvof_upload_ms"], "convert": convert,
             "submit": reported["nvof_submit_cpu_ms"],
             "download": reported["nvof_download_ms"],
             "postprocess": reported["nvof_postprocess_wait_inclusive_ms"]}
    assert all(isinstance(value, float) for value in parts.values()), parts
    assert sum(parts.values()) <= whole + 0.05, (
        f"the spans add up to more than the execute that contains them ({sum(parts.values())}"
        f" vs {whole}), so one of them is double-counting a wait")
    assert reported["nvof_dependency_wait_ms"] == of.NVOF_TIMING_UNAVAILABLE, (
        "the BGR path folds its hardware wait into the postprocess span like the sequential "
        "path does, so reporting a separate one would be reporting the same wait twice")

    engine.compute_flow_sequential(cur_bgr=cur, prev_bgr=prev, reset_temporal_hints=True)
    rgba_sequential = engine.get_last_timing()
    engine.compute_flow(prev, cur)
    rgba_timed = engine.get_last_timing()
    for name, timing_dict in (("sequential", rgba_sequential), ("timed", rgba_timed)):
        assert timing_dict["nvof_input_format"] == "rgba8_host"
        assert timing_dict["nvof_input_convert_wait_ms"] == of.NVOF_TIMING_UNAVAILABLE, (
            f"an RGBA {name} execute reported a byte-map span, so the key is being filled by "
            f"something other than the kernel it names")


# ---------------------------------------------------------------------------
# The test build: what the hardware was actually shown
# ---------------------------------------------------------------------------

_TEST_DLL_SKIP = None


def _test_dll():
    """The same source as the shipped DLL, built with the surface read-back export.

    Returns the path, or None with a reason if the toolchain is not here. Rebuilt when the
    source is newer, so a byte comparison can never pass against a binary that predates the
    kernel it is checking.
    """
    global _TEST_DLL_SKIP
    if os.path.exists(TEST_DLL) and os.path.getmtime(TEST_DLL) >= os.path.getmtime(SOURCE):
        return TEST_DLL
    if not os.path.exists(BUILD_SCRIPT):
        _TEST_DLL_SKIP = f"{BUILD_SCRIPT} is missing"
        return None
    os.makedirs(os.path.dirname(TEST_DLL), exist_ok=True)
    completed = subprocess.run(["cmd", "/c", BUILD_SCRIPT, os.path.dirname(TEST_DLL)],
                               capture_output=True, text=True, timeout=900)
    if completed.returncode != 0 or not os.path.exists(TEST_DLL):
        _TEST_DLL_SKIP = (
            f"the test build failed with {completed.returncode}: "
            + (completed.stdout + completed.stderr)[-2000:])
        return None
    return TEST_DLL


def _check(code, lib, what):
    if code != 0:
        message = lib.nvof_bridge_get_last_error()
        text = message.decode("utf-8", errors="replace") if message else "no message"
        pytest.fail(f"{what} failed with {code}: {text}")


@pytest.fixture()
def nvof_test_lib():
    """The test build, called through ctypes directly rather than through the engine.

    Going straight at the library is the point: the bytes being compared are what the bridge
    put on the GPU, so the check must not depend on the Python layer that is also under test.
    """
    path = _test_dll()
    if path is None:
        pytest.skip(_TEST_DLL_SKIP or "no C++ toolchain for the test build")
    lib = ctypes.CDLL(path)
    lib.nvof_bridge_get_abi_version.restype = ctypes.c_uint32
    assert lib.nvof_bridge_get_abi_version() == of._EXPECTED_NVOF_BRIDGE_ABI, (
        "the test build is not the ABI this checkout reads; rebuild it")
    u32, i32, f32 = ctypes.c_uint32, ctypes.c_int32, ctypes.POINTER(ctypes.c_float)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    lib.nvof_bridge_create.argtypes = [u32, u32, u32, u32, ctypes.POINTER(ctypes.c_void_p)]
    lib.nvof_bridge_create.restype = i32
    lib.nvof_bridge_execute_bgr_sequential.argtypes = [
        ctypes.c_void_p, u8p, u32, u8p, u32, f32, u32, i32,
        ctypes.POINTER(of.NvOFBridgeTiming)]
    lib.nvof_bridge_execute_bgr_sequential.restype = i32
    lib.nvof_bridge_test_read_input_surface.argtypes = [ctypes.c_void_p, i32, u8p, u32]
    lib.nvof_bridge_test_read_input_surface.restype = i32
    lib.nvof_bridge_get_last_error.restype = ctypes.c_char_p
    lib.nvof_bridge_destroy.argtypes = [ctypes.c_void_p]
    lib.nvof_bridge_destroy.restype = i32
    yield lib


@pytest.fixture()
def nvof_session(nvof_test_lib):
    handle = ctypes.c_void_p(0)
    _check(nvof_test_lib.nvof_bridge_create(SMALL_W, SMALL_H, 4, 10, ctypes.byref(handle)),
           nvof_test_lib, f"nvof_bridge_create({SMALL_W}x{SMALL_H})")
    assert handle.value, "the test build returned a null session handle"
    yield handle
    _check(nvof_test_lib.nvof_bridge_destroy(handle), nvof_test_lib, "nvof_bridge_destroy")


def _execute_bgr(lib, session, cur, prev, reset):
    out = np.zeros((SMALL_H, SMALL_W, 2), dtype=np.float32)
    timing_struct = of.NvOFBridgeTiming()
    _check(lib.nvof_bridge_execute_bgr_sequential(
        session,
        cur.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), cur.strides[0],
        (prev.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)) if prev is not None else None),
        prev.strides[0] if prev is not None else 0,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 0,
        1 if reset else 0, ctypes.byref(timing_struct)),
        lib, "nvof_bridge_execute_bgr_sequential")
    return out, timing_struct


def _read_surface(lib, session, which):
    """One ABGR8 input surface back to the host. `which` is 0 for the current, 1 for reference."""
    dst = np.zeros((SMALL_H, SMALL_W, 4), dtype=np.uint8)
    _check(lib.nvof_bridge_test_read_input_surface(
        session, which, dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), dst.strides[0]),
        lib, f"nvof_bridge_test_read_input_surface({which})")
    return dst


def test_nvof_bgr_input_surface_bytes_match_cv2_bgr_to_rgba(nvof_test_lib, nvof_session):
    """The kernel's output compared to cv2's, byte for byte, at the surface NVOF reads.

    This is the only place that claim can be checked: two different inputs can produce the
    same flow field, so the parity gate above would not notice a kernel that reordered the
    channels some other way. The named-per-byte checks below exist because a bare mismatch
    here would not say which byte was wrong.
    """
    cur, prev = _frames(SMALL_W, SMALL_H, count=2)
    _execute_bgr(nvof_test_lib, nvof_session, cur, prev, reset=True)
    for surface, frame in ((0, cur), (1, prev)):
        got = _read_surface(nvof_test_lib, nvof_session, surface)
        want = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        assert np.array_equal(got, want), (
            f"surface {surface} differs from cv2.cvtColor(BGR2RGBA) in "
            f"{int((got != want).sum())} of {got.size} bytes")
        assert np.array_equal(got[:, :, 0], frame[:, :, 2]), (
            f"surface {surface}: its first byte is not the blue byte")
        assert np.array_equal(got[:, :, 1], frame[:, :, 1]), (
            f"surface {surface}: its second byte is not the green byte")
        assert np.array_equal(got[:, :, 2], frame[:, :, 0]), (
            f"surface {surface}: its third byte is not the red byte")
        assert (got[:, :, 3] == 255).all(), (
            f"surface {surface}: the alpha byte is not the constant 255")


def test_nvof_bgr_steady_frame_keeps_previous_surface_as_reference(nvof_test_lib,
                                                                    nvof_session):
    """After a steady frame the reference surface still holds the frame before it.

    Native history in bytes rather than in a name: the second frame was mapped into the input
    surface only, and the reference surface kept whatever the reset frame put there.
    """
    first, second = _frames(SMALL_W, SMALL_H, count=2)
    _execute_bgr(nvof_test_lib, nvof_session, first, first, reset=True)
    _execute_bgr(nvof_test_lib, nvof_session, second, None, reset=False)
    reference = _read_surface(nvof_test_lib, nvof_session, 1)
    assert np.array_equal(reference, cv2.cvtColor(first, cv2.COLOR_BGR2RGBA)), (
        "the steady frame's reference surface is not the previous frame; the history the "
        "counters describe is not the history the hardware sees")
    current = _read_surface(nvof_test_lib, nvof_session, 0)
    assert np.array_equal(current, cv2.cvtColor(second, cv2.COLOR_BGR2RGBA)), (
        "the steady frame never reached its own input surface")


def test_the_bgr_surface_read_back_is_not_in_the_shipped_binary():
    """A production build must not be able to copy an input surface back off the GPU.

    The export is a test fixture. If it ever reached native/nvof_bridge.dll the pipeline would
    gain a path nothing measures, and "input surface" would start meaning two different things
    in two different binaries.
    """
    lib = ctypes.CDLL(PRODUCTION_DLL)
    with pytest.raises(AttributeError):
        getattr(lib, "nvof_bridge_test_read_input_surface")
