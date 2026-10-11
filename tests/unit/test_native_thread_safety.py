"""Gate: the residual bridge handle lifecycle has to survive real concurrency.

Everything here drives native/residual_bridge.dll through ctypes from several threads at
once and asserts on measured values - pixels, counters, timestamps and error text. "It did
not crash" is never the assertion, because the design this replaces (a handle that was a
cast pointer, a plan vector with no lock, one global error string and one global set of
counters) kept producing plausible output right up until it freed memory under a frame
that was still running.

Skipped only when residual_bridge.dll is not built on this machine; when it is present
every test below has to pass.
"""

import ctypes
import os
import sys
import threading
import time

import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline  # noqa: E402

residual_native = pytest.importorskip(
    "residual_native", reason="residual_bridge extension missing")
if not residual_native.is_available():
    pytest.skip("residual_bridge could not be probed", allow_module_level=True)

_LIB = residual_native._library()
_U8 = ctypes.POINTER(ctypes.c_uint8)
_COUNTER_FIELDS = [name for name, _ in residual_native._Counters._fields_]

# A working resolution large enough that one frame takes tens of milliseconds, so
# "another thread did this while the first call was still inside" is a window long enough
# to measure rather than a coin flip. Measured on this machine by the assertions below,
# which fail with the observed duration instead of silently passing if it shrinks.
BIG_SRC_W, BIG_SRC_H = 3840, 2160
BIG_LIVE_W, BIG_LIVE_H = 1920, 1080
MIN_LONG_FRAME_MS = 20.0


def _ptr(array):
    assert array.flags["C_CONTIGUOUS"], "the bridge rejects a non-contiguous row stride"
    return array.ctypes.data_as(_U8)


def _create(src_w, src_h, live_w, live_h):
    handle = ctypes.c_int64(0)
    code = _LIB.rr_create_plan(ctypes.byref(handle), src_w, src_h, live_w, live_h)
    assert code == 0, _error_text()
    assert handle.value > 0, "handles must be positive registry ids"
    return handle.value


def _destroy(handle):
    return _LIB.rr_destroy_plan(handle)


def _error_text():
    buf = ctypes.create_string_buffer(512)
    _LIB.rr_get_last_error(buf, ctypes.c_int32(len(buf)))
    return buf.value.decode("utf-8", "replace")


def _downsample(handle, source, dst):
    return _LIB.rr_downsample_bgr8(
        handle, _ptr(source), source.shape[1], source.shape[0],
        ctypes.c_int64(source.strides[0]),
        _ptr(dst), dst.shape[1], dst.shape[0], ctypes.c_int64(dst.strides[0]))


def _reconstruct(handle, source, low, dlss, dst, strength=1.0):
    return _LIB.rr_reconstruct_bgr8(
        handle, _ptr(source), source.shape[1], source.shape[0],
        ctypes.c_int64(source.strides[0]),
        _ptr(low), low.shape[1], low.shape[0], ctypes.c_int64(low.strides[0]),
        _ptr(dlss), dlss.shape[1], dlss.shape[0], ctypes.c_int64(dlss.strides[0]),
        ctypes.c_float(float(strength)),
        _ptr(dst), dst.shape[1], dst.shape[0], ctypes.c_int64(dst.strides[0]))


def _counters(handle):
    block = residual_native._Counters()
    code = _LIB.rr_get_counters(handle, ctypes.byref(block))
    assert code == 0, f"rr_get_counters rejected a live handle: {_error_text()}"
    return {name: getattr(block, name) for name in _COUNTER_FIELDS}


def _scene(src_w, src_h, live_w, live_h, seed):
    """Random input plus the exact output the Python reference produces for it.

    The sequence is the one the pipeline runs: working color from the source, then the
    residual between the DLSS output at working resolution and that same working color.
    """
    rng = np.random.default_rng(seed)
    source = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)
    dlss = rng.integers(0, 256, size=(live_h, live_w, 3), dtype=np.uint8)
    ref = pipeline.PythonReferenceBackend(src_w, src_h, live_w, live_h)
    try:
        want_low = np.zeros((live_h, live_w, 3), dtype=np.uint8)
        want_recon = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        ref.prepare_working_color(source, want_low)
        ref.reconstruct(source, want_low, dlss, 1.0, want_recon)
    finally:
        ref.close()
    return source, dlss, want_low, want_recon


def _long_reconstruct(seed=1):
    """One 4K reconstruct: the frame long enough to overlap another thread's work."""
    return _scene(BIG_SRC_W, BIG_SRC_H, BIG_LIVE_W, BIG_LIVE_H, seed)


# ------------------------------------------------- one plan, many threads ----

def test_same_plan_concurrent_frames_are_serialized_and_each_output_is_exact():
    """Two calls on one plan must not interleave on the plan's scratch.

    A plan owns one FP32 scratch, one residual buffer, one row accumulator set and one
    worker pool. Without the per-plan lock two reconstruct calls mix their intermediate
    rows, and the pixel that comes out belongs to neither frame. Every thread therefore
    compares its own buffers against the Python reference for its own input.
    """
    src_w, src_h, live_w, live_h = 160, 120, 80, 60
    handle = _create(src_w, src_h, live_w, live_h)
    threads, rounds = 6, 12
    scenes = [_scene(src_w, src_h, live_w, live_h, 100 + i) for i in range(threads)]
    failures = []
    barrier = threading.Barrier(threads)

    def worker(index):
        source, dlss, want_low, want_recon = scenes[index]
        got_low = np.full((live_h, live_w, 3), 3, dtype=np.uint8)
        got_recon = np.full((src_h, src_w, 3), 3, dtype=np.uint8)
        try:
            barrier.wait(timeout=30)
            for _ in range(rounds):
                if _downsample(handle, source, got_low) != 0:
                    failures.append(f"thread {index}: downsample failed: {_error_text()}")
                    return
                if not np.array_equal(got_low, want_low):
                    failures.append(f"thread {index}: downsample pixels do not match the "
                                    "Python reference")
                    return
                if _reconstruct(handle, source, got_low, dlss, got_recon) != 0:
                    failures.append(f"thread {index}: reconstruct failed: {_error_text()}")
                    return
                if not np.array_equal(got_recon, want_recon):
                    failures.append(f"thread {index}: reconstruct pixels do not match the "
                                    "Python reference")
                    return
        finally:
            barrier.abort()

    try:
        before = _counters(handle)
        workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=120)
        assert not any(w.is_alive() for w in workers), "a worker never came back"
        after = _counters(handle)
    finally:
        _destroy(handle)

    assert not failures, failures[0]
    # No lost or double counted work: the increments are atomic and exactly one per call.
    assert after["downsample_call_count"] == before["downsample_call_count"] + threads * rounds
    assert after["reconstruct_call_count"] == before["reconstruct_call_count"] + threads * rounds
    assert after["downsample_pixel_count"] == (before["downsample_pixel_count"]
                                               + threads * rounds * live_w * live_h)
    assert after["reconstruct_pixel_count"] == (before["reconstruct_pixel_count"]
                                                + threads * rounds * src_w * src_h)


# ----------------------------------------------- two plans, no interference ----

def test_a_long_frame_on_one_plan_does_not_block_another_plan():
    """The plan lock is per plan. A second backend runs while the first is mid-frame.

    The small call is issued while the 4K call is provably still inside the bridge. If the
    two shared one lock - or one scratch, or a global registry lock held across a frame -
    the small call's duration would be a large fraction of the big one's instead of a
    fraction of it. Both outputs are also checked pixel for pixel, so a fast answer that
    was wrong does not count as a pass.
    """
    big = _create(BIG_SRC_W, BIG_SRC_H, BIG_LIVE_W, BIG_LIVE_H)
    small = _create(64, 48, 32, 24)
    source, dlss, want_low, want_recon = _long_reconstruct(seed=5)
    big_low = np.zeros((BIG_LIVE_H, BIG_LIVE_W, 3), dtype=np.uint8)
    big_recon = np.zeros((BIG_SRC_H, BIG_SRC_W, 3), dtype=np.uint8)
    small_source, small_dlss, small_want_low, small_want_recon = _scene(64, 48, 32, 24, seed=6)
    small_out_low = np.full((24, 32, 3), 9, dtype=np.uint8)
    small_out_recon = np.full((48, 64, 3), 9, dtype=np.uint8)
    big_result = {}

    def run_big():
        t0 = time.perf_counter()
        big_result["started"] = True
        code = _downsample(big, source, big_low)
        t1 = time.perf_counter()
        code2 = _reconstruct(big, source, big_low, dlss, big_recon)
        t2 = time.perf_counter()
        big_result.update(down_ms=(t1 - t0) * 1e3, recon_ms=(t2 - t1) * 1e3,
                          total_ms=(t2 - t0) * 1e3, codes=(code, code2))

    try:
        t = threading.Thread(target=run_big)
        t.start()
        deadline = time.perf_counter() + 10.0
        while not big_result.get("started") and time.perf_counter() < deadline:
            time.sleep(0.001)
        assert big_result.get("started"), "the 4K thread never started"
        time.sleep(0.005)          # the downsample is inside the bridge by now
        s0 = time.perf_counter()
        assert _downsample(small, small_source, small_out_low) == 0, _error_text()
        assert _reconstruct(small, small_source, small_out_low, small_dlss,
                            small_out_recon) == 0, _error_text()
        small_ms = (time.perf_counter() - s0) * 1e3
        t.join(timeout=120)
        assert not t.is_alive(), "the 4K call never returned"
    finally:
        _destroy(big)
        _destroy(small)

    assert big_result["codes"] == (0, 0), f"the big frame failed: {big_result}"
    assert np.array_equal(big_low, want_low), "the 4K downsample was corrupted"
    assert np.array_equal(big_recon, want_recon), "the 4K reconstruct was corrupted"
    assert np.array_equal(small_out_low, small_want_low)
    assert np.array_equal(small_out_recon, small_want_recon)
    assert big_result["total_ms"] > MIN_LONG_FRAME_MS, (
        f"the 4K frame took {big_result['total_ms']:.2f} ms, too short to prove anything; "
        "raise BIG_SRC_* instead of relaxing this test")
    assert small_ms < 0.5 * big_result["total_ms"], (
        f"a small frame took {small_ms:.2f} ms while a {big_result['total_ms']:.2f} ms "
        "frame was running on another plan, so the plans are serialized")


# ------------------------------------------------- destroy against a live call ----

def test_destroy_waits_for_the_call_in_flight_and_then_invalidates_the_handle():
    """The forbidden interleaving: one thread reconstructing while another frees the plan.

    destroy has to make the handle fail to new callers immediately, and still hold the
    plan alive until the call that was already running finishes. Measured directly: the
    destroy call takes roughly the rest of the frame, not microseconds.
    """
    handle = _create(BIG_SRC_W, BIG_SRC_H, BIG_LIVE_W, BIG_LIVE_H)
    source, dlss, want_low, want_recon = _long_reconstruct(seed=9)
    low = np.array(want_low, dtype=np.uint8)     # the working color, prepared up front
    recon = np.zeros((BIG_SRC_H, BIG_SRC_W, 3), dtype=np.uint8)
    outcome = {}

    def run_frame():
        # Exactly one bridge call, so "destroy arrived while it was in flight" is a fact
        # rather than a race against the gap between two calls.
        outcome["start_ns"] = time.perf_counter_ns()
        try:
            outcome["code"] = _reconstruct(handle, source, low, dlss, recon)
        except BaseException as exc:
            # A bridge that freed the plan under this call faults instead of returning a
            # code; keep the reason so the failure below says so instead of KeyErrors.
            outcome["thread_exception"] = f"{type(exc).__name__}: {exc}"
            raise
        outcome["end_ns"] = time.perf_counter_ns()

    t = None
    try:
        t = threading.Thread(target=run_frame)
        t.start()
        deadline = time.perf_counter() + 10.0
        while "start_ns" not in outcome and time.perf_counter() < deadline:
            time.sleep(0.001)
        assert "start_ns" in outcome, "the frame thread never reached the bridge"
        time.sleep(0.005)                # the call is inside the bridge by now
        d0 = time.perf_counter_ns()
        destroy_code = _destroy(handle)
        d1 = time.perf_counter_ns()
        t.join(timeout=120)
        assert not t.is_alive(), "the in-flight frame never returned"
    except BaseException:
        if t is not None:
            t.join(timeout=120)
        _destroy(handle)
        raise

    destroy_ms = (d1 - d0) / 1e6
    assert destroy_code == 0, _error_text()
    assert "end_ns" in outcome, (
        "the in-flight frame never returned: " + outcome.get("thread_exception", "no exception"))
    frame_ms = (outcome["end_ns"] - outcome["start_ns"]) / 1e6
    assert outcome["code"] == 0, (
        f"the frame that was running when destroy arrived was rejected "
        f"({outcome['code']}): {_error_text()}")
    assert np.array_equal(recon, want_recon), "the in-flight frame did not finish correctly"
    assert frame_ms > MIN_LONG_FRAME_MS, (
        f"the frame took {frame_ms:.2f} ms, too short to prove the wait; raise BIG_SRC_*")
    # An implementation that freed the plan straight away returns in microseconds here.
    assert destroy_ms > 0.3 * frame_ms, (
        f"destroy returned after {destroy_ms:.2f} ms while the frame it was supposed to "
        f"wait for took {frame_ms:.2f} ms")
    assert _LIB.rr_get_counters(handle, ctypes.byref(residual_native._Counters())) == -5
    assert _destroy(handle) == -5, "destroying the same handle twice must be rejected"


def test_destroy_during_a_frame_never_frees_memory_under_the_running_call():
    """Hammer the destroy-versus-call race and check what every call actually returned.

    A call either started before the erase, in which case it must run to completion on a
    plan that stays alive, or after it, in which case it must be rejected with
    RR_ERR_BAD_HANDLE. Anything else - a wrong pixel, a hang, a fault - is the bug.
    """
    src_w, src_h, live_w, live_h = 320, 240, 160, 120
    source, dlss, want_low, want_recon = _scene(src_w, src_h, live_w, live_h, 21)
    rounds = 60
    accepted, rejected = [], []
    mismatched = []

    def run_one():
        handle = _create(src_w, src_h, live_w, live_h)
        low = np.zeros((live_h, live_w, 3), dtype=np.uint8)
        recon = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        coderes = _downsample(handle, source, low)
        coderes2 = _reconstruct(handle, source, low, dlss, recon)
        _destroy(handle)
        if coderes == 0 and coderes2 == 0:
            accepted.append(1)
            if not (np.array_equal(low, want_low) and np.array_equal(recon, want_recon)):
                mismatched.append("a frame that the bridge accepted produced wrong pixels")
        elif coderes == -5:
            rejected.append(1)
        else:
            mismatched.append(f"unexpected return codes {(coderes, coderes2)}: {_error_text()}")

    for _ in range(rounds):
        workers = [threading.Thread(target=run_one) for _ in range(4)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=60)
        if any(w.is_alive() for w in workers):
            mismatched.append("a worker hung")
            break

    assert not mismatched, mismatched[0]
    assert len(accepted) == rounds * 4, "a frame on a freshly created handle was rejected"
    # And now prove the rejection path is reachable at all: a handle nobody owns.
    assert _LIB.rr_get_counters(999999, ctypes.byref(residual_native._Counters())) == -5


# ------------------------------------------------------ one handle per thread ----

def test_last_error_text_is_per_calling_thread():
    """rr_get_last_error takes no handle, so its storage has to be thread local.

    The order is forced, not hoped for: thread A fails, then thread B fails, and only then
    does A read. With one shared string A reads B's message, which is exactly what a
    pipeline thread would do wrong when two frames fail at once.
    """
    handle = _create(64, 48, 32, 24)
    a_failed = threading.Event()
    b_failed = threading.Event()
    a_has_read = threading.Event()
    texts = {}

    def bad_handle_path():
        _LIB.rr_get_counters(424242, ctypes.byref(residual_native._Counters()))
        a_failed.set()
        b_failed.wait(timeout=30)           # B has failed by now, so a shared string is B's
        texts["thread_a"] = _error_text()
        a_has_read.set()

    def wrong_dims_path():
        a_failed.wait(timeout=30)
        src = np.zeros((48, 64, 3), dtype=np.uint8)
        dst = np.zeros((16, 16, 3), dtype=np.uint8)   # this plan maps to 32x24
        assert _downsample(handle, src, dst) == -4
        b_failed.set()
        a_has_read.wait(timeout=30)
        texts["thread_b"] = _error_text()

    try:
        ta = threading.Thread(target=bad_handle_path)
        tb = threading.Thread(target=wrong_dims_path)
        ta.start()
        tb.start()
        ta.join(timeout=60)
        tb.join(timeout=60)
    finally:
        _destroy(handle)

    assert not ta.is_alive() and not tb.is_alive(), "a thread was left waiting forever"
    assert set(texts) == {"thread_a", "thread_b"}, texts
    assert "invalid plan handle" in texts["thread_a"], texts["thread_a"]
    assert "called with" not in texts["thread_a"], (
        f"the bad-handle thread read the other thread's message: {texts['thread_a']!r}")
    assert "plan is 64x48" in texts["thread_b"], texts["thread_b"]
    assert "invalid plan handle" not in texts["thread_b"], (
        f"the dimension-mismatch thread read the other thread's message: {texts['thread_b']!r}")


# ------------------------------------------------------------ counter scope ----

def test_a_probe_plan_cannot_pollute_another_plans_counters():
    """is_available() builds and destroys a plan of its own. Nothing else may notice.

    The per-plan block is that plan's lifetime; the process block is where the probe
    belongs. Reporting plan_create_count = N for one backend described the library, not
    the backend, which is the misattribution this test pins down.
    """
    live = [(_create(64, 48, 32, 24), (64, 48, 32, 24)),
            (_create(96, 72, 48, 36), (96, 72, 48, 36)),
            (_create(128, 96, 64, 48), (128, 96, 64, 48))]
    try:
        base = {h: _counters(h) for h, dims in live}
        for h, (sw, sh, lw, lh) in live:
            assert base[h]["plan_create_count"] == 1
            assert base[h]["scratch_alloc_count"] == 1
            assert base[h]["plan_destroy_count"] == 0
            assert base[h]["axis_plan_count"] == 4
            assert base[h]["downsample_call_count"] == 0

        assert residual_native.is_available() is True

        after = {h: _counters(h) for h, dims in live}
        for h, _ in live:
            for name in ("plan_create_count", "scratch_alloc_count", "plan_destroy_count",
                         "worker_create_count", "scratch_bytes", "axis_plan_count",
                         "downsample_call_count", "reconstruct_call_count"):
                assert after[h][name] == base[h][name], (
                    f"the capability probe moved per-plan {name} from "
                    f"{base[h][name]} to {after[h][name]}")
            # The probe's plan shows up where it belongs: the library-wide totals.
            assert after[h]["process_plan_create_total"] == base[h]["process_plan_create_total"] + 1
            assert after[h]["process_plan_destroy_total"] == base[h]["process_plan_destroy_total"] + 1
            assert (after[h]["process_scratch_alloc_total"]
                    == base[h]["process_scratch_alloc_total"] + 1)

        # Each plan reports its own sizes, not a running sum of everybody's.
        scratch_bytes = {h: _counters(h)["scratch_bytes"] for h, _ in live}
        assert len(set(scratch_bytes.values())) == 3, "per-plan scratch is not per plan"
        totals = {h: _counters(h)["process_scratch_bytes_total"] for h, _ in live}
        assert len(set(totals.values())) == 1, "process totals differ between plans"
        assert min(totals.values()) > max(scratch_bytes.values()), (
            "the process total should be the sum over plans, not one plan's bytes")
    finally:
        for h, _ in live:
            _destroy(h)


def test_process_totals_reconcile_under_concurrent_create_and_destroy():
    """Concurrent lifecycle calls must not lose counts on either scope.

    Six threads each create a plan, run a frame and destroy it, while three long-lived
    plans keep their own scope. Afterwards the process totals have to add up exactly, and
    the long-lived plans still report one create and zero destroys.
    """
    keep = [(_create(64, 48, 32, 24), 64, 48, 32, 24),
            (_create(96, 72, 48, 36), 96, 72, 48, 36),
            (_create(128, 96, 64, 48), 128, 96, 64, 48)]
    source = {}
    for h, sw, sh, lw, lh in keep:
        source[h] = _scene(sw, sh, lw, lh, seed=h % 97 + 1)
    start = _counters(keep[0][0])
    threads, per_thread = 6, 8
    problems = []
    go = threading.Event()

    def churn(index):
        go.wait(timeout=60)
        for k in range(per_thread):
            h = _create(64, 48, 32, 24)
            src = np.zeros((48, 64, 3), dtype=np.uint8)
            dst = np.zeros((24, 32, 3), dtype=np.uint8)
            code = _downsample(h, src, dst)
            counts = _counters(h)
            _destroy(h)
            if code != 0 or counts["downsample_call_count"] != 1:
                problems.append(f"thread {index} iteration {k}: code={code} "
                                f"calls={counts['downsample_call_count']}")
                return
            if counts["plan_create_count"] != 1 or counts["scratch_alloc_count"] != 1:
                problems.append(f"thread {index} iteration {k}: per-plan scope is "
                                f"{counts['plan_create_count']}/"
                                f"{counts['scratch_alloc_count']}")
                return

    try:
        mid = []
        for h, sw, sh, lw, lh in keep:
            src, dlss, want_low, want_recon = source[h]
            low = np.zeros((lh, lw, 3), dtype=np.uint8)
            recon = np.zeros((sh, sw, 3), dtype=np.uint8)
            assert _downsample(h, src, low) == 0 and _reconstruct(h, src, low, dlss, recon) == 0
            mid.append(_counters(h))

        workers = [threading.Thread(target=churn, args=(i,)) for i in range(threads)]
        for w in workers:
            w.start()
        go.set()
        for w in workers:
            w.join(timeout=120)
        assert not any(w.is_alive() for w in workers), "a churn worker hung"
        end = _counters(keep[0][0])
    finally:
        for h, _, _, _, _ in keep:
            _destroy(h)

    assert not problems, problems[0]
    churned = threads * per_thread
    for name in ("process_plan_create_total", "process_plan_destroy_total",
                 "process_scratch_alloc_total"):
        assert end[name] - start[name] == churned, (
            f"{name} moved by {end[name] - start[name]} while {churned} plans were created "
            "and destroyed, so concurrent lifecycle calls lose counts")
    assert (end["process_plan_create_total"] - end["process_plan_destroy_total"]
            == start["process_plan_create_total"] - start["process_plan_destroy_total"]), \
        "plans opened by the churn threads were still counted as live afterwards"
    # The three long-lived plans each ran exactly one frame pair and were each created
    # exactly once, whatever the churn threads did.
    for snapshot in mid:
        assert snapshot["plan_create_count"] == 1
        assert snapshot["plan_destroy_count"] == 0
        assert snapshot["downsample_call_count"] == 1
        assert snapshot["reconstruct_call_count"] == 1


def test_destroyed_handle_never_aliases_the_next_plan():
    """Handles are registry ids, so a freed address cannot be mistaken for a new plan.

    The previous design handed out cast pointers, and an allocator that reused an address
    silently pointed the old handle at somebody else's plan.
    """
    seen = set()
    for i in range(20):
        first = _create(64, 48, 32, 24)
        assert _destroy(first) == 0
        second = _create(96 + 8 * i, 72, 48, 36)
        assert second != first, "a handle was handed out twice"
        assert first not in seen, "a live handle collided with an earlier one"
        counters = residual_native._Counters()
        assert _LIB.rr_get_counters(first, ctypes.byref(counters)) == -5, (
            "the stale handle resolved to the new plan")
        src = np.zeros((72, 96 + 8 * i, 3), dtype=np.uint8)
        dst = np.zeros((36, 48, 3), dtype=np.uint8)
        assert _downsample(first, src, dst) == -5
        assert _downsample(second, src, dst) == 0, _error_text()
        seen.add(second)
        assert _destroy(second) == 0


def test_stale_bridge_major_version_is_refused(monkeypatch):
    """A version 1 DLL must not be read with the version 2 struct.

    The struct grew by five int64 fields, so a stale library is a silent misparse rather
    than an error unless the loader checks. This drives the check with a fake library so
    it is covered even on a machine where the DLL is current.
    """
    class _FakeLib:
        def rr_get_api_version(self):
            return 0x00010000

    monkeypatch.setattr(residual_native, "_LIB", None)
    monkeypatch.setattr(residual_native, "_load_library", lambda: _FakeLib())
    with pytest.raises(residual_native.ResidualBridgeError) as excinfo:
        residual_native._library()
    assert "api version" in str(excinfo.value).lower()

    # The real library still loads afterwards: the refusal was about the stale one.
    monkeypatch.undo()
    residual_native._LIB = None
    assert (residual_native._library().rr_get_api_version() >> 16) == 2
