"""Interrupt one allocation step of rr_create_plan and report what the library did.

Child-process driver for tests/unit/test_residual_bridge_failure_paths.py. Each run loads a
DLL built from native/residual_bridge.cpp with RR_TEST_FAULT_INJECTION, arms one fault
point, calls rr_create_plan repeatedly and writes what it observed to --out as JSON.

It runs in a child process on purpose. The failure this measures is the one that used to
free the worker pool twice, and a process that corrupts its heap does not always fail an
assertion - sometimes it dies later, somewhere else. Only the exit code the parent sees can
tell "returned an error code" apart from "the exception left the library", so this script
has to exit normally for the numbers it prints to mean anything.
"""

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import psutil  # noqa: E402

import residual_native as rn  # noqa: E402

RR_OK = 0
RR_ERR_OUT_OF_MEMORY = -9
RR_ERR_INTERNAL = -7

# Mirrors TestFaultPoint in native/residual_bridge.cpp.
POINT_NAMES = {
    0: "control_no_fault",
    1: "after_plan_create",
    2: "during_axis_plan",
    3: "during_pool_thread_create",
    4: "after_row_pool_create",
    5: "during_scratch_alloc",
    6: "before_registry_publish",
    7: "during_registry_publish",
}

# Points where rr_create_plan is expected to succeed anyway. 0 arms nothing. 3 throws while
# the pool is starting, and RowPool's constructor absorbs it on purpose: a pool with fewer
# workers than asked for is still a correct pool, so the contract there is "fewer workers,
# same pixels", not "error".
SUCCEEDING_POINTS = (0, 3)

# Only points 1, 2 and 4-7 leave a plan that was never published.
PARTIAL_POOL_POINT = 3

KIND_NAMES = {0: "bad_alloc", 1: "runtime_error"}

PROCESS_KEYS = ("process_plan_create_total", "process_plan_destroy_total",
                "process_scratch_alloc_total", "process_scratch_bytes_total",
                "process_worker_create_total")


def _load(path):
    """Point residual_native at the test DLL before it caches a handle to anything.

    Reusing the product's loader keeps the ctypes signatures, the ABI major-version gate and
    the error-name table under test the same objects the application uses."""
    rn._library_path = lambda: path
    rn._LIB = None
    return rn._library()


def _declare_test_hooks(lib):
    lib.rr_test_arm_fault.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.rr_test_arm_fault.restype = ctypes.c_int32
    lib.rr_test_fault_hits.argtypes = [ctypes.c_int32]
    lib.rr_test_fault_hits.restype = ctypes.c_int32
    lib.rr_test_clear_fault.argtypes = []
    lib.rr_test_clear_fault.restype = None
    lib.rr_test_registry_size.argtypes = [ctypes.POINTER(ctypes.c_int32)]
    lib.rr_test_registry_size.restype = ctypes.c_int32
    return lib


def _counters(lib, handle):
    counters = rn._Counters()
    code = lib.rr_get_counters(handle, ctypes.byref(counters))
    if code != RR_OK:
        raise RuntimeError(f"rr_get_counters failed with {code}: {_error_text(lib)}")
    return {name: getattr(counters, name) for name, _ in rn._Counters._fields_}


def _registry_size(lib):
    value = ctypes.c_int32(0)
    if lib.rr_test_registry_size(ctypes.byref(value)) != RR_OK:
        raise RuntimeError("rr_test_registry_size failed")
    return value.value


def _error_text(lib):
    buf = ctypes.create_string_buffer(512)
    lib.rr_get_last_error(buf, ctypes.c_int32(len(buf)))
    return buf.value.decode("utf-8", "replace")


def _frame_inputs(dims):
    """Fixed inputs for one frame, identical on every call.

    A digest computed on a plan with eight workers and one computed on a plan whose pool
    only started a single worker are only comparable if the pixels going in match."""
    src_w, src_h, live_w, live_h = dims
    rng = np.random.default_rng(20260920)
    return {
        "source": rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8),
        "low": np.zeros((live_h, live_w, 3), dtype=np.uint8),
        "dlss": rng.integers(0, 256, size=(live_h, live_w, 3), dtype=np.uint8),
        "strength": 0.5,
    }


def _u8(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def _digest_on_plan(lib, handle, dims):
    """Downsample and reconstruct on an existing handle, returning the output digest."""
    src_w, src_h, live_w, live_h = dims
    f = _frame_inputs(dims)
    out = np.empty((src_h, src_w, 3), dtype=np.uint8)
    code = lib.rr_downsample_bgr8(handle, _u8(f["source"]), src_w, src_h,
                                  ctypes.c_int64(f["source"].strides[0]),
                                  _u8(f["low"]), live_w, live_h,
                                  ctypes.c_int64(f["low"].strides[0]))
    if code != RR_OK:
        raise RuntimeError(f"rr_downsample_bgr8 failed with {code}: {_error_text(lib)}")
    code = lib.rr_reconstruct_bgr8(handle,
                                   _u8(f["source"]), src_w, src_h,
                                   ctypes.c_int64(f["source"].strides[0]),
                                   _u8(f["low"]), live_w, live_h,
                                   ctypes.c_int64(f["low"].strides[0]),
                                   _u8(f["dlss"]), live_w, live_h,
                                   ctypes.c_int64(f["dlss"].strides[0]),
                                   ctypes.c_float(f["strength"]),
                                   _u8(out), src_w, src_h, ctypes.c_int64(out.strides[0]))
    if code != RR_OK:
        raise RuntimeError(f"rr_reconstruct_bgr8 failed with {code}: {_error_text(lib)}")
    return hashlib.sha1(out.tobytes()).hexdigest()


def _full_frame_via_backend(dims):
    """The same frame through the class the application actually uses."""
    src_w, src_h, live_w, live_h = dims
    f = _frame_inputs(dims)
    out = np.empty((src_h, src_w, 3), dtype=np.uint8)
    backend = rn.NativeResidualBackend(src_w, src_h, live_w, live_h)
    try:
        backend.prepare_working_color(f["source"], f["low"])
        backend.reconstruct(f["source"], f["low"], f["dlss"], f["strength"], out)
        diagnostics = backend.diagnostics()
    finally:
        backend.close()
    return hashlib.sha1(out.tobytes()).hexdigest(), diagnostics


def run(dll, point, kind, reps, dims):
    lib = _declare_test_hooks(_load(dll))
    src_w, src_h, live_w, live_h = dims
    threads_at_entry = psutil.Process().num_threads()

    # A plan that stays alive for the whole run. The process-wide totals are only readable
    # through some plan's handle, and it must not be one of the plans being created and
    # destroyed, or the measurement would include its own bookkeeping.
    observer_box = ctypes.c_int64(0)
    if lib.rr_create_plan(ctypes.byref(observer_box), src_w, src_h, live_w, live_h) != RR_OK:
        raise RuntimeError("could not create the observer plan: " + _error_text(lib))
    observer = observer_box.value

    # Before any fault is armed: proves the library works in this process at all, and gives
    # every later digest something to be equal to.
    digest_reference, _ = _full_frame_via_backend(dims)

    counters_before = _counters(lib, observer)
    registry_baseline = _registry_size(lib)
    threads_baseline = psutil.Process().num_threads()

    expected_code = RR_OK if point in SUCCEEDING_POINTS else (
        RR_ERR_OUT_OF_MEMORY if kind == 0 else RR_ERR_INTERNAL)

    reps_seen = []
    started = time.time()
    for _ in range(reps):
        lib.rr_test_arm_fault(point, kind)
        candidate = ctypes.c_int64(0)
        code = lib.rr_create_plan(ctypes.byref(candidate), src_w, src_h, live_w, live_h)
        message = _error_text(lib)
        fired = lib.rr_test_fault_hits(point)
        lib.rr_test_clear_fault()

        record = {
            "code": code,
            "handle": candidate.value,
            "hits": fired,
            "message": message,
            "registry_size": _registry_size(lib),
        }
        if code == RR_OK:
            # Read what the plan says about itself, and make it work, before letting go of
            # it. For the partial pool this is the whole point: it has to run a frame.
            plan_counters = _counters(lib, candidate.value)
            record["this_plan_worker_create_count"] = plan_counters["worker_create_count"]
            record["this_plan_chunk_scratch_ok"] = plan_counters["scratch_alloc_count"] == 1
            if point == PARTIAL_POOL_POINT:
                record["frame_digest"] = _digest_on_plan(lib, candidate.value, dims)
            lib.rr_destroy_plan(candidate.value)
        record["counters_after_this_rep"] = _counters(lib, observer)
        record["threads_after_this_rep"] = psutil.Process().num_threads()
        reps_seen.append(record)

    counters_after = _counters(lib, observer)
    digest_after, diagnostics_after = _full_frame_via_backend(dims)

    lib.rr_destroy_plan(observer)
    time.sleep(0.05)  # give a thread that was meant to be joined the chance to be joined
    threads_at_exit = psutil.Process().num_threads()

    failing = point not in SUCCEEDING_POINTS
    codes = sorted({r["code"] for r in reps_seen})
    handles = sorted({r["handle"] for r in reps_seen})
    messages = [r["message"] for r in reps_seen]
    threads_after_reps = [r["threads_after_this_rep"] for r in reps_seen]
    worker_counts = [r.get("this_plan_worker_create_count") for r in reps_seen
                     if "this_plan_worker_create_count" in r]
    partial_digests = [r["frame_digest"] for r in reps_seen if "frame_digest" in r]
    create_delta = (counters_after["process_plan_create_total"]
                    - counters_before["process_plan_create_total"])

    checks = {
        # Without this a pass could just mean the injected step was never reached.
        "fault_fired_every_rep": (point == 0) or all(r["hits"] >= 1 for r in reps_seen),
        "code_matches_contract": codes == [expected_code],
        "handle_state_matches_contract": (
            handles == [0] if failing else all(h > 0 for h in handles)),
        "registry_not_polluted": (
            max(r["registry_size"] for r in reps_seen) == registry_baseline if failing
            else max(r["registry_size"] for r in reps_seen) == registry_baseline + 1),
        "process_counters_not_moved": (
            all(counters_after[k] == counters_before[k] for k in PROCESS_KEYS)
            if failing else create_delta == reps),
        "no_thread_residue": max(threads_after_reps) <= threads_baseline + 2,
        "threads_returned_to_entry": threads_at_exit <= threads_at_entry + 2,
        "frames_still_correct_after_the_run": digest_after == digest_reference,
        # A failure has to say what kind it was: an allocator failure reported as a generic
        # internal error would send a caller chasing the wrong cause.
        "error_message_matches_kind": (not failing) or (
            ("out of memory" in messages[0]) if kind == 0
            else ("injected failure" in messages[0] and "out of memory" not in messages[0])),
        "no_stale_error_on_success": (failing) or all(m == "" for m in messages),
    }
    if point == PARTIAL_POOL_POINT:
        checks.update({
            "partial_pool_reported_fewer_workers": bool(worker_counts)
                and max(worker_counts) >= 1 and min(worker_counts) < 8,
            "partial_pool_scratch_sized_to_its_chunks": all(
                r["this_plan_chunk_scratch_ok"] for r in reps_seen),
            "partial_pool_pixels_match_the_full_pool": bool(partial_digests)
                and set(partial_digests) == {digest_reference},
        })
    return {
        "dll": os.path.abspath(dll),
        "api_version": hex(lib.rr_get_api_version()),
        "point": point,
        "point_name": POINT_NAMES[point],
        "kind": kind,
        "kind_name": KIND_NAMES[kind],
        "expected_code": expected_code,
        "reps": reps,
        "elapsed_seconds": round(time.time() - started, 3),
        "dims": list(dims),
        "observed_codes": codes,
        "observed_handles": handles,
        "worker_create_counts_reported_by_the_plans": sorted(set(worker_counts)),
        "registry_size_baseline": registry_baseline,
        "registry_size_during": {"min": min(r["registry_size"] for r in reps_seen),
                                 "max": max(r["registry_size"] for r in reps_seen)},
        "threads_at_entry": threads_at_entry,
        "threads_baseline_during_loop": threads_baseline,
        "threads_after_each_rep": {"min": min(threads_after_reps),
                                   "max": max(threads_after_reps)},
        "threads_at_exit": threads_at_exit,
        "process_counter_delta_over_loop": {
            k: counters_after[k] - counters_before[k] for k in PROCESS_KEYS},
        "first_error_message": messages[0],
        "digest_reference": digest_reference,
        "digest_after_run": digest_after,
        "frame_after_the_run": diagnostics_after,
        "checks": checks,
        "ok": all(checks.values()),
        "reps_sample": reps_seen[:2] + reps_seen[-2:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dll", required=True)
    ap.add_argument("--point", type=int, required=True)
    ap.add_argument("--kind", type=int, default=0, help="0 throws bad_alloc, 1 runtime_error")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--dims", type=int, nargs=4, default=[640, 360, 320, 180])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = run(args.dll, args.point, args.kind, args.reps, tuple(args.dims))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
