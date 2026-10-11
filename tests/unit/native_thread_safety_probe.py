"""Run the residual bridge thread-safety scenarios against a built DLL and report numbers.

test_native_thread_safety.py is the gate; this is the evidence producer behind it, and it
adds the part a gate cannot do: run the identical scenarios against an *older* build of the
same bridge and show they failed there. Without that comparison "9 passed" only proves the
current code agrees with itself, not that these scenarios catch the bug they were written
for. The old build runs in a subprocess on purpose - against the pre-fix source the destroy
scenario produces a native access violation, and that has to cost a child process, not the
report.

    python tests/unit/native_thread_safety_probe.py --out evidence.json         --control path/to/older/residual_bridge.dll
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline  # noqa: E402

_U8 = ctypes.POINTER(ctypes.c_uint8)
BIG = (3840, 2160, 1920, 1080)


def load(path):
    lib = ctypes.CDLL(path)
    lib.rr_get_api_version.argtypes = []
    lib.rr_get_api_version.restype = ctypes.c_int32
    lib.rr_get_last_error.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.rr_get_last_error.restype = ctypes.c_int32
    lib.rr_create_plan.argtypes = [ctypes.POINTER(ctypes.c_int64)] + [ctypes.c_int32] * 4
    lib.rr_create_plan.restype = ctypes.c_int32
    lib.rr_destroy_plan.argtypes = [ctypes.c_int64]
    lib.rr_destroy_plan.restype = ctypes.c_int32
    i32, i64, f32 = ctypes.c_int32, ctypes.c_int64, ctypes.c_float
    lib.rr_downsample_bgr8.argtypes = [i64, _U8, i32, i32, i64, _U8, i32, i32, i64]
    lib.rr_downsample_bgr8.restype = i32
    lib.rr_reconstruct_bgr8.argtypes = [i64, _U8, i32, i32, i64, _U8, i32, i32, i64,
                                        _U8, i32, i32, i64, f32, _U8, i32, i32, i64]
    lib.rr_reconstruct_bgr8.restype = i32
    fields = [("plan_create_count", i64), ("plan_destroy_count", i64),
              ("scratch_alloc_count", i64), ("scratch_alloc_bytes", i64),
              ("downsample_call_count", i64), ("reconstruct_call_count", i64),
              ("downsample_pixel_count", i64), ("reconstruct_pixel_count", i64),
              ("worker_create_count", i64), ("worker_count", i64),
              ("scratch_bytes", i64), ("axis_plan_count", i64),
              ("process_plan_create_total", i64), ("process_plan_destroy_total", i64),
              ("process_scratch_alloc_total", i64), ("process_scratch_bytes_total", i64),
              ("process_worker_create_total", i64)]
    version = lib.rr_get_api_version()
    shape = fields[:12] if (version >> 16) < 2 else fields

    class Counters(ctypes.Structure):
        _fields_ = shape

    lib.rr_get_counters.argtypes = [i64, ctypes.POINTER(Counters)]
    lib.rr_get_counters.restype = i32
    return lib, Counters, version


def run_scenarios(path, out_path=None):
    """Runs every scenario against one DLL and returns the report. Writes it as it goes,
    because an access violation kills the process where it happens."""
    lib, Counters, version = load(path)
    major = version >> 16
    out = {"dll": os.path.abspath(path), "api_version": f"0x{version & 0xFFFFFFFF:08x}",
           "api_major": major, "checks": {}}

    def record(name, value):
        """Write after every check: a use-after-free kills the process mid-run, and the
        checks that already completed are still evidence."""
        out["checks"][name] = value
        text = json.dumps(out, indent=2, sort_keys=True)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.write(chr(10))

    def ptr(a):
        return a.ctypes.data_as(_U8)

    def create(sw, sh, lw, lh):
        h = ctypes.c_int64(0)
        code = lib.rr_create_plan(ctypes.byref(h), sw, sh, lw, lh)
        assert code == 0, "create failed"
        return h.value

    def destroy(h):
        return lib.rr_destroy_plan(h)

    def error_text():
        buf = ctypes.create_string_buffer(512)
        lib.rr_get_last_error(buf, 512)
        return buf.value.decode("utf-8", "replace")

    def counters(h):
        block = Counters()
        code = lib.rr_get_counters(h, ctypes.byref(block))
        return code, {name: getattr(block, name) for name, _ in Counters._fields_}

    def downsample(h, src, dst):
        return lib.rr_downsample_bgr8(h, ptr(src), src.shape[1], src.shape[0],
                                      ctypes.c_int64(src.strides[0]),
                                      ptr(dst), dst.shape[1], dst.shape[0],
                                      ctypes.c_int64(dst.strides[0]))

    def reconstruct(h, src, low, dlss, dst):
        return lib.rr_reconstruct_bgr8(h, ptr(src), src.shape[1], src.shape[0],
                                       ctypes.c_int64(src.strides[0]),
                                       ptr(low), low.shape[1], low.shape[0],
                                       ctypes.c_int64(low.strides[0]),
                                       ptr(dlss), dlss.shape[1], dlss.shape[0],
                                       ctypes.c_int64(dlss.strides[0]),
                                       ctypes.c_float(1.0),
                                       ptr(dst), dst.shape[1], dst.shape[0],
                                       ctypes.c_int64(dst.strides[0]))

    def scene(sw, sh, lw, lh, seed):
        rng = np.random.default_rng(seed)
        source = rng.integers(0, 256, size=(sh, sw, 3), dtype=np.uint8)
        dlss = rng.integers(0, 256, size=(lh, lw, 3), dtype=np.uint8)
        ref = pipeline.PythonReferenceBackend(sw, sh, lw, lh)
        try:
            want_low = np.zeros((lh, lw, 3), dtype=np.uint8)
            want_recon = np.zeros((sh, sw, 3), dtype=np.uint8)
            ref.prepare_working_color(source, want_low)
            ref.reconstruct(source, want_low, dlss, 1.0, want_recon)
        finally:
            ref.close()
        return source, dlss, want_low, want_recon

    # 1 - frames from several threads on one plan
    sw, sh, lw, lh = 160, 120, 80, 60
    threads, rounds = 6, 12
    h = create(sw, sh, lw, lh)
    scenes = [scene(sw, sh, lw, lh, 100 + i) for i in range(threads)]
    mismatches = []
    bad_codes = []
    barrier = threading.Barrier(threads)

    def worker(index):
        source, dlss, want_low, want_recon = scenes[index]
        got_low = np.full((lh, lw, 3), 3, dtype=np.uint8)
        got_recon = np.full((sh, sw, 3), 3, dtype=np.uint8)
        try:
            barrier.wait(timeout=60)
            for _ in range(rounds):
                if downsample(h, source, got_low) != 0:
                    bad_codes.append(index)
                    return
                if reconstruct(h, source, got_low, dlss, got_recon) != 0:
                    bad_codes.append(index)
                    return
                if not np.array_equal(got_recon, want_recon):
                    mismatches.append(index)
                    return
        finally:
            barrier.abort()

    ws = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for w in ws:
        w.start()
    for w in ws:
        w.join(timeout=180)
    code, final = counters(h)
    destroy(h)
    record("same_plan_concurrent_frames", {
        "threads": threads, "rounds": rounds,
        "wrong_pixel_threads": sorted(set(mismatches)),
        "failed_call_threads": sorted(set(bad_codes)),
        "downsample_call_count": final.get("downsample_call_count"),
        "reconstruct_call_count": final.get("reconstruct_call_count"),
        "expected_calls": threads * rounds,
        "pass": (not mismatches and not bad_codes
                 and final.get("downsample_call_count") == threads * rounds
                 and final.get("reconstruct_call_count") == threads * rounds)})

    # 2 - a small plan while a 4K frame is running on another plan
    bsw, bsh, blw, blh = BIG
    big = create(bsw, bsh, blw, blh)
    small = create(64, 48, 32, 24)
    bsource, bdlss, bwant_low, bwant_recon = scene(bsw, bsh, blw, blh, 5)
    big_low = np.zeros((blh, blw, 3), dtype=np.uint8)
    big_recon = np.zeros((bsh, bsw, 3), dtype=np.uint8)
    ssource, sdlss, swant_low, swant_recon = scene(64, 48, 32, 24, 6)
    s_low = np.zeros((24, 32, 3), dtype=np.uint8)
    s_recon = np.zeros((48, 64, 3), dtype=np.uint8)
    info = {"started": False}

    def run_big():
        t0 = time.perf_counter()
        info["started"] = True
        c1 = downsample(big, bsource, big_low)
        c2 = reconstruct(big, bsource, big_low, bdlss, big_recon)
        info.update(codes=(c1, c2), total_ms=(time.perf_counter() - t0) * 1e3)

    t = threading.Thread(target=run_big)
    t.start()
    deadline = time.perf_counter() + 10.0
    while not info.get("started") and time.perf_counter() < deadline:
        time.sleep(0.001)
    time.sleep(0.005)
    s0 = time.perf_counter()
    rc1 = downsample(small, ssource, s_low)
    rc2 = reconstruct(small, ssource, s_low, sdlss, s_recon)
    small_ms = (time.perf_counter() - s0) * 1e3
    t.join(timeout=180)
    destroy(big)
    destroy(small)
    record("cross_plan_not_blocked", {
        "big_frame_ms": round(info.get("total_ms", -1), 3),
        "small_frame_ms": round(small_ms, 3),
        "ratio": round(small_ms / info["total_ms"], 4) if info.get("total_ms") else None,
        "small_pixels_exact": bool(np.array_equal(s_low, swant_low)
                                   and np.array_equal(s_recon, swant_recon)),
        "pass": bool(info.get("codes") == (0, 0) and (rc1, rc2) == (0, 0)
                     and small_ms < 0.5 * info.get("total_ms", 0))})

    # 3 - destroy arriving in the middle of a frame
    h = create(bsw, bsh, blw, blh)
    source, dlss, want_low, want_recon = scene(bsw, bsh, blw, blh, 9)
    low = np.array(want_low, dtype=np.uint8)
    recon = np.zeros((bsh, bsw, 3), dtype=np.uint8)
    frame = {}

    def run_frame():
        frame["start_ns"] = time.perf_counter_ns()
        try:
            frame["code"] = reconstruct(h, source, low, dlss, recon)
        except BaseException as exc:      # a pre-fix build faults here instead of returning
            frame["thread_exception"] = f"{type(exc).__name__}: {exc}"
            raise
        frame["end_ns"] = time.perf_counter_ns()

    t = threading.Thread(target=run_frame)
    t.start()
    deadline = time.perf_counter() + 10.0
    while "start_ns" not in frame and time.perf_counter() < deadline:
        time.sleep(0.001)
    time.sleep(0.005)
    d0 = time.perf_counter_ns()
    dcode = destroy(h)
    d1 = time.perf_counter_ns()
    t.join(timeout=180)
    # None rather than a fake number: if the call was killed by a native fault it has no
    # duration, and "0.0 ms" would read like a very fast frame.
    frame_ms = (None if "end_ns" not in frame
                else (frame["end_ns"] - frame["start_ns"]) / 1e6)
    destroy_ms = (d1 - d0) / 1e6
    result = {"frame_ms": None if frame_ms is None else round(frame_ms, 3),
              "destroy_ms": round(destroy_ms, 3),
              "inflight_call_code": frame.get("code"), "destroy_code": dcode,
              "frame_thread_returned": "end_ns" in frame,
              "pixels_exact_after_destroy": bool(np.array_equal(recon, want_recon)),
              "pass": bool(frame.get("code") == 0 and dcode == 0
                           and frame_ms is not None and np.array_equal(recon, want_recon)
                           and destroy_ms > 0.3 * frame_ms)}
    if "thread_exception" in frame:
        result["frame_thread_exception"] = frame["thread_exception"]
    record("destroy_waits_for_inflight_call", result)

    # 4 - last error text per calling thread
    h = create(64, 48, 32, 24)
    a_failed = threading.Event()
    b_failed = threading.Event()
    a_read = threading.Event()
    texts = {}

    def bad_handle_path():
        lib.rr_get_counters(424242, ctypes.byref(Counters()))
        a_failed.set()
        b_failed.wait(timeout=60)
        texts["thread_a"] = error_text()
        a_read.set()

    def wrong_dims_path():
        a_failed.wait(timeout=60)
        src = np.zeros((48, 64, 3), dtype=np.uint8)
        dst = np.zeros((16, 16, 3), dtype=np.uint8)
        downsample(h, src, dst)
        b_failed.set()
        a_read.wait(timeout=60)
        texts["thread_b"] = error_text()

    ta = threading.Thread(target=bad_handle_path)
    tb = threading.Thread(target=wrong_dims_path)
    ta.start()
    tb.start()
    ta.join(timeout=120)
    tb.join(timeout=120)
    destroy(h)
    record("last_error_per_thread", {
        "thread_a": texts.get("thread_a"), "thread_b": texts.get("thread_b"),
        "pass": bool("invalid plan handle" in texts.get("thread_a", "")
                     and "called with" not in texts.get("thread_a", "")
                     and "plan is 64x48" in texts.get("thread_b", ""))})

    # 5 - a probe plan must not move another plan's lifecycle counters
    live = [create(64, 48, 32, 24), create(96, 72, 48, 36), create(128, 96, 64, 48)]
    base = [counters(h)[1] for h in live]
    probe = create(16, 16, 8, 8)
    destroy(probe)
    after = [counters(h)[1] for h in live]
    for h in live:
        destroy(h)
    per_plan_stable = all(a.get("plan_create_count") == b.get("plan_create_count")
                          and a.get("scratch_alloc_count") == b.get("scratch_alloc_count")
                          and a.get("worker_create_count") == b.get("worker_create_count")
                          for a, b in zip(base, after))
    record("probe_does_not_pollute_per_plan_counters", {
        "before_first_plan": {k: base[0][k] for k in
                              ("plan_create_count", "scratch_alloc_count",
                               "worker_create_count", "scratch_bytes")},
        "after_first_plan": {k: after[0][k] for k in
                             ("plan_create_count", "scratch_alloc_count",
                              "worker_create_count", "scratch_bytes")},
        "per_plan_create_counts": [a.get("plan_create_count") for a in after],
        "pass": bool(per_plan_stable and all(a["plan_create_count"] == 1 for a in after))})

    # 6 - concurrent create/frame/destroy must not lose process-wide counts
    h0 = create(64, 48, 32, 24)
    start = counters(h0)[1]
    threads2, per = 6, 8
    problems = []
    go = threading.Event()

    def churn(index):
        go.wait(timeout=120)
        for _ in range(per):
            hh = create(64, 48, 32, 24)
            src = np.zeros((48, 64, 3), dtype=np.uint8)
            dst = np.zeros((24, 32, 3), dtype=np.uint8)
            if downsample(hh, src, dst) != 0:
                problems.append(f"thread {index}: frame rejected")
                return
            code, blk = counters(hh)
            destroy(hh)
            if code != 0 or blk["downsample_call_count"] != 1:
                problems.append(f"thread {index}: code={code} calls="
                                f"{blk.get('downsample_call_count')}")
                return

    ws = [threading.Thread(target=churn, args=(i,)) for i in range(threads2)]
    for w in ws:
        w.start()
    go.set()
    for w in ws:
        w.join(timeout=180)
    code, end = counters(h0)
    destroy(h0)
    churned = threads2 * per
    # Under api 1 these four fields were the process totals; under api 2 they are per plan
    # and the process totals have their own names. Read whichever scope applies so the
    # comparison measures the same property in both builds.
    scope = ("process_plan_create_total", "process_plan_destroy_total",
             "process_scratch_alloc_total", None) if major >= 2 else (
            "plan_create_count", "plan_destroy_count", "scratch_alloc_count",
            "worker_create_count")
    deltas = {}
    for key in scope:
        if key and key in end and key in start:
            deltas[key] = end[key] - start[key]
    per_plan = {k: end[k] for k in ("plan_create_count", "scratch_alloc_count") if k in end}
    record("concurrent_lifecycle_counts", {
        "plans_churned": churned, "process_scope_deltas": deltas,
        "final_plan_own_counts": per_plan, "problems": problems[:3],
        "pass": bool(not problems
                     and deltas.get(scope[0]) == churned
                     and deltas.get(scope[1]) == churned
                     and deltas.get(scope[2]) == churned)})

    return out


def _amend(path, extra):
    """Add fields to a report that record() already wrote, ignoring a missing file."""
    if not path or path == os.devnull:
        return
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return
    data.update(extra)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write(chr(10))


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="", help="write the merged report here")
    parser.add_argument("--control", default="",
                        help="an older residual_bridge.dll to run the same scenarios against")
    parser.add_argument("--control-note", default="",
                        help="how the control DLL was produced, recorded verbatim")
    parser.add_argument("--single", default="",
                        help="internal: run the scenarios against this DLL and exit")
    parser.add_argument("--json", default="", help="internal: report path for --single")
    args = parser.parse_args(argv)

    if args.single:
        run_scenarios(args.single, args.json or None)
        # A bridge call that never returns is one of the ways the pre-fix pool fails, and
        # CPython's exit path waits forever for the Python thread still inside it. Every
        # number worth having is already on disk by then, so say how many threads are
        # stuck and leave without them.
        stuck = sorted(t.name for t in threading.enumerate()
                       if t is not threading.main_thread() and not t.daemon)
        _amend(args.json, {"non_daemon_threads_still_running": stuck,
                           "every_scenario_call_returned": not stuck})
        sys.stdout.flush()
        os._exit(0)

    current = run_scenarios(os.path.join(_ROOT, "native", "residual_bridge.dll"),
                            (args.out + ".current.json") if args.out else None)
    failing = sorted(name for name, value in current["checks"].items() if not value["pass"])
    stuck = sorted(t.name for t in threading.enumerate()
                   if t is not threading.main_thread() and not t.daemon)
    current["non_daemon_threads_still_running"] = stuck
    current["every_scenario_call_returned"] = not stuck
    report = {"current_build": current,
              "current_build_failing": failing,
              "scenario_summary": {
                  name: {k: v for k, v in value.items() if k == "pass"}
                  for name, value in current["checks"].items()}}
    if args.control:
        # The scenarios are deterministic against the current build but not against the
        # pre-fix one: a use-after-free either returns a wrong answer or kills the process
        # where it happens. Both are evidence, so retry and keep the run that recorded the
        # most checks, along with what each attempt cost.
        tmp = (args.out or "native_thread_safety_probe") + ".control.json"
        attempts, best = [], None
        for _ in range(3):
            try:
                os.remove(tmp)
            except OSError:
                pass
            timed_out = False
            try:
                proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                                       "--single", args.control, "--json", tmp],
                                      capture_output=True, text=True, timeout=900,
                                      cwd=_ROOT)
                returncode, stderr = proc.returncode, proc.stderr
            except subprocess.TimeoutExpired as exc:
                # A child that cannot be reaped is itself a finding, and it must not cost
                # the whole report.
                timed_out = True
                raw = exc.stderr or ""
                returncode, stderr = None, raw.decode("utf-8", "replace") \
                    if isinstance(raw, bytes) else raw
            captured = None
            try:
                with open(tmp, encoding="utf-8") as handle:
                    captured = json.load(handle)
            except (OSError, ValueError):
                captured = None
            attempts.append({
                "process_returncode": returncode,
                "process_timed_out": timed_out,
                "checks_recorded": 0 if not captured else len(captured.get("checks", {})),
                "stderr_tail": [ln for ln in (stderr or "").strip().splitlines()
                                if ln.strip()][-4:],
            })
            if captured and (best is None
                             or len(captured.get("checks", {})) > len(best[0].get("checks", {}))):
                best = (captured, attempts[-1])
            if captured and captured.get("checks") and returncode == 0:
                break           # a clean run that recorded everything: no need for more
        report["control_build"] = {
            "dll": os.path.abspath(args.control),
            "how_it_was_built": args.control_note,
            "attempts": attempts,
            "report": best[0] if best else None,
            "note": "a scenario that reports pass=false here failed against the older "
                    "build, which is the point: it shows the current scenarios catch the "
                    "bug that was fixed. A pre-fix build faults natively, so an attempt "
                    "can die part-way through; checks_recorded says how far it got."}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write(chr(10))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
