"""Gate: every way out of rr_create_plan has to be a return code, not a side effect.

Creating a plan allocates filter taps, a worker pool and three FP32 buffers before it
publishes anything. Each of those steps can fail, and until now the failure handling was
itself the bug: a throw after the pool existed released the pool by hand and then again from
the destructor, the process-wide counters were raised before the registry insert, and an
allocation failure anywhere else could throw out of an extern "C" function - which on x64
Windows ends the process with no code for the caller to read.

Nothing here relies on the machine running out of memory. The failures come from
native/residual_bridge.cpp itself, compiled with RR_TEST_FAULT_INJECTION so the create path
throws at a named step, and every scenario runs in a child process: a corrupted heap does
not always fail an assertion, but it does stop the process from exiting cleanly, and the
exit code is the one piece of evidence a crashing test suite cannot fake.

Skipped when the C++ toolchain is unavailable; when the test DLL is present every test below
has to pass.

The last group of tests holds the shipped binary to its own declaration: residual_bridge.h is
read as the statement of the ABI, and the DLL and residual_native._Counters are compared
against it, because every assertion above depends on the library in front of us being the one
this tree built.
"""

import ctypes
import json
import os
import re
import subprocess
import sys

import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NATIVE_DIR = os.path.join(_ROOT, "native")
PRODUCTION_DLL = os.path.join(NATIVE_DIR, "residual_bridge.dll")
BUILD_SCRIPT = os.path.join(NATIVE_DIR, "build_residual_test.bat")
SOURCE = os.path.join(NATIVE_DIR, "residual_bridge.cpp")
HEADER = os.path.join(NATIVE_DIR, "residual_bridge.h")
PROBE = os.path.join(_UNIT_DIR, "residual_fault_probe.py")
TEST_DLL = os.path.join(_ROOT, "scratch", "native_test", "residual_bridge_test.dll")

# At least 50 repeated failures per the round's requirement: one failure passing proves the
# handler ran once, fifty passing prove nothing accumulates across them.
REPS = 50
DIMS = (640, 360, 320, 180)

RR_OK = 0
RR_ERR_OUT_OF_MEMORY = -9
RR_ERR_INTERNAL = -7

CONTROL = 0
AFTER_PLAN_CREATE = 1
DURING_AXIS_PLAN = 2
DURING_POOL_THREAD_CREATE = 3
AFTER_ROW_POOL_CREATE = 4
DURING_SCRATCH_ALLOC = 5
BEFORE_REGISTRY_PUBLISH = 6
DURING_REGISTRY_PUBLISH = 7

BAD_ALLOC = 0
RUNTIME_ERROR = 1

ALL_FAILURE_POINTS = (AFTER_PLAN_CREATE, DURING_AXIS_PLAN, AFTER_ROW_POOL_CREATE,
                      DURING_SCRATCH_ALLOC, BEFORE_REGISTRY_PUBLISH, DURING_REGISTRY_PUBLISH)

_TEST_DLL_SKIP_REASON = None


def _header_text():
    with open(HEADER, encoding="utf-8") as fh:
        return fh.read()


def header_api_version():
    """The ABI version the source declares, read from the header instead of repeated here.

    Every version assertion in this file used a literal, so a minor bump failed them while
    the two DLLs still agreed with each other - the tree had moved on, not the contract."""
    match = re.search(r"^#define\s+RR_API_VERSION\s+(0x[0-9A-Fa-f]+)", _header_text(),
                      re.MULTILINE)
    assert match, "native/residual_bridge.h no longer declares RR_API_VERSION"
    return int(match.group(1), 16)


def header_counter_fields():
    """The rr_counters members as (c type, name) in declaration order."""
    block = re.search(r"^typedef struct rr_counters \{\n(.*?)^\} rr_counters;",
                      _header_text(), re.MULTILINE | re.DOTALL)
    assert block, "native/residual_bridge.h no longer declares typedef struct rr_counters"
    return [(m.group(1), m.group(2)) for m in
            re.finditer(r"^\s*(\w+_t)\s+(\w+);", block.group(1), re.MULTILINE)]


def header_exports():
    """The entry points the header declares, in declaration order.

    Only real declarations match: the test-only exports are named in a prose comment, whose
    lines do not start with RR_API."""
    return [m.group(1) for m in
            re.finditer(r"^RR_API int32_t (\w+)\(", _header_text(), re.MULTILINE)]


# What each C type in rr_counters is mirrored by in ctypes. A type missing from here is a
# layout question this file cannot answer, so it is reported as one.
HEADER_TYPES = {"int64_t": ctypes.c_int64}


def _build_test_dll():
    """Compile the fault-injection DLL from the same source the production DLL is built from.

    Returns the path, or None with a reason recorded if the toolchain is not on this machine.
    Rebuilt whenever the source is newer than the DLL, so a test can never pass against a
    binary that predates the code it is meant to be interrupting."""
    global _TEST_DLL_SKIP_REASON
    if os.path.exists(TEST_DLL) and os.path.getmtime(TEST_DLL) >= os.path.getmtime(SOURCE):
        return TEST_DLL
    if not os.path.exists(BUILD_SCRIPT):
        _TEST_DLL_SKIP_REASON = f"{BUILD_SCRIPT} is missing"
        return None
    os.makedirs(os.path.dirname(TEST_DLL), exist_ok=True)
    completed = subprocess.run(["cmd", "/c", BUILD_SCRIPT, os.path.dirname(TEST_DLL)],
                               capture_output=True, text=True, errors="replace", timeout=600)
    if completed.returncode != 0 or not os.path.exists(TEST_DLL):
        _TEST_DLL_SKIP_REASON = (
            f"fault-injection build failed with {completed.returncode}: "
            + (completed.stdout + completed.stderr)[-2000:])
        return None
    return TEST_DLL


@pytest.fixture(scope="session")
def fault_dll():
    path = _build_test_dll()
    if path is None:
        pytest.skip(_TEST_DLL_SKIP_REASON or "no C++ toolchain for the fault-injection build")
    return path


_RUNS = {}


def _run(fault_dll, point, kind=BAD_ALLOC, reps=REPS):
    key = (point, kind, reps)
    if key in _RUNS:
        return _RUNS[key]
    out = os.path.join(_ROOT, "scratch", "native_test", f"probe_{point}_{kind}_{reps}.json")
    completed = subprocess.run(
        [sys.executable, PROBE, "--dll", fault_dll, "--point", str(point),
         "--kind", str(kind), "--reps", str(reps),
         "--dims", *[str(d) for d in DIMS], "--out", out],
        capture_output=True, text=True, errors="replace", timeout=900)
    stdout = (completed.stdout + completed.stderr)[-2000:]
    if not os.path.exists(out):
        pytest.fail(
            f"the failure probe produced no report for point {point} kind {kind}; "
            f"exit code {completed.returncode} is what the library left behind. "
            f"An escaping C++ exception or a heap fault looks exactly like this.\n{stdout}")
    with open(out, encoding="utf-8") as fh:
        report = json.load(fh)
    report["process_exit_code"] = completed.returncode
    report["process_output"] = stdout
    _RUNS[key] = report
    return report


def _matrix(fault_dll):
    """Every failure point, both exception kinds, plus the control and the pool point."""
    runs = {(CONTROL, BAD_ALLOC): _run(fault_dll, CONTROL)}
    for point in ALL_FAILURE_POINTS:
        for kind in (BAD_ALLOC, RUNTIME_ERROR):
            runs[(point, kind)] = _run(fault_dll, point, kind)
    runs[(DURING_POOL_THREAD_CREATE, BAD_ALLOC)] = _run(fault_dll, DURING_POOL_THREAD_CREATE)
    return runs


def _failed_checks(report):
    return [name for name, passed in report["checks"].items() if not passed]


def _assert_clean(report):
    """The child has to exit 0 on its own terms, not merely have written a report."""
    assert report["process_exit_code"] == 0, (
        f"point {report['point_name']} kind {report['kind_name']}: the probe exited "
        f"{report['process_exit_code']} (negative values on Windows mean the process was "
        f"taken down rather than returning)\n{report['process_output']}")
    assert not _failed_checks(report), (
        f"point {report['point_name']} kind {report['kind_name']}: "
        f"{_failed_checks(report)}\n"
        + json.dumps({k: report[k] for k in (
            "observed_codes", "observed_handles", "registry_size_during",
            "process_counter_delta_over_loop", "threads_at_entry", "threads_at_exit",
            "first_error_message")}, indent=2))


def test_create_failure_after_pool_creation_returns_error_without_crash(fault_dll):
    """The old double-free window: the pool is live, the next allocation throws.

    A return code plus a clean exit is the whole claim - the alternative was a process that
    freed the pool once here and again in the destructor."""
    report = _run(fault_dll, AFTER_ROW_POOL_CREATE)
    _assert_clean(report)
    assert report["observed_codes"] == [RR_ERR_OUT_OF_MEMORY]
    assert report["checks"]["fault_fired_every_rep"]
    assert report["expected_code"] == RR_ERR_OUT_OF_MEMORY


def test_create_failure_after_pool_creation_leaves_handle_zero(fault_dll):
    report = _run(fault_dll, AFTER_ROW_POOL_CREATE)
    _assert_clean(report)
    assert report["observed_handles"] == [0], "a failed create must not hand back a handle"
    assert report["registry_size_during"]["max"] == report["registry_size_baseline"], (
        "the registry kept an entry for a plan nobody was given a handle for")


def test_create_failure_after_pool_creation_leaves_no_worker_threads(fault_dll):
    """Eight workers are started before the failure and must all be joined by the time the
    call returns. Fifty repetitions would otherwise turn a per-call leak into 400 threads."""
    report = _run(fault_dll, AFTER_ROW_POOL_CREATE)
    _assert_clean(report)
    assert report["checks"]["no_thread_residue"]
    assert report["threads_after_each_rep"]["max"] <= report["threads_baseline_during_loop"] + 2
    assert report["reps"] == REPS


def test_create_failure_at_every_step_leaves_no_worker_threads(fault_dll):
    for point in ALL_FAILURE_POINTS + (DURING_POOL_THREAD_CREATE,):
        for kind in (BAD_ALLOC, RUNTIME_ERROR):
            if point == DURING_POOL_THREAD_CREATE and kind == RUNTIME_ERROR:
                continue
            report = _run(fault_dll, point, kind)
            _assert_clean(report)
            assert report["threads_after_each_rep"]["max"] <= (
                report["threads_baseline_during_loop"] + 2), (
                f"point {report['point_name']} kind {report['kind_name']} left threads behind")


def test_create_failure_before_registry_publish_leaves_no_handle(fault_dll):
    """A complete plan that is never published: the handle stays zero, the registry never
    saw it, and the caller's own observer plan still reports the same totals."""
    report = _run(fault_dll, BEFORE_REGISTRY_PUBLISH)
    _assert_clean(report)
    assert report["observed_handles"] == [0]
    assert report["observed_codes"] == [RR_ERR_OUT_OF_MEMORY]
    assert report["registry_size_during"]["max"] == report["registry_size_baseline"]


def test_registry_publish_failure_leaves_no_ghost_handle(fault_dll):
    """The insert itself throwing is the one step where a ghost entry was possible."""
    report = _run(fault_dll, DURING_REGISTRY_PUBLISH)
    _assert_clean(report)
    assert report["observed_handles"] == [0]
    assert report["registry_size_during"]["max"] == report["registry_size_baseline"], (
        "an entry outlived the failed create that made it")


def test_create_failure_does_not_move_process_counters(fault_dll):
    report = _run(fault_dll, AFTER_ROW_POOL_CREATE)
    _assert_clean(report)
    assert report["process_counter_delta_over_loop"] == {
        "process_plan_create_total": 0,
        "process_plan_destroy_total": 0,
        "process_scratch_alloc_total": 0,
        "process_scratch_bytes_total": 0,
        "process_worker_create_total": 0,
    }, "a create that failed was counted as if it had happened"


def test_no_failure_point_moves_process_counters(fault_dll):
    for point in ALL_FAILURE_POINTS:
        report = _run(fault_dll, point, BAD_ALLOC)
        _assert_clean(report)
        assert all(v == 0 for v in report["process_counter_delta_over_loop"].values()), (
            f"{report['point_name']} moved {report['process_counter_delta_over_loop']}")


def test_control_without_injection_moves_counters_as_expected(fault_dll):
    """Negative control: with nothing armed the same loop creates and destroys plans and the
    counters do move. Without this the zero-delta assertions above could pass because the
    harness never reached the code at all."""
    report = _run(fault_dll, CONTROL)
    _assert_clean(report)
    assert report["observed_codes"] == [RR_OK]
    assert report["process_counter_delta_over_loop"]["process_plan_create_total"] == REPS
    assert report["process_counter_delta_over_loop"]["process_plan_destroy_total"] == REPS
    assert report["process_counter_delta_over_loop"]["process_worker_create_total"] >= REPS


def test_repeated_create_failure_is_stable(fault_dll):
    report = _run(fault_dll, AFTER_ROW_POOL_CREATE)
    _assert_clean(report)
    assert report["reps"] == REPS
    assert report["observed_codes"] == [RR_ERR_OUT_OF_MEMORY], (
        "the same injected failure reported different codes across repetitions")
    assert report["observed_handles"] == [0]
    assert report["checks"]["threads_returned_to_entry"]


def test_success_after_injected_failure_still_works(fault_dll):
    """Disarm and carry on: the library must still create a plan, run a frame and produce the
    pixels it produced before anything was ever interrupted."""
    report = _run(fault_dll, AFTER_ROW_POOL_CREATE)
    _assert_clean(report)
    assert report["checks"]["frames_still_correct_after_the_run"]
    assert report["digest_after_run"] == report["digest_reference"]
    diag = report["frame_after_the_run"]
    assert diag["this_plan_create_count"] == 1
    assert diag["this_plan_scratch_alloc_count"] == 1
    assert diag["downsample_call_count"] == 1
    assert diag["reconstruct_call_count"] == 1


def test_allocation_failure_is_reported_as_out_of_memory_not_internal(fault_dll):
    report = _run(fault_dll, DURING_SCRATCH_ALLOC, BAD_ALLOC)
    _assert_clean(report)
    assert report["observed_codes"] == [RR_ERR_OUT_OF_MEMORY]
    assert "out of memory" in report["first_error_message"]


def test_other_exception_is_reported_as_internal_with_its_own_text(fault_dll):
    """An exception that is not an allocator failure must not be dressed up as one, or the
    next person to hit it will go looking for a memory problem."""
    report = _run(fault_dll, DURING_AXIS_PLAN, RUNTIME_ERROR)
    _assert_clean(report)
    assert report["observed_codes"] == [RR_ERR_INTERNAL]
    assert "injected failure" in report["first_error_message"]
    assert "out of memory" not in report["first_error_message"]


def test_pool_thread_creation_failure_leaves_a_usable_pool(fault_dll):
    """RowPool's constructor absorbs a thread that could not be started.

    Proof, not a comment: the plan reports fewer workers than were requested, its scratch is
    sized to the chunks it actually has, and the frame it computes matches the eight-worker
    run bit for bit."""
    report = _run(fault_dll, DURING_POOL_THREAD_CREATE)
    _assert_clean(report)
    assert report["observed_codes"] == [RR_OK]
    assert report["checks"]["partial_pool_reported_fewer_workers"]
    assert report["checks"]["partial_pool_scratch_sized_to_its_chunks"]
    assert report["checks"]["partial_pool_pixels_match_the_full_pool"]
    assert report["worker_create_counts_reported_by_the_plans"] == [1]
    assert report["process_counter_delta_over_loop"]["process_worker_create_total"] == REPS


def test_create_never_lets_cpp_exception_cross_c_abi(fault_dll):
    """Every failure point, with both exception types, in one pass.

    An exception leaving an extern "C" function on x64 Windows calls std::terminate, so the
    child dies without a report and the exit code is a fault status rather than 0. The matrix
    is the test: each cell has to have exited on its own and returned the code its handler
    maps that exception to."""
    matrix = _matrix(fault_dll)
    cells = []
    for (point, kind), report in sorted(matrix.items()):
        assert report["process_exit_code"] == 0, (
            f"{report['point_name']} / {report['kind_name']}: the probe exited "
            f"{report['process_exit_code']} instead of returning; an exception that crossed "
            f"the C ABI does exactly this\n{report['process_output']}")
        assert not _failed_checks(report), (
            f"{report['point_name']} / {report['kind_name']}: {_failed_checks(report)}")
        expected = RR_OK if point in (CONTROL, DURING_POOL_THREAD_CREATE) else (
            RR_ERR_OUT_OF_MEMORY if kind == BAD_ALLOC else RR_ERR_INTERNAL)
        assert report["observed_codes"] == [expected]
        cells.append(f"{report['point_name']}/{report['kind_name']}")
    assert len(cells) == len(ALL_FAILURE_POINTS) * 2 + 2


def test_production_dll_exports_no_fault_injection_hooks():
    """The shipped binary must not let a user arm an allocation failure, so the hooks have to
    be a build-time addition rather than a dormant feature of the same DLL."""
    if not os.path.exists(PRODUCTION_DLL):
        pytest.skip("production residual_bridge.dll is not built")
    lib = ctypes.CDLL(PRODUCTION_DLL)
    for name in ("rr_test_arm_fault", "rr_test_fault_hits", "rr_test_clear_fault",
                 "rr_test_registry_size"):
        with pytest.raises(AttributeError):
            getattr(lib, name)
    with open(PRODUCTION_DLL, "rb") as fh:
        blob = fh.read()
    assert b"rr_test_" not in blob, (
        "a test-only export name is present in the production DLL")
    assert b"RR_TEST_FAULT_INJECTION" not in blob


def test_shipped_dll_is_the_binary_this_header_describes():
    """The other assertions here read the header, so a stale DLL would be tested as though it
    were the new one: the version has to match and the exports it declares have to be there.

    This is the check that fails when someone edits residual_bridge.cpp and keeps benchmarking
    the DLL from before the edit."""
    if not os.path.exists(PRODUCTION_DLL):
        pytest.skip("production residual_bridge.dll is not built")
    lib = ctypes.CDLL(PRODUCTION_DLL)
    declared = header_api_version()
    reported = lib.rr_get_api_version()
    assert reported == declared, (
        f"residual_bridge.dll reports 0x{reported:08x} but the header declares "
        f"0x{declared:08x}; rebuild native/residual_bridge.dll from this tree")
    missing = [name for name in header_exports() if not hasattr(lib, name)]
    assert not missing, (
        f"the header declares exports the shipped DLL does not offer: {missing}")


def test_python_counter_mirror_matches_the_header_struct():
    """residual_native reads rr_counters by position, so a field added in the wrong place
    shifts every later value into the wrong name - a wrong number with no error on it.

    Names and types are compared, not just the count: an int32_t in place of an int64_t
    reads half of the next field."""
    import residual_native

    expected = header_counter_fields()
    unknown = sorted({c_type for c_type, _name in expected if c_type not in HEADER_TYPES})
    assert not unknown, (
        f"rr_counters gained a type this mirror cannot check: {unknown}; add it to HEADER_TYPES "
        f"and mirror it in residual_native._Counters")
    got = list(residual_native._Counters._fields_)
    assert len(got) == len(expected), (
        f"residual_native._Counters has {len(got)} fields, rr_counters has {len(expected)}")
    for i, ((name, ctype), (c_type, want_name)) in enumerate(zip(got, expected)):
        assert name == want_name, f"field {i}: python {name}, header {want_name}"
        assert ctype is HEADER_TYPES[c_type], (
            f"{want_name}: python reads it as {ctype.__name__}, the header declares {c_type}")
    assert ctypes.sizeof(residual_native._Counters) == 8 * len(expected), (
        "rr_counters is all 64-bit fields; a different size means padding, and padding is a "
        "different struct")


def test_the_harness_detects_the_original_double_free(fault_dll):
    """A suite that cannot fail is not measuring anything, so put the defect back and check.

    The shipped create path was transplanted with the pre-fix shape - hand-free the pool in
    the handler, let the destructor free it again - built from the same source with the same
    flags, and run through the same probe at the same fault point. The fixed library returns
    an error code and exits cleanly; this one has to be taken down by the heap manager.

    The anchors below are deliberately exact. If they stop matching, this test fails rather
    than silently skipping, because then the guarantee that the probe catches a double free
    is no longer established for whatever the create path became. Re-anchor it against the
    new handler and the new pool member before changing anything else."""
    source = open(SOURCE, encoding="utf-8").read()
    hoist = """    try {
        auto plan = std::make_shared<Plan>();"""
    handler = """    } catch (const std::bad_alloc &) {
        /* Reported as what it is: an allocation the plan needed could not be satisfied."""
    assert source.count(hoist) == 1, (
        "the create path no longer declares the plan inside its try block; re-anchor "
        "test_the_harness_detects_the_original_double_free against the current shape")
    assert source.count(handler) == 1, (
        "the create path no longer has a single bad_alloc handler; re-anchor "
        "test_the_harness_detects_the_original_double_free against the current shape")
    handler_head, handler_note = handler.split("\n", 1)
    broken = source.replace(hoist, """    std::shared_ptr<Plan> plan;
    try {
        plan = std::make_shared<Plan>();""")
    broken = broken.replace(handler, handler_head + """
        delete plan->pool.get();   // transplanted defect: freed again by ~Plan
""" + handler_note, 1)

    build_dir = os.path.dirname(fault_dll)
    variant = os.path.join(build_dir, "residual_bridge_doublefree.cpp")
    variant_dll = os.path.join(build_dir, "residual_bridge_doublefree.dll")
    with open(variant, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(broken)
    bat = os.path.join(build_dir, "build_doublefree.bat")
    with open(bat, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write(
            "@echo off\r\n"
            f'call "E:\\Visual Studio\\VC\\Auxiliary\\Build\\vcvars64.bat" || exit /b 1\r\n'
            f'cd /d "{build_dir}"\r\n'
            "cl.exe /nologo /LD /O2 /Oy- /EHsc /std:c++17 /DRESIDUAL_BRIDGE_EXPORTS "
            f'/DRR_TEST_FAULT_INJECTION /I"{NATIVE_DIR}" residual_bridge_doublefree.cpp '
            "/Fe:residual_bridge_doublefree.dll /link /DLL || exit /b 1\r\n")
    compiled = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True,
                              errors="replace", timeout=600)
    if compiled.returncode != 0 or not os.path.exists(variant_dll):
        pytest.fail("the regression variant did not build: "
                    + (compiled.stdout + compiled.stderr)[-1500:])

    report_path = os.path.join(build_dir, "probe_doublefree.json")
    if os.path.exists(report_path):
        os.remove(report_path)
    completed = subprocess.run(
        [sys.executable, PROBE, "--dll", variant_dll, "--point", str(AFTER_ROW_POOL_CREATE),
         "--kind", str(BAD_ALLOC), "--reps", "20",
         "--dims", *[str(d) for d in DIMS], "--out", report_path],
        # Windows crash diagnostics can contain bytes outside the console code
        # page. The regression verdict uses exit status and the JSON file;
        # unreadable diagnostic bytes must not kill the pipe-reader thread.
        capture_output=True, text=True, errors="replace", timeout=900)
    died = completed.returncode != 0 and not os.path.exists(report_path)
    assert died, (
        f"the transplanted double free survived: exit {completed.returncode}, report written "
        f"{os.path.exists(report_path)}. Either the probe no longer reaches the pool or the "
        f"failure-path tests below are not measuring what they claim to measure.")


def test_fault_injection_dll_is_built_from_the_shipped_source(fault_dll):
    """One source file, two builds: the test DLL has to carry the injected failure and the
    production DLL must not, which is what shows they came from the same tree with only the
    macro switched."""
    with open(fault_dll, "rb") as fh:
        test_blob = fh.read()
    with open(PRODUCTION_DLL, "rb") as fh:
        production_blob = fh.read()
    assert b"injected failure" in test_blob, (
        "the test DLL was built without RR_TEST_FAULT_INJECTION")
    assert b"injected failure" not in production_blob
    report = _run(fault_dll, AFTER_ROW_POOL_CREATE)
    _assert_clean(report)
    assert report["api_version"] == hex(header_api_version()), (
        "the test build reports a different ABI than the shipped one")
