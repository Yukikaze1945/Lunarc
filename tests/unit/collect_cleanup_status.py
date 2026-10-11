"""Assemble environment.json and cleanup_status.json for one cleanup round.

    python tests/unit/collect_cleanup_status.py --out perf_results/<sha>/cleanup

Reads what the gates already wrote and derives every status from it. Nothing here runs a
measurement: a file that is missing, unreadable, or shows a failed or skipped gate
becomes NOT_READY, so the status cannot be talked up by whoever edits this script. The
one thing it does measure is the environment itself, which is what environment.json is.

The status vocabulary is fixed by the task: DONE_PROVEN or NOT_READY.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GATE_SECTIONS = {
    "gate2_native_numeric": "GATE 2",
    "gate3_thread_safety": "GATE 3",
    "gate4_hardware_timing_and_avx2": "GATE 4",
    "gate5_full_pytest": "GATE 5",
    "gate_claims_guard": "GATE claims",
}
# The artifact directory is untracked by definition while it is being written, so it must
# not be reported as an unclean part of the code under test.
NON_CODE_PREFIXES = ("perf_results/",)


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _sha256(path):
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True,
                             timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _dirty_paths():
    """Unclean tracked paths, ignoring the artifact directory being written."""
    return [line for line in _git("status", "--porcelain").splitlines()
            if line.strip() and not line[3:].strip().startswith(NON_CODE_PREFIXES)]


def _cl_line(bat_path):
    """The cl.exe invocation a build script runs, so the artifact names the real command."""
    try:
        with open(bat_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "cl.exe" in line.lower():
                    return line.strip()
    except OSError:
        pass
    return None


def parse_transcript(path):
    """{section: {'pass': bool, 'summary': str}} from a gates transcript with headers."""
    report = {name: {"pass": False, "summary": "transcript missing"}
              for name in GATE_SECTIONS}
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return report
    for name, marker in GATE_SECTIONS.items():
        start = text.find("== " + marker)
        if start < 0:
            continue
        end = len(text)
        for other in GATE_SECTIONS.values():
            pos = text.find("== " + other, start + len(marker))
            if 0 <= pos < end:
                end = pos
        block = text[start:end]
        passed = [int(m.group(1)) for m in re.finditer(r"(\d+) passed", block)]
        failed = sum(int(m.group(1)) for m in re.finditer(r"(\d+) failed", block))
        skipped = sum(int(m.group(1)) for m in re.finditer(r"(\d+) skipped", block))
        errors = sum(int(m.group(1)) for m in re.finditer(r"(\d+) error", block))
        last = [ln.strip() for ln in block.splitlines() if " passed" in ln
                or " failed" in ln or " no tests ran" in ln]
        report[name] = {
            "pass": bool(passed) and not failed and not skipped and not errors,
            "tests_passed": sum(passed), "tests_failed": failed, "tests_skipped": skipped,
            "summary": last[-1] if last else block.strip().splitlines()[-1][:200],
        }
    return report


def _checks(doc):
    return ((doc or {}).get("current_build") or {}).get("checks") or {}


def derive(out_dir):
    """One status per item the task lists, each naming the evidence it read."""
    gates = parse_transcript(os.path.join(out_dir, "pytest_output.txt"))
    thread = _read_json(os.path.join(out_dir, "native_thread_safety.json"))
    nvof = _read_json(os.path.join(out_dir, "nvof_timing_semantics.json"))
    avx2 = _read_json(os.path.join(out_dir, "avx2_probe.json"))
    bench = _read_json(os.path.join(out_dir, "benchmark_quick_AD.json"))

    checks = _checks(thread)
    control_checks = ((thread or {}).get("control_build") or {}).get("report") or {}
    control_checks = control_checks.get("checks") or {}
    control_failing = sorted(k for k, v in control_checks.items() if not v.get("pass"))
    analysis = (nvof or {}).get("analysis") or {}
    results = {r.get("id"): r for r in (bench or {}).get("results") or [] if not r.get("failed")}
    gate = (bench or {}).get("performance_gate") or {}
    d_diag = (results.get("D") or {}).get("residual_diagnostics") or {}
    frames = (results.get("D") or {}).get("frames")

    statuses = {}

    statuses["nvof_timing_semantics"] = {
        "status": ("DONE_PROVEN" if analysis.get("pass") and gates["gate4_hardware_timing_and_avx2"]["pass"]
                   and analysis.get("frames_timed", 0) > 0
                   and analysis.get("retired_keys_still_absent")
                   and analysis.get("frames_with_six_distinct_spans") == analysis.get("frames_timed")
                   else "NOT_READY"),
        "evidence": {
            "hardware_frames_timed": analysis.get("frames_timed"),
            "frames_where_all_six_spans_differ": analysis.get("frames_with_six_distinct_spans"),
            "retired_alias_keys_absent": analysis.get("retired_keys_still_absent"),
            "stages_still_unmeasurable": analysis.get("always_unmeasured"),
            "problems": analysis.get("problems"),
            "gate4": gates["gate4_hardware_timing_and_avx2"]["summary"],
        },
        "meaning": "each nvof_* key equals its own QPC span; a span that cannot be timed "
                   "reads 'unavailable', never 0.0, and no number is reused across stages",
    }

    statuses["residual_bridge_thread_safety"] = {
        "status": ("DONE_PROVEN" if checks and all(v.get("pass") for v in checks.values())
                   and gates["gate3_thread_safety"]["pass"] and control_failing
                   else "NOT_READY"),
        "evidence": {
            "scenarios_on_current_build": {k: v.get("pass") for k, v in sorted(checks.items())},
            "gate3": gates["gate3_thread_safety"]["summary"],
            "pre_fix_control_dll": ((thread or {}).get("control_build") or {}).get("dll"),
            "scenarios_failing_on_the_pre_fix_build": control_failing,
            "control_note": ((thread or {}).get("control_build") or {}).get("how_it_was_built"),
        },
        "meaning": "handle lookup, per-plan serialisation, destroy-waits-for-in-flight, "
                   "thread-local last error and counter isolation all hold under the "
                   "concurrent scenarios, and the same scenarios fail on the pre-fix build",
    }

    counts_ok = (d_diag.get("this_plan_create_count") == 1
                 and d_diag.get("reconstruct_call_count") == frames
                 and d_diag.get("this_plan_scratch_alloc_count") in (1, 2)
                 and d_diag.get("process_plan_create_total", 0) >= 1)
    statuses["per_plan_counter_semantics"] = {
        "status": ("DONE_PROVEN" if counts_ok and checks.get(
            "probe_does_not_pollute_per_plan_counters", {}).get("pass")
            and checks.get("concurrent_lifecycle_counts", {}).get("pass")
            and gates["gate3_thread_safety"]["pass"] else "NOT_READY"),
        "evidence": {
            "frames_exported_in_scenario_D": frames,
            "this_plan_create_count": d_diag.get("this_plan_create_count"),
            "this_plan_scratch_alloc_count": d_diag.get("this_plan_scratch_alloc_count"),
            "this_plan_worker_create_count": d_diag.get("this_plan_worker_create_count"),
            "reconstruct_call_count": d_diag.get("reconstruct_call_count"),
            "process_plan_create_total": d_diag.get("process_plan_create_total"),
            "process_plan_open_total": d_diag.get("process_plan_open_total"),
            "probe_moves_no_counter": checks.get(
                "probe_does_not_pollute_per_plan_counters", {}).get("pass"),
            "concurrent_lifecycle_counts_own_totals": checks.get(
                "concurrent_lifecycle_counts", {}).get("pass"),
            "cross_plan_not_blocked": checks.get("cross_plan_not_blocked", {}).get("pass"),
            "per_check_meaning": {
                "probe_does_not_pollute_per_plan_counters":
                    "reading diagnostics moves no counter on the plan probed or on any other",
                "concurrent_lifecycle_counts":
                    "each of N plans created concurrently reports its own create and scratch "
                    "count, so nothing inherits another plan's number",
                "cross_plan_not_blocked":
                    "a long call on one plan does not hold up frames on other plans",
            },
        },
        "meaning": "a plan reports its own lifecycle (1 create, its own scratch and "
                   "workers) while it serves every frame; process_* stay library-wide",
    }

    statuses["avx2_probe"] = {
        "status": ("DONE_PROVEN" if avx2 and avx2.get("measured_agreement_with_numpy")
                   and not avx2.get("rows_where_the_new_probe_is_wrong")
                   and gates["gate4_hardware_timing_and_avx2"]["pass"]
                   and avx2.get("capability_reported_only_this_round")
                   and not avx2.get("kernels_compiled_with_arch_flag")
                   and not avx2.get("avx2_intrinsic_used_by_kernels")
                   else "NOT_READY"),
        "evidence": {
            "probe_answer_on_this_machine": (avx2 or {}).get("native_probe", {}
                                                             ).get("probe_answer_on_this_machine"),
            "numpy_runtime_avx2": (avx2 or {}).get("numpy_runtime_features", {}).get("AVX2"),
            "agrees_with_numpy": avx2.get("measured_agreement_with_numpy") if avx2 else None,
            "bit_tested": (avx2 or {}).get("native_probe", {}).get("answers_with"),
            "gates_walked": (avx2 or {}).get("native_probe", {}).get("gates"),
            "rows_where_the_new_probe_is_wrong": avx2.get("rows_where_the_new_probe_is_wrong")
            if avx2 else None,
            "this_machine_can_show_the_old_bug": avx2.get("this_machine_can_show_the_old_bug")
            if avx2 else None,
            "gate4": gates["gate4_hardware_timing_and_avx2"]["summary"],
        },
        "meaning": "CPUID+XGETBV at runtime, matching an independent implementation; "
                   "reported only - no AVX2 kernel, no /arch flag, no SIMD benchmark "
                   "this round",
    }

    statuses["truthful_pipeline_docs"] = {
        "status": ("DONE_PROVEN" if gates["gate_claims_guard"]["pass"]
                   and gates["gate5_full_pytest"]["pass"] else "NOT_READY"),
        "evidence": {
            "claims_guard": gates["gate_claims_guard"]["summary"],
            "full_pytest": gates["gate5_full_pytest"]["summary"],
            "retired_claim_list_lives_in": "tests/unit/test_no_misleading_claims.py",
            "snapshot_docs_exempt_only_with_banner": ["CURRENT_STATE_AUDIT.md"],
        },
        "meaning": "the export is documented as a CPU-rendered pipe into an FFmpeg NVENC "
                   "child, the device-pointer query as diagnostic, and a lifecycle test "
                   "claims only that its calls returned",
    }

    statuses["truthful_gpu_memory_metric_names"] = {
        "status": ("DONE_PROVEN" if gates["gate_claims_guard"]["pass"]
                   and "process_gpu_memory_bytes" in json.dumps(results) else "NOT_READY"),
        "evidence": {
            "claims_guard": gates["gate_claims_guard"]["summary"],
            "card_metric_name": "gpu_total_memory_used_bytes",
            "peak_metric_name": "peak_gpu_total_memory_used_bytes",
            "per_process_metric": {rid: results[rid].get("process_gpu_memory_bytes")
                                   for rid in sorted(results)},
            "per_process_source": {rid: results[rid].get("process_gpu_memory_source")
                                   for rid in sorted(results)},
        },
        "meaning": "the card's total in use is named for the card; the process figure is "
                   "asked for separately and reports unavailable with the reason",
    }

    statuses["native_numeric_regression"] = {
        "status": ("DONE_PROVEN" if gates["gate2_native_numeric"]["pass"] else "NOT_READY"),
        "evidence": {"gate2": gates["gate2_native_numeric"]["summary"],
                     "files": ["test_resampler_numeric.py", "test_native_residual.py",
                               "test_residual_backend.py", "test_preview_export_parity.py"]},
        "meaning": "bit parity against the scalar oracle, backend reporting and "
                   "preview/export parity still hold after the lifecycle change",
    }

    statuses["full_pytest"] = {
        "status": ("DONE_PROVEN" if gates["gate5_full_pytest"]["pass"] else "NOT_READY"),
        "evidence": {"summary": gates["gate5_full_pytest"]["summary"],
                     "counted_by": "pytest, not by hand"},
    }

    ok_rows = results and all((r.get("ffprobe") or {}).get("ok") for r in results.values())
    statuses["quick_performance_regression"] = {
        "status": ("DONE_PROVEN" if gate.get("passed") and ok_rows and frames == 60
                   else "NOT_READY"),
        "evidence": {
            "frames_per_scenario": frames,
            "fps_100_percent": gate.get("fps_100_percent"),
            "fps_50_percent_native": gate.get("fps_50_percent_native"),
            "ratio_50_over_100": gate.get("ratio"),
            "requirement": gate.get("requirement"),
            "ffprobe_ok": {rid: bool((r.get("ffprobe") or {}).get("ok"))
                           for rid, r in sorted(results.items())},
            "p50_ms": {rid: {k: (r.get("stages_ms") or {}).get(k, {}).get("p50")
                             for k in ("nvof_submit_cpu_ms", "nvof_upload_ms",
                                       "nvof_download_ms",
                                       "nvof_postprocess_wait_inclusive_ms",
                                       "nvof_total_native_ms", "dlss_process",
                                       "working_color_prepare_ms", "residual_pyramid",
                                       "pipe_write", "frame_total")}
                       for rid, r in sorted(results.items())},
            "note": "60-frame A/D only. The 300-frame sustained sweep was explicitly out "
                    "of scope for this round, so nothing here is a sustained result",
        },
        "meaning": "50% working must be faster than 100%, and both outputs must be the "
                   "frame count and dimensions they claim",
    }

    overall = ("DONE_PROVEN" if all(v["status"] == "DONE_PROVEN" for v in statuses.values())
               else "NOT_READY")
    return statuses, overall, gates


def environment(out_dir, gates):
    import resource_metrics

    env = resource_metrics.environment_info({
        "code_under_test_sha": _git("rev-parse", "HEAD"),
        "code_under_test_sha_short": _git("rev-parse", "--short=7", "HEAD"),
        "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_log": _git("log", "--oneline", "-7"),
        "working_tree_unclean_at_collection": _dirty_paths(),
        "dll_sha256": {
            "native/residual_bridge.dll": _sha256(os.path.join(_ROOT, "native",
                                                               "residual_bridge.dll")),
            "native/nvof_bridge.dll": _sha256(os.path.join(_ROOT, "native",
                                                           "nvof_bridge.dll")),
        },
        "how_the_shipped_dlls_were_built": {
            "native/residual_bridge.dll": _cl_line(os.path.join(_ROOT, "native",
                                                                "build_residual.bat")),
            "native/nvof_bridge.dll": _cl_line(os.path.join(_ROOT, "native", "build.bat")),
        },
        "build_reproducibility_note": "MSVC link.exe stamps the image with a timestamp and "
                                      "a CodeView GUID, so a rebuild is not byte-identical. "
                                      "The hashes above are of the binaries that ran these "
                                      "gates, which is the claim being made.",
        "pytest_summary": gates["gate5_full_pytest"]["summary"],
    })
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="the cleanup artifact directory")
    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    statuses, overall, gates = derive(out_dir)
    env = environment(out_dir, gates)

    with open(os.path.join(out_dir, "environment.json"), "w", encoding="utf-8") as handle:
        json.dump(env, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report = {
        "overall": overall,
        "code_under_test_sha": env["code_under_test_sha"],
        "code_under_test_sha_short": env["code_under_test_sha_short"],
        "utc_timestamp": env["utc_timestamp"],
        "statuses": statuses,
        "vocabulary": "DONE_PROVEN only where this round's own run produced evidence; "
                      "NOT_READY otherwise. The BLOCKED_PROVEN status the previous round "
                      "used is not available in this round.",
    }
    with open(os.path.join(out_dir, "cleanup_status.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({"overall": overall,
                      "statuses": {k: v["status"] for k, v in sorted(statuses.items())}},
                     indent=2, sort_keys=True))
    return 0 if overall == "DONE_PROVEN" else 1


if __name__ == "__main__":
    sys.exit(main())
