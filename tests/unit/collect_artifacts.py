#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the evidence directory for one code-under-test SHA (task sections 29-30).

Reads the reports the other harnesses already wrote and adds the files that are derived
rather than measured: environment.json, resource_samples.json, ffprobe_output.json,
cancel_output.json, native_residual_counters.json and final_status.json.

Nothing here re-runs a measurement, and every command is run after the code is committed,
so the whole directory describes one code-under-test SHA:

    SHA=$(git rev-parse --short=7 HEAD)
    mkdir -p perf_results/$SHA
    python -m pytest -s tests/unit > perf_results/$SHA/pytest_output.txt
    python tests/unit/benchmark_residual_backends.py --backends python \\
        --out perf_results/$SHA/reference_python_baseline.json
    python tests/unit/benchmark_residual_backends.py --backends python,native \\
        --out perf_results/$SHA/native_microbenchmark.json
    python tests/unit/benchmark_e2e.py --mode quick     --frames 60  --out perf_results/$SHA/benchmark_quick.json
    python tests/unit/benchmark_e2e.py --mode sustained --frames 300 --out perf_results/$SHA/benchmark_sustained.json
    python tests/unit/collect_artifacts.py --out perf_results/$SHA

The exit status is zero only when every gate reads DONE_PROVEN, so the commit of the
artifact directory cannot happen on evidence that does not prove what it claims.
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import resource_metrics  # noqa: E402

FIXTURES = ("tests/unit/fixture_1080p_600f.mp4", "bench_out/fixture_4k_300.mp4")
# The evidence directory is untracked by definition while it is being assembled, so it
# must not be reported as an unclean part of the code under test.
NON_CODE_PREFIXES = ("perf_results/",)


def _dirty_paths():
    return [line for line in _git("status", "--porcelain").splitlines()
            if line.strip() and not line[3:].strip().startswith(NON_CODE_PREFIXES)]


def _git(*args):
    return subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True).stdout.strip()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(directory, name):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _results(report):
    return (report or {}).get("results", [])


def _probe_stream(path):
    """Measured stream properties of a video file, or None when ffprobe cannot read it."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
             "-of", "json", path], capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return None
        stream = (json.loads(result.stdout or "{}").get("streams") or [{}])[0]
        numerator, _, denominator = str(stream.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(numerator) / float(denominator or 1)
        return {"codec": stream.get("codec_name"), "width": stream.get("width"),
                "height": stream.get("height"), "fps": round(fps, 6) if fps else None,
                "frames": int(stream.get("nb_read_frames") or 0) or None}
    except Exception:
        return None


def _encoder_settings(out_dir):
    """What the benchmark runs actually encoded with, read out of their own reports."""
    report = _load(out_dir, "benchmark_quick.json") or _load(out_dir, "benchmark_sustained.json")
    env = (report or {}).get("environment") or {}
    settings = {key: env.get(key) for key in ("encoder", "cq", "flow_engine", "flow_preset")}
    settings["observed_outputs"] = {
        row.get("id"): {"path": os.path.basename(row.get("output_path") or ""),
                        "codec": (row.get("ffprobe") or {}).get("codec"),
                        "packets": (row.get("ffprobe") or {}).get("packets"),
                        "fps": (row.get("ffprobe") or {}).get("fps"),
                        "residual_backend": (row.get("residual_diagnostics") or {}).get(
                            "residual_backend")}
        for row in _results(report)}
    return settings


def _cancel_record(out_dir):
    """The CANCEL_E2E report the end-to-end cancellation test printed into the transcript.

    Read from the log rather than from cancel_output.json so this works no matter what
    order the collector's own steps run in.
    """
    log = os.path.join(out_dir, "pytest_output.txt")
    if not os.path.exists(log):
        return None
    with open(log, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("CANCEL_E2E:"):
                return json.loads(line[len("CANCEL_E2E:"):])
    return None


def _fixture_entry(rel, digest=None):
    """One fixture's provenance row: where it is, what is in it, and who used it."""
    path = os.path.join(_ROOT, rel.replace("/", os.sep))
    entry = {"path": rel, "bytes": None, "sha256": digest, "used_by": []}
    if not os.path.exists(path):
        entry["used_by"] = "not found on disk"
        return entry
    entry["bytes"] = os.path.getsize(path)
    entry["sha256"] = digest or _sha256(path)
    probed = _probe_stream(path)
    if probed is None:
        entry["used_by"] = "present but ffprobe could not read it"
    else:
        entry.update(probed)
    return entry


def environment(out_dir):
    dirty = _dirty_paths()
    info = resource_metrics.environment_info({
        "code_under_test_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty_paths": dirty,
        "worktree_clean": not dirty,
        "commit_subject": _git("log", "-1", "--pretty=%s"),
        "commit_history": _git("log", "-6", "--pretty=%h %s").splitlines(),
    })
    fixtures = {rel: _fixture_entry(rel) for rel in FIXTURES}
    # Then whoever a report names, matched on the digest it recorded: a fixture the
    # harnesses actually used must be in this table even if it is not in the constant.
    seen = []
    for name, label in (("benchmark_quick.json", "e2e quick benchmark"),
                        ("benchmark_sustained.json", "e2e sustained benchmark")):
        env = (_load(out_dir, name) or {}).get("environment") or {}
        seen.append((label, env.get("fixture_sha256"), env.get("fixture_path")))
    cancel = _cancel_record(out_dir)
    if cancel:
        seen.append(("cancellation end-to-end test", cancel.get("fixture_sha256"), None))
    by_digest = {e["sha256"]: e for e in fixtures.values() if e.get("sha256")}
    for label, digest, rel in seen:
        rel = (rel or "").replace("\\", "/")
        entry = by_digest.get(digest)
        if entry is None and (digest or rel):
            entry = _fixture_entry(rel or f"unknown:{digest}", digest)
            fixtures[entry["path"]] = entry
            if entry.get("sha256"):
                by_digest[entry["sha256"]] = entry
        if entry is not None and isinstance(entry.get("used_by"), list):
            entry["used_by"].append(label)
    info["fixtures"] = fixtures
    info["encoder_settings"] = _encoder_settings(out_dir)
    _write(out_dir, "environment.json", info)
    return info


def _write(directory, name, payload):
    with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return os.path.join(directory, name)


def derive(out_dir, env):
    quick = _load(out_dir, "benchmark_quick.json")
    sustained = _load(out_dir, "benchmark_sustained.json")
    micro = _load(out_dir, "native_microbenchmark.json")

    ffprobe = {}
    counters = {}
    for label, report in (("quick", quick), ("sustained", sustained)):
        for row in _results(report):
            scenario = f"{label}:{row.get('id')}"
            if row.get("ffprobe"):
                ffprobe[scenario] = row["ffprobe"]
            if row.get("residual_diagnostics"):
                counters[scenario] = row["residual_diagnostics"]
    _write(out_dir, "ffprobe_output.json", ffprobe)
    _write(out_dir, "native_residual_counters.json", {
        "per_scenario": counters,
        "microbenchmark_diagnostics": [
            {"requested": r.get("requested_backend"), "actual": r.get("actual_backend"),
             "working_scale": r.get("working_scale"), "diagnostics": r.get("backend_diagnostics")}
            for r in (micro or {}).get("results", [])],
        "interpretation": "this_plan_create_count, this_plan_scratch_alloc_count and "
                          "this_plan_worker_create_count are this backend's own lifecycle and stay "
                          "at 1/1/N while downsample_call_count and reconstruct_call_count rise one "
                          "per frame; process_* are the whole-library totals, which only other "
                          "plans being created or destroyed move. The AVX2 capability probe "
                          "belongs to neither scope and changes no counter"})

    samples = (sustained or {}).get("resource_samples", {})
    _write(out_dir, "resource_samples.json", samples)
    # The run's own frame count, per scenario. The fit window is a fraction of the run, and
    # inferring it from the highest sample number would silently widen the window every time
    # the last sample landed early - which, for a run sampled every 30th frame, is the
    # difference between excluding the teardown sample and being gated by it.
    run_frames = {row.get("id"): row.get("frames") for row in _results(sustained) if row.get("id")}
    drift = {}
    for sid, groups in samples.items():
        frames = sorted(groups.get("frame_samples", []), key=lambda r: r["frame"])
        wall = sorted(groups.get("wallclock_samples", []), key=lambda r: r.get("t_sec", 0.0))
        entry = {"frame_samples": len(frames), "wallclock_samples": len(wall)}
        if len(frames) >= 3:
            total = run_frames.get(sid) or (frames[-1]["frame"] if frames else 0)
            ys = [r["rss_bytes"] for r in frames]
            tail = ys[max(0, int(0.75 * len(ys))):]
            fit = resource_metrics.fit_rss_steady_state(frames, total)
            entry.update({
                "rss_bytes_min": min(ys), "rss_bytes_max": max(ys), "rss_bytes_last": ys[-1],
                "rss_total_frames": total,
                "rss_fit_total_frames": total,
                "rss_tail_minus_head_of_tail": round(max(tail) - min(tail), 1),
                "thread_count_range": [min(r["thread_count"] for r in frames),
                                       max(r["thread_count"] for r in frames)],
                "child_process_count_range": [min(r["child_process_count"] for r in frames),
                                              max(r["child_process_count"] for r in frames)],
                "frame_range": [frames[0]["frame"], frames[-1]["frame"]],
            })
            entry.update({k: v for k, v in fit.items() if v is not None})
            # Kept beside the steady-state numbers so a reader can see what excluding the
            # warmup and the teardown changed. This one is never gated on.
            entry["rss_whole_range_slope_bytes_per_frame"] = (
                resource_metrics.fit_rss_whole_range(frames)["rss_slope_bytes_per_frame"])
        if wall:
            used = [r["gpu_total_memory_used_bytes"] for r in wall
                    if r.get("gpu_total_memory_used_bytes")]
            entry["gpu_total_memory_used_bytes_max"] = max(used) if used else None
            entry["gpu_total_memory_samples"] = len(used)
            entry["gpu_total_memory_scope"] = (
                "whole card, from nvidia-smi memory.used. Not this process: this driver "
                "reports [N/A] per process, so no run on it can attribute GPU memory to a "
                "pid. The number also covers whatever else was using the GPU meanwhile.")
        drift[sid] = entry
    stability = {}
    for sid, entry in drift.items():
        fit = {k: entry[k] for k in (
            "rss_fit_status", "rss_fit_reason", "rss_fit_sample_count", "rss_fit_frames",
            "rss_slope_bytes_per_frame", "rss_predicted_signed_growth_bytes",
            "rss_predicted_positive_growth_bytes") if k in entry}
        gate = resource_metrics.rss_stability_gate(fit, entry.get("rss_bytes_max"))
        gate["child_process_count_range"] = entry.get("child_process_count_range")
        gate["thread_count_range"] = entry.get("thread_count_range")
        stability[sid] = gate
    _write(out_dir, "resource_samples_summary.json", drift)
    _write(out_dir, "resource_stability.json", {
        sid: entry["status"] for sid, entry in stability.items()})

    cancel = _cancel_record(out_dir)
    if cancel:
        _write(out_dir, "cancel_output.json", cancel)

    summary_line = ""
    log = os.path.join(out_dir, "pytest_output.txt")
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as handle:
            lines = [l.strip() for l in handle if " passed" in l or " failed" in l or " error" in l]
        summary_line = lines[-1] if lines else ""

    gates = {
        "pytest": {"summary": summary_line,
                   "status": "DONE_PROVEN" if "passed" in summary_line
                             and "failed" not in summary_line and "error" not in summary_line
                             else "NOT_READY"},
        "performance_gate_quick": _gate_status(quick),
        "performance_gate_sustained": _gate_status(sustained),
        "native_numeric": native_numeric_status(micro, summary_line),
        "resource_reuse_over_300_frames": _reuse_status(sustained),
        "sustained_resource_stability": stability,
        "output_integrity": "DONE_PROVEN" if ffprobe and all(
            f.get("ok") for f in ffprobe.values()) else "NOT_READY",
        "cancellation": _cancellation_status(cancel),
    }
    _write(out_dir, "final_status.json", {
        "code_under_test_sha": env.get("code_under_test_sha"),
        "worktree_clean_at_collection": env.get("worktree_clean"),
        "worktree_dirty_paths_at_collection": env.get("worktree_dirty_paths"),
        "collected_at_utc": env.get("utc_timestamp"),
        "gates": gates,
        "note": "DONE_PROVEN requires production code, a numeric test, a runtime test and "
                "the artifact in this directory; anything else is NOT_READY.",
    })
    return gates


def native_numeric_status(micro, summary_line):
    """Proven only when every native microbenchmark row is bit identical to the Python
    reference at 4K and the pytest run that produced them was clean."""
    if "passed" not in summary_line or "failed" in summary_line or "error" in summary_line:
        return {"status": "NOT_READY", "reason": "pytest summary not clean",
                "pytest_summary": summary_line}
    rows = [r for r in _results(micro) if r.get("requested_backend") == "native"]
    if not rows:
        return {"status": "NOT_READY", "reason": "no native rows in the microbenchmark report",
                "pytest_summary": summary_line}
    parity = {}
    ok = True
    for row in rows:
        key = f"{row.get('working_scale')}:{row.get('source')}->{row.get('working')}"
        p = row.get("parity_with_reference_4k")
        if not isinstance(p, dict):
            parity[key] = "missing"
            ok = False
            continue
        bit_exact = (p.get("bit_equal_downsample") is True
                     and p.get("bit_equal_reconstruct") is True
                     and p.get("max_abs_diff") == 0)
        parity[key] = {
            "bit_equal_downsample": p.get("bit_equal_downsample"),
            "bit_equal_reconstruct": p.get("bit_equal_reconstruct"),
            "max_abs_diff": p.get("max_abs_diff"),
        }
        ok = ok and bit_exact
    return {"status": "DONE_PROVEN" if ok else "NOT_READY",
            "pytest_summary": summary_line,
            "parity_with_python_reference": parity}


def _gate_status(report):
    if not report:
        return {"status": "NOT_READY", "reason": "report missing"}
    gate = report.get("performance_gate", {})
    return {"status": "DONE_PROVEN" if gate.get("passed") else "NOT_READY",
            "detail": gate}


def _cancellation_status(cancel):
    """The measured cancellation facts, with the verdict derived from them."""
    if not cancel:
        return {"status": "NOT_READY", "reason": "no CANCEL_E2E record in the transcript"}
    measured = {key: cancel.get(key) for key in
                ("job_status", "frames_at_cancel", "frames_requested", "cancel_latency_ms",
                 "producer_threads_alive", "queue_thread_alive", "registry_after",
                 "pids_still_running", "written_packets", "output_bytes")}
    measured["status"] = ("DONE_PROVEN" if (
        cancel.get("job_status") == "cancelled"
        and not cancel.get("pids_still_running")
        and cancel.get("registry_after") == 0
        and not cancel.get("producer_threads_alive")
        and not cancel.get("queue_thread_alive")
        and (cancel.get("frames_at_cancel") or 0) >= 1) else "NOT_READY")
    return measured


def _reuse_status(sustained):
    rows = [r for r in _results(sustained) if r.get("residual_diagnostics")]
    if not rows:
        return {"status": "NOT_READY", "reason": "no native diagnostics in the sustained run"}
    ok = True
    detail = {}
    for row in rows:
        d = row["residual_diagnostics"]
        frames = row.get("frames", 0)
        # this_plan_* is the lifecycle of the backend that ran those frames, so it is a
        # constant; downsample/reconstruct counts are the same plan's work and must rise
        # one per frame. Reading the per-plan block instead of a process total is what
        # makes "no plan was built mid-run" a statement about this scenario.
        stable = (d.get("this_plan_create_count") == 1
                  and d.get("this_plan_scratch_alloc_count") == 1
                  and d.get("this_plan_destroy_count") == 0
                  and d.get("downsample_call_count") == frames
                  and d.get("reconstruct_call_count") == frames)
        ok = ok and stable
        detail[row["id"]] = {"frames": frames,
                             "this_plan_create_count": d.get("this_plan_create_count"),
                             "this_plan_scratch_alloc_count": d.get("this_plan_scratch_alloc_count"),
                             "this_plan_worker_create_count": d.get("this_plan_worker_create_count"),
                             "process_plan_create_total": d.get("process_plan_create_total"),
                             "process_plan_destroy_total": d.get("process_plan_destroy_total"),
                             "downsample_call_count": d.get("downsample_call_count"),
                             "reconstruct_call_count": d.get("reconstruct_call_count")}
    return {"status": "DONE_PROVEN" if ok else "NOT_READY", "per_scenario": detail}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    out_dir = args.out or os.path.join(_ROOT, "perf_results", _git("rev-parse", "--short=7", "HEAD"))
    os.makedirs(out_dir, exist_ok=True)
    env = environment(out_dir)
    gates = derive(out_dir, env)
    print(json.dumps(gates, indent=2))
    print("artifacts in", out_dir)
    missing = [f for f in glob.glob(os.path.join(out_dir, "*"))]
    print("files:", sorted(os.path.basename(f) for f in missing))
    def all_done(value):
        if isinstance(value, dict):
            if "status" in value:
                return value["status"] == "DONE_PROVEN"
            return bool(value) and all(all_done(v) for v in value.values())
        return value == "DONE_PROVEN"

    return 0 if all(all_done(v) for v in gates.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
