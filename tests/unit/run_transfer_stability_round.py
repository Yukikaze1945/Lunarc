#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run this round's Gate 6 exactly once: lock, quiet window, one batch, one monitor, one verdict.

Why a driver file exists at all
------------------------------
The previous round answered a 0.5% question with a machine doing other things. Its instrument could
not resolve that difference in either direction, and nothing it published said so, because the
question "was anybody else here while these numbers were made" had never been asked. §48 of this
round's brief asks that question in a fixed order - take the lock, find one quiet window, run twelve
60-frame executions, run four 150-frame ones - and an order is a program, not a document. Someone
would otherwise type those five commands by hand, and the one thing worth protecting here is the gap
between them: the batch has to be *inside* the lock and *inside* the monitoring window, from the first
child process to the last, with nothing in between able to be forgotten.

So this file holds the lock, runs the preflight, starts the 1 Hz monitor, launches executions one at a
time while watching for contamination, stops the monitor, and only then lets the collectors read what
happened. It never decides what any of those numbers mean: every verdict here comes from
`collect_transfer_status.py` re-deriving it from the artifacts this driver wrote.

What this file refuses to do
----------------------------
  * **Run a second batch.** §16: waiting for a quiet preflight window is not a re-roll, but once a
    measurement has been taken the batch is the batch. If the evidence directory already holds a
    pooled artifact, this driver stops rather than overwriting it, so "let's just run it again" has
    to be a decision somebody makes deliberately in a new directory, not a reflex of a script.
  * **Tune anything.** The §12 thresholds live in `benchmark_environment.THRESHOLDS`, the §34
    variance limits live in `collect_transfer_status`, and the §26 batch shape below is asserted
    against the modules that run it rather than taken from the command line. There is no
    `--frames`, no `--threshold`, no `--executions` argument to pass a nicer value through.
  * **Kill or wait out a competitor.** §13/§14: another agent's process is recorded and, if it
    stays, ends the round as NOT_READY. Nothing here signals another process, and nothing deletes a
    lock it does not own (§9).
  * **Touch the frozen evidence.** §39: the directories of previous rounds are read-only to this
    one, including the two it recalculates from.
  * **Measure on code its artifacts cannot name.** §37: the commit is taken once, at the top, and
    the run refuses to start with tracked files modified, because `code_under_test` pointing at a
    commit that does not describe the code that ran is the exact failure this round was asked to fix.

Usage:
    python tests/unit/run_transfer_stability_round.py \
        --out perf_results/<sha8>/bench_stability --pytest-transcript bench_out/pytest_output.txt
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, os.path.join(_ROOT, "src"), _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_environment as be  # noqa: E402
import benchmark_e2e as bench  # noqa: E402
import collect_transfer_ab as ab  # noqa: E402
import collect_transfer_performance as cp  # noqa: E402
import collect_transfer_status as cs  # noqa: E402

# §26, the batch this round is allowed to run. Not defaults: the values are asserted against the
# modules that do the measuring before a single child is started, and the run stops if they disagree.
PRIMARY_FRAMES = 60
PRIMARY_EXECUTIONS = 12
CORROBORATION_FRAMES = 150
CORROBORATION_EXECUTIONS = 4
RUNS_PER_MODE = 3            # §26: run 1 warms up, runs 2 and 3 are kept
WORKING_SCALE = 0.5
BASELINE = "e3aa86c"          # §42: the binaries and native sources must still be this commit's

# §39: evidence that already exists belongs to the rounds that wrote it. The recalculations this
# round reads are `740c78a` (its fixed-order batch) and `7c1768f` (its counterbalanced one).
FROZEN_EVIDENCE_DIRS = ("05f0ecc", "b1668d3", "b986788", "e48b25b", "44f4427", "740c78a",
                        "7c1768f")
HISTORY_DIR = os.path.join(_ROOT, "perf_results", "740c78a", "transfer_phase1")
HISTORY_EXECUTIONS_DIR = os.path.join(HISTORY_DIR, "ab_executions")
# The §41 recalculation is the whole of that batch, and its last two executions predate the
# per-execution subdirectory: they are the batch's own top-level artifacts. All twelve declare the
# same `code_under_test`, which `history_sources` checks rather than assumes.
HISTORY_EXTRA_FILES = ("transfer_ab_benchmark.json", "transfer_ab_benchmark_60f_e2.json")
HISTORY_SOURCE_SHA = "740c78a0f96b3927bd3fc1c27efa3fc93d5c51ee"

# §0's hard scope, restated as paths so a dirty working tree can be named rather than guessed at.
PRODUCTION_PATHS = ("src/pipeline.py", "src/optical_flow.py", "src/dlss_service.py",
                    "src/dlss_engine.py", "src/residual_native.py",
                    "native/residual_bridge.cpp", "native/residual_bridge.h",
                    "native/nvof_bridge.cpp", "native/nvof_bridge.h",
                    "native/residual_bridge.dll", "native/nvof_bridge.dll")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args):
    return cs.git(*args)


def _logger(out_dir, path):
    """Every line this driver writes goes to stdout and to a log beside it.

    The log lives in `bench_out/`, not in the evidence directory: §40 lists the artifacts a round
    owes, and a running commentary is not one of them. It is how a batch that takes forty minutes can
    be followed without a second process watching the terminal.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def say(text):
        line = text if isinstance(text, str) else json.dumps(text, sort_keys=True, default=str)
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        for chunk in line.splitlines() or [""]:
            print(f"[{stamp}] {chunk}", flush=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {line}\n")
    return say


def _write_json(path, payload):
    """Artifacts are written the same way everywhere: sorted keys, compact, never indented twice.

    `benchmark_environment.json` carries thirty GPU-client tables per window, and the two-window
    version of it that this round produced while the machine was busy was 1.8 MB at `indent=2`. The
    numbers are what the evidence is, so the numbers stay; the layout is not, and a file a reader
    has to open in an editor rather than eyeball serves nobody.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, separators=(",", ":"), sort_keys=True, default=str)
    return path


# --------------------------------------------------------------------------------------------
# What must be true before a child process starts
# --------------------------------------------------------------------------------------------

def frozen_configuration():
    """§26/§27: the batch shape, read back from the modules that will actually run it.

    Asserted rather than passed as arguments. A command line that can say `--frames 40` is a command
    line that will, eventually, say it, and then the protocol record and the gate thresholds that
    read `runs_per_mode` describe a different experiment from the one §26 defined.
    """
    checks = [
        ("fixture_resolution_is_3840x2160", (bench.FIXTURE_W, bench.FIXTURE_H) == (3840, 2160),
         [bench.FIXTURE_W, bench.FIXTURE_H]),
        ("fixture_is_60_fps", bench.FIXTURE_FPS == 60.0, bench.FIXTURE_FPS),
        ("working_scale_is_50_percent", WORKING_SCALE == 0.5, WORKING_SCALE),
        ("flow_engine_and_preset_unchanged",
         (bench.FLOW_ENGINE, bench.FLOW_PRESET) == ("nvof", "4M"),
         [bench.FLOW_ENGINE, bench.FLOW_PRESET]),
        ("encoder_is_av1_nvenc_at_cq18",
         bench.CQ == 18 and ab.EXPECTED_OUTPUT_CODEC == "av1",
         {"cq": bench.CQ, "codec": ab.EXPECTED_OUTPUT_CODEC}),
        ("primary_batch_is_twelve_executions_of_sixty_frames",
         (PRIMARY_EXECUTIONS, PRIMARY_FRAMES) == (12, 60),
         [PRIMARY_EXECUTIONS, PRIMARY_FRAMES]),
        ("corroborating_batch_is_four_executions_of_a_hundred_and_fifty_frames",
         (CORROBORATION_EXECUTIONS, CORROBORATION_FRAMES) == (4, 150),
         [CORROBORATION_EXECUTIONS, CORROBORATION_FRAMES]),
        ("three_pairs_per_run_two_of_them_kept",
         (RUNS_PER_MODE, ab.kept_run_ids(RUNS_PER_MODE), ab.warmup_run_ids(RUNS_PER_MODE))
         == (3, [2, 3], [1]),
         {"runs": RUNS_PER_MODE, "kept": ab.kept_run_ids(RUNS_PER_MODE),
          "warmup": ab.warmup_run_ids(RUNS_PER_MODE)}),
        ("the_two_lengths_are_never_pooled", PRIMARY_FRAMES != CORROBORATION_FRAMES,
         [PRIMARY_FRAMES, CORROBORATION_FRAMES]),
        ("the_protocol_is_the_counterbalanced_one",
         (ab.PROTOCOL_NAME, ab.PROTOCOL_VERSION) == ("counterbalanced-kept-runs-ab", 4),
         [ab.PROTOCOL_NAME, ab.PROTOCOL_VERSION]),
        ("a_gpu_client_needs_positive_evidence_to_be_a_conflict",
         len(be.EXTERNAL_GPU_COMPUTE_CRITERIA) == 4
         and be.PER_PROCESS_GPU_COMPUTE_PERCENT_AT_LEAST == 15.0
         and not (set(be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS)
                  & set(be.GPU_CLIENT_NON_TRIGGERING_CLASSIFICATIONS))
         and set(be.GPU_CLIENT_NON_TRIGGERING_CLASSIFICATIONS) <= set(be.GPU_CLIENT_CLASSIFICATIONS)
         and not any("*" in str(entry) for entry
                     in tuple(be.AMBIENT_OS_GRAPHICS_EXECUTABLES)
                     + tuple(be.GPU_COMPUTE_COMMAND_TOKENS)
                     + tuple(be.GPU_WORKLOAD_TOKENS)),
         {"criteria": list(be.EXTERNAL_GPU_COMPUTE_CRITERIA),
          "per_process_gpu_compute_percent_at_least": be.PER_PROCESS_GPU_COMPUTE_PERCENT_AT_LEAST,
          "may_stop_a_batch": list(be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS),
          "may_never_stop_a_batch": list(be.GPU_CLIENT_NON_TRIGGERING_CLASSIFICATIONS),
          "note": "§5: this is the frozen shape of the rule the round is measured under - four "
                  "positive criteria, one utilisation limit fixed in the module rather than chosen at "
                  "runtime, the two classes that can never refuse a window kept out of the three that "
                  "can, and no glob in any of the token tables, because a `*` in one of those is the "
                  "directory whitelist §5 forbids"}),
        ("the_counterbalance_is_deterministic",
         [ab.execution_schedule(RUNS_PER_MODE, index) for index in (1, 2)]
         == [ab.execution_schedule(RUNS_PER_MODE, index) for index in (1, 2)]
         and ab.pair_order(2, 1)[0] != ab.pair_order(2, 2)[0],
         {"execution_1_pair_2": list(ab.pair_order(2, 1)),
          "execution_2_pair_2": list(ab.pair_order(2, 2))}),
        ("the_fixture_file_is_present", os.path.exists(bench.FIXTURE_PATH),
         os.path.relpath(bench.FIXTURE_PATH, _ROOT).replace("\\", "/")),
    ]
    return {"ok": all(bool(passed) for _, passed, _ in checks),
            "checks": [{"what": what, "ok": bool(passed), "observed": observed}
                       for what, passed, observed in checks],
            "failed": [what for what, passed, _ in checks if not passed]}


def native_scope_against_baseline(baseline=BASELINE):
    """§42: are the bridges this round measures with the ones the named baseline built?

    Both the sources and the binaries, against three commits each: the working file, the baseline
    commit, HEAD. `cs.native_against_baseline` covers the DLLs and the status gate covers the rest;
    this exists so a round that is going to stop on §42 stops before spending forty minutes on a
    batch whose evidence could not be certified, rather than after.

    The binaries are compared by SHA-256 of their bytes; the sources are compared by *git blob id*,
    because a CRLF checkout of an LF-stored `.cpp` has a different file hash from the object the
    commit holds and identical content. Measured on this working tree: `native/nvof_bridge.cpp` and
    `native/residual_bridge.cpp` both differ from their committed blobs by raw hash and match them
    by blob id, so the file hash would stop a round that had changed nothing.
    """
    rows = cs.native_against_baseline(baseline)
    for _, cpp_rel in cs.NATIVE_PAIRS:
        for rel in (cpp_rel, cpp_rel[:-4] + ".h"):
            path = os.path.join(_ROOT, rel)
            working = git("hash-object", rel) if os.path.exists(path) else ""
            at_head = git("rev-parse", f"HEAD:{rel}")
            at_baseline = git("rev-parse", f"{baseline}:{rel}")
            rows[os.path.basename(rel)] = {
                "path": rel, "working_tree_git_blob": working or None,
                "git_blob_at_head": at_head or None, "git_blob_at_baseline": at_baseline or None,
                "unchanged_since_baseline": bool(working) and working == at_head == at_baseline,
            }
    changed = [line for line in git("diff", "--name-only", f"{baseline}..HEAD").splitlines()
               if line.strip()]
    touched_production = [path for path in changed
                          if path.replace("\\", "/") in PRODUCTION_PATHS]
    return {"ok": (all(row["unchanged_since_baseline"] for row in rows.values())
                   and not touched_production),
            "files": rows, "changed_since_baseline": changed,
            "production_files_touched": touched_production,
            "baseline": baseline}


def working_tree_is_clean():
    """§37 from the other side: the commit a run names has to be the code that runs.

    Only tracked files are checked. `bench_out/` and a fresh, still-untracked evidence directory are
    this round's own scratch, and refusing because of them would stop the round for a reason that has
    nothing to do with what is being measured.
    """
    dirty = [line for line in git("diff", "--name-only", "HEAD").splitlines() if line.strip()]
    staged = [line for line in git("diff", "--cached", "--name-only", "HEAD").splitlines()
              if line.strip()]
    return {"ok": not dirty and not staged,
            "modified": sorted(set(dirty) | set(staged)),
            "head": git("rev-parse", "HEAD")}


def evidence_directory_is_writable(out_dir):
    """§39: never into a previous round's directory, never over this round's finished batch."""
    rel = os.path.relpath(os.path.abspath(out_dir), _ROOT).replace("\\", "/")
    frozen = next((name for name in FROZEN_EVIDENCE_DIRS
                   if rel == f"perf_results/{name}" or rel.startswith(f"perf_results/{name}/")),
                  None)
    if frozen:
        return {"ok": False, "reason": "frozen_evidence_directory", "directory": rel,
                "note": f"§39: perf_results/{frozen}/ belongs to the round that wrote it"}
    existing = [name for name in
                ("counterbalanced_performance.json", "counterbalanced_150f.json",
                 "benchmark_runtime_monitor.json", "benchmark_environment.json",
                 "benchmark_stability_status.json")
                if os.path.exists(os.path.join(out_dir, name))]
    if existing:
        return {"ok": False, "reason": "refusing_to_reroll_a_finished_batch",
                "directory": rel, "already_present": existing,
                "note": "§16: the batch that produced these files is the batch; a second one is a "
                        "decision, not a re-run of this command"}
    return {"ok": True, "directory": rel}


# --------------------------------------------------------------------------------------------
# The measurement itself
# --------------------------------------------------------------------------------------------

def run_batch(label, indices, frames, exec_dir, monitor, say, runs=RUNS_PER_MODE,
              scale=WORKING_SCALE):
    """Launch executions one at a time, asking the monitor between each whether to go on.

    The loop is in this process on purpose. §18 allows stopping a batch only once the child that is
    currently running has ended safely, which is a thing you can only honour if you are the one
    waiting on that child; a batch launched as one subprocess could be interrupted only by killing it
    mid-run, and a killed run is neither safe nor a completed execution. So: one execution at a
    time, contamination checked in the gap between two of them, and everything already finished kept
    exactly as it is.
    """
    paths, failures = [], []
    stopped = None
    for index in indices:
        if monitor is not None and monitor.contaminated:
            stopped = (f"{label} stopped before execution {index}: the runtime monitor recorded an "
                       f"external workload inside the batch (§18)")
            say(stopped)
            break
        say(f"[{label}] execution {index} ({frames} frames x {runs} pairs, schedule "
            + json.dumps(ab.execution_schedule(runs, index), separators=(",", ":")) + ")")
        try:
            execution, error = cp.run_execution(index, frames, runs, scale, exec_dir)
        except Exception as exc:  # §29: a child that died is recorded, not retried
            failures.append(f"{label} execution {index} raised {type(exc).__name__}: {exc}")
            say(f"[{label}] execution {index} raised {type(exc).__name__}: {exc}")
            continue
        if execution is None:
            failures.append(f"{label} execution {index} produced no artifact: "
                            f"{(error or '')[-400:]}")
            say(f"[{label}] execution {index} produced no artifact")
            continue
        name = f"execution_{index:02d}_transfer_ab_benchmark.json"
        paths.append(os.path.join(exec_dir, name))
        say({"batch": label, "execution": execution.get("execution"),
             "index": index, "status": execution.get("status"),
             "ratio": execution.get("speedup_ratio_reduced_over_legacy"),
             "kept_medians": {mode: (execution.get("orders") or {}).get(mode, {})
                              .get("median_fps_of_kept_runs") for mode in cp.MODES},
             "unpoolable": execution.get("unpoolable_reasons") or None,
             "contaminated": bool(monitor.contaminated) if monitor is not None else None})
    if monitor is not None and monitor.contaminated and stopped is None:
        stopped = (f"{label}: the runtime monitor recorded an external workload inside this batch "
                   f"(§18); every execution it had launched had already finished")
    return paths, failures, stopped


def monitor_batch_record(primary_paths, corroborating_paths, stopped, corroborating_stopped,
                         failures, contaminated, preflight_passed=True):
    """The `batch` section §21 asks the monitor artifact to carry, as one importable function.

    It is a function rather than a literal inside `main` because a gate in another file judges these
    fields by name: `collect_transfer_status.runtime_contamination_gate` reads
    `stopped_because_of_contamination` to check §18's promise that nothing was launched after the
    outsider arrived. A record built here and read there can be tested against itself; a dict typed
    into the middle of a 200-line `main` can only be tested against a copy of itself, which is how a
    rename ends up silently failing a round at its last step.

    §4: `corroborating_batch_reason` is the name of what happened to the 150-frame batch, from the
    one implementation the status gate also calls. This driver can name the cases it caused - it
    either launched the batch or it did not, and it knows why - but agreement and conflict are
    statements about the arithmetic, which is pooled after this record is written. So the field is
    only filled in when the batch did not run at all, and the gate, which reads both medians, owns
    the two names that need them. A prose `*_not_run_because` sentence stays beside it because the
    reason alone does not say which execution failed to launch.
    """
    did_not_run = corroborating_batch_reason(
        len(primary_paths), len(corroborating_paths), contaminated, preflight_passed)
    return {
        "primary_frames_per_run": PRIMARY_FRAMES,
        "primary_executions_launched": len(primary_paths),
        "primary_executions_defined": PRIMARY_EXECUTIONS,
        "primary_stopped_early_because": stopped,
        # §18's promise, in the field the contamination gate reads: the driver launched no further
        # execution once the monitor said the window had an outsider in it. Which executions that
        # cost is in the two reasons beside it; this one answers "did stopping happen at all".
        "stopped_because_of_contamination": bool(contaminated),
        "corroborating_frames_per_run": CORROBORATION_FRAMES,
        "corroborating_executions_launched": len(corroborating_paths),
        "corroborating_executions_defined": CORROBORATION_EXECUTIONS,
        "corroborating_batch_not_run_because": (corroborating_stopped
                                                if not corroborating_paths else None),
        "corroborating_batch_reason": did_not_run,
        "execution_failures": list(failures),
    }


def corroborating_batch_reason(primary_launched, corroborating_launched, contaminated,
                               preflight_passed):
    """§4's name for a 150-frame batch this driver did not launch, or `None` if it did.

    Kept apart from `monitor_batch_record` so a test can ask the question "what would you have
    called it" against the same function the artifact is built with, rather than against a copy of
    the arithmetic that decides it.
    """
    if corroborating_launched:
        return None
    return be.corroboration_reason(
        primary_execution_count=primary_launched, primary_expected=PRIMARY_EXECUTIONS,
        corroborating_execution_count=0, corroborating_expected=CORROBORATION_EXECUTIONS,
        runtime_contaminated=bool(contaminated), preflight_passed=bool(preflight_passed))


def pool(out_dir, frames, executions, name, write_protocol, say, corroborate=(),
         failures=(), history=()):
    """Hand the recorded executions to the pooling collector, in a process of its own.

    The collector, not this driver, computes the medians and the verdict. Running it as a subprocess
    keeps that line visible in the log and keeps this file free of a second implementation of the
    arithmetic the gates re-read afterwards.
    """
    cmd = [sys.executable, os.path.join(_UNIT_DIR, "collect_transfer_performance.py"),
           "--out", out_dir, "--executions", "0", "--frames", str(frames),
           "--runs", str(RUNS_PER_MODE), "--scale", str(WORKING_SCALE),
           "--executions-dir", os.path.join(out_dir, "ab_executions"),
           "--name", name]
    if not write_protocol:
        cmd += ["--protocol-name", ""]
    for path in executions:
        cmd += ["--include", path]
    for path in corroborate:
        cmd += ["--corroborate", path]
    for note in failures:
        cmd += ["--execution-failure", note]
    for path in history:
        cmd += ["--recalculate-history", path]
    proc = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    tail = "\n".join((proc.stdout or "").strip().splitlines()[-25:])
    say({"pooled": name, "returncode": proc.returncode,
         "stderr": (proc.stderr or "")[-800:] or None, "tail": tail})
    return proc.returncode


def history_sources():
    """§41: the read-only sources of the earlier fixed-order batch, and what is wrong with them.

    Read, never written: `evidence_directory_is_writable` is what stops this round's *outputs* from
    landing in a previous round's directory, and this function is what proves the recalculation had
    somewhere honest to read from. A source that does not declare the commit §41 names is reported
    rather than passed on, because `historical_recalc` derives `source_code_under_test` from its
    inputs and would otherwise publish a provenance its inputs do not have.
    """
    found = sorted(glob.glob(os.path.join(HISTORY_EXECUTIONS_DIR,
                                          "execution_*_transfer_ab_benchmark.json")))
    found += [os.path.join(HISTORY_DIR, name) for name in HISTORY_EXTRA_FILES]
    problems = []
    usable = []
    for path in found:
        if not os.path.exists(path):
            problems.append({"source": path, "problem": "missing"})
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                declared = json.load(handle).get("code_under_test")
        except Exception as exc:
            problems.append({"source": path, "problem": f"unreadable: {type(exc).__name__}"})
            continue
        if declared != HISTORY_SOURCE_SHA:
            problems.append({"source": path, "problem": f"declares {declared}",
                             "expected": HISTORY_SOURCE_SHA})
            continue
        usable.append(path)
    return usable, problems


def assemble_status(out_dir, say):
    """§43-§50: let the status collector judge the round out of the artifacts on disk."""
    cmd = [sys.executable, os.path.join(_UNIT_DIR, "collect_transfer_status.py"),
           "--out", out_dir, "--round", "bench_stability"]
    proc = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    say({"status_assembly_returncode": proc.returncode,
         "stdout": "\n".join((proc.stdout or "").strip().splitlines()[-80:]),
         "stderr": (proc.stderr or "")[-2000:] or None})
    return proc.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True,
                        help="this round's evidence directory, perf_results/<sha8>/bench_stability")
    parser.add_argument("--pytest-transcript", default=None,
                        help="Gate 5's captured `pytest -q tests/unit` output, copied in as "
                             "pytest_output.txt (§40); the unit-suite gate reads it")
    parser.add_argument("--stop-after-preflight", action="store_true",
                        help="write benchmark_environment.json and stop: §16 allows waiting for a "
                             "quiet window, and this takes no measurement at all")
    parser.add_argument("--root", default=_ROOT)
    args = parser.parse_args(argv)

    out_dir = os.path.abspath(args.out)
    if sys.stdout is not None:  # the log lines quote §-numbers and process tables; cp does the same
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    say = _logger(out_dir, os.path.join(_ROOT, "bench_out", "run_transfer_stability_round.log"))
    started = utc()

    def not_ready(reason, **detail):
        say({"round_started_utc": started, "status": "NOT_READY", "not_ready_reason": reason,
             **detail})
        return 2

    # ---- everything that can stop the round before it costs the machine forty minutes ----------
    tree = working_tree_is_clean()
    if not tree["ok"]:
        return not_ready("working_tree_not_clean", **tree)
    code_under_test = tree["head"]
    writable = evidence_directory_is_writable(out_dir)
    if not writable["ok"]:
        return not_ready(writable["reason"], **{k: v for k, v in writable.items() if k != "ok"})
    configuration = frozen_configuration()
    if not configuration["ok"]:
        return not_ready("frozen_configuration_disagreed",
                         failed=configuration["failed"], checks=configuration["checks"])
    native = native_scope_against_baseline()
    if not native["ok"]:
        # §42: this is a STOP, not a caveat. Rebuilding a bridge is a different round's work.
        return not_ready("native_binaries_or_sources_changed",
                         production_files_touched=native["production_files_touched"],
                         files=native["files"])
    os.makedirs(out_dir, exist_ok=True)
    say({"round": "transfer benchmark environment stability", "code_under_test": code_under_test,
         "out": os.path.relpath(out_dir, _ROOT).replace("\\", "/"), "started_utc": started,
         "frozen_configuration": (f"{len(configuration['checks'])} checks, all passed"
                                  if configuration["ok"] else "FAILED"),
         "native_scope_against_baseline": BASELINE})

    lock = be.BenchmarkLock(root=args.root, code_under_test=code_under_test,
                            protocol=f"{ab.PROTOCOL_NAME} v{ab.PROTOCOL_VERSION}",
                            command=" ".join(sys.argv))
    try:
        lock.acquire_or_raise()
    except be.BenchmarkLockBusy as busy:
        # §9: refused inside five seconds, and the owner is named rather than waited out.
        return not_ready("benchmark_lock_busy", waited_seconds=busy.waited_seconds,
                         owner=busy.owner, lock_path=lock.path)
    say({"lock_acquired": True, "waited_seconds": lock.waited_seconds, "owner": lock.owner})

    monitor = None
    try:
        # ---- §6: is this machine even allowed to run the benchmark at full speed? ---------------
        # Before the quiet window, not after it: a machine left with an affinity mask or a 99%
        # processor cap produces a perfectly quiet preflight and a uniformly slow batch, and the
        # second measurement is the one nobody notices. Read-only by construction, and §10 says a
        # restriction found here stops the round rather than being cleared by it.
        audit = be.cpu_restriction_audit(code_under_test=code_under_test)
        _write_json(os.path.join(out_dir, "cpu_restriction_audit.json"), audit)
        say({"cpu_restriction_artifact": "cpu_restriction_audit.json",
             "system_logical_cpu_count": audit.get("system_logical_cpu_count"),
             "process_cpu_affinity_count": audit.get("process_cpu_affinity_count"),
             "process_priority_class": audit.get("process_priority_class"),
             "active_power_scheme": audit.get("active_power_scheme"),
             "processor_max_state_percent": audit.get("processor_max_state_percent"),
             "processor_max_state_state": audit.get("processor_max_state_state"),
             "job_object_membership": audit.get("job_object_membership"),
             "restrictions_detected": [row["reason"] for row in audit["restrictions_detected"]]})
        if not audit["cpu_restriction_ok"]:
            return not_ready("cpu_restricted",
                             restrictions=audit["restrictions_detected"],
                             note="§10: the audit found a limit left on the machine, the round "
                                  "stopped before any measurement, and nothing here changed it")
        # ---- §11-§16: one quiet window, up to ten minutes of waiting for it ---------------------
        preflight = be.quiet_preflight(lock=lock, code_under_test=code_under_test, log=say)
        _write_json(os.path.join(out_dir, "benchmark_environment.json"), preflight)
        say({"preflight_artifact": "benchmark_environment.json",
             "preflight_status": preflight.get("preflight_status"),
             "windows_attempted": preflight.get("preflight_window_count_attempted"),
             "cpu_median": preflight.get("cpu_median"), "cpu_p95": preflight.get("cpu_p95"),
             "gpu_util_median": preflight.get("gpu_util_median"),
             "gpu_util_p95": preflight.get("gpu_util_p95"),
             "available_ram_min_gib": preflight.get("available_ram_min_gib")})
        if args.stop_after_preflight:
            say({"stopped_after_preflight": True,
                 "note": "no execution was launched, so there is no batch to judge"})
            return 0 if preflight.get("preflight_status") == "PASS" else 2
        if preflight.get("preflight_status") != "PASS":
            return not_ready("no_quiet_preflight_window",
                             windows_attempted=preflight.get("preflight_window_count_attempted"),
                             failures_of_last_window=[row["reason"] for row
                                                      in (preflight.get("windows") or [{}])[-1]
                                                      .get("failures") or []])

        # ---- §17-§19: watched from the first child to the last ---------------------------------
        monitor = be.RuntimeMonitor(log=say)
        monitor.start()
        say({"monitor_started_utc": monitor.started_utc})
        primary_dir = os.path.join(out_dir, "ab_executions")
        corroboration_dir = os.path.join(out_dir, "ab_executions_150f")
        primary_paths, failures, stopped = run_batch(
            "primary-60f", range(1, PRIMARY_EXECUTIONS + 1), PRIMARY_FRAMES, primary_dir,
            monitor, say)
        corroborating_paths, corroborating_failures, corroborating_stopped = [], [], None
        if monitor.contaminated:
            # §30: the 150-frame batch is corroboration, and it is only admissible if the
            # measurement window was clean. A contaminated one is not rescued by measuring more.
            corroborating_stopped = ("§30's precondition failed: the runtime monitor recorded an "
                                     "external workload, so the 150-frame corroboration was not run")
            say(corroborating_stopped)
        else:
            corroborating_paths, corroborating_failures, corroborating_stopped = run_batch(
                "corroborating-150f", range(1, CORROBORATION_EXECUTIONS + 1),
                CORROBORATION_FRAMES, corroboration_dir, monitor, say)
        # Every reason a launch did not happen is published. §29's rule is that a failure is
        # recorded rather than absorbed, and a batch that stopped early has to say so in the same
        # place the pool is read from, not only in a log a reader would have to already know about.
        for note in (stopped, corroborating_stopped):
            if note:
                failures.append(note)
        failures += corroborating_failures
        monitor_artifact = monitor.stop()
        monitor_artifact["code_under_test"] = code_under_test
        monitor_artifact["batch"] = monitor_batch_record(
            primary_paths, corroborating_paths, stopped, corroborating_stopped, failures,
            monitor.contaminated,
            preflight_passed=preflight.get("preflight_status") != "FAIL")
        _write_json(os.path.join(out_dir, "benchmark_runtime_monitor.json"), monitor_artifact)
        say({"monitor_stopped_utc": monitor_artifact["measurement_end"],
             "samples": monitor_artifact["sample_count"],
             "runtime_contaminated": monitor_artifact["runtime_contaminated"],
             "contamination_reason": monitor_artifact["contamination_reason"],
             "external_workload_events": len(monitor_artifact["external_workload_events"]),
             "benchmark_children": len(monitor_artifact["benchmark_child_pids"]),
             "ffmpeg_children": len(monitor_artifact["ffmpeg_child_pids"])})

        # ---- the arithmetic, by the collectors that the gates re-read --------------------------
        history, history_problems = history_sources()
        if history_problems:
            say({"historical_recalc_sources_unusable": history_problems,
                 "note": "§41's read-only recalculation cannot be published over sources that do "
                         "not all name the commit it claims; the batch above is unaffected and "
                         "still gets its artifacts"})
        if len(history) != 12:
            failures.append(f"the §41 recalculation read {len(history)} of 12 source executions")
        pool(out_dir, CORROBORATION_FRAMES, corroborating_paths,
             "counterbalanced_150f.json", False, say)
        pool(out_dir, PRIMARY_FRAMES, primary_paths, "counterbalanced_performance.json",
             True, say, corroborate=corroborating_paths, failures=failures, history=history)

        if args.pytest_transcript:
            if not os.path.exists(args.pytest_transcript):
                say({"pytest_transcript_missing": args.pytest_transcript})
            else:
                shutil.copyfile(args.pytest_transcript,
                                os.path.join(out_dir, "pytest_output.txt"))
                say({"pytest_transcript_copied_to": "pytest_output.txt",
                     "bytes": os.path.getsize(os.path.join(out_dir, "pytest_output.txt"))})
        assemble_status(out_dir, say)
        status_path = os.path.join(out_dir, "benchmark_stability_status.json")
        verdict = None
        if os.path.exists(status_path):
            with open(status_path, encoding="utf-8") as handle:
                document = json.load(handle)
            verdict = {"status": document.get("status"),
                       "code_under_test": document.get("code_under_test"),
                       "gate_statuses": document.get("gate_statuses"),
                       "not_ready_reasons": sorted(
                           {reason for gate in (document.get("gates") or {}).values()
                            for reason in (gate.get("failed_checks") or [])})}
        say({"round_finished_utc": utc(), "status_file": verdict,
             "primary": {"legacy_and_reduced_kept_medians":
                         (json.load(open(os.path.join(out_dir, "counterbalanced_performance.json"),
                                         encoding="utf-8")).get("verdict") or {}).get(
                             "pooled_kept_median_fps")
                         if os.path.exists(os.path.join(out_dir,
                                                        "counterbalanced_performance.json"))
                         else None}})
        return 0 if (verdict or {}).get("status") == "DONE_PROVEN" else 1
    finally:
        if monitor is not None:
            monitor.stop()
        lock.release()
        say({"lock_released": True, "held_for_seconds": lock.held_for_seconds})


if __name__ == "__main__":
    sys.exit(main())
