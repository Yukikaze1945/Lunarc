#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pool repeated counterbalanced executions of the A/B protocol so the performance gate has power.

Why this exists: §41 fixes the protocol at 3 run pairs per execution with the first dropped,
which makes one execution's figure a median of two runs. On this machine the same order,
unchanged code, the same fixture measured 1.2-52% apart between runs of that protocol. The
difference the performance gate is asked about is around 1%. A median of two cannot separate
those, and an instrument that cannot separate them must not be read either way: not as a pass,
and not as a regression. So the protocol is repeated and the runs pooled.

Two properties of that pooling were gotten wrong in an earlier round of this file and both are
now enforced rather than documented:

  * **the warmup pair never enters a verdict.** An execution defines its own kept run ids -
    everything after the warmup - and publishes them as `runs_used_for_medians`. This collector
    reads that field. It does not decide for itself which runs count, and it certainly does not
    re-add the warmup runs because they happened to be measured too. A pooled figure over 36
    runs where 12 of them are warmups is not a bigger sample, it is a different experiment. The
    warmup figures stay published, labelled `included_in_performance_verdict: false`, so a reader
    can see what warming up did.
  * **the two orders must not be confounded with position.** Each execution counterbalances its
    pairs (see collect_transfer_ab.py), so across the kept pairs of a batch, legacy and reduced
    each sat in first position the same number of times. The pooled verdict checks that this
    actually happened and reports the two position strata separately, so a claim of a difference
    can be seen to hold on both sides of a pair and not only in whichever slot one order kept
    occupying.

The verdict, stated before the data is read:

  * `Primary question`: does the reduced order regress end-to-end performance under a
    counterbalanced protocol?
  * `Primary rule`: the pooled median of the reduced order's *kept* runs is at least the pooled
    median of the legacy order's kept runs.
  * `Required companion gates`: the copy gate of every execution, the output-integrity gate of
    every execution, the order-balance gate over the kept pairs, and - because the two medians
    and the paired ratios are two readings of the same instrument - that those two readings
    agree about which way the difference points.
  * `Secondary evidence`: the paired median percent change, the exact sign test over kept pairs,
    and the position-stratified ratios. These are evidence, not gates: no p-value threshold is
    applied, in either direction, because none was asked for.

An execution whose two orders do not agree on which runs were kept cannot be paired, and nothing
here guesses a correspondence or pools the intersection: it is reported and the verdict becomes
NOT_READY. Executions of a different frame count, or of a protocol that did not counterbalance,
are never pooled with these; the first are corroborations, the second are history.

Each execution keeps its own full artifact under `<out>/ab_executions/`, so every pooled figure
here traces to the run that produced it.

Usage:
    python tests/unit/collect_transfer_performance.py \
        --out perf_results/<sha>/transfer_bench_fix --executions 12 --frames 60 --runs 3
"""

import argparse
import datetime
import json
import math
import os
import re
import statistics
import subprocess
import sys

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_transfer_ab as ab  # noqa: E402

DRIVER = os.path.join(_UNIT_DIR, "collect_transfer_ab.py")
GATES = ("copy_gate", "output_integrity_gate", "performance_gate", "order_balance_gate")
# Gates a pooled verdict requires of every execution it pools. The execution's own performance
# gate is recorded but not required here: it is the pooled median that answers the question, and
# requiring each of a dozen underpowered executions to agree would be a different, stricter rule.
REQUIRED_GATES = ("copy_gate", "output_integrity_gate")
MODES = ("legacy", "reduced")


def _median(values, digits=4):
    numbers = [v for v in values if v is not None]
    if not numbers:
        return None
    out = statistics.median(numbers)
    return round(float(out), digits)


def _mean(values, digits=4):
    numbers = [v for v in values if v is not None]
    return round(sum(numbers) / len(numbers), digits) if numbers else None


def _spread_percent(values):
    """The range of a set of same-code measurements, as a percentage of its low end."""
    numbers = [v for v in values if v]
    if len(numbers) < 2:
        return None
    return round(100.0 * (max(numbers) - min(numbers)) / min(numbers), 2)


def sign_test(pairs, favouring="reduced"):
    """Exact binomial probability of an observed imbalance if both orders were equally fast.

    Ties are dropped rather than split, and `n` is the number of usable pairs, so a small `n`
    reports a small `n` instead of a confident-looking number. Both tails are returned: one is
    the evidence for a speedup, the other the evidence for a regression, so a reviewer can see
    what these runs do and do not say. Neither tail gates anything.
    """
    signs = [1 if (r["reduced_fps"] > r["legacy_fps"]) else 0
             for r in pairs if r["reduced_fps"] != r["legacy_fps"]]
    n = len(signs)
    if not n:
        return None
    wins = sum(signs)
    losses = n - wins

    def tail(k):
        return sum(math.comb(n, j) for j in range(k, n + 1)) / 2.0 ** n

    return {"pairs": n,
            "runs_favouring_reduced": wins,
            "runs_favouring_legacy": losses,
            "p_value_reduced_ahead_by_at_least_this_much": round(tail(wins), 6),
            "p_value_legacy_ahead_by_at_least_this_much": round(tail(losses), 6),
            "used_for": "secondary evidence only; no threshold in this file reads it"}


def _ids(values):
    return sorted(int(v) for v in values or [])


def trimmed_execution(payload, source):
    """What one execution contributes to the pool: its gates, its schedule and every run's FPS.

    The kept run ids are read from the execution, never recomputed here. An artifact that does
    not say which runs it kept, or that does not carry a counterbalanced schedule, is marked
    unpoolable rather than being interpreted.
    """
    modes = payload.get("modes") or {}
    protocol = payload.get("protocol") or {}
    schedule = payload.get("execution_schedule") or []
    out = {
        "source_artifact": source,
        "code_under_test": payload.get("code_under_test"),
        "status": payload.get("status"),
        "frames_per_run": (payload.get("configuration") or {}).get("frames_per_run"),
        "runs_per_mode": (payload.get("configuration") or {}).get("runs_per_mode"),
        "working_scale": (payload.get("configuration") or {}).get("working_scale"),
        "protocol": {"name": protocol.get("name"), "version": protocol.get("version")},
        "execution_order_index": payload.get("execution_order_index"),
        "execution_schedule": schedule,
        "speedup_ratio_reduced_over_legacy": payload.get("speedup_ratio_reduced_over_legacy"),
        "gates": {},
        "orders": {},
        "unpoolable_reasons": [],
    }
    if protocol.get("version") != ab.PROTOCOL_VERSION:
        out["unpoolable_reasons"].append(
            f"protocol version {protocol.get('version')} is not the counterbalanced "
            f"protocol v{ab.PROTOCOL_VERSION} this pool adjudicates")
    if not schedule:
        out["unpoolable_reasons"].append("no execution_schedule was recorded")
    for name in GATES:
        group = (payload.get("gates") or {}).get(name) or []
        out["gates"][name] = {
            "all_passed": bool(group) and all(c.get("ok") for c in group),
            "checks_passed": sum(1 for c in group if c.get("ok")),
            "checks_total": len(group),
            "failed_checks": [c for c in group if not c.get("ok")],
        }
    for mode, report in modes.items():
        every = {str(run.get("run")): run.get("fps") for run in report.get("runs") or []}
        positions = {str(run.get("run")): run.get("position_in_pair")
                     for run in report.get("runs") or []}
        kept = _ids(report.get("runs_used_for_medians"))
        if not kept:
            out["unpoolable_reasons"].append(f"{mode}: no runs_used_for_medians was published")
        out["orders"][mode] = {
            "median_fps_of_kept_runs": report.get("median_fps"),
            "runs_used_for_medians": kept,
            "warmup_run_ids": _ids(report.get("warmup_run_ids")),
            "fps_of_every_run": every,
            "position_in_pair_of_every_run": positions,
            # The noise floor is measured over the runs the verdict reads, not over the warmup
            # too, or the figure describes a sample that no conclusion is taken from.
            "inter_run_spread_percent_of_kept_runs": _spread_percent(
                [every[key] for key in every if int(key) in kept]),
            "inter_run_spread_percent_of_all_runs": _spread_percent(list(every.values())),
            "frame_total_p50_ms": (report.get("spans_ms") or {}).get("frame_total_p50_ms"),
            "frame_total_p95_ms": (report.get("spans_ms") or {}).get("frame_total_p95_ms"),
            "steady_state_counters": dict(report.get("steady_state_counters") or {}),
            "nvof_h2d_bytes_measured": (report.get("steady_state_counters") or {})
            .get("nvof_h2d_bytes_measured"),
            "cpu_bgr_to_rgba_calls": (report.get("steady_state_counters") or {})
            .get("cpu_bgr_to_rgba_calls"),
            "cpu_rgba_to_bgr_calls": (report.get("steady_state_counters") or {})
            .get("cpu_rgba_to_bgr_calls"),
            "decoder_pipe_bytes": (report.get("steady_state_counters") or {})
            .get("decoder_pipe_bytes"),
        }
    kept_match = [out["orders"].get(mode, {}).get("runs_used_for_medians") for mode in MODES]
    if len(set(map(tuple, kept_match))) > 1:
        out["unpoolable_reasons"].append(
            "the two orders kept different run ids, so there is nothing to pair: "
            + json.dumps({mode: out["orders"].get(mode, {}).get("runs_used_for_medians")
                          for mode in MODES}))
    return out


def load_execution(path):
    with open(path, encoding="utf-8") as handle:
        return trimmed_execution(json.load(handle), os.path.basename(path))


def next_execution_label(executions):
    """Continue numbering so a second invocation adds to a pooled file instead of replacing."""
    used = [int(m.group(1)) for ex in executions
            for m in [re.match(r"execution-(\d+)$", str(ex.get("execution") or ""))] if m]
    return max(used) + 1 if used else 1


def run_execution(index, frames, runs, scale, exec_dir):
    """One execution of the protocol, in the driver that defines it.

    The execution's own artifacts are kept under `exec_dir` rather than in a temporary
    directory: the pooled file is a summary of them, and a summary nobody can check the raw
    reading against is the kind of claim this round is not allowed to make.
    """
    os.makedirs(exec_dir, exist_ok=True)
    cmd = [sys.executable, DRIVER, "--out", exec_dir, "--frames", str(frames),
           "--runs", str(runs), "--scale", str(scale), "--execution-index", str(index)]
    proc = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=7200)
    artifact = os.path.join(exec_dir, "transfer_ab_benchmark.json")
    if not os.path.exists(artifact):
        return None, ((proc.stdout or "")[-1200:] + "\n" + (proc.stderr or "")[-1200:])
    kept = os.path.join(exec_dir, f"execution_{index:02d}_transfer_ab_benchmark.json")
    os.replace(artifact, kept)
    ffprobe = os.path.join(exec_dir, "ffprobe_output.json")
    if os.path.exists(ffprobe):
        os.replace(ffprobe, os.path.join(exec_dir, f"execution_{index:02d}_ffprobe_output.json"))
    execution = load_execution(kept)
    execution["source_artifact"] = os.path.relpath(kept, _ROOT).replace("\\", "/")
    return execution, None


# ---------------------------------------------------------------------------
# the two populations of runs an execution publishes
# ---------------------------------------------------------------------------

def run_rows(executions, mode, kept_only=False):
    """One row per individual run of one order, saying whether a verdict is allowed to read it.

    `kept_only` is how every statistic in this file selects its sample. The selection is the
    execution's own published decision, so the same call reads the same rows whether or not the
    caller is computing a median, a ratio or a sign test.
    """
    rows = []
    for ex in executions:
        report = ex["orders"].get(mode) or {}
        kept = set(_ids(report.get("runs_used_for_medians")))
        positions = report.get("position_in_pair_of_every_run") or {}
        for key, fps in sorted((report.get("fps_of_every_run") or {}).items(),
                               key=lambda kv: int(kv[0])):
            in_verdict = int(key) in kept
            rows.append({"execution": ex.get("execution"), "run_index": int(key),
                         "mode": mode, "fps": fps,
                         "position_in_pair": positions.get(key),
                         "frames_per_run": ex.get("frames_per_run"),
                         "included_in_performance_verdict": in_verdict})
    if kept_only:
        rows = [row for row in rows if row["included_in_performance_verdict"]]
    return rows


def pooled_stats(rows):
    """Count, median, mean and spread of a set of runs - of whichever set was passed in."""
    fps = [row["fps"] for row in rows if row["fps"]]
    by_execution = sorted({row["execution"] for row in rows})
    return {
        "run_count": len(fps),
        "median_fps": _median(fps),
        "mean_fps": _mean(fps),
        "min_fps": min(fps) if fps else None,
        "max_fps": max(fps) if fps else None,
        "spread_percent": _spread_percent(fps),
        "per_execution_inter_run_spread_percent": {
            execution: _spread_percent([row["fps"] for row in rows
                                        if row["execution"] == execution])
            for execution in by_execution},
        "executions": by_execution,
        "fps": fps,
    }


def run_census(executions):
    """How many runs there are, how many are warmups, and how many a verdict may read."""
    census = {}
    for mode in MODES:
        rows = run_rows(executions, mode)
        kept = [row for row in rows if row["included_in_performance_verdict"]]
        warm = [row for row in rows if not row["included_in_performance_verdict"]]
        census[mode] = {
            "all_run_count": len(rows),
            "kept_run_count": len(kept),
            "warmup_run_count": len(warm),
            "kept_run_ids": sorted({row["run_index"] for row in kept}),
            "warmup_run_ids": sorted({row["run_index"] for row in warm}),
        }
    return census


def paired_runs(executions, kept_only=True):
    """Pairs of runs taken from the same execution and the same pair index, so the two orders
    saw the same machine within a pair. Warmup pairs are excluded from a verdict's set.
    """
    pairs = []
    for ex in executions:
        legacy = ex["orders"].get("legacy", {}).get("fps_of_every_run", {})
        reduced = ex["orders"].get("reduced", {}).get("fps_of_every_run", {})
        kept = set(_ids(ex["orders"].get("reduced", {}).get("runs_used_for_medians"))
                   or _ids(ex["orders"].get("legacy", {}).get("runs_used_for_medians")))
        lpos = ex["orders"].get("legacy", {}).get("position_in_pair_of_every_run", {})
        rpos = ex["orders"].get("reduced", {}).get("position_in_pair_of_every_run", {})
        for key in sorted(legacy, key=int):
            if not legacy[key] or not reduced.get(key):
                continue
            in_verdict = int(key) in kept
            if kept_only and not in_verdict:
                continue
            pairs.append({"execution": ex["execution"], "run": key,
                          "frames": ex["frames_per_run"],
                          "legacy_fps": legacy[key], "reduced_fps": reduced[key],
                          "legacy_position_in_pair": lpos.get(key),
                          "reduced_position_in_pair": rpos.get(key),
                          "included_in_performance_verdict": in_verdict,
                          "reduced_minus_legacy_percent":
                              round(100.0 * (reduced[key] / legacy[key] - 1.0), 2),
                          "reduced_over_legacy_ratio": round(reduced[key] / legacy[key], 4)})
    return pairs


def order_balance(pairs):
    """How the kept pairs split by which order ran first, counted from the recorded schedules."""
    out = {"legacy_first": 0, "reduced_first": 0, "unknown_position": 0}
    for pair in pairs:
        if pair.get("reduced_position_in_pair") == 2 and pair.get("legacy_position_in_pair") == 1:
            out["legacy_first"] += 1
        elif pair.get("reduced_position_in_pair") == 1:
            out["reduced_first"] += 1
        else:
            out["unknown_position"] += 1
    out["pairs"] = len(pairs)
    out["balanced"] = (out["legacy_first"] == out["reduced_first"] > 0
                       and not out["unknown_position"])
    return out


def position_strata(pairs):
    """The same comparison split by which order went first.

    Diagnostic, not a gate: a stratum can be dominated by noise without invalidating anything, and
    requiring both halves of a 24-pair pool to point the same way would be a stricter rule than the
    one being applied. What is required is that neither stratum is empty, because an "order
    balanced" claim over a single position is not balanced.
    """
    strata = {}
    for name, predicate in (
            ("reduced_vs_legacy_when_reduced_first",
             lambda p: p.get("reduced_position_in_pair") == 1),
            ("reduced_vs_legacy_when_reduced_second",
             lambda p: p.get("reduced_position_in_pair") == 2)):
        chosen = [p for p in pairs if predicate(p)]
        ratios = [p["reduced_over_legacy_ratio"] for p in chosen]
        strata[name] = {
            "pair_count": len(chosen),
            "median_ratio": _median(ratios),
            "median_percent_change": _median(
                [p["reduced_minus_legacy_percent"] for p in chosen], 2),
            "wins": sum(1 for p in chosen if p["reduced_fps"] > p["legacy_fps"]),
            "losses": sum(1 for p in chosen if p["reduced_fps"] < p["legacy_fps"]),
            "executions": sorted({p["execution"] for p in chosen}),
        }
    return strata


def _sign(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    if numerator > denominator:
        return 1
    if numerator < denominator:
        return -1
    return 0


def adjudicate(primary, corroboration):
    """Apply the rule in the docstring to the pooled evidence of one frame count."""
    usable = [ex for ex in primary if not ex.get("unpoolable_reasons")]
    rejected = [{"execution": ex.get("execution"), "source_artifact": ex.get("source_artifact"),
                 "reasons": ex["unpoolable_reasons"]} for ex in primary
                if ex.get("unpoolable_reasons")]

    census = run_census(usable)
    kept_pairs = paired_runs(usable, kept_only=True)
    all_pairs = paired_runs(usable, kept_only=False)
    balance = order_balance(kept_pairs)
    strata = position_strata(kept_pairs)

    kept_orders = {mode: pooled_stats(run_rows(usable, mode, kept_only=True)) for mode in MODES}
    all_orders = {mode: pooled_stats(run_rows(usable, mode)) for mode in MODES}
    l = kept_orders["legacy"]["median_fps"]
    r = kept_orders["reduced"]["median_fps"]
    ratio = round(r / l, 4) if l and r else None
    pairwise_median_ratio = _median([p["reduced_over_legacy_ratio"] for p in kept_pairs])
    test = sign_test(kept_pairs)

    all_gates_ok = all(ex["gates"][g]["all_passed"] for ex in usable + corroboration
                       for g in REQUIRED_GATES)
    pooled_sign = _sign(r, l)
    pairwise_sign = _sign(pairwise_median_ratio, 1.0)
    direction_consistent = (pooled_sign is None or pairwise_sign is None
                            or pooled_sign == 0 or pairwise_sign == 0
                            or pooled_sign == pairwise_sign)

    required = {
        "kept_runs_in_the_verdict": census["legacy"]["kept_run_count"] > 0
                                    and census["legacy"]["kept_run_count"]
                                    == census["reduced"]["kept_run_count"],
        "every_execution_poolable": not rejected,
        "reduced_kept_median_at_least_legacy_kept": bool(l and r and r >= l),
        "copy_and_integrity_gates_passed_in_every_execution": all_gates_ok,
        "kept_pairs_are_position_balanced": balance["balanced"],
        "both_position_strata_have_samples": all(
            s["pair_count"] > 0 for s in strata.values()),
        "performance_direction_consistent": direction_consistent,
    }
    status = "DONE_PROVEN" if usable and all(required.values()) else "NOT_READY"
    reasons = {
        "kept_runs_in_the_verdict": f"{census['legacy']['kept_run_count']} legacy / "
                                    f"{census['reduced']['kept_run_count']} reduced kept runs",
        "every_execution_poolable": f"{len(rejected)} of {len(primary)} executions rejected",
        "reduced_kept_median_at_least_legacy_kept": f"{r} vs {l}",
        "copy_and_integrity_gates_passed_in_every_execution": "every pooled and corroborating "
                                                              "execution passed both",
        "kept_pairs_are_position_balanced": f"{balance['legacy_first']} legacy-first / "
                                            f"{balance['reduced_first']} reduced-first of "
                                            f"{balance['pairs']} kept pairs",
        "both_position_strata_have_samples": json.dumps(
            {name: s["pair_count"] for name, s in strata.items()}),
        "performance_direction_consistent": "pooled kept medians point "
                                            f"{'+' if (pooled_sign or 0) > 0 else '-' if (pooled_sign or 0) < 0 else 'nowhere'}"
                                            " and the paired median ratio points "
                                            f"{'+' if (pairwise_sign or 0) > 0 else '-' if (pairwise_sign or 0) < 0 else 'nowhere'}",
    }
    return {
        "primary_question": "Does the reduced order regress end-to-end performance under a "
                            "counterbalanced protocol?",
        "primary_rule": "the pooled median of the reduced order's kept runs is at least the "
                        "pooled median of the legacy order's kept runs",
        "required_companion_gates": ["copy", "output integrity", "order balance",
                                     "direction consistency"],
        "secondary_evidence": ["paired median percent change", "exact sign test over kept pairs",
                               "position-stratified ratios"],
        "rule": "Warmup pairs are published and excluded: an execution names its kept run ids "
                "and only those reach a median, a ratio, a pair or the sign test. The gate asks "
                "the non-regression question and nothing stronger - no p-value threshold is "
                "applied in either direction - but it does require the instrument to have been "
                "built balanced, and it requires the two readings of that instrument, the pooled "
                "medians and the paired ratios, to agree about which way they point. If they "
                "point opposite ways the honest answer is that the instrument gave a "
                "contradiction, which is NOT_READY rather than a choice of the nicer number.",
        "required_checks": [
            {"what": name, "expected": True, "actual": bool(value), "ok": bool(value),
             "detail": reasons[name]} for name, value in required.items()],
        "all_required_checks_passed": status == "DONE_PROVEN",
        "no_end_to_end_regression": bool(l and r and r >= l) and all_gates_ok
                                    and balance["balanced"] and direction_consistent,
        "performance_direction_consistent": bool(direction_consistent),
        "pooled_kept_median_fps": {"legacy": l, "reduced": r},
        "pooled_all_runs_median_fps_diagnostic": {
            mode: all_orders[mode]["median_fps"] for mode in MODES},
        "pooled_kept_ratio": ratio,
        "pooled_all_runs_ratio_diagnostic": (
            round(all_orders["reduced"]["median_fps"] / all_orders["legacy"]["median_fps"], 4)
            if all_orders["legacy"]["median_fps"] and all_orders["reduced"]["median_fps"]
            else None),
        "pairwise_median_ratio": pairwise_median_ratio,
        "median_reduced_minus_legacy_percent": _median(
            [pair["reduced_minus_legacy_percent"] for pair in kept_pairs], 2),
        "sign_test": test,
        "evidence_that_reduced_is_slower_p_value": (test or {}).get(
            "p_value_legacy_ahead_by_at_least_this_much"),
        "run_census": census,
        "kept_orders": {mode: {k: v for k, v in kept_orders[mode].items() if k != "fps"}
                        for mode in MODES},
        "all_orders_diagnostic": {mode: {k: v for k, v in all_orders[mode].items()
                                         if k != "fps"} for mode in MODES},
        "order_balance": balance,
        "position_strata": strata,
        "kept_pairs": kept_pairs,
        "warmup_pairs_diagnostics": [p for p in all_pairs
                                     if not p["included_in_performance_verdict"]],
        "rejected_executions": rejected,
        "per_execution_outcome": {
            ex["execution"]: {
                "status": ex["status"],
                "protocol": ex["protocol"],
                "unpoolable_reasons": ex.get("unpoolable_reasons") or [],
                "kept_run_ids": {mode: ex["orders"].get(mode, {}).get("runs_used_for_medians")
                                 for mode in MODES},
                "median_fps_of_kept_runs": {mode: ex["orders"].get(mode, {})
                                            .get("median_fps_of_kept_runs") for mode in MODES},
                "own_performance_gate_all_passed":
                    ex["gates"]["performance_gate"]["all_passed"],
                "copy_gate_all_passed": ex["gates"]["copy_gate"]["all_passed"],
                "output_integrity_gate_all_passed":
                    ex["gates"]["output_integrity_gate"]["all_passed"],
                "order_balance_gate_all_passed":
                    ex["gates"]["order_balance_gate"]["all_passed"],
                "kept_pair_positions": ex["execution_schedule"],
                "inter_run_spread_percent": {
                    mode: ex["orders"].get(mode, {})
                        .get("inter_run_spread_percent_of_kept_runs") for mode in MODES},
                "decoder_pipe_bytes": {mode: ex["orders"].get(mode, {})
                                       .get("decoder_pipe_bytes") for mode in MODES},
            } for ex in primary},
        "corroboration": corroboration_summary(corroboration),
        "status": status,
    }


def corroboration_summary(corroboration):
    """A different frame count, reported beside the verdict and never pooled with it."""
    out = {}
    for ex in corroboration:
        if ex.get("unpoolable_reasons"):
            out[ex["execution"]] = {"unpoolable_reasons": ex["unpoolable_reasons"],
                                    "note": "reported as recorded; it is not evidence about the "
                                            "counterbalanced protocol"}
            continue
        pairs = paired_runs([ex], kept_only=True)
        balance = order_balance(pairs)
        out[ex["execution"]] = {
            "frames_per_run": ex["frames_per_run"],
            "runs_per_mode": ex["runs_per_mode"],
            "run_census": run_census([ex]),
            "median_fps_of_kept_runs": {mode: ex["orders"].get(mode, {})
                                        .get("median_fps_of_kept_runs") for mode in MODES},
            "warmup_fps": {mode: {str(run): fps for run, fps in
                                  (ex["orders"].get(mode, {}).get("fps_of_every_run") or {}).items()
                                  if int(run) not in _ids(
                                      ex["orders"].get(mode, {}).get("runs_used_for_medians"))}
                           for mode in MODES},
            "kept_pairs": len(pairs),
            "kept_pair_positions": balance,
            "paired_runs_favouring_reduced": sum(1 for p in pairs
                                                 if p["reduced_fps"] > p["legacy_fps"]),
            "median_reduced_over_legacy_ratio": _median(
                [p["reduced_over_legacy_ratio"] for p in pairs]),
            "own_status": ex["status"],
            "source_artifact": ex["source_artifact"],
            "note": "corroboration only: never pooled with the 60-frame verdict, because runs of "
                    "a different length are not more samples of the same experiment",
        }
    return out


def protocol_record(primary, usable, args, verdict=None):
    """§36: what the protocol was, and the schedule each execution actually ran."""
    kept_pairs = paired_runs(usable, kept_only=True)
    balance = order_balance(kept_pairs)
    census = run_census(usable)
    return {
        "generated_by": "tests/unit/collect_transfer_performance.py",
        "code_under_test": ab.git("rev-parse", "HEAD"),
        "protocol": {"name": ab.PROTOCOL_NAME, "version": ab.PROTOCOL_VERSION},
        "frames_per_run": args.frames,
        "runs_per_mode": args.runs,
        "warmup_run_ids": ab.warmup_run_ids(args.runs),
        "kept_run_ids": ab.kept_run_ids(args.runs),
        "execution_count": len(primary),
        "poolable_execution_count": len(usable),
        "schedule_rule": "odd executions launch legacy first and even executions reduced first; "
                         "the kept pairs alternate, so each execution's kept pairs hold one pair "
                         "of each position. Deterministic by (execution index, pair index), not "
                         "random.",
        "identical_across_orders": "fixture, frame count, working scale, NVOF grid and preset, "
                                   "DLSS options, depth convention, encoder, CQ and output "
                                   "resolution; only transfer_mode differs",
        "per_execution_schedule": [
            {"execution": ex.get("execution"),
             "execution_order_index": ex.get("execution_order_index"),
             "code_under_test": ex.get("code_under_test"),
             "protocol": ex.get("protocol"),
             "source_artifact": ex.get("source_artifact"),
             "execution_schedule": ex.get("execution_schedule") or [],
             "unpoolable_reasons": ex.get("unpoolable_reasons") or []} for ex in primary],
        "raw_run_count_per_mode": {mode: census[mode]["all_run_count"] for mode in MODES},
        "kept_run_count_per_mode": {mode: census[mode]["kept_run_count"] for mode in MODES},
        "warmup_run_count_per_mode": {mode: census[mode]["warmup_run_count"] for mode in MODES},
        "legacy_first_kept_count": balance["legacy_first"],
        "reduced_first_kept_count": balance["reduced_first"],
        "kept_pair_count": balance["pairs"],
        "verdict_uses": "kept runs only",
    }


def historical_recalc(paths):
    """Recompute the earlier fixed-order batch with its warmups excluded, read-only.

    This is a demonstration of the pooling bug, not a verdict: that batch ran every pair
    legacy-first, so even with its warmups removed it is still a comparison where one order always
    went second. It is kept to show how large the effect of the mistake was, and nothing here
    promotes it to evidence about speed.
    """
    executions = []
    for index, path in enumerate(paths, start=1):
        with open(path, encoding="utf-8") as handle:
            execution = trimmed_execution(json.load(handle), os.path.basename(path))
        execution["execution"] = f"historical-{index}"
        execution["unpoolable_reasons"] = []  # deliberately pooling a non-counterbalanced batch
        executions.append(execution)
    census = run_census(executions)
    kept = {mode: pooled_stats(run_rows(executions, mode, kept_only=True)) for mode in MODES}
    every = {mode: pooled_stats(run_rows(executions, mode)) for mode in MODES}
    kept_pairs = paired_runs(executions, kept_only=True)
    all_pairs = paired_runs(executions, kept_only=False)

    def ratio(orders):
        l, r = orders["legacy"]["median_fps"], orders["reduced"]["median_fps"]
        return round(r / l, 4) if l and r else None

    # §4: this file is two rounds of work at once - the current collector reading last round's
    # artifacts. Those are different code, and one field cannot describe both. The collector's own
    # commit goes in `code_under_test`, the commit the numbers were made at goes in
    # `source_code_under_test`, read off the source artifacts rather than typed in here so this
    # file cannot claim a provenance its inputs do not have.
    source_shas = sorted({ex.get("code_under_test") for ex in executions
                          if ex.get("code_under_test")})
    return {
        "generated_by": "tests/unit/collect_transfer_performance.py",
        "code_under_test": ab.git("rev-parse", "HEAD"),
        "source_code_under_test": source_shas[0] if len(source_shas) == 1 else None,
        "source_code_under_test_values": source_shas,
        "source_artifacts_read_only": True,
        "provenance_note": "code_under_test is the collector that recomputed these numbers; "
                           "source_code_under_test is the code the source artifacts were measured "
                           "on. The sources were opened read-only and none of their files was "
                           "written to, so the earlier round's published verdict still stands "
                           "beside this recalculation unchanged",
        "read_only_source": sorted({os.path.dirname(os.path.abspath(p)) for p in paths}),
        "sources": [os.path.basename(p) for p in paths],
        "executions": len(executions),
        "what_this_proves": "that the earlier pooled verdict read 36 runs per order where the "
                            "protocol kept 24: it folded each execution's warmup pair back into "
                            "the median and the sign test. The two columns below are the same "
                            "runs, separated only by whether the warmup pair was allowed to "
                            "count.",
        "what_this_does_not_prove": "anything about speed under a counterbalanced protocol. Every "
                                    "pair of this batch ran legacy first, so reduced always "
                                    "occupied the second position, and removing warmups does not "
                                    "remove that.",
        "old_all_run_count_per_order": {mode: census[mode]["all_run_count"] for mode in MODES},
        "correct_kept_run_count_per_order": {mode: census[mode]["kept_run_count"]
                                             for mode in MODES},
        "warmup_run_count_per_order": {mode: census[mode]["warmup_run_count"] for mode in MODES},
        "old_all_run_median_legacy": every["legacy"]["median_fps"],
        "old_all_run_median_reduced": every["reduced"]["median_fps"],
        "correct_kept_median_legacy": kept["legacy"]["median_fps"],
        "correct_kept_median_reduced": kept["reduced"]["median_fps"],
        "old_ratio": ratio(every),
        "corrected_ratio": ratio(kept),
        "old_pair_count": len(all_pairs),
        "correct_pair_count": len(kept_pairs),
        "old_sign_test": sign_test(all_pairs),
        "corrected_sign_test": sign_test(kept_pairs),
        "old_median_reduced_minus_legacy_percent": _median(
            [p["reduced_minus_legacy_percent"] for p in all_pairs], 2),
        "corrected_median_reduced_minus_legacy_percent": _median(
            [p["reduced_minus_legacy_percent"] for p in kept_pairs], 2),
        "verdict_would_change": (ratio(every) is not None and ratio(kept) is not None
                                 and (ratio(every) >= 1.0) != (ratio(kept) >= 1.0)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="artifact directory")
    parser.add_argument("--executions", type=int, default=12,
                        help="how many times to run the protocol")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--include", action="append", default=[],
                        help="an already-recorded artifact of this same protocol to pool in")
    parser.add_argument("--execution-failure", action="append", default=[],
                        help="a failure a driver already observed and recorded, to be published "
                             "beside the pool rather than lost when the pooling is a second process")
    parser.add_argument("--corroborate", action="append", default=[],
                        help="a recorded artifact of a different frame count to report "
                             "alongside, never pooled with")
    parser.add_argument("--executions-dir", default="",
                        help="where an execution's own artifacts are kept "
                             "(default: <out>/ab_executions)")
    parser.add_argument("--name", default="counterbalanced_performance.json",
                        help="the pooled artifact's file name")
    parser.add_argument("--protocol-name", default="protocol.json",
                        help="where to write the §36 protocol record")
    parser.add_argument("--recalculate-history", action="append", default=[],
                        help="execution artifacts of an earlier protocol to recompute with the "
                             "warmup excluded, read-only")
    parser.add_argument("--history-out", default="historical_warmup_recalc.json")
    args = parser.parse_args(argv)

    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    exec_dir = args.executions_dir or os.path.join(args.out, "ab_executions")
    primary = []
    # A driver that ran the executions in its own process - so it could hold the lock and watch the
    # machine while they ran - knows about children that crashed and batches it had to stop early.
    # Those reports arrive through this flag and are published beside the pool: §29 records every
    # failure, and a pooled file that showed eleven executions and said nothing about the twelfth
    # would be the one place a stopped batch could disappear through.
    failures = list(args.execution_failure)
    for path in args.include:
        try:
            execution = load_execution(path)
        except Exception as error:
            failures.append(f"{path}: {error}")
            continue
        execution["execution"] = f"recorded-{len(primary) + 1}"
        primary.append(execution)
    first = next_execution_label(primary)
    for index in range(first, first + args.executions):
        print(f"[{datetime.datetime.now():%H:%M:%S}] execution {index} "
              f"({args.frames} frames x {args.runs} pairs per order, "
              f"schedule {json.dumps(ab.execution_schedule(args.runs, index))})", flush=True)
        execution, error = run_execution(index, args.frames, args.runs, args.scale, exec_dir)
        if execution is None:
            failures.append(f"execution {index} produced no artifact: {error}")
            continue
        execution["execution"] = f"execution-{index}"
        primary.append(execution)
        print("   " + json.dumps({
            "execution": execution["execution"],
            "ratio": execution["speedup_ratio_reduced_over_legacy"],
            "kept_medians": {mode: report["median_fps_of_kept_runs"]
                             for mode, report in sorted(execution["orders"].items())},
            "unpoolable": execution.get("unpoolable_reasons") or None}, sort_keys=True), flush=True)

    corroboration = []
    for path in args.corroborate:
        try:
            execution = load_execution(path)
        except Exception as error:
            failures.append(f"{path}: {error}")
            continue
        execution["execution"] = f"corroborating-{os.path.basename(path)}"
        corroboration.append(execution)

    same_shape = [ex for ex in primary
                  if ex["frames_per_run"] == args.frames and ex["working_scale"] == args.scale]
    usable = [ex for ex in same_shape if not ex.get("unpoolable_reasons")]
    verdict = adjudicate(same_shape, corroboration) if same_shape else {
        "status": "NOT_READY",
        "rule": "no execution of the protocol produced an artifact"}
    payload = {
        "generated_by": "tests/unit/collect_transfer_performance.py",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "code_under_test": ab.git("rev-parse", "HEAD"),
        "protocol": {"name": ab.PROTOCOL_NAME, "version": ab.PROTOCOL_VERSION,
                     "defined_by": "tests/unit/collect_transfer_ab.py",
                     "frames_per_run": args.frames,
                     "runs_per_mode": args.runs,
                     "warmup_run_ids": ab.warmup_run_ids(args.runs),
                     "kept_run_ids": ab.kept_run_ids(args.runs),
                     "child_process_per_run": True,
                     "pairs_interleaved_within_an_execution": True,
                     "schedule_counterbalanced": True,
                     "executions_pooled": len(usable),
                     "executions_rejected": len(same_shape) - len(usable)},
        "why_pooled": "one execution of the protocol is a median of two runs, and the same order "
                      "of the same code measures 1.2-52% apart between runs on this machine; that "
                      "instrument cannot resolve the difference the performance gate is asked "
                      "about, in either direction",
        "warmup_policy": "an execution publishes runs_used_for_medians; only those runs reach a "
                         "pooled median, a paired ratio or the sign test. The warmup runs are "
                         "still in this file, with included_in_performance_verdict=false",
        "executions": primary,
        "corroborating_executions": corroboration,
        "all_runs_per_order": {mode: run_rows(same_shape, mode) for mode in MODES},
        "kept_runs_per_order": {mode: run_rows(same_shape, mode, kept_only=True)
                                for mode in MODES},
        "warmup_runs_per_order": {mode: [row for row in run_rows(same_shape, mode)
                                         if not row["included_in_performance_verdict"]]
                                  for mode in MODES},
        "pooled_kept_runs_median": (verdict or {}).get("pooled_kept_median_fps")
                                   or {mode: None for mode in MODES},
        "pooled_all_runs_median_diagnostic": (verdict or {}).get(
            "pooled_all_runs_median_fps_diagnostic") or {mode: None for mode in MODES},
        "verdict": verdict,
        "execution_failures": failures,
    }
    payload["run_counts_per_order"] = {
        mode: {"all": len(payload["all_runs_per_order"][mode]),
               "kept": len(payload["kept_runs_per_order"][mode]),
               "warmup": len(payload["warmup_runs_per_order"][mode])} for mode in MODES}

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, args.name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    if args.protocol_name:
        record = protocol_record(primary, usable, args, verdict)
        with open(os.path.join(args.out, args.protocol_name), "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
    if args.recalculate_history:
        history = historical_recalc(args.recalculate_history)
        with open(os.path.join(args.out, args.history_out), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, sort_keys=True)
        print(json.dumps({"historical_recalc": args.history_out,
                          "old_ratio": history["old_ratio"],
                          "corrected_ratio": history["corrected_ratio"],
                          "old_kept": f'{history["old_all_run_count_per_order"]}',
                          "corrected_kept": f'{history["correct_kept_run_count_per_order"]}'},
                         sort_keys=True), flush=True)

    print(json.dumps({"ok": payload["verdict"]["status"] == "DONE_PROVEN",
                      "artifact": path, "status": payload["verdict"]["status"],
                      "pooled_kept_median_fps": payload["pooled_kept_runs_median"],
                      "pooled_all_runs_median_diagnostic":
                          payload["pooled_all_runs_median_diagnostic"],
                      "run_counts_per_order": payload["run_counts_per_order"],
                      "ratio": payload["verdict"].get("pooled_kept_ratio"),
                      "pairwise_median_ratio": payload["verdict"].get("pairwise_median_ratio"),
                      "order_balance": payload["verdict"].get("order_balance"),
                      "position_strata": {k: v["pair_count"] for k, v in
                                          (payload["verdict"].get("position_strata")
                                           or {}).items()},
                      "sign_test": payload["verdict"].get("sign_test"),
                      "failed_required_checks": [
                          c for c in payload["verdict"].get("required_checks", []) if not c["ok"]],
                      "failures": failures},
                     indent=2, sort_keys=True, default=str))
    return 0 if payload["verdict"]["status"] == "DONE_PROVEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
