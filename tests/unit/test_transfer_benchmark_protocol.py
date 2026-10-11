#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regressions for the transfer benchmark's evidence protocol.

What this file holds is the three mistakes the previous round shipped, each expressed as the
instrument's own behaviour rather than as a reading of its source:

  * a pooled verdict that folded each execution's warmup pair back into its medians, its pairs and
    its sign test, so a 24-run experiment was reported as a 36-run one;
  * a benchmark that always launched legacy first, so "reduced is faster" and "reduced went second"
    were the same measurement and could not be told apart;
  * a report that called a transfer REDUCED_PROVEN while its two measured numbers were equal.

So every test here builds an execution artifact the way the driver writes one, feeds it to the code
that actually adjudicates - `trimmed_execution`, `run_rows`, `paired_runs`, `adjudicate`,
`transfer_claim`, `check_transfer_claims` - and asserts on the numbers that come out. Nothing here
greps a collector or looks for a slice in a source file: those prove somebody typed something, not
that the verdict would change.

The fixtures are synthetic and the numbers are made up except where a real measurement is the point
of the test (the decoder row, the upload sizes). A test that passes on invented data and would fail
on the real protocol is exactly what is being asked for.
"""

import json
import os
import statistics
import sys
import unittest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_transfer_ab as ab  # noqa: E402
import collect_transfer_performance as perf  # noqa: E402
import collect_transfer_status as status  # noqa: E402

PAIRS = 3
FRAMES = 60
SCALE = 0.5
CODE = "0123456789abcdef0123456789abcdef01234567"

# The steady-frame counters both orders measure, as the transfer ledgers measured them: the upload
# shrank by a quarter, the CPU conversions went to zero against a non-zero control, and three rows
# - the flow download, the decoder pipe and the encoder pipe - did not move at all.
COUNTERS = {
    "legacy": {"nvof_h2d_bytes_measured": 8294400, "nvof_h2d_copy_count": 1,
               "nvof_d2h_bytes_measured": 16588800, "cpu_bgr_to_rgba_calls": 2,
               "cpu_rgba_to_bgr_calls": 1, "cpu_color_conversion_output_bytes": 22809600,
               "cpu_previous_frame_copy_bytes": 0, "decoder_pipe_bytes": 12441600,
               "encoder_pipe_bytes": 24883200},
    "reduced": {"nvof_h2d_bytes_measured": 6220800, "nvof_h2d_copy_count": 1,
                "nvof_d2h_bytes_measured": 16588800, "cpu_bgr_to_rgba_calls": 0,
                "cpu_rgba_to_bgr_calls": 0, "cpu_color_conversion_output_bytes": 0,
                "cpu_previous_frame_copy_bytes": 0, "decoder_pipe_bytes": 12441600,
                "encoder_pipe_bytes": 24883200},
}


def execution_payload(index=1, fps=None, kept=None, schedule=None, protocol_version=None,
                      gates=None, counters=None, frames=FRAMES, pairs=PAIRS):
    """One execution artifact, in the shape tests/unit/collect_transfer_ab.py writes.

    `fps` maps an order to one figure per pair, in pair order, so a test can set the warmup pair
    apart from the kept ones without touching anything else. A run records the position its
    schedule assigned it, which is what makes the position tests below about the artifact rather
    than about a comment.
    """
    schedule = schedule if schedule is not None else ab.execution_schedule(pairs, index)
    kept = list(kept) if kept is not None else ab.kept_run_ids(pairs)
    fps = fps or {"legacy": [7.0, 7.4, 7.5], "reduced": [7.1, 7.6, 7.7]}
    counters = counters or COUNTERS
    modes = {}
    for mode in ("legacy", "reduced"):
        runs = []
        for entry in schedule:
            position = (entry.get("order") or []).index(mode) + 1
            runs.append({"run": entry["run"], "mode": mode, "position_in_pair": position,
                         "execution_order_index": index,
                         "fps": fps[mode][entry["run"] - 1],
                         "included_in_medians": entry["run"] in kept,
                         "frames_recorded": frames,
                         "steady_state_counters": dict(counters[mode]),
                         "output": {"path": f"bench_out/e{index}r{entry['run']}-{mode}.mkv",
                                    "sha256": f"{mode}{entry['run']}",
                                    "bitstream_sha256": "sharedbitstream"}})
        kept_fps = [run["fps"] for run in runs if run["included_in_medians"]]
        modes[mode] = {
            "runs": runs,
            "runs_used_for_medians": [run["run"] for run in runs
                                      if run["included_in_medians"]],
            "warmup_run_ids": [run["run"] for run in runs if not run["included_in_medians"]],
            "median_fps": round(statistics.median(kept_fps), 4) if kept_fps else None,
            "fps_of_every_run": {run["run"]: run["fps"] for run in runs},
            "fps_of_kept_runs": kept_fps,
            "spans_ms": {"frame_total_p50_ms": 1000.0 / max(kept_fps or [1.0]),
                         "frame_total_p95_ms": 1100.0 / max(kept_fps or [1.0])},
            "steady_state_counters": dict(counters[mode]),
        }
    def default_group(name):
        # The execution's own performance gate is derived from its own kept medians, the way the
        # driver derives it, so a test that makes one order slower can see that gate fail.
        if name == "performance_gate":
            return [{"what": "reduced_median_fps_at_least_legacy",
                     "ok": modes["reduced"]["median_fps"] >= modes["legacy"]["median_fps"]}]
        return [{"what": name, "ok": True}]

    group = default_group if gates is None else gates
    return {
        "generated_by": "tests/unit/collect_transfer_ab.py",
        "code_under_test": CODE,
        "protocol": {"name": ab.PROTOCOL_NAME,
                     "version": protocol_version or ab.PROTOCOL_VERSION},
        "execution_order_index": index,
        "execution_schedule": schedule,
        "run_launch_sequence": [{"run": entry["run"], "mode": mode,
                                 "position_in_pair": position, "kept": entry["kept"]}
                                for entry in schedule
                                for position, mode in enumerate(entry["order"], start=1)],
        "configuration": {"frames_per_run": frames, "runs_per_mode": pairs,
                          "working_scale": SCALE,
                          "warmup_run_ids": ab.warmup_run_ids(pairs), "kept_run_ids": kept},
        "modes": modes,
        "gates": {name: group(name) for name in perf.GATES},
        "speedup_ratio_reduced_over_legacy": round(
            modes["reduced"]["median_fps"] / modes["legacy"]["median_fps"], 4),
        "status": "DONE_PROVEN" if all(check["ok"] for name in perf.GATES
                                       for check in group(name)) else "NOT_READY",
        "output_identity": {
            "bitstream_sha256_of_each_run": {
                mode: {run["run"]: run["output"]["bitstream_sha256"] for run in modes[mode]["runs"]}
                for mode in ("legacy", "reduced")},
            "container_sha256_of_each_run": {
                mode: {run["run"]: run["output"]["sha256"] for run in modes[mode]["runs"]}
                for mode in ("legacy", "reduced")}},
    }


def trimmed(index=1, **kwargs):
    payload = execution_payload(index=index, **kwargs)
    doc = perf.trimmed_execution(payload, f"execution_{index:02d}_transfer_ab_benchmark.json")
    doc["execution"] = f"execution-{index}"
    return doc


def twelve(**kwargs):
    """A batch of twelve executions of the specified protocol: 36 runs, 12 warmups, 24 kept."""
    return [trimmed(index, **kwargs) for index in range(1, 13)]


class WarmupExclusion(unittest.TestCase):
    """§2-§7: the warmup pair is excluded by every statistic, not only by the median."""

    def test_pooled_verdict_excludes_warmup_runs(self):
        executions = twelve()
        verdict = perf.adjudicate(executions, [])
        census = verdict["run_census"]
        self.assertEqual(census["legacy"]["all_run_count"], 36)
        self.assertEqual(census["legacy"]["warmup_run_count"], 12)
        self.assertEqual(census["reduced"]["all_run_count"], 36)
        self.assertEqual(census["reduced"]["warmup_run_count"], 12)
        self.assertEqual(verdict["status"], "DONE_PROVEN")
        # The figure the previous round published: the same twelve executions pooled over all of
        # their runs. Reading it back here is what shows the two are different numbers.
        self.assertEqual(verdict["pooled_all_runs_median_fps_diagnostic"]["legacy"],
                         round(statistics.median(
                             [fps for ex in executions for fps in
                              [7.0, 7.4, 7.5]]), 4))
        self.assertNotEqual(verdict["pooled_kept_median_fps"]["legacy"],
                            verdict["pooled_all_runs_median_fps_diagnostic"]["legacy"])

    def test_extreme_warmup_cannot_flip_verdict(self):
        calm = twelve()
        # One warmup pair per execution made absurdly fast for legacy and absurdly slow for
        # reduced. The previous round's pooled file read those runs, and a batch of them was enough
        # to move a median; this protocol cannot be moved by them at all.
        wild = twelve(fps={"legacy": [60.0, 7.4, 7.5], "reduced": [0.2, 7.6, 7.7]})
        calm_verdict = perf.adjudicate(calm, [])
        wild_verdict = perf.adjudicate(wild, [])
        self.assertEqual(wild_verdict["pooled_kept_median_fps"],
                         calm_verdict["pooled_kept_median_fps"])
        self.assertEqual(wild_verdict["pooled_kept_ratio"], calm_verdict["pooled_kept_ratio"])
        self.assertEqual(wild_verdict["status"], calm_verdict["status"])
        self.assertEqual(wild_verdict["sign_test"]["pairs"], calm_verdict["sign_test"]["pairs"])
        # And the warmup figures are still published, flagged as excluded, so a reader can see what
        # was left out instead of being told nothing at all.
        self.assertTrue(any(not row["included_in_performance_verdict"]
                            for row in perf.run_rows(wild, "reduced")))

    def test_pooled_run_counts_use_kept_runs_only(self):
        executions = twelve()
        census = perf.run_census(executions)
        for mode in ("legacy", "reduced"):
            kept = perf.run_rows(executions, mode, kept_only=True)
            self.assertEqual(len(kept), 24)
            self.assertEqual(census[mode]["kept_run_count"], 24)
            self.assertEqual(perf.pooled_stats(kept)["run_count"], census[mode]["kept_run_count"])
            self.assertTrue(all(row["included_in_performance_verdict"] for row in kept))
            self.assertEqual(census[mode]["kept_run_ids"], [2, 3])
            self.assertEqual(census[mode]["warmup_run_ids"], [1])
        verdict = perf.adjudicate(executions, [])
        # §7: a verdict that still reported 36 against 36 would be the old bug in new clothes.
        self.assertEqual(verdict["kept_orders"]["legacy"]["run_count"], 24)
        self.assertEqual(verdict["kept_orders"]["reduced"]["run_count"], 24)
        self.assertNotEqual(verdict["all_orders_diagnostic"]["legacy"]["run_count"],
                            verdict["kept_orders"]["legacy"]["run_count"])

    def test_paired_sign_test_excludes_warmup(self):
        executions = twelve()
        kept_pairs = perf.paired_runs(executions, kept_only=True)
        all_pairs = perf.paired_runs(executions, kept_only=False)
        self.assertEqual(len(all_pairs), 36)
        self.assertEqual(len(kept_pairs), 24)
        self.assertTrue(all(pair["included_in_performance_verdict"] for pair in kept_pairs))
        self.assertTrue(all(not pair["included_in_performance_verdict"]
                            for pair in all_pairs if pair not in kept_pairs))
        test = perf.sign_test(kept_pairs)
        self.assertEqual(test["pairs"], 24)
        self.assertEqual(test["pairs"] + sum(
            1 for p in kept_pairs if p["reduced_fps"] == p["legacy_fps"]), len(kept_pairs))
        verdict = perf.adjudicate(executions, [])
        self.assertEqual(verdict["sign_test"]["pairs"], 24)
        self.assertIsNotNone(verdict["sign_test"]["used_for"])

    def test_kept_run_ids_must_match_between_modes(self):
        # One execution that kept runs 2 and 3 for legacy but only run 2 for reduced: there is no
        # honest way to pair those, and guessing a correspondence - or pooling the intersection and
        # carrying on - is what this test forbids.
        payload = execution_payload(index=1)
        payload["modes"]["reduced"]["runs_used_for_medians"] = [2]
        broken = perf.trimmed_execution(payload, "execution_01_transfer_ab_benchmark.json")
        broken["execution"] = "execution-1"
        executions = [broken] + twelve()[1:]
        self.assertTrue(broken["unpoolable_reasons"])
        verdict = perf.adjudicate(executions, [])
        self.assertEqual(verdict["status"], "NOT_READY")
        self.assertEqual(len(verdict["rejected_executions"]), 1)
        self.assertEqual(verdict["rejected_executions"][0]["execution"], "execution-1")
        failed = {row["what"] for row in verdict["required_checks"] if not row["ok"]}
        self.assertIn("every_execution_poolable", failed)
        self.assertNotIn("execution-1", verdict["kept_orders"]["reduced"]["executions"])


class CounterbalancedOrder(unittest.TestCase):
    """§8-§13: the schedule balances by design, and the artifact says which schedule it ran."""

    def test_counterbalanced_schedule_is_deterministic(self):
        for index in range(1, 13):
            first = ab.execution_schedule(PAIRS, index)
            self.assertEqual(first, ab.execution_schedule(PAIRS, index),
                             "the same (execution, pair) must always give the same order")
        # §9's example, spelled out: odd executions open legacy-first, even ones reduced-first, and
        # the two kept pairs of one execution run opposite ways.
        self.assertEqual([entry["order"] for entry in ab.execution_schedule(3, 1)],
                         [["legacy", "reduced"], ["legacy", "reduced"], ["reduced", "legacy"]])
        self.assertEqual([entry["order"] for entry in ab.execution_schedule(3, 2)],
                         [["reduced", "legacy"], ["reduced", "legacy"], ["legacy", "reduced"]])
        self.assertEqual([entry["kept"] for entry in ab.execution_schedule(3, 1)],
                         [False, True, True])
        self.assertEqual(ab.kept_run_ids(3), [2, 3])
        self.assertEqual(ab.warmup_run_ids(3), [1])
        self.assertEqual(ab.position_in_pair(3, "legacy", 1), 2)
        self.assertEqual(ab.position_in_pair(3, "reduced", 1), 1)
        with self.assertRaises(ValueError):
            ab.pair_order(0, 1)

    def test_each_execution_balances_kept_pair_positions(self):
        for index in range(1, 13):
            rows = ab.order_balance_rows(ab.execution_schedule(PAIRS, index))
            self.assertEqual(rows["legacy_first"], 1, f"execution {index}")
            self.assertEqual(rows["reduced_first"], 1, f"execution {index}")
            self.assertEqual(len(rows["pairs"]), 2)
            self.assertEqual(sorted(pair["run"] for pair in rows["pairs"]), [2, 3])

    def test_pooled_position_counts_are_equal(self):
        executions = twelve()
        balance = perf.order_balance(perf.paired_runs(executions, kept_only=True))
        self.assertEqual(balance["legacy_first"], 12)
        self.assertEqual(balance["reduced_first"], 12)
        self.assertEqual(balance["unknown_position"], 0)
        self.assertEqual(balance["pairs"], 24)
        self.assertTrue(balance["balanced"])
        verdict = perf.adjudicate(executions, [])
        self.assertEqual(verdict["order_balance"], balance)
        self.assertTrue(next(row for row in verdict["required_checks"]
                             if row["what"] == "kept_pairs_are_position_balanced")["ok"])
        # A batch that ran every pair legacy-first is not balanced, however many runs it has.
        fixed = [trimmed(index, schedule=[{"run": run, "order": ["legacy", "reduced"],
                                           "kept": run > ab.WARMUP_PAIR_COUNT}
                                          for run in range(1, PAIRS + 1)])
                 for index in range(1, 13)]
        unbalanced = perf.adjudicate(fixed, [])
        self.assertEqual(unbalanced["status"], "NOT_READY")
        self.assertFalse(unbalanced["order_balance"]["balanced"])
        self.assertIn("kept_pairs_are_position_balanced",
                      {row["what"] for row in unbalanced["required_checks"] if not row["ok"]})

    def test_execution_artifact_records_actual_order(self):
        payload = execution_payload(index=3)
        ok, detail = status._launch_sequence_matches_schedule(payload)
        self.assertTrue(ok, detail)
        self.assertTrue(status._schedule_follows_the_rule(payload)[0])
        # The same artifact with the launch record rewritten to match a nicer schedule: the check
        # exists to catch precisely this, so it has to fail.
        tampered = json.loads(json.dumps(payload))
        tampered["run_launch_sequence"][2] = {"run": 3, "mode": "legacy",
                                             "position_in_pair": 1, "kept": True}
        self.assertFalse(status._launch_sequence_matches_schedule(tampered)[0])
        # And a schedule that was inferred after the fact rather than recorded before it.
        wrong_rule = json.loads(json.dumps(payload))
        wrong_rule["execution_schedule"] = ab.execution_schedule(PAIRS, 4)
        wrong_rule["run_launch_sequence"] = [
            {"run": entry["run"], "mode": mode, "position_in_pair": position,
             "kept": entry["kept"]}
            for entry in wrong_rule["execution_schedule"]
            for position, mode in enumerate(entry["order"], start=1)]
        self.assertFalse(status._schedule_follows_the_rule(wrong_rule)[0])
        # The pooled side reads the positions out of the artifact, so the pair knows where it sat.
        doc = perf.trimmed_execution(payload, "x.json")
        doc["execution"] = "execution-3"
        pairs = perf.paired_runs([doc], kept_only=True)
        self.assertEqual({(p["run"], p["legacy_position_in_pair"],
                           p["reduced_position_in_pair"]) for p in pairs},
                         {("2", 1, 2), ("3", 2, 1)})

    def test_conflicting_performance_directions_return_not_ready(self):
        # Pooled kept medians say reduced is faster; the paired ratios say it is slower. Two
        # readings of one instrument that disagree is a broken instrument, not a choice between
        # numbers, so the only honest answer here is NOT_READY.
        executions = []
        for index in range(1, 13):
            if index % 2:
                fps = {"legacy": [10.0, 10.0, 10.0], "reduced": [12.0, 12.0, 12.0]}
            else:
                fps = {"legacy": [5.0, 5.0, 5.0], "reduced": [3.5, 3.5, 3.5]}
            executions.append(trimmed(index, fps=fps))
        verdict = perf.adjudicate(executions, [])
        self.assertGreater(verdict["pooled_kept_ratio"], 1.0)
        self.assertLess(verdict["pairwise_median_ratio"], 1.0)
        self.assertFalse(verdict["performance_direction_consistent"])
        self.assertEqual(verdict["status"], "NOT_READY")
        self.assertIn("performance_direction_consistent",
                      {row["what"] for row in verdict["required_checks"] if not row["ok"]})
        self.assertFalse(verdict["no_end_to_end_regression"])

    def test_a_slower_execution_stays_in_the_pool(self):
        # §33: no execution is dropped because it came out badly. One execution whose reduced order
        # is slower is still pooled, still counted, and its ratio still published.
        executions = twelve()
        executions[4] = trimmed(5, fps={"legacy": [7.0, 8.0, 8.0], "reduced": [7.0, 6.0, 6.0]})
        verdict = perf.adjudicate(executions, [])
        self.assertEqual(len(verdict["kept_pairs"]), 24)
        self.assertEqual(len(verdict["per_execution_outcome"]), 12)
        self.assertLess(verdict["per_execution_outcome"]["execution-5"]["median_fps_of_kept_runs"]
                        ["reduced"],
                        verdict["per_execution_outcome"]["execution-5"]
                        ["median_fps_of_kept_runs"]["legacy"])
        self.assertFalse(verdict["per_execution_outcome"]["execution-5"]
                         ["own_performance_gate_all_passed"])
        self.assertEqual(verdict["status"], "DONE_PROVEN",
                         "one slow execution is noise the pooled medians survive, not a veto")


class StatusWords(unittest.TestCase):
    """§19-§20: the word a transfer row carries has to be the word its numbers support."""

    def test_decoder_pipe_cannot_be_reduced_when_bytes_are_unchanged(self):
        claim = status.transfer_claim("decoder_pipe", 12441600, 12441600)
        self.assertEqual(claim["status"], "BLOCKED_PROVEN")
        self.assertTrue(claim["unchanged_this_round"])
        self.assertEqual(claim["saved"], 0)
        # The exact mistake this round was called out to fix: the word the last report gave it.
        rows = status.check_transfer_claims([claim], {"decoder_pipe": "REDUCED_PROVEN"})
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["consistent"])
        self.assertIn("not below", rows[0]["problems"][0])
        # And the table the round publishes fails loudly when that word is in it.
        claims = [status.transfer_claim("decoder_pipe", 12441600, 12441600),
                  status.transfer_claim("nvof_h2d_upload", 8294400, 6220800)]
        good = status.check_transfer_claims(claims, {"decoder_pipe": "BLOCKED_PROVEN",
                                                     "nvof_h2d_upload": "REDUCED_PROVEN"})
        self.assertTrue(all(row["consistent"] for row in good))
        bad = status.check_transfer_claims(claims, {"decoder_pipe": "REDUCED_PROVEN",
                                                    "nvof_h2d_upload": "REDUCED_PROVEN"})
        self.assertEqual([row["claim"] for row in bad if not row["consistent"]],
                         ["decoder_pipe"])
        artifact = status.bench_fix_consistency(
            claims, {"decoder_pipe": "REDUCED_PROVEN", "nvof_h2d_upload": "REDUCED_PROVEN"})
        self.assertFalse(artifact["all_rows_consistent"])
        self.assertEqual([row["claim"] for row in artifact["inconsistent_rows"]],
                         ["decoder_pipe"])
        honest = status.bench_fix_consistency(
            claims, {"decoder_pipe": "BLOCKED_PROVEN", "nvof_h2d_upload": "REDUCED_PROVEN"})
        self.assertTrue(honest["all_rows_consistent"])
        self.assertEqual(honest["decoder_pipe"]["status"], "BLOCKED_PROVEN")
        self.assertEqual(honest["decoder_pipe"]["legacy_bytes"],
                         honest["decoder_pipe"]["reduced_bytes"])
        self.assertTrue(honest["decoder_pipe"]["unchanged_this_round"])
        self.assertEqual([row["claim"] for row in honest["reduced_proven_rows"]],
                         ["nvof_h2d_upload"])
        self.assertEqual(honest["reduced_proven_rows"][0]["saved"], 2073600)
        self.assertEqual(honest["reduced_proven_rows"][0]["saved_percent"], 25.0)

    def test_reduced_status_requires_after_less_than_before(self):
        self.assertEqual(status.derive_transfer_status(100, 50), "REDUCED_PROVEN")
        self.assertEqual(status.derive_transfer_status(100, 100), "BLOCKED_PROVEN")
        self.assertEqual(status.derive_transfer_status(100, 101), "NOT_READY")
        self.assertEqual(status.derive_transfer_status(100, 0, legacy_control=100), "DONE_PROVEN")
        self.assertEqual(status.derive_transfer_status(0, 0, legacy_control=0), "NOT_READY")
        self.assertEqual(status.derive_transfer_status(None, 5), "NOT_READY")
        self.assertEqual(status.derive_transfer_status(5, None), "NOT_READY")
        # Every published word that the numbers do not carry is named, not silently accepted.
        claims = [status.transfer_claim("a", 100, 50), status.transfer_claim("b", 100, 100),
                  status.transfer_claim("c", 100, 0), status.transfer_claim("d", 100, 150),
                  status.transfer_claim("e", None, 5)]
        asserted = {"a": "REDUCED_PROVEN", "b": "BLOCKED_PROVEN", "c": "ELIMINATED_PROVEN",
                    "d": "REDUCED_PROVEN", "e": "NOT_READY"}
        rows = {row["claim"]: row for row in status.check_transfer_claims(claims, asserted)}
        self.assertTrue(rows["a"]["consistent"])
        self.assertTrue(rows["b"]["consistent"])
        self.assertTrue(rows["c"]["consistent"])
        self.assertFalse(rows["d"]["consistent"])
        self.assertTrue(rows["e"]["consistent"])
        # A deletion claim over a non-zero measurement, and a verdict word over an unmeasured row.
        self.assertFalse(status.check_transfer_claims(
            [status.transfer_claim("f", 100, 7)], {"f": "ELIMINATED_PROVEN"})[0]["consistent"])
        self.assertFalse(status.check_transfer_claims(
            [status.transfer_claim("g", None, None)], {"g": "DONE_PROVEN"})[0]["consistent"])
        self.assertFalse(status.check_transfer_claims(
            [status.transfer_claim("h", 100, 100)], {"h": "SOMEHOW_FASTER"})[0]["consistent"])

    def test_claims_are_built_from_this_rounds_own_measurements(self):
        executions = twelve()
        claims, measured = status.bench_fix_claims({"executions": executions})
        by_name = {claim["claim"]: claim for claim in claims}
        self.assertEqual(by_name["decoder_pipe"]["status"], "BLOCKED_PROVEN")
        self.assertEqual(by_name["nvof_h2d_upload"]["status"], "REDUCED_PROVEN")
        self.assertEqual(by_name["cpu_bgr_to_rgba_calls"]["status"], "DONE_PROVEN")
        self.assertEqual(by_name["nvof_flow_d2h_download"]["before"], 16588800)
        reporting = measured["reduced"]["decoder_pipe_bytes"]["executions_reporting"]
        self.assertEqual(reporting, 12)
        # Every row the round publishes, published as its own derived word, is consistent.
        published = {claim["claim"]: claim["status"] for claim in claims}
        table = status.bench_fix_consistency(claims, published)
        self.assertTrue(table["all_rows_consistent"])
        self.assertEqual(table["decoder_pipe"]["status"], "BLOCKED_PROVEN")


if __name__ == "__main__":
    unittest.main()
