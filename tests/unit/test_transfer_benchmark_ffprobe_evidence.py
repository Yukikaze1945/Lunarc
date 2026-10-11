#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regressions for what the ffprobe evidence says, and for the field it is read out of.

Two of this round's three artifact bugs lived in one function. `collect_transfer_status.merge_ffprobe`
aggregated each run's codec off `ffprobe["codec_name"]`, but `benchmark_e2e.ffprobe_validate` renames
ffprobe's `codec_name` to `codec` when it builds the record, so that key was never there. A missing key
reads as None, None sorts, and the pushed evidence of the previous round published exactly that:

    "codecs_seen": [null]                       # 72 runs, every one of them encoded av1
    "runs_whose_probed_frame_count_matches": 0  # packets == expected_frames held for all 72

The second line is the same mistake in a second field: that aggregate compared `.get("frames")`, and
the record spells its counted-frame figure `packets`, with `expected_frames` beside it. Neither bug
crashed, because each produces a value of a legal type - and a test that grepped this file for the
string `codec` would have passed against both. So every test here writes per-execution ffprobe
artifacts in the shape the driver writes them, feeds them through the real `merge_ffprobe`, and
asserts on the numbers that come out. The last class goes one step further and puts a whole synthetic
round through `build_bench_fix`, so the §38 gate is decided by the same code that will decide it over
the real benchmark's output.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import collect_transfer_ab as ab  # noqa: E402
import collect_transfer_status as status  # noqa: E402
import test_transfer_benchmark_provenance as harness  # noqa: E402
import test_transfer_benchmark_protocol as fixtures  # noqa: E402

HEAD = status.git("rev-parse", "HEAD")
WIDTH, HEIGHT, SOURCE_FPS = 3840, 2160, 60.0


def reading(frames):
    """One run's ffprobe record, in the shape `benchmark_e2e.ffprobe_validate` returns it.

    `codec` and `packets` are spelled the way the instrument spells them. That is the point: a
    fixture that spelled them `codec_name` and `frames` would be a second copy of the bug.
    """
    return {"ok": True, "codec": "av1", "packets": frames, "expected_frames": frames,
            "duration_sec": round(frames / SOURCE_FPS, 4), "fps": SOURCE_FPS,
            "width": WIDTH, "height": HEIGHT,
            "checks": {"packets": True, "resolution": True, "fps_matches_source": True,
                       "duration_within_1pct": True}}


def write_json(directory, name, payload):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def artifact_dir(*executions, code_under_test=HEAD):
    """An artifact directory of per-execution ffprobe files, written as the driver writes them.

    Each entry is `(subdirectory, frames, execution_count)`; every execution carries one run per
    order, so `("ab_executions", 60, 1)` is the §2 fixture - two runs, both of them av1.
    """
    out = tempfile.mkdtemp(prefix="bench_ffprobe_")
    for subdirectory, frames, count in executions:
        directory = os.path.join(out, subdirectory)
        for index in range(1, count + 1):
            write_json(directory, f"execution_{index:02d}_ffprobe_output.json", {
                "generated_by": "tests/unit/collect_transfer_ab.py",
                "code_under_test": code_under_test,
                "status": "DONE_PROVEN",
                "expected": {"frames": frames, "width": WIDTH, "height": HEIGHT,
                             "fps": SOURCE_FPS, "codec": ab.EXPECTED_OUTPUT_CODEC},
                "runs": {mode: [{"run": 1, "kept": True, "position_in_pair": 1,
                                 "output": {
                                     "path": f"bench_out/{subdirectory}/e{index}r1-{mode}.mkv",
                                     "sha256": f"{mode}{index}",
                                     "bitstream_sha256": f"bs{index}"},
                                 "ffprobe": reading(frames)}]
                         for mode in ("legacy", "reduced")}})
    return out


def rewrite_records(out, transform, subdirectory="ab_executions", name_prefix=None):
    """Apply `transform(mode, row)` to every probe row and put the files back.

    `name_prefix` narrows the edit to one execution's artifact, so a test that wants exactly one bad
    run out of four can make exactly one bad run.
    """
    directory = os.path.join(out, subdirectory)
    for name in sorted(os.listdir(directory)):
        if not name.endswith("_ffprobe_output.json"):
            continue
        if name_prefix and not name.startswith(name_prefix):
            continue
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        for mode, rows in (doc.get("runs") or {}).items():
            for row in rows:
                transform(mode, row)
        write_json(directory, name, doc)


class CodecAggregationReadsTheFieldThatExists(unittest.TestCase):
    """§2: two runs whose records say `codec: av1` aggregate to `["av1"]`."""

    def _temp(self, out):
        self.addCleanup(shutil.rmtree, out, True)
        return out

    def test_two_av1_runs_aggregate_to_the_codec_name_they_carry(self):
        merged = status.merge_ffprobe(self._temp(artifact_dir(("ab_executions", 60, 1))))
        self.assertEqual(merged["codecs_seen"], ["av1"])
        self.assertEqual(merged["runs_with_an_ffprobe_reading"], 2)

    def test_the_null_the_old_aggregate_published_cannot_come_back(self):
        # The previous round's records are the input here, and the broken aggregate answered
        # `[null]` to this same call. That bracketed null is what a reader of ffprobe_output.json
        # saw beside `"expected": {"frames": 60}`, and it is what this test exists to prevent.
        out = self._temp(artifact_dir(("ab_executions", 60, 12)))
        merged = status.merge_ffprobe(out)
        self.assertNotIn(None, merged["codecs_seen"])
        self.assertEqual(merged["codecs_seen"], ["av1"])
        self.assertEqual(merged["rows_without_a_codec"], [])
        self.assertEqual(merged["codecs_seen_per_order"],
                         {"legacy": ["av1"], "reduced": ["av1"]})

    def test_a_record_with_no_codec_names_itself_rather_than_loosening_the_set(self):
        out = self._temp(artifact_dir(("ab_executions", 60, 1)))
        rewrite_records(out, lambda mode, row:
                        mode == "legacy" and row["ffprobe"].pop("codec", None))
        merged = status.merge_ffprobe(out)
        self.assertEqual(merged["rows_without_a_codec"],
                         ["legacy ab_executions/execution_01_ffprobe_output.json run 1"])
        self.assertFalse(merged["expected_codec_seen_for_every_run"])

    def test_the_aggregate_refuses_to_invent_a_codec_from_a_field_that_is_not_there(self):
        # The inverse guard, and the reason the fix is not simply "read the other key": records
        # spelled the way the buggy reader expected carry no `codec`, so an aggregation that read
        # `codec_name` would report success over data the instrument never produced.
        out = self._temp(artifact_dir(("ab_executions", 60, 1)))
        rewrite_records(out, lambda mode, row:
                        (row["ffprobe"].pop("codec", None),
                         row["ffprobe"].update({"codec_name": "av1"})))
        merged = status.merge_ffprobe(out)
        self.assertEqual(merged["codecs_seen"], [])
        self.assertEqual(len(merged["rows_without_a_codec"]), 2)


class FrameCountAndGeometryAreReRead(unittest.TestCase):
    """The second phantom field, and the numbers behind each probe record's own `ok` flags."""

    def _temp(self, out):
        self.addCleanup(shutil.rmtree, out, True)
        return out

    def test_the_frame_count_is_compared_as_packets_against_expected_frames(self):
        merged = status.merge_ffprobe(self._temp(artifact_dir(("ab_executions", 60, 2))))
        self.assertEqual(merged["runs_whose_probed_frame_count_matches"], 4)
        self.assertEqual(merged["runs_whose_probed_frame_count_disagrees"], [])

    def test_a_run_whose_packets_disagree_with_its_own_expectation_is_named(self):
        # The published count read 0 for 72 good runs when the field name was wrong. Here one bad
        # run in four is what should move the number, and the file should say which one it was.
        out = self._temp(artifact_dir(("ab_executions", 60, 2)))
        rewrite_records(out, lambda mode, row:
                        (mode, row["run"]) == ("legacy", 1)
                        and row["ffprobe"].update({"packets": 59}), name_prefix="execution_01_")
        merged = status.merge_ffprobe(out)
        self.assertEqual(merged["runs_whose_probed_frame_count_matches"], 3)
        self.assertEqual(len(merged["runs_whose_probed_frame_count_disagrees"]), 1)
        self.assertIn("legacy ab_executions/execution_01_ffprobe_output.json run 1",
                      merged["runs_whose_probed_frame_count_disagrees"][0])

    def test_a_record_is_checked_against_the_expectation_of_the_execution_that_wrote_it(self):
        out = self._temp(artifact_dir(("ab_executions", 60, 1)))
        rewrite_records(out, lambda mode, row:
                        mode == "reduced" and row["ffprobe"].update({"width": 1920}))
        merged = status.merge_ffprobe(out)
        self.assertEqual(merged["runs_whose_geometry_matches_its_own_expectation"], 1)
        self.assertEqual(len(merged["rows_with_the_wrong_geometry"]), 1)
        self.assertIn("width 1920 != 3840", merged["rows_with_the_wrong_geometry"][0])

    def test_a_150_frame_run_is_not_judged_against_the_60_frame_expectation(self):
        merged = status.merge_ffprobe(
            self._temp(artifact_dir(("ab_executions", 60, 1), ("ab_executions_150f", 150, 1))))
        self.assertEqual(merged["rows_with_the_wrong_geometry"], [])
        self.assertEqual(merged["runs_whose_probed_frame_count_matches"], 4)
        self.assertEqual({variant["frames"] for variant in merged["expected_variants"]},
                         {60, 150})


class BothBatchesLiveInOneFile(unittest.TestCase):
    """§40: one artifact covers two batches, so it counts them separately."""

    def test_each_batch_is_counted_by_itself(self):
        out = artifact_dir(("ab_executions", 60, 3), ("ab_executions_150f", 150, 2))
        self.addCleanup(shutil.rmtree, out, True)
        merged = status.merge_ffprobe(out)
        self.assertEqual(merged["runs_with_an_ffprobe_reading"], 10)
        self.assertEqual(merged["runs_with_an_ffprobe_reading_by_batch"],
                         {"ab_executions": 6, "ab_executions_150f": 4})
        self.assertEqual(merged["rows_by_batch"], {"ab_executions": 6, "ab_executions_150f": 4})
        self.assertEqual(merged["codecs_seen"], ["av1"])
        batches = sorted(key.split("/")[0] for key in merged["codecs_seen_per_execution"])
        self.assertEqual({name: batches.count(name) for name in set(batches)},
                         {"ab_executions": 3, "ab_executions_150f": 2})

    def test_the_merged_file_keeps_naming_the_code_it_merged(self):
        out = artifact_dir(("ab_executions", 60, 1))
        self.addCleanup(shutil.rmtree, out, True)
        self.assertEqual(status.merge_ffprobe(out)["code_under_test"], HEAD)


class TheDriverPublishesWhatItMeansToCheck(unittest.TestCase):
    """The expectation block has to carry the codec, or the comparison has only one side."""

    def test_an_execution_artifact_states_the_expected_codec(self):
        artifact = ab.ffprobe_artifact(fixtures.execution_payload(index=1), 60)
        self.assertEqual(artifact["expected"]["codec"], "av1")
        self.assertEqual(ab.EXPECTED_OUTPUT_CODEC, "av1")
        self.assertEqual(artifact["expected"]["width"], WIDTH)


class TheSchemaGateIsDecidedByTheAssembler(unittest.TestCase):
    """§38 through `build_bench_fix`, over a round the collectors really built.

    These change the per-execution artifacts the way a real run could - a file ffprobe could not
    parse, a codec nobody asked for - then re-merge and re-assemble, so the answer comes from the
    same chain of calls that will produce the committed evidence rather than from a hand-edited
    summary of it.
    """

    @classmethod
    def setUpClass(cls):
        cls.out = harness.benchmark_dir()
        cls.sha = harness.finish_round(cls.out)
        cls.snapshot = cls.out + "_snapshot"
        shutil.copytree(cls.out, cls.snapshot)

    @classmethod
    def tearDownClass(cls):
        for directory in (cls.out, cls.snapshot):
            shutil.rmtree(directory, ignore_errors=True)

    def setUp(self):
        shutil.rmtree(self.out, ignore_errors=True)
        shutil.copytree(self.snapshot, self.out)

    def _gates(self):
        write_json(self.out, "ffprobe_output.json", status.merge_ffprobe(self.out, self.sha))
        payload = status.build_bench_fix(
            self.out, required=harness.REQUIRED, key_order=status.BENCH_STABILITY_KEYS,
            baseline=status.BENCH_STABILITY_BASELINE, unit_suite_floor=287)
        return payload["gates"], payload

    def test_a_round_whose_records_carry_a_codec_passes_the_schema_gate(self):
        gates, payload = self._gates()
        gate = gates["ffprobe_schema"]
        self.assertEqual(gate["status"], "DONE_PROVEN")
        self.assertEqual(gate["failed_checks"], [])
        self.assertEqual(gate["detail"]["codecs_seen_per_execution"].get(
            "ab_executions/execution_01_ffprobe_output.json"), ["av1"])
        self.assertIn("ffprobe_schema", payload["gate_statuses"])

    def test_the_gate_fails_on_the_blank_the_previous_round_shipped(self):
        rewrite_records(self.out, lambda mode, row:
                        (row["ffprobe"].pop("codec", None),
                         row["ffprobe"].update({"codec_name": "av1"})))
        gate, _ = self._gates()
        self.assertEqual(gate["ffprobe_schema"]["status"], "NOT_READY")
        self.assertIn("codecs_seen_names_codecs_and_not_blanks",
                      gate["ffprobe_schema"]["failed_checks"])

    def test_the_gate_fails_when_a_probe_stops_running_one_of_its_checks(self):
        rewrite_records(self.out, lambda mode, row:
                        row["ffprobe"]["checks"].pop("duration_within_1pct", None))
        gate, _ = self._gates()
        self.assertIn("every_probe_ran_every_check_it_is_asked_to",
                      gate["ffprobe_schema"]["failed_checks"])
        self.assertEqual(gate["ffprobe_schema"]["status"], "NOT_READY")

    def test_the_gate_fails_when_a_codec_is_not_the_one_the_encoder_was_set_to(self):
        rewrite_records(self.out, lambda mode, row: row["ffprobe"].update({"codec": "hevc"}))
        gate, _ = self._gates()
        self.assertIn("the_observed_codecs_are_the_ones_the_encoder_was_set_to",
                      gate["ffprobe_schema"]["failed_checks"])
        self.assertNotIn("codecs_seen_names_codecs_and_not_blanks",
                         gate["ffprobe_schema"]["failed_checks"])

    def test_the_gate_fails_on_a_probe_record_whose_flags_disagree_with_its_numbers(self):
        # `ok: true` over a 3840x1080 file: the flag alone would carry this round, so the gate reads
        # the width, the height and the frame rate instead of taking the boolean's word for it.
        rewrite_records(self.out, lambda mode, row: row["ffprobe"].update({"height": 1080}))
        gate, _ = self._gates()
        self.assertIn("every_output_has_the_resolution_and_frame_rate_it_claimed",
                      gate["ffprobe_schema"]["failed_checks"])

    def test_an_unprobeable_output_fails_the_integrity_key_as_well_as_the_schema_key(self):
        dropped = []

        def drop(mode, row):
            if not dropped:
                dropped.append(f"{mode} {row['run']}")
                row["ffprobe"] = None

        rewrite_records(self.out, drop)
        gates, _ = self._gates()
        self.assertEqual(len(dropped), 1)
        self.assertIn("every_output_file_has_an_ffprobe_reading",
                      gates["ffprobe_schema"]["failed_checks"])
        self.assertIn("every_output_file_of_the_primary_batch_was_probed",
                      gates["output_integrity"]["failed_checks"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
