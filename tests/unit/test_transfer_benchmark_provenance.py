#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regressions for the benchmark evidence's provenance: who says which code they describe.

The previous round published a status file whose `code_under_test` was a real commit and whose
`artifact_shas` said, in two places, `null`. Nothing in that file was lying about a number; what it
got wrong is the sentence a reader would have built on top of it. `_shared_sha` compared the SHAs
that were present after throwing away the ones that were not, so the agreement it reported meant
"every artifact that bothered to declare a commit declares the same one" - which stays true no
matter how many of the round's artifacts silently declare nothing at all. For a reader deciding
whether twelve pooled runs and one commit describe the same experiment, those two sentences are not
the same sentence, and only the second one is worth a verdict.

So these tests put a declaration, its absence, and a contradiction through the code that actually
assembles a round - `bench_provenance`, `artifact_sha_coverage`, `build_bench_fix`,
`historical_recalc`, `merge_ffprobe` - and assert on the status word that comes out. §5 of the
brief asks for exactly three shapes: X/X/X proven, X/X/missing not ready, X/X/Y not ready.

The artifacts are written to a temporary directory and the FPS figures are synthetic. The commit
they name is this repository's real HEAD, because the collector reads it from there and a test that
invented a SHA would only be testing its own invention.
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
import collect_transfer_performance as perf  # noqa: E402
import collect_transfer_status as status  # noqa: E402
import test_transfer_benchmark_protocol as fixtures  # noqa: E402

HEAD = status.git("rev-parse", "HEAD")
OTHER_COMMIT = "ffffffffffffffffffffffffffffffffffffffff"
REQUIRED = status.BENCH_STABILITY_REQUIRED_ARTIFACTS

# One ffprobe reading per run, in the shape the driver writes: the field is `codec`, and the
# merged file's aggregate has to read it from there.
FFPROBE_READING = {"ok": True, "codec": "av1", "packets": 60, "width": 3840, "height": 2160,
                   "fps": 60.0, "duration_sec": 1.0, "expected_frames": 60,
                   "checks": {"packets": True, "resolution": True, "fps_matches_source": True,
                              "duration_within_1pct": True}}


def write_json(directory, name, payload):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def benchmark_dir(execution_count=12, code_under_test=HEAD):
    """A whole round's artifact directory, assembled by the collectors that normally make it.

    Nothing here hand-writes `counterbalanced_performance.json`: the pooled file, `protocol.json`
    and `historical_warmup_recalc.json` all come out of `collect_transfer_performance.main`, run
    over synthetic execution artifacts with `--executions 0`, so the file under test has the
    provenance fields the real collector gives it and the test cannot pass by matching a shape
    somebody guessed.
    """
    out = tempfile.mkdtemp(prefix="bench_provenance_")
    exec_dir = os.path.join(out, "ab_executions")
    paths = []
    for index in range(1, execution_count + 1):
        payload = fixtures.execution_payload(index=index)
        payload["code_under_test"] = code_under_test
        name = f"execution_{index:02d}_transfer_ab_benchmark.json"
        write_json(exec_dir, name, payload)
        paths.append(os.path.join(exec_dir, name))
        write_json(exec_dir, f"execution_{index:02d}_ffprobe_output.json", {
            "status": "DONE_PROVEN",
            "expected": {"frames": 60, "width": 3840, "height": 2160, "fps": 60.0,
                         "codec": "av1"},
            "code_under_test": code_under_test,
            "runs": {mode: [{"run": run["run"], "kept": run["included_in_medians"],
                             "output": run["output"], "ffprobe": dict(FFPROBE_READING)}
                            for run in payload["modes"][mode]["runs"]]
                     for mode in ("legacy", "reduced")}})
    # `--include` and `--recalculate-history` are append flags, one path per repetition.
    argv = ["--out", out, "--executions", "0", "--frames", "60", "--runs", "3"]
    for path in paths:
        argv += ["--include", path]
    for path in paths:
        argv += ["--recalculate-history", path]
    argv += ["--history-out", "historical_warmup_recalc.json"]
    rc = perf.main(argv)
    assert rc == 0, rc
    for name in ("counterbalanced_150f.json",):
        write_json(out, name, json.load(open(os.path.join(out, "protocol.json"),
                                             encoding="utf-8")))
    return out


def finish_round(out):
    """The derived files the status collector itself writes, so a test starts where main() would.

    Written through `collect_transfer_status`, not copied in as literals: an environment record the
    test invented would test the test.
    """
    perf_doc = status._load(out, "counterbalanced_performance.json")
    sha = status._sha_of(perf_doc)
    consistency = {"decoder_pipe": {"status": "BLOCKED_PROVEN", "legacy_bytes": 12441600,
                                    "reduced_bytes": 12441600, "unchanged_this_round": True},
                   "rows": [{"claim": "decoder_pipe", "before": 12441600, "after": 12441600,
                             "published_status": "BLOCKED_PROVEN", "consistent": True}],
                   "inconsistent_rows": [], "all_rows_consistent": True,
                   "reduced_proven_rows": [], "code_under_test": sha}
    write_json(out, "status_consistency.json", consistency)
    write_json(out, "ffprobe_output.json", status.merge_ffprobe(out, sha))
    write_json(out, "environment.json", {
        "code_under_test": sha, "code_under_test_sha": sha,
        "files_changed_since_baseline": ["tests/unit/collect_transfer_status.py"]})
    with open(os.path.join(out, "pytest_output.txt"), "w", encoding="utf-8") as handle:
        handle.write("287 passed in 134.35s (0:02:14)\n")
    # §3: the three environment records are required artifacts too. Their contents are this
    # module's business in a later commit; here they exist, and they name the round's code.
    for name in ("benchmark_environment", "benchmark_runtime_monitor", "cpu_restriction_audit"):
        write_json(out, f"{name}.json", {"code_under_test": sha})
    return sha


class SharedShaDecidesOnEveryDeclaration(unittest.TestCase):
    """§5: a missing SHA is a disagreement, not an absence to be filtered out."""

    def test_every_artifact_declaring_the_same_commit_shares_it(self):
        self.assertEqual(status._shared_sha(["X", "X", "X"]), "X")

    def test_one_artifact_with_no_declaration_shares_nothing(self):
        self.assertIsNone(status._shared_sha(["X", "X", None]))

    def test_one_artifact_naming_another_commit_shares_nothing(self):
        self.assertIsNone(status._shared_sha(["X", "X", "Y"]))

    def test_the_old_shape_of_this_bug_is_named_explicitly(self):
        # Two of last round's artifacts carried no SHA at all and the check still returned one.
        # This is that exact input, so the regression cannot come back by somebody deciding the
        # blanks were "not really disagreements".
        self.assertIsNone(status._shared_sha([HEAD, HEAD, None, None]))
        self.assertIsNone(status._shared_sha([]))


class CoverageCounts(unittest.TestCase):
    """§37: {required, present, matching, missing, mismatched}, counted off the artifacts."""

    def _coverage(self, named, sha=HEAD):
        return status.artifact_sha_coverage(named, code_under_test=sha)

    def test_a_complete_set_counts_as_complete(self):
        row = self._coverage({"a": {"code_under_test": HEAD}, "b": {"code_under_test": HEAD}})
        self.assertEqual((row["required"], row["present"], row["matching"]), (2, 2, 2))
        self.assertEqual(row["missing"], [])
        self.assertEqual(row["mismatched"], [])
        self.assertEqual(row["without_a_sha"], [])

    def test_an_absent_file_and_a_silent_file_are_counted_apart(self):
        row = self._coverage({"absent": None, "silent": {"status": "DONE_PROVEN"}})
        self.assertEqual((row["required"], row["present"], row["matching"]), (2, 1, 0))
        self.assertEqual(row["missing"], ["absent"])
        self.assertEqual(row["without_a_sha"], ["silent"])

    def test_a_different_commit_is_reported_as_mismatched(self):
        row = self._coverage({"a": {"code_under_test": HEAD},
                              "b": {"code_under_test": OTHER_COMMIT}})
        self.assertEqual(row["matching"], 1)
        self.assertEqual(row["mismatched"], ["b"])

    def test_the_environment_records_are_required_by_name_for_this_round(self):
        # §3 lists the artifacts a verdict reads; two of them did not exist last round, and a
        # gate that quietly kept its old shorter list would pass without them. Round C §6 adds the
        # third: a round that never recorded what the machine was allowed to do cannot be read.
        for name in ("benchmark_environment", "benchmark_runtime_monitor",
                     "cpu_restriction_audit"):
            self.assertIn(name, REQUIRED)
        self.assertEqual(len(REQUIRED), 10)


class ProvenanceGateIsBuiltFromArtifacts(unittest.TestCase):
    """§5/§44: the provenance key is decided by reading a real round's artifact directory."""

    @classmethod
    def setUpClass(cls):
        cls.out = benchmark_dir()
        cls.sha = finish_round(cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.out, ignore_errors=True)

    def _gates(self):
        return status.build_bench_fix(self.out, required=REQUIRED,
                                      key_order=status.BENCH_STABILITY_KEYS,
                                      baseline=status.BENCH_STABILITY_BASELINE,
                                      unit_suite_floor=287)

    def test_a_full_round_is_proven_and_its_coverage_is_complete(self):
        payload = self._gates()
        coverage = payload["artifact_sha_coverage"]
        self.assertEqual(coverage["missing"], [])
        self.assertEqual(coverage["without_a_sha"], [])
        self.assertEqual(coverage["mismatched"], [])
        self.assertEqual(coverage["required"], coverage["present"])
        self.assertEqual(coverage["matching"], coverage["required"])
        self.assertEqual(payload["code_under_test"], HEAD)
        self.assertEqual(payload["gates"]["artifact_provenance"]["status"], "DONE_PROVEN")

    def test_dropping_one_sha_makes_the_round_not_ready(self):
        """X / X / missing on the real assembler: the verdict may not be built on it."""
        quiet = json.load(open(os.path.join(self.out, "ffprobe_output.json"), encoding="utf-8"))
        self.addCleanup(write_json, self.out, "ffprobe_output.json", quiet)
        del quiet["code_under_test"]
        write_json(self.out, "ffprobe_output.json", quiet)
        payload = self._gates()
        self.assertIsNone(payload["code_under_test"])
        self.assertEqual(payload["gates"]["artifact_provenance"]["status"], "NOT_READY")
        self.assertEqual(payload["artifact_sha_coverage"]["without_a_sha"], ["ffprobe_output"])
        self.assertEqual(payload["status"], "NOT_READY")

    def test_one_artifact_from_another_commit_makes_the_round_not_ready(self):
        protocol = json.load(open(os.path.join(self.out, "protocol.json"), encoding="utf-8"))
        self.addCleanup(write_json, self.out, "protocol.json", protocol)
        protocol["code_under_test"] = OTHER_COMMIT
        write_json(self.out, "protocol.json", protocol)
        payload = self._gates()
        self.assertIsNone(payload["code_under_test"])
        self.assertEqual(payload["gates"]["artifact_provenance"]["status"], "NOT_READY")
        self.assertEqual(payload["artifact_sha_coverage"]["mismatched"], ["protocol"])

    def test_a_missing_required_artifact_makes_the_round_not_ready(self):
        monitor = os.path.join(self.out, "benchmark_runtime_monitor.json")
        saved = json.load(open(monitor, encoding="utf-8"))
        self.addCleanup(write_json, self.out, "benchmark_runtime_monitor.json", saved)
        os.remove(monitor)
        payload = self._gates()
        self.assertIn("benchmark_runtime_monitor", payload["artifact_sha_coverage"]["missing"])
        self.assertEqual(payload["gates"]["artifact_provenance"]["status"], "NOT_READY")
        self.assertIsNone(payload["code_under_test"])


class HistoricalRecalcProvenance(unittest.TestCase):
    """§4: a recalculation has two commits, and one field cannot hold both."""

    def test_the_collector_and_the_source_are_recorded_separately(self):
        out = benchmark_dir(execution_count=3)
        self.addCleanup(shutil.rmtree, out, True)
        directory = os.path.join(out, "ab_executions")
        paths = [os.path.join(directory, name) for name in sorted(os.listdir(directory))
                 if name.endswith("_transfer_ab_benchmark.json")]
        # Sources measured at a different commit than the collector reading them now.
        for path in paths:
            doc = json.load(open(path, encoding="utf-8"))
            doc["code_under_test"] = OTHER_COMMIT
            write_json(os.path.dirname(path), os.path.basename(path), doc)
        recalc = perf.historical_recalc(paths)
        self.assertEqual(recalc["code_under_test"], HEAD)
        self.assertEqual(recalc["source_code_under_test"], OTHER_COMMIT)
        self.assertNotEqual(recalc["code_under_test"], recalc["source_code_under_test"])
        self.assertTrue(recalc["source_artifacts_read_only"])

    def test_the_merged_ffprobe_file_inherits_the_sha_of_the_files_it_merged(self):
        out = benchmark_dir(execution_count=2)
        self.addCleanup(shutil.rmtree, out, True)
        merged = status.merge_ffprobe(out)
        self.assertEqual(merged["code_under_test"], HEAD)
        self.assertEqual(set(merged["code_under_test_of_each_merged_source"].values()), {HEAD})

    def test_the_merged_ffprobe_file_refuses_to_invent_a_sha_when_its_sources_disagree(self):
        out = benchmark_dir(execution_count=2)
        self.addCleanup(shutil.rmtree, out, True)
        directory = os.path.join(out, "ab_executions")
        names = sorted(n for n in os.listdir(directory) if n.endswith("_ffprobe_output.json"))
        second = json.load(open(os.path.join(directory, names[1]), encoding="utf-8"))
        second["code_under_test"] = OTHER_COMMIT
        write_json(directory, names[1], second)
        self.assertIsNone(status.merge_ffprobe(out)["code_under_test"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
