#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§22: regressions for the two instruments that decide whether a batch may be read at all.

The round before this one decided a half-percent question on a machine it could not describe. These
tests exist so that "the machine was quiet and nobody else was here" becomes something the suite can
show, on data this file makes up, rather than something a report asserts.

Everything here is synthetic. That is §22's requirement and it is also the point: a test that samples
the real CPU would pass on a quiet laptop and fail on this build host, and either result would say
nothing about the rule. So `sample_fn`, `sleep`, `monotonic`, the process-table reader and the
descendant list are all handed in, and the machine the tests run on is not consulted.

Three groups:

  * **§22's nine**, one per named behaviour: a window that passes, a CPU that is too busy, a GPU that
    is too busy, a repository workload, an external GPU compute process, a metric the driver does not
    report, an outsider that stays three samples, an outsider that appears twice and leaves, and a
    second lock owner.
  * **Attribution.** §19 says the benchmark's own FFmpeg is not contamination, and §17 says nothing
    may fail a batch for loading the machine while it runs. Both are properties of the monitor rather
    than of a filter, so both are tested by feeding the monitor rows and reading what it writes.
  * **The gates that read the artifacts.** A record a gate trusts is a record somebody can write by
    hand, so each gate is fed the stub it should refuse - a short window, a threshold edited towards
    a result, a busy window flagged PASS, a `sample_count` that disagrees with its own list - and one
    genuine artifact, produced by the same function running for real over injected samples, that it
    must accept.
  * **§8's named regressions**, added after the round that stopped itself over a process that was
    nobody's competitor: who owns which process tree and when (a tree read per sample, a recycled
    pid, the child spawned between the two reads inside one sample, the benchmark's own FFmpeg and
    FFprobe), which quiet window a gate may certify (nine refusals
    then a pass, no window at all, a window that arrives after the budget, a selected window carrying
    a conflict), the six reasons a corroboration can be missing, the six GPU client classes with
    §5's four positive criteria each tested on its own and §5's ambient identity held to one exact
    file name, and §6's three CPU readings that stop a round plus the honesty of the fields around
    them. None of these reads this file's own text: each one
    calls the classifier, the preflight loop or the gate itself.
"""

import contextlib
import datetime
import json
import os
import sys
import unittest
from unittest import mock

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_environment as be  # noqa: E402
import collect_transfer_status as status  # noqa: E402
import run_transfer_stability_round as driver  # noqa: E402

CODE = "0123456789abcdef0123456789abcdef01234567"
GIB = 1024 ** 3

# §20 and §21 list the fields their artifacts must carry, and say 至少记录 - at least. So the tests
# below check every name the brief gave, not the subset that happens to be convenient to produce:
# the artifact is what a reader audits the round against, and a field missing under its brief's own
# name is a finding whoever else noticed it.
BRIEF_20_FIELDS = ("lock_acquired", "lock_owner", "preflight_start", "preflight_end",
                   "preflight_window_count_attempted", "preflight_window_that_passed",
                   "thresholds", "all_preflight_samples", "cpu_median", "cpu_p95",
                   "gpu_util_median", "gpu_util_p95", "available_ram_min",
                   "external_compute_processes", "conflicting_repo_processes",
                   "preflight_status")
BRIEF_21_FIELDS = ("measurement_start", "measurement_end", "samples",
                   "external_workload_events", "runtime_contaminated", "contamination_reason",
                   "first_contamination_timestamp", "benchmark_child_pids", "ffmpeg_child_pids")


class FakeTime:
    """A clock that only moves when the code under test asks to wait.

    The preflight spends sixty seconds of wall clock looking for a quiet thirty-second window. A test
    that reproduced that with `time.sleep` would be a slow test that proved nothing extra, so the
    budget is simulated - and simulated *by the sleep calls*, which is what makes the ten-minute
    stop in §15 observable: the loop ends when the clock it reads has passed the budget.
    """

    def __init__(self, start=0.0):
        self.now = float(start)

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += float(seconds)


def gpu_row(pid=4242, name="explorer.exe", memory="[N/A]", created=None, parent=None, **facts):
    """One row of nvidia-smi's client table, as an *input* to the real classifier.

    The three columns `--query-compute-apps` answers with, plus whatever the OS and `pmon` are made to
    say about that pid: `command_line`, `executable`, and the engine columns `sm`, `mem`, `enc`, `dec`,
    `jpg`, `ofa` (and `engine_type`, which §5's second criterion deliberately does not read). Those
    extras exist because §5's four criteria are measured from them, so a fixture that wants to be
    refused has to say *why* it is refused.

    This helper used to write `"classification": "desktop_context"` into the dictionary it returned.
    That is the reason the suite went on passing for a round after the classification rules changed:
    those fixtures were checking a string this file typed, not the instrument's verdict (§8).
    """
    row = {"pid": int(pid), "process_name": name, "used_memory": memory,
           "process_create_time": created, "parent_pid": parent,
           "command_line": facts.pop("command_line", None),
           "executable_path": facts.pop("executable", None)}
    entry = {be.PMON_UTILISATION_COLUMNS[column]: value
             for column, value in sorted(facts.items())
             if column in be.PMON_UTILISATION_COLUMNS}
    engine_type = facts.pop("engine_type", None)
    if engine_type is not None:
        entry["pmon_engine_type_reported"] = engine_type
    if facts:
        raise TypeError("gpu_row was handed a field no criterion reads: %s" % sorted(facts))
    row["pmon"] = entry or None
    return row


def tree_row(pid, name="python.exe", created=None, parent=None):
    """One row of a §2 process tree snapshot: who, since when, whose child."""
    return {"pid": pid, "name": name, "process_create_time": created, "parent_pid": parent,
            "is_current_benchmark_descendant": True}


FIXTURE_ROOT_PID = 5001
"""The pid these fixtures call the benchmark's own root process.

`be.nvidia_compute_processes` decides descendant membership against `root_pid`, and left to itself it
uses `os.getpid()` - the pytest process running this file. §22 says the machine the tests run on is
not consulted, so the root is named here instead, and no fixture row below uses this pid.
"""


def repo_row(pid=9999, name="python.exe", created=None, parent=None):
    return {"pid": pid, "name": name, "process_create_time": created, "parent_pid": parent,
            "is_current_benchmark_descendant": False}


def sample(cpu=6.0, gpu=3.0, ram=24 * GIB, unix=1.0, repo=(), compute=(), unavailable=(),
           tree=(), baseline=(), root_pid=FIXTURE_ROOT_PID, **over):
    """One §11 row, built from the fields §11 names rather than from a real machine's answers.

    The client rows go through `be.nvidia_compute_processes` and `be.classify_gpu_processes`, so the
    classification, the two outsider lists and the ambient counts are the instrument's answer for these
    inputs. That is what makes the difference this round turned on expressible here: a windowing client
    with nothing to say about it is in `nvidia_compute_processes` and *not* in
    `external_compute_processes`, which is the pair of facts that stopped the last two rounds and could
    not be caught while this helper typed classifications by hand.

    An idle machine's default is one unproven windowing client rather than an empty table, because that
    is what this driver actually answers here - thirty-odd clients, all of them `[N/A]` - and the tests
    below that expect a quiet window to pass therefore pass *with* a client present, not by pretending
    the table was empty.
    """
    compute = list(compute) or [gpu_row()]
    processes = be.nvidia_compute_processes(
        rows=[[str(row["pid"]), row["process_name"], row["used_memory"]] for row in compute],
        tree_snapshot=list(tree),
        process_rows_by_pid={row["pid"]: {"pid": row["pid"], "name": row["process_name"],
                                         "process_create_time": row.get("process_create_time"),
                                         "parent_pid": row.get("parent_pid"),
                                         "command_line": row.get("command_line"),
                                         "executable_path": row.get("executable_path")}
                             for row in compute},
        utilisation_by_pid={row["pid"]: row["pmon"] for row in compute if row.get("pmon")},
        root_pid=root_pid)
    classified = be.classify_gpu_processes(processes, workload_rows=list(repo),
                                           baseline_identities=list(baseline))
    # §11's published shape comes from the instrument's own projection, not from a copy of it here:
    # a gate that checks what a sample carries is then checking the row a round actually writes.
    stamp = (datetime.datetime(2026, 9, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)
             + datetime.timedelta(seconds=unix)).strftime("%Y-%m-%dT%H:%M:%SZ")
    published = [be.published_gpu_row(row, stamp) for row in classified]
    conflicts = [row for row in published
                 if row["classification"] in be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS]
    counts = {kind: sum(1 for row in published if row["classification"] == kind)
              for kind in be.GPU_CLIENT_CLASSIFICATIONS}
    reading = {"utc": stamp,
               "unix_seconds": unix,
               "system_cpu_percent": cpu, "available_ram_bytes": ram,
               "gpu_utilization_percent": gpu, "gpu_memory_utilization_percent": 0.0,
               "gpu_temperature_celsius": 40.0, "gpu_graphics_clock_mhz": 600.0,
               "gpu_sm_clock_mhz": 600.0, "gpu_memory_clock_mhz": 405.0,
               "gpu_power_draw_watts": 20.0, "gpu_pstate": "P8",
               "gpu_encoder_utilization_percent": 0.0, "gpu_decoder_utilization_percent": 0.0,
               "gpu_available": True, "nvidia_smi_answered": True,
               "unavailable_fields": list(unavailable),
               "tree_of_this_sample": list(tree),
               "nvidia_compute_processes": published,
               "repo_processes": list(repo), "conflicting_repo_processes": list(repo),
               "external_compute_processes": conflicts,
               "benchmark_descendant_pids": len(tree), "gpu_process_count": len(published),
               "classification_counts": counts,
               "ambient_os_graphics_client_count": counts["ambient_os_graphics_client"],
               "ambient_or_unproven_gpu_client_count": counts["ambient_or_unproven_gpu_client"],
               # §5's criterion four is a measurement, and a fixture that handed in no `pmon` answer
               # says so instead of implying the card was asked and found nobody.
               "per_process_gpu_utilisation_answered": any(row.get("pmon") for row in compute),
               "per_process_gpu_utilisation_row_count": sum(
                   1 for row in compute if row.get("pmon"))}
    reading["per_process_gpu_utilisation_unavailable_reason"] = (
        None if reading["per_process_gpu_utilisation_answered"] else "test fixture gave no pmon table")
    reading.update(over)
    return reading


@contextlib.contextmanager
def synthetic_machine(compute_rows=(), descendants=()):
    """Cut the module off from this host's process tables for the duration of one test.

    `run_window` reads the GPU client list once per window to establish who was already there, and
    the process tree to know which processes belong to the run. Both are the real host's business,
    and §22 says a test may not depend on it, so both are handed in here.

    The GPU patch answers a *host* read - `nvidia_compute_processes()` with no table handed in - and
    leaves alone a read that was given its rows, because the fixtures above reach the real classifier
    through that same function and pass it a table. Patching both was harmless while the fixtures typed
    their own classifications; now that they do not, a patch that swallowed the classifier would
    swallow the thing under test with it.
    """
    snapshot = [tree_row(pid) for pid in descendants]
    real_gpu_table = be.nvidia_compute_processes

    def gpu_table(tree_snapshot=None, rows=None, process_rows_by_pid=None,
                  utilisation_by_pid=None, root_pid=None):
        if rows is None:
            return [dict(row) for row in compute_rows]
        return real_gpu_table(rows=rows, tree_snapshot=tree_snapshot,
                              process_rows_by_pid=process_rows_by_pid,
                              utilisation_by_pid=utilisation_by_pid, root_pid=root_pid)

    def pmon(rows=None, args=None):
        assert rows is not None, "a synthetic machine may not ask the host for pmon"
        return be.per_process_gpu_utilisation(rows=rows)

    with mock.patch.object(be, "nvidia_compute_processes", gpu_table), \
            mock.patch.object(be, "per_process_gpu_utilisation", pmon), \
            mock.patch.object(be, "process_tree_snapshot", lambda **kwargs: list(snapshot)), \
            mock.patch.object(be, "process_rows", lambda *args, **kwargs: []), \
            mock.patch.object(be.psutil, "cpu_percent", lambda interval=None: 0.0):
        yield


def preflight(rows, **over):
    """`quiet_preflight` over a list of synthetic samples, on a clock that costs nothing."""
    queue = list(rows)
    index = {"n": 0}

    def sample_fn(**kwargs):
        row = queue[index["n"] % len(queue)] if queue else sample()
        index["n"] += 1
        return dict(row)

    clock = FakeTime()
    kwargs = {"sample_fn": sample_fn, "sleep": clock.sleep, "monotonic": clock.monotonic,
              "max_wait_seconds": 90.0, "code_under_test": CODE}
    kwargs.update(over)
    with synthetic_machine():
        return be.quiet_preflight(**kwargs)


def monitor_artifact(readings, batch=None, contaminated=None, children=(), ffmpeg=(),
                     created=None):
    """Drive `RuntimeMonitor` by hand and take §21's artifact out of it."""
    monitor = be.RuntimeMonitor(sample_fn=lambda **kwargs: sample())
    snapshot = [tree_row(pid, created=created) for pid in children]
    for offset, reading in enumerate(readings, start=1):
        monitor.observe(reading=dict(reading), tree_snapshot=snapshot)
    monitor.started_utc = "2026-09-20T00:00:00Z"
    monitor.ended_utc = (datetime.datetime(2026, 9, 20, 0, 0, 0, tzinfo=datetime.timezone.utc)
                         + datetime.timedelta(seconds=len(readings))).strftime("%Y-%m-%dT%H:%M:%SZ")
    monitor.benchmark_child_pids = set(int(pid) for pid in children)
    monitor.ffmpeg_child_pids = set(int(pid) for pid in ffmpeg)
    artifact = monitor.artifact(code_under_test=CODE, batch=batch)
    artifact["measurement_start"] = monitor.started_utc
    artifact["measurement_end"] = monitor.ended_utc
    if contaminated is not None:
        artifact["runtime_contaminated"] = contaminated
    return artifact


OUTSIDER_PID, OUTSIDER_CREATED = 7000, 1000.0
"""One outsider, with the same identity in every sample of a run.

§18's rule is about *one process* persisting, so a fixture that spans samples has to hand the same
(pid, create_time) pair over and over. The create time is also what makes the parent-chain attribution
below testable: an outsider whose parent is not in this sample's tree is not forgiven for having one.
"""


def runtime_rows(seconds, outsider=None, name="ffmpeg.exe", tree=(), memory="1024"):
    """`seconds` samples at 1 Hz, with one outsider present in whichever ones are asked for.

    `tree` is the process tree of every one of those samples and `memory` is what the driver answers
    for the outsider: both are handed to §5's classifier rather than described to it, so who the
    outsider turns out to be is the instrument's decision, not this file's.
    """
    rows = []
    for offset in range(1, seconds + 1):
        compute = ([gpu_row(pid=OUTSIDER_PID, name=name, memory=memory, created=OUTSIDER_CREATED,
                            parent=FIXTURE_ROOT_PID)] if outsider == "gpu" else [])
        repo = ([repo_row(pid=OUTSIDER_PID, name=name)] if outsider == "repo" else [])
        rows.append(sample(unix=offset, repo=repo, compute=compute, tree=list(tree)))
    return rows


# --------------------------------------------------------------------------------------------
# §22's nine
# --------------------------------------------------------------------------------------------

class QuietPreflightTests(unittest.TestCase):
    """§11-§16, on machines this file invents."""

    def test_quiet_preflight_passes(self):
        """§22.1: an idle thirty-second window is accepted, once, and recorded as §20 asks."""
        doc = preflight([sample(unix=offset) for offset in range(1, 31)])
        self.assertEqual(doc["preflight_status"], "PASS")
        self.assertEqual(doc["preflight_window_that_passed"], 1)
        self.assertEqual(doc["preflight_window_count_attempted"], 1)
        self.assertEqual(doc["not_ready_reason"], None)
        # §20's field names, because the artifact is what the brief is checked against.
        for field in BRIEF_20_FIELDS:
            self.assertIn(field, doc)
        self.assertEqual(len(doc["all_preflight_samples"]), be.window_sample_count())
        self.assertEqual(doc["thresholds"], be.THRESHOLDS)
        # The waiting stops at the first window that passes: a second one would be shopping.
        self.assertTrue(all(row["preflight_window_attempt"] == 1
                            for row in doc["all_preflight_samples"]))

    def test_high_cpu_preflight_fails(self):
        """§22.2: a machine at 55% is not quiet, and the round says so instead of measuring anyway."""
        doc = preflight([sample(cpu=55.0, unix=offset) for offset in range(1, 31)])
        self.assertEqual(doc["preflight_status"], "FAIL")
        self.assertEqual(doc["not_ready_reason"], "no_quiet_preflight_window")
        reasons = {row["reason"] for row in doc["windows"][-1]["failures"]}
        self.assertIn("cpu_median_percent_violated", reasons)
        self.assertIn("cpu_p95_percent_violated", reasons)
        self.assertGreater(doc["cpu_median"], be.THRESHOLDS["cpu_median_percent_at_most"])

    def test_high_gpu_preflight_fails(self):
        """§22.3: the same for the card - §12's GPU thresholds are the real load detector here."""
        doc = preflight([sample(gpu=60.0, unix=offset) for offset in range(1, 31)])
        self.assertEqual(doc["preflight_status"], "FAIL")
        reasons = {row["reason"] for row in doc["windows"][-1]["failures"]}
        self.assertIn("gpu_utilization_median_percent_violated", reasons)
        self.assertNotIn("cpu_median_percent_violated", reasons)

    def test_external_repo_process_fails(self):
        """§22.4: another agent's pytest in this repository stops the window, and is not killed."""
        doc = preflight([sample(unix=offset, repo=[repo_row(pid=4321, name="python.exe")])
                        for offset in range(1, 31)])
        self.assertEqual(doc["preflight_status"], "FAIL")
        failures = doc["windows"][-1]["failures"]
        self.assertIn("conflicting_repo_process", {row["reason"] for row in failures})
        self.assertTrue(doc["conflicting_repo_processes"])
        # §13: the process is recorded with its pid, so the reader can see who was here.
        self.assertIn("4321 python.exe", doc["conflicting_repo_processes"][0])

    def test_external_gpu_compute_process_fails(self):
        """§22.5: a compute process the §14 policy calls a workload stops the window too."""
        doc = preflight([sample(unix=offset, compute=[gpu_row(pid=8123, name="blender.exe",
                                                              memory="3042")])
                        for offset in range(1, 31)])
        self.assertEqual(doc["preflight_status"], "FAIL")
        self.assertIn("external_gpu_compute_workload",
                      {row["reason"] for row in doc["windows"][-1]["failures"]})
        self.assertTrue(any("blender.exe" in row for row in doc["external_compute_processes"]))

    def test_missing_gpu_metric_is_unavailable_not_zero(self):
        """§22.6: a field the driver will not answer is `None`, named as unavailable, and its own
        failure - because 0 reads as "idle" and nobody knows that."""
        busy = sample(gpu=None, unavailable=["gpu_utilization_percent"], unix=2)
        self.assertIsNone(busy["gpu_utilization_percent"])
        self.assertNotEqual(busy["gpu_utilization_percent"], 0)
        summary = be.summarise([busy])
        self.assertIsNone(summary["gpu_utilization_median_percent"])
        judged = be.judge_window([busy] * be.window_sample_count())
        reasons = {row["reason"] for row in judged["failures"]}
        self.assertIn("gpu_utilization_median_percent_could_not_be_read", reasons)
        self.assertIn("gpu_utilization_p95_percent_could_not_be_read", reasons)
        # The same field §14 says to keep exactly as reported: `[N/A]` in, `None` beside it, and the
        # row classified from that answer rather than around it - no memory accounted for it and no
        # other criterion either, so it is recorded and left alone. `desktop_context`, the name this
        # assertion used to check, is gone: it said "we found no reason" without saying for which row,
        # and `ambient_or_unproven_gpu_client` says the same thing from the direction that can be
        # audited.
        raw = be.nvidia_compute_processes(tree_snapshot=[], rows=[["5", "chrome.exe", "[N/A]"]])
        rows = be.classify_gpu_processes(raw)
        self.assertEqual(rows[0]["used_memory"], "[N/A]")
        self.assertFalse(rows[0]["memory_reported_by_driver"])
        self.assertIsNone(rows[0]["used_memory_bytes"])
        self.assertEqual(rows[0]["classification"], "ambient_or_unproven_gpu_client")
        self.assertEqual(rows[0]["compute_criteria_met"], [])
        self.assertNotIn(rows[0]["classification"], be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS)

    def test_runtime_three_consecutive_conflicts_marks_contaminated(self):
        """§22.7: an outsider that stays three 1 Hz samples ends the round's claim to be clean."""
        ledger = be.ContaminationLedger()
        rows = runtime_rows(5, outsider="gpu", name="blender.exe")
        outcomes = [ledger.add(row) for row in rows]
        self.assertFalse(outcomes[0] or outcomes[1])
        self.assertTrue(outcomes[2], "the third consecutive sample is the one that crosses")
        self.assertTrue(ledger.contaminated)
        self.assertIn("blender.exe", ledger.reason)
        self.assertIn("3 consecutive", ledger.reason)
        events = ledger.event_rows()
        self.assertEqual(len(events), 1)
        # The outsider stayed the whole five seconds: the event row says how long it really was
        # here, and crossing the line once does not make the monitor forget the samples after it.
        self.assertEqual(events[0]["consecutive_samples"], 5)
        self.assertEqual(events[0]["sightings"], 5)
        self.assertTrue(events[0]["reached_contamination_threshold"])
        self.assertEqual(outcomes[3], [], "the fourth sample does not re-cross a crossed line")
        self.assertEqual(outcomes[4], [])

    def test_two_transient_conflicts_do_not_mark_contaminated(self):
        """§22.8: two sightings, a gap, two more - a `git` wrapper, not a competitor."""
        rows = (runtime_rows(2, outsider="repo") + runtime_rows(1)
                + runtime_rows(2, outsider="repo") + runtime_rows(1))
        ledger = be.judge_contamination(rows)
        self.assertFalse(ledger.contaminated, "the streak is what counts, not the total")
        self.assertEqual(ledger.event_rows()[0]["sightings"], 4)
        self.assertEqual(ledger.event_rows()[0]["consecutive_samples"], 2)
        # And the same ledger, made to streak again from the gap, does flip: the count restarts at a
        # clean sample, and three samples after that is three samples of someone else's work.
        for n in ("first", "second"):
            self.assertFalse(ledger.add(runtime_rows(1, outsider="repo")[0]),
                             f"the {n} sighting after a gap is not a streak")
        self.assertTrue(ledger.add(runtime_rows(1, outsider="repo")[0]))
        self.assertTrue(ledger.contaminated)

    def test_benchmark_lock_refuses_second_owner(self):
        """§22.9: one holder. The second is refused inside five seconds and told who holds it."""
        first = be.BenchmarkLock(root=_ROOT, code_under_test=CODE, protocol="test",
                                 command="pytest")
        self.assertTrue(first.acquire())
        try:
            clock = FakeTime()
            second = be.BenchmarkLock(root=_ROOT, timeout_seconds=5.0, code_under_test=CODE,
                                      protocol="test", command="pytest-contender",
                                      monotonic=clock.monotonic, sleep=clock.sleep)
            self.assertFalse(second.acquire())
            self.assertLessEqual(second.waited_seconds, 5.0)
            self.assertGreater(second.waited_seconds, 0.0)
            self.assertEqual(second.owner["pid"], os.getpid())
            with self.assertRaises(be.BenchmarkLockBusy) as refused:
                second.acquire_or_raise()
            self.assertIn(str(os.getpid()), str(refused.exception))
            # Releasing is not deleting: the files stay, because a waiter may be blocked on them.
            self.assertTrue(os.path.exists(second.path))
        finally:
            self.assertTrue(first.release())
        released = be.BenchmarkLock(root=_ROOT, code_under_test=CODE, command="pytest")
        try:
            self.assertTrue(released.acquire(), "the lock is takeable again once its holder exits")
        finally:
            released.release()


# --------------------------------------------------------------------------------------------
# §17-§19: who the monitor is allowed to blame
# --------------------------------------------------------------------------------------------

class MonitorAttributionTests(unittest.TestCase):
    def test_the_benchmarks_own_ffmpeg_is_not_contamination(self):
        """§19: this batch spawns transcoders on purpose; seeing one is not a competitor.

        The row is one §5 refuses an outsider for - `ffmpeg.exe`, 1024 MiB the driver accounts for -
        and the only thing that separates it from a neighbour's transcoder is the tree of the sample it
        was read in. So this drives the monitor over classified readings the real classifier produced.
        The version this replaces wrote `"classification": "benchmark_descendant"` into the dictionary
        and then checked that the monitor agreed with it, which is §8's source-string test in dict form:
        it passed at `eb61bbc`, when the instrument was doing the opposite.
        """
        root, child = 7000, 7001
        live = [live_row(child, "ffmpeg.exe", created=1000.0, parent=root),
                live_row(root, "python.exe", created=900.0)]
        ours = [tree_row(root, name="python.exe", created=900.0, parent=None)]
        readings = [classified_reading(unix, table=[[child, "ffmpeg.exe", "1024"]], tree=ours,
                                       live=live, root_pid=root) for unix in range(1, 11)]
        first = readings[0]["nvidia_compute_processes"][0]
        self.assertEqual(first["classification"], "benchmark_descendant")
        self.assertTrue(first["is_current_benchmark_descendant"])
        # The child is not in this sample's snapshot - it started after the tree walk - and is
        # attributed by the two facts the OS does give for it: this tree holds its parent, and it came
        # into existence at or after the root. §2's second method, from the row's own numbers.
        self.assertEqual(first["descendant_attribution"],
                         "parent_in_this_sample_tree_and_created_after_the_root")
        monitor = be.RuntimeMonitor(sample_fn=lambda **kwargs: sample())
        for reading in readings:
            self.assertFalse(monitor.observe(reading=reading,
                                             tree_snapshot=reading["tree_of_this_sample"])
                             ["newly_contaminated"])
        self.assertFalse(monitor.contaminated, monitor.ledger.reason)
        self.assertEqual(monitor.ledger.event_rows(), [])
        self.assertEqual(monitor.samples[-1]["external_compute_processes"], [])
        # The same table in a sample whose tree no longer holds a parent is the competitor §5 names,
        # and it is refused on two criteria rather than on the name in its image.
        gone = [classified_reading(unix, table=[[child, "ffmpeg.exe", "1024"]], tree=[], live=live,
                                   root_pid=root) for unix in range(11, 14)]
        outsider = gone[0]["external_compute_processes"][0]
        self.assertEqual(outsider["classification"], "known_compute_workload")
        self.assertEqual(outsider["compute_criteria_met"],
                         ["driver_accounts_gpu_memory_in_use",
                          "image_path_or_command_names_a_compute_workload"])
        for reading in gone:
            monitor.observe(reading=reading, tree_snapshot=reading["tree_of_this_sample"])
        self.assertTrue(monitor.contaminated)
        self.assertEqual([row["pid"] for row in monitor.ledger.event_rows()], [child])

    def test_the_measurement_query_is_not_a_workload(self):
        """The nvidia-smi that reads the table must not be the thing that stops the batch.

        A driver that listed its own query would put a fresh pid in the client table every second,
        which is precisely what §14's mid-window-arrival rule looks for.
        """
        rows = [gpu_row(pid=9000 + offset, name="nvidia-smi.exe", memory="[N/A]")
                for offset in range(3)]
        classified = be.classify_gpu_processes(rows, baseline_identities={9000})
        self.assertEqual({row["classification"] for row in classified}, {"benchmark_descendant"})
        self.assertFalse(be.looks_like_a_gpu_workload("nvidia-smi.exe"))
        self.assertTrue(be.looks_like_the_query_tool("C:\\Windows\\nvidia-smi.exe"))
        hard = [row for row in be.classify_gpu_processes(
            [gpu_row(pid=9001, name="nvidia-smi.exe", memory="[N/A]"),
             gpu_row(pid=9002, name="blender.exe", memory="120")],
            baseline_identities={9001})
            if row["classification"] in be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS]
        self.assertEqual([row["process_name"] for row in hard], ["blender.exe"])

    def test_utilisation_never_contaminates_a_batch(self):
        """§17: the benchmark loading the machine is the measurement, not a failure."""
        busy = [sample(cpu=100.0, gpu=100.0, unix=offset) for offset in range(1, 31)]
        monitor = be.RuntimeMonitor(sample_fn=lambda **kwargs: sample())
        for row in busy:
            monitor.observe(reading=row, tree_snapshot=[])
        self.assertFalse(monitor.contaminated)
        self.assertIsNone(monitor.ledger.reason)


class ContaminationGateTests(unittest.TestCase):
    def test_the_monitor_artifact_carries_every_field_the_brief_names(self):
        """§21's list, checked on the record the monitor writes - both when it saw something and
        when it did not, because the clean case is the one a reader has to trust."""
        for rows, dirty in ((runtime_rows(30), False),
                            (runtime_rows(6, outsider="gpu", name="blender.exe"), True)):
            artifact = monitor_artifact(rows, children=[7001], ffmpeg=[7001])
            for field in BRIEF_21_FIELDS:
                self.assertIn(field, artifact)
            self.assertEqual(len(artifact["samples"]), len(rows))
            # §19: this batch's own FFmpeg is named, so a transcoder in the GPU table can be told
            # apart from a neighbour's without opening this file.
            self.assertEqual(artifact["benchmark_child_pids"], [7001])
            self.assertEqual(artifact["ffmpeg_child_pids"], [7001])
            self.assertEqual(artifact["runtime_contaminated"], dirty)
            if dirty:
                self.assertIn("blender.exe", artifact["contamination_reason"])
                self.assertIsNotNone(artifact["first_contamination_timestamp"])
                self.assertEqual([row["name"] for row in artifact["external_workload_events"]],
                                 ["blender.exe"])
            else:
                self.assertIsNone(artifact["contamination_reason"])
                self.assertIsNone(artifact["first_contamination_timestamp"])
                self.assertEqual(artifact["external_workload_events"], [])

    def test_a_clean_monitor_record_passes_its_gate(self):
        gate = status.runtime_contamination_gate(
            monitor_artifact(runtime_rows(60), children=[7001], ffmpeg=[7001]))
        self.assertEqual(gate["status"], "DONE_PROVEN", gate["failed_checks"])

    def test_a_contaminated_batch_cannot_be_read_as_clean(self):
        rows = runtime_rows(4, outsider="gpu")
        artifact = monitor_artifact(rows, batch={"stopped_because_of_contamination": True})
        self.assertTrue(artifact["runtime_contaminated"])
        gate = status.runtime_contamination_gate(artifact)
        self.assertEqual(gate["status"], "NOT_READY")
        self.assertIn("the_round_is_not_reading_a_contaminated_batch_as_clean",
                      gate["failed_checks"])
        # ... and every outsider is still in the file: §18 keeps what it saw.
        self.assertEqual(len(artifact["external_workload_events"]), 1)

    def test_a_flag_that_disagrees_with_its_own_samples_is_the_finding(self):
        """The artifact says clean; the same rule applied to its samples says otherwise."""
        artifact = monitor_artifact(runtime_rows(4, outsider="gpu"), contaminated=False)
        gate = status.runtime_contamination_gate(artifact)
        self.assertEqual(gate["status"], "NOT_READY")
        self.assertIn("the_record_says_the_same_when_its_own_samples_are_rejudged",
                      gate["failed_checks"])

    def test_a_sample_count_that_disagrees_with_its_list_fails(self):
        """§1's key-collision bug, on the runtime side: a scalar written over a list."""
        artifact = monitor_artifact(runtime_rows(30))
        artifact["sample_count"] = 1
        gate = status.runtime_contamination_gate(artifact)
        self.assertIn("the_sample_count_and_the_sample_list_agree", gate["failed_checks"])

    def test_a_monitor_that_stopped_before_the_batch_did(self):
        """Three samples cannot say anything about an hour. The span is checked against the count."""
        artifact = monitor_artifact(runtime_rows(3))
        artifact["measurement_end"] = "2026-09-20T01:00:00Z"
        gate = status.runtime_contamination_gate(artifact)
        self.assertIn("the_monitor_recorded_the_length_of_the_batch", gate["failed_checks"])

    def test_a_stub_monitor_is_not_read_as_a_pass(self):
        gate = status.runtime_contamination_gate({"code_under_test": CODE})
        self.assertEqual(gate["status"], "NOT_READY")
        self.assertTrue(gate["failed_checks"])
        self.assertEqual(status.runtime_contamination_gate(None)["status"], "NOT_READY")


class EnvironmentGateTests(unittest.TestCase):
    """The two gates that read `benchmark_environment.json`."""

    def setUp(self):
        lock = be.BenchmarkLock(root=_ROOT, code_under_test=CODE, protocol="unit-test",
                                command="pytest tests/unit")
        self.assertTrue(lock.acquire(), "another benchmark is holding the lock during the suite")
        self.addCleanup(lock.release)
        # two busy windows then a quiet one: §15's waiting, and the window of record is the last.
        self.doc = preflight([sample(cpu=55.0, unix=offset) for offset in range(1, 31)]
                             * 2 + [sample(unix=offset) for offset in range(1, 31)],
                             lock=lock, max_wait_seconds=600.0)

    def test_a_genuine_artifact_passes_both_environment_gates(self):
        self.assertEqual(self.doc["preflight_status"], "PASS")
        self.assertEqual(self.doc["preflight_window_that_passed"], 3)
        self.assertEqual(len(self.doc["all_preflight_samples"]), 90)
        preflight_gate = status.quiet_preflight_gate(self.doc)
        self.assertEqual(preflight_gate["status"], "DONE_PROVEN", preflight_gate["failed_checks"])
        lock_gate = status.bench_lock_gate(self.doc)
        self.assertEqual(lock_gate["status"], "DONE_PROVEN", lock_gate["failed_checks"])

    def test_a_stub_artifact_is_not_read_as_a_pass(self):
        for gate in (status.quiet_preflight_gate, status.bench_lock_gate):
            out = gate({"code_under_test": CODE})
            self.assertEqual(out["status"], "NOT_READY")
            self.assertTrue(out["failed_checks"])

    def test_a_short_window_fails_the_gate(self):
        """§11: thirty seconds, not eleven, however quiet those eleven were."""
        doc = json.loads(json.dumps(self.doc))
        last = doc["preflight_window_that_passed"]
        doc["all_preflight_samples"] = [row for row in doc["all_preflight_samples"]
                                        if row["preflight_window_attempt"] != last][:5] \
            + [row for row in doc["all_preflight_samples"]
               if row["preflight_window_attempt"] == last][:5]
        gate = status.quiet_preflight_gate(doc)
        self.assertIn("a_full_thirty_second_window_was_measured", gate["failed_checks"])

    def test_a_tuned_threshold_fails_the_gate(self):
        """§12: a window judged against relaxed numbers is not the window this round promised."""
        doc = json.loads(json.dumps(self.doc))
        doc["thresholds"] = dict(be.THRESHOLDS, cpu_median_percent_at_most=95.0)
        gate = status.quiet_preflight_gate(doc)
        self.assertIn("the_thresholds_are_the_frozen_ones", gate["failed_checks"])

    def test_a_busy_window_flagged_passing_fails_the_gate(self):
        """The flag is a claim; the samples are the evidence, and the gate reads the evidence."""
        doc = json.loads(json.dumps(self.doc))
        last = doc["preflight_window_that_passed"]
        doc["all_preflight_samples"] = [{**row, "system_cpu_percent": 91.0}
                                        if row["preflight_window_attempt"] == last else row
                                        for row in doc["all_preflight_samples"]]
        gate = status.quiet_preflight_gate(doc)
        self.assertEqual(gate["status"], "NOT_READY")
        failed = set(gate["failed_checks"])
        self.assertIn("the_window_of_record_passes_when_its_samples_are_rejudged", failed)
        self.assertIn("the_published_figures_are_the_ones_their_own_samples_give", failed)

    def test_an_unmeasurable_field_is_its_own_failure(self):
        doc = json.loads(json.dumps(self.doc))
        last = doc["preflight_window_that_passed"]
        doc["all_preflight_samples"] = [{**row, "gpu_utilization_percent": None}
                                        if row["preflight_window_attempt"] == last else row
                                        for row in doc["all_preflight_samples"]]
        gate = status.quiet_preflight_gate(doc)
        self.assertIn("nothing_the_gate_needs_was_unmeasurable", gate["failed_checks"])


class VarianceGateShapeTests(unittest.TestCase):
    """§33-§34: the stability judgement, on a pool this file builds."""

    @staticmethod
    def execution(index, legacy_fps, reduced_fps):
        return {"execution": f"execution-{index}", "unpoolable_reasons": [],
                "orders": {mode: {"runs_used_for_medians": [2, 3],
                                  "fps_of_every_run": {"1": 1.0, "2": fps[0], "3": fps[1]},
                                  "inter_run_spread_percent_of_kept_runs": round(
                                      100.0 * abs(fps[0] - fps[1]) / min(fps), 2)}
                             for mode, fps in (("legacy", legacy_fps), ("reduced", reduced_fps))}}

    def pool(self, spreads):
        executions = [self.execution(index, (10.0, 10.0), fps)
                      for index, fps in enumerate(spreads, start=1)]
        kept = 2 * len(executions)
        return {"executions": executions,
                "verdict": {"run_census": {mode: {"kept_run_count": kept}
                                           for mode in ("legacy", "reduced")}}}

    def test_a_stable_pool_passes(self):
        gate = status.bench_variance_gate(self.pool([(10.0, 10.2)] * 12))
        self.assertEqual(gate["status"], "DONE_PROVEN", gate["failed_checks"])

    def test_a_third_of_a_noisy_pool_stops_the_round(self):
        """§34: 3 of 12 executions over 20% is exactly at the limit; 4 is not a stable machine."""
        noisy = [(10.0, 6.0)] * 3 + [(10.0, 10.1)] * 9
        self.assertEqual(status.bench_variance_gate(self.pool(noisy))["status"], "DONE_PROVEN")
        gate = status.bench_variance_gate(self.pool([(10.0, 6.0)] * 4 + [(10.0, 10.1)] * 8))
        self.assertEqual(gate["status"], "NOT_READY")
        self.assertIn("benchmark_variance_stable", gate["failed_checks"])
        self.assertEqual(len(gate["detail"]["executions_over_threshold"]), 4)

    def test_dropping_a_run_to_flatten_the_spread_is_itself_the_failure(self):
        pool = self.pool([(10.0, 10.1)] * 12)
        pool["verdict"]["run_census"]["reduced"]["kept_run_count"] = 23
        gate = status.bench_variance_gate(pool)
        self.assertIn("no_run_was_dropped_to_make_the_variance_look_stable", gate["failed_checks"])

    def test_the_gate_uses_the_sections_own_denominator_and_says_so(self):
        pool = self.pool([(10.0, 10.01)] * 12)
        row = status.kept_run_spreads(pool["executions"])[2]
        self.assertLessEqual(row["spread_percent_vs_median"], row["spread_percent_vs_lowest_run"])
        gate = status.bench_variance_gate(pool)
        check = next(item for item in gate["checks"]
                     if item["what"] == "the_spread_is_measured_by_the_definition_this_round_froze")
        self.assertEqual(check["detail"]["gate_formula"], status.VARIANCE_SPREAD_FORMULA)


class DriverBatchRecordTests(unittest.TestCase):
    """§21's `batch` section: written by the driver, judged by the gate in the paragraph above.

    The driver is the thing that runs for forty minutes and cannot be reroll-backed (§16), and this
    record is the only part of it another file reads by field name. Testing a hand-typed copy of it
    would prove the gate works and nothing about the round, so these call the driver's own function:
    a field renamed on either side fails here, in a second, rather than at the end of a batch.
    """

    def test_the_clean_record_passes_the_gate_that_reads_it(self):
        record = driver.monitor_batch_record(
            [f"execution_{i:02d}.json" for i in range(1, driver.PRIMARY_EXECUTIONS + 1)],
            [f"execution_{i:02d}.json" for i in range(1, driver.CORROBORATION_EXECUTIONS + 1)],
            None, None, [], False)
        self.assertEqual(record["primary_executions_launched"], 12)
        self.assertEqual(record["primary_executions_defined"], driver.PRIMARY_EXECUTIONS)
        self.assertEqual(record["primary_frames_per_run"], driver.PRIMARY_FRAMES)
        self.assertEqual(record["corroborating_frames_per_run"], driver.CORROBORATION_FRAMES)
        self.assertIsNone(record["primary_stopped_early_because"])
        self.assertIsNone(record["corroborating_batch_not_run_because"])
        gate = status.runtime_contamination_gate(
            monitor_artifact(runtime_rows(60), batch=record, children=[7001], ffmpeg=[7001]))
        self.assertEqual(gate["status"], "DONE_PROVEN", gate["failed_checks"])

    def test_the_contaminated_record_promises_the_stop_it_made(self):
        """§18: the finished executions stay, nothing further launches, and the record says both."""
        record = driver.monitor_batch_record(
            ["execution_01.json"], [], "primary-60f stopped before execution 2",
            "§30's precondition failed: the monitor recorded an external workload", [], True)
        self.assertTrue(record["stopped_because_of_contamination"])
        self.assertEqual(record["primary_executions_launched"], 1)
        self.assertEqual(record["corroborating_batch_not_run_because"],
                         "§30's precondition failed: the monitor recorded an external workload")
        artifact = monitor_artifact(runtime_rows(4, outsider="gpu", name="blender.exe"),
                                    batch=record)
        gate = status.runtime_contamination_gate(artifact)
        # NOT_READY, and for exactly one reason: the window was contaminated. That the driver
        # stopped launching is a check this record passes - the round is not being failed twice,
        # once for the outsider and once for a promise nobody made.
        self.assertEqual(gate["failed_checks"],
                         ["the_round_is_not_reading_a_contaminated_batch_as_clean"])
        self.assertEqual(gate["detail"]["contamination"]["runtime_contaminated"], True)

    def test_a_partial_batch_shows_in_counts_not_only_in_prose(self):
        """A batch that got partway is not a batch that didn't run: the two fields say different
        things, and a reader should not have to reconstruct which happened from a sentence."""
        halfway = driver.monitor_batch_record(["e01.json"], ["e01.json"], None,
                                              "stopped before execution 3", [], True)
        self.assertEqual(halfway["corroborating_executions_launched"], 1)
        self.assertEqual(halfway["corroborating_executions_defined"], 4)
        self.assertIsNone(halfway["corroborating_batch_not_run_because"])
        never = driver.monitor_batch_record(["e01.json"], [], None, "stopped before execution 1",
                                            [], True)
        self.assertEqual(never["corroborating_executions_launched"], 0)
        self.assertIsNotNone(never["corroborating_batch_not_run_because"])

    def test_the_record_reports_the_failures_it_was_handed(self):
        """§29: a child that died is recorded where the verdict is read, and the list stays the
        driver's - the record takes a copy rather than editing the batch it describes."""
        failures = ["primary-60f execution 7 produced no artifact"]
        record = driver.monitor_batch_record([], [], None, None, failures, False)
        self.assertEqual(record["execution_failures"], ["primary-60f execution 7 produced no "
                                                        "artifact"])
        failures.append("§18 stopped the batch")
        self.assertEqual(len(record["execution_failures"]), 1)


# --------------------------------------------------------------------------------------------
# §8: the regressions the brief names, one per named behaviour
# --------------------------------------------------------------------------------------------
#
# These are the checks §8 lists by name. Each one calls the classifier, the ledger, the preflight
# loop or the gate that the round actually runs, on a machine made of dictionaries - §8's own
# condition is "不要使用 source-string tests。必须调用真实 classifier / verdict / preflight logic",
# and reading this file back as a string would prove only that somebody typed words.

def live_row(pid, name, created=1000.0, parent=None, command_line="", executable=None):
    """One entry of `process_rows()` for a pid the GPU client table also lists."""
    return {"pid": int(pid), "name": name, "command_line": command_line,
            "process_create_time": created, "parent_pid": parent,
            "executable_path": executable or f"C:\\Program Files\\{name}"}


def classified_reading(unix, table=(), tree=(), live=(), conflicts=(), baseline=(),
                       utilisation=None, root_pid=FIXTURE_ROOT_PID, **over):
    """A monitor sample built by the real instrument, from a machine this file invents.

    `table` is nvidia-smi's raw `pid, process_name, used_memory` answer, `tree` is the process tree
    of *this* instant, `live` is what the OS reports about those pids right now, and `utilisation` is
    a raw `pmon -s u` answer - left as None when the fixture is not making that second query, which is
    a different claim from making it and being told nothing. §5's classifier, §2's attribution and
    §11's published row shape all run; nothing below is a classification typed in by hand, which is
    what made the older fixtures in this file unable to catch the bug they describe.
    """
    by_pid = {int(row["pid"]): row for row in live}
    read_pmon = utilisation is not None
    processes = be.nvidia_compute_processes(
        tree_snapshot=list(tree), rows=[list(row) for row in table],
        process_rows_by_pid=by_pid, root_pid=root_pid,
        utilisation_by_pid=(be.per_process_gpu_utilisation(rows=list(utilisation))[0]
                            if read_pmon else {}))
    classified = be.classify_gpu_processes(processes, workload_rows=list(conflicts),
                                           baseline_identities=list(baseline))
    stamp = (datetime.datetime(2026, 9, 20, tzinfo=datetime.timezone.utc)
             + datetime.timedelta(seconds=unix)).strftime("%Y-%m-%dT%H:%M:%SZ")
    published = [be.published_gpu_row(row, stamp) for row in classified]
    reading = be.runtime_reading({
        "utc": stamp, "unix_seconds": unix,
        "nvidia_compute_processes": published,
        "external_compute_processes": [row for row in published
                                       if row["classification"]
                                       in be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS],
        "conflicting_repo_processes": list(conflicts),
        "repo_processes": [], "tree_of_this_sample": list(tree),
        "benchmark_descendant_pids": len(tree), "gpu_process_count": len(published),
        "ambient_os_graphics_client_count": sum(
            1 for row in published if row["classification"] == "ambient_os_graphics_client"),
        "ambient_or_unproven_gpu_client_count": sum(
            1 for row in published if row["classification"] == "ambient_or_unproven_gpu_client"),
        "per_process_gpu_utilisation_answered": read_pmon})
    # §11 keeps the whole classified table in a preflight sample; §17's monitor record projects it
    # away to keep the file small. Both are handed back here, because these tests are about which
    # rule classified a row, and `external_compute_processes` alone only shows the hard ones.
    reading["nvidia_compute_processes"] = published
    reading.update(over)
    return reading


CPU_AUDIT_BASE = {
    "system_logical_cpu_count": 16, "logical_cpu_count_state": "measured",
    "active_processor_count": 16, "active_processor_count_state": "measured",
    "process_cpu_affinity": list(range(16)), "process_cpu_affinity_count": 16,
    "affinity_state": "measured",
    "process_priority": 32, "process_priority_class": "NORMAL_PRIORITY_CLASS",
    "process_priority_state": "measured",
    "active_power_scheme": "High performance", "power_scheme_state": "measured",
    "processor_max_state_percent": 100.0, "processor_max_state_state": "measured",
    "power_overlay_scheme": None, "power_overlay_state": "none_applied",
    "process_power_throttling": None,
    "process_power_throttling_state": "not_queryable_by_documented_api",
    "job_object_membership": False, "job_cpu_rate_limit_percent": None,
    "job_object_state_name": "no_job_object",
}


def cpu_audit(**over):
    """A §6 audit document for a machine with nothing left on it, with one reading changed.

    The published verdict is filled in by `judge_cpu_restriction` over the same numbers, because the
    gate under test re-runs that judge and fails a record whose verdict and measurements disagree -
    a test that typed both by hand would be checking its own arithmetic.
    """
    doc = dict(CPU_AUDIT_BASE, audited_utc="2026-09-20T00:00:00Z", **over)
    doc["restrictions_detected"] = be.judge_cpu_restriction(doc)
    doc["cpu_restriction_ok"] = not doc["restrictions_detected"]
    return doc


class ChildSpawnedMidQuery:
    """A host on which the benchmark's own transcoder starts *while* the GPU table is being read.

    The clock is the whole fixture. `nvidia-smi --query-compute-apps` costs this host a tenth of a
    second, and the child arrives inside that interval, so the process list either already names it
    or never will - depending only on which of the two reads the list follows. That is §2's torn read:
    at `eb61bbc` the list was taken first, so a row the card had just learned about reached the
    classifier with no parent, no creation time and no tree membership, and three of the batch's own
    processes were written up as strangers.
    """

    ROOT_CREATED = 900.0
    QUERY_SECONDS = 0.1
    CHILD = 7003
    CHILD_NAME = "ffmpeg.exe"
    CHILD_CREATED = 900.05              # inside the query, after the list a stale read would take

    def __init__(self, root_pid):
        self.root_pid = int(root_pid)
        self.now = self.ROOT_CREATED
        self.card_has_been_read = False

    def smi_rows(self, query, compute_apps=False):
        assert compute_apps, "a §11 sample asks the card for the client table"
        self.now += self.QUERY_SECONDS  # the subprocess costs this host time ...
        self.card_has_been_read = True  # ... and the spawn happened during it
        return [[str(self.root_pid), "python.exe", "[N/A]"],
                [str(self.CHILD), self.CHILD_NAME, "1024"]]

    def process_rows(self):
        rows = [{"pid": self.root_pid, "name": "python.exe",
                 "command_line": "python.exe monitor_runtime.py",
                 "process_create_time": self.ROOT_CREATED, "parent_pid": None,
                 "executable_path": "C:\\python\\python.exe"}]
        if self.card_has_been_read:
            rows.append({"pid": self.CHILD, "name": self.CHILD_NAME,
                         "command_line": "ffmpeg.exe -i in.mkv -c:v hevc_nvenc out.mkv",
                         "process_create_time": self.CHILD_CREATED,
                         "parent_pid": self.root_pid,
                         "executable_path": "C:\\tools\\ffmpeg.exe"})
        return rows


class ProcessTreeIdentityTests(unittest.TestCase):
    """§2: the tree a GPU row is excused by has to be the tree of that row's own sample."""

    TRANSCODER = [7001, "ffmpeg.exe", "1024"]

    def test_process_tree_is_sample_local(self):
        """The same client row, the same pid, two samples, two trees, two verdicts.

        This is the bug that stopped the last round: one tree was collected for the batch, so a
        transcoder that had already exited kept being vouched for - and, the other way round, a row
        could be blamed by a tree that was not the one in force when it was read.
        """
        alive = [tree_row(7001, name="ffmpeg.exe", created=1000.0, parent=7000),
                 tree_row(7000, name="python.exe", created=900.0, parent=None)]
        after = [tree_row(7000, name="python.exe", created=900.0, parent=None)]
        live = [live_row(7001, "ffmpeg.exe", created=1000.0, parent=7000),
                live_row(7000, "python.exe", created=900.0)]

        inside = classified_reading(1, table=[self.TRANSCODER], tree=alive, live=live)
        outside = classified_reading(2, table=[self.TRANSCODER], tree=after, live=live)
        self.assertEqual(inside["nvidia_compute_processes"][0]["classification"],
                         "benchmark_descendant")
        self.assertEqual(inside["external_compute_processes"], [])
        # Out of the tree, the same pid is exactly what §5 puts on the hard-fail list: this round's
        # own transcoder, seen from somebody else's process table, is another FFmpeg workload.
        self.assertEqual(outside["nvidia_compute_processes"][0]["classification"],
                         "known_compute_workload")
        self.assertEqual([row["process_name"] for row in outside["external_compute_processes"]],
                         ["ffmpeg.exe"])
        # The sample carries its own tree, so the record can be audited afterwards rather than
        # re-derived from a process table that has moved on.
        self.assertEqual({row["pid"] for row in inside["tree_of_this_sample"]}, {7000, 7001})
        self.assertEqual({row["pid"] for row in outside["tree_of_this_sample"]}, {7000})

    def test_exited_child_is_not_in_future_sample_tree(self):
        """A child that has gone cannot vouch for itself a second time, however loudly it is listed.

        The monitor remembers every pid it ever spawned (`benchmark_child_pids`); §2's rule is that
        the memory is not evidence. The driver keeps a process in its compute table for a moment
        after the process is dead, and a batch's own last transcoder was being read as a neighbour's
        first one by that mismatch.
        """
        monitor = be.RuntimeMonitor(sample_fn=lambda **kwargs: sample())
        monitor.benchmark_child_pids = {7001}          # the batch did spawn it, once
        monitor.ffmpeg_child_pids = {7001}
        live = [live_row(7001, "ffmpeg.exe", created=1000.0, parent=7000)]
        while_here = classified_reading(1, table=[self.TRANSCODER],
                                       tree=[tree_row(7001, name="ffmpeg.exe", created=1000.0,
                                                      parent=7000)],
                                       live=live)
        outcomes = [monitor.observe(reading=while_here,
                                    tree_snapshot=while_here["tree_of_this_sample"])]
        outcomes += [monitor.observe(reading=gone, tree_snapshot=gone["tree_of_this_sample"])
                     for gone in (classified_reading(unix, table=[self.TRANSCODER], tree=[],
                                                     live=live) for unix in (2, 3, 4))]
        self.assertFalse(any(row["newly_contaminated"] for row in outcomes[:1]))
        self.assertTrue(monitor.contaminated,
                        "four consecutive samples of a client nobody's tree contains is a competitor")
        # The monitor's own record is the projected one (§17), so the classification is read back
        # through the conflict list rather than through the full table a preflight sample keeps.
        self.assertEqual(monitor.samples[0]["external_compute_processes"], [])
        self.assertEqual(
            [row["classification"] for row in monitor.samples[3]["external_compute_processes"]],
            ["known_compute_workload"])
        self.assertEqual(monitor.samples[0]["benchmark_descendant_pids"], 1)
        self.assertEqual(monitor.samples[3]["benchmark_descendant_pids"], 0)

    def test_pid_reuse_does_not_inherit_old_identity(self):
        """§2: identity is (pid, create time). Two processes may share a number, not a history."""
        first = [live_row(7000, "blender.exe", created=1111.0, parent=None)]
        second = [live_row(7000, "blender.exe", created=2222.0, parent=None)]
        outsider = [7000, "blender.exe", "1024"]
        readings = ([classified_reading(unix, table=[outsider], live=first) for unix in (1, 2)]
                    + [classified_reading(unix, table=[outsider], live=second) for unix in (3, 4)])
        ledger = be.ContaminationLedger()
        crossed = [ledger.add(reading) for reading in readings]
        self.assertFalse(ledger.contaminated,
                         "two samples of one process and two of the next is not a streak of four")
        self.assertEqual(crossed, [[], [], [], []])
        events = ledger.event_rows()
        self.assertEqual(len(events), 2, "the two processes are two records")
        self.assertEqual({row["pid"] for row in events}, {7000})
        self.assertEqual(sorted(row["consecutive_samples"] for row in events), [2, 2])
        # And the same pid, the same create time, three samples in a row, is still a competitor:
        # the identity rule narrows the accusation, it does not forgive the process it names.
        again = be.ContaminationLedger()
        for unix in (1, 2, 3):
            again.add(classified_reading(unix, table=[outsider], live=first))
        self.assertTrue(again.contaminated)

    def test_benchmark_ffmpeg_is_not_external_contamination(self):
        """§5: this batch spawns transcoders on purpose. Seeing one is not a competitor."""
        live = [live_row(7001, "ffmpeg.exe", created=1000.0, parent=7000),
                live_row(7000, "python.exe", created=900.0)]
        tree = [tree_row(7000, name="python.exe", created=900.0, parent=None),
                tree_row(7001, name="ffmpeg.exe", created=1000.0, parent=7000)]
        self.assertTrue(be.looks_like_a_gpu_workload("ffmpeg.exe"),
                        "the same image name is a workload when it is not ours")
        ledger = be.judge_contamination([classified_reading(unix, table=[self.TRANSCODER],
                                                             tree=tree, live=live)
                                          for unix in range(1, 11)])
        self.assertFalse(ledger.contaminated, ledger.reason)
        self.assertEqual(ledger.event_rows(), [])

    def test_a_child_spawned_between_the_two_reads_is_attributed(self):
        """§2's fix is the *order of the reads inside one sample*, so the order is what is pinned.

        Two samples of the same host, the same child and the same GPU table. The first lets `sample()`
        work as §2 now specifies it - card first, list second - and the child is in both. The second
        hands `sample()` the list it captured *before* the card was queried, which is what this
        function did at `eb61bbc`. Nothing about the child differs between them; only the moment the
        list was read does, and the two verdicts are the whole reason a comment in `sample()` is not
        enough.
        """
        def read(host, rows=None):
            with mock.patch.object(be, "_smi_rows", host.smi_rows), \
                    mock.patch.object(be, "_pmon_table", lambda **kwargs: []), \
                    mock.patch.object(be, "process_rows", host.process_rows), \
                    mock.patch.object(be, "gpu_readings",
                                      lambda: {"gpu_available": True,
                                               "nvidia_smi_answered": True}), \
                    mock.patch.object(be.psutil, "cpu_percent", lambda interval=None: 0.0):
                return be.sample(root_pid=FIXTURE_ROOT_PID, rows=rows)

        fresh = ChildSpawnedMidQuery(FIXTURE_ROOT_PID)
        stale_host = ChildSpawnedMidQuery(FIXTURE_ROOT_PID)
        stale_list = stale_host.process_rows()
        self.assertNotIn(ChildSpawnedMidQuery.CHILD, [row["pid"] for row in stale_list],
                         "the fixture has to start where the old code did: list before card")
        old_order = read(stale_host, rows=stale_list)
        new_order = read(fresh)

        # Read after the card, the list names the child, so the child's own identity is in the tree of
        # the sample that judges it.
        mine = [row for row in new_order["nvidia_compute_processes"]
                if row["pid"] == ChildSpawnedMidQuery.CHILD][0]
        self.assertTrue(mine["is_current_benchmark_descendant"])
        self.assertEqual(mine["descendant_attribution"], "identity_in_this_sample_tree")
        self.assertEqual(mine["classification"], "benchmark_descendant")
        self.assertEqual(mine["parent_pid"], FIXTURE_ROOT_PID)
        self.assertEqual(new_order["external_compute_processes"], [],
                         "the batch's own transcoder must never reach the list that stops a round")
        self.assertEqual({row["pid"] for row in new_order["tree_of_this_sample"]},
                         {FIXTURE_ROOT_PID, ChildSpawnedMidQuery.CHILD})

        # Read before it, the same row has no parent and no creation time, and no care in the
        # classifier can recover a fact that was never read: this is the finding, written down.
        theirs = [row for row in old_order["nvidia_compute_processes"]
                  if row["pid"] == ChildSpawnedMidQuery.CHILD][0]
        self.assertIsNone(theirs["parent_pid"])
        self.assertIsNone(theirs["process_create_time"])
        self.assertIsNone(theirs["descendant_attribution"])
        self.assertFalse(theirs["is_current_benchmark_descendant"])
        self.assertEqual(theirs["classification"], "known_compute_workload")
        self.assertEqual([row["pid"] for row in old_order["external_compute_processes"]],
                         [ChildSpawnedMidQuery.CHILD])
        self.assertEqual(theirs["compute_criteria_met"],
                         ["driver_accounts_gpu_memory_in_use",
                          "image_path_or_command_names_a_compute_workload"])
        self.assertEqual({row["pid"] for row in old_order["tree_of_this_sample"]},
                         {FIXTURE_ROOT_PID})
        # The ordering is not an accident of this one sample function either: three samples of the
        # stale reading is what §18's streak asks for, and a batch's own child cannot survive it.
        self.assertTrue(be.judge_contamination([old_order] * 3).contaminated)
        self.assertFalse(be.judge_contamination([new_order] * 3).contaminated)

    def test_benchmark_ffprobe_is_not_external_contamination(self):
        """The probe the artifact gate runs is the monitor's own child, not a client of the card."""
        probing = [7002, "ffprobe.exe", "512"]
        live = [live_row(7002, "ffprobe.exe", created=1500.0, parent=7000),
                live_row(7000, "python.exe", created=900.0)]
        tree = [tree_row(7000, name="python.exe", created=900.0, parent=None),
                tree_row(7002, name="ffprobe.exe", created=1500.0, parent=7000)]
        reading = classified_reading(1, table=[probing], tree=tree, live=live)
        self.assertEqual(reading["nvidia_compute_processes"][0]["classification"],
                         "benchmark_descendant")
        # The driver accounted real memory for it, so nothing but the tree separates it from the
        # outsider it becomes one sample later.
        unwatched = classified_reading(2, table=[probing], tree=[tree[0]], live=live)
        self.assertEqual(unwatched["external_compute_processes"][0]["classification"],
                         "known_compute_workload")
        ledger = be.judge_contamination([classified_reading(unix, table=[probing], tree=tree,
                                                            live=live)
                                         for unix in range(1, 7)])
        self.assertFalse(ledger.contaminated)


class QuietWindowAdmissibilityTests(unittest.TestCase):
    """§3: the gate reads the window it accepted, and the wait that found it stays in the record."""

    BUSY = [sample(cpu=55.0, unix=(index % 30) + 1) for index in range(30 * 30)]
    QUIET = [sample(unix=(index % 30) + 1) for index in range(30)]

    def test_nine_failed_windows_then_pass_is_valid(self):
        """§3's case: nine busy windows and an idle tenth is an idle machine when it started."""
        rows = [sample(cpu=55.0, unix=(index % 30) + 1) for index in range(270)] + list(self.QUIET)
        doc = preflight(rows, max_wait_seconds=600.0)
        self.assertEqual(doc["preflight_window_count_attempted"], 10)
        self.assertEqual(doc["preflight_window_that_passed"], 10)
        self.assertEqual(doc["preflight_status"], "PASS")
        self.assertEqual(len(doc["all_preflight_samples"]), 300)
        # The history is not deleted with the failures: nine refused windows, each with its reason.
        self.assertEqual([row["ok"] for row in doc["windows"]], [False] * 9 + [True])
        self.assertTrue(all(row["failures"] for row in doc["windows"][:9]))
        gate = status.quiet_preflight_gate(doc)
        self.assertEqual(gate["status"], "DONE_PROVEN", gate["failed_checks"])
        # And the high CPU the earlier windows saw is still in the file, under its own name.
        self.assertEqual(
            doc["all_preflight_samples"][0]["system_cpu_percent"], 55.0)

    def test_no_quiet_window_is_not_ready(self):
        doc = preflight([sample(cpu=55.0, unix=(index % 30) + 1) for index in range(300)],
                        max_wait_seconds=600.0)
        self.assertEqual(doc["preflight_status"], "FAIL")
        self.assertEqual(doc["not_ready_reason"], "no_quiet_preflight_window")
        self.assertIsNone(doc["selected_passing_window"])
        gate = status.quiet_preflight_gate(doc)
        self.assertEqual(gate["status"], "NOT_READY")
        failed = set(gate["failed_checks"])
        self.assertIn("one_window_passed_within_the_ten_minute_budget", failed)
        self.assertIn("the_selected_window_itself_held_every_threshold", failed)
        self.assertLessEqual(doc["preflight_window_count_attempted"], 20)

    def test_window_after_budget_is_not_ready(self):
        """§15: a long wait is allowed; certifying a window that arrived after it is not.

        Two halves, because two components can get this wrong. `quiet_preflight` stops asking when
        its budget is spent, so a machine that goes idle in the eleventh minute yields no selection
        at all; and the gate has to refuse a document that claims one anyway - including one that
        enlarged its own `preflight_wait_budget_seconds` field to make the claim fit.
        """
        never = preflight([sample(cpu=55.0, unix=(index % 30) + 1) for index in range(30 * 20)],
                          max_wait_seconds=600.0)
        self.assertEqual(never["preflight_status"], "FAIL")
        self.assertIsNone(never["selected_passing_window"])
        self.assertEqual(never["not_ready_reason"], "no_quiet_preflight_window")
        self.assertGreaterEqual(never["preflight_window_count_attempted"], 10,
                               "the whole ten minutes was spent looking, and nothing was chosen")
        self.assertEqual(status.quiet_preflight_gate(never)["status"], "NOT_READY")

        late = json.loads(json.dumps(preflight(
            [sample(cpu=55.0, unix=(index % 30) + 1) for index in range(270)] + list(self.QUIET),
            max_wait_seconds=600.0)))
        self.assertEqual(status.quiet_preflight_gate(late)["status"], "DONE_PROVEN")
        late["selected_passing_window"]["appeared_after_seconds"] = 900.0
        late["selected_passing_window"]["within_wait_budget"] = False
        late["preflight_wait_budget_seconds"] = 3600.0
        gate = status.quiet_preflight_gate(late)
        self.assertEqual(gate["status"], "NOT_READY")
        self.assertEqual(gate["failed_checks"],
                         ["one_window_passed_within_the_ten_minute_budget"],
                         "the budget is the module's frozen ten minutes, not the document's claim")

    def test_three_samples_of_an_outsider_stop_the_window_and_two_do_not(self):
        """§18's rule, now §14's rule as well: what a window refuses is an outsider that stayed.

        The rule changed in this round because the old one refused a window for a single sighting -
        a `git` wrapper, a thumbnail generator, a chat window that opened its GPU context for two
        seconds - and a wait that refused ten of those had no quiet window left to measure on. Both
        halves are run through `quiet_preflight` and `judge_window` rather than described: the
        instrument's own summary fields are what the gate below reads, so the fixtures have to produce
        them.
        """
        outsider = [gpu_row(pid=9001, name="blender.exe", memory="1024")]
        transient = [sample(unix=index + 1, compute=outsider if index in (11, 12) else ())
                     for index in range(30)]
        doc = preflight(transient, max_wait_seconds=600.0)
        self.assertEqual(doc["preflight_status"], "PASS", doc["not_ready_reason"])
        summary = doc["selected_passing_window"]["summary"]
        self.assertEqual(summary["sustained_external_compute_processes"], [])
        self.assertTrue(any("9001" in row and "blender.exe" in row for row in
                            summary["transient_external_compute_process_sightings"]),
                        "recorded, not refused - the sightings do not disappear because they lost")
        gate = status.quiet_preflight_gate(json.loads(json.dumps(doc)))
        self.assertEqual(gate["status"], "DONE_PROVEN", gate["failed_checks"])
        # Three consecutive samples of the same identity is the other sentence, and the window says
        # so instead of measuring anyway.
        sustained = [sample(unix=index + 1, compute=outsider if index in (10, 11, 12) else ())
                     for index in range(30)]
        judged = be.judge_window(sustained)
        self.assertFalse(judged["ok"])
        self.assertIn("external_gpu_compute_workload", {row["reason"] for row in judged["failures"]})
        # With three in a row there is nothing left to tell apart: the refused list is the sustained
        # list, and the sightings it declined to refuse are gone. The sustained label carries the
        # classification beside it, which the plain sighting list does not.
        self.assertEqual(judged["summary"]["sustained_external_compute_processes"],
                         ["9001 blender.exe 1024 known_compute_workload"])
        self.assertEqual(judged["summary"]["external_compute_processes"],
                         ["9001 blender.exe 1024"])
        self.assertEqual(judged["summary"]["transient_external_compute_process_sightings"], [])
        self.assertEqual(judged["recorded_not_refused"], [])
        refused = preflight(sustained, max_wait_seconds=120.0)
        self.assertEqual(refused["preflight_status"], "FAIL")
        self.assertEqual(refused["not_ready_reason"], "no_quiet_preflight_window")

    def test_a_record_that_blames_a_window_it_accepted_is_not_ready(self):
        """The gate reads the selected window, in both directions.

        Two failures a real round cannot produce by accident and a hand-written artifact can: a
        selected window that carries a sustained outsider the waiting never refused for, and a selected
        window whose summary has no `sustained_external_compute_processes` field at all. The second is
        the one §3 exists for - an absent list and an empty list are different claims, and a gate that
        read the absence as emptiness would certify a window it never checked.
        """
        doc = json.loads(json.dumps(preflight(
            [sample(cpu=55.0, unix=(index % 30) + 1) for index in range(270)]
            + list(QuietWindowAdmissibilityTests.QUIET), max_wait_seconds=600.0)))
        self.assertEqual(status.quiet_preflight_gate(doc)["status"], "DONE_PROVEN")
        sustained = doc["selected_passing_window"]["summary"][
            "sustained_external_compute_processes"]
        self.assertEqual(sustained, [])

        forged = json.loads(json.dumps(doc))
        forged["selected_passing_window"]["summary"]["sustained_external_compute_processes"] = [
            "9001 blender.exe 1024"]
        gate = status.quiet_preflight_gate(forged)
        self.assertEqual(gate["status"], "NOT_READY")
        self.assertIn("no_sustained_external_gpu_compute_workload_was_recorded",
                      gate["failed_checks"])

        missing = json.loads(json.dumps(doc))
        del missing["selected_passing_window"]["summary"]["sustained_external_compute_processes"]
        gate = status.quiet_preflight_gate(missing)
        self.assertEqual(gate["status"], "NOT_READY")
        self.assertIn("no_sustained_external_gpu_compute_workload_was_recorded",
                      gate["failed_checks"])

        other = json.loads(json.dumps(doc))
        other["external_compute_processes_across_all_windows"] = ["9001 blender.exe 1024"]
        other["external_compute_processes"] = ["9001 blender.exe 1024"]
        # ... and leave it only in the union: §3 says the gate reads the selection, and the refused
        # windows are kept rather than hidden.
        self.assertEqual(status.quiet_preflight_gate(other)["status"], "DONE_PROVEN",
                         status.quiet_preflight_gate(other)["failed_checks"])


class CorroborationReasonTests(unittest.TestCase):
    """§4: a batch that never ran and a batch that disagreed are different sentences."""

    def test_corroboration_not_run_has_distinct_reason(self):
        """The seven-execution attempt wrote `corroboration_conflicts_with_primary_run` for a 150f
        batch that had not been launched. Every name below is produced by the real function."""
        cases = {
            # the last round, as it happened: 7 of 12 primary executions, contamination, nothing run
            "contaminated": be.corroboration_reason(
                7, 12, 0, 4, runtime_contaminated=True, preflight_passed=True),
            "preflight": be.corroboration_reason(
                0, 12, 0, 4, runtime_contaminated=False, preflight_passed=False),
            "never ran, nothing wrong": be.corroboration_reason(12, 12, 0, 4),
            "short": be.corroboration_reason(12, 12, 3, 4),
            "no direction": be.corroboration_reason(12, 12, 4, 4),
            "agrees": be.corroboration_reason(12, 12, 4, 4, primary_direction="reduced_ahead",
                                              corroborating_direction="reduced_ahead"),
            "conflicts": be.corroboration_reason(12, 12, 4, 4, primary_direction="reduced_ahead",
                                                 corroborating_direction="legacy_ahead"),
        }
        self.assertEqual(cases["contaminated"], "corroboration_not_run_due_runtime_contamination")
        self.assertEqual(cases["preflight"], "corroboration_not_run_due_preflight_failure")
        self.assertEqual(cases["never ran, nothing wrong"], "corroboration_not_run")
        self.assertEqual({cases["short"], cases["no direction"]}, {"corroboration_incomplete"})
        self.assertEqual(cases["agrees"], "corroboration_agrees_with_primary_run")
        self.assertEqual(len(set(cases.values()) - {"corroboration_incomplete"}), 5,
                         "six names, and none of them standing in for another")
        # §4: only two complete, valid batches in opposite directions may be called a conflict.
        self.assertNotEqual(cases["contaminated"], "corroboration_conflicts_with_primary_run")
        for incomplete in ("contaminated", "preflight", "never ran, nothing wrong", "short",
                          "no direction"):
            self.assertNotEqual(cases[incomplete], "corroboration_conflicts_with_primary_run",
                                f"{incomplete} is not a conflict")
        self.assertEqual(sorted(be.CORROBORATION_REASONS), sorted(set(cases.values())))


class AmbientOsGraphicsClientTests(unittest.TestCase):
    """§5: an exact identity for one named OS component, and no directory of any kind."""

    def shellhost(self, path=None, command_line="ShellHost.EXE -Embedding", memory="[N/A]"):
        return [live_row(4100, "ShellHost.exe", created=5000.0, parent=1200,
                         command_line=command_line,
                         executable=path or os.path.join(
                             be.windows_system_root() or "C:\\Windows", "System32",
                             "ShellHost.exe"))]

    def table_row(self, memory="[N/A]"):
        return ["4100", "ShellHost.exe", memory]

    def test_shellhost_exact_identity_is_ambient(self):
        """§5's conditions, each of them, on the row that this round's false alarm was built from."""
        live = self.shellhost()
        # A baseline that does not contain it: this is the row that joined the card mid-window, which
        # is the exact shape the last round read as a competitor settling in.
        everyone_else = [{"pid": 1200, "process_create_time": 500.0}]
        reading = classified_reading(1, table=[self.table_row()], live=live,
                                     baseline=everyone_else)
        row = reading["nvidia_compute_processes"][0]
        self.assertEqual(row["classification"], "ambient_os_graphics_client")
        self.assertEqual(row["ambient_identity_failed_conditions"], [])
        self.assertTrue(row["arrived_during_the_window"],
                        "it did arrive mid-window, and §5 says that alone is not a conflict")
        self.assertEqual(reading["external_compute_processes"], [])
        ledger = be.judge_contamination([classified_reading(unix, table=[self.table_row()],
                                                           live=live)
                                         for unix in range(1, 11)])
        self.assertFalse(ledger.contaminated, ledger.reason)
        self.assertEqual(ledger.event_rows(), [])

    def test_system32_wildcard_is_not_allowed(self):
        """§5: 禁止任何 System32 路径级白名单. Only one full file name is ever matched.

        The expectation in this test changed with the rule. A System32 neighbour used to be refused
        for being a neighbour, which is what made a directory prefix look tempting; it is now recorded
        as unproven unless something about it demonstrates compute. What the test has to pin is that
        location still buys nothing in *either* direction: being under System32 neither excuses a row
        that has the evidence, nor condemns one that has none, and the one named component is matched
        as one file name rather than as a directory.
        """
        system32 = os.path.join(be.windows_system_root() or "C:\\Windows", "System32")
        self.assertEqual(be.AMBIENT_OS_GRAPHICS_EXECUTABLES, ("shellhost.exe",))
        # What the path helper can produce is one file name for one table entry. It is not a prefix,
        # it takes no directory argument, and nothing in the module turns a directory into a rule.
        self.assertEqual(be.windows_ambient_component_path("shellhost.exe"),
                         os.path.join(system32, "ShellHost.exe".lower()))
        # Membership in the one-entry table is what forgives by identity, not location: these sit in
        # System32, none of them is the named component, and none of them is therefore the *narrow*
        # ambient class. They are unproven instead, which is a different claim - see the second half
        # of this test, where the same directory does not save one of them that has evidence.
        baseline = [{"pid": 1200, "process_create_time": 500.0}]
        for name in ("notepad.exe", "taskhostw.exe", "explorer.exe", "ShellHost.exe.vir",
                     "ShellHost.EXE.config"):
            row = be.classify_gpu_processes(be.nvidia_compute_processes(
                tree_snapshot=[], rows=[["4101", name, "[N/A]"]],
                process_rows_by_pid={4101: live_row(4101, name, command_line=name,
                                                    executable=os.path.join(system32, name))},
            ), baseline_identities=baseline)[0]
            self.assertNotEqual(row["classification"], "ambient_os_graphics_client", name)
            self.assertNotIn(row["classification"], be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS, name)
            self.assertEqual(row["classification"], "ambient_or_unproven_gpu_client", name)
            self.assertEqual(row["ambient_identity_failed_conditions"],
                             ["basename_is_not_in_the_ambient_table"], name)
            self.assertEqual(row["compute_criteria_met"], [], name)
        # The other identity conditions bite on the named executable itself. Each row below arrived
        # mid-window, which the old rule condemned on sight; what it now turns on is whether the
        # path, the memory or the command line can be read.
        arrived = [{"pid": 1200, "process_create_time": 500.0}]
        for label, row in (
                ("another directory", self.shellhost(path=os.path.join("D:\\Tools",
                                                                       "ShellHost.exe"))),
                ("a System32 subdirectory", self.shellhost(
                    path=os.path.join(system32, "config", "ShellHost.exe"))),
                ("no command line", self.shellhost(command_line=None))):
            classified = be.classify_gpu_processes(be.nvidia_compute_processes(
                tree_snapshot=[], rows=[self.table_row()],
                process_rows_by_pid={4100: row[0]}), baseline_identities=arrived)[0]
            self.assertNotEqual(classified["classification"], "ambient_os_graphics_client", label)
            self.assertTrue(classified["ambient_identity_failed_conditions"], label)
        # And a System32 row that *does* demonstrate compute is refused exactly as loudly as one
        # anywhere else: the same path, the same directory, the evidence on its side.
        hiding = be.classify_gpu_processes(be.nvidia_compute_processes(
            tree_snapshot=[], rows=[["4100", "ShellHost.exe", "4096"]],
            process_rows_by_pid={4100: self.shellhost()[0]}),
            baseline_identities=[{"pid": 1200, "process_create_time": 500.0}])[0]
        self.assertEqual(hiding["classification"], "unknown_external_gpu_client")
        self.assertIn("driver_accounts_gpu_memory_in_use", hiding["compute_criteria_met"])
        self.assertTrue(hiding["arrived_during_the_window"],
                        "it did arrive mid-window, and that is recorded - it is simply not the reason")

    def client(self, name="ShellHost.exe", pid=4100, memory="[N/A]", live=None, utilisation=None):
        """One GPU-table row, classified by the instrument, with only what the caller hands in.

        `live=None` means the OS answered nothing about the pid at all - no path, no command line -
        which is one of the four facts §5's criteria read rather than a verdict in itself.
        """
        reading = classified_reading(1, table=[[pid, name, memory]],
                                     live=self.shellhost() if live is None else live,
                                     utilisation=utilisation,
                                     baseline=[{"pid": 1200, "process_create_time": 500.0}])
        return reading["nvidia_compute_processes"][0]

    def test_each_of_the_four_criteria_alone_makes_a_client_a_conflict(self):
        """§5's positive-evidence rule, one criterion at a time, on the same pid and the same name.

        Nothing below differs between the cases except the single fact that changed, so what the
        classifier answers is an answer about that fact. The five cases that are *not* conflicts are
        the ones that stopped the previous round's batch: presence, arrival, and an unreadable
        identity, each of which used to be enough on its own.
        """
        cases = (
            ("criterion 1, the driver accounts for its memory",
             dict(memory="2048"), "unknown_external_gpu_client",
             "driver_accounts_gpu_memory_in_use"),
            ("criterion 2, a coding engine is occupied",
             dict(memory="[N/A]", utilisation=[{"pid": 4100, "type": "C+G", "sm": "3", "mem": "-",
                                                "enc": "62", "dec": "-", "jpg": "-", "ofa": "-",
                                                "command": "ShellHost.EXE"}]),
             "unknown_external_gpu_client", "codec_engine_occupied"),
            ("criterion 3, its command line names a transcoder",
             dict(memory="[N/A]", live=self.shellhost(
                 command_line="ShellHost.EXE -Embedding ffmpeg -i in.mp4")),
             "known_compute_workload", "image_path_or_command_names_a_compute_workload"),
            ("criterion 4, its own share of the card is at the frozen limit",
             dict(memory="[N/A]", utilisation=[{"pid": 4100, "type": "C+G", "sm": "15", "mem": "-",
                                                "enc": "-", "dec": "-", "jpg": "-", "ofa": "-",
                                                "command": "ShellHost.EXE"}]),
             "unknown_external_gpu_client",
             "per_process_gpu_utilisation_at_or_above_the_frozen_limit"),
            ("no criterion at all: nothing readable about it",
             dict(memory="[N/A]", live=[]), "ambient_or_unproven_gpu_client", None))
        for label, given, expected, criterion in cases:
            row = self.client(**given)
            self.assertEqual(row["classification"], expected, label)
            self.assertEqual(row["compute_criteria_met"], [criterion] if criterion else [], label)
            self.assertEqual(row["classification"] in be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS,
                             bool(criterion), label)
            if criterion:
                self.assertIn(criterion, row["classification_reason"], label)
        # The five fields §11's `[N/A]` rule protects are in the record either way, so the forgiven
        # rows are forgiven on the numbers a reader can see rather than on this file's word.
        forgiven = self.client(memory="[N/A]", live=[])
        self.assertIsNone(forgiven["used_memory_bytes"])
        self.assertFalse(forgiven["memory_reported_by_driver"])
        self.assertIsNone(forgiven["gpu_sm_utilisation_percent"])
        self.assertTrue(forgiven["arrived_during_the_window"],
                        "it did join the table mid-window, and that stays recorded")
        self.assertNotIn(forgiven["classification"], be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS)

    def test_pmons_type_column_is_not_evidence_of_compute(self):
        """The measured reason criterion two is not `type == "C"`: on this card every row says C+G.

        105 of 105 rows across three samples of `nvidia-smi pmon -c 1 -s u`, the shell and the editor
        included. A rule that read that column as compute would name `explorer.exe` a CUDA competitor
        in every sample of every window, forever.
        """
        ambient_band = self.client(name="ChatGPT Classic.exe", live=[],
                                   utilisation=[{"pid": 4100, "type": "C+G", "sm": "8", "mem": "-",
                                                 "enc": "-", "dec": "-", "jpg": "-", "ofa": "-",
                                                 "command": "ChatGPT Classic.EXE"}])
        self.assertEqual(ambient_band["classification"], "ambient_or_unproven_gpu_client")
        self.assertEqual(ambient_band["pmon_engine_type_reported"], "C+G")
        self.assertEqual(ambient_band["pmon_command_name"], "ChatGPT Classic.EXE")
        self.assertEqual(ambient_band["gpu_sm_utilisation_percent"], 8.0)
        self.assertLess(ambient_band["gpu_sm_utilisation_percent"],
                        be.PER_PROCESS_GPU_COMPUTE_PERCENT_AT_LEAST)
        # And the figure the driver refuses to give is a dash, which is not a zero either.
        unasked = self.client(name="ChatGPT Classic.exe", live=[],
                              utilisation=[{"pid": 4100, "type": "C+G", "sm": "-", "mem": "-",
                                            "enc": "-", "dec": "-", "jpg": "-", "ofa": "-",
                                            "command": "ChatGPT Classic.EXE"}])
        self.assertIsNone(unasked["gpu_sm_utilisation_percent"])
        self.assertEqual(unasked["compute_criteria_met"], [])

    def test_a_name_decides_nothing_unless_it_names_compute(self):
        """§5: 禁止按应用名做白名单 - and the mirror of it, which is that no name condemns either.

        Two rows identical in every field except the image name classify the same way, in both
        directions of the evidence: neither is forgiven for being called something ordinary, and
        neither is refused for being called something unusual. The one name rule that stays is the
        compute-word rule, and it is a positive claim about the workload rather than a list of names
        this file has decided to like.
        """
        for name in ("chatgpt_classic.exe", "zune_desktop.exe"):
            self.assertFalse(be.looks_like_a_gpu_workload(name),
                             "%s is a neutral name: this test is about it being neutral" % name)
            without_evidence = self.client(name=name, memory="[N/A]", live=[])
            with_memory = self.client(name=name, memory="2048", live=[])
            self.assertEqual(without_evidence["classification"], "ambient_or_unproven_gpu_client",
                             name)
            self.assertEqual(with_memory["classification"], "unknown_external_gpu_client", name)
        # The same evidence under both names gives the same verdict, which is what a whitelist of
        # either kind would make impossible.
        self.assertEqual({self.client(name=name, memory="2048", live=[])["classification"]
                          for name in ("chatgpt_classic.exe", "zune_desktop.exe")},
                         {"unknown_external_gpu_client"})
        self.assertEqual({self.client(name=name, memory="[N/A]", live=[])["classification"]
                          for name in ("chatgpt_classic.exe", "zune_desktop.exe")},
                         {"ambient_or_unproven_gpu_client"})

    def test_a_desktop_client_that_stays_the_whole_window_does_not_stop_the_round(self):
        """The round at `f7d35df` stopped itself over `ChatGPT Classic.exe`. It cannot do that twice.

        Thirty samples of a windowing client that the driver lists, accounts no memory for, and says
        nothing else about: the preflight passes on the window that saw it, the monitor never
        contaminates on it, and the row is in the artifact either way - which is the half of §5 that
        says 必须继续记录.
        """
        classic = [gpu_row(pid=4102, name="ChatGPT Classic.exe", memory="[N/A]",
                           created=1000.0, parent=900)]
        doc = preflight([sample(unix=offset, compute=classic) for offset in range(1, 31)])
        self.assertEqual(doc["preflight_status"], "PASS", doc["not_ready_reason"])
        window = doc["selected_passing_window"]
        self.assertEqual(window["summary"]["sustained_external_compute_processes"], [])
        self.assertEqual(window["summary"]["external_compute_processes"], [])
        self.assertEqual(window["summary"]["transient_external_compute_process_sightings"], [])
        self.assertTrue(any(entry["process_name"] == "ChatGPT Classic.exe"
                            and entry["classification"] == "ambient_or_unproven_gpu_client"
                            for row in [row for row in doc["all_preflight_samples"]
                                        if row.get("preflight_window_attempt")
                                        == doc["preflight_window_that_passed"]]
                            for entry in row["nvidia_compute_processes"]),
                        "§5's other half: the client is written down in the accepted window's own "
                        "samples, not quietly left out of them")
        self.assertEqual(doc["external_compute_processes"], [])
        self.assertEqual(doc["all_preflight_samples"][0]["ambient_or_unproven_gpu_client_count"], 1)
        gate = status.quiet_preflight_gate(json.loads(json.dumps(doc)))
        self.assertEqual(gate["status"], "DONE_PROVEN", gate["failed_checks"])
        ledger = be.judge_contamination([sample(unix=offset, compute=classic)
                                         for offset in range(1, 11)])
        self.assertFalse(ledger.contaminated, ledger.reason)
        self.assertEqual(ledger.event_rows(), [])

    def test_an_unproven_client_that_starts_using_the_card_is_still_refused(self):
        """The forgiveness is about the evidence, not about the app: change one field and it bites.

        The same pid, the same name, the same window - and a memory figure the driver now accounts
        for. Ten samples of that contaminate the batch and refuse the window, which is what keeps
        §5's ambient half from becoming a name list with extra steps.
        """
        idle = gpu_row(pid=4102, name="ChatGPT Classic.exe", memory="[N/A]", created=1000.0)
        working = gpu_row(pid=4102, name="ChatGPT Classic.exe", memory="2048", created=1000.0)
        ledger = be.judge_contamination([sample(unix=offset,
                                                compute=[idle if offset < 4 else working])
                                         for offset in range(1, 11)])
        self.assertTrue(ledger.contaminated)
        self.assertIn("3 consecutive", ledger.reason)
        self.assertIn("ChatGPT Classic.exe", ledger.reason)
        events = ledger.event_rows()
        self.assertEqual([row["classification"] for row in events], ["unknown_external_gpu_client"])
        self.assertEqual(events[0]["compute_criteria_met"], ["driver_accounts_gpu_memory_in_use"])
        refused = preflight([sample(unix=(index % 30) + 1,
                                    compute=[idle] if index < 2 else [working])
                             for index in range(30 * 3)], max_wait_seconds=120.0)
        self.assertEqual(refused["preflight_status"], "FAIL")
        self.assertIn("external_gpu_compute_workload",
                      {row["reason"] for row in refused["windows"][-1]["failures"]})

    def test_scsp_dlss_smoke_remains_hard_conflict(self):
        """§5's list of what must still stop a round: another CUDA/DLSS smoke workload."""
        self.assertTrue(be.looks_like_a_gpu_workload("scsp_dlss_smoke.exe"))
        row = be.classify_gpu_processes([be.nvidia_compute_processes(
            tree_snapshot=[], rows=[["4200", "scsp_dlss_smoke.exe", "[N/A]"]],
            process_rows_by_pid={4200: live_row(4200, "scsp_dlss_smoke.exe",
                                                command_line="scsp_dlss_smoke.exe --dlss")})[0]])[0]
        self.assertEqual(row["classification"], "known_compute_workload")
        self.assertIn(row["classification"], be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS)
        # Its long life is no defence: it was already there when the window started, and §5 names it
        # rather than inferring it from a directory.
        ledger = be.judge_contamination([classified_reading(
            unix, table=[["4200", "scsp_dlss_smoke.exe", "[N/A]"]],
            live=[live_row(4200, "scsp_dlss_smoke.exe", command_line="scsp_dlss_smoke.exe")],
            baseline=[{"pid": 4200, "process_create_time": None}]) for unix in (1, 2, 3)])
        self.assertTrue(ledger.contaminated)
        self.assertEqual(ledger.event_rows()[0]["classification"], "known_compute_workload")


class CpuRestrictionGateTests(unittest.TestCase):
    """§6: the three readings that stop a round, and the honesty of the fields around them."""

    def gate(self, doc):
        return status.cpu_restriction_gate(doc, bench_env=None)

    def test_cpu_affinity_restriction_blocks_benchmark(self):
        doc = cpu_audit(process_cpu_affinity=list(range(8)), process_cpu_affinity_count=8)
        self.assertEqual([row["reason"] for row in doc["restrictions_detected"]],
                         ["cpu_affinity_restricted"])
        gate = self.gate(doc)
        self.assertEqual(gate["status"], "NOT_READY")
        self.assertIn("no_cpu_restriction_was_left_on_the_machine", gate["failed_checks"])
        self.assertEqual(gate["detail"]["audit"]["process_cpu_affinity_count"], 8)

    def test_cpu_rate_limit_blocks_when_measured(self):
        doc = cpu_audit(job_object_membership=True, job_object_state_name="measured",
                        job_cpu_rate_limit_percent=50.0)
        self.assertEqual([row["reason"] for row in doc["restrictions_detected"]],
                         ["cpu_rate_limited"])
        self.assertEqual(self.gate(doc)["status"], "NOT_READY")

    def test_cpu_power_cap_blocks_when_measured(self):
        """A plan left at 99% caps the benchmark and looks exactly like an idle machine."""
        doc = cpu_audit(processor_max_state_percent=99.0)
        self.assertEqual([row["reason"] for row in doc["restrictions_detected"]],
                         ["cpu_power_cap_detected"])
        gate = self.gate(doc)
        self.assertEqual(gate["status"], "NOT_READY")
        self.assertIn("no_cpu_restriction_was_left_on_the_machine", gate["failed_checks"])

    def test_unavailable_cpu_limit_metric_is_not_zero(self):
        """§6: 如果字段无法取得：写 null / unavailable，禁止写 0."""
        unaskable = cpu_audit(job_object_membership=True,
                              job_object_state_name="in_job_and_its_limits_are_not_queryable",
                              job_cpu_rate_limit_percent=None)
        gate = self.gate(unaskable)
        self.assertEqual(gate["status"], "DONE_PROVEN", gate["failed_checks"])
        self.assertIsNone(unaskable["job_cpu_rate_limit_percent"])
        self.assertIsNone(cpu_audit(process_power_throttling=None,
                                    process_power_throttling_state=
                                    "not_queryable_by_documented_api")
                          ["process_power_throttling"])
        for value in (0, 0.0):
            lied = cpu_audit(job_object_state_name="unavailable", job_cpu_rate_limit_percent=value)
            gate = self.gate(lied)
            self.assertEqual(gate["status"], "NOT_READY")
            self.assertIn("a_field_the_machine_refused_to_answer_is_null_and_not_a_number",
                          gate["failed_checks"])
            self.assertIn("no_limit_is_reported_as_zero", gate["failed_checks"])
        # A refusal is never a pass either: it is simply not a reason to stop the round.
        self.assertNotIn("job_cpu_rate_limit_percent",
                         [row["field"] for row in be.judge_cpu_restriction(unaskable)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
