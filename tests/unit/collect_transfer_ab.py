#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the legacy-versus-reduced A/B on the 4K fixture and write its two artifacts.

Both orders process the same fixture, the same 60 frames, the same 50% working resolution, the
same NVOF preset, the same DLSS settings, the same encoder and the same CQ. Only the transfer
order differs, and each run asks for it explicitly: a silent fallback would turn this into a
comparison of a path against itself, so every run's `transfer_mode_actual` is checked against
what it requested before any number is read.

Each run happens in a fresh child process, which is what makes the per-process figures
(sessions, handles, resident memory) comparable between the two orders.

Two properties of the schedule are part of the protocol, not of the report that reads it:

  * the first pair of an execution is a warmup. It is run, kept and published, and it takes no
    part in any median, ratio, pair or sign test - here or in whatever pools this artifact. The
    run ids that are kept are written into the artifact as `kept_run_ids` so a downstream
    collector reads the decision instead of re-deriving it;
  * the two orders are counterbalanced inside each pair, deterministically. Odd executions open
    `legacy -> reduced` and even ones open `reduced -> legacy`; the last pair flips relative to
    the pair before it. For the three-pair protocol that means the two kept pairs of every
    execution are one legacy-first and one reduced-first, so a position effect - whatever makes
    the second child of a pair see a different machine than the first - is spread over both
    orders by design and does not have to be averaged away by running many executions.

`execution_schedule` is written from the schedule the driver actually launches, before the
first child starts, and every child record carries its own `run_index`, `mode`,
`position_in_pair` and `execution_order_index`. Nothing downstream has to infer the order from
an execution number.

Three gates decide the report:

  * the copy gate: the reduced order must measure zero conversions at the two CPU sites, one
    6,220,800-byte upload and one 16,588,800-byte download per steady frame, with the legacy
    order measured in the same report as the control that keeps those zeros meaningful;
  * the output integrity gate: ffprobe has to read a complete stream from both orders, and the
    two orders have to encode the same video bitstream;
  * the performance gate: the reduced median FPS over the kept pairs must be at least the
    legacy median over the same pairs. A byte saving that made the end-to-end run slower is a
    regression, not a result.

A fourth gate, the order balance gate, checks the two properties above held: the schedule that
ran is the schedule that was written, the kept pairs cover both positions, and the medians name
the same kept run ids for both orders.

The integrity gate compares the video bitstream, not the file. The first version of this script
hashed the ``.mkv`` and reported NOT_READY for reasons that had nothing to do with the transfer
order: the Matroska muxer writes a random SegmentUID per run, so two runs of the *same* order
hash differently too. The measured facts are recorded in the artifact - 44 bytes differ between
any two of these files, all of them above the first Cluster, and the extracted bitstream hashes
are equal. A container hash cannot show what it was asked to show here, so it is reported as a
diagnostic instead of a gate.

Usage:
    python tests/unit/collect_transfer_ab.py --out perf_results/<sha>/transfer_bench_fix \
        --frames 60 --scale 0.5 --runs 3 --execution-index 1
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_e2e as bench  # noqa: E402
import collect_transfer_ledger as ctl  # noqa: E402

MODES = ("legacy", "reduced")

# What an execution artifact has to say about itself for a pooled verdict to accept it. The
# earlier protocol of this round ran every pair legacy-first and let the pooled file decide on
# all three runs; those artifacts stay readable as history, they are not poolable with this one.
#
# Version 3 is the instrument, not the schedule: the counterbalance, the warmup rule and the
# primary gate are unchanged from version 2, but three of the rules that decide whether an
# environment is admissible were proven wrong and rewritten, so a batch measured under version 2
# was certified by a different collector than a batch measured under this one. Which executions
# may be pooled together is exactly what the version records:
#   * quiet-preflight admissibility -- a verdict now reads only the selected passing window, where
#     version 2 combined every window it had tried, so a version 2 round could report a conflict
#     that its own accepted window never contained, or the reverse;
#   * runtime contamination classification -- the competing-workload classes were replaced, so a
#     version 2 `external_gpu_compute_workload` flag and a version 3 one do not mean the same
#     thing and are not comparable evidence about the same batch;
#   * process-tree identity -- ownership is now decided per 1 Hz sample against that sample's own
#     tree, keyed by (pid, process_create_time), where version 2 compared every GPU event against
#     one flattened tree, which is what turned this round's own transient children into a
#     contamination that never happened.
# So the version 2 bodies of evidence -- the 7-execution contaminated attempt, 7c1768f, 740c78a
# and the fixed-order batches -- are historical evidence only. They stay readable and stay
# reported; `unpoolable_reasons` refuses them a place in a version 3 verdict.
#
# Version 4 rewrites the same two admissibility rules a second time, on evidence this round
# produced for itself, so a version 3 batch and a version 4 batch are again certified by different
# collectors and must not be pooled together:
#   * GPU-client classification -- an outside client used to be a conflict for *appearing* in the
#     driver's table, either newly or with an unreadable identity. That is now refused by four
#     positive criteria (memory the driver accounts for, an occupied codec engine, a name or path or
#     command line that says compute, or a measured per-process share at or above a limit frozen in
#     `benchmark_environment`), each held for §18's three consecutive samples. The rows that satisfy
#     none are recorded as `ambient_or_unproven_gpu_client` and may not stop anything, so a version 3
#     `external_gpu_compute_workload` flag and a version 4 one are claims about different machines;
#   * who a sample thinks it belongs to -- a reading used a process list taken before the GPU table it
#     judged, which turned the batch's own just-spawned FFmpeg into a `repo_conflict`. The reads are
#     now ordered card-then-list and a new child can be attributed by parent chain and creation time.
PROTOCOL_NAME = "counterbalanced-kept-runs-ab"
PROTOCOL_VERSION = 4
# The pairs that warm the machine up. They are measured and published, and no statistic that a
# verdict reads includes them.
WARMUP_PAIR_COUNT = 1
# What the encoder was set to, spelled the way ffprobe spells a codec name. The pipeline's default
# encoder is av1_nvenc, and `benchmark_e2e.ffprobe_validate` reports the stream's `codec_name`, so
# this is the only place the expected name and the observed name share a spelling. The aggregation
# in collect_transfer_status.merge_ffprobe compares against this rather than against a key name it
# guessed at.
EXPECTED_OUTPUT_CODEC = "av1"

# The spans the task names, spelled as the pipeline's own per-frame record spells them.
SPAN_KEYS = ("frame_total", "working_color_prepare_ms", "guidance_total", "nvof_upload_ms",
             "nvof_input_convert_wait_ms", "nvof_postprocess_wait_inclusive_ms",
             "nvof_download_ms", "dlss_process", "residual_pyramid", "pipe_write")
# The counters the gates read, plus the per-site rows that say which call went away rather
# than only that one fewer happened.
COUNTER_KEYS = ("cpu_bgr_to_rgba_calls", "cpu_rgba_to_bgr_calls",
                "cpu_bgr_to_rgba@dlss_input_calls", "cpu_bgr_to_rgba@nvof_input_calls",
                "cpu_rgba_to_bgr@residual_input_calls", "cpu_color_conversion_output_bytes",
                "nvof_h2d_bytes_measured", "nvof_h2d_copy_count", "nvof_d2h_bytes_measured",
                "nvof_d2h_copy_count", "cpu_previous_frame_copy_bytes", "encoder_pipe_bytes",
                "decoder_pipe_bytes", "nvof_bgr_staging_alloc_count_delta",
                "nvof_bgr_input_call_count_delta", "nvof_input_convert_launch_count_delta")


# ---------------------------------------------------------------------------
# the schedule: which order runs first inside each pair
# ---------------------------------------------------------------------------

def pair_order(run_index, execution_index):
    """The two orders of one pair, in the sequence this execution launches them.

    Deterministic, not random: the same (execution index, pair index) always gives the same
    answer, so a re-run reproduces a published schedule and a reader can check one.

    Odd executions open legacy-first, even ones open reduced-first, and the kept pairs then
    alternate from there. With three pairs that is exactly: the warmup and the first kept pair
    follow the execution's parity, the second kept pair is the other way round, so each
    execution's kept pairs contain one legacy-first pair and one reduced-first pair. A position
    effect - whatever makes the second child of a pair see a different machine than the first -
    is therefore split across the two orders by design instead of having to be averaged away.
    """
    if int(run_index) < 1 or int(execution_index) < 1:
        raise ValueError("run and execution indices are 1-based")
    legacy_first = bool(int(execution_index) % 2)
    # Alternate the kept pairs: the (WARMUP_PAIR_COUNT + 2)-th pair onward flips on odd pairs.
    if int(run_index) >= WARMUP_PAIR_COUNT + 2 and int(run_index) % 2 == 1:
        legacy_first = not legacy_first
    return ("legacy", "reduced") if legacy_first else ("reduced", "legacy")


def execution_schedule(runs, execution_index):
    """The full launch plan of one execution, as data, before any child is started."""
    return [{"run": int(run_index),
             "order": list(pair_order(run_index, execution_index)),
             "kept": int(run_index) > WARMUP_PAIR_COUNT}
            for run_index in range(1, int(runs) + 1)]


def kept_run_ids(runs):
    """The pair indices a verdict may read: everything after the warmup pairs."""
    return [run_index for run_index in range(1, int(runs) + 1)
            if run_index > WARMUP_PAIR_COUNT]


def warmup_run_ids(runs):
    """The pair indices kept as diagnostics only."""
    return [run_index for run_index in range(1, int(runs) + 1)
            if run_index <= WARMUP_PAIR_COUNT]


def position_in_pair(run_index, mode, execution_index):
    """1-based position `mode` was launched at inside its pair, or None if it is not in it."""
    order = pair_order(run_index, execution_index)
    return order.index(mode) + 1 if mode in order else None


def order_balance_rows(schedule):
    """How the kept pairs split by position, read back from a schedule that actually ran.

    The counts are taken from the schedule record rather than from the execution number, so a
    pooled verdict is checking what the driver launched and not what it expects to have launched.
    """
    rows = {"legacy_first": 0, "reduced_first": 0, "pairs": []}
    for entry in schedule or []:
        if not entry.get("kept"):
            continue
        order = list(entry.get("order") or [])
        first = order[0] if order else None
        if first in ("legacy", "reduced"):
            rows[f"{first}_first"] += 1
        rows["pairs"].append({"run": entry.get("run"), "order": order})
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True,
                             timeout=30)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _check(what, expected, actual):
    return {"what": what, "expected": expected, "actual": actual, "ok": actual == expected}


# The codec-name -> raw-muxer table used to pull the video bitstream out of the container.
BITSTREAM_MUXERS = {"av1": "ivf", "hevc": "hevc", "h264": "h264", "vp9": "ivf"}
# The first Cluster element, which is where a Matroska file stops being header and starts
# being payload.
CLUSTER_ID = b"\x1f\x43\xb6\x75"


def extract_bitstream(path, codec_name):
    """Hash the video bitstream alone, with the container's per-run identity taken out.

    Matroska writes a fresh random SegmentUID each time it opens a file, so a hash of the
    file describes the muxer as much as it describes the pictures. Demuxing with `-c copy`
    keeps exactly the encoded packets, which is what a bit-identity claim has to be about.
    None means this could not be read, which is reported as unavailable rather than as a
    difference.
    """
    muxer = BITSTREAM_MUXERS.get(str(codec_name))
    if not muxer or not os.path.exists(path):
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    scratch = os.path.join(_ROOT, "bench_out", f"{stem}.bitstream.{muxer}")
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-map", "0:v:0",
                           "-c", "copy", "-f", muxer, scratch],
                          capture_output=True, text=True, timeout=900)
    try:
        if proc.returncode != 0 or not os.path.exists(scratch):
            return None
        return sha256(scratch)
    finally:
        if os.path.exists(scratch):
            os.remove(scratch)


def container_diff(path_a, path_b):
    """Where two files' bytes differ, as a count and a position relative to the payload."""
    try:
        with open(path_a, "rb") as handle:
            a = handle.read()
        with open(path_b, "rb") as handle:
            b = handle.read()
    except OSError as error:
        return {"error": str(error)}
    offsets = []
    for start in range(0, min(len(a), len(b)), 1 << 20):
        chunk_a, chunk_b = a[start:start + (1 << 20)], b[start:start + (1 << 20)]
        offsets.extend(start + i for i in range(len(chunk_a)) if chunk_a[i] != chunk_b[i])
    payload_start = a.find(CLUSTER_ID)
    return {
        "size_a": len(a), "size_b": len(b), "sizes_equal": len(a) == len(b),
        "differing_bytes": len(offsets),
        "first_cluster_offset": payload_start if payload_start >= 0 else None,
        "highest_differing_offset": max(offsets) if offsets else None,
        "all_diffs_in_header": bool(offsets) and payload_start > 0
                               and len(a) == len(b) and max(offsets) < payload_start,
    }


def _median(values, digits=0):
    numbers = [v for v in values if v is not None]
    if not numbers:
        return None
    out = statistics.median(numbers)
    return int(out) if digits == 0 else round(float(out), digits)


def _steady_medians(records, keys):
    """Median of each key over the frames after the first: the reset frame uploads twice."""
    return {key: _median([r[key] for r in records[1:]
                          if key in r and isinstance(r.get(key), (int, float))])
            for key in keys}


# ---------------------------------------------------------------------------
# one run, in a child process
# ---------------------------------------------------------------------------

def run_child(mode, frames, scale, tag, execution_index=1, position=1):
    """One real export in this process: its spans, its steady counters, its output identity."""
    import pipeline

    profiler = pipeline.StageProfiler()
    run = ctl.run_export(frames, scale, transfer_mode=mode, profiler=profiler, tag=tag)
    records = run["records"]
    summary = profiler.summary()
    steady = _steady_medians(records, COUNTER_KEYS)
    steady["nvof_input_format"] = sorted({r.get("nvof_input_format") for r in records[1:]
                                          if r.get("nvof_input_format")}) or None
    out_path = run["out_path"]
    result = {
        "mode": mode,
        # Where this child believes it sits in the schedule; the parent overwrites both from the
        # order it actually launched, so a mismatch cannot survive into the artifact.
        "execution_order_index": int(execution_index),
        "position_in_pair": int(position),
        "frames_recorded": len(records),
        "elapsed_sec": run["elapsed_sec"],
        "fps": run["fps"],
        "pipeline_reported_fps": (round(summary["fps"], 4)
                                  if isinstance(summary.get("fps"), (int, float)) else None),
        "spans_ms": {key: {stat: round(summary[key][stat], 4)
                           for stat in ("mean", "p50", "p95", "min", "max")}
                     for key in SPAN_KEYS if isinstance(summary.get(key), dict)},
        "steady_state_counters": steady,
        "transfer_mode_requested": run["diagnostics"].get("transfer_mode_requested"),
        "transfer_mode_actual": run["diagnostics"].get("transfer_mode_actual"),
        "transfer_mode_fallback_reason": run["diagnostics"].get("transfer_mode_fallback_reason"),
        "residual_backend_actual": ctl.actual_residual_backend(records, run["diagnostics"]),
        "working_dims": [run["live_w"], run["live_h"]],
        "output": {"path": os.path.basename(out_path),
                   "bytes": os.path.getsize(out_path) if os.path.exists(out_path) else None,
                   "sha256": sha256(out_path) if os.path.exists(out_path) else None},
        "fixture": {"path": os.path.relpath(bench.FIXTURE_PATH, _ROOT).replace("\\", "/"),
                    "sha256": sha256(bench.FIXTURE_PATH)
                              if os.path.exists(bench.FIXTURE_PATH) else None},
        "job_settings": run["job_settings"],
    }
    print("RESULT_JSON:" + json.dumps(result), flush=True)
    return 0


def _parse_child_json(stdout):
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return None


def run_child_process(mode, run_index, frames, scale, execution_index, position):
    """One child process = one pair position of one execution."""
    cmd = [sys.executable, os.path.abspath(__file__), "--child", "--mode", mode,
           "--frames", str(frames), "--scale", str(scale),
           "--tag", f"ab-e{execution_index}r{run_index}-{mode}",
           "--execution-index", str(execution_index), "--position", str(position)]
    proc = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=7200)
    result = _parse_child_json(proc.stdout)
    if result is None:
        return None, ((proc.stdout or "")[-1500:] + "\n" + (proc.stderr or "")[-1500:])
    out_path = os.path.join(_ROOT, "bench_out", result["output"]["path"])
    # The three fields that place this run in the schedule. The child echoes what it was told,
    # and the parent - which is the one that decided the order - overwrites it, so the artifact
    # cannot claim a position the launcher did not use.
    result["run"] = run_index
    result["mode"] = mode
    result["position_in_pair"] = position
    result["execution_order_index"] = int(execution_index)
    result["output_path"] = os.path.relpath(out_path, _ROOT).replace("\\", "/")
    probe = bench.ffprobe_validate(out_path, frames, bench.FIXTURE_W, bench.FIXTURE_H)
    result["ffprobe"] = probe
    result["output"]["bitstream_sha256"] = extract_bitstream(out_path, probe.get("codec"))
    return result, None


# ---------------------------------------------------------------------------
# aggregation and gates
# ---------------------------------------------------------------------------

def summarise_mode(runs, kept_ids):
    """Every run of one order, then the medians of the kept runs only.

    `kept_ids` is the protocol's decision, made before the data (`kept_run_ids`), not something
    inferred from which runs look like outliers afterwards. Every run stays in `runs`, and each
    one carries `included_in_medians`, so the warmup figures are published and can be read, but
    no statistic here is computed from them.
    """
    kept_ids = [int(r) for r in kept_ids]
    kept = [run for run in runs if int(run["run"]) in kept_ids]
    spans = {}
    for key in SPAN_KEYS:
        for stat in ("p50", "p95"):
            spans[f"{key}_{stat}_ms"] = _median(
                [run["spans_ms"].get(key, {}).get(stat) for run in kept
                 if key in run.get("spans_ms", {})], 4)
    counters = {key: _median([run["steady_state_counters"].get(key) for run in kept])
                for key in COUNTER_KEYS}
    for run in runs:
        run["included_in_medians"] = int(run["run"]) in kept_ids
    return {
        "runs": runs,
        "runs_used_for_medians": [run["run"] for run in kept],
        "warmup_run_ids": sorted({int(run["run"]) for run in runs} - set(kept_ids)),
        "median_fps": _median([run["fps"] for run in kept], 4),
        "fps_of_every_run": {run["run"]: run["fps"] for run in runs},
        "fps_of_kept_runs": [run["fps"] for run in kept],
        "spans_ms": spans,
        "steady_state_counters": counters,
        "negotiated_modes": sorted({run["transfer_mode_actual"] for run in runs}),
        "fallback_reasons": sorted({str(run["transfer_mode_fallback_reason"]) for run in runs}),
    }


def _formats(report):
    """Every input format every run of one order reported, de-duplicated and sorted.

    Each run reports a list because a sequence could in principle change format mid-stream;
    the union across runs is what the gate has to be about.
    """
    return sorted({fmt for run in report["runs"]
                   for fmt in run["steady_state_counters"].get("nvof_input_format") or []})


def copy_gate(legacy, reduced, working_dims):
    """The measured values the reduced order has to show, checked against the run itself."""
    working_w, working_h = working_dims
    payload = working_w * working_h
    r, l = reduced["steady_state_counters"], legacy["steady_state_counters"]
    return [
        _check("transfer_mode_actual_is_what_was_asked", ["reduced"], reduced["negotiated_modes"]),
        _check("legacy_requested_legacy_and_got_it", ["legacy"], legacy["negotiated_modes"]),
        _check("cpu_bgr_to_rgba_calls", 0, r["cpu_bgr_to_rgba_calls"]),
        _check("cpu_rgba_to_bgr_calls", 0, r["cpu_rgba_to_bgr_calls"]),
        _check("cpu_bgr_to_rgba_at_dlss_input", 0, r["cpu_bgr_to_rgba@dlss_input_calls"]),
        _check("cpu_bgr_to_rgba_at_nvof_input", 0, r["cpu_bgr_to_rgba@nvof_input_calls"]),
        _check("cpu_rgba_to_bgr_at_residual_input", 0,
               r["cpu_rgba_to_bgr@residual_input_calls"]),
        _check("nvof_h2d_bytes_per_steady_frame", payload * 3, r["nvof_h2d_bytes_measured"]),
        _check("nvof_h2d_copy_count_per_steady_frame", 1, r["nvof_h2d_copy_count"]),
        _check("nvof_d2h_bytes_per_steady_frame", payload * 8, r["nvof_d2h_bytes_measured"]),
        _check("nvof_input_format", ["bgr24_host"], _formats(reduced)),
        _check("cpu_previous_frame_copy_bytes", 0, r["cpu_previous_frame_copy_bytes"]),
        _check("nvof_bgr_staging_allocations_per_frame", 0,
               r["nvof_bgr_staging_alloc_count_delta"]),
        _check("nvof_bgr_input_calls_per_frame", 1, r["nvof_bgr_input_call_count_delta"]),
        _check("nvof_input_convert_launches_per_frame", 1,
               r["nvof_input_convert_launch_count_delta"]),
        # The control: the same counters move on the legacy order, so the zeros above read as
        # an absence rather than as an instrument that stopped reporting.
        _check("legacy_control_cpu_bgr_to_rgba_calls", 2, l["cpu_bgr_to_rgba_calls"]),
        _check("legacy_control_cpu_rgba_to_bgr_calls", 1, l["cpu_rgba_to_bgr_calls"]),
        _check("legacy_control_nvof_h2d_bytes", payload * 4, l["nvof_h2d_bytes_measured"]),
        _check("legacy_control_nvof_h2d_copy_count", 1, l["nvof_h2d_copy_count"]),
        _check("legacy_control_nvof_d2h_bytes", payload * 8, l["nvof_d2h_bytes_measured"]),
        _check("legacy_control_nvof_input_format", ["rgba8_host"], _formats(legacy)),
    ]


def output_integrity_gate(legacy, reduced, frames):
    """ffprobe on both orders, and the two orders' encoded bitstreams compared."""
    checks = []
    for mode, report in (("legacy", legacy), ("reduced", reduced)):
        for run in report["runs"]:
            checks.append(_check(f"{mode} run {run['run']} ffprobe reports a complete stream",
                                 True, bool((run.get("ffprobe") or {}).get("ok"))))
    by_index = {run["run"]: run for run in reduced["runs"]}
    for run in legacy["runs"]:
        other = by_index.get(run["run"])
        checks.append(_check(
            f"run {run['run']} video bitstream bit-identical between orders",
            run["output"].get("bitstream_sha256"),
            (other or {}).get("output", {}).get("bitstream_sha256")))
    every = [run["output"].get("bitstream_sha256")
             for report in (legacy, reduced) for run in report["runs"]]
    checks.append(_check("one_bitstream_hash_across_every_run_of_both_orders", True,
                         len(set(every)) == 1 and None not in every))
    return checks


def container_hash_explanation(legacy, reduced):
    """Why the file hashes differ while the bitstreams do not, measured rather than asserted."""
    reports = {"legacy": legacy, "reduced": reduced}
    pairs = []

    def add(label, one, other):
        if one is None or other is None:
            return
        pairs.append({
            "compares": label,
            "files": [one["output"]["path"], other["output"]["path"]],
            "container_sha256s": [one["output"]["sha256"], other["output"]["sha256"]],
            "bitstream_sha256s": [one["output"].get("bitstream_sha256"),
                                  other["output"].get("bitstream_sha256")],
            "diff": container_diff(os.path.join(_ROOT, one.get("output_path", "")),
                                   os.path.join(_ROOT, other.get("output_path", ""))),
        })

    by_index = {run["run"]: run for run in reduced["runs"]}
    for run in legacy["runs"]:
        add(f"legacy run {run['run']} vs reduced run {run['run']}", run,
            by_index.get(run["run"]))
    # The control that made the container hash read as a failure: two runs of one order, which
    # differ the same way and by the same amount as the two orders differ from each other.
    for mode, report in reports.items():
        runs = report["runs"]
        if len(runs) > 1:
            add(f"{mode} run {runs[0]['run']} vs {mode} run {runs[1]['run']}", runs[0], runs[1])
    return {
        "container_sha256_of_each_run": {
            mode: {run["run"]: run["output"]["sha256"] for run in report["runs"]}
            for mode, report in reports.items()},
        "bitstream_sha256_of_each_run": {
            mode: {run["run"]: run["output"].get("bitstream_sha256") for run in report["runs"]}
            for mode, report in reports.items()},
        "pairs": pairs,
        "note": "Matroska writes a random SegmentUID and a per-run timestamp into every file, "
                "so no two exports of this pipeline hash alike, two runs of the same order "
                "included. The numbers above are measured: the count of differing bytes and "
                "their position relative to the first Cluster, which is where the payload "
                "starts. A container hash is therefore reported here as a diagnostic and the "
                "gate compares the bitstream.",
    }


def performance_gate(legacy, reduced):
    l, r = legacy["median_fps"], reduced["median_fps"]
    ratio = round(r / l, 4) if l and r else None
    return [_check("reduced_median_fps_at_least_legacy", True, bool(l and r and r >= l))], ratio


def order_balance_gate(schedule, legacy, reduced, kept_ids):
    """The schedule that ran is the schedule the artifact claims, and it is balanced.

    This gate does not correct a position effect statistically. It checks that the benchmark was
    built so the two orders each took both positions the same number of times, and it fails the
    execution when they did not, because a comparison whose second position always belongs to one
    order cannot be read either way.
    """
    kept_ids = [int(r) for r in kept_ids]
    ran = sorted({int(run["run"]) for report in (legacy, reduced) for run in report["runs"]})
    checks = [_check("schedule_covers_every_pair_that_ran", ran,
                     sorted(int(entry["run"]) for entry in schedule or []))]
    by_run = {int(entry["run"]): list(entry.get("order") or []) for entry in schedule or []}
    misplaced = []
    for mode, report in (("legacy", legacy), ("reduced", reduced)):
        for run in report["runs"]:
            order = by_run.get(int(run["run"]))
            expected = (order.index(mode) + 1) if order and mode in order else None
            if expected != run.get("position_in_pair"):
                misplaced.append({"mode": mode, "run": run["run"],
                                  "recorded": run.get("position_in_pair"),
                                  "schedule": expected})
    checks.append(_check("every_run_recorded_the_position_it_launched_at", [], misplaced))
    balance = order_balance_rows([{"run": run, "order": order, "kept": run in kept_ids}
                                  for run, order in sorted(by_run.items())])
    checks.append(_check("kept_pairs_legacy_first_equals_reduced_first",
                         True, balance["legacy_first"] == balance["reduced_first"]
                               and balance["legacy_first"] > 0))
    checks.append(_check("kept_run_ids_match_between_modes",
                         sorted(legacy["runs_used_for_medians"]),
                         sorted(reduced["runs_used_for_medians"])))
    checks.append(_check("warmup_runs_excluded_from_medians", [], [
        {"mode": mode, "run": run} for mode, report in (("legacy", legacy), ("reduced", reduced))
        for run in report["runs_used_for_medians"] if int(run) not in kept_ids]))
    return checks


def savings(legacy, reduced):
    """Where the bytes went, from the medians the two orders reported for themselves."""
    out = {}
    for key, label in (("nvof_h2d_bytes_measured", "nvof_h2d_upload"),
                       ("cpu_color_conversion_output_bytes", "cpu_color_conversion_output"),
                       ("cpu_bgr_to_rgba_calls", "cpu_bgr_to_rgba_calls_per_frame"),
                       ("cpu_rgba_to_bgr_calls", "cpu_rgba_to_bgr_calls_per_frame")):
        before = legacy["steady_state_counters"].get(key)
        after = reduced["steady_state_counters"].get(key)
        if before is None or after is None:
            out[label] = {"legacy": before, "reduced": after, "saved": None,
                          "saved_percent": None}
            continue
        saved = before - after
        out[label] = {"legacy": before, "reduced": after, "saved": saved,
                      "saved_percent": round(100.0 * saved / before, 2) if before else None}
    for key, label in (("nvof_d2h_bytes_measured", "nvof_d2h_download_unchanged"),
                       ("encoder_pipe_bytes", "encoder_pipe_unchanged"),
                       ("decoder_pipe_bytes", "decoder_pipe_unchanged")):
        out[label] = {"legacy": legacy["steady_state_counters"].get(key),
                      "reduced": reduced["steady_state_counters"].get(key)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="", help="artifact directory")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", default="reduced", choices=MODES)
    parser.add_argument("--tag", default="ab")
    parser.add_argument("--execution-index", type=int, default=1,
                        help="which execution of a repeated protocol this is; its parity picks "
                             "which order opens each pair")
    parser.add_argument("--position", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.execution_index < 1:
        parser.error("--execution-index is 1-based")

    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if args.child:
        return run_child(args.mode, args.frames, args.scale, args.tag,
                         args.execution_index, args.position)
    if not args.out:
        parser.error("--out is required unless running a child")

    bench.ensure_fixture()
    kept_ids = kept_run_ids(args.runs)
    # Recorded before the first child is launched, and launched from: what the artifact calls
    # `execution_schedule` is the plan the driver used, not a description written afterwards.
    schedule = execution_schedule(args.runs, args.execution_index)
    runs = {mode: [] for mode in MODES}
    launch_sequence = []
    failures = []
    for entry in schedule:
        for position, mode in enumerate(entry["order"], start=1):
            launch_sequence.append({"run": entry["run"], "mode": mode,
                                    "position_in_pair": position, "kept": entry["kept"]})
            result, error = run_child_process(mode, entry["run"], args.frames, args.scale,
                                              args.execution_index, position)
            if result is None:
                failures.append(f"{mode} run {entry['run']} did not finish: {error}")
            runs[mode].append(result)

    payload = {
        "generated_by": "tests/unit/collect_transfer_ab.py",
        "code_under_test": git("rev-parse", "HEAD"),
        "commit_subject": git("log", "-1", "--pretty=%s"),
        **ctl.tree_state(),
        "protocol": {"name": PROTOCOL_NAME, "version": PROTOCOL_VERSION},
        "execution_order_index": args.execution_index,
        "execution_schedule": schedule,
        "run_launch_sequence": launch_sequence,
        "configuration": {
            "fixture": {"path": os.path.relpath(bench.FIXTURE_PATH, _ROOT).replace("\\", "/"),
                        "sha256": sha256(bench.FIXTURE_PATH)
                                  if os.path.exists(bench.FIXTURE_PATH) else None,
                        "width": bench.FIXTURE_W, "height": bench.FIXTURE_H,
                        "fps": bench.FIXTURE_FPS},
            "frames_per_run": args.frames,
            "runs_per_mode": args.runs,
            "warmup_run_ids": warmup_run_ids(args.runs),
            "kept_run_ids": kept_ids,
            "runs_dropped_before_medians": len(warmup_run_ids(args.runs)),
            "working_scale": args.scale,
            "flow_engine": bench.FLOW_ENGINE,
            "flow_preset": bench.FLOW_PRESET,
            "encoder_cq": bench.CQ,
            "output_resolution": [bench.FIXTURE_W, bench.FIXTURE_H],
            "identical_across_orders": "fixture, frame count, working scale, NVOF grid and "
                                       "preset, DLSS options, depth convention, encoder, CQ "
                                       "and output resolution; only transfer_mode differs",
            "runs_interleaved": True,
            "schedule_counterbalanced": True,
            "schedule_rule": "odd executions launch legacy first, even executions reduced "
                             "first, and the kept pairs alternate, so the kept pairs of one "
                             "execution contain one pair of each position",
        },
        "span_keys": list(SPAN_KEYS),
        "counter_keys": list(COUNTER_KEYS),
        "measurement_rules": {
            "fps": "frames recorded divided by the wall clock of export_dlss_stream() alone, "
                   "the same window tests/unit/benchmark_e2e.py times, so the figures can be "
                   "read against the recorded 5.164 and 7.247 FPS baselines. Session creation "
                   "happens before the window opens and is not in either number.",
            "spans": "p50 and p95 of every frame record of the kept runs, in ms",
            "counters": "median over the frames after the first of each kept run, then median "
                        "across those runs; the reset frame uploads twice and is excluded",
            "warmup": "the warmup pairs are run and published with included_in_medians=false; "
                      "no median, ratio, pair or sign test here or downstream reads them",
            "unavailable": "None means the run did not measure it; 0 means it measured zero",
        },
        "child_failures": failures,
    }

    if not runs_complete(runs):
        payload["status"] = "NOT_READY"
        payload["incomplete_reason"] = "one or more child runs produced no result"
        payload["raw_runs"] = {mode: [r for r in runs[mode]] for mode in MODES}
    else:
        reports = {mode: summarise_mode(runs[mode], kept_ids) for mode in MODES}
        working_dims = reports["reduced"]["runs"][0]["working_dims"]
        checks = {
            "copy_gate": copy_gate(reports["legacy"], reports["reduced"], working_dims),
            "output_integrity_gate": output_integrity_gate(reports["legacy"],
                                                           reports["reduced"], args.frames),
            "order_balance_gate": order_balance_gate(schedule, reports["legacy"],
                                                     reports["reduced"], kept_ids),
        }
        perf_checks, ratio = performance_gate(reports["legacy"], reports["reduced"])
        checks["performance_gate"] = perf_checks
        payload["modes"] = reports
        payload["gates"] = checks
        payload["output_identity"] = container_hash_explanation(reports["legacy"],
                                                                reports["reduced"])
        payload["order_balance"] = order_balance_rows(schedule)
        payload["speedup_ratio_reduced_over_legacy"] = ratio
        payload["fps"] = {mode: reports[mode]["median_fps"] for mode in MODES}
        payload["bytes_saved_per_steady_frame"] = savings(reports["legacy"], reports["reduced"])
        payload["status"] = ("DONE_PROVEN"
                             if all(c["ok"] for group in checks.values() for c in group)
                             and not failures else "NOT_READY")

    os.makedirs(args.out, exist_ok=True)
    ab_path = os.path.join(args.out, "transfer_ab_benchmark.json")
    with open(ab_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    ffprobe_path = os.path.join(args.out, "ffprobe_output.json")
    with open(ffprobe_path, "w", encoding="utf-8") as handle:
        json.dump(ffprobe_artifact(payload, args.frames), handle, indent=2, sort_keys=True)

    failed = [c for name, group in payload.get("gates", {}).items() for c in group
              if not c["ok"]]
    print(json.dumps({"ok": payload["status"] == "DONE_PROVEN", "status": payload["status"],
                      "artifacts": [ab_path, ffprobe_path], "fps": payload.get("fps"),
                      "schedule": payload.get("order_balance"),
                      "speedup_ratio": payload.get("speedup_ratio_reduced_over_legacy"),
                      "failed_checks": failed, "child_failures": failures},
                     indent=2, sort_keys=True, default=str))
    return 0 if payload["status"] == "DONE_PROVEN" else 1


def runs_complete(runs):
    return all(all(run is not None for run in runs[mode]) for mode in MODES)


def ffprobe_artifact(payload, frames):
    """The probe readings of every file both orders wrote, in a file of its own."""
    reports = payload.get("modes") or {}
    runs = {mode: [{"run": run["run"], "output": run["output"], "ffprobe": run.get("ffprobe"),
                    "position_in_pair": run.get("position_in_pair"),
                    "kept": run.get("included_in_medians")}
                   for run in reports[mode]["runs"]] for mode in reports}
    return {
        "generated_by": "tests/unit/collect_transfer_ab.py",
        "code_under_test": payload["code_under_test"],
        "execution_schedule": payload.get("execution_schedule"),
        "expected": {"frames": frames, "width": bench.FIXTURE_W, "height": bench.FIXTURE_H,
                     "fps": bench.FIXTURE_FPS, "codec": EXPECTED_OUTPUT_CODEC},
        "runs": runs,
        "gate": (payload.get("gates") or {}).get("output_integrity_gate"),
        "status": payload["status"],
        "note": "the encoder ran with the same codec and CQ for both orders; the packet "
                "counts, durations, resolutions, frame rates and codecs below are ffprobe's "
                "own reading of the file each run wrote, and the hashes are of those files",
    }


if __name__ == "__main__":
    sys.exit(main())
