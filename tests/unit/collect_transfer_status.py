#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the round's status file out of the artifacts, not out of the prose.

Every key §52 names is decided here by reading the file that carries its evidence: the
ledgers, the parity run, the ABI and boundary audits, the A/B gates, the test transcripts.
A key that finds no evidence, or finds evidence that contradicts what it points at, becomes
NOT_READY rather than a judgement call. The two SHA checks are the same kind of reading -
which commit each artifact says it describes, and whether the native binaries in the working
tree are the ones the commit holds.

Nothing here runs the pipeline, so re-running this on a directory of artifacts reproduces the
same file apart from the timestamp.

Two rounds are assembled by this file. `--round phase1` (the default) reads the transfer ledgers
and the A/B artifacts of the round that reduced the transfers. `--round bench_fix` reads the
counterbalanced benchmark artifacts of the round that fixed the instrument, and writes the derived
files it owns - status_consistency.json, ffprobe_output.json, environment.json - before deciding
its own keys.

Usage:
    python tests/unit/collect_transfer_status.py --out perf_results/<sha>/transfer_phase1
    python tests/unit/collect_transfer_status.py --out perf_results/<sha>/transfer_bench_fix \
        --round bench_fix
"""

import argparse
import datetime
import hashlib
import json
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

STATUS_VALUES = ("DONE_PROVEN", "REDUCED_PROVEN", "BLOCKED_PROVEN", "NOT_READY")
VERDICT_VALUES = ("ELIMINATED_PROVEN", "REDUCED_PROVEN", "BLOCKED_PROVEN")

# The binaries this round's claims run on, and the source each one is built from.
NATIVE_PAIRS = (("native/nvof_bridge.dll", "native/nvof_bridge.cpp"),
                ("native/residual_bridge.dll", "native/residual_bridge.cpp"))
# The commits the phase-1 round made in order, and the native source each one touched. Recorded
# as a constant rather than read from git because that round is over: `git log` since then would
# describe this one.
ROUND_COMMITS = ("20fc47c", "62f5a11", "51c980c", "1352c71", "981f423", "f5a18c6")
# The benchmark-protocol round starts from the baseline its brief names, and reads its own commits
# from git so the list cannot go stale the way the one above would have.
BENCH_FIX_BASELINE = "8fa242b"
# The steady-state counters both orders of an execution measure, which is everything this round
# can put a before-and-after number on without re-running the ledgers.
CLAIM_COUNTERS = ("nvof_h2d_bytes_measured", "nvof_d2h_bytes_measured", "cpu_bgr_to_rgba_calls",
                  "cpu_rgba_to_bgr_calls", "cpu_color_conversion_output_bytes",
                  "cpu_previous_frame_copy_bytes", "decoder_pipe_bytes", "encoder_pipe_bytes")


def _ab_module():
    """Import the execution driver only when a check needs its schedule rule.

    It pulls in the pipeline's benchmarking harness, and this file is otherwise a pure reader of
    JSON; keeping the import inside the two checks that need it keeps the collector cheap.
    """
    import collect_transfer_ab as ab
    return ab


def _environment_module():
    """The lock and preflight module, imported only to recompute what its artifact published.

    The stability round's environment gates do not take the artifact's word for anything: they
    re-judge its samples against the thresholds the module still holds as constants. That needs the
    module, and it is the only reason this file knows it exists.
    """
    import benchmark_environment as be
    return be


def git_blob_sha256(rev, rel):
    """The SHA-256 of a file as one particular commit holds it, straight out of the object store."""
    try:
        out = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=_ROOT, capture_output=True,
                             timeout=300)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return hashlib.sha256(out.stdout).hexdigest()



def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True,
                             timeout=60)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def sha256(path):
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


def _entry(status, evidence, detail=None):
    assert status in STATUS_VALUES, status
    return {"status": status, "evidence": evidence, "detail": detail or {}}


def _ok(gate_group):
    return bool(gate_group) and all(check.get("ok") for check in gate_group)


# ---------------------------------------------------------------------------
# transfer claims: the status word a pair of measured numbers has to earn
# ---------------------------------------------------------------------------
#
# §34 allows four words, and the difference between two of them - REDUCED_PROVEN and
# BLOCKED_PROVEN - is entirely a statement about numbers: one says a measured quantity came down,
# the other says it did not move because something prevents it. The earlier version of this file
# took the decoder row's word from whatever the boundary audit had concluded about the decoder and
# never compared the two byte counts beside it, so a row whose legacy and reduced measurements were
# both 12,441,600 came out as REDUCED_PROVEN. Everything below exists so that cannot be written
# again: a status about a transfer is derived from its before and after, and the derivation is
# then checked a second time by an independent reader of the same two numbers.

def derive_transfer_status(before, after, legacy_control=None):
    """The one word these measurements support, or NOT_READY when none of them fits."""
    if before is None or after is None:
        return "NOT_READY"
    if after < 0 or before < 0:
        return "NOT_READY"
    if after == 0:
        # A zero is only a deletion if the same instrument reported a non-zero on the other
        # order; without that control it is an instrument that stopped reporting.
        return "DONE_PROVEN" if legacy_control else "NOT_READY"
    if after < before:
        return "REDUCED_PROVEN"
    if after == before:
        return "BLOCKED_PROVEN"
    return "NOT_READY"


def transfer_claim(name, before, after, unit="bytes/frame", evidence=(), note=None):
    """One measured comparison, its derived word, and the arithmetic a reader can check."""
    legacy_control = before if (before or 0) > 0 else None
    status = derive_transfer_status(before, after, legacy_control)
    saved = None if (before is None or after is None) else before - after
    return {
        "claim": name,
        "kind": "elimination" if after == 0 and before else (
            "reduction" if status == "REDUCED_PROVEN" else
            "unchanged" if status == "BLOCKED_PROVEN" else "unproven"),
        "unit": unit,
        "before": before,
        "after": after,
        "legacy_control": legacy_control,
        "saved": saved,
        "saved_percent": (round(100.0 * saved / before, 2)
                          if saved is not None and before else None),
        "status": status,
        "unchanged_this_round": bool(before is not None and after is not None
                                     and before == after),
        "evidence": list(evidence),
        "note": note,
    }


def check_transfer_claims(claims, asserted=None):
    """Re-read every claim's numbers and say which status words they do not support.

    `asserted` maps a claim name to the word the round actually published. Where it is omitted the
    claim carries the word its own derivation produced, which makes this a check of the arithmetic
    rather than of the report; passing the published words in makes it a check of the report.
    """
    asserted = asserted or {}
    rows = []
    for claim in claims:
        name = claim.get("claim")
        before, after = claim.get("before"), claim.get("after")
        published = asserted.get(name, claim.get("status"))
        problems = []
        if published not in STATUS_VALUES + VERDICT_VALUES:
            problems.append(f"{published} is not one of the allowed verdict words")
        if before is None or after is None:
            if published != "NOT_READY":
                problems.append("a claim with an unmeasured side cannot carry a verdict word")
        elif published == "REDUCED_PROVEN" and not after < before:
            problems.append(f"REDUCED_PROVEN but after ({after}) is not below before ({before})")
        elif published in ("ELIMINATED_PROVEN", "DONE_PROVEN") and after != 0:
            problems.append(f"{published} claims a deletion but the measured after is {after}")
        elif published == "BLOCKED_PROVEN" and after < before:
            problems.append(f"blocked, yet the measurement moved from {before} to {after}")
        rows.append({"claim": name, "published_status": published,
                     "derived_status": claim.get("status"), "before": before, "after": after,
                     "saved": claim.get("saved"),
                     "saved_percent": claim.get("saved_percent"),
                     "consistent": not problems, "problems": problems})
    return rows


def _check_value(sequences, name):
    """One named check from each parity sequence, as a list of what it measured."""
    out = []
    for seq in sequences or []:
        found = [c for c in seq.get("checks") or [] if c.get("what") == name]
        out.append({seq.get("label"): found[0].get("actual") if found else None})
    return out


def _summary_line(text):
    """pytest's own closing line, which is the only count this file is allowed to quote."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if re.search(r"\b\d+ (passed|failed|error|no tests ran)\b", line) or "seconds" in line:
            return line
    return ""


def _counts(line):
    numbers = {}
    for count, word in re.findall(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)",
                                  line):
        numbers[word.rstrip("s") if word != "errors" else "error"] = int(count)
    return numbers


def _transcript(directory, name):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    return _summary_line(text), text


def native_binaries():
    """What the tested binaries are, and whether they are the ones the commit holds.

    The rule this satisfies is that a round may not change a bridge's source and keep
    measuring an older build of it. `git hash-object` on the working file and `git rev-parse`
    on the committed path answer that directly, without anyone's word for it.
    """
    out = {}
    for dll_rel, cpp_rel in NATIVE_PAIRS:
        dll = os.path.join(_ROOT, dll_rel)
        cpp = os.path.join(_ROOT, cpp_rel)
        working_dll = git("hash-object", dll_rel) if os.path.exists(dll) else ""
        committed_dll = git("rev-parse", f"HEAD:{dll_rel}")
        working_cpp = git("hash-object", cpp_rel) if os.path.exists(cpp) else ""
        committed_cpp = git("rev-parse", f"HEAD:{cpp_rel}")
        out[os.path.basename(dll_rel)] = {
            "binary_path": dll_rel,
            "binary_sha256": sha256(dll) if os.path.exists(dll) else None,
            "binary_size_bytes": os.path.getsize(dll) if os.path.exists(dll) else None,
            "binary_git_blob_working": working_dll or None,
            "binary_git_blob_committed": committed_dll or None,
            "binary_matches_commit": bool(working_dll) and working_dll == committed_dll,
            "source_path": cpp_rel,
            "source_sha256": sha256(cpp) if os.path.exists(cpp) else None,
            "source_git_blob_working": working_cpp or None,
            "source_git_blob_committed": committed_cpp or None,
            "source_matches_commit": bool(working_cpp) and working_cpp == committed_cpp,
            "last_commit_touching_source": git("log", "-1", "--pretty=%h %s", "--", cpp_rel),
            "last_commit_touching_binary": git("log", "-1", "--pretty=%h %s", "--", dll_rel),
        }
    return out


def commits_this_round(shorts=None, baseline=None):
    """The §49 order, with the native source each commit touched (the §50 question).

    `shorts` is a recorded list, for a round whose commits are already fixed. With none given the
    list is read from git between the brief's baseline and HEAD, because a hardcoded list of the
    previous round's commits silently describes the wrong round.
    """
    if shorts is None:
        shorts = [line.split("\t")[0] for line in
                  git("log", "--reverse", "--pretty=%h%x09%s", f"{baseline}..HEAD").splitlines()
                  if line] if baseline else []
    rows = []
    for short in shorts:
        subject = git("log", "-1", "--pretty=%s", short)
        touched = git("show", "--pretty=", "--name-only", short).splitlines()
        native = [line for line in touched if line.startswith("native/")
                  and (line.endswith(".cpp") or line.endswith(".h"))]
        built = [line for line in touched if line.endswith(".dll")]
        rows.append({"sha": short, "subject": subject or None,
                     "native_source_touched": native,
                     "native_binary_in_same_commit": built})
    head = git("rev-parse", "--short", "HEAD")
    if head and head not in shorts:
        touched = git("show", "--pretty=", "--name-only", "HEAD").splitlines()
        native = [line for line in touched if line.startswith("native/")
                  and (line.endswith(".cpp") or line.endswith(".h"))]
        rows.append({"sha": head, "subject": git("log", "-1", "--pretty=%s"),
                     "native_source_touched": native,
                     "native_binary_in_same_commit":
                         [line for line in touched if line.endswith(".dll")]})
    return rows



def _sha_of(doc):
    """The commit an artifact says it describes, under either spelling of the field.

    environment.json calls it `code_under_test_sha`; every other artifact calls it
    `code_under_test`. Reading either one is a lookup and carries no judgement - the judgement
    lives in _shared_sha, which is the thing that decides whether the set of lookups agrees.
    """
    if not doc:
        return None
    return doc.get("code_under_test") or doc.get("code_under_test_sha")


def _shared_sha(artifacts):
    """The one commit every artifact says it describes, or None when any of them says nothing.

    The previous version built a set after dropping the empty entries, so a required artifact
    that never declared a SHA could not break the agreement. What it actually computed was
    "every artifact that bothers to carry a SHA says the same thing", which reads very much like
    "every required artifact declares this round's commit" and is not that sentence: two of this
    round's artifacts carried no SHA at all and the check still returned a SHA. A missing
    declaration is now treated the way a contradictory one is, because for a reader of the status
    file the two failures are indistinguishable - either way the artifact does not tell them
    which code it describes.
    """
    shas = list(artifacts)
    if not shas or any(not sha for sha in shas):
        return None
    if len(set(shas)) != 1:
        return None
    return shas[0]


def artifact_sha_coverage(named, code_under_test=None):
    """§37: how many of the required artifacts are here, and how many actually name the code.

    Published so a reader never has to trust the status word to know whether the evidence was
    complete: `required` minus `matching` is the count of artifacts whose numbers were used for a
    verdict without saying what code produced them.
    """
    missing = sorted(name for name, doc in named.items() if doc is None)
    silent = sorted(name for name, doc in named.items()
                    if doc is not None and not _sha_of(doc))
    mismatched = sorted(name for name, doc in named.items()
                        if doc is not None and _sha_of(doc) and code_under_test
                        and _sha_of(doc) != code_under_test)
    return {
        "required": len(named),
        "present": len(named) - len(missing),
        "declares_a_sha": len(named) - len(missing) - len(silent),
        "matching": len(named) - len(missing) - len(silent) - len(mismatched),
        "missing": missing,
        "without_a_sha": silent,
        "mismatched": mismatched,
        "declared_shas": {name: _sha_of(doc) for name, doc in sorted(named.items())},
    }


def build(out_dir):
    ledgers = {name: _load(out_dir, f"{name}.json") for name in
               ("transfer_ledger", "transfer_ledger_legacy", "transfer_ledger_auto")}
    parity = _load(out_dir, "nvof_bgr_parity.json")
    abi = _load(out_dir, "dlss_host_abi.json")
    boundaries = _load(out_dir, "process_boundaries.json")
    ab = _load(out_dir, "transfer_ab_benchmark.json")
    ffprobe = _load(out_dir, "ffprobe_output.json")
    executions = _ab_executions(out_dir, ab, ("transfer_ab_benchmark_60f_e2.json",
                                              "transfer_ab_benchmark_150f.json"))
    perf = _load(out_dir, "transfer_performance_pooled.json")
    env = _load(out_dir, "environment.json")

    main_pytest, main_pytest_text = _transcript(out_dir, "pytest_output.txt")
    native_pytest, _ = _transcript(out_dir, "pytest_native_numeric.txt")
    preview_pytest, _ = _transcript(out_dir, "pytest_preview_export.txt")

    names = {"transfer_ledger": ledgers["transfer_ledger"],
             "transfer_ledger_legacy": ledgers["transfer_ledger_legacy"],
             "transfer_ledger_auto": ledgers["transfer_ledger_auto"],
             "nvof_bgr_parity": parity, "dlss_host_abi": abi,
             "process_boundaries": boundaries, "transfer_ab_benchmark": ab,
             "ffprobe_output": ffprobe, "environment": env}
    present = {name: doc for name, doc in names.items() if doc is not None}
    # environment.json names the commit `code_under_test_sha` rather than `code_under_test`;
    # reading it under the wrong key would let a stale environment record agree silently.
    sha = _shared_sha([_sha_of(doc) for doc in present.values()])

    red = (ledgers["transfer_ledger"] or {}).get("measured_steady_state") or {}
    leg = (ledgers["transfer_ledger_legacy"] or {}).get("measured_steady_state") or {}
    auto = (ledgers["transfer_ledger_auto"] or {}).get("measured_steady_state") or {}
    gates = (ab or {}).get("gates") or {}
    modes = (ab or {}).get("modes") or {}
    counters = {mode: (modes.get(mode) or {}).get("steady_state_counters") or {}
                for mode in ("legacy", "reduced")}

    counts = _counts(main_pytest or "")
    suite_clean = bool(main_pytest) and not counts.get("failed") and not counts.get("error") \
        and "1 failed" not in main_pytest and "error" not in main_pytest

    statuses = {}

    # 1. the ledger itself
    ledger_ok = (red and sha is not None
                 and (ledgers["transfer_ledger"] or {}).get("transfer_mode_actual") == "reduced"
                 and red.get("nvof_h2d_bytes_measured") is not None
                 and (ledgers["transfer_ledger"] or {}).get("per_frame_records"))
    statuses["transfer_ledger"] = _entry(
        "DONE_PROVEN" if ledger_ok else "NOT_READY",
        ["transfer_ledger.json", "transfer_ledger_legacy.json", "transfer_ledger_auto.json"],
        {"measured_fields": len(red), "frames": (ledgers["transfer_ledger"] or {}).get("frames_run"),
         "record_digest_sha256": (ledgers["transfer_ledger"] or {}).get("record_digest_sha256"),
         "orders_recorded": sorted(k for k, v in ledgers.items() if v)})

    # 2. fused working preparation: the RGBA the DLSS host is handed no longer comes from a
    #    cv2 conversion at that site, measured on the frame.
    fused = red.get("cpu_bgr_to_rgba@dlss_input_calls") == 0 \
        and leg.get("cpu_bgr_to_rgba@dlss_input_calls", 0) > 0
    statuses["fused_working_prepare"] = _entry(
        "DONE_PROVEN" if fused else "NOT_READY",
        ["transfer_ledger.json measured_steady_state.cpu_bgr_to_rgba@dlss_input_calls",
         "transfer_ledger_legacy.json same field as the control",
         "pytest_output.txt"],
        {"reduced_calls_at_dlss_input_site": red.get("cpu_bgr_to_rgba@dlss_input_calls"),
         "legacy_calls_at_dlss_input_site": leg.get("cpu_bgr_to_rgba@dlss_input_calls")})

    # 3. residual reading the DLSS RGBA directly
    direct = red.get("cpu_rgba_to_bgr@residual_input_calls") == 0 \
        and leg.get("cpu_rgba_to_bgr@residual_input_calls", 0) > 0
    statuses["direct_rgba_residual"] = _entry(
        "DONE_PROVEN" if direct else "NOT_READY",
        ["transfer_ledger.json measured_steady_state.cpu_rgba_to_bgr@residual_input_calls",
         "transfer_ledger_legacy.json same field as the control"],
        {"reduced_calls_at_residual_site": red.get("cpu_rgba_to_bgr@residual_input_calls"),
         "legacy_calls_at_residual_site": leg.get("cpu_rgba_to_bgr@residual_input_calls"),
         "residual_backend_actual": (ledgers["transfer_ledger"] or {}).get(
             "residual_backend_actual")})

    # 4. NVOF BGR input, from the parity run at the real working size
    par_ok = (parity or {}).get("status") == "DONE_PROVEN" \
        and parity.get("checks_passed") == parity.get("checks_total")
    statuses["nvof_bgr_input"] = _entry(
        "DONE_PROVEN" if par_ok else "NOT_READY",
        ["nvof_bgr_parity.json", "transfer_ledger.json measured_steady_state.nvof_input_format"],
        {"checks": f"{(parity or {}).get('checks_passed')}/{(parity or {}).get('checks_total')}",
         "flow_bit_exact": _check_value(parity.get("sequences"), "flow_bit_exact_every_frame"),
         "input_format_measured_in_reduced_run": red.get("nvof_input_format"),
         "input_format_measured_in_legacy_run": leg.get("nvof_input_format")})

    # 5. the upload that got smaller
    h2d_ok = red.get("nvof_h2d_bytes_measured") == 6220800 \
        and leg.get("nvof_h2d_bytes_measured") == 8294400 \
        and red.get("nvof_h2d_copy_count") == 1
    statuses["nvof_h2d_reduction"] = _entry(
        "REDUCED_PROVEN" if h2d_ok else "NOT_READY",
        ["transfer_ab_benchmark.json gates.copy_gate",
         "transfer_ledger.json / transfer_ledger_legacy.json measured_steady_state"],
        {"reduced_bytes_per_steady_frame": red.get("nvof_h2d_bytes_measured"),
         "legacy_bytes_per_steady_frame": leg.get("nvof_h2d_bytes_measured"),
         "saved_bytes": (leg.get("nvof_h2d_bytes_measured", 0)
                         - red.get("nvof_h2d_bytes_measured", 0)),
         "saved_percent": round(100.0 * (leg.get("nvof_h2d_bytes_measured", 0)
                                         - red.get("nvof_h2d_bytes_measured", 0))
                                / leg["nvof_h2d_bytes_measured"], 2)
         if leg.get("nvof_h2d_bytes_measured") else None,
         "copy_count_reduced": red.get("nvof_h2d_copy_count"),
         "copy_count_legacy": leg.get("nvof_h2d_copy_count"),
         "note": "the download side is unchanged; this is a reduction of the upload payload "
                 "from four bytes per pixel to three, not an eliminated transfer"})

    # 6. the two CPU channel-order conversions
    conv_ok = red.get("cpu_bgr_to_rgba_calls") == 0 and red.get("cpu_rgba_to_bgr_calls") == 0 \
        and red.get("cpu_color_conversion_output_bytes") == 0 \
        and leg.get("cpu_bgr_to_rgba_calls") == 2 and leg.get("cpu_rgba_to_bgr_calls") == 1 \
        and counters["reduced"].get("cpu_bgr_to_rgba_calls") == 0 \
        and counters["legacy"].get("cpu_bgr_to_rgba_calls") == 2
    statuses["cpu_color_conversion_elimination"] = _entry(
        "DONE_PROVEN" if conv_ok else "NOT_READY",
        ["transfer_ab_benchmark.json gates.copy_gate",
         "transfer_ledger.json measured_steady_state",
         "transfer_ledger_legacy.json measured_steady_state as the control"],
        {"reduced_calls": {k: red.get(k) for k in
                           ("cpu_bgr_to_rgba_calls", "cpu_rgba_to_bgr_calls",
                            "cpu_color_conversion_output_bytes")},
         "legacy_calls": {k: leg.get(k) for k in
                          ("cpu_bgr_to_rgba_calls", "cpu_rgba_to_bgr_calls",
                           "cpu_color_conversion_output_bytes")},
         "auto_order_calls": {k: auto.get(k) for k in
                              ("cpu_bgr_to_rgba_calls", "cpu_rgba_to_bgr_calls")},
         "sites_eliminated": ["cpu_bgr_to_rgba@dlss_input", "cpu_bgr_to_rgba@nvof_input",
                              "cpu_rgba_to_bgr@residual_input"]})

    # 7. NVOF temporal history still behaves: one upload on a steady frame, two on the reset
    #    frame, the staging buffers allocated once per session, and no CPU copy of the previous
    #    frame anywhere.
    records = (ledgers["transfer_ledger"] or {}).get("per_frame_records") or []
    uploads = [r.get("nvof_h2d_bytes_measured") for r in records]
    copies = [r.get("nvof_h2d_copy_count") for r in records]
    prev_copies = [r.get("cpu_previous_frame_copy_bytes") for r in records]
    staging = [r.get("nvof_bgr_staging_alloc_count_delta") for r in records]
    history = [r.get("nvof_native_history") for r in records]
    hist_ok = (bool(records) and uploads[0] == 12441600 and copies[0] == 2
               and set(uploads[1:]) == {6220800} and set(copies[1:]) == {1}
               and set(prev_copies) == {0} and set(history) == {1}
               and set(staging[1:]) == {0} and staging[0] == 2)
    statuses["nvof_native_history_regression"] = _entry(
        "DONE_PROVEN" if hist_ok else "NOT_READY",
        ["transfer_ledger.json per_frame_records (upload bytes and copy counts, "
         "cpu_previous_frame_copy_bytes, staging allocations, native history flag)",
         "transfer_ab_benchmark.json gates.copy_gate",
         "nvof_bgr_parity.json reset-frame upload check"],
        {"reset_frame_upload_bytes": uploads[0] if uploads else None,
         "reset_frame_copy_count": copies[0] if copies else None,
         "steady_frame_upload_bytes": sorted(set(uploads[1:])) or None,
         "steady_frame_copy_count": sorted(set(copies[1:])) or None,
         "cpu_previous_frame_copy_bytes_values": sorted(set(prev_copies)) or None,
         "native_history_flag_values": sorted(set(history)) or None,
         "staging_allocations_first_frame": staging[0] if staging else None,
         "staging_allocations_steady": sorted(set(staging[1:])) or None,
         "staging_note": "the two persistent host-to-device staging buffers are allocated once "
                         "when the session opens and reused from the second frame on; a "
                         "per-frame number here would mean the path was allocating every frame",
         "parity_reset_check": _check_value((parity or {}).get("sequences"),
                                            "bgr_reset_h2d_bytes")})

    # 8. native numeric work: the residual and resampler suites in their own transcript
    native_counts = _counts(native_pytest or "")
    native_ok = bool(native_pytest) and not native_counts.get("failed") \
        and not native_counts.get("error") and bool(native_counts.get("passed"))
    statuses["native_numeric_regression"] = _entry(
        "DONE_PROVEN" if native_ok else "NOT_READY",
        ["pytest_native_numeric.txt", "pytest_output.txt"],
        {"summary_line": native_pytest or None, "included_in_full_suite": True})

    # 9. preview and export still agree
    preview_counts = _counts(preview_pytest or "")
    preview_ok = bool(preview_pytest) and not preview_counts.get("failed") \
        and not preview_counts.get("error") and bool(preview_counts.get("passed"))
    statuses["preview_export_regression"] = _entry(
        "DONE_PROVEN" if preview_ok else "NOT_READY",
        ["pytest_preview_export.txt", "pytest_output.txt"],
        {"summary_line": preview_pytest or None,
         "exports_completed_this_round": sum(
             1 for mode in modes for run in (modes[mode].get("runs") or [])
             if run.get("frames_recorded")),
         "included_in_full_suite": True})

    collected = re.search(r"(\d+) tests? collected", main_pytest_text or "")
    statuses["full_unit_suite"] = _entry(
        "DONE_PROVEN" if suite_clean else "NOT_READY",
        ["pytest_output.txt"],
        {"summary_line": main_pytest or None, "counts": counts or None,
         "collected": int(collected.group(1)) if collected else None})

    integrity = {name: (doc.get("gates") or {}).get("output_integrity_gate") or []
                 for name, doc in executions.items()}
    integrity_ok = ffprobe is not None and bool(integrity) and all(
        _ok(group) for group in integrity.values())
    statuses["legacy_reduced_output_integrity"] = _entry(
        "DONE_PROVEN" if integrity_ok else "NOT_READY",
        [f"{name} gates.output_integrity_gate" for name in sorted(integrity)]
        + ["ffprobe_output.json"],
        {"checks_per_execution": {name: f"{sum(1 for c in group if c.get('ok'))}/{len(group)}"
                                  for name, group in sorted(integrity.items())},
         "bitstream_sha256_per_run": (ab or {}).get("output_identity", {})
                                         .get("bitstream_sha256_of_each_run"),
         "container_hash_is_not_content": (ab or {}).get("output_identity", {}).get("note")})

    # 7. performance. §44 asks for the reduced order not to be slower; one execution of §41's
    #    protocol is a median of two runs, and on this machine the same order of the same code
    #    measures 10% apart between runs, so a single execution cannot answer that question in
    #    either direction. The verdict is therefore taken from the pooled runs of the same
    #    protocol - tests/unit/collect_transfer_performance.py - and every execution's own
    #    outcome, including the one whose §44 gate failed, is carried here verbatim.
    verdict = (perf or {}).get("verdict") or {}
    per_execution = verdict.get("per_execution_specified_gate_outcome") or {}
    exec_shas = {ex.get("code_under_test") for ex in (perf or {}).get("executions") or []}
    same_sha = bool(sha) and exec_shas == {sha}
    perf_status = verdict.get("status") if verdict.get("status") in STATUS_VALUES \
        else "NOT_READY"
    if not same_sha:
        perf_status = "NOT_READY"
    statuses["legacy_reduced_performance"] = _entry(
        perf_status,
        ["transfer_performance_pooled.json verdict",
         "transfer_performance_pooled.json executions[].orders",
         "transfer_ab_benchmark.json gates.performance_gate",
         "transfer_ab_benchmark_150f.json gates.performance_gate"],
        {"decision_rule": verdict.get("rule"),
         "no_end_to_end_regression": verdict.get("no_end_to_end_regression"),
         "reduced_faster_established": verdict.get("reduced_faster_established"),
         "pooled_median_fps": verdict.get("pooled_median_fps"),
         "pooled_speedup_ratio": verdict.get("pooled_speedup_ratio"),
         "pooled_runs_per_order": verdict.get("pooled_runs_per_order"),
         "paired_sign_test": verdict.get("sign_test"),
         "p_value_that_reduced_is_slower": verdict.get(
             "evidence_that_reduced_is_slower_p_value"),
         "median_paired_fps_change_percent": verdict.get(
             "median_reduced_minus_legacy_percent"),
         "executions_of_the_specified_protocol": len(per_execution),
         "executions_whose_own_44_gate_failed": sorted(
             name for name, row in per_execution.items()
             if not row.get("performance_gate_all_passed")),
         "per_execution_outcome": per_execution,
         "150_frame_corroboration": verdict.get("150_frame_corroboration"),
         "every_execution_reports_the_same_code_under_test": same_sha,
         "first_execution_code_under_test": sorted(exec_shas)[:1]})

    statuses["dlss_gpu_io"] = _entry(
        (abi or {}).get("conclusion") if (abi or {}).get("conclusion") in STATUS_VALUES
        else "NOT_READY",
        ["dlss_host_abi.json"],
        {"public_exports": (abi or {}).get("dlssnr_exports"),
         "gpu_resource_public_abi": (abi or {}).get("gpu_resource_public_abi"),
         "host_source_present": (abi or {}).get("host_source_present"),
         "sdk_headers_present": (abi or {}).get("sdk_headers_present"),
         "sdk_libs_present": (abi or {}).get("sdk_libs_present"),
         "host_pointer_path_proven": (abi or {}).get("host_pointer_path_proven"),
         "conclusion_reason": (abi or {}).get("conclusion_reason")})

    dec = (boundaries or {}).get("decoder") or {}
    enc = (boundaries or {}).get("encoder") or {}
    nvof = (boundaries or {}).get("nvof") or {}
    dlss = (boundaries or {}).get("dlss") or {}
    statuses["decoder_gpu_handoff"] = _entry(
        dec.get("status") if dec.get("status") in STATUS_VALUES else "NOT_READY",
        ["process_boundaries.json decoder"],
        # The field name is the one §47 asks for; the value it carries is the denial. The
        # handoff is blocked by the process boundary, so this records that the route does
        # not exist rather than claiming anything about a copy-free path.
        {"zero_copy_into_python_process": dec.get("zero_copy_into_python_process"),
         "gpu_resource_export_api": dec.get("gpu_resource_export_api"),
         "cuda_ipc_currently_used": dec.get("cuda_ipc_currently_used"),
         "shared_d3d_handle_currently_used": dec.get("shared_d3d_handle_currently_used"),
         "hwdownload_removed_this_round": dec.get("hwdownload_removed_this_round"),
         "unblock_requires_any_of": dec.get("unblock_requires_any_of")})
    statuses["encoder_gpu_handoff"] = _entry(
        enc.get("status") if enc.get("status") in STATUS_VALUES else "NOT_READY",
        ["process_boundaries.json encoder"],
        {"direct_gpu_encode": enc.get("direct_gpu_encode"),
         "encoder_pipe_bytes_measured_per_frame": enc.get("pipe_bytes_measured_per_frame"),
         "encoder_gpu_upload_bytes_measured": enc.get("encoder_gpu_upload_bytes_measured"),
         "unblock_requires_any_of": enc.get("unblock_requires_any_of")})

    extra = {
        "nvof_flow_download": _entry(
            nvof.get("flow_d2h_status") if nvof.get("flow_d2h_status") in STATUS_VALUES
            else "NOT_READY",
            ["process_boundaries.json nvof", "transfer_ledger.json measured_steady_state"],
            {"d2h_bytes_per_steady_frame": red.get("nvof_d2h_bytes_measured"),
             "d2h_copy_count": red.get("nvof_d2h_copy_count"),
             "why": "the flow tensor is consumed on the CPU by the DLSS host wrapper, which "
                    "takes host pointers; the upload got smaller, this one still returns"}),
        "dlss_internal_round_trip": _entry(
            dlss.get("round_trip_status") if dlss.get("round_trip_status") in STATUS_VALUES
            else "NOT_READY",
            ["process_boundaries.json dlss", "dlss_host_abi.json"],
            {"internal_h2d": dlss.get("internal_h2d_bytes_measured"),
             "internal_d2h": dlss.get("internal_d2h_bytes_measured"),
             "logical_buffer_bytes": {k: red.get(k) for k in
                                      ("dlss_input_color_bytes_logical",
                                       "dlss_input_flow_bytes_logical",
                                       "dlss_input_depth_bytes_logical",
                                       "dlss_output_bytes_logical")}}),
    }

    # The transfer rows this round can put a number on, before and after, measured on the frames
    # that ran. Each one gets its verdict word from that pair rather than from a sentence, and the
    # whole table is then read back by `check_transfer_claims` so a word that the numbers do not
    # support becomes a failure of the round rather than a line in a report.
    claims = [
        transfer_claim("nvof_h2d_upload", leg.get("nvof_h2d_bytes_measured"),
                       red.get("nvof_h2d_bytes_measured"),
                       evidence=["transfer_ledger_legacy.json", "transfer_ledger.json"],
                       note="four bytes per pixel to three: the alpha channel left the copy"),
        transfer_claim("cpu_bgr_to_rgba_calls", leg.get("cpu_bgr_to_rgba_calls"),
                       red.get("cpu_bgr_to_rgba_calls"), unit="calls/frame",
                       evidence=["transfer_ledger_legacy.json", "transfer_ledger.json"]),
        transfer_claim("cpu_rgba_to_bgr_calls", leg.get("cpu_rgba_to_bgr_calls"),
                       red.get("cpu_rgba_to_bgr_calls"), unit="calls/frame",
                       evidence=["transfer_ledger_legacy.json", "transfer_ledger.json"]),
        transfer_claim("cpu_color_conversion_output",
                       leg.get("cpu_color_conversion_output_bytes"),
                       red.get("cpu_color_conversion_output_bytes"),
                       evidence=["transfer_ledger_legacy.json", "transfer_ledger.json"]),
        transfer_claim("nvof_flow_d2h_download", leg.get("nvof_d2h_bytes_measured"),
                       red.get("nvof_d2h_bytes_measured"),
                       evidence=["transfer_ledger_legacy.json", "transfer_ledger.json"],
                       note="unchanged: DLSS takes its motion vector as a host pointer"),
        transfer_claim("decoder_pipe", leg.get("decoder_pipe_bytes"),
                       red.get("decoder_pipe_bytes"),
                       evidence=["transfer_ledger_legacy.json", "transfer_ledger.json",
                                 "process_boundaries.json decoder"],
                       note="the decoder child hands this process a byte stream; the reduced "
                            "order changed nothing about that boundary, so the row is unchanged "
                            "and is reported as blocked rather than as a reduction"),
        transfer_claim("encoder_pipe", leg.get("encoder_pipe_bytes"),
                       red.get("encoder_pipe_bytes"),
                       evidence=["transfer_ledger_legacy.json", "transfer_ledger.json"],
                       note="the encoder still reads host bytes from its stdin"),
    ]
    decoder_pipe = next(c for c in claims if c["claim"] == "decoder_pipe")
    statuses["decoder_pipe"] = _entry(
        decoder_pipe["status"],
        ["transfer_ledger.json measured_steady_state.decoder_pipe_bytes",
         "transfer_ledger_legacy.json same field", "status_consistency.json",
         "process_boundaries.json decoder"],
        {"measured_pipe_bytes_per_frame": decoder_pipe["after"],
         "legacy_pipe_bytes_per_frame": decoder_pipe["before"],
         "unchanged_this_round": decoder_pipe["unchanged_this_round"],
         "estimated_gpu_to_cpu_nv12_bytes": red.get("decoder_gpu_to_cpu_bytes_estimated"),
         "why": "the decoder runs in a child process and hands frames over a pipe, which carries "
                "bytes and not GPU resources; nothing on this path got smaller this round, so "
                "the only word the measurements support is the one that says so"})

    # The words this file publishes for those numbers, read back out of the entries above, so the
    # consistency check is checking the report and not a copy of the derivation.
    published = {
        "nvof_h2d_upload": statuses["nvof_h2d_reduction"]["status"],
        "cpu_bgr_to_rgba_calls": statuses["cpu_color_conversion_elimination"]["status"],
        "cpu_rgba_to_bgr_calls": statuses["cpu_color_conversion_elimination"]["status"],
        "decoder_pipe": statuses["decoder_pipe"]["status"],
        "nvof_flow_d2h_download": extra["nvof_flow_download"]["status"],
    }
    consistency = {
        "claims": claims,
        "checks": check_transfer_claims(claims),
        "checks_against_published_words": check_transfer_claims(claims, published),
        "rule": "a REDUCED_PROVEN row needs after strictly below before; a row that claims a "
                "transfer was deleted needs after == 0 with a non-zero measurement on the other "
                "order as its control; before == after can only ever be BLOCKED_PROVEN",
        "inconsistent": [row for row in check_transfer_claims(claims, published)
                         if not row["consistent"]],
    }

    ordered = {key: statuses[key] for key in _STATUS_ORDER}
    ordered["additional_this_round"] = extra
    verdict = ("DONE_PROVEN" if all(s["status"] != "NOT_READY" for s in statuses.values())
               and not consistency["inconsistent"] and sha else "NOT_READY")
    payload = {
        "generated_by": "tests/unit/collect_transfer_status.py",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
                                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "round": "CPU-GPU round-trip reduction phase 1",
        "code_under_test": sha,
        "code_under_test_shared_by_artifacts": sha is not None,
        "artifact_shas": {name: _sha_of(doc) for name, doc in present.items()},
        "missing_artifacts": sorted(set(names) - set(present)),
        "status": verdict,
        "native_binaries": native_binaries(),
        "commits": commits_this_round(ROUND_COMMITS),
        "gates": ordered,
        "gate_statuses": {key: ordered[key]["status"] for key in _STATUS_ORDER},
        "allowed_status_values": list(STATUS_VALUES),
        "transfer_claim_consistency": consistency,
        # §53: this round may never write these sentences, because nothing it measured earns
        # them. They are listed to be refused, not to be asserted.
        "forbidden_phrases": ["GPU pipeline complete", "zero-copy complete",
                              "CPU round trips eliminated", "full GPU acceleration complete"],
    }
    return payload


def consistency_artifact(consistency, code_under_test=None):
    """The §38 file: every measured transfer row with the word it earned and the arithmetic.

    Written separately from the round status because it answers one narrow question a reviewer
    has to be able to ask without reading prose: is there a row in this report that says
    "reduced" while its two measured numbers are equal. So it carries the bytes, the saved
    amount and the percentage beside the word, and lists the failures first.
    """
    rows = {row["claim"]: row for row in consistency["checks_against_published_words"]}
    by_name = {claim["claim"]: claim for claim in consistency["claims"]}
    dec = rows.get("decoder_pipe") or {}
    return {
        "generated_by": "tests/unit/collect_transfer_status.py",
        "code_under_test": code_under_test,
        "rule": consistency["rule"],
        "decoder_pipe": {
            "legacy_bytes": dec.get("before"),
            "reduced_bytes": dec.get("after"),
            "status": dec.get("published_status"),
            "reason": by_name.get("decoder_pipe", {}).get("note"),
            "unchanged_this_round": dec.get("before") == dec.get("after"),
        },
        "reduced_proven_rows": [
            {"claim": name, "before": by_name[name]["before"], "after": by_name[name]["after"],
             "saved": by_name[name]["saved"],
             "saved_percent": by_name[name]["saved_percent"], "unit": by_name[name]["unit"],
             "status": row["published_status"]}
            for name, row in sorted(rows.items())
            if row["published_status"] == "REDUCED_PROVEN"],
        "rows": [rows[name] for name in sorted(rows)],
        "inconsistent_rows": [row for row in (rows[n] for n in sorted(rows))
                              if not row["consistent"]],
        "all_rows_consistent": all(row["consistent"] for row in rows.values()),
    }




_STATUS_ORDER = ("transfer_ledger", "fused_working_prepare", "direct_rgba_residual",
                 "nvof_bgr_input", "nvof_h2d_reduction", "cpu_color_conversion_elimination",
                 "nvof_native_history_regression", "native_numeric_regression",
                 "preview_export_regression", "full_unit_suite",
                 "legacy_reduced_output_integrity", "legacy_reduced_performance",
                 "dlss_gpu_io", "decoder_gpu_handoff", "decoder_pipe", "encoder_gpu_handoff")


def _ab_executions(out_dir, ab, extra_names):
    """Every recorded A/B execution of this round, keyed by the file that carries it.

    The performance gate is decided on the pooled file, but the integrity gate has to hold of
    every execution individually, so this reads them all rather than the one §46 names.
    """
    out = {}
    if ab is not None:
        out["transfer_ab_benchmark.json"] = ab
    for name in extra_names:
        doc = _load(out_dir, name)
        if doc is not None:
            out[name] = doc
    return out


# ---------------------------------------------------------------------------
# the benchmark-protocol round: §41's keys, each one read out of an artifact
# ---------------------------------------------------------------------------
#
# The phase-1 status file reports gates about the transfer path. This round changed no transfer
# path; it changed the instrument that measures whether the path is affordable. So the keys below
# are gates about the instrument: whether the warmup pair is excluded everywhere a statistic is
# taken, whether each order really did sit in each position of a pair equally often, whether the
# schedule an execution published is the sequence it launched, whether the two readings of the
# same runs agree about which way they point, and whether the words this report uses for a
# transfer are the ones its own numbers support.
#
# Each key is decided by a list of checks against written artifacts, so the reason a key is
# NOT_READY is the name of the check that failed rather than anyone's summary of it.

BENCH_FIX_KEYS = ("warmup_exclusion", "kept_run_pooling", "counterbalanced_order",
                  "order_balance", "position_stratification", "copy_gate",
                  "output_integrity", "performance_non_regression",
                  "performance_direction_consistency", "decoder_status_semantics",
                  "full_unit_suite", "native_scope_unchanged")

# §50: the stability round's status file. The names §50 lists, in that order, with the four
# instrument keys from the previous round kept because a verdict still reads them. `baseline`
# moves to the commit this round started from, so the native gate compares against the binaries
# this round was told not to rebuild.
BENCH_STABILITY_BASELINE = "e3aa86c"
BENCH_STABILITY_KEYS = ("artifact_provenance", "ffprobe_schema", "benchmark_lock",
                        "cpu_restriction", "quiet_preflight", "runtime_contamination",
                        "warmup_exclusion", "kept_run_pooling", "counterbalanced_order",
                        "order_balance", "position_stratification", "benchmark_variance",
                        "copy_gate",
                        "output_integrity", "decoder_status_semantics",
                        "performance_non_regression", "performance_direction_consistency",
                        "corroboration_consistency", "native_scope_unchanged", "full_unit_suite")
# §50 lists the keys the stability status file must carry. The ones not named there -
# `kept_run_pooling`, `order_balance`, `position_stratification` and `decoder_status_semantics` -
# are carried over from the previous round because a verdict is still read out of the same
# artifacts, and a gate that round proved does not stop being worth proving because this round's
# brief happened not to repeat it. `cpu_restriction` is the fifth key outside §50's list, and it is
# not a carry-over: Round C's §6 asks for it by name, and it sits before `quiet_preflight` because
# an unbounded machine is a question the waiting cannot answer.
BENCH_STABILITY_KEYS_REQUIRED_BY_50 = ("artifact_provenance", "ffprobe_schema", "benchmark_lock",
                                       "quiet_preflight", "runtime_contamination",
                                       "benchmark_variance", "copy_gate", "output_integrity",
                                       "warmup_exclusion", "counterbalanced_order",
                                       "performance_non_regression",
                                       "performance_direction_consistency",
                                       "corroboration_consistency", "native_scope_unchanged",
                                       "full_unit_suite")
assert set(BENCH_STABILITY_KEYS_REQUIRED_BY_50) <= set(BENCH_STABILITY_KEYS), \
    "§50's names must all be keys the status file reports"

# The four questions `benchmark_e2e.ffprobe_validate` answers about every output file, named here
# so a merged artifact that reported three of them fails the schema gate instead of passing it with
# a shorter list of checks than §38 asked for.
PROBE_CHECK_FIELDS = ("packets", "resolution", "duration_within_1pct", "fps_matches_source")

# The four numbers §32 asks for, named so the report cannot quietly substitute another one.
PERFORMANCE_KEY_NUMBERS = ("A_pooled_kept_median_legacy_fps", "B_pooled_kept_median_reduced_fps",
                           "C_pooled_kept_ratio", "D_pairwise_median_ratio")


def _key(checks, evidence, detail=None):
    """One §41 key: DONE_PROVEN only when every check that decides it reads true.

    Only the two words §41 allows ever come out of here, which is the point: an instrument that
    was not built balanced does not get a softer word, it gets NOT_READY.
    """
    rows = [{"what": what, "expected": True, "actual": bool(actual), "ok": bool(actual),
             "detail": detail_row} for what, actual, detail_row in checks]
    failed = sorted(row["what"] for row in rows if not row["ok"])
    return {"status": "DONE_PROVEN" if rows and not failed else "NOT_READY",
            "evidence": list(evidence), "checks": rows, "failed_checks": failed,
            "detail": detail or {}}


def execution_artifacts(out_dir, subdirectory="ab_executions"):
    """The full artifact of each execution, as the driver that ran it wrote it.

    The pooled file is a summary of these; the questions this round asks - what schedule was
    recorded, what sequence was launched, which run sat in which position - are only answerable
    from the originals.
    """
    directory = os.path.join(out_dir, subdirectory)
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith("_transfer_ab_benchmark.json"):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as handle:
            try:
                out.append((name, json.load(handle)))
            except Exception:
                out.append((name, None))
    return out


# The artifacts a performance verdict is read out of. §3 of the stability round names them, and
# names the status file itself last because that one is what this list decides.
BENCH_FIX_REQUIRED_ARTIFACTS = ("protocol", "counterbalanced_performance",
                                "counterbalanced_150f", "historical_warmup_recalc",
                                "ffprobe_output", "environment", "status_consistency")
BENCH_STABILITY_REQUIRED_ARTIFACTS = BENCH_FIX_REQUIRED_ARTIFACTS + (
    "benchmark_environment", "benchmark_runtime_monitor", "cpu_restriction_audit")


def bench_provenance(out_dir, required=BENCH_FIX_REQUIRED_ARTIFACTS, extra_executions=()):
    """What every artifact of this round says about which code it describes.

    §5: an artifact that declares nothing is a disagreement, not an absence. Before this, two of
    the derived files carried no `code_under_test` at all and the round still reported a shared
    SHA, because the comparison had thrown the blanks away first. Here a blank, a missing file or
    a different commit all land in `shared_sha = None`, and the verdict built on them is
    NOT_READY.

    §37: the counts go out as `artifact_sha_coverage` so a reader can see how complete the
    evidence was without having to re-open ten files and diff their fields by hand.
    """
    documents = {name: _load(out_dir, f"{name}.json") for name in required}
    executions = {}
    for subdirectory in ("ab_executions", *extra_executions):
        for name, doc in execution_artifacts(out_dir, subdirectory):
            executions[f"{subdirectory}/{name}"] = doc
    everything = dict(documents)
    everything.update(executions)
    primary = _sha_of(documents.get("counterbalanced_performance"))
    return {
        "documents": documents,
        "execution_documents": executions,
        "primary_code_under_test": primary,
        "shared_sha": _shared_sha([_sha_of(doc) for doc in everything.values()]),
        "missing_required": sorted(name for name, doc in documents.items() if doc is None),
        "coverage": artifact_sha_coverage(everything, code_under_test=primary),
    }


def _launch_sequence_matches_schedule(doc):
    """Whether the artifact's recorded schedule is the sequence it says it launched.

    Compared as the four fields that matter, in order. A schedule written after the runs to match
    them would have to reproduce this equality by accident, and the point of §12 is that it cannot.
    """
    schedule = (doc or {}).get("execution_schedule") or []
    sequence = (doc or {}).get("run_launch_sequence") or []
    if not schedule or not sequence:
        return False, {"schedule_entries": len(schedule), "launched_rows": len(sequence)}
    expected = [{"run": entry.get("run"), "mode": mode, "position_in_pair": position,
                 "kept": entry.get("kept")}
                for entry in schedule
                for position, mode in enumerate(entry.get("order") or [], start=1)]
    actual = [{key: row.get(key) for key in ("run", "mode", "position_in_pair", "kept")}
              for row in sequence]
    return expected == actual, {"expected": expected, "actual": actual}


def _schedule_follows_the_rule(doc):
    """Whether an execution's recorded schedule is the deterministic one for its index."""
    ab = _ab_module()
    index = (doc or {}).get("execution_order_index")
    runs = ((doc or {}).get("configuration") or {}).get("runs_per_mode")
    if index is None or not runs:
        return False, None
    return ((doc or {}).get("execution_schedule") or []) == ab.execution_schedule(int(runs),
                                                                                  int(index)), \
        ab.execution_schedule(int(runs), int(index))


def bench_fix_claims(perf):
    """The transfer rows this round measured itself, taken from its own executions.

    Nothing is copied from the previous round's ledgers: every number here is a steady-state
    counter the A/B executions of this code wrote, and the figure a claim rests on is the median
    across the executions that produced it. That is what makes the §38 consistency table this
    round's own evidence rather than a repetition of last round's.
    """
    executions = [ex for ex in (perf or {}).get("executions") or []
                  if not ex.get("unpoolable_reasons")]
    measured = {}
    for mode in ("legacy", "reduced"):
        per_execution = {}
        for name in CLAIM_COUNTERS:
            values = [((ex.get("orders") or {}).get(mode, {})
                       .get("steady_state_counters") or {}).get(name) for ex in executions]
            values = [v for v in values if isinstance(v, (int, float))]
            per_execution[name] = {"median": statistics.median(values) if values else None,
                                   "values": sorted(values),
                                   "executions_reporting": len(values)}
        measured[mode] = per_execution

    def side(mode, name):
        value = measured[mode][name]["median"]
        return int(value) if isinstance(value, (int, float)) else None

    claims = [
        transfer_claim("nvof_h2d_upload", side("legacy", "nvof_h2d_bytes_measured"),
                       side("reduced", "nvof_h2d_bytes_measured"),
                       evidence=["counterbalanced_performance.json executions[].orders[]."
                                 "steady_state_counters.nvof_h2d_bytes_measured"],
                       note="four bytes per pixel to three on the steady frame"),
        transfer_claim("cpu_bgr_to_rgba_calls", side("legacy", "cpu_bgr_to_rgba_calls"),
                       side("reduced", "cpu_bgr_to_rgba_calls"), unit="calls/frame",
                       evidence=["the same field on both orders"]),
        transfer_claim("cpu_rgba_to_bgr_calls", side("legacy", "cpu_rgba_to_bgr_calls"),
                       side("reduced", "cpu_rgba_to_bgr_calls"), unit="calls/frame",
                       evidence=["the same field on both orders"]),
        transfer_claim("cpu_color_conversion_output",
                       side("legacy", "cpu_color_conversion_output_bytes"),
                       side("reduced", "cpu_color_conversion_output_bytes"),
                       evidence=["the same field on both orders"]),
        transfer_claim("nvof_flow_d2h_download", side("legacy", "nvof_d2h_bytes_measured"),
                       side("reduced", "nvof_d2h_bytes_measured"),
                       evidence=["the same field on both orders"],
                       note="the motion field still comes back to the host because the DLSS "
                            "wrapper takes a host pointer"),
        transfer_claim("decoder_pipe", side("legacy", "decoder_pipe_bytes"),
                       side("reduced", "decoder_pipe_bytes"),
                       evidence=["the same field on both orders",
                                 "status_consistency.json"],
                       note="the decoder is a child process writing a byte pipe; this round "
                            "changed nothing on that path, and the two measurements agree"),
        transfer_claim("encoder_pipe", side("legacy", "encoder_pipe_bytes"),
                       side("reduced", "encoder_pipe_bytes"),
                       evidence=["the same field on both orders"],
                       note="the encoder still reads host bytes from its stdin"),
    ]
    return claims, measured


def bench_fix_consistency(claims, published):
    """The §38 table: every claim, the word the round gave it, and whether the numbers agree."""
    checks = check_transfer_claims(claims, published)
    by_name = {claim["claim"]: claim for claim in claims}
    rows = {row["claim"]: row for row in checks}
    dec = rows.get("decoder_pipe") or {}
    return {
        "generated_by": "tests/unit/collect_transfer_status.py",
        "rule": "a REDUCED_PROVEN row needs after strictly below before; a row that claims a "
                "transfer was deleted needs after == 0 with a non-zero measurement on the other "
                "order as its control; before == after can only ever be BLOCKED_PROVEN",
        "decoder_pipe": {
            "legacy_bytes": dec.get("before"),
            "reduced_bytes": dec.get("after"),
            "status": dec.get("published_status"),
            "unchanged_this_round": dec.get("before") == dec.get("after"),
            "reason": by_name.get("decoder_pipe", {}).get("note"),
        },
        "reduced_proven_rows": [
            {"claim": name, "before": by_name[name]["before"], "after": by_name[name]["after"],
             "saved": by_name[name]["saved"],
             "saved_percent": by_name[name]["saved_percent"], "unit": by_name[name]["unit"],
             "status": row["published_status"]}
            for name, row in sorted(rows.items())
            if row["published_status"] == "REDUCED_PROVEN"],
        "rows": checks,
        "inconsistent_rows": [row for row in checks if not row["consistent"]],
        "all_rows_consistent": all(row["consistent"] for row in checks),
    }


def merge_ffprobe(out_dir, code_under_test=None,
                  subdirectories=("ab_executions", "ab_executions_150f")):
    """One file carrying ffprobe's reading of every output file every execution wrote.

    Each execution already wrote its own; this concatenates them so a reader who wants to check a
    frame count or a codec does not have to open thirty-six files, and so the round's integrity key
    can be decided on the count of runs actually probed rather than on a claim about it.

    §3 of this round: a merged artifact is still an artifact, so it names the code it describes.
    The SHA is passed in from the round's primary artifact rather than read from HEAD here, so
    re-assembling the derived files after the evidence has been committed cannot stamp them with
    a commit that post-dates the runs they are summarising.

    §2 of this round, on the fields. The aggregate used to read `codec_name` and `frames`, and
    neither exists on a probe record: `benchmark_e2e.ffprobe_validate` renames ffprobe's
    `codec_name` to `codec` when it builds the record, and the frame count it compares is
    `packets`. So the previous round published `"codecs_seen": [null]` beside `"expected":
    {"frames": 60}` and `"runs_whose_probed_frame_count_matches": 0` for 72 probed runs - a
    missing key reads as None and nobody noticed, which is the whole argument for counting the
    records that have no codec at all rather than folding blanks into the set of names. Reading
    a key that is not there is not a measurement of zero, and here it is not a measurement
    either: it is a bug, and it now reports itself as one.
    """
    executions = {}
    for subdirectory in subdirectories:
        directory = os.path.join(out_dir, subdirectory)
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith("_ffprobe_output.json"):
                continue
            doc = _load(directory, name)
            if doc:
                executions[f"{subdirectory}/{name}"] = doc
    runs = {mode: [{"execution": source, **row}
                   for source, doc in sorted(executions.items())
                   for row in (doc.get("runs") or {}).get(mode, [])]
            for mode in ("legacy", "reduced")}
    expected = next((doc.get("expected") for doc in executions.values() if doc.get("expected")),
                    {})
    # The 60-frame batch and the 150-frame corroboration are merged into one file, so more than one
    # expectation is normal here. Publishing the set keeps a reader from reading the 60-frame
    # `expected` block as if it applied to the 150-frame rows too.
    variants = sorted({json.dumps(doc.get("expected") or {}, sort_keys=True)
                       for doc in executions.values()})
    # What the files being merged themselves say. A merged artifact inherits the provenance of
    # its sources; it does not invent one, and it does not paper over two sources that disagree.
    declared = sorted({sha for sha in (_sha_of(doc) for doc in executions.values()) if sha})

    every_row = [row for rows in runs.values() for row in rows]
    probed = [(mode, row) for mode, rows in runs.items() for row in rows if row.get("ffprobe")]

    # §40: this one file spans both batches, so the counts are broken out by the directory each
    # execution's artifact came from. Without this the 60-frame integrity check would read a
    # probed-run count that includes the corroboration's files and find a mismatch it cannot
    # explain - which is the same class of mistake as summing a field that is not there.
    def batch_of(source):
        return (source or "").split("/")[0]

    probed_by_batch = {}
    for mode, row in probed:
        probed_by_batch[batch_of(row.get("execution"))] = \
            probed_by_batch.get(batch_of(row.get("execution")), 0) + 1
    rows_by_batch = {}
    for row in every_row:
        rows_by_batch[batch_of(row.get("execution"))] = \
            rows_by_batch.get(batch_of(row.get("execution")), 0) + 1

    def label(mode, row):
        return f"{mode} {row.get('execution')} run {row.get('run')}"

    per_execution_codecs = {
        source: sorted({probe.get("codec")
                        for mode in ("legacy", "reduced")
                        for row in (doc.get("runs") or {}).get(mode) or []
                        for probe in [row.get("ffprobe") or {}] if probe.get("codec")})
        for source, doc in sorted(executions.items())}
    codecs_seen = sorted({probe["ffprobe"].get("codec") for _, probe in probed
                          if probe["ffprobe"].get("codec")})
    # What the runs were supposed to be, as each execution's own artifact says it. Read from the
    # expectation blocks rather than from a hard-coded name, so the comparison is between two
    # published values and not between a published value and a guess made later.
    variant_docs = [json.loads(variant) for variant in variants]
    expected_codecs = sorted({code for code in (doc.get("codec") for doc in variant_docs) if code})
    expected_codec = expected.get("codec")
    rows_without_a_codec = sorted(label(mode, row) for mode, row in probed
                                  if not row["ffprobe"].get("codec"))
    rows_failing_a_probe_check = sorted(
        f"{label(mode, row)}: "
        f"{sorted(name for name, ok in (row['ffprobe'].get('checks') or {}).items() if not ok)}"
        for mode, row in probed if not all((row["ffprobe"].get("checks") or {}).values()))
    # The probe records carry their own `ok` flags, written by whoever ran ffprobe. Re-reading the
    # numbers beside those flags is the difference between trusting a boolean and checking it: a
    # record that said `"resolution": true` over a 1920x1080 file would pass the first test and
    # fail this one. Each row is compared against the expectation block of the execution that
    # wrote it, because the 60-frame and 150-frame batches share this file with different frame
    # counts and one batch's expectation cannot speak for the other.
    expected_by_source = {source: (doc.get("expected") or {})
                          for source, doc in executions.items()}

    def geometry_mismatch(mode, row):
        probe = row["ffprobe"]
        want = expected_by_source.get(row.get("execution")) or {}
        bad = []
        for key in ("width", "height"):
            if want.get(key) is not None and probe.get(key) != want[key]:
                bad.append(f"{key} {probe.get(key)} != {want[key]}")
        if want.get("fps") and probe.get("fps") is None:
            bad.append(f"no fps read against {want['fps']}")
        elif want.get("fps") and abs(probe["fps"] - want["fps"]) >= 1.0:
            bad.append(f"fps {probe['fps']} != {want['fps']}")
        return bad

    rows_with_the_wrong_geometry = sorted(
        f"{label(mode, row)}: {', '.join(geometry_mismatch(mode, row))}"
        for mode, row in probed if geometry_mismatch(mode, row))
    payload = {
        "generated_by": "tests/unit/collect_transfer_status.py",
        "code_under_test": code_under_test or (declared[0] if len(declared) == 1 else None),
        "code_under_test_of_each_merged_source": {name: _sha_of(doc)
                                                  for name, doc in sorted(executions.items())},
        "merged_from": sorted(executions),
        "expected": expected,
        "expected_variants": [json.loads(variant) for variant in variants],
        "runs_per_mode": {mode: len(rows) for mode, rows in runs.items()},
        "runs_with_an_ffprobe_reading": len(probed),
        "runs_with_an_ffprobe_reading_by_batch": probed_by_batch,
        "rows_by_batch": rows_by_batch,
        "runs_without_an_ffprobe_reading": sum(1 for row in every_row if not row.get("ffprobe")),
        "codec_field": "codec",
        "codecs_seen": codecs_seen,
        "codecs_seen_per_order": {
            mode: sorted({row["ffprobe"].get("codec") for row in rows if row.get("ffprobe")})
            for mode, rows in runs.items()},
        "codecs_seen_per_execution": per_execution_codecs,
        "rows_without_a_codec": rows_without_a_codec,
        "rows_failing_a_probe_check": rows_failing_a_probe_check,
        "runs_whose_probed_frame_count_matches": sum(
            1 for _, row in probed
            if row["ffprobe"].get("packets")
            and row["ffprobe"].get("packets") == row["ffprobe"].get("expected_frames")),
        "runs_whose_probed_frame_count_disagrees": sorted(
            f"{label(mode, row)}: ffprobe counted "
            f"{row['ffprobe'].get('packets')} packets against an expected "
            f"{row['ffprobe'].get('expected_frames')}"
            for mode, row in probed
            if row["ffprobe"].get("packets") != row["ffprobe"].get("expected_frames")),
        "expected_codec": expected_codec,
        "expected_codecs": expected_codecs,
        "expected_codec_seen_for_every_run": bool(expected_codecs) and bool(probed) and all(
            probe["ffprobe"].get("codec") in expected_codecs for _, probe in probed),
        "probe_check_fields_seen": sorted({name for _, probe in probed
                                           for name in (probe["ffprobe"].get("checks") or {})}),
        "geometry_seen": sorted({f'{probe["ffprobe"].get("width")}x{probe["ffprobe"].get("height")}'
                                 f'@{probe["ffprobe"].get("fps")}' for _, probe in probed}),
        "rows_with_the_wrong_geometry": rows_with_the_wrong_geometry,
        "runs_whose_geometry_matches_its_own_expectation":
            len(probed) - len(rows_with_the_wrong_geometry),
        "runs_whose_probe_records_no_checks": sorted(
            label(mode, row) for mode, row in probed if not row["ffprobe"].get("checks")),
        "executions": {name: {"status": doc.get("status"),
                              "runs": doc.get("runs"),
                              "expected": doc.get("expected")}
                       for name, doc in sorted(executions.items())},
        "runs": runs,
        "note": "ffprobe read each file independently of the pipeline; a run with no reading here "
                "is a file that was written and could not be parsed, which the integrity gate "
                "treats as a failure rather than as an absent row. codecs_seen is aggregated off "
                "each record's `codec` field - the field ffprobe_validate writes - and a record "
                "with no codec there is listed under rows_without_a_codec instead of becoming a "
                "null in the set of names. The frame count is `packets` against each record's own "
                "`expected_frames`, and width, height and fps are re-read against the expectation "
                "block of the execution that wrote the record, so a gate here never rests on a "
                "boolean that nobody checked against the numbers beside it",
    }
    return payload


def bench_fix_environment(out_dir, code_under_test=None, baseline=BENCH_FIX_BASELINE,
                          round_name="transfer benchmark evidence protocol fix"):
    """environment.json for this round: the machine, the binaries, the fixture the runs used.

    `code_under_test` arrives from the round's primary artifact. HEAD is only a fallback, because
    by the time a reader re-assembles the derived files the evidence commit may have moved the
    branch away from the code the runs actually used. The value is published under both spellings
    (`code_under_test` and `code_under_test_sha`) so a reader who greps for either finds it.

    `baseline` is the commit the round started from, and it is a parameter rather than a constant
    because the diff it publishes is what `native_scope_unchanged` reads: measured against the wrong
    baseline, that check either misses a production file this round touched or blames one that an
    earlier round touched.
    """
    import resource_metrics

    fixture = None
    settings = None
    for _, doc in execution_artifacts(out_dir):
        config = ((doc or {}).get("configuration") or {})
        if config.get("fixture"):
            fixture = dict(config["fixture"])
            settings = {"working_scale": config.get("working_scale"),
                        "flow_engine": config.get("flow_engine"),
                        "flow_preset": config.get("flow_preset"),
                        "encoder_cq": config.get("encoder_cq"),
                        "output_resolution": config.get("output_resolution"),
                        "frames_per_run": config.get("frames_per_run")}
            break
    sha = code_under_test or git("rev-parse", "HEAD")
    info = resource_metrics.environment_info({
        "code_under_test": sha,
        "code_under_test_sha": sha,
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit_subject": git("log", "-1", "--pretty=%s"),
        "commit_history": git("log", "-8", "--pretty=%h %s").splitlines(),
        "round": round_name,
        "baseline": baseline,
        "fixture": fixture,
        "benchmark_settings": settings,
        "files_changed_since_baseline": [line for line in git(
            "diff", "--name-only", f"{baseline}..HEAD").splitlines() if line],
        "native_binaries": native_binaries(),
    })
    return info


def native_against_baseline(baseline=BENCH_FIX_BASELINE):
    """§40: are the binaries this round's evidence was made on the ones the baseline held?

    The working file's hash is compared against the object the baseline commit stores and against
    the object HEAD stores. Three hashes agreeing is the whole question - it says the DLLs were not
    rebuilt, not merely that they were rebuilt the same way twice.
    """
    out = {}
    for dll_rel, _ in NATIVE_PAIRS:
        path = os.path.join(_ROOT, dll_rel)
        working = sha256(path) if os.path.exists(path) else None
        at_baseline = git_blob_sha256(baseline, dll_rel)
        at_head = git_blob_sha256("HEAD", dll_rel)
        out[os.path.basename(dll_rel)] = {
            "path": dll_rel,
            "working_tree_sha256": working,
            "baseline_commit": baseline,
            "sha256_at_baseline": at_baseline,
            "sha256_at_head": at_head,
            "matches_baseline": bool(working) and working == at_baseline,
            "matches_head": bool(working) and working == at_head,
            "unchanged_since_baseline": bool(working) and working == at_baseline == at_head,
            "last_commit_touching_binary": git("log", "-1", "--pretty=%h %s", "--", dll_rel),
        }
    return out


# ---------------------------------------------------------------------------
# §10-§21: was the machine ours, and was it quiet
# ---------------------------------------------------------------------------
#
# Two questions, and the round may not answer one with the other. A lock says nobody else was
# benchmarking; a quiet window says nothing else was loading the machine. The previous round had
# neither, which is why its 0.5% decision could not be read: its own spreads ran to tens of
# percent, and its unit suite took twice as long on some runs as on others.
#
# Both gates recompute from the artifact's raw samples instead of reading its verdict field. That
# is the same discipline §38 applies to ffprobe: a flag written beside a number is that program
# describing itself, and the one thing worth checking is whether the number says the same thing.
# `thresholds` is compared against the constants `benchmark_environment.THRESHOLDS` still holds, so
# "we did not tune the thresholds after seeing the results" is a check rather than a promise.

MANDATORY_LOCK_MECHANISMS = ("msvcrt.locking", "fcntl.flock")
LOCK_FILE_NAME = "bench_out/.transfer_benchmark.lock"
# §11's list of what a sample records, by the names §20's artifact uses for them.
# §2: the six things a sample has to say about every GPU client it named, so that a reader can check
# after the fact whether the process was the run's own at that moment rather than trusting whichever
# pid list happened to be lying around. A row missing one of them cannot be attributed, so the gate
# that decides "nobody else was on the card" refuses it.
GPU_ROW_EVIDENCE_FIELDS = ("sample_timestamp", "pid", "process_create_time", "parent_pid",
                           "classification", "is_current_benchmark_descendant")


def unattributed_gpu_rows(samples):
    """The GPU client rows in these samples that do not carry §2's six fields."""
    return [{"sample_utc": row.get("utc"),
             "missing": [field for field in GPU_ROW_EVIDENCE_FIELDS
                         if field not in entry]}
            for row in samples or [] for entry in (row.get("nvidia_compute_processes") or [])
            if any(field not in entry for field in GPU_ROW_EVIDENCE_FIELDS)][:3]


SAMPLE_FIELDS = ("utc", "system_cpu_percent", "available_ram_bytes", "gpu_utilization_percent",
                 "gpu_memory_utilization_percent", "gpu_temperature_celsius",
                 "gpu_graphics_clock_mhz", "gpu_sm_clock_mhz", "gpu_memory_clock_mhz",
                 "gpu_power_draw_watts", "gpu_pstate", "nvidia_compute_processes",
                 "repo_processes",
                 # §2: ownership is only checkable after the fact if the sample kept the tree it
                 # decided with, and said which of its processes that tree contained.
                 "tree_of_this_sample", "benchmark_descendant_pids")
# The five figures §12 thresholds, and the artifact's own published name for each. The pair is
# fixed here so a recomputation cannot be quietly pointed at four of them.
PUBLISHED_VS_RECOMPUTED = (("cpu_median", "cpu_median_percent"),
                           ("cpu_p95", "cpu_p95_percent"),
                           ("gpu_util_median", "gpu_utilization_median_percent"),
                           ("gpu_util_p95", "gpu_utilization_p95_percent"),
                           ("available_ram_min", "available_ram_min_bytes"))


def _gate_or_cannot_read(bench_env, build, artifact="benchmark_environment.json"):
    """Build an environment gate, or fail it for the reason that it could not be built.

    An absent artifact is a NOT_READY with a named reason, not a skipped key: §15's rule is that the
    round stops when the environment cannot be certified, and a gate that vanished with its input
    would be the one way for the round to continue without one.

    The gates that read a document hoist their data out of it before defining `build`, so they ask
    this function for the absent case first; `cannot_read_gate` is that answer on its own, for a
    round whose artifact never arrived.
    """
    if bench_env is None:
        return _key([(f"the_round_wrote_{artifact}", False, None),
                     ("the_gate_could_not_be_built", False,
                      {"reason": f"{artifact} is not in the round's artifacts"})],
                    [artifact], {"missing_artifact": artifact})
    try:
        return build(bench_env)
    except Exception as exc:  # a malformed artifact fails the gate and says how
        return _key([("the_artifact_could_be_read", False,
                      {"error": f"{type(exc).__name__}: {exc}"})], [artifact])


def cannot_read_gate(artifact):
    """The answer every environment gate gives when the artifact it judges was never written."""
    return _gate_or_cannot_read(None, None, artifact=artifact)


def bench_lock_gate(bench_env):
    """§6-§9: one holder, an OS-level mandatory lock, and a refusal that names the owner."""
    if bench_env is None:
        return cannot_read_gate("benchmark_environment.json")
    owner = bench_env.get("lock_owner") or {}
    # §9 names exactly what the holder's record must say. It is diagnostics, which is why nothing
    # about the lock's correctness depends on it - and why the gate still asks for all of it: a
    # refusal that could not name the holder is a refusal nobody can act on.
    owner_fields = ("pid", "started_utc", "code_under_test", "command", "hostname", "protocol")

    def build(doc):
        return _key([
            ("the_benchmark_declares_that_it_held_the_lock", doc.get("lock_acquired") is True,
             {"lock_acquired": doc.get("lock_acquired")}),
            ("the_lock_is_a_mandatory_os_lock",
             doc.get("lock_mechanism") in MANDATORY_LOCK_MECHANISMS,
             {"lock_mechanism": doc.get("lock_mechanism"),
              "accepted": list(MANDATORY_LOCK_MECHANISMS),
              "note": "a lockfile-existence check is not on this list, and cannot be made to be"}),
            ("the_lock_file_is_the_one_this_round_defines",
             str(doc.get("lock_path") or "").replace("\\", "/").endswith(LOCK_FILE_NAME),
             {"lock_path": doc.get("lock_path"), "expected_suffix": LOCK_FILE_NAME}),
            ("the_owner_record_carries_every_field_section_9_asks_for",
             all(bool(owner.get(field)) for field in owner_fields),
             {"required": list(owner_fields),
              "missing": sorted(field for field in owner_fields if not owner.get(field)),
              "owner": owner}),
            ("the_owner_is_a_process_and_a_moment",
             isinstance(owner.get("pid"), int) and bool(owner.get("started_utc")),
             {"pid": owner.get("pid"), "started_utc": owner.get("started_utc"),
              "hostname": owner.get("hostname")}),
        ], ["benchmark_environment.json lock_acquired / lock_owner / lock_path / lock_mechanism"],
            {"lock_record": {key: doc.get(key) for key in
                             ("lock_acquired", "lock_path", "lock_mechanism")}})
    return _gate_or_cannot_read(bench_env, build)


def quiet_preflight_gate(bench_env):
    """§11-§16: one full thirty-second window in which every frozen threshold held."""
    if bench_env is None:
        return cannot_read_gate("benchmark_environment.json")
    be = _environment_module()
    samples = list(bench_env.get("all_preflight_samples") or [])
    attempted = bench_env.get("preflight_window_count_attempted")
    passed = bench_env.get("preflight_window_that_passed")
    # The artifact's scalars describe the last window attempted, which is the passing one whenever
    # one passed: the loop stops the first time a window is accepted.
    window_of_record = [row for row in samples
                        if row.get("preflight_window_attempt") == attempted]
    # §15/§22's acceptance rule is about one window: the one in which every threshold held. A wait
    # that spent nine busy windows before an idle tenth was an idle wait when it started measuring,
    # and the failed windows stay in this artifact either way - so the conflicts that decide this
    # gate are the ones the selected window saw, not the union of everything the waiting passed
    # through. When nothing was selected there is no window to read, and every check below fails.
    selected = bench_env.get("selected_passing_window") or (
        None if not passed else (bench_env.get("windows") or [{}])[int(passed) - 1])

    def outside_the_selected_window(field):
        """What the window that was accepted saw on one of §13/§14's two conflict lists."""
        if not selected:
            return None
        return list(selected.get(field)
                    or (selected.get("summary") or {}).get(field) or [])

    recomputed = be.summarise(window_of_record)
    mismatched = [{
        "published_as": published, "published": bench_env.get(published),
        "recomputed_from_samples": recomputed.get(field),
        } for published, field in PUBLISHED_VS_RECOMPUTED
        if bench_env.get(published) != recomputed.get(field)]
    rejudged = be.judge_window(window_of_record) if window_of_record else {
        "ok": False, "failures": [{"reason": "no_samples_in_the_window_of_record", "detail": {}}]}

    def build(doc):
        return _key([
            ("a_full_thirty_second_window_was_measured",
             doc.get("preflight_window_length_seconds") == be.WINDOW_SECONDS
             and doc.get("preflight_sample_interval_seconds") == be.SAMPLE_INTERVAL_SECONDS
             and len(window_of_record) >= be.window_sample_count(),
             {"window_seconds": doc.get("preflight_window_length_seconds"),
              "interval_seconds": doc.get("preflight_sample_interval_seconds"),
              "samples_in_window_of_record": len(window_of_record),
              "samples_required": be.window_sample_count()}),
            ("the_thresholds_are_the_frozen_ones", doc.get("thresholds") == be.THRESHOLDS,
             {"recorded": doc.get("thresholds"), "frozen_in_the_module": be.THRESHOLDS,
              "note": "§12: fixed before any result was seen; a threshold changed to match a "
                      "measurement fails this check instead of passing the round"}),
            ("every_sample_records_what_section_11_asks_for",
             bool(samples) and not [
                 {"sample": index, "missing": [field for field in SAMPLE_FIELDS
                                               if field not in row]}
                 for index, row in enumerate(samples)
                 if any(field not in row for field in SAMPLE_FIELDS)][:1],
             {"samples": len(samples),
              "missing_from_the_first_bad_sample": next(
                  ({"sample": index, "missing": [field for field in SAMPLE_FIELDS
                                                 if field not in row]}
                   for index, row in enumerate(samples)
                   if any(field not in row for field in SAMPLE_FIELDS)), None)}),
            ("every_gpu_client_row_carries_the_identity_it_was_judged_on",
             bool(window_of_record) and not unattributed_gpu_rows(window_of_record),
             {"rows_missing_an_identity_field": unattributed_gpu_rows(window_of_record),
              "required": list(GPU_ROW_EVIDENCE_FIELDS),
              "note": "§2: ownership is a claim about one moment, so a sample that does not record "
                      "its own process tree cannot say who was in the way of it later"}),
            ("the_published_figures_are_the_ones_their_own_samples_give", not mismatched,
             {"mismatched": mismatched, "window_of_record": attempted}),
            ("the_window_of_record_passes_when_its_samples_are_rejudged", rejudged["ok"],
             {"failures_when_rejudged": rejudged["failures"],
              "note": "recomputed here from all_preflight_samples against the module's frozen "
                      "thresholds; `preflight_status` is the same function's own answer, and the "
                      "two agreeing is the check"}),
            ("the_artifact_and_the_rejudgement_agree",
             (doc.get("preflight_status") == "PASS") == bool(passed) == bool(rejudged["ok"]),
             {"preflight_status": doc.get("preflight_status"),
              "preflight_window_that_passed": passed,
              "rejudged_ok": rejudged["ok"]}),
            ("one_window_passed_within_the_ten_minute_budget",
             bool(passed) and doc.get("preflight_status") == "PASS"
             and bool(selected) and selected.get("attempt") == passed
             and (doc.get("preflight_window_count_attempted") or 0)
             * (doc.get("preflight_window_length_seconds") or 0)
             <= be.QUIET_WAIT_SECONDS + be.WINDOW_SECONDS
             and (selected.get("appeared_after_seconds") or 0.0) <= be.QUIET_WAIT_SECONDS,
             {"windows_attempted": attempted, "window_that_passed": passed,
              "selected_window_attempt": (selected or {}).get("attempt"),
              "selected_window_appeared_after_seconds": (selected or {}).get("appeared_after_seconds"),
              "budget_seconds_frozen_in_the_module": be.QUIET_WAIT_SECONDS,
              "budget_seconds_the_document_claims": doc.get("preflight_wait_budget_seconds"),
              "note": "§15: the wait is allowed to be long, the accepted window is not allowed to "
                      "arrive after the budget runs out. The budget is the module's, not the "
                      "document's: a round cannot widen the window it may certify by writing a "
                      "larger number into its own artifact"}),
            ("the_selected_window_itself_held_every_threshold",
             bool(selected) and not (selected.get("failures") or []),
             {"failures_of_the_selected_window": (selected or {}).get("failures"),
              "note": "this is the only conflict check §15 lets the gate make: the window the round "
                      "was certified on, not the ones the waiting refused"}),
            ("the_refused_windows_are_still_in_the_record",
             len(doc.get("windows") or []) == (attempted or 0)
             and not [row for row in (doc.get("windows") or [])
                      if not row.get("ok") and not (row.get("failures") or [])],
             {"windows_recorded": len(doc.get("windows") or []), "attempted": attempted,
              "union_external": doc.get("external_compute_processes_across_all_windows"),
              "union_repo": doc.get("conflicting_repo_processes_across_all_windows"),
              "note": "selecting one window is a rule about what the gate reads, not permission to "
                      "delete the history: every window that was refused has to still be here with "
                      "the reason it was refused"}),
            ("no_conflicting_repo_process_was_recorded",
             not outside_the_selected_window("conflicting_repo_processes"),
             {"processes": outside_the_selected_window("conflicting_repo_processes"),
              "across_the_whole_wait": doc.get("conflicting_repo_processes_across_all_windows"),
              "note": "recorded, not killed (§13); §15 stops the round instead"}),
            ("no_sustained_external_gpu_compute_workload_was_recorded",
             bool(selected) and "sustained_external_compute_processes" in (selected.get("summary")
                                                                          or {})
             and not outside_the_selected_window("sustained_external_compute_processes"),
             {"processes": outside_the_selected_window("sustained_external_compute_processes"),
              "everything_the_window_saw": outside_the_selected_window("external_compute_processes"),
              "transient_sightings_recorded_and_not_refused":
                  outside_the_selected_window(
                      "transient_external_compute_process_sightings"),
              "across_the_whole_wait": doc.get("external_compute_processes_across_all_windows"),
              "note": "§14 as revised: the refusal is the list on the left - the outsiders that stayed "
                      "three consecutive 1 Hz samples, which is §18's rule and now the window's rule "
                      "too. The two lists beside it are what the window saw and what it declined to "
                      "refuse, published because §5's other half is that an ambient or unproven client "
                      "has to be written down. A window with no `sustained_external_compute_processes` "
                      "field fails this rather than passing it: an absent list is indistinguishable "
                      "from an empty one, and that is exactly the difference this round exists to "
                      "close"}),
            ("nothing_the_gate_needs_was_unmeasurable",
             not [row for row in window_of_record
                  if row.get("system_cpu_percent") is None
                  or row.get("available_ram_bytes") is None
                  or row.get("gpu_utilization_percent") is None],
             {"samples_without_a_cpu_figure": sum(
                 1 for row in window_of_record if row.get("system_cpu_percent") is None),
              "samples_without_a_gpu_figure": sum(
                  1 for row in window_of_record if row.get("gpu_utilization_percent") is None),
              "samples_without_a_ram_figure": sum(
                  1 for row in window_of_record if row.get("available_ram_bytes") is None),
              "note": "§11: an unreadable field is recorded as unavailable, never as 0, and a "
                      "threshold nobody read cannot be certified against anything"}),
        ], ["benchmark_environment.json all_preflight_samples / thresholds / preflight_status "
            "/ selected_passing_window",
            "benchmark_environment.THRESHOLDS (re-read, not copied)"],
            {"recomputed_summary": recomputed,
             "selected_passing_window": (None if not selected else {
                 key: selected.get(key) for key in
                 ("attempt", "started_utc", "ended_utc", "sample_count", "appeared_after_seconds",
                  "within_wait_budget", "external_compute_processes",
                  "conflicting_repo_processes")}),
             "summary_of_record": (selected or {}).get("summary")
             or ((bench_env.get("windows") or [{}])[-1].get("summary"))})
    return _gate_or_cannot_read(bench_env, build)


def cpu_restriction_gate(audit, bench_env=None):
    """§6: was this machine even allowed to run at full speed while it was being measured?

    A low utilisation figure says the machine was idle, not that it was unbounded. A plan left at
    "maximum processor state 99%", an affinity mask covering half the cores, or a job object with a
    rate limit all let the benchmark run to completion at a third of the speed, so this gate reads
    the configuration the round ran inside rather than the load it saw, and it is deliberately not
    satisfied by the preflight having passed.

    The measurements are re-judged here rather than trusted: `cpu_restriction_ok` in the artifact is
    what the driver believed at the moment it wrote the file, and the whole point of §6 is that a
    belief about the machine is not evidence about it.
    """
    env = _environment_module()
    fields = list(env.CPU_RESTRICTION_FIELDS)
    state_keys = dict(env.CPU_RESTRICTION_FIELD_STATE_KEYS)
    missing_fields = [field for field in fields if field not in (audit or {})]
    # A field whose reading never says where it came from. The sentinel is a string no helper ever
    # returns, so an absent state key is reported by this gate instead of raising inside it.
    unnamed = [field for field in fields
               if (audit or {}).get(state_keys.get(field) or "", "<unsettled>")
               not in env.CPU_RESTRICTION_STATES]
    # §6's prohibition in one line: a refusal is written as null and named, never as a number,
    # because a 0 in a report about limits reads as "capped at nothing" - the opposite of "unknown".
    lied_about = sorted(field for field in fields
                        if (audit or {}).get(state_keys.get(field)) in env.CPU_RESTRICTION_REFUSALS
                        and (audit or {}).get(field) is not None)
    zeros = sorted(field for field in fields
                   if isinstance((audit or {}).get(field), (int, float))
                   and not isinstance((audit or {}).get(field), bool)
                   and (audit or {}).get(field) == 0)
    rejudged = env.judge_cpu_restriction(audit) if audit else []

    def build(doc):
        selected_start = None
        if bench_env:
            selected_start = ((bench_env.get("selected_passing_window") or {}).get("started_utc")
                              or (bench_env.get("all_preflight_samples") or [{}])[0].get("utc"))
        return _key([
            ("every_field_section_6_names_is_in_the_record", not missing_fields,
             {"missing": missing_fields, "required": fields}),
            ("every_reading_says_how_it_was_obtained", not unnamed,
             {"fields_without_a_known_state": unnamed,
              "recognised_states": list(env.CPU_RESTRICTION_STATES)}),
            ("a_field_the_machine_refused_to_answer_is_null_and_not_a_number", not lied_about,
             {"written_as_a_number_anyway": lied_about,
              "note": "§6: 无法取得的字段写 null / unavailable，禁止写 0"}),
            ("no_limit_is_reported_as_zero", not zeros,
             {"published_as_zero": zeros,
              "note": "none of these readings can legitimately be 0: a plan that allowed no "
                      "processor, an affinity of no cores, a job limit of nothing. A 0 here is an "
                      "unmeasured field wearing the shape of a measurement"}),
            ("the_three_hard_readings_are_present", not [
                field for field in ("system_logical_cpu_count", "process_cpu_affinity_count",
                                    "processor_max_state_percent") if doc.get(field) is None],
             {field: doc.get(field) for field in
              ("system_logical_cpu_count", "process_cpu_affinity_count",
               "processor_max_state_percent", "job_cpu_rate_limit_percent")}),
            ("the_recorded_verdict_is_the_one_its_own_measurements_give",
             rejudged == (doc.get("restrictions_detected") or [])
             and doc.get("cpu_restriction_ok") == (not rejudged),
             {"rejudged": rejudged, "published": doc.get("restrictions_detected"),
              "published_ok": doc.get("cpu_restriction_ok"),
              "note": "the same judge over the same document, run here instead of trusted there"}),
            ("every_detected_restriction_is_one_of_the_three_named_reasons",
             not [row for row in rejudged if row.get("reason")
                  not in env.CPU_RESTRICTION_REASONS],
             {"reasons": [row.get("reason") for row in rejudged],
              "named": list(env.CPU_RESTRICTION_REASONS)}),
            ("no_cpu_restriction_was_left_on_the_machine", not rejudged,
             {"restrictions": rejudged,
              "affinity": {"allowed": doc.get("process_cpu_affinity_count"),
                           "of": doc.get("system_logical_cpu_count")},
              "processor_max_state_percent": doc.get("processor_max_state_percent"),
              "job_cpu_rate_limit_percent": doc.get("job_cpu_rate_limit_percent"),
              "note": "§6's three hard gates. An unavailable advanced field - the job's rate limit, "
                      "EcoQoS - is recorded and never fails this on its own, because a refusal to "
                      "answer is not an accusation"}),
            ("the_audit_was_taken_before_the_measurement_it_certifies",
             bool(doc.get("audited_utc")) and not (
                 selected_start and str(doc.get("audited_utc")) > str(selected_start)),
             {"audited_utc": doc.get("audited_utc"),
              "first_quiet_preflight_sample_utc": selected_start,
              "note": "§11: the audit gates the start of the formal round, so a record written "
                      "afterwards describes a machine the round had already left behind"}),
        ], ["cpu_restriction_audit.json", "benchmark_environment.judge_cpu_restriction (re-read)"],
            {"audit": {key: doc.get(key) for key in fields},
             "states": {key: doc.get(state_keys[key]) for key in fields},
             "details": {key: doc.get(key) for key in doc
                         if key.endswith("_detail") or key.endswith("_state")},
             "restrictions_detected": rejudged})
    return _gate_or_cannot_read(audit, build, artifact="cpu_restriction_audit.json")


# ---------------------------------------------------------------------------
# §17-§19, §33-§34, §36: the three questions this round adds to the instrument's verdict
#
# All three read written artifacts and recompute the figure they judge, for the same reason the
# preflight gate re-judges its own window: a flag in an artifact records what a program believed at
# the moment it wrote the file, and this round exists because that belief was wrong twice.
# ---------------------------------------------------------------------------

# §34 froze these two numbers before the batch ran. They are literals here rather than arguments,
# and the gate below checks that they still say what the brief said - a threshold edited towards a
# result fails this round instead of rescuing it.
VARIANCE_SPREAD_PERCENT_AT_MOST = 20.0              # §33/§34: one execution's kept-run spread
VARIANCE_UNSTABLE_EXECUTION_SHARE_AT_MOST = 0.25    # §34: how many executions may be like that
VARIANCE_SPREAD_FORMULA = "(max - min) / median * 100"


def _iso_moment(text):
    """Parse a `2026-09-20T20:56:37Z` stamp, or return None when a record holds nothing parseable."""
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.datetime.strptime(str(text), fmt)
        except (TypeError, ValueError):
            continue
    return None


def kept_run_spreads(executions):
    """§33's spread for every (execution, order) pair, recomputed from that run's own FPS.

    §33 defines the figure over the *kept* runs - the warmup is not part of the sample a verdict is
    taken from - and divides by the median. The pooled benchmark publishes its own spread field
    divided by the lowest run, which is a larger number for the same data, so both are returned here
    and the gate says which one it is applying instead of comparing two formulas by accident.
    """
    rows = []
    for ex in executions:
        for mode in ("legacy", "reduced"):
            order = (ex.get("orders") or {}).get(mode) or {}
            kept = {int(run) for run in order.get("runs_used_for_medians") or []}
            fps = [value for key, value in (order.get("fps_of_every_run") or {}).items()
                   if str(key).isdigit() and int(key) in kept and value]
            median = statistics.median(fps) if fps else None
            rows.append({
                "execution": ex.get("execution"), "source_artifact": ex.get("source_artifact"),
                "mode": mode, "kept_runs": len(fps), "kept_run_ids": sorted(kept),
                "fps": sorted(fps),
                "spread_percent_vs_median": (round(100.0 * (max(fps) - min(fps)) / median, 2)
                                             if len(fps) >= 2 and median else None),
                "spread_percent_vs_lowest_run": (round(100.0 * (max(fps) - min(fps)) / min(fps), 2)
                                                 if len(fps) >= 2 and min(fps) else None),
                "published_spread_percent": order.get("inter_run_spread_percent_of_kept_runs"),
            })
    return rows


def pooled_kept_median(executions, mode):
    """The pooled median of one order's kept runs, recomputed from the runs behind it."""
    fps = []
    for ex in executions:
        order = (ex.get("orders") or {}).get(mode) or {}
        kept = {int(run) for run in order.get("runs_used_for_medians") or []}
        fps += [value for key, value in (order.get("fps_of_every_run") or {}).items()
                if str(key).isdigit() and int(key) in kept and value]
    return (round(statistics.median(fps), 4), len(fps)) if fps else (None, 0)


def runtime_contamination_gate(runtime):
    """§17-§19: who else was on the machine while the batch ran, and what that says about it."""
    if runtime is None:
        return cannot_read_gate("benchmark_runtime_monitor.json")
    be = _environment_module()
    samples = list(runtime.get("samples") or [])
    events = list(runtime.get("external_workload_events") or [])
    children = {int(pid) for pid in runtime.get("benchmark_child_pids") or []}
    ffmpeg_children = {int(pid) for pid in runtime.get("ffmpeg_child_pids") or []}
    gpu_events = [row for row in events if row.get("kind") == "gpu_compute"]
    # Re-judge the recorded samples with the module's own ledger: the artifact's flag is its claim,
    # this is the same rule applied to its own evidence, and the two disagreeing is the finding.
    ledger = be.judge_contamination(samples) if samples else None
    # §17's rule is behavioural, so it is checked behaviourally: the same samples, with the CPU and
    # GPU utilisation of every one of them pinned at 100%, must give the same answer. A monitor that
    # failed a batch for loading the machine fails here.
    loaded = [{**row, "system_cpu_percent": 100.0, "gpu_utilization_percent": 100.0,
               "gpu_memory_utilization_percent": 100.0} for row in samples]
    ledger_under_load = be.judge_contamination(loaded) if loaded else None
    started = _iso_moment(runtime.get("measurement_start"))
    ended = _iso_moment(runtime.get("measurement_end"))
    elapsed = (ended - started).total_seconds() if started and ended else None

    def build(doc):
        interval = doc.get("sample_interval_seconds") or be.SAMPLE_INTERVAL_SECONDS
        return _key([
            ("the_monitor_recorded_the_length_of_the_batch",
             bool(samples) and elapsed is not None and elapsed > 0
             and len(samples) >= (elapsed // interval) * 0.5,
             {"samples": len(samples), "measurement_start": doc.get("measurement_start"),
              "measurement_end": doc.get("measurement_end"), "elapsed_seconds": elapsed,
              "interval_seconds": interval,
              "note": "a monitor that stopped early, or never started, cannot say the batch was "
                      "clean - which is why this compares a count of samples against a span of "
                      "time instead of reading a flag"}),
            ("the_sample_count_and_the_sample_list_agree",
             doc.get("sample_count") == len(samples),
             {"sample_count": doc.get("sample_count"), "list_length": len(samples),
              "note": "the same class of mistake as §1's ffprobe aggregate: a scalar and a list "
                      "written under one key, and whichever was written last won"}),
            ("the_monitor_samples_at_one_hertz", interval == be.SAMPLE_INTERVAL_SECONDS,
             {"recorded": interval, "required_by_section_17": be.SAMPLE_INTERVAL_SECONDS}),
            ("every_sample_names_who_was_on_the_machine",
             bool(samples) and not [row for row in samples
                                    if "conflicting_repo_processes" not in row
                                    or "external_compute_processes" not in row
                                    or "tree_of_this_sample" not in row],
             {"samples_missing_a_list": sum(1 for row in samples
                                            if "conflicting_repo_processes" not in row
                                            or "external_compute_processes" not in row
                                            or "tree_of_this_sample" not in row),
              "note": "§2: the two outsider lists and the tree they were judged against, in every "
                      "sample - a verdict without the snapshot it was reached from is the thing this "
                      "round exists to stop"}),
            ("three_consecutive_samples_is_the_rule_that_was_applied",
             doc.get("contamination_consecutive_samples_required")
             == be.CONTAMINATION_CONSECUTIVE_SAMPLES,
             {"recorded": doc.get("contamination_consecutive_samples_required"),
              "in_the_module": be.CONTAMINATION_CONSECUTIVE_SAMPLES,
              "note": "§18: one transient process is not contamination; an outsider that stays is"}),
            ("the_record_says_the_same_when_its_own_samples_are_rejudged",
             ledger is not None and bool(ledger.contaminated)
             == bool(doc.get("runtime_contaminated")),
             {"published": doc.get("runtime_contaminated"),
              "recomputed": None if ledger is None else ledger.contaminated,
              "recomputed_reason": None if ledger is None else ledger.reason,
              "published_events": len(events)}),
            ("utilisation_was_not_used_to_judge_the_batch",
             doc.get("utilisation_is_not_a_failure_threshold") is True
             and ledger_under_load is not None
             and bool(ledger_under_load.contaminated) == bool(doc.get("runtime_contaminated")),
             {"contaminated_with_every_sample_at_full_utilisation":
                  None if ledger_under_load is None else ledger_under_load.contaminated,
              "published": doc.get("runtime_contaminated"),
              "note": "§17: the benchmark loading the machine *is* the measurement, so the monitor "
                      "may not read CPU or GPU utilisation as a failure. Re-running the recorded "
                      "samples with utilisation pinned to 100 must not change the verdict."}),
            ("the_benchmarks_own_children_were_never_counted_as_contamination",
             not ({int(row.get("pid") or -1) for row in events} & children),
             {"events_inside_the_tree": [(row.get("pid"), row.get("kind")) for row in events
                                         if int(row.get("pid") or -1) in children],
              "benchmark_children": len(children), "ffmpeg_children": len(ffmpeg_children),
              "note": "§19: this batch spawns FFmpeg on purpose. Its child pids are in the "
                      "artifact so a reader can see that the transcoder in the GPU table is the one "
                      "the batch started, not a neighbour's."}),
            ("only_the_three_conflict_classes_ever_contaminated_a_batch",
             not [row for row in gpu_events if row.get("classification")
                  not in be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS],
             {"events": [(row.get("pid"), row.get("name"), row.get("classification"))
                         for row in gpu_events],
              "may_stop_a_batch": list(be.GPU_CLIENT_CONFLICT_CLASSIFICATIONS),
              "may_never_stop_a_batch": list(be.GPU_CLIENT_NON_TRIGGERING_CLASSIFICATIONS),
              "ambient_clients_recorded_in_samples": sum(
                  (row.get("ambient_os_graphics_client_count") or 0)
                  + (row.get("ambient_or_unproven_gpu_client_count") or 0) for row in samples),
              "note": "§5/§13: the two classes on the right are written down and left alone - their "
                      "existence is not contamination, and neither arriving in the table nor an "
                      "unreadable identity puts a row into the left-hand list any more. Every event in "
                      "this list has to name which of the three conflict classes raised it, because an "
                      "event with no class is an event somebody blamed without saying why"}),
            ("the_batch_was_stopped_when_the_monitor_said_it_was_contaminated",
             not doc.get("runtime_contaminated")
             or bool((doc.get("batch") or {}).get("stopped_because_of_contamination")),
             {"runtime_contaminated": doc.get("runtime_contaminated"), "batch": doc.get("batch"),
              "note": "§18: keep the completed executions and do not reroll, but stop launching "
                      "further ones once the running child has finished safely"}),
            ("nothing_that_contaminated_the_round_was_deleted_from_it",
             all(bool(row.get("first_seen_utc")) and (row.get("sightings") or 0) > 0
                 for row in events),
             {"events": events,
              "note": "every outsider the batch saw stays in this file, whether or not it stayed "
                      "long enough to count (§18)"}),
            ("the_round_is_not_reading_a_contaminated_batch_as_clean",
             doc.get("runtime_contaminated") is False,
             {"runtime_contaminated": doc.get("runtime_contaminated"),
              "contamination_reason": doc.get("contamination_reason"),
              "not_ready_reason": "external_workload_during_measurement"
              if doc.get("runtime_contaminated") else None}),
        ], ["benchmark_runtime_monitor.json samples / external_workload_events / "
            "runtime_contaminated / benchmark_child_pids",
            "benchmark_environment.judge_contamination (re-applied to the recorded samples)"],
            {"contamination": {key: doc.get(key) for key in
                               ("runtime_contaminated", "contamination_reason",
                                "first_contamination_timestamp")},
             "recomputed": {"contaminated": None if ledger is None else ledger.contaminated,
                            "events": None if ledger is None else ledger.event_rows()},
             "monitor_span": {"start": doc.get("measurement_start"),
                              "end": doc.get("measurement_end"), "seconds": elapsed},
             "children": {"benchmark": len(children), "ffmpeg": sorted(ffmpeg_children)[:8]}})
    return _gate_or_cannot_read(runtime, build, artifact="benchmark_runtime_monitor.json")


def bench_variance_gate(perf):
    """§33-§34: was the instrument stable enough to read - judged without deleting anything."""
    executions = (perf or {}).get("executions") or []
    primary = [ex for ex in executions if not ex.get("unpoolable_reasons")]
    rows = kept_run_spreads(primary)
    by_execution = {}
    for row in rows:
        by_execution.setdefault(row.get("execution"), []).append(row)
    over = sorted(str(name) for name, group in by_execution.items()
                  if any((row.get("spread_percent_vs_median") or 0.0)
                         > VARIANCE_SPREAD_PERCENT_AT_MOST for row in group))
    share = round(len(over) / len(by_execution), 4) if by_execution else None
    stable = bool(by_execution) and share is not None \
        and share <= VARIANCE_UNSTABLE_EXECUTION_SHARE_AT_MOST
    verdict = (perf or {}).get("verdict") or {}
    census = verdict.get("run_census") or {}
    expected_kept = 2 * len(primary)   # §26: three pairs per execution, the first a warmup

    def build(doc):
        return _key([
            ("no_execution_was_dropped_from_the_variance_judgement",
             bool(primary) and len(primary) == len(executions),
             {"usable": len(primary), "pooled": len(executions),
              "rejected": verdict.get("rejected_executions"),
              "note": "§28/§33: an execution is not dropped because its spread was ugly"}),
            ("each_order_of_each_execution_has_enough_kept_runs_to_have_a_spread",
             all((row.get("kept_runs") or 0) >= 2 and row.get("spread_percent_vs_median")
                 is not None for row in rows),
             {"rows_without_a_spread": [row for row in rows
                                        if (row.get("kept_runs") or 0) < 2
                                        or row.get("spread_percent_vs_median") is None]}),
            ("the_spread_is_measured_by_the_definition_this_round_froze",
             all(row.get("published_spread_percent") is None
                 or (row.get("spread_percent_vs_median") or 0.0)
                 <= (row.get("spread_percent_vs_lowest_run") or 0.0) + 1e-9 for row in rows),
             {"gate_formula": VARIANCE_SPREAD_FORMULA,
              "what_the_benchmark_publishes_beside_it": "(max - min) / min * 100",
              "widest_pair": max(((str(row.get("execution")), row.get("mode"),
                                   row.get("spread_percent_vs_median"),
                                   row.get("published_spread_percent"))
                                  for row in rows if row.get("published_spread_percent")
                                  is not None),
                                 key=lambda t: (t[3] or 0) - (t[2] or 0), default=None),
              "note": "the two figures differ only in the denominator, and dividing by the median "
                      "is the smaller of the two; the gate applies §33's definition and says so "
                      "rather than choosing whichever number suits the answer"}),
            ("the_variance_thresholds_are_the_frozen_ones",
             VARIANCE_SPREAD_PERCENT_AT_MOST == 20.0
             and VARIANCE_UNSTABLE_EXECUTION_SHARE_AT_MOST == 0.25,
             {"kept_run_spread_percent_at_most": VARIANCE_SPREAD_PERCENT_AT_MOST,
              "share_of_executions_allowed_over_that": VARIANCE_UNSTABLE_EXECUTION_SHARE_AT_MOST,
              "note": "§34 fixed both before any measurement was seen"}),
            ("no_run_was_dropped_to_make_the_variance_look_stable",
             all((census.get(mode) or {}).get("kept_run_count") == expected_kept
                 for mode in ("legacy", "reduced")),
             {"kept_run_count": {mode: (census.get(mode) or {}).get("kept_run_count")
                                 for mode in ("legacy", "reduced")},
              "expected": expected_kept, "executions": len(primary),
              "note": "§26 fixes two kept runs per order per execution, so a pool of N executions "
                      "carries 2N kept runs of each order; one fewer is a run somebody removed"}),
            ("benchmark_variance_stable", stable,
             {"executions_over_the_spread": over, "share_over_the_spread": share,
              "executions_judged": len(by_execution),
              "spreads_by_execution": {
                  str(name): {row["mode"]: row["spread_percent_vs_median"] for row in group}
                  for name, group in sorted(by_execution.items(), key=lambda kv: str(kv[0]))},
              "not_ready_reason": "benchmark_variance_unstable" if not stable else None}),
        ], ["counterbalanced_performance.json executions[].orders[].fps_of_every_run and "
            "runs_used_for_medians (recomputed, not read from a flag)",
            "counterbalanced_performance.json verdict.run_census"],
            {"spreads": rows, "benchmark_variance_stable": stable,
             "share_over_threshold": share, "executions_over_threshold": over,
             "thresholds": {"spread_percent": VARIANCE_SPREAD_PERCENT_AT_MOST,
                            "execution_share": VARIANCE_UNSTABLE_EXECUTION_SHARE_AT_MOST}})
    return _gate_or_cannot_read(perf, build, artifact="counterbalanced_performance.json")


def corroboration_consistency_gate(perf, corroborating=None, runtime=None, bench_env=None):
    """§36: a second frame length either agrees with the first or makes the answer NOT_READY."""
    primary_ex = [ex for ex in (perf or {}).get("executions") or []
                  if not ex.get("unpoolable_reasons")]
    corroboration = [ex for ex in (perf or {}).get("corroborating_executions") or []
                     if not ex.get("unpoolable_reasons")]
    own = [ex for ex in ((corroborating or {}).get("executions") or [])
           if not ex.get("unpoolable_reasons")]
    verdict = (perf or {}).get("verdict") or {}
    frames = {ex.get("frames_per_run") for ex in primary_ex}
    other_frames = {ex.get("frames_per_run") for ex in corroboration}
    l60, n60 = pooled_kept_median(primary_ex, "legacy")
    r60, _ = pooled_kept_median(primary_ex, "reduced")
    l150, n150 = pooled_kept_median(corroboration, "legacy")
    r150, _ = pooled_kept_median(corroboration, "reduced")
    l_own, n_own = pooled_kept_median(own, "legacy")
    r_own, _ = pooled_kept_median(own, "reduced")
    published_own = ((corroborating or {}).get("verdict") or {}).get("pooled_kept_median_fps") or {}

    def sign(reduced, legacy):
        if not reduced or not legacy:
            return None
        return 1 if reduced > legacy else -1 if reduced < legacy else 0

    def fps_of(ex):
        # A corroboration is its own experiment: its own runs, its own ratio, its own verdict. What
        # §36 forbids is pooling the two lengths, so nothing here borrows a 150-frame pair and
        # calls it evidence about the 60-frame one.
        return {mode: ((ex.get("orders") or {}).get(mode) or {}).get("fps_of_every_run") or {}
                for mode in ("legacy", "reduced")}

    sixty, one_fifty = sign(r60, l60), sign(r150, l150)
    corroboration_ratios = []
    for ex in corroboration:
        runs = fps_of(ex)
        kept = {int(run) for run in ((ex.get("orders") or {}).get("reduced") or {})
                .get("runs_used_for_medians") or []}
        for key, reduced in runs["reduced"].items():
            legacy = runs["legacy"].get(key)
            if key.isdigit() and int(key) in kept and reduced and legacy:
                corroboration_ratios.append(reduced / legacy)
    corroboration_pairwise = (round(statistics.median(corroboration_ratios), 4)
                              if corroboration_ratios else None)
    balance_rows = [row for row in (verdict.get("corroboration") or {}).values()
                    if isinstance(row, dict) and row.get("kept_pair_positions")]
    balance = {key: sum((row["kept_pair_positions"] or {}).get(key) or 0 for row in balance_rows)
               for key in ("legacy_first", "reduced_first", "pairs")}
    balance["balanced"] = (balance["legacy_first"] == balance["reduced_first"] > 0
                           and not any((row["kept_pair_positions"] or {}).get("unknown_position")
                                       for row in balance_rows))
    agrees = sixty is not None and one_fifty is not None and sixty == one_fifty
    # §4: the reason the corroboration did not settle the question is a fact about which batch ran,
    # not a synonym for disagreement. The counts it needs come from this round's own artifacts - the
    # monitor's batch record names how many executions the protocol defines - so the gate is reading
    # what the round committed to rather than what a caller of this function remembers.
    batch = (runtime or {}).get("batch") or {}
    env = _environment_module()
    reason = env.corroboration_reason(
        primary_execution_count=len(primary_ex),
        primary_expected=batch.get("primary_executions_defined"),
        corroborating_execution_count=len(corroboration),
        corroborating_expected=batch.get("corroborating_executions_defined"),
        primary_direction=sixty, corroborating_direction=one_fifty,
        runtime_contaminated=bool((runtime or {}).get("runtime_contaminated")),
        preflight_passed=(bench_env or {}).get("preflight_status") != "FAIL")
    # The driver labels the launch decision when it makes it, before any median exists; this gate
    # labels it again from the artifacts on disk. The two are the same fact seen at different
    # moments, so they have to agree - a round whose prose says one thing and whose verdict says
    # another is the thing §4 is about.
    published_reason = batch.get("corroborating_batch_reason")
    reason_named_twice = published_reason == (None if corroboration else reason)

    def build(doc):
        return _key([
            ("the_corroborating_batch_ran_at_its_own_frame_length",
             bool(corroboration) and bool(other_frames) and not (other_frames & frames),
             {"primary_frames": sorted(str(f) for f in frames),
              "corroborating_frames": sorted(str(f) for f in other_frames),
              "corroborating_executions": len(corroboration)}),
            ("the_corroborating_executions_were_never_pooled_into_the_primary_verdict",
             n60 == 2 * len(primary_ex) and n150 == 2 * len(corroboration) and bool(primary_ex),
             {"primary_kept_runs_per_order": n60, "primary_executions": len(primary_ex),
              "corroborating_kept_runs_per_order": n150,
              "corroborating_executions": len(corroboration),
              "note": "§30: a 150-frame run is not another sample of the 60-frame experiment, so a "
                      "primary pool that carried more than two kept runs per order per execution "
                      "would have mixed the two"}),
            ("the_two_files_describe_the_same_150_frame_runs",
             bool(own) and r150 == r_own and l150 == l_own,
             {"as_corroboration_in_the_primary_file": {"legacy": l150, "reduced": r150,
                                                       "kept_runs": n150},
              "as_its_own_primary_batch": {"legacy": l_own, "reduced": r_own, "kept_runs": n_own},
              "published_by_its_own_verdict": published_own,
              "kept_run_rows_identical": sorted(ex.get("source_artifact") or ""
                                                for ex in corroboration)
              == sorted(ex.get("source_artifact") or "" for ex in own)}),
            ("the_corroborating_batch_was_counterbalanced_too",
             balance["pairs"] > 0 and balance["balanced"] is True,
             {"balance": balance,
              "expected_by_section_30": {"pairs": 8, "legacy_first": 4, "reduced_first": 4}}),
            ("the_two_lengths_point_the_same_way", agrees,
             {"60f": {"legacy_kept_median_fps": l60, "reduced_kept_median_fps": r60,
                      "direction": sixty, "passes_the_primary_rule": bool(r60 and l60
                                                                          and r60 >= l60)},
              "150f": {"legacy_kept_median_fps": l150, "reduced_kept_median_fps": r150,
                       "direction": one_fifty,
                       "pairwise_median_ratio": corroboration_pairwise,
                       "passes_the_primary_rule": bool(r150 and l150 and r150 >= l150),
                       "pairwise_rule_passed": bool(corroboration_pairwise
                                                     and corroboration_pairwise >= 1.0)},
              "not_ready_reason": None if agrees else reason,
              "corroboration_status": reason,
              "primary_executions_defined": batch.get("primary_executions_defined"),
              "corroborating_executions_defined": batch.get("corroborating_executions_defined"),
              "note": "§36: 60f passing with 150f failing is NOT_READY, and so is the other way "
                      "round. Only agreement lets the round be DONE_PROVEN; neither length may "
                      "rescue the other. The reason names what happened rather than what a "
                      "comparison would have said: a corroboration that never ran because the batch "
                      "was contaminated is not a corroboration that disagreed, and `conflicts` is "
                      "reserved for two complete batches pointing different ways"}),
            ("the_round_named_its_corroboration_the_same_way_twice", reason_named_twice,
             {"published_by_the_driver_at_launch": published_reason,
              "derived_here_from_the_artifacts": reason,
              "corroborating_executions_found": len(corroboration),
              "note": "§4: the driver labels the launch decision before any median exists and the "
                      "gate labels it again afterwards, so a corroboration that never ran cannot be "
                      "described as a conflict by whichever of the two is read"}),
        ], ["counterbalanced_performance.json executions[] / corroborating_executions[] "
            "(kept-run fps recomputed from each run)",
            "counterbalanced_performance.json verdict.corroboration[].kept_pair_positions",
            "counterbalanced_150f.json verdict.pooled_kept_median_fps"],
            {"primary": {"legacy": l60, "reduced": r60, "sign": sixty},
             "corroborating": {"legacy": l150, "reduced": r150, "sign": one_fifty,
                               "pairwise_median_ratio": corroboration_pairwise},
             "corroboration_balance": balance,
             "corroboration_reason": reason,
             "reason_is_one_of_the_named_six": reason in env.CORROBORATION_REASONS})
    return _gate_or_cannot_read(perf, build, artifact="counterbalanced_performance.json")


def build_bench_fix(out_dir, required=BENCH_FIX_REQUIRED_ARTIFACTS, key_order=None,
                    round_name="Transfer benchmark evidence protocol fix",
                    baseline=BENCH_FIX_BASELINE, unit_suite_floor=273, extra_executions=()):
    """The instrument status file: one key per question, each decided by reading artifacts.

    Both benchmark rounds are assembled here because the questions are the same ones - was the
    warmup kept out of the pool, was the schedule balanced, did the outputs match - and a second
    copy of that code is a second chance for the two rounds to disagree about what a word means.
    The stability round passes its own required-artifact list and key order; §37's provenance gate
    is shared by both. `extra_executions` names the subdirectories of raw per-execution artifacts
    beyond `ab_executions`, so the stability round's 150-frame batch is covered by the same
    provenance rule as its 60-frame one rather than escaping it by living in another folder.
    """
    prov = bench_provenance(out_dir, required, extra_executions)
    documents = prov["documents"]
    perf = documents.get("counterbalanced_performance")
    protocol = documents.get("protocol")
    corroboration = documents.get("counterbalanced_150f")
    history = documents.get("historical_warmup_recalc")
    ffprobe = documents.get("ffprobe_output")
    env = documents.get("environment")
    consistency_doc = documents.get("status_consistency")
    pytest_line, _ = _transcript(out_dir, "pytest_output.txt")

    named = {name: doc for name, doc in documents.items() if doc is not None}
    sha = prov["shared_sha"]
    coverage = prov["coverage"]

    verdict = (perf or {}).get("verdict") or {}
    census = verdict.get("run_census") or {}
    balance = verdict.get("order_balance") or {}
    strata = verdict.get("position_strata") or {}
    kept_orders = verdict.get("kept_orders") or {}
    all_orders = verdict.get("all_orders_diagnostic") or {}
    per_execution = verdict.get("per_execution_outcome") or {}
    executions = (perf or {}).get("executions") or []
    usable = [ex for ex in executions if not ex.get("unpoolable_reasons")]
    failures = (perf or {}).get("execution_failures") or []
    ab = _ab_module()

    rows = {mode: {"kept": (perf or {}).get("kept_runs_per_order", {}).get(mode, []),
                   "warm": (perf or {}).get("warmup_runs_per_order", {}).get(mode, []),
                   "all": (perf or {}).get("all_runs_per_order", {}).get(mode, [])}
            for mode in ("legacy", "reduced")}

    checks = {}

    checks["artifact_provenance"] = _key([
        ("every_required_artifact_was_written", not prov["missing_required"],
         prov["missing_required"]),
        ("every_artifact_declares_its_code_under_test", not coverage["without_a_sha"],
         coverage["without_a_sha"]),
        ("no_artifact_names_a_different_commit", not coverage["mismatched"],
         coverage["mismatched"]),
        ("all_of_them_name_one_single_commit", bool(sha),
         {"shared_sha": sha,
          "declared": sorted({v for v in coverage["declared_shas"].values() if v})}),
        ("that_commit_is_the_one_the_runs_were_made_at",
         bool(sha) and sha == prov["primary_code_under_test"],
         {"primary": prov["primary_code_under_test"], "shared": sha}),
    ], ["code_under_test in every required artifact and in every execution artifact",
        "counterbalanced_performance.json code_under_test"],
        {"artifact_sha_coverage": coverage})

    # §38: ffprobe is the only witness here that is not the pipeline describing itself, so the key
    # that covers it re-reads ffprobe's numbers rather than trusting the flags written beside them.
    # The first two checks exist because of the bug this round was told to fix: the aggregate used to
    # read a field name the records never carried, publish `[null]`, and count matched frame totals on
    # a second field that was not there either. A missing key is not a measurement, and it no longer
    # gets to look like one.
    probe_rows_missing = (ffprobe or {}).get("runs_without_an_ffprobe_reading")
    codecs_seen = list((ffprobe or {}).get("codecs_seen") or [])
    expected_codecs = list((ffprobe or {}).get("expected_codecs") or [])
    codecs_per_order = (ffprobe or {}).get("codecs_seen_per_order") or {}
    probe_fields = set((ffprobe or {}).get("probe_check_fields_seen") or [])
    rows_disagreeing_on_frames = (ffprobe or {}).get("runs_whose_probed_frame_count_disagrees") or []
    checks["ffprobe_schema"] = _key([
        ("the_merged_ffprobe_artifact_declares_its_code_under_test", bool(_sha_of(ffprobe)),
         {"code_under_test": _sha_of(ffprobe),
          "of_each_merged_source": (ffprobe or {}).get(
              "code_under_test_of_each_merged_source")}),
        ("every_output_file_has_an_ffprobe_reading",
         ffprobe is not None and bool((ffprobe or {}).get("runs_with_an_ffprobe_reading"))
         and not probe_rows_missing,
         {"runs_with_a_reading": (ffprobe or {}).get("runs_with_an_ffprobe_reading"),
          "runs_without_one": probe_rows_missing,
          "per_mode": (ffprobe or {}).get("runs_per_mode")}),
        ("codecs_seen_names_codecs_and_not_blanks",
         bool(codecs_seen) and all(codecs_seen),
         {"codecs_seen": codecs_seen,
          "rows_without_a_codec": (ffprobe or {}).get("rows_without_a_codec") or []}),
        ("the_observed_codecs_are_the_ones_the_encoder_was_set_to",
         bool(expected_codecs) and codecs_seen == expected_codecs,
         {"codecs_seen": codecs_seen, "expected": expected_codecs,
          "expected_variants": (ffprobe or {}).get("expected_variants")}),
        ("both_orders_were_encoded_the_same_way",
         len({tuple(rows) for rows in codecs_per_order.values()} or [tuple()]) == 1
         and all(codecs_per_order.get(mode) for mode in ("legacy", "reduced")),
         codecs_per_order),
        ("every_probe_ran_every_check_it_is_asked_to",
         set(PROBE_CHECK_FIELDS) <= probe_fields,
         {"fields_seen": sorted(probe_fields), "required": list(PROBE_CHECK_FIELDS)}),
        ("no_probe_record_carried_no_checks_at_all",
         not ((ffprobe or {}).get("runs_whose_probe_records_no_checks") or []),
         (ffprobe or {}).get("runs_whose_probe_records_no_checks")),
        ("no_probe_check_failed", not (ffprobe or {}).get("rows_failing_a_probe_check"),
         (ffprobe or {}).get("rows_failing_a_probe_check")),
        ("every_probed_frame_count_matched_its_own_expected_count",
         bool((ffprobe or {}).get("runs_whose_probed_frame_count_matches"))
         and not rows_disagreeing_on_frames,
         {"matching": (ffprobe or {}).get("runs_whose_probed_frame_count_matches"),
          "disagreeing": rows_disagreeing_on_frames}),
        ("every_output_has_the_resolution_and_frame_rate_it_claimed",
         not ((ffprobe or {}).get("rows_with_the_wrong_geometry") or []),
         {"geometries_seen": (ffprobe or {}).get("geometry_seen"),
          "mismatched": (ffprobe or {}).get("rows_with_the_wrong_geometry")}),
    ], ["ffprobe_output.json codecs_seen / expected_codecs / runs_by_batch",
        "ab_executions*/*_ffprobe_output.json runs[].ffprobe",
        "collect_transfer_ab.EXPECTED_OUTPUT_CODEC"],
        {"codecs_seen_per_execution": (ffprobe or {}).get("codecs_seen_per_execution")})

    # §10-§21, and the reason this round was called at all: the two claims about the machine, each
    # read out of the artifact that records it and re-derived from that artifact's own samples.
    bench_env = documents.get("benchmark_environment")
    checks["benchmark_lock"] = bench_lock_gate(bench_env)
    checks["quiet_preflight"] = quiet_preflight_gate(bench_env)
    # Round C §6: what the machine was allowed to do, as opposed to what it was busy with. Asked
    # here, beside the lock and the waiting, because it is the third claim about the environment and
    # the only one neither of the other two can substitute for.
    if key_order and "cpu_restriction" in key_order:
        checks["cpu_restriction"] = cpu_restriction_gate(
            documents.get("cpu_restriction_audit"), bench_env)
    # §17-§21 watched the machine while the batch itself ran, §33-§34 asks whether the batch was
    # steady enough to read, and §36 asks whether a second frame length says the same thing as the
    # first. All three can only be decided from artifacts this round writes, so the previous round's
    # status file has no keys for them and they are built only when they are asked for.
    if key_order and "runtime_contamination" in key_order:
        checks["runtime_contamination"] = runtime_contamination_gate(
            documents.get("benchmark_runtime_monitor"))
        checks["benchmark_variance"] = bench_variance_gate(perf)
        checks["corroboration_consistency"] = corroboration_consistency_gate(
            perf, corroboration, runtime=documents.get("benchmark_runtime_monitor"),
            bench_env=bench_env)

    checks["warmup_exclusion"] = _key([
        ("warmup_runs_were_measured_and_published",
         all(rows[mode]["warm"] for mode in rows),
         {mode: len(rows[mode]["warm"]) for mode in rows}),
        ("warmup_rows_are_flagged_out_of_the_verdict",
         all(row.get("included_in_performance_verdict") is False
             for mode in rows for row in rows[mode]["warm"]), None),
        ("kept_rows_are_flagged_into_the_verdict",
         all(row.get("included_in_performance_verdict") is True
             for mode in rows for row in rows[mode]["kept"]), None),
        ("no_run_is_both_warmup_and_kept",
         all(not ({row["run_index"] for row in rows[mode]["warm"]}
                  & {row["run_index"] for row in rows[mode]["kept"]}) for mode in rows), None),
        ("the_execution_named_its_own_kept_run_ids",
         all(sorted((ex.get("orders") or {}).get(mode, {}).get("runs_used_for_medians") or [])
             == ab.kept_run_ids(int(ex.get("runs_per_mode") or 0))
             for ex in usable for mode in ("legacy", "reduced")),
         {mode: sorted({tuple((ex.get("orders") or {}).get(mode, {})
                              .get("runs_used_for_medians") or []) for ex in usable})
          for mode in ("legacy", "reduced")}),
    ], ["counterbalanced_performance.json run_census",
        "counterbalanced_performance.json kept_runs_per_order",
        "counterbalanced_performance.json warmup_runs_per_order",
        "executions[].orders[].runs_used_for_medians"],
        {"run_census": census})

    medians_match = all(
        kept_orders.get(mode, {}).get("median_fps")
        == round(statistics.median([row["fps"] for row in rows[mode]["kept"] if row["fps"]]), 4)
        for mode in ("legacy", "reduced") if rows[mode]["kept"])
    counts_are_not_the_old_ones = all(
        census.get(mode, {}).get("kept_run_count")
        < census.get(mode, {}).get("all_run_count", 0) for mode in ("legacy", "reduced"))
    checks["kept_run_pooling"] = _key([
        ("pooled_median_recomputed_from_the_kept_rows", medians_match,
         {mode: kept_orders.get(mode, {}).get("median_fps") for mode in kept_orders}),
        ("the_verdict_does_not_read_the_warmup_runs", counts_are_not_the_old_ones,
         {mode: f"{census.get(mode, {}).get('kept_run_count')} of "
                f"{census.get(mode, {}).get('all_run_count')}" for mode in ("legacy", "reduced")}),
        ("kept_and_all_run_counts_are_both_published",
         all(k in census.get(mode, {}) for mode in ("legacy", "reduced")
             for k in ("all_run_count", "kept_run_count", "warmup_run_count")), None),
        ("the_all_runs_figure_is_labelled_a_diagnostic",
         bool(verdict.get("pooled_all_runs_median_fps_diagnostic"))
         and all(all_orders.get(mode, {}).get("run_count")
                 == census.get(mode, {}).get("all_run_count") for mode in ("legacy", "reduced")),
         {"diagnostic": verdict.get("pooled_all_runs_median_fps_diagnostic")}),
        ("the_history_recalc_shows_the_same_split",
         bool(history) and history.get("old_all_run_count_per_order")
         and sum(history["old_all_run_count_per_order"].values())
         > sum(history["correct_kept_run_count_per_order"].values()), history or None),
    ], ["counterbalanced_performance.json verdict.kept_orders",
        "counterbalanced_performance.json verdict.all_orders_diagnostic",
        "historical_warmup_recalc.json"],
        {"pooled_kept_median_fps": verdict.get("pooled_kept_median_fps"),
         "pooled_all_runs_median_fps_diagnostic":
             verdict.get("pooled_all_runs_median_fps_diagnostic")})

    schedules = [(name, doc) for name, doc in execution_artifacts(out_dir)]
    launch_ok = {name: _launch_sequence_matches_schedule(doc)[0] for name, doc in schedules}
    rule_ok = {name: _schedule_follows_the_rule(doc)[0] for name, doc in schedules}
    per_execution_balance = {
        name: ab.order_balance_rows((doc or {}).get("execution_schedule") or [])
        for name, doc in schedules}
    checks["counterbalanced_order"] = _key([
        ("every_execution_recorded_a_schedule", bool(schedules) and all(
            (doc or {}).get("execution_schedule") for _, doc in schedules), len(schedules)),
        ("every_launch_sequence_matches_its_recorded_schedule", all(launch_ok.values()),
         {k: v for k, v in launch_ok.items() if not v} or "all matched"),
        ("every_schedule_is_the_deterministic_rule", all(rule_ok.values()),
         {k: v for k, v in rule_ok.items() if not v} or "all matched"),
        ("every_execution_had_kept_pairs_of_both_positions",
         all(row.get("legacy_first") == row.get("reduced_first") > 0
             for row in per_execution_balance.values()),
         {name: {key: row.get(key) for key in ("legacy_first", "reduced_first")}
          for name, row in per_execution_balance.items()}),
        ("no_execution_ran_a_fixed_order",
         all(row.get("legacy_first") and row.get("reduced_first")
             for row in per_execution_balance.values()), None),
    ], ["ab_executions/*_transfer_ab_benchmark.json execution_schedule",
        "ab_executions/*_transfer_ab_benchmark.json run_launch_sequence",
        "counterbalanced_performance.json verdict.per_execution_outcome[].kept_pair_positions"],
        {"recorded_schedule": [doc.get("execution_schedule") for _, doc in schedules][:2]})

    checks["order_balance"] = _key([
        ("legacy_first_kept_pairs_equal_reduced_first", balance.get("balanced") is True,
         {k: balance.get(k) for k in ("legacy_first", "reduced_first", "pairs")}),
        ("no_kept_pair_has_an_unknown_position", not balance.get("unknown_position"),
         balance.get("unknown_position")),
        ("every_kept_run_is_in_exactly_one_balanced_pair",
         balance.get("pairs") == census.get("legacy", {}).get("kept_run_count")
         == census.get("reduced", {}).get("kept_run_count"),
         {"pairs": balance.get("pairs"),
          "kept": {mode: census.get(mode, {}).get("kept_run_count")
                   for mode in ("legacy", "reduced")}}),
        ("the_balance_is_not_a_statistical_correction",
         all(row.get("order_balance_gate_all_passed") for row in per_execution.values()),
         "the balance is enforced by the launch schedule, per execution"),
    ], ["counterbalanced_performance.json verdict.order_balance",
        "ab_executions/*_transfer_ab_benchmark.json gates.order_balance_gate"],
        {"order_balance": balance})

    strata_populated = {name: (row or {}).get("pair_count") for name, row in strata.items()}
    checks["position_stratification"] = _key([
        ("both_position_strata_have_samples",
         len(strata) == 2 and all(count for count in strata_populated.values()), strata_populated),
        ("the_two_strata_partition_the_kept_pairs",
         sum(strata_populated.values() or [0]) == balance.get("pairs"),
         {"strata": sum(strata_populated.values() or [0]), "pairs": balance.get("pairs")}),
        ("each_stratum_reports_its_own_ratio_and_wins",
         all({"median_ratio", "median_percent_change", "wins", "losses"} <= set(row or {})
             for row in strata.values()), None),
    ], ["counterbalanced_performance.json verdict.position_strata"],
        {"strata": strata,
         "note": "no gate requires both strata to favour either order; an empty stratum would mean "
                 "the claimed balance never happened, and that is what fails here"})

    def every_execution_value(mode, counter):
        return [((ex.get("orders") or {}).get(mode, {})
                 .get("steady_state_counters") or {}).get(counter) for ex in usable]

    red_h2d = every_execution_value("reduced", "nvof_h2d_bytes_measured")
    leg_h2d = every_execution_value("legacy", "nvof_h2d_bytes_measured")
    red_prev_copy = every_execution_value("reduced", "cpu_previous_frame_copy_bytes")
    red_conversions = [(
        ((ex.get("orders") or {}).get("reduced", {}).get("steady_state_counters") or {})
        .get("cpu_bgr_to_rgba_calls"),
        ((ex.get("orders") or {}).get("reduced", {}).get("steady_state_counters") or {})
        .get("cpu_rgba_to_bgr_calls")) for ex in usable]
    leg_conversions = [((ex.get("orders") or {}).get("legacy", {})
                        .get("steady_state_counters") or {}).get("cpu_bgr_to_rgba_calls")
                       for ex in usable]
    checks["copy_gate"] = _key([
        ("copy_gate_passed_in_every_execution",
         bool(per_execution) and all(row.get("copy_gate_all_passed")
                                     for row in per_execution.values()), len(per_execution)),
        ("reduced_upload_is_6220800_bytes_in_every_execution",
         bool(red_h2d) and set(red_h2d) == {6220800}, sorted(set(red_h2d))),
        ("legacy_upload_is_8294400_bytes_in_every_execution",
         bool(leg_h2d) and set(leg_h2d) == {8294400}, sorted(set(leg_h2d))),
        ("reduced_cpu_conversions_are_zero_in_every_execution",
         bool(red_conversions) and all(pair == (0, 0) for pair in red_conversions),
         sorted({pair for pair in red_conversions})),
        ("legacy_cpu_conversions_are_the_control",
         bool(leg_conversions) and all(v and v > 0 for v in leg_conversions),
         sorted(set(leg_conversions))),
        ("no_cpu_copy_of_the_previous_frame_anywhere",
         bool(red_prev_copy) and set(red_prev_copy) == {0}, sorted(set(red_prev_copy))),
    ], ["counterbalanced_performance.json executions[].gates.copy_gate",
        "counterbalanced_performance.json executions[].orders[].steady_state_counters"],
        {"reduced_h2d_values": sorted(set(red_h2d)),
         "legacy_h2d_values": sorted(set(leg_h2d))})

    identities = {}
    for name, doc in schedules:
        identity = (doc or {}).get("output_identity") or {}
        per_run = identity.get("bitstream_sha256_of_each_run") or {}
        identities[name] = {mode: sorted({v for v in (per_run.get(mode) or {}).values() if v})
                            for mode in ("legacy", "reduced")}
    identical = {name: (rows_two["legacy"] == rows_two["reduced"] and bool(rows_two["legacy"]))
                 for name, rows_two in identities.items()}
    probed_by_batch = (ffprobe or {}).get("runs_with_an_ffprobe_reading_by_batch") or {}
    probed_60f = probed_by_batch.get("ab_executions")
    probed_150f = probed_by_batch.get("ab_executions_150f")
    total_runs = sum(len(rows[mode]["all"]) for mode in ("legacy", "reduced"))
    corroboration_runs = sum(len(item) for item in
                             ((corroboration or {}).get("all_runs_per_order") or {}).values())
    checks["output_integrity"] = _key([
        ("output_integrity_gate_passed_in_every_execution",
         bool(per_execution) and all(row.get("output_integrity_gate_all_passed")
                                     for row in per_execution.values()), len(per_execution)),
        ("every_execution_produced_a_bitstream_for_both_orders",
         all(rows_two["legacy"] and rows_two["reduced"] for rows_two in identities.values()),
         {name: {mode: len(v) for mode, v in row.items()} for name, row in identities.items()}),
        ("legacy_and_reduced_bitstreams_are_the_same_bytes", all(identical.values()),
         sorted(name for name, ok in identical.items() if not ok) or "every execution"),
        ("every_output_file_of_the_primary_batch_was_probed",
         probed_60f == total_runs and bool(probed_60f),
         {"probed": probed_60f, "runs": total_runs,
          "note": "counted against the primary batch alone: ffprobe_output.json also carries the "
                  "corroboration's files, and a total that spans both batches would either fail "
                  "here or, worse, pass by propping one batch's count up with the other's"}),
        ("every_output_file_of_the_corroborating_batch_was_probed",
         not corroboration_runs or probed_150f == corroboration_runs,
         {"probed": probed_150f, "runs": corroboration_runs}),
        ("no_execution_failed_or_crashed", not failures, failures),
    ], ["counterbalanced_performance.json executions[].gates.output_integrity_gate",
        "ab_executions/*_transfer_ab_benchmark.json output_identity",
        "ffprobe_output.json runs_with_an_ffprobe_reading_by_batch"],
        {"bitstream_digests_per_execution": identities})

    l = verdict.get("pooled_kept_median_fps", {}).get("legacy")
    r = verdict.get("pooled_kept_median_fps", {}).get("reduced")
    slower_executions = sorted(name for name, row in per_execution.items()
                               if (row.get("median_fps_of_kept_runs") or {}).get("reduced")
                               and (row.get("median_fps_of_kept_runs") or {}).get("legacy")
                               and row["median_fps_of_kept_runs"]["reduced"]
                               < row["median_fps_of_kept_runs"]["legacy"])
    checks["performance_non_regression"] = _key([
        ("reduced_kept_median_at_least_legacy_kept", bool(l and r and r >= l),
         {"legacy": l, "reduced": r}),
        ("the_pooled_verdict_reports_no_regression",
         verdict.get("no_end_to_end_regression") is True, None),
        ("no_execution_was_dropped_to_get_there",
         len(per_execution) == len(executions),
         {"recorded": len(per_execution), "attempted": len(executions)}),
        ("every_execution_that_ran_is_still_in_the_file",
         all(ex.get("execution") in per_execution for ex in executions),
         [ex.get("execution") for ex in executions if ex.get("execution") not in per_execution]),
    ], ["counterbalanced_performance.json verdict.no_end_to_end_regression",
        "counterbalanced_performance.json verdict.pooled_kept_median_fps"],
        {"executions_whose_own_median_ratios_came_out_below_one": slower_executions,
         "kept_they_are_all_still_pooled": True,
         "note": "§33: an execution that measures the reduced order slower stays in the pool. "
                 "It is named here so a reader can count them and check none went missing."})

    checks["performance_direction_consistency"] = _key([
        ("the_pooled_medians_and_the_paired_ratios_point_the_same_way",
         verdict.get("performance_direction_consistent") is True,
         {"pooled_kept_ratio": verdict.get("pooled_kept_ratio"),
          "pairwise_median_ratio": verdict.get("pairwise_median_ratio")}),
        ("both_readings_are_published", bool(verdict.get("pooled_kept_ratio"))
         and bool(verdict.get("pairwise_median_ratio")), None),
        ("the_sign_test_is_evidence_not_a_gate",
         (verdict.get("sign_test") or {}).get("used_for") is not None,
         (verdict.get("sign_test") or {}).get("used_for")),
    ], ["counterbalanced_performance.json verdict.performance_direction_consistent",
        "counterbalanced_performance.json verdict.pairwise_median_ratio"],
        {"sign_test": verdict.get("sign_test"),
         "median_reduced_minus_legacy_percent": verdict.get(
             "median_reduced_minus_legacy_percent")})
    dec_rows = ((consistency_doc or {}).get("decoder_pipe") or {})
    checks["decoder_status_semantics"] = _key([
        ("decoder_pipe_is_not_reported_as_reduced",
         dec_rows.get("status") != "REDUCED_PROVEN" and bool(dec_rows.get("status")), dec_rows),
        ("decoder_pipe_reports_blocked_while_its_bytes_are_equal",
         dec_rows.get("status") == "BLOCKED_PROVEN"
         and dec_rows.get("legacy_bytes") == dec_rows.get("reduced_bytes")
         and dec_rows.get("unchanged_this_round") is True,
         {"legacy": dec_rows.get("legacy_bytes"), "reduced": dec_rows.get("reduced_bytes")}),
        ("no_published_word_contradicts_its_own_numbers",
         (consistency_doc or {}).get("all_rows_consistent") is True,
         (consistency_doc or {}).get("inconsistent_rows")),
        ("every_row_whose_numbers_are_equal_says_blocked",
         all(row["published_status"] == "BLOCKED_PROVEN"
             for row in (consistency_doc or {}).get("rows") or []
             if row.get("before") is not None and row["before"] == row["after"]),
         {row["claim"]: row["published_status"]
          for row in (consistency_doc or {}).get("rows") or []
          if row.get("before") is not None and row["before"] == row["after"]}),
    ], ["status_consistency.json", "counterbalanced_performance.json "
        "executions[].orders[].steady_state_counters.decoder_pipe_bytes"],
        {"claims": (consistency_doc or {}).get("rows"),
         "reduced_proven_rows": (consistency_doc or {}).get("reduced_proven_rows")})

    counts = _counts(pytest_line or "")
    checks["full_unit_suite"] = _key([
        ("the_suite_ran_and_its_transcript_is_here", bool(counts.get("passed")), pytest_line),
        ("nothing_failed", not counts.get("failed") and not counts.get("error"), counts),
        (f"at_least_the_{unit_suite_floor}_tests_the_previous_round_left_behind",
         counts.get("passed", 0) >= unit_suite_floor,
         {"floor": unit_suite_floor, "passed": counts.get("passed", 0),
          "note": "a floor, not an expected count: tests added this round raise the number and "
                  "must not require this check to be edited to match it"}),
    ], ["pytest_output.txt"], {"summary_line": pytest_line or None, "counts": counts or None})

    binaries = native_against_baseline(baseline)
    changed = (env or {}).get("files_changed_since_baseline") or []
    outside_scope = [path for path in changed
                     if not (path.startswith("tests/") or path.startswith("perf_results/"))]
    checks["native_scope_unchanged"] = _key([
        ("every_native_binary_matches_the_baseline_commit",
         all(row["unchanged_since_baseline"] for row in binaries.values()), binaries),
        ("no_native_source_was_touched",
         not [path for path in changed if path.startswith("native/")
              and (path.endswith(".cpp") or path.endswith(".h"))],
         [path for path in changed if path.startswith("native/")]),
        ("every_file_this_round_changed_is_test_layer", not outside_scope, outside_scope),
    ], [f"git object hashes at {baseline} and in the working tree",
        f"git diff --name-only {baseline}..HEAD"], {"binaries": binaries, "changed": changed})

    ordered = {key: checks[key] for key in (key_order or BENCH_FIX_KEYS)}
    # The three stability gates keep their own workings in their `detail`, and §51 asks the report
    # for three of those figures by name, so they are lifted out beside the rest of the readings
    # rather than being quoted from a summary written afterwards.
    variance_detail = (checks.get("benchmark_variance") or {}).get("detail") or {}
    corroboration_detail = (checks.get("corroboration_consistency") or {}).get("detail") or {}
    runtime_detail = (checks.get("runtime_contamination") or {}).get("detail") or {}
    status = ("DONE_PROVEN" if sha and all(row["status"] == "DONE_PROVEN"
                                           for row in ordered.values()) else "NOT_READY")
    readings = {
        "A_pooled_kept_median_legacy_fps": l,
        "B_pooled_kept_median_reduced_fps": r,
        "C_pooled_kept_ratio": verdict.get("pooled_kept_ratio"),
        "D_pairwise_median_ratio": verdict.get("pairwise_median_ratio"),
        "reduced_vs_legacy_when_reduced_first": strata.get(
            "reduced_vs_legacy_when_reduced_first"),
        "reduced_vs_legacy_when_reduced_second": strata.get(
            "reduced_vs_legacy_when_reduced_second"),
        "sign_test_over_kept_pairs": verdict.get("sign_test"),
        "run_census": census,
        "order_balance": balance,
        "all_runs_median_diagnostic": verdict.get("pooled_all_runs_median_fps_diagnostic"),
        "note": "§32 asks for all four readings plus both strata and the sign test, and forbids "
                "picking the prettiest one; the diagnostic column is the figure the warmup bug "
                "would have produced, published beside the one the verdict used",
        "150_frame_corroboration": (corroboration or {}).get("verdict"),
        # §51 asks the report for the per-execution spreads, the variance verdict and whether the
        # two frame lengths agreed, so the status file carries those three answers as readings
        # rather than leaving them to be re-derived from the gates by whoever writes the report.
        "kept_run_spreads_by_execution": variance_detail.get("spreads"),
        "benchmark_variance_stable": variance_detail.get("benchmark_variance_stable"),
        "60f_against_150f": corroboration_detail,
        "runtime_contamination": runtime_detail.get("contamination"),
    }
    return {
        "generated_by": "tests/unit/collect_transfer_status.py",
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
                                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "round": round_name,
        "baseline": baseline,
        "code_under_test": sha,
        "artifact_shas": {name: _sha_of(doc) for name, doc in named.items()},
        "missing_artifacts": prov["missing_required"],
        "artifact_sha_coverage": coverage,
        "required_artifacts": list(required),
        "status": status,
        "allowed_status_values": ["DONE_PROVEN", "NOT_READY"],
        "gate_statuses": {key: row["status"] for key, row in ordered.items()},
        "gates": ordered,
        "performance_readings": readings,
        "native_binaries": native_binaries(),
        "commits": commits_this_round(baseline=baseline),
        "primary_question": (perf or {}).get("verdict", {}).get("primary_question"),
        "rule": (perf or {}).get("verdict", {}).get("rule"),
        "what_this_round_did_not_claim": [
            "no speedup claim: this round fixed the instrument, and the sign test it reports is "
            "evidence, not a gate",
            "no reduction claim for any transfer whose two measurements are equal",
            "no GPU pipeline, zero-copy or eliminated-round-trip claim of any kind"],
        "forbidden_phrases": ["GPU pipeline complete", "zero-copy complete",
                              "CPU round trips eliminated", "full GPU acceleration complete"],
    }


# The two rounds this collector assembles, as data, so a round cannot be run with the other one's
# artifact list, key order or baseline by forgetting which `--round` was typed.
BENCH_FIX_ROUND = {
    "required": BENCH_FIX_REQUIRED_ARTIFACTS, "key_order": BENCH_FIX_KEYS,
    "round_name": "Transfer benchmark evidence protocol fix", "baseline": BENCH_FIX_BASELINE,
    "status_name": "transfer_bench_fix_status.json", "unit_suite_floor": 273,
}
BENCH_STABILITY_ROUND = dict(
    BENCH_FIX_ROUND, required=BENCH_STABILITY_REQUIRED_ARTIFACTS,
    key_order=BENCH_STABILITY_KEYS, round_name="Transfer benchmark environment stability",
    baseline=BENCH_STABILITY_BASELINE, status_name="benchmark_stability_status.json",
    # §43: a floor taken from what the previous round left behind, never the exact number this round
    # happens to reach - a round that adds tests must not have to edit a gate to admit them.
    unit_suite_floor=287,
    # §37 over both batches: the 150-frame executions are this round's evidence too, and an
    # artifact that names a different commit than the pool it sits beside is the disagreement the
    # provenance gate exists to catch.
    extra_executions=("ab_executions_150f",))


def write_bench_fix(out_dir, spec=None):
    """Assemble the round's derived files and write its status, in the order that feeds itself."""
    spec = dict(BENCH_FIX_ROUND if spec is None else spec)
    status_name = spec.pop("status_name")
    perf = _load(out_dir, "counterbalanced_performance.json")
    # Every derived file is stamped with the commit the runs happened at, which the benchmark's own
    # primary artifact already records; HEAD is only here so a re-assembly without the benchmark
    # still produces an artifact that names what it describes instead of naming nothing.
    code_under_test = _sha_of(perf) or git("rev-parse", "HEAD")
    claims, measured = bench_fix_claims(perf)
    published = {
        "nvof_h2d_upload": "REDUCED_PROVEN",
        "cpu_bgr_to_rgba_calls": "ELIMINATED_PROVEN",
        "cpu_rgba_to_bgr_calls": "ELIMINATED_PROVEN",
        "cpu_color_conversion_output": "ELIMINATED_PROVEN",
        "decoder_pipe": "BLOCKED_PROVEN",
        "encoder_pipe": "BLOCKED_PROVEN",
        "nvof_flow_d2h_download": "BLOCKED_PROVEN",
    }
    consistency = bench_fix_consistency(claims, published)
    consistency["measured_steady_state_by_order"] = measured
    consistency["code_under_test"] = code_under_test
    with open(os.path.join(out_dir, "status_consistency.json"), "w", encoding="utf-8") as handle:
        json.dump(consistency, handle, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "ffprobe_output.json"), "w", encoding="utf-8") as handle:
        json.dump(merge_ffprobe(out_dir, code_under_test), handle, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "environment.json"), "w", encoding="utf-8") as handle:
        json.dump(bench_fix_environment(out_dir, code_under_test, baseline=spec["baseline"],
                                        round_name=spec["round_name"]),
                  handle, indent=2, sort_keys=True)

    payload = build_bench_fix(out_dir, **spec)
    path = os.path.join(out_dir, status_name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(json.dumps({"ok": payload["status"] == "DONE_PROVEN", "status": payload["status"],
                      "code_under_test": payload["code_under_test"],
                      "gate_statuses": payload["gate_statuses"],
                      "failed_checks": {key: payload["gates"][key]["failed_checks"]
                                        for key in spec["key_order"]
                                        if payload["gates"][key]["failed_checks"]},
                      "performance_readings": {k: v for k, v in
                                               payload["performance_readings"].items()
                                               if k in PERFORMANCE_KEY_NUMBERS},
                      "missing_artifacts": payload["missing_artifacts"],
                      "written": path}, indent=2, sort_keys=True, default=str))
    return 0 if payload["status"] == "DONE_PROVEN" else 1





def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="the artifact directory to read and write")
    parser.add_argument("--round", choices=("phase1", "bench_fix", "bench_stability"),
                        default="phase1", help="which round's status file to assemble")
    args = parser.parse_args(argv)
    if args.round == "bench_fix":
        return write_bench_fix(args.out, BENCH_FIX_ROUND)
    if args.round == "bench_stability":
        return write_bench_fix(args.out, BENCH_STABILITY_ROUND)
    payload = build(args.out)
    path = os.path.join(args.out, "transfer_phase1_status.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    consistency_path = os.path.join(args.out, "status_consistency.json")
    with open(consistency_path, "w", encoding="utf-8") as handle:
        json.dump(consistency_artifact(payload["transfer_claim_consistency"],
                                       payload["code_under_test"]),
                  handle, indent=2, sort_keys=True)
    print(json.dumps({"ok": payload["status"] == "DONE_PROVEN", "status": payload["status"],
                      "code_under_test": payload["code_under_test"],
                      "gate_statuses": payload["gate_statuses"],
                      "missing_artifacts": payload["missing_artifacts"],
                      "written": [path, consistency_path]}, indent=2, sort_keys=True))
    return 0 if payload["status"] == "DONE_PROVEN" else 1



if __name__ == "__main__":
    sys.exit(main())
