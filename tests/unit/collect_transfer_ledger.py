#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the production export path and write its transfer ledger.

Archival: the run it accounts for is the dlssnr_host.dll session, whose wrapper
(dlss_engine.py) and binary are retired from this tree. The ledger is the record of
where those bytes moved; a GPU-host ledger is a new measurement, not this script.

The ledger is generated, not handwritten: every measured column comes from a run of this
checkout's own code, and the fields the run cannot measure arrive as None and stay that
way. The fixture, encoder, CQ, working scale and NVOF preset are the ones the benchmark
uses, so a byte figure here describes the same configuration the FPS figures describe.

Usage:
    python tests/unit/collect_transfer_ledger.py --frames 12 --out perf_results/<sha>/transfer_phase1
    python tests/unit/collect_transfer_ledger.py --frames 12 --scale 0.5 --out ... --markdown TRANSFER_LEDGER.md

The ledger always lands in `--out`; `--markdown` writes a second copy somewhere else, and
must name a different directory, because Windows treats `transfer_ledger.md` and
`TRANSFER_LEDGER.md` as one file in one folder.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import benchmark_e2e as bench  # noqa: E402
import collect_transfer_abi as abi  # noqa: E402
import transfer_ledger as tl  # noqa: E402
import transfer_profiler as tp  # noqa: E402

# Fields the ledger reads out of the per-frame records: the byte and call counts the seam
# and the native bridges publish, taken from the profiler's own field list so a new field
# is averaged here without a second list to keep in step. `nvof_input_format` is a string
# and is reported as its value set instead.
MEASURED_FIELDS = tuple(key for key in tp.FIELDS if key != "nvof_input_format")


def git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, text=True,
                             timeout=30)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def tree_state():
    """What the running code is relative to the commit it names.

    "dirty" here means a tracked file differs from HEAD, because that is the only way the
    code can differ from the commit an artifact names. Untracked scratch is reported next to
    it instead of being folded in: a directory of benchmark output does not change what ran,
    and calling it dirty would make every honest artifact claim it does not need.
    """
    tracked = git("status", "--porcelain", "--untracked-files=no")
    untracked = [abi._porcelain_path(line) for line in git("status", "--porcelain").splitlines()
                 if line.strip().startswith("??")]
    return {"working_tree_dirty": bool(tracked),
            "working_tree_paths": [abi._porcelain_path(line)
                                   for line in tracked.splitlines() if line.strip()],
            "untracked_paths": untracked}


def actual_residual_backend(records, diagnostics):
    """The residual backend that ran, read out of the frame records that ran it.

    The per-frame record calls the field `residual_backend_actual`; the export diagnostics
    dict calls it `residual_backend`. Asking the second dict for the first name returned
    None, which would read as "nobody measured it" for a stage that reported every frame.
    """
    for record in reversed(records):
        value = record.get("residual_backend_actual")
        if value:
            return value
    return diagnostics.get("residual_backend")


def steady_state_means(records):
    """Median over the frames after the first, per measured field.

    The first frame resets the temporal hints and uploads twice, so including it would
    report a steady-state path that does not exist. A field no frame measured is None.
    """
    out = {}
    for field in MEASURED_FIELDS:
        values = [r[field] for r in records[1:]
                  if field in r and isinstance(r.get(field), (int, float))]
        out[field] = int(statistics.median(values)) if values else None
    out["nvof_input_format"] = sorted({r.get("nvof_input_format") for r in records[1:]
                                       if r.get("nvof_input_format")}) or None
    return out


def run_export(frames, scale, transfer_mode=None, profiler=None, tag=""):
    """One real export of `frames` frames through the production path, with accounting.

    `profiler` is the stage timer: pass a `pipeline.StageProfiler()` to get the per-span
    mean/p50/p95 the A/B needs out of the same run that produces the byte counters, so the
    two cannot describe different runs. `tag` only names the output file.
    """
    import dlss_engine
    import pipeline
    from optical_flow import get_frame_guidance_service

    live_w, live_h = pipeline.resolve_working_dims(bench.FIXTURE_W, bench.FIXTURE_H, scale)
    job_settings = {
        "preset": 1, "style": 0, "intensity": 1.0, "local_tone": 1.0, "local_struct": 1.0,
        "skin_struct": 1.0, "use_auto_mask": 1, "ui_correction": 0,
        "working_scale": scale, "residual_pyramid": scale < 1.0, "residual_strength": 1.0,
        "multi_pass": 1, "guidance_mode": 1,
        "flow_engine": bench.FLOW_ENGINE, "flow_quality": bench.FLOW_PRESET,
    }
    if scale < 1.0:
        job_settings["residual_backend"] = "native"
    if transfer_mode:
        job_settings["transfer_mode"] = transfer_mode

    live = dlss_engine.Live(live_w, live_h, job_settings)
    guidance_service = get_frame_guidance_service(bench.FLOW_ENGINE, bench.FLOW_PRESET)
    flow_provider = pipeline.GuidanceServiceFlowProvider(guidance_service, guidance_mode=1)
    collector = tp.TransferProfiler()
    diagnostics = {}
    out_path = os.path.join(_ROOT, "bench_out",
                            f"ledger_{transfer_mode or 'auto'}_{int(scale * 100)}"
                            f"{('_' + tag) if tag else ''}.mkv")
    if os.path.exists(out_path):
        os.remove(out_path)
    # t0 sits around the export call and nowhere earlier: the session setup above is a
    # per-process cost, and folding it into the frame rate would make a 6-frame run and a
    # 60-frame run incomparable and both slower than the pipeline actually is. This is the
    # same window `tests/unit/benchmark_e2e.py` times, so the figures can be read side by side.
    t0 = time.perf_counter()
    try:
        pipeline.export_dlss_stream(
            video=bench.ensure_fixture(), live=live, lock=None, frames=frames,
            fps=bench.FIXTURE_FPS, cq=bench.CQ, with_audio=False,
            flow_provider=flow_provider,
            cancel=lambda: False, out_path=out_path, job_settings=job_settings,
            diagnostics_out=diagnostics, transfer_collector=collector, profiler=profiler)
    finally:
        elapsed = time.perf_counter() - t0
        try:
            guidance_service.close()
        except Exception:
            pass
    return {"records": collector.records, "summary": collector.summary(),
            "diagnostics": diagnostics, "out_path": out_path,
            "live_w": live_w, "live_h": live_h, "job_settings": job_settings,
            "elapsed_sec": round(elapsed, 3),
            "fps": round(len(collector.records) / max(0.001, elapsed), 4)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--scale", type=float, default=tl.WORKING_SCALE)
    parser.add_argument("--transfer-mode", default=None,
                        help="ask the pipeline for a specific transfer mode")
    parser.add_argument("--out", required=True, help="directory for transfer_ledger.json")
    parser.add_argument("--markdown", default="", help="also write the ledger here")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args(argv)

    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    run = run_export(args.frames, args.scale, args.transfer_mode)
    records = run["records"]
    if len(records) < 3:
        print(json.dumps({"ok": False, "error": f"only {len(records)} frames recorded"}))
        return 1
    measured = steady_state_means(records)
    diagnostics = run["diagnostics"]
    decode_actual = records[-1].get("decode_actual", "unknown")
    # The rows are built for the order that ran. A ledger of a reduced run drawn from the
    # legacy rows would name conversions this frame did not make, and the measured zeros in
    # its own columns would contradict the prose next to them.
    transfer_actual = diagnostics.get("transfer_mode_actual")
    if transfer_actual not in tl.TRANSFER_MODES:
        print(json.dumps({"ok": False, "error": f"run reported transfer_mode_actual="
                         f"{transfer_actual!r}, which is not a ledger order"}))
        return 1

    digest = hashlib.sha256()
    for row in records:
        digest.update(json.dumps({k: row[k] for k in MEASURED_FIELDS if k in row},
                                 sort_keys=True, default=str).encode("utf-8"))

    ledger = tl.build_ledger(
        measured=measured, source_w=bench.FIXTURE_W, source_h=bench.FIXTURE_H,
        working_w=run["live_w"], working_h=run["live_h"],
        output_w=bench.FIXTURE_W, output_h=bench.FIXTURE_H, decode_actual=decode_actual,
        transfer_mode=transfer_actual,
        extra={
            "generated_by": os.path.relpath(os.path.abspath(__file__), _ROOT).replace("\\", "/"),
            "label": args.label,
            "code_under_test": git("rev-parse", "HEAD"),
            # The ledger says which commit it describes, and a modified tracked file means the
            # running code is that commit plus whatever is uncommitted. Say so instead of
            # letting a measured figure look like it came from a clean checkout.
            **tree_state(),
            "working_scale": args.scale,
            "transfer_mode_requested": diagnostics.get("transfer_mode_requested"),
            "transfer_mode_actual": diagnostics.get("transfer_mode_actual"),
            "transfer_mode_fallback_reason": diagnostics.get("transfer_mode_fallback_reason"),
            "nvof_input_format": measured.get("nvof_input_format"),
            "residual_backend_actual": actual_residual_backend(records, diagnostics),
            "residual_backend_version": diagnostics.get("residual_backend_version"),
            "frames_run": len(records),
            "frames_used_for_steady_state": len(records) - 1,
            "fixture": {"path": os.path.relpath(bench.ensure_fixture(), _ROOT),
                        "sha256": bench.sha256(bench.ensure_fixture()),
                        "width": bench.FIXTURE_W, "height": bench.FIXTURE_H,
                        "fps": bench.FIXTURE_FPS},
            "encoder": {"cq": bench.CQ, "out_path": os.path.basename(run["out_path"]),
                        "bytes": os.path.getsize(run["out_path"])
                        if os.path.exists(run["out_path"]) else None},
            "measured_steady_state": measured,
            "per_frame_records": records,
            "transfer_summary": run["summary"],
            "record_digest_sha256": digest.hexdigest(),
        })

    json_path = tl.write_ledger(os.path.join(args.out, "transfer_ledger.json"), ledger)
    md_path = tl.write_markdown(os.path.join(args.out, "transfer_ledger.md"), ledger)
    # A second destination for the human-readable copy. Windows paths are case-insensitive,
    # so `transfer_ledger.md` and `TRANSFER_LEDGER.md` are one file: the extra copy has to
    # live in a different directory, and the guard says so instead of overwriting silently.
    extra_md = ""
    if args.markdown:
        same_dir = (os.path.dirname(os.path.abspath(args.markdown))
                    == os.path.dirname(os.path.abspath(md_path)))
        if same_dir and os.path.basename(args.markdown).lower() == os.path.basename(md_path).lower():
            extra_md = md_path
        else:
            extra_md = tl.write_markdown(args.markdown, ledger)
    print(json.dumps({"ok": True, "ledger": json_path, "markdown": md_path,
                      "markdown_copy": extra_md,
                      "frames": len(records), "decode_actual": decode_actual,
                      "transfer_mode_actual": ledger["transfer_mode_actual"],
                      "measured_steady_state": measured},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
