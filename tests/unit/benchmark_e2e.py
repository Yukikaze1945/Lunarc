#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end 4K benchmark for the residual pipeline, run in isolated processes.

Archival: the scenarios below were timed on the closed dlssnr_host.dll session, whose
wrapper (dlss_engine.py) and binary are retired from this tree, so this is the record
of how the figures were produced, not a gate a current build can run.

Quick mode (default) measures the five scenarios the task defines:

    A  100% working, NVOF 4M                      no residual stage exists here
    B   75% working, NVOF 4M, cpu_reference_native
    C   63% working, NVOF 4M, cpu_reference_native
    D   50% working, NVOF 4M, cpu_reference_native
    E   25% working, NVOF 4M, cpu_reference_native

The gate is FPS(D) > FPS(A): a smaller working resolution must actually pay for itself
after the residual is added back at full resolution. Nothing here is allowed to cheat
that comparison - the fixture, encoder, CQ, NVOF preset and DLSS settings are fixed, and
every scenario runs in a fresh process, which is what makes peak RSS a process figure.

The GPU memory columns are not process figures. nvidia-smi reports this card's total in
use, and this driver gives no per-process GPU attribution, so a fresh process does not
make that number ours; the names say so.

Sustained mode runs A and D for at least 300 frames while sampling RSS, the card's GPU
memory in use, thread count, child process count and the native residual counters every
30 frames.

Usage:
    python tests/unit/benchmark_e2e.py --mode quick     --frames 60  --out perf_results/...
    python tests/unit/benchmark_e2e.py --mode quick --only A,D --frames 60 --out ...
    python tests/unit/benchmark_e2e.py --mode sustained --frames 300 --out perf_results/...
    python tests/unit/benchmark_e2e.py --scenario 3 --frames 60      # single child process
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURE_W, FIXTURE_H, FIXTURE_FPS = 3840, 2160, 60.0
FIXTURE_FRAMES = 300
FIXTURE_PATH = os.path.join(_REPO, "bench_out", "fixture_4k_300.mp4")
CQ = 18
FLOW_ENGINE, FLOW_PRESET = "nvof", "4M"

SCENARIOS = [
    {"id": "A", "name": "100% working, NVOF 4M", "scale": 1.0, "residual_backend": None},
    {"id": "B", "name": "75% working, NVOF 4M + native residual", "scale": 0.75,
     "residual_backend": "native"},
    {"id": "C", "name": "63% working, NVOF 4M + native residual", "scale": 0.63,
     "residual_backend": "native"},
    {"id": "D", "name": "50% working, NVOF 4M + native residual", "scale": 0.50,
     "residual_backend": "native"},
    {"id": "E", "name": "25% working, NVOF 4M + native residual", "scale": 0.25,
     "residual_backend": "native"},
]

# Stage keys copied out of the per-frame profiler summary into the report. A key that the
# bridge could not measure is absent from the summary, which drops it here as well.
STAGE_KEYS = ("frame_total", "working_color_prepare_ms", "guidance_total",
              "dlss_process", "residual_pyramid", "output_convert", "pipe_write",
              "decode_wait", "nvof_submit_cpu_ms", "nvof_dependency_wait_ms",
              "nvof_postprocess_wait_inclusive_ms", "nvof_upload_ms",
              "nvof_download_ms", "nvof_total_native_ms")
COUNTER_KEYS = ("nvof_h2d_bytes_delta", "nvof_d2h_bytes_delta", "nvof_h2d_copy_count_delta",
                "nvof_d2h_copy_count_delta", "nvof_execute_count_delta",
                "nvof_native_history",
                "cpu_previous_frame_copy_bytes", "depth_missing_frames",
                "filter_plan_build_count")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_fixture():
    """Deterministic 4K test pattern, generated once and hashed into the report."""
    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    if not os.path.exists(FIXTURE_PATH):
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi",
               "-i", f"testsrc2=size={FIXTURE_W}x{FIXTURE_H}:rate={FIXTURE_FPS}"
                     f":duration={FIXTURE_FRAMES / FIXTURE_FPS:.6f}",
               "-frames:v", str(FIXTURE_FRAMES), "-c:v", "libx264", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", FIXTURE_PATH]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0 or not os.path.exists(FIXTURE_PATH):
            raise RuntimeError(f"fixture generation failed: {result.stderr[-400:]}")
    return FIXTURE_PATH


def ffprobe_validate(path, expected_frames, expected_w, expected_h):
    """Section 27: packets, duration, resolution, fps and codec must all line up."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
           "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_packets,"
                            "duration:format=duration", "-of", "json", path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr[-300:]}
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or [{}]
    stream = streams[0]
    packets = int(stream.get("nb_read_packets") or stream.get("nb_read_frames") or 0)
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    num, _, den = str(stream.get("avg_frame_rate") or "0/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    checks = {
        "packets": packets == expected_frames,
        "resolution": (stream.get("width"), stream.get("height")) == (expected_w, expected_h),
        "duration_within_1pct": abs(duration - expected_frames / FIXTURE_FPS)
                                <= 0.01 * expected_frames / FIXTURE_FPS + 0.05,
        "fps_matches_source": abs(fps - FIXTURE_FPS) < 1.0,
    }
    return {"ok": all(checks.values()), "checks": checks, "packets": packets,
            "expected_frames": expected_frames, "duration_sec": round(duration, 4),
            "fps": round(fps, 3), "codec": stream.get("codec_name"),
            "width": stream.get("width"), "height": stream.get("height")}


def run_scenario(index, frames, sample_interval=0):
    """Child process: run one scenario end to end and print RESULT_JSON:{...}."""
    sys.path.insert(0, _REPO)
    sys.path.insert(0, os.path.join(_REPO, "tests", "unit"))
    import dlss_engine
    import pipeline
    import resource_metrics
    from optical_flow import get_frame_guidance_service

    scenario = SCENARIOS[index]
    scale = scenario["scale"]
    out_path = os.path.join(_REPO, "bench_out", f"e2e_{scenario['id']}_{int(scale * 100)}.mkv")
    if os.path.exists(out_path):
        os.remove(out_path)

    live_w, live_h = pipeline.resolve_working_dims(FIXTURE_W, FIXTURE_H, scale)
    job_settings = {
        "preset": 1, "style": 0, "intensity": 1.0, "local_tone": 1.0, "local_struct": 1.0,
        "skin_struct": 1.0, "use_auto_mask": 1, "ui_correction": 0,
        "working_scale": scale, "residual_pyramid": scenario["residual_backend"] is not None,
        "residual_strength": 1.0, "multi_pass": 1, "guidance_mode": 1,
        "flow_engine": FLOW_ENGINE, "flow_quality": FLOW_PRESET,
    }
    if scenario["residual_backend"]:
        job_settings["residual_backend"] = scenario["residual_backend"]

    live = dlss_engine.Live(live_w, live_h, job_settings)
    if isinstance(live, (list, tuple)):
        raise RuntimeError(f"multi-pass sessions are not supported: got {len(live)} lives")

    guidance_service = get_frame_guidance_service(FLOW_ENGINE, FLOW_PRESET)
    flow_provider = pipeline.GuidanceServiceFlowProvider(guidance_service, guidance_mode=1)
    diagnostics = {}
    profiler = pipeline.StageProfiler()

    frame_samples = []

    def _progress(current, total, _message):
        """Cheap per-frame-boundary sampling; psutil only, so timings stay honest.

        nvidia-smi costs tens of milliseconds and would land inside the measured frame,
        so the card's GPU memory is sampled by the background thread instead.
        """
        if sample_interval and current % sample_interval == 0:
            frame_samples.append({"frame": current,
                                  "rss_bytes": resource_metrics.rss_bytes(),
                                  "peak_rss_bytes": resource_metrics.peak_rss_bytes(),
                                  "thread_count": resource_metrics.thread_count(),
                                  "child_process_count": resource_metrics.child_process_count()})

    sampler = resource_metrics.Sampler(5.0) if sample_interval else None
    if sampler:
        sampler.start()
    t0 = time.perf_counter()
    try:
        pipeline.export_dlss_stream(
            video=ensure_fixture(), live=live, lock=None, frames=frames, fps=FIXTURE_FPS,
            cq=CQ, with_audio=False, flow_provider=flow_provider,
            progress=_progress, cancel=lambda: False,
            out_path=out_path, job_settings=job_settings, profiler=profiler,
            diagnostics_out=diagnostics)
    finally:
        samples = sampler.stop() if sampler else []
        try:
            guidance_service.close()
        except Exception:
            pass
    elapsed = time.perf_counter() - t0

    summary = profiler.summary()
    counters = {key: summary.get(key, {}) for key in COUNTER_KEYS if key in summary}
    # Queried after the run because an nvidia-smi subprocess inside the frame loop would
    # land in the measured spans; this driver answers [N/A] for every process either way.
    process_gpu, process_gpu_source = resource_metrics.process_gpu_memory_bytes()
    result = {
        "id": scenario["id"],
        "name": scenario["name"],
        "working_scale": scale,
        "working_dims": [live_w, live_h],
        "frames": frames,
        "elapsed_sec": round(elapsed, 3),
        "fps": round(frames / max(0.001, elapsed), 3),
        "stages_ms": {key: summary.get(key, {}) for key in STAGE_KEYS if key in summary},
        "per_frame_counters": counters,
        "residual_diagnostics": diagnostics,
        "peak_rss_bytes": resource_metrics.peak_rss_bytes(),
        "peak_gpu_total_memory_used_bytes":
            max((s["gpu_total_memory_used_bytes"] or 0) for s in samples) if samples
            else resource_metrics.gpu_total_memory_used_bytes(),
        "peak_gpu_total_memory_source": "nvidia-smi memory.used, whole card",
        "process_gpu_memory_bytes": process_gpu,
        "process_gpu_memory_source": process_gpu_source,
        "thread_count_end": resource_metrics.thread_count(),
        "child_process_count_end": resource_metrics.child_process_count(),
        "resource_samples": samples,
        "frame_samples": frame_samples,
        "ffprobe": ffprobe_validate(out_path, frames, FIXTURE_W, FIXTURE_H),
        "output_path": out_path,
    }
    print("RESULT_JSON:" + json.dumps(result), flush=True)


def master(mode, frames, out_path, only=""):
    """Parent process: run each scenario in its own child and assemble the report."""
    import resource_metrics

    ensure_fixture()
    selected = ([0, 3] if mode == "sustained" else list(range(len(SCENARIOS))))
    if only:
        wanted = [name.strip().upper() for name in only.split(",") if name.strip()]
        selected = [i for i in selected if SCENARIOS[i]["id"] in wanted]
        if not selected:
            raise SystemExit(f"--only {only!r} matched no scenario of mode {mode}")
    results = []
    for index in selected:
        scenario = SCENARIOS[index]
        print(f"[{mode}] {scenario['id']}: {scenario['name']} ({frames} frames)...", flush=True)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--scenario", str(index),
             "--frames", str(frames), "--sample-interval",
             str(30 if mode == "sustained" else 0)],
            capture_output=True, text=True, cwd=_REPO, env=dict(os.environ, PYTHONPATH=_REPO))
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                payload = json.loads(line[len("RESULT_JSON:"):])
        if payload is None:
            print(f"  FAILED (exit {proc.returncode})\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}",
                  flush=True)
            results.append({"id": scenario["id"], "name": scenario["name"], "failed": True,
                            "stderr": proc.stderr[-2000:]})
            continue
        results.append(payload)
        print(f"  fps={payload['fps']} p50={payload['stages_ms'].get('frame_total', {}).get('p50')}ms "
              f"ffprobe_ok={payload['ffprobe']['ok']}", flush=True)

    baseline = next((r for r in results if r.get("id") == "A"), None)
    half = next((r for r in results if r.get("id") == "D"), None)
    gate = {
        "fps_100_percent": baseline.get("fps") if baseline else None,
        "fps_50_percent_native": half.get("fps") if half else None,
        "requirement": "fps_50_percent_native > fps_100_percent",
    }
    if gate["fps_100_percent"] and gate["fps_50_percent_native"]:
        gate["passed"] = bool(gate["fps_50_percent_native"] > gate["fps_100_percent"])
        gate["ratio"] = round(gate["fps_50_percent_native"] / gate["fps_100_percent"], 3)
    else:
        gate["passed"] = False
        gate["reason"] = "one of the two scenarios failed to produce a result"

    report = {
        "kind": f"e2e_{mode}_benchmark",
        "code_under_test_sha": git_sha(),
        "environment": resource_metrics.environment_info({
            "fixture_path": os.path.relpath(FIXTURE_PATH, _REPO),
            "fixture_sha256": sha256(FIXTURE_PATH),
            "fixture_width": FIXTURE_W, "fixture_height": FIXTURE_H,
            "fixture_fps": FIXTURE_FPS, "fixture_frame_count": FIXTURE_FRAMES,
            "fixture_codec": "h264",
            "encoder": "av1_nvenc (pipeline default; codec actually written is reported by ffprobe)",
            "cq": CQ,
            "flow_engine": FLOW_ENGINE, "flow_preset": FLOW_PRESET,
        }),
        "config": {"mode": mode, "frames": frames, "scenarios": [SCENARIOS[i] for i in selected]},
        "results": [{k: v for k, v in r.items() if k != "resource_samples"} for r in results],
        "performance_gate": gate,
    }
    sampled = [r for r in results if r.get("resource_samples") or r.get("frame_samples")]
    if sampled:
        report["resource_samples"] = {
            r["id"]: {"wallclock_samples": r.get("resource_samples", []),
                      "frame_samples": r.get("frame_samples", [])} for r in sampled}
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {out_path}", flush=True)
    print(json.dumps(report["performance_gate"], indent=2), flush=True)
    return 0 if gate["passed"] else 1


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=_REPO).stdout.strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=-1)
    parser.add_argument("--mode", choices=("quick", "sustained"), default="quick")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--sample-interval", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--only", default="",
                        help="run just these scenario ids, e.g. --only A,D")
    args = parser.parse_args()

    if args.scenario >= 0:
        run_scenario(args.scenario, args.frames, args.sample_interval)
        return 0
    return master(args.mode, args.frames, args.out, args.only)


if __name__ == "__main__":
    sys.exit(main())
