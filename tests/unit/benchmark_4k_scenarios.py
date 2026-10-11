#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark full export pipeline across multiple configurations in isolated processes:
1. Native 4K @ 100% Native Scale, Guidance=0
2. 4K @ 100% Native Scale, NVOF 4M (Guidance=1)
3. 4K @ 75% Working Scale, NVOF 4M + Residual
4. 4K @ 63% Working Scale, NVOF 4M + Residual
5. 4K @ 50% Working Scale, NVOF 4M + Residual
6. 4K @ 25% Working Scale, NVOF 4M + Residual

Archival: these scenarios were measured on the closed dlssnr_host.dll session, whose
wrapper (dlss_engine.py) and binary are retired from this tree, so this is the record
of how the numbers were produced, not a gate a current build can run.
"""
import os
import sys
import time
import json
import argparse
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

SCENARIOS = [
    {"name": "1. 4K @ 100% Native (Guidance=0)", "scale": 1.0, "gm": 0, "flow_engine": "nvof", "flow_q": "4M", "pyramid": False, "multi_pass": 1},
    {"name": "2. 4K @ 100% Native (NVOF 4M, Guidance=1)", "scale": 1.0, "gm": 1, "flow_engine": "nvof", "flow_q": "4M", "pyramid": False, "multi_pass": 1},
    {"name": "3. 4K @ 75% Scale (NVOF 4M + Residual)", "scale": 0.75, "gm": 1, "flow_engine": "nvof", "flow_q": "4M", "pyramid": True, "multi_pass": 1},
    {"name": "4. 4K @ 63% Scale (NVOF 4M + Residual)", "scale": 0.63, "gm": 1, "flow_engine": "nvof", "flow_q": "4M", "pyramid": True, "multi_pass": 1},
    {"name": "5. 4K @ 50% Scale (NVOF 4M + Residual)", "scale": 0.50, "gm": 1, "flow_engine": "nvof", "flow_q": "4M", "pyramid": True, "multi_pass": 1},
    {"name": "6. 4K @ 25% Scale (NVOF 4M + Residual)", "scale": 0.25, "gm": 1, "flow_engine": "nvof", "flow_q": "4M", "pyramid": True, "multi_pass": 1},
]

def create_synthetic_4k_video(path, num_frames=60):
    if os.path.exists(path):
        return
    import cv2
    import numpy as np
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 60.0, (3840, 2160))
    for i in range(num_frames):
        img = np.zeros((2160, 3840, 3), dtype=np.uint8)
        cv2.circle(img, (500 + i * 20, 500 + i * 10), 200, (0, 200, 255), -1)
        cv2.putText(img, f"Frame {i:04d}", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 4)
        out.write(img)
    out.release()

def run_single_scenario(sc_idx, num_frames=60):
    import cv2
    import numpy as np
    import dlss_engine
    import pipeline
    from optical_flow import get_frame_guidance_service

    sc = SCENARIOS[sc_idx]
    in_video = "bench_out/input_4k_test.mp4"
    name = sc["name"]
    scale = sc["scale"]
    gm = sc["gm"]
    pyramid = sc["pyramid"]
    mp = sc["multi_pass"]
    out_file = f"bench_out/out_{scale}_{gm}_{mp}.mkv"
    if os.path.exists(out_file):
        try: os.remove(out_file)
        except: pass

    w, h = 3840, 2160
    live_w, live_h = pipeline.resolve_working_dims(w, h, scale)
    settings = {
        "preset": 1, "style": 0, "intensity": 1.0, "local_tone": 1.0, "local_struct": 1.0,
        "skin_struct": 1.0, "use_auto_mask": 1, "ui_correction": 0,
        "working_scale": scale, "residual_pyramid": pyramid, "residual_strength": 1.0,
        "multi_pass": mp, "guidance_mode": gm, "flow_engine": sc["flow_engine"], "flow_quality": sc["flow_q"],
    }

    live = dlss_engine.Live(live_w, live_h, settings)
    actual_pass_count = len(live) if isinstance(live, (list, tuple)) else 1
    if actual_pass_count != mp:
        raise AssertionError(f"FAIL: actual_pass_count ({actual_pass_count}) != requested_pass_count ({mp})")

    guidance_svc = get_frame_guidance_service(sc["flow_engine"], sc["flow_q"]) if gm > 0 else None

    t_start = time.perf_counter()
    profiler = pipeline.StageProfiler()
    pipeline.export_dlss_stream(
        video=in_video,
        live=live,
        lock=None,
        frames=num_frames,
        fps=60.0,
        cq=18,
        with_audio=False,
        flow_provider=(pipeline.GuidanceServiceFlowProvider(guidance_svc, guidance_mode=sc.get("gm", 0))
                       if guidance_svc is not None else None),
        progress=lambda c, t, m: None,
        cancel=lambda: False,
        out_path=out_file,
        job_settings=settings,
        profiler=profiler,
    )
    t_elapsed = time.perf_counter() - t_start
    fps = num_frames / max(0.001, t_elapsed)
    summary = profiler.summary()
    last_rec = profiler.records[-1] if profiler.records else {}

    # Strict physical ffprobe integrity check
    ffprobe_ok = False
    ffprobe_frames = 0
    try:
        cmd_ff = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                  "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", out_file]
        res_ff = subprocess.run(cmd_ff, capture_output=True, text=True, timeout=15)
        if res_ff.returncode == 0 and res_ff.stdout.strip().isdigit():
            ffprobe_frames = int(res_ff.stdout.strip())
            ffprobe_ok = (ffprobe_frames >= num_frames)
    except Exception as e:
        pass

    result = {
        "name": name,
        "fps": round(fps, 2),
        "time": round(t_elapsed, 2),
        "num_frames": num_frames,
        "res": f"{live_w}x{live_h}",
        "requested_pass_count": mp,
        "actual_pass_count": actual_pass_count,
        "actual_working_width": last_rec.get("actual_working_width", live_w),
        "actual_working_height": last_rec.get("actual_working_height", live_h),
        "actual_dlss_feature_width": last_rec.get("actual_dlss_feature_width", live_w),
        "actual_dlss_feature_height": last_rec.get("actual_dlss_feature_height", live_h),
        "decode_actual": last_rec.get("decode_actual", "cpu"),
        "flow_engine": sc.get("flow_engine", "none") if gm > 0 else "disabled",
        "guidance_backend": last_rec.get("guidance_backend", "disabled"),
        "residual_backend": last_rec.get("residual_backend", "none"),
        "p50_ms": round(summary.get("frame_total", {}).get("p50", 0.0), 2),
        "p95_ms": round(summary.get("frame_total", {}).get("p95", 0.0), 2),
        "ffprobe_verified": ffprobe_ok,
        "ffprobe_frames": ffprobe_frames,
        "profiler_summary": summary,
    }
    print(f"RESULT_JSON:{json.dumps(result)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=-1)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    os.makedirs("bench_out", exist_ok=True)
    in_video = "bench_out/input_4k_test.mp4"
    create_synthetic_4k_video(in_video, num_frames=args.frames)

    if args.scenario >= 0:
        run_single_scenario(args.scenario, num_frames=args.frames)
        return

    # Master runner
    all_results = []
    all_profilers = {}
    python_exe = sys.executable
    print("================================================================================")
    print(f"        STARTING ISOLATED 4K FULL-PIPELINE MULTI-SCENARIO BENCHMARK ({args.frames} FRAMES)")
    print("================================================================================")

    for i, sc in enumerate(SCENARIOS):
        print(f"\n[{i+1}/{len(SCENARIOS)}] Running: {sc['name']} ...")
        proc = subprocess.run(
            [python_exe, __file__, "--scenario", str(i), "--frames", str(args.frames)],
            capture_output=True,
            text=True,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
            env=dict(os.environ, PYTHONPATH=".")
        )
        output = proc.stdout + "\n" + proc.stderr
        res = None
        for line in output.splitlines():
            if line.startswith("RESULT_JSON:"):
                res = json.loads(line[len("RESULT_JSON:"):])
                break
        if res:
            prof_summary = res.pop("profiler_summary", {})
            all_profilers[f"scenario_{i+1}"] = {
                "name": res["name"],
                "summary": prof_summary
            }
            all_results.append(res)
            print(f"  -> Finished: {res['res']} | Time: {res['time']:.2f}s | FPS: {res['fps']:.2f} | ffprobe: {'OK' if res.get('ffprobe_verified') else 'FAIL'}")
        else:
            print(f"  -> FAILED with exit code {proc.returncode}")
            print(output[-1000:])

    print("\n\n" + "="*125)
    print("                                FINAL 4K BENCHMARK SUMMARY TABLE")
    print("="*125)
    print(f"{'Scenario':<42} | {'Live Dims':<10} | {'Guidance':<24} | {'Residual':<12} | {'P50 (ms)':<9} | {'FPS':<6} | {'ffprobe':<7}")
    print("-"*125)
    for r in all_results:
        print(f"{r['name']:<42} | {r['res']:<10} | {r.get('guidance_backend', 'none'):<24} | {r.get('residual_backend', 'none'):<12} | {r.get('p50_ms', 0.0):<9.2f} | {r['fps']:<6.2f} | {'VERIFIED' if r.get('ffprobe_verified') else 'FAIL':<7}")
    print("="*125)

    # Save artifacts if out_dir specified
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "benchmark.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.out_dir, "profiler.json"), "w", encoding="utf-8") as f:
            json.dump(all_profilers, f, indent=2, ensure_ascii=False)
        print(f"\nArtifacts successfully written to {args.out_dir}")


if __name__ == "__main__":
    main()
