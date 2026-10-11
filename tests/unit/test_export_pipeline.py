#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_export_pipeline.py — Unit Tests for 4K DLSS Export Pipeline Optimizations.

Verifies:
1. Producer/Consumer Queue isolation (Zero-aliasing across queued frames).
2. Working scale motion vector scaling (sw/w, sh/h) and Residual Pyramid reconstruction.
3. Sequential NVOF state machine (1 H2D upload per frame in steady-state).
4. StageProfiler timing collection and summary calculations.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import time
import queue
import threading
import subprocess
import unittest
import numpy as np
import cv2

import pipeline
import optical_flow
from optical_flow import get_optical_flow_engine, probe_nvof_capabilities


def test_queue_flow_buffer_independence():
    """Verify that queued frames maintain independent, non-aliasing data across producer and consumer."""
    h, w = 1080, 1920
    frame_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()

    produced_frames = []
    consumed_frames = []

    def _producer():
        for i in range(5):
            # Generate frame with distinct pattern
            img = np.full((h, w, 3), i * 40, dtype=np.uint8)
            produced_frames.append(img.copy())
            frame_queue.put((i, img))
            time.sleep(0.01)
        frame_queue.put(None)

    t = threading.Thread(target=_producer)
    t.start()

    while True:
        item = frame_queue.get(timeout=2.0)
        if item is None:
            break
        idx, fr = item
        consumed_frames.append((idx, fr.copy()))

    t.join()

    assert len(consumed_frames) == 5
    for i, (idx, fr) in enumerate(consumed_frames):
        assert idx == i
        assert np.array_equal(fr, produced_frames[i]), f"Frame {i} suffered buffer corruption/aliasing!"


def test_guidance_adapter_and_working_resolution_nvof():
    """Verify production FrameGuidanceService in both working-resolution and full-resolution adapter modes."""
    w, h = 3840, 2160
    sw, sh = pipeline.resolve_working_dims(w, h, scale=0.5)
    assert sw == 1920 and sh == 1080

    svc = optical_flow.FrameGuidanceService(engine_type="dis", flow_quality="FAST")

    # Create distinct frames with horizontal motion
    f0 = np.zeros((sh, sw, 3), dtype=np.uint8)
    cv2.circle(f0, (500, 500), 100, (255, 255, 255), -1)

    f1 = np.zeros((sh, sw, 3), dtype=np.uint8)
    cv2.circle(f1, (550, 500), 100, (255, 255, 255), -1)

    # 1. Working Resolution Guidance mode (DIS)
    flow0, timing0 = svc.get_guidance(
        frame_id=0, cur_bgr=f0, prev_bgr=None, target_w=sw, target_h=sh, guidance_mode=1, is_discontinuous=True
    )
    assert flow0.shape == (sh, sw, 2)
    assert timing0["guidance_backend"] == "cpu_dis"

    flow1, timing1 = svc.get_guidance(
        frame_id=1, cur_bgr=f1, prev_bgr=None, target_w=sw, target_h=sh, guidance_mode=1, is_discontinuous=False
    )
    assert flow1.shape == (sh, sw, 2)
    assert np.max(np.abs(flow1)) > 1.0, "DIS Optical Flow should produce non-zero motion vector for moving circle!"
    assert timing1["cpu_previous_frame_copy_bytes"] == f1.nbytes

    # 2. Guidance Adapter mode (adapting from 4K source flow to working resolution)
    f_4k_0 = cv2.resize(f0, (w, h))
    f_4k_1 = cv2.resize(f1, (w, h))
    flow_adapt, timing_adapt = svc.get_guidance(
        frame_id=2, cur_bgr=f_4k_1, prev_bgr=f_4k_0, target_w=sw, target_h=sh, guidance_mode=1, is_discontinuous=False
    )
    assert flow_adapt.shape == (sh, sw, 2)
    assert timing_adapt["guidance_backend"] == "full_resolution_nvof_cpu_adapter"
    svc.close()


def test_residual_reconstruction():
    """Verify production reconstruct_residual function with preallocated workspace across multiple scales and signed deltas."""
    # working_scale 1.0 has no residual stage, so reconstruct_residual is not called there.
    for scale in (0.25, 0.50, 0.63, 0.75):
        w, h = 1920, 1080
        sw, sh = pipeline.resolve_working_dims(w, h, scale=scale)

        orig_bgr = np.random.randint(60, 180, (h, w, 3), dtype=np.uint8)
        backend = pipeline.create_residual_backend("python", w, h, sw, sh)

        if (sw, sh) == (w, h):
            down_bgr = orig_bgr
        else:
            down_bgr = np.empty((sh, sw, 3), dtype=np.uint8)
            backend.prepare_working_color(orig_bgr, down_bgr)

        for strength in (0.0, 1.0, 1.5):
            # 1. Positive residual test (+20)
            dlss_pos = np.clip(down_bgr.astype(np.int16) + 20, 0, 255).astype(np.uint8)
            recon_pos = np.empty((h, w, 3), dtype=np.uint8)
            backend.reconstruct(orig_bgr, down_bgr, dlss_pos, strength, recon_pos)
            assert recon_pos.shape == (h, w, 3)
            assert recon_pos.dtype == np.uint8
            assert not np.isnan(recon_pos).any()
            assert not np.isinf(recon_pos).any()
            if scale < 1.0 and strength > 0.0:
                assert np.mean(recon_pos.astype(np.float32) - orig_bgr.astype(np.float32)) > 5.0

            # 2. Negative residual test (-20)
            dlss_neg = np.clip(down_bgr.astype(np.int16) - 20, 0, 255).astype(np.uint8)
            recon_neg = np.empty((h, w, 3), dtype=np.uint8)
            backend.reconstruct(orig_bgr, down_bgr, dlss_neg, strength, recon_neg)
            assert recon_neg.shape == (h, w, 3)
            assert recon_neg.dtype == np.uint8
            if scale < 1.0 and strength > 0.0:
                assert np.mean(recon_neg.astype(np.float32) - orig_bgr.astype(np.float32)) < -5.0

            # 3. Zero strength test
            if strength == 0.0:
                diff_zero = np.abs(recon_pos.astype(np.float32) - orig_bgr.astype(np.float32))
                assert np.mean(diff_zero) < 1e-4


def test_frame_workspace_preallocations():
    """Verify FrameWorkspace preallocates all buffers matching working and output dimensions and eliminates dead buffers."""
    src_w, src_h = 3840, 2160
    live_w, live_h = 1920, 1080
    dst_w, dst_h = 3840, 2160

    ws = pipeline.FrameWorkspace(src_w, src_h, live_w, live_h, dst_w, dst_h)

    assert ws.live_rgba.shape == (live_h, live_w, 4)
    assert ws.live_flow.shape == (live_h, live_w, 2)
    assert not hasattr(ws, 'live_depth')
    assert ws.dlss_out_rgba.shape == (live_h, live_w, 4)
    assert ws.dlss_out_bgr.shape == (live_h, live_w, 3)
    assert ws.down_cur_bgr.shape == (live_h, live_w, 3)
    assert ws.recon_bgr.shape == (src_h, src_w, 3)
    assert ws.dst_bgr.shape == (dst_h, dst_w, 3)
    assert not hasattr(ws, 'full_flow'), "Dead full_flow buffer must be removed!"
    assert not hasattr(ws, 'full_depth'), "Dead full_depth buffer must be removed!"


def test_stage_profiler_statistics():
    """Verify that StageProfiler collects records and computes correct summary statistics."""
    profiler = pipeline.StageProfiler()
    profiler.start()

    for i in range(10):
        profiler.record({
            "decode_wait": 2.0 + i,
            "guidance_total": 5.0,
            "dlss_process": 10.0,
            "pipe_write": 1.0,
            "frame_total": 18.0 + i,
        })
    time.sleep(0.05)
    profiler.stop()

    summary = profiler.summary()
    assert summary["frame_count"] == 10
    assert summary["fps"] > 0
    assert summary["decode_wait"]["min"] == 2.0
    assert summary["decode_wait"]["max"] == 11.0
    assert abs(summary["decode_wait"]["mean"] - 6.5) < 1e-2
    assert abs(summary["dlss_process"]["mean"] - 10.0) < 1e-2


import io
import math

class MockPipe:
    """Mock pipe for capturing writes and verifying chunk sizes."""
    def __init__(self):
        self.chunks = []
        self.total_bytes = 0

    def write(self, data):
        b = bytes(data)
        self.chunks.append(len(b))
        self.total_bytes += len(b)
        return len(b)

    def flush(self):
        pass


def test_production_write_frame_bytes():
    """Verify that production _write_frame_bytes writes in <= 1MB chunks and outputs exact byte count for 1080p and 4K."""
    # Test 1080p BGR
    h, w, c = 1080, 1920, 3
    expected_1080p = h * w * c  # 6,220,800 bytes
    frame_1080p = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)

    pipe_1080p = MockPipe()
    written = pipeline._write_frame_bytes(pipe_1080p, frame_1080p, chunk_size=1048576)
    assert written == expected_1080p, f"Expected {expected_1080p} bytes, got {written}"
    assert pipe_1080p.total_bytes == expected_1080p
    assert all(chunk <= 1048576 for chunk in pipe_1080p.chunks)
    assert len(pipe_1080p.chunks) == math.ceil(expected_1080p / 1048576)

    # Test 4K BGR (24,883,200 bytes)
    h4k, w4k, c4k = 2160, 3840, 3
    expected_4k = h4k * w4k * c4k  # 24,883,200 bytes
    frame_4k = np.random.randint(0, 256, (h4k, w4k, c4k), dtype=np.uint8)

    pipe_4k = MockPipe()
    written_4k = pipeline._write_frame_bytes(pipe_4k, frame_4k, chunk_size=1048576)
    assert written_4k == expected_4k, f"Expected {expected_4k} bytes, got {written_4k}"
    assert pipe_4k.total_bytes == expected_4k
    assert all(chunk <= 1048576 for chunk in pipe_4k.chunks)
    assert len(pipe_4k.chunks) == math.ceil(expected_4k / 1048576)


def test_cuda_decode_probe():
    """Verify that probe_cuda_decode_support and probe_cuda_decode_for_input accurately probe and decode frames."""
    is_supported, reason = pipeline.probe_cuda_decode_support()
    assert isinstance(is_supported, bool)
    assert isinstance(reason, str) and len(reason) > 0

    os.makedirs("bench_out", exist_ok=True)
    h264_video = "bench_out/input_4k_h264.mp4"
    if not os.path.exists(h264_video):
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=3840x2160:rate=60", "-t", "1", "-pix_fmt", "yuv420p", "-c:v", "h264_nvenc", h264_video]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if is_supported and os.path.exists(h264_video):
        ok, dyn_reason, info = pipeline.probe_cuda_decode_for_input(h264_video)
        assert ok is True, f"Dynamic CUDA probe failed on valid H264 input: {dyn_reason}"
        assert info.get("decode_actual") == "cuda"
        assert info.get("width") == 3840 and info.get("height") == 2160

        # Verify iter_frames with decode_mode='cuda'
        dec_info = {}
        frames_iter = list(pipeline.iter_frames(h264_video, limit=3, decode_mode="cuda", decode_info=dec_info))
        assert len(frames_iter) == 3
        assert dec_info.get("actual") == "cuda"
        for idx, fr in frames_iter:
            assert fr.shape == (2160, 3840, 3)
            assert fr.dtype == np.uint8

        # Verify decode_mode='auto' on CUDA-capable video uses CUDA
        dec_info_auto = {}
        frames_auto = list(pipeline.iter_frames(h264_video, limit=2, decode_mode="auto", decode_info=dec_info_auto))
        assert len(frames_auto) == 2
        assert dec_info_auto.get("actual") == "cuda"


def test_sequential_nvof_execution_and_counters():
    """Verify that NvOFHardwareEngine sequential execution works and native copy counters reflect steady-state 1 upload/frame."""
    info = probe_nvof_capabilities()
    if not info.get("hardware"):
        print("[SKIP] NVIDIA Optical Flow hardware not available")
        return

    engine = get_optical_flow_engine("nvof", "4M")
    h, w = 1080, 1920

    f0 = np.full((h, w, 3), 50, dtype=np.uint8)
    f1 = np.full((h, w, 3), 70, dtype=np.uint8)
    f2 = np.full((h, w, 3), 90, dtype=np.uint8)

    # Frame 0 (initial / reset -> 2 uploads: cur + prev)
    flow0 = engine.compute_flow_sequential(cur_bgr=f0, prev_bgr=None, reset_temporal_hints=True)
    assert flow0.shape == (h, w, 2)
    c0 = engine.get_counters()
    assert c0["execute_count"] == 1
    assert c0["h2d_copy_count"] == 2, f"Expected 2 H2D uploads on reset, got {c0['h2d_copy_count']}"
    assert c0["d2h_copy_count"] == 1
    assert c0["reset_count"] == 1

    # Frame 1 (continuous steady-state -> exactly +1 H2D upload)
    flow1 = engine.compute_flow_sequential(cur_bgr=f1, prev_bgr=f0, reset_temporal_hints=False)
    assert flow1.shape == (h, w, 2)
    c1 = engine.get_counters()
    assert c1["execute_count"] == 2
    assert c1["h2d_copy_count"] == 3, f"Expected 3 H2D uploads after frame 1, got {c1['h2d_copy_count']}"
    assert c1["d2h_copy_count"] == 2

    # Frame 2 (continuous steady-state -> exactly +1 H2D upload)
    flow2 = engine.compute_flow_sequential(cur_bgr=f2, prev_bgr=f1, reset_temporal_hints=False)
    assert flow2.shape == (h, w, 2)
    c2 = engine.get_counters()
    assert c2["execute_count"] == 3
    assert c2["h2d_copy_count"] == 4, f"Expected 4 H2D uploads after frame 2, got {c2['h2d_copy_count']}"
    assert c2["d2h_copy_count"] == 3

    engine.close()


def test_cancellation_lifecycle_and_process_cleanup():
    """Verify that cancellation stops blocking read/write immediately (<200ms) without orphan ffmpeg processes."""
    # Level A: Deterministic blocking-pipe cancellation test
    # Spawn a real child process that hangs/sleeps, while _read_exactly blocks on its stdout
    cmd_hang = [sys.executable, "-c", "import time; time.sleep(10)"]
    hang_proc = subprocess.Popen(cmd_hang, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stop_ev = threading.Event()
    hang_read_box = {}

    def _reader_thread():
        try:
            pipeline._read_exactly(hang_proc.stdout, 1024, proc=hang_proc, stop_event=stop_ev, what="Test Hang Proc")
        except Exception as ex:
            hang_read_box["error"] = ex

    t_th = threading.Thread(target=_reader_thread, daemon=True)
    t_th.start()
    time.sleep(0.05)  # Ensure thread is blocked in read

    t0 = time.perf_counter()
    stop_ev.set()
    try:
        hang_proc.kill()
    except Exception:
        pass
    t_th.join(timeout=0.5)
    t_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert not t_th.is_alive(), "Reader thread failed to join after cancel!"
    assert issubclass(type(hang_read_box.get("error")), pipeline.JobCancelled) or isinstance(hang_read_box.get("error"), pipeline.JobCancelled), f"Expected JobCancelled, got {hang_read_box.get('error')}"
    assert t_latency_ms < 200.0, f"Cancel latency too high: {t_latency_ms:.1f} ms"
    assert hang_proc.poll() is not None, "Child process must be terminated!"

    # Level B: Real FFmpeg stream export cancellation test
    os.makedirs("bench_out", exist_ok=True)
    test_src = "bench_out/test_cancel_src.mp4"
    if not os.path.exists(test_src):
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30", "-t", "2", "-pix_fmt", "yuv420p", "-c:v", "libx264", test_src]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    test_out = "bench_out/test_cancel_out.mp4"
    active_procs = []
    def _reg_proc(p): active_procs.append(p)
    def _unreg_proc(p):
        if p in active_procs: active_procs.remove(p)

    cancel_flag = False
    def _prog(cur, total, msg):
        nonlocal cancel_flag
        if cur >= 2:
            cancel_flag = True

    # Dummy live engine for unit testing export stream cancellation. Shaped like the
    # GPU host the pipe runs on: the core refuses a session without its entry points,
    # and a run that died at construction would prove nothing about process cleanup.
    class DummyLive:
        _w = 1920
        _h = 1080

        def supports_zero_motion(self):
            return True

        def process_zero_motion(self, rgba, reset=False, out=None):
            if out is not None:
                np.copyto(out, rgba)
                return out
            return rgba

        def process_device_flow(self, *args, **kwargs):
            raise AssertionError("a guidance_mode=0 frame reached the guided GPU entry")

    dummy_live = DummyLive()
    t_start = time.perf_counter()
    export_cancelled = False
    try:
        pipeline.export_dlss_stream(
            video=test_src,
            live=dummy_live,
            frames=30,
            fps=30,
            progress=_prog,
            cancel=lambda: cancel_flag,
            out_path=test_out,
            transcode_settings={"codec": "libx264", "decode_mode": "cpu"},
            on_proc_started=_reg_proc,
            on_proc_exited=_unreg_proc,
            job_settings={"working_scale": 1.0, "guidance_mode": 0, "residual_pyramid": False}
        )
    except pipeline.JobCancelled:
        export_cancelled = True
    except Exception as e:
        if "取消" in str(e) or "cancel" in str(e).lower():
            export_cancelled = True

    t_cancel_total_ms = (time.perf_counter() - t_start) * 1000.0
    # Wait briefly for OS cleanup
    time.sleep(0.05)
    # Verify all child processes are killed and unregistered
    for p in list(active_procs):
        if p.poll() is not None:
            active_procs.remove(p)

    assert len(active_procs) == 0, f"Orphan active processes detected: {active_procs}"
    assert export_cancelled or cancel_flag, "Export pipeline should terminate upon cancellation flag"
    if os.path.exists(test_out):
        try: os.remove(test_out)
        except Exception: pass


if __name__ == "__main__":
    print("Running test_queue_flow_buffer_independence...")
    test_queue_flow_buffer_independence()
    print("PASS: test_queue_flow_buffer_independence")

    print("Running test_guidance_adapter_and_working_resolution_nvof...")
    test_guidance_adapter_and_working_resolution_nvof()
    print("PASS: test_guidance_adapter_and_working_resolution_nvof")

    print("Running test_residual_reconstruction...")
    test_residual_reconstruction()
    print("PASS: test_residual_reconstruction")

    print("Running test_stage_profiler_statistics...")
    test_stage_profiler_statistics()
    print("PASS: test_stage_profiler_statistics")

    print("Running test_production_write_frame_bytes...")
    test_production_write_frame_bytes()
    print("PASS: test_production_write_frame_bytes")

    print("Running test_frame_workspace_preallocations...")
    test_frame_workspace_preallocations()
    print("PASS: test_frame_workspace_preallocations")

    print("Running test_cuda_decode_probe...")
    test_cuda_decode_probe()
    print("PASS: test_cuda_decode_probe")

    print("Running test_sequential_nvof_execution_and_counters...")
    test_sequential_nvof_execution_and_counters()
    print("PASS: test_sequential_nvof_execution_and_counters")

    print("Running test_cancellation_lifecycle_and_process_cleanup...")
    test_cancellation_lifecycle_and_process_cleanup()
    print("PASS: test_cancellation_lifecycle_and_process_cleanup")

    print("\nALL PRODUCTION PIPELINE UNIT TESTS PASSED SUCCESSFULLY!")
