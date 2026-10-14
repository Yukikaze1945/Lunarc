"""Phase F: cancellation through the real DLSSService, not a synthetic pipe.

A job is started against the production export path, the test waits until both ffmpeg
children (decoder and encoder) are registered, then cancels. Afterwards the job must be
cancelled, the producer thread gone, both child PIDs dead, the service's process
registry empty, and the partial output must not be reported as completed.

The capability probe is a real DLSS session creation. If that fails this machine cannot
run the neural path, so the test skips; once it succeeds, any later exception fails the
test instead of being swallowed into a skip.
"""

import json
import os
import sys
import threading
import time

import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dlss_gpu_engine  # noqa: E402
import dlss_service  # noqa: E402
import pipeline  # noqa: E402

FIXTURE = os.path.join(_ROOT, "tests", "unit", "fixture_1080p_600f.mp4")
FRAMES = 600
WIDTH, HEIGHT = 1920, 1080


def _probe_live_session():
    """Return None when the neural path is usable, else the reason it is not.

    Probed on the entry the job below runs on: the GPU host's resident zero-motion
    texture, since guidance is off. A host that cannot answer that has no business
    serving this test's export either.
    """
    try:
        live = dlss_gpu_engine.Live(WIDTH, HEIGHT, {"preset": 1, "style": 0,
                                                    "intensity": 1.0, "guidance_mode": 0})
    except Exception as exc:
        return f"DLSS session unavailable: {exc}"
    try:
        if not live.supports_zero_motion():
            return ("DLSS session binds no zero-motion texture: "
                    f"{getattr(live, 'last_error', 'abi=' + str(getattr(live, 'host_abi', '?')))}")
        rgba = np.dstack([np.zeros((HEIGHT, WIDTH, 3), np.uint8),
                          np.full((HEIGHT, WIDTH), 255, np.uint8)])
        if live.process_zero_motion(rgba, reset=True) is None:
            return f"DLSS first inference failed: {getattr(live, 'last_error', 'unknown')}"
    finally:
        try:
            live.close()
        except Exception:
            pass
    return None


@pytest.fixture(scope="module")
def probe():
    reason = _probe_live_session()
    if reason:
        pytest.skip(reason)


def make_fixture():
    if os.path.exists(FIXTURE):
        return
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={WIDTH}x{HEIGHT}:rate=60:duration={FRAMES / 60:.6f}",
           "-frames:v", str(FRAMES), "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", FIXTURE]
    result = subprocess_run_or_raise(cmd)
    if not os.path.exists(FIXTURE):
        pytest.skip(f"fixture generation failed: {result}")


def subprocess_run_or_raise(cmd):
    import subprocess
    return subprocess.run(cmd, capture_output=True, text=True, timeout=1800)


def _live_threads(named):
    return [t for t in threading.enumerate() if t.name == named and t.is_alive()]


def test_cancelling_a_real_export_job_leaves_no_orphans(probe, tmp_path):
    make_fixture()
    out_path = str(tmp_path / "cancel_target.mkv")

    svc = dlss_service.DLSSService()
    svc.jobs = []
    job = {
        "id": "cancel-e2e", "name": "cancel e2e", "input_path": FIXTURE,
        "output_path": out_path, "container": "mkv", "cq": 18, "with_audio": False,
        # "pending" is the only state the queue renders (see DLSSService._run_queue);
        # this hand-built row has to enter with it or the queue would correctly skip it.
        "codec": "av1_nvenc", "status": "pending", "progress": 0.0, "current_frame": 0,
        "total_frames": FRAMES,
        "settings": {"preset": 1, "style": 0, "intensity": 1.0, "local_tone": 1.0,
                     "local_struct": 1.0, "skin_struct": 1.0, "use_auto_mask": 1,
                     "ui_correction": 0, "guidance_mode": 0,
                     "motion_scale_x": 1.0, "motion_scale_y": 1.0, "working_scale": 0.5,
                     "residual_pyramid": True, "residual_strength": 1.0,
                     "multi_pass": 1, "flow_engine": "nvof", "flow_quality": "1F",
                     "residual_backend": "native"},
        "transcode_settings": None,
    }
    svc.jobs = [job]

    started = time.perf_counter()
    assert svc.start_queue()
    queue_thread = None
    cancel_latency_ms = float("nan")
    try:
        deadline = time.time() + 90
        while time.time() < deadline and len(svc._active_procs) < 2:
            time.sleep(0.05)
        pids = [proc.pid for proc in list(svc._active_procs)]
        assert len(pids) == 2, f"decoder+encoder never both registered (got {pids})"

        # Render deep enough that the encoder has flushed packets: a cancel during
        # startup legitimately produces a zero-byte file, which would not test a
        # mid-stream cancel.
        frame_deadline = time.time() + 120
        while time.time() < frame_deadline and job["current_frame"] < 40:
            time.sleep(0.05)
        frames_at_cancel = job["current_frame"]

        cancel_start = time.perf_counter()
        svc.cancel_active_job()
        queue_thread = svc._queue_thread
        if queue_thread is not None:
            queue_thread.join(timeout=30)
        cancel_latency_ms = (time.perf_counter() - cancel_start) * 1000.0
    finally:
        # Never leave ffmpeg children behind if the body above failed early.
        svc.cancel_active_job()

    elapsed = time.perf_counter() - started
    report = {
        "code_under_test_sha": _git_sha(),
        "fixture_sha256": _sha(FIXTURE),
        "frames_requested": FRAMES,
        "frames_at_cancel": frames_at_cancel,
        "decoder_encoder_pids": pids,
        "cancel_latency_ms": round(cancel_latency_ms, 2),
        "job_status": job["status"],
        "job_error": str(job.get("error"))[:400],
        "job_error_tail": str(job.get("error_traceback") or "")[-1500:],
        "producer_threads_alive": [t.name for t in _live_threads("VideoDecoderProducer")],
        "queue_thread_alive": bool(queue_thread and queue_thread.is_alive()),
        "registry_after": len(svc._active_procs),
        "pids_still_running": sorted(pid for pid in pids if _pid_alive(pid)),
        "output_exists": os.path.exists(out_path),
        "output_bytes": os.path.getsize(out_path) if os.path.exists(out_path) else 0,
        "written_packets": (_probe_packets(out_path)
                            if os.path.exists(out_path) and os.path.getsize(out_path) else None),
        "job_elapsed_sec": round(elapsed, 3),
    }
    print("CANCEL_E2E:" + json.dumps(report), flush=True)

    assert report["producer_threads_alive"] == [], "decoder producer thread still running"
    assert not report["queue_thread_alive"], "render queue thread still running"
    assert report["registry_after"] == 0, "service still tracks child processes after cancel"
    assert report["pids_still_running"] == [], "orphan ffmpeg child processes survived cancel"
    assert job["status"] == "cancelled", f"job status was {job['status']}, expected cancelled"
    assert report["frames_at_cancel"] >= 1, \
        "cancel happened before any frame rendered, so the mid-flight path was not exercised"
    assert report["frames_at_cancel"] < FRAMES, \
        "the whole job rendered before the cancel landed, so nothing was interrupted"
    assert job.get("progress", 100.0) < 100.0, "a cancelled job reports full progress"
    # NVENC flushes in chunks, so a cancelled file may hold no readable packet yet, and
    # ffprobe reports null for a container it cannot parse. Neither is a defect; what
    # must never happen is a truncated export being presented as finished work.
    if report["written_packets"] is not None:
        assert report["written_packets"] < FRAMES, "the cancelled output looks complete"
    if report["output_bytes"]:
        assert report["output_bytes"] < os.path.getsize(FIXTURE), \
            "the cancelled output is as large as a full export of the same fixture"


def _probe_packets(path):
    """Packets ffprobe can read from the partial file, or None when it cannot.

    Returning 0 for a failed probe would make an unreadable container indistinguishable
    from a file that genuinely holds no packets.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60)
        text = result.stdout.strip()
        return int(text) if result.returncode == 0 and text.isdigit() else None
    except Exception:
        return None


def _pid_alive(pid):
    import psutil
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running() \
            and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def _sha(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=_ROOT).stdout.strip()
    except Exception:
        return "unknown"
