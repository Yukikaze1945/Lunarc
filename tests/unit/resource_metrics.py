"""Resource sampling for the residual and end-to-end benchmarks.

Windows has no usable getrusage for peak RSS, so this reads the same process counters
the OS shows in Task Manager, and adds GPU memory through nvidia-smi when present.
"""

import hashlib
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time

import psutil

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBRARIES = ("dlssnr_host.dll", "nvngx_dlssnr.dll", "native/nvof_bridge.dll",
             "native/residual_bridge.dll")

_proc = psutil.Process(os.getpid())


def rss_bytes() -> int:
    return int(_proc.memory_info().rss)


def peak_rss_bytes() -> int:
    """Peak working set of this process; monotonic for the life of the process."""
    if os.name == "nt":
        return int(_proc.memory_info().peak_wset)
    usage = _proc.memory_full_info()
    return int(getattr(usage, "peak_rss", 0) or usage.rss)


def thread_count() -> int:
    return int(_proc.num_threads())


def child_process_count() -> int:
    """Children that are still running: an exited-but-unwaited child is not an orphan."""
    try:
        live = 0
        for child in _proc.children(recursive=True):
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    live += 1
            except psutil.Error:
                continue
        return live
    except psutil.Error:
        return 0


def _nvidia_smi_rows(query, compute_apps=False):
    """Every nvidia-smi row for `query` as a list of stripped fields, or None."""
    flag = "--query-compute-apps=" if compute_apps else "--query-gpu="
    try:
        out = subprocess.run(["nvidia-smi", flag + query,
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return [[field.strip() for field in line.split(",")]
                for line in out.stdout.strip().splitlines() if line.strip()]
    except Exception:
        return None


def _nvidia_smi_fields(query):
    """First nvidia-smi row for `query` as a list of stripped fields, or None."""
    rows = _nvidia_smi_rows(query)
    return rows[0] if rows else None


def gpu_total_memory_used_bytes():
    """Memory in use on the whole first discrete GPU, or None when nvidia-smi is absent.

    Not this process' footprint: the counter is the card's, so it includes every other
    application holding GPU memory. Per-process attribution is a separate question and
    the driver does not answer it here - see process_gpu_memory_bytes().
    """
    fields = _nvidia_smi_fields("memory.used")
    try:
        return int(float(fields[0]) * 1024 * 1024) if fields else None
    except (TypeError, ValueError):
        return None


def process_gpu_memory_bytes(pid=None):
    """(bytes, source) for this process on the GPU; (None, why) when it cannot be known.

    NVIDIA's per-process table is only filled in for contexts the driver accounts
    individually. On this Windows driver every row of --query-compute-apps reports its
    memory field as [N/A] while naming plenty of processes, so the honest answer for a
    D3D12/CUDA process is None plus the source that was tried, never the card total.
    """
    pid = os.getpid() if pid is None else pid
    rows = _nvidia_smi_rows("pid,used_memory", compute_apps=True)
    if rows is None:
        return None, "unavailable: nvidia-smi did not answer"
    for row in rows:
        try:
            if int(row[0]) != int(pid):
                continue
        except (TypeError, ValueError):
            continue
        try:
            return int(float(row[1]) * 1024 * 1024), "nvidia-smi --query-compute-apps"
        except (TypeError, ValueError):
            return None, ("unavailable: nvidia-smi lists this pid but reports its memory "
                          "as %r, so no per-process GPU footprint exists on this driver"
                          % (row[1] or ""))
    return None, "unavailable: this pid is not listed by nvidia-smi --query-compute-apps"


def _cpu_model():
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameValue")[0]).strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def environment_info(extra=None) -> dict:
    import numpy as np
    try:
        import cv2
        cv_ver = cv2.__version__
    except Exception:
        cv_ver = None
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=15)
        first = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
        ffmpeg_ver = first.split("version")[1].split()[0] if "version" in first else first[:60]
    except Exception:
        ffmpeg_ver = None

    gpu_fields = _nvidia_smi_fields("name,driver_version,memory.total")
    gpu = None
    if gpu_fields and len(gpu_fields) >= 3:
        try:
            total_vram = int(float(gpu_fields[2]) * 1024 * 1024)
        except (TypeError, ValueError):
            total_vram = None
        gpu = {"name": gpu_fields[0], "driver_version": gpu_fields[1],
               "memory_total_bytes": total_vram}

    libraries = {}
    for rel in LIBRARIES:
        path = os.path.join(_REPO, rel.replace("/", os.sep))
        if not os.path.exists(path):
            libraries[rel] = None
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        libraries[rel] = {"sha256": digest.hexdigest(), "bytes": os.path.getsize(path)}

    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        # The banner reports the highest CUDA runtime the driver supports, which is the
        # number that decides whether the NVENC / NVOF paths can load. Drivers word it as
        # "CUDA Version" or, on newer branches, "CUDA UMD Version".
        match = re.search(r"CUDA(?:\s+[A-Z]{1,3})?\s+Version:\s*([0-9.]+)", out.stdout or "")
        cuda_version = match.group(1) if match else None
    except Exception:
        cuda_version = None

    process_gpu, process_gpu_source = process_gpu_memory_bytes()

    info = {
        "cuda_version": cuda_version,
        "native_libraries": libraries,
        "utc_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": sys.platform,
        "os": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_arch": platform.machine(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_bytes": int(psutil.virtual_memory().total),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv_ver,
        "ffmpeg": ffmpeg_ver,
        "gpu": gpu,
        "rss_bytes": rss_bytes(),
        "peak_rss_bytes": peak_rss_bytes(),
        "gpu_total_memory_used_bytes": gpu_total_memory_used_bytes(),
        "process_gpu_memory_bytes": process_gpu,
        "process_gpu_memory_source": process_gpu_source,
    }
    if extra:
        info.update(extra)
    return info


# ---- steady-state RSS growth -------------------------------------------------------------
#
# A sustained run has three parts, and only the middle one is a leak rate. Frames 0..~20%
# are warmup: the encoder, the model and the first plans bring RSS up to whatever the run
# will sit at. The last ~10% is where teardown starts to appear in the samples. Fitting
# across the whole run therefore measures the shape of the run, not the growth of the
# process: measured on a real 300-frame export, frames 60..270 sit inside 2.4 MB of each
# other while the sample at frame 300 is 174 MB below them, and a least-squares line over
# all ten samples came out at -307,312 bytes per frame. A gate that read the magnitude of
# that number concluded "growth" from the run finishing.
#
# So the fit is restricted to the window, and the result is reported as a signed slope plus
# the positive part of its projection. A negative slope is not proof of anything except that
# RSS came down during the window; the claim available from this fit is only that there is no
# sustained positive growth.

STEADY_FIT_METHOD = "least_squares_over_steady_state_frame_window"
STEADY_FIT_START_FRACTION = 0.20
STEADY_FIT_END_FRACTION = 0.90
MIN_STEADY_FIT_SAMPLES = 5
STEADY_GROWTH_THRESHOLD_FRACTION = 0.10


def steady_state_fit_window(total_frames):
    """(first, last) frame numbers the fit may use, inclusive.

    ceil/floor rather than rounding: for 300 frames the window is 60..270, which excludes
    the warmup region 0..59 and everything from 271 on - including frame 300, the sample
    that carries the encoder and process teardown."""
    total = int(total_frames)
    if total <= 0:
        return 0, 0
    return (int(math.ceil(total * STEADY_FIT_START_FRACTION)),
            int(math.floor(total * STEADY_FIT_END_FRACTION)))


def fit_rss_steady_state(frame_samples, total_frames):
    """Least-squares RSS slope over the steady-state window, with its own provenance.

    Samples are selected by their recorded frame number, never by position in the list: a
    run that sampled every 30th frame of 300 has eight usable samples (60, 90 ... 270) in an
    array of ten, and slicing the array by percentage would keep the teardown sample.

    Returns the window, the frames actually fitted, the signed slope, the signed and
    positive projected growth, and a status. It never returns a slope it could not justify:
    fewer than MIN_STEADY_FIT_SAMPLES frames in the window, or no spread in frame numbers,
    gives status "insufficient_samples" and no numbers to gate on.
    """
    fit_start, fit_end = steady_state_fit_window(total_frames)
    frames = sorted(int(s["frame"]) for s in frame_samples)
    by_frame = {}
    for sample in frame_samples:
        frame = int(sample["frame"])
        if fit_start <= frame <= fit_end:
            by_frame.setdefault(frame, float(sample["rss_bytes"]))
    xs = sorted(by_frame)
    ys = [by_frame[x] for x in xs]

    excluded_before = [f for f in frames if f < fit_start]
    excluded_after = [f for f in frames if f > fit_end]
    result = {
        "rss_fit_method": STEADY_FIT_METHOD,
        "rss_fit_frame_start": fit_start,
        "rss_fit_frame_end": fit_end,
        "rss_fit_frames": xs,
        "rss_fit_sample_count": len(xs),
        "rss_fit_total_frames": int(total_frames),
        "rss_fit_excludes_warmup": True,
        "rss_fit_excludes_teardown": True,
        "rss_fit_excluded_frames_before_window": excluded_before,
        "rss_fit_excluded_frames_after_window": excluded_after,
        "rss_slope_bytes_per_frame": None,
        "rss_predicted_growth_over_fit_window_bytes": None,
        "rss_predicted_signed_growth_bytes": None,
        "rss_predicted_positive_growth_bytes": None,
        "rss_fit_frame_span": (xs[-1] - xs[0]) if len(xs) >= 2 else 0,
        "rss_fit_status": "insufficient_samples",
        "rss_fit_reason": "",
    }
    if len(xs) < MIN_STEADY_FIT_SAMPLES:
        result["rss_fit_reason"] = (
            f"{len(xs)} samples inside frames {fit_start}..{fit_end}, "
            f"{MIN_STEADY_FIT_SAMPLES} are the minimum")
        return result
    denom = sum((x - sum(xs) / len(xs)) ** 2 for x in xs)
    if denom <= 0:
        result["rss_fit_reason"] = "every fitted sample carries the same frame number"
        return result
    mean_y = sum(ys) / len(ys)
    slope = sum((x - sum(xs) / len(xs)) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    span = xs[-1] - xs[0]
    signed_growth = slope * span
    positive_growth = max(0.0, signed_growth)
    result.update({
        "rss_fit_status": "ok",
        "rss_fit_reason": "",
        "rss_slope_bytes_per_frame": round(slope, 1),
        "rss_predicted_growth_over_fit_window_bytes": round(signed_growth, 1),
        "rss_predicted_signed_growth_bytes": round(signed_growth, 1),
        "rss_predicted_positive_growth_bytes": round(positive_growth, 1),
    })
    return result


def fit_rss_whole_range(frame_samples):
    """The slope the collector used to report: least squares over every sample it got.

    Kept only so the steady-state fit can be shown against what it replaced, on the same
    numbers, in the regression tests and in the recalculation artifact. Nothing gates on it:
    over a completed run it measures the ramp-up and the teardown, not the growth.
    """
    xs = sorted(int(s["frame"]) for s in frame_samples)
    if len(xs) < 2:
        return {"rss_slope_bytes_per_frame": None, "rss_fit_sample_count": len(xs)}
    by_frame = {int(s["frame"]): float(s["rss_bytes"]) for s in frame_samples}
    ys = [by_frame[x] for x in xs]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    slope = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom) if denom else 0.0
    return {"rss_slope_bytes_per_frame": round(slope, 1),
            "rss_fit_sample_count": len(xs),
            "rss_frame_range": [xs[0], xs[-1]],
            "rss_predicted_growth_over_range_bytes": round(slope * (xs[-1] - xs[0]), 1)}


def rss_stability_gate(fit, peak_rss_bytes,
                       threshold_fraction=STEADY_GROWTH_THRESHOLD_FRACTION):
    """DONE_PROVEN only when the steady-state window shows no positive growth to speak of.

    The threshold is unchanged from the previous gate: positive growth under 10% of peak RSS.
    What changed is what goes into it - the steady-state window instead of the whole run, and
    the positive part of the projection instead of its absolute value, because a run whose
    RSS came down was being failed for coming down.
    """
    peak = float(peak_rss_bytes or 0)
    if fit.get("rss_fit_status") != "ok":
        return {"status": "NOT_READY",
                "reason": "insufficient steady-state RSS samples",
                "rss_fit_sample_count": fit.get("rss_fit_sample_count", 0),
                "rss_fit_detail_reason": fit.get("rss_fit_reason", "")}
    if peak <= 0:
        return {"status": "NOT_READY",
                "reason": "no peak RSS to compare the growth against",
                "rss_fit_sample_count": fit.get("rss_fit_sample_count", 0)}
    positive = float(fit["rss_predicted_positive_growth_bytes"])
    of_peak = positive / peak
    gate = {"status": "DONE_PROVEN" if of_peak < threshold_fraction else "NOT_READY",
            "reason": ("steady-state RSS shows no sustained positive growth"
                       if of_peak < threshold_fraction else
                       "steady-state RSS projects positive growth above the threshold"),
            "rss_predicted_positive_growth_bytes": positive,
            "rss_predicted_signed_growth_bytes": fit["rss_predicted_signed_growth_bytes"],
            "of_peak": round(of_peak, 6),
            "threshold_fraction_of_peak": threshold_fraction,
            "rss_fit_sample_count": fit["rss_fit_sample_count"],
            "rss_fit_frames": fit["rss_fit_frames"]}
    return gate


class Sampler(threading.Thread):
    """Polls RSS, card GPU memory, threads and children for stability runs.

    The GPU column is the card's total in use, deliberately not labelled as this
    process' usage; the per-process figure is attempted once in environment_info().
    """

    def __init__(self, interval_sec=1.0):
        super().__init__(daemon=True)
        self.interval_sec = interval_sec
        self.samples = []
        # Never name this _stop: Thread._stop is a CPython internal that join() calls.
        self._stopping = threading.Event()

    def run(self):
        while not self._stopping.is_set():
            self.samples.append({
                "t_sec": round(time.perf_counter() - self._t0, 3) if hasattr(self, "_t0") else 0.0,
                "rss_bytes": rss_bytes(),
                "peak_rss_bytes": peak_rss_bytes(),
                "gpu_total_memory_used_bytes": gpu_total_memory_used_bytes(),
                "thread_count": thread_count(),
                "child_process_count": child_process_count(),
            })
            self._stopping.wait(self.interval_sec)

    def start(self):
        self._t0 = time.perf_counter()
        super().start()

    def stop(self):
        self._stopping.set()
        self.join(timeout=2 * self.interval_sec + 1)
        return self.samples
