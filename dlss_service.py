#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dlss_service.py — Backend Processing Core for DLSS5 Standalone.

Features:
1. Low-latency frame decoding and LRU frame cache.
2. Realtime DLSS LiveSession management with dynamic parameters.
3. Multi-Engine Optical Flow (DIS CPU under nvof/amd compat names / RAFT-Large in-memory).
4. Residual Pyramid Decomposition (Magpie-style working resolution scaling).
5. Render Queue for batch export piping CPU-rendered frames to an FFmpeg NVENC process.
"""
import os
import re
import sys
import time
import math
import uuid
import threading
import traceback
from collections import OrderedDict
import numpy as np
import cv2

os.environ.setdefault("OPENCV_LOG_LEVEL", "FATAL")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)

import pipeline
from dlss_errors import DLSSInferenceError
from optical_flow import get_optical_flow_engine, get_frame_guidance_service


def _raft_flow_options(settings):
    if settings.get('flow_engine') not in ('raft', 'raft_large'):
        return {}
    return dict(raft_analysis_edge=settings.get('raft_analysis_edge'),
                raft_iterations=settings.get('raft_iterations'))


# The native DLSS host a request that names none is served on. Every path a frame
# can take in this service is routed on the project GPU host - the preview in
# either guidance mode, and an export on either chain with NVOF device flow or
# with a host motion plane - so it is the only host left and the only value a
# caller can name. The value has to be one of pipeline.DLSS_BACKENDS; that pair,
# not this constant, is what validation checks against.
DEFAULT_DLSS_BACKEND = "gpu_flow_d3d12"


# What _execute_job hands back when it left the destination exactly as it found it.
# The queue's own state for that outcome is the same word; the reason it is a value
# rather than a bare string in two places is that a job that returns early and a job
# that was never asked to run must not be reported by the same notice.
JOB_SKIPPED = "skipped"


class FrameCache:
    """Thread-safe LRU Cache with Byte-budget for decoded video frames."""
    def __init__(self, max_bytes=256 * 1024 * 1024, maxsize=None):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key, value):
        if value is None:
            return
        val_bytes = value.nbytes if hasattr(value, "nbytes") else 1024
        with self._lock:
            if key in self._cache:
                old_val = self._cache[key]
                self.current_bytes -= (old_val.nbytes if hasattr(old_val, "nbytes") else 1024)
                self._cache.move_to_end(key)
            while self._cache and (self.current_bytes + val_bytes > self.max_bytes):
                _, evicted = self._cache.popitem(last=False)
                self.current_bytes -= (evicted.nbytes if hasattr(evicted, "nbytes") else 1024)
            self._cache[key] = value
            self.current_bytes += val_bytes

    def clear(self):
        with self._lock:
            self._cache.clear()
            self.current_bytes = 0


class DiagnosticLogger:
    """Thread-safe in-memory rolling diagnostic log buffer for complete observability."""
    def __init__(self, maxsize=1000):
        self.maxsize = maxsize
        self._logs = []
        self._lock = threading.Lock()

    def log(self, level: str, tag: str, message: str, detail: str = ""):
        entry = {
            "time": time.strftime("%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}",
            "level": level.upper(),
            "tag": tag,
            "message": message,
            "detail": detail
        }
        with self._lock:
            self._logs.append(entry)
            if len(self._logs) > self.maxsize:
                self._logs.pop(0)
        prefix = f"[{entry['time']}][{entry['level']}][{tag}] {message}"
        if level.upper() in ("ERROR", "FATAL"):
            sys.stderr.write(prefix + "\n")
            if detail:
                sys.stderr.write(f"  └─ {detail}\n")
        else:
            sys.stdout.write(prefix + "\n")

    def get_logs(self, limit=200):
        with self._lock:
            return list(self._logs[-limit:])


diagnostic_logger = DiagnosticLogger(maxsize=1000)


class NativeCallTimeout(RuntimeError):
    """_run_with_timeout gave up on a native call that is still running.

    A type, not a substring match on the text: the fuse that decides the native layer
    may be polluted reads this fact alone, and an error that happens to contain the
    word 超时 - or a hang whose label spells it differently - is not that fact.
    """


def _run_with_timeout(fn, timeout, what):
    """Run fn() in a daemon thread with a timeout.

    Native calls (dlss_gpu_host_open/close, torch weight download) can block
    forever, uninterruptible from Python. A timeout turns the hang into a
    clear error so the job fails (or the caller falls back) instead of
    wedging the queue at 0 FPS with a dead cancel button. The stray thread is
    left dangling by design; the error message tells the user to restart the
    backend if the DLL state is broken. A timeout raises NativeCallTimeout,
    which is what the native fuse reads.
    """
    box = {}

    def _w():
        try:
            box["ret"] = fn()
        except BaseException as e:  # noqa: BLE001 - must capture everything
            box["err"] = e

    t = threading.Thread(target=_w, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise NativeCallTimeout(f"{what}超时（{timeout:.0f}s 无响应，可能已在原生层挂起；建议重启后端后重试）")
    if "err" in box:
        raise box["err"]
    return box.get("ret")


def _gpu_host_provenance(live):
    """Which native DLSS host a session is actually driving, for the job record.

    Read off the live object rather than off the settings: a job's own record has to
    name the binary that drew its frames, and the host that answered can only be a
    rebuild away from the one the caller asked for by name.
    """
    return {
        "backend": DEFAULT_DLSS_BACKEND,
        "dll": getattr(live, "host_dll", None),
        "abi": getattr(live, "host_abi", None),
    }


def _hdr_request(job_settings):
    """Whether this job opted into the experimental HDR chain, and any reason it can't.

    Read through the chain's own predicate rather than a copy of its vocabulary here,
    so the service's opt-in and the backend's opt-in cannot drift apart: a flag this
    layer understood and that layer did not would open an FP16 session for a job that
    then ran the SDR chain on it. An absent key is off, which is what every job in
    every saved configuration has always meant.
    """
    htx = __import__("gpu_hdr_transfer_export")
    requested = bool(htx.hdr_requested(job_settings))
    reasons = [reason for reason in (htx.hdr_flag_reason(job_settings),
                                     htx.paper_white_reason(job_settings)) if reason]
    return requested, reasons


class DLSSService:
    def __init__(self):
        self.video_path = None
        self.video_info = None  # (total_frames, fps, width, height)
        self._cap = None
        self._cap_lock = threading.Lock()
        self._settings_lock = threading.RLock()
        self._frame_cache = FrameCache(maxsize=64)

        # Settings
        self.settings = {
            "preset": 1,
            "style": 0,
            "intensity": 1.0,
            "local_tone": 1.0,
            "local_struct": 1.0,
            "skin_struct": 1.0,
            "use_auto_mask": 1,
            "ui_correction": 0,
            "guidance_mode": 0,        # 0=off, 1=flow
            "motion_scale_x": 1.0,
            "motion_scale_y": 1.0,
            "working_scale": 1.0,      # whole percentages 25..100 of the source;
                                       # the CPU path resolves any ratio through
                                       # resolve_working_dims, and
                                       # export_backend=gpu_transfer serves that
                                       # range on the resolved extents
            "residual_pyramid": True,
            "residual_strength": 1.0,
            "post_controls": __import__("post_control").PostControls().as_dict(),
            "temporal_controls": __import__("temporal_control").TemporalControls().as_dict(),
            "flow_engine": "nvof",      # "nvof", "dis", "raft"
            "flow_quality": "1F",       # "1F", "1M", "1S", "2F" (or legacy 4M/4F)
            "export_backend": "auto",   # "auto", "legacy", "gpu_transfer"
            "neural_mode": "nr",
            "vsr_quality": 2,
            "vsr_target": "1080p",
            "vsr_output_mode": "resolution",
            "vsr_scale": 2,
            "vsr_width": 1920,
            "vsr_height": 1080,
            "vsr_runtime_path": "",
            "frame_generation": "off",
            "frame_multiplier": 2,
            "framegen_quality": 1,
            "framegen_target_fps": 60,
            "framegen_scene_detection": True,
            "dlss_dll_path": "",
            "reconstruction_mode": "single",
            "layer_scales": [1.0],
            "layer_weights": [1.0],
            "fusion_mode": "catmull_baseline",
            "fusion_beta": 0.1,
            "fusion_max_iterations": 128,
            "fusion_relative_tolerance": 0.001,
            # Which native DLSS host draws a frame. The project GPU host is the only
            # one this build ships, for both preview modes and on either export
            # chain. The video chain above is a separate choice: an auto or legacy
            # pipe job runs its frames on this host too.
            "dlss_backend": DEFAULT_DLSS_BACKEND,
            # The experimental HDR10 chain. Off is the shipped default and the only
            # value a saved configuration has ever carried: a job that does not name
            # it runs the unchanged SDR chain. Turning it on selects the HDR-aware GPU
            # Transfer branch and an FP16 Feature 18 session, and nothing that branch
            # refuses is served another way.
            "experimental_hdr_scrgb": False,
            # The diffuse white the HDR colour bridge is opened with, in scRGB
            # (8.0 = 640 cd/m2 at this bridge's reference white). Chosen once per job
            # and never adapted per frame.
            "hdr_paper_white_scrgb": 8.0,
        }

        # DLSS Live Session
        self._live_session = None
        self._live_session_dims = None
        # Which GPU-host entry point backs the cached session: "gpu_zero_motion",
        # "gpu_host_motion", "gpu_flow" (an export's device-resident flow) or
        # "gpu_hdr_fp16" (the same, on an RGBA16F session). Kept so
        # a config change rebuilds the right session instead of driving one entry
        # point through another's inputs.
        self._live_session_kind = None
        # Why the last preview session refused to open, surfaced to the operator.
        self._preview_live_error = None
        self._live_lock = threading.Lock()

        # Preview residual backend, cached per (source, working, requested) triple so
        # plans and scratch are not rebuilt while a movie plays.
        self._preview_backend = None
        self._preview_backend_key = None
        self._preview_buffers = {}

        # Render Queue
        self.jobs = []
        self.active_job_id = None
        self._cancel_requested = False
        self._queue_thread = None
        self._queue_lock = threading.Lock()
        # Subprocesses owned by the running job (export FFmpeg). Killed on cancel
        # so “终止任务” really stops a stuck job instead of only setting a flag.
        self._active_procs = []
        self._procs_lock = threading.Lock()
        # P1-07 fuse: once a native call times out, its stray daemon thread may
        # still touch global DLL/D3D12 state later. Stop reusing the poisoned
        # process for new DLSS work until the backend is restarted.
        self._native_degraded = None

    def _mark_native_degraded(self, what):
        if not self._native_degraded:
            self._native_degraded = what
            diagnostic_logger.log("FATAL", "DLSS", f"原生层疑似污染，已熔断新任务：{what}。请重启后端后再试。")

    def _ensure_native_healthy(self):
        if self._native_degraded:
            raise RuntimeError(
                f"后端原生层已熔断（{self._native_degraded}），为防污染扩散已拒绝新任务；请重启后端后重试")

    def _register_proc(self, proc):
        with self._procs_lock:
            self._active_procs.append(proc)

    def _unregister_proc(self, proc):
        with self._procs_lock:
            try:
                self._active_procs.remove(proc)
            except ValueError:
                pass

    def _kill_active_procs(self):
        with self._procs_lock:
            procs = list(self._active_procs)
        for p in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass

    def load_video(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        with self._cap_lock:
            cap = pipeline.open_capture(path)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Failed to open video: {path}")

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0: fps = 30.0
            if total <= 0: total = 1

            # P1-01 honesty: expose source bit-depth/color so callers stop
            # pretending HDR survives the fixed 8-bit SDR neural path.
            pix = pipeline.probe_pixel_info(path)
            # A rejected drag/drop must not destroy the previously loaded source.
            if self._cap is not None:
                self._cap.release()
            self._cap = cap
            self.video_path = path
            if pix["is_hdr"] or pix["bit_depth"] > 8:
                diagnostic_logger.log(
                    "WARN", "Video",
                    f"输入为 {pix['pix_fmt'] or '未知格式'}/{pix['bit_depth']}-bit"
                    f"{' HDR' if pix['is_hdr'] else ' SDR'}；原生预览主色使用 FP16，显示图像为 8-bit；导出精度由导出链路决定。")
            self.video_info = {
                "path": path,
                "filename": os.path.basename(path),
                "frames": total,
                "fps": round(fps, 3),
                "width": w,
                "height": h,
                "duration_seconds": round(total / fps, 2),
                "pix_fmt": pix["pix_fmt"],
                "bit_depth": pix["bit_depth"],
                "is_hdr": pix["is_hdr"],
                "color_transfer": pix["color_transfer"],
                "color_primaries": pix.get("color_primaries", ""),
                "color_space": pix.get("color_space", ""),
                "color_range": pix.get("color_range", ""),
                "colour_resolution": pix.get("colour_resolution"),
            }
            self._frame_cache.clear()

        # The GPU owner thread closes/reopens on its next source-key change.
        # Do not wait for an old native frame here; WS revisions suppress it.
        self._preview_revision = getattr(self, '_preview_revision', 0) + 1
        self._preview_diagnostics = {}
        self._legacy_source_changed = True

        return self.video_info

    def get_frame(self, index: int):
        """Fetch decoded BGR frame by index."""
        if not self.video_path:
            return None
        cached = self._frame_cache.get(index)
        if cached is not None:
            return cached

        with self._cap_lock:
            if self._cap is None:
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, bgr = self._cap.read()
            if ret and bgr is not None:
                self._frame_cache.put(index, bgr)
                return bgr
            return None

    @staticmethod
    def _validated_dlss_backend(value):
        """The one host this build runs, or a refusal that names what was asked.

        Reading an unknown host as "the one we have" is the silent version of a
        frame running where nobody asked: a settings file, a queued job or a script
        written before the old host was retired still says host_pointer, and answering
        that by drawing the frame on the GPU host would report a run the caller never
        requested. So a retired name is refused by name, with the migration on the
        same line, and anything else is refused as what it is - an error out loud.
        """
        requested = str(value).strip().lower()
        if requested in pipeline.RETIRED_DLSS_BACKENDS:
            raise ValueError(pipeline.retired_dlss_backend_message(requested))
        if requested not in pipeline.DLSS_BACKENDS:
            raise ValueError(
                f"dlss_backend={requested!r} is not a DLSS host this service offers; "
                f"the choices are {', '.join(pipeline.DLSS_BACKENDS)}")
        return requested

    def _job_dlss_backend(self, job_settings):
        """The host a job's settings name, or the service's own when they name none.

        A job built by a client that predates this key carries no host at all, and it
        resolves to the same GPU host a new client gets. A job that names a host gets
        that host or a refusal: an explicit value is never quietly re-routed, which is
        what makes a pre-migration caller's host_pointer request fail here rather than
        run somewhere it did not ask for.
        """
        if job_settings.get("dlss_backend") is not None:
            return self._validated_dlss_backend(job_settings["dlss_backend"])
        return self._validated_dlss_backend(
            self.settings.get("dlss_backend") or DEFAULT_DLSS_BACKEND)

    def update_settings(self, new_settings: dict):
        with self._settings_lock:
            return self._update_settings_locked(new_settings)

    def _update_settings_locked(self, new_settings: dict):
        new_settings = dict(new_settings)
        if 'dlaa_preset' in new_settings:
            from native_effects import DLAA_PRESETS
            if isinstance(new_settings['dlaa_preset'], bool) or new_settings['dlaa_preset'] not in DLAA_PRESETS:
                raise ValueError('DLAA preset must be Default, E, F, J, K, L or M')
        if 'render_pipeline' in new_settings:
            import native_effects
            if new_settings['render_pipeline'] is not None:
                new_settings['render_pipeline'] = native_effects.stages(new_settings)
        for key, allowed in (('raft_analysis_edge', (512, 720, 1080)),
                             ('raft_iterations', range(1, 25))):
            if key in new_settings:
                value = new_settings[key]
                if isinstance(value, bool) or value not in allowed:
                    raise ValueError(f'{key} has an unsupported value')
                new_settings[key] = int(value)
        if "post_controls" in new_settings:
            from post_control import controls_from_settings
            patch = new_settings["post_controls"]
            if not isinstance(patch, dict):
                raise ValueError("post_controls must be an object")
            merged_controls = dict(self.settings.get("post_controls") or {})
            merged_controls.update(patch)
            new_settings["post_controls"] = controls_from_settings(
                {"post_controls": merged_controls}).as_dict()
        if "temporal_controls" in new_settings:
            from temporal_control import temporal_from_settings
            patch = new_settings["temporal_controls"]
            if not isinstance(patch, dict):
                raise ValueError("temporal_controls must be an object")
            merged_controls = dict(self.settings.get("temporal_controls") or {})
            merged_controls.update(patch)
            new_settings["temporal_controls"] = temporal_from_settings(
                {"temporal_controls": merged_controls}).as_dict()
        new_settings.pop("depth_convention", None)
        if "guidance_mode" in new_settings and int(new_settings["guidance_mode"]) not in (0, 1):
            raise ValueError("guidance_mode supports only 0 (off) or 1 (flow)")
        if "export_backend" in new_settings:
            requested_export_backend = str(new_settings["export_backend"]).lower()
            if requested_export_backend not in pipeline.EXPORT_BACKENDS:
                raise ValueError(
                    f"export_backend={requested_export_backend!r} is not a backend this "
                    f"service offers; the choices are {', '.join(pipeline.EXPORT_BACKENDS)}")
            new_settings["export_backend"] = requested_export_backend
        if "dlss_backend" in new_settings:
            # Any caller of /api/settings may write this key, so it is validated where
            # it arrives: a value that is not the GPU host is either a typo or a
            # leftover from a build that had two hosts, and both have to be answered
            # here, at the boundary, rather than stored and met at the first frame.
            new_settings["dlss_backend"] = self._validated_dlss_backend(
                new_settings["dlss_backend"])
        # The HDR pair is checked against the settings as this patch would leave them,
        # before anything is merged: /api/settings is a partial patch, so a payload can
        # arrive with one key good and the other unusable, and storing that paper white
        # would make the next job's refusal name a number this layer just accepted.
        # Read through the chain's own predicates rather than a second copy of its
        # vocabulary, so an opt-in accepted here is the opt-in the chain acts on.
        if "experimental_hdr_scrgb" in new_settings or "hdr_paper_white_scrgb" in new_settings:
            merged = dict(self.settings)
            merged.update(new_settings)
            _hdr_reasons = _hdr_request(merged)[1]
            if _hdr_reasons:
                raise ValueError("; ".join(_hdr_reasons))
        for key, valid, default in (('preset', (1, 2, 3), 1), ('style', (0, 1, 2, 3), 0)):
            if key in new_settings:
                try:
                    value = int(new_settings[key])
                    new_settings[key] = value if value in valid else default
                except (TypeError, ValueError):
                    new_settings[key] = default
        # Settings requests must not block behind image readback. Publish a new
        # immutable snapshot; the next GPU request applies it on the owner thread.
        with self._settings_lock:
            self.settings = dict(self.settings, **new_settings)
            self._preview_revision = getattr(self, '_preview_revision', 0) + 1
            self._logged_live_err = False
        return self.settings

    def _preview_live_kind(self):
        """Which GPU-host entry point serves the current preview configuration.

        An unguided frame runs on the resident zero-motion texture the host creates
        at open; a guided one runs on the host-memory motion entry, which takes the
        same float32 HxWx2 plane a CPU flow engine produces. The two are separate
        kinds because they are separate entry points with separate capability gates;
        sharing one kind would let a session opened for one be driven through the
        other.
        """
        if int(self.settings.get("guidance_mode", 0)) == 0:
            return "gpu_zero_motion"
        return "gpu_host_motion"

    def _gpu_preview_options_key(self):
        """The create-time options a GPU host session bakes, mirroring Live._options.

        The GPU host reads settings once at open and exposes no live update
        entry, so a preview reuse has to rebuild when any of them move.
        """
        s = self.settings
        return (int(s.get("preset", 1)), int(s.get("style", 0)),
                float(s.get("intensity", 1.0)), float(s.get("local_tone", 1.0)),
                float(s.get("local_struct", 1.0)), float(s.get("skin_struct", 1.0)),
                int(s.get("use_auto_mask", 1)), int(s.get("ui_correction", 0)),
                float(s.get("motion_scale_x", 1.0)), float(s.get("motion_scale_y", 1.0)))

    # Each GPU preview kind, the capability that says the loaded host can serve
    # it, and what the operator is told when it cannot. The absence of an entry
    # point means a stale native build, and the only honest answer to that is the
    # rebuild it needs: there is no second host left to draw the frame instead.
    _GPU_PREVIEW_KINDS = {
        "gpu_zero_motion": ("supports_zero_motion",
                            "dlss_gpu_host.dll exposes no zero-motion entry point "
                            "(ABI 2); rebuild native/dlss_gpu_host.dll to preview "
                            "guidance_mode=0 on the GPU host"),
        "gpu_host_motion": ("supports_host_motion",
                            "dlss_gpu_host.dll exposes no host-motion entry point; "
                            "rebuild native/dlss_gpu_host.dll to preview "
                            "guidance_mode=1 on the GPU host"),
    }

    def _import_gpu_host(self):
        """The GPU host wrapper, with the requirement named when it is not there.

        Every frame this service draws goes through this host, so an unbuilt
        native/dlss_gpu_host.dll breaks a preview nobody configured. ctypes answers
        that with a winerr about a module it cannot find, which does not tell an
        operator which module; and the honest remedy is a rebuild, never a fallback
        to another host - this build has no other host to fall back to.
        """
        try:
            import dlss_gpu_engine
        except (ImportError, OSError) as e:
            raise DLSSInferenceError(
                f"默认 DLSS 宿主 dlss_backend={DEFAULT_DLSS_BACKEND} 需要 "
                f"native/dlss_gpu_host.dll，当前无法加载：{e}。请从 native/ 重新构建该宿主；"
                "本版本已移除旧宿主 dlssnr_host.dll，不存在可回退的第二条 DLSS 路径。") from e
        return dlss_gpu_engine

    def _open_gpu_live(self, module, w, h, kind):
        """Open the GPU host session and confirm it can serve this preview kind.

        A host that does not expose the entry the kind runs on is refused before
        any frame is drawn. Handing it to the caller anyway would leave the operator
        staring at the source picture with a DLSS label on it.
        """
        supported, refusal = self._GPU_PREVIEW_KINDS[kind]
        live = module.Live(w, h, self.settings)
        if not getattr(live, supported)():
            try:
                live.close()
            except Exception:
                pass
            raise DLSSInferenceError(refusal)
        live._gpu_options_key = self._gpu_preview_options_key()
        return live

    def _get_or_create_live(self, w, h):
        """Ensure a live session exists for the current preview kind and dims."""
        if self.active_job_id is not None:
            return None
        # The fuse refuses new DLSS work for the whole process, and preview is DLSS work:
        # a session that survived a timed-out close is the polluted state the fuse was
        # set for, so neither reusing it nor opening over it is allowed here.
        self._ensure_native_healthy()

        dims = (w, h)
        if getattr(self, '_legacy_source_changed', False):
            if self._live_session:
                self._live_session.close()
            self._live_session = None
            self._legacy_source_changed = False
        kind = self._preview_live_kind()
        self._preview_live_error = None
        session = self._live_session
        if session is not None and self._live_session_dims == dims:
            # The GPU host bakes its options at open and exposes no live update
            # entry, so a session is reusable only while the request still describes
            # the one that was opened - same entry point, same baked options.
            if (self._live_session_kind == kind
                    and getattr(session, "_gpu_options_key", None)
                    == self._gpu_preview_options_key()):
                return session

        if session:
            try:
                _run_with_timeout(session.close, 15, "DLSS 预览会话关闭")
            except Exception as ce:
                # The same rule the queue applies to its own session: an exception from
                # close is survivable, a close that never came back means a native call
                # is still running on a thread this process no longer owns. Opening the
                # next session on top of it is exactly the pollution spread the fuse
                # exists to stop, so the fuse is set here too and the frame fails.
                diagnostic_logger.log("WARN", "Preview",
                                      f"预览旧会话关闭异常/超时，继续重建: {ce}")
                if isinstance(ce, NativeCallTimeout):
                    self._mark_native_degraded(f"预览旧会话关闭超时 ({ce})")
                    raise
            self._live_session = None
            self._live_session_dims = None
            self._live_session_kind = None

        try:
            module = self._import_gpu_host()
            live = _run_with_timeout(
                lambda: self._open_gpu_live(module, w, h, kind),
                120, f"GPU {kind} 预览会话初始化")
            self._live_session = live
            self._live_session_dims = dims
            self._live_session_kind = kind
            self._logged_live_err = False
            # Which host file and ABI answered is the part an operator cannot see in
            # the picture, so the line that says a session is ready names it.
            diagnostic_logger.log(
                "INFO", "Preview",
                f"预览 DLSS 会话已就绪: kind={kind} "
                f"host={getattr(live, 'host_dll', 'dlss_gpu_host.dll')}"
                f" abi={getattr(live, 'host_abi', '?')} @ {w}x{h}")
            return live
        except Exception as e:
            self._preview_live_error = str(e)
            if not getattr(self, "_logged_live_err", False):
                print(f"[DLSS5 Warning] {kind} session create failed: {e}")
                self._logged_live_err = True
            return None

    def _preview_apply_live(self, live, frame_idx, rgba, flow):
        """Run one preview frame through the cached session's GPU-host entry.

        The same predicate that decided whether a flow array is needed picks the
        entry point, so a session opened for the zero-motion texture is never driven
        through the host-memory one. A guided frame is handed the real guidance
        plane: the host narrows it to the bound motion texture on the card, and
        nothing here zeroes it.

        The GPU host raises on a failed frame, and a raised frame means the same
        thing to a preview as a null did: show the source with a failure banner, not
        a 500 that stops the stream. The banner carries the host's own error.
        """
        if live is None:
            return None
        try:
            kind = self._preview_live_kind()
            if kind == "gpu_zero_motion":
                return live.process_zero_motion(rgba, reset=(frame_idx == 0))
            return live.process_host_motion(rgba, flow, reset=(frame_idx == 0))
        except DLSSInferenceError as e:
            if getattr(live, "last_error", None) is None:
                live.last_error = str(e)
            return None

    def _preview_live_error_detail(self, live):
        """The reason a preview frame had no DLSS result, for the failure banner."""
        return (getattr(live, "last_error", None)
                or getattr(self, "_preview_live_error", None)
                or "GPU DLSS 宿主未返回结果")

    def _preview_resizing_backend(self, pw: int, ph: int, sw: int, sh: int):
        """One residual backend for the current preview geometry, reused across frames.

        The key carries the requested backend too, so switching between the reference
        and the native path rebuilds exactly once; playing a movie at a fixed scale must
        not recreate plans or scratch.
        """
        requested = str(self.settings.get("residual_backend") or "auto")
        key = (pw, ph, sw, sh, requested)
        if self._preview_backend_key != key:
            self._release_preview_backend()
            self._preview_backend = pipeline.create_residual_backend(requested, pw, ph, sw, sh)
            self._preview_backend_key = key
            self._preview_buffers = {
                "working": np.empty((sh, sw, 3), dtype=np.uint8),
                "recon": np.empty((ph, pw, 3), dtype=np.uint8),
            }
        return self._preview_backend, self._preview_buffers

    def _release_preview_backend(self):
        if self._preview_backend is not None:
            try:
                self._preview_backend.close()
            except Exception as e:
                diagnostic_logger.log("WARN", "Preview", "预览残差后端关闭失败", str(e))
        self._preview_backend = None
        self._preview_backend_key = None
        self._preview_buffers = {}

    def _close_gpu_preview(self):
        engine = getattr(self, '_gpu_preview', None)
        if engine is not None:
            engine.close()

    def preview_diagnostics(self):
        return dict(getattr(self, '_preview_diagnostics', {}) or {})

    def _render_gpu_preview(self, frame_idx, view, split_x, zoom, pan, preview_mode, display_edge=0):
        if not self._live_lock.acquire(timeout=2):
            return None, 'GPU is busy; preview waiting, no layer reduction'
        try:
            self._ensure_native_healthy()
            if self._live_session:
                self._live_session.close()
                self._live_session = None
                self._live_session_dims = self._live_session_kind = None
            if getattr(self, '_gpu_preview', None) is None:
                from gpu_preview_engine import GpuPreview
                self._gpu_preview = GpuPreview()
            original, processed, diagnostics = self._gpu_preview.render(
                self.video_path, self.settings, frame_idx,
                need_source=view in ('orig', 'dual', 'split'), reuse_display=True,
                display_edge=(display_edge // 2 if view == 'dual' else display_edge) if zoom <= 1.01 else 0,
                **({'flow_view': True} if view == 'flow' else {}))
            self._preview_diagnostics = diagnostics
        except Exception as error:
            self._preview_diagnostics = dict(status='failed', message=str(error), frame=frame_idx)
            diagnostic_logger.log('ERROR', 'Preview', 'GPU preview failed', str(error))
            return None, str(error)
        finally:
            self._live_lock.release()
        display_start = time.perf_counter()
        if view == 'orig':
            out = original
        elif view == 'dual':
            if original.shape != processed.shape:
                original = cv2.resize(original, (processed.shape[1], processed.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
            out = np.hstack([original, processed])
        elif view == 'split':
            if original.shape != processed.shape:
                original = cv2.resize(original, (processed.shape[1], processed.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
            out = original.copy()
            column = round(np.clip(split_x, 0, 1) * out.shape[1])
            out[:, column:] = processed[:, column:]
            # Separator is a UI overlay; keep both image regions unobstructed.
        else:
            out = processed
        height, width = out.shape[:2]
        if zoom > 1.01:
            crop_w, crop_h = max(1, int(width / zoom)), max(1, int(height / zoom))
            x = max(0, min(width-crop_w, int(pan[0]*width-crop_w/2)))
            y = max(0, min(height-crop_h, int(pan[1]*height-crop_h/2)))
            out = out[y:y+crop_h, x:x+crop_w]
        elif zoom < .99:
            out = cv2.resize(out, (max(16, round(width*zoom)), max(16, round(height*zoom))),
                             interpolation=cv2.INTER_AREA)
        if display_edge and max(out.shape[:2]) > display_edge:
            scale = display_edge / max(out.shape[:2])
            out = cv2.resize(out, (max(1, round(out.shape[1]*scale)),
                max(1, round(out.shape[0]*scale))), interpolation=cv2.INTER_AREA)
        diagnostics['compose_ms'] = (time.perf_counter()-display_start)*1000
        encode_start = time.perf_counter()
        ok, encoded = cv2.imencode('.jpg', out,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80 if preview_mode else 90])
        diagnostics.update(jpeg_encode_ms=(time.perf_counter()-encode_start)*1000,
            jpeg_bytes=encoded.nbytes if ok else 0, transmitted_resolution=[out.shape[1], out.shape[0]],
            display_size_policy='viewport' if display_edge else 'full_resolution')
        self._preview_diagnostics = diagnostics
        return (encoded.tobytes(), 'ok') if ok else (None, 'Preview encoding failed')

    def render_preview(self, frame_idx: int, view: str = "dlss", split_x: float = 0.5,
                       zoom: float = 1.0, pan: tuple = (0.5, 0.5), preview_mode: bool = False, display_edge: int = 0):
        """Render preview frame according to active view and settings."""
        if not self.video_path or self.video_info is None:
            return None, "No video loaded"
        if view not in ("orig", "flow", "dlss", "split", "dual"):
            return None, f"Unsupported preview view: {view}"

        # Native device decode and the very same single/multiscale frame core as
        # export. CPU-flow single-layer preview retains its explicit old path.
        native_guidance = (int(self.settings.get('guidance_mode', 0)) == 0 or
                           self.settings.get('flow_engine', 'nvof') in ('nvof', 'fidelityfx', 'raft', 'raft_large'))
        wants_multiscale = self.settings.get('reconstruction_mode', 'single') != 'single'
        wide_input = bool(self.video_info.get('is_hdr') or
                          self.video_info.get('bit_depth', 8) > 8)
        device_flow_view = (view == 'flow' and self.settings.get('flow_engine', 'nvof')
                            in ('nvof', 'fidelityfx', 'raft', 'raft_large'))
        # Still images have no NVDEC video stream. The existing image decoder and
        # single NR host-upload path can process them without inventing a video.
        still_image = os.path.splitext(self.video_path)[1].lower() in (
            '.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')
        image_nr = (still_image and not wide_input and not wants_multiscale and
                    self.settings.get('render_pipeline', ['nr']) == ['nr'])
        if (not image_nr and self.active_job_id is None and
                (device_flow_view or (view != 'flow' and
                 (native_guidance or wants_multiscale or wide_input)))):
            return self._render_gpu_preview(frame_idx, view, split_x, zoom, pan, preview_mode, display_edge)

        cur_bgr = None
        prev_bgr = None

        gm = int(self.settings.get("guidance_mode", 0))
        need_flow = (gm == 1) or (view == "flow")

        # Fetch frames in chronological order (frame_idx - 1 first, then frame_idx)
        # to ensure seamless forward sequential decoding without GOP backward seeks
        if need_flow and frame_idx > 0:
            prev_bgr = self.get_frame(frame_idx - 1)

        cur_bgr = self.get_frame(frame_idx)
        if cur_bgr is None:
            return None, f"Frame {frame_idx} read failed"
        if prev_bgr is None:
            prev_bgr = cur_bgr

        h, w = cur_bgr.shape[:2]

        # Keep working dimensions constant for the video so the D3D12 / DLSS Live session
        # is never torn down and rebuilt between play and pause/seek.
        pw, ph = w, h
        cur_work = cur_bgr
        prev_work = prev_bgr

        if view == "orig":
            out_bgr = cur_work
        elif self.active_job_id is not None:
            # DLSS GPU engine is currently occupied by render queue export.
            # Serve source video frame for UI preview to avoid D3D12 device collision.
            if view == "split":
                out_bgr = cur_work.copy()
                sx = int(max(0.0, min(1.0, split_x)) * pw)
                line_w = max(1, int(pw / 1000))
                out_bgr[:, max(0, sx - line_w):min(pw, sx + line_w)] = [0, 215, 255]
            elif view == "dual":
                out_bgr = np.hstack([cur_work, cur_work])
            else:
                out_bgr = cur_work.copy()
            # Non-silent indicator: Explicitly inform user that GPU is dedicated to export
            b_h = max(24, int(ph * 0.035))
            cv2.rectangle(out_bgr, (0, 0), (pw, b_h), (25, 25, 25), -1)
            cv2.putText(out_bgr, "[GPU 独占渲染导出中 | 预览窗口临时呈现源片以防 D3D12 冲突]",
                        (15, int(b_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, b_h / 45.0),
                        (0, 215, 255), 1, cv2.LINE_AA)
        elif view == "flow":
            engine = get_optical_flow_engine(self.settings["flow_engine"], self.settings["flow_quality"],
                **_raft_flow_options(self.settings))
            flow = engine.compute_flow(prev_work, cur_work, reset_temporal_hints=True)
            out_bgr = pipeline.colorize_flow(flow)
        else:
            # DLSS or Split view. Working size, working color filter, residual formula,
            # residual backend and residual strength all come from the same objects the
            # render queue uses; only the decoder differs here, which is a transport
            # difference, not an image-semantics one.
            scale = float(self.settings.get("working_scale", 1.0))
            use_pyramid = bool(self.settings.get("residual_pyramid", True)) and scale < 1.0

            sw, sh = pipeline.resolve_working_dims(pw, ph, scale)

            # An unguided frame on the D3D12 backend is served by the GPU host's
            # resident zero-motion texture, so this branch must allocate no flow
            # array at all and must not build a flow provider. A guided frame on
            # that backend keeps its plane: it is handed over as the CPU float32
            # HxWx2 array the guidance services already produce, which is the
            # shape the host-motion entry reads, not a device descriptor.
            zero_motion_preview = self._preview_live_kind() == "gpu_zero_motion"

            flow_provider = None
            if need_flow:
                guidance_service = get_frame_guidance_service(
                    self.settings.get("flow_engine", "nvof"),
                    self.settings.get("flow_quality", "1F"),
                    **_raft_flow_options(self.settings),
                )
                flow_provider = pipeline.GuidanceServiceFlowProvider(guidance_service, guidance_mode=gm)

            # 1. Optical flow, only when guidance asks for it
            flow = None
            if not zero_motion_preview:
                flow_dims = (sh, sw) if use_pyramid else (ph, pw)
                if flow_provider is not None:
                    flow = flow_provider.get_flow(frame_idx, cur_work, prev_work,
                                                  flow_dims[1], flow_dims[0],
                                                  is_discontinuous=(frame_idx == 0))
                else:
                    flow = np.zeros((flow_dims[0], flow_dims[1], 2), dtype=np.float32)

            # 2. Residual Pyramid vs Full Native
            # Bug3 fix: timed lock for preview so export never deadlocks preview forever.
            # Preview waits at most 2s; on timeout serves source frame with indicator.
            _preview_locked = self._live_lock.acquire(timeout=2)
            if not _preview_locked:
                dlss_bgr = cur_work.copy()
                b_h = max(26, int(ph * 0.04))
                cv2.rectangle(dlss_bgr, (0, 0), (pw, b_h), (25, 25, 25), -1)
                cv2.putText(dlss_bgr, "[GPU 导出占用中 | 预览降级为源片]",
                            (15, int(b_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, b_h / 45.0),
                            (0, 215, 255), 1, cv2.LINE_AA)
            else:
              try:
                if use_pyramid:
                    backend, buffers = self._preview_resizing_backend(pw, ph, sw, sh)
                    down_cur = backend.prepare_working_color(cur_work, buffers["working"])

                    rgb = cv2.cvtColor(down_cur, cv2.COLOR_BGR2RGB)
                    rgba = np.dstack([rgb, np.full((sh, sw), 255, dtype=np.uint8)])

                    live = self._get_or_create_live(sw, sh)
                    res_rgba = self._preview_apply_live(live, frame_idx, rgba, flow)

                    if res_rgba is not None:
                        res_bgr_down = cv2.cvtColor(res_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                        strength = float(self.settings.get("residual_strength", 1.0))
                        dlss_bgr = backend.reconstruct(cur_work, down_cur, res_bgr_down,
                                                       strength, buffers["recon"])
                    else:
                        err_detail = self._preview_live_error_detail(live)
                        diagnostic_logger.log("ERROR", "Preview", f"第 {frame_idx} 帧金字塔 DLSS 增强失败", err_detail)
                        dlss_bgr = cur_work.copy()
                        b_h = max(26, int(ph * 0.04))
                        cv2.rectangle(dlss_bgr, (0, 0), (pw, b_h), (0, 0, 180), -1)
                        cv2.putText(dlss_bgr, f"[DLSS 增强失败: {err_detail[:60]}]",
                                    (15, int(b_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, b_h / 45.0),
                                    (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    rgb = cv2.cvtColor(cur_work, cv2.COLOR_BGR2RGB)
                    rgba = np.dstack([rgb, np.full((ph, pw), 255, dtype=np.uint8)])

                    live = self._get_or_create_live(pw, ph)
                    res_rgba = self._preview_apply_live(live, frame_idx, rgba, flow)

                    if res_rgba is not None:
                        dlss_bgr = cv2.cvtColor(res_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                    else:
                        err_detail = self._preview_live_error_detail(live)
                        diagnostic_logger.log("ERROR", "Preview", f"第 {frame_idx} 帧原生 DLSS 增强失败", err_detail)
                        dlss_bgr = cur_work.copy()
                        b_h = max(26, int(ph * 0.04))
                        cv2.rectangle(dlss_bgr, (0, 0), (pw, b_h), (0, 0, 180), -1)
                        cv2.putText(dlss_bgr, f"[DLSS 增强失败: {err_detail[:60]}]",
                                    (15, int(b_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, b_h / 45.0),
                                    (255, 255, 255), 1, cv2.LINE_AA)
              finally:
                try:
                    self._live_lock.release()
                except Exception:
                    pass

            if view == "split":
                out_bgr = cur_work.copy()
                sx = int(max(0.0, min(1.0, split_x)) * pw)
                out_bgr[:, sx:] = dlss_bgr[:, sx:]
                # Yellow separator line
                line_w = max(1, int(pw / 1000))
                out_bgr[:, max(0, sx - line_w):min(pw, sx + line_w)] = [0, 215, 255]
            elif view == "dual":
                out_bgr = np.hstack([cur_work, dlss_bgr])
            else:
                out_bgr = dlss_bgr

        # Apply digital zoom / pan if requested
        cw_full, ch_full = out_bgr.shape[1], out_bgr.shape[0]
        if zoom > 1.01:
            cw, ch = int(cw_full / zoom), int(ch_full / zoom)
            px, py = pan
            x1 = max(0, min(cw_full - cw, int(px * cw_full - cw / 2)))
            y1 = max(0, min(ch_full - ch, int(py * ch_full - ch / 2)))
            out_bgr = out_bgr[y1:y1 + ch, x1:x1 + cw]
        elif zoom < 0.99:
            target_w = max(16, int(round(cw_full * zoom)))
            target_h = max(16, int(round(ch_full * zoom)))
            out_bgr = cv2.resize(out_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Encode to JPEG for fast web transmission (quality 80 for playback, 90 for still pause/seek)
        quality = 80 if preview_mode else 90
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc = cv2.imencode(".jpg", out_bgr, encode_param)
        if not success:
            return None, "Encoding preview frame failed"
        return enc.tobytes(), "ok"

    # ---------------- Queue & Batch Export ----------------
    def add_job(self, job_data: dict):
        job_settings = dict(job_data.get("settings") or self.settings)
        # A queued job must keep its own layer vector even if the settings UI
        # changes the saved arrays before the worker starts.
        for key in ("layer_scales", "layer_weights"):
            if isinstance(job_settings.get(key), (list, tuple)):
                job_settings[key] = list(job_settings[key])
        job_settings.pop("depth_convention", None)
        if int(job_settings.get("guidance_mode", 0)) not in (0, 1):
            raise ValueError("guidance_mode supports only 0 (off) or 1 (flow)")
        requested_export_backend = str(job_settings.get("export_backend") or "auto").lower()
        if requested_export_backend not in pipeline.EXPORT_BACKENDS:
            raise ValueError(
                f"export_backend={requested_export_backend!r} is not a backend this "
                f"service offers; the choices are {', '.join(pipeline.EXPORT_BACKENDS)}")
        job_settings["export_backend"] = requested_export_backend
        if (job_settings.get("reconstruction_mode", "single") in ("dual", "triple")
                and requested_export_backend != "gpu_transfer"):
            raise ValueError("Dual / Triple 独立多尺度层仅支持 export_backend=gpu_transfer")
        # The experimental HDR chain is a mode of one backend, and it is opted into by
        # a boolean and never guessed: a job that names a half-read flag is refused at
        # the queue boundary rather than exported by the SDR chain it did not ask for.
        experimental_hdr, hdr_reasons = _hdr_request(job_settings)
        for reason in hdr_reasons:
            raise ValueError(reason)
        if experimental_hdr and requested_export_backend != "gpu_transfer":
            raise ValueError(
                "experimental_hdr_scrgb 只在 export_backend=gpu_transfer 上运行：HDR 链路"
                f"全程留在显存里，而 {requested_export_backend!r} 的管道要把帧交给 FFmpeg "
                "的 8-bit SDR 通路。请显式设 export_backend=gpu_transfer，"
                "或关掉 experimental_hdr_scrgb —— 本版本不会用另一条链路悄悄导出 HDR 源。")
        # The host is part of what the job asked for, and a client that named none
        # now gets the service's own default rather than the host its absence used
        # to imply. Written back so the queue row shows the host the job runs on.
        job_settings["dlss_backend"] = self._job_dlss_backend(job_settings)
        raw_out = job_data.get("output_path", "")
        out_path = os.path.normpath(raw_out) if raw_out else ""
        input_path = job_data.get("input_path", self.video_path)
        # Refused at the boundary rather than at the first frame: a destination that
        # resolves to the source cannot ever render, and the queue would otherwise
        # hold a job whose failure notice arrives after a GPU session was opened.
        # An empty destination is not a contradiction yet - it is derived from the
        # source at run time and checked again there.
        pipeline.assert_output_not_input(input_path, out_path)
        tot_frames = self.video_info.get("frames", 0) if self.video_info else 0
        job = {
            "id": str(uuid.uuid4())[:8],
            "name": job_data.get("name", f"Job-{len(self.jobs) + 1}"),
            "input_path": input_path,
            "output_path": out_path,
            "codec": job_data.get("codec", "av1_nvenc"),
            "container": job_data.get("container", "mkv"),
            "cq": int(job_data.get("cq", 24)),
            "preset": job_data.get("preset", "p7"),
            "with_audio": bool(job_data.get("with_audio", True)),
            "settings": job_settings,
            "transcode_settings": job_data.get("transcode_settings", None),
            "status": "pending",  # pending, running, completed, skipped, cancelled, failed
            "progress": 0.0,
            "fps": 0.0,
            "eta": 0,
            "current_frame": 0,
            "total_frames": int(job_data.get("total_frames", tot_frames)),
            # Set on a job that left its destination untouched because it already
            # existed; the status says skipped and this says which rule said so.
            "skipped_reason": None,
            "error": None,
            "error_traceback": None,
            # Which DLSS host ran or refused this job, and that host's own last error,
            # both filled in by _execute_job. The retired dlssnr_host.dll published a
            # log tail here, so a failed GPU job reported the old host's log as its own.
            "dlss_host": None,
            "dlss_host_error": None,
        }
        with self._queue_lock:
            self.jobs.append(job)
        diagnostic_logger.log("INFO", "Queue", f"作业 [{job['id']}] 已加入渲染队列: {job['name']} -> {out_path}")
        return job

    def remove_job(self, job_id: str):
        with self._queue_lock:
            job = next((j for j in self.jobs if j['id'] == job_id), None)
            if job and (job['status'] == 'running' or self.active_job_id == job_id):
                raise ValueError('任务正在导出，请先停止，结束后即可删除')
            self.jobs = [j for j in self.jobs if j["id"] != job_id]
        diagnostic_logger.log("INFO", "Queue", f"作业 [{job_id}] 已从队列移除")
        return True

    def queue_is_running(self) -> bool:
        """Whether a worker thread still owns the queue, and with it the GPU session.

        The UI gates the retry action on this: a reset that lands while a round is in
        flight would hand the running thread a job it has already passed over, which
        is the silent re-run the explicit action exists to replace.
        """
        with self._queue_lock:
            return bool(self._queue_thread and self._queue_thread.is_alive())

    def retry_job(self, job_id: str):
        """Put one terminal job back to pending, on the operator's word alone.

        The queue runs pending jobs and nothing else, so a failed job stays failed
        until this is called on it: adding a new job, or pressing render, must not
        quietly re-run somebody's earlier failure - it would burn a GPU session and
        overwrite nothing, and the row would say nothing about why it moved.

        The whole decision is made under the queue lock, which is the same lock
        ``start_queue`` takes before it spawns a worker, so a retry can neither race a
        start nor land twice: the second caller sees the job already pending and
        returns it unchanged instead of queueing a second run of one job.
        """
        with self._queue_lock:
            job = next((j for j in self.jobs if j["id"] == job_id), None)
            if job is None:
                raise KeyError(job_id)
            if job["status"] == "running" or self.active_job_id == job_id:
                raise ValueError(f"作业 [{job_id}] 正在渲染中，无法重置；请先终止当前任务")
            if self._queue_thread and self._queue_thread.is_alive():
                raise ValueError(
                    "渲染队列仍在运行，本轮结束后才能重试；如要立即停止，请终止当前任务")
            if job["status"] == "pending":
                return job
            if job["status"] not in ("failed", "cancelled", "skipped"):
                raise ValueError(
                    f"作业 [{job_id}] 状态为 {job['status']}，只有失败/已取消/已跳过的任务可以重试")
            retried_from = job["status"]
            job["status"] = "pending"
            job["progress"] = 0.0
            job["fps"] = 0.0
            job["eta"] = 0
            job["current_frame"] = 0
            job["error"] = None
            job["error_traceback"] = None
            job["skipped_reason"] = None
            # These three describe the attempt that just ended, and a reset row has to
            # describe nothing until the next attempt fills them in again.
            job["dlss_host"] = None
            job["dlss_host_error"] = None
            job["residual_backend_diagnostics"] = {}
        diagnostic_logger.log(
            "INFO", "Queue",
            f"作业 [{job_id}] 已按操作者请求从 {retried_from} 重置为待渲染：{job['name']}")
        return job

    def get_jobs(self):
        with self._queue_lock:
            return list(self.jobs)

    def cancel_active_job(self):
        # Cancel must take effect even when the worker is blocked in IO or a
        # native call: set the flag AND kill owned FFmpeg subprocesses. Pipe
        # readers observe the death promptly via _read_exactly/proc.poll().
        self._cancel_requested = True
        self._kill_active_procs()
        diagnostic_logger.log("WARN", "Queue", "收到主动取消当前渲染作业请求（已置位并杀死编码子进程）")

    def start_queue(self, progress_callback=None):
        with self._queue_lock:
            if self._queue_thread and self._queue_thread.is_alive():
                return False
            self._cancel_requested = False
            self._queue_thread = threading.Thread(
                target=self._run_queue, args=(progress_callback,), daemon=True, name="RenderQueue"
            )
            self._queue_thread.start()
            diagnostic_logger.log("INFO", "Queue", "后台渲染队列线程已启动")
            return True

    def _run_queue(self, progress_callback):
        # UI delivery must never abort the worker or strand a running row. This
        # wrapper also covers callbacks inside the encoder/export implementation.
        original_callback = progress_callback
        def notify(data):
            if original_callback:
                try:
                    original_callback(data)
                except Exception as exc:
                    diagnostic_logger.log('WARN', 'Queue', f'队列通知失败：{exc}')
        progress_callback = notify
        # P1-09: finish actions bind ONLY to jobs that completed successfully
        # in THIS run. Failures/cancels veto destructive actions, and a job that
        # skipped writes nothing at all, so it is not a finished piece of work either.
        completed_this_run = []
        run_had_failure = False
        while True:
            # Select and claim under the same lock as removal. Iterating the old
            # list used to render removed jobs and lose jobs added after removal.
            with self._queue_lock:
                if self._cancel_requested:
                    break
                job = next((j for j in self.jobs if j['status'] == 'pending'), None)
                if job is None:
                    break
                job['status'] = 'running'
                self.active_job_id = job['id']
            diagnostic_logger.log("INFO", "Queue", f"开始处理作业 [{job['id']}] {job['name']}")
            if progress_callback: progress_callback(job)

            try:
                outcome = self._execute_job(job, progress_callback)
                if self._cancel_requested:
                    job["status"] = "cancelled"
                    run_had_failure = True
                    diagnostic_logger.log("WARN", "Queue", f"作业 [{job['id']}] 已被用户中止")
                elif outcome == JOB_SKIPPED:
                    # The destination was left as it was found. Reported as skipped, not
                    # as an export that succeeded, because the difference is what a
                    # finish action and a progress bar are judged against.
                    job["status"] = "skipped"
                    job["skipped_reason"] = "output_exists"
                    diagnostic_logger.log(
                        "WARN", "Queue",
                        f"作业 [{job['id']}] 已跳过：目标文件保持原样，未渲染、未覆盖: {job['output_path']}")
                else:
                    job["status"] = "completed"
                    job["progress"] = 100.0
                    completed_this_run.append(job)
                    diagnostic_logger.log("INFO", "Queue", f"作业 [{job['id']}] 导出成功，物理校验通过: {job['output_path']}")
            except Exception as e:
                # User cancellation (flag or JobCancelled from IO paths) must
                # surface as 'cancelled', never 'failed'.
                if self._cancel_requested or "用户已取消任务" in str(e):
                    job["status"] = "cancelled"
                    job["error"] = None
                    run_had_failure = True
                    diagnostic_logger.log("WARN", "Queue", f"作业 [{job['id']}] 已被用户中止")
                else:
                    job["status"] = "failed"
                    job["error"] = str(e)
                    job["error_traceback"] = traceback.format_exc()
                    # Nothing is attached here on purpose. This used to append the
                    # tail of dlssnr_run.log - the retired dlssnr_host.dll's own log
                    # file - to every failed job, which told an operator that the old
                    # host had said something about a GPU-host run it was never part
                    # of, and did so whatever had actually failed: the log was read
                    # whether a DLSS session had been opened at all or not, and a
                    # stale one from an earlier build stayed on disk. A failed job's
                    # GPU-host diagnostics are recorded by _execute_job, from the
                    # session it opened, and only when there was one.
                    run_had_failure = True
                    diagnostic_logger.log("ERROR", "Queue", f"作业 [{job['id']}] 致命失败: {e}", traceback.format_exc())
                    traceback.print_exc()
            finally:
                self.active_job_id = None
                self._cancel_requested = False
                if progress_callback:
                    progress_callback(job)

        self.active_job_id = None
        self._cancel_requested = False
        # The final job_update says a row is terminal; only this says the round is over
        # and a reset is safe again. The UI cannot infer it from the absence of a running
        # row - between that update and this thread exiting, nothing is running and the
        # queue still owns the GPU session, which is exactly when a retry would be refused.
        # Sent ahead of the finish action so it lands even when the next line sleeps.
        if progress_callback:
            progress_callback({
                "type": "queue_finished",
                "jobs": self.get_jobs(),
                "active_id": None,
                "queue_running": False,
                "completed": len(completed_this_run),
                "run_had_failure": run_had_failure,
            })
        self._handle_finish_action(completed_this_run, run_had_failure)

    def _handle_finish_action(self, completed_jobs=None, run_had_failure=False):
        """Bug5 fix: wire finish_action (none/sleep/shutdown).
        P1-09 hardening: only jobs that COMPLETED in this queue run count, and
        any failure/cancel in the run vetoes destructive actions entirely.
        For safety, real OS power actions only run when ALLOW_POWER_ACTIONS=1;
        otherwise we log the request (still counts as wired, never a dead control).
        E2E tests set E2E_TEST=1 so they never trigger sleep/shutdown."""
        try:
            if run_had_failure:
                diagnostic_logger.log("INFO", "Queue", "本轮存在失败/取消的任务，已禁止睡眠/关机完成动作")
                return
            actions = []
            for j in completed_jobs or []:
                ts = (j.get("transcode_settings") or {})
                a = str(ts.get("finish_action", "none"))
                if a not in ("none", "", None):
                    actions.append(a)
            if not actions:
                return
            action = actions[-1]
            if os.environ.get("E2E_TEST") == "1":
                diagnostic_logger.log("INFO", "Queue", f"完成动作 {action} 在测试模式下已抑制（仅记录不执行）")
                return
            diagnostic_logger.log("INFO", "Queue", f"执行完成动作: {action}")
            if action == "sleep" and os.environ.get("ALLOW_POWER_ACTIONS") == "1":
                try:
                    import subprocess as _sp
                    _sp.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
                except Exception as e:
                    diagnostic_logger.log("ERROR", "Queue", f"睡眠动作执行失败: {e}")
            elif action == "shutdown" and os.environ.get("ALLOW_POWER_ACTIONS") == "1":
                try:
                    import subprocess as _sp
                    _sp.Popen(["shutdown", "/s", "/t", "30"])
                except Exception as e:
                    diagnostic_logger.log("ERROR", "Queue", f"关机动作执行失败: {e}")
        except Exception:
            pass

    def _create_export_live(self, width, height, settings, *, wide=False):
        """Translate precision to each session factory's actual interface."""
        import native_effects
        if native_effects.mode(settings) != 'nr':
            return native_effects.make_live(width, height, settings, wide=wide)
        live_type = self._import_gpu_host().Live
        return live_type(width, height, settings, **({'color_format': 'rgba16f'} if wide else {}))

    def _execute_job(self, job, progress_callback):
        v_path = job["input_path"]
        if not os.path.exists(v_path):
            raise FileNotFoundError(f"Input video not found: {v_path}")

        # Refused before the probe below opens a decoder and before the file rules and
        # the DLSS session: a destination that resolves to the source would be truncated
        # while still being read, whatever the overwrite settings say. A job with no
        # destination yet is not a contradiction - it is derived from the source below
        # and checked again at the export entry points that receive it.
        if job["output_path"]:
            pipeline.assert_output_not_input(v_path, job["output_path"])

        total_frames, v_fps, w, h = pipeline.video_info(v_path)
        job["total_frames"] = total_frames

        # Fail fast on invalid encoder (Bug3/11.3 fix): previously an unknown
        # codec ran 600 frames of DLSS before FFmpeg failed, exceeding the
        # 45s test timeout and looking like a deadlock. Validate now.
        ts_job_early = job.get("transcode_settings") or {}
        source_precision = pipeline.probe_pixel_info(v_path)
        import gpu_hdr_transfer_export as high_precision
        sdr10_job = (str((job.get("settings") or {}).get("export_backend")) == "gpu_transfer"
                     and not high_precision.hdr_requested(job.get("settings"))
                     and high_precision.sdr10_source(source_precision))
        codec_check = str(ts_job_early.get("codec", job.get("codec", "av1_nvenc")))
        allowed_codecs = {"av1_nvenc", "hevc_nvenc", "h264_nvenc", "libx264", "libx265", "mpeg4"}
        if codec_check not in allowed_codecs:
            raise RuntimeError(f"FFmpeg Unknown encoder / 不可用编码器参数: {codec_check}")

        # P1-01 honesty: the neural pipe is 8-bit SDR, so HDR/10-bit+ output
        # requests can only re-tag 8-bit data. Warn loudly instead of silently
        # downgrading (frontend additionally disables these options).
        _pix = str(ts_job_early.get("pixel_format", "auto")).lower()
        _col = str(ts_job_early.get("color_preset", "preserve")).lower()
        if not sdr10_job and not high_precision.hdr_requested(job.get("settings")) and (re.search(r"p0\d+|p2\d+|444p1[06]|444p16|rgb(10|12|16)", _pix)
                or _col in ("bt2020_sdr", "bt2020_pq", "bt2020_hlg")):
            diagnostic_logger.log(
                "WARN", "Queue",
                f"作业 [{job['id']}] 请求 {ts_job_early.get('pixel_format')}/{ts_job_early.get('color_preset')}，"
                f"但神经管线为 8-bit SDR：输出仅为 8-bit 数据重封装/打标签，HDR 精度无法恢复")

        # Destination path
        out_path = job["output_path"]
        if not out_path:
            base_dir = os.path.dirname(v_path)
            stem = os.path.splitext(os.path.basename(v_path))[0]
            out_path = os.path.join(base_dir, f"{stem}_dlss.{job['container']}")
            job["output_path"] = out_path

        # Bug6 fix: skip_existing / overwrite_output protection.
        # If target exists and user asked not to overwrite, preserve it and
        # skip FFmpeg entirely (FFmpeg -y would truncate on startup).
        ts_job = job.get("transcode_settings") or {}
        skip_existing = bool(ts_job.get("skip_existing", False))
        overwrite_output = ts_job.get("overwrite_output", True)
        if isinstance(overwrite_output, str):
            overwrite_output = overwrite_output.lower() not in ("false", "0", "no", "off")
        if (skip_existing or not overwrite_output) and out_path and os.path.exists(out_path):
            # The truthful record: no frame was decoded, nothing was written, the file
            # that is there is the one that was there before. So the counters stay where
            # a job that has not rendered anything belongs, and _run_queue reports the
            # outcome as skipped rather than as an export that succeeded.
            rule = "skip_existing" if skip_existing else "overwrite_output=false"
            diagnostic_logger.log("INFO", "Queue", f"作业 [{job['id']}] 跳过渲染：目标已存在且 {rule}，保留原文件: {out_path}")
            return JOB_SKIPPED

        # Setup the GPU-host session with a timed lock. One host session is alive at a
        # time because dlss_gpu_host_open refuses a second while one is active, and the
        # options are baked at open with no live-update entry, so a queued job closes
        # whatever session was cached (its own below, the preview's at the end of the
        # job) and opens one sized and configured for this job. The reuse that does
        # exist is per-frame: preview keeps the session it opened while the kind, the
        # extents and the baked options still match.
        job_settings = job["settings"]
        import native_effects
        native_effects.validate(job_settings, dict(source_precision, width=w, height=h, r_frame_rate=str(v_fps)))
        # Re-read off the job rather than off the service: a job queued before the
        # operator moved the setting runs with the host it was queued on. A job that
        # names the retired host or any other unknown value fails here, at the job
        # boundary, before a session is opened for it.
        job_settings["dlss_backend"] = self._job_dlss_backend(job_settings)
        # Set when this job asks the GPU host to answer an unguided frame. It is the
        # flag that decides which session-reuse kind the opened live is recorded
        # under, and whether the ABI-3 zero-motion entry has to be probed.
        gpu_guidance_off = False
        # Checked here rather than in the worker thread so a typo in a job's
        # settings fails the job before a session has been opened for it. Which
        # export chain a job asked for is part of what its guidance mode can be.
        requested_export_backend = str(job_settings.get("export_backend") or "auto").lower()
        if requested_export_backend not in pipeline.EXPORT_BACKENDS:
            raise ValueError(
                f"export_backend={requested_export_backend!r} is not a backend this "
                f"service offers; the choices are {', '.join(pipeline.EXPORT_BACKENDS)}")
        guidance_mode = int(job_settings.get("guidance_mode", 0))
        if guidance_mode == 0:
            # The GPU host answers an unguided frame from the zero motion texture it
            # bound at open, and both export chains drive that entry: the GPU-transfer
            # chain calls its device-resident form, the FFmpeg pipe calls
            # process_zero_motion across the CPU boundary it already has. Nothing here
            # asks for flow, so no flow engine is opened and no zero plane is uploaded
            # on either chain. The host has to be ABI 3, which is checked against the
            # session once it is open.
            gpu_guidance_off = True
        elif guidance_mode != 1:
            raise ValueError(
                "DLSS GPU 宿主的引导模式只能是 0（关闭）或 1（仅光流），"
                f"当前 guidance_mode={guidance_mode}")
        else:
            # Only the device chain needs the flow on the card. A pipe job on the
            # same GPU host hands the host's own host-motion entry the CPU float32x2
            # plane DIS and RAFT produce - the pairing the desktop window's export
            # already runs. The one setting that does break the device chain is named
            # apart, because the queue row is the only place the operator reads the
            # fix.
            if (requested_export_backend == "gpu_transfer"
                    and job_settings.get("flow_engine", "nvof") not in ('nvof', 'fidelityfx', 'raft', 'raft_large')):
                raise ValueError(
                    "GPU Transfer 需要设备光流描述符，请选择 NVIDIA 光流、AMD FidelityFX "
                    "或 RAFT CUDA，"
                    f"当前 flow_engine={job_settings.get('flow_engine')}。"
                    "若要用 DIS 的 CPU 运动平面，请把 export_backend 设为 auto 或 "
                    "legacy：该链路的 GPU 宿主有宿主内存运动入口可以直接吃它。")
        if requested_export_backend == "gpu_transfer":
            job_settings["export_backend"] = requested_export_backend
        # The HDR flag is read here as well as at the queue boundary, because a caller
        # that reaches _execute_job directly never passed through it. Like the queue's
        # reading, it is a refusal rather than a route: an FP16 session opened for a
        # job whose chain then refused it would be an 8-bit draw reported as HDR.
        experimental_hdr, hdr_reasons = _hdr_request(job_settings)
        if hdr_reasons:
            raise ValueError("; ".join(hdr_reasons))
        if experimental_hdr and requested_export_backend != "gpu_transfer":
            raise ValueError(
                "experimental_hdr_scrgb 只在 export_backend=gpu_transfer 上运行："
                f"当前 export_backend={requested_export_backend!r} 的链路会把帧交给 8-bit "
                "SDR 通路，HDR 链不会悄悄回退到它。")
        working_scale = float(job_settings.get("working_scale", 1.0))
        use_pyramid = bool(job_settings.get("residual_pyramid", True)) and (working_scale < 1.0)
        residual_strength = float(job_settings.get("residual_strength", 1.0))
        multi_pass = int(job_settings.get("multi_pass", 1))

        nr_w, nr_h = native_effects.processing_size(job_settings, (w, h))
        live_w, live_h = pipeline.resolve_working_dims(nr_w, nr_h, working_scale)

        # P1-07 fuse: refuse new DLSS work once native state may be polluted.
        self._ensure_native_healthy()
        acquired = self._live_lock.acquire(timeout=15)
        if not acquired:
            raise RuntimeError(
                "DLSS 会话锁 (_live_lock) 获取超时 15s，无法初始化 Live 会话："
                "预览线程可能持有锁或上一会话未正确释放，拒绝无限挂起"
            )
        try:
            self._close_gpu_preview()
            if self._live_session:
                try:
                    _run_with_timeout(self._live_session.close, 15, "DLSS 会话关闭")
                except Exception as ce:
                    diagnostic_logger.log("WARN", "Queue", f"旧会话关闭异常/超时，继续重建: {ce}")
                    if isinstance(ce, NativeCallTimeout):
                        self._mark_native_degraded(f"旧会话关闭超时 ({ce})")
                        raise
                self._live_session = None
                self._live_session_kind = None
            try:
                # SDR10/HDR and ordered DLAA/VSR pipelines must request the same
                # wide session as preview. The NR factory uses color_format;
                # native_effects.make_live uses wide, not that NR keyword.
                live = _run_with_timeout(
                    lambda: self._create_export_live(live_w, live_h, job_settings,
                                                    wide=experimental_hdr or sdr10_job),
                    120, "DLSS Live 会话初始化 (4K 首建需编译着色器，请耐心)")
            except Exception as e:
                if isinstance(e, NativeCallTimeout):
                    self._mark_native_degraded(f"会话初始化超时 ({e})")
                raise RuntimeError(f"GPU DLSS 会话初始化失败：{e}") from e
            self._live_session = live
            self._live_session_dims = (live_w, live_h)
            self._live_session_kind = (
                "gpu_hdr_fp16" if experimental_hdr else
                "gpu_sdr10_fp16" if sdr10_job else
                "gpu_zero_motion" if gpu_guidance_off else "gpu_flow")
            job["dlss_host"] = _gpu_host_provenance(live)
            if sdr10_job:
                job["precision_mode"] = "sdr10_bt709"
                job["dlss_session_color_format"] = getattr(live, "color_format", None)
            # What the queue row and the finished job have to be able to say about the
            # colour of the session this job drew its frames on. Written only for an
            # HDR job, so the row an SDR job has always carried is still exactly that.
            if experimental_hdr:
                job["experimental_hdr_scrgb"] = True
                job["dlss_session_color_format"] = getattr(live, "color_format", None)
                diagnostic_logger.log(
                    "INFO", "Queue",
                    f"作业 [{job['id']}] 实验性 HDR10 会话已就绪: "
                    f"format={getattr(live, 'color_format', '?')}"
                    f" paper_white={job_settings.get('hdr_paper_white_scrgb', 8.0)} scRGB"
                    f" @ {live_w}x{live_h} -> {w}x{h}")
            if gpu_guidance_off:
                probe = getattr(live, "supports_zero_motion", None)
                if not (callable(probe) and probe()):
                    # A refusal, not a fallback. This build ships one DLSS host, and
                    # the one that is loaded has no entry that serves this job's
                    # claim; drawing the frame anywhere else would report a GPU run
                    # that did not happen.
                    try:
                        live.close()
                    except Exception as ce:
                        diagnostic_logger.log(
                            "WARN", "Queue", f"被拒绝的 GPU 会话关闭异常: {ce}")
                    self._live_session = None
                    self._live_session_dims = None
                    self._live_session_kind = None
                    raise RuntimeError(
                        "dlss_backend=gpu_flow_d3d12 + guidance_mode=0 需要绑定常驻零运动纹理的 "
                        "GPU 宿主（native/dlss_gpu_host.dll ABI 3）；当前加载的会话没有该入口点。"
                        "请从 native/ 重新构建该宿主：本版本已移除旧宿主 dlssnr_host.dll，"
                        "没有可回退的第二条 DLSS 路径，或把 guidance_mode 设为 1（仅光流）")
                diagnostic_logger.log(
                    "INFO", "Queue",
                    f"作业 [{job['id']}] GPU 宿主零运动会话已就绪: "
                    f"chain={'gpu_transfer' if requested_export_backend == 'gpu_transfer' else 'legacy_pipe'}"
                    f" host={getattr(live, 'host_dll', 'dlss_gpu_host.dll')}"
                    f" abi={getattr(live, 'host_abi', '?')}"
                    f" @ {live_w}x{live_h}")
        finally:
            try:
                self._live_lock.release()
            except Exception:
                pass

        gm = guidance_mode
        need_flow = gm == 1
        if need_flow and job_settings.get('flow_engine') == 'fidelityfx' and requested_export_backend != 'gpu_transfer':
            raise ValueError('AMD FidelityFX Optical Flow requires GPU Transfer; no CPU fallback')

        guidance_service = None
        if need_flow and job_settings.get('flow_engine') != 'fidelityfx' and not (requested_export_backend == 'gpu_transfer' and job_settings.get('flow_engine') in ('raft', 'raft_large')):
            try:
                guidance_service = _run_with_timeout(
                    lambda: get_frame_guidance_service(job_settings.get("flow_engine", "nvof"), job_settings.get("flow_quality", "1F"),
                        **_raft_flow_options(job_settings)),
                    90, "光流引擎与 Guidance 初始化")
            except Exception as e:
                raise RuntimeError(f"光流引擎初始化失败：{e}（可尝试将光流引擎切换为 dis（CPU DIS）后重试）")

        flow_provider = (pipeline.GuidanceServiceFlowProvider(guidance_service, guidance_mode=gm)
                         if guidance_service is not None else None)
        if need_flow and (job_settings.get('flow_engine') == 'fidelityfx' or (requested_export_backend == 'gpu_transfer' and job_settings.get('flow_engine') in ('raft', 'raft_large'))):
            flow_provider = native_effects.make_flow_provider(job_settings, live)

        # Filled by the export so the queue, the UI and the benchmark harness can
        # report which residual implementation actually ran.
        backend_diagnostics = {}

        # Progress tracking initialization
        frames_done = 0
        t0 = time.time()

        def _progress(cur, total, msg):
            nonlocal frames_done, t0
            frames_done = cur
            elapsed = max(0.001, time.time() - t0)
            cur_fps = frames_done / elapsed
            rem = max(0, total - frames_done)
            eta = int(rem / max(0.1, cur_fps))

            job["current_frame"] = cur
            job["progress"] = round((cur / total) * 100.0, 1)
            job["fps"] = round(cur_fps, 1)
            job["eta"] = eta
            if progress_callback:
                progress_callback(job)

        try:
            res = pipeline.export_dlss_stream(
                video=v_path,
                live=live,
                lock=self._live_lock,
                frames=total_frames,
                fps=v_fps,
                cq=job["cq"],
                with_audio=job["with_audio"],
                flow_provider=flow_provider,
                progress=_progress,
                cancel=lambda: self._cancel_requested,
                out_path=out_path,
                transcode_settings=job.get("transcode_settings"),
                on_proc_started=self._register_proc,
                on_proc_exited=self._unregister_proc,
                job_settings=job_settings,
                diagnostics_out=backend_diagnostics,
            )
            if not self._cancel_requested:
                if res is None or not os.path.isfile(out_path):
                    raise RuntimeError(f"导出失败：目标文件未生成或路径无效: {out_path}")
                if os.path.getsize(out_path) == 0:
                    raise RuntimeError(f"导出失败：生成的目标文件大小为 0 字节: {out_path}")
        finally:
            # Written from the finally, not after the call: an opt-in backend that
            # refused is only diagnosable if the failed job still carries the
            # reasons it gave for refusing.
            job["residual_backend_diagnostics"] = backend_diagnostics
            # The host's own words about the run, from the session this job opened.
            # The GPU host clears last_error on a frame that succeeded, so this reads
            # None after a clean export and carries the native message after one that
            # failed - which is the whole of what a failed DLSS job can honestly be
            # shown: there is no second host's log file in this build to borrow.
            job["dlss_host_error"] = getattr(live, "last_error", None)
            # Keep the export session's bookkeeping straight: the GPU host refuses a
            # second session while one is active, so a cached session a job is done
            # with is closed and dropped here rather than left claimed by a job that
            # has finished with it.
            acquired = self._live_lock.acquire(timeout=10)
            try:
                if acquired and self._live_session is live:
                    live.close()
                    self._live_session = None
                    self._live_session_dims = None
            finally:
                if acquired:
                    try:
                        self._live_lock.release()
                    except Exception:
                        pass


# Global Singleton Service
dlss_service = DLSSService()
