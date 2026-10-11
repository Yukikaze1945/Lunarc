"""The desktop window's cached-frame video render on the GPU host, without a GPU.

Every frame this window draws is drawn by native/dlss_gpu_host.dll, the only native
DLSS host this build ships. The render is the third caller of that host's two motion
entries after the preview and the streaming export, and it routes the same way: the
guidance control names the entry, guidance off asks for the resident zero motion
texture and builds no motion plane at all, guidance on asks the host to stage the
cached `.flo`. What is apart from the other two callers is the name the one session
slot is claimed under, because that name is all that stands between this loop and a
preview frame that reset the temporal sequence mid-run. These tests pin that routing
and everything it had to keep: the PNG sequence the export-only mux consumes (name,
directory, colour order, extent), the slot and the kind that owns it, the resume that
skips cached frames, the progress and cancellation paths, and that a configuration the
host cannot serve is refused with a rebuild as its only remedy - at the session gate for
a kind with no entry, and at the boundary a retired host name still arrives at.

Two rules the one host writes itself:

A failed frame ends the run. The host's entries raise instead of returning nothing, and
a loop that carried on would leave a hole in the PNG sequence the mux stops at, with
nothing said about it.

The temporal rule follows the session, not the run: `reset` means "this frame does not
straight follow the last frame this session drew", so a run that resumed past a cached
frame in the middle starts a new sequence instead of accumulating a hole in it.

`dlss_gpu_engine.Live` is replaced by a factory handing out fake sessions that trap
every entry the route did not pick and validate their arguments the way the wrapper
does, so a frame of the wrong extent fails in the fake rather than at a ctypes
boundary. Most cases run the real `pipeline.imwrite_u` and the real PNG bytes;
`pipeline.iter_frames` and `video_info` are faked, because no test here needs a
container.
"""

import os
import queue
import sys
import threading

import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dlss_gpu_engine  # noqa: E402
import dlss_service  # noqa: E402
import gui  # noqa: E402
import pipeline  # noqa: E402
from dlss_errors import DLSSInferenceError  # noqa: E402

W, H = 64, 32
NFRAMES = 3


def _settings(**over):
    """The dict `_collect_settings` produces; the window's guidance default is off."""
    s = {'preset': 1, 'style': 0, 'intensity': 1.0, 'local_tone': 1.0,
         'local_struct': 1.0, 'skin_struct': 1.0, 'use_auto_mask': 1,
         'ui_correction': 0, 'guidance_mode': 0,
         'motion_scale_x': 1.0, 'motion_scale_y': 1.0}
    s.update(over)
    return s


def _plane(h, w, seed):
    """A non-flat float32 HxWx2 displacement plane, distinct per frame and per extent."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.ascontiguousarray(np.stack([x / 97.0 + seed, y / 61.0 - seed], axis=-1),
                                dtype=np.float32)


def _frames(count=NFRAMES, w=W, h=H):
    rng = np.random.default_rng(7)
    return [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for _ in range(count)]


class _Base:
    def __init__(self, w, h, settings=None):
        self._w, self._h = int(w), int(h)
        self.settings = dict(settings or {})
        self.closed = False
        self.last_error = None
        self.reset_flags = []
        # Read off the session, as the wrapper reads them off the library that answered:
        # which host drew a frame is invisible in the frame, and the ready line the
        # window logs is the only place an operator can read it back.
        self.host_dll = "dlss_gpu_host.dll"
        self.host_abi = 3

    def close(self):
        self.closed = True

    def is_alive(self):
        return not self.closed


class GpuLive(_Base):
    """A GPU host carrying both motion entries, as the built ABI-3 DLL does.

    The checks mirror `dlss_gpu_engine.Live`'s own argument validation, including that
    the render passes no `out=` buffer and gets one back, so a frame or plane of the
    wrong shape or extent fails here rather than at the ctypes boundary. Every other
    entry raises: they are the wrong doors, and a run that went through one is a
    failure.
    """

    def __init__(self, w, h, settings=None):
        super().__init__(w, h, settings)
        self.zero_motion_calls = 0
        self.host_motion_calls = 0
        self.seen_rgba = []
        self.seen_motion = []
        self.seen_out_is_new = []

    def supports_zero_motion(self):
        return True

    def supports_host_motion(self):
        return True

    def _check_rgba(self, rgba):
        assert isinstance(rgba, np.ndarray) and rgba.dtype == np.uint8
        assert rgba.shape == (self._h, self._w, 4), (
            f"frame {rgba.shape} handed to a session opened at {(self._h, self._w)}")
        assert rgba.flags.c_contiguous, "the GPU host rejects a non-contiguous frame"
        assert np.all(rgba[..., 3] == 255), "alpha is not opaque"

    def _render(self, rgba):
        out = rgba.copy()
        out[..., 2] = np.clip(out[..., 2].astype(np.int16) + 21, 0, 255)
        return out

    def process_zero_motion(self, rgba, reset=False, out=None):
        self._check_rgba(rgba)
        self.zero_motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.seen_rgba.append(np.array(rgba))
        return self._finish(rgba, out, self._render(rgba))

    def process_host_motion(self, rgba, motion, reset=False, out=None):
        self._check_rgba(rgba)
        assert motion is not None, "a guided frame reached the host with no motion plane"
        assert (isinstance(motion, np.ndarray) and motion.dtype == np.float32
                and motion.shape == (self._h, self._w, 2) and motion.flags.c_contiguous), (
            "motion must be a contiguous float32 HxWx2 plane at the DLSS input "
            f"resolution, got {getattr(motion, 'shape', motion)} "
            f"{getattr(motion, 'dtype', '')}")
        assert motion.strides[0] == self._w * 8, (
            "the native entry reads float32x2 rows at this pitch")
        self.host_motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.seen_rgba.append(np.array(rgba))
        self.seen_motion.append(np.array(motion))
        return self._finish(rgba, out, self._render(rgba))

    def _finish(self, rgba, out, rendered):
        """Return as the wrapper does: into the caller's buffer, or a fresh one."""
        if out is None:
            self.seen_out_is_new.append(True)
            return rendered
        self.seen_out_is_new.append(False)
        assert out.shape == rgba.shape and out.dtype == rgba.dtype
        np.copyto(out, rendered)
        return out

    def _refuse(self, *args, **kwargs):
        raise AssertionError("a video render frame reached the wrong GPU host entry")

    process_device_flow = _refuse
    process_device_color_flow = _refuse
    process_device_color_flow_resource = _refuse
    process_device_color_zero_motion_resource = _refuse

    def update(self, settings):
        # The host bakes its options at open and exposes no live update entry, so a
        # reuse that reached for one is the bug, not a no-op.
        raise AssertionError("GPU session handed the live-update entry it lacks")


class NoHostMotionGpuLive(GpuLive):
    """A host built without the host-motion entry: a guided render must be refused."""

    def supports_host_motion(self):
        return False

    def process_host_motion(self, *args, **kwargs):
        raise AssertionError("a host with no host-motion entry was asked to run one")


class Abi2GpuLive(GpuLive):
    """A host that binds no zero motion texture: an unguided render must be refused."""

    def supports_zero_motion(self):
        return False

    def process_zero_motion(self, *args, **kwargs):
        raise AssertionError("a host with no zero-motion entry was asked to run one")


class FailingGpuLive(GpuLive):
    """A host that has the entries and fails every frame, as the wrapper reports it."""

    def process_zero_motion(self, rgba, reset=False, out=None):
        self.zero_motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.last_error = "zero-motion DLSS frame failed (5): fake card error"
        raise DLSSInferenceError(self.last_error)

    def process_host_motion(self, rgba, motion, reset=False, out=None):
        self.host_motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.last_error = "host-motion DLSS frame failed (5): fake card error"
        raise DLSSInferenceError(self.last_error)


class _Factory:
    """Counts and keeps every Live the render constructs."""

    def __init__(self, cls):
        self._cls = cls
        self.built = []

    def __call__(self, w, h, settings=None):
        live = self._cls(w, h, settings)
        self.built.append(live)
        return live

    @property
    def count(self):
        return len(self.built)


class _Worker:
    """A preview worker that records what the render did to it."""

    def __init__(self):
        self.events = []
        self.paused = False

    def pause(self):
        self.events.append("pause")
        self.paused = True

    def resume(self):
        self.events.append("resume")
        self.paused = False

    def close_live(self):
        self.events.append("close_live")

    def stop(self):
        pass


class _Var:
    """A tk variable stand-in for the one the render sets when it finishes."""

    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value


class _Dialogs:
    def __init__(self):
        self.shown = []

    def showinfo(self, title, message):
        self.shown.append(("info", title, str(message)))

    def showwarning(self, title, message):
        self.shown.append(("warning", title, str(message)))

    def showerror(self, title, message):
        self.shown.append(("error", title, str(message)))

    @property
    def kinds(self):
        return [s[0] for s in self.shown]


def _app(monkeypatch, tmp_path, gpu_cls=GpuLive, settings=None, nframes=NFRAMES,
         frames=None):
    """A real App, minus Tk, wired to fake sessions and a fake preview worker."""
    app = gui.App.__new__(gui.App)
    app._dlss_lock = threading.Lock()
    app._dlss_session = None
    app._dlss_session_kind = None
    app._image_batch_running = False
    app._video_task_kind = None
    app._cancel_requested = False
    app.current_is_image = False
    app.logs, app.statuses, app.progress = [], [], []
    app.logln = lambda msg: app.logs.append(str(msg))
    app.set_status = lambda msg: app.statuses.append(str(msg))
    app.set_progress = lambda i, total, extra="": app.progress.append((i, total, extra))
    app.call_ui = lambda fn, *a, **k: fn(*a, **k)
    app.view_var = _Var("对比")
    app.dialogs = _Dialogs()
    monkeypatch.setattr(gui, "messagebox", app.dialogs)
    app.video = str(tmp_path / "clip.mp4")
    app.nframes = nframes
    app.fps = 30.0
    app._ui_queue = queue.Queue()
    # Read off the worker thread, exactly as the panel's cached copy is: the render must
    # use this dict and not re-read widgets it cannot touch from a worker thread.
    app._cached_settings = _settings(**(settings or {}))
    app._preview_worker = _Worker()

    app._gpu = _Factory(gpu_cls)
    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", app._gpu)
    # The window probes the clip for the session geometry; no test needs a real container.
    monkeypatch.setattr(pipeline, "video_info", lambda path: (nframes, 30.0, W, H))
    _install_frames(app, monkeypatch, frames if frames is not None else _frames())
    return app


def _install_frames(app, monkeypatch, frames):
    def _fake_iter(video, **kwargs):
        for idx, fr in enumerate(frames):
            yield idx, fr

    monkeypatch.setattr(pipeline, "iter_frames", _fake_iter)


def _run_render(app):
    """`_run_dlss_gui` off the main thread, the way `_do` runs it."""
    box = {}

    def run():
        try:
            box['value'] = app._run_dlss_gui()
        except BaseException as ex:  # noqa: BLE001 - reported, never swallowed
            box['error'] = ex

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(120)
    assert not thread.is_alive(), "the render never returned"
    if 'error' in box:
        raise box['error']
    return box['value']


def _dlss_dir(app):
    return pipeline.out_dirs(app.video)[1]


def _png(app, idx):
    return os.path.join(_dlss_dir(app), f"{idx:06d}.png")


def _cached_png(app, idx, colour=(9, 8, 7)):
    """Pre-seed one cached frame, as an interrupted earlier run left it."""
    os.makedirs(_dlss_dir(app), exist_ok=True)
    img = np.zeros((H, W, 3), np.uint8)
    img[:, :] = colour
    pipeline.imwrite_u(_png(app, idx), img)
    return _png(app, idx)


def _write_flo(app, frame, plane):
    flow_dir = pipeline.out_dirs(app.video)[0]
    os.makedirs(flow_dir, exist_ok=True)
    pipeline.write_flo(os.path.join(flow_dir, f"{frame:06d}.flo"), plane)


def _logged(app, needle):
    return [m for m in app.logs if needle in m]


# --- Routing ----------------------------------------------------------------

def test_an_unguided_render_runs_the_gpu_zero_motion_entry(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    assert _run_render(app) is True

    assert app._gpu.count == 1, "one run drew its frames on more than one session"
    live = app._gpu.built[0]
    assert (live.zero_motion_calls, live.host_motion_calls) == (NFRAMES, 0)
    assert live.reset_flags == [True, False, False]
    assert app._dlss_session_kind == "render_zero_motion"
    assert live is app._dlss_session
    assert all(o is True for o in live.seen_out_is_new), \
        "the render hands the host no workspace buffer"
    # The ready line is the run's record of what drew its frames.
    assert _logged(app, "GPU 宿主零运动会话已就绪") == [
        "[DLSS 渲染] GPU 宿主零运动会话已就绪: dlss_gpu_host.dll abi=3 @ 64x32"]
    assert live._gpu_options_key == dlss_gpu_engine.baked_option_key(_settings())
    assert any("DLSS 完成" in m and _dlss_dir(app) in m for m in app.logs)
    assert app.view_var.value == "DLSS"
    assert not live.closed, "the run's session is the slot's, not the render's to destroy"


def test_a_guided_render_runs_the_gpu_host_motion_entry(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": 1})
    for idx in range(NFRAMES):
        _write_flo(app, idx, _plane(H, W, idx))

    assert _run_render(app) is True

    assert app._gpu.count == 1
    live = app._gpu.built[0]
    assert (live.host_motion_calls, live.zero_motion_calls) == (NFRAMES, 0)
    assert live.reset_flags == [True, False, False]
    assert app._dlss_session_kind == "render_host_motion"
    assert _logged(app, "GPU 宿主运动向量会话已就绪") == [
        "[DLSS 渲染] GPU 宿主运动向量会话已就绪: dlss_gpu_host.dll abi=3 @ 64x32"]
    assert not any("平滑降级" in m for m in app.logs), "a full cache was reported as degraded"


def test_the_cached_flo_vectors_are_what_the_host_motion_entry_read(monkeypatch, tmp_path):
    """The plane the host stages is the plane on disk, per frame, not a silent zero."""
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": 1})
    planes = [_plane(H, W, i + 1) for i in range(NFRAMES)]
    for idx, plane in enumerate(planes):
        _write_flo(app, idx, plane)

    _run_render(app)

    handed = app._gpu.built[0].seen_motion
    assert len(handed) == NFRAMES
    for idx, plane in enumerate(planes):
        got = handed[idx]
        assert got.dtype == np.float32 and got.shape == (H, W, 2)
        assert got.strides == (W * 8, 8, 4)
        assert np.any(got), f"frame {idx} was handed flat motion"
        assert np.array_equal(got, plane), (
            f"frame {idx} bound something other than its own cached .flo")


def test_a_flow_plane_cached_at_another_extent_arrives_at_the_frame_size(monkeypatch,
                                                                        tmp_path):
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": 1})
    for idx in range(NFRAMES):
        _write_flo(app, idx, _plane(H // 2, W // 2, idx))

    _run_render(app)

    handed = app._gpu.built[0].seen_motion
    assert [g.shape for g in handed] == [(H, W, 2)] * NFRAMES
    assert all(g.strides[0] == W * 8 and np.any(g) for g in handed)


@pytest.mark.parametrize("mode, kind, entry, other", [
    (0, "render_zero_motion", "zero_motion_calls", "host_motion_calls"),
    (1, "render_host_motion", "host_motion_calls", "zero_motion_calls"),
])
def test_the_render_kind_is_read_off_the_guidance_control(monkeypatch, tmp_path, mode,
                                                         kind, entry, other):
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": mode})
    if mode == 1:
        for idx in range(NFRAMES):
            _write_flo(app, idx, _plane(H, W, idx))

    assert gui.App.video_render_kind(_settings(guidance_mode=mode)) == kind
    _run_render(app)

    live = app._gpu.built[0]
    assert app._dlss_session_kind == kind
    assert getattr(live, entry) == NFRAMES and getattr(live, other) == 0
    assert app._gpu.count == 1 and live is app._dlss_session


def test_the_three_video_callers_of_these_entries_do_not_share_a_kind():
    """The slot is keyed by kind, so the render, the export and the preview need names of
    their own or one loop silently drives another loop's temporal sequence."""
    s = _settings()
    guided = _settings(guidance_mode=1)
    names = {gui.App.video_preview_kind(s), gui.App.video_preview_kind(guided),
             gui.App.video_export_kind(s), gui.App.video_export_kind(guided),
             gui.App.video_render_kind(s), gui.App.video_render_kind(guided)}
    assert len(names) == 6
    assert names <= set(gui.App._GPU_HOST_KINDS)


def test_an_unguided_render_builds_no_cpu_motion_plane(monkeypatch, tmp_path):
    """Guidance off is answered by the zero texture the host bound at open, so a plane
    built for an argument slot nothing reads is CPU work and a copy nobody asked for."""
    app = _app(monkeypatch, tmp_path)
    seen = []
    real_zeros = np.zeros

    def spy_zeros(shape, *args, **kwargs):
        seen.append((tuple(shape), kwargs.get("dtype", args[0] if args else None)))
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", spy_zeros)

    _run_render(app)

    assert [s for s, dt in seen
            if dt is np.float32 and len(s) == 3 and s[2] == 2] == [], \
        "an unguided render still built a host motion plane"
    live = app._gpu.built[0]
    assert live.zero_motion_calls == NFRAMES and live.seen_motion == []


# --- Flow cache: the documented smooth degrade, named honestly ----------------

def test_a_frame_missing_its_cached_flow_degrades_and_is_reported(monkeypatch, tmp_path):
    """No .flo for one frame is the documented smooth-degrade case, not a failure."""
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": 1})
    _write_flo(app, 0, _plane(H, W, 0))
    _write_flo(app, 2, _plane(H, W, 2))

    assert _run_render(app) is True

    live = app._gpu.built[0]
    assert live.host_motion_calls == NFRAMES, "the route is not one frame's data"
    assert not np.any(live.seen_motion[1]), "the frame with no cache got something else"
    assert np.any(live.seen_motion[0]) and np.any(live.seen_motion[2])
    assert os.path.isfile(_png(app, 1)), "a degraded frame was not written like the others"
    degrade = [m for m in app.logs if "平滑降级" in m]
    assert len(degrade) == 2, "the operator was told twice about one event, or not at all"
    assert "第 1 帧缺少光流缓存" in degrade[0]
    assert degrade[1] == ("[平滑降级] 1 帧缺少光流缓存，这些帧以零运动向量送入渲染"), (
        "a degraded frame was described as something other than zero motion fed in")


def test_only_the_first_degraded_frame_is_announced_and_the_total_is_counted(
        monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": 1})
    _write_flo(app, 1, _plane(H, W, 1))

    _run_render(app)

    degrade = [m for m in app.logs if "平滑降级" in m]
    assert len(degrade) == 2
    assert "第 0 帧缺少光流缓存" in degrade[0]
    assert "2 帧缺少光流缓存" in degrade[1]
    assert not any("3 帧缺少光流缓存" in m for m in app.logs)


def test_a_corrupt_flow_cache_degrades_like_a_missing_one(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": 1})
    for idx in range(NFRAMES):
        _write_flo(app, idx, _plane(H, W, idx))
    with open(os.path.join(pipeline.out_dirs(app.video)[0], "000001.flo"), "wb") as fh:
        fh.write(b"not a flo file")

    assert _run_render(app) is True

    live = app._gpu.built[0]
    assert live.host_motion_calls == NFRAMES
    assert not np.any(live.seen_motion[1])
    assert any("1 帧缺少光流缓存" in m for m in app.logs)


def test_an_unguided_render_reads_no_flow_file(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    for idx in range(NFRAMES):
        _write_flo(app, idx, _plane(H, W, idx))
    reads = []
    real_read = pipeline.read_flo

    def _counting(path):
        reads.append(os.path.basename(str(path)))
        return real_read(path)

    monkeypatch.setattr(pipeline, "read_flo", _counting)

    _run_render(app)

    assert reads == [], "guidance off still walked the flow cache"


# --- Resume, skips and the temporal sequence ---------------------------------

def test_cached_frames_are_skipped_and_the_run_continues_from_them(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    seeded = _cached_png(app, 0)
    before = open(seeded, "rb").read()

    assert _run_render(app) is True

    live = app._gpu.built[0]
    assert live.zero_motion_calls == NFRAMES - 1, "a cached frame was rendered again"
    assert open(seeded, "rb").read() == before, "the skip overwrote the cached frame"
    assert [(i, t) for i, t, _ in app.progress] == \
        [(0, NFRAMES), (1, NFRAMES), (2, NFRAMES), (3, NFRAMES)], "the bar skipped a tick"
    assert live.reset_flags == [True, False], (
        "the run did not start its own sequence at the first frame it drew")


def test_a_skip_in_the_middle_starts_a_new_temporal_sequence(monkeypatch, tmp_path):
    """The frame after a hole has no history to continue, whatever the run numbering says."""
    app = _app(monkeypatch, tmp_path)
    _cached_png(app, 1)

    _run_render(app)

    live = app._gpu.built[0]
    assert live.zero_motion_calls == NFRAMES - 1
    assert live.reset_flags == [True, True], (
        "frame 2 continued a sequence frame 1 was never drawn into")


def test_a_second_render_run_starts_a_new_sequence_on_the_reused_session(monkeypatch,
                                                                        tmp_path):
    app = _app(monkeypatch, tmp_path)
    _run_render(app)
    first = app._gpu.built[0]
    for idx in range(NFRAMES):
        os.unlink(_png(app, idx))

    _run_render(app)

    assert app._gpu.count == 1, "an unchanged second render reopened the host"
    assert first is app._gpu.built[0]
    assert first.zero_motion_calls == 2 * NFRAMES
    assert first.reset_flags == [True, False, False] * 2, (
        "the second run did not start a new temporal sequence")


def test_a_run_that_skipped_every_frame_renders_nothing(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    for idx in range(NFRAMES):
        _cached_png(app, idx, colour=(idx, 0, 0))

    assert _run_render(app) is True

    live = app._gpu.built[0]
    assert (live.zero_motion_calls, live.host_motion_calls) == (0, 0)
    assert live.reset_flags == []
    assert [pipeline.imread_u(_png(app, i))[0, 0].tolist() for i in range(NFRAMES)] == \
        [[0, 0, 0], [1, 0, 0], [2, 0, 0]], "a complete cache was rewritten"


def test_a_guidance_flip_between_render_runs_rebuilds_the_session(monkeypatch, tmp_path):
    """The two render kinds are the same host at the same size; only the kind tells them."""
    app = _app(monkeypatch, tmp_path)
    for idx in range(NFRAMES):
        _write_flo(app, idx, _plane(H, W, idx))

    _run_render(app)
    unguided = app._gpu.built[0]
    for idx in range(NFRAMES):
        os.unlink(_png(app, idx))
    app._cached_settings = _settings(guidance_mode=1)
    _run_render(app)
    guided = app._gpu.built[1]

    assert app._gpu.count == 2 and unguided.closed
    assert (unguided.zero_motion_calls, unguided.host_motion_calls) == (NFRAMES, 0)
    assert (guided.host_motion_calls, guided.zero_motion_calls) == (NFRAMES, 0)
    assert guided.reset_flags == [True, False, False], (
        "a fresh session was handed a run that thought it was mid-sequence")


def test_a_baked_option_change_between_render_runs_rebuilds_the_session(monkeypatch,
                                                                       tmp_path):
    app = _app(monkeypatch, tmp_path)
    _run_render(app)
    first = app._gpu.built[0]
    for idx in range(NFRAMES):
        os.unlink(_png(app, idx))

    app._cached_settings = _settings(intensity=0.25)
    _run_render(app)

    assert app._gpu.count == 2, "the GPU host bakes intensity at open; it must reopen"
    assert first.closed


def test_a_session_marked_with_other_baked_options_is_rebuilt_not_reused(monkeypatch,
                                                                        tmp_path):
    """The reuse gate reads the key the host was opened with, not just the entry name.

    A session whose marker no longer describes the request would run the whole run with
    the old options baked in, and the host has no live update entry to correct it.
    """
    app = _app(monkeypatch, tmp_path)
    _run_render(app)
    first = app._gpu.built[0]
    for idx in range(NFRAMES):
        os.unlink(_png(app, idx))

    first._gpu_options_key = ("stale",)
    _run_render(app)

    assert app._gpu.count == 2, "a stale baked key was reused instead of reopened"
    assert first.closed
    assert app._gpu.built[1].zero_motion_calls == NFRAMES


def test_a_cached_session_marked_as_another_entry_is_never_reused(monkeypatch, tmp_path):
    """The kind marker is all that tells two sessions of the same host apart.

    Same DLL, same size, same baked options: a render handed the preview's session would
    be driving a slot it never claimed, and the preview's temporal sequence with it.
    """
    app = _app(monkeypatch, tmp_path)
    preview = app.get_dlss_session(W, H, _settings(), kind="gpu_zero_motion")
    assert app._dlss_session_kind == "gpu_zero_motion"

    live = app.get_dlss_session(W, H, _settings(), kind="render_zero_motion")

    assert preview.closed, "the preview's session was left on the slot"
    assert live is app._gpu.built[1] and live is not preview
    assert app._dlss_session_kind == "render_zero_motion"
    assert preview.zero_motion_calls == 0


# --- The PNG sequence the export-only mux consumes ---------------------------

def test_the_render_writes_the_png_sequence_the_mux_reads(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    _run_render(app)

    assert sorted(os.listdir(_dlss_dir(app))) == \
        [f"{i:06d}.png" for i in range(NFRAMES)]
    frames = _frames()
    for idx, bgr in enumerate(frames):
        got = pipeline.imread_u(_png(app, idx))
        assert got is not None and got.shape == (H, W, 3), (
            f"frame {idx} is not the extent the clip reports")
        # The host's RGBA result, dropped to BGR: the fake shifted the blue channel it was
        # handed, so a swapped channel order or a stray alpha shows up here.
        expect = np.dstack([np.clip(bgr[..., 0].astype(np.int16) + 21, 0, 255)
                            .astype(np.uint8), bgr[..., 1], bgr[..., 2]])
        assert np.array_equal(got, expect), f"frame {idx} is not colour-correctly stored"


# --- Refusals: one host, and no way around it --------------------------------

def test_the_window_names_the_one_host_and_every_entry_it_serves():
    """A refused configuration has one remedy, so the table is the whole surface.

    The kinds are the entries; the entries are the capabilities the loaded DLL binds;
    the remedy for a missing one is a rebuild, because this build ships no second host
    to draw the frame behind the caller's back instead.
    """
    assert pipeline.DLSS_BACKENDS == ("gpu_flow_d3d12",)
    assert set(gui.App._GPU_HOST_KINDS) == {
        "image", "gpu_zero_motion", "gpu_host_motion", "export_zero_motion",
        "export_host_motion", "render_zero_motion", "render_host_motion"}
    assert {entry[0] for entry in gui.App._GPU_HOST_KINDS.values()} == {
        "supports_zero_motion", "supports_host_motion"}
    for kind, (capability, refusal, ready) in gui.App._GPU_HOST_KINDS.items():
        assert "native/dlss_gpu_host.dll" in refusal and "rebuild" in refusal, kind
        assert capability.startswith("supports_") and ready, kind


@pytest.mark.parametrize("kind", list(pipeline.RETIRED_DLSS_BACKENDS) +
                         ["render_device_flow", "gpu_host_zero_motion"])
def test_a_kind_the_host_has_no_entry_for_is_refused_and_names_the_one_host(
        monkeypatch, tmp_path, kind):
    """Session bookkeeping is the boundary: an unknown kind opens nothing and says why.

    A caller still holding a retired host's name for its session, and a caller that
    misnames an entry, both get the same answer as any other unknown kind: the one host
    this build ships, and the entries it has. The window's own boundary reports it too -
    `get_dlss_session` answers None, which is what turns a stale name into a render that
    says it did not run instead of one that drew frames on something nobody chose.
    """
    assert kind not in gui.App._GPU_HOST_KINDS
    app = _app(monkeypatch, tmp_path)
    with app._dlss_lock:
        app._video_task_kind = gui.App.video_render_kind(_settings())

    with pytest.raises(DLSSInferenceError) as refused:
        app._get_or_create_session_locked(W, H, _settings(), kind)

    message = str(refused.value)
    assert f"kind={kind!r}" in message
    assert "native/dlss_gpu_host.dll" in message
    assert all(name in message for name in gui.App._GPU_HOST_KINDS)
    assert app._gpu.count == 0, "a refused kind opened a session anyway"
    assert app._dlss_session is None and app._dlss_session_kind is None
    assert app._video_task_kind == "render_zero_motion", "the refusal moved the slot claim"

    assert app.get_dlss_session(W, H, _settings(), kind=kind) is None
    assert _logged(app, "未知 DLSS 会话类型") == ["[DLSS 引擎] " + message]


def test_a_session_request_that_omits_the_kind_is_a_type_error(monkeypatch, tmp_path):
    """There is no default kind to fall back on: every caller names the entry it needs."""
    app = _app(monkeypatch, tmp_path)

    with pytest.raises(TypeError):
        app.get_dlss_session(W, H, _settings())

    assert app._gpu.count == 0


def test_a_job_that_names_a_retired_host_is_refused_where_it_is_read():
    """A retired host name has no route left, so it fails at the boundary that reads it.

    Settings files and queued jobs written while a second host still existed carry that
    name, and the answer has to be a refusal that says what is gone and what replaces it
    - the only thing an operator holding an old script can act on. Answering it with a
    frame drawn on the surviving host would report a run nobody asked for, which is the
    same guarantee this file holds for the render's own session slot.
    """
    assert pipeline.RETIRED_DLSS_BACKENDS
    assert set(pipeline.RETIRED_DLSS_BACKENDS).isdisjoint(pipeline.DLSS_BACKENDS)
    svc = dlss_service.DLSSService()
    assert svc.settings["dlss_backend"] in pipeline.DLSS_BACKENDS

    for retired in pipeline.RETIRED_DLSS_BACKENDS:
        with pytest.raises(ValueError) as refused:
            svc.update_settings({"dlss_backend": retired})
        message = str(refused.value)
        assert retired in message and pipeline.DLSS_BACKENDS[0] in message
        assert svc.settings["dlss_backend"] == pipeline.DLSS_BACKENDS[0], \
            "a refused write must not have been merged into the settings"
        with pytest.raises(ValueError, match=retired):
            svc._job_dlss_backend({"dlss_backend": retired})
        assert svc._live_session is None, "a refused job opened a session to draw anyway"


def test_a_host_without_the_zero_motion_entry_is_refused_not_replaced(monkeypatch,
                                                                     tmp_path):
    app = _app(monkeypatch, tmp_path, gpu_cls=Abi2GpuLive)

    assert _run_render(app) is False

    refused = app._gpu.built[0]
    assert app._gpu.count == 1, "an unsupported configuration opened another session"
    assert refused.closed and refused.zero_motion_calls == 0
    assert refused.host_motion_calls == 0, "the refusal reached for the other entry"
    assert app._dlss_session is None and app._dlss_session_kind is None
    assert _logged(app, "ABI 2") == [
        "[DLSS 引擎] dlss_gpu_host.dll exposes no zero-motion entry point (ABI 2); "
        "rebuild native/dlss_gpu_host.dll to render guidance_mode=0 video frames on "
        "the GPU host"]
    assert any("无法初始化 DLSS 引擎" in m for m in app.logs)
    assert os.listdir(_dlss_dir(app)) == [], "a refused render still wrote frames"
    assert app._video_task_kind is None


def test_a_host_without_the_host_motion_entry_is_refused_not_replaced(monkeypatch,
                                                                     tmp_path):
    app = _app(monkeypatch, tmp_path, gpu_cls=NoHostMotionGpuLive,
               settings={"guidance_mode": 1})
    _write_flo(app, 0, _plane(H, W, 0))

    assert _run_render(app) is False

    refused = app._gpu.built[0]
    assert app._gpu.count == 1
    assert refused.closed and refused.host_motion_calls == 0
    assert refused.zero_motion_calls == 0, (
        "a guided run that lost its entry answered with an unguided frame")
    assert _logged(app, "host-motion") == [
        "[DLSS 引擎] dlss_gpu_host.dll exposes no host-motion entry point; rebuild "
        "native/dlss_gpu_host.dll to render guidance_mode=1 video frames on the GPU host"]
    assert app._video_task_kind is None


def test_a_missing_gpu_host_dll_is_reported_and_ends_the_run(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    def _missing_library(w, h, settings=None):
        raise OSError("could not find module 'dlss_gpu_host.dll' (or one of its "
                      "dependencies)")

    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", _missing_library)

    assert _run_render(app) is False

    assert _logged(app, "dlss_gpu_host.dll") == [
        "[DLSS 引擎] could not find module 'dlss_gpu_host.dll' (or one of its dependencies)"]
    assert any("无法初始化 DLSS 引擎" in m for m in app.logs)
    assert app._dlss_session is None and app._dlss_session_kind is None
    assert os.listdir(_dlss_dir(app)) == []
    assert app._video_task_kind is None


def test_a_frame_the_host_fails_ends_the_run_and_says_so(monkeypatch, tmp_path):
    """The host's entries raise for a frame they could not draw, and the run stops.

    Carrying on would leave a hole in the PNG sequence nobody was told about, and the
    mux that reads that sequence stops at the first gap.
    """
    app = _app(monkeypatch, tmp_path, gpu_cls=FailingGpuLive)

    assert _run_render(app) is False

    live = app._gpu.built[0]
    assert live.zero_motion_calls == 1, "a run that failed on frame 0 kept going"
    assert _logged(app, "zero-motion") == [
        "[DLSS] zero-motion DLSS frame failed (5): fake card error"]
    assert any(m.startswith("DLSS 出错") for m in app.statuses)
    assert not os.path.exists(_png(app, 0)), "a frame that failed was written anyway"
    assert app._video_task_kind is None


def test_a_frame_the_decoder_grew_shuts_the_run_down_instead_of_drawing_it_wrong(
        monkeypatch, tmp_path):
    """The session opens at the extent the container reports and the host sizes every
    buffer from it; a decoder that disagreed would hand the native entry a pointer whose
    rows run past the frame it was given, which has to be an error, not a picture."""
    app = _app(monkeypatch, tmp_path)
    _install_frames(app, monkeypatch, _frames(NFRAMES, w=W * 2, h=H * 2))

    assert _run_render(app) is False

    assert app._gpu.built[0].zero_motion_calls == 0
    mismatch = _logged(app, "handed to a session opened at")
    assert len(mismatch) == 1, "the run argued with the decoder more than once, or not at all"
    assert mismatch[0].startswith(
        "[DLSS] frame (64, 128, 4) handed to a session opened at (32, 64)")
    assert app._video_task_kind is None


# --- The preview, parked and restored ---------------------------------------

def test_the_render_claims_the_slot_before_it_opens_the_session(monkeypatch, tmp_path):
    """The claim is first, so a preview frame cannot lose the run's session to it."""
    app = _app(monkeypatch, tmp_path)
    order = []
    real_get = gui.App._get_or_create_session_locked

    def _watched(self, w, h, settings, kind):
        order.append((self._video_task_kind, kind))
        return real_get(self, w, h, settings, kind)

    monkeypatch.setattr(gui.App, "_get_or_create_session_locked", _watched)

    _run_render(app)

    assert order == [("render_zero_motion", "render_zero_motion")]
    # Parking the preview belongs to `_do`, which runs before this loop starts; reaching
    # for the worker here would stop the preview for good on a run that resumed it.
    assert app._preview_worker.events == []


def test_the_preview_cannot_take_the_slot_from_a_running_render(monkeypatch, tmp_path):
    """Everything a preview could ask for is refused while the render owns the slot."""
    app = _app(monkeypatch, tmp_path)
    seen = {}
    real_iter = pipeline.iter_frames

    def _probe(video, **kwargs):
        for idx, fr in real_iter(video, **kwargs):
            if idx == 1:
                seen["during"] = [
                    app._get_or_create_session_locked(W, H, _settings(),
                                                     "gpu_zero_motion"),
                    app._get_or_create_session_locked(W, H, _settings(guidance_mode=1),
                                                     "gpu_host_motion"),
                    app._get_or_create_session_locked(W, H, _settings(),
                                                     "export_zero_motion"),
                    app._get_or_create_session_locked(W, H, _settings(), "image"),
                ]
                seen["gpu_built"] = app._gpu.count
                seen["reentry"] = app._get_or_create_session_locked(W, H, _settings(),
                                                                    "render_zero_motion")
                seen["live"] = app._dlss_session
            yield idx, fr

    monkeypatch.setattr(pipeline, "iter_frames", _probe)

    _run_render(app)

    assert seen["during"] == [None, None, None, None], "a preview took the render's slot"
    assert seen["gpu_built"] == 1
    assert seen["reentry"] is seen["live"], "the render's own kind lost its session"
    assert not seen["live"].closed


def test_a_preview_frame_after_the_run_replaces_the_render_session(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    _run_render(app)
    rendered = app._gpu.built[0]

    preview = app.get_dlss_session(W, H, _settings(), kind="gpu_zero_motion")

    assert rendered.closed, "the slot handed to the preview left the render open"
    assert preview is app._gpu.built[1] and preview is not rendered
    assert app._dlss_session_kind == "gpu_zero_motion"


def test_an_image_batch_in_progress_holds_the_slot_and_the_render_reports(
        monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    with app._dlss_lock:
        app._image_batch_running = True

    assert _run_render(app) is False

    assert app._gpu.count == 0, "the render evicted the session the image batch is on"
    assert app._dlss_session is None and app._dlss_session_kind is None
    assert any("无法初始化 DLSS 引擎" in m for m in app.logs)
    assert app._video_task_kind is None


# --- Cancellation -----------------------------------------------------------

def test_a_cancelled_render_stops_at_the_frame_and_leaves_nothing_behind(monkeypatch,
                                                                        tmp_path):
    class CancelAfterFirstFrame(GpuLive):
        def process_zero_motion(self, rgba, reset=False, out=None):
            result = super().process_zero_motion(rgba, reset=reset, out=out)
            app._cancel_requested = True
            return result

    app = _app(monkeypatch, tmp_path, gpu_cls=CancelAfterFirstFrame)

    assert _run_render(app) is False

    live = app._gpu.built[0]
    assert live.zero_motion_calls == 1, "the run did not stop when asked"
    assert os.path.isfile(_png(app, 0)), "the rendered frame was thrown away"
    assert not os.path.exists(_png(app, 1)), "the run drew a frame after the Stop"
    assert any("DLSS 处理于第 1 帧停止" in m for m in app.logs)
    assert app._video_task_kind is None
    assert [(i, t) for i, t, _ in app.progress] == [(0, NFRAMES), (1, NFRAMES)]


def test_a_render_cancelled_before_the_first_frame_writes_nothing(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    app._cancel_requested = True

    assert _run_render(app) is False

    assert app._gpu.built[0].zero_motion_calls == 0
    assert os.listdir(_dlss_dir(app)) == []
    assert app._video_task_kind is None


def test_the_progress_the_render_reports_is_the_frames_it_walked(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    _run_render(app)

    assert [(i, t) for i, t, _ in app.progress] == [(0, NFRAMES), (1, NFRAMES),
                                                   (2, NFRAMES), (3, NFRAMES)]
    assert all(extra == "DLSS" for _, _, extra in app.progress)


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
