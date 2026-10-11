"""The desktop window's streaming video export on the project GPU host, without a GPU.

The export draws every frame on `native/dlss_gpu_host.dll` — the only native DLSS host
this build ships, reached through `dlss_gpu_engine.Live` and named by
`dlss_backend="gpu_flow_d3d12"`. Guidance off asks that host for its resident
zero-motion texture and hands over no motion input at all; guidance on asks it to stage
the cached `.flo` plane. Both are the preview's two entries at the same extent, claimed
under the export's own names because the one session slot is keyed by kind: a running
export has to keep a preview frame out of the temporal sequence the encoder is being fed
from. These tests pin that routing and everything beside it: the window's own destination
and encoder settings, the slot and the kind that owns it, the preview parked for the run
and restored after it, the cancellation and failure paths releasing both, session reuse
across runs and the rebuild when geometry or a baked option moves.

The refusals are the load-bearing tests here. Which host drew a frame is invisible in the
finished file, so what a run can be held to is what it says and what it refuses: the
binary and ABI of the session it opened, the entry every frame went through, a session
missing an entry closed and refused with a rebuild as its only remedy, a request naming a
retired kind or a session the export core cannot drive answered before frame zero rather
than by whatever else happens to be on the disk.

Two seams are used deliberately. Most cases run the **real**
`pipeline.export_dlss_stream` against a fake encoder process and fake sessions, so "which
entry answered" is read off the run. A few pin what the window hands that function at all
(provider, lock, CQ, audio, frames, destination) by recording its arguments, because those
are the contract this call site has to preserve. `dlss_gpu_engine.Live` is replaced by a
factory that counts and keeps every session it builds: with one host available, that
count is the record of how many times the window decided to open one.

The window's cached-frame render has its own file (`test_gui_video_render_gpu_host.py`);
the export-only mux that consumes the PNGs that render writes is pinned at the bottom of
this file to show this path does not touch it.
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


def _frames(count=NFRAMES):
    rng = np.random.default_rng(7)
    return [rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8) for _ in range(count)]


class _Base:
    # The pair the window logs when a session is ready, so the run's record names the
    # binary and ABI that answered rather than the caller's intention.
    host_dll = "dlss_gpu_host.dll"
    host_abi = 3

    def __init__(self, w, h, settings=None):
        self._w, self._h = int(w), int(h)
        self.settings = dict(settings or {})
        self.closed = False
        self.last_error = None
        self.reset_flags = []

    def close(self):
        self.closed = True

    def is_alive(self):
        return not self.closed


class GpuLive(_Base):
    """A GPU host carrying both motion entries, as the built ABI-3 DLL does.

    The checks mirror `dlss_gpu_engine.Live`'s own argument validation, including the
    `out=` buffer the export core always passes, so a frame, plane or destination of the
    wrong shape fails here rather than at the ctypes boundary. Every other entry raises:
    they are the wrong doors, and a run that went through one is a failure.
    """

    def __init__(self, w, h, settings=None):
        super().__init__(w, h, settings)
        self.zero_motion_calls = 0
        self.host_motion_calls = 0
        self.seen_rgba = []
        self.seen_motion = []

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

    def _check_out(self, rgba, out):
        assert out is not None, "the export core always passes its workspace buffer"
        assert out.shape == rgba.shape and out.dtype == rgba.dtype
        assert out.flags.c_contiguous

    def _render(self, rgba):
        out = rgba.copy()
        out[..., 2] = np.clip(out[..., 2].astype(np.int16) + 21, 0, 255)
        return out

    def process_zero_motion(self, rgba, reset=False, out=None):
        self._check_rgba(rgba)
        self._check_out(rgba, out)
        self.zero_motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.seen_rgba.append(np.array(rgba))
        np.copyto(out, self._render(rgba))
        return out

    def process_host_motion(self, rgba, motion, reset=False, out=None):
        self._check_rgba(rgba)
        self._check_out(rgba, out)
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
        # A copy: the core hands the same reused workspace buffer every frame, so the
        # only usable record of what this frame carried has to be taken at the call.
        self.seen_motion.append(np.array(motion))
        np.copyto(out, self._render(rgba))
        return out

    def _refuse(self, *args, **kwargs):
        raise AssertionError("a stream export frame reached the wrong GPU host entry")

    process_device_flow = _refuse
    process_device_color_flow = _refuse
    process_device_color_flow_resource = _refuse
    process_device_color_zero_motion_resource = _refuse

    def process(self, *args, **kwargs):
        raise AssertionError(
            "a stream export was driven through a process() entry the GPU host lacks")

    def update(self, settings):
        raise AssertionError(
            "a stream export asked the GPU host for a live update it does not offer")


class NoHostMotionGpuLive(GpuLive):
    """A host built without the host-motion entry: a guided export must be refused."""

    def supports_host_motion(self):
        return False

    def process_host_motion(self, *args, **kwargs):
        raise AssertionError("a host with no host-motion entry was asked to run one")


class Abi2GpuLive(GpuLive):
    """A host that binds no zero motion texture: an unguided export must be refused."""

    def supports_zero_motion(self):
        return False

    def process_zero_motion(self, *args, **kwargs):
        raise AssertionError("a host with no zero-motion entry was asked to run one")


class OldHostShapedLive(_Base):
    """The retired host's shape: a `process()` entry, and no device-flow entry to have.

    It reports the capabilities the window's gate asks a kind for, so that gate cannot be
    what stops it -- the export core is what reads the one host it can drive off a
    session's entry table. Serving a frame to this shape would report a GPU-host run that
    did not happen, and no host in this build takes host pointers through `process()`
    any more.
    """

    host_dll = "dlssnr_host.dll"
    host_abi = 1

    def supports_zero_motion(self):
        return True

    def supports_host_motion(self):
        return True

    def process_zero_motion(self, rgba, reset=False, out=None):
        raise AssertionError("a session the export core must refuse drew a frame")

    def process_host_motion(self, rgba, motion, reset=False, out=None):
        raise AssertionError("a session the export core must refuse drew a frame")

    def process(self, rgba, flow=None, reset=False):
        raise AssertionError("a stream export reached a process() entry no host has")


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


class _Recorder:
    """Counts and keeps every Live a path constructs.

    With one host loadable, this count is the record of every decision the window made to
    open a session -- including the ones it must make without opening any.
    """

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
    """A preview worker that records what the export did to it."""

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


class _Dialogs:
    """messagebox stand-in: records what the operator was told, and never blocks."""

    def __init__(self):
        self.shown = []

    def _show(self, kind, title, message):
        self.shown.append((kind, title, str(message)))
        return True

    def showinfo(self, title, message):
        self._show("info", title, message)

    def showwarning(self, title, message):
        self._show("warning", title, message)

    def showerror(self, title, message):
        self._show("error", title, message)

    def askyesno(self, title, message):
        self._show("ask", title, message)
        return True

    @property
    def kinds(self):
        return [s[0] for s in self.shown]


class _PBar:
    def __init__(self):
        self.resets = 0

    def config(self, **kwargs):
        if kwargs.get("value") == 0:
            self.resets += 1


def _app(monkeypatch, tmp_path, gpu_cls=GpuLive, settings=None, nframes=NFRAMES,
         crf=17, audio=False):
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
    app.pbar = _PBar()
    app.dialogs = _Dialogs()
    monkeypatch.setattr(gui, "messagebox", app.dialogs)
    app.video = str(tmp_path / "clip.mp4")
    app.nframes = nframes
    app.fps = 30.0
    app._ui_queue = queue.Queue()
    # Read off the worker thread, exactly as the panel's cached copies are: the export
    # must use these and not the 24/True defaults of the accessors.
    app._cached_settings = _settings(**(settings or {}))
    app._cached_crf = crf
    app._cached_audio = audio
    app._preview_worker = _Worker()

    app._gpu = _Recorder(gpu_cls)
    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", app._gpu)
    # The window probes the clip for the session geometry; no test needs a real container.
    monkeypatch.setattr(pipeline, "video_info", lambda path: (nframes, 30.0, W, H))
    return app


def _install_fake_pipe(monkeypatch, frames):
    """Let the real export run, over a real pipe, with a decoder and encoder that are not video.

    `build_transcode_command` becomes a Python process that drains stdin into the output
    path, which is what proves the frames the export rendered are the bytes it delivered.
    """
    monkeypatch.setattr(pipeline, "verify_export_output",
                        lambda path, count, **kw: count)

    def _fake_iter(video, **kwargs):
        for idx, fr in enumerate(frames):
            yield idx, fr

    def _fake_command(**kwargs):
        return [sys.executable, "-c",
                "import sys; open(sys.argv[1], 'wb').write(sys.stdin.buffer.read())",
                kwargs["out_path"]]

    monkeypatch.setattr(pipeline, "iter_frames", _fake_iter)
    monkeypatch.setattr(pipeline, "build_transcode_command", _fake_command)


def _expected_out(app):
    """The destination the window has always used: its own default, unasked for."""
    base = os.path.dirname(app.video)
    stem = os.path.splitext(os.path.basename(app.video))[0]
    return os.path.join(base, stem + "_dlss.mkv")


def _flow_dir(app):
    d = pipeline.out_dirs(app.video)[0]
    os.makedirs(d, exist_ok=True)
    return d


def _write_flo(app, frame, plane):
    pipeline.write_flo(os.path.join(_flow_dir(app), f"{frame:06d}.flo"), plane)


def _run_export(app):
    """`_do_dlss_stream_export` off the main thread, the way `_in_thread` runs it."""
    box = {}

    def run():
        try:
            box['value'] = app._do_dlss_stream_export()
        except BaseException as ex:  # noqa: BLE001 - reported, never swallowed
            box['error'] = ex

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(120)
    assert not thread.is_alive(), "the export never returned"
    if 'error' in box:
        raise box['error']
    return box.get('value')


def _real_export(app, monkeypatch, frames=None):
    """Run the whole call site through the real export function."""
    _install_fake_pipe(monkeypatch, frames if frames is not None else _frames())
    _run_export(app)
    return app


def _entry_line(app, kind):
    """The ready line `_open_gpu_live` logs for the session this kind opened."""
    label = gui.App._GPU_HOST_KINDS[kind][2]
    named = [m for m in app.logs if label in m]
    assert named, f"the run never said which host answered {kind}"
    return named[0]


# --- Routing ----------------------------------------------------------------

def test_an_unguided_stream_export_runs_the_gpu_zero_motion_entry(monkeypatch, tmp_path):
    app = _real_export(_app(monkeypatch, tmp_path), monkeypatch)

    assert app._gpu.count == 1, "one host, and the run had no reason to open it twice"
    live = app._gpu.built[0]
    assert (live.zero_motion_calls, live.host_motion_calls) == (NFRAMES, 0)
    assert live.reset_flags == [True, False, False]
    assert app._dlss_session_kind == "export_zero_motion"
    # The destination is the window's own default, and the run says which host answered.
    assert os.path.isfile(_expected_out(app))
    assert os.path.getsize(_expected_out(app)) == NFRAMES * W * H * 3, (
        "the encoder was not fed the rendered frames")
    assert any("DLSS 高速流式导出成功" in m and _expected_out(app) in m for m in app.logs)
    assert any("process_zero_motion" in m and "resident_zero_texture" in m for m in app.logs)
    ready = _entry_line(app, "export_zero_motion")
    assert "dlss_gpu_host.dll" in ready and "abi=3" in ready and f"{W}x{H}" in ready, (
        "the line that names the session says nothing about the binary that answered")
    assert not live.closed, "the run's session is the slot's, not the export's to destroy"


def test_a_guided_stream_export_runs_the_gpu_host_motion_entry(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": 1})
    for idx in range(NFRAMES):
        _write_flo(app, idx, _plane(H, W, idx))

    _real_export(app, monkeypatch)

    assert app._gpu.count == 1
    live = app._gpu.built[0]
    assert (live.host_motion_calls, live.zero_motion_calls) == (NFRAMES, 0)
    assert live.reset_flags == [True, False, False]
    assert app._dlss_session_kind == "export_host_motion"
    assert any("process_host_motion" in m and "host_plane_upload" in m for m in app.logs)
    assert "dlss_gpu_host.dll" in _entry_line(app, "export_host_motion")
    assert any(_expected_out(app) in m for m in app.logs)


def test_the_cached_flo_vectors_are_what_the_host_motion_entry_read(monkeypatch, tmp_path):
    """The plane the host stages is the plane on disk, per frame, not a silent zero."""
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": 1})
    planes = [_plane(H, W, i + 1) for i in range(NFRAMES)]
    for idx, plane in enumerate(planes):
        _write_flo(app, idx, plane)

    _real_export(app, monkeypatch)

    handed = app._gpu.built[0].seen_motion
    assert len(handed) == NFRAMES
    for idx, plane in enumerate(planes):
        got = handed[idx]
        assert got.dtype == np.float32 and got.shape == (H, W, 2)
        assert got.strides == (W * 8, 8, 4)
        assert np.any(got), f"frame {idx} was handed flat motion"
        assert np.array_equal(got, plane), (
            f"frame {idx} bound something other than its own cached .flo")


def test_a_flow_plane_cached_at_another_extent_arrives_at_the_session_size(
        monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": 1})
    for idx in range(NFRAMES):
        _write_flo(app, idx, _plane(H // 2, W // 2, idx))

    _real_export(app, monkeypatch)

    handed = app._gpu.built[0].seen_motion
    assert [g.shape for g in handed] == [(H, W, 2)] * NFRAMES
    assert all(g.strides[0] == W * 8 and np.any(g) for g in handed)


@pytest.mark.parametrize("mode, kind, entry, other", [
    (0, "export_zero_motion", "zero_motion_calls", "host_motion_calls"),
    (1, "export_host_motion", "host_motion_calls", "zero_motion_calls"),
])
def test_the_export_kind_is_read_off_the_guidance_control(monkeypatch, tmp_path, mode,
                                                         kind, entry, other):
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": mode})
    if mode == 1:
        for idx in range(NFRAMES):
            _write_flo(app, idx, _plane(H, W, idx))

    assert gui.App.video_export_kind(_settings(guidance_mode=mode)) == kind
    _real_export(app, monkeypatch)

    live = app._gpu.built[0]
    assert app._dlss_session_kind == kind
    assert getattr(live, entry) == NFRAMES and getattr(live, other) == 0
    assert app._gpu.count == 1, "the other guidance entry was opened as well"


def test_the_export_kinds_are_the_preview_kinds_named_apart_for_the_slot():
    """Same host and same entry, claimed under a name of the export's own.

    The slot is keyed by kind and refuses a request naming a different one, so the two
    names are what keeps a preview frame from resetting a running export's temporal
    sequence -- not two hosts, and not two capabilities. The refusal each asks for still
    names what the operator has to rebuild, and the ready line still says which use the
    session serves.
    """
    for mode, preview_kind, export_kind in ((0, "gpu_zero_motion", "export_zero_motion"),
                                            (1, "gpu_host_motion", "export_host_motion")):
        s = _settings(guidance_mode=mode)
        assert gui.App.video_preview_kind(s) == preview_kind
        assert gui.App.video_export_kind(s) == export_kind
        preview, export = (gui.App._GPU_HOST_KINDS[k] for k in (preview_kind, export_kind))
        assert export[0] == preview[0], "the export asks for another entry than its name"
        assert ("stream-export" in export[1]
                and "rebuild native/dlss_gpu_host.dll" in export[1])
        assert "[导出 DLSS]" in export[2] and "[预览 DLSS]" in preview[2]


def test_an_unguided_export_builds_no_cpu_motion_plane(monkeypatch, tmp_path):
    """The host answers an unguided frame from the zero texture it bound at open, so
    there is no array to fill and nothing to hand over."""
    app = _app(monkeypatch, tmp_path)
    _install_fake_pipe(monkeypatch, _frames())
    seen = []
    real_zeros = np.zeros

    def spy_zeros(shape, *args, **kwargs):
        seen.append((tuple(shape), kwargs.get("dtype", args[0] if args else None)))
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", spy_zeros)
    _run_export(app)

    assert [s for s, dt in seen
            if dt is np.float32 and len(s) == 3 and s[2] == 2] == [], \
        "an unguided export still built a host motion plane"
    assert app._gpu.built[0].zero_motion_calls == NFRAMES
    assert app._gpu.built[0].seen_motion == []


def test_a_frame_missing_its_cached_flow_degrades_and_is_reported(monkeypatch, tmp_path):
    """No .flo for one frame is the documented smooth-degrade case, not a failure."""
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": 1})
    _write_flo(app, 0, _plane(H, W, 0))
    _write_flo(app, 2, _plane(H, W, 2))

    _real_export(app, monkeypatch)

    live = app._gpu.built[0]
    assert live.host_motion_calls == NFRAMES, "the route is not one frame's data"
    assert not np.any(live.seen_motion[1]), "the frame with no cache got something else"
    assert np.any(live.seen_motion[0]) and np.any(live.seen_motion[2])
    assert any("1 帧缺少光流缓存" in m for m in app.logs)
    assert any(_expected_out(app) in m for m in app.logs), "the export still delivered"


def test_the_export_opens_no_session_before_it_owns_the_slot(monkeypatch, tmp_path):
    """The claim is first, so a preview frame cannot lose the run's session to it."""
    app = _app(monkeypatch, tmp_path)
    order = []
    real_get = gui.App._get_or_create_session_locked

    def _watched(self, w, h, settings, kind):
        order.append((self._video_task_kind, kind))
        return real_get(self, w, h, settings, kind)

    monkeypatch.setattr(gui.App, "_get_or_create_session_locked", _watched)
    _real_export(app, monkeypatch)

    assert order == [("export_zero_motion", "export_zero_motion")]
    assert app._preview_worker.events[0] == "pause"


def test_a_second_export_run_reuses_the_session_and_starts_a_new_sequence(
        monkeypatch, tmp_path):
    app = _real_export(_app(monkeypatch, tmp_path), monkeypatch)
    first = app._gpu.built[0]

    _run_export(app)

    assert app._gpu.count == 1, "an unchanged second export reopened the host"
    assert first is app._gpu.built[0]
    assert first.zero_motion_calls == 2 * NFRAMES
    assert first.reset_flags == [True, False, False] * 2, (
        "the second run did not start a new temporal sequence")


def test_a_guidance_flip_between_export_runs_rebuilds_the_session(monkeypatch, tmp_path):
    """The two export kinds are the same host at the same size; only the kind tells them."""
    app = _app(monkeypatch, tmp_path)
    _install_fake_pipe(monkeypatch, _frames())
    for idx in range(NFRAMES):
        _write_flo(app, idx, _plane(H, W, idx))

    _run_export(app)
    unguided = app._gpu.built[0]
    app._cached_settings = _settings(guidance_mode=1)
    _run_export(app)
    guided = app._gpu.built[1]

    assert app._gpu.count == 2 and unguided.closed
    assert (unguided.zero_motion_calls, unguided.host_motion_calls) == (NFRAMES, 0)
    assert (guided.host_motion_calls, guided.zero_motion_calls) == (NFRAMES, 0)
    assert guided.reset_flags == [True, False, False], (
        "a fresh session was handed a run that thought it was mid-sequence")


def test_a_baked_option_change_between_export_runs_rebuilds_the_session(monkeypatch,
                                                                        tmp_path):
    app = _real_export(_app(monkeypatch, tmp_path), monkeypatch)
    first = app._gpu.built[0]

    app._cached_settings = _settings(intensity=0.25)
    _run_export(app)

    assert app._gpu.count == 2, "the GPU host bakes intensity at open; it must reopen"
    assert first.closed


def test_a_session_without_this_request_s_baked_key_is_reopened(monkeypatch, tmp_path):
    """The other half of the reuse gate, and the only half that can be faked by hand.

    The GPU host reads its options at open and offers no live update, so a cached
    session whose key does not match the request is released rather than asked to change.
    """
    app = _app(monkeypatch, tmp_path)
    first = app.get_dlss_session(W, H, _settings(), kind="export_zero_motion")
    assert first._gpu_options_key == gui.dlss_gpu_engine.baked_option_key(_settings())

    first._gpu_options_key = ("stale",)

    live = app.get_dlss_session(W, H, _settings(), kind="export_zero_motion")
    assert live is not first
    assert first.closed, "a session baked with other options was left registered"
    assert app._gpu.count == 2


# --- One host: what a refusal has to say, and what it must not lead to -------

def test_the_build_offers_one_dlss_host_and_refuses_the_retired_name():
    """No second Live is even reachable from the window, so a refusal has nowhere to go.

    Every kind the window can open is an entry of the one host, and `host_pointer` stays
    a name the API and the job boundary refuse out loud, with the migration in the
    message, because the only thing an operator holding a pre-migration script can act on
    is what to change.
    """
    assert pipeline.DLSS_BACKENDS == ("gpu_flow_d3d12",)
    assert pipeline.RETIRED_DLSS_BACKENDS == ("host_pointer",)
    assert set(gui.App._GPU_HOST_KINDS) == {
        "image", "gpu_zero_motion", "gpu_host_motion", "export_zero_motion",
        "export_host_motion", "render_zero_motion", "render_host_motion"}
    assert not hasattr(gui, "dlss_engine"), \
        "the window imports the wrapper for a host this build does not ship"

    refusal = pipeline.retired_dlss_backend_message("host_pointer")
    for named in ("host_pointer", "dlssnr_host.dll", "dlss_engine.py",
                  "native/dlss_gpu_host.dll", "gpu_flow_d3d12"):
        assert named in refusal, f"the refusal an operator acts on omits {named}"


@pytest.mark.parametrize("retired_kind", ["legacy", "host_pointer"])
def test_a_request_for_a_kind_that_is_not_a_gpu_host_entry_opens_nothing(
        retired_kind, monkeypatch, tmp_path):
    """`kind` names an entry, and there is no entry for a host that is not here.

    The retired host had a kind of its own; a caller still asking for it is refused with
    the one host that exists and the entries it serves, not answered by an implicit
    substitution onto it.
    """
    app = _app(monkeypatch, tmp_path)

    with pytest.raises(DLSSInferenceError) as refused:
        app._get_or_create_session_locked(W, H, _settings(), retired_kind)
    message = str(refused.value)
    assert retired_kind in message and "native/dlss_gpu_host.dll" in message
    assert all(kind in message for kind in gui.App._GPU_HOST_KINDS)

    assert app.get_dlss_session(W, H, _settings(), kind=retired_kind) is None
    assert any("native/dlss_gpu_host.dll" in m for m in app.logs)
    assert app._gpu.count == 0, "a request for a retired kind opened a session anyway"
    assert app._dlss_session is None and app._dlss_session_kind is None


def test_a_host_without_the_zero_motion_entry_is_refused_not_replaced(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, gpu_cls=Abi2GpuLive)

    _run_export(app)

    refused = app._gpu.built[0]
    assert refused.closed and refused.zero_motion_calls == 0
    assert app._gpu.count == 1, "an unsupported configuration opened something else"
    assert app._dlss_session is None and app._dlss_session_kind is None
    assert any("ABI 2" in m and "rebuild native/dlss_gpu_host.dll" in m
               and "stream-export" in m for m in app.logs)
    assert any("无法初始化 DLSS 会话" in m for m in app.logs)
    assert app._video_task_kind is None
    assert app._preview_worker.events == ["pause", "resume"]
    assert app._preview_worker.paused is False


def test_a_host_without_the_host_motion_entry_is_refused_not_replaced(monkeypatch,
                                                                      tmp_path):
    app = _app(monkeypatch, tmp_path, gpu_cls=NoHostMotionGpuLive,
               settings={"guidance_mode": 1})
    _write_flo(app, 0, _plane(H, W, 0))

    _run_export(app)

    refused = app._gpu.built[0]
    assert refused.closed and refused.host_motion_calls == 0
    assert app._gpu.count == 1, "a guided export found some other way to draw a frame"
    assert any("host-motion" in m and "rebuild native/dlss_gpu_host.dll" in m
               and "stream-export" in m for m in app.logs)
    assert app._video_task_kind is None
    assert app._preview_worker.events == ["pause", "resume"]


def test_a_session_the_export_core_cannot_drive_is_refused_before_frame_zero(
        monkeypatch, tmp_path):
    """The window's own capability gate cannot see this one, so the core has to.

    A session whose entry table is the retired host's -- `process()` over host pointers,
    no device-flow entry -- is refused by `export_dlss_stream` with both names in the
    message. Drawing the frame anyway would report a GPU-host run that did not happen,
    which is exactly what the picture cannot contradict afterwards.
    """
    app = _app(monkeypatch, tmp_path, gpu_cls=OldHostShapedLive)

    _real_export(app, monkeypatch)

    assert app._gpu.count == 1, "a second host was opened to answer for the refused one"
    errors = [m for m in app.logs if "流式导出错误" in m]
    assert len(errors) == 1, "the refusal the operator has to act on was not reported once"
    assert all(name in errors[0] for name in ("process_device_flow",
                                              "native/dlss_gpu_host.dll",
                                              "dlssnr_host.dll"))
    assert app.dialogs.kinds == ["error"]
    assert not any("导出成功" in m for m in app.logs)
    assert app._video_task_kind is None
    assert app._preview_worker.events == ["pause", "resume"]


def test_a_missing_gpu_host_dll_is_reported_and_not_replaced(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    def _missing_library(w, h, settings=None):
        raise OSError("could not find module 'dlss_gpu_host.dll' (or one of its dependencies)")

    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", _missing_library)

    _run_export(app)

    assert any("dlss_gpu_host.dll" in m for m in app.logs)
    assert any("无法初始化 DLSS 会话" in m for m in app.logs)
    assert app._dlss_session is None
    assert app._video_task_kind is None and app._preview_worker.events[-1] == "resume"


def test_a_frame_the_host_fails_is_reported_and_cleans_up(monkeypatch, tmp_path):
    app = _real_export(_app(monkeypatch, tmp_path, gpu_cls=FailingGpuLive), monkeypatch)

    live = app._gpu.built[0]
    assert live.zero_motion_calls == 1, "a run that failed on frame 0 kept going"
    assert any("zero-motion" in m for m in app.logs)
    assert app.dialogs.kinds == ["error"], "the operator was told twice, or not at all"
    assert "zero-motion" in app.dialogs.shown[0][2]
    assert app._video_task_kind is None
    assert app._preview_worker.events == ["pause", "resume"]
    assert app.pbar.resets == 1
    # The container FFmpeg opened stays where it was created - the pipe owns that file,
    # not this call site - so what the window must not do is call it an export.
    assert not any("导出成功" in m or "已成功完成" in m for m in app.logs)


# --- What the window hands the export ---------------------------------------

def _record_export(app, monkeypatch):
    """Run the call site with `export_dlss_stream` replaced by a recorder."""
    handed = {}

    def _record(video, **kwargs):
        handed["video"] = video
        handed.update(kwargs)
        return None

    monkeypatch.setattr(pipeline, "export_dlss_stream", _record)
    _run_export(app)
    return handed


def test_the_export_gets_the_session_the_window_opened_and_the_lock_that_owns_it(
        monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    handed = _record_export(app, monkeypatch)

    assert handed["live"] is app._dlss_session is app._gpu.built[0]
    assert handed["lock"] is app._dlss_lock
    assert callable(handed["progress"])
    assert callable(handed["cancel"]) and handed["cancel"]() is False
    app._cancel_requested = True
    assert handed["cancel"]() is True, "the export was handed a snapshot, not the Stop flag"


@pytest.mark.parametrize("mode, provider, flow_dir", [
    (0, type(None), None),
    (1, pipeline.CachedFloFlowProvider, "flow"),
])
def test_the_guidance_control_decides_the_provider_and_the_job(monkeypatch, tmp_path,
                                                               mode, provider, flow_dir):
    app = _app(monkeypatch, tmp_path, settings={"guidance_mode": mode})
    handed = _record_export(app, monkeypatch)

    assert isinstance(handed["flow_provider"], provider)
    if mode == 1:
        assert handed["flow_provider"].flow_dir == pipeline.out_dirs(app.video)[0]
    job = handed["job_settings"]
    assert job["guidance_mode"] == mode
    assert job["dlss_backend"] in pipeline.DLSS_BACKENDS, \
        "the window handed a host name this build cannot load"
    # "legacy" here is the export chain's name for the FFmpeg pipe, not a DLSS host; the
    # GPU transfer chain stays opt-in.
    assert job["export_backend"] in pipeline.EXPORT_BACKENDS
    assert not set(job) & set(pipeline.RETIRED_DLSS_BACKENDS)


def test_the_export_keeps_the_panel_s_encoder_settings_and_destination(monkeypatch,
                                                                       tmp_path):
    app = _app(monkeypatch, tmp_path, crf=11, audio=True)
    handed = _record_export(app, monkeypatch)

    assert handed["cq"] == 11 and handed["with_audio"] is True
    assert handed["frames"] == app.nframes and handed["fps"] == app.fps
    # The destination and the chain stay the window's own defaults: nothing here asks for
    # a different file, a different container, or the opt-in GPU transfer backend.
    assert "out_path" not in handed and "transcode_settings" not in handed


def test_the_progress_the_window_shows_is_the_one_the_export_reports(monkeypatch, tmp_path):
    app = _real_export(_app(monkeypatch, tmp_path), monkeypatch)

    assert [(i, t) for i, t, _ in app.progress] == [(1, NFRAMES), (2, NFRAMES),
                                                    (3, NFRAMES)]
    assert all("流式导出" in extra for _, _, extra in app.progress)


# --- The preview, parked and restored ---------------------------------------

def test_the_preview_is_parked_for_the_run_and_cannot_take_the_slot(monkeypatch, tmp_path):
    """Everything a preview could ask for is refused while the export owns the slot."""
    app = _app(monkeypatch, tmp_path)
    seen = {}

    def _record(video, **kwargs):
        live = kwargs["live"]
        seen["during"] = [
            app._get_or_create_session_locked(W, H, _settings(), "gpu_zero_motion"),
            app._get_or_create_session_locked(W, H, _settings(guidance_mode=1),
                                              "gpu_host_motion"),
            app._get_or_create_session_locked(W, H, _settings(), "image"),
        ]
        seen["gpu_built"] = app._gpu.count
        seen["reentry"] = app._get_or_create_session_locked(W, H, _settings(),
                                                           "export_zero_motion")
        seen["live"] = live
        return _expected_out(app)

    monkeypatch.setattr(pipeline, "export_dlss_stream", _record)
    _run_export(app)

    assert seen["during"] == [None, None, None], "a preview took the export's slot"
    assert seen["gpu_built"] == 1, "a refused request opened a session anyway"
    assert seen["reentry"] is seen["live"], "the export's own kind lost its session"
    assert not seen["live"].closed
    assert app._preview_worker.events == ["pause", "resume"]
    assert app._video_task_kind is None


def test_a_preview_frame_after_the_run_replaces_the_export_session(monkeypatch, tmp_path):
    app = _real_export(_app(monkeypatch, tmp_path), monkeypatch)
    exported = app._gpu.built[0]

    preview = app.get_dlss_session(W, H, _settings(), kind="gpu_zero_motion")

    assert exported.closed, "the slot handed to the preview left the export session open"
    assert preview is app._gpu.built[1] and preview is not exported
    assert app._dlss_session_kind == "gpu_zero_motion"


def test_an_image_batch_in_progress_holds_the_slot_and_the_export_reports(
        monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    with app._dlss_lock:
        app._image_batch_running = True

    _run_export(app)

    assert app._gpu.count == 0, "the export evicted the session the batch is running on"
    assert any("无法初始化 DLSS 会话" in m for m in app.logs)
    assert app._video_task_kind is None and app._preview_worker.events[-1] == "resume"


# --- Cancellation ------------------------------------------------------------

def test_a_cancelled_export_is_reported_as_an_abort_and_leaves_nothing_behind(
        monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    lives = []

    def _cancel_inside(video, **kwargs):
        lives.append(kwargs["live"])
        assert kwargs["cancel"]() is True
        raise pipeline.JobCancelled("用户已取消任务")

    monkeypatch.setattr(pipeline, "export_dlss_stream", _cancel_inside)
    app._cancel_requested = True

    _run_export(app)

    assert any("已由用户中止" in m for m in app.logs)
    assert "导出已中止" in app.statuses
    assert app.dialogs.kinds == [], "a user's own Stop click reported a failure"
    assert app._video_task_kind is None
    assert app._preview_worker.events == ["pause", "resume"]
    assert app.pbar.resets == 1
    assert not lives[0].closed


def test_a_cancel_mid_run_stops_the_real_export_and_reports_an_abort(monkeypatch,
                                                                     tmp_path):
    """The Stop flag reaches the pipe through the callback the window handed it.

    The press is modelled where it lands in practice — after a frame has been rendered —
    so the run stops on the next loop turn instead of at a point this file chooses.
    """
    class CancelAfterFirstFrame(GpuLive):
        def process_zero_motion(self, rgba, reset=False, out=None):
            out_result = super().process_zero_motion(rgba, reset=reset, out=out)
            app._cancel_requested = True
            return out_result

    app = _app(monkeypatch, tmp_path, gpu_cls=CancelAfterFirstFrame)
    _install_fake_pipe(monkeypatch, _frames())

    _run_export(app)

    assert app._gpu.built[0].zero_motion_calls == 1, "the run did not stop when asked"
    assert "导出已中止" in app.statuses
    assert app.dialogs.kinds == [], "a Stop click was reported as an export error"
    assert app._video_task_kind is None and app._preview_worker.events[-1] == "resume"
    assert app.pbar.resets == 1


# --- The slot, shared with the render that names it apart for the same reason --

@pytest.mark.parametrize("owner, other", [
    ("render_zero_motion", "export_zero_motion"),
    ("export_zero_motion", "render_zero_motion"),
])
def test_whichever_video_task_owns_the_slot_refuses_the_other(monkeypatch, tmp_path,
                                                             owner, other):
    """The guard is symmetric, and it runs between entries of one host.

    The render and the export use the same two entries at the same extent, so the only
    thing that keeps one from driving the other's session is the name each claims the
    slot under before it opens anything.
    """
    app = _app(monkeypatch, tmp_path)
    live = app.get_dlss_session(W, H, _settings(), kind=owner)
    with app._dlss_lock:
        app._video_task_kind = owner

    got = app._get_or_create_session_locked(W, H, _settings(), other)

    assert got is None and app._gpu.count == 1
    assert live is app._gpu.built[0] and not live.closed


def test_the_export_only_mux_builds_no_session_at_all(monkeypatch, tmp_path):
    """The cached-frame export path is untouched: it encodes PNGs and renders nothing."""
    app = _app(monkeypatch, tmp_path)
    handed = {}

    def _record(video, kind, frames, fps, **kwargs):
        handed.update(video=video, kind=kind, frames=frames, fps=fps, **kwargs)
        return str(tmp_path / "muxed.mkv")

    monkeypatch.setattr(pipeline, "export_video", _record)

    app._do_export_only("dlss", "DLSS")

    assert handed["kind"] == "dlss" and handed["frames"] == app.nframes
    assert handed["crf"] == app._cached_crf and handed["with_audio"] == app._cached_audio
    assert app._gpu.count == 0
    assert app._dlss_session is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
