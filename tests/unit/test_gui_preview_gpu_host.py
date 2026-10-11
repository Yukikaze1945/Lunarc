"""Video preview frames on the project GPU host, checked without a GPU.

This window has one DLSS host: `native/dlss_gpu_host.dll`, wrapped by
`dlss_gpu_engine.Live` and selected by the kind a caller hands `App.get_dlss_session`.
It answers a video preview frame in either of the two shapes guidance leaves it -
`process_zero_motion` when the guidance control is off, `process_host_motion` when it is
on - so the preview routes by guidance mode with no selector beyond that kind, and a CPU
array of zeros exists nowhere in the path. These tests pin the routing and everything
that has to hold for a preview to be worth showing: the cached .flo plane reaching the
host untouched and at the session extent, the seek reset, reuse while the geometry and the
host's create-time options hold and a rebuild when either moves, the preview cache
invalidating on a settings change, the single lock hold that keeps another route from
closing the session between lookup and call, and that a running image batch or video
render is never evicted from the slot by a preview frame.

One host also decides what a refusal means. A loaded host that does not bind the entry a
kind runs on is closed before it draws a frame and the operator is told to rebuild it,
and a caller naming a kind outside `App._GPU_HOST_KINDS` is turned away at the factory
with the entries that do exist - there is no second host to draw the picture behind the
caller's back, which matters because a picture never says which host produced it. So
every test here runs with a trap standing where a second host wrapper would be
importable, and every fake session raises on each entry its host does not serve.

Everything runs against fake sessions injected in place of the one engine. The still-image
path is pinned here only to show it shares that factory and that slot; the cached-frame
render and the streaming export have their own files (`test_gui_video_render_gpu_host.py`,
`test_gui_stream_export_gpu_host.py`).
"""

import os
import queue
import sys
import threading
import time
import types

import cv2
import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dlss_gpu_engine  # noqa: E402
import gui  # noqa: E402
import pipeline  # noqa: E402
from dlss_errors import DLSSInferenceError  # noqa: E402

W, H = 64, 32
NFRAMES = 8

# Every test below replaces dlss_gpu_engine.Live with a recorder, so the class the table
# of kinds is checked against has to be held before that happens.
_LIVE_WRAPPER = dlss_gpu_engine.Live


class SecondHostBuilt(BaseException):
    """Raised instead of letting a frame reach a second DLSS host wrapper.

    Deliberately not an Exception: the preview worker turns any Exception from a frame
    into a reported failure and keeps going, so an Exception trap here would be caught by
    the very code under test and read as a preview that merely degraded.
    """


def _settings(**over):
    """The dict _collect_settings produces; the window's guidance default is off."""
    s = {'preset': 1, 'style': 0, 'intensity': 1.0, 'local_tone': 1.0,
         'local_struct': 1.0, 'skin_struct': 1.0, 'use_auto_mask': 1,
         'ui_correction': 0, 'guidance_mode': 0,
         'motion_scale_x': 1.0, 'motion_scale_y': 1.0}
    s.update(over)
    return s


def _frame(w=W, h=H, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _plane(h, w, seed):
    """A non-flat float32 HxWx2 motion plane, distinct per frame and per extent."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.ascontiguousarray(np.stack([x / 97.0 + seed, y / 61.0 - seed], axis=-1),
                                dtype=np.float32)


class _Base:
    def __init__(self, w, h, settings=None):
        self._w, self._h = int(w), int(h)
        self.settings = dict(settings or {})
        self.closed = False
        self.last_error = None
        # The real wrapper reads both off the library that answered, and the ready line
        # the window logs quotes them: a picture does not say which host drew it.
        self.host_dll = "dlss_gpu_host.dll"
        self.host_abi = 3

    def close(self):
        self.closed = True

    def is_alive(self):
        return not self.closed


class GpuLive(_Base):
    """A GPU host carrying both motion entries, as the built ABI-3 DLL does.

    The checks mirror dlss_gpu_engine.Live's own argument validation, so a frame or
    plane handed over at the wrong size, dtype or pitch fails here rather than at the
    ctypes boundary. The traps are for entries this configuration must not reach.
    """

    def __init__(self, w, h, settings=None):
        super().__init__(w, h, settings)
        self.zero_motion_calls = 0
        self.host_motion_calls = 0
        self.reset_flags = []
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

    def _enhance(self, rgba):
        out = rgba.copy()
        out[..., 0] = np.clip(out[..., 0].astype(np.int16) + 33, 0, 255)
        return out

    def process_zero_motion(self, rgba, reset=False, out=None):
        self._check_rgba(rgba)
        self.zero_motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.seen_rgba.append(rgba.copy())
        return self._enhance(rgba)

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
        self.seen_rgba.append(rgba.copy())
        # Held by reference: a worker that re-made, scaled or zeroed the plane would
        # hand over a different object than the guidance cache produced.
        self.seen_motion.append(motion)
        return self._enhance(rgba)

    def process(self, *args, **kwargs):
        raise AssertionError("the one host was driven through a motion call it has no "
                             "entry for")

    def update(self, settings):
        raise AssertionError("GPU session handed the live-update entry it lacks")

    def _refuse_device_entry(self, *args, **kwargs):
        raise AssertionError("a host-memory preview frame reached a CUDA-descriptor entry")

    process_device_flow = _refuse_device_entry
    process_device_color_flow = _refuse_device_entry
    process_device_color_flow_resource = _refuse_device_entry
    process_device_color_zero_motion_resource = _refuse_device_entry


class NoHostMotionGpuLive(GpuLive):
    """A host built without the host-motion entry: a guided preview must be refused."""

    def supports_host_motion(self):
        return False

    def process_host_motion(self, *args, **kwargs):
        raise AssertionError("a host with no host-motion entry was asked to run one")


class Abi2GpuLive(GpuLive):
    """A host that binds no zero motion texture: an unguided preview must be refused."""

    host_abi = 2

    def supports_zero_motion(self):
        return False

    def process_zero_motion(self, *args, **kwargs):
        raise AssertionError("a host with no zero-motion entry was asked to run one")


class FailingGpuLive(GpuLive):
    """A host that has the entries and fails every frame, as the wrapper raises."""

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


class FailsOnceGpuLive(GpuLive):
    """Fails the first guided frame only: a later one has to get through."""

    def process_host_motion(self, rgba, motion, reset=False, out=None):
        if self.host_motion_calls == 0:
            self.host_motion_calls += 1
            self.reset_flags.append(bool(reset))
            self.last_error = "host-motion DLSS frame failed (5): fake card error"
            raise DLSSInferenceError(self.last_error)
        return super().process_host_motion(rgba, motion, reset=reset, out=out)


class _Recorder:
    """Counts and keeps every Live a path constructs."""

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


def _trap_second_host(monkeypatch):
    """Trap any attribute of a second host wrapper, for the length of one test.

    Importing the module by name is how an old route would get hold of it, so the name
    itself is replaced by a module whose every attribute aborts the test. Dunders stay
    ordinary AttributeErrors: the introspection every traceback walk performs asks for
    `__file__` and friends on each module it passes.
    """
    module = types.ModuleType("dlss_engine")

    def _reached(attribute):
        if attribute.startswith("__") and attribute.endswith("__"):
            raise AttributeError(attribute)
        raise SecondHostBuilt(
            f"a frame reached dlss_engine.{attribute}; this build ships one DLSS host")

    module.__getattr__ = _reached
    monkeypatch.setitem(sys.modules, "dlss_engine", module)


class _Cap:
    """A cv2.VideoCapture stand-in that serves the frames a clip would hold."""

    def __init__(self, frames):
        self.frames = frames
        self.seeked = []

    def set(self, prop, value):
        assert prop == cv2.CAP_PROP_POS_FRAMES
        self.seeked.append(int(value))
        return True

    def read(self):
        frame = self.frames.get(self.seeked[-1])
        return (True, frame.copy()) if frame is not None else (False, None)


def _app(monkeypatch, tmp_path, gpu_cls=GpuLive, frames=None,
         nframes=NFRAMES, threaded=False):
    """A real App and a real preview worker, with no Tk root.

    Unless `threaded` asks for the live thread, only Thread.start is neutralised: the
    worker's own state, cache keys and frame function are the ones the GUI runs. Every
    app here also carries the second-host trap, so a frame in any test below can only be
    drawn by the recorder standing in for the GPU host.
    """
    app = gui.App.__new__(gui.App)
    app._dlss_lock = threading.Lock()
    app._dlss_session = None
    app._dlss_session_kind = None
    app._image_batch_running = False
    app._video_task_kind = None
    app._cancel_requested = False
    app.current_is_image = False
    app.logs, app.statuses = [], []
    app.logln = lambda msg: app.logs.append(str(msg))
    app.set_status = lambda msg: app.statuses.append(str(msg))
    app.set_progress = lambda i, total, extra="": None
    app.call_ui = lambda fn, *a, **k: fn(*a, **k)
    app._cached_settings = _settings()
    app.video = str(tmp_path / "clip.mp4")
    app.nframes = nframes
    app.fps = 30.0
    app._ui_queue = queue.Queue()
    app._cap = _Cap(frames if frames is not None
                    else {i: _frame(seed=i) for i in range(nframes)})

    app._gpu = _Recorder(gpu_cls)
    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", app._gpu)
    _trap_second_host(monkeypatch)
    if threaded:
        app._worker = gui._PreviewWorker(app)
    else:
        # Park the thread for construction only, so a test that needs a real worker
        # thread still gets one when it asks for it.
        with monkeypatch.context() as parked:
            parked.setattr(threading.Thread, "start", lambda self: None)
            app._worker = gui._PreviewWorker(app)
    return app


def _preview(app, frame, settings):
    """The picture the DLSS view would show for one frame, straight from the worker."""
    return app._worker._live_dlss_image(frame, settings)


def _off(app, frame, settings=None):
    return _preview(app, frame, settings or _settings())


def _flow_dir(app, tmp_path):
    d = pipeline.out_dirs(app.video)[0]
    os.makedirs(d, exist_ok=True)
    return d


def _write_flo(app, tmp_path, frame, plane):
    pipeline.write_flo(os.path.join(_flow_dir(app, tmp_path), f"{frame:06d}.flo"), plane)


def _session(app, kind, settings=None):
    with app._dlss_lock:
        return app._get_or_create_session_locked(W, H, settings or _settings(), kind)


# --- One host, and what a refusal has to say -------------------------------

def test_the_window_ships_one_host_and_one_way_to_name_a_session():
    """What one DLSS host has to mean for every route this file drives.

    `App._GPU_HOST_KINDS` is the only selector left, so the retired session name cannot
    resolve to anything and there is no wrapper module for gui to route a frame through.
    The fakes here trap a plain `process` and a live `update`, which is only a guarantee
    while the one host's wrapper really lacks both: it reads its options at open and
    serves motion through named entries.
    """
    assert not hasattr(gui, "dlss_engine")
    assert "legacy" not in gui.App._GPU_HOST_KINDS
    assert set(gui.App._GPU_HOST_KINDS) == {
        "image", "gpu_zero_motion", "gpu_host_motion", "export_zero_motion",
        "export_host_motion", "render_zero_motion", "render_host_motion"}
    assert not hasattr(_LIVE_WRAPPER, "process")
    assert not hasattr(_LIVE_WRAPPER, "update")


def test_the_second_host_trap_every_test_here_runs_under(monkeypatch, tmp_path):
    """The trap is only evidence while it can fire, so it is proved before trusted.

    `_app` installs it for every test in this file, which is what lets a routing or a
    refusal test say that nothing but the GPU host answered a frame instead of counting
    calls on a mock of a host this build does not ship. An `import` of that module name
    resolves to this same sys.modules entry, so the attribute access below is the reach
    being tested, and it stays a trap once the file behind the name is gone.
    """
    _app(monkeypatch, tmp_path)

    with pytest.raises(SecondHostBuilt):
        sys.modules["dlss_engine"].Live


@pytest.mark.parametrize("kind", sorted(gui.App._GPU_HOST_KINDS))
def test_every_kind_is_answered_by_the_one_host_and_logged_as_such(monkeypatch, tmp_path,
                                                                   kind):
    """Every row of the selector has to be honest about the host it names.

    A row gating on a capability the wrapper does not have opens a session the caller
    cannot drive; a row whose refusal does not order a rebuild leaves the operator with
    nothing to do about a frame that was never drawn; and the ready line is the only
    record of which host answered, since the picture itself cannot say.
    """
    app = _app(monkeypatch, tmp_path)
    capability, refusal, ready = gui.App._GPU_HOST_KINDS[kind]

    assert hasattr(_LIVE_WRAPPER, capability), (
        f"kind {kind!r} gates on {capability}(), which the one host does not expose")
    assert "rebuild native/dlss_gpu_host.dll" in refusal
    assert capability in ("supports_zero_motion", "supports_host_motion")

    live = app.get_dlss_session(W, H, _settings(), kind)

    assert live is not None, app.logs
    assert app._gpu.count == 1 and app._dlss_session_kind == kind
    assert live._gpu_options_key == dlss_gpu_engine.baked_option_key(_settings())
    assert f"{ready}: dlss_gpu_host.dll abi=3 @ {W}x{H}" in app.logs


def test_a_caller_that_names_the_retired_kind_is_turned_away_at_the_factory(
        monkeypatch, tmp_path):
    """`legacy` was a second host's own session name; no kind resolves to another DLL.

    The factory is where a caller says what it needs, so the refusal belongs there, and it
    has to list the entries that do exist: the only thing a caller still holding the old
    kind can act on is the difference.
    """
    app = _app(monkeypatch, tmp_path)

    with pytest.raises(DLSSInferenceError) as refused:
        _session(app, "legacy")

    message = str(refused.value)
    assert "未知 DLSS 会话类型 kind='legacy'" in message
    assert "native/dlss_gpu_host.dll" in message
    assert ", ".join(gui.App._GPU_HOST_KINDS) in message

    assert app.get_dlss_session(W, H, _settings(), "legacy") is None
    assert any(m.startswith("[DLSS 引擎] ") and "kind='legacy'" in m for m in app.logs)
    assert app._gpu.count == 0 and app._dlss_session is None, \
        "a refused kind still opened the one host"


# --- Routing ---------------------------------------------------------------

def test_unguided_preview_runs_on_the_gpu_zero_motion_entry(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    out = _off(app, 0)

    assert out is not None and out.shape == (H, W, 3) and out.dtype == np.uint8
    assert app._dlss_session_kind == "gpu_zero_motion"
    live = app._gpu.built[0]
    assert app._gpu.count == 1
    assert live.zero_motion_calls == 1 and live.host_motion_calls == 0


def test_guided_preview_runs_on_the_gpu_host_motion_entry(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    _write_flo(app, tmp_path, 2, _plane(H, W, 2))

    out = _preview(app, 2, _settings(guidance_mode=1))

    assert out is not None
    assert app._dlss_session_kind == "gpu_host_motion"
    live = app._gpu.built[0]
    assert app._gpu.count == 1
    assert live.host_motion_calls == 1 and live.zero_motion_calls == 0


@pytest.mark.parametrize("mode, kind", [(0, "gpu_zero_motion"), (1, "gpu_host_motion")])
def test_the_preview_kind_is_read_off_guidance_mode(monkeypatch, tmp_path, mode, kind):
    app = _app(monkeypatch, tmp_path)
    assert gui.App.video_preview_kind(_settings(guidance_mode=mode)) == kind
    if kind == "gpu_host_motion":
        _write_flo(app, tmp_path, 0, _plane(H, W, 0))
    assert _off(app, 0, _settings(guidance_mode=mode)) is not None
    assert app._dlss_session_kind == kind


def test_unguided_preview_allocates_no_cpu_zero_flow_plane(monkeypatch, tmp_path):
    """The host's zero-motion texture is the whole answer: there is nothing to fill."""
    app = _app(monkeypatch, tmp_path)
    seen = []
    real_zeros = np.zeros

    def spy_zeros(shape, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        seen.append((tuple(shape), dtype))
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", spy_zeros)
    assert _off(app, 0) is not None

    assert [s for s, dt in seen
            if dt is np.float32 and len(s) == 3 and s[2] == 2] == [], \
        "an unguided preview still built a host motion plane"
    assert app._gpu.built[0].zero_motion_calls == 1


def test_the_cached_flo_plane_reaches_the_host_untouched(monkeypatch, tmp_path):
    """The plane read_flo returns is the plane the host binds, object for object.

    read_flo makes a fresh array off disk, so identity is pinned against what that
    reader gave the worker, not against the array the writer was handed.
    """
    app = _app(monkeypatch, tmp_path)
    plane = _plane(H, W, 5)
    _write_flo(app, tmp_path, 5, plane)
    from_reader = {}

    def _spy_read(path):
        from_reader["path"] = path
        return plane

    monkeypatch.setattr(gui.pipeline, "read_flo", _spy_read)

    assert _preview(app, 5, _settings(guidance_mode=1)) is not None

    assert from_reader["path"].endswith("000005.flo")
    handed = app._gpu.built[0].seen_motion[0]
    assert handed is plane, "the worker re-made the plane instead of passing it"
    assert handed.dtype == np.float32 and handed.shape == (H, W, 2)
    assert handed.flags.c_contiguous and handed.strides[0] == W * 8
    assert np.any(handed), "a guided frame was handed flat motion"
    assert np.array_equal(handed, _plane(H, W, 5))


def test_a_flow_plane_cached_at_another_size_is_scaled_to_the_session(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    _write_flo(app, tmp_path, 1, _plane(H // 2, W // 2, 1))

    assert _preview(app, 1, _settings(guidance_mode=1)) is not None

    handed = app._gpu.built[0].seen_motion[0]
    assert handed.shape == (H, W, 2) and handed.dtype == np.float32
    assert handed.flags.c_contiguous and handed.strides[0] == W * 8
    assert np.any(handed)


def test_a_frame_the_source_cannot_supply_opens_no_session(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, frames={0: _frame(seed=0)})

    assert _off(app, 4) is None
    assert app._gpu.count == 0


def test_missing_flow_cache_degrades_to_the_hint_not_a_failure(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    assert _preview(app, 3, _settings(guidance_mode=1)) is None
    assert app._worker.last_error is None, "an absent .flo is the hint case, not a failure"
    assert app._gpu.count == 0


# --- The picture the canvas gets -------------------------------------------

def test_preview_output_keeps_bgr_channel_order(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    frame = _frame(seed=8)
    app._cap = _Cap({8: frame})

    out = _off(app, 8)

    # The fake lifts the RGBA red channel, which is BGR's third plane: a conversion
    # that went the other way would move blue instead.
    assert np.array_equal(out[..., 2],
                          np.clip(frame[..., 2].astype(np.int16) + 33, 0, 255))
    assert np.array_equal(out[..., 0], frame[..., 0])
    assert np.array_equal(out[..., 1], frame[..., 1])


def test_the_frame_reaches_the_host_as_opaque_contiguous_rgba(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    frame = _frame(seed=9)
    app._cap = _Cap({9: frame})

    _off(app, 9)

    rgba = app._gpu.built[0].seen_rgba[0]
    assert rgba.shape == (H, W, 4) and rgba.dtype == np.uint8
    assert np.array_equal(rgba[..., :3], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    assert np.all(rgba[..., 3] == 255)


# --- Reuse, rebuild and reset ----------------------------------------------

def test_session_is_reused_across_a_sequence_and_reset_on_seek(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    for idx in (0, 1, 2, 0, 4):
        assert _off(app, idx) is not None

    assert app._gpu.count == 1, "session rebuilt during steady playback"
    live = app._gpu.built[0]
    assert live.zero_motion_calls == 5
    # Only a frame that follows its predecessor on the same session continues the
    # sequence: the seek back to 0 resets, and 4 after 0 is a jump, not a continuation.
    assert live.reset_flags == [True, False, False, True, True], "seek must reset"
    assert not live.closed


def test_a_session_another_route_rebuilt_is_reset_not_continued(monkeypatch, tmp_path):
    """Temporal state died with the old session, so the next frame starts over."""
    app = _app(monkeypatch, tmp_path)
    _off(app, 0)
    _off(app, 1)
    first = app._gpu.built[0]
    assert first.reset_flags == [True, False]

    app.close_dlss_session()
    assert _off(app, 2) is not None

    assert app._gpu.count == 2
    assert app._gpu.built[1].reset_flags == [True], "a fresh session inherited a sequence"
    assert first.closed


def test_baked_option_change_rebuilds_the_preview_session(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    _off(app, 0)
    first = app._gpu.built[0]

    assert _off(app, 1, _settings(intensity=0.25)) is not None

    assert app._gpu.count == 2, "the GPU host bakes intensity at open; it must reopen"
    assert first.closed
    assert app._gpu.built[1].reset_flags == [True]


def test_a_stale_baked_option_key_rebuilds_rather_than_reusing(monkeypatch, tmp_path):
    """The reuse gate is the key the host was opened with, not the frame's numbers.

    With no live-update entry, a session carrying someone else's option tuple cannot be
    corrected - only released - so the key has to be stamped at open and compared as a
    whole.
    """
    app = _app(monkeypatch, tmp_path)
    live = _session(app, "gpu_zero_motion")
    assert live._gpu_options_key == dlss_gpu_engine.baked_option_key(_settings())

    assert _session(app, "gpu_zero_motion") is live
    assert app._gpu.count == 1

    live._gpu_options_key = ("stale",)
    replaced = _session(app, "gpu_zero_motion")

    assert replaced is not live
    assert live.closed, "a session baked with other options was left open"
    assert app._gpu.count == 2


def test_frame_size_change_rebuilds_at_the_new_size(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, frames={0: _frame(W, H, seed=0),
                                              1: _frame(80, 48, seed=1)})
    _off(app, 0)
    first = app._gpu.built[0]

    assert _off(app, 1) is not None

    assert (first._w, first._h) == (W, H)
    assert (app._gpu.built[1]._w, app._gpu.built[1]._h) == (80, 48)
    assert first.closed


def test_guidance_flip_rebuilds_and_never_reuses_across_entries(monkeypatch, tmp_path):
    """The two kinds share a session slot, so they have to be told apart.

    A guided and an unguided session are the same host at the same size with the same
    baked options; only the kind says which entry point is legal.
    """
    app = _app(monkeypatch, tmp_path)
    _write_flo(app, tmp_path, 0, _plane(H, W, 0))
    _write_flo(app, tmp_path, 2, _plane(H, W, 2))

    assert _preview(app, 0, _settings(guidance_mode=1)) is not None
    guided = app._gpu.built[0]
    assert _off(app, 1) is not None
    unguided = app._gpu.built[1]

    assert guided.closed, "the guided session must go when guidance turns off"
    assert (guided.host_motion_calls, guided.zero_motion_calls) == (1, 0)
    assert (unguided.host_motion_calls, unguided.zero_motion_calls) == (0, 1)

    assert _preview(app, 2, _settings(guidance_mode=1)) is not None
    assert app._gpu.count == 3 and unguided.closed


def test_the_preview_cache_invalidates_on_a_refresh_or_a_settings_change(
        monkeypatch, tmp_path):
    """_refresh_dlss clears the cache on every settings move; the key covers the rest."""
    app = _app(monkeypatch, tmp_path)
    first = _off(app, 0)
    live = app._gpu.built[0]

    assert _off(app, 0) is first, "an identical request redrew the frame"
    assert live.zero_motion_calls == 1

    app._worker.close_live()
    assert _off(app, 0) is not None
    assert live.zero_motion_calls == 2, "close_live did not invalidate the preview cache"
    assert app._gpu.count == 1, "a cache clear alone must not reopen the host"

    assert _off(app, 0, _settings(style=1)) is not None
    assert app._gpu.count == 2, "a moved baked option has to reopen the host"


# --- A host missing an entry, and a frame that fails ------------------------

def test_a_host_without_the_entry_is_refused_and_told_to_rebuild(monkeypatch, tmp_path):
    """Nothing else draws a guided frame, so the refusal is the whole answer.

    The remedy for a missing entry point is a rebuild of the one host: the preview says
    which entry was missing, quotes the refusal on the log, and leaves the frame undrawn.
    """
    app = _app(monkeypatch, tmp_path, gpu_cls=NoHostMotionGpuLive)
    refusal = gui.App._GPU_HOST_KINDS["gpu_host_motion"][1]
    assert "no host-motion entry point" in refusal
    assert "rebuild native/dlss_gpu_host.dll" in refusal
    _write_flo(app, tmp_path, 0, _plane(H, W, 0))

    assert _preview(app, 0, _settings(guidance_mode=1)) is None

    refused = app._gpu.built[0]
    assert refused.closed and refused.host_motion_calls == 0
    assert app._worker.last_error == refusal
    assert f"[预览 DLSS] {refusal}" in app.logs
    assert app._dlss_session is None, "the refused session stayed in the slot"


def test_an_abi2_host_is_refused_for_an_unguided_preview(monkeypatch, tmp_path):
    """A host that binds no zero motion texture cannot serve guidance off either."""
    app = _app(monkeypatch, tmp_path, gpu_cls=Abi2GpuLive)

    assert _off(app, 0) is None

    refusal = gui.App._GPU_HOST_KINDS["gpu_zero_motion"][1]
    assert "no zero-motion entry point (ABI 2)" in refusal
    assert app._gpu.built[0].closed and app._gpu.built[0].zero_motion_calls == 0
    assert app._worker.last_error == refusal
    assert any("ABI 2" in m for m in app.logs)
    assert app._dlss_session is None


def test_a_missing_gpu_host_dll_is_reported_and_no_frame_is_drawn(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    def _missing_library(w, h, settings=None):
        raise OSError("could not find module 'dlss_gpu_host.dll' (or one of its dependencies)")

    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", _missing_library)

    assert _off(app, 0) is None
    assert any("dlss_gpu_host.dll" in m for m in app.logs)
    assert app._gpu.count == 0 and app._dlss_session is None


def test_a_failed_frame_is_reported_on_the_canvas(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, gpu_cls=FailsOnceGpuLive)
    settings = _settings(guidance_mode=1)
    _write_flo(app, tmp_path, 2, _plane(H, W, 2))
    _write_flo(app, tmp_path, 3, _plane(H, W, 3))

    # The failing frame goes through _render, not a separate lookup first: it is the
    # render that reports, and a retry before it would consume the one failure.
    res = app._worker._render(2, "DLSS", 0.5, 400, 200, settings)
    live = app._gpu.built[0]
    assert live.host_motion_calls == 1
    assert app._worker.last_error and "host-motion" in app._worker.last_error
    assert res["type"] == "msg" and "失败" in res["text"] and "host-motion" in res["text"], \
        "the canvas said nothing about the frame that failed"
    assert any("host-motion" in m for m in app.logs), "the failure never reached the log"

    assert _preview(app, 3, settings) is not None
    assert app._worker.last_error is None, "an old failure outlived the frame that failed"


def test_a_frame_failure_clears_the_cache_and_forces_a_reset(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path, gpu_cls=FailingGpuLive)
    assert _off(app, 0) is None

    assert _off(app, 1) is None
    assert app._gpu.built[0].reset_flags == [True, True], \
        "the retry continued a sequence whose first frame never rendered"


# --- One slot, several callers: who may evict whom -------------------------

def test_the_lookup_and_the_frame_call_share_one_lock_hold(monkeypatch, tmp_path):
    """Splitting the hold is the race the preview cannot afford.

    The host refuses a second open session, so another route that takes the slot closes
    this one: a frame called after the lookup released the lock can be handed a session
    that no longer exists.
    """
    app = _app(monkeypatch, tmp_path)
    held = []

    class _Watching(GpuLive):
        def process_zero_motion(self, rgba, reset=False, out=None):
            held.append(app._dlss_lock.locked())
            return super().process_zero_motion(rgba, reset=reset, out=out)

    app._gpu = _Recorder(_Watching)
    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", app._gpu)

    assert _off(app, 0) is not None
    assert held == [True], "the frame was drawn outside the hold its lookup took"


def test_an_image_batch_owns_the_slot_and_the_preview_drops_its_frame(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    _off(app, 0)
    preview_session = app._gpu.built[0]

    with app._dlss_lock:
        app._image_batch_running = True
    assert _off(app, 1) is None

    assert app._gpu.count == 1 and not preview_session.closed

    with app._dlss_lock:
        app._image_batch_running = False
    assert _off(app, 1) is not None


def test_a_video_render_owns_the_slot_and_the_preview_stays_out_of_it(monkeypatch,
                                                                      tmp_path):
    """The render runs on the same host as the preview; the one running holds the slot.

    A preview frame that named the render's kind would be handed the very session the
    render is drawing through and reset its temporal sequence mid-run; a frame that named
    the preview's own kind would close that session underneath it. Either way the preview
    loses a frame, which is the cheaper answer.
    """
    app = _app(monkeypatch, tmp_path)
    live = app.get_dlss_session(W, H, _settings(), "render_zero_motion")
    with app._dlss_lock:
        app._video_task_kind = "render_zero_motion"

    assert _off(app, 1) is None
    assert app._gpu.count == 1, "the preview took the slot back mid-render"
    assert not live.closed

    with app._dlss_lock:
        app._video_task_kind = None
    assert _off(app, 1) is not None
    assert live.closed, "the render's session goes once it no longer owns the slot"


def test_the_still_image_path_runs_on_the_zero_motion_entry(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)

    out = _worker_thread(lambda: app._image_dlss(_frame(W, H, seed=6)))

    assert out is not None and out.shape == (H, W, 3)
    assert app._dlss_session_kind == "image"
    assert app._gpu.count == 1
    assert app._gpu.built[0].zero_motion_calls == 1
    assert app._gpu.built[0].reset_flags == [True]


def test_close_dlss_session_releases_a_preview_session_without_leaking(monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    _off(app, 0)
    live = app._gpu.built[0]

    app.close_dlss_session()

    assert live.closed
    assert app._dlss_session is None and app._dlss_session_kind is None
    app.close_dlss_session()  # idempotent teardown on the way out


def test_an_unguided_preview_never_enters_the_optical_flow_module(monkeypatch, tmp_path):
    """No flow is computed, requested or imported for a guidance-off frame."""
    app = _app(monkeypatch, tmp_path)
    attempts = []

    def _module_getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        attempts.append(name)
        raise AssertionError(f"the preview entered optical_flow.{name}")

    stub = types.ModuleType("optical_flow")
    stub.__getattr__ = _module_getattr

    class _ImportCounter:
        def __init__(self, sink):
            self._sink = sink

        def find_spec(self, name, path=None, target=None):
            if name == "optical_flow" or name.startswith("optical_flow."):
                self._sink.append(f"import:{name}")
            return None

    counter = _ImportCounter(attempts)
    monkeypatch.setitem(sys.modules, "optical_flow", stub)
    sys.meta_path.insert(0, counter)
    try:
        assert _off(app, 0) is not None
        assert _off(app, 1) is not None
    finally:
        sys.meta_path.remove(counter)

    assert attempts == [], "the preview touched the optical-flow module"
    assert app._gpu.built[0].zero_motion_calls == 2


# --- The worker as the GUI runs it -----------------------------------------

def _worker_thread(fn):
    """Run fn off the main thread, the way _in_thread does, and surface its result."""
    box = {}

    def run():
        try:
            box['value'] = fn()
        except BaseException as ex:
            box['error'] = ex

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(60)
    assert not thread.is_alive(), "the worker never returned"
    if 'error' in box:
        raise box['error']
    return box.get('value')


def _drain(q, want, seconds):
    """Items off the UI queue, waiting up to `seconds` for `want` of them."""
    got, deadline = [], time.time() + seconds
    while time.time() < deadline and len(got) < want:
        try:
            got.append(q.get_nowait())
        except queue.Empty:
            time.sleep(0.02)
    return got


def test_a_paused_worker_draws_nothing_until_the_preview_is_resumed(monkeypatch, tmp_path):
    """Pause/resume holds for a preview that asks the GPU host for a session.

    This one lets the worker thread run for real: what is pinned is the queue
    handover and the frame reaching the host, not just the frame function.
    """
    app = _app(monkeypatch, tmp_path, threaded=True)
    worker = app._worker
    try:
        worker.pause()
        worker.request(0, "DLSS", 0.5, 400, 200, _settings())
        assert _drain(app._ui_queue, want=1, seconds=0.5) == [], \
            "a paused worker rendered anyway"
        assert app._gpu.count == 0

        worker.resume()
        items = _drain(app._ui_queue, want=1, seconds=20)
    finally:
        worker.stop()
        worker._thread.join(10)

    assert [i[0] for i in items] == ["preview"]
    res = items[0][1]
    assert res["type"] == "image" and res["bgr"].shape == (200, 400, 3)
    assert app._gpu.count == 1
    assert app._gpu.built[0].zero_motion_calls == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
