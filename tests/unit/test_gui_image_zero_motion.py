"""Still images on the project GPU host's zero-motion entry, checked without a GPU.

gui.py runs one imported picture and one whole picture folder with guidance forced off,
so both ask dlss_gpu_engine.Live for process_zero_motion at the entry named by
App._GPU_HOST_KINDS["image"] - native/dlss_gpu_host.dll, the only DLSS host this build
ships. These tests pin what that path has to keep: the pad-to-8 / trim-back geometry,
RGB order and opaque alpha, a reset for every independent picture, session reuse while
the geometry and the host's create-time options hold and a rebuild when either moves,
the batch's existing naming and skip behaviour, per-image failure and cancellation, that
no optical flow is imported or entered, and that a host which does not bind the
zero-motion entry is refused with the rebuild message instead of being replaced - there
is no second host to draw the picture behind the caller's back, and a kind outside the
table is refused before any session opens.

The App is built without a Tk root and every call is made from a worker thread, which
is how the GUI drives it through _in_thread, so the real image functions run headless.
The fake session traps the entries the GPU host does not have, so "this picture ran on
the zero-motion entry" is a failure mode here rather than a line in a call ledger.
"""

import os
import sys
import threading
import types

import cv2
import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gui
import pipeline
from dlss_errors import DLSSInferenceError


def _settings(**over):
    """The dict _collect_settings produces, with the image mode's guidance off."""
    s = {'preset': 1, 'style': 0, 'intensity': 1.0, 'local_tone': 1.0,
         'local_struct': 1.0, 'skin_struct': 1.0, 'use_auto_mask': 1,
         'ui_correction': 0, 'guidance_mode': 0,
         'motion_scale_x': 1.0, 'motion_scale_y': 1.0}
    s.update(over)
    return s


def _image(w, h, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


class GpuLive:
    """An ABI-3 GPU host session serving only the zero-motion entry a still asks for.

    process and update are traps because the real dlss_gpu_engine.Live has neither, and
    the guided entry is a trap because a picture with guidance off never earns one. The
    shape/alpha checks mirror the real wrapper so a session opened at the wrong size
    fails here rather than at the ctypes boundary.
    """

    def __init__(self, w, h, settings=None):
        self._w, self._h = int(w), int(h)
        self.settings = dict(settings or {})
        self.closed = False
        self.host_dll = "dlss_gpu_host.dll"
        self.host_abi = 3
        self.zero_motion_calls = 0
        self.reset_flags = []
        self.seen_rgba = []

    def close(self):
        self.closed = True

    def is_alive(self):
        return not self.closed

    def supports_zero_motion(self):
        return True

    def process_zero_motion(self, rgba, reset=False, out=None):
        assert isinstance(rgba, np.ndarray) and rgba.dtype == np.uint8
        assert rgba.shape == (self._h, self._w, 4), \
            f"frame {rgba.shape} handed to a session opened at {(self._h, self._w)}"
        assert rgba.flags.c_contiguous, "the GPU host rejects a non-contiguous frame"
        assert np.all(rgba[..., 3] == 255), "alpha is not opaque"
        self.zero_motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.seen_rgba.append(rgba.copy())
        result = rgba.copy()
        result[..., 0] = np.clip(result[..., 0].astype(np.int16) + 33, 0, 255)
        return result

    def process_host_motion(self, rgba, motion, reset=False, out=None):
        raise AssertionError("a still image reached the host-motion entry")

    def process(self, *args, **kwargs):
        raise AssertionError("the GPU host has no motion entry to fall back on")

    def update(self, settings):
        raise AssertionError("GPU session handed the live-update entry it lacks")


class Abi2GpuLive(GpuLive):
    """A host that binds no zero-motion texture: the configuration must be refused."""

    def __init__(self, w, h, settings=None):
        super().__init__(w, h, settings)
        self.host_abi = 2

    def supports_zero_motion(self):
        return False


class _Recorder:
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


def _app(monkeypatch, gpu_cls=GpuLive):
    """A real App whose session bookkeeping is intact and whose Tk widgets are not."""
    app = gui.App.__new__(gui.App)
    app._dlss_lock = threading.Lock()
    app._dlss_session = None
    app._dlss_session_kind = None
    app._image_batch_running = False
    app._video_task_kind = None
    app._cancel_requested = False
    app.current_is_image = True
    app.logs, app.statuses, app.progress = [], [], []
    app.logln = lambda msg: app.logs.append(str(msg))
    app.set_status = lambda msg: app.statuses.append(str(msg))
    app.set_progress = lambda i, total, extra="": app.progress.append((i, total, extra))
    app._cached_settings = _settings()

    app._gpu = _Recorder(gpu_cls)
    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", app._gpu)
    return app


def _worker(fn):
    """Run fn off the main thread, the way _in_thread does, and surface its result.

    _collect_settings reads the cached settings only for a non-main thread, which is
    what lets these tests drive the real image functions without a Tk root.
    """
    box = {}

    def run():
        try:
            box['value'] = fn()
        except BaseException as ex:  # re-raised below, on the main thread
            box['error'] = ex

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(60)
    assert not thread.is_alive(), "the image worker never returned"
    if 'error' in box:
        raise box['error']
    return box.get('value')


def _folder(tmp_path, sizes, name="pics"):
    """A folder of real PNGs at the given (w, h) sizes, plus its sorted file list."""
    src = tmp_path / name
    src.mkdir()
    for i, (w, h) in enumerate(sizes):
        assert cv2.imwrite(str(src / f"img{i}.png"), _image(w, h, seed=i))
    return src, sorted(f"img{i}.png" for i in range(len(sizes)))


class _ImportCounter:
    """A meta_path finder that records attempts to import optical_flow."""

    def __init__(self, sink):
        self._sink = sink

    def find_spec(self, name, path=None, target=None):
        if name == "optical_flow" or name.startswith("optical_flow."):
            self._sink.append(f"import:{name}")
        return None


def _ready_line():
    return gui.App._GPU_HOST_KINDS["image"][2]


# --- Routing: the image path runs on the GPU host and nowhere else -----------

def test_single_image_runs_on_the_gpu_zero_motion_entry(monkeypatch):
    app = _app(monkeypatch)
    img = _image(64, 32, seed=2)

    out = _worker(lambda: app._image_dlss(img))

    assert out is not None and out.shape == (32, 64, 3) and out.dtype == np.uint8
    live = app._gpu.built[0]
    assert (live._w, live._h) == (64, 32)
    assert app._gpu.count == 1 and app._dlss_session_kind == "image"
    assert live.zero_motion_calls == 1
    assert live.reset_flags == [True], "an independent image must reset temporal state"
    # The line the operator reads is the table's own, and it names what drew the frame.
    assert any(m.startswith(_ready_line()) and "dlss_gpu_host.dll" in m and "abi=3" in m
               and "@ 64x32" in m for m in app.logs), app.logs


def test_single_image_output_keeps_the_channel_order(monkeypatch):
    app = _app(monkeypatch)
    img = _image(64, 32, seed=3)

    out = _worker(lambda: app._image_dlss(img))

    # The fake lifts the host's red channel only, so a conversion that landed
    # differently would move blue instead of red in the BGR result.
    lifted_red = np.clip(img[..., 2].astype(np.int16) + 33, 0, 255).astype(np.uint8)
    assert np.array_equal(out[..., 2], lifted_red)
    assert np.array_equal(out[..., 1], img[..., 1])
    assert np.array_equal(out[..., 0], img[..., 0])


def test_odd_dimensions_are_padded_to_a_multiple_of_8_and_trimmed_back(monkeypatch):
    app = _app(monkeypatch)
    img = _image(33, 25, seed=4)

    out = _worker(lambda: app._image_dlss(img))

    live = app._gpu.built[0]
    assert (live._w, live._h) == (40, 32), "session not opened at the padded size"
    handed = live.seen_rgba[0]
    assert handed.shape == (32, 40, 4)
    pt, pl = (32 - 25) // 2, (40 - 33) // 2
    interior = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert np.array_equal(handed[pt:pt + 25, pl:pl + 33, :3], interior)
    # The ring is replicate padding, not a black border the host would then denoise.
    assert np.array_equal(handed[pt - 1, pl:pl + 33, :3], handed[pt, pl:pl + 33, :3])
    assert np.array_equal(handed[pt + 25, pl:pl + 33, :3], handed[pt + 24, pl:pl + 33, :3])
    assert np.array_equal(handed[pt:pt + 25, pl - 1, :3], handed[pt:pt + 25, pl, :3])
    assert np.array_equal(handed[pt:pt + 25, pl + 33, :3], handed[pt:pt + 25, pl + 32, :3])
    assert out.shape == (25, 33, 3), "result not trimmed to the original size"


def test_batch_writes_every_image_to_the_sibling_folder(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    sizes = [(50, 37), (50, 37), (64, 40)]
    src, files = _folder(tmp_path, sizes)

    _worker(lambda: app._do_images(str(src), files))

    out_dir = tmp_path / "pics_dlss"
    assert sorted(os.listdir(out_dir)) == ["img0.png", "img1.png", "img2.png"]
    for i, (w, h) in enumerate(sizes):
        written = pipeline.imread_u(str(out_dir / f"img{i}.png"))
        assert written is not None, f"img{i}.png is not a readable image"
        assert written.shape == (h, w, 3) and written.dtype == np.uint8
        assert np.array_equal(written[..., 1], _image(w, h, seed=i)[..., 1])
    assert any("图片批处理完成: 3/3" in m for m in app.logs)
    assert app._dlss_session_kind == "image", "a batch ran on another entry"


def test_batch_reuses_one_session_per_geometry_and_rebuilds_on_a_new_one(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    src, files = _folder(tmp_path, [(50, 37), (50, 37), (64, 40)])

    _worker(lambda: app._do_images(str(src), files))

    assert app._gpu.count == 2, "same geometry rebuilt, or a new geometry did not"
    first, second = app._gpu.built
    assert (first._w, first._h) == (56, 40) and (second._w, second._h) == (64, 40)
    assert [first.zero_motion_calls, second.zero_motion_calls] == [2, 1]
    assert first.closed, "the session the old geometry used was left open"
    # Each image resets: a folder of stills is not a temporal sequence.
    assert first.reset_flags == [True, True] and second.reset_flags == [True]


def test_image_session_rebuilds_when_a_baked_option_moves(monkeypatch):
    app = _app(monkeypatch)
    img = _image(64, 32, seed=5)

    _worker(lambda: app._dlss_image(img, _settings()))
    _worker(lambda: app._dlss_image(img, _settings(intensity=0.5)))

    assert app._gpu.count == 2, "the GPU host bakes intensity at open; it must reopen"
    assert app._gpu.built[0].closed
    assert app._gpu.built[1].zero_motion_calls == 1


def test_image_session_is_reused_for_an_identical_request(monkeypatch):
    app = _app(monkeypatch)
    img = _image(64, 32, seed=6)

    _worker(lambda: app._dlss_image(img, _settings()))
    _worker(lambda: app._dlss_image(img, _settings()))

    assert app._gpu.count == 1, "a second open of the same session is the churn to avoid"
    assert app._gpu.built[0].zero_motion_calls == 2


# --- Batch behaviour that already existed and had to survive -----------------

def test_batch_skips_an_existing_output_without_calling_the_host(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    src, files = _folder(tmp_path, [(64, 40), (64, 40)])
    out_dir = tmp_path / "pics_dlss"
    out_dir.mkdir()
    sentinel = _image(64, 40, seed=99)
    pipeline.imwrite_u(str(out_dir / "img0.png"), sentinel)

    _worker(lambda: app._do_images(str(src), files))

    assert app._gpu.built[0].zero_motion_calls == 1, "a cached output was re-processed"
    assert np.array_equal(pipeline.imread_u(str(out_dir / "img0.png")), sentinel)
    assert pipeline.imread_u(str(out_dir / "img1.png")) is not None


def test_batch_skips_an_unreadable_image_and_keeps_going(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    src, files = _folder(tmp_path, [(64, 40), (64, 40)])
    (src / "a_broken.png").write_bytes(b"not a png at all")
    files = sorted(files + ["a_broken.png"])

    _worker(lambda: app._do_images(str(src), files))

    assert any("跳过(无法读取): a_broken.png" in m for m in app.logs)
    assert app._gpu.built[0].zero_motion_calls == 2
    assert sorted(os.listdir(tmp_path / "pics_dlss")) == ["img0.png", "img1.png"]


def test_a_per_image_inference_failure_does_not_stop_the_batch(monkeypatch, tmp_path):
    class _Failing(GpuLive):
        def process_zero_motion(self, rgba, reset=False, out=None):
            self.attempts = getattr(self, "attempts", 0) + 1
            if self.attempts == 2:
                raise DLSSInferenceError("Feature 18 推理失败")
            return GpuLive.process_zero_motion(self, rgba, reset=reset, out=out)

    app = _app(monkeypatch, gpu_cls=_Failing)
    src, files = _folder(tmp_path, [(64, 40)] * 3)

    _worker(lambda: app._do_images(str(src), files))

    live = app._gpu.built[0]
    assert live.attempts == 3, "the batch stopped at the failing image"
    assert live.zero_motion_calls == 2
    assert sorted(os.listdir(tmp_path / "pics_dlss")) == ["img0.png", "img2.png"]
    assert any("DLSS 失败: img1.png" in m for m in app.logs)
    assert not live.closed, "one failed image does not invalidate the session"


def test_cancellation_stops_the_batch_at_the_next_image(monkeypatch, tmp_path):
    app = _app(monkeypatch)

    class _CancelAfterFirst(GpuLive):
        def process_zero_motion(self, rgba, reset=False, out=None):
            result = GpuLive.process_zero_motion(self, rgba, reset=reset, out=out)
            if self.zero_motion_calls == 1:
                app._cancel_requested = True
            return result

    recorder = _Recorder(_CancelAfterFirst)
    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", recorder)
    src, files = _folder(tmp_path, [(64, 40)] * 3)

    _worker(lambda: app._do_images(str(src), files))

    live = recorder.built[0]
    assert live.zero_motion_calls == 1, "a cancelled batch kept processing images"
    assert os.listdir(tmp_path / "pics_dlss") == ["img0.png"]
    assert any("图片批处理已由用户中止于第 1 张" in m for m in app.logs)
    assert "批处理已中止" in app.statuses[-1]


# --- A still asks for no motion of any kind ---------------------------------

def test_image_path_allocates_no_cpu_flow_or_depth_plane(monkeypatch):
    app = _app(monkeypatch)
    real_zeros = np.zeros
    seen = []

    def spy_zeros(shape, *args, **kwargs):
        try:
            dtype = kwargs["dtype"] if "dtype" in kwargs else (args[0] if args else None)
            seen.append((tuple(shape), dtype))
        except TypeError:
            pass
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", spy_zeros)
    out = _worker(lambda: app._image_dlss(_image(33, 25, seed=7)))

    assert out is not None
    # The zero-motion entry takes a colour frame and nothing else, so a two-channel
    # motion plane or a float depth plane here means the frame grew an input again.
    assert [s for s, _ in seen if len(s) == 3 and s[2] == 2] == [], \
        "zero-motion images built a host motion plane"
    assert [s for s, dt in seen if dt is np.float32 and len(s) == 2] == [], \
        "zero-motion images built a host depth plane"


def test_image_path_never_imports_or_enters_optical_flow(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    attempts = []

    def _module_getattr(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        attempts.append(name)
        raise AssertionError(f"the image path entered optical_flow.{name}")

    stub = types.ModuleType("optical_flow")
    stub.__getattr__ = _module_getattr
    counter = _ImportCounter(attempts)
    monkeypatch.setitem(sys.modules, "optical_flow", stub)
    sys.meta_path.insert(0, counter)
    try:
        src, files = _folder(tmp_path, [(64, 40)])
        _worker(lambda: app._do_images(str(src), files))
        _worker(lambda: app._image_dlss(_image(48, 32, seed=8)))
    finally:
        sys.meta_path.remove(counter)

    assert attempts == [], "the image path touched the optical-flow module"
    assert app._dlss_session_kind == "image"


# --- Unsupported configurations fail instead of migrating sideways ----------

def test_abi2_gpu_host_is_refused_for_a_single_image_with_the_rebuild_message(monkeypatch):
    _, refusal, _ = gui.App._GPU_HOST_KINDS["image"]
    app = _app(monkeypatch, gpu_cls=Abi2GpuLive)

    out = _worker(lambda: app._image_dlss(_image(64, 40, seed=9)))

    assert out is None
    assert app._gpu.built[0].closed, "the refused session must be released"
    assert app._gpu.built[0].zero_motion_calls == 0
    assert any("GPU 宿主不可用" in m and refusal in m for m in app.logs), app.logs
    assert "ABI 2" in refusal and "rebuild native/dlss_gpu_host.dll" in refusal
    assert app._dlss_session is None, "a refused session was cached as if it worked"
    assert app._gpu.count == 1, "a host that cannot serve images was reopened elsewhere"


def test_abi2_gpu_host_stops_a_batch_once_rather_than_per_image(monkeypatch, tmp_path):
    app = _app(monkeypatch, gpu_cls=Abi2GpuLive)
    src, files = _folder(tmp_path, [(64, 40)] * 4)

    _worker(lambda: app._do_images(str(src), files))

    assert app._gpu.count == 1, "a host that cannot open was reopened for every file"
    assert os.listdir(tmp_path / "pics_dlss") == []
    assert sum("GPU 宿主不可用" in m for m in app.logs) == 1


def test_a_host_that_cannot_open_surfaces_its_reason(monkeypatch):
    app = _app(monkeypatch)

    def _missing_library(w, h, settings=None):
        raise OSError("could not find module 'dlss_gpu_host.dll' (or one of its dependencies)")

    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", _missing_library)

    out = _worker(lambda: app._image_dlss(_image(64, 40, seed=10)))

    assert out is None
    assert any("GPU 宿主不可用" in m and "dlss_gpu_host.dll" in m for m in app.logs)
    assert app._dlss_session is None and app._dlss_session_kind is None


def test_a_frame_failure_is_reported_and_returns_no_result(monkeypatch):
    class _Raising(GpuLive):
        def process_zero_motion(self, rgba, reset=False, out=None):
            raise DLSSInferenceError("zero-motion DLSS frame failed (2)")

    app = _app(monkeypatch, gpu_cls=_Raising)

    assert _worker(lambda: app._image_dlss(_image(64, 40, seed=11))) is None
    assert any("推理失败" in m for m in app.logs)
    # One frame failing is not the host failing: the session stays open and cached.
    assert not app._gpu.built[0].closed
    assert app._dlss_session is app._gpu.built[0]


# --- One slot, one host: the kinds that share it ----------------------------

def test_image_and_preview_requests_hand_the_slot_over_instead_of_sharing_it(monkeypatch):
    """A still and a guidance-off preview ask the same host for the same entry.

    What separates them is the kind, and the slot is keyed on it: a batch and a played
    sequence each own their picture's temporal state, so each request that names
    another entry closes the cached session instead of driving it for someone else.
    """
    app = _app(monkeypatch)
    rgba = np.ones((32, 64, 4), np.uint8) * 7
    rgba[..., 3] = 255

    preview = app.get_dlss_session(64, 32, _settings(), "gpu_zero_motion")
    assert preview is app._gpu.built[0]
    preview.process_zero_motion(rgba, reset=False)
    assert app._gpu.count == 1

    _worker(lambda: app._image_dlss(_image(64, 32, seed=12)))
    assert app._gpu.count == 2, "the image reused the preview's session"
    assert app._dlss_session_kind == "image"
    assert preview.closed, "the preview's session survived the image taking the slot"
    assert preview.zero_motion_calls == 1
    assert app._gpu.built[1].reset_flags == [True]

    again = app.get_dlss_session(64, 32, _settings(), "gpu_zero_motion")
    assert again is app._gpu.built[2], "the preview did not reclaim the slot"
    assert app._gpu.built[1].closed, "the image session outlived the image"
    assert app._dlss_session_kind == "gpu_zero_motion"


def test_a_video_caller_rebuilds_rather_than_reusing_through_a_moved_option(monkeypatch):
    """The reuse contract, with no live-update entry to fall back on.

    dlss_gpu_engine.Live reads its options at open, so a request whose baked option
    moved is a different session; the fake's update() trap is what proves the rebuild
    was the only answer available.
    """
    app = _app(monkeypatch)

    first = app.get_dlss_session(64, 32, _settings(), "gpu_zero_motion")
    assert app.get_dlss_session(64, 32, _settings(), "gpu_zero_motion") is first
    assert app._gpu.count == 1, "an unchanged request reopened the host"

    moved = app.get_dlss_session(64, 32, _settings(intensity=0.5), "gpu_zero_motion")
    assert moved is not first and first.closed
    assert app._gpu.count == 2

    resized = app.get_dlss_session(128, 32, _settings(intensity=0.5), "gpu_zero_motion")
    assert resized is not moved and moved.closed
    assert (resized._w, resized._h) == (128, 32)
    assert app._gpu.count == 3


def test_a_kind_outside_the_gpu_host_table_is_refused_before_it_opens(monkeypatch):
    """The retired host's own kind names no entry this build has.

    There is no second host to hand the request to, so the answer is a refusal that
    lists the entries that do exist - raised by the session bookkeeping and logged by
    the caller's wrapper, which has to return None rather than half a session.
    """
    app = _app(monkeypatch)

    with pytest.raises(DLSSInferenceError) as refused:
        with app._dlss_lock:
            app._get_or_create_session_locked(64, 32, _settings(), "legacy")

    message = str(refused.value)
    assert "kind='legacy'" in message and "native/dlss_gpu_host.dll" in message
    assert all(kind in message for kind in gui.App._GPU_HOST_KINDS), message
    assert "legacy" not in gui.App._GPU_HOST_KINDS

    assert app.get_dlss_session(64, 32, _settings(), "host_pointer") is None
    assert any("未知 DLSS 会话类型" in m for m in app.logs)
    assert app._gpu.count == 0, "a refused kind still opened a host"
    assert app._dlss_session is None and app._dlss_session_kind is None


def test_close_dlss_session_releases_the_cached_gpu_session(monkeypatch):
    app = _app(monkeypatch)

    _worker(lambda: app._image_dlss(_image(64, 32, seed=13)))
    live = app._gpu.built[0]
    assert not live.closed

    app.close_dlss_session()

    assert live.closed, "shutdown leaked the GPU host session"
    assert app._dlss_session is None and app._dlss_session_kind is None
    app.close_dlss_session()  # idempotent teardown on the way out


# --- The batch owns the one session slot while a video can still be previewed --

class _WorkerSpy:
    """The preview worker's pause/resume surface, counted."""

    def __init__(self):
        self.pauses, self.resumes = 0, 0

    def pause(self):
        self.pauses += 1

    def resume(self):
        self.resumes += 1


def test_batch_parks_the_preview_worker_and_holds_the_slot_until_it_ends(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    worker = _WorkerSpy()
    app._preview_worker = worker
    frames_seen = []

    class _Observing(GpuLive):
        def process_zero_motion(self, rgba, reset=False, out=None):
            # Same thread as the batch, so reading the flag needs no lock.
            frames_seen.append(bool(app._image_batch_running))
            return GpuLive.process_zero_motion(self, rgba, reset=reset, out=out)

    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", _Recorder(_Observing))
    src, files = _folder(tmp_path, [(64, 40)] * 3)

    _worker(lambda: app._do_images(str(src), files))

    assert frames_seen == [True, True, True], "the batch did not own the slot while it ran"
    assert (worker.pauses, worker.resumes) == (1, 1)
    assert app._image_batch_running is False, "the batch left the slot claimed"


def test_a_video_request_is_handed_the_slot_back_once_the_batch_has_ended(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    app._preview_worker = _WorkerSpy()
    src, files = _folder(tmp_path, [(64, 40)])

    _worker(lambda: app._do_images(str(src), files))
    image_session = app._gpu.built[0]

    video = app.get_dlss_session(64, 40, _settings(), "gpu_zero_motion")
    assert video is app._gpu.built[1]
    assert app._dlss_session_kind == "gpu_zero_motion"
    assert image_session.closed, "the batch's session outlived the batch"


def test_a_video_request_during_a_batch_is_refused_without_reopening_the_host(monkeypatch):
    app = _app(monkeypatch)
    _worker(lambda: app._dlss_image(_image(64, 40, seed=21), _settings()))
    image_session = app._gpu.built[0]

    with app._dlss_lock:
        app._image_batch_running = True
    video = app.get_dlss_session(64, 40, _settings(), "gpu_zero_motion")

    assert video is None, "a preview frame took the slot back mid-batch"
    # dlss_gpu_host_open refuses a second live session, so a session built here would
    # have to be the batch's own replacement, evicted out from under the frame after it.
    assert app._gpu.count == 1, "the refusal still opened another host session"
    assert not image_session.closed, "the batch's GPU session was evicted"
    assert app._dlss_session is image_session

    with app._dlss_lock:
        app._image_batch_running = False
    assert app.get_dlss_session(64, 40, _settings(), "gpu_zero_motion") is not None


def test_batch_resumes_the_preview_worker_when_the_host_cannot_open(monkeypatch, tmp_path):
    app = _app(monkeypatch, gpu_cls=Abi2GpuLive)
    worker = _WorkerSpy()
    app._preview_worker = worker
    src, files = _folder(tmp_path, [(64, 40)] * 3)

    _worker(lambda: app._do_images(str(src), files))

    assert (worker.pauses, worker.resumes) == (1, 1)
    assert app._image_batch_running is False


def test_batch_resumes_the_preview_worker_when_cancelled(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    worker = _WorkerSpy()
    app._preview_worker = worker

    class _CancelAfterFirst(GpuLive):
        def process_zero_motion(self, rgba, reset=False, out=None):
            result = GpuLive.process_zero_motion(self, rgba, reset=reset, out=out)
            if self.zero_motion_calls == 1:
                app._cancel_requested = True
            return result

    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", _Recorder(_CancelAfterFirst))
    src, files = _folder(tmp_path, [(64, 40)] * 3)

    _worker(lambda: app._do_images(str(src), files))

    assert (worker.pauses, worker.resumes) == (1, 1)
    assert app._image_batch_running is False


def test_batch_resumes_the_preview_worker_when_the_body_raises(monkeypatch, tmp_path):
    app = _app(monkeypatch)
    worker = _WorkerSpy()
    app._preview_worker = worker
    src, files = _folder(tmp_path, [(64, 40)])

    def _failing_makedirs(path, *args, **kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(gui.os, "makedirs", _failing_makedirs)

    _worker(lambda: app._do_images(str(src), files))

    assert (worker.pauses, worker.resumes) == (1, 1)
    assert app._image_batch_running is False
    assert any("图片批处理错误" in m for m in app.logs)


def test_a_batch_survives_with_no_preview_worker_at_all(monkeypatch, tmp_path):
    """_do_images runs for a folder pick with no video loaded, so the worker may be absent."""
    app = _app(monkeypatch)
    src, files = _folder(tmp_path, [(64, 40)])

    _worker(lambda: app._do_images(str(src), files))

    assert os.listdir(tmp_path / "pics_dlss") == ["img0.png"]


# --- What the render path may and may not touch -------------------------------

def test_the_render_path_reads_no_widget_state_and_holds_the_session_lock(monkeypatch):
    app = _app(monkeypatch)

    def _no_widgets():
        raise AssertionError("_dlss_image re-read the Tk widgets")

    app._collect_settings = _no_widgets
    lock_held = []

    class _LockProbing(GpuLive):
        def process_zero_motion(self, rgba, reset=False, out=None):
            # The video paths take the same lock around their own process call, so a
            # frame that is not inside it can have its session evicted underneath it.
            acquired = app._dlss_lock.acquire(blocking=False)
            lock_held.append(not acquired)
            if acquired:
                app._dlss_lock.release()
            return GpuLive.process_zero_motion(self, rgba, reset=reset, out=out)

    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", _Recorder(_LockProbing))

    out = _worker(lambda: app._dlss_image(_image(64, 40, seed=22), _settings()))

    assert out is not None
    assert lock_held == [True], "session lookup and inference did not share one lock hold"
