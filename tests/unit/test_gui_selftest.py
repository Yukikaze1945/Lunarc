"""`gui.py --selftest` on the project GPU host's zero-motion entry, checked without a GPU.

The probe draws one frame the way the window's own still pictures do: it opens
`dlss_gpu_engine.Live` at 320x240 with guidance off and hands the RGBA frame to the
zero-motion entry that `App._GPU_HOST_KINDS["image"]` names. That entry is on
native/dlss_gpu_host.dll, the only DLSS host this build ships, and the probe reads both
the capability and the refusal off the window's table instead of keeping its own copy -
which is how the probe rotted once already, when it kept describing an entry the window
had moved off and died in a TypeError before a frame was drawn.

These tests pin what the probe has to keep: the frame goes to the same entry the still
images use and is refused through the same table, the retired `dlss_engine` wrapper is
never imported, each failure names the stage it failed at, the session is closed on
every path, and the report file and the returned string are the same text.

A fake GPU host stands in for `dlss_gpu_engine.Live`, so nothing here needs a card. The
real host is exercised once, in `scratch/_gui_selftest_gpu_gate.py`.
"""

import inspect
import os
import sys
import types

import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dlss_gpu_engine
import gui
from dlss_errors import DLSSInferenceError

W, H = 320, 240


class GpuLive:
    """An ABI-3 GPU host session: zero-motion entry only, options baked at open.

    `process` and `update` are traps, as in the image test's fake, because the real
    `dlss_gpu_engine.Live` has neither: a probe that reached for the old host's call
    shape fails here rather than silently asking for an entry that does not exist.
    """

    def __init__(self, w, h, settings=None):
        self._w, self._h = int(w), int(h)
        self.settings = dict(settings or {})
        self.zero_motion_calls = 0
        self.reset_flags = []
        self.closed = False

    def supports_zero_motion(self):
        return True

    def supports_host_motion(self):
        return True

    def process_zero_motion(self, rgba, reset=False, out=None):
        assert isinstance(rgba, np.ndarray) and rgba.dtype == np.uint8
        assert rgba.shape == (self._h, self._w, 4), \
            f"frame {rgba.shape} handed to a session opened at {(self._h, self._w)}"
        assert rgba.flags.c_contiguous, "the GPU host rejects a non-contiguous frame"
        assert np.all(rgba[..., 3] == 255), "alpha is not opaque"
        self.zero_motion_calls += 1
        self.reset_flags.append(bool(reset))
        result = rgba.copy()
        result[..., 1] = np.clip(result[..., 1].astype(np.int16) + 7, 0, 255)
        return result

    def process(self, *args, **kwargs):
        raise AssertionError("the GPU host has no legacy process entry")

    def update(self, *args, **kwargs):
        raise AssertionError("the GPU host bakes its options at open")

    def close(self):
        self.closed = True


class Abi2GpuLive(GpuLive):
    """A host that binds no zero-motion texture: the probe must report it, not draw."""

    def supports_zero_motion(self):
        return False


class RaisingLive(GpuLive):
    def process_zero_motion(self, rgba, reset=False, out=None):
        raise DLSSInferenceError("zero-motion DLSS frame failed (7)")


class ShortOutputLive(GpuLive):
    def process_zero_motion(self, rgba, reset=False, out=None):
        self.zero_motion_calls += 1
        return np.zeros((self._h, self._w, 3), np.uint8)


class _Recorder:
    def __init__(self, cls):
        self._cls = cls
        self.built = []

    def __call__(self, w, h, settings=None):
        live = self._cls(w, h, settings)
        self.built.append(live)
        return live


class _ImportCounter:
    """A meta_path finder that records attempts to import the retired wrapper."""

    def __init__(self, sink):
        self._sink = sink

    def find_spec(self, name, path=None, target=None):
        if name == "dlss_engine" or name.startswith("dlss_engine."):
            self._sink.append(name)
        return None


def _stub_torch(monkeypatch, version="2.7.0", cuda=True):
    """Keep the real torch out of `sys.modules`: the probe only reports it."""
    mod = types.ModuleType("torch")
    mod.__version__ = version
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: cuda
    mod.cuda = cuda_mod
    monkeypatch.setitem(sys.modules, "torch", mod)
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda_mod)


def _probe(monkeypatch, tmp_path, gpu_cls=GpuLive, report_name="selftest.txt"):
    """Run the real `gui._selftest` against a fake host and hand back its report."""
    _stub_torch(monkeypatch)
    recorder = _Recorder(gpu_cls)
    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", recorder)
    report = tmp_path / report_name
    msg = gui._selftest(report_path=str(report))
    return msg, report, recorder


def test_one_frame_runs_on_the_gpu_zero_motion_entry(monkeypatch, tmp_path):
    msg, report, recorder = _probe(monkeypatch, tmp_path)
    assert recorder.built, "the probe drew no frame"
    live = recorder.built[0]
    assert live.zero_motion_calls == 1
    assert live.reset_flags == [True], "an independent probe frame must reset"
    assert live.closed, "the probe leaked its session"
    assert msg.startswith("DLSS_OK ")
    assert report.read_text(encoding="utf-8") == msg


def test_the_probe_asks_the_host_the_default_routes_to(monkeypatch, tmp_path):
    msg, _, recorder = _probe(monkeypatch, tmp_path)
    live = recorder.built[0]
    assert (live._w, live._h) == (W, H)
    assert live.settings == {'preset': 1, 'guidance_mode': 0}, \
        "a probe asks for no guidance, which is what names the zero-motion entry"


def test_the_report_names_the_host_the_entry_and_the_output_shape(monkeypatch, tmp_path):
    msg, _, _ = _probe(monkeypatch, tmp_path)
    assert f"DLSS_OK ({H}, {W}, 4)" in msg
    assert "entry=process_zero_motion" in msg
    assert "dlss_gpu_host.dll" in msg
    assert "sha1=" in msg
    assert "torch=2.7.0 cuda=True" in msg


def test_an_abi2_host_is_refused_with_the_window_own_message(monkeypatch, tmp_path):
    capability, refusal, _ = gui.App._GPU_HOST_KINDS["image"]
    msg, _, recorder = _probe(monkeypatch, tmp_path, gpu_cls=Abi2GpuLive)
    assert msg.startswith("DLSS_ERR [capability] ")
    assert refusal in msg
    assert "rebuild native/dlss_gpu_host.dll" in refusal, \
        "the only host this build ships is the one the operator has to rebuild"
    assert recorder.built[0].closed, "a refused session has to be closed, not leaked"
    assert recorder.built[0].zero_motion_calls == 0
    assert capability == "supports_zero_motion"


def test_the_probe_reaches_no_retired_host_module(monkeypatch, tmp_path):
    """`dlss_engine` was the only seam that pulled `dlssnr_host.dll` into a process.

    Counting the import rather than a call is what survives the module's deletion: a
    probe that still needed it would fail here instead of failing to collect.
    """
    _stub_torch(monkeypatch)
    recorder = _Recorder(GpuLive)
    attempts = []
    counter = _ImportCounter(attempts)
    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", recorder)
    monkeypatch.delitem(sys.modules, "dlss_engine", raising=False)
    sys.meta_path.insert(0, counter)
    try:
        msg = gui._selftest(report_path=str(tmp_path / "selftest.txt"))
        imported = "dlss_engine" in sys.modules
    finally:
        sys.meta_path.remove(counter)
    assert msg.startswith("DLSS_OK ")
    assert attempts == [], f"the probe reached the retired wrapper through {attempts}"
    assert not imported
    assert not hasattr(gui, "dlss_engine"), "gui.py imports the retired wrapper again"


def test_the_refusal_and_capability_are_the_table_not_a_copy(monkeypatch, tmp_path):
    """The probe's refusal is the table's, not a copy it keeps of its own."""
    monkeypatch.setitem(gui.App._GPU_HOST_KINDS, "image",
                        ("supports_zero_motion", "REFUSAL-FROM-TABLE", "ready"))
    msg, _, recorder = _probe(monkeypatch, tmp_path, gpu_cls=Abi2GpuLive)
    assert "REFUSAL-FROM-TABLE" in msg
    assert recorder.built[0].zero_motion_calls == 0


def test_the_capability_name_comes_from_the_table_too(monkeypatch, tmp_path):
    """A probe with a hard-coded capability would have drawn a frame on this host."""
    monkeypatch.setitem(gui.App._GPU_HOST_KINDS, "image",
                        ("supports_device_flow_only", "unused", "ready"))
    msg, _, recorder = _probe(monkeypatch, tmp_path)
    assert msg.startswith("DLSS_ERR [capability] ")
    assert "supports_device_flow_only" in msg
    assert recorder.built[0].zero_motion_calls == 0


def test_a_missing_native_host_is_reported_as_such(monkeypatch, tmp_path):
    def missing(w, h, settings=None):
        raise OSError(126, "module could not be found")
    _stub_torch(monkeypatch)
    monkeypatch.setattr(gui.dlss_gpu_engine, "Live", missing)
    msg = gui._selftest(report_path=str(tmp_path / "selftest.txt"))
    assert msg.startswith("DLSS_ERR [host-load] ")
    assert "native/dlss_gpu_host.dll" in msg
    assert "module could not be found" in msg, "the loader's own reason has to survive"
    assert "本版本唯一的 DLSS 宿主就是它，旧宿主 dlssnr_host.dll 已移除" in msg


def test_a_failed_frame_is_attributed_to_the_frame(monkeypatch, tmp_path):
    msg, _, recorder = _probe(monkeypatch, tmp_path, gpu_cls=RaisingLive)
    assert msg.startswith("DLSS_ERR [frame] ")
    assert "zero-motion DLSS frame failed" in msg
    assert recorder.built[0].closed, "a session that raised mid-frame still has to close"


def test_an_unexpected_output_is_not_reported_as_a_pass(monkeypatch, tmp_path):
    msg, _, _ = _probe(monkeypatch, tmp_path, gpu_cls=ShortOutputLive)
    assert msg.startswith("DLSS_ERR [output] ")
    assert "DLSS_OK" not in msg


def test_main_routes_the_flag_before_any_window_exists(monkeypatch):
    calls = []
    monkeypatch.setattr(gui, "_selftest", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(gui.tk, "Tk",
                        lambda *a, **k: pytest.fail("--selftest opened a Tk root"))
    monkeypatch.setattr(sys, "argv", ["gui.py", "--selftest"])
    gui.main()
    assert calls == [((), {})]


def test_the_frame_entry_still_takes_no_second_plane(monkeypatch):
    """The reason this probe was dead stays dead: the entry has no plane slot.

    `process_zero_motion` is (rgba, reset, out), so a plane restored to the third
    positional still lands on `reset` and ends in a TypeError before a frame is drawn.
    Restoring it is not an option the probe has, so the signature that forbids it is
    pinned here rather than left to whoever next reads a TypeError.
    """
    params = list(inspect.signature(dlss_gpu_engine.Live.process_zero_motion).parameters)
    assert params == ["self", "rgba", "reset", "out"]
    assert not hasattr(dlss_gpu_engine.Live, "process"), \
        "the GPU host grew the legacy motion entry back"
    with pytest.raises(TypeError) as exc:
        dlss_gpu_engine.Live.process_zero_motion(
            None, np.zeros((H, W, 4), np.uint8), np.zeros((H, W), np.float32), reset=True)
    assert "multiple values for argument 'reset'" in str(exc.value)
