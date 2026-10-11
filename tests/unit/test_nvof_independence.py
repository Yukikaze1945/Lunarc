# -*- coding: utf-8 -*-
"""Gate: guidance_mode=0 loads, negotiates and starts without ever meeting NVOF.

The unguided frame path takes its motion from the D3D12 texture the DLSS host zeros
once at open, so no frame in a mode 0 run asks an optical flow engine for anything.
That is only independence if the *loading* path says so as well: a chain that refuses
to start because a library it will never call is missing still has that library as a
dependency, and on a machine with no optical flow engine installed it would be a chain
that cannot run unguided at all.

Four claims are checked here rather than asserted:

* an unguided bridge set binds no optical flow bridge and never looks for the module -
  the import is not merely tolerated as optional, it is not attempted;
* the guided set keeps every refusal it had: no module, no hardware, no device flow
  output export and no device RGBA input export each fail, and each says which;
* the question put to the loader is read from the same helper the gate reads the mode
  from, so the libraries and the refusal cannot disagree about what was requested;
* a set that fails to load leaves no DLL search directory registered, which is the
  partial initialisation a skipped probe must not become;
* the finished run's report carries the loading fact itself, not just the frame
  counters that would look the same if nothing had decoded.

No GPU is involved. The five bridges are stood in for by an object that answers the
binding protocol, because the claim is about what gets imported and probed, not about
what CUDA says when it is asked.
"""

import os
import sys

import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UNIT_DIR)
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gpu_transfer_export as gtx  # noqa: E402
import pipeline  # noqa: E402

# The libraries a GPU-transfer export binds, in the order it binds them. An optical
# flow bridge is not on this list at all: it is loaded by the module the guided path
# imports, which is exactly the coupling under test.
BRIDGES = ("nvdec_bridge", "nv12_rgba_bridge", "gpu_lanczos_bridge",
           "gpu_residual_bridge", "nvenc_bridge")
# The same five, as the loaded set hands them back.
BRIDGE_KEYS = ("nvdec", "nv12", "lanczos", "residual", "nvenc")

SOURCE = {"video_stream_count": 1, "codec_name": "hevc", "pix_fmt": "yuv420p",
          "width": 3840, "height": 2160, "color_range": "tv",
          "color_space": "smpte170m", "color_transfer": "bt709",
          "color_primaries": "bt709", "r_frame_rate": "30000/1001",
          "audio_streams": 1, "subtitle_streams": 0, "nb_chapters": 0,
          "metadata": True}


class _EntryPoint:
    """A bound symbol: restype and argtypes are settable as on a ctypes function.

    Called for the version exports, which have to answer with the number the loader
    reads them against or the set is refused for the wrong reason, and for the
    capability getters, which answer with what this checkout's bridges report.
    """

    _abi = {"nvdec_bridge_abi_version": gtx.NVDEC_ABI,
            "nv12_rgba_abi_version": gtx.NV12_ABI,
            "gpu_lanczos_abi_version": gtx.LANCZOS_ABI,
            "gpu_residual_abi_version": gtx.RESIDUAL_ABI,
            "nvenc_abi_version": gtx.NVENC_ABI}
    _figures = {"gpu_lanczos_max_taps": gtx.LANCZOS_MAX_TAPS}

    def __init__(self, name):
        self.name = name
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self._figures.get(self.name, self._abi.get(self.name, 0))


class _BridgeStub:
    """Answers for one bridge DLL, exporting whatever the loader asks for.

    ``absent`` names exports a particular build does not have. The loader's whole
    capability question is whether a symbol is in the binary, so a stand-in that
    fabricated one would answer a question a real build never gets asked - and the
    answer it fabricated would be the number negotiation then serves a plan to.
    """

    def __init__(self, name, absent=()):
        self._name = name + ".dll"
        self._symbols = {}
        self._absent = frozenset(absent)

    def __getattr__(self, symbol):
        # Reached only for names not already on the instance, so _name and _symbols
        # above do not come back here.
        if symbol in self._absent:
            raise AttributeError(symbol)
        found = self._symbols.get(symbol)
        if found is None:
            found = self._symbols[symbol] = _EntryPoint(symbol)
        return found


class _Cookie:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _ImportBlocker:
    """Make one module impossible to import, as an install without it would."""

    def __init__(self, name):
        self.name = name
        self.attempts = 0

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.name:
            self.attempts += 1
            raise ImportError(f"No module named '{self.name}'")
        return None


@pytest.fixture
def loaded_bridges(monkeypatch, tmp_path):
    """A load that binds five stand-ins, touches no DLL, and reports what it bound."""
    bound = []

    def fake_load(name):
        bound.append(name)
        return _BridgeStub(name)

    monkeypatch.setattr(gtx, "_load", fake_load)
    monkeypatch.setattr(gtx, "ffmpeg_dll_directories", lambda: [str(tmp_path)])
    return bound


@pytest.fixture
def no_flow_module(monkeypatch):
    """Take the optical flow module out of the interpreter for this test only."""
    monkeypatch.delitem(sys.modules, "optical_flow", raising=False)
    blocker = _ImportBlocker("optical_flow")
    monkeypatch.setattr(sys, "meta_path", [blocker] + list(sys.meta_path))
    return blocker


@pytest.fixture
def flow_probe(monkeypatch):
    """Answer the probe the guided path reads, and record every time it was asked."""
    import optical_flow as flow_module
    answers = {"available": True, "device_output": True,
               "device_rgba_input": True, "error": None}
    asked = []

    def set_answers(**changes):
        answers.update(changes)

    def fake_probe():
        asked.append(1)
        return dict(answers)

    monkeypatch.setattr(flow_module, "probe_nvof_capabilities", fake_probe)
    set_answers.asked = asked
    return set_answers


# ---------------------------------------------------------------------------
# what an unguided load binds
# ---------------------------------------------------------------------------

def test_an_unguided_load_binds_the_five_bridges_and_no_flow_engine(loaded_bridges):
    loaded = gtx.load_bridges(requires_flow=False)
    assert loaded_bridges == list(BRIDGES)
    assert loaded["nvof_probe"] is None
    assert loaded["optical_flow"] is None
    # The rest of the set is unchanged by the mode: skipping NVOF is not skipping
    # a stage the unguided chain still runs.
    for name in BRIDGE_KEYS:
        assert loaded[name] is not None


def test_a_bridge_that_reports_no_window_records_no_window(monkeypatch, tmp_path):
    """What the load records for the two ABI-2 builds of gpu_lanczos_bridge.

    The plan-driven open and gpu_lanczos_max_taps arrived in different builds under
    the same ABI number, so a loaded library can have the open and no answer for
    the window. Absence has to reach negotiation as absence: a capability filled in
    from this module's mirror of the header is a window the binary never reported,
    and it is how a 17-tap plan gets handed to the 16-wide kernel the open then
    refuses - after the job has decoded frames.
    """
    def load_with(absent):
        def fake_load(name):
            return _BridgeStub(name, absent if name == "gpu_lanczos_bridge" else ())

        monkeypatch.setattr(gtx, "_load", fake_load)
        monkeypatch.setattr(gtx, "ffmpeg_dll_directories", lambda: [str(tmp_path)])
        return gtx.load_bridges(requires_flow=False)["capabilities"]

    reported = load_with(()).get("lanczos_max_taps")
    assert reported == gtx.LANCZOS_MAX_TAPS, reported
    legacy = load_with(("gpu_lanczos_max_taps",))
    assert "lanczos_max_taps" not in legacy, legacy
    # The plan-driven open it does export is still recorded: the old build is
    # narrower, not absent, and a plan inside its window is still served.
    assert legacy["lanczos_scaled"] and legacy["residual_scaled"]


def test_an_unguided_load_succeeds_with_no_optical_flow_module_installed(
        loaded_bridges, no_flow_module):
    """The claim, in its strongest form: the module is not looked for.

    A load that caught ImportError and carried on would pass a weaker test and
    still be a load that depends on NVOF for its answer.
    """
    loaded = gtx.load_bridges(requires_flow=False)
    assert loaded["nvof_probe"] is None
    assert no_flow_module.attempts == 0


def test_an_unguided_load_does_not_probe_the_engine_even_when_a_probe_would_fail(
        loaded_bridges, flow_probe):
    flow_probe(available=False, error="nvofapi64.dll is not installed")
    loaded = gtx.load_bridges(requires_flow=False)
    assert loaded["nvof_probe"] is None
    assert flow_probe.asked == []


def test_a_guided_load_binds_the_same_bridges_and_then_the_flow_engine(
        loaded_bridges, flow_probe):
    """The contrast: mode 1 asks, and the answer is kept for the report to name."""
    flow_probe()
    loaded = gtx.load_bridges(requires_flow=True)
    assert loaded_bridges == list(BRIDGES)
    assert len(flow_probe.asked) == 1
    assert loaded["nvof_probe"]["available"] is True
    assert loaded["optical_flow"] is not None


# ---------------------------------------------------------------------------
# what a guided load refuses
# ---------------------------------------------------------------------------

def test_a_guided_load_without_the_flow_module_refuses_and_says_why(
        loaded_bridges, no_flow_module):
    with pytest.raises(gtx.BridgeError) as raised:
        gtx.load_bridges(requires_flow=True)
    message = str(raised.value)
    assert "guidance_mode=1" in message, message
    assert "optical_flow" in message, message
    assert no_flow_module.attempts >= 1


@pytest.mark.parametrize("changes,fragment", [
    ({"available": False, "error": "no NVOF hardware"}, "NVOF is unavailable"),
    ({"device_output": False}, "no device flow output export"),
    ({"device_rgba_input": False}, "no device RGBA input export"),
])
def test_a_guided_load_refuses_each_answer_the_unguided_path_cannot_run_on(
        loaded_bridges, flow_probe, changes, fragment):
    """Mode 1 keeps every validation it had, word for word."""
    flow_probe(**changes)
    with pytest.raises(gtx.BridgeError) as raised:
        gtx.load_bridges(requires_flow=True)
    assert fragment in str(raised.value), str(raised.value)


def test_an_unguided_load_accepts_the_answers_a_guided_load_refuses(
        loaded_bridges, flow_probe):
    flow_probe(available=False, device_output=False, device_rgba_input=False,
               error="no NVOF hardware")
    assert gtx.load_bridges(requires_flow=False)["nvof_probe"] is None


# ---------------------------------------------------------------------------
# the mode the loader is asked about is the mode the gate reads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("job,requires_flow", [
    ({"guidance_mode": 1}, True),
    ({"guidance_mode": 0}, False),
    ({}, False),
    ({"guidance_mode": "0"}, False),
    ({"guidance_mode": 2}, False),
    ({"guidance_mode": None}, True),
    ({"guidance_mode": "flow"}, True),
])
def test_the_loader_is_asked_the_question_the_gate_already_answered(job,
                                                                    requires_flow):
    """Read as the gate reads it, and strict when neither can read it at all.

    ``2`` is the gate's own refusal, not the loader's business; an unreadable mode
    asks for the flow bridges so that the setting which cannot be interpreted is
    the only thing the job is refused for.
    """
    assert gtx.guidance_requires_flow(job) is requires_flow


class _UnguidedCapableLive:
    """An ABI-3 session: it binds its own zero motion texture and says so."""

    def supports_zero_motion(self):
        return True

    def process_device_color_flow_resource(self, color, flow, reset=False):
        return {"d3d12_device": 1, "d3d12_resource": 2}

    def process_device_color_zero_motion_resource(self, color, reset=False):
        return {"d3d12_device": 1, "d3d12_resource": 2}


@pytest.mark.parametrize("mode,asked_for_flow", [(0, False), (1, True)])
def test_the_backend_asks_the_bridge_set_for_the_mode_the_job_asked_for(
        monkeypatch, mode, asked_for_flow):
    """The wiring, not just the function: a mode 0 job must not put the question.

    The load is stopped on purpose, because all this asserts is which way the
    backend turned the job's guidance mode when it asked for its libraries.
    """
    seen = {}

    def fake_load(requires_flow=True):
        seen["requires_flow"] = requires_flow
        raise gtx.BridgeError("stopped here on purpose")

    monkeypatch.setattr(gtx, "load_bridges", fake_load)
    monkeypatch.setattr(gtx, "probe_source",
                        lambda video, ffprobe=None: dict(SOURCE))
    monkeypatch.setattr(pipeline, "video_info",
                        lambda path: (14, 29.97, 3840, 2160))
    job = {"dlss_backend": "gpu_flow_d3d12", "guidance_mode": mode,
           "flow_engine": "nvof", "working_scale": 0.5, "residual_pyramid": True,
           "multi_pass": 1, "export_backend": "gpu_transfer"}
    with pytest.raises(RuntimeError) as raised:
        pipeline.export_gpu_transfer_stream(
            video="in.mp4", live=_UnguidedCapableLive(), frames=14, fps=29.97,
            cq=24, with_audio=False, out_path="out.mkv", transcode_settings={},
            job_settings=job, flow_provider=None, module=gtx)
    assert seen["requires_flow"] is asked_for_flow
    assert "stopped here on purpose" in str(raised.value)


# ---------------------------------------------------------------------------
# partial initialisation
# ---------------------------------------------------------------------------

def test_a_refused_load_leaves_no_dll_search_directory(
        loaded_bridges, flow_probe, monkeypatch):
    """The cookies are opened first and released on the way out, mode after mode."""
    cookies = []
    monkeypatch.setattr(os, "add_dll_directory",
                        lambda path: cookies.append(_Cookie()) or cookies[-1])
    flow_probe(available=False, error="no NVOF hardware")
    for requires_flow in (True, False):
        before = len(cookies)
        if requires_flow:
            with pytest.raises(gtx.BridgeError):
                gtx.load_bridges(requires_flow=True)
        else:
            # A set that loads must not be the one that leaks, either.
            gtx.load_bridges(requires_flow=False)
            assert all(not cookie.closed for cookie in cookies[before:])
            continue
        assert cookies[before:], "the load never got as far as opening a directory"
        assert all(cookie.closed for cookie in cookies[before:])


class _CountingLive:
    """A DLSS session that only has to answer for its own transfer counters."""

    def transfer_counters(self):
        return {}

    def close(self):
        pass


def _negotiation(uses_flow, mode):
    """What the gate returned, as the run reads it: the shape, not a re-derivation."""
    return {"plan": {"struct": {}, "codec": "av1_nvenc", "av_options": ""},
            "fps": (30000, 1001), "downscaled": False, "uses_flow": uses_flow,
            "plan_driven": False, "copy_pass": False, "guidance_mode": mode,
            "working_scale": 1.0, "declared_color_space": "smpte170m"}


def _zero_frame_run(bridges, uses_flow, mode):
    """The whole report path over a run that was asked for no frames.

    Nothing decodes, so the only thing left to record is what the set behind the
    run is and which motion source it was accepted on - which is the pair of
    claims these two tests are about.
    """
    return gtx.export_gpu_transfer(
        video="in.mp4", live=_CountingLive(), flow_provider=None,
        out_path="out.mkv",
        dims={"src_w": 3840, "src_h": 2160, "live_w": 3840, "live_h": 2160,
              "dst_w": 3840, "dst_h": 2160, "fps": 29.97},
        frames=0, cq=24, with_audio=False, transcode_settings={},
        job_settings={"guidance_mode": mode, "export_backend": "gpu_transfer"},
        negotiation=_negotiation(uses_flow, mode), bridges=bridges)[1]


def test_the_unguided_report_says_the_flow_bridge_was_never_loaded(loaded_bridges):
    """The loading fact, recorded where a reviewer of a finished run will read it.

    Zero flow calls alone would also describe a guided run that decoded nothing,
    so the report carries the fact that no engine was in the process at all.
    """
    report = _zero_frame_run(gtx.load_bridges(requires_flow=False), False, 0)
    assert report["gpu_transfer_nvof_bridge_loaded"] is False
    assert report["gpu_transfer_guidance_mode"] == 0
    assert report["gpu_transfer_uses_flow"] is False
    assert report["gpu_transfer_dlss_flow_calls"] == 0
    assert report["gpu_transfer_dlss_zero_motion_calls"] == 0
    assert loaded_bridges == list(BRIDGES)


def test_the_guided_report_says_the_flow_bridge_was_loaded(loaded_bridges,
                                                           flow_probe):
    """The same code answering the other way, so the field above is a reading."""
    flow_probe()
    report = _zero_frame_run(gtx.load_bridges(requires_flow=True), True, 1)
    assert report["gpu_transfer_nvof_bridge_loaded"] is True
    assert report["gpu_transfer_guidance_mode"] == 1
    assert report["gpu_transfer_uses_flow"] is True
