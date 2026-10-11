"""guidance_mode=0 on the FFmpeg pipe, served by the GPU host, checked without a GPU.

An unguided export on the pipe chain runs on the project GPU host's resident
zero-motion texture. These tests pin what that means at the two places the decision is
observable:

* ProductionExportCore, which must drive process_zero_motion, hand over no flow
  of any kind, reserve no CPU zero-motion plane, reset on a discontinuity, and
  refuse a host that binds no zero-motion texture;
* the queue gate, which must route a guidance_mode=0 job on the pipe to a GPU
  session, open no flow engine for it, and refuse a job that names a DLSS host this
  build does not ship.

The fakes raise if driven through the wrong entry point, which is how "no flow was
computed" and "the frame went to one entry" are proven rather than read off a call
ledger. Presence of process_device_flow is what makes a fake a GPU-host session at
all: the core refuses anything else before frame zero, because there is no second
host whose session it could drive.
"""

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

import dlss_service  # noqa: E402
import pipeline  # noqa: E402
from dlss_errors import DLSSInferenceError  # noqa: E402

SRC_W, SRC_H = 96, 64


class _HostBase:
    """Shape shared by every DLSS session: geometry, liveness, two refusing traps."""

    def __init__(self, w, h, settings=None):
        self._w, self._h = w, h
        self.settings = dict(settings or {})
        self.last_error = None
        self.closed = False
        self.zero_motion_calls = 0
        self.device_flow_calls = 0
        self.motion_calls = 0
        self.update_calls = 0
        self.reset_flags = []
        self.seen_rgba = []
        self.seen_flow = []
        self.seen_out = []

    def is_alive(self):
        return not self.closed

    def close(self):
        self.closed = True

    def _write(self, rgba, out):
        result = np.empty(rgba.shape, dtype=np.uint8) if out is None else out
        result[...] = rgba
        return result

    def process(self, rgba, flow, reset=False, raise_on_error=False, out=None):
        raise AssertionError("the retired motion call was reached: this host has no plain process entry")


class GpuZeroMotionHost(_HostBase):
    """ABI-3 GPU host: an unguided frame is answered by its resident zero texture."""

    def supports_zero_motion(self):
        return True

    def process_device_flow(self, rgba, flow, reset=False, out=None):
        raise AssertionError("a zero-motion frame reached the guided GPU entry")

    def process_zero_motion(self, rgba, reset=False, out=None):
        self.zero_motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.seen_rgba.append(np.array(rgba))
        self.seen_out.append(out)
        result = self._write(rgba, out)
        result[:, :, 0] = np.clip(rgba[:, :, 0].astype(np.int16) + 33, 0, 255)
        return result


class Abi2Host(GpuZeroMotionHost):
    """A GPU host that binds no zero-motion texture."""

    def supports_zero_motion(self):
        return False


class ProbelessHost(GpuZeroMotionHost):
    """A session that cannot answer the capability question at all."""

    supports_zero_motion = None


class FailingZeroMotionHost(GpuZeroMotionHost):
    """A GPU host whose inference fails, reported the way the wrapper reports it."""

    def process_zero_motion(self, rgba, reset=False, out=None):
        self.zero_motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.last_error = "zero-motion DLSS frame failed (simulated)"
        raise DLSSInferenceError(self.last_error)


class DeviceFlowGpuHost(GpuZeroMotionHost):
    """ABI-3 GPU host driven through its guided entry, for the mode-1 contrast."""

    def process_device_flow(self, rgba, flow, reset=False, out=None):
        self.device_flow_calls += 1
        self.reset_flags.append(bool(reset))
        self.seen_flow.append(flow)
        return self._write(rgba, out)

    def process_zero_motion(self, rgba, reset=False, out=None):
        raise AssertionError("a guided GPU frame reached the zero-motion entry")


class RetiredShapedHost(_HostBase):
    """The session shape dlssnr_host.dll's wrapper had: a plain motion call, no GPU entry.

    Kept because the refusal is the point. A caller that still hands the pipeline such a
    session - a cached object from before the migration, or a build that kept the old
    wrapper - has to be stopped by the core, not by a setting.
    """

    def process(self, rgba, flow, reset=False, raise_on_error=False, out=None):
        self.motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.seen_flow.append(None if flow is None else np.array(flow))
        return self._write(rgba, out)

    def process_zero_motion(self, rgba, reset=False, out=None):
        raise AssertionError("a session with no zero-motion entry was asked to run one")

    def update(self, settings):
        self.update_calls += 1
        self.settings.update(settings)


class _Factory:
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


class _DeviceFlowProvider(pipeline.FlowProvider):
    name = "device_stub"

    def get_flow_device(self, frame_idx, cur_bgr, width, height, is_discontinuous=False):
        return {"format": "float32x2", "width": width, "height": height,
                "cuda_context": 1, "device_ptr": 1, "pitch_bytes": width * 8}

    def get_flow(self, *args, **kwargs):
        raise AssertionError("a guided GPU frame used the host-side flow path")


def _source_frame(seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(SRC_H, SRC_W, 3), dtype=np.uint8)


def _core(live, guidance_mode=0, flow_provider=None, **extra):
    settings = {"guidance_mode": guidance_mode, "working_scale": 1.0,
                "residual_pyramid": False, "residual_strength": 1.0}
    settings.update(extra)
    return pipeline.ProductionExportCore(
        src_w=SRC_W, src_h=SRC_H, dst_w=SRC_W, dst_h=SRC_H,
        passes=[live], job_settings=settings, flow_provider=flow_provider)


# --- routing at the frame core -------------------------------------------------

def test_unguided_gpu_job_runs_every_frame_through_process_zero_motion():
    live = GpuZeroMotionHost(SRC_W, SRC_H)
    core = _core(live)
    try:
        records = [core.process_frame(i, _source_frame(i))[1] for i in range(3)]
    finally:
        core.close()

    assert live.zero_motion_calls == 3
    assert live.motion_calls == 0 and live.device_flow_calls == 0
    assert [r["dlss_entry"] for r in records] == ["process_zero_motion"] * 3
    assert all(r["dlss_backend_actual"] == "gpu_flow_d3d12" for r in records)
    assert all(r["depth_source"] == "gpu_zero_texture" for r in records)


def test_unguided_gpu_job_takes_no_flow_provider_and_no_zero_flow_plane():
    live = GpuZeroMotionHost(SRC_W, SRC_H)
    core = _core(live)
    try:
        record = core.process_frame(0, _source_frame())[1]
    finally:
        core.close()

    # The claim is about memory, not a flag: the plane a host-pointer frame is
    # handed does not exist on this branch.
    assert core.flow_provider is None
    assert core.workspace.live_flow is None
    assert record["flow_provider"] == "none"
    assert record["guidance_backend"] == "none"
    assert record["dlss_input_flow_bytes_logical"] == 0
    assert record["dlss_input_depth_bytes_logical"] == 0
    assert core.transfer_diagnostics["flow_plane_allocated"] is False


def test_a_session_that_is_not_the_gpu_host_is_refused_before_frame_zero():
    """The core is the guard, not the setting: an old-shaped session never draws a frame.

    dlss_backend is validated at the queue boundary, but the core takes the session
    object itself. A build or caller that still hands it dlssnr_host.dll's wrapper gets
    a refusal naming the host that ships, before a frame is decoded or a plane reserved.
    """
    live = RetiredShapedHost(SRC_W, SRC_H)
    with pytest.raises(ValueError, match="process_device_flow") as refused:
        _core(live)
    assert "native/dlss_gpu_host.dll" in str(refused.value)
    assert "gpu_flow_d3d12" in str(refused.value)
    assert live.motion_calls == 0 and live.zero_motion_calls == 0
    assert not live.seen_flow, "a refused session was still handed a motion plane"


def test_a_zero_flow_provider_is_never_asked_on_the_gpu_branch(monkeypatch):
    asked = []
    original = pipeline.ZeroFlowProvider.get_flow

    def _tracked(self, *args, **kwargs):
        asked.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pipeline.ZeroFlowProvider, "get_flow", _tracked)
    core = _core(GpuZeroMotionHost(SRC_W, SRC_H))
    try:
        for idx in range(2):
            core.process_frame(idx, _source_frame(idx))
    finally:
        core.close()
    assert asked == []


def test_guided_gpu_frame_still_runs_the_device_flow_entry():
    live = DeviceFlowGpuHost(SRC_W, SRC_H)
    core = _core(live, guidance_mode=1, flow_provider=_DeviceFlowProvider())
    try:
        record = core.process_frame(0, _source_frame())[1]
    finally:
        core.close()

    assert live.device_flow_calls == 1 and live.zero_motion_calls == 0
    assert record["dlss_entry"] == "process_device_flow"
    assert record["flow_provider"] == "device_stub"
    assert core.workspace.live_flow is not None


def test_reset_follows_the_first_frame_and_a_decode_discontinuity():
    live = GpuZeroMotionHost(SRC_W, SRC_H)
    core = _core(live)
    try:
        for idx, discontinuous in ((0, False), (1, False), (2, False), (9, True)):
            core.process_frame(idx, _source_frame(idx), is_discontinuous=discontinuous)
    finally:
        core.close()

    assert live.reset_flags == [True, False, False, True]


def test_an_abi2_gpu_host_is_refused_before_any_frame_and_not_substituted():
    live = Abi2Host(SRC_W, SRC_H)
    with pytest.raises(ValueError, match="ABI 3"):
        _core(live)
    assert live.zero_motion_calls == 0
    assert live.motion_calls == 0


def test_a_host_that_cannot_answer_the_capability_probe_is_refused():
    with pytest.raises(ValueError, match="zero motion texture"):
        _core(ProbelessHost(SRC_W, SRC_H))


def test_a_flow_provider_handed_to_an_unguided_gpu_job_is_refused():
    live = GpuZeroMotionHost(SRC_W, SRC_H)
    with pytest.raises(ValueError, match="no flow provider"):
        _core(live, guidance_mode=0, flow_provider=_DeviceFlowProvider())
    assert live.zero_motion_calls == 0


def test_inference_failure_propagates_and_reaches_no_other_entry():
    live = FailingZeroMotionHost(SRC_W, SRC_H)
    core = _core(live)
    try:
        with pytest.raises(DLSSInferenceError):
            core.process_frame(0, _source_frame())
        assert live.zero_motion_calls == 1
        assert live.motion_calls == 0
        # A frame that failed must not leave the core believing it denoised one.
        with pytest.raises(DLSSInferenceError):
            core.process_frame(1, _source_frame(1))
        assert live.reset_flags == [True, True]
    finally:
        core.close()


def test_the_gpu_host_is_handed_the_frame_the_pipeline_prepared():
    live = GpuZeroMotionHost(SRC_W, SRC_H)
    core = _core(live)
    try:
        frame = _source_frame(7)
        final_bgr, _ = core.process_frame(0, frame)
    finally:
        core.close()

    handed = live.seen_rgba[0]
    assert handed.shape == (SRC_H, SRC_W, 4) and handed.dtype == np.uint8
    assert np.array_equal(handed[:, :, :3], frame[:, :, ::-1]), (
        "the host was not handed the working frame the pipeline prepared")
    assert np.all(handed[:, :, 3] == 255)
    assert isinstance(live.seen_out[0], np.ndarray), (
        "the host was not given the workspace buffer it writes its output into")
    assert final_bgr.shape == (SRC_H, SRC_W, 3)
    # The host's own output reaches the encoder: this fake bumps red by 33, and a
    # frame that came back unchanged would mean the composite read something else.
    assert np.array_equal(final_bgr[:, :, 2],
                          np.clip(frame[:, :, 2].astype(np.int16) + 33, 0, 255))
    assert np.array_equal(final_bgr[:, :, :2], frame[:, :, :2])


# --- routing at the queue gate -------------------------------------------------

def _bare_service(monkeypatch, tmp_path, gpu_cls=GpuZeroMotionHost):
    """A queue with a fake GPU host and a fake export seam, and nothing preset.

    The refusal tests need to add their own jobs in their own order, so the payload and
    the add call stay in the test.
    """
    svc = dlss_service.DLSSService()
    gpu_factory = _Factory(gpu_cls)
    monkeypatch.setitem(sys.modules, "dlss_gpu_engine",
                        types.SimpleNamespace(Live=gpu_factory))
    monkeypatch.setattr(pipeline, "video_info", lambda path: (3, 30.0, SRC_W, SRC_H))
    monkeypatch.setattr(dlss_service, "get_frame_guidance_service",
                        lambda *a, **k: pytest.fail(
                            "an unguided job opened an optical flow engine"))

    def _record(**kwargs):
        with open(kwargs["out_path"], "wb") as fh:
            fh.write(b"\x00" * 4096)
        return kwargs["out_path"]

    monkeypatch.setattr(pipeline, "export_dlss_stream", _record)
    return svc, gpu_factory


def _job_payload(svc, tmp_path, name, **over):
    settings = {"guidance_mode": 0, "working_scale": 1.0, "residual_pyramid": False,
                "export_backend": "legacy"}
    settings.update(over)
    return {"name": name, "input_path": os.path.abspath(__file__),
            "output_path": str(tmp_path / f"{name}.mkv"),
            "settings": {**svc.settings, **settings}}


def _service(monkeypatch, tmp_path, *, gpu_cls=GpuZeroMotionHost, settings=None,
             seed_session=None, seed_kind="legacy"):
    """A queue run on a fake GPU host: no DLL is loaded and no FFmpeg is started.

    The export seam is replaced by a recorder that finishes the job by writing
    the file the queue then verifies, so everything up to and including the
    session that gets handed to the export is the real thing.
    """
    svc = dlss_service.DLSSService()
    job_settings = {"guidance_mode": 0, "dlss_backend": "gpu_flow_d3d12",
                    "export_backend": "legacy", "working_scale": 1.0,
                    "residual_pyramid": False, "flow_engine": "nvof"}
    job_settings.update(settings or {})

    calls = []

    def _record(**kwargs):
        # kind_at_call is read while the export seam is running, which is the only
        # moment the queue's own session ledger is observable for this job.
        seen = dict(kwargs)
        seen["kind_at_call"] = svc._live_session_kind
        seen["session_at_call"] = svc._live_session
        calls.append(seen)
        with open(kwargs["out_path"], "wb") as fh:
            fh.write(b"\x00" * 4096)
        return kwargs["out_path"]

    monkeypatch.setattr(pipeline, "video_info", lambda path: (3, 30.0, SRC_W, SRC_H))
    monkeypatch.setattr(pipeline, "export_dlss_stream", _record)
    monkeypatch.setattr(dlss_service, "get_frame_guidance_service",
                        lambda *a, **k: pytest.fail(
                            "an unguided GPU job opened an optical flow engine"))

    gpu_factory = _Factory(gpu_cls)
    monkeypatch.setitem(sys.modules, "dlss_gpu_engine",
                        types.SimpleNamespace(Live=gpu_factory))

    job = svc.add_job({
        # An input that exists but is never decoded: the export seam below is the
        # last thing this harness lets the queue reach.
        "input_path": os.path.abspath(__file__),
        "output_path": str(tmp_path / "out.mkv"),
        "settings": {**svc.settings, **job_settings},
    })
    if seed_session is not None:
        svc._live_session = seed_session
        svc._live_session_dims = (seed_session._w, seed_session._h)
        svc._live_session_kind = seed_kind

    svc._calls = calls
    return svc, job, gpu_factory


def test_the_queue_routes_an_unguided_pipe_job_to_a_gpu_session(tmp_path, monkeypatch):
    svc, job, gpu_factory = _service(monkeypatch, tmp_path)

    svc._run_queue(None)

    assert job["status"] == "completed", job["error"]
    assert gpu_factory.count == 1
    assert svc._calls[-1]["kind_at_call"] == "gpu_zero_motion"
    assert svc._calls[-1]["session_at_call"] is gpu_factory.built[0]


def test_the_export_seam_gets_the_gpu_session_and_no_flow_provider(tmp_path, monkeypatch):
    svc, job, gpu_factory = _service(monkeypatch, tmp_path)

    svc._run_queue(None)

    assert job["status"] == "completed", job["error"]
    call = svc._calls[-1]
    assert call["live"] is gpu_factory.built[0]
    assert call["flow_provider"] is None
    assert call["job_settings"]["dlss_backend"] == "gpu_flow_d3d12"
    assert call["job_settings"]["export_backend"] == "legacy"
    assert gpu_factory.built[0].closed, "the job left its GPU session open"
    assert svc._live_session is None and svc._live_session_dims is None


def test_auto_resolves_to_the_pipe_and_runs_on_the_gpu_host(tmp_path, monkeypatch):
    """export_backend=auto is the legacy chain, so the host it draws on is the GPU one."""
    svc, job, gpu_factory = _service(
        monkeypatch, tmp_path, settings={"export_backend": "auto"})

    svc._run_queue(None)

    assert job["status"] == "completed", job["error"]
    assert gpu_factory.count == 1
    assert svc._calls[-1]["flow_provider"] is None


def test_the_gpu_transfer_pairing_is_untouched_by_the_pipe_route(tmp_path, monkeypatch):
    svc, job, gpu_factory = _service(
        monkeypatch, tmp_path, settings={"export_backend": "gpu_transfer"})

    svc._run_queue(None)

    assert job["status"] == "completed", job["error"]
    assert gpu_factory.count == 1
    assert svc._calls[-1]["job_settings"]["export_backend"] == "gpu_transfer"


def test_a_guided_gpu_job_still_opens_the_flow_engine_and_keeps_the_device_path(
        tmp_path, monkeypatch):
    opened = []

    def _fake_service(engine, quality):
        opened.append((engine, quality))
        return types.SimpleNamespace(close=lambda: None)

    svc, job, gpu_factory = _service(
        monkeypatch, tmp_path, settings={"guidance_mode": 1})
    monkeypatch.setattr(dlss_service, "get_frame_guidance_service", _fake_service)

    svc._run_queue(None)

    assert job["status"] == "completed", job["error"]
    assert opened == [("nvof", svc.settings.get("flow_quality", "1F"))]
    assert isinstance(svc._calls[-1]["flow_provider"],
                      pipeline.GuidanceServiceFlowProvider)
    assert svc._calls[-1]["kind_at_call"] == "gpu_flow"


def test_an_abi2_host_fails_the_job_by_name_and_opens_nothing_else(tmp_path, monkeypatch):
    svc, job, gpu_factory = _service(monkeypatch, tmp_path, gpu_cls=Abi2Host)

    svc._run_queue(None)

    assert job["status"] == "failed"
    assert "ABI 3" in job["error"]
    assert "native/dlss_gpu_host.dll" in job["error"]
    assert "没有可回退的第二条 DLSS 路径" in job["error"], (
        "the refusal has to say there is nowhere else this job can run")
    assert gpu_factory.built[0].closed
    assert svc._live_session is None and svc._live_session_kind is None


def test_a_job_closes_a_stale_session_of_the_retired_shape(tmp_path, monkeypatch):
    """A cached session the GPU host cannot serve is released, never driven.

    The slot can still hold a non-GPU session - an object handed over before the
    retired host went away is the one way it gets there. The export opens its own
    and closes what it found.
    """
    seeded = RetiredShapedHost(SRC_W, SRC_H)
    svc, job, gpu_factory = _service(monkeypatch, tmp_path, seed_session=seeded)

    svc._run_queue(None)

    assert job["status"] == "completed", job["error"]
    assert gpu_factory.count == 1
    assert seeded.closed
    assert seeded.motion_calls == 0 and seeded.zero_motion_calls == 0
    assert svc._calls[-1]["live"] is gpu_factory.built[0]


def test_a_job_that_names_the_retired_host_is_refused_at_add_time(tmp_path, monkeypatch):
    """The migration asks for a refusal, not a re-route: the job never enters the queue.

    A client written before the old host went away still says dlss_backend=host_pointer.
    Drawing that frame on the GPU host would report a run the caller never requested, so
    the boundary answers with the removed name and the setting to use instead.
    """
    svc, gpu_factory = _bare_service(monkeypatch, tmp_path)

    with pytest.raises(ValueError) as refused:
        svc.add_job(_job_payload(svc, tmp_path, "retired", dlss_backend="host_pointer"))

    message = str(refused.value)
    assert "host_pointer" in message, "the refusal has to name what was asked for"
    assert "dlssnr_host.dll" in message
    assert "gpu_flow_d3d12" in message, "the refusal must carry the migration"
    assert svc.get_jobs() == [], "a refused job must not be queued"
    assert gpu_factory.count == 0, "a refused job must not open a session"


def test_a_refused_job_does_not_stop_the_ones_after_it(tmp_path, monkeypatch):
    """Naming no host and naming the GPU host are one route, and both still run."""
    svc, gpu_factory = _bare_service(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        svc.add_job(_job_payload(svc, tmp_path, "retired", dlss_backend="host_pointer"))

    svc.add_job(_job_payload(svc, tmp_path, "names-nothing"))
    svc.add_job(_job_payload(svc, tmp_path, "names-gpu",
                             dlss_backend="gpu_flow_d3d12"))
    svc._run_queue(None)

    jobs = svc.get_jobs()
    assert [j["status"] for j in jobs] == ["completed", "completed"], \
        [j["error"] for j in jobs]
    assert gpu_factory.count == 2, (
        "each job opens the session it runs on; the retired request opened none")
    assert {j["settings"]["dlss_backend"] for j in jobs} == {"gpu_flow_d3d12"}
    assert all(j["dlss_host"]["backend"] == "gpu_flow_d3d12" for j in jobs)
    assert all(b.closed for b in gpu_factory.built)


def test_a_cancelled_zero_motion_job_closes_the_gpu_session_it_opened(tmp_path, monkeypatch):
    svc, job, gpu_factory = _service(monkeypatch, tmp_path)

    def _cancel_inside_the_export(**kwargs):
        # What the real export does on a user cancel: stop mid-stream, leave the
        # file incomplete, and let the queue's teardown decide the session's fate.
        with open(kwargs["out_path"], "wb") as fh:
            fh.write(b"\x00" * 4096)
        svc._cancel_requested = True
        raise pipeline.JobCancelled("用户已取消任务")

    monkeypatch.setattr(pipeline, "export_dlss_stream", _cancel_inside_the_export)

    svc._run_queue(None)

    assert job["status"] == "cancelled"
    assert gpu_factory.built[0].closed, "a cancelled job left the GPU host open"
    assert svc._live_session is None
    assert svc._live_session_dims is None
