"""A guided export whose motion arrives from CPU memory, on the GPU host, without a GPU.

``ProductionExportCore`` used to know one guided entry on a ``dlss_gpu_engine.Live``
session: ``process_device_flow``, which binds an NVOF CUDA descriptor. A provider
that can only hand back a ``float32`` HxWx2 plane — ``CachedFloFlowProvider``, the
desktop GUI's offline vectors — was refused outright, so that frame could only run on
``dlssnr_host.dll``. The GPU host has ``process_host_motion`` for exactly that shape,
and the old host is gone, so these tests pin the route the core picks, per provider,
with no card attached:

* device descriptor        -> ``process_device_flow``
* CPU float32x2 plane      -> ``process_host_motion``
* guidance off on the host -> ``process_zero_motion``

and what it refuses: a host without the symbol, a provider that can answer neither, a
session that is not the GPU host at all. Every fake raises from the entry it is not the
route's, and no test here claims the plane did not cross PCIe.
"""

import os
import sys
import threading
import types

import cv2  # noqa: E402 (same OpenCV the provider resizes with)
import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import guidance_stubs as gs  # noqa: E402
import pipeline  # noqa: E402
from dlss_errors import DLSSInferenceError  # noqa: E402

SRC_W, SRC_H = 96, 64
LIVE_W, LIVE_H = 48, 32


def _plane(value=3.5):
    """A nonzero displacement plane in the convention the host entry takes."""
    flow = np.zeros((SRC_H, SRC_W, 2), dtype=np.float32)
    xs = np.arange(SRC_W, dtype=np.float32)
    flow[..., 0] = value + xs * 0.03
    flow[..., 1] = -value
    return flow


class _GpuHostBase:
    """A dlss_gpu_engine.Live stand-in: every entry exists, most of them are traps.

    ``process_host_motion`` checks what the real wrapper checks - contiguous float32
    HxWx2 at the session's own extent - so a route that handed over a device
    descriptor, a strided view or the wrong extent would fail here rather than on a
    card that is not attached.
    """

    host_motion_supported = True
    zero_motion_supported = True

    def __init__(self, w, h, settings=None):
        self._w, self._h = w, h
        self.settings = dict(settings or {})
        self.last_error = None
        self.closed = False
        self.host_motion_calls = 0
        self.device_flow_calls = 0
        self.zero_motion_calls = 0
        self.reset_flags = []
        self.seen_motion = []
        self.seen_rgba = []
        self.seen_out = []

    def is_alive(self):
        return not self.closed

    def close(self):
        self.closed = True

    def supports_host_motion(self):
        return bool(self.host_motion_supported)

    def supports_zero_motion(self):
        return bool(self.zero_motion_supported)

    def _write(self, rgba, out):
        result = np.empty(rgba.shape, dtype=np.uint8) if out is None else out
        np.copyto(result, rgba)
        return result

    def process_device_flow(self, rgba, flow, reset=False, out=None):
        self.device_flow_calls += 1
        raise AssertionError("a CPU flow plane reached the device-flow entry")

    def process_zero_motion(self, rgba, reset=False, out=None):
        self.zero_motion_calls += 1
        raise AssertionError("a guided frame reached the zero-motion entry")

    def process_host_motion(self, rgba, motion, reset=False, out=None):
        if not (isinstance(rgba, np.ndarray) and rgba.dtype == np.uint8 and
                rgba.shape == (self._h, self._w, 4) and rgba.flags.c_contiguous):
            raise ValueError("RGBA input must be contiguous uint8 HxWx4")
        if not (isinstance(motion, np.ndarray) and motion.shape == (self._h, self._w, 2)
                and motion.dtype == np.float32 and motion.flags.c_contiguous):
            raise ValueError("motion must be a contiguous float32 HxWx2 plane at the "
                             "DLSS input resolution")
        self.host_motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.seen_rgba.append(np.array(rgba))
        self.seen_motion.append(motion)
        self.seen_out.append(out)
        result = self._write(rgba, out)
        result[:, :, 0] = np.clip(rgba[:, :, 0].astype(np.int16) + 21, 0, 255)
        return result


class _NoHostMotionGpuHost(_GpuHostBase):
    """An ABI-3 host built before the additive host-motion symbol existed."""

    host_motion_supported = False


class _UnaskableHostMotionGpuHost(_GpuHostBase):
    """A session that cannot answer the capability question at all."""

    supports_host_motion = None


class _FailingHostMotionGpuHost(_GpuHostBase):
    """A host whose inference fails, reported the way the wrapper reports it."""

    def process_host_motion(self, rgba, motion, reset=False, out=None):
        self.host_motion_calls += 1
        self.reset_flags.append(bool(reset))
        self.last_error = "host-motion DLSS frame failed (simulated)"
        raise DLSSInferenceError(self.last_error)


class _DeviceFlowGpuHost(_GpuHostBase):
    """The guided device entry, kept honest as the contrast to the new route."""

    def process_device_flow(self, rgba, flow, reset=False, out=None):
        if not isinstance(flow, dict) or flow.get("format") != "float32x2":
            raise ValueError("NVOF flow must be a matching float32x2 CUDA descriptor")
        self.device_flow_calls += 1
        self.reset_flags.append(bool(reset))
        self.seen_motion.append(flow)
        return self._write(rgba, out)

    def process_host_motion(self, rgba, motion, reset=False, out=None):
        raise AssertionError("a device-resident flow descriptor was downloaded to the "
                             "host-motion entry")


class _ZeroMotionGpuHost(_GpuHostBase):
    """The guidance-off branch, whose contract must not move."""

    def process_zero_motion(self, rgba, reset=False, out=None):
        self.zero_motion_calls += 1
        self.reset_flags.append(bool(reset))
        return self._write(rgba, out)

    def process_host_motion(self, rgba, motion, reset=False, out=None):
        raise AssertionError("an unguided frame arrived with motion attached")


class _HostPointerShapedSession:
    """The shape the retired dlssnr_host.dll wrapper had: a plain ``process``, no GPU entry.

    Kept on purpose. The core classifies its session by the entry that binds motion on
    the card rather than by a setting, so this is the object that proves a session
    cannot be handed to the pipeline quietly - and it is why removing the old host's
    loader did not leave an old-shaped session drivable.
    """

    def __init__(self, w, h, settings=None):
        self._w, self._h = w, h
        self.process_calls = 0

    def is_alive(self):
        return True

    def close(self):
        pass

    def process(self, rgba, flow, reset=False, raise_on_error=False, out=None):
        self.process_calls += 1
        return np.empty(rgba.shape, dtype=np.uint8)


class _DeviceDescriptorProvider(pipeline.FlowProvider):
    """What a real NVOF engine hands the GPU host: a descriptor, never an array."""

    name = "device_stub"

    def __init__(self, timing=None):
        self.calls = []
        self._timing = dict(timing or {"guidance_backend": "nvof"})

    def get_flow_device(self, frame_idx, cur_bgr, width, height, is_discontinuous=False):
        self.calls.append((frame_idx, is_discontinuous))
        return {"format": "float32x2", "width": width, "height": height,
                "cuda_context": 1, "device_ptr": 1, "pitch_bytes": width * 8}

    def get_flow(self, *args, **kwargs):
        raise AssertionError("the device route downloaded its flow to a host plane")

    def timing(self):
        return self._timing


class _DeviceOnlyProvider(pipeline.FlowProvider):
    """A provider with no host-side answer: ``get_flow`` stays the base's refusal."""

    name = "device_only_stub"

    def get_flow_device(self, frame_idx, cur_bgr, width, height, is_discontinuous=False):
        return {"format": "float32x2", "width": width, "height": height,
                "cuda_context": 1, "device_ptr": 1, "pitch_bytes": width * 8}


class _SilentProvider(pipeline.FlowProvider):
    """A provider class that implements nothing: neither entry can be fed by it."""


def _flo_dir(tmp_path, frames, extent=None, value=None):
    """Write Middlebury .flo files the way the offline RAFT pass leaves them."""
    flow_dir = str(tmp_path / "clip_flow")
    os.makedirs(flow_dir, exist_ok=True)
    for idx in frames:
        plane = _plane(1.5 + idx) if value is None else value
        if extent:
            plane = cv2.resize(plane, extent, interpolation=cv2.INTER_LINEAR)
        pipeline.write_flo(os.path.join(flow_dir, f"{idx:06d}.flo"), plane)
    return flow_dir


def _core(live, guidance_mode=1, flow_provider=None, **extra):
    settings = {"guidance_mode": guidance_mode, "working_scale": 1.0,
                "residual_pyramid": False, "residual_strength": 1.0,
                "dlss_backend": "gpu_flow_d3d12", "export_backend": "legacy"}
    settings.update(extra)
    return pipeline.ProductionExportCore(
        src_w=SRC_W, src_h=SRC_H, dst_w=SRC_W, dst_h=SRC_H,
        passes=[live], job_settings=settings, flow_provider=flow_provider)


def _source_frame(seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(SRC_H, SRC_W, 3), dtype=np.uint8)


# --- the new route -------------------------------------------------------------

def test_a_cpu_flow_provider_on_the_gpu_host_runs_the_host_motion_entry(tmp_path):
    provider = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, range(3)))
    live = _GpuHostBase(SRC_W, SRC_H)
    core = _core(live, flow_provider=provider)
    try:
        records = [core.process_frame(i, _source_frame(i))[1] for i in range(3)]
    finally:
        core.close()

    assert live.host_motion_calls == 3
    assert live.device_flow_calls == 0 and live.zero_motion_calls == 0
    assert [r["dlss_entry"] for r in records] == ["process_host_motion"] * 3
    assert [r["flow_transport"] for r in records] == ["host_plane_upload"] * 3
    assert all(r["dlss_backend_actual"] == "gpu_flow_d3d12" for r in records)
    assert all(r["depth_source"] == "gpu_zero_texture" for r in records)
    assert all(r["flow_provider"].startswith("cached_flo:") for r in records)


def test_the_plane_handed_over_is_the_real_nonzero_vectors_at_working_dims(tmp_path):
    vectors = _plane(2.0)
    provider = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, [0], value=vectors))
    live = _GpuHostBase(SRC_W, SRC_H)
    core = _core(live, flow_provider=provider)
    try:
        core.process_frame(0, _source_frame())
    finally:
        core.close()

    handed = live.seen_motion[0]
    assert handed.shape == (SRC_H, SRC_W, 2) and handed.dtype == np.float32
    assert handed.flags.c_contiguous
    assert handed.strides == (SRC_W * 8, 8, 4)
    assert np.count_nonzero(handed) > 0, "the host was handed a silent zero plane"
    assert np.array_equal(handed, vectors), "the vectors bound by the host are not the cached ones"
    # Reused workspace, not a fresh allocation per frame: the pipeline hands the
    # provider's out= buffer straight to the host.
    assert handed is core.workspace.live_flow


def test_the_plane_is_asked_at_the_session_extent_not_the_source_extent(tmp_path):
    """A downscaled job guides DLSS at the working size, so the .flo is resized there.

    The cache on disk is bigger than both the source and the session, which is what
    makes the assertion about extent a check of the route rather than of the file.
    """
    provider = pipeline.CachedFloFlowProvider(
        _flo_dir(tmp_path, [0], extent=(SRC_W * 2, SRC_H * 2)))
    live = _GpuHostBase(LIVE_W, LIVE_H)
    core = pipeline.ProductionExportCore(
        src_w=SRC_W, src_h=SRC_H, dst_w=SRC_W, dst_h=SRC_H, passes=[live],
        job_settings={"guidance_mode": 1, "working_scale": 0.5, "residual_pyramid": True,
                      "residual_strength": 1.0, "dlss_backend": "gpu_flow_d3d12",
                      "export_backend": "legacy"},
        flow_provider=provider)
    try:
        record = core.process_frame(0, _source_frame())[1]
    finally:
        core.close()

    assert live.seen_motion[0].shape == (LIVE_H, LIVE_W, 2)
    assert np.count_nonzero(live.seen_motion[0]) > 0
    assert record["actual_working_width"] == LIVE_W
    assert record["dlss_input_flow_bytes_logical"] == LIVE_W * LIVE_H * 2 * 4


def test_reset_follows_the_first_frame_and_a_decode_discontinuity(tmp_path):
    provider = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, [0, 1, 2, 9]))
    live = _GpuHostBase(SRC_W, SRC_H)
    core = _core(live, flow_provider=provider)
    asked = []
    original = provider.get_flow

    def _tracked(*args, **kwargs):
        asked.append(kwargs.get("is_discontinuous"))
        return original(*args, **kwargs)

    provider.get_flow = _tracked
    try:
        for idx, discontinuous in ((0, False), (1, False), (2, False), (9, True)):
            core.process_frame(idx, _source_frame(idx), is_discontinuous=discontinuous)
    finally:
        core.close()

    assert live.reset_flags == [True, False, False, True]
    assert asked == [False, False, False, True], (
        "the provider was not told about the seek, so NVOF-style history reuse is blind")


def test_a_missing_cached_frame_degrades_to_zeros_and_is_counted(tmp_path):
    flow_dir = _flo_dir(tmp_path, [0])
    provider = pipeline.CachedFloFlowProvider(flow_dir)
    live = _GpuHostBase(SRC_W, SRC_H)
    core = _core(live, flow_provider=provider)
    try:
        _final, record = core.process_frame(4, _source_frame())
    finally:
        core.close()

    assert record["flow_missing_frames"] == 1
    assert np.count_nonzero(live.seen_motion[0]) == 0
    # Still the host-motion entry with a zero plane, not a different host: the route
    # is a property of the session and the provider, not of one frame's data.
    assert record["dlss_entry"] == "process_host_motion"


def test_inference_failure_propagates_and_the_session_is_left_alone(tmp_path):
    provider = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, range(2)))
    live = _FailingHostMotionGpuHost(SRC_W, SRC_H)
    core = _core(live, flow_provider=provider)
    try:
        with pytest.raises(DLSSInferenceError):
            core.process_frame(0, _source_frame())
        with pytest.raises(DLSSInferenceError):
            core.process_frame(1, _source_frame(1))
        # A frame that failed did not consume the reset: the next one still resets.
        assert live.reset_flags == [True, True]
    finally:
        core.close()


def test_a_host_without_the_host_motion_symbol_is_refused_not_substituted(tmp_path):
    provider = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, range(1)))
    live = _NoHostMotionGpuHost(SRC_W, SRC_H)
    with pytest.raises(ValueError) as refused:
        _core(live, flow_provider=provider)
    message = str(refused.value)
    assert "dlss_gpu_host_process_host_motion" in message
    assert "no second DLSS host" in message
    assert live.host_motion_calls == 0 and live.device_flow_calls == 0


def test_a_host_that_cannot_answer_the_capability_probe_is_refused(tmp_path):
    provider = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, range(1)))
    with pytest.raises(ValueError, match="host-motion entry point"):
        _core(_UnaskableHostMotionGpuHost(SRC_W, SRC_H), flow_provider=provider)


# --- what the new route must not disturb ---------------------------------------

def test_a_device_descriptor_still_takes_the_device_entry():
    provider = _DeviceDescriptorProvider()
    live = _DeviceFlowGpuHost(SRC_W, SRC_H)
    core = _core(live, flow_provider=provider)
    try:
        record = core.process_frame(0, _source_frame())[1]
    finally:
        core.close()

    assert live.device_flow_calls == 1 and live.host_motion_calls == 0
    assert record["dlss_entry"] == "process_device_flow"
    assert record["flow_transport"] == "nvof_device_descriptor"
    assert isinstance(live.seen_motion[0], dict)


def test_a_provider_that_can_answer_both_is_not_downloaded_to_the_host_entry(tmp_path):
    """GuidanceServiceFlowProvider has both methods; the resident one has to win."""
    both = _DeviceDescriptorProvider()
    both.get_flow = lambda *a, **k: pytest.fail(
        "a device-resident NVOF frame was routed through its host-side getter")
    live = _DeviceFlowGpuHost(SRC_W, SRC_H)
    core = _core(live, flow_provider=both)
    try:
        core.process_frame(0, _source_frame())
    finally:
        core.close()
    assert live.device_flow_calls == 1 and live.host_motion_calls == 0


def test_the_guidance_off_branch_still_hands_the_host_no_motion_at_all():
    live = _ZeroMotionGpuHost(SRC_W, SRC_H)
    core = _core(live, guidance_mode=0, flow_provider=None)
    try:
        record = core.process_frame(0, _source_frame())[1]
    finally:
        core.close()

    assert live.zero_motion_calls == 1 and live.host_motion_calls == 0
    assert core.workspace.live_flow is None
    assert record["dlss_entry"] == "process_zero_motion"
    assert record["flow_transport"] == "resident_zero_texture"
    assert record["dlss_input_flow_bytes_logical"] == 0


def test_a_session_that_is_not_the_gpu_host_is_refused_before_frame_zero(tmp_path):
    """The retired host's shape cannot be driven, whatever a caller asks for.

    The core classifies its session by the entry that binds motion on the card, so a
    session with only ``process`` is refused rather than run. That is what keeps a
    hand-built caller, a stale import or an archival script from drawing a frame on a
    host this build does not ship - and it holds for the unguided branch too, which
    never reaches a flow provider to argue with.
    """
    provider = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, range(1)))
    live = _HostPointerShapedSession(SRC_W, SRC_H)
    with pytest.raises(ValueError, match="process_device_flow") as refused:
        pipeline.ProductionExportCore(
            src_w=SRC_W, src_h=SRC_H, dst_w=SRC_W, dst_h=SRC_H, passes=[live],
            job_settings={"guidance_mode": 1, "working_scale": 1.0,
                          "residual_pyramid": False},
            flow_provider=provider)
    assert "native/dlss_gpu_host.dll" in str(refused.value)
    assert live.process_calls == 0

    with pytest.raises(ValueError, match="process_device_flow"):
        pipeline.ProductionExportCore(
            src_w=SRC_W, src_h=SRC_H, dst_w=SRC_W, dst_h=SRC_H, passes=[live],
            job_settings={"guidance_mode": 0, "working_scale": 1.0,
                          "residual_pyramid": False})
    assert live.process_calls == 0


def test_a_device_only_provider_on_the_gpu_host_takes_the_device_entry():
    live = _DeviceFlowGpuHost(SRC_W, SRC_H)
    core = _core(live, flow_provider=_DeviceOnlyProvider())
    try:
        record = core.process_frame(0, _source_frame())[1]
    finally:
        core.close()
    assert live.device_flow_calls == 1 and live.host_motion_calls == 0
    assert record["flow_transport"] == "nvof_device_descriptor"


def test_a_provider_that_answers_neither_is_refused_before_frame_zero(tmp_path):
    live = _GpuHostBase(SRC_W, SRC_H)
    with pytest.raises(ValueError, match="neither get_flow nor get_flow_device"):
        _core(live, flow_provider=_SilentProvider())
    assert live.host_motion_calls == 0
    assert live.device_flow_calls == 0
    assert live.zero_motion_calls == 0


def test_an_unsupported_guidance_mode_is_still_refused_by_the_core(tmp_path):
    provider = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, range(1)))
    with pytest.raises(ValueError, match="guidance_mode"):
        _core(_GpuHostBase(SRC_W, SRC_H), guidance_mode=2, flow_provider=provider)


# --- transfer accounting --------------------------------------------------------

def test_transfer_diagnostics_separate_a_cpu_upload_from_an_nvof_download(tmp_path):
    """The two guided GPU routes move different bytes and must not read alike.

    A cached plane crosses host->device once per frame and no NVOF is involved, so
    the bridge's D2H counters are absent rather than zero; a device-resident
    descriptor crosses nothing and reports the bridge's own measured zero.
    """
    cached = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, range(1)))
    gpu_live = _GpuHostBase(SRC_W, SRC_H)
    cpu_core = _core(gpu_live, flow_provider=cached)
    try:
        cpu_record = cpu_core.process_frame(0, _source_frame())[1]
    finally:
        cpu_core.close()

    nvof = _DeviceDescriptorProvider(timing={
        "guidance_backend": "nvof", "nvof_d2h_copy_count_delta": 0,
        "nvof_d2h_bytes_delta": 0, "nvof_h2d_copy_count_delta": 1})
    device_live = _DeviceFlowGpuHost(SRC_W, SRC_H)
    device_core = _core(device_live, flow_provider=nvof)
    try:
        device_record = device_core.process_frame(0, _source_frame())[1]
    finally:
        device_core.close()

    assert cpu_record["flow_transport"] == "host_plane_upload"
    assert cpu_record["dlss_input_flow_bytes_logical"] == SRC_W * SRC_H * 8
    # No engine ran, so there is no downloaded flow to count: absent, not zero.
    assert cpu_record["nvof_d2h_copy_count"] is None
    assert cpu_record["nvof_d2h_bytes_measured"] is None
    assert cpu_record["guidance_backend"] == "cached_flo"

    assert device_record["flow_transport"] == "nvof_device_descriptor"
    assert device_record["dlss_input_flow_bytes_logical"] == SRC_W * SRC_H * 8
    assert device_record["nvof_d2h_copy_count"] == 0
    assert device_record["guidance_backend"] == "nvof"
    # Neither figure is a measured transfer: the host has no motion counter, so a
    # run that crossed PCIe would look exactly like one that did not. Nothing here
    # may be read as a zero-copy claim, and this pins that the measured fields stay
    # empty on both guided GPU routes.
    assert cpu_record["dlss_gpu_transfer_bytes_measured"] is None
    assert device_record["dlss_gpu_transfer_bytes_measured"] is None


def test_the_route_is_reported_in_the_run_diagnostics_not_only_per_frame(tmp_path):
    provider = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, range(1)))
    core = _core(_GpuHostBase(SRC_W, SRC_H), flow_provider=provider)
    try:
        diagnostics = dict(core.transfer_diagnostics)
    finally:
        core.close()

    assert diagnostics["dlss_entry"] == "process_host_motion"
    assert diagnostics["flow_transport"] == "host_plane_upload"
    assert diagnostics["dlss_backend_actual"] == "gpu_flow_d3d12"
    assert diagnostics["flow_plane_allocated"] is True
    assert diagnostics["transfer_mode_actual"] == "legacy"


def test_a_cpu_flow_provider_does_not_claim_the_reduced_transfer_order(tmp_path):
    """Reduced is a claim about the frame copies that went away, and this route has
    no bearing on it: the reduction is still negotiated from its own capabilities."""
    provider = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, range(1)))
    live = _GpuHostBase(LIVE_W, LIVE_H)
    core = pipeline.ProductionExportCore(
        src_w=SRC_W, src_h=SRC_H, dst_w=SRC_W, dst_h=SRC_H, passes=[live],
        job_settings={"guidance_mode": 1, "working_scale": 0.5, "residual_pyramid": True,
                      "residual_backend": "python", "dlss_backend": "gpu_flow_d3d12"},
        flow_provider=provider)
    try:
        diagnostics = dict(core.transfer_diagnostics)
    finally:
        core.close()

    assert diagnostics["transfer_mode_actual"] == "legacy"
    assert "fused_working_bgr_and_rgba_store" in diagnostics["transfer_mode_fallback_reason"]


# --- wiring through pipeline.export_dlss_stream ---------------------------------

def test_export_dlss_stream_routes_a_cached_flow_job_to_the_host_motion_entry(tmp_path,
                                                                              monkeypatch):
    """The function the callers actually use, end to end, with a fake encoder.

    A pipe run on a GPU session and a cached .flo directory: what leaves this
    function is the same route the core decided, reported in the diagnostics the
    queue and the UI read.
    """
    frames = [np.full((SRC_H, SRC_W, 3), 40 + i, dtype=np.uint8) for i in range(3)]
    monkeypatch.setattr(pipeline, "video_info", lambda path: (3, 30.0, SRC_W, SRC_H))
    monkeypatch.setattr(pipeline, "verify_export_output",
                        lambda path, count, **kw: count)

    def _fake_frames(video, **kwargs):
        for idx, fr in enumerate(frames):
            yield idx, fr

    def _fake_command(**kwargs):
        return [sys.executable, "-c",
                "import sys; open(sys.argv[1], 'wb').write(sys.stdin.buffer.read())",
                kwargs["out_path"]]

    monkeypatch.setattr(pipeline, "iter_frames", _fake_frames)
    monkeypatch.setattr(pipeline, "build_transcode_command", _fake_command)

    provider = pipeline.CachedFloFlowProvider(_flo_dir(tmp_path, range(3)))
    live = _GpuHostBase(SRC_W, SRC_H)
    diagnostics = {}
    out_path = str(tmp_path / "out.mkv")
    result = pipeline.export_dlss_stream(
        video=str(tmp_path / "clip.mp4"), live=live, frames=3, fps=30.0,
        flow_provider=provider, out_path=out_path,
        transcode_settings={"codec": "libx264"},
        job_settings={"guidance_mode": 1, "working_scale": 1.0, "residual_pyramid": False,
                      "dlss_backend": "gpu_flow_d3d12", "export_backend": "legacy"},
        diagnostics_out=diagnostics)

    assert result == out_path
    assert live.host_motion_calls == 3
    assert diagnostics["dlss_entry"] == "process_host_motion"
    assert diagnostics["flow_transport"] == "host_plane_upload"
    assert diagnostics["export_backend_actual"] == "legacy_pipe"
    dst_w, dst_h = pipeline.resolve_output_dims(SRC_W, SRC_H, {"codec": "libx264"})
    assert os.path.getsize(out_path) == 3 * dst_w * dst_h * 3, (
        "the encoder was not fed the rendered frames")


def test_the_service_still_hands_its_guided_gpu_job_a_device_provider(tmp_path, monkeypatch):
    """No service routing moved: an explicit guided GPU job still gets NVOF's provider."""
    import dlss_service

    svc = dlss_service.DLSSService()
    seen = {}

    def _record(**kwargs):
        seen.update(kwargs)
        with open(kwargs["out_path"], "wb") as fh:
            fh.write(b"\x00" * 4096)
        return kwargs["out_path"]

    monkeypatch.setattr(pipeline, "video_info", lambda path: (3, 30.0, SRC_W, SRC_H))
    monkeypatch.setattr(pipeline, "export_dlss_stream", _record)
    monkeypatch.setattr(dlss_service, "get_frame_guidance_service",
                        lambda engine, quality: types.SimpleNamespace(close=lambda: None))
    monkeypatch.setitem(sys.modules, "dlss_gpu_engine",
                        types.SimpleNamespace(Live=_GpuHostBase))

    job = svc.add_job({
        "input_path": os.path.abspath(__file__),
        "output_path": str(tmp_path / "out.mkv"),
        "settings": {**svc.settings, "guidance_mode": 1, "flow_engine": "nvof",
                     "dlss_backend": "gpu_flow_d3d12", "export_backend": "legacy",
                     "working_scale": 1.0, "residual_pyramid": False},
    })
    svc._run_queue(None)

    assert job["status"] == "completed", job["error"]
    assert isinstance(seen["flow_provider"], pipeline.GuidanceServiceFlowProvider)
    assert svc._live_session is None, "the job left its GPU session registered"


def test_the_shared_stub_live_is_routed_like_the_gpu_host():
    """Every other contract that builds a core off ``guidance_stubs`` inherits its route.

    If the shared stub ever stops looking like the only host this build drives, the
    contracts written against it start describing a run that cannot happen; this says
    out loud which one it is pinned to.
    """
    live = gs.StubLive(SRC_W, SRC_H)
    core = pipeline.ProductionExportCore(
        src_w=SRC_W, src_h=SRC_H, dst_w=SRC_W, dst_h=SRC_H, passes=[live],
        job_settings={"guidance_mode": 1, "working_scale": 1.0, "residual_pyramid": False},
        flow_provider=ZeroPlaneProvider())
    try:
        record = core.process_frame(0, np.zeros((SRC_H, SRC_W, 3), np.uint8))[1]
    finally:
        core.close()
    assert record["dlss_entry"] == "process_host_motion"
    assert record["flow_transport"] == "host_plane_upload"
    assert record["depth_source"] == "gpu_zero_texture"
    assert record["dlss_backend_actual"] == "gpu_flow_d3d12"
    assert record["dlss_host_dll"] == "dlss_gpu_host.dll"


class ZeroPlaneProvider(pipeline.FlowProvider):
    name = "zero_plane_stub"

    def get_flow(self, frame_idx, cur_bgr, prev_bgr, width, height,
                 is_discontinuous=False, out=None, input_format="rgba8_host"):
        plane = out if out is not None else np.zeros((height, width, 2), np.float32)
        plane[...] = 0.0
        return plane


# --- the span between the two GPU entries -----------------------------------------

class _LockAwareDeviceProvider(_DeviceDescriptorProvider):
    """The device route, asked whether the session lock was held while it ran."""

    def __init__(self, lock, seen, timing=None):
        super().__init__(timing=timing)
        self._lock, self._seen = lock, seen

    def get_flow_device(self, frame_idx, cur_bgr, width, height, is_discontinuous=False):
        self._seen.append(("flow", self._lock.locked()))
        return super().get_flow_device(frame_idx, cur_bgr, width, height,
                                       is_discontinuous=is_discontinuous)


class _LockAwareDeviceHost(_DeviceFlowGpuHost):
    """The device entry, with the same question asked of it and an optional failure."""

    def __init__(self, w, h, lock, seen, fail=False):
        super().__init__(w, h)
        self._lock, self._seen, self._fail = lock, seen, fail

    def process_device_flow(self, rgba, flow, reset=False, out=None):
        self._seen.append(("inference", self._lock.locked()))
        if self._fail:
            self.last_error = "device-flow DLSS frame failed (simulated)"
            raise DLSSInferenceError(self.last_error)
        return super().process_device_flow(rgba, flow, reset=reset, out=out)


def _device_core(live, provider, lock):
    return pipeline.ProductionExportCore(
        src_w=SRC_W, src_h=SRC_H, dst_w=SRC_W, dst_h=SRC_H, passes=[live],
        job_settings={"guidance_mode": 1, "working_scale": 1.0, "residual_pyramid": False,
                      "residual_strength": 1.0, "dlss_backend": "gpu_flow_d3d12",
                      "export_backend": "legacy"},
        flow_provider=provider, lock=lock)


def test_a_guided_frame_holds_the_session_lock_from_flow_through_inference():
    """The descriptor's lifetime, not the inference, is what the lock protects.

    ``get_flow_device`` hands back a descriptor the NVOF engine owns and its next
    execute destroys, so a caller that computed flow between the two entries would
    leave this frame guided by memory that is already gone and quietly rewrite the
    shared engine's temporal history. The span that has to be atomic is therefore
    flow→inference, which is what the two probes below read. The lock must also be
    back on the shelf when the frame is done, or the next caller waits 15 s for
    nothing.
    """
    lock = threading.Lock()
    seen = []
    provider = _LockAwareDeviceProvider(lock, seen)
    live = _LockAwareDeviceHost(SRC_W, SRC_H, lock, seen)
    core = _device_core(live, provider, lock)
    try:
        record = core.process_frame(0, _source_frame())[1]
    finally:
        core.close()

    assert seen == [("flow", True), ("inference", True)]
    assert record["dlss_entry"] == "process_device_flow"
    assert not lock.locked()


def test_a_failed_inference_does_not_leave_the_session_lock_held():
    """A frame that fails midway still gives the session back."""
    lock = threading.Lock()
    seen = []
    provider = _LockAwareDeviceProvider(lock, seen)
    live = _LockAwareDeviceHost(SRC_W, SRC_H, lock, seen, fail=True)
    core = _device_core(live, provider, lock)
    try:
        with pytest.raises(DLSSInferenceError):
            core.process_frame(0, _source_frame())
    finally:
        core.close()

    assert seen == [("flow", True), ("inference", True)]
    assert not lock.locked()
