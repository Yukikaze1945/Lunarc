# -*- coding: utf-8 -*-
"""Gate: the two transfer orders are two different sets of bytes, and the run says which ran.

ProductionExportCore negotiates between them at construction. Legacy converts the working
plane to RGBA for the DLSS host, hands the flow engine a second CPU-reordered copy of that
plane, and converts the host's RGBA output back to BGR before the composite reads it.
Reduced keeps none of those three copies: one filter pass publishes both plane orders, the
BGR plane goes to the bridge as the pipeline holds it, and the composite reads the host's
output where the host wrote it.

What makes a "does not convert" claim checkable is that the number comes from the route the
conversion would have taken. Two of these gates therefore install one recorder at both
entry points into OpenCV - the module attribute and the profiler's own bound reference,
because the seam holds the second and patching only the first would leave the production
conversions unobserved - and each gate runs legacy in the same test as the control. A zero
alongside a control that converts is a deleted copy; a zero without one is a broken
recorder.

The last gate is the one that leaves the CPU: it runs the production core with a real NVOF
engine and asserts, from the bridge's own counters, that three bytes per pixel went up and
the CPU-built RGBA copy count is zero. It is skipped only when the probe says there is no
NVOF hardware; a GPU whose bridge exports no BGR entry point is a failure, because that
entry point is what the shipped binary is supposed to provide.
"""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import guidance_stubs as gs  # noqa: E402
import nvof_timing_semantics as timing  # noqa: E402
import optical_flow as of  # noqa: E402
import pipeline  # noqa: E402
import transfer_ledger as tl  # noqa: E402
import transfer_profiler as tp  # noqa: E402

SRC_W, SRC_H = 64, 48
LIVE_W, LIVE_H = 32, 24

PRESENT, PROBE = timing.hardware_present()
requires_nvof = pytest.mark.skipif(
    not PRESENT, reason=f"no NVIDIA optical flow hardware to feed (probe said {PROBE})")


class ConversionRecorder:
    """Every cv2.cvtColor call, whoever made it, with the code it was given."""

    def __init__(self):
        self._real = cv2.cvtColor
        self.codes = []

    def __call__(self, src, code, dst=None, **kwargs):
        self.codes.append(int(code))
        if dst is not None:
            return self._real(src, int(code), dst=dst)
        return self._real(src, int(code), **kwargs)

    def count(self, code):
        return sum(1 for seen in self.codes if seen == int(code))


@pytest.fixture
def conversions(monkeypatch):
    recorder = ConversionRecorder()
    monkeypatch.setattr(cv2, "cvtColor", recorder)
    monkeypatch.setattr(tp, "_cvtColor", recorder)
    tp.reset_conversions()
    return recorder


def _core(mode, live_w=LIVE_W, live_h=LIVE_H, src_w=SRC_W, src_h=SRC_H, flow_service=None,
          flow_provider=None, backend=None):
    if flow_provider is None and flow_service is None:
        flow_service = gs.StubFlowService()
    return gs.make_core(
        live_w=live_w, live_h=live_h, src_w=src_w, src_h=src_h, guidance_mode=1,
        flow_service=flow_service, flow_provider=flow_provider,
        passes=[gs.StubLive(live_w, live_h)],
        job_settings_extra={"transfer_mode": mode, "residual_backend": backend})


def _one_frame(core, idx=0):
    frame = np.full((core.src_h, core.src_w, 3), 90, dtype=np.uint8)
    frame[:, :core.src_w // 3] = 200
    return core.process_frame(idx, frame, is_discontinuous=(idx == 0))


# ---------------------------------------------------------------------------
# the two §38 gates: the reduced order makes no channel conversion
# ---------------------------------------------------------------------------

def test_production_reduced_path_does_not_call_cv2_bgr_to_rgba(conversions):
    """No BGR2RGBA anywhere in a reduced frame - with legacy as the control below it.

    The recorder sits on the global entry point, so this counts conversions the pipeline
    makes and conversions a helper makes, not only the ones that go through the seam.
    """
    legacy_core = _core("legacy")
    try:
        conversions.codes = []
        _out, legacy_record = _one_frame(legacy_core)
    finally:
        legacy_core.close()
    legacy_conversions = conversions.count(cv2.COLOR_BGR2RGBA)

    conversions.codes = []
    reduced_core = _core("reduced")
    try:
        _out, reduced_record = _one_frame(reduced_core)
    finally:
        reduced_core.close()

    assert legacy_conversions >= 1, (
        "the control frame made no BGR2RGBA conversion, so the zero under it is a recorder "
        "that is not wired up rather than a copy that was deleted")
    assert conversions.count(cv2.COLOR_BGR2RGBA) == 0, (
        f"a reduced frame still calls cv2.cvtColor(BGR2RGBA) "
        f"{conversions.count(cv2.COLOR_BGR2RGBA)} times, so the DLSS input is still built "
        "by a second pass over the pixels")
    # The same frame's own counters, read from the seam the ledger rows point at.
    assert reduced_record["cpu_bgr_to_rgba@dlss_input_calls"] == 0
    assert reduced_record["cpu_bgr_to_rgba@nvof_input_calls"] == 0
    assert reduced_record["cpu_bgr_to_rgba_calls"] == 0
    assert reduced_record["cpu_color_conversion_output_bytes"] == 0
    # The plane the host is handed still exists at full size: what went away is the
    # conversion, not the input.
    assert reduced_record["dlss_input_color_bytes_logical"] == LIVE_W * LIVE_H * 4
    assert legacy_record["cpu_bgr_to_rgba@dlss_input_calls"] == 1
    assert legacy_record["cpu_bgr_to_rgba@dlss_input_output_bytes"] == LIVE_W * LIVE_H * 4


def test_production_reduced_path_does_not_call_cv2_rgba_to_bgr(conversions):
    """The composite reads the host's RGBA output in place, so nothing reorders it first."""
    legacy_core = _core("legacy")
    try:
        conversions.codes = []
        _out, legacy_record = _one_frame(legacy_core)
    finally:
        legacy_core.close()
    legacy_conversions = conversions.count(cv2.COLOR_RGBA2BGR)

    conversions.codes = []
    reduced_core = _core("reduced")
    try:
        _out, reduced_record = _one_frame(reduced_core)
    finally:
        reduced_core.close()

    assert legacy_conversions >= 1, (
        "the control frame made no RGBA2BGR conversion, so the zero under it proves nothing")
    assert conversions.count(cv2.COLOR_RGBA2BGR) == 0, (
        "a reduced frame still converts the host's output to BGR before reconstructing")
    assert reduced_record["cpu_rgba_to_bgr@residual_input_calls"] == 0
    assert reduced_record["cpu_rgba_to_bgr@residual_input_output_bytes"] == 0
    assert reduced_record["cpu_rgba_to_bgr_calls"] == 0
    # The reduced order also stops reserving the buffer that conversion used to write, so
    # the size the legacy record reports has no allocation behind it any more.
    assert legacy_record["cpu_rgba_to_bgr@residual_input_calls"] == 1
    assert legacy_record["cpu_rgba_to_bgr@residual_input_output_bytes"] == LIVE_W * LIVE_H * 3


def test_only_the_reduced_run_leaves_the_bgr_output_buffer_unreserved():
    legacy_core, reduced_core = _core("legacy"), _core("reduced")
    try:
        assert legacy_core.workspace.dlss_out_bgr is not None
        assert reduced_core.workspace.dlss_out_bgr is None
        assert legacy_core.transfer_reduced is False
        assert reduced_core.transfer_reduced is True
    finally:
        legacy_core.close()
        reduced_core.close()


# ---------------------------------------------------------------------------
# the orders must differ in the bytes that move, not in the picture
# ---------------------------------------------------------------------------

def test_legacy_and_reduced_produce_identical_output():
    """Same frames, same settings, same pass: byte-for-byte the same exported picture.

    The claim this protects is the one every saving in this round rests on: a transfer order
    may change where pixels get reordered, never what they become.
    """
    rng = np.random.default_rng(21)
    frames = [rng.integers(0, 256, size=(SRC_H, SRC_W, 3), dtype=np.uint8) for _ in range(6)]
    results = {}
    for mode in ("legacy", "reduced"):
        core = _core(mode)
        try:
            results[mode] = []
            for index, frame in enumerate(frames):
                out, _record = core.process_frame(index, frame, is_discontinuous=(index == 0))
                results[mode].append(np.array(out))
        finally:
            core.close()
    for index, (a, b) in enumerate(zip(results["legacy"], results["reduced"])):
        assert a.shape == b.shape
        assert np.array_equal(a, b), (
            f"frame {index}: the two transfer orders disagree by up to "
            f"{int(np.abs(a.astype(int) - b.astype(int)).max())} per channel, so one of "
            "them is not moving the same pixels")
    # Identical output from a frame the residual has nothing to add would pass for the
    # wrong reason.
    assert float(results["reduced"][-1].astype(np.int32).std()) > 0.0


# ---------------------------------------------------------------------------
# negotiation: auto falls back and says why, an explicit request refuses
# ---------------------------------------------------------------------------

class UnansweredFlowProvider(pipeline.FlowProvider):
    """A provider that never answers the input-format question."""

    name = "unanswered"

    def get_flow(self, frame_idx, cur_bgr, prev_bgr, width, height,
                 is_discontinuous=False, out=None, input_format="rgba8_host"):
        return np.zeros((height, width, 2), dtype=np.float32)


def test_auto_takes_the_reduced_order_only_when_every_capability_is_there():
    core = _core("auto")
    try:
        diagnostics = core.transfer_diagnostics
        assert diagnostics["transfer_mode_actual"] == "reduced"
        assert diagnostics["transfer_mode_fallback_reason"] is None
        assert all(diagnostics["transfer_capabilities"].values())
    finally:
        core.close()


def test_auto_falls_back_to_legacy_and_names_the_missing_capability():
    """A provider that has not answered is offered legacy, and the reason says whose answer."""
    core = _core("auto", flow_provider=UnansweredFlowProvider())
    try:
        diagnostics = core.transfer_diagnostics
        assert diagnostics["transfer_mode_requested"] == "auto"
        assert diagnostics["transfer_mode_actual"] == "legacy"
        assert diagnostics["transfer_mode_fallback_reason"] == (
            "missing_capability: flow_input_needs_no_rgba_copy")
        assert diagnostics["transfer_capabilities"]["flow_input_needs_no_rgba_copy"] is False
    finally:
        core.close()


def test_auto_falls_back_when_the_residual_backend_cannot_do_the_fused_pass():
    core = _core("auto", backend="python")
    try:
        diagnostics = core.transfer_diagnostics
        assert diagnostics["transfer_mode_actual"] == "legacy"
        missing = diagnostics["transfer_mode_fallback_reason"]
        assert "fused_working_bgr_and_rgba_store" in missing
        assert "rgba_read_in_place_by_reconstruct" in missing
    finally:
        core.close()


def test_explicit_reduced_refuses_rather_than_running_legacy():
    """The mode is a claim about bytes, so a half-available one has to be an error."""
    with pytest.raises(RuntimeError) as refused:
        _core("reduced", backend="python")
    message = str(refused.value)
    assert "fused_working_bgr_and_rgba_store" in message
    assert "Refusing rather than running the legacy order" in message

    with pytest.raises(RuntimeError):
        _core("reduced", flow_provider=UnansweredFlowProvider())

    with pytest.raises(RuntimeError):
        _core("reduced", live_w=SRC_W, live_h=SRC_H)


def test_an_explicit_legacy_request_is_recorded_as_a_choice_not_a_fallback():
    core = _core("legacy", backend="python")
    try:
        diagnostics = core.transfer_diagnostics
        assert diagnostics["transfer_mode_actual"] == "legacy"
        assert diagnostics["transfer_mode_fallback_reason"] == "requested_legacy"
    finally:
        core.close()


def test_an_unknown_mode_raises_instead_of_being_read_as_auto():
    with pytest.raises(ValueError):
        _core("reducd")


def test_a_provider_that_cannot_be_asked_fails_closed():
    """The base answer is True: silence gets legacy, not the reduction."""
    assert pipeline.FlowProvider().requires_rgba_input() is True
    assert pipeline.ZeroFlowProvider().requires_rgba_input() is False
    assert pipeline.CachedFloFlowProvider("").requires_rgba_input() is False
    # A guidance mode that never reaches an engine has no input format to be handed.
    assert pipeline.GuidanceServiceFlowProvider(gs.StubFlowService(),
                                                guidance_mode=0).requires_rgba_input() is False
    assert pipeline.GuidanceServiceFlowProvider(gs.StubFlowService(),
                                                guidance_mode=1).requires_rgba_input() is False
    old_bridge = gs.StubFlowService(nvof_rgba_input_only=True)
    assert pipeline.GuidanceServiceFlowProvider(old_bridge,
                                                guidance_mode=1).requires_rgba_input() is True


def test_the_reduced_order_asks_the_service_for_the_bgr_plane():
    """Which input was requested is recorded by the thing that receives it."""
    for mode, expected in (("auto", "bgr24_host"), ("legacy", "rgba8_host")):
        service = gs.StubFlowService()
        core = _core(mode, flow_service=service)
        try:
            _one_frame(core)
        finally:
            core.close()
        assert core.transfer_diagnostics["transfer_mode_actual"] == (
            "reduced" if expected == "bgr24_host" else "legacy")
        assert service.input_formats == [expected]


def test_which_order_ran_is_in_every_frame_record():
    for mode, expected in (("auto", "reduced"), ("legacy", "legacy")):
        core = _core(mode)
        try:
            records = [_one_frame(core, idx=i)[1] for i in range(3)]
        finally:
            core.close()
        assert [r["transfer_mode_actual"] for r in records] == [expected] * 3
        assert {r["transfer_mode_requested"] for r in records} == {mode}
        for record in records:
            assert "transfer_mode_fallback_reason" in record
            assert "nvof_input_convert_wait_ms" in record


def test_a_cancelled_export_still_reports_which_order_it_ran_in():
    """The mode is decided at construction, so a run that never finishes still says what ran."""
    source = os.path.join(_ROOT, "bench_out", "test_transfer_mode_src.mp4")
    os.makedirs(os.path.dirname(source), exist_ok=True)
    if not os.path.exists(source):
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                        "testsrc=size=1920x1080:rate=30", "-t", "1", "-pix_fmt", "yuv420p",
                        "-c:v", "libx264", source],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    class PassthroughLive:
        """A GPU-host session that draws nothing but copies; guidance is off here."""
        _w, _h = 960, 540

        def supports_zero_motion(self):
            return True

        def supports_host_motion(self):
            return True

        def process_device_flow(self, rgba, flow, reset=False, out=None):
            raise AssertionError("an unguided frame reached the device-flow entry")

        def process_zero_motion(self, rgba, reset=False, out=None):
            if out is not None:
                np.copyto(out, rgba[:, :, :out.shape[2]])
                return out
            return rgba

    diagnostics = {}
    seen = []

    def _progress(current, total, message):
        seen.append(current)

    with pytest.raises(pipeline.JobCancelled):
        pipeline.export_dlss_stream(
            video=source, live=PassthroughLive(), frames=3, fps=30, progress=_progress,
            cancel=lambda: len(seen) >= 1, out_path=source + ".out.mkv",
            transcode_settings={"codec": "libx264", "decode_mode": "cpu"},
            job_settings={"working_scale": 0.5, "guidance_mode": 0, "residual_pyramid": True,
                          "residual_strength": 1.0, "residual_backend": "auto"},
            diagnostics_out=diagnostics)

    assert diagnostics["transfer_mode_requested"] == "auto"
    assert "transfer_mode_actual" in diagnostics
    assert diagnostics["transfer_mode_actual"] in tl.TRANSFER_MODES
    assert diagnostics["transfer_capabilities"]


# ---------------------------------------------------------------------------
# the ledger describes one order at a time
# ---------------------------------------------------------------------------

def test_a_ledger_names_the_order_it_describes():
    working = {"working_w": tl.WORKING_W, "working_h": tl.WORKING_H}
    legacy = tl.build_ledger(transfer_mode="legacy", **working)
    reduced = tl.build_ledger(transfer_mode="reduced", **working)
    assert legacy["transfer_mode"] == "legacy" and reduced["transfer_mode"] == "reduced"
    # Same rows in the same order: what changes is who writes the buffer and how many
    # bytes cross, never whether the audit looked at it.
    assert [r["name"] for r in legacy["rows"]] == [r["name"] for r in reduced["rows"]]

    legacy_rows = {r["name"]: r for r in legacy["rows"]}
    reduced_rows = {r["name"]: r for r in reduced["rows"]}
    assert legacy_rows["nvof_input_upload"]["format"] == "rgba8"
    assert legacy_rows["nvof_input_upload"]["payload_bytes"] \
        == tl.PAYLOAD_REFERENCE["working_rgba8"]
    assert reduced_rows["nvof_input_upload"]["format"] == "bgr24"
    assert reduced_rows["nvof_input_upload"]["payload_bytes"] \
        == tl.PAYLOAD_REFERENCE["working_bgr24"]
    # The plane the host reads exists in both orders; only its producer differs.
    assert reduced_rows["working_rgba_for_dlss"]["payload_bytes"] \
        == legacy_rows["working_rgba_for_dlss"]["payload_bytes"]
    assert "fused" in reduced_rows["working_bgr"]["producer"]
    assert "fused" in reduced_rows["working_rgba_for_dlss"]["producer"]


def test_a_reduced_ledger_reports_the_deleted_copies_as_measured_zeros():
    reduced = tl.build_ledger(measured={
        "cpu_bgr_to_rgba@dlss_input_calls": 0, "cpu_bgr_to_rgba@dlss_input_output_bytes": 0,
        "cpu_bgr_to_rgba@nvof_input_calls": 0, "cpu_bgr_to_rgba@nvof_input_output_bytes": 0,
        "cpu_rgba_to_bgr@residual_input_calls": 0,
        "cpu_rgba_to_bgr@residual_input_output_bytes": 0,
        "nvof_h2d_bytes_measured": tl.PAYLOAD_REFERENCE["working_bgr24"],
    }, transfer_mode="reduced")
    rows = {r["name"]: r for r in reduced["rows"]}
    for name in ("working_rgba_for_dlss", "working_rgba_for_nvof", "dlss_output_bgr"):
        assert rows[name]["measurement"] == "measured", name
        assert rows[name]["cpu_call_count"] == 0, name
        assert rows[name]["transfer_bytes"] == 0, name
    assert rows["nvof_input_upload"]["transfer_bytes"] == 6220800


def test_a_ledger_refuses_an_order_it_has_no_rows_for():
    with pytest.raises(ValueError):
        tl.build_ledger(transfer_mode="auto")


def test_both_ledgers_render_their_order_into_the_header():
    for mode in tl.TRANSFER_MODES:
        text = tl.render_markdown(tl.build_ledger(transfer_mode=mode))
        assert f"transfer order `{mode}`" in text
        assert tl.TRANSFER_MODE_MEANING[mode] in text


# ---------------------------------------------------------------------------
# hardware: the reduced order is the one that stops uploading a fourth channel
# ---------------------------------------------------------------------------

def _require_bgr_input():
    probe = of.probe_nvof_capabilities()
    assert probe.get("available"), f"NVOF hardware went away mid-run: {probe.get('error')}"
    assert probe.get("bgr_input"), (
        "the loaded nvof_bridge exports no nvof_bridge_execute_bgr_sequential, so the "
        "production reduced path cannot run; rebuild native/nvof_bridge.dll from this "
        "checkout's source")
    return probe


@requires_nvof
def test_production_reduced_path_uploads_three_bytes_per_pixel_to_nvof(conversions):
    """The production core, a real NVOF engine, and the bridge's own counters.

    The byte figure is arithmetic on the working size, checked against a counter that lives
    in the C++ session. That direction is the only one that keeps both sides honest: a
    ledger may publish what a counter measured, and a test may check a measurement only
    against something independent of it.
    """
    probe = _require_bgr_input()
    # The guided NVOF frame keeps its vectors on the card, which is what makes this the
    # production route. A bridge that cannot answer with a descriptor would run the
    # host-plane entry instead and still satisfy every byte assertion below with timings
    # from a different schedule, so the gate says which bridge it measured.
    if not probe.get("device_output"):
        pytest.skip("the loaded nvof_bridge exports no device-descriptor flow entry, so "
                    "this gate cannot reach the route it describes")
    src_w, src_h, live_w, live_h = 1280, 720, 640, 360
    # A service of this test's own, not get_frame_guidance_service's cached one: that
    # registry hands out a zombie to whoever asks after another test has closed it, and a
    # closed service running no engine would answer this gate with None instead of with a
    # format. It is not closed afterwards either, for the same reason in the other
    # direction - the engine underneath it is process-wide.
    service = of.FrameGuidanceService("nvof", "4M")
    engine = service._flow_engine
    assert engine is not None, "the guidance service built no flow engine at all"
    provider = engine.provider_info()
    assert provider.get("hardware"), (
        f"the guidance service is not on NVOF hardware, so this gate would measure nothing "
        f"about the bridge: {provider}")
    core = pipeline.ProductionExportCore(
        src_w=src_w, src_h=src_h, dst_w=src_w, dst_h=src_h,
        passes=[gs.StubLive(live_w, live_h)],
        job_settings={"guidance_mode": 1, "working_scale": 0.5, "residual_pyramid": True,
                      "residual_strength": 1.0, "residual_backend": "native",
                      "transfer_mode": "reduced"},
        flow_provider=pipeline.GuidanceServiceFlowProvider(service, guidance_mode=1),
        )
    rng = np.random.default_rng(5)
    frames = [rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8) for _ in range(4)]
    try:
        records = [core.process_frame(index, frame, is_discontinuous=(index == 0))[1]
                   for index, frame in enumerate(frames)]
    finally:
        core.close()

    steady = records[-1]
    assert steady["transfer_mode_actual"] == "reduced"
    assert steady["residual_backend_actual"] == "cpu_reference_native", (
        "the reduced order needs the native bridge's fused pass and in-place RGBA read; a "
        "run on the Python reference is not the path this gate is about")
    assert steady["nvof_input_format"] == "bgr24_host"
    assert steady["nvof_h2d_copy_count"] == 1
    assert steady["nvof_h2d_bytes_measured"] == live_w * live_h * 3, (
        "a steady reduced frame did not upload exactly one BGR24 working plane")
    assert steady["nvof_d2h_bytes_measured"] == 0, (
        "the guided NVOF frame hands the host a device descriptor, so its vectors are "
        "never downloaded; a nonzero figure here means the run left the device route")
    assert steady["cpu_bgr_to_rgba@nvof_input_calls"] == 0
    assert conversions.count(cv2.COLOR_BGR2RGBA) == 0, (
        "the engine still builds an RGBA copy of the working frame on the CPU")
    assert steady["nvof_bgr_input_call_count_delta"] == 1
    assert steady["nvof_input_convert_launch_count_delta"] == 1
    assert steady["nvof_bgr_staging_alloc_count_delta"] == 0, (
        "a frame allocated its own staging buffer; the session keeps two for the job")
    assert steady["nvof_input_convert_wait_ms"] is None, (
        "the device-output entry point reports no separate input-convert wait; a number "
        "here means the frame came back as a host plane and the route above is a lie")
    # The reset frame is the other half of the schedule: two uploads, both BGR.
    assert records[0]["nvof_h2d_bytes_measured"] == live_w * live_h * 3 * 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
