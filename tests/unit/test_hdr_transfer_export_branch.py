# -*- coding: utf-8 -*-
"""Gate and lifecycle: the experimental HDR10 export chain is opt-in and refuses.

Nothing in here needs a GPU, a display or an HDR source. The HDR chain's decision is
made from the same facts the SDR backend's is - one ffprobe, the resolved extents, what
the loaded libraries advertise about themselves and the colour format of the Feature 18
session that is already open - so it is testable without hardware, and the real export
stays the single run it is meant to be.

What is checked rather than asserted:

* the flag is opt-in only, in both directions: a job that does not name it runs the SDR
  chain untouched, and a job that names something unreadable is refused rather than
  guessed at;
* a request this chain cannot serve names every unmet condition and does not turn into a
  successful export by another route - an 8-bit frame drawn from a PQ source is
  indistinguishable from an HDR export in every field but this chain's own;
* the fixed parameters stay fixed: the support ceiling, the encode clamp and the paper
  white are the job's constants, not derived from MaxCLL and not adapted per frame;
* the descriptors the chain hands across the two boundaries it does cross - colour bridge
  to Feature 18, Feature 18 back to the bridge - satisfy the shipping host's own
  validation, at the working extent and eight bytes a pixel;
* a byte or component figure in the report is a bridge's own number or None, and the
  stages a run never opened contribute no key at all.
"""

import ctypes as C
import itertools
import json
import os
import subprocess
import sys
import tempfile

import pytest
import gpu_memory


@pytest.fixture(autouse=True)
def no_native_memory_query_for_fake_devices(monkeypatch):
    # This file intentionally supplies fake D3D12 addresses and no real GPU.
    monkeypatch.setattr(gpu_memory,'capture',lambda *args,**kwargs:None)

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_UNIT_DIR)
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dlss_gpu_engine as dge            # noqa: E402
import gpu_hdr_transfer_export as htx    # noqa: E402
import gpu_transfer_export as gtx        # noqa: E402
import hdr_fusion_engine as hfe          # noqa: E402
import hdr_gpu_engine as hdr             # noqa: E402
import pipeline                          # noqa: E402


@pytest.fixture(autouse=True)
def scene_observation_uses_no_real_device_in_cpu_contract_tests(monkeypatch):
    # These owners carry synthetic device pointers. Native scene detection has
    # its own real-GPU gate; the routing suite must never dereference a stub.
    from temporal_control import GpuTemporalControl
    monkeypatch.setattr(GpuTemporalControl, 'observe', lambda self, **kwargs:
        {'scene_cut': False, 'classification': 'test_not_measured'})

# The one source this chain was built for: the accepted HDR10 fixture's own ffprobe
# answer, at the working scale the final gate runs - 50% of 1920x1080 is 960x544,
# resolved by the shared rule rather than by a scale list of the HDR chain's own.
HDR_DIMS = {"src_w": 1920, "src_h": 1080, "live_w": 960, "live_h": 544,
            "dst_w": 1920, "dst_h": 1080, "fps": 25.0}
HDR_SOURCE = {"video_stream_count": 1, "codec_name": "hevc", "pix_fmt": "yuv420p10le",
              "width": 1920, "height": 1080, "color_range": "tv",
              "color_space": "bt2020nc", "color_transfer": "smpte2084",
              "color_primaries": "bt2020", "r_frame_rate": "25/1",
              "audio_streams": 0, "subtitle_streams": 0, "nb_chapters": 0,
              "metadata": True}
# The request the encode alone can deliver: nothing to carry from the source, so the
# shared helper reports no copy pass and the chain writes its deliverable itself.
NO_MUX = {"copy_metadata": False, "copy_attachments": False, "copy_chapters": False,
          "subtitle_mode": "drop"}
HDR_JOB = {"export_backend": "gpu_transfer", "dlss_backend": "gpu_flow_d3d12",
           "guidance_mode": 1, "flow_engine": "nvof", "working_scale": 0.5,
           "multi_pass": 1, "reconstruction_mode": "single",
           "experimental_hdr_scrgb": True}
# The five declarations are the installed entry's own answer, and no number in here is
# a Python opinion about what P010LE or BT2020 non-constant-luminance equals: a stub
# that returns a different value has to be honoured, not corrected.
HDR_CAPS = {
    "nvdec": {"abi": 3, "pixel_readback_entry_bound": False},
    "nvenc": {"hdr_abi": 1, "sei_abi": 1},
    "defaults_rc": 0,
    "struct_sizes_accepted": {"NvencHdrDefaults": True,
                              "NvencHdrSeiCapability": True,
                              "NvencHdrCapability": True},
    "encoder_defaults": {"sw_pixel_format": 0, "color_range": 1,
                         "color_primaries": 9, "color_trc": 16, "color_space": 9,
                         "profile": "main10"},
    "sei_capability": {"writer_available": 1, "supports_partial_mastering": 0,
                       "payload_type_mastering": 137,
                       "payload_type_content_light": 144},
    "capability": {"encoder_present": 1, "p010_supported": 1, "supported": 1},
    "colour_bridge": {"abi": 1, "path": "native/gpu_hdr_scrgb_bridge.dll",
                      "bound_exports": ["gpu_hdr_decode", "gpu_hdr_encode"],
                      "pixel_readback_exports_bound": [],
                      # The optional multiscale foundation, as hdr.probe_multiscale
                      # answers it: by symbol name, with a version and no opinion
                      # about the ABI it shares the file with.
                      "multiscale": {"supported": True, "version": 1, "missing": [],
                                     "reason": ""}},
    # The typed half of the fusion bridge, read off the library the bridge set
    # loaded. A set that loaded no fusion library carries no key at all, which is a
    # different answer from this one and refuses a data-consistency job.
    "fusion_bridge": {"supported": True, "version": 1, "missing": [], "reason": "",
                      "path": "native/gpu_multiscale_fusion_bridge.dll"},
    "missing": [],
}


TEARDOWN = []
"""Who released what, in the order they released it.

Each stub records itself here on close, so one list proves the reverse-owner order the
borrowed context and the imported texture depend on - a decoder released before the
encoder that reads its memory is not a bug a log can show any other way.
"""


class Fp16Live:
    """A session that validates a colour descriptor with the shipping host's own rule.

    ``_check_color`` and ``_check_flow`` are the real methods of
    ``dlss_gpu_engine.Live``, borrowed against the FP16 session's own byte count. A
    chain that handed an 8-bit plane, a full-size plane or a short pitch to inference
    would be refused here exactly as the real host refuses it.
    """

    color_format = "rgba16f"
    bytes_per_pixel = 8
    _check_color = dge.Live._check_color
    _check_flow = dge.Live._check_flow
    # The host's own id for a session is a counter over the shared core, so a fake
    # that hands one out gives one out: distinct per session and not a pointer.
    _sessions = itertools.count(1)

    def __init__(self, width=960, height=544, color_format="rgba16f"):
        self._w, self._h = width, height
        if color_format != "rgba16f":
            self.color_format = color_format
            self.bytes_per_pixel = 4
        self.session_id = next(Fp16Live._sessions)
        # The keys a real session bakes at open, so a layer session can be shown to
        # have been opened on this job's request rather than on the host's defaults.
        self.settings = {"preset": 1, "intensity": 1.0, "ui_correction": 0}
        self.resets = []
        self.descriptors = []
        self.flows = []

    def d3d12_device(self):
        return 0x3000

    def supports_zero_motion(self):
        return True

    def layer_capability(self):
        """The shipping host's answer for a session that opened as an FP16 layer:
        the four separate facts ``dlss_gpu_engine.Live.layer_capability`` reports
        rather than one boolean, because a build with layer sessions and no
        format-aware layer open is a different refusal and the gate names it."""
        return {"shared_layer_sessions": True, "format_aware_layer_open": True,
                "rgba16f_sessions": True, "reportable_session_id": True,
                "host_color_capabilities": {"fp16_row_bridge_ready": 1}}

    def transfer_counters(self):
        return {"host_color_stage_count": 0, "host_color_stage_bytes": 0,
                "device_color_copy_count": 6, "device_color_copy_bytes": 25_165_824,
                "output_readback_count": 0, "output_readback_bytes": 0}

    def _output(self, reset):
        self.resets.append(bool(reset))
        # dxgi_format 10 is R16G16B16A16_FLOAT, which the colour bridge insists on
        # before it reads the texture, and the extent is the working one it opened with.
        # The texture and the fence are this session's own, named by its own id: a
        # layer whose observation read another layer's texture is then visible as a
        # different resource, and not only as a different extent.
        return {"width": self._w, "height": self._h, "dxgi_format": 10,
                "d3d12_device": 0x3000,
                "d3d12_resource": 0x4000 + 0x1000 * self.session_id,
                "d3d12_fence": 0x5000 + 0x1000 * self.session_id,
                "fence_value": len(self.resets)}

    def process_device_color_flow_resource(self, color, flow, reset=False):
        self._check_color(color)
        self._check_flow(flow)
        self.descriptors.append(dict(color))
        self.flows.append(dict(flow))
        return self._output(reset)

    def process_device_color_zero_motion_resource(self, color, reset=False):
        self._check_color(color)
        self.descriptors.append(dict(color))
        return self._output(reset)

    def close(self):
        TEARDOWN.append("feature18")
        self.closed = True


class Sdr8BitLive(Fp16Live):
    """The session every job has always gotten: open, alive, and the wrong colour."""

    def __init__(self):
        super().__init__(color_format="rgba8")


class RaisingLive(Fp16Live):
    def process_device_color_flow_resource(self, color, flow, reset=False):
        raise dge.DLSSInferenceError("Feature 18 refused the FP16 frame")


class DeviceFlowProvider:
    """An NVOF provider whose engine reads the guidance surface where it lies."""

    def __init__(self, width=960, height=544, counters=(0, 0)):
        self.size = (width, height)
        self._counters = counters
        self.calls = []
        self.extents = []
        self.discontinuities = []
        self.layers = []
        self.closed_layers = 0

    def new_independent_layer(self):
        """A second engine with its own history, as the real provider returns one.

        One per extra layer, and nothing about the extent is fixed here: the real
        service takes the size from the descriptor it is handed per frame, which is
        what lets a layer be guided at the extent that layer resolved to. The list is
        what lets a test say which engine a frame's motion came from.
        """
        layer = DeviceFlowProvider(self.size[0], self.size[1], self._counters)
        layer.anchored_to = self
        self.layers.append(layer)
        return layer

    def close_independent_layer(self):
        """Hand this engine's history back, and say so in the teardown ledger.

        Only a layer's own provider has this method to be called on: the root provider
        belongs to the caller, so a chain that closed it would be releasing an object it
        never opened.
        """
        self.closed_layers += 1
        TEARDOWN.append("flow layer")

    def get_flow_device_rgba(self, frame_idx, color, cuda_context,
                             is_discontinuous=False):
        """Flow at the extent the descriptor says, as the real engine answers it.

        Nothing is fixed to the provider's own construction size: a layer is guided at
        the extent that layer resolved to, and an engine that could not follow the
        descriptor would be the bug this chain has to be caught doing.
        """
        assert color.get("format") == "rgba8_cuda", color.get("format")
        width, height = int(color["width"]), int(color["height"])
        assert int(color.get("pitch_bytes", 0)) >= width * 4
        self.calls.append(frame_idx)
        self.extents.append((width, height))
        self.discontinuities.append(bool(is_discontinuous))
        return {"format": "float32x2", "width": width, "height": height,
                "cuda_context": cuda_context or 0x9000, "device_ptr": 0x7000,
                "pitch_bytes": width * 8}

    def supports_device_rgba_flow(self):
        return True

    def device_input_counters(self):
        return self._counters

    def timing(self):
        return {"guidance_backend": "working_resolution_nvof_device_rgba"}


class BannedFlowProvider(DeviceFlowProvider):
    def get_flow_device_rgba(self, *args, **kwargs):
        raise AssertionError("an unguided HDR frame asked a flow provider for motion")


class HostOnlyFlow(DeviceFlowProvider):
    def supports_device_rgba_flow(self):
        return False


def _negotiate(job=None, source=None, dims=None, transcode=None, live=None,
               provider="default", caps=None, out_path="out.mkv",
               with_audio=False, **kwargs):
    provider = DeviceFlowProvider() if provider == "default" else provider
    kwargs.setdefault("bridges_ok", True)
    return htx.negotiate_hdr(
        dict(dims or HDR_DIMS), dict(source or HDR_SOURCE), dict(job or HDR_JOB),
        dict(transcode or NO_MUX), 24, with_audio, out_path, live or Fp16Live(),
        hdr_caps=dict(caps if caps is not None else HDR_CAPS),
        flow_provider=provider, **kwargs)


# ---------------------------------------------------------------------------
# the opt-in, and the gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (True, True), (1, True), ("true", True), ("True", True), ("yes", True),
    (False, False), (None, False), ("", False), (0, False), ("false", False),
    ("off", False), ("nonsense", False)])
def test_the_hdr_chain_is_opted_into_by_an_explicit_flag_and_nothing_else(value,
                                                                          expected):
    """An absent key is off: every saved job has always meant that, and still does."""
    assert htx.hdr_requested({"experimental_hdr_scrgb": value}) is expected


def test_a_flag_that_is_neither_on_nor_off_is_a_refusal_rather_than_a_guess():
    assert htx.hdr_flag_reason({"experimental_hdr_scrgb": "maybe"})
    assert htx.hdr_flag_reason({}) is None
    assert htx.hdr_flag_reason({"experimental_hdr_scrgb": True}) is None


@pytest.mark.parametrize("value", [0.5, 0.0, -1.0, float("nan"), float("inf"), "x"])
def test_a_paper_white_that_is_not_finite_and_at_least_one_is_refused(value):
    assert htx.paper_white_reason({"hdr_paper_white_scrgb": value})


def test_a_request_the_hdr_chain_can_serve_is_accepted_with_no_reason_at_all():
    negotiation = _negotiate()
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["reasons"] == []
    assert negotiation["uses_flow"] is True
    assert negotiation["downscaled"] is True
    assert negotiation["copy_pass"] is False


def test_the_fixed_parameters_are_the_job_constants_and_not_something_derived():
    policy = _negotiate()["hdr_policy"]
    assert policy["support_ceiling_scrgb"] == 250.0
    assert policy["clip_ceiling_scrgb"] == 125.0
    assert policy["paper_white_scrgb"] == 8.0
    assert policy["pq_top_scrgb"] == 125.0
    assert policy["derived_legal_peak_scrgb"] == pytest.approx(207.53)
    assert policy["nits_per_scrgb"] == 80.0
    assert policy["single_layer"] is True and policy["tonemap"] is False
    assert policy["output_resize"] is False
    # The mux is the shared pass's video copy, not a second encode: the phrase the
    # policy is allowed to use is the one that keeps it honest.
    assert "build_mux_command" in policy["mux"] and "-c:v copy" in policy["mux"]


def test_a_job_that_named_a_paper_white_uses_it_once_and_the_gate_still_accepts():
    policy = _negotiate(job=dict(HDR_JOB, hdr_paper_white_scrgb=4.0))["hdr_policy"]
    assert policy["paper_white_scrgb"] == 4.0


def test_an_sdr_backend_is_refused_rather_than_fallen_back_to():
    negotiation = _negotiate(job=dict(HDR_JOB, export_backend="legacy"))
    assert not negotiation["supported"]
    assert any("never falls back" in reason for reason in negotiation["reasons"])


def test_an_open_rgba8_session_is_refused_because_the_proxy_is_fp16():
    """The requirement in one test: never silently use an already-open 8-bit session."""
    negotiation = _negotiate(live=Sdr8BitLive())
    assert not negotiation["supported"]
    assert any("'rgba16f'" in reason for reason in negotiation["reasons"])


@pytest.mark.parametrize("change,fragment", [
    ({"pix_fmt": "yuv420p"}, "10-bit"),
    ({"color_transfer": "arib-std-b67"}, "HLG"),
    ({"color_transfer": "bt709"}, "smpte2084"),
    ({"color_range": ""}, "limited or full range"),
    ({"color_range": "unknown"}, "limited or full range"),
    ({"color_primaries": "bt709"}, "bt2020"),
    ({"color_space": "bt2020cl"}, "constant-luminance"),
    ({"color_space": ""}, "bt2020ncl"),
    ({"video_stream_count": 2}, "exactly one"),
])
def test_a_source_outside_the_domain_this_bridge_converts_is_refused(change, fragment):
    negotiation = _negotiate(source=dict(HDR_SOURCE, **change))
    assert not negotiation["supported"]
    assert any(fragment in reason for reason in negotiation["reasons"]), fragment


@pytest.mark.parametrize("range_name", ["tv", "mpeg", "limited", "pc", "jpeg", "full"])
def test_hdr_source_explicit_range_is_normalized_by_decode(range_name):
    negotiation = _negotiate(source=dict(HDR_SOURCE, color_range=range_name))
    assert negotiation["supported"], negotiation["reasons"]


@pytest.mark.parametrize("job_change,fragment", [
    ({"reconstruction_mode": "dual"}, "layer_scales"),
    ({"reconstruction_mode": "quadruple"}, "single, dual, or triple"),
    ({"multi_pass": 2}, "multi_pass"),
    ({"guidance_mode": 2}, "guidance_mode"),
    ({"flow_engine": "dis"}, "no device path"),
    ({"hdr_paper_white_scrgb": 0.5}, "1.0 scRGB"),
])
def test_a_mode_this_chain_does_not_implement_is_refused_not_degraded(job_change,
                                                                     fragment):
    negotiation = _negotiate(job=dict(HDR_JOB, **job_change))
    assert not negotiation["supported"]
    assert any(fragment in reason for reason in negotiation["reasons"]), fragment


def test_a_tonemap_or_colour_preset_is_refused_because_the_master_is_unclamped():
    for preset in ("hdr_to_sdr", "bt709_sdr", "tone_map_reactive"):
        negotiation = _negotiate(transcode=dict(NO_MUX, color_preset=preset))
        assert not negotiation["supported"]
        assert any("color_preset" in reason for reason in negotiation["reasons"])


def test_final_output_resize_is_accepted_and_wrong_container_remains_refused():
    negotiation = _negotiate(dims=dict(HDR_DIMS, dst_w=1280, dst_h=720))
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation['hdr_policy']['output_resize'] is True
    negotiation = _negotiate(out_path="out.mp4")
    assert any("output container" in reason for reason in negotiation["reasons"])


@pytest.mark.parametrize("scale,accepted", [(0.5, True), (0.25, True), (1.0, True),
                                            (0.6, True), (0.2, False), (1.1, False)])
def test_the_working_extent_is_resolved_by_the_same_rule_the_cpu_plan_uses(scale,
                                                                          accepted):
    """No HDR scale list: the shared resolve_working_dims, and the 25..100 window."""
    src_w, src_h = 1920, 1080
    live = pipeline.resolve_working_dims(src_w, src_h, scale)
    dims = dict(HDR_DIMS, src_w=src_w, src_h=src_h, live_w=live[0], live_h=live[1],
                dst_w=src_w, dst_h=src_h)
    negotiation = _negotiate(job=dict(HDR_JOB, working_scale=scale), dims=dims)
    assert negotiation["supported"] is accepted, negotiation["reasons"]


def test_extents_that_are_not_what_that_scale_resolves_to_are_refused():
    negotiation = _negotiate(job=dict(HDR_JOB, working_scale=0.6),
                             dims=dict(HDR_DIMS, live_w=1152, live_h=648))
    assert negotiation["supported"], negotiation["reasons"]
    negotiation = _negotiate(job=dict(HDR_JOB, working_scale=0.6),
                             dims=dict(HDR_DIMS, live_w=960, live_h=544))
    assert not negotiation["supported"]
    assert any("declared" in reason for reason in negotiation["reasons"])


class NoLayerLive(Fp16Live):
    """An FP16 session from a wrapper that answers no question about layer sessions."""

    layer_capability = None


class SdrLayerOnlyLive(Fp16Live):
    """A host with independent layer sessions and no format-aware layer open.

    The shape a dual-layer SDR job runs on and a dual-layer HDR job cannot: sharing
    the core is one export and opening an RGBA16F session on it is a second.
    """

    def layer_capability(self):
        return {"shared_layer_sessions": True, "format_aware_layer_open": False,
                "rgba16f_sessions": False, "reportable_session_id": True,
                "host_color_capabilities": {"fp16_row_bridge_ready": 0}}


class SharedHistoryFlow(DeviceFlowProvider):
    """An NVOF provider with one engine and no way to give a layer its own history."""

    new_independent_layer = None
    close_independent_layer = None


def _caps(without=()):
    caps = {key: value for key, value in HDR_CAPS.items() if key not in without}
    for key, subkey in (("colour_bridge", "multiscale"),):
        if subkey in without and key in caps:
            caps[key] = {k: v for k, v in caps[key].items() if k != subkey}
    return caps


def _multi(mode="dual", scales=(0.5, 0.25), **settings):
    job = dict(HDR_JOB, reconstruction_mode=mode, layer_scales=list(scales))
    job.update(settings)
    return job


def test_a_dual_job_on_a_build_that_carries_the_parts_is_two_layers_and_says_which():
    negotiation = _negotiate(job=_multi())
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["layers"] == 2
    assert negotiation["hdr_policy"]["single_layer"] is False
    assert negotiation["hdr_policy"]["reconstruction_mode"] == "dual"
    # The extents are the shared resolver's, not a scale list of this chain's.
    assert negotiation["multi_scale"]["dims"] == [(960, 544), (480, 272)]


def test_a_triple_job_takes_a_100_percent_layer_on_the_identity_plan_alone():
    """No whitelist of served ratios and no special case at the top of the range:
    1.0 resolves to the full extent, which is the shape the bridge already takes."""
    negotiation = _negotiate(job=_multi("triple", (0.5, 0.25, 1.0)))
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["layers"] == 3
    assert negotiation["multi_scale"]["dims"][2] == (1920, 1080)


def test_a_data_consistency_dual_job_is_accepted_onto_the_typed_fusion_group():
    negotiation = _negotiate(job=_multi(
        fusion_mode="data_consistency_experimental", fusion_beta=0.05,
        fusion_relative_tolerance=0.001, fusion_max_iterations=64))
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["hdr_policy"]["fusion_mode"] == "data_consistency_experimental"
    assert negotiation["multi_scale"]["beta"] == 0.05
    assert negotiation["multi_scale"]["max_iterations"] == 64


def test_a_catmull_baseline_dual_job_needs_no_typed_fusion_read_at_all():
    """The group the solver would need is asked for only by the mode that would call
    it, so a baseline job on a build with no fusion library is still two layers."""
    negotiation = _negotiate(job=_multi(), caps=_caps(without=("fusion_bridge",)))
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["multi_scale"]["fusion"] == "catmull_baseline"


@pytest.mark.parametrize("patch,fragment", [
    (dict(colour_bridge=_caps(without=("multiscale",))["colour_bridge"]),
     "no HDR multiscale capability read"),
    (dict(colour_bridge=dict(HDR_CAPS["colour_bridge"], multiscale={
        "supported": False, "version": 0, "missing": ["gpu_hdr_observe"],
        "reason": "gpu_hdr_scrgb_bridge.dll does not export gpu_hdr_observe"})),
     "does not export gpu_hdr_observe"),
    (dict(fusion_bridge={"supported": False, "version": 0,
                         "missing": ["gpu_fusion_process_hdr"],
                         "reason": "the fusion bridge exports no typed linear-scRGB "
                                   "group"}), "typed linear"),
    ({"fusion_bridge": None}, "no typed HDR fusion capability read"),
])
def test_a_second_layer_on_a_build_that_cannot_answer_for_it_is_refused(patch,
                                                                        fragment):
    """A missing part is said as the missing part - including the build's own words
    about which symbol it does not export - and is never answered by running one
    layer instead.

    One capability per case, so the refusal names the one thing that was not there.
    """
    job = _multi(fusion_mode="data_consistency_experimental")
    if fragment.startswith("no HDR multiscale") or "gpu_hdr_observe" in fragment:
        job = _multi()
    caps = dict(HDR_CAPS)
    for key, value in patch.items():
        if value is None:
            caps.pop(key, None)
        else:
            caps[key] = value
    negotiation = _negotiate(job=job, caps=caps)
    assert not negotiation["supported"]
    assert any(fragment in reason for reason in negotiation["reasons"]), (
        fragment, negotiation["reasons"])


@pytest.mark.parametrize("live,fragment", [
    (NoLayerLive(), "reports no layer capability"),
    (SdrLayerOnlyLive(), "rgba16f_sessions"),
])
def test_a_second_layer_the_host_cannot_open_is_refused_by_the_fact_that_is_missing(
        live, fragment):
    negotiation = _negotiate(job=_multi(), live=live)
    assert not negotiation["supported"]
    assert any(fragment in reason for reason in negotiation["reasons"]), (
        fragment, negotiation["reasons"])
    # The same session still runs the layer it was opened for.
    assert _negotiate(job=dict(HDR_JOB), live=live)["supported"]


def test_a_second_layer_sharing_one_history_is_refused_and_zero_motion_is_not():
    """guidance_mode=1 needs a layer per engine; guidance_mode=0 needs none, because
    each layer's own session carries its resident zero motion."""
    refused = _negotiate(job=_multi(), provider=SharedHistoryFlow())
    assert not refused["supported"]
    assert any("independent NVOF layer" in reason for reason in refused["reasons"])
    unguided = _negotiate(job=_multi(guidance_mode=0), provider=SharedHistoryFlow())
    assert unguided["supported"], unguided["reasons"]


def test_the_layer_vector_is_the_shared_helpers_own_verdict_word_for_word():
    """Nothing about a percentage, a weight, a tolerance or an iteration budget is
    re-decided for HDR: the refusal text is ``multi_scale_config``'s, so the two
    backends cannot disagree about a job they were both given."""
    for change, fragment in (
            ({"layer_scales": [0.5, 0.5]}, "distinct"),
            ({"layer_scales": [0.5, 1 / 3]}, "integer percent"),
            ({"layer_scales": [0.5]}, "exactly 2 layer_scales"),
            ({"layer_weights": [1.0, 0.0]}, "finite and positive"),
            ({"fusion_mode": "gauss_seidel"}, "unknown fusion_mode"),
            ({"fusion_beta": 0.0}, "beta>0"),
            ({"fusion_relative_tolerance": 1.5}, "0<tolerance<1"),
            ({"fusion_max_iterations": 0}, "1..1000"),
            ({"working_scale": 0.6}, "first layer scale")):
        negotiation = _negotiate(job=_multi(**change))
        assert not negotiation["supported"], change
        assert any("multi-scale configuration" in reason or
                   fragment in reason for reason in negotiation["reasons"]), (
            change, negotiation["reasons"])


def test_a_single_layer_job_needs_none_of_the_multiscale_parts():
    """The original path did not become conditional: a colour bridge that exports no
    multiscale group, a host that opens no layer session and a flow provider with one
    engine still run the one layer they always ran, with the same answer."""
    negotiation = _negotiate(caps=_caps(without=("multiscale", "fusion_bridge")),
                             live=NoLayerLive(), provider=SharedHistoryFlow())
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["layers"] == 1
    assert negotiation["hdr_policy"]["single_layer"] is True
    assert negotiation["multi_scale"]["mode"] == "single"
    assert negotiation["multi_scale"]["fusion"] == "catmull_baseline"


def test_a_job_that_needs_streams_carried_across_is_accepted_onto_the_copy_pass():
    """Attachments, format metadata and chapters are the copy pass's, not a refusal.

    The gate asks the same shared helper the SDR chain asks, so what this chain admits
    is what that helper can express - and the answer is recorded as ``copy_pass``, which
    is what the run reads to decide whether the encoder's file is the deliverable.
    """
    negotiation = _negotiate(transcode=dict(NO_MUX, copy_attachments=True))
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["copy_pass"] is True


def test_a_source_with_audio_is_accepted_and_named_as_needing_the_copy_pass():
    """A soundtrack is carried by reusing the mapping pass, never dropped silently."""
    negotiation = _negotiate(source=dict(HDR_SOURCE, audio_streams=1),
                             with_audio=True)
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["copy_pass"] is True
    # The same request without the answer to "keep the sound?" has nothing to carry,
    # so the flag - not the mode's default - decides that no pass runs.
    quiet = _negotiate(source=dict(HDR_SOURCE, audio_streams=1), with_audio=False)
    assert quiet["copy_pass"] is False


def test_a_request_the_encode_alone_delivers_still_needs_no_copy_pass():
    assert _negotiate()["copy_pass"] is False
    subtitles = _negotiate(source=dict(HDR_SOURCE, subtitle_streams=1))
    assert subtitles["supported"] and subtitles["copy_pass"] is False


def test_a_chapter_only_request_is_a_copy_pass_request_and_not_a_silent_drop():
    """Two chapters and nothing else to carry is still a job the pass has to run.

    The audit case was the narrow one: audio removed, subtitles dropped, no attachment,
    no format metadata, chapters wanted. Every other clause of ``needs_copy_pass`` says
    no, so the whole job rested on ``nb_chapters`` - and the shared probe read that from
    ``format.nb_chapters``, a field a real Matroska file is free not to write. The gate
    then reported no copy pass, the encoder's own file became the deliverable, and the
    chapters were gone without a word.
    """
    chapters_only = dict(NO_MUX, copy_chapters=True)
    probed = dict(HDR_SOURCE, nb_chapters=2)

    negotiation = _negotiate(source=probed, transcode=chapters_only)
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["copy_pass"] is True

    # What the same gate does with the count the file never wrote: it is not a refusal,
    # and that is exactly why the count has to come from -show_chapters.
    uncounted = _negotiate(source=dict(HDR_SOURCE, nb_chapters=0),
                           transcode=chapters_only)
    assert uncounted["copy_pass"] is False

    # And a job that did not ask for chapters is not rewrapped for them.
    assert _negotiate(source=probed)["copy_pass"] is False


def test_the_hdr_entry_asks_the_probe_for_the_facts_the_encode_cannot_carry(
        monkeypatch, tmp_path):
    """The two opt-ins are read on the HDR route and not on the SDR one.

    ``format.nb_chapters`` and the first video stream's tags are not free: the first
    needs a second ``-show_chapters`` dump and the second is a field no other caller
    reads. Asking for them is the HDR chain's decision because its deliverable is a
    stream the encoder bridge wrote, and the SDR entry is asserted here to keep asking
    exactly what it always asked.
    """
    seen = []

    class RecordingGtx:
        INTERMEDIATE_CONTAINER = ".mkv"

        def guidance_requires_flow(self, js):
            return False

        def load_bridges(self, requires_flow=False):
            return {"nvdec": None, "nvenc": None, "cookies": [],
                    "option_available": lambda codec, name: True}

        def probe_source(self, path, include_chapters=False,
                         include_video_metadata=False):
            seen.append({"include_chapters": include_chapters,
                         "include_video_metadata": include_video_metadata})
            return dict(HDR_SOURCE, nb_chapters=2,
                        video_tags={"title": "Master shot", "ENCODER": "Lavc"},
                        video_disposition={})

        def negotiate(self, *args, **kwargs):
            raise AssertionError("the SDR gate negotiated an HDR job")

    class RefusingHdr:
        def hdr_requested(self, js):
            return True

        def load_hdr_capabilities(self, bridges, width, height):
            return dict(HDR_CAPS)

        def negotiate_hdr(self, dims, source, *args, **kwargs):
            seen.append({"source": source})
            return {"supported": False, "reasons": ["stub refuses"], "plan": {},
                    "fps": (25, 1), "uses_flow": False, "copy_pass": False,
                    "hdr_video_stream_metadata": None}

        def release_loader_cookies(self, bridges):
            return 0

    monkeypatch.setattr(pipeline, "video_info",
                        lambda path: (6, 25.0, 1920, 1080))
    source = os.path.join(str(tmp_path), "src.mkv")
    open(source, "wb").write(b"x" * 4096)
    with pytest.raises(RuntimeError, match="stub refuses"):
        pipeline.export_gpu_hdr_transfer_stream(
            video=source, live=Fp16Live(), dims=dict(HDR_DIMS), frames=6, fps=25.0,
            cq=24, with_audio=False, out_path=os.path.join(str(tmp_path), "out.mkv"),
            transcode_settings=dict(NO_MUX, copy_chapters=True),
            job_settings=dict(HDR_JOB), flow_provider=DeviceFlowProvider(),
            module=RefusingHdr(), transfer_module=RecordingGtx())

    asked = seen[0]
    assert asked == {"include_chapters": True, "include_video_metadata": True}
    # The chapter count the gate was handed is the list's length, and the video stream's
    # own title reached the gate narrowed: the technical tag stayed with the source.
    probed = seen[1]["source"]
    assert probed["nb_chapters"] == 2
    assert htx.descriptive_video_metadata_request(
        probed, dict(NO_MUX, copy_chapters=True, copy_metadata=True)) == {
        "title": "Master shot", "disposition": []}


def test_the_sdr_entry_still_asks_the_probe_for_exactly_what_it_always_asked(
        monkeypatch, tmp_path):
    """The opt-ins are the HDR route's; the SDR call site is unchanged."""
    seen = []

    class RecordingGtx:
        INTERMEDIATE_CONTAINER = ".mkv"

        def guidance_requires_flow(self, js):
            return False

        def load_bridges(self, requires_flow=False):
            return {"nvdec": None, "nvenc": None, "cookies": [],
                    "option_available": lambda codec, name: True}

        def probe_source(self, path, **kwargs):
            seen.append(kwargs)
            return dict(HDR_SOURCE)

        def negotiate(self, dims, source, *args, **kwargs):
            return {"supported": False, "reasons": ["stub refuses"],
                    "plan": {}, "fps": (25, 1), "copy_pass": False}

    monkeypatch.setattr(pipeline, "video_info",
                        lambda path: (6, 25.0, 1920, 1080))
    source = os.path.join(str(tmp_path), "sdr.mkv")
    open(source, "wb").write(b"x" * 4096)
    with pytest.raises(RuntimeError, match="stub refuses"):
        pipeline.export_gpu_transfer_stream(
            video=source, live=Fp16Live(), frames=6, fps=25.0, cq=24,
            with_audio=False, out_path=os.path.join(str(tmp_path), "out.mkv"),
            transcode_settings={}, job_settings={"export_backend": "gpu_transfer",
                                                 "guidance_mode": 0,
                                                 "dlss_backend": "gpu_flow_d3d12",
                                                 "working_scale": 1.0},
            flow_provider=DeviceFlowProvider(), module=RecordingGtx())
    assert seen == [{}], "the SDR probe asks for nothing the HDR route added"


def test_a_container_that_is_not_the_encoder_or_the_rewrap_is_still_refused():
    negotiation = _negotiate(out_path="out.mp4")
    assert not negotiation["supported"]
    assert any("output container" in reason for reason in negotiation["reasons"])


@pytest.mark.parametrize("extra", [
    # The clause the pass owns for the picture: a later -c:v wins in FFmpeg and the
    # Main 10 bitstream the chain exists to preserve is re-encoded.
    "-c:v libx265", "-c:v=libx265", "-codec:v libx264", "-vcodec=libx264",
    # A mapping clause could pull the source's own video into the deliverable.
    "-map 1:v:0", "-map=1:v",
    # A second, contradicting colour declaration.
    "-color_trc bt709", "-pix_fmt yuv420p", "-colorspace smpte170m",
    # A container, a stream count, a rate, an overwrite the queue already decided.
    "-f mp4", "-an", "-y", "-r 60", "-s 1280x720", "-frames:v 1",
    # The one that a name list cannot catch and this chain used to let through: a
    # bitstream filter that deletes the ST 2086 and CTA-861.3 SEI NAL units from every
    # packet, which is a refusal the deliverable cannot show.
    "-bsf:v filter_units=remove_types=39|40",
    "-bsf:v=filter_units=remove_types=39",
    "bsf:v filter_units=remove_types=40",
    # A filter graph, which is a decode of the copied bitstream.
    "-filter_complex [1:v]scale=960:-2", "-filter:v scale=960:-2", "-vf tmppornv",
    # An extra bare output file, and a field that is not a clause at all.
    "second_output.mkv", "-metadata title=HDR", "-map_metadata 0:s:0",
    # Quoted, escaped and comment-shaped values: the contract does not read the argument
    # list, so none of these forms is an exception. shlex.split would drop the second
    # line as a comment (its comments= default is False), which is exactly the kind of
    # disagreement between a parser and FFmpeg this refusal ends.
    '"-c:v" libx265', "-c\\:v libx265", "# comment\n-c:v libx265",
    "-c:v 'libx265 -y'", "  -sn  ", "\t-metadata x=1\t",
])
def test_any_extra_arg_at_all_is_refused_before_the_gpu_work(extra):
    """This stage carries no caller-written FFmpeg arguments, in any form.

    The chain used to keep a list of option names it would not let a caller repeat, and
    ``-bsf:v filter_units=remove_types=39|40`` was not on it: the argument list a gate
    like that approves can still delete the mastering-display and content-light SEIs
    from every packet of the deliverable. Refusing every non-empty value is the only
    contract here that does not depend on parsing FFmpeg's grammar better than FFmpeg.
    """
    negotiation = _negotiate(transcode=dict(NO_MUX, extra_args=extra))
    assert not negotiation["supported"]
    reason = next(r for r in negotiation["reasons"] if "extra_args" in r)
    assert "no extra FFmpeg arguments at all" in reason


@pytest.mark.parametrize("extra", ["", "   ", "\n", "\t", " ", None])
def test_an_empty_extra_args_field_is_no_request_at_all(extra):
    """The refusal is of a value a caller named, not of the field existing."""
    negotiation = _negotiate(transcode=dict(NO_MUX, extra_args=extra))
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["copy_pass"] is False


def test_the_shared_pass_still_appends_a_callers_extra_args_for_sdr():
    """Refusing them is this chain's gate, not a change to what the muxer does.

    The GPU-transfer copy pass has always appended ``extra_args`` after its own
    clauses; the narrower HDR rule refuses a job before it reaches that builder, so a
    non-HDR job's command is exactly the one it was before.
    """
    command = gtx.build_mux_command(
        "ffmpeg", "enc.mkv", "src.mkv",
        {"extra_args": "-bsf:v filter_units=remove_types=39|40"}, "out.mkv", True)
    assert "-bsf:v" in command and "filter_units=remove_types=39|40" in command
    assert command.index("-bsf:v") < command.index("out.mkv")
    assert command[-1] == "out.mkv"


# The source the encoder's own stream cannot inherit: a track named in a language, with
# the flags the source set - and the technical tags the source's encoder wrote.
NAMED_SOURCE = dict(HDR_SOURCE, video_tags={"title": "Master shot", "language": "jpn",
                                            "ENCODER": "Lavc61.19.101 libx265",
                                            "BPS": "18750000",
                                            "DURATION": "00:00:00.240000000"},
                    video_disposition={"default": 1, "original": 1, "dub": 0})
BARE_NAMED_SOURCE = dict(NAMED_SOURCE, video_disposition={})
# The request that asks for the description: NO_MUX with the metadata switch on, since
# the fields are derived from it and not from a key the form has to offer.
NAMED_MUX = dict(NO_MUX, copy_metadata=True)


def test_the_source_videos_own_track_description_is_carried_without_a_new_setting():
    """Nobody has to turn this on: it follows the metadata switch the job already has.

    The encoder writes a new stream, so the deliverable would otherwise say nothing about
    the track the operator named in the source. A key the user must also set is a new
    setting, and a setting the form does not offer is one nobody can make - so the
    request is derived from ``copy_metadata``, which already means "describe the material
    rather than only contain it".
    """
    negotiation = _negotiate(source=NAMED_SOURCE, transcode=NAMED_MUX)
    assert negotiation["supported"], negotiation["reasons"]
    assert negotiation["copy_pass"] is True
    assert negotiation["hdr_video_stream_metadata"] == {
        "title": "Master shot", "language": "jpn",
        "disposition": ["default", "original"]}

    # Declining the metadata declines the video track's description with it, which is the
    # one control the operator does have - and with nothing else to carry, no rewrap runs.
    declined = _negotiate(source=NAMED_SOURCE, transcode=NO_MUX)
    assert declined["hdr_video_stream_metadata"] is None
    assert declined["copy_pass"] is False

    # And the narrow key overrides in either direction without ever being required -
    # which is also why it is not refused as an unknown encoder-plan setting.
    forced_off = _negotiate(source=NAMED_SOURCE,
                            transcode=dict(NAMED_MUX, copy_video_stream_metadata=False))
    assert forced_off["supported"], forced_off["reasons"]
    assert forced_off["hdr_video_stream_metadata"] is None
    # The override closes the video clause; it does not un-ask for the container's own
    # metadata, which is the thing copy_metadata was always the switch for.
    assert forced_off["copy_pass"] is True
    forced_on = _negotiate(source=NAMED_SOURCE,
                           transcode=dict(NO_MUX, copy_video_stream_metadata=True))
    assert forced_on["supported"], forced_on["reasons"]
    assert forced_on["hdr_video_stream_metadata"]["title"] == "Master shot"
    # Nothing else in this request needs the source, so the track description alone is
    # what put the rewrap back on the plan.
    assert forced_on["copy_pass"] is True


def test_the_source_encoders_technical_tags_are_never_repeated_as_this_runs():
    """``ENCODER``, ``BPS`` and ``DURATION`` describe the file the source *was*.

    Stamping them onto the stream this chain encoded would report somebody else's
    measurement as this run's, so the clause set is the track's name and language and
    nothing else.
    """
    named = htx.descriptive_video_metadata_request(NAMED_SOURCE, NAMED_MUX)
    assert set(named) <= {"title", "language", "disposition"}
    for technical in ("ENCODER", "BPS", "DURATION"):
        assert technical not in named and technical not in str(named.values())


def test_a_source_that_states_no_video_flags_does_not_inherit_the_muxers_own():
    """An all-zero disposition is a claim, and an empty one is the encoder's guess.

    The fixture this chain was gated on has nineteen disposition keys on its video track
    and not one of them set. Emitting no clause there leaves whatever flag the container
    assumed for the encoder's stream in the deliverable, which reads back as the source's
    own. The pass states the zero instead, in the spelling FFmpeg accepts.
    """
    named = htx.descriptive_video_metadata_request(BARE_NAMED_SOURCE, NAMED_MUX)
    assert named == {"title": "Master shot", "language": "jpn", "disposition": []}
    command = gtx.build_mux_command("ffmpeg", "enc.mkv", "src.mkv", dict(NO_MUX),
                                    "out.mkv", True, video_descriptive_metadata=named)
    assert ["-disposition:v:0", "-default"] in [
        command[i:i + 2] for i in range(len(command) - 1)]

    # A track with no name, no language and no flag is not a reason to rewrap anything.
    assert htx.descriptive_video_metadata_request(
        dict(HDR_SOURCE, video_tags={}, video_disposition={}), NAMED_MUX) is None


def test_the_descriptive_clause_names_the_encoded_stream_and_the_sdr_command_is_unchanged():
    """One clause list for the HDR run, and the SDR chain's command argument for argument.

    The rewrap maps the encoder's video as input 0 and the source as input 1, so without
    the clause the deliverable's video track is anonymous; with it, the track is named and
    the packets are still the encoder's - the source's video is never mapped. A request
    that never derived the fields, which is every SDR job, gets the command it always got,
    because nothing in the shared builder reads the HDR key.
    """
    plain = gtx.build_mux_command("ffmpeg", "enc.mkv", "src.mkv", dict(NO_MUX),
                                  "out.mkv", True)
    named = htx.descriptive_video_metadata_request(NAMED_SOURCE, NAMED_MUX)
    clauses = gtx.video_descriptive_clauses(named)
    with_clauses = gtx.build_mux_command("ffmpeg", "enc.mkv", "src.mkv", dict(NO_MUX),
                                        "out.mkv", True,
                                        video_descriptive_metadata=named)
    assert clauses, "the request produced no clause to find"
    stripped = list(with_clauses)
    for clause in reversed(clauses):
        stripped.remove(clause)
    assert stripped == plain
    assert "-metadata:s:v:0" not in plain and "-disposition:v:0" not in plain
    assert not any(arg.startswith("-map") for arg in clauses)
    assert "1:v" not in with_clauses


@pytest.mark.parametrize("change,fragment", [
    ({"audio_mode": "mp3"}, "audio_mode=mp3"),
    ({"audio_mode": "copy"}, None),
    ({"subtitle_mode": "ass"}, "subtitle_mode=ass"),
    ({"subtitle_mode": "convert"}, None),
])
def test_a_mode_the_shared_pass_does_not_express_is_refused_at_the_gate(
        change, fragment):
    negotiation = _negotiate(transcode=dict(NO_MUX, copy_attachments=True, **change))
    if fragment is None:
        assert negotiation["supported"], negotiation["reasons"]
    else:
        assert not negotiation["supported"]
        assert any(fragment in reason for reason in negotiation["reasons"])


def test_an_additive_entry_the_loaded_bridge_does_not_export_is_a_refusal():
    caps = dict(HDR_CAPS, missing=["nvdec_bridge.dll exports no "
                                  "nvdec_bridge_get_hdr_metadata"])
    negotiation = _negotiate(caps=caps)
    assert not negotiation["supported"]
    assert negotiation["reasons"][0].startswith("nvdec_bridge.dll")
    assert _negotiate(caps={})["supported"] is False


def test_a_flow_provider_that_cannot_be_handed_a_device_surface_is_a_refusal():
    """A provider that only takes a host plane would move the frame off the GPU.

    The chain's colour never leaves, so the guidance must not be the thing that
    downloads it: a provider without the device entry is refused, not worked around.
    """
    negotiation = _negotiate(provider=HostOnlyFlow())
    assert not negotiation["supported"]
    assert any("device RGBA input entry point" in reason
               for reason in negotiation["reasons"])
    assert _negotiate(provider=BannedFlowProvider(),
                      job=dict(HDR_JOB, guidance_mode=0))["supported"]


def test_the_encoder_request_is_the_installed_entrys_own_declaration():
    plan = _negotiate()["plan"]
    assert plan["codec"] == "hevc_nvenc" and plan["profile"] == "main10"
    struct = plan["struct"]
    assert (struct["sw_pixel_format"], struct["color_range"],
            struct["color_primaries"], struct["color_trc"],
            struct["color_space"]) == (0, 1, 9, 16, 9)
    assert struct["codec"] == b"hevc_nvenc" and struct["profile"] == b"main10"
    assert not any(name in htx.LOCKED_HDR_AV_OPTIONS
                   for name, _ in plan["av_options"])


@pytest.mark.parametrize("probe,expected", [(False, "hq"), (True, "uhq"),
                                            (None, "uhq")])
def test_the_gate_forwards_the_tune_probe_the_encoder_request_is_judged_by(probe,
                                                                           expected):
    """The plan holds the downgrade rule, the caller holds the probe, the gate connects them.

    The entry forwards tune verbatim, so a uhq the installed libavcodec does not list
    is an EINVAL that refuses the open after the decode and the reconstruction have
    already run - and nobody answering for the codec must not become a refusal either.
    """
    plan = _negotiate(transcode=dict(NO_MUX, tune="uhq"),
                      **({} if probe is None else
                         {"tune_available": lambda codec: probe}))["plan"]
    assert plan["struct"]["tune"] == expected.encode()


@pytest.mark.parametrize("transcode,fragment", [
    ({"codec": "h264_nvenc"}, "codec=h264_nvenc"),
    ({"profile": "main"}, "main10"),
    ({"pixel_format": "nv12"}, "P010"),
    ({"rate_mode": "unknown"}, "rate_mode=unknown"),
    ({"two_pass": "on"}, "no equivalent"),
])
def test_an_encoder_setting_that_is_not_main10_p010_is_a_refusal_not_a_downgrade(
        transcode, fragment):
    negotiation = _negotiate(transcode=dict(NO_MUX, **transcode))
    assert not negotiation["supported"]
    assert any(fragment in reason for reason in negotiation["reasons"]), fragment


def test_an_entry_that_names_a_different_profile_refuses_before_the_first_frame():
    caps = dict(HDR_CAPS, encoder_defaults=dict(HDR_CAPS["encoder_defaults"],
                                                 profile="main"))
    negotiation = _negotiate(caps=caps)
    assert not negotiation["supported"]
    assert any("names its profile" in reason for reason in negotiation["reasons"])


# ---------------------------------------------------------------------------
# the chain, driven against stubs that keep the contracts
# ---------------------------------------------------------------------------

def _write_into(target, source):
    C.memmove(target, C.byref(source), C.sizeof(source))


def _read_from(source, instance):
    """Copy the struct the chain was handed into an instance this test can read.

    The stub receives exactly what a real bridge receives: a ``byref`` for an output
    slot, the struct itself where ctypes passes a structure by reference. Both are the
    same memory, so this is how a test sees the numbers the production code wrote.
    """
    if isinstance(source, C.Structure):
        source = C.byref(source)
    C.memmove(C.byref(instance), source, C.sizeof(instance))
    return instance


def _device_frame(index):
    frame = gtx._DeviceFrame()
    frame.struct_size = C.sizeof(gtx._DeviceFrame)
    frame.width, frame.height = 1920, 1080
    frame.software_format = 2                      # P010
    frame.y_device_ptr, frame.uv_device_ptr = 0xA000, 0xB000
    frame.y_pitch_bytes, frame.uv_pitch_bytes = 4096, 4096
    frame.cuda_context = C.c_void_p(0x9000)
    frame.pts, frame.time_base_num, frame.time_base_den = index, 1, 25
    frame.color_range, frame.color_primaries = 1, 9
    frame.color_transfer, frame.color_space = 16, 9
    return frame


def _fixture_metadata(**overrides):
    """The valid fixture's own HDR declarations, as the decoder reports them.

    BT.2020 primaries, D65 white, 1000/0.005 cd/m2, MaxCLL 900 and MaxFALL 250 - the
    numbers ``HDR_STAGE_B_VALID_FIXTURE.md`` records for ``hdr_valid_src.mkv``, kept as
    rationals so the handoff can be checked without rounding anything first.
    """
    fields = dict(struct_size=C.sizeof(htx.HdrMetadata), has_mastering=1,
                  mastering_has_primaries=1, mastering_has_luminance=1,
                  has_content_light=1, max_cll=900, max_fall=250,
                  max_luminance_num=1000, max_luminance_den=1,
                  min_luminance_num=5, min_luminance_den=1000,
                  white_x_num=15635, white_x_den=50000,
                  white_y_num=16493, white_y_den=50000)
    fields.update(overrides)
    metadata = htx.HdrMetadata(**fields)
    metadata.primary_x_num = (13250, 3750, 1789)      # G, B, R of BT.2020
    metadata.primary_y_num = (34500, 1500, 1465)
    metadata.primary_x_den = (50000, 50000, 50000)
    metadata.primary_y_den = (50000, 50000, 50000)
    return metadata


class StubNvdec:
    def __init__(self, frames=6, metadata=None, software_format=2):
        self.total = frames
        self.metadata = metadata if metadata is not None else _fixture_metadata()
        self.software_format = software_format
        self.symbols = []
        self.index = 0
        self.counters = gtx._NvdecCounters()
        self.counters.struct_size = C.sizeof(gtx._NvdecCounters)
        self.counters.frames_decoded = frames
        self.counters.device_frame_count = frames
        self.counters.bitstream_bytes_submitted = 61_440

    def _log(self, name):
        self.symbols.append(name)

    def nvdec_bridge_open_hdr(self, path, out):
        self._log("nvdec_bridge_open_hdr")
        _write_into(out, C.c_void_p(0x1111))
        return 0

    def nvdec_bridge_next(self, handle, frame):
        self._log("nvdec_bridge_next")
        if self.index >= self.total:
            return 0
        current = _device_frame(self.index)
        current.software_format = self.software_format
        _write_into(frame, current)
        self.index += 1
        return 1

    def nvdec_bridge_get_hdr_metadata(self, handle, out):
        self._log("nvdec_bridge_get_hdr_metadata")
        _write_into(out, self.metadata)
        return 0

    def nvdec_bridge_get_transfer_counters(self, handle, out):
        _write_into(out, self.counters)
        return 0

    def nvdec_bridge_cancel(self, handle):
        self._log("nvdec_bridge_cancel")
        return 0

    def nvdec_bridge_close(self, handle):
        self._log("nvdec_bridge_close")
        TEARDOWN.append("decoder")


class StubNvenc:
    def __init__(self, metadata_changed_at=None, writes_file=True):
        self.symbols = []
        self.options = []
        self.attached = []
        self.submitted = []
        self.contexts = []
        self.paths = []
        # An entry that opened and ran but left nothing on disk: the copy pass has to
        # refuse that instead of wrapping a file that is not there.
        self.writes_file = writes_file
        self.metadata_changed_at = metadata_changed_at
        self.counters = htx.HdrCounters(C.sizeof(htx.HdrCounters))
        self.counters.frames_submitted = 6
        self.counters.packets_written = 6
        self.counters.p010_device_copy_bytes = 6 * 1920 * 1080 * 3
        self.counters.host_pixel_bytes = 0
        self.counters.rgb_conversion_bytes = 0
        self.counters.mastering_side_data = 68
        self.counters.content_light_side_data = 8
        self.contract = htx.HdrContract(C.sizeof(htx.HdrContract))
        self.contract.metadata_written_to = 8 | 16
        self.contract.has_mastering = 1
        self.contract.actual_bits_per_raw_sample = 10
        self.sei = htx.SeiStatus(C.sizeof(htx.SeiStatus))
        self.sei.writer_available = 1
        self.sei.metadata_requested = 1
        self.sei.mastering_serialized = 1
        self.sei.content_light_serialized = 1
        self.sei.packets_seen = 6
        self.sei.nals_inserted = 12
        self.sei.mastering_inserted = 6
        self.sei.content_light_inserted = 6

    def nvenc_hdr_open(self, context, width, height, fps_num, fps_den, tb_num, tb_den,
                       out_path, options, metadata, handle):
        self.symbols.append("nvenc_hdr_open")
        # The bridge is handed ``os.fsencode`` bytes, which is what it asks for.
        self.paths.append(os.fsdecode(out_path))
        self.contexts.append(int(context.value or 0))
        self.options.append(_read_from(options, htx.HdrOptions(
            C.sizeof(htx.HdrOptions))))
        self.attached.append(_read_from(metadata, htx.HdrMetadata(
            C.sizeof(htx.HdrMetadata))) if metadata else None)
        if self.writes_file:
            with open(out_path, "wb") as handle_out:
                handle_out.write(b"\0" * 2048)
        _write_into(handle, C.c_void_p(0x2222))
        return 0

    def nvenc_encode_p010(self, encoder, y_ptr, y_pitch, uv_ptr, uv_pitch, pts):
        self.symbols.append("nvenc_encode_p010")
        self.submitted.append((int(y_ptr), int(y_pitch), int(uv_ptr), int(uv_pitch),
                               int(pts)))
        return 0

    def nvenc_hdr_get_counters(self, encoder, out):
        _write_into(out, self.counters)
        return 0

    def nvenc_hdr_get_contract(self, encoder, out):
        _write_into(out, self.contract)
        return 0

    def nvenc_hdr_sei_get(self, encoder, out):
        _write_into(out, self.sei)
        return 0

    def nvenc_last_error(self):
        return b"stub encoder refusal"

    def nvenc_finish(self, encoder):
        self.symbols.append("nvenc_finish")
        return 0

    def nvenc_close(self, encoder):
        self.symbols.append("nvenc_close")
        TEARDOWN.append("encoder")


class FakeBridge:
    """The colour bridge, recording what it was opened with and asked for.

    The planes it returns are the real ``hdr_gpu_engine.DevicePlane`` objects, so the
    descriptors the chain hands on are the real shape: a wrong format string, a
    full-size proxy or a short pitch fails the shipping host's own validation.

    The multiscale half - ``import_master``, ``observe``, ``set_composite`` - validates
    what it is handed through the real wrapper's own checks, so an extra layer that was
    fed the 8-bit proxy, a plane of another extent, a texture from another layer's
    session or a fused composite with no fence behind it fails for the reason the
    compiled bridge would give, and every instance answers for itself: the ``calls``
    ledger names the working extent of the bridge each entry was made on.
    """

    calls = []
    # A colour bridge that cannot allocate its working plane is a real open failure,
    # and the only way to see what a half-opened layer vector does about it.
    constructed = 0
    refuse_after = None
    fail_close_tags = set()
    report_readback_by_tag = {}

    def __init__(self, *, d3d12_device, cuda_context, full_width, full_height,
                 working_width, working_height, paper_white_scrgb, library=None):
        type(self).constructed += 1
        if (type(self).refuse_after is not None
                and type(self).constructed > type(self).refuse_after):
            raise hdr.HdrBridgeError(
                f"gpu_hdr_open reported -1 for bridge "
                f"{type(self).constructed}: the working plane could not be allocated")
        type(self).calls.append({"d3d12_device": d3d12_device,
                                "cuda_context": cuda_context,
                                "full": (full_width, full_height),
                                "working": (working_width, working_height),
                                "paper_white_scrgb": paper_white_scrgb})
        self._full = (full_width, full_height)
        self._working = (working_width, working_height)
        self._cuda = int(cuda_context)
        self.tag = "%dx%d" % (working_width, working_height)
        self.pitches = {"full": full_width * 8, "working": working_width * 8}
        self.closed = False
        # What this bridge was actually asked to do. The counters below are read off
        # these tallies, so a per-layer traffic figure in a report is a consequence of
        # the calls this run made on this bridge and not a constant the stub recites.
        self.passes = 0
        self.decodes = 0
        self.downscales = 0
        self.proxies = 0
        self.observes = 0
        self.reconstructs = 0
        self.encodes = 0
        self.imports = 0
        self.composites = 0

    # The bridge's own rules for a borrowed producer texture, taken from the class the
    # chain would have called. A working-extent texture handed to a full-extent entry,
    # an 8-bit format or a composite without its fence is refused here exactly as the
    # compiled bridge refuses it, in its own words rather than the test's.
    _producer_texture = hdr.GpuHdrBridge._producer_texture

    @property
    def contract(self):
        return {"scrgb_per_nit": 0.0125, "nits_per_scrgb": 80.0}

    def _plane(self, fmt, size, bpp, plane, pitch):
        return hdr.DevicePlane(fmt=fmt, pixel_format=fmt, width=size[0],
                               height=size[1], cuda_context=0x9000,
                               device_ptr=0xC000 + plane * 16, pitch_bytes=pitch,
                               bytes_per_pixel=bpp, plane=plane)

    def decode(self, **kwargs):
        type(self).calls.append(("decode", kwargs, self.tag))
        self.passes += 1
        self.decodes += 1
        return self._plane(hdr.FORMAT_RGBA16F_CUDA, self._full, 8, 2,
                           self.pitches["full"])

    def source_report(self):
        return {"support_ceiling_scrgb": 250.0, "paper_white_scrgb": 8.0,
                "nonfinite_samples": 0, "negative_components": 221184,
                "above_ceiling_components": 0, "min_component": -0.31,
                "max_component": 13.99, "pixels": 1920 * 1080}

    def downscale(self):
        type(self).calls.append(("downscale", self.tag))
        self.passes += 1
        self.downscales += 1
        return self._plane(hdr.FORMAT_RGBA16F_CUDA, self._working, 8, 3,
                           self.pitches["working"])

    def proxy(self):
        type(self).calls.append(("proxy", self.tag))
        self.passes += 1
        self.proxies += 1
        return (self._plane(hdr.FORMAT_RGBA16F_CUDA, self._working, 8, 4,
                            self.pitches["working"]),
                self._plane(hdr.FORMAT_RGBA8_CUDA, self._working, 4, 5,
                            self._working[0] * 4))

    def proxy_report(self):
        return {"negative_components": 221184, "pinned_components": 0,
                "dead_zone_components": 128, "above_actionable_components": 0,
                "above_saturation_components": 0, "min_code": 0.0, "max_code": 1.0,
                "pixels": 960 * 544}

    def reconstruct(self, dlss_output, residual_strength=1.0):
        texture, _fence, value = self._producer_texture(
            dlss_output, None, None, extent=self._working,
            extent_name="working", label="Feature 18 output",
            role="reconstruction")
        type(self).calls.append(("reconstruct", {
            "dxgi_format": int(dlss_output["dxgi_format"]),
            "residual_strength": float(residual_strength),
            "texture": int(texture), "fence_value": int(value)}, self.tag))
        self.passes += 1
        self.reconstructs += 1
        return self._plane(hdr.FORMAT_RGBA16F_CUDA, self._full, 8, 7,
                           self.pitches["full"])

    def import_master(self, source):
        """Take this layer's colour from a borrowed pristine master, validated.

        The check is the real wrapper's: format, extent, pitch and CUDA context. A
        chain that handed an extra layer its own 8-bit proxy, the working plane of
        another layer or a plane from another context is refused here, in the words
        ``hdr_gpu_engine`` uses, and never copies anything as if it were HDR colour.
        """
        plane = hdr._require_scrgb16f_plane(
            source, extent=self._full, cuda_context=self._cuda,
            label="imported master")
        type(self).calls.append(("import_master", plane, self.tag))
        self.imports += 1
        return self._plane(hdr.FORMAT_RGBA16F_CUDA, self._full, 8,
                           hdr.PLANE_MASTER_FULL, self.pitches["full"])

    def observe(self, dlss_output, fence=None, fence_value=None):
        """This layer's Feature 18 output -> its own signed working residual.

        The returned plane is half-float at *this* bridge's working extent, which is
        the shape the residual has: a linear scRGB number that may be negative, not a
        code in [0, 1] and not the proxy difference of another layer.
        """
        self._producer_texture(dlss_output, fence, fence_value,
                               extent=self._working, extent_name="working",
                               label="Feature 18 output", role="observation")
        type(self).calls.append(("observe", {
            "texture": int(dlss_output["d3d12_resource"]),
            "fence_value": int(dlss_output["fence_value"])}, self.tag))
        self.passes += 1
        self.observes += 1
        return self._plane(hdr.FORMAT_RGBA16F_CUDA, self._working, 8,
                           hdr.PLANE_WORKING_RESIDUAL, self.pitches["working"])

    def set_composite(self, composite_output, fence=None, fence_value=None):
        """The fused full-resolution composite -> this bridge's composited plane.

        A producer fence and a non-zero value are required, as the real entry requires
        them: the solver writes the texture on another queue, and an import that did
        not wait for that store would encode a half-written frame.
        """
        texture, fence_ptr, value = self._producer_texture(
            composite_output, fence, fence_value, extent=self._full,
            extent_name="full", label="fused composite", role="composite import")
        if not int(fence_ptr or 0) or not int(value or 0):
            raise hdr.HdrBridgeError(
                "the fused composite needs its producer fence and value: the solver "
                "is still writing it on another queue")
        type(self).calls.append(("set_composite", {
            "texture": int(texture), "fence": int(fence_ptr),
            "fence_value": int(value), "dxgi_format": int(
                composite_output["dxgi_format"])}, self.tag))
        self.composites += 1
        return self._plane(hdr.FORMAT_RGBA16F_CUDA, self._full, 8,
                           hdr.PLANE_COMPOSITED, self.pitches["full"])

    def reconstruct_report(self):
        return {"applied_components": 300, "gain_cap_components": 7,
                "rejected_pinned_components": 0, "rejected_dead_components": 128,
                "rejected_negative_components": 221184, "nonfinite_components": 0,
                "composite_out_of_domain_components": 0,
                "composite_nonfinite_components": 0, "max_abs_stops": 1.5,
                "max_abs_residual_scrgb": 0.4, "composite_min_component": -0.31,
                "composite_max_component": 13.99, "pixels": 1920 * 1080}

    def encode(self, *, clip_ceiling_scrgb):
        type(self).calls.append(("encode", clip_ceiling_scrgb, self.tag))
        self.passes += 1
        self.encodes += 1
        return hdr.P010Output(
            self._plane(hdr.FORMAT_P010_CUDA, self._full, 2, 8, 4096),
            self._plane(hdr.FORMAT_P010_CUDA, (self._full[0] // 2, self._full[1] // 2),
                        4, 9, 4096))

    def encode_report(self):
        return {"clip_ceiling_scrgb": 125.0, "clamped_high_components": 0,
                "clamped_low_components": 4096, "out_of_gamut_components": 12,
                "nonfinite_components": 0, "max_linear_component": 13.99,
                "max_code_y": 1023.0, "min_code_y": 64.0, "max_code_uv": 960.0,
                "min_code_uv": 64.0}

    def transfer_counters(self):
        full, working = self._full, self._working
        return {"process_count": self.passes,
                "source_device_copy_bytes": self.decodes * full[0] * full[1] * 3,
                # An imported frame of colour is the same full-resolution FP16 write
                # a decoded one is, which is what the compiled bridge counts here.
                "source_device_write_bytes": (self.decodes + self.imports)
                * full[0] * full[1] * 8,
                "working_device_write_bytes": self.downscales * working[0]
                * working[1] * 8,
                "guidance_device_write_bytes": self.proxies * working[0] * working[1] * 4,
                "composited_device_write_bytes": (self.reconstructs + self.composites)
                * full[0] * full[1] * 8,
                "encode_device_write_bytes": self.encodes * full[0] * full[1] * 3,
                # report_readback_bytes is the bridge's counter for image planes the
                # diagnostic gpu_hdr_readback_* entries copied out. Production binds
                # none of them and answers 0; this stub answers a non-zero on purpose,
                # so a report that shows it proves the field is wired to the bridge
                # rather than hardcoded. It has never been a measure of the per-pass
                # scalar counter readback.
                "report_readback_bytes": type(self).report_readback_by_tag.get(
                    self.tag, self.passes * 96)}

    def is_alive(self):
        return not self.closed

    @property
    def full_extent(self):
        return self._full

    @property
    def working_extent(self):
        return self._working

    @property
    def paper_white(self):
        return 8.0

    def multiscale_counters(self):
        """The optional group's own attribution, read off what this bridge was asked.

        The shape of the answer is the compiled bridge's: an imported master is a
        full-resolution FP16 volume and no dispatch, an observation copies its Feature
        18 texture in and writes a residual of the same size at the working extent,
        and ``pass_dispatches`` counts only the entries that ran a shader - so a layer
        that imported and observed five times reports five dispatches more than a
        bridge that only imported, and the scalar counter readback each dispatch
        crosses stays a range rather than a claim of nothing.
        """
        full, working = self._full, self._working
        return {"master_provenance": 2 if self.imports else (1 if self.decodes else 0),
                "master_import_count": self.imports,
                "master_import_device_bytes": self.imports * full[0] * full[1] * 8,
                "observe_count": self.observes,
                "observe_copy_bytes": self.observes * working[0] * working[1] * 8,
                "observe_residual_bytes": self.observes * working[0] * working[1] * 8,
                "composite_import_count": self.composites,
                "composite_import_bytes": self.composites * full[0] * full[1] * 8,
                "pass_dispatches": self.passes}

    def close(self):
        type(self).calls.append(("close", self.tag))
        self.closed = True
        TEARDOWN.append("colour bridge")
        if self.tag in type(self).fail_close_tags:
            raise RuntimeError(f"colour bridge close failed at {self.tag}")


class StubCuda:
    pushed = []
    popped = []

    def __init__(self):
        pass

    def init(self):
        return 0

    def push(self, context):
        type(self).pushed.append(int(context))
        return True

    def pop(self):
        type(self).popped.append(len(type(self).pushed))
        TEARDOWN.append("context")
        return 0


class LayerSession(Fp16Live):
    """A Feature 18 session opened *as a layer*, by the chain's own construction call.

    Stands in for ``dlss_gpu_engine.Live`` and insists on the arguments that make a
    layer a layer rather than a second call on one session: its own extent, the anchor
    whose core it shares, and the eight-byte colour an FP16 proxy needs. ``refuse_at``
    fails one open, which is the only way a partial-open rollback is observable.
    """

    created = []
    refuse_at = None
    fail_close_sizes = set()

    def __init__(self, width, height, settings=None, *, shared_with=None,
                 color_format="rgba8", **kwargs):
        if type(self).refuse_at == len(type(self).created):
            raise dge.DLSSInferenceError(
                f"open_layer_ex refused layer {len(type(self).created)}: the anchor's "
                "core reports no second Feature 18 session")
        super().__init__(width=width, height=height, color_format=color_format)
        self.shared_with = shared_with
        self.settings_given = settings
        # Which of the two the chain asked for, kept separate from what this session
        # ended up as: asking for the 8-bit colour and getting an FP16 session would be
        # a bug, and so is the reverse, and neither is visible in one field.
        self.color_format_given = color_format
        type(self).created.append(self)

    def close(self):
        TEARDOWN.append(f"feature18[{self._w}x{self._h}]")
        self.closed = True
        if (self._w, self._h) in type(self).fail_close_sizes:
            raise RuntimeError(f"layer session close failed at {self._w}x{self._h}")


class Solver:
    """The typed data-consistency solver, opened on the layer vector it was given.

    ``layer_options`` and ``_require_scrgb16f_plane`` are the real wrapper's own,
    borrowed the way the FP16 session borrows the host's descriptor checks: a layer
    vector that could not open the compiled solver is refused here, and so is a master
    or a residual that is not a half-float plane at the extent it is attributed to.
    What it hands back is the shape ``set_composite`` consumes - a full-extent
    R16G16B16A16_FLOAT texture in COPY_SOURCE with its producer fence and a non-zero
    value - and the solve it reports is converged against the float32 tolerance it was
    actually sent. Whether the solve happened, and what it was fed, is in ``calls``.
    """

    opened = []
    calls = []
    closed = 0
    refuse_open = False
    solve_fails_at = None
    unrepresentable = 0

    def __init__(self, *, d3d12_device, full_width, full_height, layers, beta,
                 residual_strength, relative_tolerance, max_iterations,
                 pixel_bytes=hfe.HDR_PIXEL_BYTES, library=None):
        if type(self).refuse_open:
            raise hfe.HdrFusionError(
                "gpu_fusion_open_hdr reported 5 (the first allocation failed)")
        self.options = hfe.layer_options(full_width=full_width,
                                         full_height=full_height, layers=layers)
        # The same four scalars the real constructor marshals onto the options after the
        # layer table is built, in the same half-precision shape: the solver compares its
        # own ratio against the stored float32, so the tolerance a frame is judged by is
        # this value and not the caller's Python double.
        self.options.beta = C.c_float(beta)
        self.options.residual_strength = C.c_float(residual_strength)
        self.options.relative_tolerance = C.c_float(relative_tolerance)
        self.options.max_iterations = int(max_iterations)
        self._full = (int(full_width), int(full_height))
        self._layers = list(layers)
        self.device = int(d3d12_device)
        self.solves = 0
        self.master_bytes = 0
        self.observation_bytes = 0
        self.scalar_bytes = 0
        self.unrepresentable_total = 0
        type(self).opened.append({
            "d3d12_device": self.device, "full": self._full,
            "layers": [{"dims": tuple(layer["dims"]), "weight": layer["weight"],
                        "plan": type(layer["plan"]).__name__}
                       for layer in self._layers],
            "beta": self.beta, "residual_strength": self.residual_strength,
            "relative_tolerance": self.relative_tolerance,
            "max_iterations": self.max_iterations, "library": library})

    @property
    def beta(self):
        return float(self.options.beta)

    @property
    def residual_strength(self):
        return float(self.options.residual_strength)

    @property
    def full_extent(self):
        return self._full

    @property
    def working_extents(self):
        return [(int(layer["dims"][0]), int(layer["dims"][1]))
                for layer in self._layers]

    @property
    def relative_tolerance(self):
        return float(self.options.relative_tolerance)

    @property
    def max_iterations(self):
        return int(self.options.max_iterations)

    def process(self, *, cuda_context, master, observations):
        context = int(cuda_context)
        source = hfe._require_scrgb16f_plane(
            master, extent=self._full, cuda_context=context,
            label="fused scRGB master")
        if len(observations) != len(self._layers):
            raise hfe.HdrFusionError(
                f"the bridge was opened for {len(self._layers)} layers and "
                f"{len(observations)} observations were given")
        handed = []
        for index, observation in enumerate(observations):
            dims = self._layers[index]["dims"]
            handed.append(hfe._require_scrgb16f_plane(
                observation, extent=(int(dims[0]), int(dims[1])),
                cuda_context=context,
                label=f"layer {index} signed working residual"))
        frame = len(type(self).calls)
        type(self).calls.append({"cuda_context": context, "frame": frame,
                                 "master": source, "observations": handed})
        if type(self).solve_fails_at == frame:
            raise hfe.HdrFusionError(
                "gpu_fusion_process_hdr reported 7 (the observation could not be "
                "read): this bridge is marked failed and will refuse further frames")
        tolerance = self.relative_tolerance
        initial = 0.5367
        relative = tolerance * 0.4
        unrepresentable = int(type(self).unrepresentable)
        self.solves += 1
        self.master_bytes += source["width"] * source["height"] * 8
        self.observation_bytes += sum(plane["width"] * plane["height"] * 8
                                      for plane in handed)
        # Two dot reductions per CG iteration, each a counted 256-byte counter run.
        self.scalar_bytes += 2 * 256 * 6
        self.unrepresentable_total += unrepresentable
        values = {"iterations": 6, "initial_residual_norm": initial,
                  "final_residual_norm": initial * relative,
                  "relative_residual": relative, "converged": 1,
                  "unrepresentable_components": unrepresentable,
                  "output_format": hfe.DXGI_R16G16B16A16_FLOAT,
                  "output_width": self._full[0], "output_height": self._full[1],
                  "output_state": hfe.D3D12_COPY_SOURCE,
                  "source_device_copy_bytes": self.master_bytes,
                  "observation_device_copy_bytes": self.observation_bytes,
                  "scalar_readback_bytes": self.scalar_bytes}
        return {"d3d12_resource": 0x6000 + frame, "d3d12_fence": 0x7000,
                "fence_value": 100 + frame,
                "dxgi_format": values["output_format"],
                "width": self._full[0], "height": self._full[1],
                "state": values["output_state"],
                "report": {name: values[name] for name in hfe.REPORT_FIELDS},
                "iterations": values["iterations"], "converged": True,
                "unrepresentable_components": unrepresentable}

    def counters(self):
        values = {"process_count": self.solves,
                  "source_device_copy_bytes": self.master_bytes,
                  "observation_device_copy_bytes": self.observation_bytes,
                  "scalar_readback_bytes": self.scalar_bytes,
                  "unrepresentable_components": self.unrepresentable_total}
        return {name: values[name] for name in hfe.COUNTER_FIELDS}

    def close(self):
        type(self).closed += 1
        TEARDOWN.append("solver")


@pytest.fixture
def stubbed(monkeypatch):
    FakeBridge.calls = []
    FakeBridge.constructed = 0
    FakeBridge.refuse_after = None
    FakeBridge.fail_close_tags = set()
    FakeBridge.report_readback_by_tag = {}
    StubCuda.pushed = []
    StubCuda.popped = []
    del TEARDOWN[:]
    monkeypatch.setattr(htx.hdr, "GpuHdrBridge", FakeBridge)
    monkeypatch.setattr(htx, "_Cuda", StubCuda)


@pytest.fixture
def multiscale(stubbed, monkeypatch):
    """Every seam a dual or triple job additionally constructs, stood in for.

    The chain builds a layer session and a solver by name inside the modules that own
    them, so patching those two names is the same seam the production code reads - and
    the colour bridge the layers are opened on comes from ``stubbed``.
    """
    LayerSession.created = []
    LayerSession.refuse_at = None
    LayerSession.fail_close_sizes = set()
    Solver.opened = []
    Solver.calls = []
    Solver.closed = 0
    Solver.refuse_open = False
    Solver.solve_fails_at = None
    Solver.unrepresentable = 0
    monkeypatch.setattr(dge, "Live", LayerSession)
    monkeypatch.setattr(hfe, "HdrFusionBridge", Solver)


def _bridges(tag=None):
    """The colour-bridge ledger, as ``(kind, bridge)`` pairs, optionally one bridge's.

    Every entry names the bridge it was made on by that bridge's working extent, so
    one list says both which stage ran and *which layer's* bridge ran it - the claim a
    multi-scale run has to be able to make.
    """
    out = []
    for call in FakeBridge.calls:
        if not isinstance(call, tuple):
            continue
        entry = (call[0], call[-1] if len(call) > 1 else None)
        if tag is None or entry[1] == tag:
            out.append(entry)
    return out


def _kinds(tag=None):
    """The whole ledgers kinds in call order, or one bridge's.

    With no tag, a run that opened n colour bridges ends in n closes: each layer gives
    its bridge back first, by reverse ownership, and the primary bridge is the parent's
    to release last.
    """
    return [kind for kind, _bridge in _bridges(tag)]


def _payloads(kind, tag=None):
    """What the bridge was handed on every entry of one kind, in call order.

    Only the entries that carry a payload are read, and the payload is the *validated*
    description - the numbers the compiled bridge would have copied against - so these
    are the assertions that say which plane a layer was fed, not merely that a call
    happened.
    """
    return [call[1] for call in FakeBridge.calls
            if isinstance(call, tuple) and len(call) >= 3 and call[0] == kind
            and (tag is None or call[-1] == tag)]


def _run(tmp_path, bridges=None, negotiation=None, frames=6, cancel=None,
         flow=None, live=None, verify=None, dims=None, transcode=None,
         with_audio=False, on_proc_started=None, on_proc_exited=None, job=None,
         diagnostics=None, caps=None):
    out_path = os.path.join(str(tmp_path), "hdr_out.mkv")
    source = os.path.join(str(tmp_path), "hdr_valid_src.mkv")
    open(source, "wb").write(b"x" * 4096)
    nvdec = (bridges or {}).get("nvdec") or StubNvdec(frames=frames)
    nvenc = (bridges or {}).get("nvenc") or StubNvenc()
    bridges = dict({"nvdec": nvdec, "nvenc": nvenc, "cookies": []}, **(bridges or {}))
    transcode = dict(NO_MUX if transcode is None else transcode)
    # The gate is negotiated from the same job the run is given, so a dual or triple
    # test asks for two things - a layer vector and a build that can serve it - and
    # gets both from one place, the way the product's entry point does.
    caps = dict(HDR_CAPS if caps is None else caps)
    negotiation = negotiation or _negotiate(job=job, dims=dims, transcode=transcode,
                                            with_audio=with_audio, caps=caps)
    progress = []
    result = htx.export_gpu_hdr_transfer(
        video=source, live=live or Fp16Live(),
        flow_provider=flow if flow is not None else DeviceFlowProvider(),
        out_path=out_path, dims=dict(dims or HDR_DIMS), frames=frames, cq=24,
        with_audio=with_audio, transcode_settings=transcode,
        job_settings=dict(job or HDR_JOB),
        negotiation=negotiation, progress=lambda cur, total, msg: progress.append(
            (cur, total)), cancel=cancel, bridges=bridges, hdr_caps=caps,
        on_proc_started=on_proc_started, on_proc_exited=on_proc_exited,
        diagnostics_out=diagnostics,
        verify=verify or (lambda path: (frames, 25.0, 1920, 1080)))
    return result, out_path, nvdec, nvenc, progress


def test_the_chain_drives_every_stage_in_the_order_the_borrows_require(stubbed,
                                                                       tmp_path):
    (_out, report), out_path, nvdec, nvenc, progress = _run(tmp_path)
    assert nvdec.symbols[:3] == ["nvdec_bridge_open_hdr", "nvdec_bridge_next",
                                 "nvdec_bridge_get_hdr_metadata"]
    assert nvdec.symbols.count("nvdec_bridge_next") == 6       # stops at the count
    assert nvdec.symbols.count("nvdec_bridge_get_hdr_metadata") == 6
    assert nvenc.symbols == ["nvenc_hdr_open"] + ["nvenc_encode_p010"] * 6 + \
        ["nvenc_finish", "nvenc_close"]
    kinds = [call[0] for call in FakeBridge.calls if isinstance(call, tuple)]
    assert kinds == ["decode", "downscale", "proxy", "reconstruct", "encode"] * 6 + \
        ["close"]
    assert progress[-1] == (6, 6)
    assert report["hdr_frames_requested"] == 6
    assert report["hdr_frames_processed_measured"] == 6
    assert report["experimental_hdr_scrgb"] is True
    assert report["export_backend_actual"] == "gpu_transfer"
    assert report["transfer_mode_actual"] == "gpu_hdr_transfer"
    assert report["export_backend_fallback_reason"] is None


def test_the_colour_bridge_is_opened_on_the_sessions_device_and_the_jobs_constants(
        stubbed, tmp_path):
    _run(tmp_path)
    opened = FakeBridge.calls[0]
    assert opened["full"] == (1920, 1080) and opened["working"] == (960, 544)
    assert opened["paper_white_scrgb"] == 8.0
    assert opened["d3d12_device"] == 0x3000        # the Live session's own device
    assert opened["cuda_context"] == 0x9000        # the decoder's, borrowed
    assert StubCuda.pushed == [0x9000]
    decodes = [call for call in FakeBridge.calls if isinstance(call, tuple)
               and call[0] == "decode"]
    assert decodes[0][1]["support_ceiling_scrgb"] == 250.0
    # The decoder's own colour enums, not a guess at them.
    assert decodes[0][1]["color_space"] == 9 and decodes[0][1]["color_transfer"] == 16
    encodes = [call for call in FakeBridge.calls if isinstance(call, tuple)
               and call[0] == "encode"]
    assert {call[1] for call in encodes} == {125.0}


def test_the_fp16_proxy_and_the_device_flow_satisfy_the_hosts_own_validation(stubbed,
                                                                            tmp_path):
    live = Fp16Live()
    flow = DeviceFlowProvider()
    _run(tmp_path, live=live, flow=flow)
    assert len(live.descriptors) == 6 and len(flow.calls) == 6
    descriptor = live.descriptors[0]
    assert descriptor["format"] == "rgba16f_cuda"
    assert (descriptor["width"], descriptor["height"]) == (960, 544)
    assert descriptor["pitch_bytes"] >= 960 * 8
    assert live.resets == [True] + [False] * 5
    assert flow.discontinuities == [True] + [False] * 5
    # The composite the bridge then reads is the FP16 texture, at the working extent.
    reconstructs = [call for call in FakeBridge.calls
                    if isinstance(call, tuple) and call[0] == "reconstruct"]
    assert {call[1]["dxgi_format"] for call in reconstructs} == {10}


def test_the_encoder_is_opened_once_with_the_sources_own_metadata(stubbed, tmp_path):
    """One open for the job, carrying the fixture's numbers rather than defaults.

    The mastering display and CLL/FALL the source declared have to reach the encoder
    verbatim, or the SEI the writer authors is a claim about a display nobody graded.
    """
    nvenc = StubNvenc()
    _run(tmp_path, bridges={"nvenc": nvenc})
    assert len(nvenc.options) == 1 == len(nvenc.attached)
    attached = nvenc.attached[0]
    assert (attached.has_mastering, attached.has_content_light) == (1, 1)
    assert attached.max_cll == 900 and attached.max_fall == 250
    source = _fixture_metadata()
    assert list(attached.primary_x_num) == list(source.primary_x_num)
    assert list(attached.primary_y_den) == list(source.primary_y_den)
    assert (attached.max_luminance_num, attached.max_luminance_den) == (
        source.max_luminance_num, source.max_luminance_den)
    assert (attached.white_x_num, attached.white_y_den) == (
        source.white_x_num, source.white_y_den)
    options = nvenc.options[0]
    # The five numeric declarations come from the installed entry's own defaults, so
    # the Python side cannot be the place an enum is invented.
    assert (options.color_range, options.color_primaries) == (1, 9)
    assert (options.color_trc, options.color_space) == (16, 9)
    assert options.constant_quality == 24


def test_the_encoder_is_handed_the_borrowed_p010_planes_and_the_frame_pts(stubbed,
                                                                          tmp_path):
    (_out, _report), _path, _nvdec, nvenc, _progress = _run(tmp_path)
    assert [entry[4] for entry in nvenc.submitted] == list(range(6))
    assert all(entry[1] == 4096 and entry[3] == 4096 for entry in nvenc.submitted)
    attached = nvenc.attached[0]
    assert attached.max_cll == 900 and attached.max_fall == 250
    assert len(nvenc.options) == 1                     # opened once, never per frame


def test_a_source_that_declares_nothing_gets_nothing_attached(stubbed, tmp_path):
    """Absence stays absence: the open is made with NULL rather than a default."""
    nvdec = StubNvdec(metadata=htx.HdrMetadata(C.sizeof(htx.HdrMetadata)))
    nvenc = StubNvenc()
    _run(tmp_path, bridges={"nvdec": nvdec, "nvenc": nvenc})
    assert nvenc.attached[0] is None


def test_metadata_that_moves_mid_job_is_refused_rather_than_reported_as_preserved(
        stubbed, tmp_path):
    first = _fixture_metadata()
    later = _fixture_metadata(max_cll=800)

    class Moving(StubNvdec):
        def nvdec_bridge_get_hdr_metadata(self, handle, out):
            _write_into(out, first if self.index <= 1 else later)
            self._log("nvdec_bridge_get_hdr_metadata")
            return 0

    with pytest.raises(htx.HdrExportError) as raised:
        _run(tmp_path, bridges={"nvdec": Moving()})
    assert "changed at frame" in str(raised.value)
    assert "max_cll" in str(raised.value)


def test_a_partial_mastering_display_is_refused_before_the_encoder_opens(stubbed,
                                                                        tmp_path):
    half = _fixture_metadata(mastering_has_luminance=0)
    nvenc = StubNvenc()
    with pytest.raises(htx.HdrExportError) as raised:
        _run(tmp_path, bridges={"nvdec": StubNvdec(metadata=half), "nvenc": nvenc})
    assert "half present" in str(raised.value)
    assert nvenc.symbols == []


def test_an_8_bit_frame_from_the_decoder_is_refused_not_read_as_10_bit(stubbed,
                                                                       tmp_path):
    with pytest.raises(htx.HdrExportError) as raised:
        _run(tmp_path, bridges={"nvdec": StubNvdec(software_format=1)})
    assert "software_format" in str(raised.value)


def test_the_unguided_chain_drives_the_zero_motion_entry_and_no_flow_provider(stubbed,
                                                                             tmp_path):
    negotiation = _negotiate(job=dict(HDR_JOB, guidance_mode=0))
    assert negotiation["supported"], negotiation["reasons"]
    (_out, report), _p, _n, _e, _pr = _run(
        tmp_path, negotiation=negotiation, flow=BannedFlowProvider(),
        live=Fp16Live())
    assert report["hdr_dlss_zero_motion_calls_measured"] == 6
    assert report["hdr_dlss_flow_calls_measured"] == 0


def test_a_guided_frame_is_never_drawn_from_zero_motion_when_the_flow_engine_refuses(
        stubbed, tmp_path):
    class Refusing(DeviceFlowProvider):
        def get_flow_device_rgba(self, *args, **kwargs):
            raise RuntimeError("NVOF device RGBA input is unavailable")

    with pytest.raises(RuntimeError) as raised:
        _run(tmp_path, flow=Refusing())
    assert "NVOF" in str(raised.value)


def test_a_feature18_refusal_is_the_sessions_own_words_and_not_retried_elsewhere(
        stubbed, tmp_path):
    with pytest.raises(dge.DLSSInferenceError) as raised:
        _run(tmp_path, live=RaisingLive())
    assert "FP16 frame" in str(raised.value)


def test_close_releases_in_reverse_owner_order_even_after_a_failed_frame(stubbed,
                                                                        tmp_path):
    nvdec, nvenc = StubNvdec(), StubNvenc()
    with pytest.raises(dge.DLSSInferenceError):
        _run(tmp_path, bridges={"nvdec": nvdec, "nvenc": nvenc}, live=RaisingLive())
    # The frame died at inference, so the encoder never saw a plane - but it still
    # goes first, because it was opened last.
    assert nvenc.symbols == ["nvenc_hdr_open", "nvenc_close"]
    assert TEARDOWN == ["encoder", "colour bridge", "feature18", "context", "decoder"]
    assert FakeBridge.calls[-1] == ("close", "960x544")
    assert StubCuda.popped == [1]
    assert nvdec.symbols[-1] == "nvdec_bridge_close"


def test_a_cancelled_job_stops_between_frames_and_takes_the_decoder_with_it(stubbed,
                                                                           tmp_path):
    nvdec, nvenc = StubNvdec(), StubNvenc()
    flags = [False, False, True]

    with pytest.raises(pipeline.JobCancelled):
        _run(tmp_path, bridges={"nvdec": nvdec, "nvenc": nvenc},
             cancel=lambda: flags.pop(0) if flags else True)
    # Cancel is asked before the next frame is borrowed, and the decoder is told to
    # stop before anything is released, so the reading thread never sits in the call
    # the teardown is about to free.
    assert "nvdec_bridge_cancel" in nvdec.symbols
    assert nvdec.symbols.index("nvdec_bridge_cancel") < nvdec.symbols.index(
        "nvdec_bridge_close")
    assert "nvenc_finish" not in nvenc.symbols       # a cancelled job flushes nothing
    assert nvenc.symbols.count("nvenc_encode_p010") == 2   # the two frames it finished
    assert nvenc.symbols[0] == "nvenc_hdr_open"
    assert nvenc.symbols[-1] == "nvenc_close"
    assert TEARDOWN == ["encoder", "colour bridge", "feature18", "context", "decoder"]


def test_the_report_never_fabricates_a_zero_for_a_boundary_that_never_ran(stubbed,
                                                                         tmp_path):
    (_out, report), _p, _n, _e, _pr = _run(tmp_path)
    assert report["decode_host_copy_bytes_measured"] == 0     # the stub's own answer
    assert report["hdr_pixel_readback_bytes_measured"] is None
    assert report["encoder_host_pixel_bytes_measured"] == 0
    assert report["encoder_rgb_conversion_bytes_measured"] == 0
    assert report["hdr_metadata_stable_across_frames"] is True
    assert report["hdr_metadata_snapshot_count_measured"] == 6
    assert report["encoder_in_bitstream_sei_written"] is True
    assert "in_bitstream_prefix_sei" in report["encoder_metadata_destinations"]
    assert "container_colour_tags" in report["encoder_metadata_destinations"]
    assert report["hdr_pixel_readback_exports_bound"] == []
    assert report["decoder_pixel_readback_entry_bound"] is False
    # A stage that never opened has no key: nothing was counted because nothing ran.
    assert "dlss_host_color_stage_bytes_measured" in report
    for name in ("total_source_negative_components", "total_proxy_pinned_components",
                 "total_reconstruct_rejected_negative_components",
                 "total_encode_clamped_low_components",
                 "total_encode_out_of_gamut_components"):
        assert name in report, name
    assert report["total_source_negative_components"] == 6 * 221184
    assert report["total_reconstruct_rejected_negative_components"] == 6 * 221184
    assert report["total_encode_clamped_low_components"] == 6 * 4096
    assert report["total_source_max_component_envelope"] == 13.99


def test_the_image_readback_counter_is_not_reported_as_the_scalar_boundary(stubbed,
                                                                          tmp_path):
    """Two boundaries the accepted record held in one field, read apart again.

    ``report_readback_bytes`` is the bridge's count of whole image planes copied out by
    the diagnostic ``gpu_hdr_readback_*`` entries, which this chain never binds. The
    per-pass counter readback is something else: scalars that come to the host on every
    executed dispatch, measured by no field of the ABI. So the report carries the first
    under its own name, answers the second with None, and names the source-derived
    per-dispatch figure in a key that cannot be read as a measurement.
    """
    (_out, report), _p, _n, _e, _pr = _run(tmp_path)
    assert report["hdr_diagnostic_image_readback_bytes_measured"] == 6 * 5 * 96
    assert report["hdr_scalar_report_readback_bytes_measured"] is None
    scalar = report["hdr_scalar_report_readback"]
    assert scalar["crosses_host"] is True
    assert scalar["bytes_measured"] is None
    assert scalar["measured_by_this_report"] is False
    assert scalar["per_dispatch_bytes_from_source"] == 256
    # The complete picture-byte figure stays unanswered rather than being borrowed from
    # either counter.
    assert report["hdr_pixel_readback_bytes_measured"] is None
    # And nothing folds an unmeasured boundary into a total.
    assert not [key for key in report if key.startswith("total_")
                and "readback" in key]
    assert "host_traffic" in report["hdr_not_claimed"]
    assert "one, two or three reconstruction layers are supported" in \
        report["hdr_not_claimed"]["layers"]
    assert "multi_pass greater than 1 is not supported" in \
        report["hdr_not_claimed"]["layers"]
    assert "Not every source and scale combination has been verified" in \
        report["hdr_not_claimed"]["layers"]


def test_counters_a_bridge_does_not_answer_stay_unanswered_rather_than_becoming_zero(
        stubbed, tmp_path):
    class Silent(StubNvenc):
        def nvenc_hdr_get_counters(self, encoder, out):
            return 1                     # the bridge declined to answer

    (_out, report), _p, _n, _e, _pr = _run(tmp_path,
                                           bridges={"nvenc": Silent()})
    # A 0 would read as "this boundary was crossed no times", which is a claim only the
    # encoder may make; the encoder's silence is reported as its own answer.
    assert report["encoder_frames_submitted_measured"] is None
    assert report["encoder_p010_device_copy_bytes_measured"] is None
    # what the same encoder did answer, on another entry, is still its numbers
    assert report["encoder_in_bitstream_sei_written"] is True


def test_a_short_decode_is_tolerated_only_within_the_containers_own_rounding(stubbed,
                                                                            tmp_path):
    with pytest.raises(RuntimeError) as raised:
        _run(tmp_path, frames=60, bridges={"nvdec": StubNvdec(frames=6)})
    assert "提前终止" in str(raised.value)


def test_a_run_one_frame_short_of_the_requested_count_is_not_a_failure(stubbed,
                                                                       tmp_path):
    """The container's rounding, not a licence to truncate.

    One frame short of six is inside the tolerance the piped backend already applies, so
    the run is judged on the file it wrote instead of failing on an index that never
    landed. A shortfall past the tolerance is the test above.
    """
    (_out, report), _p, _n, _e, _pr = _run(
        tmp_path, frames=6, bridges={"nvdec": StubNvdec(frames=5)},
        verify=lambda path: (5, 25.0, 1920, 1080))
    assert report["hdr_frames_requested"] == 6
    assert report["hdr_frames_processed_measured"] == 5


def test_a_run_that_wrote_nothing_is_not_verified_as_a_success(stubbed, tmp_path):
    with pytest.raises(Exception):
        _run(tmp_path, verify=lambda path: (0, 25.0, 1920, 1080))


def test_output_that_resolves_to_the_source_refuses_before_anything_opens(tmp_path):
    source = os.path.join(str(tmp_path), "hdr_valid_src.mkv")
    open(source, "wb").write(b"x" * 4096)
    with pytest.raises(ValueError):
        htx.export_gpu_hdr_transfer(
            video=source, live=Fp16Live(), flow_provider=DeviceFlowProvider(),
            out_path=source, dims=dict(HDR_DIMS), frames=6, cq=24, with_audio=False,
            transcode_settings=dict(NO_MUX), job_settings=dict(HDR_JOB),
            negotiation=_negotiate())


# ---------------------------------------------------------------------------
# the reused copy pass: what the run hands it, and what a failure leaves behind
# ---------------------------------------------------------------------------

RICH = dict(NO_MUX, copy_attachments=True, copy_metadata=True, copy_chapters=True,
            subtitle_mode="copy", audio_mode="copy")


def _clause(command, clause):
    """Whether the clause appears in the command, in order and next to itself."""
    return [command[i:i + len(clause)] for i in range(len(command) - len(clause) + 1)
            ].count(list(clause)) > 0


class FakeFfmpeg:
    """The remux process, recorded and answered without a binary in the way.

    ``Popen`` is the only thing the copy pass touches outside its own command builder,
    so standing in for it lets a test read the real ``gtx.build_mux_command`` output for
    the real job rather than a command the test wrote itself. What it keeps is what the
    pass was asked to do and when it was asked, which is how the ordering, the cleanup
    and the failure paths are observable at all.
    """

    def __init__(self, returncode=0, stderr=b"", hang=False, deliver=True):
        self.commands = []
        self.teardown_at_call = []
        self.procs = []
        self.kills = 0
        self.returncode = returncode
        self.stderr = stderr
        self.hang = hang
        self.deliver = deliver

    def popen(self, command, **kwargs):
        self.commands.append(list(command))
        # Whatever the encoder had already released by the time FFmpeg is started: a
        # remux that began while the bridge still held the file open would be reading a
        # bitstream whose trailer had not been written.
        self.teardown_at_call.append(list(TEARDOWN))
        proc = self._Proc(self, command)
        self.procs.append(proc)
        return proc

    class _Proc:
        def __init__(self, owner, command):
            self._owner = owner
            self._command = command
            self.returncode = None

        def communicate(self, timeout=None):
            if self._owner.hang:
                self._owner.hang = False
                raise subprocess.TimeoutExpired(self._command, timeout)
            self.returncode = self._owner.returncode
            if self.returncode == 0 and self._owner.deliver:
                with open(self._command[-1], "wb") as handle:
                    handle.write(b"m" * 4096)
            return None, self._owner.stderr

        def kill(self):
            self._owner.kills += 1

    @property
    def command(self):
        return self.commands[0]


@pytest.fixture
def remux_factory(monkeypatch):
    def build(**kwargs):
        ffmpeg = FakeFfmpeg(**kwargs)
        monkeypatch.setattr(gtx.subprocess, "Popen", ffmpeg.popen)
        monkeypatch.setattr(pipeline, "_bundle_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(pipeline, "_bundle_ffprobe", lambda: "ffprobe")
        monkeypatch.setattr(gtx.subprocess, "check_output", lambda *args, **kw: json.dumps({
            "streams": [{"time_base": "1/1000"}],
            "packets": [{"pts": pts, "duration": 40}
                        for pts in (0, 40, 80, 120, 160, 200)]}).encode())
        return ffmpeg
    return build


@pytest.fixture
def remux(remux_factory):
    return remux_factory()


def _rich_run(tmp_path, remux, verify=None, **kwargs):
    """One accepted copy-pass job, with the path it verified at the end recorded."""
    verified = []

    def probe(path):
        verified.append(path)
        return (6, 25.0, 1920, 1080)

    result, out_path, _nvdec, nvenc, _progress = _run(
        tmp_path, transcode=RICH, with_audio=True, verify=verify or probe, **kwargs)
    return result, out_path, nvenc, verified


def test_a_copy_pass_job_encodes_into_a_unique_intermediate_and_delivers_the_rewrap(
        stubbed, tmp_path, remux):
    (out, report), out_path, nvenc, verified = _rich_run(tmp_path, remux)
    encoded = nvenc.paths[0]
    assert encoded != out_path, "the encoder must not write the deliverable itself"
    assert encoded.endswith(gtx.INTERMEDIATE_CONTAINER)
    assert os.path.basename(encoded).startswith("dlss_gpu_hdr_")
    # Input 0 is the encode and input 1 is the source: the original video is never
    # mapped into the file the caller is handed.
    command = remux.command
    assert command[0] == "ffmpeg" and command[-1] == out_path
    assert _clause(command, ["-i", encoded])
    assert _clause(command, ["-i", os.path.join(str(tmp_path), "hdr_valid_src.mkv")])
    assert _clause(command, ["-map", "0:v:0"]) and _clause(command, ["-c:v", "copy"])
    assert "1:v" not in command, "the source's own video must not reach the deliverable"
    # The streams the request asked for, expressed by the shared builder rather than
    # by a mapping rule this chain invented.
    for clause in (["-map", "1:a?"], ["-map", "1:s?"], ["-map", "1:t?"],
                   ["-map_metadata", "1"], ["-map_chapters", "1"],
                   ["-c:a", "copy"], ["-c:s", "copy"],
                   ["-c:t", "copy"]):
        assert _clause(command, clause), clause
    assert "-shortest" not in command
    assert _clause(command, ["-t", "0.24"])
    assert report["hdr_copy_pass"] is True
    assert report["hdr_deliverable_written_by"] == "ffmpeg_map_and_copy_pass"
    assert report["hdr_mux_command"] == command
    # The verification is of the file the caller was promised, after the rewrap.
    assert verified == [out_path]
    assert os.path.exists(out_path)
    assert not os.path.exists(encoded), "the intermediate does not outlive the run"


def test_the_run_hands_the_video_track_description_to_the_pass_that_writes_it(
        stubbed, tmp_path, remux):
    """The negotiated fields reach the command, and the report says which ones.

    A gate that derives the clause and a run that forgets to pass it on is the same
    silent loss in a different file, so the whole path is asserted here: negotiation,
    ``gtx.run_copy_pass``, the argument list FFmpeg was given, and the report's own
    record of what it asked for.
    """
    fields = {"title": "Master shot", "language": "jpn",
              "disposition": ["default", "original"]}
    negotiation = _negotiate(transcode=RICH, with_audio=True)
    negotiation["hdr_video_stream_metadata"] = fields
    (_out, report), _path, _nvenc, _verified = _rich_run(tmp_path, remux,
                                                        negotiation=negotiation)
    command = remux.command
    for pair in (["-metadata:s:v:0", "title=Master shot"],
                 ["-metadata:s:v:0", "language=jpn"],
                 ["-disposition:v:0", "+default+original"]):
        assert _clause(command, pair), pair
    assert report["hdr_video_stream_metadata"] == fields
    # The clause names the first output video stream; the encoder's packets still come
    # from input 0 and the source's video is still not mapped.
    assert _clause(command, ["-map", "0:v:0"]) and "1:v" not in command


def test_the_rewrap_starts_only_after_the_encoder_has_released_the_intermediate(
        stubbed, tmp_path, remux):
    _rich_run(tmp_path, remux)
    assert "encoder" in remux.teardown_at_call[0], remux.teardown_at_call[0]
    assert "colour bridge" in remux.teardown_at_call[0]


def test_the_pass_hooks_reach_the_remux_process_so_a_cancel_can_reach_it(
        stubbed, tmp_path, remux):
    started, exited = [], []
    _rich_run(tmp_path, remux, on_proc_started=started.append,
              on_proc_exited=exited.append)
    assert len(started) == 1 and len(exited) == 1
    assert started[0] is exited[0]


def test_a_run_that_needs_nothing_from_the_source_never_starts_ffmpeg(stubbed,
                                                                     tmp_path,
                                                                     remux):
    verified = []
    (out, report), out_path, _nvdec, nvenc, _progress = _run(
        tmp_path, verify=lambda path: (verified.append(path) or (6, 25.0, 1920, 1080)))
    assert report["hdr_copy_pass"] is False
    assert report["hdr_deliverable_written_by"] == "nvenc_hdr_bridge"
    assert remux.commands == [] and "hdr_mux_command" not in report
    assert nvenc.paths[0] == out_path, "the encode is the deliverable"
    assert verified == [out_path]


def test_a_rewrap_that_exits_non_zero_is_a_failed_job_and_leaves_nothing_behind(
        stubbed, tmp_path, remux_factory):
    """No silent fallback: the encoder's video-only file is not a deliverable."""
    ffmpeg = remux_factory(returncode=1, stderr=b"mkv muxer refused the attachment")
    with pytest.raises(gtx.BridgeError) as caught:
        _rich_run(tmp_path, ffmpeg)
    assert "rc=1" in str(caught.value)
    assert "mkv muxer refused" in str(caught.value)
    assert not os.path.exists(os.path.join(str(tmp_path), "hdr_out.mkv"))
    assert not _leftovers(tmp_path)


def test_a_rewrap_that_stops_responding_is_terminated_and_reported(
        stubbed, tmp_path, remux_factory):
    """A hung FFmpeg cannot outlive the job that is already being failed."""
    ffmpeg = remux_factory(hang=True)
    with pytest.raises(gtx.BridgeError) as caught:
        _rich_run(tmp_path, ffmpeg)
    assert "超时" in str(caught.value)
    assert ffmpeg.kills == 1
    assert not _leftovers(tmp_path)


def test_an_encode_that_left_no_intermediate_refuses_before_ffmpeg_is_named(
        stubbed, tmp_path, remux_factory):
    ffmpeg = remux_factory()
    with pytest.raises(htx.HdrExportError) as caught:
        _rich_run(tmp_path, ffmpeg,
                  bridges={"nvenc": StubNvenc(writes_file=False)})
    assert "no intermediate to wrap" in str(caught.value)
    assert ffmpeg.commands == []


def test_a_missing_ffmpeg_is_a_refusal_and_not_a_video_only_delivery(
        stubbed, tmp_path, remux, monkeypatch):
    monkeypatch.setattr(pipeline, "_bundle_ffmpeg", lambda: None)
    with pytest.raises(htx.HdrExportError) as caught:
        _rich_run(tmp_path, remux)
    assert "ffmpeg not found" in str(caught.value)
    assert remux.commands == []
    assert not _leftovers(tmp_path)


def test_a_cancel_between_the_encode_and_the_rewrap_leaves_no_intermediate(
        stubbed, tmp_path, remux):
    """The file the encoder wrote is this run's own, and a cancelled run is over."""
    with pytest.raises(pipeline.JobCancelled):
        _rich_run(tmp_path, remux, cancel=lambda: True)
    assert remux.commands == []
    assert not os.path.exists(os.path.join(str(tmp_path), "hdr_out.mkv"))
    assert not _leftovers(tmp_path)


def _leftovers(tmp_path):
    """Anything this run created under its own unique name that it did not remove."""
    return [name for name in os.listdir(str(tmp_path))
            if name.startswith("dlss_gpu_hdr_")]


@pytest.fixture
def reserved_in_tmp_path(monkeypatch, tmp_path):
    """Hand the run its intermediate name from a directory this test can read.

    ``tempfile.mkstemp`` answers from the system temp directory, where a leftover is
    invisible to an assertion about the test's own tree. Redirected into ``tmp_path``,
    the test can name the exact path the run reserved, prove the encoder really wrote
    that file, and say whether the run left it behind.
    """
    reserved = []
    real = tempfile.mkstemp

    def mkstemp(suffix="", prefix="tmp", dir=None, text=False):
        handle, path = real(suffix=suffix, prefix=prefix, dir=str(tmp_path),
                            text=text)
        reserved.append(path)
        return handle, path

    monkeypatch.setattr(htx.tempfile, "mkstemp", mkstemp)
    return reserved


class NvdecFailingAfter(StubNvdec):
    """A decoder that yields ``fail_at`` frames and then refuses the next one."""

    def __init__(self, fail_at=2, **kwargs):
        StubNvdec.__init__(self, **kwargs)
        self.fail_at = fail_at

    def nvdec_bridge_next(self, handle, frame):
        if self.index >= self.fail_at:
            raise RuntimeError("NVDEC refused the next access unit")
        return StubNvdec.nvdec_bridge_next(self, handle, frame)


class NvencFailingFinish(StubNvenc):
    def nvenc_finish(self, encoder):
        self.symbols.append("nvenc_finish")
        raise RuntimeError("the encoder refused to close its GOP")


class NvencFailingClose(StubNvenc):
    def nvenc_close(self, encoder):
        self.symbols.append("nvenc_close")
        TEARDOWN.append("encoder")
        raise RuntimeError("the encoder handle would not release")


def _assert_intermediate_written_and_gone(nvenc, reserved):
    """The encoder wrote the file the run reserved, and the run removed that file."""
    assert len(reserved) == 1, reserved
    assert "nvenc_hdr_open" in nvenc.symbols, nvenc.symbols
    # The name the run reserved is the name the encoder was handed, so the file whose
    # absence is asserted below is the file this run created - not a path the test
    # invented and nothing ever wrote.
    assert nvenc.paths == reserved, (nvenc.paths, reserved)
    assert not os.path.exists(reserved[0]), reserved[0]


def test_a_frame_failure_after_the_encoder_wrote_its_file_removes_it(
        stubbed, tmp_path, remux, reserved_in_tmp_path):
    """A decode that dies mid-job is not an exemption from cleanup.

    The failure is after the encoder created its file, which is the only point at which
    there is something to leak: a test that cancels before the first frame proves
    nothing about a file that was never written.
    """
    nvdec = NvdecFailingAfter(fail_at=2, frames=6)
    nvenc = StubNvenc()
    with pytest.raises(RuntimeError, match="refused the next access unit"):
        _rich_run(tmp_path, remux, bridges={"nvdec": nvdec, "nvenc": nvenc})
    assert nvenc.symbols.count("nvenc_encode_p010") == 2, "two frames were written"
    _assert_intermediate_written_and_gone(nvenc, reserved_in_tmp_path)


def test_a_cancel_after_the_first_frame_removes_the_intermediate(
        stubbed, tmp_path, remux, reserved_in_tmp_path):
    """The cancel the loop honours between frames leaves no HDR master behind.

    The old cancellation case stopped the job before its first frame, so it could not
    show a file being left behind. This one asks the cancel to wait until the encoder
    has written something.
    """
    nvenc = StubNvenc()
    asks = []

    def cancel():
        asks.append(1)
        return len(asks) > 1

    with pytest.raises(pipeline.JobCancelled):
        _rich_run(tmp_path, remux, cancel=cancel, bridges={"nvenc": nvenc})
    assert len(asks) == 2, "the run got one frame out before it was cancelled"
    assert nvenc.symbols.count("nvenc_encode_p010") == 1
    _assert_intermediate_written_and_gone(nvenc, reserved_in_tmp_path)


def test_a_finish_that_raises_still_removes_the_intermediate(stubbed, tmp_path, remux,
                                                             reserved_in_tmp_path):
    nvenc = NvencFailingFinish()
    with pytest.raises(RuntimeError, match="close its GOP"):
        _rich_run(tmp_path, remux, bridges={"nvenc": nvenc})
    _assert_intermediate_written_and_gone(nvenc, reserved_in_tmp_path)


def test_a_close_that_raises_still_removes_the_intermediate(stubbed, tmp_path, remux,
                                                            reserved_in_tmp_path):
    """A teardown that fails halfway is the case a per-block cleanup cannot cover."""
    nvenc = NvencFailingClose()
    with pytest.raises(RuntimeError, match="would not release"):
        _rich_run(tmp_path, remux, bridges={"nvenc": nvenc})
    _assert_intermediate_written_and_gone(nvenc, reserved_in_tmp_path)
    # Preserve the first failure while attempting every later owner in order;
    # an encoder cleanup error must not leak the model and decoder context.
    assert TEARDOWN == ["encoder", "colour bridge", "feature18", "context", "decoder"], TEARDOWN
    assert nvenc.symbols == ["nvenc_hdr_open"] + ["nvenc_encode_p010"] * 6 + \
        ["nvenc_finish", "nvenc_close"], nvenc.symbols


def test_a_decoder_open_that_raises_still_removes_the_intermediate(
        stubbed, tmp_path, remux, reserved_in_tmp_path):
    """Nothing was written yet, and the guard says so without hiding the reason."""
    nvdec = StubNvdec(frames=6)

    def refusing_open(path, out):
        raise RuntimeError("nvdec_bridge_open_hdr refused the source")

    nvdec.nvdec_bridge_open_hdr = refusing_open
    with pytest.raises(RuntimeError, match="refused the source"):
        _rich_run(tmp_path, remux, bridges={"nvdec": nvdec})
    assert reserved_in_tmp_path and not _leftovers(tmp_path)
    assert not os.path.exists(reserved_in_tmp_path[0])


def test_a_chain_that_cannot_be_built_still_removes_the_intermediate(
        stubbed, tmp_path, remux, reserved_in_tmp_path, monkeypatch):
    """Construction is inside the guard too, and its error is the one that propagates."""
    class Refusing:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no colour bridge for these extents")

    monkeypatch.setattr(htx, "HdrChain", Refusing)
    with pytest.raises(RuntimeError, match="no colour bridge"):
        _rich_run(tmp_path, remux)
    assert len(reserved_in_tmp_path) == 1
    assert not _leftovers(tmp_path)


def test_a_report_that_cannot_be_aggregated_still_removes_the_intermediate(
        stubbed, tmp_path, remux, reserved_in_tmp_path, monkeypatch):
    """The run's own record is a stage of the run, not a stage after its duties.

    Aggregating the per-frame reports sits outside the loop's ``finally`` and before the
    remux block, which is exactly where a cleanup that belongs to the remux alone would
    miss a file the encoder had already written.
    """
    def refusing(frame_reports):
        raise RuntimeError("the report refused to build")

    nvenc = StubNvenc()
    monkeypatch.setattr(htx, "aggregate_hdr_reports", refusing)
    with pytest.raises(RuntimeError, match="refused to build"):
        _rich_run(tmp_path, remux, bridges={"nvenc": nvenc})
    # The chain had fully released its handles by then, so the only thing left to do
    # was the one thing a remux-scoped cleanup would never have reached.
    assert TEARDOWN == ["encoder", "colour bridge", "feature18", "context", "decoder"], \
        TEARDOWN
    _assert_intermediate_written_and_gone(nvenc, reserved_in_tmp_path)
    assert remux.commands == [], "the report failed before the rewrap was named"


def test_a_verification_failure_still_removes_the_intermediate(
        stubbed, tmp_path, remux, reserved_in_tmp_path):
    """The last check before the job is called a success is inside the guard as well.

    The delivered file answers with no readable frame count, which is the rule
    ``verify_export_output`` refuses on - not a stub that never ran, so the rewrap is
    provably the thing that came before the refusal.
    """
    nvenc = StubNvenc()
    with pytest.raises(RuntimeError, match="拒绝交付"):
        _rich_run(tmp_path, remux, bridges={"nvenc": nvenc},
                  verify=lambda path: (0, 25.0, 1920, 1080))
    _assert_intermediate_written_and_gone(nvenc, reserved_in_tmp_path)
    assert remux.commands, "the rewrap did run before the file was read back"


def test_the_cleanup_leaves_the_source_and_the_deliverable_alone(
        stubbed, tmp_path, remux, reserved_in_tmp_path):
    """What the guard removes is only the file this run created under its own name."""
    source = os.path.join(str(tmp_path), "hdr_valid_src.mkv")
    (out, _report), out_path, nvenc, verified = _rich_run(tmp_path, remux)
    assert out == out_path
    assert os.path.exists(out_path), "the deliverable is not the intermediate"
    # The source is input 1 of the copy pass and was read while out_path was written;
    # it is not this run's to remove, and it still holds exactly what ``_run`` planted.
    assert open(source, "rb").read() == b"x" * 4096
    assert verified == [out_path]
    _assert_intermediate_written_and_gone(nvenc, reserved_in_tmp_path)


def test_a_source_alias_is_refused_before_the_encoder_opens_even_with_a_mux(
        tmp_path, remux):
    """The copy pass reads the source while writing out_path, so an alias is a fuse."""
    source = os.path.join(str(tmp_path), "hdr_valid_src.mkv")
    open(source, "wb").write(b"x" * 4096)
    with pytest.raises(ValueError):
        htx.export_gpu_hdr_transfer(
            video=source, live=Fp16Live(), flow_provider=DeviceFlowProvider(),
            out_path=source, dims=dict(HDR_DIMS), frames=6, cq=24, with_audio=True,
            transcode_settings=dict(RICH), job_settings=dict(HDR_JOB),
            negotiation=_negotiate(transcode=dict(RICH), with_audio=True))
    assert remux.commands == []


class FakeCookie:
    """One ``os.add_dll_directory`` answer, reduced to what a release can do to it."""

    def __init__(self, name="ffmpeg"):
        self.name = name
        self.closes = 0

    @property
    def closed(self):
        return self.closes > 0

    def close(self):
        self.closes += 1


def test_a_refused_hdr_job_releases_the_cookies_its_bridge_set_loaded(
        monkeypatch, tmp_path):
    """A refusal loads libraries and never runs them, so somebody has to give the
    DLL-directory handles back. The entry point owns them until the chain takes the
    set, and the chain's ``finally`` is the only other place that may release them."""
    seen = {}
    cookies = [FakeCookie("a"), FakeCookie("b")]

    class Stub:
        def hdr_requested(self, js):
            return True

        def load_hdr_capabilities(self, bridges, width, height):
            return dict(HDR_CAPS)

        def negotiate_hdr(self, *args, **kwargs):
            return {"supported": False, "reasons": ["the encoder carries no HDR10"],
                    "hdr_policy": {}, "fps": (25, 1), "plan": {},
                    "working_scale": 0.5}

        def release_loader_cookies(self, bridges):
            # the real release, through the seam the product calls
            seen["released"] = htx.release_loader_cookies(bridges)
            return seen["released"]

        def export_gpu_hdr_transfer(self, **kwargs):
            raise AssertionError("the chain ran for a job the gate refused")

    class TransferStub:
        def guidance_requires_flow(self, js):
            return True

        def load_bridges(self, requires_flow=False):
            bridges = {"nvdec": StubNvdec(), "nvenc": StubNvenc(),
                       "cookies": cookies,
                       "option_available": lambda codec, name: True}
            seen["bridges"] = bridges
            return bridges

        def probe_source(self, path):
            return dict(HDR_SOURCE)

    monkeypatch.setattr(pipeline, "video_info", lambda path: (6, 25.0, 1920, 1080))
    source = os.path.join(str(tmp_path), "src.mkv")
    out_path = os.path.join(str(tmp_path), "out.mkv")
    open(source, "wb").write(b"x" * 4096)
    with pytest.raises(RuntimeError, match="carries no HDR10"):
        pipeline.export_gpu_transfer_stream(
            video=source, live=Fp16Live(), frames=6, fps=25.0, cq=24,
            with_audio=False, out_path=out_path, transcode_settings=dict(NO_MUX),
            job_settings=dict(HDR_JOB), flow_provider=DeviceFlowProvider(),
            hdr_module=Stub(), module=TransferStub())
    assert seen["released"] == 2
    assert [cookie.closes for cookie in cookies] == [1, 1]
    assert seen["bridges"]["cookies"] == []


def test_a_negotiation_that_raises_releases_the_cookies_before_it_propagates(
        monkeypatch, tmp_path):
    """The release is not a refusal-path detail: anything that keeps the job from the
    hand-over line leaves the handles owned by this function."""
    released = []
    cookies = [FakeCookie("only")]

    class Stub:
        def hdr_requested(self, js):
            return True

        def load_hdr_capabilities(self, bridges, width, height):
            return dict(HDR_CAPS)

        def negotiate_hdr(self, *args, **kwargs):
            raise ValueError("the gate itself faulted")

        def release_loader_cookies(self, bridges):
            released.append(htx.release_loader_cookies(bridges))
            return released[-1]

    class TransferStub:
        def guidance_requires_flow(self, js):
            return True

        def load_bridges(self, requires_flow=False):
            return {"nvdec": None, "nvenc": None, "cookies": cookies,
                    "option_available": lambda codec, name: True}

        # The HDR entry's own probe call, keyword for keyword. A stub that took only the
        # path would raise TypeError inside a try that turns it into source_error, and
        # this test would still pass while proving nothing about the source it hands on.
        def probe_source(self, path, include_chapters=False,
                         include_video_metadata=False):
            assert include_chapters and include_video_metadata
            return dict(HDR_SOURCE)

    monkeypatch.setattr(pipeline, "video_info", lambda path: (6, 25.0, 1920, 1080))
    source = os.path.join(str(tmp_path), "src.mkv")
    open(source, "wb").write(b"x" * 4096)
    with pytest.raises(ValueError, match="the gate itself faulted"):
        pipeline.export_gpu_hdr_transfer_stream(
            video=source, live=Fp16Live(), dims=dict(HDR_DIMS), frames=6, fps=25.0,
            cq=24, with_audio=False, out_path=os.path.join(str(tmp_path), "out.mkv"),
            transcode_settings=dict(NO_MUX), job_settings=dict(HDR_JOB),
            flow_provider=DeviceFlowProvider(), module=Stub(),
            transfer_module=TransferStub())
    assert released == [1]
    assert cookies[0].closes == 1


def test_release_loader_cookies_touches_the_handles_and_nothing_else():
    """The set's libraries are another session's to keep: closing a cookie removes a
    search directory, and this helper must not reach for anything else."""
    decoder, encoder = StubNvdec(), StubNvenc()
    cookies = [FakeCookie("a"), FakeCookie("b")]
    bridges = {"nvdec": decoder, "nvenc": encoder, "cookies": cookies}
    assert htx.release_loader_cookies(bridges) == 2
    assert [cookie.closes for cookie in cookies] == [1, 1]
    # Releasing the same set again - the shape of a chain finally after an entry point
    # that already let go - closes nothing a second time.
    assert htx.release_loader_cookies(bridges) == 0
    assert [cookie.closes for cookie in cookies] == [1, 1]
    assert decoder.symbols == [] and encoder.symbols == []
    assert htx.release_loader_cookies(None) == 0
    assert htx.release_loader_cookies({}) == 0


def test_the_product_branch_routes_an_hdr_job_to_the_hdr_gate(monkeypatch, tmp_path):
    """The one delegation point, exercised without a GPU: the flag selects the gate.

    ``pipeline.export_gpu_transfer_stream`` is what the service calls; a job that
    opted into HDR has to leave it for the HDR negotiation before the SDR gate -
    which refuses a 10-bit PQ source by design - ever sees it. Both seams are stubbed,
    so nothing here loads a library or touches a device.
    """
    seen = {}
    cookies = [FakeCookie("handed")]

    class Stub:
        def hdr_requested(self, js):
            return bool(js.get("experimental_hdr_scrgb"))

        def load_hdr_capabilities(self, bridges, width, height):
            seen["caps_extent"] = (width, height)
            return dict(HDR_CAPS)

        def negotiate_hdr(self, dims, source, js, ts, cq, audio, out_path, live,
                          **kwargs):
            seen["negotiation_args"] = (dims, source, js, live, kwargs)
            return {"supported": True, "reasons": [], "uses_flow": True,
                    "hdr_policy": {"support_ceiling_scrgb": 250.0},
                    "fps": (25, 1), "plan": {"codec": "hevc_nvenc"},
                    "working_scale": 0.5}

        def release_loader_cookies(self, bridges):
            seen.setdefault("released_before_handoff",
                            htx.release_loader_cookies(bridges))
            return seen["released_before_handoff"]

        def export_gpu_hdr_transfer(self, **kwargs):
            seen.update(kwargs)
            # What the chain's own finally does with the set it was handed.
            seen["released_by_chain"] = htx.release_loader_cookies(
                kwargs["bridges"])
            return kwargs["out_path"], {"export_backend_actual": "gpu_transfer"}

    class TransferStub:
        def guidance_requires_flow(self, js):
            return int(js.get("guidance_mode", 0)) == 1

        def load_bridges(self, requires_flow=False):
            seen["requires_flow"] = requires_flow
            return {"nvdec": None, "nvenc": None, "cookies": cookies,
                    "option_available": lambda codec, name: True}

        def probe_source(self, path):
            return dict(HDR_SOURCE)

    monkeypatch.setattr(pipeline, "video_info",
                        lambda path: (6, 25.0, 1920, 1080))
    out_path = os.path.join(str(tmp_path), "out.mkv")
    source = os.path.join(str(tmp_path), "src.mkv")
    open(source, "wb").write(b"x" * 4096)
    finished = pipeline.export_gpu_transfer_stream(
        video=source, live=Fp16Live(), frames=6, fps=25.0, cq=24, with_audio=False,
        out_path=out_path, transcode_settings=dict(NO_MUX), job_settings=dict(HDR_JOB),
        flow_provider=DeviceFlowProvider(), hdr_module=Stub(),
        module=TransferStub())
    assert finished == out_path
    assert seen["dims"]["live_w"] == 960 and seen["dims"]["live_h"] == 544
    assert seen["dims"]["src_w"] == 1920 and seen["dims"]["dst_w"] == 1920
    # the negotiation the gate produced is the one the chain was handed
    assert seen["negotiation"]["uses_flow"] is True
    assert seen["caps_extent"] == (1920, 1080)
    assert seen["requires_flow"] is True
    # Ownership, the other direction: an accepted job hands the cookies over untouched,
    # and the chain is the one that releases them.
    assert "released_before_handoff" not in seen
    assert seen["released_by_chain"] == 1
    assert cookies[0].closes == 1


def test_the_product_branch_refuses_an_hdr_job_the_hdr_gate_rejects_without_reaching_sdr(
        monkeypatch, tmp_path):
    """A refusal is a refusal: the SDR chain is never given the job back.

    Falling through to the 8-bit path would produce a file whose every field reads as
    success while its picture is a PQ source tonemapped by accident.
    """
    reached = []

    class Stub:
        def hdr_requested(self, js):
            return bool(js.get("experimental_hdr_scrgb"))

        def load_hdr_capabilities(self, bridges, width, height):
            return dict(HDR_CAPS)

        def negotiate_hdr(self, *args, **kwargs):
            return {"supported": False, "reasons": ["source is 8-bit"],
                    "hdr_policy": {}, "fps": (25, 1), "plan": {},
                    "working_scale": 0.5}

        def release_loader_cookies(self, bridges):
            return htx.release_loader_cookies(bridges)

        def export_gpu_hdr_transfer(self, **kwargs):
            reached.append(kwargs)
            raise AssertionError("the SDR-side export ran after an HDR refusal")

    class TransferStub:
        def guidance_requires_flow(self, js):
            return True

        def load_bridges(self, requires_flow=False):
            return {"nvdec": None, "nvenc": None, "cookies": [],
                    "option_available": lambda codec, name: True}

        def probe_source(self, path):
            return dict(HDR_SOURCE)

        def negotiate(self, *args, **kwargs):
            reached.append(kwargs)
            raise AssertionError("the SDR gate negotiated an HDR job")

    monkeypatch.setattr(pipeline, "video_info",
                        lambda path: (6, 25.0, 1920, 1080))
    source = os.path.join(str(tmp_path), "src.mkv")
    out_path = os.path.join(str(tmp_path), "out.mkv")
    open(source, "wb").write(b"x" * 4096)
    diagnostics = {}
    with pytest.raises(RuntimeError) as raised:
        pipeline.export_gpu_transfer_stream(
            video=source, live=Fp16Live(), frames=6, fps=25.0, cq=24,
            with_audio=False, out_path=out_path,
            transcode_settings=dict(NO_MUX), job_settings=dict(HDR_JOB),
            flow_provider=DeviceFlowProvider(), hdr_module=Stub(),
            module=TransferStub(), diagnostics_out=diagnostics)
    assert "source is 8-bit" in str(raised.value)
    assert "未回退" in str(raised.value)
    assert reached == []
    assert diagnostics["export_backend_actual"] == "refused"
    assert diagnostics["transfer_mode_requested"] == "gpu_hdr_transfer"
    assert diagnostics["experimental_hdr_scrgb"] is True


def test_the_product_branch_leaves_an_sdr_job_on_the_sdr_gate(monkeypatch, tmp_path):
    calls = []

    class SdrStub:
        INTERMEDIATE_CONTAINER = ".mkv"

        def guidance_requires_flow(self, js):
            return int(js.get("guidance_mode", 0)) == 1

        def load_bridges(self, requires_flow=False):
            return {"nvdec": None, "nvenc": None, "cookies": [],
                    "option_available": lambda codec, name: True}

        def probe_source(self, path):
            return dict(HDR_SOURCE)

        def negotiate(self, *args, **kwargs):
            calls.append(kwargs)
            return {"supported": False, "reasons": ["stub refuses"]}

    monkeypatch.setattr(pipeline, "video_info",
                        lambda path: (6, 25.0, 1920, 1080))
    source = os.path.join(str(tmp_path), "sdr.mkv")
    out_path = os.path.join(str(tmp_path), "out.mkv")
    open(source, "wb").write(b"x" * 4096)
    with pytest.raises(RuntimeError) as raised:
        pipeline.export_gpu_transfer_stream(
            video=source, live=Fp16Live(), frames=6, fps=25.0, cq=24,
            with_audio=False, out_path=out_path, transcode_settings={},
            job_settings={"export_backend": "gpu_transfer", "guidance_mode": 1,
                          "dlss_backend": "gpu_flow_d3d12", "working_scale": 0.5},
            flow_provider=DeviceFlowProvider(), module=SdrStub())
    assert calls and not any(c.get("experimental_hdr_scrgb") for c in calls)
    assert "stub refuses" in str(raised.value)


# ---------------------------------------------------------------------------
# the multi-scale run: who opens what, in which order, and what each of them owns
# ---------------------------------------------------------------------------

# The two shapes a second layer can fuse into. The baseline is the one that needs no
# solver at all, so the pair of them is what says a layer was run rather than a mode.
FUSED = _multi(fusion_mode="data_consistency_experimental")
TRIPLE_FUSED = _multi("triple", (0.5, 0.25, 1.0),
                      fusion_mode="data_consistency_experimental")
# What the chain's own decode leaves in the master plane, and what an observation
# leaves in the residual plane: the bridge's numbered device addresses.
MASTER_PTR = 0xC000 + hdr.PLANE_MASTER_FULL * 16
RESIDUAL_PTR = 0xC000 + hdr.PLANE_WORKING_RESIDUAL * 16
# A dual frame, as the colour bridges recorded being asked for it. The extra layer's
# three calls sit between the primary's proxy and the first observation because that
# is where the pristine master is still the master every layer must share.
DUAL_FRAME = ["decode", "downscale", "proxy", "import_master", "downscale", "proxy"]
DUAL_BASELINE = DUAL_FRAME + ["observe", "observe", "reconstruct", "encode"]
DUAL_FUSED_FRAME = DUAL_FRAME + ["observe", "observe", "set_composite", "encode"]


def test_a_dual_frame_runs_the_baseline_over_both_layers_and_composites_one(
        multiscale, tmp_path):
    """Two layers, one decode, one decision: the order the borrows require.

    The frame is checked as one line rather than stage by stage because the claim that
    matters is the sequence: a second decode would measure the source twice, an
    observation after the reconstruct would be an observation of a frame that had
    already been committed, and a composite that summed two layers is a different
    algorithm than the one the job named.
    """
    (_out, report), _p, nvdec, nvenc, _pr = _run(tmp_path, job=_multi(), frames=3)
    assert _kinds() == DUAL_BASELINE * 3 + ["close"] * 2      # both bridges, in order
    assert _kinds("480x272") == ["import_master", "downscale", "proxy",
                                 "observe"] * 3 + ["close"]
    assert _kinds("960x544") == ["decode", "downscale", "proxy", "observe",
                                 "reconstruct", "encode"] * 3 + ["close"]
    # The source is decoded once per frame, never once per layer.
    assert nvdec.symbols.count("nvdec_bridge_next") == 3
    assert _kinds().count("decode") == 3
    assert nvenc.symbols == (["nvenc_hdr_open"] + ["nvenc_encode_p010"] * 3
                             + ["nvenc_finish", "nvenc_close"])
    assert report["hdr_frames_processed_measured"] == 3
    assert report["hdr_layers"] == 2
    assert report["hdr_fusion_modes_actual"] == ["catmull_baseline_layer_0"]


def test_the_fused_dual_frame_hands_every_observation_to_the_solver_before_it_imports(
        multiscale, tmp_path):
    (_out, report), _p, _n, nvenc, _pr = _run(tmp_path, job=FUSED, frames=3)
    assert _kinds() == DUAL_FUSED_FRAME * 3 + ["close"] * 2
    assert _kinds().count("reconstruct") == 0    # the baseline is not a fallback here
    assert nvenc.symbols == (["nvenc_hdr_open"] + ["nvenc_encode_p010"] * 3
                             + ["nvenc_finish", "nvenc_close"])
    assert len(Solver.calls) == 3
    assert report["hdr_fusion_modes_actual"] == ["typed_gpu_data_consistency_solve"]
    assert report["hdr_frames_processed_measured"] == 3


def test_an_extra_layer_is_fed_the_pristine_master_it_cannot_produce_for_itself(
        multiscale, tmp_path):
    """The imported plane is the decode's full-resolution half-float master, in the
    bridge's own words: extent, pitch, pointer. Not the working master, not the proxy,
    and nothing resampled on the way in - which is the whole reason a second layer may
    be trusted to measure a residual against its own scale.
    """
    _run(tmp_path, job=_multi(), frames=2)
    imports = _payloads("import_master", tag="480x272")
    assert len(imports) == 2
    assert imports == [{"device_ptr": MASTER_PTR, "width": 1920, "height": 1080,
                        "pitch_bytes": 1920 * 8}] * 2
    # The primary's own master is the same plane the layers were given, so there is one
    # measurement of the source and every layer is derived from it.
    assert _payloads("decode")[0]["support_ceiling_scrgb"] == 250.0


def test_each_layer_is_opened_on_its_own_extent_one_policy_and_the_decoders_context(
        multiscale, tmp_path):
    _run(tmp_path, job=_multi("dual", (0.5, 0.25),
                              hdr_paper_white_scrgb=4.0), frames=1)
    opens = [call for call in FakeBridge.calls if isinstance(call, dict)]
    assert [bridge["working"] for bridge in opens] == [(960, 544), (480, 272)]
    assert [bridge["full"] for bridge in opens] == [(1920, 1080)] * 2
    # The job's fixed reference white, handed to both and never adapted per layer.
    assert [bridge["paper_white_scrgb"] for bridge in opens] == [4.0, 4.0]
    assert [bridge["cuda_context"] for bridge in opens] == [0x9000, 0x9000]
    assert [bridge["d3d12_device"] for bridge in opens] == [0x3000, 0x3000]


def test_every_extra_layer_is_its_own_session_on_the_anchor_and_its_own_history(
        multiscale, tmp_path):
    """A layer is a session, not the same Feature 18 asked twice.

    The anchor is what the layer shares - its core, and the settings the job was asked
    with - and the extent, colour and motion history are each the layer's own, because
    a second layer drawn from one session would be one Feature wearing a layer's name.
    """
    live, flow = Fp16Live(), DeviceFlowProvider()
    (_out, report), _p, _n, _e, _pr = _run(
        tmp_path, job=_multi("triple", (0.5, 0.25, 1.0)), frames=2, live=live,
        flow=flow)
    assert [(session._w, session._h) for session in LayerSession.created] == \
        [(480, 272), (1920, 1080)]
    assert [session.color_format_given for session in LayerSession.created] == \
        ["rgba16f", "rgba16f"]
    # Each layer receives a resolved snapshot, so later per-layer changes do not
    # mutate the anchor's settings object. Sharing the device is independent.
    assert all(session.settings_given is not live.settings for session in LayerSession.created)
    assert all(session.settings_given.get('intensity') == live.settings.get('intensity', 1.)
               for session in LayerSession.created)
    assert [session.shared_with is live for session in LayerSession.created] == \
        [True, True]
    # The motion each layer was guided by came from an engine of its own, at its own
    # extent - not from the root provider, and not from another layer's history.
    assert len(flow.layers) == 2
    assert flow.extents == [(960, 544)] * 2
    assert [layer.extents for layer in flow.layers] == \
        [[(480, 272)] * 2, [(1920, 1080)] * 2]
    assert report["hdr_dlss_flow_calls_measured"] == 6      # three layers, two frames
    assert report["hdr_layers_measured"] == 3
    assert report["hdr_layer_sessions_distinct"] is True
    assert len(set(report["hdr_layer_session_ids"])) == 3


def test_an_observation_is_the_layers_own_signed_plane_and_the_solver_gets_one_per_layer(
        multiscale, tmp_path):
    live = Fp16Live()
    (_out, _r), _p, _n, _e, _pr = _run(tmp_path, job=FUSED, frames=2, live=live)
    for call in Solver.calls:
        assert call["cuda_context"] == 0x9000
        assert (call["master"]["width"], call["master"]["height"]) == (1920, 1080)
        assert call["master"]["device_ptr"] == MASTER_PTR
        assert [(plane["width"], plane["height"])
                for plane in call["observations"]] == [(960, 544), (480, 272)]
        assert [plane["device_ptr"] for plane in call["observations"]] == \
            [RESIDUAL_PTR, RESIDUAL_PTR]
    # Each layer read its own session's texture: the anchor's and the layer's differ by
    # session, and no layer ever observed the other's Feature 18 output.
    textures = [0x4000 + 0x1000 * session.session_id
                for session in LayerSession.created]
    observes = _payloads("observe")
    assert [entry["texture"] for entry in observes] == \
        [0x4000 + 0x1000 * live.session_id, textures[0]] * 2
    assert [entry["fence_value"] for entry in observes] == [1, 1, 2, 2]


def test_the_fused_composite_is_imported_over_a_real_producer_fence_then_encoded(
        multiscale, tmp_path):
    """The wait is the point: an import that did not name the store it retired would
    encode a half-written frame, so the fence and its non-zero value are what make the
    solve a completed picture rather than a queued one.
    """
    (_out, report), _p, _n, nvenc, _pr = _run(tmp_path, job=FUSED, frames=3)
    composites = _payloads("set_composite", tag="960x544")
    assert [(entry["texture"], entry["fence"], entry["fence_value"],
             entry["dxgi_format"]) for entry in composites] == \
        [(0x6000 + index, 0x7000, 100 + index, hdr.DXGI_R16G16B16A16_FLOAT)
         for index in range(3)]
    assert all(entry["fence_value"] for entry in composites)
    kinds = _kinds()
    for index in range(3):
        frame = kinds[index * len(DUAL_FUSED_FRAME):(index + 1) * len(DUAL_FUSED_FRAME)]
        assert frame.index("set_composite") > max(i for i, kind in enumerate(frame)
                                                 if kind == "observe")
        assert frame.index("set_composite") < frame.index("encode")
    assert report["hdr_process_count_measured"] == 15     # the primary's five per frame


def test_the_solver_is_opened_on_the_jobs_layer_vector_and_the_plans_those_extents_got(
        multiscale, tmp_path):
    live = Fp16Live()
    _run(tmp_path, job=_multi(fusion_mode="data_consistency_experimental",
                              layer_weights=[2.0, 0.5], fusion_beta=0.05,
                              fusion_relative_tolerance=0.001,
                              fusion_max_iterations=64, residual_strength=0.75),
         frames=1, live=live)
    assert len(Solver.opened) == 1
    opened = Solver.opened[0]
    assert opened["full"] == (1920, 1080)
    assert opened["d3d12_device"] == live.d3d12_device()
    assert [layer["dims"] for layer in opened["layers"]] == [(960, 544), (480, 272)]
    assert [layer["weight"] for layer in opened["layers"]] == [2.0, 0.5]
    assert {layer["plan"] for layer in opened["layers"]} == {"TapPlans"}
    assert opened["beta"] == pytest.approx(float(C.c_float(0.05).value))
    assert opened["max_iterations"] == 64
    assert opened["residual_strength"] == 0.75
    # The tolerance the solve is judged by is the one the bridge was actually sent.
    assert opened["relative_tolerance"] == pytest.approx(
        float(C.c_float(0.001).value))


def test_a_triple_vector_drives_three_sessions_three_bridges_and_three_residuals(
        multiscale, tmp_path):
    (_out, report), _p, _n, _e, _pr = _run(tmp_path, job=TRIPLE_FUSED, frames=2)
    assert _kinds("960x544") == ["decode", "downscale", "proxy", "observe",
                                 "set_composite", "encode"] * 2 + ["close"]
    assert len(LayerSession.created) == 2
    assert len(Solver.calls) == 2
    assert [(plane["width"], plane["height"])
            for plane in Solver.calls[0]["observations"]] == \
        [(960, 544), (480, 272), (1920, 1080)]
    assert report["hdr_layer_scales"] == [0.5, 0.25, 1.0]
    assert report["hdr_layer_working_dims"] == [[960, 544], [480, 272], [1920, 1080]]
    assert report["hdr_layers_measured"] == 3
    assert len(report["hdr_layer_transfers_measured"]) == 2   # the extras, by name


def test_a_zero_residual_strength_reaches_the_baseline_reconstruct_as_zero(
        multiscale, tmp_path):
    _run(tmp_path, job=_multi(residual_strength=0.0), frames=2)
    assert [entry["residual_strength"] for entry in _payloads("reconstruct")] == \
        [0.0, 0.0]
    assert {entry["dxgi_format"] for entry in _payloads("reconstruct")} == \
        {hdr.DXGI_R16G16B16A16_FLOAT}


def test_a_zero_residual_strength_reaches_the_solver_as_zero_and_still_solves(
        multiscale, tmp_path):
    (_out, report), _p, _n, _e, _pr = _run(
        tmp_path, job=_multi(fusion_mode="data_consistency_experimental",
                             residual_strength=0.0), frames=2)
    assert Solver.opened[0]["residual_strength"] == 0.0
    assert len(Solver.calls) == 2
    assert report["hdr_solver_frames_measured"] == 2


def test_the_layer_traffic_is_attributed_to_the_bridge_that_moved_it_and_the_scalar(
        multiscale, tmp_path):
    """Per bridge, per session, per engine - and the counter readback apart.

    An import is a full-resolution FP16 write that dispatches no shader, an observation
    is a working-extent copy in and a residual out, and the composite never touched the
    layer's own bridge. Those three facts are what makes the scalar boundary statable as
    a range at all, so they are asserted as numbers the bridges gave rather than as
    numbers this file expected.
    """
    full, working = 1920 * 1080 * 8, 480 * 272 * 8
    (_out, report), _p, _n, _e, _pr = _run(
        tmp_path, job=FUSED, frames=3, flow=DeviceFlowProvider(counters=(7, 1234)))
    layer = report["hdr_layer_transfers_measured"][0]
    assert layer["working_dims"] == [480, 272]
    assert layer["master_provenance_measured"] == 2      # last write was an import
    assert layer["master_import_count_measured"] == 3
    assert layer["master_import_device_bytes_measured"] == 3 * full
    assert layer["observe_count_measured"] == 3
    assert layer["observe_copy_bytes_measured"] == 3 * working
    assert layer["observe_residual_bytes_measured"] == 3 * working
    # The layer never composited and never encoded: that is the primary bridge's frame.
    assert layer["composite_import_count_measured"] == 0
    assert layer["composited_device_write_bytes_measured"] == 0
    assert layer["encode_device_write_bytes_measured"] == 0
    assert layer["pass_dispatches_measured"] == 9        # downscale, proxy, observe
    assert layer["hdr_diagnostic_image_readback_bytes_measured"] == 9 * 96
    assert layer["nvof_device_input_counters_measured"] == [7, 1234]
    assert layer["dlss_measured"] is not None

    primary = report["hdr_scalar_report_readback"]
    assert primary["pass_dispatches_measured"] == {"primary": 15, "layer_1": 9}
    assert primary["pass_dispatches_total_measured"] == 24
    assert primary["range_bytes_from_completed_dispatch_count"] == 24 * 256
    assert primary["fusion_scalar_readback_bytes_measured"] == 3 * 2 * 256 * 6
    # Never folded into a measurement, and never answered with the image counter.
    assert report["hdr_scalar_report_readback_bytes_measured"] is None
    assert report["hdr_diagnostic_image_readback_bytes_measured"] == 15 * 96
    assert report["hdr_composited_device_write_bytes_measured"] == 3 * full
    assert report["hdr_fusion_counters_measured"]["process_count"] == 3
    assert report["hdr_fusion_counters_measured"]["scalar_readback_bytes"] == 9216
    assert report["hdr_fusion_counters_measured"]["source_device_copy_bytes"] == 3 * full
    assert report["hdr_fusion_counters_measured"]["observation_device_copy_bytes"] == \
        3 * (960 * 544 * 8 + working)


def test_extra_bridge_diagnostic_image_readback_preserves_nonzero_and_none_per_layer(
        multiscale, tmp_path):
    FakeBridge.report_readback_by_tag = {
        "480x272": 4321,
        "1920x1080": None,
    }
    (_out, report), _p, _n, _e, _pr = _run(
        tmp_path, job=TRIPLE_FUSED, frames=1)
    layers = report["hdr_layer_transfers_measured"]
    assert [layer["working_dims"] for layer in layers] == [[480, 272], [1920, 1080]]
    assert [layer["hdr_diagnostic_image_readback_bytes_measured"] for layer in layers] == [
        4321, None]


def test_a_dual_job_that_carries_the_sources_streams_still_remuxes_the_fused_encode(
        multiscale, tmp_path, remux):
    """The tail is the unchanged one: a solve changes the composite, never the mux.

    The encoder's file is an intermediate and the deliverable is the map-and-copy pass's
    output, read back after it - the same rule a single-layer job runs on.
    """
    (out, report), out_path, nvenc, verified = _rich_run(
        tmp_path, remux, job=FUSED, frames=2)
    encoded = nvenc.paths[0]
    assert out == out_path
    assert encoded != out_path
    assert encoded.endswith(gtx.INTERMEDIATE_CONTAINER)
    assert report["hdr_copy_pass"] is True
    assert report["hdr_deliverable_written_by"] == "ffmpeg_map_and_copy_pass"
    assert verified == [out_path]
    assert _kinds().count("set_composite") == 2
    assert len(Solver.calls) == 2


def test_a_frame_that_restarts_the_picture_resets_every_active_layer_at_once(
        multiscale, tmp_path):
    """One discontinuity, every session told about it, on the same frame.

    A layer whose history carried a picture the others had forgotten would be a
    reconstruction of two moments, so the reset flags are read per layer and the
    sessions are asked what they were actually given.
    """
    live = Fp16Live()
    layer_flow = DeviceFlowProvider()
    (_out, report), _p, _n, _e, _pr = _run(
        tmp_path, job=_multi(), frames=6, live=live, flow=layer_flow,
        bridges={"nvdec": GapNvdec(frames=6)})
    flags = [True, False, True, False, False, False]
    assert report["hdr_layer_reset_flags"] == [flags, flags]
    assert report["hdr_feature_resets_measured"] == 2
    assert live.resets == flags
    assert [session.resets for session in LayerSession.created] == [flags]
    assert layer_flow.discontinuities == flags
    assert layer_flow.layers[0].discontinuities == flags


class GapNvdec(StubNvdec):
    """Six frames whose third PTS is one short, which is a gap the chain must answer."""

    def nvdec_bridge_next(self, handle, frame):
        self._log("nvdec_bridge_next")
        if self.index >= self.total:
            return 0
        current = _device_frame(self.index)
        current.pts = self.index + (1 if self.index >= 2 else 0)
        current.software_format = self.software_format
        _write_into(frame, current)
        self.index += 1
        return 1


class MovingNvdec(StubNvdec):
    """A decoder that yields a different frame size in the middle of the job."""

    def nvdec_bridge_next(self, handle, frame):
        self._log("nvdec_bridge_next")
        if self.index >= self.total:
            return 0
        current = _device_frame(self.index)
        current.software_format = self.software_format
        if self.index == 2:
            current.width, current.height = 1920, 960
        _write_into(frame, current)
        self.index += 1
        return 1


def test_a_source_that_changes_extent_rebuilds_the_layers_instead_of_reusing_their_history(
        multiscale, tmp_path):
    with pytest.raises(htx.HdrExportError) as raised:
        _run(tmp_path, job=_multi(), frames=4,
             bridges={"nvdec": MovingNvdec(frames=4)})
    assert "rebuild" in str(raised.value)
    # Refused before the frame was decoded, so no layer ever drew a mismatched master.
    assert _kinds().count("decode") == 2
    assert _kinds().count("encode") == 2
    assert [tag for kind, tag in _bridges() if kind == "close"] == \
        ["480x272", "960x544"]


def test_an_unguided_dual_leaves_the_flow_provider_alone_and_uses_each_sessions_motion(
        multiscale, tmp_path):
    """guidance_mode=0 is resident zero motion *per session*, which needs no engine.

    The provider is the one object this test makes loud if it is touched: a chain that
    asked for motion on a job that refused it is caught by the assertion below, and a
    layer that borrowed the anchors zero motion would show as one session's count.
    """
    (_out, report), _p, _n, _e, _pr = _run(
        tmp_path, job=_multi(guidance_mode=0), frames=2, flow=BannedFlowProvider())
    assert report["hdr_layer_zero_motion_calls_measured"] == [2, 2]
    assert report["hdr_layer_flow_calls_measured"] == [0, 0]
    assert report["hdr_dlss_zero_motion_calls_measured"] == 4
    assert report["hdr_dlss_flow_calls_measured"] == 0
    assert report["hdr_uses_flow"] is False
    # No layer was built on the provider, so none is given back either.
    assert "flow layer" not in TEARDOWN


def test_a_finished_dual_job_gives_every_layer_back_before_the_parent_that_owns_it(
        multiscale, tmp_path):
    """Reverse ownership, once each, and never the borrowed root.

    The solver's texture is what the primary bridge imported, a layer's bridge imports
    that layer's Feature 18 output, and that session was fed by that layer's own motion
    layer - so that is the order they go, and the anchor's session, the primary bridge,
    the context, the decoder and the caller's provider stay the parents to release.
    """
    live, flow = Fp16Live(), DeviceFlowProvider()
    (_out, report), _p, nvdec, nvenc, _pr = _run(tmp_path, job=FUSED, frames=2,
                                                 live=live, flow=flow)
    assert TEARDOWN == ["solver", "colour bridge", "feature18[480x272]",
                        "flow layer", "encoder", "colour bridge", "feature18",
                        "context", "decoder"]
    assert [tag for kind, tag in _bridges() if kind == "close"] == \
        ["480x272", "960x544"]          # each bridge closed exactly once
    assert Solver.closed == 1
    assert flow.closed_layers == 0                    # the caller's provider stays alive
    assert flow.layers[0].closed_layers == 1          # only the layer's own engine goes
    assert live.closed is True and LayerSession.created[0].closed is True
    assert nvdec.symbols[-1] == "nvdec_bridge_close"
    assert nvenc.symbols[-1] == "nvenc_close"
    assert report["hdr_frames_processed_measured"] == 2


def test_cleanup_attempts_every_owned_resource_and_reports_close_failures(
        multiscale, monkeypatch, tmp_path):
    FakeBridge.fail_close_tags = {"480x272"}
    LayerSession.fail_close_sizes = {(480, 272)}
    original_flow_close = DeviceFlowProvider.close_independent_layer

    def fail_after_flow_close(self):
        original_flow_close(self)
        raise RuntimeError("independent flow close failed")

    monkeypatch.setattr(DeviceFlowProvider, "close_independent_layer", fail_after_flow_close)
    with pytest.raises(htx.HdrExportError, match="cleanup failed") as raised:
        _run(tmp_path, job=FUSED, frames=1)

    assert "colour bridge close failed" in str(raised.value)
    assert "layer session close failed" in str(raised.value)
    assert "independent flow close failed" in str(raised.value)
    # Each extra owner and every parent resource was attempted despite earlier errors.
    assert TEARDOWN == ["solver", "colour bridge", "feature18[480x272]",
                        "flow layer", "encoder", "colour bridge", "feature18",
                        "context", "decoder"]


def test_partial_layer_open_preserves_open_error_when_rollback_close_fails(
        multiscale, tmp_path):
    LayerSession.refuse_at = 1
    LayerSession.fail_close_sizes = {(480, 272)}
    with pytest.raises(dge.DLSSInferenceError, match="open_layer_ex refused layer 1"):
        _run(tmp_path, job=TRIPLE_FUSED, frames=1)
    assert TEARDOWN[:3] == ["flow layer", "feature18[480x272]", "flow layer"]


def test_multiscale_plan_construction_failure_releases_all_unadopted_extras(
        multiscale, tmp_path, monkeypatch):
    live = Fp16Live()
    flow = DeviceFlowProvider()
    LayerSession.fail_close_sizes = {(480, 272)}
    plans = []

    def fail_second_plan(*args):
        plans.append(args)
        if len(plans) == 2:
            raise ValueError("injected layer-plan construction failure")
        return "TapPlans"

    monkeypatch.setattr(htx.gtx, "TapPlans", fail_second_plan)
    with pytest.raises(ValueError, match="injected layer-plan construction failure") as raised:
        _run(tmp_path, job=TRIPLE_FUSED, frames=1, live=live, flow=flow)

    assert len(LayerSession.created) == 2
    assert all(session.closed for session in LayerSession.created)
    assert [provider.closed_layers for provider in flow.layers] == [1, 1]
    assert flow.closed_layers == 0
    assert TEARDOWN == ["feature18[1920x1080]", "flow layer",
                        "feature18[480x272]", "flow layer"]
    # Cleanup notes use Python 3.11+ Exception.add_note; older runtimes still
    # preserve the primary error and execute every close, as asserted above.
    if hasattr(raised.value, "add_note"):
        assert any("layer session close failed at 480x272" in note
                   for note in getattr(raised.value, "__notes__", []))
    else:
        assert str(raised.value) == "injected layer-plan construction failure"


def test_multiscale_run_refuses_missing_or_repeated_session_ids(multiscale, tmp_path,
                                                                monkeypatch):
    live = Fp16Live()
    original_init = LayerSession.__init__

    def repeated_id(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.session_id = live.session_id

    monkeypatch.setattr(LayerSession, "__init__", repeated_id)
    with pytest.raises(htx.HdrExportError, match="distinct, positive reportable"):
        _run(tmp_path, job=_multi(), frames=1, live=live)
    assert len(LayerSession.created) == 1 and LayerSession.created[0].closed
    assert FakeBridge.constructed == 1  # primary bridge only; no unverified layer bridge


def test_multiscale_run_refuses_zero_session_id(multiscale, tmp_path, monkeypatch):
    original_init = LayerSession.__init__

    def zero_id(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.session_id = 0

    monkeypatch.setattr(LayerSession, "__init__", zero_id)
    with pytest.raises(htx.HdrExportError, match="distinct, positive reportable"):
        _run(tmp_path, job=_multi(), frames=1)
    assert LayerSession.created[0].closed


@pytest.mark.parametrize("knob,job,expected", [
    ("bridge", FUSED, ["feature18[480x272]", "flow layer", "encoder",
                       "colour bridge", "feature18", "context", "decoder"]),
    ("solver", FUSED, ["colour bridge", "feature18[480x272]", "flow layer",
                       "encoder", "colour bridge", "feature18", "context", "decoder"]),
    ("session", TRIPLE_FUSED, ["flow layer", "feature18[480x272]", "flow layer"]),
])
def test_a_layer_that_cannot_be_built_hands_back_exactly_what_was_built(multiscale,
                                                                       tmp_path,
                                                                       knob, job,
                                                                       expected):
    """Three doors a layer can fail at, and the same rule behind all three.

    The bridge that never opened has nothing to close, a solver whose open failed is not
    a handle to release, and a layer vector built before the first frame unwinds in the
    order it was built - which is why the decoder is untouched in the last case and the
    encoder is closed in the other two.
    """
    if knob == "bridge":
        FakeBridge.refuse_after = 1
    elif knob == "solver":
        Solver.refuse_open = True
    else:
        LayerSession.refuse_at = 1
    raises = {"bridge": hdr.HdrBridgeError, "solver": hfe.HdrFusionError,
              "session": dge.DLSSInferenceError}[knob]
    nvdec, nvenc = StubNvdec(frames=2), StubNvenc()
    with pytest.raises(raises):
        _run(tmp_path, job=job, frames=2, bridges={"nvdec": nvdec, "nvenc": nvenc})
    assert TEARDOWN == expected
    if knob == "session":
        assert nvdec.symbols == [] and nvenc.symbols == []
        assert FakeBridge.calls == []
    else:
        assert nvenc.symbols[-1] == "nvenc_close"
    # No solver was ever adopted, so none was closed: an open that raised hands back no
    # handle, and a chain that closed one would be releasing what it never opened.
    assert Solver.opened == [] and Solver.closed == 0


def test_a_cancelled_dual_job_stops_between_frames_and_rolls_the_layers_back(
        multiscale, tmp_path):
    flags = [False, False, True]
    with pytest.raises(pipeline.JobCancelled):
        _run(tmp_path, job=FUSED, frames=4,
             cancel=lambda: flags.pop(0) if flags else True)
    assert len(Solver.calls) == 2                      # stopped between frames
    assert _kinds().count("encode") == 2
    assert TEARDOWN == ["solver", "colour bridge", "feature18[480x272]",
                        "flow layer", "encoder", "colour bridge", "feature18",
                        "context", "decoder"]


def test_a_solve_that_refuses_ends_the_job_and_is_not_answered_by_another_algorithm(
        multiscale, tmp_path):
    """No fallback, no retry: the frame that could not be solved is not exported.

    A chain that drew the Catmull baseline when the typed solve refused would report a
    data-consistency job whose picture is a single-layer reconstruction, which is the
    one substitution this brief forbids.
    """
    Solver.solve_fails_at = 1
    with pytest.raises(hfe.HdrFusionError) as raised:
        _run(tmp_path, job=FUSED, frames=3)
    assert "could not be read" in str(raised.value)
    assert len(Solver.calls) == 2          # attempted on the second frame and refused
    assert _kinds().count("set_composite") == 1
    assert _kinds().count("reconstruct") == 0
    assert _kinds().count("encode") == 1
    # The last bridge call of the job is the primary's close, after the layer's: the
    # rollback unwinds in reverse ownership and reorders nothing.
    assert FakeBridge.calls[-1] == ("close", "960x544")
    assert Solver.closed == 1


def test_a_build_that_loses_the_group_between_the_gate_and_the_run_refuses_before_a_layer_bridge(
        multiscale, tmp_path):
    """The run asks the loaded library again, in its own words.

    A negotiation is made of the answers the entry point got; a frame is driven on the
    handles that hold them. If the second no longer says what the first did, importing a
    master on the strength of a stale answer is the failure this guards.

    The layer's Feature 18 session is built with the rest of the plan, before a frame,
    and so it is alive at the refusal; what the gate has to keep unbuilt is the colour
    bridge that would have owned the imports, which is why the rollback hands the
    session and its motion layer back and never reaches a bridge close for 480x272.
    """
    with pytest.raises(htx.HdrExportError) as raised:
        _run(tmp_path, negotiation=_negotiate(job=_multi()), frames=2,
             caps=_caps(without=("multiscale",)))
    assert "no HDR multiscale capability read" in str(raised.value)
    assert FakeBridge.constructed == 1        # the primary, and no layer built on it
    assert [tag for kind, tag in _bridges() if kind == "close"] == ["960x544"]
    assert len(LayerSession.created) == 1 and LayerSession.created[0].closed is True
    assert Solver.opened == []
    assert TEARDOWN == ["feature18[480x272]", "flow layer", "encoder",
                        "colour bridge", "feature18", "context", "decoder"]


def test_a_single_layer_job_builds_no_session_no_extra_bridge_and_no_solver(
        multiscale, tmp_path):
    """The original path did not become conditional on the new seams being present.

    With every multi seam patched in - a host that records layer opens, a solver that
    records opens and a provider that records handed-out histories - a single job still
    runs the five calls it always ran, on one bridge, and touches none of them.
    """
    flow = DeviceFlowProvider()
    (_out, report), _p, _n, _e, _pr = _run(tmp_path, frames=6, flow=flow)
    assert LayerSession.created == [] and Solver.opened == []
    assert flow.layers == []
    assert FakeBridge.constructed == 1
    assert _kinds() == ["decode", "downscale", "proxy", "reconstruct",
                        "encode"] * 6 + ["close"]
    assert report["hdr_layers"] == 1
    assert "hdr_layers_measured" not in report
    assert "hdr_layer_transfers_measured" not in report
    assert "hdr_fusion_modes_actual" not in report
    assert TEARDOWN == ["encoder", "colour bridge", "feature18", "context", "decoder"]


def test_the_multi_frame_record_is_the_single_record_with_the_vector_audited_on_top(
        multiscale, tmp_path):
    """Compatibility the old readers keep, and the new facts nobody can read as silence.

    Every key a single-layer report carried is still there and still means the same
    thing; the layer vector sits beside it with its own session ids, provenance and
    solver evidence. An imported master is reported as imported and carries no source
    report, because only the bridge that ran the decode pass measured the source.
    """
    diagnostics = {}
    (_out, report), _p, _n, _e, _pr = _run(tmp_path, job=FUSED, frames=2,
                                           diagnostics=diagnostics)
    frames = diagnostics["hdr_frame_reports"]
    assert {"frame_idx", "pts", "time_base", "colour_fields_sent", "reset",
            "motion_source", "source_report", "proxy_report", "encode_report",
            "planes", "feature18_output"} <= set(frames[0])
    assert frames[0]["reconstruction_mode"] == "dual"
    assert frames[0]["fusion_mode_actual"] == "typed_gpu_data_consistency_solve"
    assert "reconstruct_report" not in frames[0]
    assert [layer["layer"] for layer in frames[0]["layers"]] == [0, 1]
    assert [layer["master_provenance"] for layer in frames[0]["layers"]] == \
        ["decoded_pristine_master", "imported_pristine_master"]
    assert "source_report" not in frames[0]["layers"][1]
    assert [layer["working_dims"] for layer in frames[0]["layers"]] == \
        [[960, 544], [480, 272]]
    assert [layer["paper_white_scrgb"] for layer in frames[0]["layers"]] == [8.0, 8.0]
    assert len({layer["session_id"] for layer in frames[0]["layers"]}) == 2
    for layer in frames[0]["layers"]:
        assert layer["planes"]["observation"]["format"] == hdr.FORMAT_RGBA16F_CUDA
        assert layer["planes"]["observation"]["bytes_per_pixel"] == 8
        assert [layer["planes"]["observation"]["width"],
                layer["planes"]["observation"]["height"]] == layer["working_dims"]
        assert layer["feature18_output"]["dxgi_format"] == \
            hdr.DXGI_R16G16B16A16_FLOAT
    solver = frames[0]["solver"]
    assert set(solver["report"]) == set(hfe.REPORT_FIELDS)
    assert solver["converged"] is True and solver["iterations"] == 6
    assert solver["output"]["fence_value_present"] is True
    assert solver["output"]["state"] == hfe.D3D12_COPY_SOURCE
    assert solver["relative_tolerance"] == pytest.approx(
        float(C.c_float(0.001).value))
    # The old per-frame aggregation still reads the same top-level reports.
    assert report["total_proxy_negative_components"] == 2 * 221184
    assert report["hdr_solver_converged_all_frames"] is True
    assert report["hdr_solver_frames_measured"] == 2


def test_a_counted_unrepresentable_fallback_is_still_a_frame_and_stays_counted(
        multiscale, tmp_path):
    """The half grid cannot hold every fused value; the answer is a number, not a gap.

    A sample that fell back to the pristine master is the contracts own honest case, so
    the frame is delivered and the count is reported - never dropped because a non-zero
    here could be read as a failed solve.
    """
    Solver.unrepresentable = 5
    diagnostics = {}
    (_out, report), _p, _n, _e, _pr = _run(tmp_path, job=FUSED, frames=3,
                                           diagnostics=diagnostics)
    assert report["hdr_frames_processed_measured"] == 3
    assert report["hdr_solver_unrepresentable_components_total_measured"] == 15
    assert report["hdr_solver_converged_all_frames"] is True
    assert [frame["solver"]["unrepresentable_components"]
            for frame in diagnostics["hdr_frame_reports"]] == [5, 5, 5]
    # A counted fallback is still a delivered frame: every solve became a composite and
    # every composite an encode, so nothing was quietly dropped from the picture.
    assert _kinds().count("set_composite") == 3
    assert _kinds().count("encode") == 3


def test_the_multiscale_libraries_are_loaded_only_for_a_job_that_named_layers(
        monkeypatch, tmp_path):
    """A second layer is a second set of libraries, and only where it was asked for.

    ``load_bridges`` is the seam that decides which entries get bound, so the claim that
    a single-layer export still loads exactly what it always loaded is read here, off the
    arguments the route passed, without a device.
    """
    loads = []

    class Stub:
        def load_hdr_capabilities(self, bridges, width, height):
            return dict(HDR_CAPS)

        def negotiate_hdr(self, dims, source, js, ts, cq, audio, out_path, live,
                          **kwargs):
            return {"supported": False, "reasons": ["stub gate refuses"],
                    "hdr_policy": {}, "fps": (25, 1), "plan": {},
                    "working_scale": 0.5}

        def release_loader_cookies(self, bridges):
            return htx.release_loader_cookies(bridges)

    class TransferStub:
        def guidance_requires_flow(self, js):
            return int(js.get("guidance_mode", 0)) == 1

        def load_bridges(self, requires_flow=False, requires_multiscale=False):
            loads.append((requires_flow, requires_multiscale))
            return {"nvdec": None, "nvenc": None, "cookies": [],
                    "option_available": lambda codec, name: True}

        def probe_source(self, path, **kwargs):
            assert kwargs["include_chapters"] and kwargs["include_video_metadata"]
            return dict(HDR_SOURCE)

    monkeypatch.setattr(pipeline, "video_info", lambda path: (6, 25.0, 1920, 1080))
    source = os.path.join(str(tmp_path), "src.mkv")
    open(source, "wb").write(b"x" * 4096)
    for mode, scales in (("single", None), ("dual", [0.5, 0.25])):
        job = dict(HDR_JOB, reconstruction_mode=mode)
        if scales:
            job["layer_scales"] = scales
        with pytest.raises(RuntimeError) as raised:
            pipeline.export_gpu_hdr_transfer_stream(
                video=source, live=Fp16Live(), dims=dict(HDR_DIMS), frames=6,
                fps=25.0, cq=24, with_audio=False,
                out_path=os.path.join(str(tmp_path), f"out_{mode}.mkv"),
                transcode_settings=dict(NO_MUX), job_settings=job,
                flow_provider=DeviceFlowProvider(), module=Stub(),
                transfer_module=TransferStub())
        assert "stub gate refuses" in str(raised.value)
    assert loads == [(True, False), (True, True)]
