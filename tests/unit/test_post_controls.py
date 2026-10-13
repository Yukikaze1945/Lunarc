import ctypes
import pytest
from post_control import (PostControls, GpuPostControl, controls_from_settings,
                          native_options, _Options, _Candidate, PostControlError)


def test_neutral_strict_identity_without_library_or_any_copy():
    candidate = {"device_ptr": 111, "pitch_bytes": 512, "fence_value": 10}
    bridge = GpuPostControl(hdr=True)
    assert bridge.process(device=1, cuda_context=2, source_ptr=3,
        source_pitch=512, width=64, height=32, candidate=candidate) is candidate
    assert bridge._lib is None and bridge._handle is None
    assert bridge.diagnostics()["neutral_bypass"]
    bridge.close()
    with pytest.raises(PostControlError, match="closed"):
        bridge.process(device=1, cuda_context=2, source_ptr=3, source_pitch=512,
                       width=64, height=32, candidate=candidate)


@pytest.mark.parametrize("field,value", [("total_gain", -1), ("hue_protection", 2),
    ("soft_compression", float("nan")), ("roi_feather", float("inf")),
    ("roi_kind", True), ("debug_mode", 4), ("total_gain", "1"),
    ("roi_bounds", [0, 0, 0, 1]), ("roi_bounds", [0, 0, 2, 1]),
    ("roi_bounds", [0, 0, 1])])
def test_invalid_controls_refused(field, value):
    with pytest.raises(ValueError):
        controls_from_settings({"post_controls": {field: value}})


def test_migrate_old_settings_without_touching_other_parameters():
    old = {"style": 2, "reconstruction_mode": "triple"}
    assert controls_from_settings(old) == PostControls()
    assert old == {"style": 2, "reconstruction_mode": "triple"}
    assert not controls_from_settings({"post_controls": {"lightness_gain": .5}}).neutral


def test_explicit_unknown_settings_error():
    with pytest.raises(ValueError, match="unknown"):
        controls_from_settings({"post_controls": {"invented": 1}})


def test_native_abi_controls_and_geometry():
    assert ctypes.sizeof(_Options) == 88
    assert ctypes.sizeof(_Candidate) == 48
    options = native_options(controls_from_settings({"post_controls": {
        "roi_kind": 2, "roi_bounds": [.1, .2, .8, .9], "debug_mode": 3}}), 160, 90, 8)
    assert options.struct_size == 88 and options.pixel_bytes == 8 and options.color_mode == 1
    assert native_options(controls_from_settings({}), 160, 90, 8, sdr10=True).color_mode == 2
    assert options.roi_kind == 2 and options.debug_mode == 3
    assert tuple(options.roi_bounds) == pytest.approx((.1, .2, .8, .9))


def test_service_partial_controls_patch_keeps_previous_values_and_rejects_atomically():
    from dlss_service import DLSSService
    service = DLSSService()
    service.update_settings({"post_controls": {"total_gain": .5},
                             "temporal_controls": {"scene_detection": True}})
    service.update_settings({"post_controls": {"hue_protection": .4},
                             "temporal_controls": {"history_weight": .2}})
    assert service.settings["post_controls"]["total_gain"] == .5
    assert service.settings["post_controls"]["hue_protection"] == .4
    assert service.settings["temporal_controls"]["scene_detection"] is True
    before = dict(service.settings["post_controls"])
    with pytest.raises(ValueError):
        service.update_settings({"post_controls": {"total_gain": 99}})
    assert service.settings["post_controls"] == before


def test_sdr_chain_temporal_precedes_final_post_and_never_reuses_post_output():
    from types import SimpleNamespace
    from gpu_transfer_export import Chain

    temporal_inputs = []
    post_inputs = []
    stable_outputs = []
    final_outputs = []

    class Temporal:
        def process(self, **kwargs):
            temporal_inputs.append(kwargs["candidate"])
            stable = {"stage": "temporal", "frame": len(temporal_inputs)}
            stable_outputs.append(stable)
            return stable

    class Post:
        def process(self, **kwargs):
            post_inputs.append(kwargs["candidate"])
            final = {"stage": "post", "frame": len(post_inputs),
                     "roi_and_zero_strength_final": True}
            final_outputs.append(final)
            return final

    chain = object.__new__(Chain)
    chain.temporal_control = Temporal()
    chain.post_control = Post()
    chain._frame_reset = False
    frame = SimpleNamespace(cuda_context="ctx")
    dlss_first = {"stage": "dlss", "frame": 1, "d3d12_device": "device"}
    dlss_second = {"stage": "dlss", "frame": 2, "d3d12_device": "device"}
    frame.width, frame.height = 64, 32

    result_first = chain._post_process(frame, 100, 512, dlss_first)
    result_second = chain._post_process(frame, 200, 512, dlss_second)

    assert temporal_inputs == [dlss_first, dlss_second]
    assert temporal_inputs[1] is dlss_second
    assert final_outputs[0] not in temporal_inputs
    assert post_inputs == stable_outputs
    assert result_first is final_outputs[0]
    assert result_second is final_outputs[1]


def test_hdr_chain_temporal_then_post_then_set_composite():
    from types import SimpleNamespace
    from gpu_hdr_transfer_export import HdrChain

    events = []
    composited = SimpleNamespace(stage="composited", device_ptr=11, pitch_bytes=64)
    stable = {"stage": "temporal", "device_ptr": 22}
    controlled = {"stage": "post", "roi_and_zero_strength_final": True}
    returned = {"stage": "set_composite", "value": 33}

    class Temporal:
        controls = SimpleNamespace(mode="static_low_frequency")

        def process(self, **kwargs):
            events.append(("temporal", kwargs["candidate"]))
            return stable

    class Post:
        controls = SimpleNamespace(neutral=False)

        def process(self, **kwargs):
            events.append(("post", kwargs["candidate"]))
            return controlled

    class Live:
        def d3d12_device(self):
            return "device"

    class CompositeBridge:
        def set_composite(self, candidate):
            events.append(("set_composite", candidate))
            return returned

    chain = object.__new__(HdrChain)
    chain.temporal_control = Temporal()
    chain.post_control = Post()
    chain.live = Live()
    chain.bridge = CompositeBridge()
    chain.context = "ctx"
    chain.src = (64, 32)
    chain.processing_size = chain.src
    chain._frame_reset = False
    chain.temporal_control.controls.mode = "static_low_frequency"
    master = SimpleNamespace(device_ptr=7, pitch_bytes=64)
    stages = {}

    result = chain._post_process(master, composited, stages)

    assert events == [("temporal", {"device_ptr": 11, "pitch_bytes": 64}), ("post", stable),
                      ("set_composite", controlled)]
    assert result is returned
    assert "post_control" in stages


def test_hdr_chain_strict_neutral_and_temporal_off_bypass_every_stage():
    from types import SimpleNamespace
    from gpu_hdr_transfer_export import HdrChain

    class MustNotRun:
        def process(self, **kwargs):
            pytest.fail("neutral HDR bypass must not open or call native stages")

    class MustNotComposite:
        def set_composite(self, candidate):
            pytest.fail("strict neutral bypass must preserve the original composite")

    chain = object.__new__(HdrChain)
    chain.temporal_control = SimpleNamespace(controls=SimpleNamespace(mode="off"),
                                             process=MustNotRun().process)
    chain.post_control = SimpleNamespace(controls=SimpleNamespace(neutral=True),
                                         process=MustNotRun().process)
    chain.bridge = MustNotComposite()
    original = object()
    assert chain._post_process(object(), original, {}) is original
