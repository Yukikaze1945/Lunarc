import ctypes as C
import pytest
from temporal_control import (TemporalControls, temporal_from_settings, GpuTemporalControl, _Observe, _Process)


def test_default_does_not_open_any_library_or_change_output():
    owner = GpuTemporalControl(hdr=True)
    args = dict(device=1, cuda_context=2, source_ptr=3, source_pitch=256,
                width=32, height=32, reset=False)
    candidate = {"device_ptr": 55}
    assert owner.observe(**args)["scene_cut"] is False
    assert owner.process(**args, candidate=candidate) is candidate
    assert owner._lib is None
    owner.close()
    from post_control import PostControlError
    with pytest.raises(PostControlError, match="closed"):
        owner.observe(**args)
    with pytest.raises(PostControlError, match="closed"):
        owner.process(**args, candidate=candidate)


@pytest.mark.parametrize("raw", [False, [], {"mode": "flow"}, {"scene_detection": 1},
    {"history_weight": 1}, {"history_weight": float("nan")}, {"history_weight": True}, {"unknown": 1}])
def test_invalid_controls_explicit_failure(raw):
    with pytest.raises(ValueError):
        temporal_from_settings({"temporal_controls": raw})


def test_old_settings_migrate_disabled_and_abi_layouts():
    assert temporal_from_settings({"style": 2}) == TemporalControls()
    assert C.sizeof(_Observe) == 64 and C.sizeof(_Process) == 56
    assert temporal_from_settings({"temporal_controls": {"scene_detection": True}}).enabled
