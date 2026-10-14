import ctypes as C
import pytest
from temporal_control import (TemporalControls, temporal_from_settings, GpuTemporalControl, _Observe, _Process)


def test_explicitly_disabled_does_not_open_any_library_or_change_output():
    owner = GpuTemporalControl(TemporalControls(scene_detection=False), hdr=True)
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


def test_default_scene_observation_and_abi_layouts():
    assert temporal_from_settings({"style": 2}) == TemporalControls()
    assert temporal_from_settings({}).scene_detection is True
    assert temporal_from_settings({}).mode == 'off'
    assert temporal_from_settings({'temporal_controls': {'scene_detection': False}}).enabled is False
    assert C.sizeof(_Observe) == 64 and C.sizeof(_Process) == 56
    assert temporal_from_settings({"temporal_controls": {"scene_detection": True}}).enabled


def test_temporal_close_failure_does_not_retry_released_handle():
    from types import SimpleNamespace
    calls = []
    def fail(handle):
        calls.append(handle)
        raise OSError('device removed during release')
    owner = GpuTemporalControl()
    owner._handle = 123
    owner._lib = SimpleNamespace(gpu_temporal_close=fail)
    with pytest.raises(OSError, match='device removed'):
        owner.close()
    owner.close()
    assert calls == [123] and owner._handle is None and owner._closed


def test_switching_stability_off_releases_histories_without_closing_owner():
    from types import SimpleNamespace
    calls = []
    owner = GpuTemporalControl(TemporalControls(mode='static_low_frequency'))
    owner._handle, owner._identity = 123, (1, 2, 32, 32, 8)
    owner._lib = SimpleNamespace(gpu_temporal_close=lambda handle: calls.append(handle))
    owner.update_controls(TemporalControls())
    assert calls == [123] and owner._handle is None and owner._identity is None
    assert not owner._closed and owner.controls.scene_detection
    owner.update_controls(TemporalControls())
    assert calls == [123]
    owner.close()
