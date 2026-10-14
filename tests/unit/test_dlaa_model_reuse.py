import ctypes as C
from types import SimpleNamespace

import pytest

import dlss_gpu_engine as nr
from dlaa_gpu_engine import Live
from native_effects import update_dlaa_model


def make_live(code=0):
    calls = []
    def recreate(handle, options):
        value = C.cast(options, C.POINTER(nr._Options)).contents
        calls.append((handle.value, value.preset))
        return code
    live = Live.__new__(Live)
    live._handle = C.c_void_p(123)
    live.settings = {'preset': 1, 'dlaa_preset': 12}
    live.render_preset, live.session_id = 12, 5
    live._lib = SimpleNamespace(dlss_gpu_host_recreate_dlaa=recreate,
        dlss_gpu_host_session_id=lambda _: 6, dlss_gpu_host_last_error=lambda: b'create failed')
    return live, calls


def test_model_change_reuses_handle_and_updates_actual_session_id():
    live, calls = make_live()
    assert live.recreate_preset({'preset': 1, 'dlaa_preset': 13})
    assert calls == [(123, 13)]
    assert live.is_alive() and live.render_preset == 13 and live.session_id == 6


def test_failed_recreation_does_not_claim_requested_model_was_applied():
    live, _ = make_live(3)
    with pytest.raises(RuntimeError, match='create failed'):
        live.recreate_preset({'preset': 1, 'dlaa_preset': 13})
    assert live.render_preset == 12 and live.settings['dlaa_preset'] == 12


def test_borrowed_primary_and_stage_recreate_exactly_once():
    live, calls = make_live()
    stage = SimpleNamespace(selected='nr_dlaa', effect=live, settings=live.settings)
    chain = SimpleNamespace(live=live, output_effect=SimpleNamespace(effects=[stage]))
    assert update_dlaa_model(chain, {'preset': 1, 'dlaa_preset': 13})
    assert calls == [(123, 13)] and stage.settings['dlaa_preset'] == 13


def test_old_host_is_detected_before_any_model_changes():
    live, calls = make_live()
    old = SimpleNamespace(_lib=SimpleNamespace(), recreate_preset=lambda _: pytest.fail('old DLL called'))
    stage_a = SimpleNamespace(selected='nr_dlaa', effect=live)
    stage_b = SimpleNamespace(selected='nr_dlaa', effect=old)
    chain = SimpleNamespace(live=SimpleNamespace(), output_effect=SimpleNamespace(effects=[stage_a, stage_b]))
    assert update_dlaa_model(chain, {'dlaa_preset': 13}) is False
    assert calls == []
