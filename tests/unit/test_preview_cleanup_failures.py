import ctypes
import threading
from types import SimpleNamespace

import pytest

from dlss_gpu_engine import Live
from gpu_preview_engine import GpuPreview
from gpu_transfer_export import Chain
from dlss_service import DLSSService


def test_native_close_fault_never_reuses_deleted_host_address():
    calls = []
    def fail(handle):
        calls.append(handle.value)
        raise OSError('exception: access violation reading 0x910')
    live = Live.__new__(Live)
    live._handle = ctypes.c_void_p(123)
    live._lib = SimpleNamespace(dlss_gpu_host_close=fail)
    with pytest.raises(OSError):
        live.close()
    live.close()
    assert calls == [123] and not live.is_alive()


def test_failed_display_cleanup_still_releases_chain_and_loader(monkeypatch):
    preview = GpuPreview()
    calls = []
    def fail():
        calls.append('display')
        raise OSError('display close failed')
    preview.native_presenter = SimpleNamespace(close=fail)
    preview.chain = SimpleNamespace(close=lambda: calls.append('chain'))
    preview.live = SimpleNamespace(close=lambda: calls.append('anchor'))
    preview.provider = SimpleNamespace(close_independent_layer=lambda: calls.append('flow'))
    preview.bridges = {'cookie': object()}
    monkeypatch.setattr('gpu_preview_engine.htx.release_loader_cookies', lambda _: calls.append('loader'))
    try:
        with pytest.raises(RuntimeError, match='display close failed'):
            preview.close()
        assert calls == ['display', 'chain', 'anchor', 'flow', 'loader']
        preview.close()
        assert calls == ['display', 'chain', 'anchor', 'flow', 'loader']
    finally:
        preview.shutdown()


def test_failed_effect_close_keeps_decoder_last_and_cleanup_idempotent():
    events = []
    def fail():
        events.append('post')
        raise OSError('post cleanup failure')
    chain = Chain.__new__(Chain)
    chain.post_control = SimpleNamespace(close=fail)
    chain.temporal_control = SimpleNamespace(close=lambda: events.append('temporal'))
    chain.live = SimpleNamespace(close=lambda: events.append('anchor'))
    chain.flow_provider = None
    chain.decoder = 123
    chain.b = {'nvdec': SimpleNamespace(nvdec_bridge_close=lambda _: events.append('decoder'))}
    with pytest.raises(RuntimeError, match='post cleanup failure'):
        chain.close()
    assert events == ['post', 'temporal', 'anchor', 'decoder']
    chain.close()
    assert events == ['post', 'temporal', 'anchor', 'decoder']


def test_failed_cache_cleanup_is_not_retried_on_deleted_decoder_context():
    preview = GpuPreview()
    calls = []
    def fail():
        calls.append('old_cuda_context')
        raise RuntimeError('CUDA cache close failed')
    preview.post_cache = SimpleNamespace(close=fail)
    try:
        with pytest.raises(RuntimeError, match='CUDA cache close failed'):
            preview.close()
        preview.close()
        assert calls == ['old_cuda_context']
        assert preview.post_cache.identity is None
    finally:
        preview.shutdown()


def test_preview_native_fault_stops_subsequent_dll_calls():
    calls = []
    def fail(*args, **kwargs):
        calls.append('native')
        raise OSError('exception: access violation reading 0x910')
    service = DLSSService.__new__(DLSSService)
    service.active_job_id = service._live_session_owner = service._live_session = None
    service._native_degraded = None
    service._live_lock = threading.Lock()
    service._settings_lock = threading.Lock()
    service.settings, service.video_path = {}, 'fixture.mkv'
    service._gpu_preview = SimpleNamespace(render=fail, lifecycle_phase='open new pipeline')
    service._get_gpu_preview = lambda: service._gpu_preview
    first = service._render_gpu_preview(0, 'split', .5, 1., (.5, .5), False)
    second = service._render_gpu_preview(0, 'split', .5, 1., (.5, .5), False)
    assert 'access violation' in first[1] and '重启后端' in second[1]
    assert calls == ['native']
