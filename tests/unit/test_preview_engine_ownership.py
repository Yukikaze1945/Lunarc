import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import gpu_preview_engine
from dlss_service import DLSSService


def test_layout_and_render_startup_share_exactly_one_engine(monkeypatch):
    service=DLSSService.__new__(DLSSService)
    entered,release=threading.Event(),threading.Event()
    created=[]
    def construct():
        owner=object();created.append(owner);entered.set()
        assert release.wait(2)
        return owner
    monkeypatch.setattr(gpu_preview_engine,'GpuPreview',construct)
    with ThreadPoolExecutor(max_workers=2) as workers:
        render=workers.submit(service._get_gpu_preview)
        assert entered.wait(2)
        layout=workers.submit(service._get_gpu_preview)
        release.set()
        assert render.result(2) is layout.result(2)
    assert len(created)==1


def test_host_image_entry_closes_device_owner_before_reusing_legacy_session():
    service=DLSSService.__new__(DLSSService)
    events=[]
    service.active_job_id=None
    service._ensure_native_healthy=lambda:None
    service._gpu_preview=SimpleNamespace(live=object(),close=lambda:events.append('close_device'))
    service._live_session=object()
    service._live_session_dims=(320,192)
    service._live_session_kind='gpu_zero_motion'
    session=SimpleNamespace(_gpu_options_key=('stable',))
    service._live_session=session
    service._preview_live_kind=lambda:'gpu_zero_motion'
    service._gpu_preview_options_key=lambda:('stable',)
    assert service._get_or_create_live(320,192) is session
    assert events==['close_device']


def test_native_preview_cannot_close_export_session_after_acquiring_lock():
    service=DLSSService.__new__(DLSSService)
    events=[]
    service._live_lock=threading.Lock()
    service.active_job_id=None
    service._live_session_owner='export'
    service._live_session=SimpleNamespace(close=lambda:events.append('closed'))
    result=service._render_gpu_preview(0,'split',.5,1.,(.5,.5),False,native=True)
    assert result==(None,'GPU_EXPORT_BUSY') and events==[]
    assert service._live_lock.acquire(blocking=False)
    service._live_lock.release()


def test_export_claim_arriving_while_preview_waits_for_lock_is_respected():
    service=DLSSService.__new__(DLSSService)
    service.active_job_id=None
    service._live_session_owner=None
    class Lock:
        def acquire(self,**_):service.active_job_id='export';return True
        def release(self):pass
    service._live_lock=Lock()
    service._live_session=SimpleNamespace(close=lambda:(_ for _ in ()).throw(AssertionError('export closed')))
    assert service._render_gpu_preview(0,'split',.5,1.,(.5,.5),False,native=True)==(None,'GPU_EXPORT_BUSY')


def test_failed_export_setup_releases_owned_host_without_opening_preview():
    import pytest
    service=DLSSService.__new__(DLSSService);events=[]
    service._live_lock=threading.Lock()
    def fail(*_):
        service._live_session_owner='export'
        service._live_session=SimpleNamespace(close=lambda:events.append('closed'))
        raise ValueError('guidance setup failed')
    service._execute_job_owned=fail
    with pytest.raises(ValueError,match='guidance setup failed'):service._execute_job({},None)
    assert events==['closed'] and service._live_session_owner is None and service._live_session is None
