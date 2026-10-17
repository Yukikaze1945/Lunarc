import asyncio
import pytest

def test_pipeline_cleanup_freezes_before_context_close_and_explicit_release_retires_it():
    from types import SimpleNamespace as NS
    from gpu_preview_engine import GpuPreview
    preview=GpuPreview();events=[]
    held=NS(freeze=lambda:events.append('freeze') or True,close=lambda:events.append('display close'))
    preview.native_presenter=held
    preview.live=NS(close=lambda:events.append('context close'))
    try:
        preview.executor.submit(preview._close,preserve_display=True).result()
        assert events==['freeze','context close']
        assert preview.native_presenter is None and preview.frozen_presenter is held
        preview.close()
        assert events==['freeze','context close','display close']
        assert preview.frozen_presenter is None
    finally:preview.shutdown()
from native_preview import validate_layout
from preview_player import PreviewPlayer
from tests.unit.test_preview_player import Service,Socket,until,close


def test_native_layout_checks_geometry_and_nonfinite_controls():
    layout=dict(parent='123',x=0,y=0,width=100,height=50,view='split')
    assert validate_layout(layout)['zoom']==1.
    for change in ({'parent':'-1'},{'width':0},{'height':20000},{'zoom':float('nan')},{'pan_x':2},{'view':'flow'}):
        with pytest.raises(ValueError):validate_layout(dict(layout,**change))


def test_native_transport_keeps_ack_without_transmitting_picture_bytes():
    async def scenario():
        service,socket=Service(),Socket()
        player=PreviewPlayer(socket,service)
        player.command(dict(action='seek',frame=2,request_id=1,transport='native',ack=True))
        task=asyncio.create_task(player.run());await until(service.entered.is_set)
        service.release.set();await until(lambda:any(m['type']=='frame' for m in socket.messages))
        frame=next(m for m in socket.messages if m['type']=='frame')
        assert frame['encoding']=='native_fp16' and frame['jpeg_bytes']==0
        assert 'image' not in frame and not socket.images
        assert service.calls[0]['native'] is True
        assert player.awaiting==1
        player.command(dict(action='presented',sequence=1))
        assert player.awaiting is None
        await close(player,task)
    asyncio.run(scenario())


def test_export_busy_pauses_native_player_without_error_or_retry_loop():
    async def scenario():
        service,socket=Service(),Socket()
        service.render_preview=lambda **_: (None,'GPU_EXPORT_BUSY')
        player=PreviewPlayer(socket,service)
        player.command(dict(action='play',frame=2,request_id=1,transport='native',ack=True))
        task=asyncio.create_task(player.run())
        await until(lambda:any(m['type']=='busy' for m in socket.messages))
        assert not player.playing and not player.dirty
        assert not socket.images and not any(m['type']=='error' for m in socket.messages)
        await close(player,task)
    asyncio.run(scenario())


def test_neutral_sdr_texture_enters_native_preview_with_its_full_extent(monkeypatch):
    from types import SimpleNamespace as NS
    from gpu_preview_engine import GpuPreview
    import native_effects
    calls=[]
    class Exchange:
        def __init__(self,device,context,size,wide):
            calls.append((device,context,size,wide))
        def color(self,texture):
            assert (texture['width'],texture['height'])==(320,192)
            return dict(format='rgba8_cuda',width=320,height=192,cuda_context=123,device_ptr=456,pitch_bytes=1280)
    monkeypatch.setattr(native_effects,'Exchange',Exchange)
    preview=object.__new__(GpuPreview)
    preview.hdr=preview.sdr10=False
    preview.live=NS(d3d12_device=lambda:99)
    preview._native_candidate(dict(d3d12_resource=77),NS(width=320,height=192,cuda_context=123))
    assert calls==[(99,123,(320,192),False)]
    assert preview.native_result['format']=='rgba8_cuda'


def test_cached_display_device_loss_is_reported_and_releases_service_lock(monkeypatch):
    import threading
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    import image_render
    from dlss_service import DLSSService
    service=DLSSService.__new__(DLSSService)
    service._live_lock=threading.Lock()
    service._ensure_native_healthy=lambda:None
    service.active_job_id=None;service.video_path='image.png'
    result=NS(source=('image.png',1,1))
    service._gpu_preview=NS(image_results=NS(get=lambda _:result),
        present_image_result=Mock(side_effect=RuntimeError('FP16 present: 0x887a0005 (removed reason 0x887a0006)')))
    marked=[];service._mark_native_degraded=marked.append
    monkeypatch.setattr(image_render,'source_key',lambda _:result.source)
    with pytest.raises(RuntimeError,match='FP16 present'):
        service.compare_image_results('result')
    assert marked==['GPU 设备已被驱动重置']
    assert service._live_lock.acquire(blocking=False)
    service._live_lock.release()
    assert service._gpu_preview.image_results.get('result') is result
