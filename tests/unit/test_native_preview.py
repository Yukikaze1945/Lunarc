import asyncio
import pytest
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
