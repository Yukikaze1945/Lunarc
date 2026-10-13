"""Parameter refresh must not starve completed frames or rebuild model owners."""
import asyncio
from types import SimpleNamespace
import ctypes as C

from preview_player import PreviewPlayer
from preview_cache import PostCache, _Copy2D
from gpu_preview_engine import GpuPreview
from tests.unit.test_preview_player import Service, Socket, until, close


def test_parameter_refresh_preserves_ack_and_delivers_completed_frame():
    async def scenario():
        service, socket = Service(), Socket()
        player = PreviewPlayer(socket, service)
        player.command(dict(action='seek', frame=2, request_id=1, transport='binary', ack=True))
        task = asyncio.create_task(player.run())
        await until(service.entered.is_set)
        for _ in range(5):
            service._preview_revision += 1
            player.command(dict(action='refresh',request_id=1))
        service.release.set()
        await until(lambda: len(socket.images)==1)
        first = next(m for m in socket.messages if m['type']=='frame')
        assert first['preview']['settings_current'] is False
        assert len(service.calls)==1
        player.command(dict(action='refresh',request_id=1))
        assert player.awaiting == 1
        player.command(dict(action='presented',sequence=1))
        await until(lambda: len(socket.images)==2)
        latest = [m for m in socket.messages if m['type']=='frame'][-1]
        assert latest['preview']['settings_current'] is True
        assert latest['preview']['settings_revision']==5
        await close(player,task)
    asyncio.run(scenario())


def test_parameter_refresh_cannot_publish_reloaded_source():
    async def scenario():
        service, socket = Service(), Socket()
        service._preview_source_revision = 0
        player = PreviewPlayer(socket,service)
        player.command(dict(action='seek',request_id=1,frame=1,transport='binary',ack=True))
        task=asyncio.create_task(player.run())
        await until(service.entered.is_set)
        player.command(dict(action='refresh',request_id=1))
        service._preview_source_revision += 1
        service._preview_revision += 1
        service.release.set()
        await until(lambda: len(socket.images)==1)
        assert len(service.calls)==2
        assert next(m for m in socket.messages if m['type']=='frame')['preview']['stale_results_discarded']==1
        await close(player,task)
    asyncio.run(scenario())


def test_post_edits_on_combined_chain_do_not_update_model_options():
    preview=GpuPreview()
    updates=[]
    preview.live=SimpleNamespace(update_runtime_options=lambda settings:updates.append(settings))
    preview.multi={'scales':[1.]}
    preview.chain=SimpleNamespace(post_control=SimpleNamespace(update_controls=lambda _:None),
        temporal_control=SimpleNamespace(controls=None))
    old={'render_pipeline':['nr','vsr'],'post_controls':{'chroma_gain':1}}
    preview.active_settings=old
    preview.post_cache.valid=True
    try:
        assert preview._update_runtime(dict(old,post_controls={'chroma_gain':0}))
        assert preview.post_only_update and preview.post_cache.valid and updates==[]
        assert preview._update_runtime(dict(preview.active_settings,residual_strength=.5))
        assert preview.chain.strength==.5 and not preview.post_cache.valid
    finally:
        preview.chain=preview.live=None
        preview.shutdown()


def test_gpu_cache_keeps_own_rows_when_borrowed_inputs_change():
    cache=PostCache()
    source=(C.c_ubyte*32)(*range(32));candidate=(C.c_ubyte*32)(*range(32,64))
    owned_source=(C.c_ubyte*512)();owned_candidate=(C.c_ubyte*512)()
    def copy(pointer):
        row=C.cast(pointer,C.POINTER(_Copy2D)).contents
        for y in range(row.Height):
            C.memmove(row.dstDevice+y*row.dstPitch,row.srcDevice+y*row.srcPitch,row.WidthInBytes)
        return 0
    cache.cuda=SimpleNamespace(cuMemcpy2D_v2=copy)
    cache._push=cache._pop=lambda:None
    def setup(*args):
        cache.device,cache.context,cache.width,cache.height,cache.wide=args
        cache.identity=args;cache.pixel_bytes=4;cache.pitch=256
        cache.source_ptr=C.addressof(owned_source);cache.candidate_ptr=C.addressof(owned_candidate)
    cache._open=setup
    cache.capture(device=1,context=2,width=4,height=2,wide=False,
        source_ptr=C.addressof(source),source_pitch=16,
        candidate=dict(device_ptr=C.addressof(candidate),pitch_bytes=16),frame='frame')
    C.memset(C.addressof(source),0,32);C.memset(C.addressof(candidate),0,32)
    assert bytes(owned_source[:16])==bytes(range(16))
    assert bytes(owned_candidate[256:272])==bytes(range(48,64))
    assert cache.report()['host_image_bytes']==0 and cache.copies==2
    cache.identity=None
