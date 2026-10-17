"""Pure/mocked regressions. Do not run GPU or image exports for these contracts."""
import gc
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

import image_render as images


def entry(number, path='source.png', hdr=False):
    pixels=np.full((4,8,4),12.5 if hdr else number,np.float16 if hdr else np.uint8)
    return images.ImageResult(str(number),(path,1,2),'settings',
        np.full((4,8,4),255,np.uint8),pixels,hdr,{},number)


def test_result_cache_keeps_only_current_and_previous_and_reclaims_old_array():
    cache=images.ImageResults()
    first=entry(1);old=weakref.ref(first.pixels)
    cache.put(first);del first
    cache.put(entry(2));assert cache.previous('2').id=='1'
    cache.put(entry(3));gc.collect()
    assert list(cache.entries)==['2','3']
    assert old() is None
    assert cache.previous('3').id=='2'
    with pytest.raises(ValueError,match='已清理'):cache.get('1')
    assert not cache.get('3').pixels.flags.writeable


def test_memory_passes_preserve_order_fp16_and_extended_hdr_without_tiling():
    settings=dict(render_pipeline=['dlaa','nr','truehdr','vsr:second'],truehdr_enabled=True,
                  vsr_output_mode='scale',vsr_scale=2)
    original=np.full((5,7,4),100,np.uint8)
    calls=[]
    def nr(pixels,options):
        calls.append(('nr',pixels.dtype,pixels.shape))
        return pixels.copy(),False,dict(stage='nr',resources_released=True)
    def stage(pixels,hdr,options,kind):
        calls.append((kind,pixels.dtype,pixels.shape))
        assert options.get('image_model_tile') is None  # No automatic tiling.
        if kind=='truehdr':
            result=np.full(pixels.shape,12.5,np.float16);result[0,0,0]=-.25
            return result,True,dict(stage=kind,resources_released=True)
        if hdr:
            assert pixels[0,0,0]==-.25 and pixels[0,0,1]==12.5
        return pixels.copy(),hdr,dict(stage=kind,resources_released=True)
    result,hdr,reports,transfers=images.render_passes(original,settings,nr,stage)
    assert [item[0] for item in calls]==['dlaa','nr','truehdr','vsr:second']
    assert hdr and result.dtype==np.float16
    assert result.shape==(5,7,4) and result[0,0,0]==-.25
    assert transfers>0 and all(item['resources_released'] for item in reports)


def test_failed_pass_cannot_run_following_model():
    called=[]
    def stage(pixels,hdr,settings,kind):
        called.append(kind);raise RuntimeError('cleanup failed')
    with pytest.raises(RuntimeError,match='cleanup failed'):
        images.render_passes(np.ones((8,8,4),np.uint8),
            {'render_pipeline':['dlaa','vsr']},None,stage)
    assert called==['dlaa']


def test_signature_ignores_execution_mode_but_not_pixel_parameters(monkeypatch):
    settings=dict(render_pipeline=['vsr'],vsr_quality=4)
    assert images.signature(settings)==images.signature(dict(settings,image_memory_mode='fast'))
    assert images.signature(settings)!=images.signature(dict(settings,vsr_quality=3))


def test_cpu_cache_status_and_staleness_never_open_gpu(tmp_path):
    path=tmp_path/'source.png';path.write_bytes(b'mock source')
    cache=images.ImageResults();settings=dict(render_pipeline=['vsr'],vsr_quality=4)
    result=entry(1)
    result.source=images.source_key(str(path));result.settings_key=images.signature(settings)
    cache.put(result)
    assert cache.status(str(path),settings)['current']
    assert not cache.status(str(path),dict(settings,vsr_quality=3))['current']
    path.write_bytes(b'changed source')
    assert not cache.status(str(path),settings)['source_unchanged']


def test_cached_format_conversion_does_not_acquire_gpu_even_during_export(tmp_path,monkeypatch):
    from dlss_service import DLSSService
    import image_source
    source=tmp_path/'source.png';source.write_bytes(b'mock source')
    settings=dict(render_pipeline=['vsr'],vsr_quality=4)
    result=entry(1);result.source=images.source_key(str(source));result.settings_key=images.signature(settings)
    cache=images.ImageResults();cache.put(result)
    service=DLSSService.__new__(DLSSService)
    service.settings=settings;service._settings_lock=threading.RLock()
    service._gpu_preview=SimpleNamespace(image_results=cache)
    service.active_job_id='other GPU export'
    class ForbiddenLock:
        def acquire(self,**_):pytest.fail('cached format conversion acquired GPU ownership')
    service._live_lock=ForbiddenLock()
    service._ensure_native_healthy=lambda:pytest.fail('cached export checked/created a GPU device')
    monkeypatch.setattr(image_source,'validate_destination',lambda *a,**k:None)
    calls=[]
    monkeypatch.setattr(images,'save_result',lambda *a,**k:calls.append(a[0].id) or {'path':'result.png'})
    saved=service.export_image(str(source),'result.png',render_source='preview',result_id='1')
    assert saved['reused_render'] and calls==['1']
    service.settings=dict(settings,vsr_quality=3)
    with pytest.raises(ValueError,match='参数已改变'):
        service.export_image(str(source),'result.png',render_source='preview',result_id='1')
    saved=service.export_image(str(source),'result.png',render_source='preview',result_id='1',allow_stale=True)
    assert not saved['settings_current']
