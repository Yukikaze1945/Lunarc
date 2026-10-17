"""HDR still master, display and processing routing contracts; no GPU."""
from types import SimpleNamespace

import numpy as np
import pytest

import hdr_image_decode
import image_render
import image_source
import native_effects


def master():
    pixels=np.full((5,7,4),12.5,np.float16)
    pixels[0,0,0]=-.25
    pixels[:,:,3]=.5
    return pixels


def test_display_copy_does_not_replace_or_modify_fp16_master(monkeypatch):
    pixels=master();before=pixels.copy()
    monkeypatch.setattr(hdr_image_decode,'probe_hdr',lambda _: {'bit_depth':12})
    monkeypatch.setattr(hdr_image_decode,'decode_hdr',lambda *a: pixels)
    assert image_source.read_pixels('source.avif') is pixels
    display=image_source.read_rgba('source.avif')
    assert display.dtype==np.uint8 and display.shape==pixels.shape
    assert np.all(display[:,:,3]==128)
    assert np.array_equal(pixels,before)


def test_processing_probe_distinguishes_source_pq12_from_linear_fp16(monkeypatch):
    source=dict(width=7,height=5,bit_depth=12,color_transfer='smpte2084',is_hdr=True)
    monkeypatch.setattr(hdr_image_decode,'probe_hdr',lambda _:source)
    probe=image_source.probe('source.avif')
    assert probe['source_colour'] is source
    assert probe['pix_fmt']=='rgba16f' and probe['color_transfer']=='linear'
    assert probe['is_hdr'] and (probe['width'],probe['height'])==(8,8)
    native_effects.validate(dict(render_pipeline=['dlaa'],export_backend='gpu_transfer'),probe)


def test_incomplete_pq_metadata_cannot_fall_back_to_sdr(monkeypatch):
    import pipeline
    monkeypatch.setattr(pipeline,'probe_pixel_info',lambda _:dict(pix_fmt='yuv444p',
        bit_depth=8,color_transfer='smpte2084',is_hdr=False))
    with pytest.raises(ValueError,match='不会自动降为 SDR'):
        image_source.read_pixels('incomplete.avif')


def test_hdr_memory_pass_starts_fp16_and_never_enters_sdr_nr(tmp_path):
    original=master()
    calls=[]
    def forbidden(*_):raise AssertionError('HDR entered SDR NR')
    def process(pixels,hdr,settings,stage):
        calls.append(stage)
        assert hdr and pixels.dtype==np.float16 and np.all(pixels[:,:,3]==1)
        assert pixels[0,0,0]==-.25 and pixels[0,0,1]==12.5
        return pixels.copy(),hdr,dict(stage=stage)
    settings=dict(render_pipeline=['nr','dlaa'])
    pixels,hdr,_,_=image_render.render_passes(original,settings,forbidden,process)
    assert hdr and calls==['nr','dlaa'] and pixels.shape==original.shape
    source=tmp_path/'source.avif';source.write_bytes(b'mocked source')
    result=image_render.make_result(str(source),image_render.source_key(source),settings,
                                   original,pixels,hdr,{})
    assert np.all(result.pixels[:,:,3]==.5)
    assert result.pixels[0,0,0]==-.25 and result.pixels[0,0,1]==12.5


def test_original_comparison_uses_hdr_mapping(monkeypatch):
    flags=[]
    def display(pixels,hdr):
        flags.append(hdr)
        return np.zeros((*pixels.shape[:2],3),np.uint8)
    monkeypatch.setattr(image_render,'display_bgr',display)
    result=image_render.ImageResult('one',('source.avif',1,1),'settings',master(),master(),True,{},0)
    assert image_render.preview_pair(result)
    assert flags==[True,True]


def test_hdr_source_can_save_hdr_without_truehdr(monkeypatch):
    import image_hdr_export
    monkeypatch.setattr(hdr_image_decode,'probe_hdr',lambda _:dict(is_hdr=True))
    monkeypatch.setattr(image_hdr_export,'capabilities',lambda:dict(avif=dict(available=True,alpha=True)))
    image_hdr_export.validate_request(dict(render_pipeline=['dlaa']),'avif','source.avif',True)


def test_native_original_upload_keeps_hdr_descriptor(monkeypatch):
    uploads=[]
    class Upload:
        def __init__(self,pixels,context=None):
            self.context=SimpleNamespace(value=1)
            uploads.append(pixels.dtype)
        def close(self):pass
    monkeypatch.setattr(image_render,'PixelUpload',Upload)
    monkeypatch.setattr(native_effects,'Passthrough',lambda *a,**k:SimpleNamespace(close=lambda:None))
    result=image_render.ImageResult('one',('source.avif',1,1),'settings',master(),master(),True,{},0)
    resource=image_render.DisplayResources(result)
    try:
        assert resource.source_hdr and resource.result_hdr
        assert uploads==[np.float16,np.float16]
    finally:resource.close()


def test_fast_hdr_image_reuses_owner_and_retires_it_on_setting_change(tmp_path,monkeypatch):
    from gpu_preview_engine import GpuPreview
    source=tmp_path/'source.avif';source.write_bytes(b'mocked source')
    monkeypatch.setattr(image_source,'read_pixels',lambda _:master())
    monkeypatch.setattr(image_source,'probe',lambda _:dict(media_type='image',pix_fmt='rgba16f',
        is_hdr=True,color_transfer='linear',width=8,height=8))
    owners=[]
    class Resident:
        def __init__(self,original,settings):
            self.original=original;self.closed=False;self.calls=0
            owners.append(self)
        def render(self):
            assert not self.closed
            self.calls+=1
            return self.original.copy(),True,dict(stages=[dict(stage='dlaa')])
        def close(self):self.closed=True
    monkeypatch.setattr(image_render,'HdrImagePipeline',Resident)
    engine=GpuPreview()
    def forbidden(*a,**k):raise AssertionError('HDR still entered video decoder/P010 path')
    monkeypatch.setattr(engine,'_render',forbidden)
    try:
        settings=dict(render_pipeline=['dlaa'],dlaa_preset=12)
        engine._render_image(str(source),settings,'fast')
        engine._render_image(str(source),settings,'fast')
        assert len(owners)==1 and owners[0].calls==2 and not owners[0].closed
        result=engine._render_image(str(source),dict(settings,dlaa_preset=13),'fast')
        assert owners[0].closed and len(owners)==2 and result['hdr']
        assert len(engine.image_results.entries)==2
    finally:engine.shutdown()
    assert owners[-1].closed
