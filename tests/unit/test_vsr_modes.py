import pytest
from types import SimpleNamespace as NS
import native_effects
from vsr_config import configuration, validate_sdk_options


@pytest.mark.parametrize('family,levels', [('standard',[1,2,3,4]),('denoise',[8,9,10,11]),
    ('deblur',[12,13,14,15]),('high_bitrate',[16,17,18,19])])
def test_family_mapping(family,levels):
    for quality,level in enumerate(levels,1):
        assert configuration(dict(vsr_mode=family,vsr_quality=quality))['sdk_level']==level
        validate_sdk_options(level,None,(640,360),(640,360) if family in ('denoise','deblur') else (1280,720))


def test_streaming_and_bicubic_are_exact_official_modes():
    assert configuration(dict(vsr_mode='streaming',vsr_quality=2))['sdk_level']==21
    assert configuration(dict(vsr_mode='streaming',vsr_quality=4))['sdk_level']==23
    for q in (1,3):
        with pytest.raises(ValueError): configuration(dict(vsr_mode='streaming',vsr_quality=q))
    assert configuration(dict(vsr_mode='bicubic'))['sdk_level']==0
    assert configuration({})==configuration({'vsr_mode':'standard','vsr_quality':2,'vsr_strength':None})


def test_same_size_mode_preserves_saved_upscale_selection():
    original=dict(render_pipeline=['vsr'],vsr_mode='denoise',vsr_output_mode='scale',vsr_scale=4)
    assert native_effects.output_size(original,(640,360))==(640,360)
    assert original['vsr_scale']==4
    assert native_effects.output_size(dict(original,vsr_mode='standard'),(640,360))==(2560,1440)


def test_same_size_mode_change_releases_only_vsr_model():
    owner=native_effects.OutputEffect.__new__(native_effects.OutputEffect)
    owner.selected='vsr';owner.source_size=(640,360);owner.target=(1280,720)
    closed=[];owner.effect=NS(close=lambda:closed.append('vsr'))
    owner.input=object();owner.output=object();resources=(owner.input,owner.output)
    settings=dict(render_pipeline=['vsr'],vsr_mode='high_bitrate',vsr_output_mode='scale',vsr_scale=2)
    assert owner.update_vsr(settings) and closed==['vsr'] and owner.effect is None
    assert (owner.input,owner.output)==resources
    assert not owner.update_vsr(dict(settings,vsr_mode='denoise'))


def test_ordered_vsr_stage_is_found_by_preview_update():
    updates=[]
    owner=NS(selected='vsr',source_size=(640,360),target=(1280,720),
             update_vsr=lambda settings: updates.append(settings) or True)
    chain=NS(output_effect=NS(effects=[owner]))
    settings=dict(render_pipeline=['vsr'],vsr_mode='streaming',vsr_quality=2,vsr_output_mode='scale',vsr_scale=2)
    assert native_effects.update_vsr_options(chain,settings) and updates==[settings]


@pytest.mark.parametrize('fails', [False, True])
def test_decoder_reuse_is_adopted_once_or_released_on_rebuild_failure(fails):
    from gpu_preview_engine import GpuPreview
    events=[]
    preview=GpuPreview()
    old_chain=NS(decoder=123)
    preview.chain=old_chain
    preview.bridges={'nvdec':NS(nvdec_bridge_close=lambda handle:events.append(('decoder',handle)))}
    preview.source={'width':640,'height':360};preview.source_origin=0
    original_close=preview._close
    def close():
        assert old_chain.decoder is None
        events.append('close old effects')
    def open(video,settings,decoder_reuse):
        assert decoder_reuse['handle']==123
        if fails: raise ValueError('new stage failed')
        decoder_reuse['handle']=None
        events.append('adopt decoder')
    preview._close=close;preview._open=open
    try:
        if fails:
            with pytest.raises(ValueError):preview._reopen_vsr_with_decoder('video',{})
            assert events==['close old effects',('decoder',123)]
        else:
            preview._reopen_vsr_with_decoder('video',{})
            assert events==['close old effects','adopt decoder']
    finally:
        preview._close=original_close;preview.chain=preview.bridges=None;preview.shutdown()


@pytest.mark.parametrize('patch', [{'vsr_mode':'unknown'}, {'vsr_mode':'streaming','vsr_quality':1},
    {'vsr_strength':float('nan')}, {'vsr_strength':True}])
def test_settings_api_refuses_bad_modes_before_persisting(patch):
    from dlss_service import DLSSService
    service=DLSSService.__new__(DLSSService)
    service.settings=dict(vsr_mode='standard',vsr_quality=2,vsr_strength=None)
    before=dict(service.settings)
    with pytest.raises(ValueError):service._update_settings_locked(patch)
    assert service.settings==before
