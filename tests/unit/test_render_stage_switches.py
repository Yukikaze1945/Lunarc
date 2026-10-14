from copy import deepcopy

import pytest
import native_effects as effects


def settings():
    return dict(render_pipeline=['dlaa','nr','truehdr','vsr','vfg'],
                render_stage_enabled={'nr':False,'truehdr':False,'vsr':False,'vfg':False},
                truehdr_enabled=True,frame_generation='vfg_dynamic',working_scale=.5,
                reconstruction_mode='triple',layer_scales=[.4,.67,.85],intensity=1.7,
                vsr_scale=3,framegen_target_fps=120,guidance_mode=1,
                export_backend='gpu_transfer')


def test_disabled_stages_keep_configuration_but_are_not_executed():
    original=settings();before=deepcopy(original)
    assert effects.configured_stages(original)==['dlaa','nr','truehdr','vsr','vfg']
    assert effects.stages(original)==['dlaa']
    runtime=effects.execution_settings(original)
    assert effects.stages(runtime)==['dlaa']
    assert runtime['frame_generation']=='off' and not runtime['truehdr_enabled']
    assert runtime['working_scale']==1 and runtime['reconstruction_mode']=='single'
    assert runtime['guidance_mode']==1 and original==before
    original['render_stage_enabled']['nr']=True
    restored=effects.execution_settings(original)
    assert restored['working_scale']==.5 and restored['reconstruction_mode']=='triple'
    assert restored['intensity']==1.7


def test_all_off_opens_passthrough_and_not_a_neural_feature(monkeypatch):
    original=settings();original['render_stage_enabled']['dlaa']=False
    runtime=effects.execution_settings(original)
    assert effects.stages(runtime)==[] and runtime['guidance_mode']==0
    marker=object()
    monkeypatch.setattr(effects,'Passthrough',lambda *args,**kw:marker)
    assert effects.make_live(1280,720,original,wide=True) is marker


def test_disabled_vsr_preserves_source_extent_and_skips_invalid_parameters():
    original=settings()
    original['render_stage_enabled']['dlaa']=False
    original['vsr_scale']=-10
    assert effects.output_size(original,(1280,720))==(1280,720)
    effects.validate(original,dict(pix_fmt='yuv420p10le',width=1280,height=720,color_transfer='bt709'))


def test_service_keeps_disabled_cards_and_restores_vfg_mode():
    from dlss_service import DLSSService
    service=DLSSService()
    service.update_settings(dict(render_pipeline=['nr','truehdr','vfg'],frame_generation='vfg_dynamic'))
    service.update_settings(dict(render_stage_enabled={'truehdr':False,'vfg':False}))
    assert service.settings['render_pipeline']==['nr','truehdr','vfg']
    assert not service.settings['truehdr_enabled'] and service.settings['frame_generation']=='off'
    service.update_settings(dict(render_stage_enabled={'truehdr':True,'vfg':True}))
    assert service.settings['truehdr_enabled'] and service.settings['frame_generation']=='vfg_dynamic'


@pytest.mark.parametrize('switches',[{'nr':'false'},{'unknown':False},[],{'nr':0}])
def test_bad_switches_fail_instead_of_silently_enabling(switches):
    with pytest.raises(ValueError):effects.stages(dict(render_pipeline=['nr'],render_stage_enabled=switches))
