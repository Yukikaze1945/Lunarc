"""Ordered VFG scheduling: every placement, temporal timestamps and borrows."""
from itertools import permutations
from types import SimpleNamespace as NS
import ctypes as C
import pytest
import native_effects as effects
import ordered_vfg
import native_runtime_paths


@pytest.mark.parametrize('order',list(permutations(('nr','dlaa','vsr','vfg'))))
def test_every_vfg_position_builds_exact_gpu_graph(monkeypatch,order):
    class Series:
        def __init__(self,live,settings,size,ordered,provider,**kwargs):
            self.order=list(ordered)
            self.target=effects.output_size(settings,size) if 'vsr' in ordered else tuple(size)
    monkeypatch.setattr(effects,'EffectSeries',Series)
    monkeypatch.setattr(ordered_vfg,'EffectSeries',Series)
    monkeypatch.setattr(native_runtime_paths,'wide_vfx_runtime',lambda *a,**k:'runtime')
    chain=NS(preview_consumer=None,process=lambda *a,**k:None,working=(640,360))
    live=NS(bytes_per_pixel=8)
    settings=dict(render_pipeline=list(order),frame_generation='vfg',frame_multiplier=2,
        vsr_output_mode='scale',vsr_scale=1.5)
    effects.install(chain,live,settings,(1280,720),None)
    fg=order.index('vfg')
    assert chain.framegen_stage_index==fg
    assert chain.framegen_stage_size==((1920,1080) if order.index('vsr')<fg else (1280,720))
    if order.index('nr')<fg:
        spatial=list(order[:fg]);suffix=getattr(chain,'framegen_tail',None)
        assert (suffix.order if suffix else [])==list(order[fg+1:])
    else:
        spatial=list(order[fg+1:]);prefix=chain.ordered_framegen.prefix
        assert (prefix.order if prefix else [])==list(order[:fg])
    nr=spatial.index('nr')
    assert getattr(chain,'input_effect',NS(order=[])).order==spatial[:nr]
    assert getattr(chain,'output_effect',NS(order=[])).order==spatial[nr+1:]
    assert chain.render_order==list(order)


class Frame(C.Structure):
    _fields_=[('width',C.c_uint32),('height',C.c_uint32),('pts',C.c_int64),
              ('time_base_num',C.c_int32),('time_base_den',C.c_int32)]


def test_generated_picture_enters_downstream_with_output_clock_and_independent_index():
    events=[]
    chain=NS()
    driver=object.__new__(ordered_vfg.BeforeReconstruction)
    driver.chain,driver.provider=chain,'flow'
    driver.frame=Frame(1280,720,40,1,1000)
    driver.tb,driver.reset,driver.emitted=(1,2000),True,0
    driver.core_process=lambda frame,provider,index,reset: events.append(
        (frame.pts,frame.time_base_den,index,reset,provider,chain._ordered_color['device_ptr']))
    for pts,kind in [(80,'original'),(120,'generated'),(160,'original'),(200,'tail_hold')]:
        driver.emit(dict(device_ptr=55),pts,kind)
        assert chain._ordered_color is None
    assert events==[(80,2000,0,True,'flow',55),(120,2000,1,False,'flow',55),
                    (160,2000,2,True,'flow',55),(200,2000,3,False,'flow',55)]
    assert driver.frame.pts==40 and driver.frame.time_base_den==1000


def test_failed_downstream_does_not_leave_a_recycled_picture_installed():
    driver=object.__new__(ordered_vfg.BeforeReconstruction)
    driver.chain,driver.provider=NS(),'flow'
    driver.frame,driver.tb,driver.reset,driver.emitted=Frame(1280,720,0,1,1000),(1,2000),False,0
    def fail(*args):raise RuntimeError('downstream failed')
    driver.core_process=fail
    with pytest.raises(RuntimeError,match='downstream failed'):
        driver.emit(dict(device_ptr=55),0,'original')
    assert driver.chain._ordered_color is None and driver.emitted==0


def test_vfg_only_owns_no_nr_or_dlaa_feature(monkeypatch):
    calls=[]
    monkeypatch.setattr(effects,'Passthrough',lambda w,h,s,owner,wide:calls.append((owner,wide)) or 'owner')
    assert effects.make_live(1280,720,dict(render_pipeline=['vfg'],frame_generation='vfg'),wide=True)=='owner'
    assert calls==[(None,True)]


def test_legacy_vfg_migrates_to_tail_once_and_duplicate_rejected():
    assert effects.stages(dict(render_pipeline=['nr','vsr'],frame_generation='vfg'))==['nr','vsr','vfg']
    assert effects.stages(dict(render_pipeline=['vfg','nr'],frame_generation='vfg'))==['vfg','nr']
    with pytest.raises(ValueError):effects.stages(dict(render_pipeline=['vfg','vfg']))

def test_passthrough_does_not_report_an_ngx_inference_call():
    from gpu_hdr_transfer_export import HdrChain
    chain=object.__new__(HdrChain)
    chain.uses_flow=False
    chain.zero_motion_calls=0
    live=NS(native_passthrough=True,process_device_color_zero_motion_resource=lambda color,reset:color)
    plane=NS(as_descriptor=lambda:dict(format='rgba16f_cuda',device_ptr=55))
    result,_=chain._guidance_and_inference(live,None,plane,None,NS(),0,True,True,{})
    assert result['device_ptr']==55 and chain.zero_motion_calls==0


def test_vsr_vfg_truehdr_nr_decodes_once_and_uses_the_enlarged_passthrough(monkeypatch):
    class Series:
        def __init__(self,live,settings,size,ordered,provider,**kwargs):
            self.order = list(ordered)
            self.target = effects.output_size(settings,size) if 'vsr' in ordered else tuple(size)
    monkeypatch.setattr(effects,'EffectSeries',Series)
    monkeypatch.setattr(ordered_vfg,'EffectSeries',Series)
    monkeypatch.setattr(native_runtime_paths,'wide_vfx_runtime',lambda *a,**k:'runtime')
    chain = NS(preview_consumer=None,process=lambda *a:None,working=(320,192),
        post_control=NS(update_controls=lambda _:None),temporal_control=NS(update_controls=lambda _:None))
    live = NS(bytes_per_pixel=4)
    settings = dict(render_pipeline=['vsr','vfg','truehdr','nr'],frame_generation='vfg',
        frame_multiplier=2,vsr_output_mode='scale',vsr_scale=2)
    effects.install(chain,live,settings,(320,192),None)
    assert chain.working == chain.processing_size == (640,384)
    assert (live._w,live._h) == (640,384)
    assert chain.ordered_framegen.prefix.order == ['vsr']
    assert chain.output_effect.order == ['truehdr','nr']
