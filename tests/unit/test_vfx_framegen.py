"""Offline interpolation timebase, borrowed-picture lifetime, and EOF policy."""
from types import SimpleNamespace as NS
import pytest
from vfx_framegen import Interpolator, config
import native_runtime_paths as paths


@pytest.mark.parametrize('settings', [dict(frame_generation='xess'), dict(frame_generation='vfg',frame_multiplier=True),
    dict(frame_generation='vfg',frame_multiplier=9), dict(frame_generation='vfg',framegen_quality=3),
    dict(frame_generation='vfg_dynamic',framegen_target_fps='nan'),
    dict(frame_generation='vfg_dynamic',framegen_target_fps=True),
    dict(frame_generation='vfg',framegen_scene_detection=1)])
def test_unknown_or_unsupported_choices_are_refused(settings):
    with pytest.raises(ValueError):
        config(settings)


@pytest.mark.parametrize('multiplier', range(2,9))
def test_real_intermediates_precede_current_frame_and_tail_is_reported_separately(multiplier):
    events, emitted = [], []
    stage = object.__new__(Interpolator)
    stage.multiplier, stage.fps, stage.size = multiplier, (25,1), (128,64)
    stage.previous_pts = stage.time_base = stage.cancel = None
    stage.sources = stage.generated = stage.emitted = stage.tail_holds = 0
    stage.recent_pts = []
    stage.input_codec = stage.output_codec = None
    stage.previous, stage.generated_ptr, stage.linear_output, stage.last_picture, stage.linear_row = 2,3,3,4,512
    stage.buffers = NS(copy=lambda *args: events.append(('copy', args)), call=lambda *args: None)
    stage.model = NS(process=lambda *args: events.append(('run',args[-2:])))
    color = dict(device_ptr=1,pitch_bytes=512,width=128,height=64)
    emit = lambda color, pts, kind: emitted.append((pts,kind,color['device_ptr']))
    stage.submit(color, NS(pts=0,time_base_num=1,time_base_den=1000),False,emit)
    stage.submit(color, NS(pts=40,time_base_num=1,time_base_den=1000),True,emit)
    stage.finish(emit)
    stage.finish(emit)
    assert len(emitted) == 2*multiplier
    assert [p for p,_,_ in emitted] == list(range(0,80*multiplier,40))
    assert stage.generated == multiplier-1 and stage.tail_holds == multiplier-1
    assert [kind for _,kind,_ in emitted] == ['original']+['generated']*(multiplier-1)+['original']+['tail_hold']*(multiplier-1)
    assert all(args == (i,True) for i,(_,args) in enumerate([e for e in events if e[0]=='run'],1))
    # Retain current input before any generated output can recycle its producer.
    assert emitted[0][2] == emitted[multiplier][2] == 4
    assert events[2][0] == 'copy' and events[3][0] == 'run'


def test_nonuniform_or_reversed_time_is_not_silently_converted():
    stage = object.__new__(Interpolator)
    stage.time_base, stage.previous_pts, stage.fps = (1,1000), 40, (25,1)
    stage.size, stage.linear_row, stage.last_picture = (128,64),512,4
    stage.buffers = NS(copy=lambda *args: None)
    stage.input_codec = None
    stage.cancel = None
    with pytest.raises(ValueError, match='uniform'):
        stage.submit(dict(device_ptr=1,pitch_bytes=512),NS(pts=100,time_base_num=1,time_base_den=1000),False,None)


def test_cancellation_before_submit_or_emit_does_not_touch_picture_or_consumer():
    import pipeline
    stage = object.__new__(Interpolator)
    stage.cancel = lambda: True
    with pytest.raises(pipeline.JobCancelled):
        stage.submit({},None,False,None)
    with pytest.raises(pipeline.JobCancelled):
        stage._emit({},0,'generated',None)


def test_split_sdk_discovery_and_missing_feature_are_explicit(tmp_path):
    for name in ['bin/NVVideoEffects.dll','bin/NVCVImage.dll','bin/nvngxruntime.dll','bin/nvrtc64_120_0.dll',
                 'features/nvvfxvideosuperres/bin/nvVFXVideoSuperRes.dll','features/nvvfxvideosuperres/include/nvVFXVideoSuperRes.h']:
        file = tmp_path/name
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_bytes(b'fixture')
    assert paths.wide_vfx_runtime(tmp_path) == str(tmp_path.resolve())
    with pytest.raises(FileNotFoundError,match='incomplete'):
        paths.wide_vfx_runtime(tmp_path,framegen=True)
    (tmp_path/'features/nvvfxvideosuperres/bin/nvVFXVideoSuperRes.dll').unlink()
    feature = tmp_path/'features/nvvfxvideoframegeneration/bin/nvVFXVideoFrameGeneration.dll'
    feature.parent.mkdir(parents=True)
    feature.write_bytes(b'fixture')
    assert paths.wide_vfx_runtime(tmp_path,framegen=True) == str(tmp_path.resolve())


@pytest.mark.parametrize('target,source,expected', [('60',(25,1),(60,1)),('60000/1001',(24000,1001),(60000,1001)),('59.94',(25,1),(2997,50))])
def test_dynamic_encoder_uses_exact_target_rate_and_reciprocal_timebase(target,source,expected):
    from vfx_framegen import encoder_timing
    settings = dict(frame_generation='vfg_dynamic',framegen_target_fps=target)
    assert config(settings) == (0,1)
    assert encoder_timing(settings,source,(1,1000)) == (expected,expected[::-1])
    with pytest.raises(ValueError,match='greater'):
        encoder_timing(settings,expected,(1,1000))


@pytest.mark.parametrize('scene_detection', [True,False])
def test_dynamic_25_to_60_uses_real_timestep_and_outputs_fifteen_ticks_for_six_sources(scene_detection):
    from fractions import Fraction
    phases, emitted = [], []
    stage = object.__new__(Interpolator)
    stage.multiplier, stage.dynamic, stage.fps, stage.target_rate = 0,True,(25,1),Fraction(60)
    stage.scene_detection,stage.size = scene_detection,(128,64)
    stage.previous_pts = stage.time_base = stage.cancel = None
    stage.sources = stage.generated = stage.emitted = stage.tail_holds = 0
    stage.next_tick = stage.origin_tick = 0
    stage.finished = False
    stage.recent_pts,stage.recent_timesteps = [],[]
    stage.input_codec = stage.output_codec = None
    stage.previous,stage.generated_ptr,stage.linear_output,stage.last_picture,stage.linear_row = 2,3,3,4,512
    stage.buffers = NS(copy=lambda *args: None,call=lambda *args: None)
    stage.model = NS(process_time=lambda *args: phases.append(args[-2:]))
    emit = lambda color,pts,kind: emitted.append((pts,kind,color['device_ptr']))
    for index in range(6):
        stage.submit(dict(device_ptr=1,pitch_bytes=512),NS(pts=1000+40*index,time_base_num=1,time_base_den=1000),True,emit)
    stage.finish(emit)
    stage.finish(emit)
    assert [p for p,_,_ in emitted] == list(range(60,75))
    assert len(phases) == stage.generated == 11
    assert stage.tail_holds == 2
    assert [p for p,k,_ in emitted if k=='original'] == [60,72]
    assert phases[0] == (Fraction(5,12),scene_detection)
    assert phases[1] == (Fraction(5,6),scene_detection)
    assert all(0 < p < 1 and reset == scene_detection for p,reset in phases)


def test_dynamic_nonuniform_source_is_refused_before_interpolation():
    stage = object.__new__(Interpolator)
    stage.time_base,stage.previous_pts,stage.fps = (1,1000),40,(25,1)
    stage.size,stage.linear_row,stage.last_picture = (128,64),512,4
    stage.buffers = NS(copy=lambda *args: None)
    stage.input_codec,stage.cancel = None,None
    with pytest.raises(ValueError,match='uniform'):
        stage.submit(dict(device_ptr=1,pitch_bytes=512),NS(pts=120,time_base_num=1,time_base_den=1000),False,None)
