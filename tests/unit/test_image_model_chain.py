import json
import pytest
import native_effects as effects
from image_model_store import ModelStore


def config():
    return dict(render_pipeline=['vsr','nr','vsr:second'],vsr_mode='standard',
        vsr_output_mode='scale',vsr_scale=2,
        render_stage_options={'vsr:second':{'vsr_scale':4}})


def test_independent_vsr_sizes_and_core_boundary():
    settings=config()
    assert effects.output_size(settings,(128,72))==(1024,576)
    assert effects.processing_size(settings,(128,72))==(256,144)
    assert effects.output_size(settings,(128,72),['vsr'])==(256,144)
    assert effects.stage_settings(settings,'vsr:second')['vsr_scale']==4
    assert effects.stage_settings(settings,'vsr')['vsr_scale']==2
    settings['render_stage_enabled']={'vsr':False}
    assert effects.output_size(settings,(128,72))==(512,288)
    assert effects.processing_size(settings,(128,72))==(128,72)


def test_remaining_instance_without_legacy_vsr():
    settings=config();settings['render_pipeline']=['vsr:second']
    assert effects.has_stage(settings,'vsr')
    assert effects.output_size(settings,(128,72))==(512,288)
    with pytest.raises(ValueError):effects.stages(dict(render_pipeline=['vsr:second','vsr:second']))
    with pytest.raises(ValueError):effects.stages(dict(render_pipeline=['dlaa:other']))
    with pytest.raises(ValueError):effects.stage_settings(dict(render_stage_options={'vsr':{'render_pipeline':[]}}),'vsr')


def test_image_models_geometry_and_bypass():
    settings=dict(render_pipeline=['animesharp','nr','illustration'])
    assert effects.output_size(settings,(128,72))==(1024,576)
    assert effects.processing_size(settings,(128,72))==(256,144)
    settings['render_stage_enabled']={'illustration':False}
    assert effects.output_size(settings,(128,72))==(256,144)


def test_model_import_hash_and_path_guard(tmp_path):
    store=ModelStore(tmp_path/'models')
    source=tmp_path/'author.safetensors';source.write_bytes(b'placeholder weights')
    store.import_model('animesharp',source)
    target=store.resolve('animesharp');target.write_bytes(b'changed')
    with pytest.raises(ValueError):store.resolve('animesharp')
    record=tmp_path/'models/animesharp.json'
    record.write_text(json.dumps(dict(file='../author.safetensors',sha256='irrelevant')))
    with pytest.raises(ValueError):store.resolve('animesharp')
