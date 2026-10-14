import pytest
from truehdr_engine import parameters
from native_effects import stages


def test_truehdr_controls_use_official_ranges_and_invert_debanding_flag():
    assert parameters({}) == (100,100,50,650,0)
    assert parameters({'truehdr_debanding':False})[-1] == 1
    for key,value in [('truehdr_contrast',201),('truehdr_saturation',-1),
                      ('truehdr_middle_gray',9),('truehdr_peak_nits',2001)]:
        with pytest.raises(ValueError): parameters({key:value})


def test_truehdr_can_be_the_only_stage_and_stays_after_other_processing():
    assert stages(dict(render_pipeline=[],truehdr_enabled=True)) == ['truehdr']
    assert stages(dict(render_pipeline=['dlaa','nr'],truehdr_enabled=True)) == ['dlaa','nr','truehdr']


def test_explicit_truehdr_retains_position_without_legacy_double_append():
    assert stages(dict(render_pipeline=['dlaa','truehdr','nr'],truehdr_enabled=True)) == ['dlaa','truehdr','nr']
    assert stages(dict(render_pipeline=['truehdr'])) == ['truehdr']
    with pytest.raises(ValueError): stages(dict(render_pipeline=['truehdr','truehdr']))


def test_nr_after_conversion_uses_its_own_layer_config_not_the_sdr_decoder_core():
    from gpu_transfer_export import multi_scale_config
    from native_effects import nr_effect_settings, nr_as_effect, processing_size
    settings = dict(render_pipeline=['truehdr','nr'], working_scale=.5, reconstruction_mode='triple',
                    layer_scales=[.5,.75,1.], fusion_mode='sequential')
    assert nr_as_effect(settings)
    assert multi_scale_config(settings, 320, 192)['dims'] == [(320,192)]
    assert multi_scale_config(nr_effect_settings(settings), 320, 192)['dims'] == [(160,96),(240,144),(320,192)]
    assert processing_size(dict(settings, render_pipeline=['vsr','truehdr','nr']), (320,192)) == (320,192)


def test_ordered_encoder_refuses_a_lost_hdr_descriptor_instead_of_converting_twice():
    from types import SimpleNamespace
    from truehdr_pipeline import Tail
    tail = Tail(SimpleNamespace(), {}, ordered=True)
    with pytest.raises(RuntimeError, match='lost its HDR'):
        tail.process(dict(format='rgba16f_cuda'), None)
    picture = dict(output_hdr=True)
    assert tail.process(picture, None) is picture


def test_primary_feature_closes_before_its_borrowed_motion_provider():
    from types import SimpleNamespace
    from native_effects import Passthrough
    events = []
    live = object.__new__(Passthrough)
    live.exchange = None
    live.device = True
    live.owner = SimpleNamespace(close=lambda: events.append('feature'))
    live.stage_providers = [SimpleNamespace(close_independent_layer=lambda: events.append('flow'))]
    live.close()
    live.close()
    assert events == ['feature','flow']


def test_sdr10_diagnostics_preserve_the_truehdr_output_domain():
    from gpu_hdr_transfer_export import sdr10_report
    report = sdr10_report(dict(truehdr={'effect':{'stage':'truehdr'}}, sdr10_not_claimed={}))
    assert 'TrueHDR HDR10' in report['sdr10_not_claimed']['formal_hdr']
    assert 'PQ / BT.2020' in report['sdr10_not_claimed']['domains']


def test_inline_hdr_nr_parameter_edit_updates_only_the_selected_session(monkeypatch):
    from types import SimpleNamespace as NS
    import native_effects
    from gpu_preview_engine import GpuPreview
    updates = []
    owners = [NS(update_runtime_options=lambda settings, index=i:
                 updates.append((index, settings['intensity'])) or True) for i in range(3)]
    stage = NS(owner=owners[0], multi={'scales':[.5,.75,1.]},
               core=NS(extra_layers=[{'live':owner} for owner in owners[1:]]), update=lambda _:None)
    monkeypatch.setattr(native_effects, 'inline_nr_stage', lambda _:stage)
    preview = GpuPreview()
    old = dict(render_pipeline=['truehdr','nr'], working_scale=.5, reconstruction_mode='triple',
               layer_scales=[.5,.75,1.], fusion_mode='sequential', intensity=1., layer_settings=[{},{},{}])
    preview.active_settings = old
    preview.live = NS()
    preview.multi = {'scales':[1.]}
    preview.chain = NS(post_control=NS(update_controls=lambda _:None), temporal_control=NS(controls=None, update_controls=lambda _:None))
    try:
        assert preview._update_runtime(dict(old, layer_settings=[{}, {'intensity':.25}, {}]))
        assert updates == [(1,.25)]
        assert preview.runtime_updated_layers == {1} and not preview.post_only_update
    finally:
        preview.chain = preview.live = None
        preview.shutdown()
