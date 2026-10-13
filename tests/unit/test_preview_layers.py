"""Offline regression coverage for independent parameters and NR reuse."""
from types import SimpleNamespace
import ctypes as C
import pytest

import gpu_preview_engine as preview_module
import gpu_hdr_transfer_export as htx
from layer_settings import resolve_layer_settings, layer_parameters
from preview_layer_cache import evaluate
from fusion_runtime import update_parameters


def settings():
    return dict(render_pipeline=['nr'], reconstruction_mode='triple',
                working_scale=.4, layer_scales=[.4, .67, .85], layer_weights=[1., 1., 1.],
                fusion_mode='data_consistency_experimental',
                **layer_parameters({}), layer_settings=[{}, {'intensity': .25}, {'intensity': 1.5}])


def test_nr_overrides_do_not_replace_global_colour_or_other_layers():
    s = settings()
    s['post_controls'] = {'chroma_gain': .7}
    assert [resolve_layer_settings(s, i)['intensity'] for i in range(3)] == [1., .25, 1.5]
    assert all(resolve_layer_settings(s, i)['post_controls'] == s['post_controls'] for i in range(3))
    with pytest.raises(ValueError):
        resolve_layer_settings(dict(s, layer_settings=[{}, {'color_format': 'rgba8'}]), 1)
    with pytest.raises(ValueError):
        resolve_layer_settings(dict(s, layer_settings=[{}, {'intensity': float('nan')}]), 1)


def test_wide_layer_factory_receives_independent_params(monkeypatch):
    made = []
    def live(w, h, resolved, **kwargs):
        made.append((w, h, resolved, kwargs))
        return SimpleNamespace(settings=resolved, close=lambda: None)
    monkeypatch.setattr('dlss_gpu_engine.Live', live)
    anchor = SimpleNamespace(settings=settings())
    config = {'scales': [.4, .67, .85], 'dims': [(40, 20), (68, 34), (86, 44)]}
    layers = htx._open_extra_layers(config, anchor, None, False, 100, 50)
    assert len(layers) == 2
    assert [item[2]['intensity'] for item in made] == [.25, 1.5]
    assert all(item[3] == dict(shared_with=anchor, color_format='rgba16f') for item in made)


def test_editing_second_layer_updates_only_its_owner():
    preview = preview_module.GpuPreview()
    calls = []
    def owner(i):
        def update(s):
            calls.append((i, s['intensity']))
            return True
        return SimpleNamespace(update_runtime_options=update)
    preview.live = owner(0)
    preview.multi = {'scales': [.4, .67, .85]}
    preview.chain = SimpleNamespace(extra_layers=[{'live': owner(1)}, {'live': owner(2)}])
    old = settings()
    preview.active_settings = old
    changed = dict(old, layer_settings=[{}, {'intensity': .6}, {'intensity': 1.5}])
    try:
        assert preview._update_runtime(changed)
        assert calls == [(1, .6)]
        assert preview.runtime_updated_layers == {1}
        assert not preview.post_only_update
    finally:
        preview.chain = preview.live = None
        preview.shutdown()


def test_paused_layer_reuse_does_not_infer_other_sessions_and_export_does_not_cache():
    chain = SimpleNamespace(_preview_nr_cache={}, _preview_reuse_layers=set())
    calls = []
    def infer(i):
        calls.append(i)
        return {'session': i, 'value': len(calls)}
    for i in range(3):
        evaluate(chain, i, lambda: infer(i))
    chain._preview_reuse_layers = {0, 2}
    reused = [evaluate(chain, i, lambda: infer(i))[1] for i in range(3)]
    assert reused == [True, False, True] and calls == [0, 1, 2, 1]
    # A new frame clears all borrowed outputs before any Evaluate.
    chain._preview_nr_cache.clear()
    assert not evaluate(chain, 0, lambda: infer(0))[1]
    export = SimpleNamespace()
    assert not evaluate(export, 0, lambda: infer(0))[1]
    assert not evaluate(export, 0, lambda: infer(0))[1]


def test_scalar_solver_update_preserves_handle_and_forwards_all_parameters():
    calls = []
    class Fn:
        def __call__(self, *args):
            calls.append(args)
            return 0
    lib = SimpleNamespace(gpu_fusion_update_parameters=Fn())
    config = dict(weights=[.5, 2., 1.], beta=.2, relative_tolerance=.001, max_iterations=64)
    handle = C.c_void_p(123)
    assert update_parameters(lib, handle, config, .75)
    args = calls[0]
    assert args[0] is handle and list(args[1]) == [.5, 2., 1.]
    assert args[2:] == (3, .2, .75, .001, 64)
    assert not update_parameters(SimpleNamespace(), handle, config, .75)
