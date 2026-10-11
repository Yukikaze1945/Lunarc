import numpy as np
import pytest
import optical_flow as flow


def test_invalid_cached_configuration_does_not_mutate_active_model(monkeypatch):
    from types import SimpleNamespace
    model = SimpleNamespace(analysis_edge=720, iterations=6)
    wrapper = SimpleNamespace(_engine=model)
    monkeypatch.setattr(flow, '_ENGINES', {('raft', 'Balanced', 'None', 'None', None, None): wrapper})
    with pytest.raises(ValueError):
        flow.get_optical_flow_engine('raft', 'Balanced', raft_analysis_edge=720, raft_iterations=0)
    assert model.analysis_edge == 720 and model.iterations == 6


def test_torch_context_restores_native_owner_when_computation_fails():
    from raft_gpu_flow import FlowProvider
    from types import SimpleNamespace
    calls = []
    provider = FlowProvider({})
    provider.context = 123
    provider.cuda = SimpleNamespace(
        cuCtxPushCurrent_v2=lambda context: calls.append(('push', context)) or 0,
        cuCtxPopCurrent_v2=lambda target: calls.append(('pop', None)) or 0)
    with pytest.raises(RuntimeError, match='inference'):
        with provider._torch_context():
            raise RuntimeError('inference')
    assert calls == [('push', 123), ('pop', None)]


def test_reconfigured_facade_shares_weights_without_history_or_weight_reload(monkeypatch):
    weights = object()
    monkeypatch.setattr(flow.RaftOpticalFlowEngine, '_init_raft',
        lambda self: pytest.fail('must not load weights again'))
    one = flow.RaftOpticalFlowEngine('Fast', 512, 4, shared_weights=weights)
    two = flow.RaftOpticalFlowEngine('Quality', 1080, 8, shared_weights=weights)
    one._prepared_tensor = object()
    assert one._raft_model is two._raft_model is weights
    assert two._prepared_tensor is None and two.analysis_edge == 1080 and two.iterations == 8


@pytest.mark.parametrize('size,edge,expected', [
    ((3840,2160),512,(512,288)), ((3840,2160),720,(720,408)),
    ((3840,2160),1080,(1080,608)), ((2160,3840),720,(408,720)),
    ((320,240),512,(320,240)), ((4096,64),512,(512,8)),
])
def test_long_edge_preserves_orientation_and_never_stretches_to_minimum(size, edge, expected):
    assert flow.raft_analysis_size(*size, edge) == expected


@pytest.mark.parametrize('edge,iterations', [(480,6),(512,0),(720,25)])
def test_invalid_raft_configuration_is_refused_before_loading_weights(edge, iterations):
    with pytest.raises(ValueError):
        flow.RaftOpticalFlowEngine('Balanced', edge, iterations)


def test_normalized_tensor_cache_is_content_safe_and_restores_each_vector_axis():
    torch = pytest.importorskip('torch')
    engine = flow.RaftOpticalFlowEngine.__new__(flow.RaftOpticalFlowEngine)
    engine.analysis_edge, engine.iterations, engine._device = 512, 4, 'cpu'
    engine._prepared_source = engine._prepared_tensor = None
    engine._infer = lambda cur, prev, size: torch.ones((1,2,size[1],size[0]))
    before = np.zeros((1080,1920,3), np.uint8)
    current = np.full_like(before, 70)
    result = engine.compute_flow(before, current)
    assert np.allclose(result[...,0],1920/512)
    assert np.allclose(result[...,1],1080/288)
    engine.compute_flow(current, before)
    assert engine.get_last_timing()['previous_tensor_reused']
    before[:] = 33  # Same allocation, changed contents; cannot reuse old tensor.
    engine.compute_flow(before, current)
    assert not engine.get_last_timing()['previous_tensor_reused']
    engine.compute_flow(current, before, reset_temporal_hints=True)
    assert not engine.get_last_timing()['previous_tensor_reused']
