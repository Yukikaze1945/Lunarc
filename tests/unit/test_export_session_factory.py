"""Queue session creation must preserve precision for every rendering order."""
from itertools import permutations
from types import SimpleNamespace

import pytest
import native_effects
from dlss_service import DLSSService


ORDERS = [list(order) for count in (1, 2, 3)
          for order in permutations(('nr', 'dlaa', 'vsr'), count)]


@pytest.mark.parametrize('wide', [False, True])
@pytest.mark.parametrize('settings', [dict(render_pipeline=order) for order in ORDERS]
                         + [dict(neural_mode=mode) for mode in native_effects.MODES])
def test_queue_factory_maps_precision_without_losing_order(monkeypatch, settings, wide):
    service = DLSSService()
    calls = []
    result = object()

    def nr(width, height, options, **kwargs):
        calls.append(('nr', width, height, options, kwargs))
        return result

    def native(width, height, options, *, wide=False):
        calls.append(('native', width, height, options, {'wide': wide}))
        return result

    monkeypatch.setattr(native_effects, 'make_live', native)
    monkeypatch.setattr(service, '_import_gpu_host', lambda: SimpleNamespace(Live=nr))
    assert service._create_export_live(960, 540, settings, wide=wide) is result
    if native_effects.mode(settings) == 'nr':
        assert calls == [('nr', 960, 540, settings, {'color_format': 'rgba16f'} if wide else {})]
    else:
        assert calls == [('native', 960, 540, settings, {'wide': wide})]
