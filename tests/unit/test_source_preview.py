"""Source-only display must not acquire or resume an enhancement GPU session."""
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from dlss_service import DLSSService


def service(tmp_path):
    source = tmp_path / 'source.mkv'
    source.write_bytes(b'mock source')
    owner = DLSSService.__new__(DLSSService)
    owner._cap_lock = threading.Lock()
    owner.video_path = str(source)
    owner.video_info = dict(frames=100, fps=24, is_hdr=False)
    owner._preview_suspended = True
    owner.active_job_id = 'active video export'
    owner._cap = SimpleNamespace(set=lambda *_: True,
        read=lambda: (True, np.full((64, 128, 3), 100, np.uint8)))
    def forbidden(*_args, **_kwargs):
        pytest.fail('source display entered the GPU enhancement path')
    owner._get_gpu_preview = owner._render_gpu_preview = owner.resume_preview_memory = forbidden
    return owner


def test_suspended_source_displays_during_export_without_opening_gpu(tmp_path):
    owner = service(tmp_path)
    payload = owner.source_preview(owner.video_path, 3)
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (64, 128, 3)
    assert abs(float(image.mean())-100) < 2
    assert owner._preview_suspended and owner.active_job_id == 'active video export'


def test_source_cache_does_not_decode_again_and_stale_path_is_rejected(tmp_path):
    owner = service(tmp_path)
    payload = owner.source_preview(owner.video_path, 3)
    owner._cap.read = lambda: pytest.fail('cached frame was decoded again')
    assert owner.source_preview(owner.video_path, 3) == payload
    with pytest.raises(ValueError, match='已更换'):
        owner.source_preview('previous.mkv', 3)


def test_hdr_source_uses_display_mapping_and_small_cpu_decode(tmp_path, monkeypatch):
    import comparison_media
    owner = service(tmp_path)
    owner.video_info['is_hdr'] = True
    calls = []
    class Decoder:
        def _probe_metadata(self, path): return {'width':7680,'height':4320,'is_hdr':True}
        def _origin(self, *_): return 2.
        def _decode_linear(self, media, pts, edge):
            calls.append((media.path, pts, edge))
            return np.full((32, 64, 3), 12.5, np.float32)
    monkeypatch.setattr(comparison_media, 'MediaComparisonService', Decoder)
    payload = owner.source_preview(owner.video_path, 24, 1200)
    assert calls == [(owner.video_path, 3., 1200)]
    assert payload and owner._preview_suspended
