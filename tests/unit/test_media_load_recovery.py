from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import pipeline
from dlss_service import DLSSService


def test_rejected_drop_preserves_loaded_capture(tmp_path, monkeypatch):
    bad = tmp_path / 'broken.png'
    bad.write_bytes(b'not an image')
    svc = DLSSService()
    previous = Mock()
    svc._cap = previous
    svc.video_path = 'previous.mkv'
    svc.video_info = {'filename': 'previous.mkv'}
    svc._preview_suspended = False
    rejected = Mock()
    rejected.isOpened.return_value = False
    monkeypatch.setattr(pipeline, 'open_capture', lambda path: rejected)
    with pytest.raises(RuntimeError, match='Failed to open'):
        svc.load_video(str(bad))
    assert svc._cap is previous
    assert svc.video_path == 'previous.mkv'
    previous.release.assert_not_called()
    assert not svc.preview_memory_state()['suspended']
    rejected.release.assert_called_once()


def test_unicode_still_loads_and_previews_without_nvdec(tmp_path):
    image = tmp_path / '图片.png'
    pipeline.imwrite_u(str(image), np.full((144, 256, 3), 100, np.uint8))
    svc = DLSSService()
    try:
        info = svc.load_video(str(image))
        assert info['frames'] == 1
        assert (info['width'], info['height']) == (256, 144)
        assert svc.preview_memory_state()['suspended']
        assert svc.render_preview(0, view='orig') == (None, 'GPU_PREVIEW_RELEASED')
        svc.resume_preview_memory()
        # Original view checks actual decoding/display without needing a GPU.
        jpeg, status = svc.render_preview(0, view='orig')
        assert status == 'ok'
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (144, 256, 3)
        assert abs(float(decoded.mean()) - 100) < 2
        svc.load_video(str(image))
        assert svc.render_preview(0, view='orig') == (None, 'GPU_PREVIEW_RELEASED')
    finally:
        if svc._cap: svc._cap.release()
