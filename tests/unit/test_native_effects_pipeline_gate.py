import numpy as np
import pytest

from tests.helpers.native_effects_pipeline_gate import _require_picture


def test_quality_gate_rejects_black_or_flat_first_frame():
    with pytest.raises(AssertionError, match="black or lacks source detail"):
        _require_picture("first VSR frame", np.zeros((64, 64, 3), dtype=np.uint8))
    flat = np.full((64, 64, 3), 96, dtype=np.uint8)
    with pytest.raises(AssertionError, match="black or lacks source detail"):
        _require_picture("first VSR frame", flat)


def test_quality_gate_accepts_a_textured_source_frame():
    pattern = np.indices((64, 64)).sum(axis=0).astype(np.uint8) * 4
    image = np.repeat(pattern[:, :, None], 3, axis=2)
    metrics = _require_picture("first VSR frame", image)
    assert metrics["std"] > 5
    assert metrics["max"] >= 32
