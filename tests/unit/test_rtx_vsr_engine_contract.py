import pytest

from rtx_vsr_engine import RtxVsr, RtxVsrError


@pytest.mark.parametrize("level", [0, 5, -1])
def test_rejects_non_vendor_quality_levels_before_loading_native_runtime(level):
    with pytest.raises(ValueError, match="quality_level"):
        RtxVsr("missing.dll", "missing-runtime", 1, (1280, 720), (1920, 1080), level)


def test_hdr_without_packed10_is_explicitly_refused_before_loading_runtime():
    with pytest.raises(RtxVsrError, match="HDR VSR requires packed RGB10A2"):
        RtxVsr("missing.dll", "missing-runtime", 1, (1920, 1080), (3840, 2160), 2, is_hdr=True)
