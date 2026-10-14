import pytest

from rtx_vsr_engine import RtxVsr, RtxVsrError


@pytest.mark.parametrize("level", [5, 6, 7, 20, 22, 24, -1, True])
def test_rejects_non_vendor_quality_levels_before_loading_native_runtime(level):
    with pytest.raises(ValueError, match="quality_level"):
        RtxVsr("missing.dll", "missing-runtime", 1, (1280, 720), (1920, 1080), level)


def test_hdr_without_packed10_is_explicitly_refused_before_loading_runtime():
    with pytest.raises(RtxVsrError, match="HDR VSR requires packed RGB10A2"):
        RtxVsr("missing.dll", "missing-runtime", 1, (1920, 1080), (3840, 2160), 2, is_hdr=True)


@pytest.mark.parametrize('strength', [-.1, 1.1, float('nan'), True])
def test_invalid_strength_is_refused_before_loading(strength):
    with pytest.raises(ValueError, match='strength'):
        RtxVsr('missing.dll','runtime',1,(640,360),(1280,720),2,strength=strength)


def test_denoise_deblur_cannot_upscale_before_loading():
    for level in range(8,16):
        with pytest.raises(ValueError, match='match input size'):
            RtxVsr('missing.dll','runtime',1,(640,360),(1280,720),level)
