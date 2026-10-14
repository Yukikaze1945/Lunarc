"""The new SDR colour domain reuses FP16/fusion/remux without becoming HDR."""
from copy import deepcopy

import pytest
import gpu_hdr_transfer_export as hp
import hdr_gpu_engine as engine
from test_hdr_transfer_export_branch import HDR_CAPS, HDR_DIMS, HDR_JOB, HDR_SOURCE, NO_MUX, Fp16Live


def facts():
    source = dict(HDR_SOURCE, color_transfer="bt709", color_primaries="bt709",
                  color_space="bt709", codec_name="av1")
    job = dict(HDR_JOB, experimental_hdr_scrgb=False, guidance_mode=0)
    caps = deepcopy(HDR_CAPS)
    caps["encoder_defaults"].update(color_trc=1, color_primaries=1, color_space=1)
    return source, job, caps


def gate(source=None, job=None, **tx):
    src, js, caps = facts()
    return hp.negotiate_sdr10(HDR_DIMS, source or src, job or js,
        dict(NO_MUX, **tx), 24, False, "out.mkv", Fp16Live(), hdr_caps=caps)


def test_sdr10_uses_relative_linear_and_sdr_encoder_declaration():
    result = gate()
    assert result["supported"], result["reasons"]
    assert result["hdr_policy"]["paper_white_scrgb"] == 1
    assert result["hdr_policy"]["nits_per_scrgb"] is None
    assert result["hdr_policy"]["clip_ceiling_scrgb"] == 1
    declaration = result["hdr_encoder_contract_declared"]
    assert [declaration[k] for k in ("color_range", "color_primaries", "color_trc", "color_space")] == [1]*4


@pytest.mark.parametrize("key,value", [("color_transfer", "smpte2084"),
    ("color_transfer", "arib-std-b67"), ("color_transfer", ""),
    ("color_primaries", "bt2020"), ("color_range", "pc"),
    ("color_space", "bt2020nc"), ("pix_fmt", "yuv420p")])
def test_sdr10_refuses_wrong_or_missing_colour(key, value):
    source, _, _ = facts()
    source[key] = value
    assert not gate(source=source)["supported"]


def test_hdr_and_sdr_requests_stay_separate():
    assert not gate(job=HDR_JOB)["supported"]
    source, job, caps = facts()
    assert not hp.negotiate_hdr(HDR_DIMS, source, dict(job, experimental_hdr_scrgb=True),
        NO_MUX, 24, False, "out.mkv", Fp16Live(), hdr_caps=caps)["supported"]


@pytest.mark.parametrize("tx", [{"codec": "h264_nvenc"}, {"pixel_format": "yuv420p"},
    {"profile": "main"}, {"color_preset": "hdr_to_sdr"}])
def test_sdr10_cannot_silently_downgrade(tx):
    assert not gate(**tx)["supported"]


def test_sdr_units_and_metadata_are_not_hdr_claims():
    report = hp.sdr10_report({"hdr_frames_processed_measured": 6,
        "experimental_hdr_scrgb": False, "hdr_policy": {"nits_per_scrgb": None},
        "hdr_not_claimed": {}, "hdr_source_metadata": {"has_mastering": 0}})
    assert report["experimental_hdr_scrgb"] is False
    assert report["sdr10_frames_processed_measured"] == 6
    assert report["sdr10_policy"]["nits_per_relative_linear"] is None
    assert report["sdr10_source_metadata"]["has_mastering"] == 0


def test_fixed_reference_white_is_enforced_before_native_open():
    with pytest.raises(engine.HdrBridgeError, match="fixed relative"):
        engine.GpuSdr10Bridge(paper_white_scrgb=2)
