import pytest

from tests.helpers.sdr10_service_gate import validate_result


def _facts():
    job = {"precision_mode": "sdr10_bt709", "dlss_session_color_format": "rgba16f"}
    diag = {
        "export_backend_actual": "gpu_transfer",
        "transfer_mode_actual": "gpu_sdr10_transfer",
        "sdr10_frames_processed_measured": 4,
        "decode_host_copy_bytes_measured": 0,
        "dlss_host_color_stage_bytes_measured": 0,
        "dlss_output_readback_bytes_measured": 0,
        "encoder_host_pixel_bytes_measured": 0,
        "sdr10_diagnostic_image_readback_bytes_measured": 0,
        "sdr10_pixel_readback_bytes_measured": None,
    }
    streams = [{"codec_type": "video", "codec_name": "hevc", "profile": "Main 10",
                "pix_fmt": "yuv420p10le", "color_primaries": "bt709",
                "color_transfer": "bt709", "color_space": "bt709"},
               {"codec_type": "audio", "codec_name": "aac"}]
    frames = [{"best_effort_timestamp": i} for i in range(4)]
    source_audio = [{"size": "10", "data_hash": f"SHA256:{i:064x}"} for i in range(3)]
    return job, diag, streams, frames, source_audio, source_audio[:2]


def test_service_acceptance_checker_accepts_real_sdr10_contract_facts():
    facts = _facts()
    summary = validate_result(*facts, expected_frames=4)
    assert summary["precision_mode"] == "sdr10_bt709"
    assert summary["audio_packets_copied"] == 2
    assert summary["host_image_boundaries_zero"] is True


@pytest.mark.parametrize("field,value", [
    ("precision_mode", "sdr8"),
    ("dlss_session_color_format", "rgba8"),
])
def test_service_acceptance_checker_rejects_wrong_precision_session(field, value):
    job, diag, streams, frames, source_audio, output_audio = _facts()
    job[field] = value
    with pytest.raises(AssertionError):
        validate_result(job, diag, streams, frames, source_audio, output_audio, 4)


def test_service_acceptance_checker_rejects_audio_that_was_not_copied():
    job, diag, streams, frames, source_audio, output_audio = _facts()
    output_audio = [{"size": "11", "data_hash": "SHA256:" + "f" * 64}]
    with pytest.raises(AssertionError, match="copied source prefix"):
        validate_result(job, diag, streams, frames, source_audio, output_audio, 4)


def test_service_acceptance_checker_rejects_any_measured_host_image_copy():
    job, diag, streams, frames, source_audio, output_audio = _facts()
    diag["encoder_host_pixel_bytes_measured"] = 16
    with pytest.raises(AssertionError, match="host image boundary"):
        validate_result(job, diag, streams, frames, source_audio, output_audio, 4)
