"""Focused checks for the CPU PQ AVIF decoder."""

import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import hdr_image_decode as decoder
import pipeline


def _metadata(**overrides):
    info = {
        "pix_fmt": "yuv444p12le", "bit_depth": 12,
        "color_space": "bt2020nc", "color_transfer": "smpte2084",
        "color_primaries": "bt2020", "color_range": "pc",
        "width": 2, "height": 1, "is_hdr": True,
        "side_data_list": [{"side_data_type": "Content light level metadata",
                             "max_content": 590, "max_average": 65}],
    }
    info.update(overrides)
    return info


def test_probe_keeps_pipeline_metadata_and_only_admits_pq_avif(monkeypatch):
    info = _metadata()
    monkeypatch.setattr(pipeline, "probe_pixel_info", lambda path: info)
    assert decoder.probe_hdr("frame.AVIF") is info
    assert decoder.probe_hdr("frame.heic") is None
    monkeypatch.setattr(pipeline, "probe_pixel_info",
                        lambda path: _metadata(color_transfer="arib-std-b67"))
    assert decoder.probe_hdr("frame.avif") is None


def test_decode_reorders_gbra_and_preserves_signed_hdr_values(monkeypatch):
    # Planes are G, B, R, A; unequal sentinels expose accidental channel swaps.
    planar = np.array([[[2.0, 3.0]], [[-0.25, 0.5]],
                       [[1.5, 4.0]], [[1.0, 0.25]]], dtype="<f4")
    monkeypatch.setattr(pipeline, "_bundle_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(decoder.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=planar.tobytes(), stderr=b""))
    result = decoder.decode_hdr("frame.avif", _metadata())
    assert result.dtype == np.float16
    assert result.shape == (1, 2, 4)
    assert np.array_equal(result[0], np.array([[1.5, 2.0, -0.25, 1.0],
                                               [4.0, 3.0, 0.5, 0.25]], dtype=np.float16))


def test_decode_uses_explicit_pq_to_linear_scrgb_filter(monkeypatch):
    monkeypatch.setattr(pipeline, "_bundle_ffmpeg", lambda: "ffmpeg")
    seen = {}

    def run(command, **kwargs):
        seen["command"] = command
        return SimpleNamespace(returncode=0, stdout=np.ones((4, 1, 2), dtype="<f4").tobytes(), stderr=b"")

    monkeypatch.setattr(decoder.subprocess, "run", run)
    decoder.decode_hdr("frame.avif", _metadata())
    vf = seen["command"][seen["command"].index("-vf") + 1]
    for required in ("rangein=full", "matrixin=bt2020nc", "transferin=smpte2084",
                     "primariesin=bt2020", "primaries=bt709", "transfer=linear",
                     "npl=80", "gbrapf32le"):
        assert required in vf


@pytest.mark.parametrize("payload, message", [
    (b"short", "expected"),
    (np.array([[[np.nan, 0]], [[0, 0]], [[0, 0]], [[1, 1]]], dtype="<f4").tobytes(), "non-finite"),
    (np.array([[[0, 0]], [[0, 0]], [[0, 0]], [[1.1, 1]]], dtype="<f4").tobytes(), "alpha"),
])
def test_decode_rejects_malformed_component_data(monkeypatch, payload, message):
    monkeypatch.setattr(pipeline, "_bundle_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(decoder.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=payload, stderr=b""))
    with pytest.raises(ValueError if message != "expected" else RuntimeError, match=message):
        decoder.decode_hdr("frame.avif", _metadata())


def test_pq_reference_filter_maps_100_nits_to_1_25_scrgb():
    """Known ST 2084 100-nit sample checks zscale's absolute npl=80 scale."""
    ffmpeg = pipeline._bundle_ffmpeg()
    if not ffmpeg:
        pytest.skip("bundled FFmpeg unavailable")
    filters = subprocess.run([ffmpeg, "-hide_banner", "-filters"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, timeout=10)
    if "zscale" not in filters.stdout:
        pytest.skip("bundled FFmpeg has no zscale filter")

    # ST 2084 OETF for 100 cd/m² (L=0.01 relative to 10,000 cd/m²).
    m1, m2 = 2610 / 16384, 2523 / 32
    c1, c2, c3 = 3424 / 4096, 2413 / 128, 2392 / 128
    linear_pq = 0.01
    code = ((c1 + c2 * linear_pq**m1) / (1 + c3 * linear_pq**m1))**m2
    y = np.full((2, 2), round(code * 4095), dtype="<u2")
    chroma = np.full((2, 2), 2048, dtype="<u2")
    raw = y.tobytes() + chroma.tobytes() + chroma.tobytes()
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
        "-pixel_format", "yuv444p12le", "-video_size", "2x2", "-framerate", "1",
        "-color_range", "pc", "-colorspace", "bt2020nc", "-color_primaries", "bt2020",
        "-color_trc", "smpte2084", "-i", "pipe:0", "-frames:v", "1", "-vf",
        "zscale=rangein=full:matrixin=bt2020nc:primariesin=bt2020:transferin=smpte2084:"
        "range=full:matrix=gbr:primaries=bt709:transfer=linear:npl=80,format=gbrapf32le",
        "-f", "rawvideo", "-pix_fmt", "gbrapf32le", "pipe:1",
    ]
    result = subprocess.run(command, input=raw, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=30, check=False)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    planes = np.frombuffer(result.stdout, dtype="<f4").reshape(4, 2, 2)
    np.testing.assert_allclose(planes[:3], 1.25, atol=0.015, rtol=0)
    np.testing.assert_allclose(planes[3], 1.0, atol=0, rtol=0)
