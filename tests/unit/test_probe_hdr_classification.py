"""Source HDR classification must not confuse bit depth with transfer function."""
import json
from types import SimpleNamespace

import pytest
import pipeline


@pytest.mark.parametrize("pix_fmt,transfer,expected", [
    ("yuv420p", "bt709", False),
    ("yuv420p10le", "bt709", False),
    ("yuv420p12le", "bt709", False),
    ("p010le", "bt2020-10", False),
    ("yuv420p12le", "bt2020-12", False),
    ("yuv420p10le", "", False),
    ("yuv420p10le", "smpte2084", True),
    ("yuv420p10le", "smpte-st-2084", True),
    ("yuv420p10le", "arib-std-b67", True),
])
def test_probe_classifies_transfer_independently_of_bit_depth(monkeypatch, pix_fmt, transfer, expected):
    monkeypatch.setattr(pipeline, "_bundle_ffprobe", lambda: "ffprobe")
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs:
        SimpleNamespace(stdout=json.dumps({"streams": [{
            "pix_fmt": pix_fmt, "color_transfer": transfer,
            "color_primaries": "bt2020" if "2020" in transfer else "bt709",
        }]})))
    info = pipeline.probe_pixel_info("source.mkv")
    assert info["is_hdr"] is expected
    assert info["color_transfer"] == transfer
    assert info["bit_depth"] == (12 if "12" in pix_fmt else 10 if "10" in pix_fmt else 8)
