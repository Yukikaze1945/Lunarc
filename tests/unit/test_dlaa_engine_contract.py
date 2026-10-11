"""Offline contracts for the official equal-resolution DLAA wrapper."""
from pathlib import Path

import pytest

import dlaa_gpu_engine
import dlss_gpu_engine


def test_dlaa_selects_fp16_for_hdr_and_keeps_sdr8_contract(monkeypatch, tmp_path):
    dll = tmp_path / "nvngx_dlss.dll"
    dll.write_bytes(b"signed SDK placeholder for wrapper contract")
    opened = []

    def fake_init(self, width, height, settings=None, *, color_format="rgba8", **kwargs):
        opened.append((width, height, color_format))
        self._w, self._h, self.color_format = width, height, color_format

    monkeypatch.setattr(dlss_gpu_engine.Live, "__init__", fake_init)
    wide = dlaa_gpu_engine.Live(64, 36, dll_path=str(dll), is_hdr=True)
    assert wide.color_format == 'rgba16f'
    assert opened == [(64, 36, 'rgba16f')]

    live = dlaa_gpu_engine.Live(64, 36, dll_path=str(dll))
    assert opened == [(64, 36, 'rgba16f'), (64, 36, "rgba8")]
    assert live.dlss_dll == str(dll.resolve())


def test_dlaa_counter_wrapper_reports_native_lifecycle_counts():
    class CounterFunction:
        argtypes = None
        restype = None

        def __call__(self, values):
            values[0], values[1], values[2] = 2, 7, 2
            return 0

    class Library:
        dlss_gpu_host_dlaa_counters = CounterFunction()

    live = object.__new__(dlaa_gpu_engine.Live)
    live._lib = Library()
    live.render_preset = 11
    assert live.counters() == {
        "creates": 2,
        "successful_evaluates": 7,
        "release_calls": 2,
        "preset_requested": 11,
        "preset_effective": None,
    }


@pytest.mark.parametrize('shared', [False, True])
def test_fp16_open_uses_format_info_and_the_correct_anchor(shared):
    import ctypes as C
    from types import SimpleNamespace
    live = object.__new__(dlaa_gpu_engine.Live)
    live.bytes_per_pixel = 8
    live.color_format = 'rgba16f'
    live._w, live._h = 64, 40
    live.dlss_dll = 'verified.dll'
    live._handle = C.c_void_p()
    live._options = lambda: dlss_gpu_engine._Options()
    calls = []
    class Entry:
        def __call__(self, *args):
            calls.append(args)
            info = args[-2]._obj
            assert info.color_format == dlss_gpu_engine._COLOR_FORMAT['rgba16f']
            return 0
    entry = Entry()
    name = 'dlss_gpu_host_open_dlaa_layer_ex' if shared else 'dlss_gpu_host_open_dlaa_ex'
    live._lib = SimpleNamespace(**{name: entry})
    anchor = SimpleNamespace(_handle=C.c_void_p(99)) if shared else None
    assert live._open(anchor) == 0
    assert calls[0][0:3] == (anchor._handle,64,40) if shared else calls[0][0:2] == (64,40)


def test_native_dlaa_contract_uses_equal_resolution_and_official_quality_code():
    source = (Path(__file__).resolve().parents[2] / "native" / "dlss_gpu_host.cpp").read_text(
        encoding="utf-8")
    start = source.index("bool configure_create(")
    end = source.index("const char* widths[]", start)
    dlaa_branch = source[start:end]
    assert "NVSDK_NGX_Parameter_Width, width" in dlaa_branch
    assert "NVSDK_NGX_Parameter_Height, height" in dlaa_branch
    assert "NVSDK_NGX_Parameter_OutWidth, width" in dlaa_branch
    assert "NVSDK_NGX_Parameter_OutHeight, height" in dlaa_branch
    assert "NVSDK_NGX_PerfQuality_Value_DLAA" in dlaa_branch
    assert "dlss_gpu_host_open_dlaa" in source
    assert "dlss_gpu_host_dlaa_counters" in source
