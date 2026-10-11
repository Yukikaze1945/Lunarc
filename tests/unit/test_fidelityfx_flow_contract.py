from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fidelityfx_flow_engine", ROOT / "fidelityfx_flow_engine.py")
ENGINE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = ENGINE
SPEC.loader.exec_module(ENGINE)


def test_flow_contract_exposes_full_resolution_borrowed_cuda_float2():
    assert ENGINE.FlowOutput.__dataclass_fields__["format"].default == "float32x2"
    assert ENGINE.FlowOutput.__dataclass_fields__["units"].default == "input_pixels"
    assert ENGINE.FlowOutput.__dataclass_fields__["direction"].default == "current_to_previous"
    names = dict(ENGINE._Counters._fields_)
    assert names["output_pitch_bytes"] is ctypes.c_uint32
    assert names["flow_format_float32x2"] is ctypes.c_uint32
    assert names["scd_scene_change_bits"] is ctypes.c_uint64


@pytest.mark.parametrize('quality', ['Performance', 'Quality', '1F'])
def test_layer_clones_keep_selected_quality_and_independent_history(quality):
    parent = ENGINE.FlowProvider(123, quality)
    child = parent.new_independent_layer()
    assert child is not parent
    assert child.quality == ('Quality' if quality == '1F' else quality)
    child.frames_since_reset = 4
    assert parent.frames_since_reset == 0


def test_unsupported_amd_quality_is_rejected_without_fallback():
    with pytest.raises(ValueError, match='Performance or Quality'):
        ENGINE.FlowProvider(123, 'Balanced')


def test_missing_fidelityfx_adapter_is_reported_without_creating_sessions(monkeypatch):
    import bundle_paths

    def missing(*args):
        raise RuntimeError('FidelityFX bridge missing')

    monkeypatch.setattr(bundle_paths, 'require_native', missing)
    assert ENGINE.probe_bridge() == {
        'available': False, 'error': 'FidelityFX bridge missing',
        'availability_scope': 'adapter ABI; actual device admission occurs per job'}


def test_repeated_capability_refresh_reuses_library_without_gpu_open(monkeypatch, tmp_path):
    import bundle_paths

    bridge = tmp_path / 'flow.dll'
    bridge.touch()
    monkeypatch.setattr(bundle_paths, 'require_native', lambda *args: str(bridge))
    monkeypatch.setattr(ENGINE, '_PROBE_LIBRARIES', {})
    loaded = []

    class Version:
        restype = None
        def __call__(self):
            return 1

    class Library:
        ffx_flow_abi_version = Version()
        # No GPU entry points: calling one would fail this capability-only test.

    def load(path):
        loaded.append(path)
        return Library()

    monkeypatch.setattr(ENGINE.ctypes, 'WinDLL', load)
    assert ENGINE.probe_bridge()['available'] is True
    assert ENGINE.probe_bridge()['available'] is True
    assert loaded == [str(bridge.resolve())]


def test_native_header_and_implementation_keep_contract_and_reset_dispatch():
    header = (ROOT / "native" / "fidelityfx_flow_bridge.h").read_text(encoding="utf-8")
    cpp = (ROOT / "native" / "fidelityfx_flow_bridge.cpp").read_text(encoding="utf-8")
    assert "uint64_t input_cuda_rgba8" in header
    assert "uint64_t* out_cuda_float32x2" in header
    assert "dd.reset=reset!=0" in cpp
    assert "opticalFlowVector" in (ROOT / "native/vendor/fidelityfx/sdk/include/FidelityFX/host/ffx_opticalflow.h").read_text(encoding="utf-8")
    assert "cuMemcpyDtoH" not in cpp and "cuMemcpyHtoD" not in cpp
