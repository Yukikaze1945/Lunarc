"""CPU-only validation rules for native DLAA/VSR routing."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import native_effects  # noqa: E402
import pytest  # noqa: E402


def _settings(**updates):
    settings = {
        "neural_mode": "dlaa",
        "export_backend": "gpu_transfer",
        "reconstruction_mode": "single",
        "working_scale": 1.0,
        "guidance_mode": 0,
        "vsr_quality": 2,
        "vsr_target": "1080p",
    }
    settings.update(updates)
    return settings


def _source(**updates):
    source = {"pix_fmt": "yuv420p", "color_transfer": "bt709",
              "width": 640, "height": 360, "bit_depth": 8}
    source.update(updates)
    return source


@pytest.mark.parametrize("source", [
    _source(color_transfer="smpte2084"),
    _source(color_transfer="arib-std-b67"),
])
def test_native_effects_refuse_invalid_hdr_sources(source):
    with pytest.raises(ValueError, match="HLG|10-bit P010"):
        native_effects.validate(_settings(), source)


@pytest.mark.parametrize('format', ['yuv420p10le', 'p010le'])
def test_dlaa_accepts_sdr10_but_old_vsr_requires_a_verified_wide_interface(format, monkeypatch):
    source = _source(pix_fmt=format, bit_depth=10)
    native_effects.validate(_settings(), source)
    import native_runtime_paths
    def missing(*args, **kwargs):
        raise FileNotFoundError('VFX 1.3 is missing')
    monkeypatch.setattr(native_runtime_paths, 'wide_vfx_runtime', missing)
    with pytest.raises(FileNotFoundError, match='VFX 1.3'):
        native_effects.validate(_settings(neural_mode='vsr'), source)
    monkeypatch.setattr(native_runtime_paths, 'wide_vfx_runtime', lambda *a, **kw: 'sdk')
    native_effects.validate(_settings(neural_mode='vsr'), source)


@pytest.mark.parametrize("backend", ["auto", "legacy", ""])
def test_dlaa_requires_gpu_transfer(backend):
    with pytest.raises(ValueError, match="requires GPU Transfer"):
        native_effects.validate(_settings(export_backend=backend), _source())


@pytest.mark.parametrize("patch", [
    {"reconstruction_mode": "dual"},
    {"reconstruction_mode": "triple"},
    {"working_scale": 0.75},
])
def test_standalone_dlaa_refuses_multiscale_or_reduced_working_input(patch):
    with pytest.raises(ValueError, match="Single at 100%"):
        native_effects.validate(_settings(**patch), _source())


@pytest.mark.parametrize("quality", [0, 5, -1])
def test_rtx_vsr_refuses_quality_outside_the_supported_range(quality):
    with pytest.raises(ValueError, match="quality must be 1–4"):
        native_effects.validate(_settings(neural_mode="vsr", vsr_quality=quality),
                                _source())


def test_rtx_vsr_uses_the_same_standalone_scale_and_guidance_refusals():
    with pytest.raises(ValueError, match="Single at 100%"):
        native_effects.validate(_settings(neural_mode="vsr", reconstruction_mode="dual"),
                                _source())
    with pytest.raises(ValueError, match="does not consume optical-flow"):
        native_effects.validate(_settings(neural_mode="vsr", guidance_mode=1), _source())


def test_nr_remains_valid_on_wide_sources_without_native_effect_constraints():
    native_effects.validate(_settings(neural_mode="nr", export_backend="legacy",
                                      reconstruction_mode="triple", working_scale=0.5),
                            _source(pix_fmt="p010le", color_transfer="smpte2084",
                                    bit_depth=10))


def test_vsr_scale_and_custom_resolution_reuse_output_size():
    scale = _settings(neural_mode='vsr',vsr_output_mode='scale',vsr_scale=1.5)
    assert native_effects.output_size(scale,(1280,720)) == (1920,1080)
    custom = _settings(neural_mode='vsr',vsr_output_mode='resolution',vsr_target='custom',vsr_width=2560,vsr_height=1440)
    assert native_effects.output_size(custom,(1280,720)) == (2560,1440)
    for patch in [dict(vsr_scale=0),dict(vsr_scale=5),dict(vsr_scale=float('nan'))]:
        with pytest.raises(ValueError):
            native_effects.output_size(dict(scale,**patch),(1280,720))
    for size in [(1921,1080),(640,360),(2000,1080),(7680,4320)]:
        with pytest.raises(ValueError):
            native_effects.output_size(dict(custom,vsr_width=size[0],vsr_height=size[1]),(1280,720))
