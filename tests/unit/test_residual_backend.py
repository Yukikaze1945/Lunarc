"""Section 4/17 contract: one residual backend, chosen in exactly one place.

Preview, export and the benchmarks must obtain their working-color filter and residual
formula from create_residual_backend(), and what a run reports has to be what actually
executed. Requesting "native" before the bridge exists must fail loudly rather than
quietly running the reference and calling it native.
"""

import os
import sys

import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline  # noqa: E402
from resampler_oracle import quantized, scalar_lanczos2_downsample  # noqa: E402


def _ramp(h, w):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :, 0] = (np.arange(h * w) % 251).reshape(h, w)
    a[:, :, 1] = (np.arange(h * w) // 3 % 251).reshape(h, w)
    a[:, :, 2] = (np.arange(h * w) * 7 % 251).reshape(h, w)
    return a


def test_factory_returns_python_reference_when_requested():
    backend = pipeline.create_residual_backend("python", 16, 12, 8, 6)
    assert backend.name == "cpu_reference_python"
    info = {}
    assert pipeline.create_residual_backend("cpu_reference_python", 16, 12, 8, 6,
                                            diagnostics_sink=info) is not None
    assert info["residual_backend_requested"] == "cpu_reference_python"
    assert info["residual_backend_actual"] == "cpu_reference_python"
    assert info["residual_backend_fallback_reason"] == "requested_python"


def test_factory_reports_the_reference_honestly_when_native_is_missing(monkeypatch):
    monkeypatch.setattr(pipeline, "_load_native_backend", lambda *a: None)
    info = {}
    backend = pipeline.create_residual_backend("auto", 16, 12, 8, 6, diagnostics_sink=info)
    assert backend.name == "cpu_reference_python"
    assert info["residual_backend_actual"] == "cpu_reference_python"
    assert info["residual_backend_fallback_reason"] == "native_load_failed"


def test_factory_refuses_to_pretend_native_when_requested(monkeypatch):
    monkeypatch.setattr(pipeline, "_load_native_backend", lambda *a: None)
    with pytest.raises(RuntimeError):
        pipeline.create_residual_backend("native", 16, 12, 8, 6)
    with pytest.raises(RuntimeError):
        pipeline.create_residual_backend("cpu_reference_native", 16, 12, 8, 6)


def test_factory_reports_native_when_the_bridge_loads(monkeypatch):
    class FakeNative(pipeline.ResidualBackend):
        name = "cpu_reference_native"
        version = "fake-1"

    monkeypatch.setattr(pipeline, "_load_native_backend", lambda *a: FakeNative())
    info = {}
    backend = pipeline.create_residual_backend("auto", 16, 12, 8, 6, diagnostics_sink=info)
    assert backend.name == "cpu_reference_native"
    assert info["residual_backend_actual"] == "cpu_reference_native"
    assert info["residual_backend_fallback_reason"] == "native_available"


def test_python_backend_working_color_matches_the_oracle():
    backend = pipeline.PythonReferenceBackend(16, 12, 8, 6)
    src = _ramp(12, 16)
    dst = np.zeros((6, 8, 3), dtype=np.uint8)
    got = backend.prepare_working_color(src, dst)

    assert got is dst, "prepare_working_color must fill the caller's buffer"
    expected = quantized(scalar_lanczos2_downsample(src, 8, 6))
    assert np.abs(got.astype(np.float64) - expected.astype(np.float64)).max() <= 1.0


def test_python_backend_rejects_the_wrong_source_size():
    backend = pipeline.PythonReferenceBackend(16, 12, 8, 6)
    with pytest.raises(ValueError):
        backend.prepare_working_color(_ramp(20, 20), np.zeros((6, 8, 3), np.uint8))
    with pytest.raises(ValueError):
        backend.reconstruct(_ramp(20, 20), _ramp(6, 8), _ramp(6, 8), 1.0,
                            np.zeros((12, 16, 3), np.uint8))


def test_backend_diagnostics_count_calls_not_allocations():
    backend = pipeline.PythonReferenceBackend(16, 12, 8, 6)
    src = _ramp(12, 16)
    low = np.zeros((6, 8, 3), dtype=np.uint8)
    out = np.zeros((12, 16, 3), dtype=np.uint8)
    diag = backend.diagnostics()
    assert diag["downsample_call_count"] == 0 and diag["reconstruct_call_count"] == 0
    scratch_before = diag["python_scratch_bytes"]
    assert scratch_before > 0 and diag["axis_plan_builds"] == 4

    for _ in range(4):
        backend.prepare_working_color(src, low)
        backend.reconstruct(src, low, low, 1.0, out)

    after = backend.diagnostics()
    assert after["downsample_call_count"] == 4
    assert after["reconstruct_call_count"] == 4
    assert after["python_scratch_bytes"] == scratch_before, "scratch grew during the frame loop"
    assert after["axis_plan_builds"] == 4, "plans were rebuilt while processing frames"

    backend.close()
    assert backend.closed and backend.context.res_diff_low_f32 is None


def test_export_core_uses_one_backend_for_color_and_residual():
    """The color the DLSS session sees and the color the residual subtracts are the
    same array, produced by the same backend that reconstructs."""
    import guidance_stubs as gs

    backend = pipeline.PythonReferenceBackend(gs.SRC_W, gs.SRC_H, gs.LIVE_W, gs.LIVE_H)
    live = gs.StubLive(gs.LIVE_W, gs.LIVE_H)
    core = pipeline.ProductionExportCore(
        src_w=gs.SRC_W, src_h=gs.SRC_H, dst_w=gs.SRC_W, dst_h=gs.SRC_H,
        passes=[live],
        job_settings={"guidance_mode": 0, "working_scale": 0.5, "residual_pyramid": True,
                      "residual_strength": 1.0},
        residual_backend=backend,
    )
    frame = np.full((gs.SRC_H, gs.SRC_W, 3), 90, dtype=np.uint8)
    _final, record = core.process_frame(0, frame)

    assert np.array_equal(live.seen_rgba[0][:, :, :3], core.workspace.down_cur_bgr), \
        "DLSS was fed a different working color than the residual subtracts"
    assert record["residual_backend"] == "cpu_reference_python"
    assert record["residual_backend_actual"] == "cpu_reference_python"
    assert backend.diagnostics()["downsample_call_count"] == 1
    assert backend.diagnostics()["reconstruct_call_count"] == 1


def test_export_core_rejects_multiple_passes():
    import guidance_stubs as gs

    passes = [gs.StubLive(gs.LIVE_W, gs.LIVE_H), gs.StubLive(gs.LIVE_W, gs.LIVE_H)]
    with pytest.raises(ValueError):
        pipeline.ProductionExportCore(
            src_w=gs.SRC_W, src_h=gs.SRC_H, dst_w=gs.SRC_W, dst_h=gs.SRC_H,
            passes=passes,
            job_settings={"guidance_mode": 0, "working_scale": 0.5, "residual_pyramid": True},
        )


def test_zero_flow_provider_delivers_a_zero_plane_at_the_requested_size():
    p = pipeline.ZeroFlowProvider()
    got = p.get_flow(0, np.zeros((6, 8, 3), np.uint8), None, 8, 6)
    assert got.shape == (6, 8, 2) and got.dtype == np.float32 and not got.any()
    buf = np.full((6, 8, 2), 5.0, dtype=np.float32)
    assert p.get_flow(0, None, None, 8, 6, out=buf) is buf and not buf.any()


def test_cached_flo_provider_reads_real_flow_files(tmp_path):
    flow = np.zeros((12, 16, 2), dtype=np.float32)
    flow[..., 0] = 2.0
    flow[..., 1] = -1.5
    pipeline.write_flo(os.path.join(str(tmp_path), "000004.flo"), flow)

    p = pipeline.CachedFloFlowProvider(str(tmp_path))
    assert p.has_flow(4) and not p.has_flow(5)
    got = p.get_flow(4, None, None, 16, 12)
    assert np.array_equal(got, flow), "cached motion vectors were altered"

    missing = p.get_flow(5, None, None, 16, 12)
    assert not missing.any() and p.missing_frame_count == 1 and p.loaded_frame_count == 1
    assert p.timing()["guidance_backend"] == "cached_flo"


def test_cached_flo_provider_resizes_to_the_working_extent(tmp_path):
    flow = np.zeros((12, 16, 2), dtype=np.float32)
    flow[:, :, 0] = 4.0
    flow[:, :, 1] = -4.0
    pipeline.write_flo(os.path.join(str(tmp_path), "000000.flo"), flow)

    got = pipeline.CachedFloFlowProvider(str(tmp_path)).get_flow(0, None, None, 8, 6)
    assert got.shape == (6, 8, 2)
    assert np.allclose(got[..., 0], 4.0) and np.allclose(got[..., 1], -4.0)


def test_export_default_is_auto_and_reports_what_actually_ran():
    """No residual_backend in job_settings must resolve through the one factory, and the
    record has to say which implementation ran rather than what was asked for."""
    import guidance_stubs as gs

    live = gs.StubLive(gs.LIVE_W, gs.LIVE_H)
    core = pipeline.ProductionExportCore(
        src_w=gs.SRC_W, src_h=gs.SRC_H, dst_w=gs.SRC_W, dst_h=gs.SRC_H,
        passes=[live],
        job_settings={"guidance_mode": 0, "working_scale": 0.5, "residual_pyramid": True},
    )
    _final, record = core.process_frame(0, gs.source_frame())
    try:
        assert record["residual_backend_requested"] == "auto"
        assert record["residual_backend_actual"] == core.residual_backend.name
        assert record["residual_backend"] == core.residual_backend.name
        assert core.backend_diagnostics["residual_backend_fallback_reason"] in (
            "native_available", "native_load_failed")
    finally:
        core.close()


def test_export_reports_python_when_native_bridge_is_absent(monkeypatch):
    """auto must never claim to be native."""
    import guidance_stubs as gs

    monkeypatch.setattr(pipeline, "_load_native_backend", lambda *a: None)
    live = gs.StubLive(gs.LIVE_W, gs.LIVE_H)
    core = pipeline.ProductionExportCore(
        src_w=gs.SRC_W, src_h=gs.SRC_H, dst_w=gs.SRC_W, dst_h=gs.SRC_H,
        passes=[live],
        job_settings={"guidance_mode": 0, "working_scale": 0.5, "residual_pyramid": True},
    )
    try:
        assert core.residual_backend.name == "cpu_reference_python"
        assert core.backend_diagnostics["residual_backend_fallback_reason"] == "native_load_failed"
    finally:
        core.close()


def test_export_can_be_told_to_use_native_and_says_so():
    import guidance_stubs as gs

    live = gs.StubLive(gs.LIVE_W, gs.LIVE_H)
    core = pipeline.ProductionExportCore(
        src_w=gs.SRC_W, src_h=gs.SRC_H, dst_w=gs.SRC_W, dst_h=gs.SRC_H,
        passes=[live],
        job_settings={"guidance_mode": 0, "working_scale": 0.5, "residual_pyramid": True,
                      "residual_backend": "native"},
    )
    _final, record = core.process_frame(0, gs.source_frame())
    try:
        assert record["residual_backend_actual"] == "cpu_reference_native", record
        assert record["residual_backend_version"].startswith("api-0x")
    finally:
        core.close()
