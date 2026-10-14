"""Gate 2: tiny numeric tests that compare the production resamplers against an
independent scalar oracle (see resampler_oracle.py).

Every test here compares real pixel values. Tolerances are set by what float32
accumulation can explain: production keeps weights and intermediates in FP32 while
the oracle runs FP64, so a <=1 code value difference is expected only when a value
sits exactly on a rounding boundary. A mean difference under 0.05 across the whole
image means the two agree everywhere except occasional boundary pixels.
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline  # noqa: E402
from resampler_oracle import (  # noqa: E402
    IMAGE_INPUTS,
    checkerboard,
    deterministic_random,
    edge_impulse,
    impulse,
    quantized,
    ramp,
    scalar_catmull_rom_upsample,
    scalar_composite,
    scalar_lanczos2_downsample,
    signed_residual,
)


# ------------------------------------------------------------- Lanczos gates ----

def _assert_matches_lanczos_oracle(src, dst_w, dst_h):
    got = pipeline.downsample_lanczos2(src, (dst_w, dst_h))
    expected = quantized(scalar_lanczos2_downsample(src, dst_w, dst_h))
    assert got.shape == (dst_h, dst_w, 3)
    diff = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    assert diff.max() <= 1.0, f"max diff {diff.max()} at {np.unravel_index(diff.argmax(), diff.shape)}"
    assert diff.mean() < 0.05, f"mean diff {diff.mean()}"


def test_lanczos_8x8_to_4x4_matches_scalar_oracle():
    for maker in IMAGE_INPUTS:
        _assert_matches_lanczos_oracle(maker(8, 8), 4, 4)


def test_lanczos_8x8_to_3x3_matches_scalar_oracle():
    for maker in IMAGE_INPUTS:
        _assert_matches_lanczos_oracle(maker(8, 8), 3, 3)


def test_lanczos_non_integer_scale_matches_scalar_oracle():
    for maker in IMAGE_INPUTS:
        _assert_matches_lanczos_oracle(maker(7, 11), 4, 6)


def test_lanczos_tap_count_widens_with_downscale_ratio():
    """A fixed 4-tap window cannot anti-alias: support is 2/scale in source space."""
    half = pipeline._build_lanczos2_axis_plan(8, 4)
    quarter = pipeline._build_lanczos2_axis_plan(8, 2)
    assert set(half.tap_counts.tolist()) == {8}, half.tap_counts
    assert set(quarter.tap_counts.tolist()) == {16}, quarter.tap_counts
    assert half.max_taps == 8 and quarter.max_taps == 16
    for plan in (half, quarter):
        assert plan.weights.sum(axis=1) == pytest.approx(np.ones(plan.dst_len), abs=1e-6)


def test_lanczos_dst_buffer_identity():
    src = ramp(8, 8)
    dst = np.full((4, 4, 3), 123, dtype=np.uint8)
    result = pipeline.downsample_lanczos2(src, (4, 4), dst=dst)
    assert result is dst
    assert not np.all(dst == 123), "dst was never written"
    expected = quantized(scalar_lanczos2_downsample(src, 4, 4))
    assert np.abs(dst.astype(np.float64) - expected.astype(np.float64)).max() <= 1.0


# --------------------------------------------------------- Catmull-Rom gates ----

def _residual_case(low_w, low_h, high_w, high_h, kind, strength=1.0, seed=11):
    """Run production reconstruction and return it next to the oracle's answer."""
    rng = np.random.default_rng(seed)
    source = rng.integers(0, 256, size=(high_h, high_w, 3), dtype=np.uint8)
    down_cur = rng.integers(0, 256, size=(low_h, low_w, 3), dtype=np.uint8)
    delta = signed_residual(low_h, low_w, kind).astype(np.int32)
    dlss_out = np.clip(down_cur.astype(np.int32) + delta, 0, 255).astype(np.uint8)

    backend = pipeline.PythonReferenceBackend(high_w, high_h, low_w, low_h)
    out = np.zeros((high_h, high_w, 3), dtype=np.uint8)
    got = backend.reconstruct(source, down_cur, dlss_out, strength, out)
    residual_low = dlss_out.astype(np.float64) - down_cur.astype(np.float64)
    residual_full = scalar_catmull_rom_upsample(residual_low, high_w, high_h)
    return got, source, residual_full, backend


def _assert_composite(got, source, residual_full, strength):
    expected = scalar_composite(source, residual_full, strength)
    diff = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    assert got.shape == source.shape
    assert diff.max() <= 1.0, f"max diff {diff.max()}"
    assert diff.mean() < 0.05, f"mean diff {diff.mean()}"


def test_catmull_rom_4x4_to_8x8_matches_scalar_oracle():
    for kind in ("positive", "negative", "mixed"):
        got, source, residual_full, _ = _residual_case(4, 4, 8, 8, kind)
        _assert_composite(got, source, residual_full, 1.0)


def test_catmull_rom_non_integer_scale_matches_scalar_oracle():
    got, source, residual_full, _ = _residual_case(5, 3, 11, 7, "mixed")
    _assert_composite(got, source, residual_full, 1.0)


def test_catmull_rom_preserves_negative_lobes():
    plan = pipeline._build_catmull_rom_axis_plan(4, 8)
    active = np.zeros_like(plan.weights, dtype=bool)
    for d in range(plan.dst_len):
        active[d, :plan.tap_counts[d]] = True
    masked = np.where(active, plan.weights, 0.0)
    assert (masked < 0.0).any(), "Catmull-Rom plan lost its negative lobes"
    assert set(plan.tap_counts.tolist()) == {4}
    assert plan.indices.min() >= 0 and plan.indices.max() <= plan.src_len - 1


def test_signed_residual_preserves_negative_values():
    """A negative residual must darken the composite, not be clamped away."""
    for kind in ("positive", "negative", "mixed"):
        got, source, residual_full, backend = _residual_case(4, 4, 8, 8, kind)
        assert (backend.context.res_diff_high_f32 < 0.0).any(), f"{kind}: residual lost its sign"
        _assert_composite(got, source, residual_full, 1.0)
        if kind == "negative":
            assert (got.astype(np.int32) < source.astype(np.int32)).any(), \
                "negative residual never darkened any pixel"


def test_residual_strength_zero():
    rng = np.random.default_rng(3)
    source = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    down_cur = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    dlss_out = rng.integers(0, 256, size=(4, 4, 3), dtype=np.uint8)
    backend = pipeline.PythonReferenceBackend(8, 8, 4, 4)
    out = np.zeros((8, 8, 3), dtype=np.uint8)
    got = backend.reconstruct(source, down_cur, dlss_out, 0.0, out)
    assert np.array_equal(got, source), "strength 0 must return the untouched source"


def test_residual_strength_one():
    got, source, residual_full, _ = _residual_case(4, 4, 8, 8, "mixed", strength=1.0)
    _assert_composite(got, source, residual_full, 1.0)


def test_residual_strength_one_point_five():
    got, source, residual_full, _ = _residual_case(4, 4, 8, 8, "mixed", strength=1.5)
    _assert_composite(got, source, residual_full, 1.5)


# ------------------------------------------------------- oracle-free invariants ----

def test_lanczos_preserves_constant_color():
    """Any normalized filter must map a flat field onto the same value, at any ratio."""
    cases = [((8, 8), (4, 4)), ((8, 8), (3, 3)), ((7, 11), (6, 4)), ((13, 9), (11, 7)),
             ((8, 8), (12, 5))]
    for (src_w, src_h), (dst_w, dst_h) in cases:
        src = np.full((src_h, src_w, 3), 177, dtype=np.uint8)
        got = pipeline.downsample_lanczos2(src, (dst_w, dst_h))
        assert np.unique(got).tolist() == [177], f"{src_w}x{src_h}->{dst_w}x{dst_h} lost the DC level"


def test_lanczos_reproduces_linear_ramp():
    """At 2:1 every window is symmetric and normalized, so a ramp must stay a ramp.

    Independent of the oracle: the expected destination value is the source
    coordinate it samples, 2j + 0.5. A wrong sampling coordinate, a dropped tap or a
    broken edge clamp all move that line.
    """
    src = np.zeros((64, 64, 3), dtype=np.uint8)
    src[:, :, 0] = np.arange(64)[None, :]
    src[:, :, 1] = np.arange(64)[:, None]
    got = pipeline.downsample_lanczos2(src, (32, 32)).astype(np.float64)

    expected_cols = np.arange(32) * 2.0 + 0.5
    err_h = np.abs(got[:, :, 0] - expected_cols[None, :])
    err_v = np.abs(got[:, :, 1] - expected_cols[:, None])
    assert err_h.max() <= 1.0, f"horizontal ramp off by {err_h.max()}"
    assert err_v.max() <= 1.0, f"vertical ramp off by {err_v.max()}"


# ------------------------------------------------------------ plan lifecycle ----

def test_filter_plan_build_count_does_not_grow_per_frame():
    src = ramp(8, 8)
    backend = pipeline.create_residual_backend("python", 8, 8, 4, 4)
    after_build = pipeline.filter_plan_build_count
    assert backend.context.resampler_plan.axis_plan_builds == 4
    assert backend.name == "cpu_reference_python"

    dst = np.empty((4, 4, 3), dtype=np.uint8)
    recon = np.empty((8, 8, 3), dtype=np.uint8)
    for _ in range(5):
        backend.prepare_working_color(src, dst)
        backend.reconstruct(src, dst, dst, 1.0, recon)
    assert pipeline.filter_plan_build_count == after_build, \
        "resampler plans were rebuilt while processing frames"


def test_the_tap_width_query_answers_what_the_plans_actually_carry():
    """The width a gate compares against a kernel's window, without building one.

    A GPU bridge is opened with a plan whose widest row decides whether the kernel
    can walk it, so the figure is asked here rather than by a second copy of the
    window arithmetic - and the copy that would have to agree is this test. The
    query builds nothing, because ``filter_plan_build_count`` measures plans
    rebuilt per frame and a gate that only asked a question must not move it.
    """
    for src_len, dst_len in ((8, 4), (11, 6), (2160, 1616), (2160, 1448),
                             (2160, 1360), (3840, 2576), (2160, 540)):
        before = pipeline.filter_plan_build_count
        width = pipeline.lanczos2_axis_taps(src_len, dst_len)
        assert pipeline.filter_plan_build_count == before, (
            "the width query built a plan")
        assert width == pipeline._build_lanczos2_axis_plan(
            src_len, dst_len).max_taps, (src_len, dst_len)

    for src_len, dst_len in ((4, 8), (1080, 2160), (1448, 2160), (1360, 2160),
                             (540, 2160)):
        assert pipeline.CATMULL_ROM_TAPS == pipeline._build_catmull_rom_axis_plan(
            src_len, dst_len).max_taps, (src_len, dst_len)
    with pytest.raises(ValueError):
        pipeline.lanczos2_axis_taps(2160, 0)


def test_axis_plan_rejects_mismatched_source_length():
    """A working->source plan must never be applied to a source->working reshape."""
    plan = pipeline._build_catmull_rom_axis_plan(4, 8)
    values = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        pipeline._apply_axis_plan(values, plan, axis=1)


def test_axis_plan_pads_taps_with_zero_weights():
    """Padded columns must be inert, so tap_counts is the only window definition."""
    plan = pipeline._build_lanczos2_axis_plan(11, 6)
    assert plan.indices.shape == (6, plan.max_taps)
    assert plan.weights.shape == (6, plan.max_taps)
    for d in range(plan.dst_len):
        padding = plan.weights[d, plan.tap_counts[d]:]
        assert (padding == 0.0).all()
    assert plan.weights.dtype == np.float32 and plan.indices.dtype == np.int32
