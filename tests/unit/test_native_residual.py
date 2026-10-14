"""Gate: the native hot path must be numerically the reference, not merely close.

Every test compares real pixels. Where a Python and a native FP32 accumulation order
could differ, both are also compared against the independent scalar oracle, and the
allowed error is the error the Python reference already has against that oracle - never
larger (task section 16).

Skipped only when residual_bridge.dll is not built on this machine; when it is present,
a mismatch is a failure.
"""

import ctypes
import os
import sys

import cv2
import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline  # noqa: E402
import resampler_oracle as oracle  # noqa: E402

residual_native = pytest.importorskip(
    "residual_native", reason="residual_bridge extension missing")
if not residual_native.is_available():
    pytest.skip("residual_bridge could not be probed", allow_module_level=True)

NATIVE = "cpu_reference_native"
PYREF = "cpu_reference_python"


def _native(src_w, src_h, live_w, live_h):
    return residual_native.NativeResidualBackend(src_w, src_h, live_w, live_h)


def _python(src_w, src_h, live_w, live_h):
    return pipeline.PythonReferenceBackend(src_w, src_h, live_w, live_h)


def _down_both(src, dst_w, dst_h):
    """Run the same downsample through both backends. Returns (native, python)."""
    src_h, src_w = src.shape[:2]
    n = _native(src_w, src_h, dst_w, dst_h)
    p = _python(src_w, src_h, dst_w, dst_h)
    try:
        out_n = np.full((dst_h, dst_w, 3), 77, dtype=np.uint8)
        out_p = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        got_n = n.prepare_working_color(src, out_n)
        got_p = p.prepare_working_color(src, out_p)
        assert got_n is out_n and got_p is out_p, "backend must fill the caller's buffer"
        return out_n, out_p
    finally:
        n.close()
        p.close()


def _recon_both(source, low, dlss, strength):
    src_h, src_w = source.shape[:2]
    live_h, live_w = low.shape[:2]
    n = _native(src_w, src_h, live_w, live_h)
    p = _python(src_w, src_h, live_w, live_h)
    try:
        out_n = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        out_p = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        return n.reconstruct(source, low, dlss, strength, out_n), p.reconstruct(source, low, dlss, strength, out_p)
    finally:
        n.close()
        p.close()


def _assert_matches_oracle(native, expected, label):
    """Same acceptance the Python reference gets from the oracle: max 1, mean < 0.05."""
    diff = np.abs(native.astype(np.float64) - expected.astype(np.float64))
    assert diff.max() <= 1.0, f"{label}: max diff {diff.max()}"
    assert diff.mean() < 0.05, f"{label}: mean diff {diff.mean()}"


# The working scales the round is specified against: 25, 33, 50, 63, 75 and 83 per cent,
# plus the identity scale where a plan's two resolutions are the same size.
WORKING_SCALES = (1.0, 0.83, 0.75, 0.63, 0.5, 0.33, 0.25)

# (src_w, src_h, live_w, live_h) shapes that no scale arithmetic reaches: the smallest
# extents the bridge accepts, and non-rectangular ratios at the same time.
SMALL_SHAPES = ((16, 16, 8, 8), (8, 8, 8, 8), (16, 16, 5, 4), (11, 7, 4, 6),
                (5, 3, 2, 1), (4, 4, 1, 1))


def _fused_prepare(src, dst_w, dst_h):
    """One rr_prepare_working_bgr_rgba8 call. Returns the (bgr, rgba) the library wrote."""
    src_h, src_w = src.shape[:2]
    n = _native(src_w, src_h, dst_w, dst_h)
    try:
        bgr = np.full((dst_h, dst_w, 3), 77, dtype=np.uint8)
        rgba = np.full((dst_h, dst_w, 4), 77, dtype=np.uint8)
        got_bgr, got_rgba = n.prepare_working_color_rgba(src, bgr, rgba)
        assert got_bgr is bgr and got_rgba is rgba, "backend must fill the caller's buffers"
        return bgr, rgba
    finally:
        n.close()


# --------------------------------------------------- Lanczos against the oracle ----

def test_native_lanczos_8x8_to_4x4_matches_scalar_oracle():
    for maker in oracle.IMAGE_INPUTS:
        src = maker(8, 8)
        native, python = _down_both(src, 4, 4)
        _assert_matches_oracle(native, oracle.quantized(oracle.scalar_lanczos2_downsample(src, 4, 4)),
                               f"{maker.__name__} 8x8->4x4")
        assert np.array_equal(native, python), f"{maker.__name__}: native diverged from the reference"


def test_native_lanczos_8x8_to_3x3_matches_scalar_oracle():
    for maker in oracle.IMAGE_INPUTS:
        src = maker(8, 8)
        native, python = _down_both(src, 3, 3)
        _assert_matches_oracle(native, oracle.quantized(oracle.scalar_lanczos2_downsample(src, 3, 3)),
                               f"{maker.__name__} 8x8->3x3")
        assert np.array_equal(native, python)


def test_native_lanczos_non_integer_scale_matches_scalar_oracle():
    for maker in oracle.IMAGE_INPUTS:
        src = maker(7, 11)
        native, python = _down_both(src, 4, 6)
        _assert_matches_oracle(native, oracle.quantized(oracle.scalar_lanczos2_downsample(src, 4, 6)),
                               f"{maker.__name__} 11x7->4x6")
        assert np.array_equal(native, python)


def test_native_plan_reports_four_axis_plans():
    backend = _native(32, 24, 16, 12)
    try:
        diag = backend.diagnostics()
        assert diag["axis_plan_count"] == 4, "expected lanczos_x/y and catmull_x/y"
        assert diag["native_scratch_bytes"] > 0
        # A fixed pool of workers exists from plan creation; the calling thread joins the
        # chunk split too, so worker_count is one more than the pool size.
        assert diag["this_plan_worker_count"] >= 2, diag
        assert diag["this_plan_worker_create_count"] >= 1
        # The lifecycle block describes this plan, so it is a constant while it is open:
        # one plan, one scratch allocation, nothing destroyed.
        assert diag["this_plan_create_count"] == 1
        assert diag["this_plan_scratch_alloc_count"] == 1
        assert diag["this_plan_destroy_count"] == 0
        assert diag["this_plan_scratch_alloc_bytes"] == diag["native_scratch_bytes"]
    finally:
        backend.close()


def test_native_downsample_matches_oracle_at_every_working_scale():
    src_h, src_w = 64, 96
    for scale in WORKING_SCALES:
        live_w, live_h = pipeline.resolve_working_dims(src_w, src_h, scale)
        for maker in (oracle.ramp, oracle.checkerboard, oracle.deterministic_random, oracle.edge_impulse):
            src = maker(src_h, src_w)
            native, python = _down_both(src, live_w, live_h)
            _assert_matches_oracle(native,
                                   oracle.quantized(oracle.scalar_lanczos2_downsample(src, live_w, live_h)),
                                   f"{maker.__name__} scale {scale} -> {live_w}x{live_h}")
            assert np.array_equal(native, python), f"scale {scale} {maker.__name__}"


# -------------------------------------------- fused BGR+RGBA preparation ----

def test_native_prepare_bgr_rgba_matches_old_bgr():
    """The fused store is the same Lanczos result written twice, not a second implementation.

    Compared against the two-pass native path and against the Python reference at every
    working scale, over every oracle image, so a change that only reached one of the two
    stores cannot pass.
    """
    src_h, src_w = 64, 96
    for scale in WORKING_SCALES:
        live_w, live_h = pipeline.resolve_working_dims(src_w, src_h, scale)
        for maker in oracle.IMAGE_INPUTS:
            src = maker(src_h, src_w)
            fused, _rgba = _fused_prepare(src, live_w, live_h)
            native, python = _down_both(src, live_w, live_h)
            assert np.array_equal(fused, native), (
                f"scale {scale} {maker.__name__}: the fused BGR store diverged from "
                f"rr_downsample_bgr8")
            assert np.array_equal(fused, python), (
                f"scale {scale} {maker.__name__}: the fused BGR store diverged from the "
                f"Python reference")


def test_native_prepare_bgr_rgba_matches_old_bgr_at_small_shapes():
    for src_w, src_h, live_w, live_h in SMALL_SHAPES:
        for maker in oracle.IMAGE_INPUTS:
            src = maker(src_h, src_w)
            fused, _rgba = _fused_prepare(src, live_w, live_h)
            native, python = _down_both(src, live_w, live_h)
            assert np.array_equal(fused, native), (
                f"{src_w}x{src_h}->{live_w}x{live_h} {maker.__name__}: fused BGR store "
                f"diverged from rr_downsample_bgr8")
            assert np.array_equal(fused, python), (
                f"{src_w}x{src_h}->{live_w}x{live_h} {maker.__name__}: fused BGR store "
                f"diverged from the Python reference")


def test_native_prepare_bgr_rgba_matches_old_bgr_at_4k():
    """The size the round is measured at. A fused store that only matches on thumbnails is
    a vectorisation bug waiting for the first real frame."""
    src_w, src_h = 3840, 2160
    for scale in (0.5, 0.25):
        live_w, live_h = pipeline.resolve_working_dims(src_w, src_h, scale)
        src = oracle.deterministic_random(src_h, src_w)
        fused, _rgba = _fused_prepare(src, live_w, live_h)
        native, python = _down_both(src, live_w, live_h)
        assert np.array_equal(fused, native), f"4K at scale {scale}: fused vs rr_downsample_bgr8"
        assert np.array_equal(fused, python), f"4K at scale {scale}: fused vs Python reference"


def test_native_prepare_bgr_rgba_output_matches_oracle():
    """The fused path still answers to the independent scalar oracle, not only to the two
    implementations that share this file's tap builder."""
    for src_w, src_h, live_w, live_h in ((8, 8, 4, 4), (8, 8, 3, 3), (11, 7, 4, 6)):
        for maker in oracle.IMAGE_INPUTS:
            src = maker(src_h, src_w)
            fused, _rgba = _fused_prepare(src, live_w, live_h)
            _assert_matches_oracle(
                fused,
                oracle.quantized(oracle.scalar_lanczos2_downsample(src, live_w, live_h)),
                f"fused {src_w}x{src_h}->{live_w}x{live_h} {maker.__name__}")


def test_native_prepare_rgba_matches_cv2_conversion():
    """The RGBA plane is what cv2.cvtColor(BGR2RGBA) would have produced from the BGR plane
    of the same call, byte for byte, at every working scale.

    This is the equivalence that lets the conversion be deleted: anything less and the
    reduced path would be feeding DLSS a different image.
    """
    src_h, src_w = 64, 96
    for scale in WORKING_SCALES:
        live_w, live_h = pipeline.resolve_working_dims(src_w, src_h, scale)
        for maker in oracle.IMAGE_INPUTS:
            src = maker(src_h, src_w)
            fused_bgr, fused_rgba = _fused_prepare(src, live_w, live_h)
            expected = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGBA)
            assert np.array_equal(fused_rgba, expected), (
                f"scale {scale} {maker.__name__}: fused RGBA differs from "
                f"cv2.cvtColor(BGR2RGBA) in {(fused_rgba != expected).sum()} bytes")
            assert np.all(fused_rgba[..., 3] == 255), "alpha must be opaque, not a leftover"


def test_native_prepare_rgba_matches_cv2_conversion_at_small_shapes_and_4k():
    for src_w, src_h, live_w, live_h in SMALL_SHAPES:
        src = oracle.checkerboard(src_h, src_w)
        fused_bgr, fused_rgba = _fused_prepare(src, live_w, live_h)
        assert np.array_equal(fused_rgba, cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGBA)), (
            f"{src_w}x{src_h}->{live_w}x{live_h}")

    src_w, src_h = 3840, 2160
    live_w, live_h = pipeline.resolve_working_dims(src_w, src_h, 0.5)
    src = oracle.deterministic_random(src_h, src_w)
    fused_bgr, fused_rgba = _fused_prepare(src, live_w, live_h)
    expected = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGBA)
    assert fused_rgba.shape == (live_h, live_w, 4)
    assert np.array_equal(fused_rgba, expected), "the RGBA store differs at 4K"


# ------------------------------------------------- Catmull-Rom against the oracle ----

def _residual_triple(low_w, low_h, high_w, high_h, kind, seed=11):
    rng = np.random.default_rng(seed)
    source = rng.integers(0, 256, size=(high_h, high_w, 3), dtype=np.uint8)
    # A random low plane matters: with a constant one the residual is a copy of the DLSS
    # output, and reading the wrong plane at the wrong stride still looks correct.
    low = rng.integers(0, 256, size=(low_h, low_w, 3), dtype=np.uint8)
    if kind == "checkerboard":
        yy, xx = np.indices((low_h, low_w))
        delta = (np.where((xx + yy) % 2, 1, -1)[..., None] * 40).astype(np.int64)
    elif kind == "deterministic_random":
        dlss = oracle.deterministic_random(low_h, low_w)
    else:
        delta = oracle.signed_residual(low_h, low_w, kind).astype(np.int32)
    if kind != "deterministic_random":
        dlss = np.clip(low.astype(np.int32) + delta, 0, 255).astype(np.uint8)
    residual_full = oracle.scalar_catmull_rom_upsample(
        dlss.astype(np.float64) - low.astype(np.float64), high_w, high_h)
    return source, low, dlss, residual_full


# The residual shapes the composite is gated over: both signs, a sign that alternates in
# space so a half-pixel offset cannot hide, and pixels no pattern predicts.
RESIDUAL_KINDS = ("positive", "negative", "mixed", "checkerboard", "deterministic_random")
STRENGTHS = (0.0, 0.5, 1.0, 1.5)


def _recon_input_shapes(source, low, dlss, strength):
    """Reconstruct one frame twice: DLSS output as BGR, and as the RGBA of that same BGR.

    Returns (bgr_out, rgba_out). The RGBA plane is derived from the BGR plane rather than
    invented, so the pair can only agree if the direct read maps channels by meaning.
    """
    src_h, src_w = source.shape[:2]
    live_h, live_w = low.shape[:2]
    n = _native(src_w, src_h, live_w, live_h)
    try:
        rgba = cv2.cvtColor(dlss, cv2.COLOR_BGR2RGBA)
        out_bgr = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        out_rgba = np.zeros((src_h, src_w, 3), dtype=np.uint8)
        n.reconstruct(source, low, dlss, strength, out_bgr)
        n.reconstruct_from_rgba(source, low, rgba, strength, out_rgba)
        return out_bgr, out_rgba
    finally:
        n.close()


def test_native_reconstruct_rgba_matches_old_bgr_path():
    """Reading the DLSS output as RGBA must not move a single byte of the composite.

    Every case compares the two entry points on one plan over the same frame, and checks the
    result against the scalar oracle as well, so a pair of paths that agree with each other
    and disagree with the maths cannot pass.
    """
    for low_w, low_h, high_w, high_h in ((4, 4, 8, 8), (20, 20, 40, 40), (5, 3, 11, 7),
                                         (32, 24, 64, 48)):
        for kind in RESIDUAL_KINDS:
            source, low, dlss, residual_full = _residual_triple(low_w, low_h, high_w, high_h, kind)
            for strength in STRENGTHS:
                bgr_out, rgba_out = _recon_input_shapes(source, low, dlss, strength)
                label = (f"{low_w}x{low_h}->{high_w}x{high_h} {kind} strength {strength}")
                assert np.array_equal(bgr_out, rgba_out), (
                    f"{label}: reading RGBA directly changed {(bgr_out != rgba_out).sum()} "
                    f"bytes of {(bgr_out != rgba_out).size}")
                if strength == 1.0:
                    _assert_matches_oracle(rgba_out,
                                           oracle.scalar_composite(source, residual_full, 1.0),
                                           f"{label} vs oracle")


def test_native_reconstruct_rgba_matches_old_bgr_path_at_4k():
    source, low, dlss, _residual = _residual_triple(1920, 1080, 3840, 2160, "mixed")
    for strength in (1.0, 1.5):
        bgr_out, rgba_out = _recon_input_shapes(source, low, dlss, strength)
        assert np.array_equal(bgr_out, rgba_out), (
            f"4K at strength {strength}: {(bgr_out != rgba_out).sum()} bytes differ")


def test_native_reconstruct_rgba_matches_python_reference():
    """The direct read is allowed to replace the conversion, so it has to answer to the
    reference the conversion was feeding."""
    source, low, dlss, _residual = _residual_triple(20, 20, 40, 40, "mixed")
    rgba = cv2.cvtColor(dlss, cv2.COLOR_BGR2RGBA)
    n = _native(40, 40, 20, 20)
    p = _python(40, 40, 20, 20)
    try:
        for strength in STRENGTHS:
            out_n = np.zeros((40, 40, 3), np.uint8)
            out_p = np.zeros((40, 40, 3), np.uint8)
            n.reconstruct_from_rgba(source, low, rgba, strength, out_n)
            p.reconstruct(source, low, dlss, strength, out_p)
            assert np.array_equal(out_n, out_p), f"strength {strength} vs the reference"
    finally:
        n.close()
        p.close()


def test_native_reconstruct_rgba_ignores_the_alpha_channel():
    """Alpha carries no colour, so no value it holds may reach the composite.

    A version that folded the fourth byte into the residual - which a SIMD rewrite could do
    by accident - would still produce plausible pixels.
    """
    source, low, dlss, _residual = _residual_triple(20, 20, 40, 40, "mixed")
    rgba = cv2.cvtColor(dlss, cv2.COLOR_BGR2RGBA)
    other = rgba.copy()
    rng = np.random.default_rng(31)
    other[..., 3] = rng.integers(0, 256, size=other[..., 3].shape, dtype=np.uint8)
    n = _native(40, 40, 20, 20)
    try:
        first = np.zeros((40, 40, 3), np.uint8)
        second = np.zeros((40, 40, 3), np.uint8)
        n.reconstruct_from_rgba(source, low, rgba, 1.0, first)
        n.reconstruct_from_rgba(source, low, other, 1.0, second)
        assert np.array_equal(first, second), "the alpha plane changed the picture"
    finally:
        n.close()


def test_native_reconstruct_rgba_preserves_negative_residual():
    """The negative half of the residual survives the direct read.

    The residual is never clamped where it is formed - the chain has exactly one clamp, at
    the end - so a darkened working pixel has to darken the output, and the darkening has to
    stop at zero rather than wrap. The claim is made about the working-resolution residual:
    Catmull-Rom has negative lobes, so a handful of upsampled values in an all-negative field
    come out slightly positive, and a test that demanded otherwise would be measuring the
    kernel rather than the sign of the residual.
    """
    source, low, dlss, residual_full = _residual_triple(20, 20, 40, 40, "negative")
    working_residual = dlss.astype(np.int32) - low.astype(np.int32)
    assert (working_residual <= 0).all(), "the case stopped being a negative residual"
    assert (working_residual < 0).any(), "the case has no negative residual to preserve"
    rgba = cv2.cvtColor(dlss, cv2.COLOR_BGR2RGBA)
    n = _native(40, 40, 20, 20)
    try:
        out = np.zeros((40, 40, 3), np.uint8)
        n.reconstruct_from_rgba(source, low, rgba, 1.0, out)
        assert (out.astype(np.int32) < source.astype(np.int32)).any(), \
            "a negative residual never darkened anything through the RGBA path"
        _assert_matches_oracle(out, oracle.scalar_composite(source, residual_full, 1.0),
                               "negative residual via RGBA")
        dark = out.astype(np.int32)
        assert dark.min() >= 0, "a channel went below the single clamp of the chain"
        combined = source.astype(np.float64) + residual_full
        assert (combined < 0).any(), "nothing reached the clamp, so the case proves nothing"
        # np.rint is half-to-even, so a composite of exactly a half is also stored as zero.
        assert np.array_equal(dark == 0, combined <= 0.5), \
            "the clamp landed somewhere other than where the composite fell to or below a half"
    finally:
        n.close()


def test_rgba_reconstruct_reuses_plan_and_counts_its_own_entry_point():
    """Ten frames through the direct RGBA read: one plan, one scratch, and the library says so.

    The two reconstruct counters have to separate the entry points, because that distinction
    is the whole claim of the reduced path. A single counter that rose for both would be
    satisfied by a call that still converted on the way in, so the BGR-only call at the end is
    what makes the RGBA count a measurement. The Python totals are compared against the native
    ones rather than trusted: they are two separate readings of the same call.
    """
    backend = _native(64, 48, 32, 24)
    rng = np.random.default_rng(7)
    source = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    low = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    dlss = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    rgba = cv2.cvtColor(dlss, cv2.COLOR_BGR2RGBA)
    out = np.empty((48, 64, 3), dtype=np.uint8)
    try:
        before = backend.diagnostics()
        for _ in range(10):
            backend.reconstruct_from_rgba(source, low, rgba, 1.0, out)
        via_rgba = backend.diagnostics()
        backend.reconstruct(source, low, dlss, 1.0, out)
        via_bgr = backend.diagnostics()
    finally:
        backend.close()

    for step, label in ((via_rgba, "after ten RGBA-input calls"),
                        (via_bgr, "after one more BGR-input call")):
        assert step["process_plan_create_total"] == before["process_plan_create_total"], \
            f"{label}: a plan was built mid-run"
        assert step["process_scratch_alloc_total"] == before["process_scratch_alloc_total"], \
            f"{label}: scratch was reallocated"
        assert step["process_scratch_bytes_total"] == before["process_scratch_bytes_total"], \
            f"{label}: scratch bytes changed"
        assert step["process_worker_create_total"] == before["process_worker_create_total"], \
            f"{label}: a worker thread was started"
        assert step["this_plan_create_count"] == 1, f"{label}: this plan was rebuilt"
        assert step["this_plan_scratch_alloc_count"] == 1, f"{label}: this plan reallocated"
        assert step["native_scratch_bytes"] == before["native_scratch_bytes"]

    assert via_rgba["reconstruct_rgba_input_call_count"] == \
        before["reconstruct_rgba_input_call_count"] + 10
    assert via_rgba["reconstruct_bgr_only_call_count"] == \
        before["reconstruct_bgr_only_call_count"], \
        "an RGBA-input reconstruct was counted as a BGR-input one"
    assert via_bgr["reconstruct_bgr_only_call_count"] == \
        via_rgba["reconstruct_bgr_only_call_count"] + 1, \
        "the BGR-only entry point is not counted where it runs"
    assert via_bgr["reconstruct_rgba_input_call_count"] == \
        via_rgba["reconstruct_rgba_input_call_count"], \
        "a BGR-input reconstruct was counted as reading RGBA"
    assert via_rgba["reconstruct_call_count"] == before["reconstruct_call_count"] + 10, \
        "the two entry points do not add up to the total the library kept"
    assert via_rgba["reconstruct_pixel_count"] == before["reconstruct_pixel_count"] + 10 * 48 * 64
    assert via_rgba["py_rgba_reconstruct_call_count"] == 10
    assert via_rgba["py_rgba_reconstruct_call_count"] - before["py_rgba_reconstruct_call_count"] \
        == via_rgba["reconstruct_rgba_input_call_count"] - \
        before["reconstruct_rgba_input_call_count"], \
        "the Python and native counts of the same call disagree"


def test_native_catmull_rom_4x4_to_8x8_matches_scalar_oracle():
    source, low, dlss, residual_full = _residual_triple(4, 4, 8, 8, "mixed")
    native, python = _recon_both(source, low, dlss, 1.0)
    _assert_matches_oracle(native, oracle.scalar_composite(source, residual_full, 1.0), "4x4->8x8")
    assert np.array_equal(native, python)


def test_native_catmull_rom_non_integer_scale_matches_scalar_oracle():
    source, low, dlss, residual_full = _residual_triple(5, 3, 11, 7, "mixed")
    native, python = _recon_both(source, low, dlss, 1.0)
    _assert_matches_oracle(native, oracle.scalar_composite(source, residual_full, 1.0), "5x3->11x7")
    assert np.array_equal(native, python)


def test_native_signed_residual_positive_negative_mixed():
    for kind in ("positive", "negative", "mixed"):
        source, low, dlss, residual_full = _residual_triple(20, 20, 40, 40, kind)
        native, python = _recon_both(source, low, dlss, 1.0)
        _assert_matches_oracle(native, oracle.scalar_composite(source, residual_full, 1.0), kind)
        assert np.array_equal(native, python)
        if kind == "negative":
            assert (native.astype(np.int32) < source.astype(np.int32)).any(), \
                "a negative residual never darkened anything"
        if kind == "positive":
            assert (native.astype(np.int32) > source.astype(np.int32)).any(), \
                "a positive residual never brightened anything"


def test_native_strength_zero():
    source, low, dlss, _residual = _residual_triple(20, 20, 40, 40, "mixed")
    native, python = _recon_both(source, low, dlss, 0.0)
    assert np.array_equal(native, source), "strength 0 must return the untouched source"
    assert np.array_equal(native, python)


def test_native_strength_one():
    source, low, dlss, residual_full = _residual_triple(20, 20, 40, 40, "mixed")
    native, python = _recon_both(source, low, dlss, 1.0)
    _assert_matches_oracle(native, oracle.scalar_composite(source, residual_full, 1.0), "strength 1")
    assert np.array_equal(native, python)


def test_native_strength_one_point_five():
    source, low, dlss, residual_full = _residual_triple(20, 20, 40, 40, "mixed")
    native, python = _recon_both(source, low, dlss, 1.5)
    _assert_matches_oracle(native, oracle.scalar_composite(source, residual_full, 1.5), "strength 1.5")
    assert np.array_equal(native, python)


def test_native_matches_python_reference_random_cases():
    rng = np.random.default_rng(20260919)
    for trial in range(12):
        high_w = int(rng.integers(20, 60))
        high_h = int(rng.integers(15, 45))
        scale = float(rng.choice([0.25, 0.33, 0.5, 0.63, 0.75, 0.83]))
        live_w, live_h = pipeline.resolve_working_dims(high_w, high_h, scale)
        source = rng.integers(0, 256, size=(high_h, high_w, 3), dtype=np.uint8)
        low = rng.integers(0, 256, size=(live_h, live_w, 3), dtype=np.uint8)
        dlss = rng.integers(0, 256, size=(live_h, live_w, 3), dtype=np.uint8)
        strength = float(rng.choice([0.0, 0.5, 1.0, 1.5, 2.0]))

        down_n, down_p = _down_both(source, live_w, live_h)
        assert np.array_equal(down_n, down_p), f"trial {trial} downsample"
        rec_n, rec_p = _recon_both(source, low, dlss, strength)
        assert np.array_equal(rec_n, rec_p), f"trial {trial} reconstruct scale {scale}"

        residual_full = oracle.scalar_catmull_rom_upsample(
            dlss.astype(np.float64) - low.astype(np.float64), high_w, high_h)
        _assert_matches_oracle(rec_n, oracle.scalar_composite(source, residual_full, strength),
                               f"trial {trial} oracle")


# -------------------------------------------------------- loader reporting ----

def test_factory_reports_native_when_it_really_loaded():
    info = {}
    backend = pipeline.create_residual_backend("native", 32, 24, 16, 12, diagnostics_sink=info)
    try:
        assert isinstance(backend, residual_native.NativeResidualBackend)
        assert info["residual_backend_actual"] == NATIVE
        assert info["residual_backend_requested"] == "native"
        assert info.get("residual_backend_fallback_reason") is None
        assert backend.backend_info()["residual_backend_version"].startswith("api-0x")
    finally:
        backend.close()


def test_auto_prefers_native_and_records_why():
    info = {}
    backend = pipeline.create_residual_backend("auto", 32, 24, 16, 12, diagnostics_sink=info)
    try:
        assert backend.name == NATIVE
        assert info["residual_backend_fallback_reason"] == "native_available"
    finally:
        backend.close()


def test_python_still_selectable_and_never_misreported():
    info = {}
    backend = pipeline.create_residual_backend("python", 32, 24, 16, 12, diagnostics_sink=info)
    try:
        assert backend.name == PYREF
        assert info["residual_backend_actual"] == PYREF
    finally:
        backend.close()


# ------------------------------------------------------- validation and errors ----

def test_native_rejects_non_contiguous_input():
    backend = _native(32, 24, 16, 12)
    try:
        big = np.zeros((24, 34, 3), dtype=np.uint8)
        view = big[:, 1:33, :]
        assert not view.flags["C_CONTIGUOUS"]
        with pytest.raises(residual_native.ResidualBridgeError):
            backend.prepare_working_color(view, np.zeros((12, 16, 3), np.uint8))
    finally:
        backend.close()


def test_native_rejects_wrong_dimensions():
    backend = _native(32, 24, 16, 12)
    try:
        with pytest.raises(residual_native.ResidualBridgeError):
            backend.prepare_working_color(np.zeros((24, 40, 3), np.uint8),
                                          np.zeros((12, 16, 3), np.uint8))
        with pytest.raises(residual_native.ResidualBridgeError):
            backend.prepare_working_color(np.zeros((24, 32, 3), np.uint8),
                                          np.zeros((12, 20, 3), np.uint8))
        with pytest.raises(residual_native.ResidualBridgeError):
            backend.reconstruct(np.zeros((24, 32, 3), np.uint8), np.zeros((12, 16, 3), np.uint8),
                                np.zeros((12, 16, 3), np.uint8), 1.0, np.zeros((24, 32, 4), np.uint8))
    finally:
        backend.close()


def test_native_rejects_use_after_close_and_bad_handles():
    import ctypes

    backend = _native(16, 16, 8, 8)
    backend.close()
    with pytest.raises(residual_native.ResidualBridgeError):
        backend.prepare_working_color(np.zeros((16, 16, 3), np.uint8),
                                      np.zeros((8, 8, 3), np.uint8))

    lib = residual_native._library()
    bogus = ctypes.c_int64(12345)
    counters = residual_native._Counters()
    assert lib.rr_get_counters(bogus, ctypes.byref(counters)) == -5
    assert lib.rr_destroy_plan(bogus) == -5
    src = np.zeros((16, 16, 3), np.uint8)
    dst = np.zeros((8, 8, 3), np.uint8)
    assert lib.rr_downsample_bgr8(bogus, src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                  16, 16, 48,
                                  dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                  8, 8, 24) == -5
    buf = ctypes.create_string_buffer(256)
    assert lib.rr_get_last_error(buf, 256) > 0, "a rejected call must explain itself"


# --------------------------------------------------- fused preparation rejections ----

def test_fused_prepare_rejects_buffers_it_cannot_write():
    """A rejected fused call must not be counted, and must not leave a half-written frame.

    The reduced path is only evidence if the counter moved with the store: if a call that
    returned an error still bumped prepare_bgr_rgba_call_count, the counter would stop
    distinguishing the paths it exists to tell apart.
    """
    backend = _native(64, 48, 32, 24)
    try:
        before = backend.diagnostics()
        src = np.zeros((48, 64, 3), dtype=np.uint8)
        low = np.full((24, 32, 3), 77, dtype=np.uint8)
        bad_cases = {
            "rgba with three channels": (src, low, np.full((24, 32, 3), 77, np.uint8)),
            "rgba at another resolution": (src, low, np.full((48, 64, 4), 77, np.uint8)),
            "non-contiguous source": (np.zeros((48, 64, 4), np.uint8)[..., :3], low,
                                      np.full((24, 32, 4), 77, np.uint8)),
            "float source": (np.zeros((48, 64, 3), np.float32), low,
                             np.full((24, 32, 4), 77, np.uint8)),
        }
        for label, (s, bgr, rgba) in bad_cases.items():
            with pytest.raises(residual_native.ResidualBridgeError):
                backend.prepare_working_color_rgba(s, bgr, rgba)
            assert np.all(bgr == 77) and np.all(rgba == 77), (
                f"{label}: a rejected call wrote one of the caller's buffers")
        after = backend.diagnostics()
        assert after["prepare_bgr_rgba_call_count"] == before["prepare_bgr_rgba_call_count"], \
            "a rejected fused call was counted as a store"
        assert after["py_fused_prepare_call_count"] == before["py_fused_prepare_call_count"]
        assert after["fused_rgba_store_bytes"] == before["fused_rgba_store_bytes"]
        assert after["downsample_call_count"] == before["downsample_call_count"]

        closed = _native(16, 16, 8, 8)
        closed.close()
        with pytest.raises(residual_native.ResidualBridgeError):
            closed.prepare_working_color_rgba(np.zeros((16, 16, 3), np.uint8),
                                              np.zeros((8, 8, 3), np.uint8),
                                              np.zeros((8, 8, 4), np.uint8))
    finally:
        backend.close()


def test_fused_prepare_raises_instead_of_falling_back(monkeypatch):
    """Asked for the fused store on a bridge that cannot do it, the caller gets an error.

    The alternative is the bug this whole phase is trying to remove: an AttributeError that
    quietly runs the two-pass path, so the diagnostics keep reporting a frame count while
    the BGR-to-RGBA conversion the phase was supposed to delete is still there.
    """
    assert residual_native.capabilities()["prepare_bgr_rgba"], (
        "this test needs a bridge that can do the fused store, otherwise it proves nothing")
    monkeypatch.setitem(residual_native._CAPABILITIES, "prepare_bgr_rgba", False)
    backend = _native(64, 48, 32, 24)
    try:
        before = backend.diagnostics()
        with pytest.raises(residual_native.ResidualBridgeError) as caught:
            backend.prepare_working_color_rgba(np.zeros((48, 64, 3), np.uint8),
                                               np.zeros((24, 32, 3), np.uint8),
                                               np.zeros((24, 32, 4), np.uint8))
        assert "rr_prepare_working_bgr_rgba8" in str(caught.value)
        after = backend.diagnostics()
        assert after["prepare_bgr_rgba_call_count"] == before["prepare_bgr_rgba_call_count"]
        assert after["downsample_call_count"] == before["downsample_call_count"], \
            "the refused call still reached the library"
    finally:
        backend.close()


def test_rgba_reconstruct_refuses_a_plane_it_cannot_read_as_rgba():
    """A three-channel plane is not an RGBA plane, and reading one as such is the mistake
    this entry point could make most expensively: the stride is wrong by a quarter, so every
    pixel after the first would take its colour from its neighbour's alpha byte.
    """
    backend = _native(40, 40, 20, 20)
    source = np.zeros((40, 40, 3), np.uint8)
    low = np.zeros((20, 20, 3), np.uint8)
    out = np.full((40, 40, 3), 77, np.uint8)
    wide = np.zeros((20, 20, 8), np.uint8)[:, :, :4]
    try:
        before = backend.diagnostics()
        for bad in (np.zeros((20, 20, 3), np.uint8),
                    wide,
                    np.zeros((20, 20, 4), np.float32),
                    np.zeros((21, 21, 4), np.uint8)):
            with pytest.raises(residual_native.ResidualBridgeError):
                backend.reconstruct_from_rgba(source, low, bad, 1.0, out)
        assert np.all(out == 77), "a rejected call wrote the output"
        after = backend.diagnostics()
        assert after["reconstruct_rgba_input_call_count"] == \
            before["reconstruct_rgba_input_call_count"], \
            "a rejected call was counted as a reconstruct"
        assert after["reconstruct_call_count"] == before["reconstruct_call_count"], \
            "a rejected call still reached the library"
    finally:
        backend.close()


def test_rgba_reconstruct_raises_instead_of_falling_back(monkeypatch):
    """A bridge without the direct read says so; it does not convert and carry on.

    Silently falling back here would keep the RGBA-to-BGR conversion alive while the
    diagnostics still counted a reconstruct, which is exactly the outcome the reduced path
    is meant to make impossible.
    """
    assert residual_native.capabilities()["reconstruct_rgba"], (
        "this test needs a bridge that can read RGBA, otherwise it proves nothing")
    monkeypatch.setitem(residual_native._CAPABILITIES, "reconstruct_rgba", False)
    backend = _native(64, 48, 32, 24)
    try:
        before = backend.diagnostics()
        with pytest.raises(residual_native.ResidualBridgeError) as caught:
            backend.reconstruct_from_rgba(np.zeros((48, 64, 3), np.uint8),
                                          np.zeros((24, 32, 3), np.uint8),
                                          np.zeros((24, 32, 4), np.uint8), 1.0,
                                          np.zeros((48, 64, 3), np.uint8))
        assert "rr_reconstruct_from_rgba8" in str(caught.value)
        assert backend.rgba_reconstruct_call_count == 0, \
            "the Python side counted a call the library never ran"
        after = backend.diagnostics()
        assert after["reconstruct_rgba_input_call_count"] == \
            before["reconstruct_rgba_input_call_count"]
        assert after["reconstruct_call_count"] == before["reconstruct_call_count"], \
            "the refused call still reached the library"
    finally:
        backend.close()


# ------------------------------------------------------ resource reuse (section 15) ----

def test_native_reuses_plan_and_scratch_across_ten_frames():
    backend = _native(64, 48, 32, 24)
    try:
        before = backend.diagnostics()
        rng = np.random.default_rng(4)
        source = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
        low = np.empty((24, 32, 3), dtype=np.uint8)
        recon = np.empty((48, 64, 3), dtype=np.uint8)
        for _ in range(10):
            backend.prepare_working_color(source, low)
            backend.reconstruct(source, low, low, 1.0, recon)
        after = backend.diagnostics()
    finally:
        backend.close()

    assert after["process_plan_create_total"] == before["process_plan_create_total"], \
        "a plan was built mid-run"
    assert after["process_scratch_alloc_total"] == before["process_scratch_alloc_total"], \
        "scratch was reallocated"
    assert after["process_scratch_bytes_total"] == before["process_scratch_bytes_total"]
    assert after["process_worker_create_total"] == before["process_worker_create_total"]
    assert after["this_plan_create_count"] == before["this_plan_create_count"] == 1
    assert after["this_plan_scratch_alloc_count"] == before["this_plan_scratch_alloc_count"] == 1
    assert after["native_scratch_bytes"] == before["native_scratch_bytes"]
    assert after["downsample_call_count"] == before["downsample_call_count"] + 10
    assert after["reconstruct_call_count"] == before["reconstruct_call_count"] + 10
    assert after["downsample_pixel_count"] == before["downsample_pixel_count"] + 10 * 32 * 24
    assert after["reconstruct_pixel_count"] == before["reconstruct_pixel_count"] + 10 * 64 * 48


def test_fused_prepare_reuses_plan_and_scratch():
    """Ten frames through the fused call: one plan, one scratch, and the counters say so.

    The per-plan counters come from inside the library, so they report which of the two
    preparation entry points actually ran. The BGR-only half of the test is what makes that
    a measurement rather than a tautology: a single counter that went up for both calls
    could not tell the reduced path from the two-pass one.
    """
    backend = _native(64, 48, 32, 24)
    rng = np.random.default_rng(4)
    source = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    try:
        before = backend.diagnostics()
        low = np.empty((24, 32, 3), dtype=np.uint8)
        rgba = np.empty((24, 32, 4), dtype=np.uint8)
        for _ in range(10):
            backend.prepare_working_color_rgba(source, low, rgba)
        fused = backend.diagnostics()
        backend.prepare_working_color(source, low)
        after = backend.diagnostics()
    finally:
        backend.close()

    for step, label in ((fused, "after ten fused calls"),
                        (after, "after one more BGR-only call")):
        assert step["process_plan_create_total"] == before["process_plan_create_total"], \
            f"{label}: a plan was built mid-run"
        assert step["process_scratch_alloc_total"] == before["process_scratch_alloc_total"], \
            f"{label}: scratch was reallocated"
        assert step["process_scratch_bytes_total"] == before["process_scratch_bytes_total"], \
            f"{label}: scratch bytes changed"
        assert step["process_worker_create_total"] == before["process_worker_create_total"], \
            f"{label}: a worker thread was started"
        assert step["this_plan_create_count"] == 1, f"{label}: this plan was rebuilt"
        assert step["this_plan_scratch_alloc_count"] == 1, f"{label}: this plan reallocated"
        assert step["native_scratch_bytes"] == before["native_scratch_bytes"]

    assert fused["prepare_bgr_rgba_call_count"] == before["prepare_bgr_rgba_call_count"] + 10
    assert fused["prepare_bgr_only_call_count"] == before["prepare_bgr_only_call_count"], \
        "a fused call was counted as a BGR-only preparation"
    assert fused["fused_rgba_store_bytes"] == before["fused_rgba_store_bytes"] + 10 * 24 * 32 * 4
    assert fused["downsample_call_count"] == before["downsample_call_count"] + 10
    assert fused["downsample_pixel_count"] == before["downsample_pixel_count"] + 10 * 32 * 24
    assert fused["py_fused_prepare_call_count"] == before["py_fused_prepare_call_count"] + 10

    assert after["prepare_bgr_only_call_count"] == fused["prepare_bgr_only_call_count"] + 1
    assert after["prepare_bgr_rgba_call_count"] == fused["prepare_bgr_rgba_call_count"]
    assert after["fused_rgba_store_bytes"] == fused["fused_rgba_store_bytes"], \
        "the two-pass path reported an RGBA store it did not write"
    assert after["downsample_call_count"] == fused["downsample_call_count"] + 1


def test_close_releases_the_plan():
    """Destroying one plan moves only the process total, and its handle stops resolving.

    The per-plan block cannot show this: a plan that is gone answers no more queries, so
    the count that has to move is the library-wide one.
    """
    keep_open = _native(16, 16, 8, 8)
    doomed = _native(32, 32, 16, 16)
    doomed_handle = doomed._handle.value
    try:
        before = keep_open.diagnostics()["process_plan_destroy_total"]
        assert keep_open.diagnostics()["this_plan_destroy_count"] == 0, \
            "another plan's destroy leaked into this plan's counters"
        doomed.close()
        after = keep_open.diagnostics()
        assert after["process_plan_destroy_total"] == before + 1
        assert after["this_plan_destroy_count"] == 0
        doomed.close()          # idempotent, does not double count
        assert keep_open.diagnostics()["process_plan_destroy_total"] == before + 1

        lib = residual_native._library()
        counters = residual_native._Counters()
        assert lib.rr_get_counters(doomed_handle, ctypes.byref(counters)) == -5, \
            "a destroyed handle still resolves"
        src = np.zeros((32, 32, 3), np.uint8)
        dst = np.zeros((16, 16, 3), np.uint8)
        assert lib.rr_downsample_bgr8(
            doomed_handle,
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 32, 32, 96,
            dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 16, 16, 48) == -5
    finally:
        keep_open.close()


# ---------------------------------------------------------- parallel safety ----

def test_native_parallel_output_is_deterministic_and_matches_reference():
    """Row-parallel kernels must not leak a chunk's scratch into another chunk's rows.

    The same 4K-shaped case is run repeatedly: every run has to agree with the Python
    reference bit for bit, which a shared-accumulator or missing-barrier bug breaks.
    """
    src_w, src_h, live_w, live_h = 640, 360, 320, 180
    rng = np.random.default_rng(77)
    source = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)
    low = rng.integers(0, 256, size=(live_h, live_w, 3), dtype=np.uint8)
    dlss = rng.integers(0, 256, size=(live_h, live_w, 3), dtype=np.uint8)

    python = _python(src_w, src_h, live_w, live_h)
    want_down = np.zeros((live_h, live_w, 3), np.uint8)
    want_recon = np.zeros((src_h, src_w, 3), np.uint8)
    python.prepare_working_color(source, want_down)
    python.reconstruct(source, want_down, dlss, 1.0, want_recon)
    python.close()

    native = _native(src_w, src_h, live_w, live_h)
    try:
        for attempt in range(12):
            got_down = np.full((live_h, live_w, 3), 5, np.uint8)
            got_recon = np.zeros((src_h, src_w, 3), np.uint8)
            native.prepare_working_color(source, got_down)
            native.reconstruct(source, got_down, dlss, 1.0, got_recon)
            assert np.array_equal(got_down, want_down), f"run {attempt} working color"
            assert np.array_equal(got_recon, want_recon), f"run {attempt} reconstruct"
    finally:
        native.close()


def test_native_handles_more_chunks_than_rows():
    """Tiny frames still split correctly when there are fewer rows than workers."""
    for live_w, live_h in ((2, 1), (4, 2), (8, 3)):
        src_w, src_h = live_w * 2, live_h * 2
        source = np.full((src_h, src_w, 3), 200, dtype=np.uint8)
        native = _native(src_w, src_h, live_w, live_h)
        python = _python(src_w, src_h, live_w, live_h)
        try:
            a = np.zeros((live_h, live_w, 3), np.uint8)
            b = np.zeros((live_h, live_w, 3), np.uint8)
            native.prepare_working_color(source, a)
            python.prepare_working_color(source, b)
            assert np.array_equal(a, b), f"{live_w}x{live_h}"
            assert np.unique(a).tolist() == [200], "DC must survive a 1-row chunk split"
        finally:
            native.close()
            python.close()
