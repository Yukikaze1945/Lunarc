"""Gate: the stage-A proxy slope and saturation numbers must be their own.

Two defects were audited into ``native/hdr_proxy_math.py`` and this file is the
evidence against both. Neither was visible from the identity gate: with
``neural == input`` the delta is zero, so a wrong slope cancels out of
``0 / slope`` and the reconstruction comes back bit-exact anyway. That is what
made them dangerous - every recorded pass criterion stayed green while the
numbers that decide *how far a real correction moves a channel* were wrong.

1. ``code_slope_per_stop`` normalized the scene to paper-white units and then
   handed the normalized value back to ``proxy_code``, which normalizes again.
   The two arms of the central difference therefore measured a scene a factor
   ``paper_white`` darker than the one being asked about, so every stop, cap
   and plateau rejection at ``paperWhite`` 4.0 and 8.0 was computed from the
   wrong derivative. Worse, it reported *nonzero* slope inside the saturation
   plateau - at ``paperWhite`` 4.0 a scene of 8.0 (two relative whites, code
   pinned at 1.0) came back with a slope of 0.2188 - which is exactly the
   condition the noise guard exists to refuse.

2. ``saturation_relative`` solved the shoulder's *linear* value for a half-grid
   step, but the proxy rounds after the sRGB encode, where the grid is coarser
   per unit of linear headroom. The true ceiling is the encoded midpoint
   ``1 - 2**-12`` pulled back through the transfer function; the linear-domain
   answer (1.9511 whites) was ~7% too generous, calling recoverable scene
   levels that were already a single code.

Everything asserted below comes from ``native/hdr_proxy_slope_oracle.py``, a
second implementation written from the specification in float64 with its own
knee, transfer function and rounding. It imports nothing from the module under
test, so agreement here is not a tautology.
"""

import os
import sys

import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, os.path.join(_ROOT, "native")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hdr_proxy_math as proxy           # noqa: E402
import hdr_proxy_slope_oracle as oracle  # noqa: E402

# The scenes the stage-A gate actually measures, plus values straddling the
# knee and the plateau at each session constant.
SCENES = (0.0125, 0.1, 0.5, 0.75, 1.0, 1.49, 1.5, 1.6, 1.8,
          1.80859375, 1.8095703125, 2.0, 4.0, 8.0, 16.0, 64.0)
PAPER_WHITES = (0.05, 1.0, 4.0, 8.0, 16.0)
STALE_LINEAR_DOMAIN_SATURATION = 0.75 + np.log(1024.0) / 5.770780


def _module_slope(scene, paper_white):
    return float(np.asarray(
        proxy.code_slope_per_stop(np.float32(scene), paper_white),
        dtype=np.float64))


def test_measured_slope_equals_the_independent_finite_difference():
    """No tolerance: both sides quantize to the same half codes, exactly."""
    offenders = []
    for paper_white in PAPER_WHITES:
        for scene in SCENES:
            measured = _module_slope(scene, paper_white)
            expected = oracle.oracle_slope_per_stop(scene, paper_white)
            if measured != expected:
                offenders.append((paper_white, scene, measured, expected,
                                  oracle.slope_tolerance(scene, paper_white)))
    assert not offenders, (
        "code_slope_per_stop disagrees with the independently written central "
        f"difference: {offenders[:8]}")


def test_perturbation_is_applied_to_the_scene_not_to_a_second_normalized_copy():
    """A scene and its paper-white-scaled twin are the same relative value.

    ``proxy_code`` normalizes exactly once, so scene ``paperWhite * x`` at
    ``paperWhite`` must be scene ``x`` at 1.0 in every respect, slope included.
    The double division broke this by measuring the slope at ``x / paperWhite``
    instead, which is what made highlights look informative and shadows look
    dead.
    """
    for paper_white in (1.0, 4.0, 8.0):
        for scene in (0.1, 0.5, 1.0, 1.5, 2.0, 4.0):
            scaled = scene * paper_white
            assert _module_slope(scaled, paper_white) == pytest.approx(
                _module_slope(scene, 1.0), abs=0.0), (
                f"slope({scaled} @ pw {paper_white}) != slope({scene} @ pw 1.0):"
                " the paper-white normalization is being applied twice")


def test_plateau_reports_zero_slope_at_every_paper_white():
    """Above the saturation point the code is one constant, so slope is zero.

    A nonzero value here is the audited defect, and it is not cosmetic: it lets
    a rounding difference be divided into a fabricated stop offset.
    """
    saturation = oracle.saturation_relative()
    for paper_white in PAPER_WHITES:
        for scene in SCENES:
            relative = scene / max(paper_white, proxy.PAPER_FLOOR)
            measured = _module_slope(scene, paper_white)
            assert measured >= 0.0, (
                f"scene {scene} at paperWhite {paper_white}: the proxy is "
                f"monotone, so a negative measured slope means {measured}")
            if relative >= saturation:
                assert measured == 0.0, (
                    f"scene {scene} at paperWhite {paper_white} is "
                    f"{relative:.6f} relative whites, at or beyond saturation "
                    f"{saturation:.6f}, yet reports slope {measured}")
            if relative <= proxy.KNEE:
                assert measured > 0.0, (
                    f"scene {scene} at paperWhite {paper_white} is "
                    f"{relative:.6f} relative whites, at or below the knee "
                    "where the proxy is the identity, yet measures no slope")


def test_dead_zone_above_the_knee_is_measured_not_assumed():
    """Between the knee and the ceiling the code can stall without pinning.

    A 1/64-stop nudge crosses a half-grid boundary only when one lies inside
    the window, and near the top the boundaries are roughly 1/30 of a stop
    apart, so the reading alternates. The guard turns each of those into a
    rejection; the point of this test is that the extent is recorded rather
    than waved away.
    """
    reference = proxy.grid_analysis(1.0)
    assert reference["levels_knee_to_saturation_dead"] > 0
    assert reference["first_dead_level_above_knee_relative"] > proxy.KNEE
    assert reference["last_dead_level_below_saturation_relative"] \
        < reference["relative_saturation_point"]
    for paper_white in (4.0, 8.0):
        other = proxy.grid_analysis(paper_white)
        # The dead zone is a property of the relative value, so it must not
        # move with the session constant.
        assert other["first_dead_level_above_knee_relative"] == pytest.approx(
            reference["first_dead_level_above_knee_relative"], rel=1e-3)
        assert other["levels_knee_to_saturation_dead"] == \
            reference["levels_knee_to_saturation_dead"]


def test_saturation_threshold_is_the_encoded_boundary_not_a_linear_step():
    """1 - 2**-12 in the *encoded* domain, decoded back, is what pins the code."""
    assert proxy.saturation_relative() == pytest.approx(
        oracle.saturation_relative(), abs=1e-9)
    assert proxy.saturation_relative() < STALE_LINEAR_DOMAIN_SATURATION - 0.1, (
        "saturation_relative is back on the linear-domain formula "
        f"({STALE_LINEAR_DOMAIN_SATURATION:.4f}); the rounding under audit "
        "happens after the sRGB encode, not before it")


@pytest.mark.parametrize("paper_white", (1.0, 4.0, 8.0))
def test_saturation_threshold_is_verified_by_exhaustive_enumeration(paper_white):
    """Every half level at or above the threshold pins, and none below does."""
    grid = proxy.grid_analysis(paper_white)
    assert grid["levels_at_or_above_saturation_not_pinned"] == 0
    assert grid["levels_below_saturation_pinned_to_one"] == 0
    assert grid["codes_used_by_scene_above_saturation"] == 1.0
    # The first half level that actually pins is the smallest grid point at or
    # above the analytic threshold: the two derivations agree on the grid too.
    first = grid["first_grid_level_pinned_to_one"]
    assert first >= grid["scene_saturation_point"]
    below = oracle.enumerate_half_grid()
    below = below[below < first]
    assert proxy.proxy_code(below, paper_white).max() < 1.0


@pytest.mark.parametrize("paper_white", (0.05, 1.0, 4.0, 8.0))
def test_proxy_code_agrees_with_the_oracle_across_the_half_grid(paper_white):
    """At most one level may differ, and only by one adjacent code.

    The audited chain runs in float32 and the oracle in float64, so a scene
    value within a float32 epsilon of a rounding midpoint can legitimately land
    on the other side. Anything wider than adjacency is a real disagreement.
    """
    grid = oracle.enumerate_half_grid()
    module = proxy.proxy_code(grid, paper_white).astype(np.float64)
    expected = np.array([oracle.oracle_code(float(v), paper_white)
                         for v in grid], dtype=np.float64)
    step = np.maximum(np.abs(np.nextafter(expected.astype(np.float32),
                                         np.float32(2.0 ** 16),
                                         dtype=np.float32)
                             - expected.astype(np.float32)), 2.0 ** -13)
    difference = np.abs(module - expected)
    offenders = np.nonzero(difference > step)[0]
    assert offenders.size == 0, (
        f"{offenders.size} half levels differ from the oracle by more than one "
        "grid step, e.g. scene "
        f"{float(grid[offenders[0]]) if offenders.size else None}: "
        f"module {module[offenders[0]] if offenders.size else None} vs oracle "
        f"{expected[offenders[0]] if offenders.size else None}")
    assert int((difference > 0).sum()) <= 1, (
        "float32 and float64 chains disagree on more than the single "
        "rounding-midpoint level they are expected to straddle")


def test_identity_gate_cannot_catch_a_wrong_slope():
    """Why this file exists: the shipped gate stays green with a broken slope.

    Recorded so a later edit cannot claim the identity test covers the
    derivative. It does not, and only the oracle above does.
    """
    scenes = [np.float32(v) for v in (0.1, 1.0, 4.0)]
    rows = proxy.identity_gate(scenes, 8.0)
    for method in (proxy.REC_LIN, proxy.REC_LUMA):
        assert rows[method]["pass"], method
    # ... while a fabricated correction in the same place is caught: a scene
    # whose code is pinned at the ceiling, given a nonzero delta, must be
    # refused rather than divided by the cliff's own slope.
    scene = np.float32(4.0)
    anchor = proxy.proxy_code(scene, 1.0)
    assert float(anchor) == 1.0, "expected a pinned code for this scene"
    moved = anchor.astype(np.float32) - np.float32(2.0 ** -11)
    result = proxy.reconstruct_local_linear(scene, anchor, moved, 1.0)
    assert float(np.asarray(result["stops"])) == 0.0
    assert bool(np.asarray(result["rejected_code_noise"]))
    assert float(proxy.code_slope_per_stop(scene, 1.0)) == 0.0
