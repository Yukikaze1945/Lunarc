"""Independent reimplementation of the stage-A display proxy, for auditing it.

Deliberately shares no code with ``hdr_proxy_math.py``: no import, no reused
helpers. The chain below is written from the stated specification in float64
with its own knee, transfer function and binary16 rounding, so a defect in the
audited module cannot cancel itself out here. This is what makes the comparison
an oracle rather than a self-check: ``hdr_proxy_math`` is the implementation
under test, and this file is a second, separately derived opinion about what
its numbers must be.

Two things are reproduced here and nothing else:

``oracle_code``
    scene value -> bounded display proxy code on the binary16 grid.

``oracle_slope_per_stop``
    the same central difference in stops that ``code_slope_per_stop`` claims to
    measure, taken through ``oracle_code``.

Both are scalar functions; callers loop. Exact float64 arithmetic is used up to
the final rounding, which is why a one-grid-step disagreement with the audited
float32 module near a rounding boundary is expected and tolerated, while a
difference in the *relative value being perturbed* -- the defect this file was
written for -- is not.
"""
from __future__ import annotations

import math

import numpy as np

Half = np.float16

KNEE = 0.75
SPAN = 0.25
SHOULDER_K = 5.770780
PAPER_FLOOR = 0.05
HALF_STEP_BELOW_ONE = 2.0 ** -11


def round_to_half(value) -> float:
    """Round a real number to the nearest-even binary16 value."""
    return float(Half(np.float32(value)))


def relative_of(scene, paper_white: float) -> float:
    x = float(scene)
    if not math.isfinite(x) or x < 0.0:
        x = 0.0
    return x / max(float(paper_white), PAPER_FLOOR)


def oracle_code(scene, paper_white: float) -> float:
    """Bounded proxy code of a linear scene value, on the half grid."""
    x = relative_of(scene, paper_white)
    if x > KNEE:
        y = KNEE + SPAN * (1.0 - math.exp(-SHOULDER_K * (x - KNEE)))
    else:
        y = x
    y = min(max(y, 0.0), 1.0)
    if y <= 0.0031308:
        encoded = 12.92 * y
    else:
        encoded = 1.055 * y ** (1.0 / 2.4) - 0.055
    return round_to_half(min(max(encoded, 0.0), 1.0))


def oracle_slope_per_stop(scene, paper_white: float,
                          step: float = 1.0 / 64.0) -> float:
    """Central difference of the proxy code with respect to stops of scene.

    A code at the bounded ceiling has no magnitude left to carry, so its slope
    is taken as zero: with only the lower arm free to move, a bare central
    difference would report the shape of the saturation cliff as if it were a
    signal. That follows from what a bounded proxy *is*, and is derived here
    rather than copied.
    """
    base = oracle_code(scene, paper_white)
    if base == 1.0:
        return 0.0
    factor = 2.0 ** step
    high = oracle_code(float(scene) * factor, paper_white)
    low = oracle_code(float(scene) / factor, paper_white)
    return (high - low) / (2.0 * step)


def slope_tolerance(scene, paper_white: float) -> float:
    """One binary16 step of disagreement, expressed as slope.

    The audited module evaluates the chain in float32; a value that lands
    within a float32 epsilon of a half-grid midpoint may round the other way
    here. One grid step on each side of the central difference, over the
    2 * step denominator, is therefore the largest honest disagreement.
    """
    code = oracle_code(scene, paper_white)
    ulp = float(Half(np.nextafter(np.float32(code), np.float32(1e30),
                                  dtype=np.float32))) - code
    if not math.isfinite(ulp) or ulp <= 0.0:
        ulp = HALF_STEP_BELOW_ONE
    return 2.0 * ulp / (2.0 * (1.0 / 64.0))


def analytic_slope_per_stop(scene, paper_white: float) -> float:
    """Exact-arithmetic derivative, unquantized, for the sanity bound only."""
    x = max(relative_of(scene, paper_white), 1e-12)
    if x > KNEE:
        remaining = SPAN * math.exp(-SHOULDER_K * (x - KNEE))
        y = 1.0 - remaining
        dy_dx = SHOULDER_K * remaining
    else:
        y, dy_dx = x, 1.0
    if y <= 0.0031308:
        dcode_dy = 12.92
    else:
        dcode_dy = 1.055 * y ** (1.0 / 2.4 - 1.0) / 2.4
    return math.log(2.0) * x * dy_dx * dcode_dy


def saturation_relative() -> float:
    """Relative value from which every code pins at 1.0, derived independently.

    Rounding happens *after* the transfer function, so the boundary that
    matters is the encoded midpoint between 1.0 and the half value below it,
    pulled back through the transfer function into the bounded linear domain.
    """
    boundary = 1.0 - HALF_STEP_BELOW_ONE / 2.0
    linear = ((boundary + 0.055) / 1.055) ** 2.4
    return KNEE + math.log(SPAN / (1.0 - linear)) / SHOULDER_K


def enumerate_half_grid(scene_max: float = 65504.0):
    """Every positive finite binary16 magnitude, as float32."""
    bits = np.arange(1, 0x7C00, dtype=np.uint16)
    grid = bits.view(Half).astype(np.float32)
    return grid[(grid > 0.0) & np.isfinite(grid) & (grid <= scene_max)]


if __name__ == "__main__":
    import json
    print(json.dumps({"relative_saturation_point": saturation_relative()}))
