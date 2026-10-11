"""Bounded display-proxy math and HDR reconstruction prototype (stage A).

Pure numbers: no GPU, no NGX, no production import path. Everything here works
on the binary16 grid because that is the grid the isolated probe's FP16 color
texture actually holds.

The proxy (arm B)
-----------------
A scene-referred linear value ``v`` (scRGB, paper white at 1.0 by convention)
is first put in *paper-white-relative* units, ``x = v / paperWhite``. That is
the only place the display's diffuse white enters; it is a session constant,
never a per-frame measurement.

Below ``KNEE`` the curve is the identity: relative values keep their ratio, so
nothing in the SDR range is compressed and the mapping is exactly invertible.

Above the knee an exponential shoulder takes over and climbs the remaining
``SHOULDER_SPAN`` with a decaying rate. In exact arithmetic it never reaches
1.0, so the map is strictly monotone and injective over the whole nonnegative
half line -- that is the whole point of a *soft* shoulder: it is bounded
without clipping. The value is then put through the sRGB electrical transfer
and rounded to the half grid.

Rounding is where information actually dies, and it happens *after* the sRGB
encode, so the threshold has to be derived in the encoded domain. The binary16
value below 1.0 is ``1 - 2**-11`` and the round-to-nearest boundary they share
is ``1 - 2**-12``; any encoded value at or above it rounds up to the identical
code 1.0. Pulling that boundary back through the transfer function gives the
bounded-linear ceiling ``1 - srgb_decode(1 - 2**-12)``, about 5.553e-4, and the
shoulder leaves ``SHOULDER_SPAN * exp(-SHOULDER_K * (x - KNEE))`` of headroom,
so the proxy pins at 1.0 from

    ``x = KNEE + log(SHOULDER_SPAN / ceiling) / SHOULDER_K`` = 1.80873 whites.

Solving the same condition in the bounded-linear domain instead -- treating
``1 - y`` as if it were the grid step -- would put the ceiling at 1.9511 whites
and miss the ~7% of the scene below it that is already unrecoverable, because
the encode expands the last sliver of linear headroom by 1/(2.4) of its slope.
Above the threshold the proxy is a hard constant, not a soft one.
``code_slope_per_stop`` measures the surviving slope on the quantized grid, so
it reports that plateau as exactly zero rather than as the tiny
exact-arithmetic derivative.

Reconstruction
--------------
Neither reconstructor inverts the proxy. A saturated code carries no
highlight magnitude, and feeding it to an inverse would invent one. Instead the
pristine HDR master supplies the magnitude and the network supplies only a
signed *change* measured in the proxy domain, which is converted to a relative
stop offset by the local quantized slope and applied multiplicatively. A zero
change therefore reproduces the master bit-exactly -- that is the identity
gate -- a change measured where the code cannot resolve a stop is dropped
outright instead of amplified, and an applied change can never move further
than ``MAX_STOPS``, which is a stated budget rather than a property of the
proxy.
"""
from __future__ import annotations

import numpy as np

Half = np.float16

KNEE = 0.75
SHOULDER_SPAN = 0.25
# Knee steepness of the audited reference shape: 4 * log2(e), which makes the
# shoulder leave the knee at slope SHOULDER_SPAN * SHOULDER_K = log2(e).
SHOULDER_K = 5.770780
PAPER_FLOOR = 0.05
# Stated reconstruction budget: a proxy-domain change may move a channel by at
# most this many stops in either direction.
MAX_STOPS = 1.0
# Binary16 relative quantum at unity: the largest error any half rounding
# inside this module can introduce into a value near 1.0.
HALF_ULP_AT_ONE = 2.0 ** -11
# A proxy code that does not advance by at least one binary16 step when the
# scene moves a full stop cannot carry a measured correction: anything read
# there is the rounding of the code itself, not a signal from the network.
MIN_SLOPE_PER_STOP = HALF_ULP_AT_ONE
SLOPE_STOP_STEP = 1.0 / 64.0


def to_half(values) -> np.ndarray:
    """Round float32 (or exact half) samples onto the binary16 grid, RNE."""
    return np.asarray(values, dtype=np.float32).astype(Half)


def finite_nonnegative(values) -> np.ndarray:
    """NaN, infinity and negative scRGB become 0; magnitudes are float32."""
    v = np.asarray(values, dtype=np.float32)
    return np.maximum(np.where(np.isfinite(v), v, np.float32(0.0)), 0.0)


def _relative(values, paper_white: float) -> np.ndarray:
    return finite_nonnegative(values) / float(max(paper_white, PAPER_FLOOR))


def paper_relative(values, paper_white: float) -> np.ndarray:
    """Scene value in paper-white-relative units, before any shoulder."""
    return _relative(values, paper_white)


def shoulder(relative) -> np.ndarray:
    """Map relative values onto [0, 1) with a soft exponential knee."""
    x = np.asarray(relative, dtype=np.float32)
    soft = KNEE + SHOULDER_SPAN * (1.0 - np.exp(
        -SHOULDER_K * np.maximum(x - KNEE, np.float32(0.0))))
    return np.clip(np.where(x > KNEE, soft, x), 0.0, 1.0)


def srgb_encode(value) -> np.ndarray:
    v = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v,
                    1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def srgb_decode(code) -> np.ndarray:
    """Electrical code back to the *bounded* linear proxy domain, [0, 1]."""
    v = np.clip(np.asarray(code, dtype=np.float32), 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92,
                    np.power((v + 0.055) / 1.055, 2.4))


def proxy_code(scene, paper_white: float) -> np.ndarray:
    """Bounded display proxy of a linear HDR scene value, on the half grid."""
    coded = srgb_encode(shoulder(_relative(scene, paper_white)))
    return to_half(coded)


def saturation_relative(paper_white: float = 1.0) -> float:
    """Relative value from which every code rounds to exactly 1.0.

    The rounding that destroys magnitude is the one applied to the *encoded*
    value, so the ceiling is the encoded binary16 midpoint adjacent to unity,
    ``1 - 2**-12``, decoded back into the bounded linear domain -- not a linear
    half step. Returned in relative units, which is why ``paper_white`` only
    documents the unit and does not enter the value.
    """
    boundary = np.float64(1.0) - np.float64(HALF_ULP_AT_ONE) / 2.0
    ceiling = 1.0 - ((boundary + 0.055) / 1.055) ** np.float64(2.4)
    return float(KNEE + np.log(np.float64(SHOULDER_SPAN) / ceiling)
                 / np.float64(SHOULDER_K))


def code_slope_per_stop(scene, paper_white: float,
                        step: float = SLOPE_STOP_STEP) -> np.ndarray:
    """Measured d(code)/d(log2 scene) on the quantized grid.

    Central difference in *stops*, so a flat section of the shoulder reports a
    true zero instead of a vanishing analytic derivative. The perturbation is
    applied to the scene value itself and every arm then goes through
    ``proxy_code``, which does the paper-white normalization: normalizing first
    and re-normalizing inside the proxy would divide by ``paper_white`` twice
    and report the slope of a scene ``paper_white`` times darker than the one
    being measured. Because a stop is a ratio, the derivative with respect to
    log2 scene and log2 paper-white-relative scene are the same number.

    A channel whose own code already sits at the bounded ceiling reports zero
    regardless of the difference. There the central difference is measuring the
    cliff, not a signal: the upper arm is pinned and only the lower arm moves,
    so an uncorrected difference hands a nonzero slope to exactly the values
    that cannot carry any magnitude at all.
    """
    scene = finite_nonnegative(scene)
    base = proxy_code(scene, paper_white)
    factor = np.float32(np.exp2(step))
    high = proxy_code(scene * factor, paper_white)
    low = proxy_code(scene / factor, paper_white)
    slope = (high.astype(np.float32) - low.astype(np.float32)) / np.float32(
        2.0 * step)
    return np.where(base == Half(1.0), np.float32(0.0), slope)


# --- HDR reconstruction ---------------------------------------------------

REC_LIN = "local_linear_stops"
REC_LUMA = "luminance_ratio"
LUMA_WEIGHTS = np.array([0.212639, 0.715169, 0.072192], dtype=np.float32)


def _luma(values) -> np.float32:
    """Rec.709 weights over the last axis; a scalar gray is its own luminance."""
    v = np.asarray(values, dtype=np.float32)
    if v.ndim and v.shape[-1] == 3:
        return np.dot(v, LUMA_WEIGHTS)
    return v


def _stops(delta, slope):
    """Signed proxy-domain change to signed stops, three explicit dispositions.

    A region the quantized proxy cannot resolve (measured slope below one half
    step per stop) contributes no correction at all: there the code carries
    rounding, not a signal, and inverting through it would manufacture HDR
    magnitude the proxy never held. An applied correction keeps its sign. A
    correction beyond the stated budget is clipped to it and flagged. Returns
    (stops, gain_cap_hit, rejected_noise).
    """
    slope = np.asarray(slope, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    informative = np.abs(slope) >= MIN_SLOPE_PER_STOP
    raw = np.divide(delta, np.where(informative, slope, np.float32(1.0)),
                    out=np.zeros_like(delta), where=informative)
    applied = informative & np.isfinite(raw)
    stops = np.where(applied, np.clip(raw, -MAX_STOPS, MAX_STOPS), 0.0)
    cap_hit = applied & (np.abs(raw) > MAX_STOPS)
    rejected = (~applied) & (delta != 0.0)
    return stops.astype(np.float32), cap_hit, rejected


def reconstruct_local_linear(master, input_proxy, neural_proxy,
                             paper_white: float) -> dict:
    """Per-channel signed correction through the capped local gain.

    Carries chroma change because every channel moves on its own; the master
    magnitude is never replaced by an inverse of the proxy.
    """
    master = finite_nonnegative(master)
    neural = np.clip(np.asarray(neural_proxy, dtype=np.float32), 0.0, 1.0)
    anchor = np.clip(np.asarray(input_proxy, dtype=np.float32), 0.0, 1.0)
    delta = neural - anchor
    slope = code_slope_per_stop(master, paper_white)
    stops, cap_hit, rejected = _stops(delta, slope)
    out = master * np.exp2(stops).astype(np.float32)
    return {"method": REC_LIN, "hdr": to_half(out), "delta_code": delta,
            "stops": stops, "gain_cap_hit": cap_hit,
            "rejected_code_noise": rejected, "slope_per_stop": slope}


def reconstruct_luminance_ratio(master, input_proxy, neural_proxy) -> dict:
    """Single explainable luminance ratio, applied to the master chroma.

    Both proxies are decoded to the *bounded* linear proxy domain first, so the
    ratio is a light-level ratio and the master keeps its own colour ratios.
    """
    master = finite_nonnegative(master)
    neural_y = _luma(srgb_decode(neural_proxy))
    input_y = _luma(srgb_decode(input_proxy))
    ratio = np.divide(neural_y, input_y, out=np.ones_like(neural_y),
                      where=input_y > 1e-6)
    stops, cap_hit, rejected = _stops(np.log2(np.maximum(ratio, 1e-6)),
                                      np.ones_like(ratio))
    out = master * np.exp2(stops).astype(np.float32)
    return {"method": REC_LUMA, "hdr": to_half(out), "ratio": ratio,
            "stops": stops, "gain_cap_hit": cap_hit,
            "rejected_code_noise": rejected}


def reconstruct(master, input_proxy, neural_proxy, paper_white: float,
                method: str) -> dict:
    if method == REC_LIN:
        return reconstruct_local_linear(master, input_proxy, neural_proxy,
                                       paper_white)
    if method == REC_LUMA:
        return reconstruct_luminance_ratio(master, input_proxy, neural_proxy)
    raise ValueError(f"unknown reconstruction method {method!r}")


def identity_gate(samples, paper_white: float, methods=(REC_LIN, REC_LUMA)):
    """Neural proxy == input proxy must return the master untouched."""
    rows = {}
    for method in methods:
        worst = 0.0
        exact = True
        for scene in samples:
            scene = np.asarray(scene, dtype=np.float32)
            anchor = proxy_code(scene, paper_white).astype(np.float32)
            out = reconstruct(scene, anchor, anchor, paper_white,
                              method)["hdr"].astype(np.float32)
            master = to_half(scene).astype(np.float32)
            gap = np.max(np.abs(out - master))
            rel = np.max(gap / np.maximum(np.abs(master), HALF_ULP_AT_ONE))
            worst = max(worst, float(rel))
            exact &= bool(np.array_equal(out, master))
        rows[method] = {"max_relative_error": worst,
                        "bit_exact": exact,
                        "tolerance_relative": HALF_ULP_AT_ONE,
                        "pass": bool(exact or worst <= HALF_ULP_AT_ONE)}
    return rows


def grid_analysis(paper_white: float, scene_max: float = 65504.0) -> dict:
    """How much of the half domain survives proxy rounding.

    Every positive finite binary16 magnitude is pushed through the proxy; the
    surviving code count tells us the real information budget, including the
    plateau where the whole highlight range collapses to one code. The
    analytic saturation point is checked against that enumeration in both
    directions rather than assumed: no level at or above it may escape the
    plateau, and no level below it may already be inside it.
    """
    bits = np.arange(1, 0x7C00, dtype=np.uint16)
    grid = bits.view(Half).astype(np.float32)
    grid = grid[(grid > 0.0) & np.isfinite(grid) & (grid <= scene_max)]
    codes = proxy_code(grid, paper_white).astype(np.float32)
    knee_scene = np.float32(KNEE * paper_white)
    sat_scene = np.float32(saturation_relative() * paper_white)
    above = codes[grid >= sat_scene]
    between = codes[(grid >= knee_scene) & (grid < sat_scene)]
    pinned = float(np.unique(above).size) if above.size else 0.0
    at_one = codes == 1.0
    first_pinned = float(grid[at_one].min()) if at_one.any() else None
    window = (grid >= knee_scene) & (grid < sat_scene)
    dead = window & (code_slope_per_stop(grid, paper_white) == 0.0)
    return {
        "paper_white": float(paper_white),
        "scene_levels_tested": int(grid.size),
        "distinct_codes_whole_domain": int(np.unique(codes).size),
        "distinct_codes_knee_to_saturation": int(np.unique(between).size)
        if between.size else 0,
        "scene_levels_above_knee": int((grid >= knee_scene).sum()),
        "relative_saturation_point": saturation_relative(),
        "scene_saturation_point": float(sat_scene),
        "codes_used_by_scene_above_saturation": pinned,
        "scene_levels_above_saturation": int((grid >= sat_scene).sum()),
        "highlight_bits_lost": float(np.log2(max(
            int((grid >= sat_scene).sum()), 1) / max(pinned, 1.0))),
        "first_grid_level_pinned_to_one": first_pinned,
        "first_grid_level_pinned_relative": None if first_pinned is None
        else first_pinned / paper_white,
        "levels_at_or_above_saturation_not_pinned": int(
            ((grid >= sat_scene) & ~at_one).sum()),
        "levels_below_saturation_pinned_to_one": int(
            ((grid < sat_scene) & at_one).sum()),
        # The quantized dead zone: codes still below the ceiling that a
        # 1/64-stop nudge cannot advance at all, so the guard refuses a
        # correction there even though magnitude technically survives.
        "levels_knee_to_saturation_tested": int(window.sum()),
        "levels_knee_to_saturation_dead": int(dead.sum()),
        "first_dead_level_above_knee_relative": None if not dead.any()
        else float(grid[dead].min() / paper_white),
        "last_dead_level_below_saturation_relative": None if not dead.any()
        else float(grid[dead].max() / paper_white),
    }


if __name__ == "__main__":
    import json

    scalars = [0.1, 0.5, 1.0, 1.5, 2.0, 4.0]
    colored = [[4.0, 0.5, 0.25], [1.5, 2.0, 0.5], [0.2, 0.8, 2.0],
               [0.5, 4.0, 1.5], [0.1, 1.0, 1.5]]
    report = {}
    for paper in (1.0, 4.0, 8.0):
        flat = [np.float32(v) for v in scalars] + \
               [np.array(c, dtype=np.float32) for c in colored]
        report[f"paper_{paper}"] = {
            "proxy_codes": {str(v): [float(x) for x in
                                     proxy_code(np.array([v], np.float32),
                                               paper)] for v in scalars},
            "identity": identity_gate(flat, paper),
            "grid": grid_analysis(paper),
        }
    print(json.dumps(report, indent=2, sort_keys=True))
