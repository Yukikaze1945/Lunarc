"""Independent scalar reference for the resampler numeric tests.

Every function here is a direct transcription of the Magpie kernels in
``Magpie/src/Magpie.Core/DLSSNRFilter.cpp``, written with Python loops and double
precision so it shares no code path with ``pipeline.py``:

  DownsampleColorVertical / DownsampleColorHorizontal    -> scalar_lanczos2_downsample
  UpsampleResidualHorizontal / CompositeResidualVertical -> scalar_catmull_rom_upsample
  CompositeResidualVertical saturate()                   -> scalar_composite

Test-only. Never import this from production code, and never import production
resampler helpers here -- an oracle that calls the code under test proves nothing.
"""

import math

import numpy as np


def sinc(x):
    px = math.pi * x
    if abs(px) < 1e-5:
        return 1.0
    return math.sin(px) / px


def lanczos2(x):
    if abs(x) >= 2.0:
        return 0.0
    return sinc(x) * sinc(0.5 * x)


def catmull_rom(x):
    x = abs(x)
    if x < 1.0:
        return ((1.5 * x - 2.5) * x) * x + 1.0
    if x < 2.0:
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    return 0.0


def _clamp_index(i, last):
    return 0 if i < 0 else (last if i > last else i)


def scalar_lanczos2_downsample(src, dst_w, dst_h):
    """Vertical dispatch first, then horizontal, each pixel normalized by its own weight sum."""
    src_h, src_w = src.shape[:2]
    f = src.astype(np.float64)
    channels = src.shape[2]

    scale_y = dst_h / float(src_h)
    support_y = 2.0 / scale_y
    vert = np.zeros((dst_h, src_w, channels), dtype=np.float64)
    for y in range(dst_h):
        position = (y + 0.5) / scale_y - 0.5
        acc = np.zeros((src_w, channels), dtype=np.float64)
        total_weight = 0.0
        for i in range(int(math.ceil(position - support_y)), int(math.floor(position + support_y)) + 1):
            w = lanczos2((i - position) * scale_y)
            acc += w * f[_clamp_index(i, src_h - 1)]
            total_weight += w
        vert[y] = acc / (total_weight if abs(total_weight) > 1e-6 else 1.0)

    scale_x = dst_w / float(src_w)
    support_x = 2.0 / scale_x
    out = np.zeros((dst_h, dst_w, channels), dtype=np.float64)
    for x in range(dst_w):
        position = (x + 0.5) / scale_x - 0.5
        acc = np.zeros((dst_h, channels), dtype=np.float64)
        total_weight = 0.0
        for i in range(int(math.ceil(position - support_x)), int(math.floor(position + support_x)) + 1):
            w = lanczos2((i - position) * scale_x)
            acc += w * vert[:, _clamp_index(i, src_w - 1)]
            total_weight += w
        out[:, x] = acc / (total_weight if abs(total_weight) > 1e-6 else 1.0)
    return out


def scalar_catmull_rom_upsample(low, dst_w, dst_h):
    """Horizontal axis first, then vertical. Signed throughout, never clamped."""
    low_h, low_w = low.shape[:2]
    f = low.astype(np.float64)
    channels = low.shape[2]

    horiz = np.zeros((low_h, dst_w, channels), dtype=np.float64)
    for x in range(dst_w):
        position = (x + 0.5) * low_w / float(dst_w) - 0.5
        center = math.floor(position)
        acc = np.zeros((low_h, channels), dtype=np.float64)
        total_weight = 0.0
        for offset in (-1, 0, 1, 2):
            index = center + offset
            w = catmull_rom(position - index)
            acc += w * f[:, _clamp_index(index, low_w - 1)]
            total_weight += w
        horiz[:, x] = acc / (total_weight if abs(total_weight) > 1e-6 else 1.0)

    out = np.zeros((dst_h, dst_w, channels), dtype=np.float64)
    for y in range(dst_h):
        position = (y + 0.5) * low_h / float(dst_h) - 0.5
        center = math.floor(position)
        acc = np.zeros((dst_w, channels), dtype=np.float64)
        total_weight = 0.0
        for offset in (-1, 0, 1, 2):
            index = center + offset
            w = catmull_rom(position - index)
            acc += w * horiz[_clamp_index(index, low_h - 1), :]
            total_weight += w
        out[y] = acc / (total_weight if abs(total_weight) > 1e-6 else 1.0)
    return out


def scalar_composite(source, residual_full, strength):
    """The single clamp of the chain: source + signed residual * strength -> uint8."""
    combined = source.astype(np.float64) + residual_full * strength
    return np.rint(np.clip(combined, 0.0, 255.0)).astype(np.uint8)


def quantized(values):
    """The uint8 result a shader would store after saturate()."""
    return np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)


# ------------------------------------------------------- deterministic inputs ----

def ramp(h, w):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :, 0] = (np.arange(h * w) % 251).reshape(h, w)
    a[:, :, 1] = (np.arange(h * w) // 3 % 251).reshape(h, w)
    a[:, :, 2] = (np.arange(h * w) * 7 % 251).reshape(h, w)
    return a


def impulse(h, w):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[h // 2, w // 2] = 255
    return a


def edge_impulse(h, w):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, w // 2:] = 255
    return a


def checkerboard(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.where(((xx // 2) + (yy // 2)) % 2 == 0, 240, 12).astype(np.uint8)
    return np.repeat(a[:, :, None], 3, axis=2)


def deterministic_random(h, w):
    rng = np.random.default_rng(20260919)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def signed_residual(h, w, kind, seed=7):
    """Signed low-resolution residuals: positive, negative or mixed."""
    rng = np.random.default_rng(seed)
    base = rng.integers(-60, 61, size=(h, w, 3))
    if kind == "positive":
        return np.abs(base)
    if kind == "negative":
        return -np.abs(base)
    return base


IMAGE_INPUTS = (ramp, impulse, edge_impulse, checkerboard, deterministic_random)
