"""Fixed HDR-to-SDR display transform; never used for processing or export."""
import numpy as np

# scRGB uses 80 nit per unit. Use a fixed 203-nit display reference rather than
# treating every value above SDR's 80-nit white as almost-white in the viewport.
HDR_DISPLAY_REFERENCE_SCRGB = 203.0 / 80.0


def hdr_to_sdr_linear(rgb):
    values = np.nan_to_num(np.asarray(rgb, dtype=np.float32), nan=0.,
                           posinf=125., neginf=0.)
    luminance = np.maximum(values @ np.array([.2126, .7152, .0722], np.float32), 0.)
    # Luminance shoulder, same scale for all channels. Fixed exposure across
    # frames preserves temporal brightness; clipping is display gamut only.
    return values / (HDR_DISPLAY_REFERENCE_SCRGB + luminance[..., None])
