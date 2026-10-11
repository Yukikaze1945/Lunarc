"""Independent scalar checks for the BT.709 limited-range P010 colour path."""
import math


KR, KB = 0.2126, 0.0722
KG = 1.0 - KR - KB


def bt709_oetf(linear):
    return 4.5 * linear if linear < 0.018 else 1.099 * linear ** 0.45 - 0.099


def bt709_eotf(encoded):
    return encoded / 4.5 if encoded < 0.081 else ((encoded + 0.099) / 1.099) ** (1 / 0.45)


def rgb_to_yuv(rgb):
    r, g, b = (bt709_oetf(float(c)) for c in rgb)
    y = KR * r + KG * g + KB * b
    return y, (b - y) / (2 * (1 - KB)), (r - y) / (2 * (1 - KR))


def quantize_limited(y, cb, cr):
    return (round(max(0, min(1, y)) * 876 + 64),
            round(max(-0.5, min(0.5, cb)) * 896 + 512),
            round(max(-0.5, min(0.5, cr)) * 896 + 512))


def test_bt709_piecewise_transfer_roundtrips_reference_points():
    for linear in (0.0, 0.001, 0.018, 0.18, 0.5, 1.0):
        assert math.isclose(bt709_eotf(bt709_oetf(linear)), linear,
                            rel_tol=2e-14, abs_tol=2e-14)
    assert bt709_eotf(-0.045) == -0.01  # signed linear extension


def test_limited_range_black_white_and_neutral_codes():
    assert quantize_limited(*rgb_to_yuv((0.0, 0.0, 0.0))) == (64, 512, 512)
    assert quantize_limited(*rgb_to_yuv((1.0, 1.0, 1.0))) == (940, 512, 512)
    for level in (0.01, 0.18, 0.5):
        y, cb, cr = rgb_to_yuv((level, level, level))
        yc, cbc, crc = quantize_limited(y, cb, cr)
        assert 64 <= yc <= 940
        assert cbc == crc == 512


def test_chroma_reference_averages_oetf_encoded_pixels():
    # 4:2:0 chroma is averaged in the encoded RGB domain. This saturated pair
    # makes the nonlinear distinction visible and guards against averaging
    # linear RGB before applying the OETF.
    linear_pixels = ((1.0, 0.0, 0.0), (0.0, 0.5, 0.0),
                     (0.0, 0.0, 0.2), (0.18, 0.18, 0.18))
    encoded_rgb_mean = tuple(sum(bt709_oetf(p[i]) for p in linear_pixels) / 4
                             for i in range(3))
    y = KR * encoded_rgb_mean[0] + KG * encoded_rgb_mean[1] + KB * encoded_rgb_mean[2]
    expected = ((encoded_rgb_mean[2] - y) / (2 * (1 - KB)),
                (encoded_rgb_mean[0] - y) / (2 * (1 - KR)))
    linear_mean = tuple(sum(p[i] for p in linear_pixels) / 4 for i in range(3))
    wrong_rgb = tuple(bt709_oetf(v) for v in linear_mean)
    wrong_y = KR * wrong_rgb[0] + KG * wrong_rgb[1] + KB * wrong_rgb[2]
    wrong = ((wrong_rgb[2] - wrong_y) / (2 * (1 - KB)),
             (wrong_rgb[0] - wrong_y) / (2 * (1 - KR)))
    assert max(abs(a - b) for a, b in zip(expected, wrong)) > 0.01
    assert quantize_limited(y, *expected)[1:] == tuple(
        round(max(-0.5, min(0.5, c)) * 896 + 512) for c in expected)
