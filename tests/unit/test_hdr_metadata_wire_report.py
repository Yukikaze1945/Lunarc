"""Audit metadata quantization separately from picture processing."""
from copy import deepcopy
from fractions import Fraction

import gpu_hdr_transfer_export as htx


def metadata():
    return dict(has_mastering=1, mastering_has_primaries=1,
        mastering_has_luminance=1,
        primary_x_num=[11374617, 2055209, 9259681],
        primary_x_den=[16777216, 8388608, 67108864],
        primary_y_num=[675241, 11794383, 3489661],
        primary_y_den=[2097152, 16777216, 67108864],
        white_x_num=1311475, white_x_den=4194304,
        white_y_num=11038737, white_y_den=33554432,
        max_luminance_num=1000, max_luminance_den=1,
        min_luminance_num=0, min_luminance_den=1)


def test_container_floats_report_bounded_error_and_keep_original_rationals():
    source = metadata()
    original = deepcopy(source)
    report = htx.mastering_wire_report(source)
    assert source == original
    assert report['quantized_fields'] == 8
    assert [f['wire_units'] for f in report['fields']] == [
        33899, 16099, 12250, 35150, 6899, 2600, 15634, 16449, 10000000, 0]
    for field in report['fields']:
        difference = abs(Fraction(field['wire']) - Fraction(field['source']))
        assert Fraction(field['absolute_error']) == difference
        assert difference <= Fraction(1, 2 * field['wire_denominator'])


def test_exact_grid_values_report_no_change():
    source = metadata()
    source.update(primary_x_num=[34000, 13250, 26500], primary_x_den=[50000]*3,
        primary_y_num=[16000, 34500, 10000], primary_y_den=[50000]*3,
        white_x_num=15635, white_x_den=50000,
        white_y_num=16450, white_y_den=50000)
    report = htx.mastering_wire_report(source)
    assert report['quantized_fields'] == 0
    assert all(f['absolute_error'] == '0' for f in report['fields'])


def test_absent_or_partial_mastering_does_not_invent_payload_values():
    assert htx.mastering_wire_report(None)['fields'] == []
    source = metadata()
    source['mastering_has_luminance'] = 0
    assert htx.mastering_wire_report(source)['fields'] == []


def test_invalid_rational_is_reported_without_dividing_by_zero():
    source = metadata()
    source['white_x_den'] = 0
    fields = htx.mastering_wire_report(source)['fields']
    assert next(f for f in fields if f['field'] == 'white_x') == {
        'field': 'white_x', 'valid_rational': False}
