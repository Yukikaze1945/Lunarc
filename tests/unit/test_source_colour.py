"""Untagged SDR10 must stay P010; colour defaults never erase declarations."""
import json
from types import SimpleNamespace

import pytest
import gpu_transfer_export as gtx
import pipeline
from source_colour import resolve_source_colour, resolve_frame_colour


def source(**patch):
    return dict(pix_fmt='yuv420p10le', width=1920, height=1080, color_range='tv',
                color_transfer=None, color_space=None, color_primaries=None, **patch)


def test_hd_untagged_resolution_is_recorded_and_does_not_mutate_source():
    raw = source()
    resolved = resolve_source_colour(raw)
    assert all(resolved[name] == 'bt709' for name in ('color_space', 'color_transfer', 'color_primaries'))
    assert resolved['colour_resolution']['source_declared']['color_transfer'] is None
    assert raw['color_transfer'] is None
    assert resolve_source_colour(resolved) == resolved


@pytest.mark.parametrize('patch', [
    {'pix_fmt': 'yuv420p'}, {'color_range': 'pc'}, {'color_range': None},
    {'width': 720, 'height': 480}, {'color_transfer': 'smpte2084'},
    {'color_transfer': 'arib-std-b67'}, {'color_space': 'bt2020ncl'},
    {'color_primaries': 'bt2020'}, {'is_hdr': True},
    {'side_data_list': [{'side_data_type': 'Mastering display metadata'}]},
    {'side_data_list': [{'side_data_type': 'Content light level metadata'}]},
    {'side_data_list': [{'side_data_type': 'DOVI configuration record'}]},
])
def test_defaults_do_not_guess_range_sd_matrix_or_hdr(patch):
    raw = dict(source(), **patch)
    assert resolve_source_colour(raw) == raw


def test_partial_missing_tags_keep_existing_bt709_declaration():
    resolved = resolve_source_colour(dict(source(), color_transfer='bt709'))
    assert resolved['colour_resolution']['assumed_fields'] == ['color_space', 'color_primaries']
    colour = dict(color_range=1, color_space=2, color_primaries=2, color_transfer=1)
    assert resolve_frame_colour(colour, resolved['colour_resolution']) == dict.fromkeys(colour, 1)
    assert colour['color_space'] == 2
    conflict = dict(colour, color_space=9)
    assert resolve_frame_colour(conflict, resolved['colour_resolution'])['color_space'] == 9
    assert resolve_frame_colour(colour, None) == colour


def test_preview_export_and_service_probe_apply_same_policy(monkeypatch):
    stream = dict(source(), codec_type='video', codec_name='hevc', r_frame_rate='24000/1001')
    payload = json.dumps({'streams': [stream]})
    monkeypatch.setattr(gtx.subprocess, 'check_output', lambda *a, **k: payload.encode())
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=payload))
    monkeypatch.setattr(pipeline, '_bundle_ffprobe', lambda: 'ffprobe')
    preview = gtx.probe_source('untagged.mp4')
    service = pipeline.probe_pixel_info('untagged.mp4')
    assert service['bit_depth'] == 10 and service['is_hdr'] is False
    assert preview['colour_resolution']['effective'] == service['colour_resolution']['effective']
    assert preview['colour_resolution']['assumed_fields'] == service['colour_resolution']['assumed_fields']
    from gpu_hdr_transfer_export import sdr10_source
    assert sdr10_source(preview) and sdr10_source(service)
