"""Native option parsing is authoritative; CLI and shared libraries may differ."""
from types import SimpleNamespace

import pytest
import gpu_transfer_export as gtx
import gpu_hdr_transfer_export as htx


def test_loaded_encoder_value_probe_is_cached_and_codec_specific():
    calls = []
    def value(codec, option, setting):
        calls.append((codec, option, setting))
        return int(codec == b'hevc_nvenc')
    probe = gtx.native_tune_probe(SimpleNamespace(nvenc_option_value_available=value))
    assert probe('hevc_nvenc') and probe('hevc_nvenc')
    assert not probe('av1_nvenc')
    assert calls == [(b'hevc_nvenc', b'tune', b'uhq'), (b'av1_nvenc', b'tune', b'uhq')]


def test_probe_error_is_not_converted_to_supported_or_downgraded():
    library = SimpleNamespace(nvenc_option_value_available=lambda *a: -1,
                              nvenc_last_error=lambda: b'probe failed')
    with pytest.raises(gtx.BridgeError, match='probe failed'):
        gtx.native_tune_probe(library)('av1_nvenc')
    with pytest.raises(gtx.BridgeError, match='rebuilt'):
        gtx.native_tune_probe(library, value_probe=False)('hevc_nvenc')


@pytest.mark.parametrize('codec', ['hevc_nvenc', 'av1_nvenc'])
@pytest.mark.parametrize('wide', [False, True])
def test_uhq_is_retained_for_supported_codec_and_adjustment_is_recorded(codec, wide):
    probe = gtx.native_tune_probe(SimpleNamespace(
        nvenc_option_value_available=lambda actual, *a: int(actual == b'hevc_nvenc')))
    plan_fn = htx.plan_hdr_encoder if wide else gtx.plan_encoder
    kwargs = {'hdr_caps': {'p010_available': True, 'encoder_codec': codec}} if wide else {}
    plan = plan_fn({'codec': codec, 'tune': 'uhq', 'b_frames': 'auto'}, 24, {},
                   tune_available=probe, **kwargs)
    assert not plan['unsupported'], plan['unsupported']
    assert plan['struct']['tune'] == (b'uhq' if codec == 'hevc_nvenc' else b'hq')
    assert plan['tune_adjustment'] == (None if codec == 'hevc_nvenc' else {
        'requested': 'uhq', 'applied': 'hq', 'reason': 'loaded encoder does not support UHQ'})
    assert not any(option == 'bf' for option, _ in plan['av_options'])
