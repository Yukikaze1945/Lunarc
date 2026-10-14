"""Codec/color-mode independence and compatibility, without a GPU."""
from copy import deepcopy
import ctypes as C
from types import SimpleNamespace

import pytest
import gpu_hdr_transfer_export as htx
import gpu_transfer_export as gtx
from test_hdr_transfer_export_branch import HDR_CAPS, HDR_SOURCE, NO_MUX


def caps(codec):
    result=deepcopy(HDR_CAPS)
    result.update(p010_available=True,encoder_codec=codec)
    result['encoder_defaults']['profile']=htx.P010_PROFILES[codec]
    return result


@pytest.mark.parametrize('codec,profile', [('hevc_nvenc','main10'),('av1_nvenc','main')])
def test_codec_selects_its_own_profile_and_option_probe(codec,profile):
    queried=[]
    plan=htx.plan_hdr_encoder(dict(NO_MUX,codec=codec,profile=profile),24,HDR_SOURCE,
        lambda actual,name: queried.append((actual,name)) or True,caps(codec))
    assert not plan['unsupported'],plan['unsupported']
    assert plan['struct']['codec']==codec.encode() and plan['struct']['profile']==profile.encode()
    assert queried and all(actual==codec for actual,_ in queried)


@pytest.mark.parametrize('codec', ['hevc_nvenc','av1_nvenc'])
@pytest.mark.parametrize('probe,tune', [(None,'uhq'),(False,'hq'),(True,'uhq')])
def test_uhq_survives_only_where_the_probed_encoder_lists_it(codec,probe,tune):
    """Which codecs tune uhq is a property of the installed libavcodec, not of here.

    av1_nvenc gained it after 7.1, the entry forwards tune verbatim, and an enumerator
    the option table does not list is an EINVAL that refuses the open after the decode
    and the reconstruction have already run. Nobody answering is not a downgrade.
    """
    profile=htx.P010_PROFILES[codec]
    plan=htx.plan_hdr_encoder(dict(NO_MUX,codec=codec,profile=profile,tune='uhq'),24,
        HDR_SOURCE,lambda *a: True,caps(codec),
        None if probe is None else (lambda actual: probe))
    assert not plan['unsupported'],plan['unsupported']
    assert plan['struct']['tune']==tune.encode()


@pytest.mark.parametrize('patch', [dict(profile='main10'),dict(weighted_prediction=True),
    dict(b_ref_mode='each'),dict(cq=64),dict(cq=-1)])
def test_invalid_av1_options_are_refused(patch):
    plan=htx.plan_hdr_encoder({**NO_MUX,'codec':'av1_nvenc','profile':'main',**patch},24,
        HDR_SOURCE,lambda *a: True,caps('av1_nvenc'))
    assert plan['unsupported']


def test_av1_has_63_cq_limit_and_does_not_send_disabled_hevc_prediction_option():
    plan=htx.plan_hdr_encoder(dict(NO_MUX,codec='av1_nvenc',profile='main',cq=63,
        weighted_prediction=False),24,HDR_SOURCE,lambda codec,name:name!='weighted_pred',caps('av1_nvenc'))
    assert not plan['unsupported'] and ('weighted_pred','0') not in plan['av_options']
    assert plan['struct']['constant_quality']==63


def test_av1_rejects_old_bridge_and_wrong_codec_capabilities():
    for capability in (HDR_CAPS,caps('hevc_nvenc')):
        plan=htx.plan_hdr_encoder(dict(NO_MUX,codec='av1_nvenc',profile='main'),24,
                                 HDR_SOURCE,None,capability)
        assert plan['unsupported']


@pytest.mark.parametrize('sdr,color_mode', [(True,0),(False,1)])
def test_new_p010_open_receives_explicit_color_mode(sdr,color_mode):
    seen=[]
    library=SimpleNamespace(nvenc_p010_open=lambda *a: seen.append(a) or 0,
                            nvenc_p010_abi_version=lambda:1)
    options=htx.HdrOptions(C.sizeof(htx.HdrOptions));options.codec=b'av1_nvenc'
    assert htx.open_p010_encoder(library,sdr,123,1280,720,25,1,1,1000,b'out.mkv',
        C.byref(options),None,C.byref(C.c_void_p()))==0
    assert seen[0][-2]==color_mode


def test_old_bridge_keeps_hevc_entry_and_never_falls_back_for_av1():
    calls=[]
    library=SimpleNamespace(nvenc_hdr_open=lambda *a:calls.append('hevc') or 0)
    options=htx.HdrOptions(C.sizeof(htx.HdrOptions));options.codec=b'hevc_nvenc'
    args=(1,320,192,25,1,1,1000,b'out.mkv',C.byref(options),None,C.byref(C.c_void_p()))
    assert htx.open_p010_encoder(library,False,*args)==0 and calls==['hevc']
    options.codec=b'av1_nvenc'
    with pytest.raises(gtx.BridgeError,match='rebuilt P010'): htx.open_p010_encoder(library,False,*args)
    assert calls==['hevc']


def test_quantization_error_report_retains_float_precision():
    status=htx.P010MetadataStatus(C.sizeof(htx.P010MetadataStatus))
    status.max_chromaticity_error=0.000003
    assert htx._fields(status)['max_chromaticity_error']==0.000003
