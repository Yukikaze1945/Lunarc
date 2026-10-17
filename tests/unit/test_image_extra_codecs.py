"""Image codec routing/color contracts; no native library or GPU is opened."""
from pathlib import Path
import numpy as np
import pytest

import image_extra_codecs as codecs
import image_hdr_export as hdr


def test_uhdr_preserves_absolute_white_and_gamut_mapping_luminance():
    white,report=codecs.uhdr_pixels(np.ones((2,2,3),np.float16))
    assert np.allclose(white[:,:,:3],80/203,atol=.001)
    assert report['scrgb_nits_per_unit']==80 and report['codec_reference_white_nits']==203
    rgb=np.array([[[-1,1,.1]]],np.float32)
    result,report=codecs.uhdr_pixels(rgb)
    expected=(rgb@codecs._SRGB_XYZ[1]).item()
    actual=(result[0,0,:3].astype(np.float32)@codecs._P3_XYZ[1])*203/80
    assert report['gamut_compressed_pixels']==1
    assert result[:,:,:3].min()>=0 and abs(actual-expected)<.002


def test_uhdr_rejects_nonfinite_input():
    with pytest.raises(ValueError,match='非有限'):
        codecs.uhdr_pixels(np.full((2,2,3),np.nan,np.float32))


def test_explicit_codec_directory_never_falls_back(tmp_path,monkeypatch):
    monkeypatch.setenv('LUNARC_IMAGE_CODECS_DIR',str(tmp_path))
    assert not codecs.capabilities()['jxl']['available']
    with pytest.raises(FileNotFoundError):codecs.codec_file('uhdr.dll')


def test_uhdr_alpha_is_rejected_before_encoding(tmp_path,monkeypatch):
    monkeypatch.setattr(hdr,'capabilities',lambda:{'uhdr':dict(available=True,alpha=False)})
    monkeypatch.setattr(codecs,'save_uhdr',lambda *a:pytest.fail('must not discard requested alpha'))
    destination=tmp_path/'result.jpg'
    with pytest.raises(ValueError,match='透明度'):
        hdr.save_hdr_image(str(tmp_path/'source.png'),np.ones((2,2,3),np.float16),str(destination),
            'uhdr',source_alpha=np.zeros((2,2),np.uint8))
    assert not destination.exists()


def test_hdr_jxl_uses_original_float_master_and_atomic_destination(tmp_path,monkeypatch):
    monkeypatch.setattr(hdr,'capabilities',lambda:{'jxl':dict(available=True,alpha=True)})
    rgb=np.full((2,3,3),12.5,np.float16);destination=tmp_path/'result.jxl'
    def encode(path,master,alpha,quality,**kwargs):
        assert master is rgb and master.max()==12.5 and kwargs['hdr']
        Path(path).write_bytes(b'validated JXL')
        return dict(bit_depth=16,has_alpha=False)
    monkeypatch.setattr(codecs,'save_jxl',encode)
    result=hdr.save_hdr_image(str(tmp_path/'source.png'),rgb,str(destination),'jxl',
        source_alpha=np.full((2,3),255,np.uint8))
    assert result['bit_depth']==16 and destination.read_bytes()==b'validated JXL'
