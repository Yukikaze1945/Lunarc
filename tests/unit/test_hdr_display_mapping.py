import numpy as np
from hdr_display_mapping import hdr_to_sdr_linear


def test_fixed_display_shoulder_preserves_order_chroma_and_source():
    rgb = np.array([[[.1,.2,.3],[1.,2.,3.],[10.,20.,30.]]],np.float32)
    original = rgb.copy()
    mapped = hdr_to_sdr_linear(rgb)
    assert np.array_equal(rgb,original)
    assert np.all(np.diff(mapped[0],axis=0)>0)
    assert np.allclose(mapped[...,0]/mapped[...,1],.5)
    assert np.allclose(mapped[...,2]/mapped[...,1],1.5)


def test_display_reference_does_not_adapt_to_frame_peak():
    neutral = np.array([[[203/80]*3]],np.float32)
    alone = hdr_to_sdr_linear(neutral)
    with_highlight = hdr_to_sdr_linear(np.concatenate((neutral,np.full_like(neutral,125.)),axis=1))
    assert np.allclose(alone,.5)
    assert np.array_equal(alone[0,0],with_highlight[0,0])
