"""Small deterministic contracts for GPU preview display and reset behavior."""
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gpu_preview_engine as preview_module  # noqa: E402


def test_display_linear_is_fixed_display_only_mapping():
    linear = np.array([[[0.0, 0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0]]], dtype=np.float32)
    sdr = preview_module.display_linear(linear, hdr=False)
    assert sdr.shape == (1, 2, 3)
    assert np.array_equal(sdr[0, 0], [0, 0, 0])
    assert np.array_equal(sdr[0, 1], [255, 255, 255])

    # scRGB white is 80 nits; fixed Reinhard-like display mapping yields 0.5
    # linear before the same Rec.709 display OETF. It is intentionally not an
    # adaptive tone map and remains separate from the source/output buffers.
    hdr = preview_module.display_linear(linear, hdr=True)
    expected = round((1.099 * (0.5 ** 0.45) - 0.099) * 255)
    assert np.all(np.abs(hdr[0, 1].astype(int) - expected) <= 1)
    assert np.array_equal(linear[0, 1, :3], [1.0, 1.0, 1.0])


class _Nvdec:
    def __init__(self):
        self.seeks = []

    def nvdec_bridge_seek_us(self, _decoder, timestamp):
        self.seeks.append(timestamp)
        return 0


class _Chain:
    def __init__(self):
        self.frames = [0, 1]
        self.resets = []
        self.closed = False
        self.decoder = object()

    def close(self):
        self.closed = True

    def next_frame(self, frame):
        if not self.frames:
            return 0
        frame.width, frame.height = 8, 4
        frame.pts = self.frames.pop(0)
        frame.time_base_num, frame.time_base_den = 1, 30
        return 1

    def process(self, frame, _provider, index, reset):
        self.resets.append((index, bool(reset), int(frame.pts)))
        return {"stages": {"decode": 0.01}}

    def layer_diagnostics(self):
        return {"layer_count": 1, "layer_session_ids": [17],
                "layer_working_dims": [[4, 2]]}


class _Live:
    session_id = 17


def test_preview_requests_reset_on_first_frame_and_repeat_without_reopening(tmp_path):
    clip = tmp_path / "tiny.mkv"
    clip.write_bytes(b"fixture-key-only")
    preview = preview_module.GpuPreview()
    chain = _Chain()
    chain.nv12 = object()  # Already opened converter in this decoder-only fake.
    nvdec = _Nvdec()

    def open_stub(self, _video, _settings):
        self.multi = {"scales": [1.0], "dims": [(4, 2)],
                      "fusion": "catmull_baseline"}
        self.chain, self.live = chain, _Live()
        self.bridges = {"nvdec": nvdec}
        self.hdr = self.sdr10 = False
        self.fps = 30.0
        self.source = {"width": 8, "height": 4}
        self.generation += 1

    preview._open = types.MethodType(open_stub, preview)
    def source_stub(self, _outcome, _frame):
        self.result_bgr = np.zeros((4, 8, 3), np.uint8)
        return self.result_bgr.copy()

    preview._source_bgr = types.MethodType(source_stub, preview)
    preview._consume_texture = lambda *_args: None
    try:
        preview._render(str(clip), {"a": 1}, 0)
        preview._render(str(clip), {"a": 1}, 1)
        preview._render(str(clip), {"a": 1}, 1)
        assert [reset for _, reset, _ in chain.resets] == [True, False, True]
        assert len(nvdec.seeks) == 0, "sequential and repeated frame should reuse decode surface"
        assert preview.generation == 1
    finally:
        preview.shutdown()
