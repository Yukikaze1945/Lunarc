"""Lifecycle contracts for the production preview session, using an injected fake chain."""
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gpu_preview_engine as preview_module  # noqa: E402
from comparison_media import linear_rgb_to_display_bgr  # noqa: E402
from tests.helpers.preview_export_parity_gate import _compare  # noqa: E402


class _Nvdec:
    def __init__(self):
        self.seeks = []

    def nvdec_bridge_seek_us(self, _decoder, timestamp):
        self.seeks.append(timestamp)
        return 0


class _Chain:
    def __init__(self, dims, session_ids):
        self.frames = list(range(20))
        self.closed = False
        self.decoder = object()
        self.nv12 = object()
        self.dims, self.session_ids = dims, session_ids
        self.calls, self.resets = 0, []
        self.frame_reports, self.fusion_reports, self.metadata_snapshots = [], [], []
        self.layer_reset_flags = [[] for _ in dims]

    def close(self):
        self.closed = True
        for owner in getattr(self, "owners", []):
            owner.close()

    def next_frame(self, frame):
        if not self.frames:
            return 0
        frame.width, frame.height = 8, 4
        frame.pts = self.frames.pop(0)
        frame.time_base_num, frame.time_base_den = 1, 30
        return 1

    def process(self, frame, _provider, index, reset):
        self.calls += 1
        self.resets.append((index, bool(reset)))
        for flags in self.layer_reset_flags:
            flags.append(bool(reset))
        return {"stages": {"process": 0.1}}

    def layer_diagnostics(self):
        return {"layer_count": len(self.dims), "layer_session_ids": self.session_ids,
                "layer_working_dims": [list(d) for d in self.dims],
                "layer_dlss_calls": [self.calls] * len(self.dims)}


class _Owner:
    def __init__(self, session_id=None):
        self.session_id = session_id
        self.closed = False

    def close(self):
        self.closed = True

    def close_independent_layer(self):
        self.closed = True

    def timing(self):
        return {"sdk_counters": {"successful_dispatches": 0}}


def test_hdr_parity_uses_same_fixed_mapping_as_preview_and_comparison_media():
    linear_rgb = np.array([[[0.0, 0.0, 0.0], [0.25, 0.5, 1.0],
                            [2.0, 1.0, 0.5]]], dtype=np.float32)
    rgba = np.concatenate((linear_rgb, np.ones((*linear_rgb.shape[:2], 1), np.float32)), axis=2)
    preview = preview_module.display_linear(rgba, hdr=True)
    compared = linear_rgb_to_display_bgr(linear_rgb, hdr=True)
    assert np.array_equal(preview, compared)


def test_flow_and_picture_views_cannot_share_cached_display(tmp_path):
    clip = tmp_path / 'flow.mkv'
    clip.write_bytes(b'fake source')
    preview, opened = preview_module.GpuPreview(), []
    _inject_open(preview, opened)
    preview._flow_bgr = lambda descriptor: (np.full((4, 8, 3), 90, np.uint8), 256)
    settings = dict(_settings(), guidance_mode=1)
    try:
        _, picture, _ = preview._render(str(clip), settings, 0, reuse_display=True)
        _, motion, diag = preview._render(str(clip), settings, 0, reuse_display=True, flow_view=True)
        assert np.all(picture == 40) and np.all(motion == 90)
        assert diag['flow_diagnostic_readback_bytes'] == 256 and not diag['cached_display']
        _, cached, diag = preview._render(str(clip), settings, 0, reuse_display=True, flow_view=True)
        assert np.all(cached == 90) and diag['cached_display']
        _, picture, diag = preview._render(str(clip), settings, 0, reuse_display=True)
        assert np.all(picture == 40) and not diag['cached_display']
    finally:
        preview.shutdown()


def test_parity_checker_marks_flat_input_without_using_correlation():
    flat_preview = np.zeros((12, 16, 3), np.uint8)
    flat_export = np.ones((12, 16, 3), np.uint8)
    flat = _compare(flat_preview, flat_export)
    assert flat["frame_classification"] == "flat_or_near_black_diagnostic_only"
    assert flat["correlation"] is None and flat["within_lossy_tolerance"]

    textured = np.indices((12, 16)).sum(axis=0).astype(np.uint8) * 8
    textured = np.repeat(textured[..., None], 3, axis=2)
    failed = _compare(textured, np.zeros_like(textured))
    assert failed["frame_classification"] == "visible_detail"
    assert not failed["within_lossy_tolerance"]


def _settings(mode="single", marker=0):
    scales = {"single": [0.5], "dual": [0.5, 0.75], "triple": [0.4, 0.67, 0.85]}[mode]
    return {"reconstruction_mode": mode, "working_scale": scales[0],
            "layer_scales": scales, "marker": marker}


def test_export_framegen_controls_do_not_reload_or_reject_preview(tmp_path):
    clip = tmp_path/'clip.mkv'
    clip.write_bytes(b'fixture')
    preview,opened = preview_module.GpuPreview(),[]
    _inject_open(preview,opened)
    try:
        preview._render(str(clip),dict(_settings(),frame_generation='off'),0)
        preview._render(str(clip),dict(_settings(),frame_generation='vfg_dynamic',framegen_target_fps='',framegen_scene_detection=False),1)
        preview._render(str(clip),dict(_settings(),frame_generation='vfg',frame_multiplier=8,framegen_quality=2),2)
        assert len(opened) == 1
        assert not any(k.startswith('framegen') or k == 'frame_generation' for k in opened[0]['settings'])
    finally:
        preview.shutdown()


def _inject_open(preview, opened):
    def open_stub(self, video, settings):
        mode = settings["reconstruction_mode"]
        scales = settings["layer_scales"]
        dims = [(max(1, round(8 * s)), max(1, round(4 * s))) for s in scales]
        ids = list(range(100 + len(opened) * 10, 100 + len(opened) * 10 + len(scales)))
        chain = _Chain(dims, ids)
        live, provider = _Owner(ids[0]), _Owner()
        chain.owners = [live]
        opened.append({"video": video, "settings": dict(settings), "chain": chain,
                       "live": live, "provider": provider, "ids": ids})
        self.chain, self.live, self.provider = chain, live, provider
        process = chain.process
        def process_picture(*args):
            self.result_bgr = np.full((4, 8, 3), 40, np.uint8)
            return process(*args)
        chain.process = process_picture
        self.multi = {"scales": scales, "dims": dims, "fusion": "data_consistency"}
        self.bridges = {"nvdec": _Nvdec()}
        self.hdr = self.sdr10 = False
        self.fps, self.source = 30.0, {"width": 8, "height": 4}
        self.generation += 1

    preview._open = types.MethodType(open_stub, preview)

    def source_stub(self, _outcome, _frame):
        self.result_bgr = np.full((4, 8, 3), 40, np.uint8)
        return self.result_bgr.copy()

    preview._source_bgr = types.MethodType(source_stub, preview)


def test_preview_reuses_session_resets_seek_and_closes_on_settings_or_video_change(tmp_path):
    clips = [tmp_path / f"{n}.mkv" for n in ("a", "b")]
    for clip in clips:
        clip.write_bytes(b"stable fake source")
    preview, opened = preview_module.GpuPreview(), []
    _inject_open(preview, opened)
    try:
        s = _settings()
        for index in (0, 1, 1, 0):
            _, image, diag = preview._render(str(clips[0]), s, index)
            assert image.shape == (4, 8, 3)
            assert diag["generation"] == 1
        first = opened[0]
        assert first["chain"].resets == [(0, True), (1, False), (1, True), (0, True)]
        assert first["provider"].closed is False
        assert first["chain"].closed is False
        assert first["chain"].decoder is not None
        assert first["chain"].layer_diagnostics()["layer_session_ids"] == [100]
        assert len(preview.bridges["nvdec"].seeks) == 1

        preview._render(str(clips[0]), _settings(marker=1), 0)
        assert first["chain"].closed and first["provider"].closed and first["live"].closed
        second = opened[1]
        preview._render(str(clips[1]), _settings(marker=1), 0)
        assert second["chain"].closed and second["provider"].closed
        assert opened[2]["video"] == str(clips[1])
        assert preview.generation == 3
    finally:
        preview.shutdown()


def test_single_dual_triple_sessions_are_independent_and_shutdown_releases_all(tmp_path):
    clip = tmp_path / "layers.mkv"
    clip.write_bytes(b"stable fake source")
    preview, opened = preview_module.GpuPreview(), []
    _inject_open(preview, opened)
    try:
        for mode, expected in (("single", 1), ("dual", 2), ("triple", 3)):
            _, _, diag = preview._render(str(clip), _settings(mode), 0)
            assert diag["layer_count"] == expected
            ids = diag["layers"]["layer_session_ids"]
            assert len(ids) == expected and len(set(ids)) == expected
            assert len(diag["layers"]["layer_working_dims"]) == expected
        assert [row["chain"].closed for row in opened] == [True, True, False]
    finally:
        preview.shutdown()
    assert all(row["chain"].closed and row["provider"].closed and row["live"].closed
               for row in opened)


def test_cached_display_is_bounded_and_runtime_nr_changes_reset_without_recreating(tmp_path):
    clip = tmp_path / 'cached.mkv'
    clip.write_bytes(b'source')
    preview, opened = preview_module.GpuPreview(), []
    _inject_open(preview, opened)
    try:
        settings = dict(_settings(), intensity=1.)
        preview._render(str(clip), settings, 0, reuse_display=True)
        chain, live = preview.chain, preview.live
        live.update_runtime_options = lambda settings: True
        _, _, cached = preview._render(str(clip), settings, 0, reuse_display=True)
        assert cached['cached_display'] and chain.calls == 1
        _, _, changed = preview._render(str(clip), dict(settings, intensity=.8), 1, reuse_display=True)
        assert preview.chain is chain and changed['generation'] == 1
        assert changed['reset'] and not changed['cached_display']
        assert chain.resets[-1] == (1, True)
        from post_control import GpuPostControl
        from temporal_control import GpuTemporalControl
        chain.post_control = GpuPostControl()
        chain.temporal_control = GpuTemporalControl()
        _, _, adjusted = preview._render(str(clip), dict(settings, intensity=.8,
            post_controls={'total_gain': .7}, temporal_controls={'scene_detection': True}),
            2, reuse_display=True)
        assert preview.chain is chain and adjusted['generation'] == 1 and adjusted['reset']
        assert chain.post_control.diagnostics()['controls']['total_gain'] == .7
        assert chain.temporal_control.controls.scene_detection
        # Structural changes retain the safe recreation boundary.
        preview._render(str(clip), dict(settings, preset=2), 1, reuse_display=True)
        assert chain.closed and preview.generation == 2
    finally:
        preview.shutdown()
