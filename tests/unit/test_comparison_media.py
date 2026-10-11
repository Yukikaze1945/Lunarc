import json
import os
import subprocess
import sys
import shutil
import wave

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from comparison_media import (  # noqa: E402
    MediaComparisonEOF, MediaComparisonError, MediaComparisonService,
    linear_rgb_to_display_bgr,
)
from comparison_audio import audio_chunks, wav_range  # noqa: E402


class FakeRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), dict(kwargs)))
        assert kwargs["shell"] is False
        if args[0] == "fake-ffprobe":
            path = args[-1]
            exported = path.endswith("exported.mkv")
            origin = 2.0 if exported else 10.0
            if "-show_streams" in args:
                stream = {"index": 0, "codec_type": "video", "codec_name": "h264",
                    "profile": "High", "width": 2, "height": 1,
                    "pix_fmt": "yuv420p", "bits_per_raw_sample": "8",
                    "avg_frame_rate": "30/1", "r_frame_rate": "30/1",
                    "time_base": "1/1000", "start_time": str(origin),
                    "duration": "4.0", "color_range": "tv",
                    "color_space": "bt709", "color_transfer": "bt709",
                    "color_primaries": "bt709"}
                audio = {"index": 1, "codec_type": "audio", "codec_name": "aac",
                         "channels": 2, "sample_rate": "48000"}
                payload = {"streams": [stream, audio], "format": {
                    "format_name": "matroska,webm", "duration": "4.0",
                    "size": "1234", "start_time": str(origin)}}
            elif "%+#3" in args[args.index("-read_intervals") + 1]:
                payload = {"frames": [{"best_effort_timestamp_time": str(origin)}]}
            else:
                pts = ([origin + .966667, origin + 1.0, origin + 1.033333]
                       if not exported else [origin + .95, origin + 1.0, origin + 1.05])
                payload = {"frames": [{"best_effort_timestamp_time": str(t),
                    "pkt_duration_time": "0.033333", "key_frame": 1 if i == 0 else 0}
                    for i, t in enumerate(pts)]}
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")
        # gbrpf32le: three float planes in G, B, R order.
        planes = np.zeros((3, 1, 2), dtype="<f4")
        planes[2, :, :] = 1.0
        return subprocess.CompletedProcess(args, 0, planes.tobytes(), b"")


def _pair(tmp_path):
    original = tmp_path / "original.mkv"
    exported = tmp_path / "exported.mkv"
    original.write_bytes(b"original")
    exported.write_bytes(b"exported")
    return original, exported


def test_pair_uses_actual_nearby_pts_and_decodes_selected_file_frame(tmp_path):
    runner = FakeRunner()
    service = MediaComparisonService("fake-ffprobe", "fake-ffmpeg", runner=runner)
    original, exported = _pair(tmp_path)
    opened = service.open_pair(str(original), str(exported))

    assert opened["original"]["codec"] == "h264"
    assert opened["original"]["frame_rate"] == 30.0
    assert opened["original"]["audio"]["codec"] == "aac"
    assert opened["duration_seconds"] == 4.0

    timeline = service.get_timeline(opened["session"], 1.01)
    assert timeline["sides"]["original"]["actual_time"] == pytest.approx(1.0)
    assert timeline["sides"]["exported"]["actual_time"] == pytest.approx(1.0)
    assert timeline["sides"]["original"]["previous_time"] == pytest.approx(.966667)
    assert timeline["sides"]["original"]["next_time"] == pytest.approx(1.033333)

    frame = service.get_frame(opened["session"], 1.01, "exported")
    assert frame["actual_pts"] == pytest.approx(3.0)
    assert frame["actual_time"] == pytest.approx(1.0)
    assert frame["metadata"]["name"] == "exported.mkv"
    assert frame["jpeg"].startswith(b"\xff\xd8")
    decoded = cv2.imdecode(np.frombuffer(frame["jpeg"], np.uint8), cv2.IMREAD_COLOR)
    assert decoded[0, 0, 2] > 245 and decoded[0, 0, 0] < 8, "GBR planes were reordered incorrectly"
    ffmpeg_call = next(args for args, _ in runner.calls if args[0] == "fake-ffmpeg")
    assert ffmpeg_call[ffmpeg_call.index("-ss") + 1] == "3.000000000"
    assert all(kwargs["shell"] is False for _, kwargs in runner.calls)
    assert service.audio_source(opened["session"], "original")["stream_index"] == 1


def test_local_file_validation_eof_and_expired_session(tmp_path):
    runner = FakeRunner()
    service = MediaComparisonService("fake-ffprobe", "fake-ffmpeg", runner=runner)
    with pytest.raises(MediaComparisonError, match="本机"):
        service.open_pair("https://example.invalid/video.mkv", "C:\\out.mkv")
    assert not runner.calls

    original, exported = _pair(tmp_path)
    opened = service.open_pair(str(original), str(exported))
    with pytest.raises(MediaComparisonEOF):
        service.get_frame(opened["session"], 4.2, "original")
    assert service.close_session(opened["session"])
    with pytest.raises(MediaComparisonError, match="会话"):
        service.get_timeline(opened["session"], 1.0)


def test_fixed_hdr_display_mapping_is_shared_and_does_not_mutate_source():
    rgb = np.array([[[1.0, 1.0, 1.0], [0.5, 0.25, 0.0]]], dtype=np.float32)
    before = rgb.copy()
    sdr = linear_rgb_to_display_bgr(rgb, hdr=False)
    hdr = linear_rgb_to_display_bgr(rgb, hdr=True)
    assert np.array_equal(sdr[0, 0], [255, 255, 255])
    assert np.all(hdr[0, 0] < sdr[0, 0])
    assert np.array_equal(rgb, before)
    assert np.all(np.isfinite(linear_rgb_to_display_bgr(
        np.array([[[np.nan, np.inf, -np.inf]]], np.float32), hdr=True)))


def test_audio_wav_ranges_preserve_header_and_virtual_pcm_byte_offsets(monkeypatch):
    import comparison_audio

    pcm = bytes(range(192))

    class FakeProcess:
        def __init__(self, command, **kwargs):
            self.command = command
            seek = float(command[command.index('-ss') + 1]) - source['origin_pts']
            sample = round(seek * 48000)
            self.stdout = __import__('io').BytesIO(pcm[sample * 4:])
            self.returncode = None
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            self.returncode = 0
            return self.returncode
        def terminate(self):
            self.returncode = 0
        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(comparison_audio.subprocess, 'Popen', FakeProcess)
    source = {'duration_seconds': .001, 'origin_pts': 3.0,
              'path': 'source.mkv', 'stream_index': 1}
    header, start, end, total = wav_range(source, 'bytes=40-55')
    assert total == 44 + 192
    assert b''.join(audio_chunks('ffmpeg', source, header, start, end)) == header[40:44] + pcm[:12]

    # A range beginning halfway through a PCM sample still uses the virtual WAV
    # byte offset and trims precisely the leading decoder bytes.
    header, start, end, _ = wav_range(source, 'bytes=86-92')
    body = b''.join(audio_chunks('ffmpeg', source, header, start, end))
    assert body == pcm[42:49]



def test_audio_decode_failure_is_not_silently_replaced_with_silence(monkeypatch):
    import comparison_audio

    class FailedProcess:
        def __init__(self, command, **kwargs):
            self.stdout = __import__('io').BytesIO(b'')
            self.returncode = None
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            self.returncode = 2
            return self.returncode
        def terminate(self):
            self.returncode = 2
        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(comparison_audio.subprocess, 'Popen', FailedProcess)
    source = {'duration_seconds': .001, 'origin_pts': 0.0,
              'path': 'broken.mkv', 'stream_index': 1}
    header, start, end, _ = wav_range(source)
    with pytest.raises(RuntimeError, match='FFmpeg audio decode failed'):
        b''.join(audio_chunks('ffmpeg', source, header, 44, end))


def test_audio_chunks_decode_a_real_short_wav_when_ffmpeg_is_available(tmp_path):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        pytest.skip('ffmpeg executable is unavailable')
    wav_path = tmp_path / 'short.wav'
    samples = (np.sin(np.arange(480) * (2 * np.pi / 48)) * 12000).astype('<i2')
    stereo = np.column_stack((samples, -samples)).astype('<i2')
    with wave.open(str(wav_path), 'wb') as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(48000)
        out.writeframes(stereo.tobytes())
    source = {'duration_seconds': .01, 'origin_pts': 0.0,
              'path': str(wav_path), 'stream_index': 0}
    header, start, end, total = wav_range(source)
    stream = b''.join(audio_chunks(ffmpeg, source, header, start, end))
    assert len(stream) == total
    decoded = np.frombuffer(stream[44:], dtype='<i2').reshape(-1, 2)
    assert np.any(decoded)
    assert np.array_equal(decoded[:, 0], -decoded[:, 1])


def test_real_vfr_pair_reports_nearest_timestamps_and_union_steps(tmp_path):
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        pytest.skip('ffmpeg tools are unavailable')
    original, exported = tmp_path / 'vfr.mkv', tmp_path / 'slow.mp4'
    subprocess.run([ffmpeg, '-hide_banner', '-v', 'error', '-y', '-f', 'lavfi',
        '-i', 'testsrc2=size=64x48:rate=10:duration=1', '-vf',
        "setpts='if(eq(N,0),0,if(eq(N,1),1,if(eq(N,2),4,N+4)))'",
        '-fps_mode', 'vfr', '-c:v', 'ffv1', '-output_ts_offset', '2',
        str(original)], check=True, capture_output=True)
    subprocess.run([ffmpeg, '-hide_banner', '-v', 'error', '-y', '-f', 'lavfi',
        '-i', 'testsrc2=size=64x48:rate=5:duration=1', '-output_ts_offset', '2',
        '-c:v', 'mpeg4', '-q:v', '3', str(exported)], check=True, capture_output=True)

    service = MediaComparisonService(ffprobe, ffmpeg)
    opened = service.open_pair(str(original), str(exported))
    try:
        assert opened['original']['frame_rate'] != opened['exported']['frame_rate']
        assert opened['original']['start_time'] >= 1.9
        timeline = service.get_timeline(opened['session'], .11)
        a = timeline['sides']['original']
        b = timeline['sides']['exported']
        assert a['actual_time'] == pytest.approx(.1, abs=.02)
        assert b['actual_time'] == pytest.approx(.2, abs=.02)
        assert a['next_time'] is not None and b['next_time'] is not None
        assert min(a['next_time'], b['next_time']) == pytest.approx(.4, abs=.03)
    finally:
        service.close_session(opened['session'])


@pytest.mark.parametrize('depth,hdr,pixel_format', [
    (8, False, 'yuv444p'),
    (10, False, 'yuv444p10le'),
    (10, True, 'yuv444p10le'),
])
def test_real_frame_decode_selects_vfr_pts_and_uses_shared_hdr_mapping(
        tmp_path, depth, hdr, pixel_format):
    import pipeline

    ffmpeg, ffprobe = pipeline._bundle_ffmpeg(), pipeline._bundle_ffprobe()
    raw = tmp_path / f'known-{depth}-{hdr}.rgb'
    media_path = tmp_path / f'known-{depth}-{hdr}.mkv'
    colors = np.array([[240, 20, 20], [20, 240, 20],
                       [20, 20, 240], [230, 210, 20]], dtype=np.uint8)
    frames = np.repeat(colors[:, None, None, :], 32 * 24, axis=1).reshape(4, 24, 32, 3)
    raw.write_bytes(frames.tobytes())
    trc, primaries, matrix = (('smpte2084', 'bt2020', 'bt2020nc') if hdr else
                              ('bt709', 'bt709', 'bt709'))
    encoder = (['-c:v', 'libx265', '-preset', 'ultrafast', '-x265-params',
                'lossless=1', '-bsf:v',
                'hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9']
               if hdr else ['-c:v', 'ffv1'])
    filter_chain = ("setpts='if(eq(N,0),0,if(eq(N,1),1,if(eq(N,2),4,N+4)))',"
                    f'setparams=color_primaries={primaries}:color_trc={trc}:colorspace={matrix}')
    subprocess.run([ffmpeg, '-hide_banner', '-v', 'error', '-y', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '32x24', '-framerate', '10',
        '-i', str(raw), '-vf', filter_chain,
        '-fps_mode', 'vfr', *encoder, '-pix_fmt', pixel_format,
        '-color_primaries', primaries, '-color_trc', trc, '-colorspace', matrix,
        '-output_ts_offset', '2', str(media_path)], check=True, capture_output=True)

    service = MediaComparisonService(ffprobe, ffmpeg)
    opened = service.open_pair(str(media_path), str(media_path))
    try:
        info = opened['original']
        assert info['bit_depth'] == depth
        assert info['is_hdr'] is hdr, (info['color_transfer'],
            info['color_primaries'], info['color_matrix'], info['pixel_format'])
        assert info['start_time'] >= 1.9
        timeline = service.get_timeline(opened['session'], .11)
        assert timeline['sides']['original']['actual_time'] == pytest.approx(.1, abs=.02)
        for side in ('original', 'exported'):
            frame = service.get_frame(opened['session'], .11, side)
            assert frame['actual_time'] == pytest.approx(.1, abs=.02)
            assert frame['actual_pts'] == pytest.approx(info['start_time'] + .1, abs=.02)
            image = cv2.imdecode(np.frombuffer(frame['jpeg'], np.uint8), cv2.IMREAD_COLOR)
            center = image[12, 16].astype(int)
            assert center[1] > center[2] * 2 and center[1] > center[0] * 2
            assert frame['display_mapping'] == (
                'fixed_hdr_to_sdr_uncalibrated' if hdr else 'linear_rec709_to_sdr_display')
    finally:
        service.close_session(opened['session'])


@pytest.mark.parametrize('pixel_format', ['yuv444p', 'yuv444p10le'])
def test_sdr_display_preserves_midgray_code_values(tmp_path, pixel_format):
    """Playback must not darken SDR by mixing display gamma and camera OETF."""
    import pipeline

    ffmpeg, ffprobe = pipeline._bundle_ffmpeg(), pipeline._bundle_ffprobe()
    raw, media = tmp_path / 'gray.rgb', tmp_path / 'gray.mkv'
    raw.write_bytes(np.full((24, 32, 3), 128, np.uint8).tobytes())
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '32x24', '-i', str(raw),
        '-frames:v', '1', '-c:v', 'ffv1', '-pix_fmt', pixel_format,
        '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-colorspace', 'bt709', str(media)], check=True, capture_output=True)
    service = MediaComparisonService(ffprobe, ffmpeg)
    pair = service.open_pair(str(media), str(media))
    try:
        for side in ('original', 'exported'):
            frame = service.get_frame(pair['session'], 0, side)
            decoded = cv2.imdecode(np.frombuffer(frame['jpeg'], np.uint8), cv2.IMREAD_COLOR)
            assert np.all(np.abs(decoded[12, 16].astype(int) - 128) <= 2)
    finally:
        service.close_session(pair['session'])
