import json
import struct
import subprocess

import cv2
import numpy as np
import pytest

from comparison_media import MediaComparisonService
from comparison_playback import ComparisonPlayback


def unpack(packet):
    length = struct.unpack('!I', packet[:4])[0]
    data = json.loads(packet[4:4+length])
    first = 4+length
    for side in ('original', 'exported'):
        size = data[side]['size']
        image = cv2.imdecode(np.frombuffer(packet[first:first+size], np.uint8), cv2.IMREAD_COLOR)
        assert image is not None
        data[side]['pixels'] = image
        first += size
    assert first == len(packet)
    return data


@pytest.fixture
def service_pair(tmp_path):
    service = MediaComparisonService()
    original, exported = tmp_path/'source.mp4', tmp_path/'output.mkv'
    for file, fps, offset in ((original, 12, 2), (exported, 24, 5)):
        subprocess.run([service.ffmpeg, '-hide_banner', '-v', 'error', '-y',
            '-f', 'lavfi', '-i', f'testsrc2=size=96x64:rate={fps}:duration=1',
            '-c:v', 'mpeg4', '-q:v', '3', '-output_ts_offset', str(offset), str(file)],
            check=True, capture_output=True)
    return service, service.open_pair(str(original), str(exported))


def test_continuous_decode_preserves_pts_and_closes_bounded_readers(service_pair):
    service, pair = service_pair
    player = ComparisonPlayback(service, pair['session'], 0, 1)
    try:
        times = [0, .1, .3, .6, .8]
        rows = [unpack(player.packet(at)) for at in times]
        for at, row in zip(times, rows):
            assert row['time'] == at
            for side, fps, origin in (('original', 12, 2), ('exported', 24, 5)):
                assert abs(row[side]['actualTime'] - at) <= 1/fps
                assert row[side]['actualPts'] - row[side]['actualTime'] == pytest.approx(origin)
        assert len({row['original']['actualPts'] for row in rows}) == len(times)
        assert len({row['exported']['actualPts'] for row in rows}) == len(times)
        assert all(decoder.frames.qsize() <= 2 for decoder in player.decoders)
    finally:
        player.close()
    assert all(decoder.process.poll() is not None for decoder in player.decoders)
    assert not any(thread.is_alive() for decoder in player.decoders for thread in decoder.threads)


def test_seek_starts_at_requested_relative_time_and_invalid_rate_is_refused(service_pair):
    service, pair = service_pair
    player = ComparisonPlayback(service, pair['session'], .5, .5)
    try:
        row = unpack(player.packet(.5))
        assert row['original']['actualTime'] == pytest.approx(.5, abs=.04)
        assert row['exported']['actualTime'] == pytest.approx(.5, abs=.04)
    finally:
        player.close()
    for at, rate in ((float('nan'), 1), (0, float('inf')), (-1, 1), (0, 0)):
        with pytest.raises(ValueError):
            ComparisonPlayback(service, pair['session'], at, rate)


def test_failed_gpu_startup_reports_software_fallback_and_keeps_pts(service_pair, monkeypatch):
    service, pair = service_pair
    # Force the GPU branch; the failed process must be replaced before any
    # pictures are published, without leaking the old pipe readers.
    for media in service._pair(pair['session']):
        media.metadata.update(codec='h264',pixel_format='yuv420p',bit_depth=8)
    real_popen = subprocess.Popen
    failed = []
    def launch(command, **options):
        if '-hwaccel_output_format' in command:
            proc = real_popen([service.ffmpeg,'-v','error','-invalid_gpu_startup_probe'],**options)
            failed.append(proc)
            return proc
        return real_popen(command,**options)
    monkeypatch.setattr('comparison_playback.subprocess.Popen',launch)
    player = ComparisonPlayback(service,pair['session'],.5,1.)
    try:
        row = unpack(player.packet(.5))
        assert len(failed)==2
        for side in ('original','exported'):
            assert row['playback'][side]['decoder']=='software'
            assert row['playback'][side]['fallback_reason']
            assert row[side]['actualTime']==pytest.approx(.5,abs=1/12)
    finally:
        player.close()
    assert all(proc.poll() is not None for proc in failed)
    assert not any(thread.is_alive() for decoder in player.decoders for thread in decoder.threads)
