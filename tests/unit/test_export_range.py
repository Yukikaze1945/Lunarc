import ctypes as C
from types import SimpleNamespace
import pytest
from export_range import SelectedDecoder, validate_range


def test_selection_validation():
    assert validate_range({'start_frame': 2, 'end_frame': 5}, 5)['end_frame'] == 5
    for value in ({'start_frame': True, 'end_frame': 5}, {'start_frame': 5, 'end_frame': 5},
                  {'start_frame': -1, 'end_frame': 2}, {'start_frame': 0, 'end_frame': 6}):
        with pytest.raises(ValueError):
            validate_range(value, 5)


def test_seek_preroll_is_discarded_and_selected_pts_rebased():
    pts = iter([2000, 2000, 2100, 2200, 2300])
    class Seek:
        def __call__(self, handle, at):
            assert at == 200000
            return 0
    def read(frame):
        frame.pts = next(pts)
        return 1
    chain = SimpleNamespace(next_frame=read, b={'nvdec': SimpleNamespace(nvdec_bridge_seek_us=Seek())}, decoder=None)
    selected = SelectedDecoder(chain, dict(start_frame=2, end_frame=4), (10, 1))
    frame = SimpleNamespace(pts=0, time_base_num=1, time_base_den=1000)
    assert selected.next(frame) == 1 and frame.pts == 0
    assert selected.next(frame) == 1 and frame.pts == 100
    assert selected.source_first_seconds == pytest.approx(.2)


def test_vfr_selection_refused_instead_of_wrong_frame_index():
    pts = iter([0, 100, 240])
    def read(frame):
        frame.pts = next(pts)
        return 1
    selected = SelectedDecoder(SimpleNamespace(next_frame=read), dict(start_frame=0, end_frame=3), 10)
    frame = SimpleNamespace(pts=0, time_base_num=1, time_base_den=1000)
    selected.next(frame); selected.next(frame)
    with pytest.raises(ValueError, match='可变帧率'):
        selected.next(frame)


def test_remux_selection_applies_only_to_source_input():
    from gpu_transfer_export import build_mux_command
    command = build_mux_command('ffmpeg', 'encode.mkv', 'source.mkv', dict(
        _source_range_start_seconds=2.5, _source_range_duration_seconds=.5), 'output.mkv')
    assert command[command.index('encode.mkv') + 1:command.index('source.mkv')] == [
        '-ss', '2.500000000', '-t', '0.500000000', '-i']
    assert command[command.index('-copypriorss:a')+1] == '0'
    assert command[command.index('-avoid_negative_ts')+1] == 'disabled'
    assert command[command.index('-bsf:s')+1].startswith('setts=')


def test_cache_green_requires_completed_job_existing_file_and_current_identity(tmp_path):
    from preview_segments import PreviewSegments
    source, output = tmp_path / 'source.mkv', tmp_path / 'cache.mkv'
    source.write_bytes(b'source'); output.write_bytes(b'encoded')
    job = dict(id='123', status='running', progress=20)
    jobs = [job]
    cache = PreviewSegments(SimpleNamespace(get_jobs=lambda: jobs), tmp_path)
    identity = cache.identity(source, {'intensity': 1}, {})
    cache.records['key'] = dict(key='key', identity=identity, job_id='123', output_path=str(output))
    assert not cache.list(source, {'intensity': 1}, {})[0]['ready']
    job['status'] = 'completed'
    assert cache.list(source, {'intensity': 1}, {})[0]['ready']
    jobs.clear()
    assert cache.list(source, {'intensity': 1}, {})[0]['ready']
    assert PreviewSegments(SimpleNamespace(get_jobs=lambda: []), tmp_path).list(source, {'intensity': 1}, {})[0]['ready']
    assert cache.list(source, {'intensity': .8}, {}) == []
    source.write_bytes(b'replaced source')
    assert cache.list(source, {'intensity': 1}, {}) == []


def test_mux_time_accounts_for_video_and_container_origins(monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, 'check_output', lambda *args, **kwargs: b'{"format":{"start_time":"5.0"}}')
    selected = SelectedDecoder(SimpleNamespace(video='source.mkv'), dict(start_frame=6,end_frame=12), 30)
    selected.origin, selected.source_first_seconds = 5.021, .2
    assert selected.mux_settings({}, 'ffprobe')['_source_range_start_seconds'] == pytest.approx(.221)
