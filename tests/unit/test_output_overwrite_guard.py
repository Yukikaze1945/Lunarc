"""Finding 1: an export may never be pointed at the file it reads.

Each test proves the refusal by making the next step explode - a probe, a bridge load, a
subprocess. If the code reached that step it would fail the test loudly instead of
letting a truncated source hide behind a passed assertion.
"""
import ctypes
import os
import subprocess
import threading

import pytest

import gpu_transfer_export as gtx
import pipeline
from dlss_service import DLSSService


def _boom(*args, **kwargs):
    raise AssertionError(
        f"the export reached a stage past the guard: {args!r} {sorted(kwargs)!r}")


class _ExplodingFFmpeg:
    """Any Popen here is a real encoder opening both files."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("an FFmpeg process was started after the refusal")


class _ExplodingModule:
    """Standing in for gpu_transfer_export: touching it at all is the failure."""

    def __getattr__(self, name):
        raise AssertionError(f"the device chain reached {name} after the refusal")


def _source(tmp_path, name="clip.mkv"):
    src = tmp_path / name
    src.write_bytes(b"the only copy of this file")
    return src


# --- path identity -------------------------------------------------------------

def test_the_same_path_is_refused(tmp_path):
    src = _source(tmp_path)
    with pytest.raises(pipeline.OutputOverwritesInputError):
        pipeline.assert_output_not_input(str(src), str(src))


def test_windows_spelling_differences_are_the_same_file(tmp_path):
    src = _source(tmp_path)
    text = str(src)
    aliases = {
        "forward slashes": text.replace("\\", "/"),
        "upper case": text.upper(),
        "trailing dots": text + "...",
        "trailing space": text + " ",
        "dot segment": os.path.join(os.path.dirname(text), ".", os.path.basename(text)),
        "extended length": "\\\\?\\" + text,
    }
    if os.name != "nt":
        aliases = {"forward slashes": aliases["forward slashes"],
                   "dot segment": aliases["dot segment"]}
    for label, alias in aliases.items():
        assert pipeline.same_underlying_file(text, alias), label
        with pytest.raises(pipeline.OutputOverwritesInputError):
            pipeline.assert_output_not_input(text, alias)


def test_a_renamed_or_moved_target_is_a_different_file(tmp_path):
    """The honest other half: distinct valid files must not be rejected.

    A different name, a different folder, or only a different container are all
    perfectly writable targets, and a guard that refuses them turns the panel unusable.
    """
    src = _source(tmp_path)
    (tmp_path / "out").mkdir()
    distinct = [
        str(tmp_path / "other.mkv"),
        str(tmp_path / "out" / src.name),
        str(tmp_path / (src.stem + ".mp4")),
    ]
    for other in distinct:
        assert not pipeline.same_underlying_file(str(src), other), other
        pipeline.assert_output_not_input(str(src), other)


def test_an_8_3_short_name_alias_is_the_same_file(tmp_path):
    src = _source(tmp_path, "a_long_source_name.mkv")
    if os.name != "nt":
        pytest.skip("short names are a Windows path feature")
    buffer = ctypes.create_unicode_buffer(260)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(src), buffer, 260):
        pytest.skip("GetShortPathNameW is unavailable here")
    short = buffer.value
    if not short or short.lower() == str(src).lower():
        pytest.skip("short-name generation is disabled on this volume")
    assert pipeline.same_underlying_file(str(src), short)


def test_a_symlink_alias_is_the_same_file(tmp_path):
    src = _source(tmp_path)
    link = tmp_path / "alias.mkv"
    try:
        os.symlink(str(src), str(link))
    except OSError as e:
        pytest.skip(f"this checkout cannot create symlinks: {e}")
    assert pipeline.same_underlying_file(str(src), str(link))


def test_a_hardlink_is_the_same_file(tmp_path):
    src = _source(tmp_path)
    twin = tmp_path / "twin.mkv"
    try:
        os.link(str(src), str(twin))
    except (OSError, AttributeError) as e:
        pytest.skip(f"hard links are unavailable here: {e}")
    assert str(src) != str(twin), "the two names are two directory entries"
    if not (os.stat(str(src)).st_ino and os.stat(str(twin)).st_ino):
        pytest.skip("this filesystem reports no file index, so one record behind two "
                    "names is not detectable - and two names are then two files")
    assert pipeline.same_underlying_file(str(src), str(twin))


def test_no_source_is_never_a_match(tmp_path):
    src = _source(tmp_path)
    assert not pipeline.same_underlying_file("", str(src))
    assert not pipeline.same_underlying_file(str(src), None)
    pipeline.assert_output_not_input(str(src), "")
    pipeline.assert_output_not_input("", "")


# --- the guard precedes every stage that could write ---------------------------

def test_build_transcode_command_refuses_before_ffmpeg_is_located(tmp_path, monkeypatch):
    # The command is what carries -y; if it were built, the caller already holds a
    # truncating command line.
    src = str(_source(tmp_path))
    monkeypatch.setattr(pipeline, "_bundle_ffmpeg", _boom)
    with pytest.raises(pipeline.OutputOverwritesInputError):
        pipeline.build_transcode_command(src, src, 1920, 1080, 60.0, {}, 16, True)


def test_legacy_pipe_refuses_before_any_process_starts(tmp_path, monkeypatch):
    src = str(_source(tmp_path))
    monkeypatch.setattr(subprocess, "Popen", _ExplodingFFmpeg)
    monkeypatch.setattr(pipeline, "video_info", _boom)
    with pytest.raises(pipeline.OutputOverwritesInputError):
        pipeline.export_dlss_stream(src, live=None, frames=[0, 1], fps=60.0,
                                    cq=16, with_audio=False, out_path=src)


def test_the_derived_default_target_reaches_the_encoder_lookup(tmp_path, monkeypatch):
    """With no destination named, the default is derived and must not be refused.

    The default is ``<stem>_dlss.<ext>`` beside the source - a different file - so this
    test is the guard's other face: it has to let a normal job through to the next step.
    The next step is faked to fail in a way that says so, and the source stays intact.
    """
    src = str(_source(tmp_path))
    monkeypatch.setattr(pipeline, "video_info", lambda path: (2, 60.0, 1920, 1080))
    monkeypatch.setattr(pipeline, "_bundle_ffmpeg", _boom)

    with pytest.raises(AssertionError) as reached:
        pipeline.export_dlss_stream(src, live=None, frames=[0, 1], fps=60.0,
                                    cq=16, with_audio=False, out_path=None)

    assert "past the guard" in str(reached.value)
    assert os.path.exists(src)


def test_gpu_transfer_pipe_refuses_before_the_bridges_load(tmp_path):
    src = str(_source(tmp_path))
    with pytest.raises(pipeline.OutputOverwritesInputError):
        pipeline.export_gpu_transfer_stream(
            video=src, live=None, frames=[0, 1], fps=60.0, cq=16, with_audio=False,
            out_path=src, transcode_settings={}, job_settings={}, flow_provider=None,
            module=_ExplodingModule())


def test_device_chain_refuses_before_the_encoder_runs(tmp_path):
    src = str(_source(tmp_path))
    with pytest.raises(pipeline.OutputOverwritesInputError):
        gtx.export_gpu_transfer(src, None, None, src, {}, [0], 16, False,
                                {}, {}, {}, bridges=_ExplodingModule())


def test_the_copy_pass_refuses_before_assembling_a_command(tmp_path):
    # This is the one that would have run -y with the source as input 1.
    src = str(_source(tmp_path))
    with pytest.raises(pipeline.OutputOverwritesInputError):
        gtx.build_mux_command("ffmpeg", str(tmp_path / "encoded.mkv"), src,
                              {"codec": "av1_nvenc"}, src)


def test_add_job_refuses_before_the_job_enters_the_queue(tmp_path):
    src = str(_source(tmp_path))
    service = DLSSService()

    with pytest.raises(ValueError, match="同一个文件"):
        service.add_job({"input_path": src, "output_path": src,
                         "settings": {**service.settings}})

    assert service.get_jobs() == [], "a refused job must not sit in the queue at all"


def test_the_queue_refuses_before_probing_the_source(tmp_path, monkeypatch):
    src = str(_source(tmp_path))
    service = DLSSService()
    job = service.add_job({"input_path": src,
                           "output_path": str(tmp_path / "different.mkv"),
                           "settings": {**service.settings}})
    # Written after the queue accepted it, which is how a settings file or a second
    # client can end up naming the source: the run-time check is what has to catch it.
    job["output_path"] = src
    monkeypatch.setattr(pipeline, "video_info", _boom)

    service._run_queue(None)

    assert job["status"] == "failed"
    assert "同一个文件" in (job["error"] or "")
    assert job["progress"] == 0.0
    assert open(src, "rb").read() == b"the only copy of this file", "source untouched"


def test_server_reports_the_refusal_as_a_reason_not_a_crash(tmp_path, monkeypatch):
    import asyncio
    import server
    from fastapi import HTTPException

    src = str(_source(tmp_path))
    service = DLSSService()
    monkeypatch.setattr(server, "dlss_service", service)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(server.add_to_queue({
            "input_path": src, "output_path": src, "settings": {**service.settings}}))

    assert raised.value.status_code == 400
    assert "同一个文件" in raised.value.detail
    assert service.get_jobs() == []


def test_the_guard_answers_the_same_on_the_worker_thread(tmp_path):
    """gui.py builds its command inside a worker thread, so path identity must not
    depend on which thread asks: a different answer there is how a source gets cut."""
    src = str(_source(tmp_path))
    seen = {}

    def worker():
        try:
            pipeline.assert_output_not_input(src, src)
            seen["raised"] = False
        except pipeline.OutputOverwritesInputError:
            seen["raised"] = True

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert seen["raised"] is True


def test_a_refusal_names_the_file_and_says_nothing_was_written(tmp_path):
    src = _source(tmp_path)
    with pytest.raises(pipeline.OutputOverwritesInputError) as refused:
        pipeline.assert_output_not_input(str(src), str(src))

    text = str(refused.value)
    assert src.name in text
    assert "源素材未被改动" in text
    # The two moves that fix it, named as the controls the operator can find.
    assert "【导出目标文件夹】" in text and "【文件名】" in text
    assert src.read_bytes() == b"the only copy of this file"
