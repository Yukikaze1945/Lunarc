"""The online FFmpeg component must carry the native runtime as well as CLI tools."""
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import component_install as installer


def make_archive(path, *, shared=True, duplicate=False):
    with zipfile.ZipFile(path, 'w') as package:
        for name in ('ffmpeg.exe','ffprobe.exe', *(installer.FFMPEG_LIBRARIES if shared else ())):
            package.writestr('ffmpeg/bin/' + name, b'fixture')
        package.writestr('ffmpeg/LICENSE', b'license')
        if duplicate:
            package.writestr('another/bin/avcodec-62.dll', b'duplicate')


def test_shared_component_installs_cli_and_dlls_and_can_replace_previous_notices(tmp_path, monkeypatch):
    archive = tmp_path / 'shared.zip'
    make_archive(archive)
    def run(args, **kwargs):
        return SimpleNamespace(stdout=' T.. zscale description\n T.. scale description\n T.. select description\n T.. showinfo description\n')
    monkeypatch.setattr(installer.subprocess, 'run', run)
    runtime = tmp_path / 'runtime'
    for _ in range(2):
        installer.install_ffmpeg(archive, runtime)
        for name in ('ffmpeg.exe','ffprobe.exe', *installer.FFMPEG_LIBRARIES):
            assert (runtime / 'tools' / name).read_bytes() == b'fixture'
    assert not list(runtime.glob('ffmpeg-staging-*'))


@pytest.mark.parametrize('duplicate', [False, True])
def test_cli_only_or_duplicate_runtime_archive_cannot_complete(tmp_path, duplicate):
    archive = tmp_path / 'bad.zip'
    make_archive(archive, shared=duplicate, duplicate=duplicate)
    with pytest.raises(ValueError, match='shared libraries absent|Duplicate'):
        installer.install_ffmpeg(archive, tmp_path / 'runtime')
    assert not (tmp_path / 'runtime/tools/ffmpeg.exe').exists()


def test_managed_shared_library_directory_precedes_unrelated_path(tmp_path, monkeypatch):
    import gpu_transfer_export as gtx
    tools = tmp_path / 'runtime/tools'
    tools.mkdir(parents=True)
    (tools / 'avcodec-62.dll').write_bytes(b'fixture')
    legacy = tmp_path / 'old'
    legacy.mkdir()
    (legacy / 'avcodec-61.dll').write_bytes(b'fixture')
    monkeypatch.setenv('LUNARC_RUNTIME_DIR', str(tools.parent))
    monkeypatch.delenv('FFMPEG_SDK_DIR', raising=False)
    monkeypatch.setenv('PATH', str(legacy))
    monkeypatch.setattr(gtx.pipeline, '_bundle_ffmpeg', lambda: str(legacy / 'ffmpeg.exe'))
    assert gtx.ffmpeg_dll_directories()[0] == str(tools)
