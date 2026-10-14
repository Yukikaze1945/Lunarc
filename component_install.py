"""Bounded install checks run with the managed Python, never in the live backend."""
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import subprocess
import sys
import zipfile
import tempfile


FFMPEG_LIBRARIES = ('avcodec-62.dll', 'avformat-62.dll', 'avfilter-11.dll',
                    'avutil-60.dll', 'swresample-6.dll', 'swscale-9.dll')


def unpack_ffmpeg(archive, staging, licenses):
    """Keep command line tools and their shared runtime from the same archive."""
    tools = ('ffmpeg.exe', 'ffprobe.exe')
    seen = set()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as package:
            for name in package.namelist():
                path = PurePosixPath(name)
                if path.parent.name == 'bin' and (path.name in tools or path.suffix.lower() == '.dll'):
                    target = staging / path.name
                elif path.name.lower().startswith(('license', 'copying', 'readme')) and not name.endswith('/'):
                    target = licenses / path.name
                else:
                    continue
                key = (target.parent, target.name.lower())
                if key in seen:
                    raise ValueError('Duplicate FFmpeg archive member')
                seen.add(key)
                with package.open(name) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
        return
    # Windows ships libarchive tar with 7z support; no extra Python package or
    # developer-installed 7-Zip is required in the managed environment.
    extractor = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/tar.exe'
    result = subprocess.run([str(extractor), '-tf', str(archive)], check=True,
                            capture_output=True, text=True, encoding='utf-8', timeout=30)
    selected = []
    for member in result.stdout.splitlines():
        path = PurePosixPath(member)
        if path.is_absolute() or '..' in path.parts or ':' in member or '\\' in member:
            raise ValueError('Invalid FFmpeg archive path')
        if path.parent.name == 'bin' and (path.name in tools or path.suffix.lower() == '.dll'):
            selected.append(member)
        elif path.name.lower().startswith(('license', 'copying', 'readme')) and not member.endswith('/'):
            selected.append(member)
    if not selected:
        raise ValueError('Unexpected FFmpeg archive layout')
    subprocess.run([str(extractor), '-xf', str(archive), '-C', str(staging), *selected],
                   check=True, capture_output=True, timeout=60)
    for member in selected:
        source = staging / member
        if source.is_symlink() or not source.is_file():
            raise ValueError('Invalid FFmpeg archive file')
        target = staging / source.name if source.parent.name == 'bin' else licenses / source.name
        key = (target.parent, target.name.lower())
        if key in seen:
            raise ValueError('Duplicate FFmpeg archive member')
        seen.add(key)
        shutil.copyfile(source, target)


def install_ffmpeg(archive, runtime):
    runtime = Path(runtime).resolve()
    tools = runtime / 'tools'
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ffmpeg-staging-', dir=runtime) as directory:
        _install_ffmpeg(archive, runtime, Path(directory), tools)


def _install_ffmpeg(archive, runtime, staging, tools):
    licenses = runtime / 'licenses' / 'ffmpeg'
    licenses.mkdir(parents=True, exist_ok=True)
    unpack_ffmpeg(archive, staging, licenses)
    if not all((staging / name).is_file() for name in FFMPEG_LIBRARIES):
        raise ValueError('FFmpeg shared libraries absent; CLI-only archives cannot serve native encoding')
    for name in ('ffmpeg.exe', 'ffprobe.exe'):
        subprocess.run([str(staging / name), '-version'], check=True, capture_output=True, timeout=15)
    result = subprocess.run([str(staging / 'ffmpeg.exe'), '-hide_banner', '-filters'],
                            check=True, capture_output=True, text=True, timeout=15)
    present = {row.split()[1] for row in result.stdout.splitlines() if len(row.split()) > 2}
    if not {'zscale', 'scale', 'select', 'showinfo'}.issubset(present):
        raise ValueError('Required FFmpeg filters absent')
    tools.mkdir(parents=True, exist_ok=True)
    for name in ('ffmpeg.exe', 'ffprobe.exe', *FFMPEG_LIBRARIES, 'avdevice-62.dll'):
        if not (staging / name).exists():
            continue
        os.replace(staging / name, tools / name)


def verify_raft(site, model):
    sys.path.insert(0, site)
    import torch
    import torchvision
    assert torch.cuda.is_available(), 'CUDA unavailable; check NVIDIA driver'
    sample = torch.ones(8, device='cuda') + 1
    torch.cuda.synchronize()
    assert sample.sum().item() == 16
    from torchvision.models.optical_flow import raft_large
    instance = raft_large(weights=None)
    instance.load_state_dict(torch.load(model, map_location='cpu', weights_only=True))
    print(f'Torch {torch.__version__}; torchvision {torchvision.__version__}; RAFT weights loaded')


if __name__ == '__main__':
    if sys.argv[1] == 'ffmpeg':
        install_ffmpeg(*sys.argv[2:4])
    elif sys.argv[1] == 'raft':
        verify_raft(*sys.argv[2:4])
    else:
        raise ValueError('Unknown component')
