"""Bounded install checks run with the managed Python, never in the live backend."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def install_ffmpeg(archive, runtime):
    runtime = Path(runtime).resolve()
    tools = runtime / 'tools'
    staging = runtime / 'ffmpeg-staging'
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        for name in ('ffmpeg.exe', 'ffprobe.exe'):
            matches = [n for n in names if n.endswith('/bin/' + name)]
            if len(matches) != 1:
                raise ValueError('Unexpected FFmpeg archive layout')
            with package.open(matches[0]) as source, (staging / name).open('wb') as target:
                shutil.copyfileobj(source, target)
        licenses = runtime / 'licenses' / 'ffmpeg'
        licenses.mkdir(parents=True, exist_ok=True)
        for name in names:
            leaf = Path(name).name
            if leaf.lower().startswith(('license', 'copying', 'readme')):
                (licenses / leaf).write_bytes(package.read(name))
    for name in ('ffmpeg.exe', 'ffprobe.exe'):
        subprocess.run([str(staging / name), '-version'], check=True, capture_output=True, timeout=15)
    result = subprocess.run([str(staging / 'ffmpeg.exe'), '-hide_banner', '-filters'],
                            check=True, capture_output=True, text=True, timeout=15)
    present = {row.split()[1] for row in result.stdout.splitlines() if len(row.split()) > 2}
    if not {'zscale', 'scale', 'select', 'showinfo'}.issubset(present):
        raise ValueError('Required FFmpeg filters absent')
    tools.mkdir(parents=True, exist_ok=True)
    for name in ('ffmpeg.exe', 'ffprobe.exe'):
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
