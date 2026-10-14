"""Fetch the pinned shared FFmpeg build, including matching MSVC headers/import libs."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import urllib.request

VERSION = '8.1.2'
NAME = f'ffmpeg-{VERSION}-full_build-shared'
SHA256 = 'cba748035c21ce1431d0823c7a3a711f38616f89f87a265dceddf9b7f6749d2d'
URL = f'https://www.gyan.dev/ffmpeg/builds/packages/{NAME}.7z'


def prepare(output, archive=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = Path(archive).resolve() if archive else output / (NAME + '.7z')
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as source, archive.open('wb') as target:
            shutil.copyfileobj(source, target)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('FFmpeg SDK archive checksum mismatch')
    tar = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/tar.exe'
    listing = subprocess.run([str(tar), '-tf', str(archive)], check=True,
                             capture_output=True, text=True, timeout=30).stdout
    for name in listing.splitlines():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name or path.parts[0] != NAME:
            raise ValueError('Invalid FFmpeg SDK archive layout')
    subprocess.run([str(tar), '-xf', str(archive), '-C', str(output)], check=True, timeout=60)
    sdk = output / NAME
    for name in ('include/libavcodec/avcodec.h','lib/avcodec.lib','bin/avcodec-62.dll'):
        if not (sdk / name).is_file():
            raise ValueError('FFmpeg SDK development files absent')
    (output / 'manifest.json').write_text(json.dumps({'version': VERSION, 'source': URL,
        'sha256': SHA256, 'sdk': NAME}, indent=2), encoding='utf-8')
    print(sdk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(Path(__file__).resolve().parents[1] / 'build/ffmpeg-sdk'))
    parser.add_argument('--archive')
    args = parser.parse_args()
    prepare(args.output, args.archive)
