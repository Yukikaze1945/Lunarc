"""Collect the small, pinned environment installer; never collect Torch or Python."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

VERSION = '0.13.0'
URL = f'https://github.com/astral-sh/uv/releases/download/{VERSION}/uv-x86_64-pc-windows-msvc.zip'
SHA256 = '088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c'


def prepare(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'uv.zip'
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        with urllib.request.urlopen(URL, timeout=60) as response, archive.open('wb') as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('uv archive checksum mismatch')
    with zipfile.ZipFile(archive) as package:
        members = [n for n in package.namelist() if n.split('/')[-1] == 'uv.exe']
        if len(members) != 1:
            raise ValueError('uv.exe absent or duplicated')
        (output / 'uv.exe').write_bytes(package.read(members[0]))
    executable = output / 'uv.exe'
    (output / 'manifest.json').write_text(json.dumps({'schema': 1, 'version': VERSION,
        'source': URL, 'archive_sha256': SHA256, 'files': {'uv.exe': {
            'bytes': executable.stat().st_size,
            'sha256': hashlib.sha256(executable.read_bytes()).hexdigest()}}}, indent=2), encoding='utf8')
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args().output)
