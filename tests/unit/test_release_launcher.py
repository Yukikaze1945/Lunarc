"""Exercise the Windows release launcher with small, real hashed data volumes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

LAUNCHER = Path(__file__).resolve().parents[2] / 'packaging' / 'Install-Lunarc.ps1'
pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows release launcher')


def entry(name, data):
    return dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def package(root, parts_root):
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(LAUNCHER, root / LAUNCHER.name)
    setup = b'test installer -- never executed'
    parts = [b'first volume\x00', b'second volume\xff']
    (root / 'Setup.exe').write_bytes(setup)
    for index, data in enumerate(parts, 1):
        (parts_root / f'data.7z.{index:03d}').write_bytes(data)
    manifest = dict(schema=1, installer=entry('Setup.exe', setup),
                    payload=entry('data.7z', b''.join(parts)),
                    parts=[entry(f'data.7z.{i:03d}', p) for i, p in enumerate(parts, 1)])
    (root / 'installer-payload.json').write_text(json.dumps(manifest), encoding='utf-8')
    return b''.join(parts)


def verify(root):
    powershell = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    return subprocess.run([str(powershell), '-NoProfile', '-ExecutionPolicy', 'Bypass',
                           '-File', str(root / LAUNCHER.name), '-VerifyOnly'],
                          capture_output=True, timeout=30)


@pytest.mark.parametrize('nested', [False, True])
def test_default_explorer_extraction_and_same_directory(tmp_path, nested):
    root = tmp_path / 'Installer' if nested else tmp_path
    expected = package(root, tmp_path)
    result = verify(root)
    assert result.returncode == 0, result.stderr
    assert (root / 'data.7z').read_bytes() == expected
    # A second launch checks the already assembled payload rather than rebuilding.
    assert verify(root).returncode == 0


@pytest.mark.parametrize('damage', ['missing', 'truncated', 'hash', 'traversal'])
def test_bad_download_is_refused_before_assembly(tmp_path, damage):
    root = tmp_path / 'Installer'
    package(root, tmp_path)
    part = tmp_path / 'data.7z.002'
    if damage == 'missing':
        part.unlink()
    elif damage == 'truncated':
        part.write_bytes(b'bad')
    elif damage == 'hash':
        part.write_bytes(b'x' * part.stat().st_size)
    else:
        manifest_path = root / 'installer-payload.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['parts'][0]['name'] = '../data.7z.001'
        manifest_path.write_text(json.dumps(manifest))
    assert verify(root).returncode != 0
    assert not (root / 'data.7z').exists()
    assert not (root / 'data.7z.rebuilding').exists()
