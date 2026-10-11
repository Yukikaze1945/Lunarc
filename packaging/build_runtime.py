"""Build an isolated app-local runtime from a validated Windows Python environment.

Copies only the dependency closure declared below, including RECORD license files,
CPython/stdlib and PE import dependencies. No activation scripts, source environment
paths, registry Python, pip or user site are needed on the receiving computer.
Use --source-python to select a build environment; never modifies that environment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ['numpy', 'opencv-python', 'torch', 'torchvision', 'fastapi', 'uvicorn',
            'pydantic', 'Pillow', 'av']
INVENTORY = r'''
import importlib.metadata as m, json, sys, sysconfig
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
todo=json.loads(sys.argv[1]); found={}
while todo:
    name=todo.pop(); key=canonicalize_name(name)
    if key in found: continue
    dist=m.distribution(name)
    files=[str(dist.locate_file(f).resolve()) for f in dist.files or []]
    if not files: raise RuntimeError('No file manifest: '+name)
    found[key]={'name':dist.metadata['Name'],'version':dist.version,'files':files}
    for text in dist.requires or []:
        req=Requirement(text)
        if req.marker is None or req.marker.evaluate({'extra':''}): todo.append(req.name)
print(json.dumps({'prefix':sys.base_prefix,'site':sysconfig.get_path('purelib'),
 'version':sys.version.split()[0],'minor':f'{sys.version_info.major}{sys.version_info.minor}',
 'distributions':list(found.values())}))
'''


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    # Repeated builds reuse identical files, but changed inputs are always copied.
    if target.exists() and source.stat().st_size == target.stat().st_size and source.stat().st_mtime_ns == target.stat().st_mtime_ns:
        return
    shutil.copy2(source, target)


def copy_tree(source, target):
    for path in source.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            copy_file(path, target / path.relative_to(source))


def imports(path):
    import pefile
    try:
        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT'],
                                              pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT']])
        names = [i.dll.decode('ascii') for key in ('DIRECTORY_ENTRY_IMPORT', 'DIRECTORY_ENTRY_DELAY_IMPORT')
                 for i in getattr(pe, key, [])]
        pe.close()
        return names
    except pefile.PEFormatError:
        return []


def collect_dependencies(output, search_roots, native):
    """Resolve PE imports, omitting OS/driver DLLs. Fail on unresolved runtime DLLs."""
    sys.path.insert(0, str(ROOT))
    import bundle_paths
    never = {n.lower() for n in bundle_paths.NEVER_BUNDLE}
    system = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32'
    candidates = {}
    for directory in search_roots:
        if directory.is_dir():
            for path in directory.glob('*.dll'):
                candidates.setdefault(path.name.lower(), path)
    # Modules normally load their own private .libs/torch/lib dependencies; these
    # remain in their wheel layout rather than being flattened or taken from PATH.
    carried = {p.name.lower() for p in output.rglob('*') if p.suffix.lower() in ('.dll', '.pyd')}
    carried.update(p.name.lower() for p in native.glob('*.dll'))
    todo = [p for p in output.rglob('*') if p.suffix.lower() in ('.exe', '.pyd', '.dll')]
    todo.extend(native.glob('*.dll'))
    seen = set()
    dependencies = []
    while todo:
        path = todo.pop()
        if str(path) in seen:
            continue
        seen.add(str(path))
        for name in imports(path):
            key = name.lower()
            private_package = path.is_relative_to(output / 'python/Lib/site-packages')
            if ((key in carried and private_package) or (path.parent / name).is_file()
                    or (output / 'python' / name).is_file() or key in never
                    or key.startswith(('api-ms-', 'ext-ms-'))
                    or key in {p.name.lower() for p in native.glob('*.dll')}):
                continue
            source = candidates.get(key)
            # VC runtime is app-local if the build environment supplies it. OS
            # libraries remain external; never copy from Windows/System32.
            if source is None and (system / name).is_file():
                continue
            if source is None:
                raise FileNotFoundError(f'{path.name} requires {name}; provide its runtime directory using --dll-dir')
            target = output / 'python' / name
            copy_file(source, target)
            carried.add(key)
            dependencies.append(target.relative_to(output).as_posix())
            todo.append(target)
    return dependencies


def build(args):
    result = subprocess.run([args.source_python, '-c', INVENTORY, json.dumps(PACKAGES)],
                            check=True, capture_output=True, text=True, encoding='utf-8')
    inventory = json.loads(result.stdout)
    prefix, site = Path(inventory['prefix']), Path(inventory['site'])
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not (output / 'build-in-progress.json').is_file() and not (output / 'manifest.json').is_file():
        raise ValueError('Output must be a new directory or an existing generated runtime')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'build-in-progress.json').write_text(json.dumps({'schema': 1}), encoding='utf-8')
    python = output / 'python'
    for name in ('python.exe', 'python3.dll', f"python{inventory['minor']}.dll"):
        copy_file(prefix / name, python / name)
    for name in ('LICENSE_PYTHON.txt', 'LICENSE.txt'):
        if (prefix / name).is_file():
            copy_file(prefix / name, python / name)
    for path in (prefix / 'Lib').rglob('*'):
        relative = path.relative_to(prefix / 'Lib')
        if path.is_file() and relative.parts[0] not in ('site-packages', 'test', 'idlelib', 'tkinter', 'ensurepip', 'venv') and '__pycache__' not in relative.parts:
            copy_file(path, python / 'Lib' / relative)
    for path in (prefix / 'DLLs').glob('*'):
        if path.suffix.lower() in ('.pyd', '.dll') and not ('test' in path.name or '_tkinter' in path.name):
            copy_file(path, python / path.name)
    for dist in inventory['distributions']:
        print(f"Collecting {dist['name']} {dist['version']}", flush=True)
        for filename in dist.pop('files'):
            path = Path(filename)
            # RECORD may include ../../../Scripts or share/. Do not copy console
            # launchers with baked-in prefixes or write outside the private runtime.
            if path.is_file() and path.is_relative_to(site) and '__pycache__' not in path.parts:
                copy_file(path, python / 'Lib/site-packages' / path.relative_to(site))
    # _pth implies isolated mode: ignores PYTHONHOME/PYTHONPATH, user site and
    # registry. Relative application root works in both source and packaged layout.
    (python / f"python{inventory['minor']}._pth").write_text('.\nLib\nLib/site-packages\n../..\n', encoding='utf-8')
    ffmpeg = Path(args.ffmpeg_dir).resolve()
    for name in ('ffmpeg.exe', 'ffprobe.exe'):
        copy_file(ffmpeg / name, output / 'tools' / name)
    for path in ffmpeg.glob('*.dll'):
        copy_file(path, output / 'tools' / path.name)
    for path in ffmpeg.parent.glob('*LICENSE*'):
        if path.is_file():
            copy_file(path, output / 'licenses/ffmpeg' / path.name)
    for path in ffmpeg.parent.glob('*COPYING*'):
        if path.is_file():
            copy_file(path, output / 'licenses/ffmpeg' / path.name)
    search = [prefix, prefix / 'DLLs', prefix / 'Library/bin', ffmpeg] + [Path(p) for p in args.dll_dir]
    dependencies = collect_dependencies(output, search, ROOT / 'native')
    raft = Path(args.raft_weights or ROOT / 'torch_home/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth')
    if not raft.is_file():
        raise FileNotFoundError('Provide the official RAFT-Large checkpoint using --raft-weights')
    copy_file(raft, output / 'torch_home/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth')
    # Proprietary runtimes are opt-in inputs for private/local bundles. Public
    # redistribution must independently satisfy the vendor's license.
    if args.vfx_runtime:
        source = Path(args.vfx_runtime)
        for directory in ('bin', 'features', 'nvvfx', 'license'):
            if (source / directory).is_dir():
                copy_tree(source / directory, output / 'vfx' / directory)
    if args.dlaa_dll:
        copy_file(Path(args.dlaa_dll), output / 'dlss/nvngx_dlss.dll')
    print('Validating isolated imports and CUDA...', flush=True)
    code = 'import numpy,cv2,fastapi,uvicorn,pydantic,torch,torchvision,av,PIL,ssl,sqlite3; assert torch.cuda.is_available(); x=torch.ones(8,device="cuda")+1; torch.cuda.synchronize(); assert x.sum().item()==16; print(torch.__version__)'
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX')}
    env['PATH'] = str(python) + os.pathsep + str(output / 'tools') + os.pathsep + str(Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32')
    subprocess.run([str(python / 'python.exe'), '-c', code], check=True, env=env, timeout=90)
    manifest = {'schema': 1, 'python': inventory['version'], 'packages': inventory['distributions'],
                'pe_dependencies': dependencies, 'proprietary_vfx_included': bool(args.vfx_runtime),
                'proprietary_dlaa_included': bool(args.dlaa_dll), 'files': {}}
    print('Writing integrity manifest...', flush=True)
    for path in output.rglob('*'):
        if path.is_file() and path.name not in ('manifest.json', 'build-in-progress.json') and '__pycache__' not in path.parts:
            manifest['files'][path.relative_to(output).as_posix()] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'build-in-progress.json').unlink()
    print(f'Runtime complete: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-python', default=sys.executable)
    parser.add_argument('--output', default=str(ROOT / 'build/standalone-runtime'))
    parser.add_argument('--ffmpeg-dir', required=True, help='Real bin directory; no Scoop shims')
    parser.add_argument('--dll-dir', action='append', default=[])
    parser.add_argument('--raft-weights')
    parser.add_argument('--vfx-runtime')
    parser.add_argument('--dlaa-dll')
    build(parser.parse_args())
