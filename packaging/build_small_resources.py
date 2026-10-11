"""Prepare the small installer: verified uv and native PE import closure only."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import bundle_paths
from build_runtime import imports, digest
from prepare_component_tools import prepare


def build(output,prefix):
    output=Path(output).resolve();prefix=Path(prefix).resolve()
    output.mkdir(parents=True,exist_ok=True)
    prepare(output/'component-tools')
    shutil.copytree(ROOT/'packaging/components',output/'components',dirs_exist_ok=True)
    for name in ('uv-LICENSE-MIT.txt','uv-LICENSE-APACHE.txt'):
        shutil.copy2(ROOT/'LICENSES'/name,output/'component-tools'/name)
    native=ROOT/'native';target=output/'native-deps';target.mkdir(exist_ok=True)
    candidates={}
    for directory in (prefix/'Library/bin',prefix,prefix/'DLLs',prefix/'Lib/site-packages/torch/lib'):
        for p in directory.glob('*.dll'):
            candidates.setdefault(p.name.lower(),p)
    own={p.name.lower() for p in native.glob('*.dll')}
    never={n.lower() for n in bundle_paths.NEVER_BUNDLE}
    system=Path(os.environ['SystemRoot'])/'System32'
    pending=list(native.glob('*.dll'));seen=set();copied={}
    while pending:
        file=pending.pop()
        if str(file) in seen:continue
        seen.add(str(file))
        for name in imports(file):
            key=name.lower()
            if key in own or key in never or key.startswith(('api-ms-','ext-ms-')):continue
            source=candidates.get(key)
            if source is None and (system/name).is_file():continue
            if source is None:raise FileNotFoundError(f'{file.name} requires {name}')
            if key in copied:continue
            shutil.copy2(source,target/source.name);copied[key]=source;pending.append(source)
    # Retain the exact Conda package notices and recipes for the shared libraries.
    # Package archives are build inputs, excluded from the installer.
    import zstandard
    cache=output/'notice-archives';cache.mkdir(exist_ok=True)
    notices=output/'native-deps/licenses';notices.mkdir(exist_ok=True)
    packages=[]
    for meta in (prefix/'conda-meta').glob('*.json'):
        m=json.loads(meta.read_text('utf-8'))
        files={f.replace('\\','/').lower() for f in m.get('files',[])}
        selected={p.relative_to(prefix).as_posix().lower() for p in copied.values()}
        if not files.intersection(selected):continue
        url=m.get('url','');sha=m.get('sha256','')
        if not url.startswith('https://conda.anaconda.org/conda-forge/') or len(sha)!=64:
            raise ValueError(f'Unverified package source: {m["name"]}')
        archive=cache/Path(url).name
        if not archive.is_file() or digest(archive)!=sha:
            print('Collect notices:',m['name'],flush=True)
            with urllib.request.urlopen(url,timeout=60) as r,archive.open('wb') as w:shutil.copyfileobj(r,w)
        if digest(archive)!=sha:raise ValueError('Package checksum mismatch')
        directory=notices/f'{m["name"]}-{m["version"]}'
        directory.mkdir(exist_ok=True)
        if archive.suffix=='.conda':
            with zipfile.ZipFile(archive) as z:
                info=next(n for n in z.namelist() if n.startswith('info-') and n.endswith('.tar.zst'))
                data=zstandard.ZstdDecompressor().stream_reader(io.BytesIO(z.read(info)))
        else:
            data=archive.open('rb')
        with data, tarfile.open(fileobj=data,mode='r|*') as tar:
            for member in tar:
                if not member.isfile() or not member.name.startswith(('info/licenses/','info/recipe/')):continue
                relative=Path(member.name)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid notice path')
                destination=directory/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as r,destination.open('wb') as w:shutil.copyfileobj(r,w)
        packages.append({k:m.get(k) for k in ('name','version','build','license','url','sha256')})
    (notices/'packages.json').write_text(json.dumps(packages,indent=2),encoding='utf-8')
    files={p.relative_to(output).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)}
           for folder in ('component-tools','components','native-deps') for p in (output/folder).rglob('*') if p.is_file() and p.name!="uv.zip"}
    (output/'manifest.json').write_text(json.dumps({'schema':1,'mode':'components','files':files},indent=2),encoding='utf-8')
    print('Small resources:',sum(f['bytes'] for f in files.values()),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--native-runtime-prefix',required=True)
    args=parser.parse_args();build(args.output,args.native_runtime_prefix)
