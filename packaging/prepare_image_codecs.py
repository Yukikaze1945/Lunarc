"""Copy the pinned optional image codec set with its original license notices."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def prepare(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    manifest_path=Path(__file__).with_name('image-codecs.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    # Validate the complete set before modifying the destination.
    for name,expected in manifest['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid codec manifest path')
        path=(source/relative).resolve()
        if not path.is_relative_to(source) or not path.is_file():raise FileNotFoundError(path)
        if path.stat().st_size!=expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError('Image codec checksum mismatch: '+name)
    for name in manifest['files']:
        destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,destination)
    shutil.copy2(manifest_path,output/'manifest.json')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(prepare(args.source,args.output))
