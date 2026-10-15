"""Activate only the verified optional AI overlay installed by the desktop app."""
import hashlib
import json
import os
from pathlib import Path
import sys


def configure():
    if os.environ.get('LUNARC_COMPONENT_MODE') != '1':
        return
    root = Path(os.environ['LUNARC_COMPONENT_ROOT']).resolve()
    resources = Path(__file__).resolve().parent
    try:
        marker = json.loads((root / 'raft-ready.json').read_text(encoding='utf-8'))
        digest = hashlib.sha256(b'3.10.18')
        digest.update((resources / 'components/raft.lock').read_text(encoding='utf-8').replace('\r\n','\n').encode('utf-8'))
        digest.update((resources / 'components/raft-model.json').read_bytes())
        if marker.get('identity') != digest.hexdigest():
            return
        site = root / 'raft-v1/site'
        if (site / 'torch/__init__.py').is_file():
            sys.path.insert(0, str(site))
            try:
                image_marker=json.loads((root/'image-models-ready.json').read_text(encoding='utf-8'))
                image_hash=hashlib.sha256(b'3.10.18')
                image_hash.update((resources/'components/image-models.lock').read_text(encoding='utf-8').replace('\r\n','\n').encode('utf-8'))
                image_site=root/'image-models-v1/site'
                if image_marker.get('identity')==image_hash.hexdigest() and (image_site/'spandrel/__init__.py').is_file():
                    sys.path.insert(0,str(image_site))
            except (OSError,ValueError):pass
    except (OSError, ValueError):
        return
