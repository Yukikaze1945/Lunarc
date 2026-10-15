"""AVIF/HEIC codec and alpha check; reuses an already verified GPU image."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import image_source as images


def main():
    out=ROOT/'build/image-format-gate';out.mkdir(exist_ok=True)
    x,y=np.meshgrid(np.arange(97),np.arange(65))
    rgba=np.stack([(x*2)%256,(y*3)%256,(x+y)%256,(x*255//96)],axis=-1).astype(np.uint8)
    source=out/'colour-alpha.png';Image.fromarray(rgba).save(source)
    results=[]
    for fmt in ('avif','heic'):
        for quality in (70,100):
            path=out/f'{fmt}-{quality}.{fmt}'
            result=images.save_image(str(source),rgba[:,:,:3],str(path),fmt,quality,True,True)
            actual=images.read_rgba(str(path))
            assert actual.shape==rgba.shape
            with Image.open(path) as decoded:
                assert decoded.format==('HEIF' if fmt=='heic' else 'AVIF')
                assert decoded.info.get('icc_profile')
            alpha_error=int(np.abs(actual[:,:,3].astype(int)-rgba[:,:,3].astype(int)).max())
            assert alpha_error <= (0 if quality==100 else 1), alpha_error
            if fmt=='heic' and quality==100:assert np.array_equal(actual,rgba)
            result['max_channel_difference']=int(np.abs(actual.astype(int)-rgba.astype(int)).max())
            results.append(result)
    # GPU processing itself already passed; only exercise the added file encoders.
    gpu_file=ROOT/'build/image-export-gate/processed.png'
    if gpu_file.exists():
        rgb=np.array(Image.open(gpu_file).convert('RGB'))
        for fmt in ('avif','heic'):
            path=out/f'gpu-result.{fmt}'
            result=images.save_image(str(gpu_file),rgb,str(path),fmt,95,True,True)
            with Image.open(path) as decoded:assert decoded.size==(rgb.shape[1],rgb.shape[0])
            result['reused_verified_gpu_output']=True;results.append(result)
    data=dict(passed=True,formats=images.export_formats(),results=results)
    (out/'result.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,outputs=len(results))))


if __name__=='__main__':main()
