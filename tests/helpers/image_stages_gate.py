"""Small odd-size image validates multilayer, VSR and TrueHDR image routing."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gpu_preview_engine import GpuPreview


def main():
    out=ROOT/'build/image-export-gate';out.mkdir(parents=True,exist_ok=True)
    source=out/'odd.png'
    image=np.zeros((289,513,4),np.uint8)
    image[:,:,:3]=np.linspace(0,255,513,dtype=np.uint8)[None,:,None]
    image[:,:,3]=255;image[0,0,3]=0
    Image.fromarray(image).save(source)
    base=dict(guidance_mode=0,dlss_backend='gpu_flow_d3d12',export_backend='gpu_transfer',
        working_scale=1,reconstruction_mode='single',residual_strength=1,
        vsr_runtime_path=r'F:\DLSSC\VideoFX',vsr_mode='standard',vsr_quality=3,
        vsr_output_mode='scale',vsr_scale=2)
    cases=[dict(base,render_pipeline=['nr'],reconstruction_mode='dual',working_scale=.5,
            layer_scales=[.5,.75],layer_weights=[1,1],fusion_mode='sequential'),
           dict(base,render_pipeline=['vsr']),
           dict(base,render_pipeline=['truehdr'],truehdr_enabled=True)]
    results=[]
    for index,settings in enumerate(cases):
        preview=GpuPreview()
        try:
            result=preview.export_image(str(source),settings,str(out/f'case-{index}.png'),'png',95,True,True)
            expected=(1026,578) if index==1 else (513,289)
            with Image.open(result['path']) as saved:
                assert saved.size==expected
                assert np.array(saved.convert('RGBA'))[0,0,3]==0
            if index==0:assert result['diagnostics']['layer_count']==2
            if index==2:assert result['hdr_mapped_to_sdr'] is True
            results.append(result)
        finally:preview.shutdown()
    (out/'stages.json').write_text(json.dumps(dict(passed=True,cases=results),indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,pipelines=[s['render_pipeline'] for s in cases])))


if __name__=='__main__':main()
