"""Real author weights, independent VSR instances and image export on CUDA."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gpu_preview_engine import GpuPreview

def main():
    out=ROOT/'build/image-model-gate';out.mkdir(parents=True,exist_ok=True)
    rgba=np.zeros((73,65,4),np.uint8)
    rgba[:,:,:3]=np.arange(65,dtype=np.uint8)[None,:,None]*3
    rgba[15:55,10:50,:3]=[30,160,220];rgba[:,:,3]=255;rgba[0,0,3]=0
    source=out/'source.png';Image.fromarray(rgba).save(source)
    base=dict(guidance_mode=0,dlss_backend='gpu_flow_d3d12',export_backend='gpu_transfer',
        working_scale=1,reconstruction_mode='single',residual_strength=1,
        vsr_runtime_path=r'F:\DLSSC\VideoFX',vsr_mode='standard',vsr_quality=3,
        vsr_output_mode='scale',vsr_scale=2)
    cases=[('animesharp',['animesharp'],{},2),('illustration',['illustration'],{},4),
        ('vsr-two',['vsr','vsr:second'],{'render_stage_options':{'vsr:second':{'vsr_scale':2,'vsr_quality':2}}},4),
        ('model-vsr',['animesharp','vsr'],{},4),
        ('model-nr-vsr',['animesharp','nr','vsr'],{},4)]
    results=[]
    for name,order,extra,factor in cases:
        preview=GpuPreview()
        try:
            result=preview.export_image(str(source),dict(base,render_pipeline=order,**extra),str(out/(name+'.png')),'png',95,True,True)
            with Image.open(result['path']) as actual:
                assert actual.size==(65*factor,73*factor),actual.size
                assert np.array(actual)[0,0,3]==0
            stages=result['diagnostics']['native_output_effect']['stages']
            stages=[*(result['diagnostics'].get('native_input_effect') or {}).get('stages',[]),*stages]
            assert [stage['stage'] for stage in stages]==[stage for stage in order if stage!='nr']
            assert all(stage['calls']==1 and stage['host_image_bytes']==0 for stage in stages)
            results.append(result)
            print(name,'passed',flush=True)
        finally:preview.shutdown()
    (out/'result.json').write_text(json.dumps(dict(passed=True,cases=results),indent=2),encoding='utf-8')

if __name__=='__main__':main()
