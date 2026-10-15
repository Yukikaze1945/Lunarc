"""Small real GPU lifecycle gate; simulated pressure, never force a GPU OOM."""
import json
from pathlib import Path
import subprocess
import sys
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import preview_memory
from gpu_preview_engine import GpuPreview

def main():
    out=ROOT/'build/preview-memory-gate';out.mkdir(parents=True,exist_ok=True)
    free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
    if free<4096:
        result=dict(skipped=True,reason='Keep 4 GiB free for a bounded lifecycle check; no memory stress test')
    else:
        source=out/'small.png';Image.new('RGB',(128,128),(64,128,192)).save(source)
        settings=dict(render_pipeline=['animesharp'],working_scale=1,reconstruction_mode='single',
            guidance_mode=0,export_backend='gpu_transfer',dlss_backend='gpu_flow_d3d12')
        preview=GpuPreview();real_sample=preview_memory.sample
        try:
            preview.render(str(source),settings,0,need_source=False)
            generation=preview.generation
            report=preview.release_memory()
            assert preview.live is None and preview.chain is None and not preview.raft_shared
            assert report['after']['torch']['allocated_bytes']<report['before']['torch']['allocated_bytes']
            assert report['after']['torch']['reserved_bytes']<report['before']['torch']['reserved_bytes']
            preview.render(str(source),settings,0,need_source=False)
            assert preview.generation==generation+1
            def simulated(live=None):
                sample=real_sample(live)
                if live is not None:sample['process']=dict(local_usage=15*preview_memory.GIB,local_budget=16*preview_memory.GIB)
                return sample
            preview_memory.sample=simulated
            preview.render(str(source),dict(settings,image_model_tile=128),0,need_source=False)
            assert preview.generation==generation+2
            assert preview.memory_release_report['reason']=='memory_pressure_model_switch'
            result=dict(passed=True,pressure='injected threshold sample; physical GPU not filled',manual=report,
                automatic=preview.memory_release_report,generation=preview.generation)
        finally:
            preview_memory.sample=real_sample
            preview.release_memory();preview.shutdown()
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
