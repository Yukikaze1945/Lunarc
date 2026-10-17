"""One small image, two generations; no model matrix or performance benchmark."""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('LUNARC_DATA_DIR',str(Path(os.environ['APPDATA'])/'DaVinci_DLSS5_Studio'))

import numpy as np
from PIL import Image
from gpu_preview_engine import GpuPreview
from image_render import preview_pair,save_result


def main():
    folder=ROOT/'build/image-memory-smoke'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'source.png'
    pixels=np.full((145,257,4),255,np.uint8)
    pixels[:,:,:3]=np.linspace(0,255,257,dtype=np.uint8)[None,:,None]
    pixels[:16,:16,3]=0
    Image.fromarray(pixels).save(path)
    settings=dict(guidance_mode=0,dlss_backend='gpu_flow_d3d12',export_backend='gpu_transfer',
        working_scale=1,reconstruction_mode='single',residual_strength=1,
        render_pipeline=['nr','truehdr','vsr'],truehdr_enabled=True,
        vsr_runtime_path=r'F:\DLSSC\VideoFX',vsr_mode='bicubic',vsr_quality=3,
        vsr_output_mode='scale',vsr_scale=2)
    engine=GpuPreview()
    try:
        first=engine.render_image(str(path),settings,'memory')
        previous=engine.image_results.get(first['id'])
        assert previous.hdr and previous.pixels.dtype==np.float16
        assert previous.pixels.shape==(290,514,4)
        assert previous.pixels[0,0,3]==0
        assert engine.live is None and engine.chain is None
        assert all(stage['resources_released'] for stage in first['diagnostics']['image_passes'])
        # A second HDR generation tests release/reopen and previous-result comparison.
        second=engine.render_image(str(path),dict(settings,truehdr_contrast=120),'memory')
        current=engine.image_results.get(second['id'])
        assert engine.image_results.previous(second['id']).id==previous.id
        assert len(engine.image_results.entries)==2
        pair=preview_pair(current,previous)
        (folder/'comparison.png').write_bytes(pair)
        saved=save_result(current,str(folder/'cached.png'),'png',95,True,True,'sdr')
        hdr=save_result(current,str(folder/'cached.heic'),'heic',95,True,True,'hdr10')
        assert engine.live is None and engine.chain is None
        with Image.open(saved['path']) as image:
            assert image.size==(514,290) and image.convert('RGBA').getpixel((0,0))[3]==0
        report=dict(passed=True,source_size=[257,145],output_size=[514,290],
            stages=settings['render_pipeline'],generations=2,fp16=True,
            per_stage_resources_released=True,model_chain_closed=True,
            retained_results=len(engine.image_results.entries),png=saved,hdr=hdr,
            first=first,second=second)
        (folder/'evidence.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
        print(json.dumps(dict(passed=True,output_size=report['output_size'],generations=2,
            retained_results=2,per_stage_resources_released=True,
            hdr_save={key:hdr[key] for key in ('bit_depth','color_mode','has_alpha','hdr_mapped_to_sdr')})))
    finally:
        engine.shutdown()


if __name__=='__main__':main()
