"""Real VSR preview quality gate; display readback is explicit, no quality claim."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gpu_preview_engine import GpuPreview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--directory',required=True,type=Path)
    args=parser.parse_args()
    args.directory.mkdir(parents=True,exist_ok=False)
    preview=GpuPreview();images={};rows=[]
    try:
        for quality in (1,4):
            settings=dict(render_pipeline=['vsr'],neural_mode='nr',guidance_mode=0,
                working_scale=1.,reconstruction_mode='single',export_backend='gpu_transfer',
                vsr_output_mode='scale',vsr_scale=2,vsr_quality=quality)
            for index in range(2):
                original,image,diag=preview.render(str(args.source),settings,index,
                    need_source=True,display_edge=0)
            effect=diag['native_output_effect']['stages'][0]
            assert effect['counters']['quality_level']==quality
            assert effect['counters']['successful_runs']==effect['calls']==2
            assert diag['cached_display'] is False and effect['counters']['model_loaded']==1
            assert image.shape[:2]==(original.shape[0]*2,original.shape[1]*2)
            images[quality]=image.copy()
            cv2.imwrite(str(args.directory/f'quality_{quality}.png'),image)
            cv2.imwrite(str(args.directory/'source.png'),original)
            rows.append(diag)
        delta=np.abs(images[1].astype(np.float32)-images[4].astype(np.float32))
        result=dict(status='PASS',diagnostics=rows,
            low_vs_ultra_mean_display_code_difference=float(delta.mean()),
            low_vs_ultra_different_pixels=int(np.count_nonzero(np.any(delta!=0,axis=2))),
            pixel_comparison='full output display image before JPEG; not a fidelity metric',
            display_host_image_readback='nonzero, explicitly reported by preview diagnostics',
            performance='NOT_CHARACTERIZED')
        (args.directory/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf8')
        print(json.dumps({k:v for k,v in result.items() if k!='diagnostics'}))
    finally:preview.close()


if __name__=='__main__':main()
