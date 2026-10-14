"""Two short SDR10 frames verify the production memory ledger on its owner thread."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from gpu_preview_engine import GpuPreview
import gpu_memory

def main():
    directory=ROOT/'build/backend-recovery';directory.mkdir(parents=True,exist_ok=True)
    settings=dict(render_pipeline=['nr','dlaa'],dlaa_preset=12,guidance_mode=0,
        export_backend='gpu_transfer',dlss_backend='gpu_flow_d3d12',working_scale=1,
        reconstruction_mode='single',hdr_paper_white_scrgb=8.,experimental_hdr_scrgb=False)
    preview=GpuPreview()
    try:
        for frame in (0,1):
            _,image,diag=preview.render(str(ROOT/'build/p010-codec-check/sdr709_hd_source.mkv'),
                settings,frame,display_edge=320,need_source=True)
            assert image is not None
        report=preview.executor.submit(lambda:gpu_memory.capture(preview.chain,preview.live,
            phase='preview',preview_cache=preview.post_cache,force=True)).result()
        assert 'error' not in report,report
        assert report['process']['local_budget']>0 and report['process']['local_usage']>0
        rows=[row for row in report['components'] if row['kind'] in ('DLAA','DLSSNR')]
        assert {row['kind'] for row in rows}=={'DLAA','DLSSNR'},rows
        assert len(set(row['session_id'] for row in rows))==len(rows)
        assert all(row['host_staging_bytes']==0 for row in rows)
        assert report['known_device_allocations_bytes']>0
        (directory/'memory-gate.json').write_text(json.dumps({'passed':True,'memory':report},indent=2),encoding='utf-8')
        print(json.dumps({'passed':True,'local_usage':report['process']['local_usage'],
            'known_allocations':report['known_device_allocations_bytes'],'sessions':len(rows)}))
    finally:preview.shutdown()

if __name__=='__main__':main()
