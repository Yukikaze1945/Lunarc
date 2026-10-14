"""Bounded reproduction of the saved render chain around a reported frame."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['LUNARC_GPU_DIAGNOSTICS'] = '1'
import pipeline
import native_effects
import gpu_transfer_export as gtx
import gpu_hdr_transfer_export as htx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video')
    parser.add_argument('settings', type=Path)
    parser.add_argument('--first', type=int, default=470)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--temporal-only', action='store_true')
    parser.add_argument('--transcode', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = json.loads(args.settings.read_text('utf-8'))
    settings.update(_frame_range={'start_frame':args.first,'end_frame':args.first+args.frames})
    if args.temporal_only:
        settings.update(render_pipeline=[], _pipeline_bypass=True, truehdr_enabled=False,
                        guidance_mode=0, post_controls={})
    source = gtx.probe_source(args.video)
    size = native_effects.processing_size(settings, (source['width'],source['height']))
    live = native_effects.make_live(*size, settings, wide=True)
    flow = native_effects.make_flow_provider(settings, live)
    evidence = {}
    original = htx.export_gpu_hdr_transfer
    def capture(**kwargs):
        result, report = original(**kwargs)
        evidence['report'] = report
        return result, report
    htx.export_gpu_hdr_transfer = capture
    tx = dict(codec='av1_nvenc',profile='main',pixel_format='p010le',container='.mkv',
              preset='p7',tune='uhq',rate_mode='cq',cq=24,aq_mode='off',
              audio_mode='remove',subtitle_mode='remove',copy_metadata=False,
              copy_chapters=False,copy_attachments=False,resolution_mode='original')
    if args.transcode:
        tx.update(json.loads(args.transcode.read_text('utf-8')))
        tx.update(audio_mode='remove',subtitle_mode='remove',copy_metadata=False,
                  copy_chapters=False,copy_attachments=False)
    try:
        pipeline.export_gpu_transfer_stream(video=args.video, live=live, frames=args.frames,
            fps=120, cq=24, with_audio=False, out_path=str(args.output/'output.mkv'),
            transcode_settings=tx, job_settings=settings, flow_provider=flow)
        evidence['passed'] = True
        print('SHORT EXPORT PASSED', flush=True)
    except Exception:
        evidence['error'] = traceback.format_exc()
        raise
    finally:
        live.close()
        if flow:
            flow.close_independent_layer()
        (args.output/'result.json').write_text(json.dumps(evidence,indent=2,default=str),encoding='utf-8')


if __name__ == '__main__':
    main()
