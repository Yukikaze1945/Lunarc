"""Short production preview/export routing checks for VSR mode families."""
import ctypes as C
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pipeline
import native_effects
import gpu_hdr_transfer_export as htx
from gpu_preview_engine import GpuPreview
from vsr_config import configuration


def main():
    directory=ROOT/'build/vsr-modes/integration';directory.mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser();ap.add_argument('--colours',nargs='+',choices=['sdr8','sdr10','hdr'],default=['sdr8','sdr10','hdr'])
    ap.add_argument('--export-only',action='store_true')
    options=ap.parse_args()
    result=json.loads((directory/'result.json').read_text('utf-8')) if (directory/'result.json').exists() else {'rows':[],'performance':'NOT_CHARACTERIZED'}
    result['rows']=[row for row in result['rows'] if (row.get('export') not in options.colours if options.export_only
        else row.get('colour',row.get('export')) not in options.colours)]
    result.pop('passed',None)
    for colour in options.colours:
        source=(ROOT/'build/p010-codec-check/sdr709_hd_source.mkv' if colour=='sdr10'
                else ROOT/'build/hdr_probe/hdr_valid_src.mkv' if colour=='hdr' else directory/f'{colour}.mkv')
        import gpu_transfer_export as gtx
        if colour != 'sdr8' and not source.exists():
            raise FileNotFoundError('Run the existing P010/HDR fixture gates before the wide preview check')
        if colour=='sdr8' and not source.exists():
            tags=['bt2020','smpte2084','bt2020nc'] if colour=='hdr' else ['bt709']*3
            encoder_options=[] if colour=='sdr8' else ['-x265-params',f'log-level=error:colorprim={tags[0]}:transfer={tags[1]}:colormatrix={tags[2]}',
                '-bsf:v','hevc_metadata=colour_primaries=9:transfer_characteristics=16:matrix_coefficients=9' if colour=='hdr'
                else 'hevc_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1']
            subprocess.run([pipeline._bundle_ffmpeg(),'-v','error','-y','-f','lavfi','-i',
                'testsrc2=size=640x360:rate=30','-frames:v','4','-c:v','libx264' if colour=='sdr8' else 'libx265',
                '-preset','ultrafast','-pix_fmt','yuv420p' if colour=='sdr8' else 'yuv420p10le',
                '-color_primaries',tags[0],'-color_trc',tags[1],'-colorspace',tags[2],'-color_range','tv',*encoder_options,str(source)],
                check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        facts=gtx.probe_source(str(source));width,height=facts['width'],facts['height']
        preview=GpuPreview();decoder=context=base_generation=None
        modes=['standard','high_bitrate','streaming','denoise','deblur','bicubic'] if colour=='sdr8' else ['high_bitrate','denoise','streaming']
        if options.export_only:modes=[]
        try:
            for family in modes:
                settings=dict(render_pipeline=['vsr'],guidance_mode=0,working_scale=1,
                    reconstruction_mode='single',experimental_hdr_scrgb=colour=='hdr',
                    export_backend='gpu_transfer',dlss_backend='gpu_flow_d3d12',hdr_paper_white_scrgb=8.,vsr_runtime_path=r'F:\DLSSC\VideoFX',
                    vsr_mode=family,vsr_quality=2,vsr_strength=None if family=='standard' else .6,
                    vsr_output_mode='scale',vsr_scale=2)
                for frame in (0,1):
                    original,image,diag=preview.render(str(source),settings,frame,need_source=True,display_edge=640,reuse_display=True)
                assert np.any(image) and original.shape[1]/original.shape[0]==width/height
                effective=diag['native_output_effect']['stages'][0]
                config=configuration(settings)
                assert effective['counters']['quality_level']==config['sdk_level']
                assert effective['counters']['successful_runs']==2
                assert tuple(effective['output_size'])==((width,height) if config['same_size'] else (width*2,height*2))
                identity=preview.executor.submit(lambda:(preview.chain.decoder.value,preview.current_frame.cuda_context)).result()
                if decoder is not None: assert identity==(decoder,context),'mode switch replaced decoder/context'
                decoder,context=identity
                if family=='standard':base_generation=diag['generation']
                if colour=='sdr8' and family in ('high_bitrate','streaming'):assert diag['generation']==base_generation
                result['rows'].append(dict(colour=colour,family=family,diagnostics=diag,decoder=decoder,context=context))
                (directory/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
                print(colour,family,'generation',diag['generation'],'decoder retained',flush=True)
            preview.close()
        finally:preview.shutdown()
        if colour!='sdr8':
            # Same-size denoise export must use the original GPU P010 route.
            settings=dict(render_pipeline=['vsr'],vsr_mode='denoise',vsr_quality=2,vsr_strength=.6,
                vsr_runtime_path=r'F:\DLSSC\VideoFX',export_backend='gpu_transfer',dlss_backend='gpu_flow_d3d12',
                experimental_hdr_scrgb=colour=='hdr',hdr_paper_white_scrgb=8.,guidance_mode=0,
                working_scale=1,reconstruction_mode='single',vsr_output_mode='scale',vsr_scale=2)
            live=native_effects.make_live(width,height,settings,wide=True)
            original_export=htx.export_gpu_hdr_transfer;reports=[]
            def capture(**kwargs):
                value,report=original_export(**kwargs);reports.append(report);return value,report
            htx.export_gpu_hdr_transfer=capture
            try:
                target=directory/f'{colour}_denoise_av1.mkv'
                pipeline.export_gpu_transfer_stream(video=str(source),live=live,frames=3,fps=30,cq=24,
                    with_audio=False,out_path=str(target),job_settings=settings,flow_provider=None,
                    transcode_settings=dict(codec='av1_nvenc',profile='main',pixel_format='p010le',container='.mkv',
                        tune='hq',preset='p4',resolution_mode='original',audio_mode='remove',subtitle_mode='remove',
                        copy_metadata=False,copy_chapters=False,copy_attachments=False))
                facts=json.loads(subprocess.check_output([pipeline._bundle_ffprobe(),'-v','error','-show_streams',
                    '-count_frames','-of','json',str(target)]))
                video=facts['streams'][0]
                assert video['codec_name']=='av1' and video['pix_fmt']=='yuv420p10le'
                assert (video['width'],video['height'],int(video['nb_read_frames']))==(width,height,3)
                assert video['color_transfer']==('smpte2084' if colour=='hdr' else 'bt709')
                assert reports[0]['encoder_host_pixel_bytes_measured']==0
                result['rows'].append(dict(export=colour,report=reports[0],probe=facts))
            finally:
                htx.export_gpu_hdr_transfer=original_export;live.close()
        (directory/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    result['passed']=True
    (directory/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')


if __name__=='__main__':main()
