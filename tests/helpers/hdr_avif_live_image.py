"""One user-authorized full-size retry from a saved current-settings snapshot."""
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
os.environ.setdefault('LUNARC_DATA_DIR',str(Path(os.environ['APPDATA'])/'DaVinci_DLSS5_Studio'))
from gpu_preview_engine import GpuPreview
import image_render
import image_hdr_export
import pipeline


def main():
    folder=ROOT/'build/hdr-avif-fix'
    state=json.loads((folder/'live-input.json').read_text(encoding='utf-8'))
    video=state['video']['video'];settings=state['settings']
    assert video['media_type']=='image' and Path(video['path']).is_file()
    # Read fresh state before sharing this adapter; do not interrupt video jobs.
    key=(Path(os.environ['LUNARC_DATA_DIR'])/'.session_token').read_text().strip()
    def api(route,data=None):
        request=urllib.request.Request('http://127.0.0.1:18923/api/'+route,
            data=json.dumps(data).encode() if data is not None else None,
            headers={'X-DLSS-Token':key,'Content-Type':'application/json'})
        return json.load(urllib.request.urlopen(request,timeout=60))
    queue=api('queue')
    assert not queue.get('active_id') and not queue.get('active_job_id'),'another GPU job is running'
    assert not any(j.get('status')=='running' for j in queue.get('jobs',[]))
    assert api('video/current')['video']['path']==video['path'],'source changed after capture'
    api('preview/memory/release',{})  # Keeps both CPU results, frees only preview GPU resources.
    original_passes=image_render.render_passes
    def passes(original,options,nr_runner):
        def effect(pixels,hdr,local,stage):
            print(f'Start {stage}: {pixels.shape[1]}x{pixels.shape[0]}',flush=True)
            result=image_render.effect_pass(pixels,hdr,local,stage)
            print(f'Completed/released {stage}',flush=True)
            return result
        def nr(pixels,local):
            print(f'Start NR: {pixels.shape[1]}x{pixels.shape[0]}',flush=True)
            result=nr_runner(pixels,local);print('Completed/released NR',flush=True);return result
        return original_passes(original,options,nr,effect)
    image_render.render_passes=passes
    engine=GpuPreview();started=time.perf_counter()
    output=folder/(Path(video['path']).stem+'_HDR_fixed.avif')
    assert not output.exists(),'do not overwrite an existing validation output'
    try:
        info=engine.render_image(video['path'],settings,'memory')
        result=engine.image_results.get(info['id'])
        assert result.hdr and result.pixels.shape[:2]==(4320,7680)
        print('Full 8K FP16 result ready; encoding HDR AVIF on CPU',flush=True)
        saved=image_render.save_result(result,str(output),'avif',95,True,False,'hdr10')
        probe=image_hdr_export._run([pipeline._bundle_ffprobe(),'-v','error','-show_streams','-of','json',str(output)])
        assert probe.returncode==0,probe.stderr.decode('utf-8','replace')
        streams=json.loads(probe.stdout)['streams'];color=streams[0]
        assert all(color.get(k)==v for k,v in dict(codec_name='av1',pix_fmt='yuv444p10le',
            width=7680,height=4320,color_primaries='bt2020',color_transfer='smpte2084',color_space='bt2020nc',color_range='pc').items())
        trace=image_hdr_export._run([pipeline._bundle_ffmpeg(),'-v','trace','-i',str(output),'-map','0:v:0',
            '-c','copy','-bsf:v','trace_headers','-f','null','-'],timeout=30)
        headers='\n'.join(line for line in trace.stderr.decode('utf-8','replace').splitlines()
            if any(key in line for key in ('color_primaries','transfer_characteristics','matrix_coefficients','color_range')))
        assert trace.returncode==0
        (folder/'8k-evidence.json').write_text(json.dumps(dict(passed=True,source=video['path'],
            elapsed_seconds=time.perf_counter()-started,saved=saved,stream=color,sequence_header=headers),
            ensure_ascii=False,indent=2,default=str),encoding='utf-8')
        print(json.dumps(dict(passed=True,path=str(output),bytes=saved['bytes'],
            stream={k:color[k] for k in ('width','height','pix_fmt','color_primaries','color_transfer','color_space','color_range')}),ensure_ascii=False),flush=True)
    finally:engine.shutdown()


if __name__=='__main__':main()
