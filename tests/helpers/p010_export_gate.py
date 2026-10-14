"""Six-frame AV1 SDR10/HDR10 export, remux and VFG regression evidence."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from fractions import Fraction

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import p010_codec_gate as gate
import gpu_transfer_export as gtx
import gpu_hdr_transfer_export as htx
import pipeline
import native_effects


def recheck(root):
    """Recheck saved GPU evidence; no model, decoder or encoder is run."""
    data=json.loads((root/'export.json').read_text('utf-8'))
    for label,case in data.items():
        report=case['report']
        assert report['encoder_contract']['codec']=='av1_nvenc'
        assert report['encoder_contract']['requested_profile']==report['encoder_contract']['actual_profile']==0
        assert report['encoder_contract']['actual_bits_per_raw_sample']==10
        assert report['encoder_frames_submitted_measured']==(12 if label=='vfg' else 6)
        for name in ('decode_host_copy_bytes_measured','dlss_host_color_stage_bytes_measured',
                     'dlss_output_readback_bytes_measured','encoder_host_pixel_bytes_measured',
                     'encoder_rgb_conversion_bytes_measured'):
            assert report[name]==0,(label,name,report[name])
        assert report[('hdr_' if label=='hdr' else 'sdr10_')+'diagnostic_image_readback_bytes_measured']==0
        assert report[('hdr_' if label=='hdr' else 'sdr10_')+'scalar_report_readback']['crosses_host']
        facts=case['probe']; video=next(s for s in facts['streams'] if s['codec_type']=='video')
        assert [video[k] for k in ('color_primaries','color_transfer','color_space','color_range')]==(
            ['bt2020','smpte2084','bt2020nc','tv'] if label=='hdr' else ['bt709','bt709','bt709','tv'])
        pts=case['display_pts']; step=1/(50 if label=='vfg' else 25)
        assert all(abs((b-a)-step)<.0011 for a,b in zip(pts,pts[1:])),(label,pts)
        if label=='hdr':
            source=report['hdr_source_metadata']
            # Repair report-only wire-grid labeling using the captured source,
            # and compare it to the actual native writer values and raw OBUs.
            conversion=htx.mastering_wire_report(source,'av1_nvenc')
            report['hdr_mastering_wire_conversion']=conversion
            values=report['p010_metadata_writer']['wire_values']
            assert [f['wire_units'] for f in conversion['fields']]==values[:10]
            raw_blocks=[s for f in case['elementary_probe']['frames'] for s in f.get('side_data_list',[])]
            names=['red_x','red_y','green_x','green_y','blue_x','blue_y',
                   'white_point_x','white_point_y','max_luminance','min_luminance']
            scales=[65536]*8+[256,16384]
            mastering=[s for s in raw_blocks if s['side_data_type']=='Mastering display metadata']
            lights=[s for s in raw_blocks if s['side_data_type']=='Content light level metadata']
            assert mastering and lights
            for block in mastering:
                assert [Fraction(block[name]) for name in names]==[Fraction(v,scale) for v,scale in zip(values,scales)]
            for block in lights:
                assert block['max_content']==source['max_cll'] and block['max_average']==source['max_fall']
            assert report['encoder_in_bitstream_obu_written'] and not report['encoder_in_bitstream_sei_written']
    (root/'export.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('Saved metadata/PTS/host-image evidence recheck passed',flush=True)


def fixtures(root):
    subtitle=root/'source.srt'; chapter=root/'chapters.txt'; attachment=root/'note.txt'
    subtitle.write_text('1\n00:00:00,000 --> 00:00:00,240\nLunarc codec check\n',encoding='utf-8')
    chapter.write_text(';FFMETADATA1\ntitle=Lunarc codec fixture\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=240\ntitle=Check\n',encoding='utf-8')
    attachment.write_text('Lunarc codec metadata check\n',encoding='utf-8')
    files=[]
    for hdr in (False,True):
        out=root/f'source_{"hdr" if hdr else "sdr709_hd"}.mkv';files.append(out)
        if out.exists(): continue
        raw=ROOT/'build/hdr_probe/hdr_valid_src.mkv' if hdr else ROOT/'build/p010-codec-check/sdr709_hd_source.mkv'
        subprocess.run([pipeline._bundle_ffmpeg(),'-v','error','-y','-i',str(raw),
            '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=0.24',
            '-i',str(subtitle),'-i',str(chapter),'-map','0:v:0','-map','1:a:0','-map','2:s:0',
            '-map_metadata','3','-map_chapters','3','-c:v','copy','-c:a','aac','-c:s','srt',
            '-metadata:s:v:0','title=Source picture','-metadata:s:v:0','language=jpn',
            '-metadata:s:a:0','language=jpn','-metadata:s:s:0','language=eng',
            '-attach',str(attachment),'-metadata:s:t:0','mimetype=text/plain',str(out)],check=True)
    return files


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'build/p010-export-check')
    ap.add_argument('--dll',type=Path,default=ROOT/'native/nvenc_bridge.dll')
    ap.add_argument('--cases',nargs='+',choices=['sdr','hdr','vfg'],default=['sdr','hdr','vfg'])
    ap.add_argument('--recheck',action='store_true')
    ap.add_argument('--settings',type=Path,help='Optional JSON of local runtime settings (or a settings snapshot)')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.recheck:
        recheck(args.output);return
    # Replace only this test process's library resolution.
    original=gtx._load
    def load(name):
        if name=='nvenc_bridge':
            lib=C.CDLL(str(args.dll.resolve()));lib._dll_path=str(args.dll.resolve());return lib
        return original(name)
    gtx._load=load
    sources=fixtures(args.output)
    saved_remove=htx._remove_intermediate
    intermediates=[]
    def preserve(file):
        if file and Path(file).exists():
            target=args.output/(current_label+'_intermediate.mkv')
            shutil.copy2(file,target);intermediates.append(target)
        saved_remove(file)
    htx._remove_intermediate=preserve
    # Reuse local trusted runtime locations, not user render parameters.
    snapshot={"dlss_backend":"gpu_flow_d3d12","export_backend":"gpu_transfer",
              "multi_pass":1,"hdr_paper_white_scrgb":8,"residual_strength":1,
              "preset":1,"style":0,"intensity":1,"local_tone":0,
              "local_struct":1,"skin_struct":0,"use_auto_mask":1}
    if args.settings:
        stored=json.loads(args.settings.read_text('utf-8'))
        stored=stored.get('settings',stored)
        for key in ('dlss_dll_path','vsr_runtime_path'):
            if key in stored:snapshot[key]=stored[key]
    results={}
    export=htx.export_gpu_hdr_transfer
    def capture(**kwargs):
        finished,report=export(**kwargs)
        results[current_label]={'report':report}
        return finished,report
    htx.export_gpu_hdr_transfer=capture
    for label in args.cases:
        current_label=label
        hdr=label=='hdr';source=sources[1 if hdr else 0];out=args.output/(label+'_av1.mkv')
        settings=dict(snapshot,render_pipeline=['nr','vfg'] if label=='vfg' else ['nr'],
            experimental_hdr_scrgb=hdr,truehdr_enabled=False,working_scale=1,
            reconstruction_mode='single',layer_scales=[1],layer_settings=[],guidance_mode=0,
            frame_generation='vfg' if label=='vfg' else 'off',frame_multiplier=2,
            framegen_quality=1,flow_engine='nvof',layer_execution='sequential')
        tx=dict(codec='av1_nvenc',profile='main',pixel_format='p010le',container='.mkv',
            preset='p4',tune='hq',rate_mode='cq',aq_mode='off',gop_mode='auto',
            audio_mode='copy',subtitle_mode='copy',copy_metadata=True,copy_chapters=True,
            copy_attachments=True,color_preset='preserve',resolution_mode='original')
        fps=gtx.probe_source(str(source)); live=native_effects.make_live(fps['width'],fps['height'],settings,wide=True)
        before=len(intermediates)
        try:
            finished=pipeline.export_gpu_transfer_stream(video=str(source),live=live,frames=6,
                fps=25,cq=20,with_audio=True,out_path=str(out),transcode_settings=tx,
                job_settings=settings,flow_provider=None)
            assert Path(finished).exists()
        finally: live.close()
        report=results[label]['report']; facts=gate.probe(out)
        entries=facts.get('packets_and_frames',[])
        pictures=[f for f in entries if f['type']=='frame' and f.get('media_type')=='video']
        expected=12 if label=='vfg' else 6
        assert len(pictures)==expected,(label,len(pictures))
        video=next(s for s in facts['streams'] if s['codec_type']=='video')
        assert video['codec_name']=='av1' and video['pix_fmt']=='yuv420p10le'
        assert len(intermediates)>before
        intermediate=gate.probe(intermediates[-1]);source_probe=gate.probe(source)
        def packets(data,kind):
            return [p['data_hash'] for p in data['packets_and_frames'] if p['type']=='packet' and p['codec_type']==kind]
        assert packets(facts,'video')==packets(intermediate,'video')
        for kind in ('audio','subtitle'): assert packets(facts,kind)==packets(source_probe,kind),(label,kind)
        assert facts['chapters']==source_probe['chapters']
        source_attach=[s for s in source_probe['streams'] if s['codec_type']=='attachment']
        final_attach=[s for s in facts['streams'] if s['codec_type']=='attachment']
        assert [(s['extradata_hash'],s['tags']) for s in final_attach]==[(s['extradata_hash'],s['tags']) for s in source_attach]
        assert facts['format']['tags']['title']==source_probe['format']['tags']['title']
        pts=[float(f['best_effort_timestamp_time']) for f in pictures]
        assert pts==sorted(set(pts)) and abs(pts[0])<.002
        raw=args.output/(label+'.obu')
        subprocess.run([pipeline._bundle_ffmpeg(),'-v','error','-y','-i',str(out),'-map','0:v:0','-c:v','copy','-f','obu',str(raw)],check=True)
        raw_probe=json.loads(subprocess.check_output([pipeline._bundle_ffprobe(),'-v','error','-show_frames','-of','json',str(raw)],encoding='utf-8'))
        side=[s for f in raw_probe['frames'] for s in f.get('side_data_list',[])]
        if hdr:
            assert any(s['side_data_type']=='Mastering display metadata' for s in side)
            assert any(s['side_data_type']=='Content light level metadata' for s in side)
            assert report['p010_metadata_writer']['format']=='av1_metadata_obu'
        else: assert not any('Mastering' in s['side_data_type'] or 'Content light' in s['side_data_type'] for s in side)
        results[label].update(probe=facts,intermediate=intermediate,source=source_probe,
                             elementary_probe=raw_probe,display_pts=pts)
        (args.output/'export.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(label+' export/remux passed',flush=True)
    recheck(args.output)


if __name__=='__main__': main()
