"""Short actual NVDEC/DLSSNR/NVENC range export and comparison alignment gate."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pipeline
from dlss_service import DLSSService
from comparison_media import MediaComparisonService
from comparison_playback import ComparisonPlayback
from preview_segments import PreviewSegments

evidence = ROOT / 'build' / 'segment-preview-check'
evidence.mkdir(parents=True, exist_ok=True)
ffmpeg, ffprobe = pipeline._bundle_ffmpeg(), pipeline._bundle_ffprobe()
meta = evidence / 'chapters.txt'
meta.write_text(';FFMETADATA1\ntitle=range fixture\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=200\ntitle=before\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=200\nEND=600\ntitle=selected\n', encoding='utf-8')
proof=evidence/'proof.json'
results=json.loads(proof.read_text(encoding='utf-8')) if proof.exists() else []
for depth in ([int(sys.argv[1])] if len(sys.argv)>1 else (8,10)):
    source=evidence / f'source{depth}.mkv'
    if depth == 10:
        # Reuse the verified, explicitly tagged BT.709 Main10 fixture; repeating
        # its encoded packets creates a longer CFR source without reinterpretation.
        subprocess.run([ffmpeg,'-v','error','-y','-stream_loop','5','-i',str(ROOT/'build/sequential-check/input10.mkv'),
            '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=0.6','-i',str(meta),
            '-map','0:v','-map','1:a','-map_metadata','2','-map_chapters','2','-c:v','copy','-t','0.6','-c:a','aac',str(source)],
            check=True,capture_output=True)
        from gpu_hdr_transfer_export import sdr10_source
        assert sdr10_source(pipeline.probe_pixel_info(str(source)))
    else:
        subprocess.run([ffmpeg,'-v','error','-y','-f','lavfi','-i','testsrc2=size=320x192:rate=30:duration=0.6',
            '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=0.6','-i',str(meta),
            '-map','0:v','-map','1:a','-map_metadata','2','-map_chapters','2',
            '-c:v','libx265','-x265-params','log-level=error:colorprim=1:transfer=1:colormatrix=1:range=limited','-pix_fmt',f'yuv420p{depth}le' if depth==10 else 'yuv420p',
            '-bsf:v','hevc_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv','-c:a','aac',str(source)],
            check=True,capture_output=True)
    service=DLSSService();service.load_video(str(source))
    settings=dict(service.settings,export_backend='gpu_transfer',guidance_mode=0,working_scale=.5,render_pipeline=['nr'])
    tx=dict(codec='hevc_nvenc',container='.mkv',preset='p4',tune='hq',multipass='qres',rate_mode='cq',cq=24,
        pixel_format='auto',profile='auto',color_preset='preserve',audio_mode='copy',subtitle_mode='copy',
        copy_metadata=True,copy_chapters=True,copy_attachments=True)
    cache=PreviewSegments(service,evidence / 'cache')
    record=cache.create(str(source),settings,tx,dict(start_frame=6,end_frame=12))
    job=next(j for j in service.get_jobs() if j['id']==record['job_id'])
    try:
        service._execute_job(job,None);job['status']='completed'
        rows=cache.list(str(source),settings,tx);assert rows[0]['ready']
        probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_streams','-show_chapters',
            '-show_format','-show_frames','-of','json',record['output_path']],encoding='utf-8'))
        frames=[f for f in probe['frames'] if f['media_type']=='video']
        assert len(frames)==6,len(frames)
        assert max(float(f['best_effort_timestamp_time']) for f in frames)-min(float(f['best_effort_timestamp_time']) for f in frames)<.18
        assert job['total_frames']==6 and job['residual_backend_diagnostics']['frame_range']['emitted_input_frames']==6
        assert probe['chapters'] and all(float(c['end_time'])<=.201 for c in probe['chapters'])
        compare=MediaComparisonService()
        pair=compare.open_pair(str(source),record['output_path'],record['original_offset_seconds'],record['duration_seconds'])
        timeline=compare.get_timeline(pair['session'],0)
        assert abs(timeline['sides']['original']['actual_time'])<.02,timeline
        assert abs(timeline['sides']['original']['actual_pts']-.2)<.04,timeline
        player=ComparisonPlayback(compare,pair['session'],0,1)
        try:
            packet=player.packet(.05);assert len(packet)>1000
        finally:player.close();compare.close_session(pair['session'])
        row=dict(depth=depth,record=record,video_pts=[f['best_effort_timestamp_time'] for f in frames],
            chapters=probe['chapters'],streams=probe['streams'],timeline=timeline,diagnostics=job['residual_backend_diagnostics'])
        results=[item for item in results if item["depth"]!=depth]+[row]
        proof.write_text(json.dumps(results,indent=2,default=str,ensure_ascii=False),encoding="utf-8")
        print(f'PASS SDR{depth}: 6 selected frames; cache valid; aligned continuous comparison',flush=True)
    finally:
        if getattr(service,"_gpu_preview_engine",None):service._gpu_preview_engine.shutdown()
        if service._cap:service._cap.release()
(evidence/'proof.json').write_text(json.dumps(results,indent=2,default=str,ensure_ascii=False),encoding='utf-8')
