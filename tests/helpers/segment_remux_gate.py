"""Offline recheck the original rendered pictures after a copy-pass fix."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pipeline
import gpu_transfer_export as gtx
from comparison_media import MediaComparisonService

root=ROOT/'build/segment-preview-check'
ffmpeg,ffprobe=pipeline._bundle_ffmpeg(),pipeline._bundle_ffprobe()
def probe(file,*args):
    return json.loads(subprocess.check_output([ffprobe,'-v','error',*args,'-of','json',str(file)],encoding='utf8'))

results=[]
for depth in (8,10):
    # The first successful picture run for each precision; no model is rerun.
    candidates=list((root/'cache').glob('*.mkv'))
    final=None
    for file in candidates:
        stream=next((s for s in probe(file,'-show_streams')['streams'] if s['codec_type']=='video'),{})
        if stream.get('pix_fmt')==('yuv420p10le' if depth==10 else 'yuv420p'):
            final=file;break
    assert final,depth
    frames=probe(final,'-select_streams','v:0','-show_frames')['frames']
    first=min(float(f['best_effort_timestamp_time']) for f in frames)
    encoded=root/f'remux-input{depth}.mkv'
    subprocess.run([ffmpeg,'-v','error','-y','-ss',f'{first:.9f}','-i',str(final),
        '-map','0:v','-c:v','copy','-an','-sn','-avoid_negative_ts','disabled',str(encoded)],check=True,capture_output=True)
    output=root/f'corrected{depth}.mkv'; source=root/f'source{depth}.mkv'
    ts=dict(codec='hevc_nvenc',audio_mode='copy',subtitle_mode='copy',copy_metadata=True,copy_chapters=True,
        copy_attachments=True,_source_range_start_seconds=.2,_source_range_duration_seconds=.2)
    command=gtx.run_copy_pass(ffmpeg,str(encoded),str(source),ts,str(output),True,
        video_authoritative=True,ffprobe=ffprobe)
    info=probe(output,'-show_frames','-show_streams','-show_chapters','-show_format')
    pictures=[f for f in info['frames'] if f['media_type']=='video']
    assert len(pictures)==6
    assert abs(min(float(f['best_effort_timestamp_time']) for f in pictures))<.002
    assert float(info['format']['duration'])<.24
    packets=probe(output,'-show_packets','-show_data_hash','sha256')['packets']
    original=probe(source,'-show_packets','-show_data_hash','sha256')['packets']
    audio=[p for p in packets if p['codec_type']=='audio']
    assert audio and all(float(p['pts_time'])>=-.001 for p in audio)
    source_hashes={p['data_hash'] for p in original if p['codec_type']=='audio'}
    assert all(p['data_hash'] in source_hashes for p in audio)
    service=MediaComparisonService()
    pair=service.open_pair(str(source),str(output),.2,.2)
    timeline=service.get_timeline(pair['session'],0)
    assert abs(timeline['sides']['original']['actual_time'])<.01
    assert abs(timeline['sides']['exported']['actual_pts'])<.002
    service.close_session(pair['session'])
    results.append(dict(depth=depth,original_picture_evidence=str(final),output=str(output),
        video_pts=[f['best_effort_timestamp_time'] for f in pictures],duration=info['format']['duration'],
        audio_pts=[p['pts_time'] for p in audio],audio_payloads_preserved=True,
        chapters=info['chapters'],command=command,timeline=timeline))
    print(f'PASS SDR{depth} offline remux: first video PTS zero; six original rendered pictures; copied selected audio',flush=True)
(root/'remux-proof.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')
