"""Short TrueHDR P010 export with UHQ requested; inspect the actual loaded codec."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_hdr_transfer_export as htx
import gpu_transfer_export as gtx
import native_effects
import pipeline
from p010_codec_gate import probe


def check_case(evidence, output, codec, *, uhq_supported=False):
    facts = probe(output)
    video = next(s for s in facts['streams'] if s['codec_type'] == 'video')
    frames = [f for f in facts['packets_and_frames'] if f['type'] == 'frame']
    assert len(frames) == 2
    assert video['codec_name'] == ('av1' if codec == 'av1_nvenc' else 'hevc')
    assert video['pix_fmt'] == 'yuv420p10le'
    assert [video[k] for k in ('color_primaries','color_transfer','color_space')] == ['bt2020','smpte2084','bt2020nc']
    adjustment = evidence['sdr10_encoder_tune_adjustment']
    assert adjustment == ({'requested': 'uhq', 'applied': 'hq',
        'reason': 'loaded encoder does not support UHQ'} if codec == 'av1_nvenc' and not uhq_supported else None)
    assert evidence['truehdr']['encoder_counters']['frames_submitted'] == 2
    assert evidence['truehdr']['encoder_counters']['host_pixel_bytes'] == 0
    evidence['output_probe'] = facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/nvenc-tune-check')
    parser.add_argument('--codecs', nargs='+', choices=['av1_nvenc','hevc_nvenc'],
                        default=['av1_nvenc','hevc_nvenc'])
    parser.add_argument('--require-uhq', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = args.output / 'result.json'
    report = json.loads(record.read_text('utf-8')) if record.exists() else {}
    original = htx.export_gpu_hdr_transfer
    def capture(**kwargs):
        finished, evidence = original(**kwargs)
        report[current_codec] = evidence
        return finished, evidence
    htx.export_gpu_hdr_transfer = capture
    source = gtx.probe_source(str(args.source))
    for current_codec in args.codecs:
        settings = dict(export_backend='gpu_transfer', dlss_backend='gpu_flow_d3d12',
            render_pipeline=['truehdr'], truehdr_enabled=True, vsr_runtime_path=args.runtime,
            experimental_hdr_scrgb=False, working_scale=1, reconstruction_mode='single',
            guidance_mode=0, frame_generation='off')
        tx = dict(codec=current_codec, profile=htx.P010_PROFILES[current_codec],
                  pixel_format='p010le', container='.mkv', preset='p4', tune='uhq',
                  rate_mode='cq', aq_mode='off', b_frames='auto', resolution_mode='original',
                  color_preset='preserve', audio_mode='remove', subtitle_mode='remove',
                  copy_metadata=False, copy_chapters=False, copy_attachments=False)
        live = native_effects.make_live(source['width'], source['height'], settings, wide=True)
        output = args.output / (current_codec + '.mkv')
        try:
            pipeline.export_gpu_transfer_stream(video=str(args.source), live=live, frames=2,
                fps=25, cq=24, with_audio=False, out_path=str(output), transcode_settings=tx,
                job_settings=settings, flow_provider=None)
        finally:
            live.close()
        evidence = report[current_codec]
        record.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        if args.require_uhq:
            assert evidence['sdr10_encoder_tune_adjustment'] is None
        check_case(evidence, output, current_codec, uhq_supported=args.require_uhq)
        record.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        print(current_codec, 'TrueHDR export passed', evidence['sdr10_encoder_tune_adjustment'], flush=True)


if __name__ == '__main__':
    main()
