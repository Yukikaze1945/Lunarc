"""Short real GPU export gate. No benchmark, no production image readback."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if '--runtime' in sys.argv:
    sys.path.insert(0, str(Path(sys.argv[sys.argv.index('--runtime') + 1]).resolve()))
import dlss_gpu_engine as dge
import pipeline
import optical_flow
import native_effects


def probe(path, *args):
    return json.loads(subprocess.check_output([pipeline._bundle_ffprobe(), '-v', 'error',
        *args, '-of', 'json', str(path)], text=True, encoding='utf-8'))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--mode', choices=['single', 'dual', 'triple', 'hdr', 'sdr8'], default='single')
    parser.add_argument('--frames', type=int, default=6)
    parser.add_argument('--flow', action='store_true')
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--recheck', action='store_true', help='Verify existing output without GPU work')
    parser.add_argument('--stages', nargs='+', choices=['nr', 'dlaa', 'vsr', 'vfg'])
    parser.add_argument('--vsr-target', choices=['720p','1080p','1440p','4k','8k'], default='1080p')
    parser.add_argument('--vsr-size', type=int, nargs=2)
    parser.add_argument('--frame-multiplier', type=int, choices=range(1,9), default=1)
    parser.add_argument('--framegen-fps', help='Dynamic VFG target fps, decimal or rational')
    parser.add_argument('--no-framegen-scene-detection', action='store_true')
    parser.add_argument('--vsr-scale', type=float)
    parser.add_argument('--dlaa-preset', type=int, default=11)
    parser.add_argument('--output-size', type=int, nargs=2)
    parser.add_argument('--scale-algorithm', choices=['nearest','bilinear','bicubic','lanczos'], default='lanczos')
    a = parser.parse_args()
    a.directory.mkdir(parents=True, exist_ok=True)
    out = a.directory / 'output.mkv'
    if out.exists() and not a.recheck:
        raise RuntimeError('Refuse to overwrite gate evidence')
    n, fps, w, h = pipeline.video_info(str(a.source))
    scales = [0.4, 0.67, 0.85] if a.mode == 'triple' else [0.5, 0.75] if a.mode == 'dual' else [0.5]
    settings = dict(preset=1, style=0, intensity=1, local_tone=0, local_struct=1,
        skin_struct=0, use_auto_mask=1, ui_correction=0, guidance_mode=int(a.flow),
        flow_engine='nvof', flow_quality='4M', working_scale=scales[0],
        residual_pyramid=True, residual_strength=1, multi_pass=1,
        dlss_backend='gpu_flow_d3d12', export_backend='gpu_transfer',
        reconstruction_mode=a.mode if a.mode in ('dual', 'triple') else 'single',
        layer_scales=scales, layer_weights=[1]*len(scales),
        fusion_mode='data_consistency_experimental', fusion_beta=.1,
        fusion_max_iterations=128, fusion_relative_tolerance=.001,
        experimental_hdr_scrgb=a.mode == 'hdr', hdr_paper_white_scrgb=8)
    if a.frame_multiplier > 1:
        settings.update(frame_generation='vfg', frame_multiplier=a.frame_multiplier, framegen_quality=1)
    if a.framegen_fps:
        settings.update(frame_generation='vfg_dynamic', framegen_target_fps=a.framegen_fps)
    settings['framegen_scene_detection'] = not a.no_framegen_scene_detection
    if a.vsr_scale:
        settings.update(vsr_output_mode='scale',vsr_scale=a.vsr_scale)
    if a.stages:
        settings.update(render_pipeline=a.stages, dlaa_preset=a.dlaa_preset, vsr_target=a.vsr_target)
        if a.vsr_size:
            settings.update(vsr_output_mode='resolution',vsr_target='custom',vsr_width=a.vsr_size[0],vsr_height=a.vsr_size[1])
        if 'nr' not in a.stages:
            scales = [1.0]
            settings.update(working_scale=1., reconstruction_mode='single', layer_scales=scales)
    tx = dict(codec='hevc_nvenc', preset='p7', tune='hq', multipass='qres',
        pixel_format='auto', profile='auto', color_preset='preserve',
        container='mkv', rate_mode='cq', cq=24, aq_mode='spatial',
        gop_mode='auto', decode_mode='cuda', audio_mode='copy',
        subtitle_mode='copy', copy_attachments=True, copy_metadata=True, copy_chapters=True)
    working = pipeline.resolve_working_dims(*native_effects.processing_size(settings, (w,h)), scales[0])
    if a.output_size:
        tx.update(resolution_mode='custom_size', custom_width=a.output_size[0], custom_height=a.output_size[1],
            keep_aspect=False, scale_algorithm=a.scale_algorithm)
    live = None if a.recheck else native_effects.make_live(*working, settings, wide=a.mode != 'sdr8')
    service = optical_flow.get_frame_guidance_service('nvof', '4M') if a.flow and not a.recheck else None
    provider = pipeline.GuidanceServiceFlowProvider(service, guidance_mode=1) if service else None
    diagnostics = json.loads((a.directory / 'diagnostics.json').read_text(encoding='utf-8')) if a.recheck else {}
    try:
        if not a.recheck:
            pipeline.export_gpu_transfer_stream(video=str(a.source), live=live,
            frames=a.frames, fps=fps, cq=24, with_audio=True, out_path=str(out),
            transcode_settings=tx, job_settings=settings, flow_provider=provider,
            diagnostics_out=diagnostics)
    finally:
        if live is not None:
            live.close()
            (a.directory / 'diagnostics.json').write_text(json.dumps(diagnostics, indent=2,
                ensure_ascii=False, default=str), encoding='utf-8')
    facts = probe(out, '-show_streams', '-show_frames', '-show_packets', '-show_chapters', '-show_format')
    video = next(s for s in facts['streams'] if s['codec_type'] == 'video')
    video_frames = [f for f in facts.get('frames', []) if f.get('media_type') == 'video']
    video_packets = [p for p in facts.get('packets', []) if p.get('codec_type') == 'video']
    # Combined ffprobe output may put both types in packets_and_frames.
    mixed = facts.get('packets_and_frames', [])
    video_frames += [f for f in mixed if f.get('type') == 'frame' and f.get('media_type') == 'video']
    video_packets += [p for p in mixed if p.get('type') == 'packet' and p.get('codec_type') == 'video']
    source_video = next(s for s in probe(a.source,'-show_streams')['streams'] if s['codec_type']=='video')
    source_rate = Fraction(source_video.get('r_frame_rate') or str(fps))
    rate = Fraction(a.framegen_fps) if a.framegen_fps else source_rate*a.frame_multiplier
    extent = Fraction(a.frames,1)*rate/source_rate
    count = -(-extent.numerator//extent.denominator)
    assert len(video_frames) == len(video_packets) == count, (len(video_frames), len(video_packets))
    if a.frame_multiplier > 1 or a.framegen_fps:
        fg = diagnostics['frame_generation']
        assert fg['source_frames'] == a.frames and fg['output_frames'] == count
        if not a.framegen_fps:
            assert fg['generated_frames'] == (a.frames-1)*(a.frame_multiplier-1)
            assert fg['tail_hold_frames'] == a.frame_multiplier-1
        else:
            assert fg['mode'] == 'dynamic' and fg['generated_frames'] > 0
            assert all(0 < t < 1 for t in fg['recent_timesteps'])
        assert fg['automatic_shot_change_detection'] == (not a.no_framegen_scene_detection)
        assert fg['model']['successful_runs'] == fg['generated_frames']
        assert fg['host_image_bytes'] == 0
        assert abs(float(Fraction(video['avg_frame_rate'])) - float(rate)) < .001
    assert (video['width'], video['height']) == (tuple(a.output_size) if a.output_size else native_effects.output_size(settings, (w,h)))
    if a.output_size and tuple(a.output_size) != native_effects.output_size(settings, (w,h)):
        assert diagnostics['export_resize']['calls'] == count
        assert diagnostics['export_resize']['algorithm'] == a.scale_algorithm
        assert diagnostics['export_resize']['host_image_bytes'] == 0
    pts = [int(f['best_effort_timestamp']) for f in video_frames]
    assert pts == sorted(set(pts)), pts
    if a.mode != 'sdr8':
        assert video['pix_fmt'] == 'yuv420p10le' and video['profile'] == 'Main 10', video
        colour = ('bt2020nc', 'smpte2084', 'bt2020') if a.mode == 'hdr' else ('bt709',)*3
        assert tuple(video[k] for k in ('color_space', 'color_transfer', 'color_primaries')) == colour
        if a.mode != 'hdr':
            assert not any(side.get('side_data_type') in ('Mastering display metadata', 'Content light level metadata')
                for frame in video_frames for side in frame.get('side_data_list', []))
            assert diagnostics['experimental_hdr_scrgb'] is False
            assert diagnostics['transfer_mode_actual'] == 'gpu_sdr10_transfer'
            assert diagnostics['sdr10_frames_processed_measured'] == a.frames
            sys.path.insert(0, str(ROOT / 'tests' / 'unit'))
            from hdr_production_export_gate import _sei_payloads
            _, payloads, _ = _sei_payloads(str(out), str(a.directory / 'video.hevc'))
            assert payloads == [], payloads
        else:
            sys.path.insert(0, str(ROOT / 'tests' / 'unit'))
            from hdr_production_export_gate import _sei_payloads
            # Inspect only the source extent exercised by this short gate.
            # Parsing an entire multi-minute 4K source adds no evidence for
            # the four/six output frames and makes offline checks very slow.
            source_sample = a.directory / 'source_sample.hevc'
            subprocess.check_call([pipeline._bundle_ffmpeg(), '-v','error','-y',
                '-i',str(a.source),'-map','0:v:0','-frames:v',str(a.frames),
                '-c:v','copy','-f','hevc',str(source_sample)])
            _, source_sei, _ = _sei_payloads(str(source_sample), str(a.directory / 'source.hevc'))
            _, final_sei, _ = _sei_payloads(str(out), str(a.directory / 'video.hevc'))
            expected = {(s['payload_type'], s['bytes']) for s in source_sei}
            actual = {(s['payload_type'], s['bytes']) for s in final_sei}
            # Some MKV sources declare HDR metadata only in track side data.
            # Build the expected HEVC payload from that source declaration,
            # independently of the production conversion/report code.
            source_side = {s['side_data_type']: s for s in source_video.get('side_data_list', [])}
            if not any(t == 137 for t, _ in expected) and 'Mastering display metadata' in source_side:
                md = source_side['Mastering display metadata']
                values = []
                for name in ('green_x','green_y','blue_x','blue_y','red_x','red_y',
                             'white_point_x','white_point_y','max_luminance','min_luminance'):
                    scale = 10000 if 'luminance' in name else 50000
                    rational = Fraction(md[name]) * scale
                    whole, remainder = divmod(rational.numerator, rational.denominator)
                    values.append(whole + int(2 * remainder >= rational.denominator))
                payload = b''.join(v.to_bytes(2,'big') for v in values[:8]) + \
                    b''.join(v.to_bytes(4,'big') for v in values[8:])
                expected.add((137, payload.hex()))
            if not any(t == 144 for t, _ in expected) and 'Content light level metadata' in source_side:
                cll = source_side['Content light level metadata']
                payload = int(cll['max_content']).to_bytes(2,'big') + int(cll['max_average']).to_bytes(2,'big')
                expected.add((144, payload.hex()))
            assert {t for t, _ in expected} == {137, 144}
            assert actual == expected, (actual, expected)
            assert any(side.get('side_data_type') == 'Mastering display metadata'
                for frame in video_frames for side in frame.get('side_data_list', []))
    if a.mode in ('dual', 'triple'):
        ids = diagnostics['sdr10_layer_session_ids']
        assert len(set(ids)) == len(scales), ids
        for record in diagnostics['sdr10_frame_reports']:
            assert record['fusion_mode_actual'] == 'typed_gpu_data_consistency_solve'
            assert record['solver']['converged'], record['solver']
    if a.mode != 'sdr8':
        for key in ('decode_host_copy_bytes_measured', 'dlss_host_color_stage_bytes_measured',
                    'dlss_output_readback_bytes_measured', 'encoder_host_pixel_bytes_measured',
                    ('hdr' if a.mode == 'hdr' else 'sdr10') + '_diagnostic_image_readback_bytes_measured'):
            assert diagnostics[key] == 0, (key, diagnostics[key])
    if a.stages:
        expected_order = a.stages + (['vfg'] if (a.frame_multiplier > 1 or a.framegen_fps) and 'vfg' not in a.stages else [])
        assert diagnostics['render_pipeline'] == expected_order
        effects = [diagnostics.get(key) for key in ('native_input_effect', 'native_output_effect')]
        fg_pipeline = (diagnostics.get('frame_generation') or {}).get('pipeline') or {}
        effects += [fg_pipeline.get('prefix'), fg_pipeline.get('suffix')]
        if a.mode != 'sdr8' and 'vfg' in expected_order:
            prefix = 'hdr' if a.mode == 'hdr' else 'sdr10'
            downstream_frames = count if 'nr' not in expected_order or expected_order.index('vfg') < expected_order.index('nr') else a.frames
            if 'nr' in expected_order:
                calls = '_dlss_flow_calls_measured' if a.flow else '_dlss_zero_motion_calls_measured'
                assert diagnostics[prefix + calls] == downstream_frames * len(scales)
            else:
                assert diagnostics[prefix + '_dlss_zero_motion_calls_measured'] == 0
        for effect in effects:
            for stage in (effect or {}).get('stages', []):
                fg_at = expected_order.index('vfg') if 'vfg' in expected_order else len(expected_order)
                expected_calls = count if expected_order.index(stage['stage']) > fg_at else a.frames
                assert stage['calls'] == expected_calls and stage['host_image_bytes'] == 0, stage
                assert stage['processing_format'] == ('rgba8' if a.mode == 'sdr8' else 'rgba16f'), stage
    source_audio = probe(a.source, '-select_streams', 'a', '-show_packets', '-show_data_hash', 'sha256')['packets']
    final_audio = probe(out, '-select_streams', 'a', '-show_packets', '-show_data_hash', 'sha256')['packets']
    # A short export trims at the requested video extent. Copied audio remains
    # an exact packet prefix; a source fixture can have audio longer than video.
    assert len(final_audio) <= len(source_audio)
    assert [(p['size'], p['data_hash']) for p in source_audio[:len(final_audio)]] == [(p['size'], p['data_hash']) for p in final_audio]
    assert not source_audio or final_audio
    result = dict(status='PASS', source=str(a.source), output=str(out), mode=a.mode,
        frames=a.frames, source_frames=n, dimensions=[w,h], source_fps=fps,
        video=video, audio_packets_copied=len(final_audio), display_pts=pts,
        streams=[(s['codec_type'],s.get('codec_name')) for s in facts['streams']],
        performance='NOT_CHARACTERIZED')
    (a.directory / 'ffprobe.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')
    (a.directory / 'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    run()
