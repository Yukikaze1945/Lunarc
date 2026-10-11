"""Short real preview/export/remux gate; no benchmark or model substitution."""
import argparse
import cv2
import json
import numpy as np
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pipeline
from dlss_service import DLSSService


def _pixel_metrics(image):
    pixels = np.asarray(image)[..., :3]
    return dict(mean=float(pixels.mean()), std=float(pixels.std()),
                min=int(pixels.min()), max=int(pixels.max()))


def _require_picture(label, image):
    metrics = _pixel_metrics(image)
    if metrics['mean'] < 8 or metrics['std'] < 5 or metrics['max'] < 32:
        raise AssertionError(f'{label} is black or lacks source detail: {metrics}')
    return metrics


def _decode_frames(ffmpeg, path, width, height):
    result = subprocess.run([ffmpeg, '-hide_banner', '-v', 'error', '-nostdin',
        '-i', str(path), '-map', '0:v:0', '-fps_mode', 'passthrough',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'],
        check=True, capture_output=True)
    frame_bytes = width * height * 3
    if len(result.stdout) % frame_bytes:
        raise AssertionError(f'{path} decoded to a partial frame')
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def run(directory, modes, overrides=None):
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / 'source720.mkv'
    if not source.exists():
        subprocess.run([pipeline._bundle_ffmpeg(), '-hide_banner', '-v', 'error', '-nostdin',
            '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=24:duration=0.25',
            '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=0.25',
            '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p',
            '-color_range', 'tv', '-colorspace', 'bt709', '-color_trc', 'bt709',
            '-color_primaries', 'bt709', '-c:a', 'pcm_s16le', '-metadata', 'title=Native effects gate',
            '-metadata:s:a:0', 'language=jpn', str(source)], check=True)
    rows = []
    source_pixels = None
    for selected in modes:
        output = directory / (selected + '.mkv')
        if output.exists():
            raise RuntimeError('Refuse to overwrite gate evidence: ' + str(output))
        service = DLSSService()
        service.load_video(str(source))
        settings = dict(service.settings, neural_mode=selected, export_backend='gpu_transfer',
            guidance_mode=0, working_scale=1, reconstruction_mode='single',
            layer_scales=[1], layer_weights=[1],
            dlss_dll_path=r'F:\dlss\DLSS-310.6-dll\nvngx_dlss.dll',
            vsr_runtime_path=r'F:\Magpie-Experimental-x64', vsr_quality=2, vsr_target='1080p')
        if '-' in selected:
            settings.update(render_pipeline=selected.split('-'), neural_mode='nr',
                            dlss_dll_path='', vsr_runtime_path='')
        if selected.startswith('nr_') or ('-' in selected and 'nr' in settings['render_pipeline']):
            settings.update(working_scale=.5, layer_scales=[.5, .75],
                layer_weights=[1, 1], reconstruction_mode='dual',
                fusion_mode='data_consistency_experimental')
        settings.update(overrides or {})
        service.update_settings(settings)
        row = dict(mode=selected, previews=[], preview_pixels=[], export_pixels=[])
        try:
            if source_pixels is None:
                source_pixels = _decode_frames(pipeline._bundle_ffmpeg(), source, 1280, 720)
                if len(source_pixels) != 6:
                    raise AssertionError('source fixture did not decode to six frames')
            for index in (0, 1, 1):
                blob, status = service.render_preview(index, view='dlss')
                if status != 'ok':
                    raise RuntimeError(status)
                (directory / f'{selected}_preview_{len(row["previews"])}.jpg').write_bytes(blob)
                row['previews'].append(service.preview_diagnostics())
                if '-' in selected:
                    diag = row['previews'][-1]
                    assert diag['render_pipeline'] == selected.split('-')
                    for position in ('native_input_effect','native_output_effect'):
                        if diag.get(position):
                            assert all(s['calls'] > 0 for s in diag[position]['stages'])
                preview = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                if preview is None:
                    raise AssertionError(f'{selected} preview {index} is not a JPEG image')
                row['preview_pixels'].append(_require_picture(
                    f'{selected} preview request {index}', preview))
            service._gpu_preview.shutdown(); service._gpu_preview = None
            tx = dict(codec='hevc_nvenc', preset='p4', tune='hq', container='mkv',
                audio_mode='copy', subtitle_mode='copy', copy_chapters=True,
                copy_metadata=True, copy_attachments=True)
            job = service.add_job(dict(name=selected, input_path=str(source),
                output_path=str(output), codec='hevc_nvenc', container='mkv', cq=24,
                with_audio=True, settings=settings, transcode_settings=tx))
            service._execute_job(job, None)
            row['job'] = job
            probe = json.loads(subprocess.check_output([pipeline._bundle_ffprobe(), '-v', 'error',
                '-show_streams', '-show_frames', '-show_format', '-of', 'json', str(output)], text=True))
            videos = [s for s in probe['streams'] if s['codec_type'] == 'video']
            audios = [s for s in probe['streams'] if s['codec_type'] == 'audio']
            expected = [1920,1080] if 'vsr' in selected else [1280,720]
            assert [videos[0]['width'], videos[0]['height']] == expected
            assert len([f for f in probe['frames'] if f.get('media_type') == 'video']) == 6
            assert len(audios) == 1 and audios[0]['codec_name'] == 'pcm_s16le'
            assert audios[0]['tags']['language'] == 'jpn'
            decoded = _decode_frames(pipeline._bundle_ffmpeg(), output,
                                     videos[0]['width'], videos[0]['height'])
            if len(decoded) != 6:
                raise AssertionError(f'{selected} export did not decode to six full frames')
            for frame_index, pixels in enumerate(decoded):
                row['export_pixels'].append(_require_picture(
                    f'{selected} export frame {frame_index}', pixels))
            aligned = cv2.resize(decoded[0], (1280, 720), interpolation=cv2.INTER_AREA)
            source_gray = cv2.cvtColor(source_pixels[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
            output_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
            correlation = float(np.corrcoef(source_gray.ravel(), output_gray.ravel())[0, 1])
            row['first_frame_source_correlation'] = correlation
            if not np.isfinite(correlation) or correlation < .35:
                raise AssertionError(f'{selected} export first frame lost source structure: {correlation}')
            row['probe'], row['status'] = probe, 'PASS'
        except Exception as exc:
            row['status'], row['error'] = 'FAIL', str(exc)
            raise
        finally:
            rows.append(row)
            (directory / 'report.json').write_text(json.dumps(rows, indent=2, default=str), encoding='utf-8')
            if getattr(service, '_gpu_preview', None):
                service._gpu_preview.shutdown()
            if service._cap:
                service._cap.release()
    print(json.dumps(dict(status='PASS', modes=modes, evidence=str(directory))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    p.add_argument('--modes', nargs='+', default=['dlaa','vsr','nr_dlaa','nr_vsr'])
    args = p.parse_args()
    run(args.directory, args.modes)
