"""Three distinct sequential GPU preview frames; no FPS/quality extrapolation."""
import argparse
import json
from pathlib import Path
import sys
import time
import ctypes as C
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--mode', choices=['dual', 'triple', 'single'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fusion-dll', type=Path)
    args = parser.parse_args()
    assert not args.out.exists(), 'Fresh output required'
    if args.fusion_dll:
        assert args.fusion_dll.is_file()
        def redirect(original):
            def load(name, *values, **keywords):
                if Path(str(name)).name.lower() == 'gpu_multiscale_fusion_bridge.dll':
                    name = str(args.fusion_dll.resolve())
                return original(name, *values, **keywords)
            return load
        C.WinDLL = redirect(C.WinDLL)
        C.CDLL = redirect(C.CDLL)
    import dlss_service
    service = dlss_service.DLSSService()
    scales = {'single': [.5], 'dual': [.5, .75], 'triple': [.4, .67, .85]}[args.mode]
    rows = []
    try:
        service.load_video(str(args.source.resolve()))
        settings = dict(service.settings)
        settings.update(guidance_mode=0, flow_engine='nvof', neural_mode='nr',
                        render_pipeline=['nr'], frame_generation='off',
                        dlss_backend='gpu_flow_d3d12', export_backend='gpu_transfer',
                        reconstruction_mode=args.mode, working_scale=scales[0],
                        layer_scales=scales, layer_weights=[1.]*len(scales),
                        fusion_mode='data_consistency_experimental', fusion_beta=.1,
                        fusion_relative_tolerance=.001, fusion_max_iterations=128,
                        residual_pyramid=True, residual_strength=1.,
                        experimental_hdr_scrgb=False)
        service.update_settings(settings)
        assert int(service.video_info['frames']) >= 3
        for frame in range(3):
            start = time.perf_counter()
            blob, status = service.render_preview(frame, view='dlss', preview_mode=True)
            assert blob and status == 'ok', status
            diag = service.preview_diagnostics()
            assert diag.get('frame') == frame and not diag.get('cached_display')
            layers = diag.get('layers') or {}
            if args.fusion_dll:
                loaded = service._gpu_preview.chain.b['fusion']._name
                assert Path(loaded).resolve() == args.fusion_dll.resolve(), loaded
            ids = layers.get('layer_session_ids') or []
            assert len(ids) == len(scales) and len(set(ids)) == len(ids)
            if len(scales) > 1:
                assert diag['solver']['converged']
            rows.append({'frame': frame, 'wall_ms': (time.perf_counter()-start)*1000,
                         'diagnostics': diag})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'passed': True, 'mode': args.mode,
            'fusion_dll': str(args.fusion_dll or ROOT/'native/gpu_multiscale_fusion_bridge.dll'),
            'fusion_sha256': hashlib.sha256((args.fusion_dll or ROOT/'native/gpu_multiscale_fusion_bridge.dll').read_bytes()).hexdigest(),
            'scope': 'three sequential previews on one GPU; first frame includes setup',
            'rows': rows}, indent=2), encoding='utf-8')
        print(json.dumps([{'frame': r['frame'], 'stages': r['diagnostics']['stages'],
                          'solver': r['diagnostics'].get('solver')} for r in rows]))
    finally:
        service._close_gpu_preview()


if __name__ == '__main__':
    run()
