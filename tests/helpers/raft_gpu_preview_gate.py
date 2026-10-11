"""Manual short CUDA/D3D12 preview gate; requires CUDA PyTorch + torchvision.

Example:
  python tests/helpers/raft_gpu_preview_gate.py --source fixture.mkv \
    --modes single dual triple --directory bench_out/raft-preview-gate

Supply a 3+ frame SDR8, SDR10 or HDR10 fixture. This checks independent motion
histories, actual fusion, source/processed viewport display, and hot controls.
It does not benchmark long-run throughput or validate general RAFT quality.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gpu_preview_engine import GpuPreview
from gpu_transfer_export import probe_source

SCALES = {'single': [.5], 'dual': [.5, .75], 'triple': [.4, .67, .85]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--modes', choices=list(SCALES), nargs='+', default=list(SCALES))
    parser.add_argument('--frames', type=int, choices=range(2, 7), default=3)
    parser.add_argument('--edge', type=int, choices=(512, 720, 1080), default=512)
    parser.add_argument('--iterations', type=int, choices=range(1, 25), default=4)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    metadata = probe_source(str(source))
    preview, rows, shared_model = GpuPreview(), [], None
    try:
        for mode in args.modes:
            scales = SCALES[mode]
            settings = dict(neural_mode='nr', guidance_mode=1, flow_engine='raft',
                flow_quality='Balanced', raft_analysis_edge=args.edge, raft_iterations=args.iterations,
                reconstruction_mode=mode, working_scale=scales[0], layer_scales=scales,
                layer_weights=[1.] * len(scales), residual_strength=1., residual_pyramid=True,
                experimental_hdr_scrgb=metadata.get('color_transfer') == 'smpte2084',
                fusion_mode='catmull_baseline' if mode == 'single' else 'data_consistency_experimental',
                fusion_beta=.1, fusion_relative_tolerance=.001, fusion_max_iterations=128)
            diagnostics, elapsed = [], []
            for index in range(args.frames):
                start = time.perf_counter()
                original, processed, diag = preview.render(str(source), settings, index,
                    need_source=True, display_edge=512)
                elapsed.append((time.perf_counter() - start) * 1000)
                assert original is not None and processed is not None
                ids = diag['layers']['layer_session_ids']
                assert len(set(ids)) == len(ids) == len(scales)
                if mode != 'single':
                    assert diag['solver']['converged'] is True
                    assert diag['fusion_mode_actual'] in (
                        'typed_gpu_data_consistency_solve', 'gpu_data_consistency_solve')
                providers = [preview.provider] + [layer['flow_provider']
                    for layer in getattr(preview.chain, 'extra_layers', [])]
                assert len({provider.previous.data_ptr() for provider in providers}) == len(scales)
                for provider in providers:
                    timing = provider.timing()
                    assert provider.calls == index + 1 and provider.resets == 1
                    assert timing['host_image_bytes_measured'] == timing['host_flow_readback_bytes'] == 0
                    assert timing['direction'] == 'current_to_previous'
                    assert provider.shared is preview.provider.shared
                diag['provider_diagnostics'] = [provider.timing() for provider in providers]
                diagnostics.append(diag)
            model = preview.raft_shared['model']
            assert shared_model is None or shared_model is model
            shared_model = model
            controls = dict(settings, intensity=.65, post_controls={'total_gain': .7},
                            temporal_controls={'scene_detection': True})
            hot = []
            for gain in (.7, .8):
                controls['post_controls'] = {'total_gain': gain}
                start = time.perf_counter()
                _, _, diag = preview.render(str(source), controls, args.frames-1,
                    need_source=True, display_edge=512)
                assert diag['generation'] == diagnostics[-1]['generation'] and diag['reset']
                hot.append({'elapsed_ms': (time.perf_counter()-start)*1000, 'diagnostics': diag})
            counters = preview.executor.submit(preview.chain.counters).result()
            rows.append(dict(mode=mode, settings=settings, frames=diagnostics, elapsed_ms=elapsed,
                             hot_controls=hot, counters=counters, status='PASS'))
            (args.directory / 'report.json').write_text(json.dumps(dict(source=str(source),
                metadata=metadata, rows=rows), indent=2), encoding='utf-8')
            print(f'{mode}: PASS; {len(scales)} independent sessions/histories', flush=True)
    finally:
        preview.shutdown()


if __name__ == '__main__':
    main()
