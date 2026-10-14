"""Short real-GPU NR removal / standalone DLAA guidance regression.

Pass a local JSON containing video/settings and an evidence output path.
Uses external runtimes and production bridges. No encoding or remux.
"""
import json
from copy import deepcopy
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpu_preview_engine import GpuPreview


def main():
    snapshot, output = map(Path, sys.argv[1:3])
    args = json.loads(snapshot.read_text(encoding='utf-8'))
    rows = []
    preview = GpuPreview()
    settings = dict(args['settings'], truehdr_enabled=False, working_scale=1,
                    layer_scales=[1], reconstruction_mode='single',
                    guidance_mode=1, flow_engine='nvof', flow_quality='1F',
                    frame_generation='off', dlaa_preset=12)
    cases = [(['nr', 'dlaa'], 0, 12), (['dlaa'], 0, 12),
             (['dlaa'], 1, 12), (['dlaa'], 1, 13), (['dlaa'], 5, 13)]
    try:
        previous_evaluates = 0
        for order, index, preset in cases:
            _, _, diag = preview.render(args['video'],
                dict(settings, render_pipeline=order, dlaa_preset=preset),
                index, display_edge=640)
            counts = diag.get('dlaa_process_global_counters')
            stage = diag['native_output_effect']['stages'][0]
            counts = counts or stage['counters']
            evaluates = counts['successful_evaluates']
            assert evaluates - previous_evaluates == 1, (order, counts)
            previous_evaluates = evaluates
            rows.append(deepcopy(dict(order=order, frame=index, preset=preset,
                             generation=diag['generation'], reset=diag['reset'],
                             format=diag['processing_format'], counters=counts,
                             stage=stage, processing_ms=diag['processing_ms'])))
            output.write_text(json.dumps(rows, indent=2), encoding='utf-8')
            print(json.dumps(rows[-1]), flush=True)
    except Exception:
        rows.append(dict(error=traceback.format_exc()))
        raise
    finally:
        try:
            preview.shutdown()
            rows.append(dict(cleanup='closed'))
        except Exception:
            rows.append(dict(cleanup_error=traceback.format_exc()))
            raise
        finally:
            output.write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
