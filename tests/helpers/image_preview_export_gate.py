"""Actual user's PNG through NR/DLAA preview and lossless image export."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image
from gpu_preview_engine import GpuPreview


def main():
    source = sys.argv[1]
    target = ROOT / 'build/image-export-gate'
    target.mkdir(parents=True, exist_ok=True)
    settings = dict(render_pipeline=['nr','dlaa'], guidance_mode=1, flow_engine='nvof',
        dlss_backend='gpu_flow_d3d12', export_backend='gpu_transfer', working_scale=1,
        reconstruction_mode='single', preset=0, style=0, intensity=1, local_tone=0,
        local_struct=1, skin_struct=0, dlaa_preset=0, residual_strength=1)
    preview = GpuPreview()
    try:
        original, processed, diagnostics = preview.render(source, settings, 0, display_edge=640)
        assert original is not None and processed is not None
        assert diagnostics['render_pipeline'] == ['nr','dlaa']
        assert diagnostics['image_source']['guidance'] == 'zero_motion'
        assert diagnostics['image_source']['host_upload_bytes'] > 0
        assert diagnostics['native_output_effect']['stages'][0]['stage'] == 'dlaa'
        result = preview.export_image(source, settings, str(target/'processed.png'),
            'png',95,True,True)
        with Image.open(source) as a, Image.open(target/'processed.png') as b:
            assert a.size == b.size and b.format == 'PNG'
        assert result['diagnostics']['generation'] == diagnostics['generation']
        (target/'result.json').write_text(json.dumps(dict(preview=diagnostics,export=result,
            passed=True),indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(dict(passed=True,size=[result['width'],result['height']],
            generation=diagnostics['generation'],pipeline=diagnostics['render_pipeline'])))
    finally:
        preview.shutdown()


if __name__ == '__main__':
    main()
