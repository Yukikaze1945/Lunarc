"""Direct GPU D/D^T pairing including signed values, highlights and clamp edges."""
import importlib.util
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('fixture_gate',ROOT/'tests/unit/hdr_multiscale_fusion_a2_gate.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from multiscale_fusion import LanczosObservation


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    assert not out.exists(), 'Fresh evidence required'
    results=[]
    for width,height,layers in [(64,48,(((32,24),1.),((16,12),.5))),
                               (65,49,(((32,24),.7),((45,33),1.8)))]:
        directory=out/f'{width}x{height}'
        fixture=directory/'fixture';fixture.mkdir(parents=True)
        module.FULL_W=width;module.FULL_H=height;module.LAYERS=layers
        module.build_fixture(str(fixture))
        dumps=directory/'gpu'
        subprocess.run([str(ROOT/'build/fusion_optimization/gpu_fusion_operator_probe.exe'),
            str(fixture),str(dumps),str(ROOT/'native/gpu_multiscale_fusion_bridge.dll')],check=True,cwd=ROOT/'native')
        summary=module._parse_summary(str(dumps/'summary.txt'))
        assert int(summary['failures'])==0
        assert int(summary['debug_layer_available'])==1
        for index,((low_w,low_h),weight) in enumerate(layers):
            item=LanczosObservation.from_extents(width,height,np.zeros((low_h,low_w,3),np.float32),weight)
            forward=f'layer{index}_forward';adjoint=f'layer{index}_adjoint'
            x=np.fromfile(dumps/(forward+'_input.bin'),np.float32).reshape(height,width,4)[...,:3]
            y=np.fromfile(dumps/(adjoint+'_input.bin'),np.float32).reshape(low_h,low_w,4)[...,:3]
            def output(prefix,w,h):
                pitch=int(summary[prefix+'_pitch'])//4
                return np.fromfile(dumps/(prefix+'_output.bin'),np.float32).reshape(h,pitch)[:,:w*4].reshape(h,w,4)[...,:3]
            dx=output(forward,low_w,low_h);aty=output(adjoint,width,height)
            np.testing.assert_allclose(dx,item.down(x),atol=4e-6,rtol=3e-6)
            np.testing.assert_allclose(aty,item.adjoint(y),atol=4e-6,rtol=3e-6)
            lhs=float(np.sum(dx.astype(np.float64)*y));rhs=float(np.sum(x.astype(np.float64)*aty))
            error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1.)
            assert error <= 3e-6,(lhs,rhs,error)
            results.append(dict(full=[width,height],low=[low_w,low_h],adjoint_relative_error=error))
    (out/'result.json').write_text(json.dumps(dict(passed=True,d3d12_errors=0,rows=results),indent=2))
    print(json.dumps(results))


if __name__=='__main__':run()
