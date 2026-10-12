"""Same synthetic 4K HDR observations, four completed solves per path.

NVML reports sampled whole-device allocation, including other applications.
It is distinct from the solver's exact logical/committed texture allocation.
"""
import argparse
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


class MemoryInfo(C.Structure):
    _fields_=[('total',C.c_uint64),('free',C.c_uint64),('used',C.c_uint64)]


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--fixture',type=Path)
    parser.add_argument('--modes',nargs='+',choices=['legacy','host','auto','primal','observation','observation_jacobi'],
                        default=['legacy','primal','observation','observation_jacobi'])
    args=parser.parse_args()
    out=args.out.resolve()
    assert not out.exists(), 'Fresh evidence required'
    spec=importlib.util.spec_from_file_location('fixture_gate',ROOT/'tests/unit/hdr_multiscale_fusion_a2_gate.py')
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    gate.FULL_W=3840;gate.FULL_H=2160
    gate.LAYERS=(((1920,1080),1.),((1536,864),1.),((960,540),1.))
    fixture=args.fixture.resolve() if args.fixture else out/'fixture'
    if args.fixture:
        assert 'full_width=3840' in (fixture/'manifest.txt').read_text()
        out.mkdir(parents=True)
    else:gate.build_fixture(str(fixture))
    nvml=C.CDLL('nvml.dll')
    assert nvml.nvmlInit_v2()==0
    device=C.c_void_p()
    nvml.nvmlDeviceGetHandleByIndex_v2.argtypes=[C.c_uint,C.POINTER(C.c_void_p)]
    nvml.nvmlDeviceGetMemoryInfo.argtypes=[C.c_void_p,C.POINTER(MemoryInfo)]
    assert nvml.nvmlDeviceGetHandleByIndex_v2(0,C.byref(device))==0
    name=C.create_string_buffer(128)
    nvml.nvmlDeviceGetName.argtypes=[C.c_void_p,C.c_char_p,C.c_uint]
    assert nvml.nvmlDeviceGetName(device,name,len(name))==0
    rows=[]
    try:
        for mode in args.modes:
            directory=out/mode
            environment=dict(os.environ)
            environment['LUNARC_FUSION_SOLVER']='observation' if mode.startswith('observation') else '' if mode=='auto' else 'legacy' if mode=='host' else mode
            environment['LUNARC_FUSION_PRECONDITIONER']='jacobi' if mode.endswith('jacobi') else ''
            dll=args.baseline.resolve() if mode=='legacy' else ROOT/'native/gpu_multiscale_fusion_bridge.dll'
            samples=[];stop=threading.Event()
            def sample():
                while not stop.is_set():
                    info=MemoryInfo()
                    if nvml.nvmlDeviceGetMemoryInfo(device,C.byref(info))==0:samples.append(info.used)
                    stop.wait(.01)
            sampler=threading.Thread(target=sample);sampler.start()
            try:
                subprocess.run([str(ROOT/'build/fusion_optimization/gpu_fusion_profile.exe'),
                    str(fixture),str(directory),str(dll)],env=environment,cwd=ROOT/'native',check=True)
            finally:stop.set();sampler.join()
            summary=gate._parse_summary(str(directory/'summary.txt'))
            warm_ms=[float(summary[f'frame{i}_ms']) for i in range(1,4)]
            assert all(float(summary[f'frame{i}_residual'])<=gate.RELTOL for i in range(4))
            rows.append(dict(mode=mode,median_ms=float(np.median(warm_ms)),summary=summary,
                sampled_device_peak_bytes=max(samples),sample_count=len(samples),dll=str(dll)))
        pitch=int(rows[0]['summary']['output_pitch'])//2
        outputs=[np.fromfile(out/r['mode']/'output.bin',np.float16).reshape(2160,pitch)[:,:3840*4] for r in rows]
        for row,pixels in zip(rows[1:],outputs[1:]):
            assert np.isfinite(pixels).all()
            row['max_composite_difference_from_legacy']=float(np.max(np.abs(pixels.astype(np.float32)-outputs[0].astype(np.float32))))
            assert row['max_composite_difference_from_legacy']<=.002
        (out/'result.json').write_text(json.dumps(dict(passed=True,gpu=name.value.decode(),
            full=[3840,2160],layer_scales=[.5,.4,.25],scope='synthetic fusion only; repeated same observations; completed CPU wall time; device VRAM sampled at 10 ms including other apps',rows=rows),indent=2))
        print(json.dumps([{k:r[k] for k in ['mode','median_ms','sampled_device_peak_bytes']} for r in rows]))
    finally:nvml.nvmlShutdown()


if __name__=='__main__':run()
