"""GPU filter reference checks, including signed HDR and >24 downsample taps."""
import ctypes as C
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from export_resize import Resize
from native_effects import _library


def matrix(src, dst, algorithm):
    result = np.zeros((dst,src),np.float64)
    for q in range(dst):
        centre = (q+.5)*src/dst-.5
        if algorithm == 'nearest':
            result[q,min(src-1,int((q+.5)*src/dst))] = 1
            continue
        scale = max(1,src/dst)
        radius = (1 if algorithm == 'bilinear' else 2)*scale
        for k in range(int(np.ceil(centre-radius)),int(np.floor(centre+radius))+1):
            x = abs((k-centre)/scale)
            if algorithm == 'bilinear': a = max(1-x,0)
            elif algorithm == 'bicubic': a = 1.5*x**3-2.5*x*x+1 if x<1 else -.5*x**3+2.5*x*x-4*x+2 if x<2 else 0
            else: a = np.sinc(x)*np.sinc(x/2) if x<2 else 0
            result[q,min(src-1,max(0,k))] += a
        result[q] /= result[q].sum()
    return result


def run():
    torch.cuda.init()
    owner = torch.empty(1,device='cuda')
    cuda = C.WinDLL('nvcuda.dll')
    cuda.cuCtxGetCurrent.argtypes = [C.POINTER(C.c_void_p)]
    context = C.c_void_p(); assert cuda.cuCtxGetCurrent(C.byref(context)) == 0
    lib = _library(); device = C.c_void_p()
    assert lib.gpu_effect_device_open(C.byref(device)) == 0
    rows=[]
    try:
        for wide in (False,True):
            for algorithm in ('nearest','bilinear','bicubic','lanczos'):
                for size,target in (((66,50),(100,72)),((66,50),(32,24)),((130,50),(8,6))):
                    rng=np.random.default_rng(7)
                    source = rng.uniform(-2,8,(size[1],size[0],4)).astype(np.float16) if wide else rng.integers(0,256,(size[1],size[0],4),dtype=np.uint8)
                    if wide:
                        source[...,0] = -.5; source[...,1] = 4
                    image=torch.from_numpy(source).cuda()
                    resize=Resize(device.value,context.value,size,target,algorithm,wide=wide)
                    try:
                        for _ in range(2):
                            out=resize.color(dict(format='rgba16f_cuda' if wide else 'rgba8_cuda',width=size[0],height=size[1],device_ptr=image.data_ptr(),pitch_bytes=size[0]*(8 if wide else 4),cuda_context=context.value))
                            raw=np.empty((target[1],out['pitch_bytes']),np.uint8)
                            cuda.cuMemcpyDtoH_v2.argtypes=[C.c_void_p,C.c_uint64,C.c_size_t]
                            assert cuda.cuMemcpyDtoH_v2(raw.ctypes.data,out['device_ptr'],raw.nbytes)==0
                            actual=raw[:,:target[0]*(8 if wide else 4)].copy().view(np.float16 if wide else np.uint8).reshape(target[1],target[0],4)
                        x=source.astype(np.float64)
                        reference=np.einsum('yh,hwc,xw->yxc',matrix(size[1],target[1],algorithm),x,matrix(size[0],target[0],algorithm))
                        if wide:
                            np.testing.assert_allclose(actual,reference,atol=.004,rtol=.002)
                            assert actual.min()<0 and actual.max()>1
                        else:
                            reference=np.rint(np.clip(reference,0,255)).astype(np.uint8)
                            assert np.abs(actual.astype(int)-reference.astype(int)).max()<=1
                        rows.append(dict(wide=wide,algorithm=algorithm,source=size,target=target,report=resize.report()))
                    finally:resize.close()
    finally:lib.gpu_effect_device_close(device)
    out=ROOT/'bench_out'/'export_resize_gpu';out.mkdir(parents=True,exist_ok=True)
    (out/'result.json').write_text(json.dumps(dict(status='PASS',cases=rows),indent=2))
    print('PASS: 24 GPU/reference cases, repeat calls, signed HDR and long filters')


if __name__=='__main__':run()
