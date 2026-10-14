"""Real VSR mode/strength boundary on synthetic CUDA images; no DLSS or encoder."""
import ctypes as C
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import bundle_paths
from rtx_vsr_engine import RtxVsr
from vsr_config import LEVELS
from preview_cache import _Copy2D


class Memory(C.Structure):
    _fields_=[(key,C.c_uint64) for key in ('luid','local_usage','local_budget','nonlocal_usage','nonlocal_budget')]


def main():
    directory=ROOT/'build/vsr-modes';directory.mkdir(parents=True,exist_ok=True)
    bundle_paths.add_dll_directory(bundle_paths.native_dir())
    cuda=C.WinDLL('nvcuda.dll')
    for name,args in {
        'cuDeviceGet':[C.POINTER(C.c_int),C.c_int],
        'cuCtxCreate_v2':[C.POINTER(C.c_void_p),C.c_uint,C.c_int],
        'cuCtxDestroy_v2':[C.c_void_p],
        'cuMemAlloc_v2':[C.POINTER(C.c_uint64),C.c_size_t],
        'cuMemFree_v2':[C.c_uint64],
        'cuMemcpy2D_v2':[C.POINTER(_Copy2D)],
        'cuCtxSynchronize':[],
    }.items():getattr(cuda,name).argtypes=args
    device,context=C.c_int(),C.c_void_p()
    assert cuda.cuInit(0)==cuda.cuDeviceGet(C.byref(device),0)==0
    assert cuda.cuCtxCreate_v2(C.byref(context),0,device)==0
    memory=C.CDLL(str(ROOT/'native/build/memory/gpu_process_memory_probe.dll'))
    memory.gpu_process_memory_query.argtypes=[C.c_void_p,C.POINTER(Memory)]
    def sample():
        value=Memory();assert memory.gpu_process_memory_query(None,C.byref(value))==0
        return {key:getattr(value,key) for key,_ in value._fields_}
    result=dict(rows=[],timing_scope='completed CUDA VSR wall time, second call excludes model loading',
                allocation_scope='DXGI LOCAL process samples, not adapter-wide',performance='NOT_CHARACTERIZED')
    try:
        for packed10 in (False,True):
            w,h=640,360
            yy,xx=np.mgrid[:h,:w]
            value=((xx*13+yy*7)%256).astype(np.uint32)
            image=((value*4)|((value*3)<<10)|((value*2)<<20)|np.uint32(3<<30)).astype(np.uint32) if packed10 else (
                np.stack((value,(value*3)%256,(value*7)%256,np.full_like(value,255)),axis=-1).astype(np.uint8))
            src=C.c_uint64();assert cuda.cuMemAlloc_v2(C.byref(src),image.nbytes)==0
            cp=_Copy2D(srcMemoryType=1,srcHost=image.ctypes.data,srcPitch=w*4,
                dstMemoryType=2,dstDevice=src.value,dstPitch=w*4,WidthInBytes=w*4,Height=h)
            assert cuda.cuMemcpy2D_v2(C.byref(cp))==0
            try:
                for level in sorted(LEVELS):
                    ow,oh=(w,h) if 8<=level<=15 else (w*2,h*2)
                    dst=C.c_uint64();assert cuda.cuMemAlloc_v2(C.byref(dst),ow*oh*4)==0
                    effect=None
                    try:
                        strength=None if level==0 else .6
                        before=sample()
                        effect=RtxVsr(ROOT/'native/rtx_vsr_bridge.dll',r'F:\DLSSC\VideoFX',context.value,
                            (w,h),(ow,oh),level,packed10=packed10,strength=strength)
                        times=[]
                        for index in range(2):
                            mark=time.perf_counter();effect.process(src.value,w*4,dst.value,ow*4)
                            assert cuda.cuCtxSynchronize()==0
                            times.append((time.perf_counter()-mark)*1000)
                        counters=effect.counters()
                        assert counters['successful_runs']==2 and counters['quality_level']==level
                        assert counters['quality_effective'] in (None,level)
                        if strength is not None and counters['strength_effective'] is not None:
                            assert abs(counters['strength_effective']-strength)<1e-5
                        # Explicit diagnostic readback to reject all-zero/invalid output.
                        image_out=np.empty((oh,ow),np.uint32)
                        cp=_Copy2D(srcMemoryType=2,srcDevice=dst.value,srcPitch=ow*4,
                            dstMemoryType=1,dstHost=image_out.ctypes.data,dstPitch=ow*4,WidthInBytes=ow*4,Height=oh)
                        assert cuda.cuMemcpy2D_v2(C.byref(cp))==0 and np.any(image_out & 0x3fffffff)
                        row=dict(level=level,packed10=packed10,counters=counters,times_ms=times,
                            before=before,loaded=sample(),test_image_readback_bytes=image_out.nbytes)
                        result['rows'].append(row)
                        print('mode',level,'RGB10A2' if packed10 else 'RGBA8',times[-1],'ms',flush=True)
                    finally:
                        if effect:effect.close()
                        cuda.cuMemFree_v2(dst)
                    (directory/'native-modes.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            finally:cuda.cuMemFree_v2(src)
        result['passed']=True
    finally:
        (directory/'native-modes.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        cuda.cuCtxDestroy_v2(context)


if __name__=='__main__':main()
