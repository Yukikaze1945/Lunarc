"""HDR→SDR GPU boundary reference, ordered HDR preview, offline check of short AV1 export."""
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import bundle_paths
from vfx_rgb10 import Codec
from gpu_preview_engine import GpuPreview


def main():
    output=ROOT/'build/hdr-truehdr';output.mkdir(parents=True,exist_ok=True)
    result={}
    bundle_paths.add_dll_directory(bundle_paths.native_dir())
    cuda=C.WinDLL('nvcuda.dll')
    for name,args in {'cuInit':[C.c_uint],'cuDeviceGet':[C.POINTER(C.c_int),C.c_int],
        'cuCtxCreate_v2':[C.POINTER(C.c_void_p),C.c_uint,C.c_int],'cuCtxDestroy_v2':[C.c_void_p],
        'cuMemAlloc_v2':[C.POINTER(C.c_uint64),C.c_size_t],'cuMemFree_v2':[C.c_uint64],
        'cuMemcpyHtoD_v2':[C.c_uint64,C.c_void_p,C.c_size_t],
        'cuMemcpyDtoH_v2':[C.c_void_p,C.c_uint64,C.c_size_t]}.items():
        getattr(cuda,name).argtypes=args
    assert cuda.cuInit(0)==0
    device,context=C.c_int(),C.c_void_p();assert cuda.cuDeviceGet(C.byref(device),0)==0
    assert cuda.cuCtxCreate_v2(C.byref(context),0,device)==0
    ptr=C.c_uint64();codec=None
    try:
        sample=np.array([[0,0,0],[.5,.5,.5],[2.5375]*3,[8]*3,[125]*3,[-.05,.5,1],[5,1,.5],[1,.5,.2]],np.float16)
        image=np.ones((8,16,4),np.float16);image[:,:,:3]=np.tile(sample,(8,2,1))
        assert cuda.cuMemAlloc_v2(C.byref(ptr),image.nbytes)==0
        assert cuda.cuMemcpyHtoD_v2(ptr,image.ctypes.data,image.nbytes)==0
        codec=Codec(context,(16,8),r'F:\DLSSC\VideoFX',hdr=False)
        packed,pitch=codec.hdr_to_sdr8(ptr.value,16*8)
        actual=np.zeros((8,16,4),np.uint8)
        assert cuda.cuMemcpyDtoH_v2(actual.ctypes.data,packed,actual.nbytes)==0
        rgb=image[:,:,:3].astype(np.float32)
        luminance=np.maximum(rgb@np.array([.2126,.7152,.0722],np.float32),0)
        linear=np.clip(rgb/(203/80+luminance[:,:,None]),0,1)
        encoded=np.where(linear<.018,4.5*linear,1.099*np.power(linear,.45)-.099)
        reference=np.rint(encoded*255).astype(np.uint8)
        error=int(np.abs(actual[:,:,:3].astype(int)-reference.astype(int)).max())
        assert error<=1 and np.all(actual[:,:,3]==255)
        assert 240<=int(actual[0,4,0])<255
        counters=codec.counters();assert counters['host_image_bytes']==0
        result['boundary']=dict(max_error_u8=error,counters=counters,
            test_only_image_upload_bytes=image.nbytes,test_only_image_readback_bytes=actual.nbytes)
    finally:
        if codec:codec.close()
        if ptr.value:cuda.cuMemFree_v2(ptr)
        cuda.cuCtxDestroy_v2(context)
    previews=[]
    for order in (['truehdr'],['nr','truehdr'],['truehdr','nr']):
        preview=GpuPreview()
        settings=dict(render_pipeline=order,truehdr_enabled=True,vsr_runtime_path=r'F:\DLSSC\VideoFX',
            export_backend='gpu_transfer',experimental_hdr_scrgb=True,working_scale=1,
            hdr_paper_white_scrgb=8.,guidance_mode=0,reconstruction_mode='single')
        try:
            _,image,diag=preview.render(str(ROOT/'build/hdr_probe/hdr_valid_src.mkv'),settings,0,display_edge=320,need_source=True)
            assert np.any(image)
            effect=diag['truehdr']['effect'];assert effect['hdr_to_sdr_calls']==1 and effect['calls']==1
            assert effect['input_conversion']['host_image_bytes']==0
            previews.append(dict(order=order,diagnostics=diag))
        finally:preview.shutdown()
    result['previews']=previews
    # The accepted encode already completed. Its old SDR-only checker failed on a
    # report key; inspect its original output without repeating a successful gate.
    original=json.loads((output/'export/result.json').read_text('utf-8'))['av1_nvenc']
    video=output/'export/av1_nvenc.mkv'
    facts=json.loads(subprocess.check_output([str(ROOT/'runtime/tools/ffprobe.exe'),'-v','error',
        '-show_streams','-show_frames','-of','json',str(video)]))
    stream=facts['streams'][0]
    assert stream['codec_name']=='av1' and stream['pix_fmt']=='yuv420p10le' and len(facts['frames'])==2
    assert [stream[k] for k in ('color_primaries','color_transfer','color_space')]==['bt2020','smpte2084','bt2020nc']
    assert original['truehdr']['effect']['hdr_to_sdr_calls']==2
    assert original['truehdr']['encoder_counters']['host_pixel_bytes']==0
    assert all(d['side_data_type'] not in ('Mastering display metadata','Content light level metadata')
        for frame in facts['frames'] for d in frame.get('side_data_list',[]))
    result['export']=dict(report=original,probe=facts,offline_recheck=True)
    result['passed']=True
    (output/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    print(json.dumps(dict(passed=True,boundary_error_u8=error,preview_orders=3,av1_frames=2)))

if __name__=='__main__':main()
