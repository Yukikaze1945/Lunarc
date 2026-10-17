"""Small cached SDR/HDR pair in an offscreen native window; no model inference."""
import ctypes as C
from ctypes import wintypes as W
from pathlib import Path
import sys
import time
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from image_render import DisplayResources,ImageResult
from native_preview import Presenter


def main():
    user=C.WinDLL('user32',use_last_error=True)
    user.CreateWindowExW.argtypes=[W.DWORD,W.LPCWSTR,W.LPCWSTR,W.DWORD,C.c_int,C.c_int,C.c_int,C.c_int,W.HWND,W.HMENU,W.HINSTANCE,C.c_void_p]
    user.CreateWindowExW.restype=W.HWND
    user.DestroyWindow.argtypes=[W.HWND]
    user.PeekMessageW.argtypes=[C.POINTER(W.MSG),W.HWND,W.UINT,W.UINT,W.UINT]
    user.DispatchMessageW.argtypes=[C.POINTER(W.MSG)]
    # Use a separate parent message thread, like Electron. Do not block the
    # parent's sent-message handling while querying the child's native mutex.
    parent_ready=threading.Event();stop=threading.Event();parents=[]
    def parent_loop():
        parent=user.CreateWindowExW(0,'STATIC','Lunarc cache display check',0x90000000,-32000,-32000,320,180,None,None,None,None)
        parents.append(parent);parent_ready.set()
        if not parent:return
        try:
            while not stop.is_set():
                msg=W.MSG()
                while user.PeekMessageW(C.byref(msg),None,0,0,1):user.DispatchMessageW(C.byref(msg))
                stop.wait(.01)
        finally:user.DestroyWindow(parent)
    thread=threading.Thread(target=parent_loop);thread.start();parent_ready.wait(5)
    parent=parents[0] if parents else None
    if not parent:stop.set();thread.join();raise RuntimeError('test parent window creation failed')
    if '--staged' in sys.argv:
        import bundle_paths
        original=bundle_paths.require_native
        bundle_paths.require_native=lambda name,purpose:str(Path(__file__).resolve().parents[2]/'native/build/presenter'/name) if name=='gpu_native_presenter.dll' else original(name,purpose)
    source=np.full((145,257,4),200,np.uint8)
    pixels=np.full((290,514,4),1.5,np.float16)
    source[:,:,3]=255;pixels[:,:,3]=1
    result=ImageResult('test',('memory',1,1),'test',source,pixels,True,{},time.time())
    resources=presenter=None
    try:
        resources=DisplayResources(result)
        presenter=Presenter(resources.live.d3d12_device(),dict(parent=str(parent),x=0,y=0,width=320,height=180,
            visible=True,view='split',zoom=1,pan_x=.5,pan_y=.5))
        resources.submit(presenter)
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            report=presenter.report()
            if report['present_calls']>=1:
                presenter.freeze()
                resources.close();resources=None
                presenter.configure(dict(parent=str(parent),x=0,y=0,width=330,height=190,
                    visible=True,view='split',zoom=1,pan_x=.5,pan_y=.5))
                assert presenter.report()['uploaded_pairs']==1
                resources=DisplayResources(result)
                replacement=Presenter(resources.live.d3d12_device(),dict(parent=str(parent),x=0,y=0,width=330,height=190,
                    visible=True,view='split',zoom=1,pan_x=.5,pan_y=.5))
                try:
                    resources.submit(replacement)
                    assert replacement.report()['present_calls']>=1
                    presenter.close();presenter=replacement;replacement=None
                finally:
                    if replacement:replacement.close()
                print('PASS: RGBA8/FP16 display frozen before old context release and atomically replaced by a new presenter',flush=True)
                return
            time.sleep(.02)
        raise AssertionError(f'native display did not present the cached pair: {report}')
    finally:
        if presenter:presenter.close()
        if resources:resources.close()
        stop.set();thread.join(5)


if __name__=='__main__':main()
