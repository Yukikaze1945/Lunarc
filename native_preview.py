"""Owned GPU display pair and Win32 child panel; no host image data."""
import ctypes as C
import math
import bundle_paths

VIEWS = {'orig':0,'dlss':1,'split':2,'dual':3}

def validate_layout(value):
    if not isinstance(value,dict):
        raise ValueError('原生显示布局无效')
    result = dict(value)
    for key in ('x','y','width','height'):
        number = value.get(key,0 if key in ('x','y') else 1)
        if isinstance(number,bool) or not isinstance(number,int) or abs(number)>16384:
            raise ValueError('原生显示尺寸无效')
        result[key]=number
    if result['width']<1 or result['height']<1:
        raise ValueError('原生显示区域不能为空')
    for key,default,low,high in (('zoom',1.,.1,32.),('pan_x',.5,0.,1.),('pan_y',.5,0.,1.)):
        number=value.get(key,default)
        if isinstance(number,bool) or not isinstance(number,(int,float)) or not math.isfinite(number) or not low<=number<=high:
            raise ValueError('原生显示缩放参数无效')
        result[key]=float(number)
    if result.get('view','split') not in VIEWS:
        raise ValueError('原生显示不支持此视图')
    result['view']=result.get('view','split')
    result['visible']=bool(result.get('visible',True))
    parent=value.get('parent')
    if not isinstance(parent,str) or not parent.isdecimal() or not 0<int(parent)<2**64:
        raise ValueError('原生显示窗口无效')
    return result


class Stats(C.Structure):
    _fields_=[('size',C.c_uint32),('hdr',C.c_uint32),('width',C.c_uint32),('height',C.c_uint32),
        ('frames',C.c_uint64),('presents',C.c_uint64),('drag_presents',C.c_uint64),('host_image_bytes',C.c_uint64),
        ('split',C.c_float),('sdr_white',C.c_float),('hwnd',C.c_uint64),('pointer_downs',C.c_uint32),('down_x',C.c_float)]


class Presenter:
    def __init__(self,device,layout):
        self.parent=layout['parent']
        bundle_paths.add_dll_directory(bundle_paths.native_dir())
        self.lib=C.CDLL(bundle_paths.require_native('gpu_native_presenter.dll','原生 FP16 预览'))
        self.lib.gpu_native_presenter_error.restype=C.c_char_p
        self.lib.gpu_native_presenter_open.argtypes=[C.c_void_p,C.c_uint64,C.POINTER(C.c_void_p)]
        self.lib.gpu_native_presenter_config.argtypes=[C.c_void_p]+[C.c_int]*6+[C.c_float]*3
        self.lib.gpu_native_presenter_frame.argtypes=[C.c_void_p,C.c_void_p,C.c_uint64]+[C.c_uint32]*5+[C.c_uint64]+[C.c_uint32]*5
        self.lib.gpu_native_presenter_stats.argtypes=[C.c_void_p,C.POINTER(Stats)]
        self.lib.gpu_native_presenter_close.argtypes=[C.c_void_p]
        self.freeze_fn=getattr(self.lib,'gpu_native_presenter_freeze',None)
        if self.freeze_fn:self.freeze_fn.argtypes=[C.c_void_p]
        self.handle=C.c_void_p()
        self._check(self.lib.gpu_native_presenter_open(device,int(self.parent),C.byref(self.handle)))
        try:self.configure(layout)
        except Exception:self.close();raise

    def _check(self,code):
        if code:
            detail=self.lib.gpu_native_presenter_error()
            raise RuntimeError('原生预览：'+(detail.decode('utf8','replace') if detail else str(code)))

    def configure(self,layout):
        self._check(self.lib.gpu_native_presenter_config(self.handle,layout['x'],layout['y'],
            layout['width'],layout['height'],int(layout['visible']),VIEWS[layout['view']],
            layout['zoom'],layout['pan_x'],layout['pan_y']))

    def frame(self,source,result,source_hdr,result_hdr):
        def plane(d,hdr):
            return (d['device_ptr'],d['pitch_bytes'],d['width'],d['height'],
                    8 if d['format']=='rgba16f_cuda' else 4,int(hdr))
        self._check(self.lib.gpu_native_presenter_frame(self.handle,source['cuda_context'],
            *plane(source,source_hdr),*plane(result,result_hdr)))

    def report(self):
        out=Stats(size=C.sizeof(Stats))
        self._check(self.lib.gpu_native_presenter_stats(self.handle,C.byref(out)))
        return dict(native=True,hdr_active=bool(out.hdr),display_format='rgba16f/scRGB',
            display_resolution=[out.width,out.height],uploaded_pairs=out.frames,
            present_calls=out.presents,drag_present_calls=out.drag_presents,
            display_host_image_bytes=out.host_image_bytes,split_x=out.split,
            sdr_white_scrgb=out.sdr_white,hwnd=str(out.hwnd),
            pointer_downs=out.pointer_downs,pointer_down_x=out.down_x,
            output_mode='HDR scRGB' if out.hdr else 'SDR display mapping')

    def freeze(self):
        if not self.freeze_fn:return False  # Older installed DLLs close normally.
        self._check(self.freeze_fn(self.handle))
        return True

    def close(self):
        if self.handle:
            self.lib.gpu_native_presenter_close(self.handle);self.handle=None
