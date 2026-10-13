"""NVIDIA TrueHDR: explicit RGBA8 SDK boundary, PQ RGB10A2 output to linear FP16."""
import ctypes as C
from pathlib import Path
import bundle_paths

DEFAULTS = dict(truehdr_contrast=100, truehdr_saturation=100,
                truehdr_middle_gray=50, truehdr_peak_nits=650, truehdr_debanding=True)


def parameters(settings):
    result = []
    for key, low, high in [('truehdr_contrast',0,200), ('truehdr_saturation',0,200),
                           ('truehdr_middle_gray',10,100), ('truehdr_peak_nits',400,2000)]:
        value = settings.get(key, DEFAULTS[key])
        if isinstance(value, bool) or not isinstance(value, (int,float)) or int(value) != value or not low <= value <= high:
            raise ValueError(f'{key} must be an integer in {low}..{high}')
        result.append(int(value))
    enabled = settings.get('truehdr_debanding', True)
    if not isinstance(enabled, bool):
        raise ValueError('truehdr_debanding must be a boolean')
    return (*result, int(not enabled))


def runtime_directory(settings):
    from native_runtime_paths import vsr_runtime
    path = Path(vsr_runtime(settings.get('vsr_runtime_path')))
    if not any((path / suffix / 'nvVFXTrueHDR.dll').is_file()
               for suffix in ('features/nvvfxtruehdr/bin','bin','')):
        raise FileNotFoundError('缺少 NVIDIA TrueHDR 特效模块和模型；请在 VFX SDK 中安装 nvvfxtruehdr 组件')
    return str(path)


class TrueHDR:
    def __init__(self, device, context, size, settings, *, wide=False):
        self.device, self.context, self.size, self.wide = device, context, tuple(size), wide
        self.runtime = runtime_directory(settings)
        self.options = parameters(settings)
        self.lib = C.WinDLL(bundle_paths.require_native('rtx_vsr_bridge.dll', 'NVIDIA TrueHDR'))
        self.lib.rtx_vsr_last_error.restype = C.c_char_p
        self.lib.vfx_truehdr_open.argtypes = [C.c_void_p] + [C.c_uint32]*7 + [C.c_wchar_p,C.POINTER(C.c_void_p)]
        self.lib.vfx_truehdr_update.argtypes = [C.c_void_p] + [C.c_uint32]*5
        self.lib.vfx_truehdr_process.argtypes = [C.c_void_p,C.c_uint64,C.c_uint32,C.c_uint64,C.c_uint32]
        self.lib.rtx_vsr_close.argtypes = [C.c_void_p]
        self.handle = C.c_void_p()
        self.input = self.output = self.input_codec = self.output_codec = None
        self.calls = 0
        self._check(self.lib.vfx_truehdr_open(context,*size,*self.options,self.runtime,C.byref(self.handle)))
        try:
            from native_effects import Exchange
            from vfx_rgb10 import Codec
            self.input = Exchange(device,context,size,wide=wide)
            self.output = Exchange(device,context,size,wide=True)
            if wide:
                self.input_codec = Codec(context,size,self.runtime,hdr=False)
            self.output_codec = Codec(context,size,self.runtime,hdr=True)
        except Exception:
            self.close()
            raise

    def _check(self, code):
        if code:
            detail = self.lib.rtx_vsr_last_error()
            raise RuntimeError('NVIDIA TrueHDR: '+(detail.decode('utf-8','replace') if detail else str(code)))

    def update(self, settings):
        options = parameters(settings)
        if options != self.options:
            self._check(self.lib.vfx_truehdr_update(self.handle,*options))
            self.options = options

    def process(self, candidate, frame, reset=False):
        color = candidate if 'device_ptr' in candidate else self.input.color(candidate)
        ptr, pitch = color['device_ptr'], color['pitch_bytes']
        if self.wide:
            ptr, pitch = self.input_codec.sdr8(ptr,pitch)
        packed, packed_pitch = self.output_codec.pointer()
        self._check(self.lib.vfx_truehdr_process(self.handle,ptr,pitch,packed,packed_pitch))
        output, row = self.output.pointer()
        self.output_codec.unpack(packed,packed_pitch,output,row)
        self.calls += 1
        result = self.output.texture(output,row)
        result.update(format='rgba16f_cuda',cuda_context=self.context,device_ptr=output,
                      pitch_bytes=row,output_hdr=True)
        return result

    def report(self):
        return dict(stage='truehdr',calls=self.calls,output_hdr=True,
                    input_encoding='RGBA8 SDR (explicit NVIDIA SDK boundary)',
                    output_encoding='BT.2020 PQ RGB10A2 → linear scRGB FP16',
                    parameters=dict(zip(DEFAULTS,(*self.options[:4],not self.options[4]))),
                    input_conversion=self.input_codec.counters() if self.input_codec else None)

    def close(self):
        for name in ('output_codec','input_codec','output','input'):
            obj = getattr(self,name,None)
            if obj:
                obj.close(); setattr(self,name,None)
        if self.handle:
            self.lib.rtx_vsr_close(self.handle); self.handle=None
