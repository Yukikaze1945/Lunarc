"""Final GPU resampling after the entire user-ordered rendering chain."""
import ctypes as C
import time
import bundle_paths

ALGORITHMS = {'nearest': 0, 'bilinear': 1, 'bicubic': 2, 'lanczos': 3}


def validate(size, algorithm):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 or v > 16384 or v % 2 for v in size):
        raise ValueError('导出分辨率必须为 2–16384 范围内的偶数宽高')
    if algorithm not in ALGORITHMS:
        raise ValueError('Unknown export scaling algorithm: ' + str(algorithm))


class Resize:
    def __init__(self, device, context, source_size, size, algorithm='lanczos', *, wide=False):
        validate(size, algorithm)
        self.size, self.source_size = tuple(size), tuple(source_size)
        self.device, self.context, self.wide = device, context, wide
        self.algorithm, self.calls, self.elapsed_ms = algorithm, 0, 0.
        self.input_exchange = self.output_exchange = None
        self.lib = C.CDLL(bundle_paths.require_native('gpu_export_resize.dll', 'GPU export resizing'))
        self.lib.gpu_export_resize_error.restype = C.c_char_p
        self.lib.gpu_export_resize_open.argtypes = [C.c_void_p, C.c_void_p, *([C.c_uint32]*6), C.POINTER(C.c_void_p)]
        self.lib.gpu_export_resize_run.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)]
        self.lib.gpu_export_resize_close.argtypes = [C.c_void_p]
        self.handle = C.c_void_p()
        self._check(self.lib.gpu_export_resize_open(device, context, *source_size, *size,
            8 if wide else 4, ALGORITHMS[algorithm], C.byref(self.handle)))

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.gpu_export_resize_error().decode('utf-8', 'replace'))

    def color(self, color):
        if (color['width'], color['height']) != self.source_size or color['format'] != ('rgba16f_cuda' if self.wide else 'rgba8_cuda'):
            raise ValueError('Export resize input extent/precision changed')
        start = time.perf_counter()
        pointer, pitch = C.c_uint64(), C.c_uint32()
        self._check(self.lib.gpu_export_resize_run(self.handle, color['device_ptr'], color['pitch_bytes'], C.byref(pointer), C.byref(pitch)))
        self.calls += 1
        self.elapsed_ms += (time.perf_counter()-start)*1000
        return dict(color, width=self.size[0], height=self.size[1], device_ptr=pointer.value, pitch_bytes=pitch.value)

    def texture(self, texture):
        from native_effects import Exchange
        if self.input_exchange is None:
            self.input_exchange = Exchange(self.device, self.context, self.source_size, wide=self.wide)
        return self.as_texture(self.color(self.input_exchange.color(texture)))

    def as_texture(self, color):
        from native_effects import Exchange
        if self.output_exchange is None:
            self.output_exchange = Exchange(self.device, self.context, self.size, wide=self.wide)
        return self.output_exchange.texture(color['device_ptr'], color['pitch_bytes'])

    def report(self):
        return dict(source_size=list(self.source_size), output_size=list(self.size),
            algorithm=self.algorithm, calls=self.calls, elapsed_ms=self.elapsed_ms,
            host_image_bytes=0, scalar_readback_bytes=0,
            device_copy_bytes=self.calls*self.source_size[0]*self.source_size[1]*(8 if self.wide else 4)
            + sum(e.device_copy_bytes for e in (self.input_exchange,self.output_exchange) if e))

    def close(self):
        for e in (self.output_exchange, self.input_exchange):
            if e: e.close()
        self.output_exchange = self.input_exchange = None
        if self.handle:
            self.lib.gpu_export_resize_close(self.handle)
            self.handle = None
