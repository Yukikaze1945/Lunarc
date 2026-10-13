"""One owned GPU source/result pair for same-frame post-control edits.

Never cache borrowed processing pointers: reconstruction and post processing
reuse their allocations. This owner closes before the decoder CUDA context.
"""
import ctypes as C
import time

class CacheMemoryUnavailable(RuntimeError):
    pass


class _Copy2D(C.Structure):
    _fields_ = [('srcXInBytes',C.c_size_t),('srcY',C.c_size_t),('srcMemoryType',C.c_uint),
        ('srcHost',C.c_void_p),('srcDevice',C.c_uint64),('srcArray',C.c_void_p),('srcPitch',C.c_size_t),
        ('dstXInBytes',C.c_size_t),('dstY',C.c_size_t),('dstMemoryType',C.c_uint),
        ('dstHost',C.c_void_p),('dstDevice',C.c_uint64),('dstArray',C.c_void_p),('dstPitch',C.c_size_t),
        ('WidthInBytes',C.c_size_t),('Height',C.c_size_t)]


class PostCache:
    def __init__(self):
        self.valid = False
        self.identity = None
        self.exchange = None
        self.source_ptr = self.candidate_ptr = 0
        self.copies = self.reuses = 0
        self.capture_ms = 0.
        self.outcome = None
        self.disabled_identity = None
        self.disabled_reason = None

    def invalidate(self):
        self.valid = False
        self.outcome = None

    def _check(self, code):
        if code:
            raise RuntimeError('GPU preview cache CUDA error: ' + str(code))

    def _push(self):
        self._check(self.cuda.cuCtxPushCurrent_v2(self.context))

    def _pop(self):
        old = C.c_void_p()
        self._check(self.cuda.cuCtxPopCurrent_v2(C.byref(old)))

    def _open(self, device, context, width, height, wide):
        identity = (device,context,width,height,wide)
        if self.identity == identity:
            return
        self.close()
        self.cuda = C.WinDLL('nvcuda.dll')
        self.cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        self.cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        self.cuda.cuMemAlloc_v2.argtypes = [C.POINTER(C.c_uint64),C.c_size_t]
        self.cuda.cuMemFree_v2.argtypes = [C.c_uint64]
        self.cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(_Copy2D)]
        self.identity = identity
        self.device,self.context,self.width,self.height,self.wide = identity
        self.pixel_bytes = 8 if wide else 4
        self.pitch = (width*self.pixel_bytes+255)&~255
        try:
            self._push()
            try:
                for name in ('source_ptr','candidate_ptr'):
                    value = C.c_uint64()
                    code = self.cuda.cuMemAlloc_v2(C.byref(value),self.pitch*height)
                    if code == 2:
                        raise CacheMemoryUnavailable('insufficient GPU memory for optional preview cache')
                    self._check(code)
                    setattr(self,name,value.value)
            finally:
                self._pop()
        except Exception:
            self.close()
            raise

    def _copy(self, pointer, pitch, output):
        desc = _Copy2D()
        desc.srcMemoryType = desc.dstMemoryType = 2
        desc.srcDevice,desc.srcPitch = pointer,pitch
        desc.dstDevice,desc.dstPitch = output,self.pitch
        desc.WidthInBytes,desc.Height = self.width*self.pixel_bytes,self.height
        self._check(self.cuda.cuMemcpy2D_v2(C.byref(desc)))
        self.copies += 1

    def capture(self, *, device, context, width, height, wide, source_ptr, source_pitch, candidate, frame):
        start = time.perf_counter()
        self.valid = False
        identity = (device,context,width,height,wide)
        if identity == self.disabled_identity:
            return
        try:
            self._open(device,context,width,height,wide)
        except CacheMemoryUnavailable as error:
            self.disabled_identity, self.disabled_reason = identity, str(error)
            return
        if 'device_ptr' in candidate:
            color = candidate
        else:
            if self.exchange is None:
                from native_effects import Exchange
                self.exchange = Exchange(device,context,(width,height),wide=wide)
            color = self.exchange.color(candidate)
        self._push()
        try:
            self._copy(source_ptr,source_pitch,self.source_ptr)
            self._copy(color['device_ptr'],color['pitch_bytes'],self.candidate_ptr)
        finally:
            self._pop()
        self.frame = frame
        self.capture_ms = (time.perf_counter()-start)*1000
        self.valid = True

    def descriptor(self, pointer):
        return dict(format='rgba16f_cuda' if self.wide else 'rgba8_cuda',width=self.width,height=self.height,
            cuda_context=self.context,device_ptr=pointer,pitch_bytes=self.pitch)

    def reprocess(self, chain):
        if not self.valid or self.outcome is None:
            raise RuntimeError('GPU preview post cache is not valid')
        start = time.perf_counter()
        candidate = self.descriptor(self.candidate_ptr)
        controlled = chain.post_control.process(device=self.device,cuda_context=self.context,
            source_ptr=self.source_ptr,source_pitch=self.pitch,width=self.width,height=self.height,
            candidate=candidate)
        stages = dict(post_control=(time.perf_counter()-start)*1000)
        if self.wide:
            from hdr_gpu_engine import DevicePlane
            if 'device_ptr' in controlled:
                composited = DevicePlane(fmt='rgba16f_cuda',pixel_format='cached linear NR result',
                    width=self.width,height=self.height,cuda_context=self.context,
                    device_ptr=controlled['device_ptr'],pitch_bytes=controlled['pitch_bytes'],bytes_per_pixel=8)
            else:
                composited = chain.bridge.set_composite(controlled)
            composited = chain._native_post_process(composited,stages,self.frame)
            preview = chain._consume_preview_output(self.frame,composited)
        else:
            if 'device_ptr' in controlled:
                if self.exchange is None:
                    from native_effects import Exchange
                    self.exchange = Exchange(self.device,self.context,(self.width,self.height))
                controlled = self.exchange.texture(controlled['device_ptr'],controlled['pitch_bytes'])
            preview = chain._submit_output(self.frame,controlled,stages)
        self.reuses += 1
        return dict(self.outcome,stages=stages,preview_output=preview)

    def report(self):
        return dict(valid=self.valid,reuses=self.reuses,device_copy_calls=self.copies,
            disabled_reason=self.disabled_reason,
            private_plane_bytes=2*self.pitch*self.height if self.identity else 0,
            capture_ms=self.capture_ms,host_image_bytes=0)

    def close(self):
        self.invalidate()
        self.disabled_identity = self.disabled_reason = None
        if self.exchange:
            self.exchange.close()
            self.exchange = None
        if self.identity:
            self._push()
            try:
                for name in ('source_ptr','candidate_ptr'):
                    pointer = getattr(self,name)
                    if pointer:
                        self._check(self.cuda.cuMemFree_v2(pointer))
                        setattr(self,name,0)
            finally:
                self._pop()
            self.identity = None
