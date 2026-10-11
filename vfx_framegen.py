"""NVIDIA VFG offline output; original frames, real interpolation, explicit tail hold."""
import ctypes as C
from fractions import Fraction
import bundle_paths
from rtx_vsr_engine import Counters


def config(settings):
    engine = settings.get('frame_generation', 'off')
    if engine not in ('off', 'vfg', 'vfg_dynamic'):
        raise ValueError('Frame generation engine must be off, vfg or vfg_dynamic')
    if engine == 'off':
        return 1, 1
    multiplier = settings.get('frame_multiplier', 2)
    quality = settings.get('framegen_quality', 1)
    if engine == 'vfg_dynamic':
        target_fps(settings)
        multiplier = 0
    elif isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier not in range(2, 9):
        raise ValueError('Frame multiplier must be an integer from 2 through 8')
    if isinstance(quality, bool) or quality not in (0, 1, 2):
        raise ValueError('VFG quality must be low, medium or high')
    if not isinstance(settings.get('framegen_scene_detection', True), bool):
        raise ValueError('VFG scene detection must be a boolean')
    return multiplier, quality


def target_fps(settings):
    value = settings.get('framegen_target_fps', 60)
    try:
        if isinstance(value, bool):
            raise ValueError()
        rate = Fraction(str(value))
        if rate <= 0 or rate.numerator > 2147483647 or rate.denominator > 2147483647:
            raise ValueError()
        return rate
    except (ValueError, ZeroDivisionError, TypeError):
        raise ValueError('VFG target fps must be a positive finite number or rational') from None


def output_fps(settings, fps):
    multiplier, _ = config(settings)
    source = Fraction(*fps)
    result = target_fps(settings) if multiplier == 0 else source * multiplier
    if multiplier == 0 and result <= source:
        raise ValueError('VFG target fps must be greater than source fps')
    if result.numerator > 2147483647 or result.denominator > 2147483647:
        raise ValueError('VFG output frame rate exceeds encoder limits')
    return result.numerator, result.denominator


def encoder_timing(settings, fps, tb):
    multiplier, _ = config(settings)
    rate = output_fps(settings, fps)
    # Fixed mode retains the existing multiplied timestamp grid. Dynamic mode
    # emits sequential ticks on an exact reciprocal output-rate time base.
    time_base = (rate[1], rate[0]) if multiplier == 0 else (tb[0], tb[1] * multiplier)
    if max(time_base) > 2147483647:
        raise ValueError('VFG output time base exceeds encoder limits')
    return rate, time_base


def output_frame_count(settings, fps, frames):
    extent = frames * Fraction(*output_fps(settings, fps)) / Fraction(*fps)
    return -(-extent.numerator // extent.denominator)


class Copy2D(C.Structure):
    _fields_ = [('srcXInBytes', C.c_size_t), ('srcY', C.c_size_t), ('srcMemoryType', C.c_int),
        ('srcHost', C.c_void_p), ('srcDevice', C.c_uint64), ('srcArray', C.c_void_p), ('srcPitch', C.c_size_t),
        ('dstXInBytes', C.c_size_t), ('dstY', C.c_size_t), ('dstMemoryType', C.c_int),
        ('dstHost', C.c_void_p), ('dstDevice', C.c_uint64), ('dstArray', C.c_void_p), ('dstPitch', C.c_size_t),
        ('WidthInBytes', C.c_size_t), ('Height', C.c_size_t)]


class DeviceBuffers:
    def __init__(self, context):
        self.context, self.pointers, self.device_copy_bytes = context, [], 0
        self.lib = C.WinDLL('nvcuda.dll')
        for name, args in {
            'cuCtxPushCurrent_v2': [C.c_void_p], 'cuCtxPopCurrent_v2': [C.POINTER(C.c_void_p)],
            'cuMemAlloc_v2': [C.POINTER(C.c_uint64), C.c_size_t], 'cuMemFree_v2': [C.c_uint64],
            'cuMemcpy2D_v2': [C.POINTER(Copy2D)], 'cuStreamSynchronize': [C.c_void_p],
        }.items():
            getattr(self.lib, name).argtypes = args

    @staticmethod
    def check(rc):
        if rc:
            raise RuntimeError(f'VFG device operation failed: CUDA {rc}')

    def call(self, function, *args):
        self.check(self.lib.cuCtxPushCurrent_v2(self.context))
        try:
            self.check(getattr(self.lib, function)(*args))
        finally:
            previous = C.c_void_p()
            self.check(self.lib.cuCtxPopCurrent_v2(C.byref(previous)))

    def alloc(self, size):
        ptr = C.c_uint64()
        self.call('cuMemAlloc_v2', C.byref(ptr), size)
        self.pointers.append(ptr.value)
        return ptr.value

    def copy(self, src, pitch, dst, row, height):
        copy = Copy2D(srcMemoryType=2, srcDevice=src, srcPitch=pitch,
            dstMemoryType=2, dstDevice=dst, dstPitch=row, WidthInBytes=row, Height=height)
        self.call('cuMemcpy2D_v2', C.byref(copy))
        self.device_copy_bytes += row * height

    def close(self):
        while self.pointers:
            self.call('cuMemFree_v2', self.pointers.pop())


class FrameGenerator:
    def __init__(self, context, size, runtime, multiplier, quality, *, wide, scene_detection=True):
        self.lib = C.WinDLL(bundle_paths.require_native('rtx_vsr_bridge.dll', 'NVIDIA VFG'))
        self.lib.rtx_vsr_last_error.restype = C.c_char_p
        self.lib.vfx_framegen_open2.argtypes = [C.c_void_p] + [C.c_uint32]*6 + [C.c_wchar_p, C.POINTER(C.c_void_p)]
        self.lib.vfx_framegen_process.argtypes = [C.c_void_p, C.c_uint64, C.c_uint32,
            C.c_uint64, C.c_uint32, C.c_uint64, C.c_uint32, C.c_uint32, C.c_uint32]
        self.lib.vfx_framegen_process_time.argtypes = [C.c_void_p, C.c_uint64, C.c_uint32,
            C.c_uint64, C.c_uint32, C.c_uint64, C.c_uint32, C.c_float, C.c_uint32]
        self.lib.rtx_vsr_get_counters.argtypes = [C.c_void_p, C.POINTER(Counters)]
        self.lib.rtx_vsr_close.argtypes = [C.c_void_p]
        self.handle = C.c_void_p()
        self.check(self.lib.vfx_framegen_open2(context, *size, multiplier, quality, int(wide),
            int(scene_detection), runtime, C.byref(self.handle)))

    def check(self, rc):
        if rc:
            raise RuntimeError(f'NVIDIA VFG ({rc}): ' + self.lib.rtx_vsr_last_error().decode('utf-8', 'replace'))

    def process(self, previous, ppitch, current, cpitch, output, opitch, index, reset):
        self.check(self.lib.vfx_framegen_process(self.handle, previous, ppitch, current,
            cpitch, output, opitch, index, int(reset)))

    def process_time(self, previous, ppitch, current, cpitch, output, opitch, timestep, reset):
        self.check(self.lib.vfx_framegen_process_time(self.handle, previous, ppitch, current,
            cpitch, output, opitch, float(timestep), int(reset)))

    def counters(self):
        block = Counters()
        self.check(self.lib.rtx_vsr_get_counters(self.handle, C.byref(block)))
        return {name: getattr(block, name) for name, _ in Counters._fields_}

    def close(self):
        if self.handle:
            self.lib.rtx_vsr_close(self.handle)
            self.handle = None


class Interpolator:
    def __init__(self, context, size, settings, fps, *, wide, hdr=False, cancel=None):
        from native_runtime_paths import wide_vfx_runtime
        self.multiplier, quality = config(settings)
        self.dynamic = self.multiplier == 0
        self.target_rate = Fraction(*output_fps(settings, fps))
        self.scene_detection = settings.get('framegen_scene_detection', True)
        self.next_tick = self.origin_tick = 0
        self.finished = False
        self.recent_timesteps = []
        self.context, self.size, self.fps, self.wide = context, tuple(size), tuple(fps), wide
        self.cancel = cancel
        self.previous_pts = self.time_base = None
        self.sources = self.generated = self.emitted = self.tail_holds = 0
        self.recent_pts = []
        self.input_codec = self.output_codec = self.model = None
        self.buffers = DeviceBuffers(context)
        self.runtime = wide_vfx_runtime(settings.get('vsr_runtime_path'), framegen=True)
        try:
            row, height = size[0]*4, size[1]
            self.previous = self.buffers.alloc(row*height)
            self.generated_ptr = self.buffers.alloc(row*height)
            self.linear_row = size[0]*(8 if wide else 4)
            self.last_picture = self.buffers.alloc(self.linear_row*height)
            self.linear_output = self.buffers.alloc(self.linear_row*height) if wide else self.generated_ptr
            if wide:
                from vfx_rgb10 import Codec
                self.input_codec = Codec(context, size, self.runtime, hdr=hdr)
                self.output_codec = Codec(context, size, self.runtime, hdr=hdr)
            self.model = FrameGenerator(context, size, self.runtime, self.multiplier, quality,
                wide=wide, scene_detection=self.scene_detection)
        except Exception:
            self.close()
            raise

    def _cancel(self):
        if self.cancel and self.cancel():
            import pipeline
            raise pipeline.JobCancelled('用户已取消任务')

    def _emit(self, descriptor, pts, kind, emit):
        self._cancel()
        emit(descriptor, int(pts), kind)
        self.emitted += 1
        self.recent_pts = (self.recent_pts + [int(pts)])[-64:]

    def submit(self, color, frame, reset, emit):
        self._cancel()
        pts, tb = int(frame.pts), (int(frame.time_base_num), int(frame.time_base_den))
        if self.time_base is not None and tb != self.time_base:
            raise ValueError('VFG source time base changed')
        row = self.size[0]*4
        # Consumer encoding can recycle the bridge's borrowed current composite.
        # Retain it before emitting any intermediate frame.
        self.buffers.copy(color['device_ptr'], color['pitch_bytes'], self.last_picture, self.linear_row, self.size[1])
        color = dict(color, device_ptr=self.last_picture, pitch_bytes=self.linear_row)
        if self.input_codec:
            current, cpitch = self.input_codec.pack(color['device_ptr'], color['pitch_bytes'])
        else:
            current, cpitch = color['device_ptr'], color['pitch_bytes']
        if self.previous_pts is not None:
            delta = pts - self.previous_pts
            expected = Fraction(self.fps[1]*tb[1], self.fps[0]*tb[0])
            if delta <= 0 or abs(Fraction(delta)-expected) > 1:
                raise ValueError('VFG export requires uniform source timestamps; no silent frame-rate conversion')
            for index in range(1, self.multiplier):
                self._cancel()
                self.model.process(self.previous, row, current, cpitch, self.generated_ptr, row, index,
                    reset if getattr(self, 'scene_detection', True) else False)
                self.buffers.call('cuStreamSynchronize', None)
                if self.output_codec:
                    self.output_codec.unpack(self.generated_ptr, row, self.linear_output, self.linear_row)
                descriptor = dict(color, device_ptr=self.linear_output, pitch_bytes=self.linear_row)
                self._emit(descriptor, self.previous_pts*self.multiplier + delta*index, 'generated', emit)
                self.generated += 1
        if getattr(self, 'dynamic', False):
            self._submit_dynamic(color, current, cpitch, pts, tb, reset, emit)
        else:
            self._emit(color, pts*self.multiplier, 'original', emit)
        self.buffers.copy(current, cpitch, self.previous, row, self.size[1])
        self.previous_pts, self.time_base, self.sources = pts, tb, self.sources+1
        self.last_descriptor = dict(color, device_ptr=self.last_picture, pitch_bytes=self.linear_row)

    def _submit_dynamic(self, color, current, cpitch, pts, tb, reset, emit):
        source_rate = Fraction(*self.fps)
        if self.sources == 0:
            self.origin_tick = round(pts * Fraction(*tb) * self.target_rate)
        # CFR source ticks may be rounded by the container (e.g. milliseconds).
        # Use the verified nominal source grid so rounding does not accumulate.
        current_time = Fraction(self.sources, 1) / source_rate
        previous_time = Fraction(self.sources - 1, 1) / source_rate
        while Fraction(self.next_tick, 1) / self.target_rate <= current_time:
            at = Fraction(self.next_tick, 1) / self.target_rate
            if at == current_time:
                self._emit(color, self.origin_tick + self.next_tick, 'original', emit)
            else:
                self._cancel()
                phase = (at - previous_time) * source_rate
                self.model.process_time(self.previous, self.size[0]*4, current, cpitch,
                    self.generated_ptr, self.size[0]*4, phase,
                    reset if self.scene_detection else False)
                self.buffers.call('cuStreamSynchronize', None)
                if self.output_codec:
                    self.output_codec.unpack(self.generated_ptr, self.size[0]*4, self.linear_output, self.linear_row)
                self._emit(dict(color, device_ptr=self.linear_output, pitch_bytes=self.linear_row),
                    self.origin_tick + self.next_tick, 'generated', emit)
                self.generated += 1
                self.recent_timesteps = (self.recent_timesteps + [float(phase)])[-64:]
            self.next_tick += 1

    def finish(self, emit):
        if getattr(self, 'dynamic', False):
            if self.previous_pts is None or self.finished:
                return
            duration = Fraction(self.sources, 1) / Fraction(*self.fps)
            while Fraction(self.next_tick, 1) / self.target_rate < duration:
                self._emit(self.last_descriptor, self.origin_tick + self.next_tick, 'tail_hold', emit)
                self.next_tick += 1
                self.tail_holds += 1
            self.finished = True
            return
        if self.previous_pts is None or self.tail_holds:
            return
        # No future source frame exists at EOF. Explicit holds preserve the source
        # duration; these are reported separately and never called generated frames.
        step = Fraction(self.fps[1]*self.time_base[1], self.fps[0]*self.time_base[0])
        for i in range(1, self.multiplier):
            self._emit(self.last_descriptor, self.previous_pts*self.multiplier + round(step*i), 'tail_hold', emit)
            self.tail_holds += 1

    def report(self):
        return dict(engine='NVIDIA VideoFrameGeneration', multiplier=self.multiplier,
            mode='dynamic' if self.dynamic else 'fixed', output_fps=str(self.target_rate),
            automatic_shot_change_detection=self.scene_detection, recent_timesteps=self.recent_timesteps,
            runtime=self.runtime, source_frames=self.sources, generated_frames=self.generated,
            tail_hold_frames=self.tail_holds, output_frames=self.emitted, recent_output_pts=self.recent_pts,
            model=self.model.counters() if self.model else None,
            encoding='RGB10A2' if self.wide else 'RGBA8', host_image_bytes=0,
            device_copy_bytes=self.buffers.device_copy_bytes,
            colour_counters=self.input_codec.counters() if self.input_codec else None)

    def close(self):
        for item in (self.model, self.output_codec, self.input_codec, self.buffers):
            if item:
                item.close()
