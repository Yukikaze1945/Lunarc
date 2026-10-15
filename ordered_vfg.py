"""VFG at the selected GPU stage boundary, with synchronous borrowed pictures.

Before NR, generated CUDA colour enters the existing reconstruction directly;
there is no intermediate encode/decode or host picture copy. After NR, the
remaining spatial stages consume every emitted picture, including EOF holds.
"""
import ctypes as C
import time
from native_effects import EffectSeries, frame_at_size, output_size, nr_as_effect
from vfx_framegen import Interpolator, encoder_timing


def install_ordered_vfg(chain, live, settings, source_size, provider, order):
    boundary = order.index('vfg')
    before, after = order[:boundary], order[boundary + 1:]
    size = output_size(settings,source_size,before)
    chain.framegen_stage_index = boundary
    chain.framegen_stage_size = size
    chain.framegen_hdr = 'truehdr' in before
    tail = getattr(chain, 'truehdr_tail', None)
    if 'nr' in before:
        chain.framegen_tail = EffectSeries(live, settings, size, after, provider,
            wide=True if chain.framegen_hdr else None, hdr=chain.framegen_hdr, truehdr_tail=tail) if after else None
        return before
    driver = BeforeReconstruction(chain, live, settings, source_size, size, before, provider, after)
    chain.ordered_framegen = driver
    chain.ordered_input_size = size
    if 'nr' not in after or nr_as_effect(settings):
        chain.processing_size = size
        chain.working = size
        live._w, live._h = size
    chain.process = driver.process
    chain.framegen_emit = driver.emit
    return after


def stage_report(chain):
    return dict(stage_index=chain.framegen_stage_index, input_size=list(chain.framegen_stage_size),
                render_order=chain.render_order,
                prefix=chain.ordered_framegen.prefix.report()
                    if getattr(chain, 'ordered_framegen', None) and chain.ordered_framegen.prefix else None,
                suffix=(chain.ordered_framegen.suffix.report()
                    if getattr(getattr(chain, 'ordered_framegen', None), 'suffix', None)
                    else chain.framegen_tail.report() if getattr(chain, 'framegen_tail', None) else None))


class BeforeReconstruction:
    def __init__(self, chain, live, settings, source_size, size, before, provider, after=None):
        self.chain, self.live, self.settings = chain, live, dict(settings)
        self.source_size, self.size = tuple(source_size), tuple(size)
        self.prefix = EffectSeries(live, settings, source_size, before, provider,
            truehdr_tail=getattr(chain, 'truehdr_tail', None)) if before else None
        self.hdr_direct = chain.framegen_hdr
        self.suffix = EffectSeries(live, settings, size, after, provider,
            wide=True, hdr=True, truehdr_tail=getattr(chain, 'truehdr_tail', None)) if self.hdr_direct and after else None
        self.core_process = chain.process
        self.wide = getattr(live, 'bytes_per_pixel', 4) == 8
        self.emitted = 0
        self.frame = self.provider = None
        self.scene_observer = None

    def process(self, frame, flow_provider, index, is_discontinuous):
        chain = self.chain
        start = time.perf_counter()
        self.frame = frame_at_size(frame, (frame.width, frame.height))
        self.provider = flow_provider
        self.reset = bool(is_discontinuous or index == 0)
        if self.wide:
            if chain.first:
                chain._open_stages(frame)
                chain.first = False
            else:
                chain._assert_metadata_stable()
            colour = {name: int(getattr(frame, name)) for name in
                      ('color_range', 'color_primaries', 'color_transfer', 'color_space')}
            if chain.sdr10:
                from source_colour import resolve_frame_colour
                colour = resolve_frame_colour(colour, chain.policy.get('source_colour_resolution'))
            decoder = chain.decode_bridge or chain.bridge
            master = decoder.decode(p010_y_ptr=frame.y_device_ptr, p010_y_pitch=frame.y_pitch_bytes,
                p010_uv_ptr=frame.uv_device_ptr, p010_uv_pitch=frame.uv_pitch_bytes,
                support_ceiling_scrgb=float(chain.policy['support_ceiling_scrgb']), **colour)
            color = master.as_descriptor()
            self.source_report = decoder.source_report()
        else:
            if chain.nv12 is None:
                chain._open_color_stages(frame)
            ptr, pitch = C.c_uint64(), C.c_uint32()
            rc = chain.b['nv12'].nv12_rgba_convert(chain.nv12, frame.y_device_ptr,
                frame.y_pitch_bytes, frame.uv_device_ptr, frame.uv_pitch_bytes, C.byref(ptr), C.byref(pitch))
            if rc:
                from gpu_transfer_export import _bridge_error
                _bridge_error(chain.b['nv12'], 'nv12_rgba_last_error', rc)
            color = dict(format='rgba8_cuda', cuda_context=frame.cuda_context,
                         width=frame.width, height=frame.height, device_ptr=ptr.value, pitch_bytes=pitch.value)
        if chain.temporal_control.controls.enabled:
            if self.scene_observer is None:
                from temporal_control import GpuTemporalControl
                self.scene_observer = GpuTemporalControl(chain.temporal_control.controls, hdr=self.wide)
            scene = self.scene_observer.observe(device=self.live.d3d12_device(),
                cuda_context=frame.cuda_context, source_ptr=color['device_ptr'],
                source_pitch=color['pitch_bytes'], width=color['width'], height=color['height'], reset=self.reset)
            self.reset = bool(self.reset or scene['scene_cut'])
        if self.prefix:
            color = self.prefix.color(color, frame, self.reset)
        if chain.framegen is None:
            chain.framegen = Interpolator(frame.cuda_context, self.size, self.settings, chain.fps,
                wide=self.wide or chain.framegen_hdr,
                hdr=chain.framegen_hdr or (self.wide and not chain.sdr10), cancel=getattr(chain, 'cancel', None))
        self.rate, self.tb = encoder_timing(self.settings, chain.fps,
            (int(frame.time_base_num), int(frame.time_base_den)))
        chain._encoder_fps = self.rate
        chain.framegen.submit(color, frame, self.reset, self.emit)
        return dict(stages={'ordered_vfg_and_downstream': (time.perf_counter() - start) * 1000,
                           'decode_pts': int(frame.pts)},
                    flow=self.last_outcome.get('flow'), nvof_timing=self.last_outcome.get('nvof_timing'))

    def emit(self, color, pts, kind):
        frame = frame_at_size(self.frame, (self.frame.width, self.frame.height))
        frame.pts, frame.time_base_num, frame.time_base_den = pts, *self.tb
        # A scene/discontinuity boundary resets downstream history on the new
        # original picture. The very first emission also starts fresh history.
        reset = self.emitted == 0 or (kind == 'original' and self.reset)
        if getattr(self, 'hdr_direct', False):
            from native_effects import Exchange
            if not hasattr(self, 'hdr_output'):
                self.hdr_output = Exchange(self.live.d3d12_device(), frame.cuda_context, self.size, wide=True)
            texture = self.hdr_output.texture(color['device_ptr'], color['pitch_bytes'])
            texture['output_hdr'] = True
            if self.suffix:
                texture = self.suffix.process(texture, frame, reset)
            self.chain.truehdr_tail.encode(self.chain, texture, frame, pts)
            self.last_outcome = {}
            self.emitted += 1
            return
        self.chain._ordered_color = color
        try:
            self.last_outcome = self.core_process(frame, self.provider, self.emitted, reset) or {}
        finally:
            self.chain._ordered_color = None
        self.emitted += 1

    def close(self):
        for item in (self.suffix, getattr(self, 'hdr_output', None)):
            if item: item.close()
        if self.scene_observer:
            self.scene_observer.close()
            self.scene_observer = None
        if self.prefix:
            self.prefix.close()
            self.prefix = None
