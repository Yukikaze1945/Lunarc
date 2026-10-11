"""Ordered native models with format-matched GPU transfers.

DLAA accepts RGBA8 or linear FP16. VFX 1.3 VideoSuperRes uses RGB10A2 for wide
pictures; only its GPU encoding boundary quantises the FP16 master.
"""
import ctypes as C
import os

import bundle_paths

MODES = ('nr', 'dlaa', 'vsr', 'nr_dlaa', 'nr_vsr')
LEGACY_STAGES = {'nr': ['nr'], 'dlaa': ['dlaa'], 'vsr': ['vsr'],
                 'nr_dlaa': ['nr', 'dlaa'], 'nr_vsr': ['nr', 'vsr']}
DLAA_PRESETS = (0, 5, 6, 10, 11, 12, 13)


def stages(settings):
    value = settings.get('render_pipeline')
    if value is None:
        selected = str(settings.get('neural_mode') or 'nr')
        if selected not in LEGACY_STAGES:
            raise ValueError('Unknown native processing mode: ' + selected)
        value = list(LEGACY_STAGES[selected])
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('Render pipeline must contain at least one stage')
    if any(stage not in ('nr', 'dlaa', 'vsr', 'vfg') for stage in value) or len(set(value)) != len(value):
        raise ValueError('Render pipeline stages must be unique NR, DLAA, VSR or VFG')
    ordered = list(value)
    if settings.get('frame_generation', 'off') != 'off' and 'vfg' not in ordered:
        ordered.append('vfg')
    return ordered


def mode(settings):
    if settings.get('render_pipeline') is not None:
        ordered = stages(settings)
        return next((key for key, value in LEGACY_STAGES.items() if value == ordered), 'pipeline')
    value = str(settings.get('neural_mode') or 'nr')
    if value not in MODES:
        raise ValueError('Unknown native processing mode: ' + value)
    return value


def validate(settings, source):
    from vfx_framegen import config
    multiplier, _ = config(settings)
    if multiplier != 1:
        from vfx_framegen import output_fps
        from gpu_transfer_export import fps_rational
        output_fps(settings, fps_rational(source) or (int(source.get('fps', 30)), 1))
        if settings.get('export_backend') != 'gpu_transfer':
            raise ValueError('VFG requires GPU Transfer')
        from native_runtime_paths import wide_vfx_runtime
        wide_vfx_runtime(settings.get('vsr_runtime_path'), framegen=True)
    ordered = stages(settings)
    if ordered == ['nr']:
        return
    wide = source.get('pix_fmt') in ('yuv420p10le', 'p010le', 'p010')
    if source.get('color_transfer') == 'arib-std-b67':
        raise ValueError('HLG is unsupported by the existing native HDR10/PQ chain')
    if source.get('color_transfer') == 'smpte2084' and not wide:
        raise ValueError('HDR10 native stages require a 10-bit P010 source')
    if 'vsr' in ordered and wide:
        from native_runtime_paths import wide_vfx_runtime
        wide_vfx_runtime(settings.get('vsr_runtime_path'))
    if source.get('pix_fmt') not in ('yuv420p', 'nv12', 'yuv420p10le', 'p010le', 'p010'):
        raise ValueError('Native stages require a supported 8-bit or 10-bit YUV source')
    if settings.get('export_backend') != 'gpu_transfer':
        raise ValueError('Native DLAA / RTX VSR requires GPU Transfer')
    if 'dlaa' in ordered and (isinstance(settings.get('dlaa_preset', 0), bool) or
                             settings.get('dlaa_preset', 0) not in DLAA_PRESETS):
        raise ValueError('DLAA preset must be Default, E, F, J, K, L or M')
    if int(settings.get('guidance_mode', 0)) and settings.get('flow_engine', 'nvof') not in ('nvof','fidelityfx','raft','raft_large'):
        raise ValueError('Native DLAA / NR guidance requires device optical flow')
    if 'nr' not in ordered and (
            settings.get('reconstruction_mode', 'single') != 'single' or
            float(settings.get('working_scale', 1)) != 1):
        raise ValueError('Standalone DLAA / RTX VSR requires Single at 100% input')
    if not any(s in ordered for s in ('nr', 'dlaa')) and int(settings.get('guidance_mode', 0)):
        raise ValueError('RTX VSR standalone does not consume optical-flow guidance')
    if 'vsr' in ordered:
        quality = int(settings.get('vsr_quality', 2))
        if quality not in (1, 2, 3, 4):
            raise ValueError('RTX VSR quality must be 1–4')
        output_size(settings, (source['width'], source['height']))


def output_size(settings, size):
    if 'vsr' not in stages(settings):
        return tuple(size)
    import math
    mode = settings.get('vsr_output_mode', 'resolution')
    choices = {'720p': (1280, 720), '1080p': (1920, 1080),
               '1440p': (2560, 1440), '4k': (3840, 2160), '8k': (7680, 4320)}
    if mode == 'scale':
        scale = settings.get('vsr_scale', 2)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 1 < scale <= 4:
            raise ValueError('RTX VSR scale must be greater than 1 and at most 4')
        result = tuple(int(math.floor(value * scale / 2 + .5)) * 2 for value in size)
    elif mode == 'resolution':
        target = str(settings.get('vsr_target', '1080p'))
        if target == 'custom':
            result = (settings.get('vsr_width', 1920), settings.get('vsr_height', 1080))
        elif target in choices:
            result = choices[target]
        else:
            raise ValueError('Unknown RTX VSR target')
    else:
        raise ValueError('RTX VSR output mode must be scale or resolution')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 64 or v > 16384 or v % 2 for v in result):
        raise ValueError('RTX VSR output dimensions must be even integers in 64..16384')
    if size[0] >= result[0] or size[1] >= result[1]:
        raise ValueError('RTX VSR target must enlarge both source dimensions')
    if result[0] > size[0] * 4 or result[1] > size[1] * 4:
        raise ValueError('RTX VSR supports at most 4x enlargement per axis')
    if abs(size[0] / size[1] - result[0] / result[1]) > .001:
        raise ValueError('RTX VSR currently requires matching source/output aspect ratios')
    return result


def processing_size(settings, size):
    """Extent entering NR (or passthrough when NR is absent)."""
    ordered = stages(settings)
    if 'nr' in ordered and 'vsr' in ordered[:ordered.index('nr')]:
        return output_size(settings, size)
    return tuple(size)


def frame_at_size(frame, size):
    clone = type(frame).from_buffer_copy(frame)
    clone.width, clone.height = size
    return clone


def make_live(width, height, settings, *, wide=False):
    ordered = stages(settings)
    if wide and ordered == ['dlaa']:
        from dlaa_gpu_engine import Live
        return Live(width, height, settings, dll_path=settings.get('dlss_dll_path'), is_hdr=True)
    if (settings.get('render_pipeline') is not None or 'vfg' in ordered) and 'nr' not in ordered:
        owner = None
        if 'dlaa' in ordered:
            from dlaa_gpu_engine import Live
            size = output_size(settings, (width, height)) if 'vsr' in ordered[:ordered.index('dlaa')] else (width, height)
            owner = Live(*size, settings, dll_path=settings.get('dlss_dll_path'), is_hdr=wide)
        try:
            return Passthrough(width, height, settings, owner=owner, wide=wide)
        except Exception:
            if owner:
                owner.close()
            raise
    selected = mode(settings)
    if selected == 'vsr':
        return Passthrough(width, height, settings, wide=wide)
    if selected == 'dlaa':
        from dlaa_gpu_engine import Live
        return Live(width, height, settings, dll_path=settings.get('dlss_dll_path'), is_hdr=wide)
    from dlss_gpu_engine import Live
    return Live(width, height, settings, color_format='rgba16f' if wide else 'rgba8')


def make_flow_provider(settings, live, *, raft_shared=None):
    if int(settings.get('guidance_mode', 0)) == 0:
        return None
    if settings.get('flow_engine') == 'fidelityfx':
        from fidelityfx_flow_engine import FlowProvider
        return FlowProvider(live.d3d12_device(), settings.get('flow_quality', 'Quality'))
    if settings.get('flow_engine') in ('raft', 'raft_large'):
        from raft_gpu_flow import FlowProvider
        return FlowProvider(settings, raft_shared)
    from optical_flow import FrameGuidanceService
    from pipeline import GuidanceServiceFlowProvider
    service = FrameGuidanceService(settings.get('flow_engine', 'nvof'),
        settings.get('flow_quality', '1F'), dedicated=True)
    return GuidanceServiceFlowProvider(service, 1)


def _library():
    bundle_paths.add_dll_directory(bundle_paths.native_dir())
    lib = C.CDLL(bundle_paths.require_native('gpu_rgba_exchange.dll', 'native effect GPU exchange'))
    lib.gpu_rgba_exchange_last_error.restype = C.c_char_p
    lib.gpu_effect_device_open.argtypes = [C.POINTER(C.c_void_p)]
    lib.gpu_effect_device_close.argtypes = [C.c_void_p]
    lib.gpu_rgba_exchange_open.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32,
                                          C.c_uint32, C.POINTER(C.c_void_p)]
    lib.gpu_rgba_exchange_close.argtypes = [C.c_void_p]
    lib.gpu_rgba_exchange_import.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
        C.c_uint64, C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)]
    lib.gpu_rgba_exchange_export.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t,
                                           C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]
    lib.gpu_rgba_exchange_pointer.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)]
    lib.gpu_rgba_exchange_fence.argtypes = [C.c_void_p]
    lib.gpu_rgba_exchange_fence.restype = C.c_void_p
    return lib


def _check(lib, rc):
    if rc:
        raise RuntimeError(lib.gpu_rgba_exchange_last_error().decode('utf-8', 'replace'))


class Exchange:
    def __init__(self, device, context, size, *, wide=False):
        self.lib = _library()
        self.device, self.context, self.size = device, context, tuple(size)
        self.wide = wide
        self.handle = C.c_void_p()
        self.device_copy_bytes = 0
        if wide:
            if not hasattr(self.lib, 'gpu_rgba_exchange_open2'):
                raise RuntimeError('FP16 exchange requires the updated native bridge')
            self.lib.gpu_rgba_exchange_open2.argtypes = [C.c_void_p, C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p)]
            _check(self.lib, self.lib.gpu_rgba_exchange_open2(device, context, *self.size, 8, C.byref(self.handle)))
        else:
            _check(self.lib, self.lib.gpu_rgba_exchange_open(device, context, *self.size, C.byref(self.handle)))

    def pointer(self):
        ptr, pitch = C.c_uint64(), C.c_uint32()
        _check(self.lib, self.lib.gpu_rgba_exchange_pointer(self.handle, C.byref(ptr), C.byref(pitch)))
        return ptr.value, pitch.value

    def color(self, texture):
        ptr, pitch = C.c_uint64(), C.c_uint32()
        _check(self.lib, self.lib.gpu_rgba_exchange_import(self.handle, texture['d3d12_resource'],
            texture.get('d3d12_fence'), texture.get('fence_value', 0), C.byref(ptr), C.byref(pitch)))
        self.device_copy_bytes += self.size[0] * self.size[1] * (8 if self.wide else 4)
        return dict(format='rgba16f_cuda' if self.wide else 'rgba8_cuda', width=self.size[0], height=self.size[1],
                    cuda_context=self.context, device_ptr=ptr.value, pitch_bytes=pitch.value)

    def texture(self, ptr, pitch):
        texture, value = C.c_void_p(), C.c_uint64()
        _check(self.lib, self.lib.gpu_rgba_exchange_export(self.handle, ptr, pitch,
                                                        C.byref(texture), C.byref(value)))
        self.device_copy_bytes += self.size[0] * self.size[1] * (8 if self.wide else 4)
        return dict(d3d12_device=self.device, d3d12_resource=texture.value,
                    d3d12_fence=self.lib.gpu_rgba_exchange_fence(self.handle), fence_value=value.value,
                    width=self.size[0], height=self.size[1],
                    dxgi_format=10 if self.wide else 28, state=0x800, producer_completed=True)

    def close(self):
        if self.handle:
            self.lib.gpu_rgba_exchange_close(self.handle)
            self.handle = None


class Passthrough:
    """GPU resource owner for standalone VSR. Opens no NGX feature."""
    color_format, bytes_per_pixel, session_id = 'rgba8', 4, 0
    native_passthrough = True
    host_dll, host_abi, last_error = 'gpu_rgba_exchange.dll', 1, None
    def __init__(self, width, height, settings, owner=None, *, wide=False):
        self._w, self._h, self.settings = width, height, dict(settings)
        self.color_format, self.bytes_per_pixel = ('rgba16f', 8) if wide else ('rgba8', 4)
        self.lib, self.device, self.exchange = _library(), C.c_void_p(), None
        self.owner = owner
        if owner:
            self.device = C.c_void_p(owner.d3d12_device())
        else:
            _check(self.lib, self.lib.gpu_effect_device_open(C.byref(self.device)))

    def d3d12_device(self):
        return self.device.value

    def is_alive(self):
        return bool(self.device)

    def supports_zero_motion(self):
        return True

    def process_device_color_zero_motion_resource(self, color, reset=False):
        if not self.exchange:
            self.exchange = Exchange(self.device.value, color['cuda_context'], (self._w, self._h), wide=self.bytes_per_pixel == 8)
        return self.exchange.texture(color['device_ptr'], color['pitch_bytes'])

    def transfer_counters(self):
        return dict(host_color_stage_count=0, host_color_stage_bytes=0,
                    output_readback_count=0, output_readback_bytes=0,
                    device_color_copy_count=None, device_color_copy_bytes=None,
                    vendor_feature='none; ordered model stages run separately')

    def close(self):
        if self.exchange:
            self.exchange.close(); self.exchange = None
        if self.device:
            if self.owner:
                self.owner.close(); self.owner = None
            else:
                self.lib.gpu_effect_device_close(self.device)
            self.device = None


class OutputEffect:
    """Runs only after the final NR composite; returns a borrowed GPU texture."""
    def __init__(self, live, settings, source_size, flow_provider=None, *, kind=None):
        self.live, self.settings, self.source_size = live, dict(settings), tuple(source_size)
        self.selected = ('nr_dlaa' if kind == 'dlaa' else 'vsr') if kind else mode(settings)
        self.target = self.source_size if self.selected == 'nr_dlaa' else output_size(settings, source_size)
        self.flow_provider = flow_provider
        self.input = self.output = self.effect = self.guidance = None
        self.calls, self.resets = 0, []
        self.owns_effect = True
        self.wide = getattr(live, 'bytes_per_pixel', 4) == 8
        self.guide_bridge = None
        self.input_codec = self.output_codec = None

    def process(self, candidate, frame, reset):
        if self.input is None:
            self.input = Exchange(candidate['d3d12_device'], frame.cuda_context, self.source_size, wide=self.wide)
            if self.selected == 'nr_dlaa':
                from dlaa_gpu_engine import Live
                owner = getattr(self.live, 'owner', None)
                if owner:
                    self.effect, self.owns_effect = owner, False
                else:
                    self.effect = Live(*self.source_size, self.settings,
                        dll_path=self.settings.get('dlss_dll_path'), shared_with=self.live, is_hdr=self.wide)
                if int(self.settings.get('guidance_mode', 0)):
                    if not callable(getattr(self.flow_provider, 'new_independent_layer', None)):
                        raise ValueError('DLAA guidance requires an independent device-flow history')
                    self.guidance = self.flow_provider.new_independent_layer()
            else:
                from rtx_vsr_engine import RtxVsr
                from native_runtime_paths import vsr_runtime
                runtime = vsr_runtime(self.settings.get('vsr_runtime_path'))
                self.runtime = runtime
                self.output = Exchange(candidate['d3d12_device'], frame.cuda_context, self.target, wide=self.wide)
                if self.wide:
                    from native_runtime_paths import wide_vfx_runtime
                    from vfx_rgb10 import Codec
                    wide_vfx_runtime(runtime)
                    hdr_picture = int(getattr(frame, 'color_transfer', 0)) == 16
                    self.input_codec = Codec(frame.cuda_context, self.source_size, runtime, hdr=hdr_picture)
                    self.output_codec = Codec(frame.cuda_context, self.target, runtime, hdr=hdr_picture)
                self.effect = RtxVsr(bundle_paths.require_native('rtx_vsr_bridge.dll', 'RTX VSR'),
                    runtime, frame.cuda_context, self.source_size, self.target,
                    int(self.settings.get('vsr_quality', 2)), packed10=self.wide)
        color = self.input.color(candidate)
        if self.selected == 'nr_dlaa':
            if self.guidance:
                guide_color = color
                if self.wide:
                    # Keep the optional wide ABI behind the existing bundle's
                    # explicit hidden import, as the primary HDR chain does.
                    hdr = __import__('hdr_gpu_engine')
                    if self.guide_bridge is None:
                        self.guide_bridge = hdr.GpuHdrBridge(d3d12_device=candidate['d3d12_device'],
                            cuda_context=frame.cuda_context, full_width=self.source_size[0], full_height=self.source_size[1],
                            working_width=self.source_size[0], working_height=self.source_size[1],
                            paper_white_scrgb=float(self.settings.get('hdr_paper_white_scrgb', 8)))
                    self.guide_bridge.import_master(color)
                    self.guide_bridge.downscale()
                    _, guide = self.guide_bridge.proxy()
                    guide_color = guide.as_descriptor()
                flow = self.guidance.get_flow_device_rgba(self.calls, guide_color, frame.cuda_context,
                                                          is_discontinuous=reset)
                result = self.effect.process_device_color_flow_resource(color, flow, reset=reset)
            else:
                result = self.effect.process_device_color_zero_motion_resource(color, reset=reset)
        else:
            ptr, pitch = self.output.pointer()
            if self.wide:
                packed, row = self.input_codec.pack(color['device_ptr'], color['pitch_bytes'])
                target, target_row = self.output_codec.pointer()
                self.effect.process(packed, row, target, target_row)
                self.output_codec.unpack(target, target_row, ptr, pitch)
            else:
                self.effect.process(color['device_ptr'], color['pitch_bytes'], ptr, pitch)
            result = self.output.texture(ptr, pitch)
        self.calls += 1
        self.resets.append(bool(reset))
        self.resets = self.resets[-16:]
        return result

    def report(self):
        return dict(mode=self.selected, input_size=self.source_size, output_size=self.target,
                    stage='dlaa' if self.selected == 'nr_dlaa' else 'vsr',
                    preset_requested=self.settings.get('dlaa_preset', 0) if self.selected == 'nr_dlaa' else None,
                    preset_effective=None,
                    runtime=(getattr(self.effect, 'dlss_dll', None) if self.selected == 'nr_dlaa'
                             else getattr(self, 'runtime', None)),
                    calls=self.calls, recent_resets=self.resets,
                    counters=self.effect.counters() if self.effect else None,
                    host_image_bytes=0, scalar_readback_bytes=None,
                    processing_format='rgba16f' if self.wide else 'rgba8',
                    motion_proxy_only=bool(self.guide_bridge),
                    motion_proxy_counters=self.guide_bridge.transfer_counters() if self.guide_bridge else None,
                    vfx_encoding='rgb10a2' if self.input_codec else None,
                    vfx_colour_counters=self.input_codec.counters() if self.input_codec else None,
                    limitation=('VSR quality selects SDK processing strength, not a sharpening amount'
                        if self.selected != 'nr_dlaa' else
                        'DLAA uses video flow and zero depth/jitter; requested model may be overridden'))

    def close(self):
        for item in (self.effect if self.owns_effect else None, self.guidance, self.guide_bridge,
                     self.output_codec, self.input_codec, self.output, self.input):
            if item:
                closer = getattr(item, 'close_independent_layer', None) or item.close
                closer()
        self.effect = self.guidance = self.output = self.input = None


class EffectSeries:
    """Ordered borrowed GPU textures. Owns stages, never the shared device owner."""
    def __init__(self, live, settings, source_size, ordered, provider):
        self.effects, self.order = [], list(ordered)
        size = tuple(source_size)
        for kind in ordered:
            effect = OutputEffect(live, settings, size, provider, kind=kind)
            self.effects.append(effect)
            size = effect.target
        self.target = size

    def process(self, candidate, frame, reset):
        for effect in self.effects:
            candidate = effect.process(candidate, frame, reset)
        return candidate

    def color(self, color, frame, reset):
        # Enter from NV12 conversion, and exit back to a CUDA RGBA plane for NR.
        if not hasattr(self, 'entry'):
            self.entry = Exchange(self.effects[0].live.d3d12_device(), frame.cuda_context,
                                  (color['width'], color['height']), wide=self.effects[0].wide)
        result = self.process(self.entry.texture(color['device_ptr'], color['pitch_bytes']), frame, reset)
        if not hasattr(self, 'exit'):
            self.exit = Exchange(result['d3d12_device'], frame.cuda_context, self.target, wide=self.effects[-1].wide)
        return self.exit.color(result)

    def report(self):
        return dict(order=self.order, output_size=self.target,
                    stages=[effect.report() for effect in self.effects], host_image_bytes=0)

    def close(self):
        for item in (getattr(self, 'exit', None), *reversed(self.effects), getattr(self, 'entry', None)):
            if item:
                item.close()


def install(chain, live, settings, source_size, provider):
    from vfx_framegen import config
    multiplier, _ = config(settings)
    if multiplier != 1 and getattr(chain, 'preview_consumer', None) is None:
        from native_runtime_paths import wide_vfx_runtime
        wide_vfx_runtime(settings.get('vsr_runtime_path'), framegen=True)
        chain.framegen_settings = dict(settings)
    ordered = stages(settings)
    active_vfg = multiplier != 1 and getattr(chain, 'preview_consumer', None) is None
    spatial = [s for s in ordered if s != 'vfg']
    if active_vfg:
        from ordered_vfg import install_ordered_vfg
        spatial = install_ordered_vfg(chain, live, settings, source_size, provider, ordered)
    if 'vsr' in ordered and getattr(live, 'bytes_per_pixel', 4) == 8:
        from native_runtime_paths import wide_vfx_runtime
        wide_vfx_runtime(settings.get('vsr_runtime_path'))
    # A legacy standalone DLAA remains the core itself, preserving saved sessions.
    if ordered == ['dlaa'] and (settings.get('render_pipeline') is None or getattr(live, 'bytes_per_pixel', 4) == 8):
        return
    split = spatial.index('nr') if 'nr' in spatial else 0
    before = spatial[:split] if 'nr' in spatial else []
    after = spatial[split + 1:] if 'nr' in spatial else spatial
    chain.processing_size = (getattr(chain, 'ordered_input_size', source_size)
        if active_vfg and 'nr' not in spatial else processing_size(settings, source_size))
    chain.render_order = ordered
    if before:
        chain.input_effect = EffectSeries(live, settings, getattr(chain, 'ordered_input_size', source_size), before, provider)
    if after:
        chain.output_effect = EffectSeries(live, settings, chain.processing_size, after, provider)
