"""Display adapter for the existing GPU export frame processors.

One dedicated thread owns decoder/CUDA/NGX lifetime. Only the finished composite
and source display image are read back, explicitly counted; no encoder or remux
is started. Processing precision is independent of the final 8-bit JPEG display.
"""
import ctypes as C
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import bundle_paths
import dlss_gpu_engine as dge
import gpu_transfer_export as gtx
import gpu_hdr_transfer_export as htx
import pipeline


def display_linear(rgba, hdr=False, display_edge=0):
    # Display resampling is independent from full-resolution neural processing.
    # Average linear radiance before the display transfer, never the HDR master.
    if display_edge and max(rgba.shape[:2]) > display_edge:
        scale = display_edge / max(rgba.shape[:2])
        rgba = cv2.resize(rgba.astype(np.float32),
            (max(1, round(rgba.shape[1]*scale)), max(1, round(rgba.shape[0]*scale))),
            interpolation=cv2.INTER_AREA)
    if not hdr and rgba.dtype == np.float16:
        # Exact FP16 code -> SDR display lookup. Avoid two full FP32 planes and
        # a power evaluation per pixel; this never modifies the HDR master.
        return np.ascontiguousarray(_SDR_HALF_DISPLAY[rgba[..., 2::-1].view(np.uint16)])
    rgb = np.asarray(rgba[..., :3], dtype=np.float32)
    if hdr:
        # Fixed display-only luminance mapping for every frame and both views.
        # scRGB is linear Rec.709, white=80 nits. Never feeds processing/export.
        luminance = np.maximum(rgb @ np.array([.2126, .7152, .0722]), 0)
        rgb = rgb / (1 + luminance[..., None])
    rgb = np.clip(rgb, 0, 1)
    encoded = np.where(rgb < .018, 4.5 * rgb, 1.099 * rgb ** .45 - .099)
    return np.ascontiguousarray(np.rint(encoded[..., ::-1] * 255), dtype=np.uint8)


with np.errstate(invalid='ignore', over='ignore'):
    _half = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    _half = np.nan_to_num(np.clip(_half, 0, 1), nan=0)
    _SDR_HALF_DISPLAY = np.rint(np.where(_half < .018, 4.5 * _half,
        1.099 * _half ** .45 - .099) * 255).astype(np.uint8)
del _half


class TextureReader:
    def __init__(self, device, width, height, fmt=28):
        self.lib = C.CDLL(bundle_paths.require_native('gpu_preview_readback.dll',
                                                       'display GPU preview'))
        self.lib.gpu_preview_open.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32,
                                             C.c_uint32, C.POINTER(C.c_void_p)]
        self.lib.gpu_preview_read.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p,
                                              C.c_uint64, C.c_void_p, C.c_size_t]
        self.lib.gpu_preview_close.argtypes = [C.c_void_p]
        self.lib.gpu_preview_last_error.restype = C.c_char_p
        self.handle = C.c_void_p()
        self.image = np.empty((height, width, 4), dtype=np.uint8 if fmt == 28 else np.float16)
        self._check(self.lib.gpu_preview_open(device, width, height, fmt, C.byref(self.handle)))
        self.bytes = 0

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.gpu_preview_last_error().decode('utf-8', 'replace'))

    def read(self, candidate):
        if candidate.get('state', 0x800) != 0x800:
            raise RuntimeError('preview consumer requires COPY_SOURCE')
        self._check(self.lib.gpu_preview_read(self.handle, candidate['d3d12_resource'],
            candidate.get('d3d12_fence'), candidate.get('fence_value', 0),
            self.image.ctypes.data, self.image.strides[0]))
        self.bytes += self.image.nbytes
        return self.image  # Borrowed until next read; cvtColor below owns its output.

    def close(self):
        if self.handle:
            self.lib.gpu_preview_close(self.handle)
            self.handle = None


class _Copy2D(C.Structure):
    _fields_ = [('srcXInBytes', C.c_size_t), ('srcY', C.c_size_t),
        ('srcMemoryType', C.c_uint), ('srcHost', C.c_void_p), ('srcDevice', C.c_uint64),
        ('srcArray', C.c_void_p), ('srcPitch', C.c_size_t), ('dstXInBytes', C.c_size_t),
        ('dstY', C.c_size_t), ('dstMemoryType', C.c_uint), ('dstHost', C.c_void_p),
        ('dstDevice', C.c_uint64), ('dstArray', C.c_void_p), ('dstPitch', C.c_size_t),
        ('WidthInBytes', C.c_size_t), ('Height', C.c_size_t)]


class ViewportReader:
    def __init__(self, device, context, width, height, edge, pixel_bytes):
        bundle_paths.add_dll_directory(bundle_paths.native_dir())
        self.lib = C.CDLL(bundle_paths.require_native('gpu_preview_display.dll', 'GPU preview display'))
        self.lib.gpu_preview_display_open.argtypes = [C.c_void_p, C.c_void_p] + [C.c_uint32]*5 + [C.POINTER(C.c_void_p)]
        self.lib.gpu_preview_display_read.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.c_uint32, C.c_void_p, C.c_size_t]
        self.lib.gpu_preview_display_close.argtypes = [C.c_void_p]
        self.lib.gpu_preview_display_error.restype = C.c_char_p
        scale = min(1., edge/max(width, height))
        w, h = max(1, round(width*scale)), max(1, round(height*scale))
        self.image = np.empty((h, w, 4), dtype=np.uint8)
        self.handle = C.c_void_p()
        self.bytes = 0
        self._check(self.lib.gpu_preview_display_open(device, context, width, height,
            w, h, pixel_bytes, C.byref(self.handle)))

    def _check(self, code):
        if code:
            raise RuntimeError(self.lib.gpu_preview_display_error().decode('utf8', 'replace'))

    def read(self, descriptor, hdr):
        self._check(self.lib.gpu_preview_display_read(self.handle, descriptor['device_ptr'],
            descriptor['pitch_bytes'], int(hdr), self.image.ctypes.data, self.image.strides[0]))
        self.bytes += self.image.nbytes
        return cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)

    def close(self):
        if self.handle:
            self.lib.gpu_preview_display_close(self.handle)
            self.handle = None


class GpuPreview:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='GPUPreview')
        self.chain = None
        self.live = None
        self.provider = None
        self.bridges = None
        self.reader = None
        self.key = None
        self.last_index = None
        self.current_frame = None
        self.diag = {}
        self.display_bytes = 0
        self.generation = 0
        self.pending_layers = []
        self.display_buffers = {}
        self.cached_display = None
        self.options = {}
        self.active_settings = None
        self.display_edge = 0
        self.cached_edge = None
        self.viewport_readers = {}
        self.display_exchange = None
        # One bounded weight cache on the preview owner thread. Histories and
        # native sessions still reset independently when source/mode changes.
        self.raft_shared = {}

    def render(self, video, settings, frame_index, *, need_source=True, reuse_display=False, display_edge=0, flow_view=False):
        return self.executor.submit(self._render, video, dict(settings), int(frame_index),
                                    need_source, reuse_display, display_edge, flow_view).result()

    def close(self):
        self.executor.submit(self._close).result()

    def shutdown(self):
        self.close()
        self.executor.submit(self.raft_shared.clear).result()
        self.executor.shutdown(wait=True)

    def _close(self):
        # Display reader borrows the processing device and closes before owners.
        for _, reader in self.viewport_readers.values():
            reader.close()
        self.viewport_readers.clear()
        if self.display_exchange:
            self.display_exchange.close()
            self.display_exchange = None
        if self.reader:
            self.reader.close()
            self.reader = None
        try:
            if self.chain:
                self.chain.close()
            else:
                for layer in reversed(self.pending_layers):
                    if layer.get('flow_provider'):
                        layer['flow_provider'].close_independent_layer()
                    if layer.get('live'):
                        layer['live'].close()
                if self.live:
                    self.live.close()
        finally:
            self.pending_layers = []
            self.chain = self.live = None
            if self.provider:
                self.provider.close_independent_layer()
                self.provider = None
            if self.bridges:
                htx.release_loader_cookies(self.bridges)
                self.bridges = None
            self.key = self.last_index = self.current_frame = None
            self.display_buffers.clear()
            self.cached_display = None
            self.active_settings = None

    def _open(self, video, settings):
        # Frame generation is an export setting; it must not load VFG or reject
        # an otherwise valid preview when its target rate is being edited.
        settings = dict(settings, frame_generation='off')
        source = gtx.probe_source(video)
        import native_effects
        native_effects.validate(settings, source)
        width, height = source['width'], source['height']
        process_w, process_h = native_effects.processing_size(settings, (width, height))
        self.multi = gtx.multi_scale_config(settings, process_w, process_h)
        uses_flow = gtx.guidance_requires_flow(settings)
        self.bridges = gtx.load_bridges(requires_flow=uses_flow,
            requires_multiscale=len(self.multi['scales']) > 1)
        self.hdr = source.get('color_transfer') == 'smpte2084'
        # Precision selects the decoder path; colour admission belongs to that
        # path's gate. Never send a 10-bit source to the NV12-only gate.
        self.sdr10 = source.get('pix_fmt') == 'yuv420p10le' and not self.hdr
        if source.get('color_transfer') == 'arib-std-b67':
            raise ValueError('HLG GPU preview is not validated; no SDR conversion fallback')
        wide = self.hdr or self.sdr10
        self.live = native_effects.make_live(*self.multi['dims'][0], settings, wide=wide)
        uses_flow = uses_flow and not isinstance(self.live, native_effects.Passthrough)
        self.provider = native_effects.make_flow_provider(settings, self.live,
            **({'raft_shared': self.raft_shared} if settings.get('flow_engine') in ('raft', 'raft_large') else {}))
        target = native_effects.output_size(settings, (width, height))
        dims = dict(src_w=width, src_h=height, live_w=self.multi['dims'][0][0],
            live_h=self.multi['dims'][0][1], dst_w=target[0], dst_h=target[1])
        local = dict(settings, dlss_backend='gpu_flow_d3d12', export_backend='gpu_transfer')
        tx = dict(codec='hevc_nvenc', container='mkv', color_preset='preserve')
        if wide:
            tx.update(profile='main10', pixel_format='p010le')
            caps = (htx.load_sdr10_capabilities if self.sdr10 else htx.load_hdr_capabilities)(
                self.bridges, width, height)
            local['experimental_hdr_scrgb'] = self.hdr
            local['hdr_paper_white_scrgb'] = settings.get('hdr_paper_white_scrgb', 1.0)
            negotiate = htx.negotiate_sdr10 if self.sdr10 else htx.negotiate_hdr
            negotiation = negotiate(dims, source, local, tx, 24, False,
                'preview-display.mkv', self.live, hdr_caps=caps,
                option_available=self.bridges['option_available'], flow_provider=self.provider)
        else:
            negotiation = gtx.negotiate(dims, source, local, tx, 24, False,
                'preview-display.mkv', self.live,
                option_available=self.bridges['option_available'], flow_provider=self.provider,
                capabilities=self.bridges['capabilities'])
        if not negotiation['supported']:
            raise ValueError('GPU preview refused: ' + '; '.join(negotiation['reasons']))
        strength = float(settings.get('residual_strength', 1))
        if wide:
            extra = htx._open_extra_layers(self.multi, self.live, self.provider,
                                            uses_flow, process_w, process_h)
            self.pending_layers = extra
            args = (self.bridges, caps, video, None, self.live, negotiation['plan'],
                    negotiation['fps'], negotiation['hdr_policy'], dims)
            kwargs = dict(flow_provider=self.provider, residual_strength=strength,
                          uses_flow=uses_flow, preview_consumer=self._consume_linear)
            if extra:
                self.chain = htx.HdrMultiScaleChain(*args, **kwargs, multi=self.multi,
                    extra_layers=extra, fusion_plans=[gtx.TapPlans(process_w, process_h, *d)
                                                       for d in self.multi['dims']])
            else:
                self.chain = htx.HdrChain(*args, **kwargs)
        else:
            extra = []
            self.pending_layers = extra
            for scale, d in zip(self.multi['scales'][1:], self.multi['dims'][1:]):
                layer = dict(dims=d, scale=scale, live=None,
                    flow_provider=self.provider.new_independent_layer() if uses_flow else None,
                    lanczos=None, low_residual=None, tap_plans=None)
                extra.append(layer)
                layer['live'] = dge.Live(*d, settings, shared_with=self.live)
                layer['tap_plans'] = gtx.TapPlans(process_w, process_h, *d)
            args = (self.bridges, video, None, self.live, negotiation['plan'],
                    negotiation['fps'], strength, self.multi['dims'][0], self.provider)
            kwargs = dict(downscaled=negotiation['downscaled'], uses_flow=uses_flow,
                tap_plans=gtx.TapPlans(process_w, process_h, *self.multi['dims'][0])
                    if negotiation['plan_driven'] else None,
                preview_consumer=self._consume_texture)
            if extra:
                self.chain = gtx.MultiScaleObservationChain(*args, **kwargs,
                    extra_layers=extra, config=self.multi,
                    fusion_plans=[gtx.TapPlans(process_w, process_h, *d) for d in self.multi['dims']])
            else:
                self.chain = gtx.Chain(*args, **kwargs)
        from post_control import GpuPostControl, controls_from_settings
        self.pending_layers = []  # The constructed chain now owns every layer.
        from temporal_control import GpuTemporalControl, temporal_from_settings
        self.chain.post_control = GpuPostControl(controls_from_settings(settings), hdr=wide)
        self.chain.temporal_control = GpuTemporalControl(temporal_from_settings(settings), hdr=wide)
        native_effects.install(self.chain, self.live, settings, (width, height), self.provider)
        self.chain.open_decoder()
        lib = self.bridges['nvdec']
        lib.nvdec_bridge_seek_us.argtypes = [C.c_void_p, C.c_int64]
        lib.nvdec_bridge_seek_us.restype = C.c_int32
        self.source = source
        # NVDEC reports absolute PTS; the UI clock and seek ABI are source-relative.
        import subprocess
        raw = subprocess.check_output([pipeline._bundle_ffprobe(), '-v', 'error',
            '-select_streams', 'v:0', '-read_intervals', '%+#1', '-show_frames',
            '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', video])
        frames = json.loads(raw).get('frames', [])
        self.source_origin = float(frames[0]['best_effort_timestamp_time']) if frames else 0.0
        self.fps = negotiation['fps'][0] / negotiation['fps'][1]
        self.generation += 1
        self.display_bytes = 0

    def _consume_texture(self, candidate, frame):
        mark = time.perf_counter()
        if self.display_edge:
            import native_effects
            if self.display_exchange is None:
                self.display_exchange = native_effects.Exchange(candidate['d3d12_device'], frame.cuda_context,
                    (candidate['width'], candidate['height']))
            color = self.display_exchange.color(candidate)
            self.result_bgr = self._viewport('processed', color, candidate['d3d12_device'], False)
            self.processed_display_ms = (time.perf_counter()-mark)*1000
            return
        if self.reader is None:
            self.reader = TextureReader(candidate['d3d12_device'], candidate['width'], candidate['height'])
        raw = self.reader.read(candidate)
        self.frame_readback_ms += (time.perf_counter()-mark)*1000
        self.result_bgr = cv2.cvtColor(raw[..., :3], cv2.COLOR_RGB2BGR)
        self.processed_display_ms = (time.perf_counter()-mark)*1000

    def _update_runtime(self, settings):
        runtime = {'style', 'intensity', 'local_tone', 'local_struct', 'skin_struct',
                   'use_auto_mask', 'ui_correction', 'motion_scale_x', 'motion_scale_y',
                   'post_controls', 'temporal_controls'}
        import native_effects
        if self.active_settings is None or native_effects.stages(settings) != ['nr']:
            return False
        old = {k: v for k, v in self.active_settings.items() if k not in runtime}
        new = {k: v for k, v in settings.items() if k not in runtime}
        if old != new:
            return False
        from post_control import controls_from_settings
        from temporal_control import temporal_from_settings
        post, temporal = controls_from_settings(settings), temporal_from_settings(settings)
        owners = [self.live] + [layer['live'] for layer in getattr(self.chain, 'extra_layers', [])]
        if any(not hasattr(owner, 'update_runtime_options') for owner in owners):
            return False
        for owner in owners:
            if not owner.update_runtime_options(settings):
                return False
        if hasattr(self.chain, 'post_control'):
            self.chain.post_control.update_controls(post)
        if hasattr(self.chain, 'temporal_control'):
            self.chain.temporal_control.controls = temporal
        self.active_settings = dict(settings)
        self.cached_display = None
        return True

    def _linear_plane(self, plane):
        bridge = (getattr(self.chain, 'encode_bridge', None) if plane == 7
                  else getattr(self.chain, 'decode_bridge', None)) or self.chain.bridge
        lib = bridge._lib
        lib.gpu_hdr_readback_scrgb.argtypes = [C.c_void_p, C.c_int32, C.c_void_p, C.c_size_t]
        size = getattr(self.chain, 'output_size', None) if plane == 7 else None
        shape = (size[1], size[0], 4) if size else (self.source['height'], self.source['width'], 4)
        if plane not in self.display_buffers or self.display_buffers[plane].shape != shape:
            self.display_buffers[plane] = np.empty(shape, dtype=np.float16)
        image = self.display_buffers[plane]
        mark = time.perf_counter()
        code = lib.gpu_hdr_readback_scrgb(bridge._handle, plane,
                                         image.ctypes.data, image.strides[0])
        if code:
            raise RuntimeError('FP16 display readback failed')
        self.display_bytes += image.nbytes
        self.frame_readback_ms += (time.perf_counter() - mark) * 1000
        return image

    def _consume_linear(self, candidate, frame):
        mark = time.perf_counter()
        self.result_bgr = (self._viewport('processed', candidate, candidate['d3d12_device'], self.hdr)
            if self.display_edge else display_linear(self._linear_plane(7), self.hdr))
        self.processed_display_ms = (time.perf_counter() - mark) * 1000

    def _viewport(self, kind, descriptor, device, hdr):
        key = (device, descriptor['cuda_context'], descriptor['width'], descriptor['height'],
               self.display_edge, 8 if descriptor['format'] == 'rgba16f_cuda' else 4)
        previous = self.viewport_readers.get(kind)
        if previous is None or previous[0] != key:
            if previous:
                previous[1].close()
            reader = ViewportReader(*key)
            self.viewport_readers[kind] = key, reader
        reader = self.viewport_readers[kind][1]
        mark = time.perf_counter()
        result = reader.read(descriptor, hdr)
        self.frame_readback_ms += (time.perf_counter()-mark)*1000
        self.display_bytes += reader.image.nbytes
        return result

    def _source_bgr(self, outcome, frame):
        if self.display_edge:
            if self.hdr or self.sdr10:
                descriptor = outcome['_preview_master']
            else:
                ptr, pitch = outcome['_full_color']
                descriptor = dict(format='rgba8_cuda', cuda_context=frame.cuda_context,
                    width=frame.width, height=frame.height, device_ptr=ptr, pitch_bytes=pitch)
            return self._viewport('source', descriptor, self.live.d3d12_device(), self.hdr)
        if self.hdr or self.sdr10:
            return display_linear(self._linear_plane(2), self.hdr, self.display_edge)
        ptr, pitch = outcome['_full_color']
        image = np.empty((frame.height, frame.width, 4), dtype=np.uint8)
        cuda = C.WinDLL('nvcuda.dll')
        cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(_Copy2D)]
        if cuda.cuCtxPushCurrent_v2(frame.cuda_context):
            raise RuntimeError('preview source CUDA context push failed')
        try:
            copy = _Copy2D(srcMemoryType=2, srcDevice=ptr, srcPitch=pitch,
                dstMemoryType=1, dstHost=image.ctypes.data, dstPitch=image.strides[0],
                WidthInBytes=frame.width * 4, Height=frame.height)
            if cuda.cuMemcpy2D_v2(C.byref(copy)):
                raise RuntimeError('preview source display copy failed')
        finally:
            popped = C.c_void_p()
            cuda.cuCtxPopCurrent_v2(C.byref(popped))
        self.display_bytes += image.nbytes
        return cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR)

    def _flow_bgr(self, descriptor):
        """Diagnostic-only readback, on the native session owner thread.

        The float2 resource stays borrowed until this synchronous copy completes.
        Never create a second CPU-facing RAFT/NVOF engine for the flow view.
        """
        if not descriptor or descriptor.get('format') != 'float32x2':
            raise ValueError('Native preview did not return a float2 flow resource')
        width, height = int(descriptor['width']), int(descriptor['height'])
        image = np.empty((height, width, 2), dtype=np.float32)
        cuda = C.WinDLL('nvcuda.dll')
        cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        cuda.cuMemcpy2D_v2.argtypes = [C.POINTER(_Copy2D)]
        if cuda.cuCtxPushCurrent_v2(descriptor['cuda_context']):
            raise RuntimeError('flow diagnostic context push failed')
        try:
            copy = _Copy2D(srcMemoryType=2, srcDevice=descriptor['device_ptr'],
                srcPitch=descriptor['pitch_bytes'], dstMemoryType=1,
                dstHost=image.ctypes.data, dstPitch=image.strides[0],
                WidthInBytes=width*8, Height=height)
            if cuda.cuMemcpy2D_v2(C.byref(copy)):
                raise RuntimeError('flow diagnostic readback failed')
        finally:
            popped = C.c_void_p()
            cuda.cuCtxPopCurrent_v2(C.byref(popped))
        if not np.isfinite(image).all():
            raise ValueError('flow diagnostic contains nonfinite motion')
        return pipeline.colorize_flow(image), image.nbytes

    def _render(self, video, settings, index, need_source=True, reuse_display=False, display_edge=0, flow_view=False):
        if index < 0:
            raise ValueError('preview frame must be nonnegative')
        settings = {key:value for key,value in settings.items() if key not in (
            'frame_generation','frame_multiplier','framegen_quality','framegen_target_fps','framegen_scene_detection')}
        # A diagnostic flow view requires motion even with enhancement guidance
        # disabled; this local snapshot does not change the user's setting.
        if flow_view:
            settings = dict(settings, guidance_mode=1)
        key = (os.path.realpath(video), os.stat(video).st_mtime_ns,
               json.dumps(settings, sort_keys=True, default=str))
        start = time.perf_counter()
        self.frame_readback_ms = self.processed_display_ms = 0.0
        self.display_edge = max(0, int(display_edge))
        try:
            setup = time.perf_counter()
            runtime_changed = False
            if key != self.key:
                same_source = self.key is not None and self.key[:2] == key[:2]
                runtime_changed = same_source and self._update_runtime(settings)
                if not runtime_changed:
                    self._close()
                    self._open(video, settings)
                self.key = key
                self.active_settings = dict(settings)
            setup_ms = (time.perf_counter() - setup) * 1000
            if (reuse_display and self.cached_display is not None and self.last_index == index
                    and self.cached_edge == self.display_edge
                    and getattr(self, 'cached_flow_view', False) == flow_view
                    and (not need_source or self.cached_display[0] is not None)):
                original, result, cached = self.cached_display
                return original, result, dict(cached, cached_display=True,
                    request_ms=(time.perf_counter()-start)*1000)
            discontinuity = runtime_changed or self.last_index is None or index != self.last_index + 1
            # Repeat requests reuse the decode surface but re-evaluate with reset:
            # changed parameters rebuild, and no previous enhancement is returned.
            if self.last_index != index:
                decode_start = time.perf_counter()
                target = index / self.fps
                # Short forward gaps decode from the current surface. Only an
                # initial/backward/large jump needs GOP seek; temporal reset is
                # independent from decoder positioning.
                seek = ((self.last_index is None and index > 0) or
                        (self.last_index is not None and (index <= self.last_index or
                         index - self.last_index > max(8, int(self.fps * 2)))))
                if seek:
                    code = self.bridges['nvdec'].nvdec_bridge_seek_us(self.chain.decoder,
                                                                     round(target * 1e6))
                    if code:
                        gtx._bridge_error(self.bridges['nvdec'], 'nvdec_bridge_last_error', code)
                while True:
                    frame = gtx._DeviceFrame(struct_size=C.sizeof(gtx._DeviceFrame))
                    code = self.chain.next_frame(frame)
                    if code < 0:
                        gtx._bridge_error(self.bridges['nvdec'], 'nvdec_bridge_last_error', code)
                    if not code:
                        raise ValueError('preview reached end of source')
                    timestamp = frame.pts * frame.time_base_num / frame.time_base_den - getattr(self, 'source_origin', 0.0)
                    if timestamp + .5 / self.fps >= target:
                        self.current_frame = frame
                        break
                decode_ms = (time.perf_counter() - decode_start) * 1000
            else:
                decode_ms, seek = 0.0, False
            frame = self.current_frame
            if not (self.hdr or self.sdr10) and self.chain.nv12 is None:
                self.chain._open_color_stages(frame)
            process_start = time.perf_counter()
            outcome = self.chain.process(frame, self.provider, index, discontinuity)
            process_ms = (time.perf_counter()-process_start)*1000
            display_start = time.perf_counter()
            source_bgr = self._source_bgr(outcome, frame) if need_source else None
            source_display_ms = (time.perf_counter()-display_start)*1000
            flow_readback_bytes = 0
            if flow_view:
                self.result_bgr, flow_readback_bytes = self._flow_bgr(outcome.get('flow'))
            self.last_index = index
            elapsed = time.perf_counter() - start
            layers = (self.chain.layer_diagnostics() if hasattr(self.chain, 'layer_diagnostics')
                      else dict(layer_count=1, layer_session_ids=[self.live.session_id],
                                layer_working_dims=[list(self.multi['dims'][0])]))
            reports = getattr(self.chain, 'frame_reports', [])
            report = reports[-1] if reports else {}
            solver = (self.chain.fusion_reports[-1] if getattr(self.chain, 'fusion_reports', []) else None)
            self.diag = dict(frame=index, pts=int(frame.pts),
                time_base=[int(frame.time_base_num), int(frame.time_base_den)],
                source_resolution=[frame.width, frame.height],
                preview_resolution=[self.result_bgr.shape[1], self.result_bgr.shape[0]], quality_mode='full_resolution',
                colour_mode='hdr10' if self.hdr else 'sdr10_bt709' if self.sdr10 else 'sdr8',
                processing_format='rgba16f' if self.hdr or self.sdr10 else 'rgba8',
                source_colour_resolution=self.source.get('colour_resolution'),
                display_mode='fixed_HDR_to_SDR_non_calibrated' if self.hdr else 'SDR8_display',
                processing_ms=elapsed * 1000, processing_fps=1 / max(elapsed, 1e-9),
                generation=self.generation, reset=discontinuity,
                cached_display=False, decoder_seek=seek, setup_ms=setup_ms,
                decode_ms=decode_ms, process_with_display_ms=process_ms,
                readback_ms=self.frame_readback_ms,
                processed_display_ms=self.processed_display_ms,
                display_readback_scope='GPU display conversion/resampling plus viewport readback' if self.display_edge else 'full plane readback',
                source_display_ms=source_display_ms,
                flow_diagnostic_readback_bytes=flow_readback_bytes,
                flow_visualization=bool(flow_view),
                gpu_pipeline_completed_wall_ms=process_ms-self.processed_display_ms,
                timing_clock='CPU wall scopes around completed native work; not kernel timestamps',
                layer_count=len(self.multi['scales']), layers=layers,
                fusion_mode_requested=self.multi['fusion'],
                fusion_mode_actual=report.get('fusion_mode_actual') or
                    ('gpu_data_consistency_solve' if solver else 'catmull_baseline_layer_0'),
                solver=report.get('solver') or solver,
                display_readback_bytes=self.display_bytes + (self.reader.bytes if self.reader else 0),
                frame_report=report, stages=outcome['stages'])
            self.diag['neural_mode'] = settings.get('neural_mode', 'nr')
            self.diag['source_relative_time'] = frame.pts * frame.time_base_num / frame.time_base_den - getattr(self, 'source_origin', 0.0)
            self.diag['seek_clock'] = 'nominal frame index; actual decoded PTS reported'
            if self.diag['neural_mode'] in ('dlaa', 'vsr'):
                self.diag['fusion_mode_actual'] = 'native_single_effect'
            self.diag['motion'] = self.provider.timing() if self.provider else None
            if self.diag['neural_mode'] == 'dlaa':
                self.diag['dlaa_process_global_counters'] = self.live.counters()
            self.diag['native_output_effect'] = (self.chain.output_effect.report()
                if getattr(self.chain, 'output_effect', None) else None)
            self.diag['native_input_effect'] = (self.chain.input_effect.report()
                if getattr(self.chain, 'input_effect', None) else None)
            import native_effects
            self.diag['render_pipeline'] = native_effects.stages(settings)
            if getattr(self.chain, 'scene_observer', None):
                self.chain.scene_observer.observations = self.chain.scene_observer.observations[-16:]
            if 'nr' not in self.diag['render_pipeline']:
                self.diag['fusion_mode_actual'] = 'none; native model stages only'
            # Preview has no final export report. Keep recent diagnostics bounded
            # without changing per-layer counters or the metadata reference.
            for name in ('frame_reports', 'fusion_reports', 'metadata_snapshots'):
                values = getattr(self.chain, name, None)
                if values is not None and len(values) > 16:
                    del values[:-16]
            for values in getattr(self.chain, 'layer_reset_flags', []):
                if len(values) > 16:
                    del values[:-16]
            temporal = getattr(self.chain, 'temporal_control', None)
            for name in ('observations', 'process_reports'):
                values = getattr(temporal, name, None)
                if values is not None and len(values) > 16:
                    del values[:-16]
            self.cached_display = source_bgr, self.result_bgr, dict(self.diag)
            self.cached_edge = self.display_edge
            self.cached_flow_view = flow_view
            return source_bgr, self.result_bgr, dict(self.diag)
        except Exception:
            self._close()
            raise
