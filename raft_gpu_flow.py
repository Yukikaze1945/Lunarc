"""RAFT CUDA guidance for GPU Transfer; no host pixel or flow readback.

Torch owns its primary CUDA context. Decoder colour is copied with explicit
source/destination contexts, never reinterpreted as a foreign Torch pointer.
Independent layer histories share only immutable model weights; calls serialize
on the pipeline owner thread. Output lifetime extends through the native consumer.
"""
import ctypes as C
import time
from contextlib import contextmanager


class _PeerCopy(C.Structure):
    _fields_ = [(name, kind) for side in ('src', 'dst') for name, kind in (
        (side+'XInBytes', C.c_size_t), (side+'Y', C.c_size_t),
        (side+'Z', C.c_size_t), (side+'LOD', C.c_size_t),
        (side+'MemoryType', C.c_uint), (side+'Host', C.c_void_p),
        (side+'Device', C.c_uint64), (side+'Array', C.c_void_p),
        (side+'Context', C.c_void_p), (side+'Pitch', C.c_size_t),
        (side+'Height', C.c_size_t))] + [
        ('WidthInBytes', C.c_size_t), ('Height', C.c_size_t), ('Depth', C.c_size_t)]


def _check(code, label):
    if code:
        raise RuntimeError(f'{label}: CUDA driver error {code}; no host fallback')


class FlowProvider:
    name = 'RAFT-Large CUDA current-to-previous'
    borrows_decoder_context = False

    def __init__(self, settings, shared=None):
        self.settings = dict(settings)
        self.shared = shared if shared is not None else {}
        self.previous = self.output = self.input = None
        self.geometry = None
        self.context = None
        self.calls = self.copy_bytes = self.resets = 0
        self._timing = {}

    def new_independent_layer(self):
        return FlowProvider(self.settings, self.shared)

    def supports_device_rgba_flow(self):
        return True

    def _open(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('RAFT GPU Transfer requires CUDA-enabled PyTorch/torchvision; CPU fallback is disabled')
        self.cuda = C.WinDLL('nvcuda.dll')
        self.cuda.cuDevicePrimaryCtxRetain.argtypes = [C.POINTER(C.c_void_p), C.c_int]
        self.cuda.cuDevicePrimaryCtxRelease_v2.argtypes = [C.c_int]
        self.cuda.cuMemcpy3DPeer.argtypes = [C.POINTER(_PeerCopy)]
        self.cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
        self.cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
        self.cuda.cuCtxGetDevice.argtypes = [C.POINTER(C.c_int)]
        self.cuda.cuDeviceGetByPCIBusId.argtypes = [C.POINTER(C.c_int), C.c_char_p]
        # The decoder can already be current on this owner thread. Never let
        # Torch allocate in that borrowed context: its caching allocator would
        # retain allocations after the decoder destroys the context. Resolve
        # Torch's logical device by PCI identity, including CUDA_VISIBLE_DEVICES.
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        pci = f'{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0'
        ordinal = C.c_int()
        _check(self.cuda.cuDeviceGetByPCIBusId(C.byref(ordinal), pci.encode('ascii')), 'resolve Torch device')
        self.device_index = ordinal.value
        self.context = C.c_void_p()
        _check(self.cuda.cuDevicePrimaryCtxRetain(C.byref(self.context), self.device_index), 'retain Torch context')
        try:
            with self._torch_context():
                from optical_flow import RaftOpticalFlowEngine
                config = (self.settings.get('flow_quality', 'Balanced'),
                          self.settings.get('raft_analysis_edge'), self.settings.get('raft_iterations'))
                if 'model' not in self.shared or self.shared.get('config') != config:
                    previous = self.shared.get('model')
                    weights = (previous._raft_model if previous is not None and
                               self.shared.get('device_index') == self.device_index else None)
                    self.shared['model'] = RaftOpticalFlowEngine(self.settings.get('flow_quality', 'Balanced'),
                        self.settings.get('raft_analysis_edge'), self.settings.get('raft_iterations'),
                        shared_weights=weights)
                    self.shared['config'] = config
                    self.shared['device_index'] = self.device_index
                self.model = self.shared['model']
        except Exception:
            _check(self.cuda.cuDevicePrimaryCtxRelease_v2(self.device_index), 'release failed Torch initialization')
            self.context = None
            raise

    @contextmanager
    def _torch_context(self):
        _check(self.cuda.cuCtxPushCurrent_v2(self.context), 'enter Torch primary context')
        try:
            yield
        finally:
            popped = C.c_void_p()
            _check(self.cuda.cuCtxPopCurrent_v2(C.byref(popped)), 'restore native owner context')

    def get_flow_device_rgba(self, frame_idx, color, cuda_context, is_discontinuous=False):
        import torch
        import torch.nn.functional as functional
        from optical_flow import raft_analysis_size
        if color.get('format') != 'rgba8_cuda':
            raise ValueError('RAFT motion takes independent RGBA8 guidance, never the FP16 master')
        w, h = int(color['width']), int(color['height'])
        pitch = int(color['pitch_bytes'])
        if w <= 0 or h <= 0 or pitch < w*4 or not color.get('device_ptr') or not cuda_context:
            raise ValueError('Invalid RAFT CUDA colour descriptor')
        if self.context is None:
            self._open()
        with self._torch_context():
            return self._compute(frame_idx, color, cuda_context, is_discontinuous)

    def _compute(self, frame_idx, color, cuda_context, is_discontinuous):
        import torch
        import torch.nn.functional as functional
        from optical_flow import raft_analysis_size
        w, h = int(color['width']), int(color['height'])
        pitch = int(color['pitch_bytes'])
        _check(self.cuda.cuCtxPushCurrent_v2(cuda_context), 'inspect decode context')
        try:
            ordinal = C.c_int()
            _check(self.cuda.cuCtxGetDevice(C.byref(ordinal)), 'inspect decode device')
            if ordinal.value != self.device_index:
                raise ValueError('RAFT and decoder must use the same physical CUDA device')
        finally:
            popped = C.c_void_p()
            _check(self.cuda.cuCtxPopCurrent_v2(C.byref(popped)), 'restore CUDA context')
        started = time.perf_counter()
        size = raft_analysis_size(w, h, self.model.analysis_edge)
        geometry = (w, h, size)
        reset = is_discontinuous or geometry != self.geometry or self.previous is None
        if self.input is None or tuple(self.input.shape) != (h, w, 4):
            self.input = torch.empty((h, w, 4), dtype=torch.uint8, device='cuda')
        copy = _PeerCopy(srcMemoryType=2, srcDevice=int(color['device_ptr']),
            srcContext=cuda_context, srcPitch=pitch, srcHeight=h,
            dstMemoryType=2, dstDevice=self.input.data_ptr(), dstContext=self.context.value,
            dstPitch=w*4, dstHeight=h, WidthInBytes=w*4, Height=h, Depth=1)
        _check(self.cuda.cuMemcpy3DPeer(C.byref(copy)), 'RGBA guidance context-to-context copy')
        copied = time.perf_counter()
        current = self.input[..., :3].permute(2, 0, 1)[None].float()
        current = functional.interpolate(current, size=(size[1], size[0]),
                                          mode='bilinear', align_corners=False)
        current.div_(255).mul_(2).sub_(1)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        if reset:
            flow = torch.zeros((1, 2, size[1], size[0]), device='cuda')
            self.resets += 1
        else:
            flow = self.model._infer(current, self.previous, size)
        flow = functional.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
        flow[:, 0].mul_(w/size[0]); flow[:, 1].mul_(h/size[1])
        self.output = flow[0].permute(1, 2, 0).contiguous()
        end.record(); end.synchronize()  # Native consumer uses its own stream/context.
        self.previous, self.geometry = current, geometry
        self.calls += 1; self.copy_bytes += w*h*4
        self._timing = dict(guidance_backend='raft_cuda_device_rgba',
            analysis_dimensions=list(size), iterations=self.model.iterations,
            precision='float32', direction='current_to_previous', units='input_pixels',
            reset=bool(reset), previous_tensor_reused=not reset,
            flow_ms=(time.perf_counter()-started)*1000, input_device_copy_ms=(copied-started)*1000,
            inference_and_restore_gpu_ms=begin.elapsed_time(end),
            host_image_bytes_measured=0, host_flow_readback_bytes=0,
            input_device_copy_bytes=self.copy_bytes, reset_count=self.resets,
            synchronization='completed CUDA event before native D3D12 flow import')
        return dict(format='float32x2', cuda_context=self.context.value,
            device_ptr=self.output.data_ptr(), pitch_bytes=w*8, width=w, height=h,
            direction='current_to_previous', units='input_pixels')

    def device_input_counters(self):
        return self.calls, self.copy_bytes

    def timing(self):
        return dict(self._timing)

    def close_independent_layer(self):
        if self.context is not None:
            import torch
            with self._torch_context():
                torch.cuda.synchronize()
                self.previous = self.output = self.input = None
                self.model=None  # Immutable weights may still live in shared cache.
            _check(self.cuda.cuDevicePrimaryCtxRelease_v2(self.device_index), 'release Torch context retain')
            self.context = None
