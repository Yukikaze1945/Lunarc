"""Official NGX SuperSampling DLAA mode, equal input/output RGBA8 or FP16 resources.

The signed DLSS binary is discovered from the local NGX cache/application, or an override. No
Feature 18 session or resize substitutes for missing DLAA. Video currently uses
zero depth/jitter; this is explicitly not a game's engine motion contract.
"""
import ctypes
import os
import bundle_paths
import dlss_gpu_engine as nr


class Live(nr.Live):
    native_dlaa = True
    def __init__(self, width, height, settings=None, *, dll_path=None, is_hdr=False,
                 shared_with=None):
        from native_runtime_paths import dlss_runtime
        from native_effects import DLAA_PRESETS
        self.render_preset = (settings or {}).get('dlaa_preset', 0)
        if isinstance(self.render_preset, bool) or self.render_preset not in DLAA_PRESETS:
            raise ValueError('DLAA preset must be Default, E, F, J, K, L or M')
        self.dlss_dll = dlss_runtime(dll_path)
        super().__init__(width, height, settings, color_format='rgba16f' if is_hdr else 'rgba8', shared_with=shared_with)

    def _options(self):
        options = super()._options()
        options.preset = int(self.render_preset)
        return options

    def _open(self, shared_with):
        self.runtime_dll = self.dlss_dll
        if self.bytes_per_pixel == 8:
            name = 'dlss_gpu_host_open_dlaa_layer_ex' if shared_with is not None else 'dlss_gpu_host_open_dlaa_ex'
            if not hasattr(self._lib, name):
                raise RuntimeError('DLAA FP16 requires the updated native host')
            call = getattr(self._lib, name)
            call.argtypes = ([ctypes.c_void_p] if shared_with is not None else []) + [
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_wchar_p,
                ctypes.POINTER(nr._Options), ctypes.POINTER(nr._OpenInfo), ctypes.POINTER(ctypes.c_void_p)]
            call.restype = ctypes.c_int32
            options = self._options()
            info = nr._OpenInfo(ctypes.sizeof(nr._OpenInfo), nr._COLOR_FORMAT[self.color_format], 0)
            args = [self._w, self._h, self.dlss_dll, ctypes.byref(options), ctypes.byref(info), ctypes.byref(self._handle)]
            if shared_with is not None:
                args.insert(0, shared_with._handle)
            return call(*args)
        if shared_with is not None:
            self._lib.dlss_gpu_host_open_dlaa_layer.argtypes = [ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_uint32, ctypes.c_wchar_p,
                ctypes.POINTER(nr._Options), ctypes.POINTER(ctypes.c_void_p)]
            self._lib.dlss_gpu_host_open_dlaa_layer.restype = ctypes.c_int32
            options = self._options()
            return self._lib.dlss_gpu_host_open_dlaa_layer(shared_with._handle,
                self._w, self._h, self.dlss_dll, ctypes.byref(options),
                ctypes.byref(self._handle))
        self._lib.dlss_gpu_host_open_dlaa.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_wchar_p, ctypes.POINTER(nr._Options), ctypes.POINTER(ctypes.c_void_p)]
        self._lib.dlss_gpu_host_open_dlaa.restype = ctypes.c_int32
        options = self._options()
        return self._lib.dlss_gpu_host_open_dlaa(self._w, self._h, self.dlss_dll,
            ctypes.byref(options), ctypes.byref(self._handle))

    def counters(self):
        values = (ctypes.c_uint64 * 3)()
        self._lib.dlss_gpu_host_dlaa_counters.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        if self._lib.dlss_gpu_host_dlaa_counters(values):
            raise RuntimeError('Native DLAA counters unavailable')
        return dict(zip(('creates', 'successful_evaluates', 'release_calls'), values),
                    preset_requested=self.render_preset, preset_effective=None)

    def transfer_counters(self):
        result = super().transfer_counters()
        result.update(vendor_feature='NGX SuperSampling / DLAA (equal extent)',
                      dlaa_process_global_counters=self.counters())
        return result
