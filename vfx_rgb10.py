"""GPU conversion only at the VFX RGB10A2 boundary; the master stays FP16."""
import ctypes as C
import bundle_paths


class Codec:
    def __init__(self, context, size, runtime, *, hdr):
        self.lib = C.WinDLL(bundle_paths.require_native('rtx_vsr_bridge.dll', 'VFX 10-bit'))
        self.lib.vfx_rgb10_last_error.restype = C.c_char_p
        self.lib.vfx_rgb10_open.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32, C.c_uint32,
            C.POINTER(C.c_float), C.c_wchar_p, C.POINTER(C.c_void_p)]
        self.lib.vfx_rgb10_pack.argtypes = [C.c_void_p, C.c_uint64, C.c_uint32,
            C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)]
        self.lib.vfx_rgb10_pointer.argtypes = [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)]
        self.lib.vfx_rgb10_unpack.argtypes = [C.c_void_p, C.c_uint64, C.c_uint32, C.c_uint64, C.c_uint32]
        self.lib.vfx_rgb10_counters.argtypes = [C.c_void_p, C.POINTER(C.c_uint64)]
        self.lib.vfx_rgb10_close.argtypes = [C.c_void_p]
        # Bind canonical colour constants from the currently loaded HDR bridge.
        engine = __import__('hdr_gpu_engine')
        contract = engine.GpuHdrContract(C.sizeof(engine.GpuHdrContract))
        library = engine.load()
        rc = library.gpu_hdr_fill_contract(C.byref(contract))
        if rc:
            raise RuntimeError('Cannot read the existing HDR colour contract')
        values = list(contract.to_bt2020) + list(contract.to_scrgb) + [contract.pq_m1,
            contract.pq_m2, contract.pq_c1, contract.pq_c2, contract.pq_c3, contract.nits_per_scrgb / 10000]
        self.handle = C.c_void_p()
        self._check(self.lib.vfx_rgb10_open(context, *size, int(hdr),
            (C.c_float * 24)(*values), runtime, C.byref(self.handle)))

    def _check(self, rc):
        if rc:
            raise RuntimeError(self.lib.vfx_rgb10_last_error().decode('utf-8', 'replace'))

    def pack(self, ptr, pitch):
        output, row = C.c_uint64(), C.c_uint32()
        self._check(self.lib.vfx_rgb10_pack(self.handle, ptr, pitch, C.byref(output), C.byref(row)))
        return output.value, row.value

    def pointer(self):
        ptr, pitch = C.c_uint64(), C.c_uint32()
        self._check(self.lib.vfx_rgb10_pointer(self.handle, C.byref(ptr), C.byref(pitch)))
        return ptr.value, pitch.value

    def sdr8(self, ptr, pitch):
        self.lib.vfx_sdr8_pack.argtypes = [C.c_void_p, C.c_uint64, C.c_uint32,
            C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)]
        output, row = C.c_uint64(), C.c_uint32()
        self._check(self.lib.vfx_sdr8_pack(self.handle, ptr, pitch, C.byref(output), C.byref(row)))
        return output.value, row.value

    def unpack(self, ptr, pitch, output, opitch):
        self._check(self.lib.vfx_rgb10_unpack(self.handle, ptr, pitch, output, opitch))

    def counters(self):
        values = (C.c_uint64 * 4)()
        self._check(self.lib.vfx_rgb10_counters(self.handle, values))
        return dict(pack_calls=values[0], encoded_boundary_clipped_components=values[1],
            scalar_readback_bytes=values[2], host_image_bytes=values[3])

    def close(self):
        if self.handle:
            self.lib.vfx_rgb10_close(self.handle)
            self.handle = None
