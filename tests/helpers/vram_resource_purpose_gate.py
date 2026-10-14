"""GPU resource-purpose equivalence and allocation check; readbacks are test-only."""
import ctypes as C
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bundle_paths
import hdr_gpu_engine as hdr
import native_effects
from preview_cache import _Copy2D
from temporal_control import GpuTemporalControl, TemporalControls


class Memory(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
                ('luid', 'local_usage', 'local_budget', 'nonlocal_usage', 'nonlocal_budget')]


def main():
    output = ROOT/'build/vram-purposes'
    output.mkdir(parents=True, exist_ok=True)
    bundle_paths.add_dll_directory(bundle_paths.native_dir())
    lib = native_effects._library()
    device = C.c_void_p()
    assert lib.gpu_effect_device_open(C.byref(device)) == 0
    cuda = C.WinDLL('nvcuda.dll')
    signatures = {
        'cuDeviceGet': [C.POINTER(C.c_int), C.c_int],
        'cuCtxCreate_v2': [C.POINTER(C.c_void_p), C.c_uint, C.c_int],
        'cuCtxDestroy_v2': [C.c_void_p],
        'cuMemAlloc_v2': [C.POINTER(C.c_uint64), C.c_size_t],
        'cuMemFree_v2': [C.c_uint64],
        'cuMemcpy2D_v2': [C.POINTER(_Copy2D)],
    }
    for name, args in signatures.items():
        getattr(cuda, name).argtypes = args
    assert cuda.cuInit(0) == 0
    ordinal, context = C.c_int(), C.c_void_p()
    assert cuda.cuDeviceGet(C.byref(ordinal), 0) == 0
    assert cuda.cuCtxCreate_v2(C.byref(context), 0, ordinal) == 0
    memory = C.CDLL(str(ROOT/'native/build/memory/gpu_process_memory_probe.dll'))
    memory.gpu_process_memory_query.argtypes = [C.c_void_p, C.POINTER(Memory)]
    def sample():
        m = Memory()
        assert memory.gpu_process_memory_query(device, C.byref(m)) == 0
        return {key: getattr(m, key) for key, _ in Memory._fields_}
    def allocate(image):
        image = np.ascontiguousarray(image)
        ptr = C.c_uint64()
        pitch = image.strides[0]
        assert cuda.cuMemAlloc_v2(C.byref(ptr), image.nbytes) == 0
        cp = _Copy2D(srcMemoryType=1, srcHost=image.ctypes.data, srcPitch=pitch,
                     dstMemoryType=2, dstDevice=ptr.value, dstPitch=pitch,
                     WidthInBytes=pitch, Height=image.shape[0])
        assert cuda.cuMemcpy2D_v2(C.byref(cp)) == 0
        return ptr.value, pitch
    def read(plane, dtype, channels=1):
        image = np.empty((plane.height, plane.width, channels), dtype=dtype)
        cp = _Copy2D(srcMemoryType=2, srcDevice=plane.device_ptr, srcPitch=plane.pitch_bytes,
                     dstMemoryType=1, dstHost=image.ctypes.data, dstPitch=image.strides[0],
                     WidthInBytes=image.strides[0], Height=image.shape[0])
        assert cuda.cuMemcpy2D_v2(C.byref(cp)) == 0
        return image.tobytes()
    def exchange_bytes(exchange):
        fn = exchange.lib.gpu_rgba_exchange_allocated_bytes
        fn.argtypes, fn.restype = [C.c_void_p], C.c_uint64
        return int(fn(exchange.handle))
    def bridge(kind, purpose, size):
        return kind(d3d12_device=device.value, cuda_context=context.value,
                      full_width=size[0], full_height=size[1], working_width=size[0],
                      working_height=size[1], paper_white_scrgb=1 if kind is hdr.GpuSdr10Bridge else 8,
                      purpose=purpose)

    result = {'scope': 'same-device DEFAULT heap allocations and bitwise GPU-output comparisons',
              'test_image_readbacks': True, 'rows': []}
    try:
        for kind in (hdr.GpuHdrBridge, hdr.GpuSdr10Bridge):
            w, h = 128, 72
            y = np.linspace(64, 940, w*h, dtype=np.uint16).reshape(h, w) << 6
            uv = np.full((h//2, w), 512 << 6, dtype=np.uint16)
            yp, ypitch = allocate(y); uvp, uvpitch = allocate(uv)
            image = np.zeros((h, w, 4), np.float16)
            image[..., 0] = np.linspace(-.08, .95, w*h).reshape(h, w)
            image[..., 1] = .5; image[..., 2] = 1.6; image[..., 3] = 1
            if kind is hdr.GpuHdrBridge: image[..., :3] *= 3
            ptr, pitch = allocate(image)
            producer = native_effects.Exchange(device.value, context.value, (w, h), wide=True)
            texture = producer.texture(ptr, pitch)
            source = dict(format='rgba16f_cuda', width=w, height=h, cuda_context=context.value,
                          device_ptr=ptr, pitch_bytes=pitch, bytes_per_pixel=8)
            reference = {}
            for purpose in ('full', 'decode', 'guidance', 'encode'):
                owner = bridge(kind, purpose, (w, h))
                try:
                    row = {'colour': kind.__name__, 'purpose': purpose,
                           'allocated_device_bytes': owner.allocated_device_bytes()}
                    values = {}
                    if purpose in ('full', 'decode'):
                        master = owner.decode(p010_y_ptr=yp, p010_y_pitch=ypitch, p010_uv_ptr=uvp,
                            p010_uv_pitch=uvpitch, color_range=1,
                            color_primaries=1 if kind is hdr.GpuSdr10Bridge else 9,
                            color_transfer=1 if kind is hdr.GpuSdr10Bridge else 16,
                            color_space=1 if kind is hdr.GpuSdr10Bridge else 9,
                            support_ceiling_scrgb=2 if kind is hdr.GpuSdr10Bridge else 125)
                        values['decode'] = read(master, np.float16, 4)
                    if purpose in ('full', 'guidance'):
                        owner.import_master(source)
                        working = owner.downscale(); proxy, guidance = owner.proxy()
                        values.update(working=read(working, np.float16, 4),
                                      proxy=read(proxy, np.float16, 4),
                                      guidance=read(guidance, np.uint8, 4))
                        rgba = np.frombuffer(values['guidance'], np.uint8).reshape(h, w, 4)
                        assert np.all(rgba[..., 0] == rgba[..., 1]) and np.all(rgba[..., 1] == rgba[..., 2])
                        assert np.all(rgba[..., 3] == 255), 'guidance must pack exactly one RGBA8 pixel per Store'
                    if purpose in ('full', 'encode'):
                        owner.import_master(source); owner.set_composite(texture)
                        encoded = owner.encode(clip_ceiling_scrgb=2 if kind is hdr.GpuSdr10Bridge else 125)
                        values.update(encode_y=read(encoded.y, np.uint16), encode_uv=read(encoded.uv, np.uint16, 2))
                    if purpose == 'full': reference = values
                    else:
                        for key, value in values.items():
                            if reference[key] != value:
                                dt = np.uint8 if key == 'guidance' else np.uint16 if key.startswith('encode') else np.float16
                                a, b = np.frombuffer(reference[key], dt), np.frombuffer(value, dt)
                                print('DIFFERENCE', row, key, 'count', int(np.count_nonzero(a != b)),
                                      'max', float(np.max(np.abs(a.astype(float)-b.astype(float)))),
                                      'first', a[:12], b[:12], flush=True)
                        assert all(reference[key] == value for key, value in values.items()), row
                        row['bitwise_equal_to_full'] = sorted(values)
                    row['production_image_host_bytes'] = owner.transfer_counters()['report_readback_bytes']
                    assert row['production_image_host_bytes'] == 0
                    if purpose == 'decode':
                        try: owner.downscale()
                        except hdr.HdrBridgeError: row['omitted_stage_refused'] = True
                        else: raise AssertionError('omitted downscale accepted')
                    result['rows'].append(row)
                finally: owner.close()
            producer.close()
            for p in (yp, uvp, ptr): assert cuda.cuMemFree_v2(p) == 0

        w, h = 3840, 2160
        # Sequential owners avoid combining allocations from unrelated test cases.
        for purpose in ('full', 'guidance', 'decode', 'encode'):
            before = sample(); mark = time.perf_counter()
            owner = bridge(hdr.GpuHdrBridge, purpose, (w, h))
            result['rows'].append({'size': [w,h], 'purpose': purpose,
                'allocated_device_bytes': owner.allocated_device_bytes(), 'dxgi_before': before,
                'dxgi_open': sample(), 'setup_ms': (time.perf_counter()-mark)*1000})
            owner.close()
        for wide in (False, True):
            owner = native_effects.Exchange(device.value, context.value, (w,h), wide=wide)
            before = exchange_bytes(owner)
            image = np.ones((h,w,4), np.float16 if wide else np.uint8)
            ptr, pitch = allocate(image)
            texture = owner.texture(ptr, pitch)
            after = exchange_bytes(owner)
            importer = native_effects.Exchange(device.value, context.value, (w,h), wide=wide)
            imported_before = exchange_bytes(importer)
            color = importer.color(texture)
            assert exchange_bytes(importer) == imported_before
            plane = hdr.DevicePlane(fmt=color['format'], pixel_format='test', width=w, height=h,
                cuda_context=context.value, device_ptr=color['device_ptr'], pitch_bytes=color['pitch_bytes'],
                bytes_per_pixel=8 if wide else 4, plane=0)
            assert read(plane, image.dtype, 4) == image.tobytes()
            assert after > before
            result['rows'].append({'exchange': 'FP16' if wide else 'RGBA8',
                'import_only_bytes': before, 'export_bytes': after, 'bitwise_equal': True})
            importer.close(); owner.close(); cuda.cuMemFree_v2(ptr)
        owner = GpuTemporalControl(TemporalControls(), hdr=True)
        owner._open(device.value, context.value, w, h)
        small = owner.allocated_device_bytes()
        image = np.ones((h,w,4), np.float16); ptr, pitch = allocate(image)
        args = dict(device=device.value, cuda_context=context.value, width=w, height=h,
                    source_ptr=ptr, source_pitch=pitch, reset=True)
        owner.observe(**args)
        assert owner.allocated_device_bytes() == small
        owner.update_controls(TemporalControls(mode='static_low_frequency'))
        owner.process(**args, candidate=dict(format='rgba16f_cuda', width=w, height=h,
            cuda_context=context.value, device_ptr=ptr, pitch_bytes=pitch))
        large = owner.allocated_device_bytes()
        owner.update_controls(TemporalControls())
        assert owner.allocated_device_bytes() == 0
        owner.observe(**args)
        assert owner.allocated_device_bytes() == small
        result['rows'].append({'temporal_size': [w,h], 'scene_only_bytes': small,
            'stability_bytes': large, 'on_off_on_resources_released': True})
        owner.close(); cuda.cuMemFree_v2(ptr)
        result['passed'] = True
    finally:
        (output/'resource-gate.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        cuda.cuCtxDestroy_v2(context)
        lib.gpu_effect_device_close(device)
    print('Resource-purpose equivalence, exchange and temporal allocation checks passed')


if __name__ == '__main__': main()
