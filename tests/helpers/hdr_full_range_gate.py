"""Test-only P010 uploads/readbacks validate full/limited HDR normalization."""
import ctypes as C
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bundle_paths
import hdr_gpu_engine as hdr
import native_effects
from preview_cache import _Copy2D


def main():
    output = ROOT / 'build/hdr-full-range'
    output.mkdir(parents=True, exist_ok=True)
    bundle_paths.add_dll_directory(bundle_paths.native_dir())
    # A running workbench may hold the installed DLL; validate the new build.
    library = C.CDLL(str(ROOT / 'native/build/hdr_colour/gpu_hdr_scrgb_bridge.dll'))
    exchange = native_effects._library()
    device = C.c_void_p()
    assert exchange.gpu_effect_device_open(C.byref(device)) == 0
    cuda = C.WinDLL('nvcuda.dll')
    for name, args in {
        'cuDeviceGet': [C.POINTER(C.c_int), C.c_int],
        'cuCtxCreate_v2': [C.POINTER(C.c_void_p), C.c_uint, C.c_int],
        'cuCtxDestroy_v2': [C.c_void_p],
        'cuMemAlloc_v2': [C.POINTER(C.c_uint64), C.c_size_t],
        'cuMemFree_v2': [C.c_uint64],
        'cuMemcpy2D_v2': [C.POINTER(_Copy2D)],
    }.items():
        getattr(cuda, name).argtypes = args
    assert cuda.cuInit(0) == 0
    ordinal, context = C.c_int(), C.c_void_p()
    assert cuda.cuDeviceGet(C.byref(ordinal), 0) == 0
    assert cuda.cuCtxCreate_v2(C.byref(context), 0, ordinal) == 0
    allocated = []
    owner = None

    def upload(image):
        image = np.ascontiguousarray(image)
        ptr = C.c_uint64()
        assert cuda.cuMemAlloc_v2(C.byref(ptr), image.nbytes) == 0
        allocated.append(ptr.value)
        copy = _Copy2D(srcMemoryType=1, srcHost=image.ctypes.data, srcPitch=image.strides[0],
            dstMemoryType=2, dstDevice=ptr.value, dstPitch=image.strides[0],
            WidthInBytes=image.strides[0], Height=image.shape[0])
        assert cuda.cuMemcpy2D_v2(C.byref(copy)) == 0
        return ptr.value, image.strides[0]

    rows = []
    try:
        owner = hdr.GpuHdrBridge(d3d12_device=device.value, cuda_context=context.value,
            full_width=16, full_height=8, working_width=16, working_height=8,
            paper_white_scrgb=8, purpose='decode', library=library)
        contract = owner.contract
        # Exact endpoints, neutral midtones and chroma offsets test both scales.
        for input_range in (2, 1, 2):
            y_codes = np.tile(np.array([0, 0, 128, 128, 400, 400, 700, 700,
                1023, 1023, 512, 512, 600, 600, 512, 512]), (8, 1))
            uv_codes = np.tile(np.array([512, 512] * 5 + [560, 480, 450, 540, 700, 300]), (4, 1))
            if input_range == 1:
                y_codes = np.rint(64 + y_codes * (876 / 1023))
                uv_codes = np.rint(512 + (uv_codes - 512) * (896 / 1023))
            yp, ypitch = upload(y_codes.astype(np.uint16) << 6)
            uvp, uvpitch = upload(uv_codes.astype(np.uint16) << 6)
            args = dict(p010_y_ptr=yp, p010_y_pitch=ypitch, p010_uv_ptr=uvp,
                p010_uv_pitch=uvpitch, color_range=input_range, color_primaries=9,
                color_transfer=16, color_space=9, support_ceiling_scrgb=250)
            plane = owner.decode(**args)
            actual = np.empty((8, 16, 4), np.float16)
            copy = _Copy2D(srcMemoryType=2, srcDevice=plane.device_ptr, srcPitch=plane.pitch_bytes,
                dstMemoryType=1, dstHost=actual.ctypes.data, dstPitch=actual.strides[0],
                WidthInBytes=actual.strides[0], Height=8)
            assert cuda.cuMemcpy2D_v2(C.byref(copy)) == 0
            y = y_codes / 1023 if input_range == 2 else (y_codes - 64) / 876
            chroma = (uv_codes - 512) / (1023 if input_range == 2 else 896)
            cb = np.repeat(np.repeat(chroma[:, 0::2], 2, axis=0), 2, axis=1)
            cr = np.repeat(np.repeat(chroma[:, 1::2], 2, axis=0), 2, axis=1)
            kr, kg, kb = (contract[key] for key in ('kr', 'kg', 'kb'))
            r, b = y + 2 * (1 - kr) * cr, y + 2 * (1 - kb) * cb
            rgb = np.stack([r, (y - kr * r - kb * b) / kg, b], axis=-1)
            n = np.maximum(rgb, 0) ** (1 / contract['pq_m2'])
            relative = (np.maximum(n - contract['pq_c1'], 0) /
                (contract['pq_c2'] - contract['pq_c3'] * n)) ** (1 / contract['pq_m1'])
            reference = relative @ np.array(contract['to_scrgb']).reshape(3, 3).T * 125
            expected = reference.astype(np.float16)
            # At most one FP16 rounding step away from a double reference.
            bits = actual[:, :, :3].view(np.uint16).astype(np.int32)
            error = int(np.abs(bits - expected.view(np.uint16).astype(np.int32)).max())
            assert error <= 1, error
            assert np.all(actual[:, :, 3] == 1)
            assert np.all(actual[0, 0, :3] == 0)
            assert np.all(actual[0, 8, :3] == 125)
            if input_range == 2 and len(rows) == 2:
                assert np.array_equal(actual, full_result)
            elif input_range == 2:
                full_result = actual.copy()
            rows.append(dict(input_range=input_range, max_fp16_steps=error,
                source_report=owner.source_report()))
            for bad in (0, 3):
                try:
                    owner.decode(**dict(args, color_range=bad))
                except hdr.HdrBridgeError as exc:
                    assert 'rejected range' in str(exc)
                else:
                    raise AssertionError('unsupported range accepted')
        counters = owner.transfer_counters()
        result = dict(passed=True, rows=rows, counters=counters,
            scope='decode only; test-only CPU uploads and readbacks; no NR/export performance claim')
        (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result))
    finally:
        if owner:
            owner.close()
        for ptr in allocated:
            cuda.cuMemFree_v2(ptr)
        cuda.cuCtxDestroy_v2(context)
        exchange.gpu_effect_device_close(device)


if __name__ == '__main__':
    main()
