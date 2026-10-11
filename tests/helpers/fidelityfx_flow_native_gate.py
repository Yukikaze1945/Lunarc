"""Opt-in real D3D12/CUDA FidelityFX Optical Flow direction/reset smoke.

The gate creates D3D12 on the adapter whose LUID matches the selected CUDA
device, submits three SDK dispatches, and reads back only the small SCD counters
plus diagnostic flow samples. It does not substitute a CPU flow implementation.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fidelityfx_flow_engine import FidelityFXFlow


class GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class AdapterDesc1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32), ("SubSysId", ctypes.c_uint32), ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuid", LUID), ("Flags", ctypes.c_uint32)]


def _guid(text: str) -> GUID:
    u = uuid.UUID(text)
    return GUID.from_buffer_copy(u.bytes_le)


def _com(ptr, index, restype, argtypes, *args):
    vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[index])
    return fn(ptr, *args)


def _check(code, name):
    if code:
        raise RuntimeError(f"{name} returned CUDA error {code}")


def _device(cuda, device_ordinal):
    _check(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    _check(cuda.cuDeviceGet(ctypes.byref(dev), device_ordinal), "cuDeviceGet")
    luid = (ctypes.c_ubyte * 8)()
    node = ctypes.c_uint()
    _check(cuda.cuDeviceGetLuid(luid, ctypes.byref(node), dev.value), "cuDeviceGetLuid")
    return dev, bytes(luid), node.value


def _matching_d3d12(cuda_luid: bytes):
    dxgi = ctypes.WinDLL("dxgi.dll")
    create_factory = dxgi.CreateDXGIFactory1
    create_factory.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create_factory.restype = ctypes.c_long
    factory = ctypes.c_void_p()
    hr = create_factory(ctypes.byref(_guid("770aae78-f26f-4dba-a829-253c83d1b387")), ctypes.byref(factory))
    if hr < 0:
        raise OSError(f"CreateDXGIFactory1 failed: 0x{hr & 0xffffffff:08x}")
    d3d12 = ctypes.WinDLL("d3d12.dll")
    create_device = d3d12.D3D12CreateDevice
    create_device.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create_device.restype = ctypes.c_long
    device = ctypes.c_void_p()
    try:
        for index in range(32):
            adapter = ctypes.c_void_p()
            hr = _com(factory, 12, ctypes.c_long, [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint], index, ctypes.byref(adapter), 0)
            if hr < 0:
                break
            desc = AdapterDesc1()
            hr = _com(adapter, 10, ctypes.c_long, [ctypes.POINTER(AdapterDesc1)], ctypes.byref(desc))
            if hr >= 0 and ctypes.string_at(ctypes.byref(desc, AdapterDesc1.AdapterLuid.offset), 8) == cuda_luid:
                hr = create_device(adapter, 0xB000, ctypes.byref(_guid("189819f1-1db6-4b57-be54-1821339b85f7")), ctypes.byref(device))
                _com(adapter, 2, ctypes.c_uint, [])
                if hr < 0:
                    raise OSError(f"D3D12CreateDevice on CUDA adapter failed: 0x{hr & 0xffffffff:08x}")
                return device
            _com(adapter, 2, ctypes.c_uint, [])
        raise RuntimeError("No DXGI adapter LUID matches the selected CUDA device")
    finally:
        _com(factory, 2, ctypes.c_uint, [])


def _image(width: int, height: int, dx: int = 0, dy: int = 0) -> bytes:
    # Translate a central textured patch while keeping most of the frame static
    # to stay below the SDK scene-change detector threshold. In that patch the
    # current frame samples the previous image at (x-dx,y-dy).
    data = bytearray(width * height * 4)
    for y in range(height):
        for x in range(width):
            moving = width // 5 <= x < 4 * width // 5 and height // 5 <= y < 4 * height // 5
            sx, sy = (x - dx, y - dy) if moving else (x, y)
            i = (y * width + x) * 4
            if 0 <= sx < width and 0 <= sy < height:
                tile = ((sx // 19) * 73 + (sy // 23) * 151 + ((sx // 7) ^ (sy // 9)) * 37) & 255
                fine = ((sx * 13 + sy * 29 + (sx * sy) % 97) & 63)
                checker = 96 if ((sx // 31 + sy // 27) & 1) else 0
                data[i:i+4] = bytes(((tile + fine + checker) & 255,
                                     (tile * 3 + fine * 2 + checker) & 255,
                                     (tile * 5 + fine + checker * 2) & 255, 255))
            else:
                data[i:i+4] = b"\x00\x00\x00\xff"
    return bytes(data)


def run(args) -> dict:
    outdir = args.evidence_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    cuda = ctypes.WinDLL("nvcuda.dll")
    def cu(name, argtypes, restype=ctypes.c_int):
        fn = getattr(cuda, name); fn.argtypes = argtypes; fn.restype = restype; return fn
    cuinit = cu("cuInit", [ctypes.c_uint]); getdev = cu("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int])
    getluid = cu("cuDeviceGetLuid", [ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_uint), ctypes.c_int])
    retain = cu("cuDevicePrimaryCtxRetain", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int])
    release = cu("cuDevicePrimaryCtxRelease_v2", [ctypes.c_int])
    setctx = cu("cuCtxSetCurrent", [ctypes.c_void_p])
    alloc = cu("cuMemAlloc_v2", [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t])
    free = cu("cuMemFree_v2", [ctypes.c_uint64])
    h2d = cu("cuMemcpyHtoD_v2", [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t])
    d2h = cu("cuMemcpyDtoH_v2", [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t])
    dev, luid, node = _device(cuda, args.cuda_device)
    context, d3d = ctypes.c_void_p(), ctypes.c_void_p()
    srcptr = ctypes.c_uint64()
    flow = None
    failure = None
    move_x, move_y = 8, 4
    report = {"status": "running", "sdk": "FidelityFX SDK 1.1.4", "adapter_luid": luid.hex(), "cuda_node_mask": node,
              "resolution": [args.width, args.height], "translation_pixels": [move_x, move_y], "expected_current_to_previous": [-move_x, -move_y]}
    try:
        _check(retain(ctypes.byref(context), dev.value), "cuDevicePrimaryCtxRetain")
        _check(setctx(context), "cuCtxSetCurrent")
        d3d = _matching_d3d12(luid)
        pitch = args.width * 4
        _check(alloc(ctypes.byref(srcptr), pitch * args.height), "cuMemAlloc(input)")
        flow = FidelityFXFlow(args.bridge, d3d.value, context.value, args.width, args.height, args.quality)
        report['quality'] = flow.quality
        analysis_size = [((v+1)//2 if flow.quality == 'Performance' else v) for v in (args.width,args.height)]
        report['analysis_resolution'] = analysis_size
        frames = []
        # Official SDK intentionally returns zero flow for frameIndex <= 5.
        # Keep that algorithm intact and validate motion after six-frame startup.
        requests = [(0, 0, i == 0) for i in range(7)] + [
            (move_x, move_y, False), (move_x, move_y, True)]
        for dx, dy, reset in requests:
            raw = _image(args.width, args.height, dx, dy)
            host = ctypes.create_string_buffer(raw)
            _check(h2d(srcptr.value, host, len(raw)), "cuMemcpyHtoD(input)")
            result = flow.process(srcptr.value, pitch, reset)
            if result.width != args.width or result.height != args.height or result.pitch_bytes < args.width * 8:
                raise RuntimeError(f"invalid output extent or pitch: {result}")
            output_size = result.pitch_bytes * result.height
            host_out = bytearray(output_size)
            _check(d2h((ctypes.c_ubyte * output_size).from_buffer(host_out), result.deviceptr, output_size), "cuMemcpyDtoH(flow diagnostic)")
            samples_x, samples_y = [], []
            for y in range(2 * args.height // 5, 3 * args.height // 5, 8):
                for x in range(2 * args.width // 5, 3 * args.width // 5, 8):
                    off = y * result.pitch_bytes + x * 8
                    vx, vy = ctypes.c_float.from_buffer_copy(host_out, off).value, ctypes.c_float.from_buffer_copy(host_out, off + 4).value
                    if abs(vx) < 100 and abs(vy) < 100:
                        samples_x.append(vx); samples_y.append(vy)
            frame_fact = {"reset": reset, "sample_count": len(samples_x),
                          "median_flow_xy": [statistics.median(samples_x), statistics.median(samples_y)] if samples_x else None,
                          "sample_range_xy": [[min(samples_x), max(samples_x)], [min(samples_y), max(samples_y)]] if samples_x else None,
                          "output_pitch_bytes": result.pitch_bytes}
            frames.append(frame_fact)
            report["frames"] = frames
        counters = flow.counters()
        observed = frames[7]["median_flow_xy"]
        report["counters"] = counters
        if [counters['grid_width'], counters['grid_height']] != [(v+7)//8 for v in analysis_size]:
            raise RuntimeError('SDK grid does not match requested analysis resolution')
        if observed is None or abs(observed[0] + move_x) > 2.0 or abs(observed[1] + move_y) > 2.0:
            raise RuntimeError(f"translation flow direction/magnitude outside tolerance: {observed}")
        if frames[7]["sample_count"] < 24:
            raise RuntimeError(f"too few finite flow diagnostic samples: {frames[7]['sample_count']}")
        if counters["successful_dispatches"] != 9 or counters["reset_dispatches"] != 2 or counters["flow_format_float32x2"] != 1:
            raise RuntimeError(f"unexpected native counters: {counters}")
        report.update(status="pass", frames=frames, counters=counters, output_contract={"format": "float32x2", "units": "input_pixels", "direction": "current_to_previous", "full_resolution": True})
    except Exception as exc:
        failure = exc
        report.update(status="fail", error=f"{type(exc).__name__}: {exc}")
    finally:
        if flow: flow.close()
        if srcptr.value: _check(free(srcptr.value), "cuMemFree(input)")
        if d3d.value: _com(d3d, 2, ctypes.c_uint, [])
        if context.value:
            _check(setctx(None), "clear CUDA context"); _check(release(dev.value), "cuDevicePrimaryCtxRelease")
    evidence = outdir / "fidelityfx_flow_native_gate.json"
    evidence.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failure:
        raise RuntimeError(f"Optical Flow native gate failed; evidence: {evidence}") from failure
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bridge", type=Path, default=ROOT / "bench_out/amd_flow_20261010/bridge/fidelityfx_flow_bridge.dll")
    p.add_argument("--evidence-dir", type=Path, default=ROOT / "bench_out/amd_flow_20261010/native_gate")
    p.add_argument("--cuda-device", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--quality", choices=('Performance', 'Quality'), default='Quality')
    args = p.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "evidence_dir": str(args.evidence_dir.resolve()),
                      "frames": len(report["frames"]), "counters": report["counters"]}, indent=2))


if __name__ == "__main__":
    main()
