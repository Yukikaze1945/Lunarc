"""Guidance-off hardware gate for the project GPU host (DLSS Feature 18).

Run from the repository root with ``python native/dlss_gpu_zero_motion_smoke.py``
after building dlss_gpu_host.dll at ABI 3. The gate needs no dlssnr_host.dll: the
final arm compares against that retired host only while its wrapper module is
still in the tree, and reports the arm as unavailable instead of assuming it.
Nothing here creates an NVOF engine:
the unguided entries bind the zero motion texture the host made at open, so the
process must never load the NVOF API, must never stage a motion frame, and must
still hand back a usable D3D12 output texture.

Four claims are checked, not assumed:

* the device-color device-output entry works with no flow at all, and the
  output it borrows back is one persistent resource with an advancing fence;
* the transfer counters say the two device frames crossed no host boundary, and
  the only readback in that phase is the one this smoke asked for by name;
* the output is a real image, not the buffer that happened to be left over;
* when the retired host's wrapper is still importable, the bytes match what it
  answers for the same guidance-off request, frame for frame.
"""
import ctypes as C
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dlss_gpu_host_smoke import (DeviceOutput, Options, TransferCounters,  # noqa: E402
                                 load_gpu_host, make_frames, rgba_from_bgr)

WIDTH, HEIGHT = 1920, 1080
DXGI_R8G8B8A8_UNORM = 28
# The host's own options, mirrored into the legacy session so the two are
# configured the same way for the byte comparison.
OPTIONS = Options(C.sizeof(Options), 0, 0, 1, 1, 1, 0, 0, 0, 1, 1)
LEGACY_SETTINGS = {"preset": 0, "style": 0, "intensity": 1.0, "local_tone": 1.0,
                   "local_struct": 1.0, "skin_struct": 0.0, "use_auto_mask": 0,
                   "ui_correction": 0, "guidance_mode": 0,
                   "motion_scale_x": 1.0, "motion_scale_y": 1.0}
NVOF_MODULES = ("nvofapi64.dll", "nvof_bridge.dll")


class Copy2D(C.Structure):
    _fields_ = [
        ("srcXInBytes", C.c_size_t), ("srcY", C.c_size_t),
        ("srcMemoryType", C.c_int), ("srcHost", C.c_void_p),
        ("srcDevice", C.c_uint64), ("srcArray", C.c_void_p),
        ("srcPitch", C.c_size_t), ("dstXInBytes", C.c_size_t),
        ("dstY", C.c_size_t), ("dstMemoryType", C.c_int),
        ("dstHost", C.c_void_p), ("dstDevice", C.c_uint64),
        ("dstArray", C.c_void_p), ("dstPitch", C.c_size_t),
        ("WidthInBytes", C.c_size_t), ("Height", C.c_size_t),
    ]


def module_loaded(name):
    """Whether this process already has ``name`` mapped, without loading it."""
    kernel32 = C.windll.kernel32
    kernel32.GetModuleHandleW.restype = C.c_void_p
    kernel32.GetModuleHandleW.argtypes = [C.c_wchar_p]
    return bool(kernel32.GetModuleHandleW(name))


class DeviceColor:
    """A primary-context RGBA8 allocation, which is what a producer stage hands over.

    Own context on purpose: the guided path borrows the NVOF engine's, and an
    unguided frame must work without one.
    """

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.cuda = C.WinDLL("nvcuda.dll")
        self._prototype("cuInit", C.c_int, [C.c_uint])
        self._prototype("cuDevicePrimaryCtxRetain", C.c_int,
                        [C.POINTER(C.c_void_p), C.c_int])
        self._prototype("cuCtxPushCurrent_v2", C.c_int, [C.c_void_p])
        self._prototype("cuCtxPopCurrent_v2", C.c_int,
                        [C.POINTER(C.c_void_p)])
        self._prototype("cuMemAllocPitch_v2", C.c_int, [C.POINTER(C.c_uint64),
            C.POINTER(C.c_size_t), C.c_size_t, C.c_size_t, C.c_uint])
        self._prototype("cuMemcpy2D_v2", C.c_int, [C.POINTER(Copy2D)])
        self._prototype("cuMemFree_v2", C.c_int, [C.c_uint64])
        self._prototype("cuDevicePrimaryCtxRelease_v2", C.c_int, [C.c_int])
        assert self.cuda.cuInit(0) == 0, "cuInit failed"
        self.context = C.c_void_p()
        code = self.cuda.cuDevicePrimaryCtxRetain(C.byref(self.context), 0)
        assert code == 0, f"cuDevicePrimaryCtxRetain failed ({code})"
        self.pointer = C.c_uint64()
        self.pitch = C.c_size_t()
        self._call_in_context(self.cuda.cuMemAllocPitch_v2, C.byref(self.pointer),
            C.byref(self.pitch), width * 4, height, 16)

    def _prototype(self, name, restype, argtypes):
        function = getattr(self.cuda, name)
        function.restype = restype
        function.argtypes = argtypes

    def _call_in_context(self, function, *arguments):
        assert self.cuda.cuCtxPushCurrent_v2(self.context) == 0
        try:
            return function(*arguments)
        finally:
            previous = C.c_void_p()
            assert self.cuda.cuCtxPopCurrent_v2(C.byref(previous)) == 0

    def upload(self, rgba):
        assert rgba.flags.c_contiguous, "the uploaded frame has to be contiguous"
        copy = Copy2D()
        copy.srcMemoryType = 1
        copy.srcHost = rgba.ctypes.data
        copy.srcPitch = rgba.strides[0]
        copy.dstMemoryType = 2
        copy.dstDevice = self.pointer.value
        copy.dstPitch = self.pitch.value
        copy.WidthInBytes = self.width * 4
        copy.Height = self.height
        assert self._call_in_context(self.cuda.cuMemcpy2D_v2,
                                     C.byref(copy)) == 0

    def descriptor(self):
        return {"format": "rgba8_cuda", "cuda_context": self.context,
                "device_ptr": self.pointer.value,
                "pitch_bytes": self.pitch.value,
                "width": self.width, "height": self.height}

    def close(self):
        if self.pointer.value:
            self._call_in_context(self.cuda.cuMemFree_v2, self.pointer.value)
            self.pointer = C.c_uint64()
        self.cuda.cuDevicePrimaryCtxRelease_v2(0)


def prototype_zero_motion(lib):
    """The guidance-off entries, which the guided smoke's loader does not bind."""
    lib.dlss_gpu_host_process_zero_motion.argtypes = [C.c_void_p, C.c_void_p,
        C.c_size_t, C.c_int32, C.c_void_p, C.c_size_t]
    lib.dlss_gpu_host_process_zero_motion.restype = C.c_int32
    lib.dlss_gpu_host_process_device_color_zero_motion_resource.argtypes = [
        C.c_void_p, C.c_void_p, C.c_uint64, C.c_size_t, C.c_int32,
        C.POINTER(DeviceOutput)]
    lib.dlss_gpu_host_process_device_color_zero_motion_resource.restype = C.c_int32


def counters_of(lib, handle):
    block = TransferCounters()
    block.struct_size = C.sizeof(TransferCounters)
    assert lib.dlss_gpu_host_get_transfer_counters(handle, C.byref(block)) == 0
    return {name: getattr(block, name) for name, _ in
            TransferCounters._fields_[1:]}


def describe(name, value):
    print(f"{name}={value}", flush=True)


def host_pointer_phase(lib, handle, colors):
    """Two frames through the CPU-pointer unguided entry, which needs no CUDA."""
    outputs = []
    for index, color in enumerate(colors[:2]):
        out = np.zeros_like(color)
        code = lib.dlss_gpu_host_process_zero_motion(handle, color.ctypes.data,
            color.strides[0], int(index == 0), out.ctypes.data, out.strides[0])
        if code:
            raise AssertionError(f"host-pointer frame {index} failed ({code}): "
                                 f"{lib.dlss_gpu_host_last_error()!r}")
        outputs.append(out)
    return outputs


def device_output_phase(lib, handle, colors, device):
    """The production shape: CUDA color in, borrowed D3D12 texture out, no flow."""
    resources = []
    fences = []
    for index, color in enumerate(colors[1:3]):
        device.upload(color)
        descriptor = device.descriptor()
        output = DeviceOutput()
        output.struct_size = C.sizeof(DeviceOutput)
        code = lib.dlss_gpu_host_process_device_color_zero_motion_resource(
            handle, descriptor["cuda_context"], descriptor["device_ptr"],
            descriptor["pitch_bytes"], 0, C.byref(output))
        if code:
            raise AssertionError(f"device-output frame {index} failed ({code}): "
                                 f"{lib.dlss_gpu_host_last_error()!r}")
        assert (output.width, output.height) == (WIDTH, HEIGHT), output.width
        assert output.dxgi_format == DXGI_R8G8B8A8_UNORM, output.dxgi_format
        assert output.d3d12_device and output.d3d12_resource and output.d3d12_fence
        describe(f"device_frame{index}_output_resource", hex(output.d3d12_resource))
        describe(f"device_frame{index}_fence_value", output.fence_value)
        resources.append(output.d3d12_resource)
        fences.append(output.fence_value)
    assert len(set(resources)) == 1, "the output texture was not reused across frames"
    assert fences == sorted(fences) and len(set(fences)) == len(fences), (
        f"the output fence did not advance per frame: {fences}")
    readback = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    code = lib.dlss_gpu_host_readback_output(handle, readback.ctypes.data,
                                             readback.strides[0])
    if code:
        raise AssertionError(f"explicit readback failed ({code}): "
                             f"{lib.dlss_gpu_host_last_error()!r}")
    return readback


def legacy_phase(colors):
    """What the legacy DLL answers for the same guidance-off request.

    One session, one teardown: this DLL is known to hang a process on a second
    init/shutdown cycle, so the guidance_mode comparison the GPU host makes
    possible is not run back to back in here.
    """
    from dlss_engine import Live
    live = Live(WIDTH, HEIGHT, dict(LEGACY_SETTINGS))
    try:
        return [live.process(color, np.zeros((HEIGHT, WIDTH, 2), np.float32),
                             reset=(index == 0), raise_on_error=True).copy()
                for index, color in enumerate(colors[:2])]
    finally:
        live.close()


def main():
    colors = [rgba_from_bgr(frame) for frame in make_frames()]
    assert len(colors) >= 3, "the gate needs three distinct frames"
    lib = load_gpu_host()
    abi = lib.dlss_gpu_host_abi_version()
    describe("abi", abi)
    if abi < 3:
        raise AssertionError("dlss_gpu_host.dll has no zero-motion entries; "
                             "rebuild it from native/ before running this gate")
    prototype_zero_motion(lib)
    handle = C.c_void_p()
    code = lib.dlss_gpu_host_open(WIDTH, HEIGHT, str(ROOT / "nvngx_dlssnr.dll"),
                                  C.byref(OPTIONS), C.byref(handle))
    if code:
        raise AssertionError(f"GPU host open {code}: "
                             f"{lib.dlss_gpu_host_last_error()!r}")
    device = DeviceColor(WIDTH, HEIGHT)
    try:
        host_outputs = host_pointer_phase(lib, handle, colors)
        after_host = counters_of(lib, handle)
        describe("counters_after_host_pointer_phase", after_host)
        frame_bytes = WIDTH * HEIGHT * 4
        # A CPU-pointer frame stages its color in and reads its result out; that
        # is the round trip the device entries below are supposed to not add.
        assert after_host["host_color_stage_count"] == 2, after_host
        assert after_host["host_color_stage_bytes"] == 2 * frame_bytes, after_host
        assert after_host["device_color_copy_count"] == 0, after_host
        assert after_host["output_readback_count"] == 2, after_host

        readback = device_output_phase(lib, handle, colors, device)
        after_device = counters_of(lib, handle)
        describe("counters_after_device_output_phase", after_device)
        assert after_device["host_color_stage_count"] == 2, after_device
        assert after_device["output_readback_count"] == 3, after_device
        assert after_device["output_readback_bytes"] == 3 * frame_bytes, after_device
        assert after_device["device_color_copy_count"] == 2, after_device
        describe("device_frames_with_host_transfer", 0)

        assert readback[..., :3].any(), "the unguided output is all black"
        assert int(readback[..., :3].std()) > 1, "the output carries no image"
        describe("output_alpha_min", int(readback[..., 3].min()))
        describe("output_byte_sum", int(readback[..., :3].sum()))

        for name in NVOF_MODULES:
            assert not module_loaded(name), (
                f"{name} was loaded by an unguided run")
        describe("nvof_modules_loaded", 0)
    finally:
        lib.dlss_gpu_host_close(handle)
        device.close()

    # Cleanup is only a claim if the next session can still be opened: close
    # released the singleton, the device and every texture, including the zero
    # planes, and open rebuilt them at a different extent.
    second = C.c_void_p()
    code = lib.dlss_gpu_host_open(1280, 720, str(ROOT / "nvngx_dlssnr.dll"),
                                  C.byref(OPTIONS), C.byref(second))
    if code:
        raise AssertionError(f"reopen after close failed ({code}): "
                             f"{lib.dlss_gpu_host_last_error()!r}")
    try:
        small = np.zeros((720, 1280, 4), np.uint8)
        small[..., 0] = 40
        small[..., 3] = 255
        out = np.zeros_like(small)
        code = lib.dlss_gpu_host_process_zero_motion(second, small.ctypes.data,
            small.strides[0], 1, out.ctypes.data, out.strides[0])
        if code:
            raise AssertionError(f"second session frame failed ({code}): "
                                 f"{lib.dlss_gpu_host_last_error()!r}")
    finally:
        lib.dlss_gpu_host_close(second)
    describe("reopen_after_close", "ok")

    # The byte-for-byte arm against dlssnr_host.dll is archival: the wrapper module
    # was deleted with the retired host, so a current tree reports that it compared
    # nothing rather than passing an unrun comparison as a result.
    try:
        legacy_outputs = legacy_phase(colors)
    except ImportError as exc:
        describe("ab_vs_legacy_guidance_off", f"unavailable: {exc}")
        legacy_outputs = None
    for index, (gpu, old) in enumerate(zip(host_outputs, legacy_outputs or [])):
        if not np.array_equal(gpu, old):
            difference = np.abs(gpu.astype(np.int16) - old.astype(np.int16))
            raise AssertionError(
                f"frame {index} differs from the legacy guidance-off output: "
                f"mean={difference.mean():.4f} max={difference.max()} "
                f"differing_bytes={np.count_nonzero(difference)} "
                f"(GPU byte_sum={int(gpu.sum())}, legacy={int(old.sum())})")
        describe(f"frame{index}_vs_legacy_guidance_off",
                 f"identical byte_sum={int(gpu.sum())}")
    describe("zero_motion_temporal_change",
             "differs" if not np.array_equal(host_outputs[0], host_outputs[1])
             else "identical")
    print("zero_motion_smoke=PASS", flush=True)


if __name__ == "__main__":
    main()
