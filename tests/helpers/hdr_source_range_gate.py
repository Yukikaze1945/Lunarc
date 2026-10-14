"""One borrowed NVDEC frame through the rebuilt HDR decoder, without NR/VFX."""
import ctypes as C
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bundle_paths
import gpu_transfer_export as gtx
import hdr_gpu_engine as hdr
import native_effects


def main():
    source = sys.argv[1]
    output = ROOT / 'build/hdr-full-range/source-result.json'
    cookies = [os.add_dll_directory(p) for p in gtx.ffmpeg_dll_directories()[:1]]
    bundle_paths.add_dll_directory(bundle_paths.native_dir())
    nvdec = gtx._load('nvdec_bridge')
    nvdec.nvdec_bridge_open_hdr.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
    nvdec.nvdec_bridge_next.argtypes = [C.c_void_p, C.POINTER(gtx._DeviceFrame)]
    nvdec.nvdec_bridge_close.argtypes = [C.c_void_p]
    nvdec.nvdec_bridge_last_error.restype = C.c_char_p
    library = C.CDLL(str(ROOT / 'native/build/hdr_colour/gpu_hdr_scrgb_bridge.dll'))
    exchange = native_effects._library()
    decoder, device = C.c_void_p(), C.c_void_p()
    owner = None
    try:
        code = nvdec.nvdec_bridge_open_hdr(os.fsencode(source), C.byref(decoder))
        assert code == 0, nvdec.nvdec_bridge_last_error()
        frame = gtx._DeviceFrame(C.sizeof(gtx._DeviceFrame))
        assert nvdec.nvdec_bridge_next(decoder, C.byref(frame)) == 1, nvdec.nvdec_bridge_last_error()
        assert frame.software_format == 2
        assert exchange.gpu_effect_device_open(C.byref(device)) == 0
        owner = hdr.GpuHdrBridge(d3d12_device=device.value, cuda_context=frame.cuda_context,
            full_width=frame.width, full_height=frame.height, working_width=frame.width,
            working_height=frame.height, paper_white_scrgb=8, purpose='decode', library=library)
        colour = {key: int(getattr(frame, key)) for key in
            ('color_range', 'color_primaries', 'color_transfer', 'color_space')}
        plane = owner.decode(p010_y_ptr=frame.y_device_ptr, p010_y_pitch=frame.y_pitch_bytes,
            p010_uv_ptr=frame.uv_device_ptr, p010_uv_pitch=frame.uv_pitch_bytes,
            support_ceiling_scrgb=250, **colour)
        report = owner.source_report()
        assert report['nonfinite_samples'] == report['above_ceiling_components'] == 0
        result = dict(passed=True, source=source, decoder_colour=colour,
            extent=[plane.width, plane.height], source_report=report,
            counters=owner.transfer_counters(), image_host_readback=False,
            scope='one decoded source frame, no NR/VFX/export')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(result, ensure_ascii=True))
    finally:
        if owner:
            owner.close()
        if device.value:
            exchange.gpu_effect_device_close(device)
        if decoder.value:
            nvdec.nvdec_bridge_close(decoder)
        for cookie in cookies:
            cookie.close()


if __name__ == '__main__':
    main()
