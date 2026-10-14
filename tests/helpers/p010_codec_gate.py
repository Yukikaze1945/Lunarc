"""Short P010 codec boundary / AV1 metadata gate using production ABI.

Artifacts and fixtures are local and ignored. --dll permits checking a staged
build while the application holds the installed DLL. No DLSS model in boundary.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import gpu_transfer_export as gtx
import gpu_hdr_transfer_export as htx
import pipeline


def load(dll):
    original = gtx._load
    def replacement(name):
        if name == 'nvenc_bridge':
            lib = C.CDLL(str(dll.resolve())); lib._dll_path = str(dll.resolve())
            return lib
        return original(name)
    gtx._load = replacement
    return gtx.load_bridges(requires_flow=False)


def metadata():
    md = htx.HdrMetadata(C.sizeof(htx.HdrMetadata))
    md.has_mastering = md.mastering_has_primaries = md.mastering_has_luminance = md.has_content_light = 1
    md.primary_x_num[:] = [6400,3000,1500]; md.primary_y_num[:] = [3300,6000,600]
    md.primary_x_den[:] = md.primary_y_den[:] = [10000]*3
    md.white_x_num,md.white_x_den,md.white_y_num,md.white_y_den = 3127,10000,3290,10000
    md.min_luminance_num,md.min_luminance_den = 1,10000
    md.max_luminance_num,md.max_luminance_den = 1000,1
    md.max_cll,md.max_fall = 1000,400
    return md


def cpu_gate(lib):
    fn=lib.nvenc_av1_metadata_apply_packet
    fn.argtypes=[C.POINTER(htx.HdrMetadata),C.c_void_p,C.c_uint32,C.c_void_p,
                 C.c_uint32,C.POINTER(C.c_uint32),C.POINTER(htx.P010MetadataStatus)]
    fn.restype=C.c_int32
    def apply(md, data, fails=False):
        size=C.c_uint32(); status=htx.P010MetadataStatus(C.sizeof(htx.P010MetadataStatus))
        out=C.create_string_buffer(len(data)+256)
        code=fn(C.byref(md),data,len(data),out,len(out),C.byref(size),C.byref(status))
        assert (code < 0) == fails, (code,htx._fields(status))
        return out.raw[:size.value], htx._fields(status)
    frame=b'\x32\x01\x00'  # OBU_FRAME, opaque test payload, never decoded here.
    md=metadata(); result,status=apply(md,frame)
    assert result[-len(frame):] == frame
    assert status['mastering_inserted']==status['content_light_inserted']==1
    expected=[]
    for i in range(3):
        for nums,dens in ((md.primary_x_num,md.primary_x_den),(md.primary_y_num,md.primary_y_den)):
            q=Fraction(nums[i],dens[i])*65536
            expected.append((q.numerator*2+q.denominator)//(q.denominator*2))
    assert status['wire_values'][:6] == expected
    assert status['wire_values'][8:]==[256000,2,1000,400]
    unchanged,existing=apply(md,result)
    assert unchanged==result and existing['bytes_inserted']==0
    assert existing['mastering_already_present']==existing['content_light_already_present']==1
    # The standard allows multi-byte LEB128 for small values as well.
    expanded=b'\x2a\x1b\x82\x00'+result[3:28]+result[28:]
    unchanged,existing=apply(md,expanded)
    assert unchanged==expanded and existing['mastering_already_present']==1
    prefix=b'\x0a\x01\x00\x2a\x03\x04\x99\x80'
    augmented,_=apply(md,prefix+frame)
    assert augmented.startswith(prefix) and augmented.endswith(frame)
    altered=bytearray(result); altered[4]^=1; apply(md,bytes(altered),True)
    apply(md,result[:28]+result,True)
    for malformed in (b'\x32\x80',b'\x32\x81\x01',b'\x30\x00',b'\xaa\x00',b'\x2a\x01\x02',b'\x36\x07\x01\x00'):
        apply(md,malformed,True)
    no_md=htx.HdrMetadata(C.sizeof(htx.HdrMetadata)); unchanged,_=apply(no_md,frame)
    assert unchanged==frame
    apply(no_md,result,True)
    partial=metadata(); partial.mastering_has_luminance=0; apply(partial,frame,True)
    invalid=metadata(); invalid.primary_x_den[0]=0; apply(invalid,frame,True)
    overflow=metadata(); overflow.primary_x_num[0]=10001; apply(overflow,frame,True)
    light=metadata(); light.has_mastering=0
    light_only,light_status=apply(light,frame)
    assert light_status['mastering_inserted']==0 and light_only[-3:]==frame
    absent,_=apply(md,b'\x12\x00')  # Temporal delimiter only.
    assert absent==b'\x12\x00'
    return status


def probe(path):
    return json.loads(subprocess.check_output([pipeline._bundle_ffprobe(),'-v','error',
        '-show_streams','-show_frames','-show_packets','-show_data_hash','sha256',
        '-show_chapters','-show_format','-of','json',str(path)],encoding='utf-8'))


def boundary(b,source,out,codec,hdr,legacy=False):
    dec,enc=b['nvdec'],b['nvenc']
    gtx._bind_optional(dec,htx._HDR_NVDEC); gtx._bind_optional(enc,htx._HDR_NVENC)
    gtx._bind_optional(enc,htx._P010_NVENC)
    dec.nvdec_bridge_open_p010.argtypes=dec.nvdec_bridge_open_hdr.argtypes
    dec.nvdec_bridge_open_p010.restype=dec.nvdec_bridge_open_hdr.restype
    decoder=C.c_void_p(); encoder=C.c_void_p(); cuda=htx._Cuda(); pushed=False
    try:
        code=dec.nvdec_bridge_open_p010(os.fsencode(source),C.byref(decoder))
        assert code==0,dec.nvdec_bridge_last_error()
        pts=[]
        for i in range(4):
            frame=gtx._DeviceFrame(C.sizeof(gtx._DeviceFrame))
            assert dec.nvdec_bridge_next(decoder,C.byref(frame))==1
            if i==0:
                cuda.push(frame.cuda_context); pushed=True
                md=htx.HdrMetadata(C.sizeof(htx.HdrMetadata))
                assert dec.nvdec_bridge_get_hdr_metadata(decoder,C.byref(md))==0
                caps=htx.P010Capability(C.sizeof(htx.P010Capability))
                assert enc.nvenc_p010_capability(frame.cuda_context,frame.width,frame.height,codec.encode(),C.byref(caps))==0
                assert caps.supported==1,htx._fields(caps)
                opts=htx.HdrOptions(C.sizeof(htx.HdrOptions))
                opts.codec=codec.encode(); opts.profile=htx.P010_PROFILES[codec].encode()
                opts.preset=b'p4';opts.tune=b'hq';opts.rate_control=b'vbr';opts.constant_quality=20
                attach=C.byref(md) if hdr else None
                args=(frame.cuda_context,frame.width,frame.height,25,1,frame.time_base_num,
                      frame.time_base_den,os.fsencode(out),C.byref(opts),attach,C.byref(encoder))
                if legacy:
                    fn=enc.nvenc_hdr_open if hdr else enc.nvenc_sdr10_open
                    fn.argtypes=enc.nvenc_hdr_open.argtypes; fn.restype=C.c_int32
                    code=fn(*args)
                else: code=htx.open_p010_encoder(enc,not hdr,*args)
                assert code==0,enc.nvenc_last_error()
            assert enc.nvenc_encode_p010(encoder,frame.y_device_ptr,frame.y_pitch_bytes,
                       frame.uv_device_ptr,frame.uv_pitch_bytes,frame.pts)==0,enc.nvenc_last_error()
            pts.append(frame.pts)
        assert enc.nvenc_finish(encoder)==0,enc.nvenc_last_error()
        counters=htx.HdrCounters(C.sizeof(htx.HdrCounters)); contract=htx.HdrContract(C.sizeof(htx.HdrContract))
        status=htx.P010MetadataStatus(C.sizeof(htx.P010MetadataStatus))
        assert enc.nvenc_hdr_get_counters(encoder,C.byref(counters))==0
        assert enc.nvenc_p010_get_contract(encoder,C.byref(contract))==0
        assert enc.nvenc_p010_get_metadata_status(encoder,C.byref(status))==0
        assert counters.host_pixel_bytes==counters.rgb_conversion_bytes==0
        result=dict(counters=htx._fields(counters),contract=htx._fields(contract),
                    metadata=htx._fields(status),submitted_pts=pts)
    finally:
        if encoder: enc.nvenc_close(encoder)
        if pushed: cuda.pop()
        if decoder: dec.nvdec_bridge_close(decoder)
    facts=probe(out); streams=facts['streams']; video=next(s for s in streams if s['codec_type']=='video')
    frames=[x for x in facts['packets_and_frames'] if x['type']=='frame' and x['media_type']=='video']
    assert len(frames)==4
    assert video['codec_name']==('av1' if codec=='av1_nvenc' else 'hevc')
    assert video['pix_fmt']=='yuv420p10le' and video['profile']==('Main' if codec=='av1_nvenc' else 'Main 10')
    assert [video[k] for k in ('color_primaries','color_transfer','color_space','color_range')]==(
        ['bt2020','smpte2084','bt2020nc','tv'] if hdr else ['bt709','bt709','bt709','tv'])
    side=[side for f in frames for side in f.get('side_data_list',[])]
    if hdr: assert any(s['side_data_type']=='Mastering display metadata' for s in side)
    else: assert not any('Mastering' in s['side_data_type'] or 'Content light' in s['side_data_type'] for s in side)
    result['probe']=facts
    return result


def capabilities(b, source):
    dec,enc=b['nvdec'],b['nvenc']
    gtx._bind_optional(dec,htx._HDR_NVDEC);gtx._bind_optional(enc,htx._P010_NVENC)
    decoder=C.c_void_p()
    assert dec.nvdec_bridge_open_hdr(os.fsencode(source),C.byref(decoder))==0
    try:
        frame=gtx._DeviceFrame(C.sizeof(gtx._DeviceFrame))
        assert dec.nvdec_bridge_next(decoder,C.byref(frame))==1
        result={}
        for codec in ('hevc_nvenc','av1_nvenc'):
            cap=htx.P010Capability(C.sizeof(htx.P010Capability))
            assert enc.nvenc_p010_capability(frame.cuda_context,frame.width,frame.height,
                                           codec.encode(),C.byref(cap))==0
            assert cap.supported==1,enc.nvenc_last_error()
            result[codec]=htx._fields(cap)
        cap=htx.P010Capability(C.sizeof(htx.P010Capability))
        assert enc.nvenc_p010_capability(frame.cuda_context,frame.width,frame.height,b'h264_nvenc',C.byref(cap))<0
        return result
    finally:dec.nvdec_bridge_close(decoder)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dll',type=Path,default=ROOT/'native/nvenc_bridge.dll')
    ap.add_argument('--output',type=Path,default=ROOT/'build/p010-codec-check')
    ap.add_argument('--capability-only',action='store_true')
    ap.add_argument('--cpu-only',action='store_true');args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    sdr=args.output/'sdr709_hd_source.mkv'
    if not sdr.exists():
        subprocess.run([pipeline._bundle_ffmpeg(),'-v','error','-y','-f','lavfi','-i',
            'testsrc2=size=1280x720:rate=25','-frames:v','6','-c:v','libx265','-preset','ultrafast',
            '-x265-params','log-level=error:colorprim=bt709:transfer=bt709:colormatrix=bt709','-pix_fmt','yuv420p10le','-color_primaries','bt709',
            '-color_trc','bt709','-colorspace','bt709','-color_range','tv',str(sdr)],check=True)
    b=load(args.dll); result={}
    try:
        if args.cpu_only:
            result['cpu']=cpu_gate(b['nvenc'])
            (args.output/'av1-cpu-final.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print('AV1 metadata CPU checks passed',flush=True);return
        if args.capability_only:
            result=capabilities(b,ROOT/'build/hdr_probe/hdr_valid_src.mkv')
            (args.output/'driver-capabilities.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print('HEVC/AV1 real-driver P010 opens passed (no files encoded)',flush=True)
            return
        result['cpu']=cpu_gate(b['nvenc']); print('AV1 metadata CPU checks passed',flush=True)
        for hdr in (False,True):
            source=ROOT/'build/hdr_probe/hdr_valid_src.mkv' if hdr else sdr
            for codec in ('hevc_nvenc','av1_nvenc'):
                label=f'{codec}_{"hdr" if hdr else "sdr"}'
                result[label]=boundary(b,source,args.output/f'{label}.mkv',codec,hdr,legacy=codec=='hevc_nvenc')
                print(label+' passed',flush=True)
                (args.output/'boundary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    finally:
        htx.release_loader_cookies(b)
        if not args.capability_only and not args.cpu_only:
            (args.output/'boundary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__': main()
