"""Optional CPU-only JPEG Ultra HDR and JPEG XL encoders.

UHDR uses libultrahdr's HDR-only API, not a pre-compressed display thumbnail.
JXL receives tagged 16-bit PQ RGB(A), following Starshot's HDR JPEG XL route.
"""
import ctypes as C
import os
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


def codec_file(name):
    import bundle_paths
    override=os.environ.get('LUNARC_IMAGE_CODECS_DIR')
    if override:
        roots=[Path(override).expanduser()]
    else:
        root=Path(bundle_paths.app_dir())
        roots=[Path(bundle_paths.contents_dir())/'image-codecs',
               root/'runtime/tools/image-codecs',root/'build/standalone-runtime/tools/image-codecs']
        runtime=os.environ.get('LUNARC_RUNTIME_DIR')
        if runtime:roots.insert(0,Path(runtime)/'tools/image-codecs')
    for folder in roots:
        path=folder/name
        if path.is_file():return str(path.resolve())
    raise FileNotFoundError('缺少图片编码组件：'+name+'；请安装 UHDR/JXL 编码组件')


def capabilities():
    result={}
    for kind,names in dict(uhdr=['uhdr.dll'],jxl=['cjxl.exe','jxlinfo.exe','djxl.exe','jxl.dll','jxl_threads.dll',
                                                      'brotlicommon.dll','brotlidec.dll','brotlienc.dll']).items():
        try:
            for name in names:codec_file(name)
            result[kind]=dict(available=True,alpha=kind=='jxl',reason='')
        except OSError as error:
            result[kind]=dict(available=False,alpha=kind=='jxl',reason=str(error))
    return result


def run(command,timeout=600):
    result=subprocess.run(command,capture_output=True,timeout=timeout,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:
        message=(result.stderr+result.stdout).decode('utf-8','replace')[-1800:]
        raise ValueError(f'图片编码组件运行失败（{result.returncode}）：{message}')
    return (result.stdout+result.stderr).decode('utf-8','replace')


def save_jxl(path,rgb,alpha,quality,*,hdr=False):
    from image_hdr_export import scrgb_to_pq16
    pixels,conversion=scrgb_to_pq16(rgb) if hdr else (rgb,{})
    with tempfile.TemporaryDirectory(prefix='.lunarc-jxl-',dir=Path(path).parent) as directory:
        source=Path(directory)/'input.pam'
        h,w=pixels.shape[:2];channels=4 if alpha is not None else 3
        depth=16 if hdr else 8
        header=f'P7\nWIDTH {w}\nHEIGHT {h}\nDEPTH {channels}\nMAXVAL {65535 if hdr else 255}\nTUPLTYPE {"RGB_ALPHA" if alpha is not None else "RGB"}\nENDHDR\n'
        with source.open('wb') as output:
            output.write(header.encode('ascii'))
            for row in range(0,h,128):
                block=np.empty((min(128,h-row),w,channels),dtype='>u2' if hdr else np.uint8)
                block[:,:,:3]=pixels[row:row+128]
                if alpha is not None:
                    block[:,:,3]=alpha[row:row+128].astype(np.uint16)*257 if hdr else alpha[row:row+128]
                output.write(block.tobytes())
        command=[codec_file('cjxl.exe'),str(source),str(path),'-q',str(quality),'-e','5',
            '--num_threads=8','--container=1','--premultiply=0','-x',
            'color_space=RGB_D65_202_Rel_PeQ' if hdr else 'color_space=RGB_D65_SRG_Rel_SRG']
        if hdr:command+=['--intensity_target=10000']
        run(command)
    info=run([codec_file('jxlinfo.exe'),'-v',str(path)],timeout=30)
    expected=[f'{w}x{h}',f'{depth}-bit RGB','PQ transfer function' if hdr else 'sRGB transfer function']
    if hdr:expected+=['Rec.2100 primaries','intensity_target: 10000.']
    if alpha is not None:expected+=['Alpha','Non-premultiplied']
    if any(value not in info for value in expected):
        raise ValueError('JXL 的尺寸、精度、透明度或色彩声明与请求不符，未保存')
    return dict(bit_depth=depth,color_primaries='bt2020' if hdr else 'bt709',
        color_transfer='smpte2084' if hdr else 'iec61966-2-1',color_space='rgb',
        color_range='pc',chroma='RGB',has_alpha=alpha is not None,
        hdr_encoding='PQ RGB16' if hdr else None,lossless_encoding=quality==100,
        hdr_mapped_to_sdr=False,conversion=conversion)


class Error(C.Structure):
    _fields_=[('code',C.c_int),('has_detail',C.c_int),('detail',C.c_char*256)]


class RawImage(C.Structure):
    _fields_=[('fmt',C.c_int),('cg',C.c_int),('ct',C.c_int),('range',C.c_int),
        ('w',C.c_uint),('h',C.c_uint),('planes',C.c_void_p*3),('stride',C.c_uint*3)]


class CompressedImage(C.Structure):
    _fields_=[('data',C.c_void_p),('data_sz',C.c_size_t),('capacity',C.c_size_t),
              ('cg',C.c_int),('ct',C.c_int),('range',C.c_int)]


class GainMetadata(C.Structure):
    _fields_=[(name,C.c_float*3) for name in ('max_boost','min_boost','gamma','offset_sdr','offset_hdr')]+[
        ('capacity_min',C.c_float),('capacity_max',C.c_float),('use_base_cg',C.c_int)]


def uhdr_library():
    path=codec_file('uhdr.dll')
    library=C.CDLL(path)
    for name,args in {
        'uhdr_enc_set_raw_image':[C.c_void_p,C.POINTER(RawImage),C.c_int],
        'uhdr_enc_set_quality':[C.c_void_p,C.c_int,C.c_int],
        'uhdr_enc_set_using_multi_channel_gainmap':[C.c_void_p,C.c_int],
        'uhdr_enc_set_gainmap_scale_factor':[C.c_void_p,C.c_int],
        'uhdr_enc_set_target_display_peak_brightness':[C.c_void_p,C.c_float],
        'uhdr_encode':[C.c_void_p],
        'uhdr_dec_set_image':[C.c_void_p,C.POINTER(CompressedImage)],
        'uhdr_dec_probe':[C.c_void_p],
    }.items():
        function=getattr(library,name);function.argtypes=args;function.restype=Error
    for name in ('uhdr_create_encoder','uhdr_create_decoder'):
        getattr(library,name).argtypes=[];getattr(library,name).restype=C.c_void_p
    for name in ('uhdr_release_encoder','uhdr_release_decoder'):
        getattr(library,name).argtypes=[C.c_void_p];getattr(library,name).restype=None
    library.uhdr_get_encoded_stream.argtypes=[C.c_void_p]
    library.uhdr_get_encoded_stream.restype=C.POINTER(CompressedImage)
    library.uhdr_dec_get_gainmap_metadata.argtypes=[C.c_void_p]
    library.uhdr_dec_get_gainmap_metadata.restype=C.POINTER(GainMetadata)
    for name in ('image_width','image_height','gainmap_width','gainmap_height'):
        function=getattr(library,'uhdr_dec_get_'+name);function.argtypes=[C.c_void_p];function.restype=C.c_int
    library.is_uhdr_image.argtypes=[C.c_void_p,C.c_int];library.is_uhdr_image.restype=C.c_int
    return library


def check(error):
    if error.code:
        raise ValueError('UHDR 编码组件：'+(error.detail.decode('utf-8','replace') if error.has_detail else str(error.code)))


# D65 transforms derived from the standard chromaticities. Convert before
# compressing negative channels, as in Starshot's shared Display-P3 working node.
_SRGB_XYZ=np.array([[.4123907993,.3575843394,.1804807884],
    [.2126390059,.7151686788,.0721923154],[.0193308187,.1191947798,.9505321522]])
_P3_XYZ=np.array([[.4865709486,.2656676932,.1982172852],
    [.2289745641,.6917385218,.0792869141],[0,.0451133819,1.0439443689]])
SCRGB_TO_P3=np.linalg.solve(_P3_XYZ,_SRGB_XYZ).astype(np.float32)
REFERENCE_WHITE_NITS=203.


def uhdr_pixels(rgb):
    if rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('UHDR 需要线性 RGB 图像')
    output=np.ones((*rgb.shape[:2],4),np.float16);compressed=above=0;peak=REFERENCE_WHITE_NITS
    for row in range(0,rgb.shape[0],128):
        values=rgb[row:row+128].astype(np.float32)
        if not np.isfinite(values).all():raise ValueError('HDR 图像包含非有限亮度值，未保存')
        values=values@SCRGB_TO_P3.T
        y=values@_P3_XYZ[1].astype(np.float32)
        # One chroma scale per pixel preserves luminance and the direction to gray.
        pull=np.max(np.clip(np.maximum(-values,0)/np.maximum(y[:,:,None]-values,1e-6),0,1),axis=2)
        compressed+=int(np.count_nonzero(pull))
        values=values+(np.maximum(y,0)[:,:,None]-values)*pull[:,:,None]
        above+=int(np.count_nonzero(values>125))
        values=np.clip(values,0,125)
        peak=max(peak,float(values.max())*80)
        output[row:row+128,:,:3]=values*(80/REFERENCE_WHITE_NITS)
    return output,dict(scrgb_nits_per_unit=80,codec_reference_white_nits=REFERENCE_WHITE_NITS,
        working_gamut='Display-P3',gamut_compressed_pixels=compressed,
        above_10000_nits_components_clipped=above,codec_target_peak_nits=peak)


def probe_uhdr(path,library=None):
    library=library or uhdr_library();data=Path(path).read_bytes();buffer=C.create_string_buffer(data)
    if len(data)>2**31-1 or not library.is_uhdr_image(buffer,len(data)):
        raise ValueError('UHDR 文件未包含有效 HDR 增益图，未保存')
    decoder=library.uhdr_create_decoder()
    if not decoder:raise ValueError('无法创建 UHDR 检查器')
    try:
        image=CompressedImage(C.addressof(buffer),len(data),len(data),-1,-1,-1)
        check(library.uhdr_dec_set_image(decoder,C.byref(image)));check(library.uhdr_dec_probe(decoder))
        metadata=library.uhdr_dec_get_gainmap_metadata(decoder)
        if not metadata:raise ValueError('UHDR 缺少增益图元数据，未保存')
        info=metadata.contents
        values=[*info.max_boost,*info.min_boost,*info.gamma,info.capacity_min,info.capacity_max]
        if not np.isfinite(values).all() or min(values)<=0 or info.capacity_min>info.capacity_max:
            raise ValueError('UHDR 增益图元数据无效，未保存')
        return dict(width=library.uhdr_dec_get_image_width(decoder),height=library.uhdr_dec_get_image_height(decoder),
            gainmap_width=library.uhdr_dec_get_gainmap_width(decoder),gainmap_height=library.uhdr_dec_get_gainmap_height(decoder),
            capacity_min=info.capacity_min,capacity_max=info.capacity_max)
    finally:library.uhdr_release_decoder(decoder)


def save_uhdr(path,rgb,quality):
    library=uhdr_library();pixels,conversion=uhdr_pixels(rgb);h,w=pixels.shape[:2]
    encoder=library.uhdr_create_encoder()
    if not encoder:raise ValueError('无法创建 UHDR 编码器')
    try:
        image=RawImage(4,1,0,1,w,h,(C.c_void_p*3)(pixels.ctypes.data,0,0),(C.c_uint*3)(w,0,0))
        check(library.uhdr_enc_set_raw_image(encoder,C.byref(image),0))
        for label in (2,3):check(library.uhdr_enc_set_quality(encoder,quality,label))
        check(library.uhdr_enc_set_using_multi_channel_gainmap(encoder,1))
        check(library.uhdr_enc_set_gainmap_scale_factor(encoder,1))
        check(library.uhdr_enc_set_target_display_peak_brightness(encoder,conversion['codec_target_peak_nits']))
        check(library.uhdr_encode(encoder));encoded=library.uhdr_get_encoded_stream(encoder)
        if not encoded or not encoded.contents.data_sz:raise ValueError('UHDR 编码未返回图像')
        Path(path).write_bytes(C.string_at(encoded.contents.data,encoded.contents.data_sz))
    finally:library.uhdr_release_encoder(encoder)
    metadata=probe_uhdr(path,library)
    if (metadata['width'],metadata['height'])!=(w,h) or min(metadata['gainmap_width'],metadata['gainmap_height'])<1:
        raise ValueError('UHDR 基图或增益图尺寸与请求不符，未保存')
    return dict(bit_depth=8,color_primaries='display-p3',color_transfer='sRGB + HDR gain map',
        color_space='jpeg',color_range='pc',has_alpha=False,hdr_encoding='JPEG gain map',
        hdr_mapped_to_sdr=False,conversion=conversion,gainmap=metadata,
        sdr_base='libultrahdr tone mapping; HDR retained in gain map')
