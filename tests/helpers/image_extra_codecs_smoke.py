"""CPU-only cache-format checks; no GPU, source inference or model loading."""
import ctypes as C
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from image_render import ImageResult,save_result
from image_extra_codecs import uhdr_library,check,CompressedImage,RawImage,probe_uhdr,run,codec_file,uhdr_pixels
from image_source import read_rgba


def main():
    folder=ROOT/'build/image-extra-codecs';folder.mkdir(parents=True,exist_ok=True)
    source=folder/'source.png'
    original=np.full((32,64,4),255,np.uint8)
    original[:,:,:3]=np.linspace(10,240,64,dtype=np.uint8)[None,:,None]
    original[0,0,3]=0;Image.fromarray(original).save(source)
    hdr=np.ones((32,64,4),np.float16)
    hdr[:,:,:3]=np.linspace(.05,8,64,dtype=np.float16)[None,:,None]
    hdr[0,0,3]=0
    record=ImageResult('hdr',(str(source),1,1),'settings',original,hdr,True,{},time.time())
    results=[]
    for fmt,color,alpha in [('uhdr','hdr10',False),('jxl','hdr10',True),('jxl','sdr',True)]:
        destination=folder/f'cached-{fmt}-{color}.{ "jpg" if fmt=="uhdr" else "jxl"}'
        result=save_result(record,str(destination),fmt,95,alpha,True,color)
        results.append(result)
        if fmt=='jxl':
            text=run([codec_file('jxlinfo.exe'),'-v',str(destination)],timeout=30)
            assert 'Alpha' in text
            if color=='sdr':
                loaded=read_rgba(str(destination))
                assert loaded.shape==original.shape and loaded[0,0,3]==0
            else:
                decoded=folder/'jxl-hdr-decoded.png'
                run([codec_file('djxl.exe'),str(destination),str(decoded),'--bits_per_sample=16','--num_threads=4'])
                import cv2
                image=cv2.imdecode(np.fromfile(decoded,np.uint8),cv2.IMREAD_UNCHANGED)
                assert image.dtype==np.uint16 and image.shape==(32,64,4) and image[0,0,3]==0
    # Check absolute luminance with a real UHDR FP16 decode. It is relative to
    # 203 nits, rather than scRGB's 80 nits, and must not gain an exposure factor.
    path=Path(results[0]['path']);lib=uhdr_library();data=path.read_bytes();buf=C.create_string_buffer(data)
    decoder=lib.uhdr_create_decoder()
    for name,args in {'uhdr_dec_set_out_img_format':[C.c_void_p,C.c_int],
                      'uhdr_dec_set_out_color_transfer':[C.c_void_p,C.c_int],
                      'uhdr_decode':[C.c_void_p]}.items():
        getattr(lib,name).argtypes=args;getattr(lib,name).restype=__import__('image_extra_codecs').Error
    lib.uhdr_get_decoded_image.argtypes=[C.c_void_p];lib.uhdr_get_decoded_image.restype=C.POINTER(RawImage)
    try:
        compressed=CompressedImage(C.addressof(buf),len(data),len(data),-1,-1,-1)
        check(lib.uhdr_dec_set_image(decoder,C.byref(compressed)))
        check(lib.uhdr_dec_set_out_img_format(decoder,4));check(lib.uhdr_dec_set_out_color_transfer(decoder,0))
        check(lib.uhdr_decode(decoder));image=lib.uhdr_get_decoded_image(decoder).contents
        raw=C.string_at(image.planes[0],image.stride[0]*image.h*8)
        decoded=np.frombuffer(raw,np.float16).reshape(image.h,image.stride[0],4)[:,:image.w,:3].astype(np.float32)
        expected,_=uhdr_pixels(hdr[:,:,:3]);target=expected[:,:,:3].astype(np.float32)
        # JPEG and gain-map compression are lossy. Check the bulk of the ramp,
        # away from the first endpoint, and rule out a white-level multiplier.
        ratio=decoded[:,8:]/target[:,8:]
        median=float(np.median(ratio));assert .9<median<1.1,median
    finally:lib.uhdr_release_decoder(decoder)
    metadata=probe_uhdr(path)
    evidence=dict(passed=True,dimensions=[64,32],uhdr_decoded_luminance_ratio=median,
        uhdr_metadata=metadata,cases=results)
    (folder/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(json.dumps(dict(passed=True,cases=len(results),uhdr_decoded_luminance_ratio=median,
                         uhdr_gainmap=[metadata['gainmap_width'],metadata['gainmap_height']]),ensure_ascii=False))


if __name__=='__main__':main()
