"""Still-image decoding and one reusable RGB GPU upload; no video conversion."""
import ctypes as C
import io
import os
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps, features

EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff', '.avif', '.heic', '.heif')


def heif_plugin():
    try:
        import pillow_heif
    except ImportError as error:
        raise ValueError('当前运行环境缺少 pillow-heif 图片编码组件，请安装项目依赖后重启') from error
    pillow_heif.register_heif_opener()
    return pillow_heif


def export_formats():
    heic = False
    try:
        heic = bool(heif_plugin().libheif_info().get('HEIF'))
    except ValueError:
        pass
    return dict(png=True, jpeg=True, webp=features.check('webp'),
                avif=features.check('avif'), heic=heic)


def is_image(path):
    return os.path.splitext(path)[1].lower() in EXTENSIONS


def read_rgba(path):
    # Pillow can silently reduce a 16-bit RGB PNG to 8-bit. Check before converting.
    extension = os.path.splitext(path)[1].lower()
    if extension in ('.heic','.heif'):
        source = heif_plugin().open_heif(path, convert_hdr_to_8bit=False)
        if source.info.get('bit_depth',8) > 8:
            raise ValueError('当前图片入口支持 8-bit 图片；高位深 HEIC 不会自动降为 8-bit')
    elif extension == '.avif':
        # Pillow's AVIF decoder always returns 8-bit; verify original precision first.
        import pipeline
        info = pipeline.probe_pixel_info(path)
        if not info.get('pix_fmt'):
            raise ValueError('无法确认 AVIF 输入位深')
        if info.get('bit_depth',8)>8 or info.get('is_hdr'):
            raise ValueError('当前图片入口支持 8-bit SDR；高位深或 HDR AVIF 不会自动降精度')
    else:
        raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError('无法读取图片')
        if raw.dtype != np.uint8:
            raise ValueError('当前图片入口支持 8-bit 图片；16-bit 图片尚未接入，不会自动降为 8-bit')
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        alpha = image.convert('RGBA').getchannel('A')
        profile = opened.info.get('icc_profile')
        if profile:
            image = ImageCms.profileToProfile(image if image.mode in ('RGB','CMYK','L') else image.convert('RGB'),
                ImageCms.ImageCmsProfile(io.BytesIO(profile)), ImageCms.createProfile('sRGB'),
                outputMode='RGB')
        else:
            image = image.convert('RGB')
        image.putalpha(alpha)
        return np.asarray(image).copy()


def probe(path):
    rgba = read_rgba(path)
    h, w = rgba.shape[:2]
    return dict(media_type='image', width=(w+7)//8*8, height=(h+7)//8*8,
        image_width=w, image_height=h, pix_fmt='rgba', video_stream_count=1,
        color_range='pc', color_space='gbr', color_transfer='iec61966-2-1',
        color_primaries='bt709', r_frame_rate='1/1', audio_streams=0,
        subtitle_streams=0, nb_chapters=0, metadata=False, codec_name='image')


class Capture:
    def __init__(self, path):
        self.rgba = read_rgba(path)
        self.position = 0

    def isOpened(self):
        return self.rgba is not None

    def get(self, key):
        h, w = self.rgba.shape[:2]
        return {cv2.CAP_PROP_FRAME_COUNT: 1, cv2.CAP_PROP_FPS: 1,
            cv2.CAP_PROP_FRAME_WIDTH: w, cv2.CAP_PROP_FRAME_HEIGHT: h}.get(key, 0)

    def set(self, key, value):
        if key == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return True

    def read(self):
        if self.position != 0 or self.rgba is None:
            return False, None
        self.position = 1
        return True, cv2.cvtColor(self.rgba, cv2.COLOR_RGBA2BGR)

    def release(self):
        self.rgba = None


class DeviceImage:
    def __init__(self, path):
        self.context, self.pointer = C.c_void_p(), C.c_uint64()
        self.cuda = C.WinDLL('nvcuda.dll')
        for name, args in {
            'cuInit': [C.c_uint], 'cuDeviceGet': [C.POINTER(C.c_int), C.c_int],
            'cuCtxCreate_v2': [C.POINTER(C.c_void_p), C.c_uint, C.c_int],
            'cuCtxDestroy_v2': [C.c_void_p], 'cuMemAlloc_v2': [C.POINTER(C.c_uint64), C.c_size_t],
            'cuMemFree_v2': [C.c_uint64],
            'cuMemcpyHtoD_v2': [C.c_uint64, C.c_void_p, C.c_size_t],
        }.items():
            getattr(self.cuda, name).argtypes = args
        rgba = read_rgba(path)
        h, w = rgba.shape[:2]
        self.width, self.height = (w+7)//8*8, (h+7)//8*8
        image = np.pad(rgba, ((0,self.height-h), (0,self.width-w), (0,0)), mode='edge')
        image[:, :, 3] = 255  # Models process colour; export restores source transparency.
        self.host_upload_bytes = image.nbytes
        self.pitch = self.width * 4
        try:
            self.check(self.cuda.cuInit(0))
            device = C.c_int()
            self.check(self.cuda.cuDeviceGet(C.byref(device), 0))
            self.check(self.cuda.cuCtxCreate_v2(C.byref(self.context), 0, device))
            self.check(self.cuda.cuMemAlloc_v2(C.byref(self.pointer), image.nbytes))
            self.check(self.cuda.cuMemcpyHtoD_v2(self.pointer, image.ctypes.data, image.nbytes))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f'图片 GPU 上传失败，CUDA 错误 {code}')

    def frame(self, frame):
        frame.width, frame.height = self.width, self.height
        frame.cuda_context = self.context.value
        frame.time_base_num = frame.time_base_den = 1
        frame.pts = 0
        frame.image_rgba = (self.pointer.value, self.pitch)
        return 1

    def close(self):
        if self.pointer.value:
            self.cuda.cuMemFree_v2(self.pointer)
            self.pointer.value = 0
        if self.context.value:
            self.cuda.cuCtxDestroy_v2(self.context)
            self.context.value = None


def output_pixels(bgr, source, settings):
    """Trim source padding; fixed VSR output resolutions need no trimming."""
    import native_effects
    from vsr_config import configuration
    fixed=any(native_effects.stage_kind(stage)=='vsr' and
        not configuration(native_effects.stage_settings(settings,stage))['same_size'] and
        native_effects.stage_settings(settings,stage).get('vsr_output_mode','resolution')=='resolution'
        for stage in native_effects.stages(settings))
    if fixed:
        return bgr
    w = round(bgr.shape[1] * source['image_width'] / source['width'])
    h = round(bgr.shape[0] * source['image_height'] / source['height'])
    return bgr[:max(1,h), :max(1,w)]


FORMATS = {'png': ('.png', 'PNG'), 'jpeg': ('.jpg', 'JPEG'), 'webp': ('.webp', 'WEBP'),
           'avif': ('.avif', 'AVIF'), 'heic': ('.heic', 'HEIF')}


def validate_destination(path, output_path, fmt, quality, overwrite=False):
    if fmt not in FORMATS or not 1 <= quality <= 100:
        raise ValueError('图片格式或质量参数无效')
    if fmt == 'heic':
        heif_plugin()
    if fmt == 'avif' and not features.check('avif'):
        raise ValueError('当前 Pillow 缺少 AVIF 编码支持，请安装项目依赖后重启')
    if os.path.splitext(output_path)[1].lower() not in (FORMATS[fmt][0], '.jpeg' if fmt == 'jpeg' else FORMATS[fmt][0]):
        raise ValueError('文件扩展名与导出格式不一致')
    if os.path.normcase(os.path.realpath(path)) == os.path.normcase(os.path.realpath(output_path)):
        raise ValueError('不能覆盖原图片')
    if not os.path.isdir(os.path.dirname(os.path.abspath(output_path))):
        raise ValueError('保存目录不存在')
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError('目标文件已存在；请更换文件名或开启覆盖')


def save_image(path, rgba, output_path, fmt, quality=95, preserve_alpha=True, overwrite=False):
    validate_destination(path, output_path, fmt, quality, overwrite)
    image = Image.fromarray(rgba)
    source = read_rgba(path)
    if preserve_alpha and fmt != 'jpeg' and np.any(source[:, :, 3] != 255):
        alpha = Image.fromarray(source[:, :, 3]).resize(image.size, Image.Resampling.BILINEAR)
        image.putalpha(alpha)
    elif fmt == 'jpeg' and np.any(source[:, :, 3] != 255):
        alpha = Image.fromarray(source[:, :, 3]).resize(image.size, Image.Resampling.BILINEAR)
        image = Image.composite(image, Image.new('RGB', image.size, 'white'), alpha)
    parent = os.path.dirname(os.path.abspath(output_path))
    fd, temporary = tempfile.mkstemp(prefix='.lunarc-image-', dir=parent)
    os.close(fd)
    try:
        opts = {'quality':quality} if fmt != 'png' else {}
        opts['icc_profile'] = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        if fmt == 'jpeg': opts['subsampling'] = 0
        if fmt == 'webp': opts.update(lossless=quality == 100, exact=True)
        if fmt == 'avif': opts.update(subsampling='4:4:4',speed=6,max_threads=8)
        if fmt == 'heic':
            opts.update(chroma=444,color_primaries=1,transfer_characteristic=13,full_range_flag=1)
            if quality == 100: opts.update(quality=-1,matrix_coefficients=0)
        image.save(temporary, format=FORMATS[fmt][1], **opts)
        if overwrite:
            os.replace(temporary, output_path)
        else:
            # Atomic no-overwrite publication, including a file created during render.
            os.link(temporary, output_path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return dict(path=output_path, width=image.width, height=image.height, format=fmt,
                bytes=os.path.getsize(output_path), bit_depth=8, color_mode='sRGB SDR')
