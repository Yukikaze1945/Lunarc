"""Bounded discovery of local NVIDIA runtimes; explicit overrides never fall back."""
import os
from pathlib import Path
import bundle_paths


def dlss_runtime(override=None):
    override = override or os.environ.get('LUNARC_DLSS_DLL')
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'DLAA DLL override does not exist: {path}')
        return str(path)
    cache = Path(os.environ.get('PROGRAMDATA', r'C:\ProgramData')) / 'NVIDIA/NGX/models/dlss_override/versions'
    candidates = sorted(cache.glob('*/files/*/nvngx_dlss.dll'),
                        key=lambda p: p.stat().st_mtime_ns, reverse=True) if cache.exists() else []
    candidates += [Path(bundle_paths.collected('nvngx_dlss.dll')),
                   Path(bundle_paths.app_dir()) / 'runtime/dlss/nvngx_dlss.dll']
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    raise FileNotFoundError('未找到本机 DLAA 运行库；请安装 NVIDIA DLSS 运行库或在高级中指定 DLL')


def vsr_runtime(override=None, *, _framegen=False):
    override = override or os.environ.get('LUNARC_VSR_RUNTIME')
    effect = 'nvVFXVideoFrameGeneration.dll' if _framegen else 'nvVFXVideoSuperRes.dll'
    feature = 'nvvfxvideoframegeneration' if _framegen else 'nvvfxvideosuperres'
    required = ('NVVideoEffects.dll', 'NVCVImage.dll', effect, 'nvngxruntime.dll')
    def valid(path):
        return (all((path / name).is_file() for name in required) or
                all((path / 'bin' / name).is_file() for name in ('NVVideoEffects.dll', 'NVCVImage.dll', 'nvngxruntime.dll')) and
                (path / 'features' / feature / 'bin' / effect).is_file())
    if override:
        path = Path(override).expanduser().resolve()
        if not valid(path):
            raise FileNotFoundError(f'RTX VSR runtime override is incomplete: {path}')
        return str(path)
    roots = [Path(bundle_paths.app_dir()) / 'runtime/vfx', Path(bundle_paths.app_dir()),
             Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'NVIDIA Corporation/NVIDIA Video Effects',
             Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'NVIDIA Corporation/Video Effects']
    # Portable Magpie packages at drive roots are common local deployments. No recursive disk scan.
    if os.name == 'nt':
        for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
            root = Path(f'{letter}:/')
            if root.exists():
                roots.insert(0, root / 'DLSSC/VideoFX')
                roots.extend(sorted(root.glob('Magpie*')))
    for root in roots:
        for path in (root, root / 'bin'):
            if valid(path):
                return str(path.resolve())
    raise FileNotFoundError('未找到本机 RTX VSR 运行库；显卡驱动不保证包含 Video Effects 模型，请在高级中指定已安装运行库')


def wide_vfx_runtime(override=None, *, framegen=False):
    path = Path(vsr_runtime(override, _framegen=framegen))
    feature = 'nvvfxvideoframegeneration/bin/nvVFXVideoFrameGeneration.dll' if framegen else 'nvvfxvideosuperres/include/nvVFXVideoSuperRes.h'
    if not (path / 'features' / feature).is_file() or not (path / 'bin/nvrtc64_120_0.dll').is_file():
        raise FileNotFoundError('此功能需要 VFX 1.3 SDK Core 和对应特效模块；请在高级中选择新版 SDK 根目录')
    return str(path)
