"""Read-only, bounded environment checks. DLL checks never open render sessions.

Detailed checks run in a disposable process so a broken CUDA/runtime library
cannot poison the renderer's existing sessions. Ready dependencies are not a
claim that a particular SDK effect or codec has executed successfully.
"""
import ctypes
from functools import lru_cache
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE_MODULES = (('numpy', 'NumPy'), ('cv2', 'OpenCV'), ('fastapi', 'FastAPI'),
                ('uvicorn', 'Uvicorn'), ('websockets', 'WebSocket'), ('pydantic', 'Pydantic'))
BRIDGES = {
    'dlss_gpu_host.dll': ('dlss_gpu_host_abi_version', 3),
    'nvof_d3d12_interop.dll': ('nvof_d3d12_abi_version', 1),
    'nvof_bridge.dll': ('nvof_bridge_get_abi_version', 3),
    'nvdec_bridge.dll': ('nvdec_bridge_abi_version', 3),
    'nvenc_bridge.dll': ('nvenc_abi_version', 2),
    'nv12_rgba_bridge.dll': ('nv12_rgba_abi_version', 3),
    'gpu_lanczos_bridge.dll': ('gpu_lanczos_abi_version', 2),
    'gpu_residual_bridge.dll': ('gpu_residual_abi_version', 1),
    'gpu_low_residual_bridge.dll': ('gpu_low_residual_abi_version', 1),
    'gpu_multiscale_fusion_bridge.dll': ('gpu_fusion_abi_version', 1),
    'gpu_post_control_bridge.dll': ('gpu_post_control_abi_version', 1),
    'gpu_temporal_bridge.dll': ('gpu_temporal_abi_version', 1),
    'gpu_hdr_scrgb_bridge.dll': ('gpu_hdr_abi_version', 1),
    'gpu_preview_readback.dll': ('gpu_preview_open', None),
    'gpu_preview_display.dll': ('gpu_preview_display_open', None),
    'gpu_rgba_exchange.dll': ('gpu_rgba_exchange_open', None),
    'rtx_vsr_bridge.dll': ('rtx_vsr_abi_version', 1),
    'fidelityfx_flow_bridge.dll': ('ffx_flow_abi_version', 1),
}


def item(key, label, status, detail='', action=''):
    return dict(id=key, label=label, status=status, detail=detail, action=action)


def bootstrap():
    checks = []
    for module, label in BASE_MODULES:
        try:
            value = importlib.import_module(module)
            checks.append(item(module, label, 'ok', str(getattr(value, '__version__', '已加载'))))
        except Exception as exc:
            checks.append(item(module, label, 'missing', str(exc), '重新安装完整运行环境'))
    return {'ready': all(c['status'] == 'ok' for c in checks), 'checks': checks,
            'python': platform.python_version(), 'executable': sys.executable,
            'isolated': bool(sys.flags.isolated)}


def quick_system_info():
    info = {'gpu': None, 'vram_total_gb': None, 'vram_used_gb': None,
            'gpu_error': None, 'os': sys.platform, 'python_version': platform.python_version(),
            'backend_ready': True}
    try:
        import torch
        if not torch.cuda.is_available():
            info['gpu_error'] = '当前 PyTorch 未检测到可用的 CUDA 设备'
        else:
            info['gpu'] = torch.cuda.get_device_name(0)
            info['vram_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            # Allocated bytes are already included in reserved bytes. This is this
            # process's allocator, not the whole GPU's usage.
            info['vram_used_gb'] = round(torch.cuda.memory_reserved(0) / 1024**3, 2)
    except Exception as exc:
        info['gpu_error'] = str(exc)
    info['vram_scope'] = 'process_reserved'
    info['service'] = 'lunarc'
    info['backend_api'] = 1
    info['runtime_id'] = runtime_id()
    return info


@lru_cache(maxsize=1)
def runtime_id():
    path = Path(os.environ.get('LUNARC_RUNTIME_DIR') or ROOT / 'runtime') / 'manifest.json'
    if not path.is_file():
        return None
    digest = hashlib.sha256(path.read_bytes())
    for module in sorted(ROOT.glob('*.py')):
        digest.update(module.name.encode('utf-8') + b'\0')
        digest.update(module.read_bytes())
    return digest.hexdigest()


def run_tool(path, *args):
    return subprocess.run([str(path), *args], capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=12,
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))


def worker():
    import bundle_paths
    import pipeline
    checks = bootstrap()['checks']
    gpu = quick_system_info()
    checks.append(item('python', 'Python', 'ok', f'{platform.python_version()} · {platform.machine()}'))
    checks.append(item('cuda', 'CUDA 运算', 'missing', gpu['gpu_error'] or '', '安装支持此显卡的 NVIDIA 驱动'))
    try:
        import torch
        if torch.cuda.is_available():
            # Actually execute and synchronize, not merely import a CUDA wheel.
            sample = torch.ones(8, device='cuda') + 1
            torch.cuda.synchronize()
            if sample.sum().item() != 16:
                raise RuntimeError('CUDA 运算结果错误')
            checks[-1] = item('cuda', 'CUDA 运算', 'ok',
                              f"{gpu['gpu']} · PyTorch {torch.__version__} · CUDA {torch.version.cuda}")
    except Exception as exc:
        checks[-1] = item('cuda', 'CUDA 运算', 'error', str(exc), '检查 NVIDIA 驱动与随软件携带的 CUDA 运行库')
    try:
        driver = ctypes.WinDLL('nvcuda.dll')
        version = ctypes.c_int()
        driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        result = driver.cuDriverGetVersion(ctypes.byref(version))
        if result:
            raise RuntimeError(f'cuDriverGetVersion={result}')
        checks.append(item('driver', 'NVIDIA 驱动接口', 'ok', f'CUDA Driver API {version.value}'))
    except Exception as exc:
        checks.append(item('driver', 'NVIDIA 驱动接口', 'missing', str(exc), '安装 NVIDIA 显卡驱动'))

    runtime_dir = Path(os.environ.get('LUNARC_RUNTIME_DIR') or ROOT / 'runtime')
    tools_dir = runtime_dir / 'tools'
    # Native import dependencies (shared FFmpeg + CUDA interop) need retained handles.
    for directory in (bundle_paths.native_dir(), str(runtime_dir / 'python'), str(tools_dir), os.path.dirname(pipeline._bundle_ffmpeg() or '')):
        bundle_paths.add_dll_directory(directory)
    for key, name, finder in [('ffmpeg', 'FFmpeg', pipeline._bundle_ffmpeg), ('ffprobe', 'FFprobe', pipeline._bundle_ffprobe)]:
        path = finder()
        try:
            if not path:
                raise FileNotFoundError('未找到程序')
            result = run_tool(path, '-version')
            if result.returncode:
                raise RuntimeError(result.stderr)
            checks.append(item(key, name, 'ok', result.stdout.splitlines()[0]))
            if key == 'ffmpeg':
                filters = run_tool(path, '-hide_banner', '-filters')
                required = ('zscale', 'scale', 'showinfo', 'select')
                missing = [f for f in required if not any(len(l.split()) >= 2 and l.split()[1] == f for l in filters.stdout.splitlines())]
                checks.append(item('ffmpeg_filters', '预览与色彩转换', 'missing' if missing else 'ok',
                                   '缺少：' + ', '.join(missing) if missing else 'zscale / scale / showinfo / select', '安装随软件提供的完整 FFmpeg' if missing else ''))
        except Exception as exc:
            checks.append(item(key, name, 'missing', str(exc), '重新安装随软件提供的 FFmpeg'))

    libraries = {}
    for filename, (symbol, expected) in BRIDGES.items():
        try:
            path = bundle_paths.require_native(filename, '运行此功能')
            lib = ctypes.WinDLL(path)
            fn = getattr(lib, symbol)
            actual = None
            if expected is not None:
                fn.restype = ctypes.c_uint32
                fn.argtypes = []
                actual = fn()
                if actual != expected:
                    raise RuntimeError(f'ABI {actual}，需要 {expected}')
            libraries[filename] = lib
            checks.append(item(filename, filename, 'ok', f'已加载 · ABI {actual}' if actual is not None else f'已加载 · {symbol}'))
        except Exception as exc:
            checks.append(item(filename, filename, 'missing', str(exc), '重新安装匹配版本的原生组件'))
    # All checks here are load/interface probes, not feature session initialization.
    try:
        from dlssnr_runtime import manager
        runtime = manager().status()
        error = runtime['error']
        detail = error['message'] if error else f"{runtime['active']['version']} · NVIDIA 签名有效 · SHA-256 已核对；设备支持在创建会话时验证"
        checks.append(item('dlssnr_runtime', 'DLSSNR 运行库', 'missing' if error else 'ok',
                           detail, '在“运行与诊断 → DLSSNR 运行库”导入并选择运行库' if error else ''))
    except Exception as exc:
        checks.append(item('dlssnr_runtime', 'DLSSNR 运行库', 'error', str(exc), '检查运行库缓存'))
    from native_runtime_paths import dlss_runtime, vsr_runtime, wide_vfx_runtime
    for key, label, resolver in [('dlaa_runtime', 'DLAA 运行库', dlss_runtime),
                                 ('vsr_runtime', 'RTX VSR 运行库', vsr_runtime),
                                 ('vfg_runtime', 'NVIDIA VFG 运行库', lambda: wide_vfx_runtime(framegen=True))]:
        try:
            location = resolver()
            checks.append(item(key, label, 'ok', f'已找到：{location}；模型支持在创建会话时验证'))
        except Exception as exc:
            checks.append(item(key, label, 'missing', str(exc), '安装对应 NVIDIA 运行库，或在高级设置中选择已安装目录'))
    try:
        import torchvision
        weights = [Path(bundle_paths.app_dir()) / 'models/raft_large_c_t.pth',
                   Path(os.environ.get('TORCH_HOME', str(ROOT / 'torch_home'))) / 'hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth']
        found = next((p for p in weights if p.is_file()), None)
        checks.append(item('raft', 'RAFT-Large', 'ok' if found else 'missing',
                           f'torchvision {torchvision.__version__} · {found}' if found else '未找到模型权重',
                           '' if found else '安装完整模型组件'))
    except Exception as exc:
        checks.append(item('raft', 'RAFT-Large', 'missing', str(exc), '重新安装完整 GPU 运行环境'))
    index = {c['id']: c for c in checks}
    features = []
    base = ['cuda', 'driver', 'nvdec_bridge.dll', 'nvenc_bridge.dll', 'gpu_rgba_exchange.dll']
    for key, label, deps in [
        ('dlssnr', 'DLSSNR', ['cuda', 'driver', 'dlss_gpu_host.dll', 'dlssnr_runtime']),
        ('gpu_transfer', 'GPU 解码与编码', base),
        ('hdr', '10-bit / HDR', base + ['gpu_hdr_scrgb_bridge.dll']),
        ('fidelityfx', 'AMD 光流', ['cuda', 'fidelityfx_flow_bridge.dll']),
        ('nvof', 'NVIDIA 光流', ['driver', 'nvof_bridge.dll', 'nvof_d3d12_interop.dll']),
        ('raft', 'RAFT-Large', ['cuda', 'raft']),
        ('dlaa', 'DLAA', ['cuda', 'driver', 'dlss_gpu_host.dll', 'dlaa_runtime']),
        ('vsr', 'RTX VSR', ['cuda', 'rtx_vsr_bridge.dll', 'vsr_runtime']),
        ('vfg', 'NVIDIA VFG', ['cuda', 'rtx_vsr_bridge.dll', 'vfg_runtime']),
    ]:
        missing = [index[d]['label'] for d in deps if index[d]['status'] != 'ok']
        features.append(item(key, label, 'missing' if missing else 'pending',
                             '缺少：' + '、'.join(missing) if missing else '依赖检查通过，实际设备与模型支持在运行时验证'))
    return {'schema': 1, 'checked_at': time.time(), 'gpu': gpu, 'checks': checks,
            'features': features, 'python': sys.executable,
            'isolated': bool(sys.flags.isolated), 'standalone': (runtime_dir / 'manifest.json').is_file(),
            'scope': 'imports_cuda_kernel_dll_abi_tools; effect_sessions_not_opened'}


_cache = None
_cache_key = None
_lock = threading.Lock()


def environment_report(refresh=False, overrides=None):
    global _cache, _cache_key
    overrides = {k: str(v) for k, v in (overrides or {}).items() if k in ('LUNARC_DLSS_DLL', 'LUNARC_VSR_RUNTIME') and v}
    from dlssnr_runtime import selection_key
    key = (tuple(sorted(overrides.items())), selection_key())
    with _lock:
        if _cache is not None and not refresh and key == _cache_key:
            return _cache
        _cache_key = key
        try:
            result = subprocess.run([sys.executable, str(ROOT / 'runtime_environment.py'), '--worker'],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
                env={**os.environ, **overrides},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            line = next((l[len('LUNARC_ENV='):] for l in reversed(result.stdout.splitlines()) if l.startswith('LUNARC_ENV=')), None)
            if result.returncode or not line:
                raise RuntimeError((result.stderr or result.stdout)[-2000:] or f'检测进程退出：{result.returncode}')
            _cache = json.loads(line)
        except Exception as exc:
            _cache = {'schema': 1, 'checked_at': time.time(), 'gpu': {'gpu': None},
                      'checks': [item('probe', '环境检测', 'error', str(exc), '检查运行环境或重新安装')],
                      'features': [], 'scope': 'probe_failed'}
        return _cache


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    result = bootstrap() if '--bootstrap' in sys.argv else worker()
    print('LUNARC_ENV=' + json.dumps(result, ensure_ascii=False), flush=True)
    if '--bootstrap' in sys.argv and not result['ready']:
        sys.exit(2)
