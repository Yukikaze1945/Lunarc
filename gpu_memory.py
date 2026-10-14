"""Owner-thread memory samples. DXGI usage and allocated resources are separate scopes."""
import copy
import ctypes as C
import sys
import threading
import time

_lock = threading.Lock()
_latest = None
_library = None
_last_sample = 0.0


class MemoryInfo(C.Structure):
    _fields_ = [('struct_size', C.c_uint32)] + [(key, C.c_uint64) for key in
        ('adapter_luid', 'local_usage', 'local_budget', 'nonlocal_usage', 'nonlocal_budget')]


class ResourceInfo(C.Structure):
    _fields_ = [('struct_size', C.c_uint32)] + [(key, C.c_uint64) for key in
        ('color_bytes', 'depth_bytes', 'motion_bytes', 'output_bytes', 'staging_bytes')]


def process_memory(device):
    global _library
    if _library is None:
        import bundle_paths
        lib = C.WinDLL(bundle_paths.require_native('dlss_gpu_host.dll', 'GPU memory diagnostics'))
        lib.dlss_gpu_memory_query.argtypes = [C.c_void_p, C.POINTER(MemoryInfo)]
        lib.dlss_gpu_memory_query.restype = C.c_int32
        _library = lib
    info = MemoryInfo(struct_size=C.sizeof(MemoryInfo))
    code = _library.dlss_gpu_memory_query(device, C.byref(info))
    if code:
        raise RuntimeError(f'DXGI memory query failed: 0x{code & 0xffffffff:08x}')
    result = {key: int(getattr(info, key)) for key, _ in info._fields_[1:]}
    result['local_headroom'] = max(0, info.local_budget-info.local_usage)
    return result


def resource_rows(root):
    rows, seen = [], set()
    def visit(owner, label, depth=0):
        if owner is None or id(owner) in seen or depth > 12:
            return
        seen.add(id(owner))
        if isinstance(owner, dict):
            for key, value in owner.items():
                if key in ('live', 'bridge', 'low_residual', 'flow_provider'):
                    visit(value, label+'.'+key, depth+1)
            return
        lib, handle = getattr(owner, '_lib', None), getattr(owner, '_handle', None)
        row = None
        if handle and lib and callable(getattr(owner, 'transfer_counters', None)):
            fn = getattr(lib, 'dlss_gpu_host_resource_info', None)
            if fn:
                fn.argtypes, fn.restype = [C.c_void_p, C.POINTER(ResourceInfo)], C.c_int32
                info = ResourceInfo(struct_size=C.sizeof(ResourceInfo))
                if fn(handle, C.byref(info)) == 0:
                    detail = {key: int(getattr(info,key)) for key,_ in info._fields_[1:]}
                    row = dict(owner=label, kind='DLAA' if getattr(owner,'native_dlaa',False) else 'DLSSNR',
                        session_id=getattr(owner,'session_id',None), allocated_device_bytes=sum(
                            detail[key] for key in ('color_bytes','depth_bytes','motion_bytes','output_bytes')),
                        host_staging_bytes=detail['staging_bytes'], resources=detail,
                        scope='session frame textures; excludes NGX internals and transport')
        allocated = getattr(owner,'allocated_device_bytes',None)
        if row is None and callable(allocated):
            row = dict(owner=label, kind=owner.__class__.__name__, allocated_device_bytes=allocated())
        if row is None and owner.__class__.__name__ == 'Exchange':
            fn = getattr(owner.lib,'gpu_rgba_exchange_allocated_bytes',None)
            if fn and owner.handle:
                fn.argtypes,fn.restype=[C.c_void_p],C.c_uint64
                row=dict(owner=label,kind='Exchange',allocated_device_bytes=int(fn(owner.handle)))
        if row is None and owner.__class__.__name__ in ('HdrFusionBridge','GpuFusionBridge') and lib and handle:
            from fusion_diagnostics import read_solver_diagnostics
            report=read_solver_diagnostics(lib,handle)
            if report:
                row=dict(owner=label,kind='Fusion',allocated_device_bytes=report['workspace_committed_bytes'],
                         logical_bytes=report['workspace_logical_bytes'],scope=report['workspace_scope'])
        if row is None and getattr(owner,'fusion_handle',None):
            from fusion_diagnostics import read_solver_diagnostics
            report=read_solver_diagnostics(owner.b['fusion'],owner.fusion_handle)
            if report:
                row=dict(owner=label,kind='Fusion',allocated_device_bytes=report['workspace_committed_bytes'],
                         logical_bytes=report['workspace_logical_bytes'],scope=report['workspace_scope'])
        if row:
            rows.append(row)
        for name in ('live','owner','effect','bridge','decode_bridge','encode_bridge','guide_bridge',
                     'input','output','input_effect','output_effect','truehdr_tail','temporal_control',
                     'core','exchange','fusion','inline_nr','chain','native_exchange','display_exchange'):
            visit(getattr(owner,name,None),label+'.'+name,depth+1)
        for name in ('effects','extra_layers'):
            for index,item in enumerate(getattr(owner,name,[]) or []):
                visit(item,f'{label}.{name}.{index}',depth+1)
    visit(root,'chain')
    return rows


def capture(chain, live, *, phase, preview_cache=None, force=False):
    global _latest, _last_sample
    now=time.monotonic()
    if not force and _latest and _latest['active'] and _latest['phase']==phase and now-_last_sample < 1:
        return snapshot()
    _last_sample=now
    report=dict(sampled_at=time.time(), phase=phase, active=True,
        sampling_scope='frame boundary on GPU owner thread; excludes transient kernel peaks',
        unattributed=['NGX model/workspace','VSR/VFG/TrueHDR SDK internals','NVDEC/NVENC and NVOF pools',
                      'unqueried display/post-processing/transport allocations'])
    try:
        report['process']=process_memory(live.d3d12_device())
        report['components']=resource_rows(chain)
        report['known_device_allocations_bytes']=sum(row.get('allocated_device_bytes') or 0 for row in report['components'])
        if preview_cache is not None:
            report['preview_cache']=preview_cache.report()
        torch=sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            report['torch']=dict(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
                scope='PyTorch allocator only; included in process usage')
    except Exception as error:
        report['error']=str(error)
    with _lock:
        _latest=report
    return copy.deepcopy(report)


def snapshot():
    with _lock:
        return copy.deepcopy(_latest)


def inactive(phase):
    with _lock:
        if _latest and _latest['phase']==phase:
            _latest['active']=False
