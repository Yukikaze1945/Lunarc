"""Keep Windows Torch allocations out of borrowed native CUDA contexts."""
import ctypes as C
import sys
from contextlib import contextmanager


def _check(code, operation):
    if code:
        raise RuntimeError(f'{operation}: CUDA driver error {code}')


@contextmanager
def torch_primary_context(device):
    if not str(device).startswith('cuda') or sys.platform != 'win32':
        yield
        return
    import torch
    cuda = C.WinDLL('nvcuda.dll')
    cuda.cuDeviceGetByPCIBusId.argtypes = [C.POINTER(C.c_int), C.c_char_p]
    cuda.cuDevicePrimaryCtxRetain.argtypes = [C.POINTER(C.c_void_p), C.c_int]
    cuda.cuDevicePrimaryCtxRelease_v2.argtypes = [C.c_int]
    cuda.cuCtxPushCurrent_v2.argtypes = [C.c_void_p]
    cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(C.c_void_p)]
    properties = torch.cuda.get_device_properties(torch.device(device))
    pci = f'{properties.pci_domain_id:04x}:{properties.pci_bus_id:02x}:{properties.pci_device_id:02x}.0'
    ordinal, context = C.c_int(), C.c_void_p()
    _check(cuda.cuDeviceGetByPCIBusId(C.byref(ordinal), pci.encode('ascii')), 'resolve Torch device')
    _check(cuda.cuDevicePrimaryCtxRetain(C.byref(context), ordinal.value), 'retain Torch primary context')
    pushed = False
    try:
        _check(cuda.cuCtxPushCurrent_v2(context), 'enter Torch primary context')
        pushed = True
        yield
    finally:
        try:
            if pushed:
                popped = C.c_void_p()
                _check(cuda.cuCtxPopCurrent_v2(C.byref(popped)), 'restore native CUDA context')
        finally:
            _check(cuda.cuDevicePrimaryCtxRelease_v2(ordinal.value), 'release Torch primary retain')
