import ctypes
import sys
from types import SimpleNamespace

import pytest
import torch_cuda_context as contexts


def test_primary_scope_restores_foreign_context_and_releases_retain_on_failure(monkeypatch):
    calls = []

    def driver_call(name):
        def invoke(*args):
            calls.append(name)
            if name == 'resolve':
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int))[0] = 2
                assert args[1] == b'0000:03:00.0'
            elif name == 'retain':
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = 123
                assert args[1] == 2
            return 0
        return invoke

    cuda = SimpleNamespace(**{function: driver_call(name) for function, name in (
        ('cuDeviceGetByPCIBusId', 'resolve'), ('cuDevicePrimaryCtxRetain', 'retain'),
        ('cuDevicePrimaryCtxRelease_v2', 'release'), ('cuCtxPushCurrent_v2', 'push'),
        ('cuCtxPopCurrent_v2', 'pop'))})
    torch = SimpleNamespace(device=lambda value: value, cuda=SimpleNamespace(
        get_device_properties=lambda value: SimpleNamespace(
            pci_domain_id=0, pci_bus_id=3, pci_device_id=0)))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(contexts.C, 'WinDLL', lambda name: cuda)
    with pytest.raises(ValueError, match='inference failure'):
        with contexts.torch_primary_context('cuda'):
            calls.append('allocate')
            raise ValueError('inference failure')
    assert calls == ['resolve', 'retain', 'push', 'allocate', 'pop', 'release']


def test_cpu_scope_does_not_load_cuda_driver(monkeypatch):
    monkeypatch.setattr(contexts.C, 'WinDLL', lambda name: pytest.fail('CPU must not load CUDA'))
    with contexts.torch_primary_context('cpu'):
        pass
