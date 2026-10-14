"""Bounded real-video DLAA preset switching with in-process DXGI memory samples.

Usage: python dlaa_model_memory_gate.py <video-and-settings.json> <output.json>
Requires the test-only DLLs under native/build/memory, external model runtimes,
and an idle GPU. Settings/source are read from a local snapshot, never modified.
"""
import ctypes as C
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import dlss_gpu_engine
from gpu_preview_engine import GpuPreview


class Memory(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in
                ('luid', 'local_usage', 'local_budget', 'nonlocal_usage', 'nonlocal_budget')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--modes', nargs='+', choices=['solo', 'mixed'], default=['solo', 'mixed'])
    parser.add_argument('--presets', nargs='+', type=int, default=[12, 13, 12, 13, 12])
    options = parser.parse_args()
    snapshot, output = options.snapshot, options.output
    args = json.loads(snapshot.read_text('utf-8'))
    dlss_gpu_engine._LIBRARY = str(ROOT/'native/build/memory/dlss_gpu_host_cleanup_probe.dll')
    probe = C.CDLL(str(ROOT/'native/build/memory/gpu_process_memory_probe.dll'))
    probe.gpu_process_memory_query.argtypes = [C.c_void_p, C.POINTER(Memory)]
    probe.gpu_process_memory_query.restype = C.c_int32
    rows = []
    def sample(p, label, **extra):
        def dxgi():
            memory = Memory()
            device = p.live.d3d12_device() if p.live else None
            code = probe.gpu_process_memory_query(device, C.byref(memory))
            if code:
                raise RuntimeError(f'DXGI memory query failed: {code:#x}')
            return {key: int(getattr(memory, key)) for key, _ in Memory._fields_}
        command = ("Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
                   f"| Where-Object {{$_.Name -like 'pid_{os.getpid()}_*'}} "
                   "| Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted | ConvertTo-Json -Compress")
        windows = subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command], text=True).strip()
        row = dict(phase=label, pid=os.getpid(), dxgi=p.executor.submit(dxgi).result(),
                   windows=json.loads(windows) if windows else None,
                   adapter_used_mib=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip(),
                   **extra)
        rows.append(row)
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    try:
        for mode in options.modes:
            order = ['dlaa'] if mode == 'solo' else ['dlaa', 'nr']
            p = GpuPreview()
            try:
                sample(p, 'baseline', order=order)
                for preset in options.presets:
                    settings = dict(args['settings'], render_pipeline=order, dlaa_preset=preset,
                                    truehdr_enabled=False, working_scale=1, layer_scales=[1],
                                    reconstruction_mode='single', guidance_mode=0)
                    _, _, diag = p.render(args['video'], settings, 0, display_edge=640)
                    sample(p, 'live', order=order, preset=preset, generation=diag['generation'],
                           dlaa=diag.get('dlaa_process_global_counters'),
                           input_effect=diag.get('native_input_effect'))
                p.close()
                sample(p, 'closed', order=order)
            finally:
                p.shutdown()
    except Exception:
        rows.append({'error': traceback.format_exc()})
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
