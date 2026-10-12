"""Optional native solver metrics, independent of the existing solve/report ABI."""
import ctypes as C


class SolverDiagnostics(C.Structure):
    _fields_ = [('struct_size', C.c_uint32), ('version', C.c_uint32),
                ('solver_path', C.c_uint32), ('batch_size', C.c_uint32),
                ('workspace_logical_bytes', C.c_uint64),
                ('workspace_committed_bytes', C.c_uint64),
                ('scalar_sync_count', C.c_uint64), ('cpu_wait_ms', C.c_double),
                ('horizontal_taps', C.c_uint32 * 3), ('vertical_taps', C.c_uint32 * 3)]


def read_solver_diagnostics(library, handle):
    getter = getattr(library, 'gpu_fusion_get_solver_diagnostics', None)
    if getter is None:
        return None
    getter.argtypes = [C.c_void_p, C.POINTER(SolverDiagnostics)]
    getter.restype = C.c_int32
    report = SolverDiagnostics()
    report.struct_size = C.sizeof(report)
    code = getter(handle, C.byref(report))
    if code:
        raise RuntimeError(f'fusion solver diagnostics failed: {code}')
    result = {name: getattr(report, name) for name, _ in report._fields_[1:]}
    result['horizontal_taps'] = list(report.horizontal_taps)
    result['vertical_taps'] = list(report.vertical_taps)
    result['solver_path'] = {1: 'primal_host_cg', 2: 'primal_gpu_cg',
                             3: 'observation_gpu_cg',
                             4: 'observation_gpu_pcg'}.get(report.solver_path, 'unknown')
    result['workspace_scope'] = 'solver FP32 textures; excludes NR and transport resources'
    return result
