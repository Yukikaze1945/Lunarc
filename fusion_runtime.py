"""Optional scalar-only solver update shared by SDR and linear-HDR preview."""
import ctypes as C


def update_parameters(library, handle, config, strength):
    fn = getattr(library, 'gpu_fusion_update_parameters', None)
    if fn is None:
        return False
    fn.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_uint32,
                   C.c_float, C.c_float, C.c_float, C.c_uint32]
    fn.restype = C.c_int32
    weights = (C.c_float * len(config['weights']))(*config['weights'])
    code = fn(handle, weights, len(weights), config['beta'], strength,
              config['relative_tolerance'], config['max_iterations'])
    if code:
        from gpu_transfer_export import _bridge_error
        _bridge_error(library, 'gpu_fusion_last_error', code)
    return True
