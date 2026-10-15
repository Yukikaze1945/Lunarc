"""Classify explicit GPU reset/allocation failures, without guessing from VRAM usage."""
def recovery_reason(error):
    message=str(error).lower()
    if 'gpu preview cleanup failed:' in message or 'gpu resource cleanup failed:' in message:
        return 'GPU 资源未能安全释放'
    if any(code in message for code in ('0x887a0005','0x887a0006','0x887a0007')):
        return 'GPU 设备已被驱动重置'
    if any(code in message for code in ('cuda_error_out_of_memory','cuda out of memory',
            'cuda error: out of memory','e_outofmemory','0x8007000e')):
        return 'GPU 内存分配失败'
    if 'access violation' in message:
        return '原生层访问异常'
    return None
