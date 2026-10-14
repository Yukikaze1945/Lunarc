"""Windows child containment: backend death must also stop owned FFmpeg children."""
import sys

_job = None


def contain_children():
    global _job
    if sys.platform != 'win32' or _job is not None:
        return
    import ctypes as C
    from ctypes import wintypes as W
    class Basic(C.Structure):
        _fields_ = [('process_time',C.c_int64),('job_time',C.c_int64),('flags',W.DWORD),
            ('min_working',C.c_size_t),('max_working',C.c_size_t),('active_limit',W.DWORD),
            ('affinity',C.c_size_t),('priority',W.DWORD),('scheduling',W.DWORD)]
    class Extended(C.Structure):
        _fields_ = [('basic',Basic),('io',C.c_uint64*6),('process_limit',C.c_size_t),
            ('job_limit',C.c_size_t),('peak_process',C.c_size_t),('peak_job',C.c_size_t)]
    kernel=C.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.argtypes=[C.c_void_p,W.LPCWSTR];kernel.CreateJobObjectW.restype=W.HANDLE
    kernel.SetInformationJobObject.argtypes=[W.HANDLE,C.c_int,C.c_void_p,W.DWORD]
    kernel.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE]
    kernel.GetCurrentProcess.restype=W.HANDLE
    kernel.CloseHandle.argtypes=[W.HANDLE]
    job=kernel.CreateJobObjectW(None,None)
    if not job:raise C.WinError(C.get_last_error())
    limits=Extended();limits.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job,9,C.byref(limits),C.sizeof(limits)):
        error=C.WinError(C.get_last_error());kernel.CloseHandle(job);raise error
    if not kernel.AssignProcessToJobObject(job,kernel.GetCurrentProcess()):
        error=C.WinError(C.get_last_error());kernel.CloseHandle(job);raise error
    # Non-inheritable handle lives until OS process teardown, including native crash.
    _job=job
