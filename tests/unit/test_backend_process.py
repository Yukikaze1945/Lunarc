import ctypes as C
import subprocess
import sys
from pathlib import Path

import pytest

@pytest.mark.skipif(sys.platform!='win32',reason='Windows job containment')
def test_backend_death_terminates_owned_children():
    code="from backend_process import contain_children; contain_children(); import subprocess,sys; child=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.read()'],stdin=subprocess.PIPE); print(child.pid,flush=True); sys.stdin.read()"
    parent=subprocess.Popen([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    kernel=C.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[C.c_uint32,C.c_bool,C.c_uint32];kernel.OpenProcess.restype=C.c_void_p
    kernel.WaitForSingleObject.argtypes=[C.c_void_p,C.c_uint32]
    kernel.CloseHandle.argtypes=[C.c_void_p]
    handle=None
    try:
        pid=parent.stdout.readline().strip()
        assert pid,parent.stderr.read()
        handle=kernel.OpenProcess(0x100000,False,int(pid))
        assert handle
        parent.terminate();parent.wait(timeout=5)
        assert kernel.WaitForSingleObject(handle,5000)==0,'orphaned child after backend exit'
    finally:
        if parent.poll() is None:parent.kill();parent.wait(timeout=5)
        if handle:kernel.CloseHandle(handle)
