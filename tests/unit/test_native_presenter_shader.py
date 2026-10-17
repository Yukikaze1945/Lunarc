"""Compile the real display shader and inspect resource guards, without a GPU."""
import ctypes as C
from pathlib import Path
import re
import sys

import pytest

pytestmark=pytest.mark.skipif(sys.platform!='win32',reason='uses the Windows HLSL compiler')


def disassemble(source):
    lib=C.WinDLL('d3dcompiler_47.dll')
    lib.D3DCompile.argtypes=[C.c_void_p,C.c_size_t,C.c_char_p,C.c_void_p,C.c_void_p,
        C.c_char_p,C.c_char_p,C.c_uint,C.c_uint,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    lib.D3DDisassemble.argtypes=[C.c_void_p,C.c_size_t,C.c_uint,C.c_char_p,C.POINTER(C.c_void_p)]
    def method(blob,index,result,*args):
        table=C.cast(blob,C.POINTER(C.POINTER(C.c_void_p))).contents
        return C.WINFUNCTYPE(result,C.c_void_p,*args)(table[index])
    def pointer(blob):return method(blob,3,C.c_void_p)(blob)
    code=errors=assembly=None
    try:
        code,errors,assembly=C.c_void_p(),C.c_void_p(),C.c_void_p()
        encoded=source.encode()
        rc=lib.D3DCompile(encoded,len(encoded),b'presenter-test',None,None,b'ps',b'ps_5_1',32768,0,C.byref(code),C.byref(errors))
        assert not rc,C.string_at(pointer(errors)).decode() if errors else hex(rc&0xffffffff)
        size=method(code,4,C.c_size_t)(code)
        assert not lib.D3DDisassemble(pointer(code),size,0,None,C.byref(assembly))
        return C.string_at(pointer(assembly)).decode()
    finally:
        for blob in (code,errors,assembly):
            if blob:method(blob,2,C.c_ulong)(blob)


def unguarded_resource_loads(assembly):
    """Every A/B load must belong to opposite arms of a resource-selection if."""
    root=[];arm=root;stack=[];nodes=[];loads=set()
    for index,line in enumerate(assembly.splitlines()):
        text=line.strip()
        if text.startswith(('if_nz ','if_z ')):
            node=[[],[]];arm.append(node);nodes.append(node);stack.append((arm,node));arm=node[0]
        elif text=='else':arm=stack[-1][1][1]
        elif text=='endif':arm,_=stack.pop()
        elif text.startswith('ld_raw '):
            resource=re.search(r'T([01])\[',text)
            assert resource,text
            load=(index,resource.group(1));loads.add(load);arm.append(load)
    def all_loads(items):
        result=set()
        for item in items:
            if isinstance(item,tuple):result.add(item)
            else:result.update(all_loads(item[0])|all_loads(item[1]))
        return result
    covered=set()
    for left,right in nodes:
        a,b=all_loads(left),all_loads(right)
        if ({r for _,r in a},{r for _,r in b}) in (({'0'},{'1'}),({'1'},{'0'})):
            covered.update(a|b)
    assert loads,'compiler output did not contain resource loads'
    return loads-covered


def test_real_shader_never_speculatively_loads_both_raw_planes():
    source=Path('native/gpu_native_presenter.cpp').read_text(encoding='utf-8')
    shader=re.search(r'const char\* shader=R"\((.*?)\)";',source,re.S).group(1)
    assert not unguarded_resource_loads(disassemble(shader))


def test_guard_checker_reproduces_old_ternary_compiler_bug():
    shader='''ByteAddressBuffer A:register(t0);ByteAddressBuffer B:register(t1);
float4 ps(float4 p:SV_Position):SV_Target{
uint address=(uint)p.y*64+(uint)p.x*8;
uint2 v=p.x<16?A.Load2(address):B.Load2(address);
return float4(f16tof32(v.x&65535),0,0,1);}'''
    assert unguarded_resource_loads(disassemble(shader))
