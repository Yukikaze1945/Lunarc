"""MCP protocol/adapter checks use fake backend results; never render media."""
import asyncio
import sys
from pathlib import Path

import pytest

from lunarc_mcp import Backend,create_server


def test_wait_consumes_nonterminal_states_internally_and_returns_one_final_result():
    backend=Backend()
    calls=[]
    async def request(path,data=None,timeout=35):
        calls.append(data)
        return dict(done=len(calls)==2,jobs=[dict(id='one',status='completed' if len(calls)==2 else 'running')])
    backend.request=request
    result=asyncio.run(backend.wait(['one'],1))
    assert result['done'] and len(calls)==2 and all(c['job_ids']==['one'] for c in calls)


def test_sdk_tools_return_structured_results_without_loading_backend_modules():
    pytest.importorskip('mcp')
    class FakeBackend:
        async def request(self,*a,**k):return dict(presets=[dict(name='Shared',settings={'intensity':1})],revision=2)
    async def run():
        server=create_server(FakeBackend())
        assert {'render_video','render_image','wait_for_jobs','resolve_settings','save_preset'} <= {t.name for t in await server.list_tools()}
        result=await server.call_tool('list_presets',{})
        assert not result.is_error and result.structured_content==dict(presets=['Shared'],revision=2)
    asyncio.run(run())


def test_real_stdio_handshake_and_tool_errors_without_running_app(tmp_path):
    pytest.importorskip('mcp')
    from mcp import ClientSession,StdioServerParameters
    from mcp.client.stdio import stdio_client
    async def run():
        script=Path(__file__).resolve().parents[2]/'lunarc_mcp.py'
        params=StdioServerParameters(command=sys.executable,args=[str(script),'--token-file',str(tmp_path/'missing-token')])
        async with stdio_client(params) as (reader,writer):
            async with ClientSession(reader,writer) as session:
                initialized=await session.initialize()
                assert initialized.server_info.name=='Lunarc'
                tools=await session.list_tools()
                assert len(tools.tools)>=12
                result=await session.call_tool('list_presets',{})
                assert result.is_error
                assert '尚未启动' in result.content[0].text
    asyncio.run(run())


def test_wait_network_failure_preserves_job_and_request_ids():
    pytest.importorskip('mcp')
    class FakeBackend:
        async def request(self,*a,**k):return dict(job=dict(id='queued-job',status='pending'),reused=False)
        async def wait(self,*a,**k):raise ValueError('temporary connection loss')
    async def run():
        result=await create_server(FakeBackend()).call_tool('render_video',dict(
            input_path='D:/source.mkv',output_path='D:/output.mkv',request_id='stable-original-request'))
        assert not result.is_error
        data=result.structured_content
        assert data['job']['id']=='queued-job' and data['request_id']=='stable-original-request'
        assert data['recoverable'] and not data['done'] and data['wait_error']=='temporary connection loss'
    asyncio.run(run())
