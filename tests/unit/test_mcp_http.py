"""MCP HTTP transport on an isolated loopback port; no user backend or GPU."""
import asyncio
import socket

import httpx2
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from mcp_control import MCPControl
from preset_store import PresetStore
from lunarc_mcp import create_http_app


def test_http_handshake_and_tools_use_existing_routes_without_client_secret(tmp_path,monkeypatch):
    import server
    control=MCPControl(tmp_path/'mcp.json');control.set_enabled(True)
    store=PresetStore(tmp_path/'presets.json');store.save('Shared',{'intensity':1.25})
    monkeypatch.setattr(server.automation,'control',control)
    monkeypatch.setattr(server.automation,'presets',store)
    http_app=create_http_app(server.app,server.LOCAL_TOKEN)
    monkeypatch.setattr(server,'_mcp_http',http_app)
    monkeypatch.setattr(server.app.router.routes[-1],'app',http_app)
    async def run():
        # Only the transport lifespan: do not run backend startup/token-file/GPU cleanup.
        async with http_app.router.lifespan_context(http_app),httpx2.AsyncClient(trust_env=False) as http:
            with socket.socket() as listener:
                listener.bind(('127.0.0.1',0))
                url=f'http://127.0.0.1:{listener.getsockname()[1]}/mcp'
                instance=uvicorn.Server(uvicorn.Config(server.app,lifespan='off',ws='none',log_level='critical'))
                task=asyncio.create_task(instance.serve(sockets=[listener]))
                try:
                    for _ in range(150):
                        if instance.started:break
                        if task.done():await task
                        await asyncio.sleep(.02)
                    assert instance.started
                    await check_tools(http,url)
                finally:
                    instance.should_exit=True
                    await asyncio.wait_for(task,5)
    async def check_tools(http,url):
        async with streamable_http_client(url,http_client=http) as streams:
            async with ClientSession(*streams) as client:
                assert (await client.initialize()).server_info.name=='Lunarc'
                tools=await client.list_tools();assert len(tools.tools)==13
                result=await client.call_tool('list_presets',{})
                assert not result.is_error and result.structured_content['presets']==['Shared']
                status=await client.call_tool('get_status',{})
                assert not status.is_error and 'jobs' in status.structured_content
                draft=await client.call_tool('resolve_settings',{'settings':{'intensity':.7}})
                assert not draft.is_error and draft.structured_content['settings']['intensity']==.7
                invalid=await client.call_tool('resolve_settings',{'settings':{'typo_intensity':1}})
                assert invalid.is_error and 'typo_intensity' in invalid.content[0].text
        control.set_enabled(False)
        disabled=await http.post(url,json={})
        assert disabled.status_code==403
    asyncio.run(run())


def test_http_blocks_remote_peers_browser_requests_and_rebinding(tmp_path,monkeypatch):
    import server
    control=MCPControl(tmp_path/'mcp.json');control.set_enabled(True)
    monkeypatch.setattr(server.automation,'control',control)
    async def run():
        for peer,url,headers in (
            ('192.168.1.10','http://127.0.0.1:18923/mcp',{}),
            ('127.0.0.1','http://malicious.example/mcp',{}),
            ('127.0.0.1','http://127.0.0.1:18923/mcp',{'Origin':'https://untrusted.example'}),
            ('127.0.0.1','http://127.0.0.1:18923/mcp',{'Origin':'http://127.0.0.1:4318'}),
            ('127.0.0.1','http://127.0.0.1:18923/mcp',{'Sec-Fetch-Site':'same-origin'}),
        ):
            async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=server.app,client=(peer,54321))) as http:
                response=await http.post(url,headers=headers,json={})
                assert response.status_code==403,(peer,url,headers)
        # Token exemption is exactly /mcp, never REST or the permission switch.
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=server.app,client=('127.0.0.1',54321))) as http:
            assert (await http.get('http://127.0.0.1:18923/api/automation/status')).status_code==401
            assert (await http.post('http://127.0.0.1:18923/api/mcp',json={'enabled':True})).status_code==401
    asyncio.run(run())
