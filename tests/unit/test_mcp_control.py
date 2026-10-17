"""MCP permission, generated configuration and backend enforcement; no GPU calls."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from automation import Automation,create_router
from dlss_service import DLSSService
from mcp_control import MCPControl
from preset_store import PresetStore


def test_permission_defaults_off_and_survives_restart(tmp_path):
    path=tmp_path/'mcp.json'
    control=MCPControl(path)
    assert control.enabled is False
    control.set_enabled(True)
    assert MCPControl(path).enabled is True
    control.set_enabled(False)
    assert MCPControl(path).enabled is False


@pytest.mark.parametrize('value',['true',1,None,[]])
def test_permission_does_not_accept_truthy_non_boolean(tmp_path,value):
    control=MCPControl(tmp_path/'mcp.json')
    with pytest.raises(ValueError):control.set_enabled(value)
    assert not control.enabled and not control.path.exists()


def test_corrupt_permission_fails_closed_without_rewriting_file(tmp_path):
    path=tmp_path/'mcp.json';path.write_text('{invalid',encoding='utf-8')
    control=MCPControl(path)
    assert not control.enabled and control.info()['error']
    assert path.read_text(encoding='utf-8')=='{invalid'


def test_failed_save_does_not_change_permission(tmp_path,monkeypatch):
    control=MCPControl(tmp_path/'mcp.json')
    def fail(*args):raise OSError('write refused')
    monkeypatch.setattr('mcp_control.os.replace',fail)
    with pytest.raises(OSError):control.set_enabled(True)
    assert not control.enabled
    assert not list(tmp_path.glob('*.tmp'))


def test_copied_prompt_is_short_and_uses_url_only(tmp_path):
    control=MCPControl(tmp_path/'mcp.json')
    info=control.info(19123)
    cfg=info['config']['mcpServers']['lunarc']
    assert cfg=={'url':'http://127.0.0.1:19123/mcp'}
    assert info['transport']=='streamable-http'
    assert info['prompt']==('Add a Streamable HTTP MCP server named "lunarc" at '
        'http://127.0.0.1:19123/mcp to my user-level MCP configuration on this PC. '
        'If you cannot configure MCP servers directly, show me where to add it manually.')
    assert len(info['prompt'])<250 and str(tmp_path) not in info['prompt']


def test_disabled_gate_blocks_all_agent_routes_but_not_workbench_presets(tmp_path):
    service=DLSSService()
    automation=Automation(service,PresetStore(tmp_path/'presets.json'))
    app=FastAPI();router=create_router(automation,lambda _:None);app.include_router(router)
    with TestClient(app,base_url='http://127.0.0.1:18923') as client:
        for route in router.routes:
            if route.path.startswith('/api/automation/'):
                method=next(m for m in route.methods if m in ('GET','POST'))
                kwargs={'json':{}} if method=='POST' else {}
                assert client.request(method,route.path,**kwargs).status_code==403,route.path
        assert client.get('/api/presets').status_code==200
        assert not client.get('/api/mcp').json()['enabled']
        assert client.post('/api/mcp',json={'enabled':'true'}).status_code==400
        assert client.post('/api/mcp',json={'enabled':True}).json()['enabled']
        assert client.get('/api/automation/status').status_code==200
        assert client.get('/api/automation/presets').status_code==200
        assert client.post('/api/mcp',json={'enabled':False}).status_code==200
        assert client.get('/api/automation/status').status_code==403
        assert not service.jobs and service._queue_thread is None


def test_connection_settings_require_existing_backend_auth(tmp_path,monkeypatch):
    import server
    monkeypatch.setattr(server.automation,'control',MCPControl(tmp_path/'mcp.json'))
    # No lifespan: importing/checking routes must not overwrite a running app's session file.
    client=TestClient(server.app,base_url='http://127.0.0.1:18923')
    assert client.get('/api/mcp').status_code==401
    assert client.post('/api/mcp',json={'enabled':True}).status_code==401
    headers={'X-DLSS-Token':server.LOCAL_TOKEN}
    assert client.get('/api/mcp',headers=headers).status_code==200
    assert client.get('/api/mcp',headers={**headers,'Origin':'https://untrusted.example'}).status_code==403
    assert not server.automation.control.enabled
