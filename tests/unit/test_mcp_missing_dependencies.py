"""An older runtime missing the new optional integration must still open the app."""
import builtins

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lunarc_mcp import create_http_app
from mcp_control import MCPControl


@pytest.mark.parametrize('missing',['mcp','httpx'])
def test_missing_dependency_only_disables_mcp(monkeypatch,tmp_path,missing):
    original=builtins.__import__
    def load(name,*args,**kwargs):
        if name==missing or name.startswith(missing+'.'):
            raise ModuleNotFoundError(f"No module named '{missing}'")
        return original(name,*args,**kwargs)
    app=FastAPI()
    @app.get('/core-ready')
    def ready():return {'ready':True}
    with monkeypatch.context() as context:
        context.setattr(builtins,'__import__',load)
        http_app=create_http_app(app,'test-token')
    assert missing in http_app.state.mcp_error
    control=MCPControl(tmp_path/'mcp.json');control.runtime_error=http_app.state.mcp_error
    assert missing in control.info()['error']
    with pytest.raises(RuntimeError):control.set_enabled(True)
    assert not control.enabled
    app.mount('/',http_app)
    with TestClient(app) as client:
        assert client.get('/core-ready').json()=={'ready':True}
        result=client.post('/mcp',json={})
        assert result.status_code==503 and missing in result.json()['detail']
