import asyncio
import importlib
import threading
import pytest
from tests.unit.test_preview_memory import fake_service
from tests.unit.test_preview_player import until
from preview_player import render_gate

@pytest.fixture
def server(monkeypatch,tmp_path):
    # Importing the API in a test must never touch a live desktop session token.
    monkeypatch.setenv('LUNARC_DATA_DIR',str(tmp_path))
    monkeypatch.setenv('DLSS_LOCAL_TOKEN','isolated-memory-control-test')
    module=importlib.import_module('server')
    monkeypatch.setattr(module,'dlss_service',fake_service())
    return module

def test_release_waits_for_render_gate_and_stays_paused_after_completion(server):
    async def run():
        service=server.dlss_service;calls=[]
        service.release_preview_memory=lambda:calls.append('native close') or {'released':True}
        async with render_gate():
            task=asyncio.create_task(server.release_preview_memory())
            await until(lambda:service.preview_memory_state()['suspended'])
            assert not calls
        result=await task
        assert result['suspended'] and not result['releasing'] and calls==['native close']
        assert not (await server.resume_preview_memory())['suspended']
    asyncio.run(run())

def test_cancelled_http_task_retains_gpu_gate_until_native_close_returns(server):
    async def run():
        entered,finish=threading.Event(),threading.Event()
        def native():entered.set();assert finish.wait(3);return {}
        server.dlss_service.release_preview_memory=native
        task=asyncio.create_task(server.release_preview_memory())
        await until(entered.is_set)
        task.cancel()
        async def successor():
            async with render_gate():return 'acquired'
        next_task=asyncio.create_task(successor())
        try:
            await asyncio.sleep(.02)
            assert not next_task.done()
            finish.set()
            with pytest.raises(asyncio.CancelledError):await task
            assert await next_task=='acquired'
            assert server.dlss_service.preview_memory_state()==dict(suspended=True,releasing=False,error=None)
        finally:finish.set()
    asyncio.run(run())

def test_export_owner_cannot_be_released_by_preview_api(server):
    server.dlss_service.active_job_id='export'
    with pytest.raises(server.HTTPException) as error:asyncio.run(server.release_preview_memory())
    assert error.value.status_code==409
    assert not server.dlss_service.preview_memory_state()['suspended']
