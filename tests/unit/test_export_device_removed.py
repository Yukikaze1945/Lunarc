from types import SimpleNamespace

import pytest
from dlss_service import DLSSService


@pytest.mark.parametrize('error', ['GPU process: 0x887a0005', 'removed reason 0x887a0006'])
def test_export_device_removal_fuses_further_gpu_work(error):
    service = DLSSService()
    service._execute_job_owned = lambda *a: (_ for _ in ()).throw(RuntimeError(error))
    service._live_session_owner = 'export'
    closed = []
    service._live_session = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(RuntimeError, match='0x887a'):
        service._execute_job({}, None)
    assert closed == [True] and service._live_session is None
    with pytest.raises(RuntimeError, match='驱动重置'):
        service._ensure_native_healthy()


def test_ordinary_encoder_option_failure_does_not_fuse_gpu():
    service = DLSSService()
    service._execute_job_owned = lambda *a: (_ for _ in ()).throw(RuntimeError('tune: Invalid argument'))
    with pytest.raises(RuntimeError, match='tune'):
        service._execute_job({}, None)
    service._ensure_native_healthy()
