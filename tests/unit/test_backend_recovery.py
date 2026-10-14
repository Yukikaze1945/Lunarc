import json
from types import SimpleNamespace

import pytest

from dlss_service import DLSSService
from queue_store import QueueStore
from gpu_faults import recovery_reason
import gpu_memory


def add(service, name):
    return service.add_job({'name':name,'input_path':__file__,'output_path':name+'.mkv'})


def test_restart_preserves_pending_and_marks_running_interrupted(tmp_path):
    path=tmp_path/'queue.json'
    service=DLSSService(queue_path=path)
    first,second=add(service,'first'),add(service,'second')
    first['status']='running'
    first['current_frame']=91
    service._mark_native_degraded('GPU reset')
    restored=DLSSService(queue_path=path)
    a,b=restored.get_jobs()
    assert a['id']==first['id'] and a['status']=='failed' and a['interrupted']
    assert a['current_frame']==91 and b['status']=='pending'
    assert restored.backend_health()['healthy']
    restored.retry_job(a['id'])
    restored.remove_job(b['id'])
    final=DLSSService(queue_path=path).get_jobs()
    assert len(final)==1 and final[0]['status']=='pending' and 'interrupted' not in final[0]
    fault=json.loads((tmp_path/'last-backend-fault.json').read_text('utf-8'))
    assert fault['incident_id']==service.backend_health()['incident_id']


def test_fuse_stops_queue_before_claiming_next_task(tmp_path):
    service=DLSSService(queue_path=tmp_path/'queue.json')
    first,second=add(service,'first'),add(service,'second')
    called=[]
    def execute(job, callback):
        called.append(job['id'])
        service._mark_native_degraded('GPU reset')
        raise RuntimeError('GPU reset')
    service._execute_job=execute
    service.start_queue()
    service._queue_thread.join(5)
    assert called==[first['id']] and first['status']=='failed' and second['status']=='pending'
    with pytest.raises(RuntimeError,match='熔断'):service.start_queue()


def test_bad_archive_is_preserved_and_blocks_new_exports(tmp_path):
    path=tmp_path/'queue.json';path.write_text('broken')
    service=DLSSService(queue_path=path)
    with pytest.raises(RuntimeError,match='存档'):add(service,'new')
    assert path.read_text()=='broken'
    assert service.backend_health()['queue_storage_error']


def test_atomic_write_failure_preserves_last_checkpoint(tmp_path, monkeypatch):
    store=QueueStore(tmp_path/'queue.json')
    store.save([])
    original=store.path.read_bytes()
    def fail(*args):raise OSError('disk unavailable')
    monkeypatch.setattr('queue_store.os.replace',fail)
    with pytest.raises(OSError):store.save([{'id':'new'}])
    assert store.path.read_bytes()==original and not list(tmp_path.glob('*.tmp'))


def test_recovery_requires_explicit_fault_evidence():
    assert recovery_reason('GPU temporal map 0x887a0005 reason 0x887a0006')
    assert recovery_reason('CUDA error: out of memory')
    assert recovery_reason('access violation reading 0x910')
    assert recovery_reason('4K model slow, 15 GiB used') is None
    assert recovery_reason('tune: Invalid argument (-22)') is None


def test_memory_reader_does_not_query_native_and_returns_detached_data(monkeypatch):
    monkeypatch.setattr(gpu_memory,'_latest',{'phase':'preview','components':[{'owner':'one'}]})
    def fail(*args):raise AssertionError('control plane queried native')
    monkeypatch.setattr(gpu_memory,'process_memory',fail)
    view=gpu_memory.snapshot();view['components'].clear()
    assert len(gpu_memory.snapshot()['components'])==1


def test_memory_ledger_deduplicates_shared_owners():
    owner=SimpleNamespace(allocated_device_bytes=lambda:4096)
    root=SimpleNamespace(input_effect=owner, output_effect=owner)
    rows=gpu_memory.resource_rows(root)
    assert len(rows)==1 and rows[0]['allocated_device_bytes']==4096


def test_storage_failure_does_not_acknowledge_a_task_that_was_not_saved(tmp_path,monkeypatch):
    service=DLSSService(queue_path=tmp_path/'queue.json')
    first=add(service,'first')
    def fail(*args):raise OSError('disk full')
    monkeypatch.setattr(service._queue_store,'save',fail)
    with pytest.raises(RuntimeError,match='未添加'):add(service,'second')
    assert service.get_jobs()==[first]
    with pytest.raises(RuntimeError,match='未删除'):service.remove_job(first['id'])
    assert service.get_jobs()==[first]
