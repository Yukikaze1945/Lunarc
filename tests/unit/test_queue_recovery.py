import threading

import pytest

from dlss_service import DLSSService


def add(service, name):
    return service.add_job({'name': name, 'input_path': __file__,
                            'output_path': name + '.mkv'})


def test_remove_pending_and_add_after_failure_during_round():
    service = DLSSService()
    first = add(service, 'first')
    removed = add(service, 'removed')
    entered, resume = threading.Event(), threading.Event()
    executed = []
    def execute(job, callback):
        executed.append(job['name'])
        if job is first:
            entered.set()
            assert resume.wait(5)
            raise RuntimeError('encoder failed')
    service._execute_job = execute
    assert service.start_queue()
    assert entered.wait(5)
    with pytest.raises(ValueError, match='正在导出'):
        service.remove_job(first['id'])
    service.remove_job(removed['id'])
    next_job = add(service, 'new')
    resume.set()
    service._queue_thread.join(5)
    assert not service.queue_is_running()
    assert service.active_job_id is None
    assert executed == ['first', 'new']
    assert first['status'] == 'failed'
    assert next_job['status'] == 'completed'
    service.remove_job(first['id'])
    last = add(service, 'last')
    assert service.start_queue()
    service._queue_thread.join(5)
    assert last['status'] == 'completed'


def test_notification_failure_does_not_strand_running_task():
    service = DLSSService()
    job = add(service, 'notification')
    def broken_notification(data):
        raise RuntimeError('socket gone')
    def execute(job, callback):
        callback(job)
    service._execute_job = execute
    assert service.start_queue(broken_notification)
    service._queue_thread.join(5)
    assert service.active_job_id is None
    assert not service.queue_is_running()
    assert job['status'] == 'completed'
    service.remove_job(job['id'])
    assert service.get_jobs() == []
