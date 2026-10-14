"""Atomic queue checkpoints. Native handles and per-frame evidence stay out of JSON."""
import json
import os
import tempfile
from pathlib import Path

from queue_payload import queue_job


class QueueStore:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return []
        document = json.loads(self.path.read_text(encoding='utf-8'))
        if document.get('version') != 1 or not isinstance(document.get('jobs'), list):
            raise ValueError('无法读取导出队列：存档格式不受支持')
        jobs, ids = [], set()
        for raw in document['jobs']:
            if (not isinstance(raw, dict) or not isinstance(raw.get('id'), str)
                    or raw['id'] in ids or not isinstance(raw.get('settings'), dict)
                    or raw.get('status') not in ('pending', 'running', 'completed', 'failed', 'cancelled', 'skipped')):
                raise ValueError('无法读取导出队列：任务记录损坏')
            job = dict(raw)
            ids.add(job['id'])
            if job['status'] == 'running':
                job.update(status='failed', fps=0, eta=0, interrupted=True,
                           error='后端重启，导出已中断；可点击重试从头导出', error_traceback=None)
            jobs.append(job)
        return jobs

    def save(self, jobs):
        payload = {'version': 1, 'jobs': [queue_job(dict(job)) for job in jobs]}
        self._write(self.path, payload)

    def save_fault(self, fault):
        self._write(self.path.with_name('last-backend-fault.json'), fault)

    @staticmethod
    def _write(path, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                    prefix=path.name+'.', suffix='.tmp', delete=False) as file:
                temporary = file.name
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
