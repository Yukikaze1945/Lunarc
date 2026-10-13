"""Encoded preview segments submitted to the existing single-owner export queue."""
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from export_range import validate_range


class PreviewSegments:
    def __init__(self, service, root=None):
        self.service = service
        self.root = Path(root or Path(os.environ.get('LOCALAPPDATA') or Path.home()) / 'Lunarc' / 'render-cache')
        self.lock = threading.RLock()
        self.records = {}
        try:
            saved = json.loads((self.root / 'segments.json').read_text(encoding='utf-8'))
            for record in saved:
                path = Path(record['output_path']).resolve()
                if record['status'] == 'completed' and path.is_relative_to(self.root.resolve()):
                    self.records[record['key']] = record
        except (OSError, ValueError, TypeError, KeyError):
            pass

    def persist(self):
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / 'segments.json'
        temporary = self.root / 'segments.json.tmp'
        temporary.write_text(json.dumps([r for r in self.records.values() if r['status'] == 'completed'],
            ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(target)

    @staticmethod
    def identity(source, settings, tx):
        if not isinstance(source, (str, Path)) or not source:
            raise ValueError('请先载入视频素材')
        if not isinstance(settings, dict) or not isinstance(tx, dict):
            raise ValueError('渲染参数或导出参数无效')
        path = Path(source).resolve(strict=True)
        if not path.is_file():
            raise ValueError('视频素材不是文件')
        stat = path.stat()
        root = Path(__file__).resolve().parent
        runtime = []
        for name in ('export_range.py', 'gpu_transfer_export.py', 'gpu_hdr_transfer_export.py',
                     'native_effects.py', 'truehdr_pipeline.py', 'pipeline.py', 'dlss_gpu_engine.py',
                     'native/dlss_gpu_host.dll', 'native/gpu_hdr_scrgb_bridge.dll', 'native/nvenc_bridge.dll'):
            file = root / name
            if file.is_file():
                state = file.stat()
                runtime.append((name, state.st_size, state.st_mtime_ns))
        payload = dict(source=str(path), size=stat.st_size, modified_ns=stat.st_mtime_ns, runtime=runtime,
                       settings=settings, transcode=tx)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(',', ':')).encode()).hexdigest()

    def list(self, source, settings, tx):
        identity = self.identity(source, settings, tx)
        jobs = {j['id']: j for j in self.service.get_jobs()}
        with self.lock:
            rows = []
            changed = False
            for record in self.records.values():
                if record['identity'] != identity:
                    continue
                job = jobs.get(record['job_id'])
                status = job['status'] if job else record.get('status', 'missing')
                path = Path(record['output_path'])
                try:
                    stat = path.stat()
                    if status == 'completed' and record.get('status') != 'completed':
                        record.update(status=status, output_size=stat.st_size, output_modified_ns=stat.st_mtime_ns)
                        changed = True
                    ready = (status == 'completed' and stat.st_size > 0 and
                        record.get('output_size') == stat.st_size and record.get('output_modified_ns') == stat.st_mtime_ns)
                except OSError:
                    ready = False
                rows.append(dict(record, status=status, ready=ready,
                    progress=job.get('progress', 0) if job else 100,
                    error=job.get('error') if job else None))
            if changed:
                self.persist()
            return rows

    def create(self, source, settings, tx, selection):
        settings, tx = copy.deepcopy(settings), copy.deepcopy(tx)
        if settings.get('export_backend') != 'gpu_transfer':
            raise ValueError('选区预渲染使用 GPU 导出，请在工作台开启选区')
        import pipeline
        total, fps, _, _ = pipeline.video_info_uncached(source)
        selection = validate_range(selection, total)
        if selection is None:
            raise ValueError('请先选择预渲染范围')
        identity = self.identity(source, settings, tx)
        key = hashlib.sha256((identity + json.dumps(selection, sort_keys=True)).encode()).hexdigest()
        with self.lock:
            matches = self.list(source, settings, tx)
            existing = next((r for r in matches if r['key'] == key and
                (r['ready'] or r['status'] in ('pending', 'running'))), None)
            if existing:
                return existing
            self.root.mkdir(parents=True, exist_ok=True)
            # Unique paths keep incomplete retries from becoming a valid cache.
            ext = str(tx.get('container') or '.mkv').lstrip('.')
            if ext not in ('mkv', 'mp4', 'mov', 'webm'):
                raise ValueError('不支持此预渲染封装格式')
            output = str(self.root / f'{key[:20]}-{time.time_ns()}.{ext}')
            effective_tx = dict(tx, finish_action='none', skip_existing=False, overwrite_output=True)
            job = self.service.add_job(dict(name=f'预渲染 · {selection["start_frame"] + 1}–{selection["end_frame"]}',
                input_path=source, output_path=output, settings=settings, transcode_settings=effective_tx,
                frame_range=selection, codec=tx.get('codec', 'av1_nvenc'), container=ext,
                with_audio=tx.get('audio_mode', 'copy') != 'remove',
                cq=tx.get('cq', 16), preset=tx.get('preset', 'p7'), preview_segment=True))
            record = dict(key=key, identity=identity, job_id=job['id'], output_path=output,
                frame_range=selection, original_offset_seconds=selection['start_frame'] / fps,
                duration_seconds=(selection['end_frame'] - selection['start_frame']) / fps,
                status='pending', ready=False, progress=0)
            self.records[key] = record
            return dict(record)
