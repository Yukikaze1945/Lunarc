"""Real signed user runtime -> independent cache -> two GPU frames.

Not a performance benchmark. Produces local evidence; no runtime redistribution.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dll', required=True)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--catalog-download', action='store_true')
    args = parser.parse_args()
    out = Path(args.evidence).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ['LUNARC_DATA_DIR'] = str(out / 'user-data')
    os.environ['LUNARC_STANDALONE'] = '1'
    import numpy as np
    from dlssnr_runtime import manager, sha256
    from dlss_gpu_engine import Live
    value = manager()
    start = time.perf_counter()
    if args.catalog_download:
        from dlssnr_runtime import KNOWN_VERSIONS
        digest = sha256(args.dll)
        record = value.import_file(catalog_id=digest, confirmed=True)
        assert record['id'] == digest
        def offline(*args):
            raise AssertionError('verified archive cache should not download again')
        value.downloader = offline
        assert value.import_file(catalog_id=digest, confirmed=True)['id'] == digest
        evidence = dict(passed=True, mode='catalog_download', version=record['version'],
            signature=record['signature'], sha256=digest,
            archive_sha256=KNOWN_VERSIONS[digest]['archive_sha256'],
            bytes=record['bytes'], offline_archive_reuse=True, source_preserved=sha256(args.dll) == digest)
        (out / 'download-result.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        print(json.dumps(evidence))
        return
    record = value.import_file(args.dll)
    value.activate(record['id'])
    imported_ms = (time.perf_counter() - start) * 1000
    rows, cols = np.indices((144, 256))
    frame = np.stack([cols % 256, rows % 256, (rows + cols) % 256,
                      np.full_like(rows, 255)], axis=-1).astype(np.uint8)
    reports = []
    live = Live(256, 144, {'style': 0, 'intensity': 1., 'use_auto_mask': 1})
    try:
        assert Path(live.runtime_dll) == Path(record['path'])
        for index in range(2):
            result = live.process_zero_motion(np.roll(frame, index, axis=1), reset=index == 0)
            assert result.shape == frame.shape and result.dtype == np.uint8
            assert result[:, :, :3].max() > 0
            reports.append({'frame': index, 'reset': index == 0, 'process_ms': live.last_process_ms,
                            'sum_rgb': int(result[:, :, :3].sum())})
        evidence = {'passed': True, 'signature': record['signature'], 'version': record['version'],
                    'sha256': record['sha256'], 'cache_path': live.runtime_dll,
                    'source_preserved': sha256(args.dll) == record['sha256'],
                    'session_id': live.session_id, 'host_abi': live.host_abi,
                    'import_and_validation_ms': imported_ms, 'frames': reports,
                    'Performance': 'NOT_CHARACTERIZED'}
    finally:
        live.close()
    (out / 'result.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=True))


if __name__ == '__main__':
    main()
