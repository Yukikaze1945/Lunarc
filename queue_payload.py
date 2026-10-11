"""Detached queue messages and lossless JSON representation of native diagnostics."""
import base64


def diagnostic_json(value, *, summary=False):
    # Encoder plans contain UTF-8 C-string options (codec, preset, tune, ...).
    # Arbitrary binary evidence must retain its bytes, not become repr(bytes).
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        try:
            text = raw.decode('utf-8')
            if all(ord(char) >= 32 or char in '\t\n\r' for char in text):
                return text
        except UnicodeDecodeError:
            pass
        return {'encoding': 'base64', 'byte_length': len(raw),
                'data': base64.b64encode(raw).decode('ascii')}
    if isinstance(value, dict):
        return {key: diagnostic_json(item, summary=summary) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if summary and len(value) > 64:
            result = {'item_count': len(value), 'details_omitted': True}
            if all(isinstance(item, dict) and 'converged' in item for item in value):
                result['all_converged'] = all(bool(item['converged']) for item in value)
            return result
        return [diagnostic_json(item, summary=summary) for item in value]
    return value


def queue_job(job):
    # Full frame reports belong to the diagnostics endpoint. Traversing them on
    # each queue refresh stalls the control plane after a long export.
    snapshot = {key: value for key, value in job.items()
                if key != 'residual_backend_diagnostics'}
    snapshot['diagnostics_available'] = bool(job.get('residual_backend_diagnostics'))
    return diagnostic_json(snapshot)


def queue_message(data):
    snapshot = dict(data)
    if isinstance(snapshot.get('job'), dict):
        snapshot['job'] = queue_job(snapshot['job'])
    if isinstance(snapshot.get('jobs'), (list, tuple)):
        snapshot['jobs'] = [queue_job(job) for job in snapshot['jobs']]
    return diagnostic_json(snapshot)
