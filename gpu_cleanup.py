"""Attempt every owned release in caller-defined dependency order."""

def release_owned(owner, name, errors, callback=None, method='close'):
    resource = owner.get(name) if isinstance(owner, dict) else getattr(owner, name, None)
    if resource is None:
        return
    if isinstance(owner, dict):
        owner[name] = None
    else:
        setattr(owner, name, None)
    try:
        if callback:
            callback(resource)
        else:
            getattr(resource, method)()
    except Exception as error:
        errors.append((name, error))


def finish_cleanup(errors):
    if errors:
        raise RuntimeError('GPU resource cleanup failed: ' + '; '.join(
            f'{name}: {error}' for name, error in errors)) from errors[0][1]
