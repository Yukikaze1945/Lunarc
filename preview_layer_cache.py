"""Borrow session-owned NR outputs only for the same paused source frame.

The preview owner clears this cache before a new frame or pipeline. A different
layer's Evaluate does not overwrite the cached session's private output texture.
Export never enables the cache.
"""


def evaluate(chain, number, inference):
    cache = getattr(chain, '_preview_nr_cache', None)
    if cache is not None and number in getattr(chain, '_preview_reuse_layers', set()) and number in cache:
        return cache[number], True
    result = inference()
    if cache is not None:
        cache[number] = result
    return result, False
