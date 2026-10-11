"""Shared, recorded colour interpretation for untagged HD SDR10 media.

Missing colour tags are not a reason to read a P010 picture as NV12. For HD
10-bit 4:2:0 with an explicit limited range and no conflicting colour/HDR
declaration, use the BT.709 SDR interpretation. Keep the original declarations
and list every assumed field; this does not establish the source's true colour.
"""

COLOUR_FIELDS = ('color_range', 'color_space', 'color_transfer', 'color_primaries')
MISSING_TAGS = (None, '', 'unknown', 'unspecified')


def resolve_source_colour(source):
    result = dict(source)
    if source.get('colour_resolution'):
        return result
    if (source.get('pix_fmt') != 'yuv420p10le'
            or source.get('color_range') not in ('tv', 'mpeg', 'limited')
            or source.get('is_hdr')
            or max(int(source.get('width') or 0), int(source.get('height') or 0)) < 1280):
        return result
    fields = COLOUR_FIELDS[1:]
    if any(source.get(name) not in (*MISSING_TAGS, 'bt709') for name in fields):
        return result
    side_data = source.get('side_data_list') or []
    if any(any(marker in str(item.get('side_data_type', '')).lower()
               for marker in ('mastering display', 'content light', 'dovi', 'hdr'))
           for item in side_data):
        return result
    assumed = [name for name in fields if source.get(name) in MISSING_TAGS]
    if not assumed:
        return result
    for name in assumed:
        result[name] = 'bt709'
    result['colour_resolution'] = {
        'policy': 'untagged_hd_sdr10_bt709',
        'source_declared': {name: source.get(name) for name in COLOUR_FIELDS},
        'effective': {name: result.get(name) for name in COLOUR_FIELDS},
        'assumed_fields': assumed,
    }
    return result


def resolve_frame_colour(colour, resolution):
    """Resolve only AVColor* UNSPECIFIED (2), never replace an explicit enum."""
    result = dict(colour)
    for name in (resolution or {}).get('assumed_fields', []):
        if name in COLOUR_FIELDS[1:] and result.get(name) == 2:
            result[name] = 1  # AVColorSpace/Transfer/Primaries BT709
    return result
