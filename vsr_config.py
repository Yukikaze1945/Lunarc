"""NVIDIA VideoSuperRes families, SDK selectors and extent rules."""
import math

FAMILIES = {'standard': (1, 2, 3, 4), 'denoise': (8, 9, 10, 11),
            'deblur': (12, 13, 14, 15), 'high_bitrate': (16, 17, 18, 19),
            'streaming': (21, 23), 'bicubic': (0,)}
SAME_SIZE = ('denoise', 'deblur')
LEVELS = {number for values in FAMILIES.values() for number in values}


def configuration(settings):
    family = settings.get('vsr_mode', 'standard')
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError('Unknown RTX VSR mode family')
    quality = settings.get('vsr_quality', 2)
    if type(quality) is not int or quality not in (1, 2, 3, 4):
        raise ValueError('RTX VSR quality must be 1–4')
    if family == 'streaming' and quality not in (2, 4):
        raise ValueError('Streaming VSR supports medium or ultra quality only')
    strength = settings.get('vsr_strength')
    if strength is not None and (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('RTX VSR strength must be finite in 0..1 or SDK default')
    level = (0 if family == 'bicubic' else (21 if quality == 2 else 23) if family == 'streaming'
             else FAMILIES[family][quality-1])
    return dict(family=family, quality=quality, sdk_level=level,
                strength=None if family == 'bicubic' else strength, same_size=family in SAME_SIZE)


def validate_sdk_options(level, strength, input_size, output_size):
    if type(level) is not int or level not in LEVELS:
        raise ValueError('quality_level is not an NVIDIA VideoSuperRes mode')
    if strength is not None and (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError('VSR strength must be finite in 0..1')
    if level in range(8, 16) and tuple(input_size) != tuple(output_size):
        raise ValueError('VSR denoise/deblur output must match input size')
