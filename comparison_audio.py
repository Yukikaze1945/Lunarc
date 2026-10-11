"""Seekable display-only PCM stream; never writes or remuxes a source file."""
import math
import re
import struct
import subprocess
import tempfile


def wav_range(source, range_header=None):
    samples = math.ceil(float(source['duration_seconds']) * 48000)
    size = samples * 4
    if size > 0xffffffff - 36:
        raise ValueError('The display audio duration exceeds the WAV range limit')
    header = (b'RIFF' + struct.pack('<I', size + 36) + b'WAVEfmt ' +
              struct.pack('<IHHIIHH', 16, 1, 2, 48000, 192000, 4, 16) +
              b'data' + struct.pack('<I', size))
    total = size + 44
    start, end = 0, total - 1
    if range_header:
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', range_header.strip())
        if not match or not any(match.groups()):
            raise ValueError('Unsupported audio range')
        first, last = match.groups()
        if not first:
            start = max(0, total - int(last))
        else:
            start = int(first)
            end = min(end, int(last)) if last else min(end, start + 4 * 1024 * 1024 - 1)
        if start >= total or end < start:
            raise ValueError('Audio range is outside the stream')
    return header, start, end, total


def audio_chunks(ffmpeg, source, header, start, end):
    if start < 44:
        yield header[start:min(44, end + 1)]
    position = max(44, start) - 44
    remaining = end + 1 - max(44, start)
    if remaining <= 0:
        return
    # Whole sample alignment, discard only the byte offset requested by HTTP.
    sample = position // 4
    discard = position % 4
    pts = float(source['origin_pts']) + sample / 48000
    # FFmpeg's output -t is timestamp based and can round a partial sample down.
    # Request whole samples through the final byte, then trim to the HTTP range.
    duration = math.ceil((remaining + discard) / 4) * 4 / 192000
    command = [ffmpeg, '-hide_banner', '-v', 'error', '-nostdin',
        '-i', source['path'], '-ss', f'{pts:.9f}',
        '-map', f"0:{source['stream_index']}", '-vn', '-sn', '-t', f'{duration:.9f}',
        '-af', 'aresample=48000:async=1:first_pts=0', '-ac', '2', '-ar', '48000',
        '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1']
    # Keep stderr out of a pipe so a verbose decoder error cannot deadlock while
    # this generator is draining stdout. A clean, shorter audio stream is padded
    # to the video's requested timeline; a failed decoder is always an error.
    error_log = tempfile.TemporaryFile()
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=error_log)
    try:
        discarded = process.stdout.read(discard) if discard else b''
        ended_early = len(discarded) != discard
        while remaining:
            chunk = b'' if ended_early else process.stdout.read(min(65536, remaining))
            if not chunk:
                return_code = process.wait()
                if return_code:
                    error_log.seek(0, 2)
                    error_log.seek(max(0, error_log.tell() - 1200))
                    detail = error_log.read().decode('utf-8', 'replace').strip()
                    raise RuntimeError('FFmpeg audio decode failed' +
                                       (f': {detail}' if detail else ''))
                # A successful decoder may finish because the audio ends before
                # the video. Only this clean EOF is represented as silence.
                chunk = bytes(min(65536, remaining))
                ended_early = True
            remaining -= len(chunk)
            yield chunk
        if process.poll() is None:
            process.wait()
        if process.returncode:
            error_log.seek(0, 2)
            error_log.seek(max(0, error_log.tell() - 1200))
            detail = error_log.read().decode('utf-8', 'replace').strip()
            raise RuntimeError('FFmpeg audio decode failed' +
                               (f': {detail}' if detail else ''))
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()
        error_log.close()
