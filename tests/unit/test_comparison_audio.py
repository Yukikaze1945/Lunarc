import shutil
import subprocess
import wave

import numpy as np
import pytest

from comparison_audio import audio_chunks, wav_range


FFMPEG = shutil.which("ffmpeg")


@pytest.mark.skipif(not FFMPEG or not __import__("os").path.isfile(FFMPEG),
                    reason="ffmpeg executable is unavailable")
def test_nonzero_pts_audio_ranges_match_full_pcm_decode(tmp_path):
    source_wav = tmp_path / "audio.wav"
    delayed = tmp_path / "delayed.mka"
    samples = (np.sin(np.arange(9600) * 2 * np.pi / 37) * 11000).astype("<i2")
    stereo = np.column_stack((samples, -samples)).astype("<i2")
    with wave.open(str(source_wav), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(48000)
        out.writeframes(stereo.tobytes())
    subprocess.run([FFMPEG, "-hide_banner", "-v", "error", "-y", "-itsoffset", "1.25",
                    "-i", str(source_wav), "-c:a", "pcm_s16le", str(delayed)],
                   check=True, capture_output=True)

    source = {"duration_seconds": .2, "origin_pts": 1.25,
              "path": str(delayed), "stream_index": 0}
    header, start, end, total = wav_range(source)
    full = b"".join(audio_chunks(FFMPEG, source, header, start, end))
    assert len(full) == total
    # Include an unaligned byte start and ranges crossing multiple read chunks.
    for first, last in ((44 + 1, 44 + 801), (44 + 7981, 44 + 19004),
                        (44 + 32767, 44 + 40031)):
        _, range_start, range_end, _ = wav_range(source, f"bytes={first}-{last}")
        actual = b"".join(audio_chunks(FFMPEG, source, header, range_start, range_end))
        assert actual == full[range_start:range_end + 1], (first, last,
            actual[:16].hex(), full[range_start:range_start + 16].hex())
