"""Gate: claims that were removed as misleading must not grow back.

Each phrase below is a sentence that used to describe this pipeline and turned out not to
mean what it sounded like:

* "Direct-to-NVENC" / "Zero-Disk-IO" - the export writes frames into an FFmpeg process'
  stdin as rawvideo. No intermediate PNG, but every frame sits in CPU memory on its way,
  so it is a pipe, not a GPU path.
* "zero-copy Direct GPU access" for nvof_bridge_get_output_deviceptr - the call reports an
  address. DLSSNR takes host pointers, nothing consumes that value, and the flow the
  pipeline uses arrives by a D2H copy.
* "Elimination of per-frame host memory allocations" - the buffer that is allocated once
  per session is on the GPU, and the per-frame copy out of it is the opposite of zero-copy.
* "均平稳无泄漏" in a lifecycle test - ten cycles that return are not evidence about
  memory being freed.
* "residual_backend = cpu_fallback" with an int16 OpenCV description - the residual is
  computed by the native bridge in FP32.

The renames of the GPU memory metrics are checked the same way: a number that belongs to
the card must be named for the card, and the per-process figure has to be asked for
explicitly instead of being represented by the card total.
"""

import os
import re

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SKIP_DIRS = {".git", "Magpie", "node_modules", "dist", "perf_results", "scratch",
             "__pycache__", ".pytest_cache", "bench_out", ".venv", "venv", "env", "build"}
SUFFIXES = {".py", ".md", ".ts", ".tsx", ".cpp", ".h", ".bat", ".json", ".yaml", ".yml"}

# Case-insensitive; matched against the whole file text, so they fail wherever they return.
RETIRED_PHRASES = [
    "Direct-to-NVENC",
    "Zero-Disk-IO",
    "zero-copy Direct GPU access",
    "zero-copy GPU flow tensor",
    "Elimination of per-frame host memory allocations",
    "Direct GPU memory query for NVOF -> DLSS resource sharing",
    "Query GPU device pointer and pitch for zero-copy",
    "residual_backend = \"cpu_fallback\"",
    "均平稳无泄漏",
    "纯显存",
    "显存直推",
]

# Names that used to stand for the whole card's memory while reading as process memory.
RETIRED_METRIC_NAMES = ["peak_vram_bytes", "vram_bytes_max", "def vram_bytes(",
                        "\"vram_bytes\""]

# A dated audit snapshot is allowed to quote a claim it then corrects, but only while it
# says out loud that it is a snapshot of an older commit. Without the banner the phrases
# below stop being history and start being documentation of the current tree.
SNAPSHOT_BANNERS = {
    "CURRENT_STATE_AUDIT.md": "这份文件是那一天的快照",
}

# This module quotes every retired phrase in order to forbid it.
SELF = os.path.abspath(__file__)


def _source_files():
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if os.path.splitext(name)[1].lower() in SUFFIXES:
                yield os.path.join(dirpath, name)


def _text(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _candidates():
    """Every source file that has to be free of the retired claims."""
    for path in _source_files():
        if os.path.abspath(path) == SELF:
            continue
        rel = os.path.relpath(path, _ROOT).replace("\\", "/")
        text = _text(path)
        banner = SNAPSHOT_BANNERS.get(rel)
        if banner:
            assert banner in text, (
                f"{rel} is exempt from the retired-claim gate only because it carries the "
                f"banner {banner!r} telling the reader it is a dated snapshot")
            continue
        yield rel, text


def test_no_file_still_carries_a_retired_claim():
    hits = []
    for rel, text in _candidates():
        for phrase in RETIRED_PHRASES:
            if phrase.lower() in text.lower():
                hits.append(f"{rel}: {phrase}")
    assert not hits, ("these sentences were removed because they described something this "
                      f"code does not do: {hits}")


def test_no_metric_uses_the_old_card_total_name():
    hits = []
    for rel, text in _candidates():
        if not rel.endswith(".py"):
            continue
        for name in RETIRED_METRIC_NAMES:
            if name in text:
                hits.append(f"{rel}: {name}")
    assert not hits, (
        f"GPU memory metrics must be named for whose memory they are: {hits}")


def test_gpu_memory_metrics_state_their_scope():
    """The card figure is renamed, and the process figure is asked for separately."""
    path = os.path.join(_ROOT, "tests", "unit", "resource_metrics.py")
    text = _text(path)
    assert "def gpu_total_memory_used_bytes(" in text, "the card counter must be renamed"
    assert "def process_gpu_memory_bytes(" in text, (
        "per-process GPU memory has to be a question of its own; nvidia-smi on this "
        "driver answers [N/A] per process, so the honest return is None plus the reason")
    assert "process_gpu_memory_source" in text, (
        "'unavailable' must be recorded as such instead of being left out")


def test_the_export_docstring_describes_the_pipe_it_runs():
    import pipeline

    doc = pipeline.export_dlss_stream.__doc__ or ""
    assert "NVENC" in doc and "rawvideo" in doc, (
        "the docstring must say what the encoder child process is fed")
    assert re.search(r"not a GPU path|CPU memory", doc), (
        "the sentence that keeps the pipe from reading like a GPU path is the one that "
        "carries the whole correction; dropping it re-introduces the old claim")


def test_zero_copy_wording_only_survives_as_a_denial():
    """Where the term is still allowed, it must be saying that the thing does not exist."""
    offenders = []
    for rel, text in _candidates():
        for match in re.finditer(r"(?i)zero[- _]?copy", text):
            around = text[max(0, match.start() - 260):match.end() + 260]
            if not re.search(r"(?i)(not|never|no)\b|BLOCKED", around):
                offenders.append(rel)
                break
    assert not offenders, (
        "'zero-copy' appears as a positive claim again in "
        f"{sorted(set(offenders))}. If something really became zero-copy, that needs a "
        "measurement behind it, not just the word")


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
