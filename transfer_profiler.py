"""Transfer accounting for the export path: which bytes moved, and who says so.

StageProfiler answers "how long did a stage take". This module answers a different
question - "how many bytes crossed a boundary, in which direction, and is that number
measured, estimated, logical or simply not observable". The two are kept apart because a
duration and a byte count can disagree (a stage can be fast and still move 24 MB), and
because a byte figure that nobody measured must never look like one that somebody did.

Every field has a declared basis in FIELD_BASIS, and a field whose basis is
"unavailable" carries the value None. 0 is reserved for "measured, and it really was 0",
which is why the counters here are incremented at the call site that does the work
rather than being asserted by whoever reads them.
"""

import cv2
import threading

# ---------------------------------------------------------------------------
# CPU colour-conversion instrumentation
# ---------------------------------------------------------------------------

# Codes the pipeline actually uses. Anything else routed through convert_color() is
# counted under its own name so a new conversion cannot hide inside an existing bucket.
CONVERSION_NAMES = {
    int(cv2.COLOR_BGR2RGBA): "cpu_bgr_to_rgba",
    int(cv2.COLOR_BGRA2RGBA): "cpu_bgra_to_rgba",
    int(cv2.COLOR_BGR2BGRA): "cpu_bgr_to_bgra",
    int(cv2.COLOR_RGBA2BGR): "cpu_rgba_to_bgr",
    int(cv2.COLOR_BGRA2BGR): "cpu_bgra_to_bgr",
    int(cv2.COLOR_GRAY2RGBA): "cpu_gray_to_rgba",
    int(cv2.COLOR_YUV2BGR_NV12): "cpu_yuv_to_bgr",
}

# The conversions this ledger tracks bytes for: the ones that exist only to change a
# buffer's channel order between two stages, i.e. the ones a transfer reduction can
# delete. The decoder's NV12 -> BGR is a pixel-format decode, not a channel shuffle, so
# it is counted under its own name but stays out of cpu_color_conversion_output_bytes.
TRACKED_CONVERSIONS = ("cpu_bgr_to_rgba", "cpu_rgba_to_bgr")

# The call sites a tracked conversion can come from. Seeded into every delta so a site
# that ran without calling reads as a measured zero instead of as a missing row.
TRACKED_SITES = {
    "cpu_bgr_to_rgba": ("dlss_input", "nvof_input"),
    "cpu_rgba_to_bgr": ("residual_input",),
}

_lock = threading.Lock()
_counts = {}
_output_bytes = {}

# The real OpenCV entry point. Tests replace it to prove the counters are driven by the
# calls that actually happen: if the hook and the counter do not move together, the
# counter is decoration.
_cvtColor = cv2.cvtColor


def conversion_name(code):
    return CONVERSION_NAMES.get(int(code), f"cpu_cvtColor_{int(code)}")


def _key(name, site):
    return f"{name}@{site}" if site else name


def _base(key):
    return key.split("@", 1)[0]


def convert_color(src, code, dst=None, site=None):
    """The single cv2.cvtColor seam used by the production frame path.

    Counting happens here, at the call site, on the way to the real conversion; a call
    site that is deleted stops incrementing, which is the only reason the resulting zero
    means anything. `site` names the caller, so a ledger row can show which of two
    conversions went away instead of only that one fewer happened.
    """
    name = conversion_name(code)
    if name in TRACKED_CONVERSIONS and not site:
        raise ValueError(f"{name} must name its call site: pass site=...")
    key = _key(name, site)
    out = _cvtColor(src, code, dst=dst) if dst is not None else _cvtColor(src, code)
    nbytes = int(out.nbytes)
    with _lock:
        _counts[key] = _counts.get(key, 0) + 1
        _output_bytes[key] = _output_bytes.get(key, 0) + nbytes
    return out


def conversion_snapshot():
    """Cumulative call counts and output byte totals, keyed by conversion name and site."""
    with _lock:
        return (dict(_counts), dict(_output_bytes))


def reset_conversions():
    with _lock:
        _counts.clear()
        _output_bytes.clear()


def conversion_deltas(previous):
    """The tracked-conversion fields as deltas since `previous`.

    The per-name totals answer "did this frame shuffle channels" and the per-site fields
    answer "which shuffle", which is what a ledger row has to be able to point at. Every
    site in TRACKED_SITES is reported whether or not it was reached, so a site that stops
    converting reads as a measured 0 instead of as a row that vanished from the ledger.
    """
    previous_counts, previous_bytes = previous
    counts, output_bytes = conversion_snapshot()
    fields = {}
    total_bytes = 0
    for name in TRACKED_CONVERSIONS:
        keys = {_key(name, site) for site in TRACKED_SITES.get(name, ())}
        keys |= {k for k in set(counts) | set(previous_counts) if _base(k) == name}
        calls = 0
        for key in sorted(keys):
            delta = counts.get(key, 0) - previous_counts.get(key, 0)
            delta_bytes = output_bytes.get(key, 0) - previous_bytes.get(key, 0)
            fields[key + "_calls"] = delta
            fields[key + "_output_bytes"] = delta_bytes
            calls += delta
            total_bytes += delta_bytes
        fields[f"{name}_calls"] = calls
    fields["cpu_color_conversion_output_bytes"] = total_bytes
    return fields, (counts, output_bytes)


# ---------------------------------------------------------------------------
# Child-process byte pipes
# ---------------------------------------------------------------------------

class PipeByteLedger:
    """Bytes moved across one child-process byte stream, per frame and cumulative.

    The producer thread owns `note_frame`; the consuming thread reads per-frame values
    back with `take_frame`. Both go through one lock, and the cumulative total is what a
    caller uses when it needs a figure that does not depend on frame attribution.
    """

    def __init__(self, name):
        self.name = name
        self._lock = threading.Lock()
        self._total = 0
        self._frames = {}
        self._frame_count = 0

    def note_frame(self, idx, nbytes):
        with self._lock:
            self._total += int(nbytes)
            self._frames[idx] = int(nbytes)
            self._frame_count += 1

    def take_frame(self, idx):
        with self._lock:
            value = self._frames.pop(idx, None)
        return value

    def snapshot(self):
        with self._lock:
            return {"total_bytes": self._total, "frames": self._frame_count,
                    "unattributed_frames": len(self._frames)}

    def delta_since(self, previous_total):
        current = self.snapshot()["total_bytes"]
        return current - int(previous_total)


DECODER_PIPE = PipeByteLedger("decoder_stdout")
ENCODER_PIPE = PipeByteLedger("encoder_stdin")


def reset_pipes():
    """Drop the per-frame attribution; the cumulative totals stay, they are history."""
    for ledger in (DECODER_PIPE, ENCODER_PIPE):
        with ledger._lock:
            ledger._frames.clear()


def unmeasured_pipe_fields():
    """The pipe keys, empty. A frame record always carries them, so a reader looking
    for a number that is not there finds None instead of a missing key."""
    return {"encoder_pipe_bytes": None, "decoder_pipe_bytes": None,
            "decoder_gpu_to_cpu_bytes_estimated": None}


def pipe_fields(frame_idx, gpu_decoded):
    """The two child-process byte totals for one frame, as that frame moved them.

    `gpu_decoded` says whether the decoder child was downloading frames off the GPU at
    all. When it was not there is no GPU to CPU payload to report, so the field is None
    instead of a zero that nobody measured.
    """
    encoder = ENCODER_PIPE.take_frame(frame_idx)
    decoder = DECODER_PIPE.take_frame(frame_idx)
    return {
        "encoder_pipe_bytes": encoder,
        "decoder_pipe_bytes": decoder,
        "decoder_gpu_to_cpu_bytes_estimated": decoder if gpu_decoded else None,
    }


# ---------------------------------------------------------------------------
# Field basis
# ---------------------------------------------------------------------------

# What each per-frame field is, in one word, and where the number comes from. Written
# into every summary so a reader never has to guess whether a byte count was measured.
FIELD_BASIS = {
    "decoder_pipe_bytes": (
        "measured", "bytes read out of the decoder child process' stdout, summed from "
                    "the reads that produced the frame"),
    "decoder_gpu_to_cpu_bytes_estimated": (
        "estimated", "NV12 payload the hwdownload filter writes when CUDA decode runs; "
                     "FFmpeg exposes no transfer counter, so this is arithmetic on the "
                     "frame format, not a measurement"),
    "cpu_bgr_to_rgba_calls": (
        "measured", "incremented at the convert_color() call site that does the work"),
    "cpu_rgba_to_bgr_calls": (
        "measured", "incremented at the convert_color() call site that does the work"),
    "cpu_color_conversion_output_bytes": (
        "measured", "output buffer size of each tracked channel-order conversion"),
    "nvof_input_format": (
        "reported", "the format the flow engine says it handed to the NVOF bridge"),
    "nvof_h2d_copy_count": (
        "measured", "nvof_bridge_get_counters, read either side of the frame"),
    "nvof_h2d_bytes_measured": (
        "measured", "nvof_bridge_get_counters, read either side of the frame"),
    "nvof_d2h_copy_count": (
        "measured", "nvof_bridge_get_counters, read either side of the frame"),
    "nvof_d2h_bytes_measured": (
        "measured", "nvof_bridge_get_counters, read either side of the frame"),
    "dlss_input_color_bytes_logical": (
        "logical", "size of the host buffer passed as the DLSS colour input; the host "
                   "wrapper's internal transfers are not observable"),
    "dlss_input_flow_bytes_logical": (
        "logical", "size of the host buffer passed as the DLSS motion input"),
    "dlss_input_depth_bytes_logical": (
        "logical", "size of the host buffer passed as the DLSS depth input"),
    "dlss_output_bytes_logical": (
        "logical", "size of the host buffer the DLSS output was written into"),
    "dlss_gpu_transfer_bytes_measured": (
        "unavailable", "the GPU host does report staging and readback counters "
                       "(dlss_gpu_host_get_transfer_counters), but this per-frame field "
                       "belongs to the pipe backend, which never reads them; the "
                       "GPU-transfer backend publishes the same counters under "
                       "dlss_host_color_stage_bytes_measured and "
                       "dlss_output_readback_bytes_measured. There is no number here, and "
                       "the buffer sizes above are not one"),
    "encoder_pipe_bytes": (
        "measured", "bytes returned by the write into the encoder child process' stdin"),
    "encoder_gpu_upload_bytes_measured": (
        "unavailable", "FFmpeg uploads the rawvideo it reads, and exposes no counter "
                       "for that copy"),
}

# One basis line per tracked call site, derived from TRACKED_SITES so a site cannot be
# added to the seam without also being given a meaning in the summary. These are the
# fields a ledger row points at: a row is about one conversion in one place, not about
# how many conversions the frame happened to make.
for _conversion, _sites in TRACKED_SITES.items():
    for _site in _sites:
        _keyed = f"{_conversion}@{_site}"
        FIELD_BASIS[f"{_keyed}_calls"] = (
            "measured", f"{_conversion} calls made by the {_site} conversion site of the "
                        f"production path, incremented at that site")
        FIELD_BASIS[f"{_keyed}_output_bytes"] = (
            "measured", f"bytes of the buffer the {_site} {_conversion} conversion wrote")

FIELDS = tuple(FIELD_BASIS)


class TransferProfiler:
    """Collects one transfer record per frame and reports the bytes, with their basis."""

    def __init__(self):
        self.records = []

    def record(self, stage_dict):
        self.records.append(dict(stage_dict))

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _stats(values):
        ordered = sorted(values)
        n = len(ordered)
        return {
            "frames_observed": n,
            "total": int(sum(ordered)),
            "mean": sum(ordered) / n,
            "min": int(ordered[0]),
            "p50": int(ordered[n // 2]),
            "p95": int(ordered[min(n - 1, int(round(0.95 * (n - 1))))]),
            "max": int(ordered[-1]),
        }

    def summary(self):
        """Per-field stats over all frames and over steady-state frames only.

        A field that was None on some frames is reported on the frames that measured it,
        and frames_measured says how many that was. A field that was None on every frame
        is reported as None rather than as a run of zeros.
        """
        out = {}
        for key in FIELDS:
            values = [r[key] for r in self.records
                      if key in r and isinstance(r[key], (int, float))]
            steady = [r[key] for r in self.records[1:]
                      if key in r and isinstance(r[key], (int, float))]
            entry = {
                "basis": FIELD_BASIS[key][0],
                "source": FIELD_BASIS[key][1],
                "frames_in_records": len(self.records),
                "frames_measured": len(values),
                "all_frames": self._stats(values) if values else None,
                "steady_state": self._stats(steady) if steady else None,
            }
            if key == "nvof_input_format":
                entry["values"] = sorted({r.get(key) for r in self.records if r.get(key)})
            out[key] = entry
        return out

    def as_dict(self):
        return {"frame_count": len(self.records),
                "records": [dict(r) for r in self.records],
                "summary": self.summary()}
