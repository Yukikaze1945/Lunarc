# -*- coding: utf-8 -*-
"""Gate: the transfer counters measure the path, and an unmeasured figure stays unmeasured.

Two rules hold this whole phase together, and both are about honesty rather than speed.

1. A count has to come from the calls that ran. The conversion counter is checked against
   an independent observer wired into the same seam, so a record that simply asserted 0
   while the conversion still happened cannot pass.
2. A field nothing measured is None, never 0. 0 means "measured, and it really was zero",
   which is exactly the claim a fabricated zero would be making.

The ledger rows are checked against arithmetic (the 4K and 1080p payload sizes the phase
is specified against) and against the stage order, because a ledger that silently drops a
stage is how a round trip gets lost from a report instead of from the code.
"""

import os
import sys

import cv2
import numpy as np
import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import guidance_stubs as gs  # noqa: E402
import pipeline  # noqa: E402
import transfer_ledger as tl  # noqa: E402
import transfer_profiler as tp  # noqa: E402


class ConverterLog:
    """A stand-in conversion that delegates and remembers being called.

    Installed only inside a test. It exists so the counter can be compared against
    something that is not the counter: if the log and the counter disagree, the counter
    is decoration.
    """

    def __init__(self):
        self.calls = []

    def __call__(self, src, code, dst=None):
        self.calls.append((int(code), src.shape, None if dst is None else tuple(dst.shape)))
        return cv2.cvtColor(src, code, dst=dst) if dst is not None else cv2.cvtColor(src, code)

    def names(self):
        return [tp.conversion_name(code) for code, _shape, _dst in self.calls]


@pytest.fixture
def converter_log(monkeypatch):
    log = ConverterLog()
    monkeypatch.setattr(tp, "_cvtColor", log)
    tp.reset_conversions()
    return log


def _frame(h=24, w=32, channels=3):
    return np.full((h, w, channels), 90, dtype=np.uint8)


# ---------------------------------------------------------------------------
# the conversion seam
# ---------------------------------------------------------------------------

def test_counter_matches_the_converter_that_was_reached(converter_log):
    before = tp.conversion_snapshot()
    src = _frame()
    gray = np.full((24, 32), 90, dtype=np.uint8)
    tp.convert_color(src, cv2.COLOR_BGR2RGBA, dst=np.empty((24, 32, 4), np.uint8),
                     site="dlss_input")
    tp.convert_color(src, cv2.COLOR_BGR2RGBA, dst=np.empty((24, 32, 4), np.uint8),
                     site="nvof_input")
    tp.convert_color(gray, cv2.COLOR_GRAY2RGBA, dst=np.empty((24, 32, 4), np.uint8),
                     site="decode")

    fields, _after = tp.conversion_deltas(before)
    assert converter_log.names() == ["cpu_bgr_to_rgba", "cpu_bgr_to_rgba", "cpu_gray_to_rgba"]
    # The grey conversion lands in its own bucket instead of borrowing the tracked one:
    # the counter follows the calls, and a call that is not in a tracked bucket cannot be
    # reported as one that is.
    assert fields["cpu_bgr_to_rgba_calls"] == 2
    assert fields["cpu_rgba_to_bgr_calls"] == 0
    assert fields["cpu_color_conversion_output_bytes"] == 2 * 24 * 32 * 4
    # ...and the two RGBA calls are two different places, which is the only way a ledger
    # row can say that one of them went away.
    assert fields["cpu_bgr_to_rgba@dlss_input_calls"] == 1
    assert fields["cpu_bgr_to_rgba@nvof_input_calls"] == 1
    assert fields["cpu_bgr_to_rgba@nvof_input_output_bytes"] == 24 * 32 * 4


def test_a_tracked_conversion_cannot_be_called_without_naming_its_site():
    """An anonymous conversion would land in a bucket no row reads, which is how a
    round trip disappears from a report without disappearing from the code."""
    tp.reset_conversions()
    src = _frame()
    with pytest.raises(ValueError):
        tp.convert_color(src, cv2.COLOR_BGR2RGBA, dst=np.empty((24, 32, 4), np.uint8))
    with pytest.raises(ValueError):
        tp.convert_color(src, cv2.COLOR_RGBA2BGR, dst=np.empty((24, 32, 3), np.uint8))
    # Nothing was counted, because the seam refused to run the conversion at all.
    assert tp.conversion_snapshot() == ({}, {})


def test_an_untracked_conversion_does_not_inflate_a_tracked_field(converter_log):
    before = tp.conversion_snapshot()
    # A real NV12 plane: single channel, height * 1.5 rows, which is what the decoder's
    # YUV2BGR call is handed.
    nv12 = np.full((12, 8), 128, dtype=np.uint8)
    tp.convert_color(nv12, cv2.COLOR_YUV2BGR_NV12, site="decode")
    fields, _after = tp.conversion_deltas(before)
    assert fields["cpu_bgr_to_rgba_calls"] == 0
    assert fields["cpu_rgba_to_bgr_calls"] == 0
    assert fields["cpu_color_conversion_output_bytes"] == 0
    assert tp.conversion_snapshot()[0].get("cpu_yuv_to_bgr@decode") == 1


# ---------------------------------------------------------------------------
# the frame path, not the source text
# ---------------------------------------------------------------------------

def _run_one_frame():
    """One legacy-order frame: these tests count the conversions the legacy path makes.

    The mode is pinned rather than left to auto because the point of these gates is the
    counter mechanism, and a counter test whose ground moves when the default order changes
    stops testing the counter and starts testing the default. The reduced order has its own
    gates in test_transfer_modes.py.
    """
    core = gs.make_core(live_w=32, live_h=24, src_w=64, src_h=48, guidance_mode=1,
                        flow_service=gs.StubFlowService(),
                        job_settings_extra={"transfer_mode": "legacy"})
    _final, record = core.process_frame(0, gs.source_frame(64, 48))
    core.close()
    return record


def test_record_counts_the_conversions_of_the_frame_that_ran(converter_log):
    """The per-frame record is compared against the log, not against a hoped-for number."""
    core = gs.make_core(live_w=32, live_h=24, src_w=64, src_h=48, guidance_mode=1,
                        flow_service=gs.StubFlowService(),
                        job_settings_extra={"transfer_mode": "legacy"})
    try:
        tp.reset_conversions()
        _final, first = core.process_frame(0, gs.source_frame(64, 48))
        calls_at_first = len(converter_log.calls)
        _final, second = core.process_frame(1, gs.source_frame(64, 48))
    finally:
        core.close()

    assert calls_at_first == 2, "one into the DLSS input, one out of the DLSS output"
    for record in (first, second):
        assert record["cpu_bgr_to_rgba_calls"] == 1
        assert record["cpu_rgba_to_bgr_calls"] == 1
        assert record["cpu_color_conversion_output_bytes"] == 32 * 24 * 4 + 32 * 24 * 3
    # The log grew by exactly what the two frames reported: nothing counted twice,
    # nothing counted that did not happen.
    assert len(converter_log.calls) == 4
    # Each counted call is the buffer the log was handed, so the byte figure is the
    # conversion's own output size rather than a width times a height written elsewhere.
    logged_rgba = [dst for _code, _s, dst in converter_log.calls if dst and dst[2] == 4]
    logged_bgr = [dst for _code, _s, dst in converter_log.calls if dst and dst[2] == 3]
    assert len(logged_rgba) == 2 and len(logged_bgr) == 2
    assert first["cpu_bgr_to_rgba@dlss_input_output_bytes"] == 32 * 24 * 4
    assert first["cpu_rgba_to_bgr@residual_input_output_bytes"] == 32 * 24 * 3


def test_a_site_the_frame_never_reached_reads_as_measured_zero(converter_log):
    """The stub flow service converts nothing, so the NVOF site must report a zero.

    This is the shape the whole phase rests on: when a call site goes away the row stays
    and says 0, which is only distinguishable from "nobody looked" because the other two
    sites in the same record say 1.
    """
    record = _run_one_frame()
    assert len(converter_log.calls) == 2, "the frame ran two conversions, no more"
    assert record["cpu_bgr_to_rgba_calls"] == 1
    assert record["cpu_bgr_to_rgba@dlss_input_calls"] == 1
    assert record["cpu_bgr_to_rgba@nvof_input_calls"] == 0
    assert record["cpu_bgr_to_rgba@nvof_input_output_bytes"] == 0
    assert record["cpu_rgba_to_bgr@residual_input_calls"] == 1

    profiler = tp.TransferProfiler()
    profiler.record(record)
    summary = profiler.summary()["cpu_bgr_to_rgba@nvof_input_calls"]
    assert summary["basis"] == "measured"
    assert summary["frames_measured"] == 1
    assert summary["all_frames"]["total"] == 0


def test_logical_dlss_sizes_are_reported_as_sizes():
    record = _run_one_frame()
    assert record["dlss_input_color_bytes_logical"] == 32 * 24 * 4
    assert record["dlss_input_flow_bytes_logical"] == 32 * 24 * 2 * 4
    # The GPU host answers depth from the zero texture it made at open, so this is a
    # measured nothing rather than a size: there is no depth array in the run at all.
    assert record["depth_source"] == "gpu_zero_texture"
    assert record["dlss_input_depth_bytes_logical"] == 0
    assert record["dlss_output_bytes_logical"] == 32 * 24 * 4
    assert record["dlss_output_bytes_logical"] == 32 * 24 * 4
    assert all(tp.FIELD_BASIS[k][0] == "logical" for k in (
        "dlss_input_color_bytes_logical", "dlss_input_flow_bytes_logical",
        "dlss_input_depth_bytes_logical", "dlss_output_bytes_logical"))


# ---------------------------------------------------------------------------
# None, not 0
# ---------------------------------------------------------------------------

def test_fields_nothing_measured_arrive_as_none_and_summary_says_so():
    record = _run_one_frame()
    # The stub flow service has no native counters, so these were never measured: they
    # must not read as "the GPU moved nothing".
    for key in ("nvof_h2d_bytes_measured", "nvof_d2h_bytes_measured",
                "nvof_h2d_copy_count", "nvof_d2h_copy_count", "nvof_input_format",
                "dlss_gpu_transfer_bytes_measured", "encoder_gpu_upload_bytes_measured",
                "encoder_pipe_bytes", "decoder_pipe_bytes",
                "decoder_gpu_to_cpu_bytes_estimated"):
        assert key in record, f"{key} must be present so its absence is visible"
        assert record[key] is None, f"{key} was not measured and must not claim {record[key]!r}"

    profiler = tp.TransferProfiler()
    profiler.record(record)
    summary = profiler.summary()
    for key in ("dlss_gpu_transfer_bytes_measured", "encoder_gpu_upload_bytes_measured",
                "nvof_h2d_bytes_measured"):
        assert summary[key]["basis"] in ("unavailable", "measured")
        assert summary[key]["frames_measured"] == 0
        assert summary[key]["all_frames"] is None


def test_pipe_fields_report_nothing_for_a_frame_nobody_counted():
    fields = tp.pipe_fields(4242, gpu_decoded=True)
    assert fields == {"encoder_pipe_bytes": None, "decoder_pipe_bytes": None,
                      "decoder_gpu_to_cpu_bytes_estimated": None}


def test_a_cpu_decode_leaves_the_gpu_download_field_empty_not_zero():
    ledger = tl.build_ledger(measured={"decoder_pipe_bytes": tl.PAYLOAD_REFERENCE["4k_bgr24"]},
                             decode_actual="ffmpeg_cpu")
    rows = {r["name"]: r for r in ledger["rows"]}
    download = rows["decoder_gpu_to_cpu_download"]
    assert download["measurement"] == "not_reached"
    assert download["transfer_bytes"] is None
    assert rows["decoder_rawvideo"]["format"] == "bgr24"


# ---------------------------------------------------------------------------
# the encoder and decoder pipes
# ---------------------------------------------------------------------------

class ByteCountingPipe:
    def __init__(self):
        self.written = 0

    def write(self, chunk):
        self.written += len(chunk)
        return len(chunk)


def test_encoder_bytes_are_what_the_write_call_returned():
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    pipe = ByteCountingPipe()
    written = pipeline._write_frame_bytes(pipe, frame, chunk_size=1 << 20)
    assert written == pipe.written == tl.PAYLOAD_REFERENCE["4k_bgr24"]

    tp.ENCODER_PIPE.note_frame(7, written)
    fields = tp.pipe_fields(7, gpu_decoded=False)
    assert fields["encoder_pipe_bytes"] == 24883200
    # The frame is gone from the ledger once read, so a second read cannot reuse it.
    assert tp.pipe_fields(7, gpu_decoded=False)["encoder_pipe_bytes"] is None


def test_decoder_bytes_are_attributed_to_the_frame_that_was_read():
    tp.DECODER_PIPE.note_frame(100, 12441600)
    tp.DECODER_PIPE.note_frame(101, 12441600)
    assert tp.pipe_fields(101, gpu_decoded=True)["decoder_gpu_to_cpu_bytes_estimated"] == 12441600
    assert tp.DECODER_PIPE.snapshot()["total_bytes"] >= 24883200


# ---------------------------------------------------------------------------
# the ledger itself
# ---------------------------------------------------------------------------

def test_payload_arithmetic_matches_the_specified_sizes():
    assert tl.PAYLOAD_REFERENCE == {
        "4k_bgr24": 24883200,
        "4k_nv12": 12441600,
        "working_bgr24": 6220800,
        "working_rgba8": 8294400,
        "working_flow_f32x2": 16588800,
        "working_depth_f32": 8294400,
    }


def test_ledger_covers_every_stage_in_frame_order():
    ledger = tl.build_ledger()
    stages_in_rows = [r["stage"] for r in ledger["rows"]]
    for stage in tl.STAGES:
        assert stage in stages_in_rows, f"{stage} vanished from the ledger"
    order = {s: i for i, s in enumerate(tl.STAGES)}
    mapped = [order[s] for s in stages_in_rows]
    assert mapped == sorted(mapped), "the ledger must stay in the order a frame touches them"


def test_ledger_rows_carry_the_audit_fields():
    required = {"producer", "consumer", "width", "height", "format", "bytes_per_pixel",
                "owner_process", "location", "allocation_owner", "transfer_direction",
                "transfer_bytes", "measurement", "lifetime", "allocation", "status"}
    for row in tl.build_ledger()["rows"]:
        missing = required - set(row)
        assert not missing, f"{row['name']} is missing {missing}"


def test_a_missing_measurement_leaves_the_row_unavailable_never_zero():
    ledger = tl.build_ledger(measured={"nvof_h2d_bytes_measured": None,
                                       "encoder_pipe_bytes": None})
    rows = {r["name"]: r for r in ledger["rows"]}
    assert rows["nvof_input_upload"]["measurement"] == "unavailable"
    assert rows["nvof_input_upload"]["transfer_bytes"] is None
    assert rows["encoder_rawvideo"]["transfer_bytes"] is None
    assert 0 not in [rows["nvof_input_upload"]["transfer_bytes"],
                     rows["encoder_rawvideo"]["transfer_bytes"]]


def test_measured_rows_pick_up_the_run_figures():
    measured = {"nvof_h2d_bytes_measured": 8294400, "nvof_d2h_bytes_measured": 16588800,
                "decoder_pipe_bytes": 12441600, "encoder_pipe_bytes": 24883200,
                "cpu_bgr_to_rgba@dlss_input_calls": 1,
                "cpu_bgr_to_rgba@dlss_input_output_bytes": 8294400,
                "cpu_bgr_to_rgba@nvof_input_calls": 1,
                "cpu_bgr_to_rgba@nvof_input_output_bytes": 8294400,
                "cpu_rgba_to_bgr@residual_input_calls": 1,
                "cpu_rgba_to_bgr@residual_input_output_bytes": 6220800,
                "cpu_bgr_to_rgba_calls": 2, "cpu_rgba_to_bgr_calls": 1,
                "dlss_input_color_bytes_logical": 8294400,
                "dlss_input_flow_bytes_logical": 16588800,
                "dlss_input_depth_bytes_logical": 8294400,
                "dlss_output_bytes_logical": 8294400}
    ledger = tl.build_ledger(measured=measured)
    rows = {r["name"]: r for r in ledger["rows"]}
    assert rows["nvof_input_upload"]["transfer_bytes"] == 8294400
    assert rows["nvof_flow_download"]["transfer_bytes"] == 16588800
    assert rows["decoder_rawvideo"]["transfer_bytes"] == 12441600
    assert rows["encoder_rawvideo"]["transfer_bytes"] == 24883200
    # Two RGBA conversions of the same frame, two rows, each with its own number. A ledger
    # that could only report "2 conversions" could not show which one a phase deleted.
    assert rows["working_rgba_for_dlss"]["cpu_call_count"] == 1
    assert rows["working_rgba_for_dlss"]["transfer_bytes"] == 8294400
    assert rows["working_rgba_for_nvof"]["cpu_call_count"] == 1
    assert rows["working_rgba_for_nvof"]["transfer_bytes"] == 8294400
    assert rows["dlss_output_bgr"]["cpu_call_count"] == 1
    # The row nobody can measure stays nobody's number.
    assert rows["dlss_internal_gpu_io"]["transfer_bytes"] is None
    assert rows["dlss_internal_gpu_io"]["measurement"] == "unavailable"


def test_a_deleted_conversion_site_shows_as_zero_not_as_a_missing_row():
    """What the reduced path must look like: the row is still there, and it says 0."""
    measured = {"cpu_bgr_to_rgba@dlss_input_calls": 0,
                "cpu_bgr_to_rgba@dlss_input_output_bytes": 0,
                "cpu_bgr_to_rgba@nvof_input_calls": 0,
                "cpu_bgr_to_rgba@nvof_input_output_bytes": 0,
                "cpu_rgba_to_bgr@residual_input_calls": 0,
                "cpu_rgba_to_bgr@residual_input_output_bytes": 0}
    ledger = tl.build_ledger(measured=measured, transfer_mode=tl.REDUCED)
    rows = {r["name"]: r for r in ledger["rows"]}
    for name in ("working_rgba_for_dlss", "working_rgba_for_nvof", "dlss_output_bgr"):
        assert rows[name]["measurement"] == "measured", name
        assert rows[name]["transfer_bytes"] == 0, name
        assert rows[name]["cpu_call_count"] == 0, name


def test_every_ledger_row_reads_a_field_the_run_publishes():
    """A row pointed at a field nobody writes would read as unavailable forever, and the
    ledger would look honest while reporting nothing. So the names are checked against the
    profiler's own field list."""
    published = set(tp.FIELDS)
    for row in tl.build_ledger()["rows"]:
        for key in ("measured_field", "measured_calls"):
            field = row[key]
            if field:
                assert field in published, f"{row['name']} reads {field}, which is not a field"


def test_a_tracked_site_without_a_ledger_row_is_visible():
    """Adding a conversion to the seam without adding a row would hide a round trip."""
    row_fields = {r["measured_calls"] for r in tl.build_ledger()["rows"]}
    for name, sites in tp.TRACKED_SITES.items():
        for site in sites:
            assert f"{name}@{site}_calls" in row_fields, f"{name}@{site} has no ledger row"


def test_markdown_renders_without_losing_a_row():
    ledger = tl.build_ledger(measured={"nvof_h2d_bytes_measured": 8294400})
    text = tl.render_markdown(ledger)
    for row in ledger["rows"]:
        assert f"`{row['name']}`" in text
    assert "not_yet_evaluated" in text
    assert "zero-copy" not in text.lower()


def test_transfer_profiler_stays_out_of_the_stage_profiler():
    """The two profilers answer different questions; one must not be a view of the other."""
    assert not issubclass(tp.TransferProfiler, pipeline.StageProfiler)
    assert not hasattr(pipeline.StageProfiler, "summary_basis")


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
