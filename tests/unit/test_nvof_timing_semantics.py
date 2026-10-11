"""Gate: one number per optical-flow stage, and no fabricated zeros.

The profiler used to publish a dict for NVOF in which several keys carried the same
measurement and two keys that nothing measures carried 0.0. A p50 averaged over that dict
therefore described a stage that was never timed, and a reader could not tell an upload from
a download because they were the same number. The rule now is that every reported key comes
from exactly one QueryPerformanceCounter span in the bridge, and a span the bridge cannot
time by itself is reported as the string "unavailable".

nvof_timing_semantics.collect_rows() runs real executes on the real engine and reads both
the reported dict and the raw native struct, so the comparison is between the number the
pipeline publishes and the span the driver produced. hardware_present() is the only thing
that can excuse these tests from running: if the GPU is there and the bridge fails, that is
a failure, not a skip.

Some spans belong to one entry point only. The rule still holds there, in both directions:
a span a path does not produce has to read as unavailable on that path, and a path that
produces it has to report the bridge's own number for it rather than a neighbour's.
"""

import pytest

import nvof_timing_semantics as timing

PRESENT, PROBE = timing.hardware_present()
pytestmark = pytest.mark.skipif(
    not PRESENT,
    reason=f"no NVIDIA optical flow hardware to time (probe said {PROBE})")


@pytest.fixture(scope="module")
def rows():
    """Real executes, each paired with the raw struct the bridge filled in."""
    return timing.collect_rows(frames=4, sequential_frames=2)


@pytest.fixture(scope="module")
def analysis(rows):
    return timing.analyze(rows)


def test_the_gate_actually_ran_on_hardware(rows):
    """A skip that looks like a pass would be the worst outcome here."""
    assert rows, "collect_rows returned nothing while the probe said hardware is present"
    assert all(row["provider"]["hardware"] is True for row in rows), (
        "a software provider was timed while the probe reported hardware; the numbers "
        "would not be NVOF's")
    assert all(row["reported"]["nvof_total_native_ms"] != "unavailable" for row in rows), (
        "the native side never filled its timing struct, so every stage comparison below "
        "would be comparing placeholders")


def test_every_reported_key_equals_its_own_native_span(analysis):
    assert analysis["pass"] is True, "problems: " + "; ".join(analysis["problems"])
    assert not analysis["problems"], analysis["problems"]


def test_no_two_stages_share_a_number(analysis):
    """The aliasing this round removed: one measurement written into several keys."""
    assert analysis["frames_timed"] > 0
    assert analysis["frames_with_six_distinct_spans"] == analysis["frames_timed"], (
        f"{analysis['frames_timed'] - analysis['frames_with_six_distinct_spans']} of "
        f"{analysis['frames_timed']} frames published fewer distinct numbers than the six "
        f"stages they claim to time, which means two stages shared one measurement")


def test_a_span_that_cannot_be_timed_is_never_zero(rows):
    """The two spans nothing measures say so, and the six the hardware does measure cannot
    honestly come out at zero for a 1080p execute.

    Scoped to those keys on purpose: a Python-side span like prev_bgr_to_rgba_ms really is
    zero when the caller handed over already-converted pixels, and writing the rule as
    "no key may be 0.0" would have been its own kind of placeholder.
    """
    for row in rows:
        reported = row["reported"]
        for key in timing.ALWAYS_UNMEASURED:
            assert reported.get(key) == timing.of.NVOF_TIMING_UNAVAILABLE, (
                f"{key} must say it was not measured; it said {reported.get(key)!r}")
        for key in timing.FIELD_TO_KEY.values():
            value = reported.get(key)
            if isinstance(value, float):
                assert value > 0.0, (
                    f"{key} reported exactly 0.0 for a span the bridge says it timed; "
                    f"that is how a fabricated placeholder used to look")


def test_the_retired_alias_keys_do_not_come_back(analysis):
    assert analysis["retired_keys_still_absent"] is True, (
        "a key the old profiler used to alias or fabricate reappeared; it would be read as "
        "a measured stage by anything comparing history across rounds")


def test_each_stage_stays_inside_the_native_total(rows):
    """A part larger than the whole means one of the two names is on the wrong span."""
    whole_key = timing.FIELD_TO_KEY["total_native_ms"]
    for row in rows:
        reported = row["reported"]
        whole = reported.get(whole_key)
        assert isinstance(whole, float), f"{whole_key} was not a measured number: {whole!r}"
        for key in timing.FIELD_TO_KEY.values():
            if key == whole_key or not isinstance(reported.get(key), float):
                continue
            assert reported[key] <= whole + 0.05, (
                f"frame {row['frame']}: {key} = {reported[key]} ms does not fit inside "
                f"{whole_key} = {whole} ms")


def test_the_python_frame_time_covers_the_native_work(rows):
    """Python's span has to be at least the native span it wraps, or a clock is wrong."""
    for row in rows:
        whole = row["reported"]["python_total_ms"]
        part = row["reported"]["nvof_total_native_ms"]
        assert part == "unavailable" or whole + 1.0 >= part, (
            f"python_total_ms {whole} does not contain nvof_total_native_ms {part}")


def test_the_sequential_path_keeps_its_own_semantics(rows):
    """compute_flow_sequential has no second frame to wait for, and must not claim one."""
    sequential = [row for row in rows if row["path"] == "sequential"]
    assert sequential, "the sequential path was never exercised"
    for row in sequential:
        assert row["reported"]["nvof_dependency_wait_ms"] == "unavailable", (
            "the sequential path reported a dependency wait it does not perform")


def test_the_bgr_input_path_times_its_own_convert_span(rows, analysis):
    """The one span this round added, held to the same rule as the six that came before.

    A row is evidence only if the bridge itself filled the field: the reported key has to
    equal raw input_convert_wait_ms, which analyze() checks, and it has to be a number the
    BGR execute produced rather than a leftover from a previous call.
    """
    key = timing.PATH_SPECIFIC_FIELDS["input_convert_wait_ms"][0]
    bgr = [row for row in rows if row["path"] == "bgr"]
    assert bgr, (
        f"no BGR-input execute was collected while the probe said {key} exists; the shipped "
        f"nvof_bridge.dll would be one that has no BGR entry point")
    assert analysis["frames_bgr"] == len(bgr)
    for row in bgr:
        value = row["reported"].get(key)
        assert isinstance(value, float), (
            f"frame {row['frame']}: the BGR path reported {value!r} for the span that times "
            f"its own byte-map kernel")
        assert value > 0.0, (
            f"frame {row['frame']}: {key} measured {value} ms for a kernel and a synchronize "
            f"that the bridge says it ran")
        assert value <= row["reported"]["nvof_total_native_ms"] + 0.05, (
            f"frame {row['frame']}: {key} = {value} ms does not fit inside the native execute "
            f"it is a stage of")


def test_the_rgba_paths_do_not_report_a_convert_span(rows):
    """Nothing on those entry points maps bytes, so a number there is another stage's."""
    key = timing.PATH_SPECIFIC_FIELDS["input_convert_wait_ms"][0]
    for row in rows:
        if row["path"] == "bgr":
            continue
        assert row["reported"].get(key) == "unavailable", (
            f"the {row['path']} path reported {row['reported'].get(key)!r} for {key}, which "
            f"only a BGR-input execute performs")
