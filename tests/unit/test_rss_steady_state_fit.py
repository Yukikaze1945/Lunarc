"""Gate: the steady-state RSS fit has to measure growth, not the shape of a run.

A sustained export brings RSS up, holds it, and drops it again as the run finishes. Fitting
one line through all of that produced -307,312 bytes per frame on a real 300-frame run whose
interior samples were flat to within 2.4 MB, because the single sample at frame 300 sat
174 MB below the plateau. The stability gate read the magnitude of that number, so the
direction the slope pointed told you nothing about growth: a deeper teardown fails a flat
steady state (case A below), and the two committed runs passed only because their drops
happened to land under the threshold.

Every case below is synthetic and exact: the numbers going in are the ones being asserted
on, so a change to the window rule, the sample selection or the positive-growth semantics
shows up as a specific failure rather than a shifted summary. The last group re-runs the fit
over a committed historical artifact and proves it stays read-only.
"""

import glob
import hashlib
import json
import os
import sys

import pytest

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import resource_metrics as rm  # noqa: E402

TOTAL = 300


def _samples(pairs):
    """Minimal frame_samples records: the fit reads frame and rss_bytes, nothing else."""
    return [{"frame": f, "rss_bytes": r} for f, r in pairs]


def _plateau(step=30, end_frame=270):
    """The shape a sustained run has before it finishes: a ramp, then a flat interior.

    The warm-up sample is a single low point at frame 0 (model and encoder not resident
    yet), and the interior rises 1,000 bytes per frame - 210 KB across the whole window,
    against the 2.4 MB spread the committed 300-frame artifact actually shows.
    """
    out = [(0, 900_000_000)]
    out += [(f, 1_100_000_000 + max(0, f - 60) * 1_000)
            for f in range(step, end_frame + 1, step)]
    return out


def _case_a():
    """_plateau plus a teardown sample: the run empties 500 MB of buffers at frame 300."""
    return _samples(_plateau() + [(TOTAL, 600_000_000)])


# ---- the window rule ---------------------------------------------------------------------

def test_window_for_300_frames_is_60_to_270():
    assert rm.steady_state_fit_window(300) == (60, 270)


@pytest.mark.parametrize("total,expected", [
    (60, (12, 54)),
    (100, (20, 90)),
    (301, (61, 270)),       # ceil(60.2) and floor(270.9)
    (7, (2, 6)),
])
def test_window_uses_ceil_and_floor_of_the_fractions(total, expected):
    assert rm.steady_state_fit_window(total) == expected


def test_window_rule_fractions_are_the_documented_ones():
    assert rm.STEADY_FIT_START_FRACTION == 0.20
    assert rm.STEADY_FIT_END_FRACTION == 0.90
    assert rm.STEADY_GROWTH_THRESHOLD_FRACTION == 0.10, (
        "the threshold was deliberately left alone this round")
    assert rm.MIN_STEADY_FIT_SAMPLES == 5


# ---- case A: warmup, plateau, teardown ----------------------------------------------------

def test_case_a_teardown_excluded_and_slope_is_flat():
    samples = _case_a()
    fit = rm.fit_rss_steady_state(samples, TOTAL)
    assert fit["rss_fit_frames"] == [60, 90, 120, 150, 180, 210, 240, 270]
    assert TOTAL not in fit["rss_fit_frames"]
    assert fit["rss_fit_excluded_frames_before_window"] == [0, 30]
    assert fit["rss_fit_excluded_frames_after_window"] == [TOTAL]
    assert fit["rss_fit_status"] == "ok"
    # 1000 bytes per frame over a 210-frame span, against a peak of ~1.1 GB.
    assert abs(fit["rss_slope_bytes_per_frame"]) < 2_000
    gate = rm.rss_stability_gate(fit, max(r["rss_bytes"] for r in samples))
    assert gate["status"] == "DONE_PROVEN"
    assert gate["reason"] == "steady-state RSS shows no sustained positive growth"


def test_case_a_differs_markedly_from_the_whole_range_fit():
    """The point of the window: the same eleven numbers, fitted two ways, disagree in sign.

    Without this the new function could be a relabelling, and the old gate's behaviour -
    failing a run because its RSS came down - would still be reachable."""
    samples = _case_a()
    steady = rm.fit_rss_steady_state(samples, TOTAL)
    whole = rm.fit_rss_whole_range(samples)
    peak = max(s["rss_bytes"] for s in samples)
    assert whole["rss_slope_bytes_per_frame"] < -400_000, (
        "the whole-range fit is supposed to be dominated by the teardown")
    assert whole["rss_slope_bytes_per_frame"] * steady["rss_slope_bytes_per_frame"] < 0, (
        "the two fits disagree in direction, which is the defect being fixed")
    # The verdict the replaced gate would have given this run, computed the way it computed
    # it: the magnitude of the whole-range projection against 10% of peak.
    old_growth = abs(whole["rss_predicted_growth_over_range_bytes"])
    assert old_growth >= 0.10 * peak, (
        f"expected the old gate to fail this run, got {old_growth / peak:.3f} of peak")
    assert rm.rss_stability_gate(steady, peak)["status"] == "DONE_PROVEN", (
        "the run the old gate failed for finishing is the one the new gate passes")
    assert steady["rss_predicted_positive_growth_bytes"] / peak < 0.01


# ---- case B: real growth inside the window ------------------------------------------------

def test_case_b_true_growth_is_detected_and_gates_not_ready():
    growth_per_frame = 2_000_000  # 2 MB per frame: an unambiguous leak
    pairs = []
    for frame in range(0, TOTAL + 1, 30):
        rss = 1_000_000_000 + (frame * growth_per_frame)
        pairs.append((frame, rss))
    fit = rm.fit_rss_steady_state(_samples(pairs), TOTAL)
    assert fit["rss_fit_status"] == "ok"
    assert fit["rss_slope_bytes_per_frame"] == pytest.approx(growth_per_frame, rel=1e-6)
    assert fit["rss_predicted_positive_growth_bytes"] > 0
    assert fit["rss_predicted_signed_growth_bytes"] > 0
    gate = rm.rss_stability_gate(fit, max(r for _, r in pairs))
    assert gate["status"] == "NOT_READY"
    assert "positive growth" in gate["reason"]


def test_case_b_growth_under_the_threshold_still_passes():
    """The gate is 10% of peak, unchanged: a small positive slope is not a failure."""
    pairs = [(frame, 1_000_000_000 + frame * 1_000) for frame in range(0, TOTAL + 1, 30)]
    fit = rm.fit_rss_steady_state(_samples(pairs), TOTAL)
    assert fit["rss_predicted_positive_growth_bytes"] > 0
    assert rm.rss_stability_gate(fit, 1_100_000_000)["status"] == "DONE_PROVEN"


# ---- case C: a falling steady state -------------------------------------------------------

def test_case_c_negative_slope_is_not_a_leak_failure():
    # -1 MB per frame across the window: RSS visibly came down, and that is not growth.
    pairs = [(frame, 1_500_000_000 - frame * 1_000_000) for frame in range(60, 271, 30)]
    fit = rm.fit_rss_steady_state(_samples(pairs), TOTAL)
    peak = max(r for _, r in pairs)
    assert fit["rss_slope_bytes_per_frame"] < 0
    assert fit["rss_predicted_signed_growth_bytes"] < 0
    assert fit["rss_predicted_positive_growth_bytes"] == 0
    gate = rm.rss_stability_gate(fit, peak)
    assert gate["status"] == "DONE_PROVEN"
    # The old gate compared abs(growth) against 10% of peak: this same run, a 210 MB
    # signed swing on a 1.44 GB peak, would have been reported as growth.
    assert abs(fit["rss_predicted_signed_growth_bytes"]) > 0.10 * peak


def test_case_c_signed_information_survives():
    pairs = [(frame, 1_500_000_000 - frame * 1_000_000) for frame in range(60, 271, 30)]
    fit = rm.fit_rss_steady_state(_samples(pairs), TOTAL)
    gate = rm.rss_stability_gate(fit, 1_500_000_000)
    assert gate["rss_predicted_signed_growth_bytes"] < 0
    assert gate["rss_predicted_positive_growth_bytes"] == 0
    assert fit["rss_slope_bytes_per_frame"] == pytest.approx(
        fit["rss_predicted_signed_growth_bytes"] / fit["rss_fit_frame_span"], rel=1e-9)


# ---- case D: not enough samples to say anything -------------------------------------------

@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_case_d_insufficient_samples_cannot_be_done_proven(count):
    pairs = [(60 + i * 10, 1_000_000_000 + i * 40_000_000) for i in range(count)]
    fit = rm.fit_rss_steady_state(_samples(pairs), TOTAL)
    assert fit["rss_fit_sample_count"] == count
    assert fit["rss_fit_status"] == "insufficient_samples"
    assert fit["rss_slope_bytes_per_frame"] is None
    gate = rm.rss_stability_gate(fit, 1_200_000_000)
    assert gate["status"] == "NOT_READY"
    assert gate["reason"] == "insufficient steady-state RSS samples"


def test_five_samples_is_enough_to_be_measured():
    pairs = [(60, 1_000_000_000), (110, 1_001_000_000), (160, 1_002_000_000),
             (210, 1_003_000_000), (270, 1_004_000_000)]
    fit = rm.fit_rss_steady_state(_samples(pairs), TOTAL)
    assert fit["rss_fit_sample_count"] == 5
    assert fit["rss_fit_status"] == "ok"


def test_samples_outside_the_window_do_not_rescue_an_insufficient_run():
    """Nine samples, six of them warm-up: the fit may not quietly widen itself to pass."""
    pairs = [(frame, 900_000_000 + frame * 2_000_000) for frame in (0, 10, 20, 30, 40, 50)]
    pairs += [(frame, 1_000_000_000) for frame in (120, 150, 180)]
    fit = rm.fit_rss_steady_state(_samples(pairs), TOTAL)
    assert fit["rss_fit_frames"] == [120, 150, 180]
    assert fit["rss_fit_status"] == "insufficient_samples"


def test_repeated_samples_of_one_frame_cannot_produce_a_slope():
    """Duplicate frame numbers collapse to a single point, and one point is no trend."""
    pairs = [(100, 1_000_000_000 + i) for i in range(6)]
    fit = rm.fit_rss_steady_state(_samples(pairs), TOTAL)
    assert fit["rss_fit_frames"] == [100]
    assert fit["rss_fit_sample_count"] == 1
    assert fit["rss_fit_status"] == "insufficient_samples"
    assert "60..270" in fit["rss_fit_reason"]


# ---- case E: irregular frame indices ------------------------------------------------------

IRREGULAR = [0, 31, 62, 93, 127, 181, 229, 268, 300]


def _irregular_samples():
    return _samples([(frame, 1_000_000_000 + frame * 1_500) for frame in IRREGULAR])


def test_case_e_selects_by_frame_number_not_list_position():
    samples = _irregular_samples()
    fit = rm.fit_rss_steady_state(samples, TOTAL)
    assert fit["rss_fit_frames"] == [62, 93, 127, 181, 229, 268]
    assert 0 not in fit["rss_fit_frames"] and 31 not in fit["rss_fit_frames"]
    assert 300 not in fit["rss_fit_frames"]
    assert fit["rss_fit_excluded_frames_before_window"] == [0, 31]
    assert fit["rss_fit_excluded_frames_after_window"] == [300]
    # What a percentage slice of the same list would have kept instead: frame 31 belongs to
    # warm-up by number, and it enters the fit only because of where it sits in the list.
    # Which samples a slice drops depends on how many there were, not on the run's shape.
    by_position = samples[int(0.20 * len(samples)):int(0.90 * len(samples))]
    kept = [s["frame"] for s in by_position]
    assert kept == [31, 62, 93, 127, 181, 229, 268]
    assert kept != fit["rss_fit_frames"]
    assert 31 in kept and 31 not in fit["rss_fit_frames"]


def test_case_e_slope_is_invariant_to_how_the_frames_were_sampled():
    """Same underlying line, sampled at 30-frame intervals and at odd ones: same slope."""
    regular = _samples([(frame, 1_000_000_000 + frame * 1_500)
                        for frame in range(60, 271, 30)])
    irregular = _irregular_samples()
    a = rm.fit_rss_steady_state(regular, TOTAL)
    b = rm.fit_rss_steady_state(irregular, TOTAL)
    assert a["rss_slope_bytes_per_frame"] == pytest.approx(1_500, rel=1e-9)
    assert b["rss_slope_bytes_per_frame"] == pytest.approx(1_500, rel=1e-6)


def test_unsorted_input_is_fitted_against_the_frame_axis_not_arrival_order():
    samples = _samples(list(reversed([(f, 1_000_000_000 + f * 1_500)
                                      for f in range(60, 271, 30)])))
    fit = rm.fit_rss_steady_state(samples, TOTAL)
    assert fit["rss_fit_frames"] == sorted(fit["rss_fit_frames"])
    assert fit["rss_slope_bytes_per_frame"] == pytest.approx(1_500, rel=1e-9)


# ---- metadata the artifact has to carry ---------------------------------------------------

def test_fit_metadata_names_every_frame_it_used():
    samples = _case_a()
    fit = rm.fit_rss_steady_state(samples, TOTAL)
    required = {"rss_fit_method", "rss_fit_frame_start", "rss_fit_frame_end",
                "rss_fit_sample_count", "rss_fit_frames", "rss_slope_bytes_per_frame",
                "rss_predicted_growth_over_fit_window_bytes",
                "rss_predicted_signed_growth_bytes",
                "rss_predicted_positive_growth_bytes",
                "rss_fit_excludes_warmup", "rss_fit_excludes_teardown"}
    assert required <= set(fit)
    assert fit["rss_fit_method"] == "least_squares_over_steady_state_frame_window"
    assert fit["rss_fit_excludes_warmup"] is True
    assert fit["rss_fit_excludes_teardown"] is True
    assert fit["rss_fit_sample_count"] == len(fit["rss_fit_frames"])
    assert fit["rss_predicted_growth_over_fit_window_bytes"] == (
        fit["rss_predicted_signed_growth_bytes"])
    # Every sample the run took is accounted for: fitted, or named as excluded from one end.
    accounted = (set(fit["rss_fit_frames"])
                 | set(fit["rss_fit_excluded_frames_before_window"])
                 | set(fit["rss_fit_excluded_frames_after_window"]))
    assert accounted == {s["frame"] for s in samples}


def test_a_run_without_a_frame_count_in_the_report_still_bounds_the_window():
    """The collector falls back to the highest sample number; the fit must not care."""
    samples = _case_a()
    fit = rm.fit_rss_steady_state(samples, max(s["frame"] for s in samples))
    assert (fit["rss_fit_frame_start"], fit["rss_fit_frame_end"]) == (60, 270)


def test_gate_without_a_peak_cannot_claim_proven():
    fit = rm.fit_rss_steady_state(_samples(_plateau()), TOTAL)
    assert fit["rss_fit_status"] == "ok"
    gate = rm.rss_stability_gate(fit, 0)
    assert gate["status"] == "NOT_READY"


# ---- the committed 300-frame artifact, recalculated read-only -----------------------------

HISTORICAL = [p for p in glob.glob(os.path.join(_ROOT, "perf_results", "*",
                                                "resource_samples.json"))]


@pytest.mark.parametrize("path", HISTORICAL or [pytest.param("", marks=pytest.mark.skip(
    reason="no historical resource_samples.json in perf_results"))], ids=os.path.basename)
def test_historical_artifact_excludes_its_last_frame(path):
    if not path:
        pytest.skip("no historical resource_samples.json in perf_results")
    before = _tree_digest(os.path.dirname(path))
    with open(path, encoding="utf-8") as handle:
        groups = json.load(handle)
    assert groups, path

    for scenario, group in groups.items():
        samples = group.get("frame_samples") or []
        frames = sorted(s["frame"] for s in samples)
        last = frames[-1]
        total = _historical_total_frames(os.path.dirname(path), scenario) or last
        fit = rm.fit_rss_steady_state(samples, total)
        assert last in frames, "the artifact is supposed to end at the teardown sample"
        assert last not in fit["rss_fit_frames"], (
            f"{scenario}: frame {last} entered the fit window")
        assert fit["rss_fit_excluded_frames_after_window"], (
            f"{scenario}: nothing was excluded from the end of the run")
        assert fit["rss_fit_frames"] == [f for f in frames
                                         if fit["rss_fit_frame_start"] <= f
                                         <= fit["rss_fit_frame_end"]]
        whole = rm.fit_rss_whole_range(samples)
        # Every committed run of this shape ended below its own plateau, so the whole-range
        # slope is negative while the steady state is flat to slightly positive.
        assert (whole["rss_slope_bytes_per_frame"] or 0) < 0, f"{scenario}: {whole}"
        assert fit["rss_predicted_positive_growth_bytes"] == 0 or (
            fit["rss_predicted_positive_growth_bytes"] < 0.10 * max(
                s["rss_bytes"] for s in samples)), f"{scenario}: {fit}"

    assert _tree_digest(os.path.dirname(path)) == before, (
        "a historical evidence directory was modified")


def test_historical_artifact_recalculation_lands_in_a_separate_directory(tmp_path):
    """Recomputing old numbers writes them somewhere new; the old directory is input."""
    path = HISTORICAL[0] if HISTORICAL else ""
    if not path:
        pytest.skip("no historical resource_samples.json in perf_results")
    source_dir = os.path.dirname(path)
    before = _tree_digest(source_dir)
    with open(path, encoding="utf-8") as handle:
        groups = json.load(handle)
    out = {}
    for scenario, group in groups.items():
        samples = group.get("frame_samples") or []
        total = _historical_total_frames(source_dir, scenario) or max(
            s["frame"] for s in samples)
        fit = rm.fit_rss_steady_state(samples, total)
        out[scenario] = {
            "source_artifact_path": os.path.relpath(path, _ROOT).replace("\\", "/"),
            "source_total_frames": total,
            "all_sample_frames": sorted(s["frame"] for s in samples),
            "rss_fit_frame_start": fit["rss_fit_frame_start"],
            "rss_fit_frame_end": fit["rss_fit_frame_end"],
            "rss_fit_frames": fit["rss_fit_frames"],
            "old_whole_range_slope": rm.fit_rss_whole_range(
                samples)["rss_slope_bytes_per_frame"],
            "new_steady_state_slope": fit["rss_slope_bytes_per_frame"],
            "frame_300_excluded": 300 not in fit["rss_fit_frames"],
        }
    target = tmp_path / "historical_rss_recalc.json"
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    assert target.exists()
    assert _tree_digest(source_dir) == before


def _historical_total_frames(directory, scenario):
    report = os.path.join(directory, "benchmark_sustained.json")
    if not os.path.exists(report):
        return None
    with open(report, encoding="utf-8") as handle:
        data = json.load(handle)
    rows = (data or {}).get("results") or []
    for row in rows:
        scenarios = [row] + list(row.get("scenarios") or [])
        for entry in scenarios:
            if entry.get("id") == scenario and entry.get("frames"):
                return int(entry["frames"])
    return None


def _tree_digest(directory):
    """Hash every file in a directory: used to prove the evidence was only read."""
    digest = hashlib.sha256()
    for root, _dirs, files in os.walk(directory):
        for name in sorted(files):
            full = os.path.join(root, name)
            digest.update(name.encode("utf-8"))
            with open(full, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()
