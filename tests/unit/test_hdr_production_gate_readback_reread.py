# -*- coding: utf-8 -*-
"""The offline inspector's read-back of the two readback boundaries, with no GPU.

``tests/unit/hdr_production_export_gate.py`` drives the HDR export once and otherwise
re-reads the report that run left on disk. Only that second half is covered here: the
lines the inspector draws about host traffic, against every shape a saved report can
have. Nothing in this file starts a device, loads a bridge or drives an export - the
gate module's imports are the same ones the CPU flow tests already use, and the checks
are called on dictionaries.

The distinction under test is the one HDR_STAGE_B_BACKEND_AUDIT_REPAIR.md records: the
bridge's ``report_readback_bytes`` counts image planes copied out by the diagnostic
readback entries, while the per-pass counter readback is scalar traffic that crosses on
every dispatch and that no counter measures. A pre-repair saved report filed the first
number in the second boundary's slot, so re-reading one must relabel it rather than
repeat the claim.
"""
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

GATE = os.path.join(_ROOT, "tests", "unit", "hdr_production_export_gate.py")


def _gate():
    spec = importlib.util.spec_from_file_location("hdr_production_export_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_current_report_files_the_two_boundaries_apart():
    gate = _gate()
    report = {"hdr_diagnostic_image_readback_bytes_measured": 0,
              "hdr_scalar_report_readback_bytes_measured": None}
    image = gate.image_readback_counter(report)
    assert image["key"] == gate.IMAGE_READBACK_KEY
    assert image["value"] == 0
    assert "OFFLINE" not in image["provenance"]
    scalar = gate.scalar_readback_claim(report)
    assert scalar["state"] == "unanswered"
    assert scalar["value"] is None


def test_a_pre_repair_report_is_re_read_under_the_corrected_meaning():
    """The shape of the report the accepted six-frame export left on disk.

    Its one readback field held the image-plane counter under a scalar name. Reading it
    now answers the image boundary truthfully and calls the scalar boundary
    present-and-unmeasured, and every line says it re-read rather than measured.
    """
    gate = _gate()
    report = {"hdr_scalar_report_readback_bytes_measured": 0,
              "hdr_pixel_readback_bytes_measured": None}
    image = gate.image_readback_counter(report)
    assert image["key"] == gate.HISTORICAL_IMAGE_READBACK_KEY
    assert image["value"] == 0
    assert "OFFLINE RE-READ" in image["provenance"]
    assert "no new measurement" in image["provenance"]
    scalar = gate.scalar_readback_claim(report)
    assert scalar["state"] == "misfiled"
    assert "OFFLINE RE-READ" in scalar["detail"]


def test_the_same_counter_booked_as_a_measured_scalar_is_the_failing_shape():
    gate = _gate()
    report = {"hdr_diagnostic_image_readback_bytes_measured": 0,
              "hdr_scalar_report_readback_bytes_measured": 0}
    assert gate.scalar_readback_claim(report)["state"] == "doubled"


def test_a_report_that_answers_neither_name_reads_as_missing_not_as_zero():
    gate = _gate()
    image = gate.image_readback_counter({})
    assert image["value"] is None
    assert "neither name" in image["provenance"]
    assert gate.scalar_readback_claim({})["state"] == "absent"


def test_the_inspector_no_longer_demands_a_zero_from_the_scalar_boundary():
    """Guards the removal, not the wording: the false line must not come back.

    The check that used to read "report structs cost no readback, and were read anyway"
    asked a positive number of nothing and asserted that a report struct never leaves
    the device, which the bridge source contradicts on every dispatch.
    """
    with open(GATE, encoding="utf-8") as handle:
        source = handle.read()
    assert "cost no readback" not in source
    assert "scalar counter readback is present and not reported as a measured zero" \
        in source
    assert "diagnostic image-plane readback is the bridge's own zero" in source
