"""Stage B of the HDR10 chain: the two settings keys the UI can write.

The gate itself is covered by test_hdr_transfer_export_branch.py. What is not covered
there is the path a click takes before the gate: ``POST /api/settings`` merges a partial
patch, ``add_job`` snapshots it, and the queue re-reads the snapshot. A refusal that
landed after the merge would leave an unusable paper white stored and reported as the
value the next job would open its colour bridge with, and a snapshot that dropped a key
would send an HDR job down the SDR chain it just opted out of.

Every assertion here reads the answer through the chain's own predicates
(``hdr_requested`` / ``hdr_flag_reason`` / ``paper_white_reason``) rather than through a
copy of its vocabulary, so a mirror that drifts from the backend fails instead of
agreeing with itself. No GPU, no FFmpeg, no display: nothing in this file reaches a
device.
"""
import ast
import pathlib
import re

import pytest

import gpu_hdr_transfer_export as htx
import gpu_transfer_export as gtx
from dlss_service import DLSSService
from server import update_settings

ROOT = pathlib.Path(__file__).resolve().parents[2]
HDR_TS = ROOT / "frontend" / "src" / "services" / "hdrExport.ts"
SCALE_TS = ROOT / "frontend" / "src" / "services" / "workingScale.ts"


def _service():
    return DLSSService()


def _ts_array(source, name):
    """The literal array written for ``export const NAME``, as Python sees it."""
    match = re.search(rf"export const {name}(?::[^=]*)?\s*=\s*(\[.*?\])",
                      source, re.DOTALL)
    assert match, f"{name} is no longer a literal array the test can read"
    return ast.literal_eval(match.group(1))


def _ts_value(source, name):
    match = re.search(rf"export const {name}(?::[^=]*)?\s*=\s*([^;\n]+)", source)
    assert match, f"{name} is no longer a literal the test can read"
    return ast.literal_eval(match.group(1).strip())


# --- the opt-in is a setting the service can be asked to remember ------------------

def test_the_service_ships_the_hdr_chain_off_with_the_backends_own_default():
    service = _service()

    assert service.settings[htx.HDR_SETTING] is False
    assert service.settings[htx.HDR_PAPER_WHITE_SETTING] == htx.HDR_PAPER_WHITE_SCRGB
    # Read through the chain's predicate, not a copy of its vocabulary.
    assert htx.hdr_requested(service.settings) is False


def test_an_opt_in_reaches_the_job_snapshot_the_queue_actually_reads():
    service = _service()

    saved = service.update_settings({htx.HDR_SETTING: True, "export_backend": "gpu_transfer",
                                     htx.HDR_PAPER_WHITE_SETTING: 20.0})
    job = service.add_job({"settings": saved})
    # Later UI moves must not rewrite a job already in the queue.
    service.update_settings({htx.HDR_SETTING: False, htx.HDR_PAPER_WHITE_SETTING: 8.0})

    assert htx.hdr_requested(job["settings"]) is True, "the snapshot dropped the opt-in"
    assert job["settings"][htx.HDR_PAPER_WHITE_SETTING] == 20.0, \
        "the snapshot re-read the live paper white instead of the job's own"
    assert job["settings"]["export_backend"] == "gpu_transfer"


def test_a_per_job_override_beats_the_live_settings_in_both_directions():
    service = _service()
    service.update_settings({htx.HDR_SETTING: True, "export_backend": "gpu_transfer"})

    off = service.add_job({"settings": {**service.settings, htx.HDR_SETTING: False}})

    assert htx.hdr_requested(off["settings"]) is False
    assert htx.hdr_requested(service.settings) is True


# --- a refusal lands before anything is stored ------------------------------------

def test_an_unusable_paper_white_is_refused_and_never_merged():
    service = _service()

    with pytest.raises(ValueError, match="hdr_paper_white_scrgb") as refused:
        service.update_settings({htx.HDR_PAPER_WHITE_SETTING: ""})

    assert htx.paper_white_reason(service.settings) is None
    assert service.settings[htx.HDR_PAPER_WHITE_SETTING] == htx.HDR_PAPER_WHITE_SCRGB
    # The reason an operator sees is the backend's own words, not a paraphrase.
    assert str(refused.value) == htx.paper_white_reason(
        {htx.HDR_PAPER_WHITE_SETTING: ""})


def test_a_patch_with_one_good_key_and_one_bad_key_stores_neither():
    """/api/settings is a partial patch, so the pair can arrive half-valid."""
    service = _service()

    with pytest.raises(ValueError):
        service.update_settings({htx.HDR_SETTING: True,
                                 htx.HDR_PAPER_WHITE_SETTING: "banana"})

    assert htx.hdr_requested(service.settings) is False, "the opt-in was merged anyway"
    assert service.settings[htx.HDR_PAPER_WHITE_SETTING] == htx.HDR_PAPER_WHITE_SCRGB


def test_a_flag_that_is_neither_on_nor_off_is_refused_by_name():
    service = _service()

    with pytest.raises(ValueError, match="experimental_hdr_scrgb"):
        service.update_settings({htx.HDR_SETTING: "maybe"})

    assert htx.hdr_requested(service.settings) is False
    assert service.settings[htx.HDR_SETTING] is False


def test_the_service_only_checks_the_pair_a_patch_can_reach_it_with():
    """A job that never mentions HDR is not policed by the HDR chain."""
    service = _service()
    service.update_settings({htx.HDR_SETTING: True, "export_backend": "gpu_transfer"})

    # A patch that touches neither key (a preset change, a guidance change) must not be
    # refused because of a paper white this write says nothing about.
    merged = service.update_settings({"guidance_mode": 1})

    assert htx.hdr_requested(merged) is True
    assert htx.paper_white_reason(merged) is None


def test_the_http_boundary_answers_a_refused_hdr_write_with_the_reason():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        update_settings({htx.HDR_PAPER_WHITE_SETTING: 0.5})

    assert raised.value.status_code == 400
    assert htx.HDR_PAPER_WHITE_SETTING in str(raised.value.detail)


# --- the queue boundary refuses what the page could send -------------------------

def test_an_hdr_job_on_the_sdr_chain_is_refused_at_the_boundary_not_re_routed():
    service = _service()

    with pytest.raises(ValueError) as refused:
        service.add_job({
            "input_path": str(__file__),
            "output_path": str(ROOT / "build" / "unused_hdr_legacy.mkv"),
            "settings": {**service.settings, htx.HDR_SETTING: True,
                         "export_backend": "legacy"},
        })

    message = str(refused.value)
    assert "gpu_transfer" in message
    assert htx.HDR_SETTING in message, "the refusal names the switch to turn off"
    assert not service.get_jobs(), "a refused job must not sit in the queue"


@pytest.mark.parametrize(("mode", "scales", "weights"), [
    ("dual", [0.5, 0.75], [1.0, 0.8]),
    ("triple", [0.4, 0.67, 0.85], [1.0, 0.8, 1.2]),
])
def test_hdr_multilayer_job_is_queued_with_its_requested_layer_vector(mode, scales, weights):
    service = _service()
    job = service.add_job({
        "input_path": str(__file__),
        "output_path": str(ROOT / "build" / f"unused_hdr_{mode}.mkv"),
        "settings": {**service.settings, htx.HDR_SETTING: True,
                     "export_backend": "gpu_transfer", "reconstruction_mode": mode,
                     "working_scale": scales[0], "layer_scales": scales,
                     "layer_weights": weights, "fusion_mode": "data_consistency_experimental"},
    })

    assert job["settings"]["reconstruction_mode"] == mode
    assert job["settings"]["layer_scales"] == scales
    assert job["settings"]["layer_weights"] == weights
    assert job["settings"]["fusion_mode"] == "data_consistency_experimental"


def test_the_switch_alone_is_enough_for_a_job_to_be_an_hdr_job():
    """Opting in is one checkbox; the chain it needs is already the service default.

    The panel writes export_backend=gpu_transfer as part of the same patch, but a job
    that arrives with only the flag set and the default chain must still be an HDR job -
    otherwise the checkbox would be decoration.
    """
    service = _service()
    service.update_settings({"export_backend": "gpu_transfer"})

    job = service.add_job({"input_path": str(__file__),
                           "output_path": str(ROOT / "build" / "unused_hdr_ok.mkv"),
                           "settings": {**service.settings, htx.HDR_SETTING: True}})

    assert htx.hdr_requested(job["settings"]) is True
    assert job["settings"]["export_backend"] == "gpu_transfer"


# --- the client mirror must not drift from the gate it mirrors --------------------

def test_the_front_end_mirror_carries_the_backends_vocabulary():
    source = HDR_TS.read_text(encoding="utf-8")

    assert _ts_array(source, "HDR_ALLOWED_PROFILES") == list(
        ("auto", "", "None", "none", htx.HDR_PROFILE, "Main 10", "main 10", "Main10"))
    assert _ts_array(source, "HDR_ALLOWED_PIXEL_FORMATS") == list(htx.HDR_ALLOWED_PIXEL_FORMATS)
    assert _ts_array(source, "HDR_ALLOWED_COLOR_PRESETS") == list(htx.HDR_ALLOWED_COLOR_PRESETS)
    assert _ts_array(source, "HDR_HLG_TRANSFERS") == list(htx.HLG_TRANSFERS)
    assert _ts_value(source, "HDR_SOURCE_PIX_FMT") in htx.HDR_SOURCE_PIX_FMTS
    assert _ts_value(source, "HDR_SOURCE_TRANSFER") in htx.HDR_SOURCE_TRANSFER
    assert _ts_value(source, "HDR_SETTING") == htx.HDR_SETTING
    assert _ts_value(source, "HDR_PAPER_WHITE_SETTING") == htx.HDR_PAPER_WHITE_SETTING
    assert _ts_value(source, "HDR_PAPER_WHITE_SCRGB") == htx.HDR_PAPER_WHITE_SCRGB
    assert _ts_value(source, "NITS_PER_SCRGB") == htx.NITS_PER_SCRGB
    assert _ts_value(source, "HDR_SUPPORT_CEILING_SCRGB") == htx.HDR_SUPPORT_CEILING_SCRGB
    assert _ts_value(source, "HDR_CLIP_CEILING_SCRGB") == htx.HDR_CLIP_CEILING_SCRGB
    assert _ts_value(source, "HDR_DERIVED_LEGAL_PEAK_SCRGB") == htx.DERIVED_LEGAL_PEAK_SCRGB
    assert _ts_value(source, "HDR_CODEC") == htx.HDR_CODEC
    assert _ts_value(source, "HDR_PROFILE") == htx.HDR_PROFILE
    assert _ts_value(source, "HDR_CONTAINER") == htx.HDR_CONTAINER


def test_the_mirror_carries_the_reused_copy_pass_vocabulary():
    """The rewrap is ``gtx``'s own helper, so its vocabularies are the helper's.

    A mode the mirror refuses that the shared pass serves would show the operator a
    refusal that never lands; a clause the mirror lets through while the gate names it
    would queue a job that re-encodes the bitstream this chain exists to preserve.
    """
    source = HDR_TS.read_text(encoding="utf-8")

    assert _ts_array(source, "HDR_MUX_AUDIO_MODES") == list(gtx.MUX_AUDIO_MODES)
    assert _ts_array(source, "HDR_MUX_SUBTITLE_MODES") == list(gtx.MUX_SUBTITLE_MODES)

    # And the list is the helper's because the helper is the only thing that reads it:
    # every mode the mirror names has a branch in build_mux_command.
    helper = pathlib.Path(gtx.__file__).read_text(encoding="utf-8")
    body = helper[helper.index("def build_mux_command"):helper.index("def run_copy_pass")]
    for clause in ['-c:a", "copy', '"aac"', '"libopus"', '"flac"', '"pcm_s24le"',
                   '-an', '-c:s", "copy', '-sn', '"srt"']:
        assert clause in body, f"the shared pass no longer expresses {clause}"


def test_the_mirror_refuses_extra_args_the_way_the_gate_does_and_for_the_same_reason():
    """One question on both sides - is the field non-empty - and no second parser.

    The mirror used to carry its own option-name denylist and its own shell tokenizer, and
    that was the defect: ``-bsf:v filter_units=remove_types=39|40`` named nothing on the
    list, so the client called a ready job while the backend would have shipped a
    deliverable with its mastering-display and content-light SEIs stripped. A refusal that
    reads the field's emptiness cannot drift from the gate's, because there is nothing left
    in it to drift.
    """
    source = HDR_TS.read_text(encoding="utf-8")
    gate = pathlib.Path(htx.__file__).read_text(encoding="utf-8")

    for abandoned in ("HDR_MUX_PROTECTED_OPTIONS", "shellTokens",
                      "claimsProtectedOption"):
        assert abandoned not in source, f"{abandoned} is a second parser in the page"

    refusal = source[source.index("export function hdrExtraArgsReason"):
                     source.index("export interface HdrFormInput")]
    # The one escape from the refusal is an empty field, and nothing analyses tokens.
    assert refusal.count("return null") == 1
    assert "if (!extra) return null" in refusal
    assert "tokens" not in refusal and ".filter(" not in refusal

    gate = gate[gate.index("def hdr_mux_refusals"):gate.index("def wants_video_descriptive")]
    assert "if named:" in gate and "shlex" not in gate and "split(" not in gate


def test_the_mirror_accepts_exactly_the_rate_and_aq_modes_the_encoder_plan_reads():
    """plan_hdr_encoder's if/elif ladder is the vocabulary; the mirror lists it.

    A mode the ladder refuses lands in ``unsupported``; a mode the mirror refuses that
    the ladder accepts would show the operator a refusal that never happens.
    """
    source = HDR_TS.read_text(encoding="utf-8")
    plan = pathlib.Path(htx.__file__).read_text(encoding="utf-8")
    body = plan[plan.index("def plan_hdr_encoder"):plan.index("# --- the gate")]

    rate_modes = _ts_array(source, "HDR_RATE_MODES")
    aq_modes = _ts_array(source, "HDR_AQ_MODES")
    for mode in rate_modes:
        assert f'rate_mode == "{mode}"' in body or f'rate_mode in ("{mode}"' in body, (
            f"the mirror offers rate_mode={mode}, which the encoder plan does not handle")
    for mode in aq_modes:
        assert f'"{mode}"' in body, (
            f"the mirror offers aq_mode={mode}, which the encoder plan does not handle")
    # The ladders' else branches are the refusals; nothing outside these lists is
    # accepted by either side.
    assert 'rate_mode = str(ts.get("rate_mode", "cq"))' in body
    assert 'aq_mode = str(ts.get("aq_mode", "spatial"))' in body


def test_the_mirror_uses_the_working_scale_range_the_gate_serves():
    source = HDR_TS.read_text(encoding="utf-8")
    scale_source = SCALE_TS.read_text(encoding="utf-8")

    assert _ts_value(scale_source, "WORKING_PERCENT_MIN") == gtx.WORKING_PERCENT_MIN
    assert _ts_value(scale_source, "WORKING_PERCENT_MAX") == gtx.WORKING_PERCENT_MAX
    # The mirror imports the range instead of restating it, so a gate that widens or
    # narrows cannot leave the panel quoting the old numbers.
    assert "WORKING_PERCENT_MIN" in source and "WORKING_PERCENT_MAX" in source
    assert "workingPercent" in source
