"""Pure regressions for the offline export acceptance checker."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "optimization_export_gate", ROOT / "tests/hardware/optimization_export_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def packet(index, pts, digest, duration=0.04):
    return {"pts_time": str(pts), "duration_time": str(duration),
            "sha256": digest, "complete": True}


def facts(video_digests=("v0", "v1", "v2"), audio_digests=("a0", "a1", "a2"),
          subtitle_digests=("s0", "s1"), chapter_end=500):
    video_stream = {"codec_type": "video", "color_range": "tv",
                    "color_space": "bt2020nc", "color_transfer": "smpte2084",
                    "color_primaries": "bt2020", "side_data_list": [
                        {"side_data_type":"Mastering display metadata","max_luminance":1000},
                        {"side_data_type":"Content light level metadata","max_content":1000}]}
    audio = {"codec_type": "audio", "codec_name": "aac", "tags": {"language": "jpn"},
             "disposition": {"default": 1}}
    subtitle = {"codec_type": "subtitle", "codec_name": "subrip",
                "tags": {"language": "eng"}, "disposition": {"original": 1}}
    attach = {"codec_type": "attachment"}
    pts = (0.0, 0.04, 0.08)
    chapter_data = [{"start": 0, "end": 100, "time_base": "1/1000",
                     "tags": {"title": "early"}},
                    {"start": 100, "end": 500, "time_base": "1/1000",
                     "tags": {"title": "crosses cutoff"}}]
    return {"video": video_stream, "streams": [video_stream, audio, subtitle, attach],
            "video_packets": [packet(i, pts[i], d) for i, d in enumerate(video_digests)],
            "frames": [{"best_effort_timestamp_time": str(x)} for x in pts[:len(video_digests)]],
            "packets_by_kind": {
                "audio": [packet(0, p, d) for p, d in zip((-0.01, .03, .15),
                                                             ("a0", "a1", "a2"))
                          if d in audio_digests],
                "subtitle": [packet(0, p, d, .5) for p, d in zip((.04, .24),
                                                                     ("s0", "s1"))
                             if d in subtitle_digests]},
            "chapters": [{**chapter_data[0]},
                         {**chapter_data[1], "end": chapter_end}],
            "format": {"tags": {"title": "fixture", "COMMENT": "kept"}},
            "attachment_sha256": ["attachment-content-hash"]}


def hdr_sei():
    return [{"payload_type": 137, "bytes": "deadbeef"},
            {"payload_type": 144, "bytes": "cafe"}]


def all_pass(checks):
    return [name for name, (ok, _) in checks.items() if not ok]


def test_checker_accepts_full_video_and_video_bounded_copied_packets():
    source, intermediate, final = facts(), facts(), facts()
    # The remux keeps packet starts before 120 ms; later audio/subtitle packets
    # are excluded while a crossing subtitle's original duration is not rewritten.
    final["packets_by_kind"]["audio"] = [packet(0, -.01, "a0"), packet(0, .03, "a1")]
    final["packets_by_kind"]["subtitle"] = [packet(0, .04, "s0", .5)]
    final["chapters"][1]["end"] = 120
    checks = gate.evaluate_artifacts(source, intermediate, final, 3, "hdr",
                                     ["ffmpeg", "-t", "0.12"],
                                     hdr_sei(), hdr_sei(), hdr_sei())
    assert all_pass(checks) == []


def test_checker_rebases_nonzero_source_timeline_and_keeps_boundary_chapter():
    source, intermediate, final = facts(), facts(), facts()
    source["format"]["start_time"] = "0.021"
    source["packets_by_kind"] = {
        "audio": [packet(0, .021, "a0"), packet(0, .051, "a1"), packet(0, .241, "a2")],
        "subtitle": [packet(0, .061, "s0", .5)]}
    final["packets_by_kind"] = {
        "audio": [packet(0, 0, "a0"), packet(0, .030, "a1")],
        "subtitle": [packet(0, .040, "s0", .5)]}
    source["chapters"] = [{"start": 21, "end": 141, "time_base": "1/1000", "tags": {"title": "early"}},
                          {"start": 141, "end": 261, "time_base": "1/1000", "tags": {"title": "boundary"}}]
    final["chapters"] = [{"start": 0, "end": 120, "time_base": "1/1000", "tags": {"title": "early"}},
                         {"start": 120, "end": 120, "time_base": "1/1000", "tags": {"title": "boundary"}}]
    checks = gate.evaluate_artifacts(source, intermediate, final, 3, "hdr",
                                     ["ffmpeg", "-t", "0.12"],
                                     hdr_sei(), hdr_sei(), hdr_sei())
    assert all_pass(checks) == []
    final["packets_by_kind"]["audio"][0]["pts_time"] = ".010"
    assert not gate.evaluate_artifacts(source, intermediate, final, 3, "hdr",
        ["ffmpeg", "-t", "0.12"], hdr_sei(), hdr_sei(), hdr_sei())["audio packet payload/cutoff check"][0]


def test_checker_rejects_video_packet_or_display_extent_shortfall():
    source, intermediate, final = facts(), facts(), facts(("v0", "v1"))
    checks = gate.evaluate_artifacts(source, intermediate, final, 3, "hdr",
                                     ["ffmpeg", "-t", "0.12"],
                                     hdr_sei(), hdr_sei(), hdr_sei())
    assert not checks["complete encoded video packets preserved"][0]
    assert not checks["display PTS/count and full video extent preserved"][0]


def test_checker_rejects_nonincreasing_video_timestamps():
    source, intermediate, final = facts(), facts(), facts()
    final["frames"][1]["best_effort_timestamp_time"] = "0.0"
    checks = gate.evaluate_artifacts(source, intermediate, final, 3, "hdr",
                                     ["ffmpeg", "-t", "0.12"],
                                     hdr_sei(), hdr_sei(), hdr_sei())
    assert not checks["display PTS/count and full video extent preserved"][0]


def test_checker_rejects_late_cue_and_unbounded_chapter_tail():
    source, intermediate, final = facts(), facts(), facts()
    final["packets_by_kind"]["subtitle"].append(packet(0, .24, "s1", .5))
    final["chapters"][1]["end"] = 500
    checks = gate.evaluate_artifacts(source, intermediate, final, 3, "hdr",
                                     ["ffmpeg", "-t", "0.12"],
                                     hdr_sei(), hdr_sei(), hdr_sei())
    assert not checks["subtitle packet payload/cutoff check"][0]
    assert not checks["chapters retained with video-span cutoff semantics"][0]


def test_checker_rejects_sei_rewrite_and_shortest_policy():
    source, intermediate, final = facts(), facts(), facts()
    altered = hdr_sei()
    altered[0] = {"payload_type": 137, "bytes": "baad"}
    checks = gate.evaluate_artifacts(source, intermediate, final, 3, "hdr",
                                     ["ffmpeg", "-t", "0.12", "-shortest"],
                                     hdr_sei(), hdr_sei(), altered)
    assert not checks["HDR SEI 137/144 exact source/intermediate/final"][0]
    assert not checks["video-authoritative remux cutoff equals encoded span"][0]


def test_hdr_side_data_equivalent_rationals_normalize_and_deduplicate():
    source, intermediate, final = facts(), facts(), facts()
    source_side = [{"side_data_type": "Mastering display metadata",
                    "blue_x": "6550/50000", "max_luminance": "10000000/10000"}]
    remux_side = [{"side_data_type": "Mastering display metadata",
                   "blue_x": "131/1000", "max_luminance": "1000/1"}]
    source["video"]["side_data_list"] = source_side
    source["streams"][0]["side_data_list"] = source_side
    intermediate["video"]["side_data_list"] = remux_side
    intermediate["streams"][0]["side_data_list"] = remux_side
    final["video"]["side_data_list"] = remux_side
    final["streams"][0]["side_data_list"] = remux_side

    source_facts = gate.hdr_side_data_facts(source)
    assert source_facts == gate.hdr_side_data_facts(intermediate)
    assert source_facts == gate.hdr_side_data_facts(final)
    assert len(source_facts["Mastering display metadata"]) == 1


def test_hdr_side_data_rejects_real_rational_change():
    source, intermediate, final = facts(), facts(), facts()
    source_side = [
        {"side_data_type": "Mastering display metadata", "blue_x": "6550/50000"},
        {"side_data_type": "Content light level metadata", "max_content": 900}]
    intermediate_side = [
        {"side_data_type": "Mastering display metadata", "blue_x": "132/1000"},
        {"side_data_type": "Content light level metadata", "max_content": 900}]
    source["video"]["side_data_list"] = source_side
    intermediate["video"]["side_data_list"] = intermediate_side
    final["video"]["side_data_list"] = intermediate_side
    checks = gate.evaluate_artifacts(source, intermediate, final, 3, "hdr",
                                     ["ffmpeg", "-t", "0.12"],
                                     hdr_sei(), hdr_sei(), hdr_sei())
    assert not checks["HDR mastering/CLL side data exact source/intermediate/final"][0]


def test_inactive_roi_bounds_do_not_mark_post_controls_active():
    checks = gate.evaluate_control_diagnostics(
        {}, {"post_controls": {"roi_bounds": [0.1, 0.1, 0.9, 0.9]}}, 3)
    assert "requested post-control settings reported" not in checks


def test_cutoff_duration_uses_encoded_span_when_video_has_nonzero_start():
    source, intermediate, final = facts(), facts(), facts()
    for artifact in (intermediate, final):
        for packet_item in artifact["video_packets"]:
            packet_item["pts_time"] = str(float(packet_item["pts_time"]) + 10.0)
        for frame in artifact["frames"]:
            frame["best_effort_timestamp_time"] = str(
                float(frame["best_effort_timestamp_time"]) + 10.0)
    final["packets_by_kind"] = source["packets_by_kind"]
    final["chapters"] = source["chapters"]
    checks = gate.evaluate_artifacts(source, intermediate, final, 3, "hdr",
                                     ["ffmpeg", "-t", "0.12"],
                                     hdr_sei(), hdr_sei(), hdr_sei())
    assert checks["video-authoritative remux cutoff equals encoded span"][0]


def test_requested_post_and_temporal_settings_are_verified_without_accepting_every_frame():
    job = {"post_controls": {"total_gain": 0.8, "hue_protection": 0.5},
           "temporal_controls": {"scene_detection": True,
                                 "mode": "static_low_frequency",
                                 "history_weight": 0.6}}
    controls = {"total_gain": .8, "lightness_gain": 1.0, "chroma_gain": 1.0,
        "low_frequency_gain": 1.0, "high_frequency_gain": 1.0,
        "hue_protection": .5, "dark_protection": 0.0,
        "highlight_protection": 0.0, "soft_compression": 0.0,
        "roi_kind": 0, "roi_bounds": [0.0,0.0,1.0,1.0],
        "roi_feather": .02, "debug_mode": 0}
    diagnostics = {
        "post_control": {"enabled": True, "controls": controls,
                          "process_count": 3, "host_image_bytes": 0},
        "temporal_control": {"enabled": True,
            "controls": {"scene_detection": True, "mode": "static_low_frequency",
                         "history_weight": .6},
                "observations": [{"classification":"first", "scalar_readback_bytes":32},
                                 {"classification":"translation", "scalar_readback_bytes":32},
                                 {"classification":"ambiguous", "scalar_readback_bytes":32}],
                "process_reports": [{"scalar_readback_bytes":16} for _ in range(3)],
                "host_image_bytes": 0}}
    checks = gate.evaluate_control_diagnostics(diagnostics, job, 3)
    assert all_pass(checks) == []
    # Observations can legitimately reject history on a moving source. The checker
    # verifies diagnostics are present, not that the filter accepts every frame.
    assert [x["classification"] for x in diagnostics["temporal_control"]["observations"]] == [
        "first", "translation", "ambiguous"]


def test_scene_observation_only_temporal_reports_scalar_bytes_not_frame_acceptance():
    temporal = {"controls": {"scene_detection": True, "mode": "off",
                              "history_weight": .6},
                "enabled": True, "host_image_bytes": 0,
                "observations": [{"classification":"ambiguous",
                                  "scalar_readback_bytes":32} for _ in range(3)],
                "process_reports": []}
    checks = gate.evaluate_control_diagnostics(
        {"temporal_control": temporal}, {"temporal_controls": {
            "scene_detection": True, "mode": "off", "history_weight": .6}}, 3)
    assert checks["requested temporal settings and diagnostics reported"][0]


def test_temporal_processing_requires_one_report_and_16_scalar_bytes_per_frame():
    controls = {"scene_detection": False, "mode": "static_low_frequency",
                "history_weight": .6}
    temporal = {"controls": controls, "enabled": True, "host_image_bytes": 0,
                "observations": [],
                "process_reports": [{"scalar_readback_bytes":16} for _ in range(3)]}
    settings = {"temporal_controls": controls}
    checks = gate.evaluate_control_diagnostics({"temporal_control": temporal}, settings, 3)
    assert checks["requested temporal settings and diagnostics reported"][0]
    temporal["process_reports"][1]["scalar_readback_bytes"] = 32
    checks = gate.evaluate_control_diagnostics({"temporal_control": temporal}, settings, 3)
    assert not checks["requested temporal settings and diagnostics reported"][0]


def test_scene_observer_cannot_report_first_for_every_frame():
    controls = {"scene_detection": True, "mode": "static_low_frequency",
                "history_weight": .6}
    temporal = {"controls": controls, "enabled": True, "host_image_bytes": 0,
                "observations": [{"classification":"first", "scene_cut":True,
                                  "scalar_readback_bytes":32} for _ in range(3)],
                "process_reports": [{"scalar_readback_bytes":16} for _ in range(3)]}
    checks = gate.evaluate_control_diagnostics(
        {"temporal_control": temporal}, {"temporal_controls": controls}, 3)
    assert not checks["requested temporal settings and diagnostics reported"][0]


def _multiscale_diagnostics(case="sdr"):
    rows = [[True, False, False], [True, False, False]]
    if case == "sdr":
        return {"layer_count": 2, "layer_scales": [.5, .75],
                "layer_working_dims": [[960, 544], [1440, 808]],
                "layer_session_ids": [11, 12], "layer_reset_flags": rows,
                "layer_dlss_calls": [3, 3], "fusion_mode": "data_consistency_experimental",
                "fusion_solver_frames": [
                    {"iterations": 7, "relative_residual": .0004, "converged": True}
                    for _ in range(3)],
                "fusion_scalar_readback_bytes_measured": 120,
                "layer_transfer_counters": [{"dlss": {
                    "host_color_stage_bytes": 0, "output_readback_bytes": 0}}],
                "decode_host_copy_bytes_measured": 0,
                "dlss_host_color_stage_bytes_measured": 0,
                "dlss_output_readback_bytes_measured": 0,
                "residual_output_readback_bytes_measured": 0,
                "encoder_host_pixel_bytes_measured": 0}
    return {"hdr_layers_measured": 2, "hdr_layer_scales": [.5, .75],
            "hdr_layer_working_dims": [[960, 544], [1440, 808]],
            "hdr_layer_session_ids": [21, 22], "hdr_layer_reset_flags": rows,
            "hdr_layer_calls_measured": [3, 3],
            "hdr_fusion_mode_requested": "data_consistency_experimental",
            "hdr_layer_transfers_measured": [{
                "process_count_measured": 3, "working_dims": [1440, 808],
                "hdr_diagnostic_image_readback_bytes_measured": 0,
                "dlss_measured": {"host_color_stage_bytes": 0,
                                   "output_readback_bytes": 0}}],
            "hdr_scalar_report_readback": {"fusion_scalar_readback_bytes_measured": 120},
            "hdr_frame_reports": [{"frame_idx":i, "pts":i*40,
                                   "time_base":[1, 1000], "reset":i == 0,
                                   "fusion_mode_actual":"typed_gpu_data_consistency_solve",
                                   "solver":{"iterations":7,"converged":True,
                                             "report":{"relative_residual":.0004}}
                                   } for i in range(3)],
            "decode_host_copy_bytes_measured": 0,
            "dlss_host_color_stage_bytes_measured": 0,
            "dlss_output_readback_bytes_measured": 0,
            "hdr_diagnostic_image_readback_bytes_measured": 0,
            "encoder_host_pixel_bytes_measured": 0,
            "hdr_pixel_readback_exports_bound": []}


def test_sdr_multiscale_checks_layer_lifecycle_solver_and_image_boundaries():
    settings = {"layer_scales": [.5, .75], "fusion_relative_tolerance": .001,
                "fusion_max_iterations": 128}
    d = _multiscale_diagnostics("sdr")
    checks = gate.evaluate_multiscale_diagnostics(
        d, settings, 3, "sdr", "dual", [[960, 544], [1440, 808]])
    assert all_pass(checks) == []
    d["layer_session_ids"] = [11, 11]
    d["layer_reset_flags"][1] = [False, False, False]
    d["layer_transfer_counters"][0]["dlss"]["output_readback_bytes"] = 1
    checks = gate.evaluate_multiscale_diagnostics(
        d, settings, 3, "sdr", "dual", [[960, 544], [1440, 808]])
    assert not checks["SDR layers use distinct DLSS sessions"][0]
    assert not checks["SDR layers reset consistently across frames"][0]
    assert not checks["SDR primary and extra-layer image host boundaries are zero"][0]


def test_hdr_multiscale_checks_distinct_sessions_and_typed_solver():
    settings = {"layer_scales": [.5, .75], "fusion_relative_tolerance": .001,
                "fusion_max_iterations": 128}
    d = _multiscale_diagnostics("hdr")
    checks = gate.evaluate_multiscale_diagnostics(
        d, settings, 3, "hdr", "dual", [[960, 544], [1440, 808]])
    assert all_pass(checks) == []
    d["hdr_frame_reports"][0]["solver"]["converged"] = False
    checks = gate.evaluate_multiscale_diagnostics(
        d, settings, 3, "hdr", "dual", [[960, 544], [1440, 808]])
    assert not checks["HDR fusion solver converges within configured limits"][0]


def test_multiscale_reset_sequences_follow_scene_cut_observations():
    d = _multiscale_diagnostics("sdr")
    d["temporal_control"] = {"controls": {"scene_detection": True},
        "observations": [{"scene_cut": True}, {"scene_cut": False},
                         {"scene_cut": True}]}
    d["layer_reset_flags"] = [[True, False, True], [True, False, True]]
    settings = {"layer_scales": [.5, .75], "fusion_relative_tolerance": .001,
                "fusion_max_iterations": 128, "temporal_controls": {
                    "scene_detection": True}}
    checks = gate.evaluate_multiscale_diagnostics(
        d, settings, 3, "sdr", "dual", [[960, 544], [1440, 808]])
    assert checks["SDR layers reset consistently across frames"][0]
    d["layer_reset_flags"][1] = [True, True, True]
    checks = gate.evaluate_multiscale_diagnostics(
        d, settings, 3, "sdr", "dual", [[960, 544], [1440, 808]])
    assert not checks["SDR layers reset consistently across frames"][0]


def test_hdr_multiscale_frame_reports_validate_continuous_pts_and_scene_resets():
    d = _multiscale_diagnostics("hdr")
    d["temporal_control"] = {"controls": {"scene_detection": True},
        "observations": [{"scene_cut": True}, {"scene_cut": False},
                         {"scene_cut": True}]}
    d["hdr_layer_reset_flags"] = [[True, False, True], [True, False, True]]
    for i, row in enumerate(d["hdr_frame_reports"]):
        row.update(frame_idx=i, pts=i * 40, time_base=[1, 1000],
                   reset=(i == 0 or i == 2))
    settings = {"layer_scales": [.5, .75], "fusion_relative_tolerance": .001,
                "fusion_max_iterations": 128, "temporal_controls": {
                    "scene_detection": True}}
    checks = gate.evaluate_multiscale_diagnostics(
        d, settings, 3, "hdr", "dual", [[960, 544], [1440, 808]])
    assert checks["HDR layers reset consistently across frames"][0]
    assert checks["HDR frame report timestamps and indices are continuous"][0]
    d["hdr_frame_reports"][2]["pts"] = 20
    checks = gate.evaluate_multiscale_diagnostics(
        d, settings, 3, "hdr", "dual", [[960, 544], [1440, 808]])
    assert not checks["HDR frame report timestamps and indices are continuous"][0]


def test_runner_cleanup_closes_owned_service_after_live_even_if_live_close_fails():
    calls = []

    class Owner:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def close(self):
            calls.append(self.name)
            if self.fail:
                raise RuntimeError("close failed")

    error = gate._close_owned_runner_resources(
        Owner("live", fail=True), Owner("service"))
    assert calls == ["live", "service"]
    assert error == "session close failed: close failed"


def test_baseline_setup_comparison_normalizes_json_tuple_list_shapes():
    runtime = {"working_dims": ((768, 432), [1288, 720]),
               "job_settings": {"scales": (.4, .67, .85)}}
    saved = {"working_dims": [[768, 432], [1288, 720]],
             "job_settings": {"scales": [.4, .67, .85]}}
    assert gate.json_shape(runtime) == gate.json_shape(saved)
    saved["working_dims"][1][0] = 1280
    assert gate.json_shape(runtime) != gate.json_shape(saved)


def test_decoded_hdr_comparison_reports_small_and_rejects_large_differences():
    baseline = np.zeros((2, 8), dtype=np.uint16)
    small = np.full((2, 8), 4, dtype=np.uint16)
    metrics = gate.decoded_delta_metrics(baseline, small)
    assert metrics["within_diagnostic_bounds"]
    assert metrics["mean_abs_code"] == 4
    assert metrics["rms_code"] == 4
    huge = np.full((2, 8), 600, dtype=np.uint16)
    metrics = gate.decoded_delta_metrics(baseline, huge)
    assert not metrics["within_diagnostic_bounds"]
    assert metrics["max_abs_code"] == 600
