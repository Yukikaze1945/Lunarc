# -*- coding: utf-8 -*-
"""Short hardware acceptance harness for current SDR/HDR GPU exports.

Run (PowerShell):
  python tests/hardware/optimization_export_gate.py hdr triple --fixture bench_out/hdr_remux_export/hdr_remux_src.mkv --evidence-dir bench_out/optimization_20261009/export_hdr_triple
  python tests/hardware/optimization_export_gate.py sdr dual --fixture <sdr-fixture.mkv> --evidence-dir bench_out/optimization_20261009/export_sdr_dual
  python tests/hardware/optimization_export_gate.py hdr triple --fixture bench_out/hdr_remux_export/hdr_remux_src.mkv --evidence-dir bench_out/optimization_20261009/export_hdr_triple_baseline --fusion-dll bench_out/optimization_20261009/fusion_baseline/gpu_multiscale_fusion_bridge.dll
  python tests/hardware/optimization_export_gate.py hdr triple --fixture bench_out/hdr_remux_export/hdr_remux_src.mkv --evidence-dir bench_out/optimization_20261009/export_hdr_triple_new --compare-video-evidence bench_out/optimization_20261009/export_hdr_triple_baseline

The default rich fixture is HDR PQ. An SDR run must supply a fixture admitted by
the production SDR path. Settings JSON may shallow-override ``job_settings`` and
``transcode_settings`` and set ``frames``, ``cq``, or ``with_audio``. No export
is run by ``--recheck``. Evidence directories must be fresh and under bench_out.

The encoded video is authoritative: intermediate/final video packet payloads,
display PTS, frame count and extent must match completely. Copied audio/subtitle
packet payloads are compared with source packets whose start is before the video
end. Cue duration and audio codec padding remain intact and can extend container
duration. Chapters starting at or before the encoded video end are retained and
their ends are clipped to it; FFmpeg can retain a zero-duration boundary chapter.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = ROOT / "bench_out/hdr_remux_export/hdr_remux_src.mkv"
COLOR_FIELDS = ("color_range", "color_space", "color_transfer", "color_primaries")
DYNAMIC_TAGS = {"encoder", "duration", "number_of_frames", "bps"}
DECODED_DELTA_LIMITS_10BIT = {"max_abs_code": 512, "mean_abs_code": 32.0,
                             "rms_code": 64.0}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_shape(value):
    """Compare runtime tuples and persisted JSON arrays with the same shape."""
    if isinstance(value, dict):
        return {str(key): json_shape(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_shape(item) for item in value]
    return value


def decoded_delta_metrics(baseline, current):
    """Summarize decoded 10-bit YUV samples, with explicit large-error limits."""
    import numpy as np
    a = np.asarray(baseline, dtype=np.int32)
    b = np.asarray(current, dtype=np.int32)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"decoded arrays must be nonempty and equal-shaped: {a.shape}/{b.shape}")
    delta = np.abs(a - b)
    rms = float(np.sqrt(np.mean(np.square(a.astype(np.float64) - b.astype(np.float64)))))
    result = {"sample_count": int(a.size),
              "baseline_range": [int(a.min()), int(a.max())],
              "current_range": [int(b.min()), int(b.max())],
              "max_abs_code": int(delta.max()),
              "mean_abs_code": float(delta.mean()),
              "rms_code": rms,
              "limits_10bit_code": dict(DECODED_DELTA_LIMITS_10BIT)}
    result["within_diagnostic_bounds"] = (
        result["max_abs_code"] <= DECODED_DELTA_LIMITS_10BIT["max_abs_code"] and
        result["mean_abs_code"] <= DECODED_DELTA_LIMITS_10BIT["mean_abs_code"] and
        result["rms_code"] <= DECODED_DELTA_LIMITS_10BIT["rms_code"])
    if a.ndim >= 2:
        frames = a.reshape((a.shape[0], -1))
        other = b.reshape((b.shape[0], -1))
        result["per_frame"] = [
            {"frame": index,
             "max_abs_code": int(np.abs(frames[index]-other[index]).max()),
             "mean_abs_code": float(np.abs(frames[index]-other[index]).mean()),
             "rms_code": float(np.sqrt(np.mean(np.square(
                 frames[index].astype(np.float64)-other[index].astype(np.float64)))))}
            for index in range(frames.shape[0])]
    return result


def compare_decoded_hdr_video(ffmpeg, baseline_path, current_path, evidence_dir,
                              baseline_facts, current_facts, frames):
    """Decode comparison outputs to offline 10-bit planar YUV for numeric review."""
    import numpy as np
    base_video = baseline_facts.get("video", {})
    current_video = current_facts.get("video", {})
    width, height = int(base_video.get("width") or 0), int(base_video.get("height") or 0)
    dims_match = (width > 0 and height > 0 and width % 2 == 0 and height % 2 == 0 and
                  width == int(current_video.get("width") or 0) and
                  height == int(current_video.get("height") or 0))
    base_pts, current_pts = _display_pts(baseline_facts), _display_pts(current_facts)
    timeline_match = (len(base_pts) == len(current_pts) == frames and
                      all(abs(a-b) <= 0.0011 for a,b in zip(base_pts, current_pts)))
    result = {"format": "yuv420p10le", "width": width, "height": height,
              "frames_requested": frames, "baseline_display_pts": base_pts,
              "current_display_pts": current_pts,
              "dimensions_match": dims_match, "display_pts_match": timeline_match,
              "packet_payloads_bitexact":
                  [p.get("sha256") for p in baseline_facts.get("video_packets", [])] ==
                  [p.get("sha256") for p in current_facts.get("video_packets", [])]}
    if not dims_match or not timeline_match:
        result["within_diagnostic_bounds"] = False
        result["error"] = "decoded comparison requires matching dimensions, frame count and display PTS"
        return result

    evidence_dir = Path(evidence_dir)
    base_raw = evidence_dir / "comparison_baseline_yuv420p10le.raw"
    current_raw = evidence_dir / "comparison_current_yuv420p10le.raw"
    expected_values = frames * width * height * 3 // 2
    raw_info = []
    for label, path, raw in (("baseline", baseline_path, base_raw),
                             ("current", current_path, current_raw)):
        _run([ffmpeg, "-v", "error", "-y", "-i", str(path), "-map", "0:v:0",
              "-frames:v", str(frames), "-fps_mode", "passthrough",
              "-pix_fmt", "yuv420p10le", "-f", "rawvideo", str(raw)])
        if raw.stat().st_size != expected_values * 2:
            result["within_diagnostic_bounds"] = False
            result["error"] = (f"{label} decoded {raw.stat().st_size} bytes; expected "
                               f"{expected_values*2}")
            return result
        raw_info.append({"label": label, "path": str(raw),
                         "sha256": sha256_file(raw), "bytes": raw.stat().st_size})
    a = np.fromfile(base_raw, dtype="<u2").reshape((frames, -1))
    b = np.fromfile(current_raw, dtype="<u2").reshape((frames, -1))
    result.update(decoded_delta_metrics(a, b))
    result["raw_diagnostics"] = raw_info
    return result


def hexdump_bytes(text):
    out = bytearray()
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        offset, rest = line.strip().split(":", 1)
        try:
            at = int(offset, 16)
        except ValueError:
            continue
        if at != len(out):
            raise ValueError(f"ffprobe hexdump offset {at} != {len(out)}")
        digits = "".join(x for x in rest.split("  ", 1)[0].split()
                         if x and all(c in "0123456789abcdefABCDEF" for c in x))
        if len(digits) % 2:
            raise ValueError("ragged ffprobe hexdump")
        out.extend(bytes.fromhex(digits))
    return bytes(out)


def stable_tags(tags):
    return {str(k).lower(): str(v) for k, v in (tags or {}).items()
            if str(k).lower() not in DYNAMIC_TAGS}


def hdr_side_data_facts(facts):
    wanted = {"Mastering display metadata", "Content light level metadata"}
    found = defaultdict(set)
    streams = facts.get("streams", [])
    video_streams = [stream for stream in streams
                     if stream.get("codec_type") == "video"]
    values = [facts.get("video", {}).get("side_data_list", [])]
    values += [stream.get("side_data_list", []) for stream in video_streams]
    values += [frame.get("side_data_list", []) for frame in facts.get("frames", [])]

    def canonicalize(value):
        if isinstance(value, dict):
            return {key: canonicalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonicalize(item) for item in value]
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+/[+-]?\d+", value):
            rational = Fraction(value)
            return f"{rational.numerator}/{rational.denominator}"
        return value

    for group in values:
        for item in group or []:
            kind = item.get("side_data_type")
            if kind in wanted:
                normalized = canonicalize(item)
                found[kind].add(json.dumps(normalized, sort_keys=True,
                                           separators=(",", ":")))
    return {kind: sorted(items) for kind, items in found.items()}


def video_extent(facts):
    packets = facts.get("video_packets", [])
    starts = [float(p["pts_time"]) for p in packets if p.get("pts_time") is not None]
    ends = [float(p["pts_time"]) + float(p.get("duration_time") or 0)
            for p in packets if p.get("pts_time") is not None]
    return (min(starts), max(ends)) if starts and ends else (math.nan, math.nan)


def _display_pts(facts):
    return [float(x["best_effort_timestamp_time"]) for x in facts.get("frames", [])
            if x.get("best_effort_timestamp_time") is not None]


def _chapter_time(chapter, key):
    numerator, denominator = (int(x) for x in
                              (chapter.get("time_base") or "1/1").split("/"))
    return int(chapter[key]) * numerator / denominator


def expected_chapter_facts(chapters, video_end, source_offset=0.0):
    result = []
    for chapter in chapters:
        start, end = _chapter_time(chapter, "start"), _chapter_time(chapter, "end")
        start, end = start-source_offset, end-source_offset
        if start <= video_end:
            result.append((start, min(end, video_end),
                           stable_tags(chapter.get("tags")).get("title")))
    return result


def evaluate_artifacts(source, intermediate, final, frames, case, mux_command,
                       source_sei=None, intermediate_sei=None, final_sei=None,
                       transcode_settings=None, with_audio=True):
    """Pure checker: facts are synthetic-testable and require no subprocess/GPU."""
    checks = {}
    src_video, int_video, out_video = (f.get("video", {}) for f in
                                       (source, intermediate, final))
    int_packets, out_packets = (intermediate.get("video_packets", []),
                               final.get("video_packets", []))
    ih = [p.get("sha256") for p in int_packets]
    oh = [p.get("sha256") for p in out_packets]
    int_pts, out_pts = _display_pts(intermediate), _display_pts(final)
    start, end = video_extent(intermediate)
    pts_match = len(int_pts) == len(out_pts) == frames and all(
        abs(a-b) <= 0.0011 for a, b in zip(int_pts, out_pts))
    checks["complete encoded video packets preserved"] = (
        len(int_packets) == len(out_packets) == frames and
        all(p.get("complete") for p in int_packets + out_packets) and ih == oh,
        f"packet counts {len(int_packets)}/{len(out_packets)}, equal={ih == oh}")
    checks["display PTS/count and full video extent preserved"] = (
        pts_match and all(b > a for a, b in zip(out_pts, out_pts[1:])) and
        math.isfinite(start) and math.isfinite(end),
        f"PTS {int_pts} -> {out_pts}; extent [{start},{end})")
    checks["color declaration exact source/intermediate/final"] = (
        all(int_video.get(k) == out_video.get(k) and
            (src_video.get(k) == int_video.get(k) or
             (case == "sdr" and src_video.get(k) is None))
            for k in COLOR_FIELDS),
        json.dumps({k: [src_video.get(k), int_video.get(k), out_video.get(k)]
                    for k in COLOR_FIELDS}))
    source_video_tags = stable_tags(src_video.get("tags"))
    final_video_tags = stable_tags(out_video.get("tags"))
    video_descriptive = (source_video_tags == final_video_tags and
                         src_video.get("disposition") == out_video.get("disposition"))
    if case == "hdr":
        checks["HDR video descriptive metadata/disposition retained"] = (
            video_descriptive, f"source={source_video_tags}; final={final_video_tags}")

    def by_kind(facts, kind):
        return [s for s in facts.get("streams", []) if s.get("codec_type") == kind]

    ts = dict(transcode_settings or {})
    # FFmpeg's normal input timeline rebases by the input format start_time.
    # Compare copied payloads and timestamps on that same timeline; packets
    # can precede the declared video start because of audio encoder priming.
    source_offset = (0.0 if "-copyts" in (mux_command or []) else
                     float(source.get("format", {}).get("start_time") or 0.0))
    audio_mode = "remove" if not with_audio else str(ts.get("audio_mode", "copy"))
    subtitle_mode = str(ts.get("subtitle_mode", "copy"))
    expect_audio = len(by_kind(source, "audio")) if audio_mode != "remove" else 0
    expect_subtitle = len(by_kind(source, "subtitle")) if subtitle_mode != "drop" else 0
    expect_attachment = len(by_kind(source, "attachment")) if ts.get("copy_attachments", True) else 0
    mappings = {kind: (len(by_kind(source, kind)), len(by_kind(final, kind)))
                for kind in ("video", "audio", "subtitle", "attachment")}
    checks["stream mapping counts follow requested mapping modes"] = (
        mappings["video"] == (1, 1) and mappings["audio"][1] == expect_audio and
        mappings["subtitle"][1] == expect_subtitle and
        mappings["attachment"][1] == expect_attachment,
        json.dumps({"counts":mappings,"expected_final":{"audio":expect_audio,
                     "subtitle":expect_subtitle,"attachment":expect_attachment}}))
    for kind in ("audio", "subtitle"):
        source_packets = source.get("packets_by_kind", {}).get(kind, [])
        final_packets = final.get("packets_by_kind", {}).get(kind, [])
        actual = [p.get("sha256") for p in final_packets]
        starts_bounded = all(p.get("pts_time") is not None and math.isfinite(end)
                             and float(p["pts_time"]) < end for p in final_packets)
        copy_mode = (audio_mode == "copy" if kind == "audio" else
                     subtitle_mode == "copy")
        if copy_mode:
            expected = [p.get("sha256") for p in source_packets
                        if p.get("pts_time") is not None and math.isfinite(end) and
                        float(p["pts_time"])-source_offset < end]
            expected_pts = [float(p["pts_time"])-source_offset for p in source_packets
                            if p.get("pts_time") is not None and math.isfinite(end) and
                            float(p["pts_time"])-source_offset < end]
            packet_ok = (actual == expected and starts_bounded and
                         all(p.get("complete") for p in final_packets) and
                         len(expected_pts) == len(final_packets) and all(
                             abs(expected_time-float(p["pts_time"])) <= .002
                             for expected_time, p in zip(expected_pts, final_packets)))
            detail = f"expected={len(expected)} final={len(actual)} starts_bounded={starts_bounded} cutoff={end} source_offset={source_offset}"
        else:
            packet_ok = all(p.get("complete") for p in final_packets)
            detail = f"mode={audio_mode if kind == 'audio' else subtitle_mode}; " \
                     f"payload equality not applicable; packets={len(actual)}"
        checks[f"{kind} packet payload/cutoff check"] = (packet_ok, detail)
        old, new = by_kind(source, kind), by_kind(final, kind)
        stream_copy = (audio_mode == "copy" if kind == "audio" else
                       subtitle_mode == "copy")
        removed = (audio_mode == "remove" if kind == "audio" else
                   subtitle_mode == "drop")
        metadata_equal = (not new if removed else
            len(old) == len(new) and all(
                (not stream_copy or a.get("codec_name") == b.get("codec_name")) and
                stable_tags(a.get("tags")) == stable_tags(b.get("tags")) and
                a.get("disposition") == b.get("disposition")
                for a, b in zip(old, new)))
        checks[f"{kind} codec/tags/disposition preserved"] = (
            metadata_equal, f"streams {len(old)}/{len(new)}")
    source_tags = stable_tags(source.get("format", {}).get("tags"))
    final_tags = stable_tags(final.get("format", {}).get("tags"))
    metadata_ok = (all(final_tags.get(k) == v for k, v in source_tags.items())
                   if ts.get("copy_metadata", True) else not final_tags)
    checks["global metadata follows copy_metadata setting"] = (
        metadata_ok, f"copy_metadata={ts.get('copy_metadata', True)}; "
        f"source={source_tags}, final={final_tags}")

    expected = (expected_chapter_facts(source.get("chapters", []), end, source_offset)
                if ts.get("copy_chapters", True) and math.isfinite(end) else [])
    actual = [(_chapter_time(c, "start"), _chapter_time(c, "end"),
               stable_tags(c.get("tags")).get("title"))
              for c in final.get("chapters", [])]
    chapters_ok = len(expected) == len(actual) and all(
        a[2] == e[2] and abs(a[0]-e[0]) <= .002 and abs(a[1]-e[1]) <= .002
        for e, a in zip(expected, actual))
    checks["chapters retained with video-span cutoff semantics"] = (
        chapters_ok, f"expected={expected}; final={actual}; cutoff={end}")
    source_attachments = source.get("attachment_sha256", [])
    final_attachments = final.get("attachment_sha256", [])
    attachments_ok = (source_attachments == final_attachments
                      if ts.get("copy_attachments", True) else not final_attachments)
    checks["attachment bytes preserved"] = (
        attachments_ok,
        f"source={source_attachments}; final={final_attachments}")

    if case == "hdr":
        source_side = hdr_side_data_facts(source)
        inter_side = hdr_side_data_facts(intermediate)
        final_side = hdr_side_data_facts(final)
        required_side = {"Mastering display metadata", "Content light level metadata"}
        checks["HDR mastering/CLL side data exact source/intermediate/final"] = (
            set(source_side) == set(inter_side) == set(final_side) == required_side and
            source_side == inter_side == final_side,
            json.dumps({"source":source_side,"intermediate":inter_side,"final":final_side}))
        per_type = {}
        for ptype in (137, 144):
            groups = [[x.get("bytes") for x in listing or []
                       if x.get("payload_type") == ptype]
                      for listing in (source_sei, intermediate_sei, final_sei)]
            exact = (all(groups) and all(len(set(g)) == 1 for g in groups) and
                     groups[0][0] == groups[1][0] == groups[2][0] and
                     len(groups[1]) == len(groups[2]))
            per_type[str(ptype)] = {"exact_payload": bool(exact),
                                    "counts": [len(g) for g in groups]}
        checks["HDR SEI 137/144 exact source/intermediate/final"] = (
            all(x["exact_payload"] for x in per_type.values()), json.dumps(per_type))
    else:
        checks["SDR SEI validation skipped"] = (True, "HDR SEIs are not required for SDR")
    command = [str(x) for x in mux_command or []]
    duration_matches = False
    duration_detail = repr(command)
    if "-t" in command and command.index("-t") + 1 < len(command) and math.isfinite(start) and math.isfinite(end):
        try:
            requested_duration = float(command[command.index("-t") + 1])
            expected_duration = end - start
            duration_matches = abs(requested_duration-expected_duration) <= max(1e-6, expected_duration*1e-6)
            duration_detail = f"-t={requested_duration:.9f}; encoded span={expected_duration:.9f}"
        except (ValueError, TypeError):
            pass
    checks["video-authoritative remux cutoff equals encoded span"] = (
        duration_matches and "-shortest" not in command, duration_detail)
    return checks


def evaluate_control_diagnostics(diagnostics, job_settings, frames):
    """Verify requested post/temporal controls are named in the export report."""
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    job_settings = job_settings if isinstance(job_settings, dict) else {}
    checks = {}
    def normalized(value):
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (list, tuple)):
            return [normalized(x) for x in value]
        if isinstance(value, dict):
            return {str(k): normalized(v) for k, v in value.items()}
        return value
    post_defaults = {"total_gain":1.0,"lightness_gain":1.0,"chroma_gain":1.0,
        "low_frequency_gain":1.0,"high_frequency_gain":1.0,"hue_protection":0.0,
        "dark_protection":0.0,"highlight_protection":0.0,"soft_compression":0.0,
        "roi_kind":0,"roi_bounds":[0.0,0.0,1.0,1.0],"roi_feather":0.02,"debug_mode":0}
    post_requested = dict(job_settings.get("post_controls") or {})
    expected_post = dict(post_defaults, **post_requested)
    post_active = any(expected_post[k] != post_defaults[k]
                      for k in post_defaults
                      if k not in {"roi_bounds", "roi_feather"})
    if expected_post["roi_kind"] != 0:
        post_active = post_active or any(
            expected_post[k] != post_defaults[k]
            for k in ("roi_bounds", "roi_feather"))
    post = diagnostics.get("post_control")
    if post_active:
        actual_controls = post.get("controls") if isinstance(post, dict) else None
        exact = actual_controls is not None and normalized(actual_controls) == \
            normalized(expected_post)
        post_count = post.get("process_count") if isinstance(post, dict) else None
        checks["requested post-control settings reported"] = (
            isinstance(post, dict) and post.get("enabled") is True and exact and
            post_count == frames and post.get("host_image_bytes") == 0,
            f"enabled={post.get('enabled') if isinstance(post,dict) else None}; "
            f"process_count={post_count}/{frames}; "
            f"host_image_bytes={post.get('host_image_bytes') if isinstance(post,dict) else None}; "
            f"controls={actual_controls}")
    elif post is not None:
        checks["post-control diagnostics recorded when present"] = (
            isinstance(post, dict) and isinstance(post.get("controls"), dict),
            repr(post))

    temporal_defaults = {"scene_detection":False,"mode":"off","history_weight":0.6}
    expected_temporal = dict(temporal_defaults, **dict(job_settings.get("temporal_controls") or {}))
    temporal_active = bool(expected_temporal["scene_detection"] or
                           expected_temporal["mode"] != "off")
    temporal = diagnostics.get("temporal_control")
    if temporal_active:
        actual = temporal.get("controls") if isinstance(temporal, dict) else None
        exact = actual is not None and normalized(actual) == normalized(expected_temporal)
        observations = temporal.get("observations", []) if isinstance(temporal, dict) else []
        process_reports = temporal.get("process_reports", []) if isinstance(temporal, dict) else []
        obs_scalars = ([x.get("scalar_readback_bytes") for x in observations]
                       if isinstance(observations, list) else [])
        process_scalars = ([x.get("scalar_readback_bytes") for x in process_reports]
                           if isinstance(process_reports, list) else [])
        observations_ok = (not expected_temporal["scene_detection"] or
                           (isinstance(observations, list) and len(observations) == frames and
                            all(value == 32 for value in obs_scalars)))
        process_ok = (expected_temporal["mode"] == "off" or
                      (isinstance(process_reports, list) and len(process_reports) == frames and
                       all(value == 16 for value in process_scalars)))
        classifications = ([item.get("classification") for item in observations]
                           if isinstance(observations, list) else [])
        scene_observations_valid = (
            not expected_temporal["scene_detection"] or frames <= 1 or
            (len(classifications) == frames and
             any(value != "first" for value in classifications[1:])))
        checks["requested temporal settings and diagnostics reported"] = (
            isinstance(temporal, dict) and temporal.get("enabled") is True and exact and
            observations_ok and process_ok and scene_observations_valid and
            temporal.get("host_image_bytes") == 0,
            f"enabled={temporal.get('enabled') if isinstance(temporal,dict) else None}; "
            f"controls={actual}; observations="
            f"{len(observations) if isinstance(observations,list) else None}/"
            f"{frames if expected_temporal['scene_detection'] else 0} scalars={obs_scalars}; "
            f"process_reports={len(process_reports) if isinstance(process_reports,list) else None}/"
            f"{frames if expected_temporal['mode'] != 'off' else 0} scalars={process_scalars}; "
            f"classifications={classifications}; "
            f"host_image_bytes="
            f"{temporal.get('host_image_bytes') if isinstance(temporal,dict) else None}")
    elif temporal is not None:
        checks["temporal diagnostics recorded when present"] = (
            isinstance(temporal, dict) and isinstance(temporal.get("controls"), dict),
            repr(temporal))
    return checks


def evaluate_multiscale_diagnostics(diagnostics, job_settings, frames, case,
                                    mode, expected_dims):
    """Verify layer identity, timeline and image-vs-scalar transfer boundaries."""
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    settings = job_settings if isinstance(job_settings, dict) else {}
    checks = {}
    expected_count = {"single": 1, "dual": 2, "triple": 3}[mode]
    expected_scales = (list(settings.get("layer_scales") or [
        settings.get("working_scale", 0.5)]) if mode != "single" else
        [settings.get("working_scale", 0.5)])
    expected_scales = [float(x) for x in expected_scales]
    expected_dims = [list(map(int, d)) for d in (expected_dims or [])]

    if case == "hdr":
        count = diagnostics.get("hdr_layers_measured")
        scales = diagnostics.get("hdr_layer_scales")
        dims = diagnostics.get("hdr_layer_working_dims")
        sessions = diagnostics.get("hdr_layer_session_ids")
        resets = diagnostics.get("hdr_layer_reset_flags")
        calls = diagnostics.get("hdr_layer_calls_measured")
        transfer_layers = diagnostics.get("hdr_layer_transfers_measured")
        fusion_mode = diagnostics.get("hdr_fusion_mode_requested")
        transfer_valid = isinstance(transfer_layers, list) and len(transfer_layers) == expected_count - 1
        per_layer_image_zero = []
        per_layer_calls_valid = []
        if transfer_valid:
            for item in transfer_layers:
                dlss = item.get("dlss_measured") or {}
                per_layer_calls_valid.append(
                    item.get("process_count_measured") == frames and
                    item.get("working_dims") in expected_dims[1:])
                per_layer_image_zero.append(
                    item.get("hdr_diagnostic_image_readback_bytes_measured") == 0 and
                    dlss.get("host_color_stage_bytes") == 0 and
                    dlss.get("output_readback_bytes") == 0)
        scalar = (diagnostics.get("hdr_scalar_report_readback") or {}).get(
            "fusion_scalar_readback_bytes_measured")
        image_keys = ("decode_host_copy_bytes_measured",
                      "dlss_host_color_stage_bytes_measured",
                      "dlss_output_readback_bytes_measured",
                      "hdr_diagnostic_image_readback_bytes_measured",
                      "encoder_host_pixel_bytes_measured")
        image_zero = all(diagnostics.get(key) == 0 for key in image_keys)
        image_zero = image_zero and diagnostics.get("hdr_pixel_readback_exports_bound") == []
        labels = "HDR"
    else:
        count = diagnostics.get("layer_count")
        scales = diagnostics.get("layer_scales")
        dims = diagnostics.get("layer_working_dims")
        sessions = diagnostics.get("layer_session_ids")
        resets = diagnostics.get("layer_reset_flags")
        calls = diagnostics.get("layer_dlss_calls")
        transfer_layers = diagnostics.get("layer_transfer_counters")
        fusion_mode = diagnostics.get("fusion_mode")
        transfer_valid = isinstance(transfer_layers, list) and len(transfer_layers) == expected_count - 1
        per_layer_image_zero = []
        per_layer_calls_valid = []
        if transfer_valid:
            for item in transfer_layers:
                dlss = item.get("dlss") or {}
                per_layer_image_zero.append(
                    dlss.get("host_color_stage_bytes") == 0 and
                    dlss.get("output_readback_bytes") == 0)
        scalar = diagnostics.get("fusion_scalar_readback_bytes_measured")
        image_keys = ("decode_host_copy_bytes_measured",
                      "dlss_host_color_stage_bytes_measured",
                      "dlss_output_readback_bytes_measured",
                      "residual_output_readback_bytes_measured",
                      "encoder_host_pixel_bytes_measured")
        image_zero = all(diagnostics.get(key) == 0 for key in image_keys)
        labels = "SDR"

    shape_ok = (count == expected_count and isinstance(scales, list) and
                len(scales) == expected_count and
                all(abs(a-b) < 1e-7 for a, b in zip(scales, expected_scales)) and
                dims == expected_dims)
    checks[f"{labels} multiscale layer count/scales/dimensions"] = (
        shape_ok,
        f"count={count}/{expected_count}; scales={scales}/{expected_scales}; "
        f"dims={dims}/{expected_dims}")
    expected_fusion = settings.get("fusion_mode", "data_consistency_experimental")
    checks[f"{labels} requested fusion mode reported"] = (
        fusion_mode == expected_fusion,
        f"fusion_mode={fusion_mode}; requested={expected_fusion}")
    ids_ok = (isinstance(sessions, list) and len(sessions) == expected_count and
              all(isinstance(x, int) for x in sessions) and
              len(set(sessions)) == expected_count)
    checks[f"{labels} layers use distinct DLSS sessions"] = (
        ids_ok, f"session_ids={sessions}")
    temporal = diagnostics.get("temporal_control") or {}
    temporal_settings = (temporal.get("controls") or
                         settings.get("temporal_controls") or {})
    observations = temporal.get("observations") or []
    if temporal_settings.get("scene_detection"):
        scene_flags = ([item.get("scene_cut") for item in observations]
                       if isinstance(observations, list) else [])
        expected_resets = ([True] + [bool(flag) for flag in scene_flags[1:]]
                           if len(scene_flags) == frames and
                           all(isinstance(flag, bool) for flag in scene_flags) else None)
    else:
        expected_resets = [True] + [False] * (frames - 1)
    reset_ok = (expected_resets is not None and isinstance(resets, list) and
                len(resets) == expected_count and
                all(isinstance(row, list) and len(row) == frames for row in resets) and
                bool(resets) and all(row == resets[0] for row in resets) and
                resets[0] == expected_resets)
    checks[f"{labels} layers reset consistently across frames"] = (
        reset_ok, f"reset_flags={resets}; expected={expected_resets} from scene observations")
    if case == "hdr":
        reports = diagnostics.get("hdr_frame_reports") or []
        report_pts = []
        report_resets = []
        report_order_ok = isinstance(reports, list) and len(reports) == frames
        for index, item in enumerate(reports if isinstance(reports, list) else []):
            tb = item.get("time_base") or []
            try:
                report_pts.append(Fraction(int(item["pts"])) *
                                  Fraction(int(tb[0]), int(tb[1])))
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                report_order_ok = False
            report_order_ok = report_order_ok and item.get("frame_idx") == index
            report_resets.append(item.get("reset"))
        timestamps_increase = (len(report_pts) == frames and
                               all(b > a for a, b in zip(report_pts, report_pts[1:])))
        checks["HDR frame report timestamps and indices are continuous"] = (
            report_order_ok and timestamps_increase and report_resets == expected_resets,
            f"frame_idx={[r.get('frame_idx') for r in reports] if isinstance(reports,list) else None}; "
            f"pts_seconds={[float(x) for x in report_pts]}; "
            f"resets={report_resets}; expected_resets={expected_resets}")
    calls_ok = (isinstance(calls, list) and len(calls) == expected_count and
                all(value == frames for value in calls))
    checks[f"{labels} every layer processes all requested frames"] = (
        calls_ok and (case != "hdr" or all(per_layer_calls_valid)),
        f"calls={calls}; frames={frames}; extra_layer_calls={per_layer_calls_valid}")
    checks[f"{labels} primary and extra-layer image host boundaries are zero"] = (
        image_zero and transfer_valid and all(per_layer_image_zero),
        f"primary_image_bytes={[diagnostics.get(k) for k in image_keys]}; "
        f"extra_layers={len(transfer_layers) if isinstance(transfer_layers,list) else None}/"
        f"{expected_count-1}; extra_image_zero={per_layer_image_zero}")
    checks[f"{labels} scalar solver readback is separately reported"] = (
        isinstance(scalar, int) and scalar >= 0,
        f"scalar_readback_bytes={scalar}; reported separately from image boundaries")

    if mode != "single" and fusion_mode == "data_consistency_experimental":
        solver_rows = (diagnostics.get("hdr_frame_reports") or [] if case == "hdr"
                       else diagnostics.get("fusion_solver_frames") or [])
        tolerance = float(settings.get("fusion_relative_tolerance", 0.001))
        max_iterations = int(settings.get("fusion_max_iterations", 128))
        solver_ok = isinstance(solver_rows, list) and len(solver_rows) == frames
        for row in solver_rows:
            if case == "hdr":
                solver = (row.get("solver") or {}).get("report") or {}
                iterations = (row.get("solver") or {}).get("iterations")
                converged = (row.get("solver") or {}).get("converged")
                residual = solver.get("relative_residual")
                actual_mode = row.get("fusion_mode_actual")
                solver_ok = solver_ok and actual_mode == "typed_gpu_data_consistency_solve"
            else:
                iterations, converged = row.get("iterations"), row.get("converged")
                residual = row.get("relative_residual")
            solver_ok = (solver_ok and isinstance(iterations, int) and
                         0 <= iterations <= max_iterations and converged is True and
                         isinstance(residual, (float, int)) and math.isfinite(residual) and
                         0 <= residual <= tolerance)
        checks[f"{labels} fusion solver converges within configured limits"] = (
            bool(solver_ok), f"frames={len(solver_rows) if isinstance(solver_rows,list) else None}/"
            f"{frames}; tolerance={tolerance}; max_iterations={max_iterations}")
    return checks


def evaluate_hdr_solver_diagnostics(diagnostics, job_settings, frames, mode):
    if mode == "single":
        return {}
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    settings = job_settings if isinstance(job_settings, dict) else {}
    records = diagnostics.get("hdr_frame_reports") or []
    tolerance = float(settings.get("fusion_relative_tolerance", .001))
    max_iterations = int(settings.get("fusion_max_iterations", 128))
    rows, valid = [], len(records) == frames
    for item in records:
        solver = item.get("solver") or {}
        report = solver.get("report") or {}
        residual = report.get("relative_residual")
        iterations, converged = solver.get("iterations"), solver.get("converged")
        valid = valid and item.get("fusion_mode_actual") == "typed_gpu_data_consistency_solve"
        valid = valid and isinstance(iterations, int) and 0 <= iterations <= max_iterations
        valid = valid and converged is True and isinstance(residual, (int,float)) and \
            math.isfinite(float(residual)) and 0 <= float(residual) <= tolerance
        rows.append({"iterations":iterations,"converged":converged,
                     "relative_residual":residual,"threshold":tolerance})
    return {"HDR multiscale solver frames converge within configured limits":
            (bool(valid), f"frames={len(records)}/{frames}; solver={rows}")}


def _run(argv, timeout=180):
    return subprocess.run([str(x) for x in argv], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout, check=True)


def _probe_json(ffprobe, path, *args):
    return json.loads(_run([ffprobe, "-v", "error", "-of", "json", *args,
                            str(path)]).stdout.decode("utf-8", "replace"))


def probe_artifact(ffprobe, path):
    stream_dump = _probe_json(ffprobe, path, "-show_streams", "-show_data",
                              "-show_format", "-show_chapters")
    packet_dump = _probe_json(ffprobe, path, "-show_packets", "-show_data",
                              "-show_entries", "packet=stream_index,pts,pts_time,dts,dts_time,duration,duration_time,size,data")
    frame_dump = _probe_json(ffprobe, path, "-select_streams", "v:0", "-show_frames")
    streams = stream_dump.get("streams", [])
    videos = [s for s in streams if s.get("codec_type") == "video"]
    if len(videos) != 1:
        raise ValueError(f"expected exactly one video stream in {path}; saw {len(videos)}")
    index_kind = {int(s["index"]): s.get("codec_type") for s in streams}
    video_packets, by_kind = [], defaultdict(list)
    for packet in packet_dump.get("packets", []):
        raw = hexdump_bytes(packet.get("data") or "")
        item = {k: packet.get(k) for k in
                ("stream_index", "pts", "pts_time", "dts", "dts_time",
                 "duration", "duration_time")}
        item.update(complete=len(raw) == int(packet.get("size") or 0),
                    sha256=hashlib.sha256(raw).hexdigest(), kind=
                    index_kind.get(int(packet.get("stream_index", -1))))
        if item["kind"] == "video":
            video_packets.append(item)
        elif item["kind"] in ("audio", "subtitle"):
            by_kind[item["kind"]].append(item)
    attachments = []
    for stream in streams:
        if stream.get("codec_type") == "attachment":
            attachments.append(hashlib.sha256(
                hexdump_bytes(stream.get("extradata") or "")).hexdigest())
    return {"path": str(path), "sha256": sha256_file(path), "video": videos[0],
            "streams": streams, "video_packets": video_packets,
            "packets_by_kind": by_kind,
            "frames": [f for f in frame_dump.get("frames", [])
                       if f.get("media_type", "video") == "video"],
            "chapters": stream_dump.get("chapters", []),
            "format": stream_dump.get("format") or {},
            "attachment_sha256": attachments}


def _unescape_ebsp(data):
    out, i = bytearray(), 0
    while i < len(data):
        if data[i:i+3] == b"\0\0\3":
            out.extend(b"\0\0")
            i += 3
        else:
            out.append(data[i])
            i += 1
    return bytes(out)


def probe_hdr_sei(ffmpeg, path, evidence_dir, label):
    elementary = evidence_dir / f"{label}.hevc"
    _run([ffmpeg, "-v", "error", "-y", "-i", str(path), "-map", "0:v:0",
          "-c:v", "copy", "-f", "hevc", str(elementary)])
    data, starts, cursor, found = elementary.read_bytes(), [], 0, []
    while True:
        i = data.find(b"\0\0\1", cursor)
        if i < 0:
            break
        starts.append(i+3)
        cursor = i+3
    for start, stop in zip(starts, starts[1:] + [len(data)]):
        nal = _unescape_ebsp(data[start:stop])
        if len(nal) < 2 or ((nal[0] >> 1) & 0x3f) != 39:
            continue
        body, offset = nal[2:], 0
        while offset < len(body) and body[offset] not in (255, 128):
            kind, size = body[offset], 0
            offset += 1
            while offset < len(body) and body[offset] == 255:
                size += 255
                offset += 1
            if offset >= len(body):
                break
            size += body[offset]
            offset += 1
            payload = body[offset:offset+size]
            offset += size
            if kind in (137, 144):
                found.append({"payload_type": kind, "bytes": payload.hex()})
    return found


def default_settings(case, mode):
    scales = {"single": [.5], "dual": [.5,.75], "triple": [.4,.67,.85]}[mode]
    job = {"preset": 1, "style": 0, "intensity": 1.0, "local_tone": 1.0,
           "local_struct": 1.0, "skin_struct": 1.0, "use_auto_mask": 1,
           "ui_correction": 0, "motion_scale_x": 1.0, "motion_scale_y": 1.0,
           "guidance_mode": 1, "flow_engine": "nvof", "flow_quality": "4M",
           "working_scale": scales[0], "residual_pyramid": True,
           "residual_strength": 1.0, "multi_pass": 1, "dlss_backend": "gpu_flow_d3d12",
           "export_backend": "gpu_transfer", "reconstruction_mode": mode}
    if mode != "single":
        job.update(layer_scales=scales, layer_weights=[1.0]*len(scales),
                   fusion_mode="data_consistency_experimental", fusion_beta=.1,
                   fusion_relative_tolerance=.001, fusion_max_iterations=128)
    if case == "hdr":
        job.update(experimental_hdr_scrgb=True, hdr_paper_white_scrgb=8.0)
        ts = {"codec":"hevc_nvenc", "preset":"p7", "container":"mkv",
              "rate_mode":"cq", "aq_mode":"spatial", "gop_mode":"auto",
              "decode_mode":"cuda", "profile":"main10", "pixel_format":"p010le",
              "color_preset":"preserve", "resolution_mode":"original",
              "custom_color_space":"auto", "custom_color_primaries":"auto",
              "custom_color_transfer":"auto", "custom_color_range":"auto",
              "audio_mode":"copy", "subtitle_mode":"copy", "copy_attachments":True,
              "copy_metadata":True, "copy_chapters":True, "audio_bitrate":"320k",
              "extra_args":""}
    else:
        ts = {"codec":"hevc_nvenc", "preset":"p7", "container":"mkv",
              "decode_mode":"cuda", "audio_mode":"copy", "subtitle_mode":"copy",
              "copy_attachments":True, "copy_metadata":True, "copy_chapters":True,
              "color_preset":"preserve", "extra_args":""}
    return job, ts


def _overrides(path):
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"job_settings", "transcode_settings", "frames", "cq", "with_audio"}
    if not isinstance(data, dict) or set(data)-allowed:
        raise ValueError(f"settings JSON keys must be from {sorted(allowed)}")
    return data


def _close_owned_runner_resources(live, service, error=None):
    """Tear down the harness-owned DLSS session before its guidance engine."""
    for owner, label in ((live, "session"), (service, "frame guidance service")):
        if owner is None:
            continue
        try:
            owner.close()
        except Exception as exc:
            error = error or f"{label} close failed: {exc}"
    return error


def run_hardware(case, mode, fixture, evidence_dir, overrides,
                 fusion_dll=None, compare_video_evidence=None):
    sys.path.insert(0, str(ROOT))
    import dlss_gpu_engine
    import gpu_transfer_export as gtx
    import optical_flow
    import pipeline
    import bundle_paths

    ffmpeg, ffprobe = pipeline._bundle_ffmpeg(), pipeline._bundle_ffprobe()
    if not ffmpeg or not Path(ffmpeg).is_file():
        ffmpeg = shutil.which("ffmpeg") or ""
    if not ffprobe or not Path(ffprobe).is_file():
        ffprobe = shutil.which("ffprobe") or ""
    if not Path(ffmpeg).is_file() or not Path(ffprobe).is_file():
        raise FileNotFoundError("FFmpeg/ffprobe unavailable")
    job, ts = default_settings(case, mode)
    job.update(overrides.get("job_settings", {}))
    ts.update(overrides.get("transcode_settings", {}))
    job["reconstruction_mode"] = mode
    if mode == "single":
        scales = [float(job.get("working_scale", .5))]
    else:
        scales = list(job.get("layer_scales", job["layer_scales"]))
        if len(scales) != {"dual":2,"triple":3}[mode]:
            raise ValueError(f"wrong number of scales for {mode}: {scales}")
        job["layer_scales"], job["working_scale"] = scales, scales[0]
    job["export_backend"] = "gpu_transfer"
    if case == "hdr":
        job["experimental_hdr_scrgb"] = True
    else:
        job.pop("experimental_hdr_scrgb", None)
    frames, cq = int(overrides.get("frames", 3)), int(overrides.get("cq", 24))
    with_audio = bool(overrides.get("with_audio", True))
    count, fps, width, height = pipeline.video_info(str(fixture))
    if not 0 < frames <= count:
        raise ValueError(f"requested {frames} frames from {count}-frame fixture")
    source = probe_artifact(ffprobe, fixture)
    hdr_source = source["video"].get("color_transfer") in ("smpte2084", "arib-std-b67")
    if hdr_source != (case == "hdr"):
        raise ValueError(f"case={case} conflicts with color_transfer="
                         f"{source['video'].get('color_transfer')!r}")
    dims = [pipeline.resolve_working_dims(width, height, float(scale)) for scale in scales]
    live = dlss_gpu_engine.Live(*dims[0], job,
                                color_format="rgba16f" if case == "hdr" else "rgba8")
    # This acceptance run owns a dedicated service. The app-level getter returns a
    # process-wide singleton; closing that here would leave its cache pointing at a
    # closed engine for any later in-process run.
    service = optical_flow.FrameGuidanceService(
        job["flow_engine"], job["flow_quality"], dedicated=True)
    provider = pipeline.GuidanceServiceFlowProvider(service,
        guidance_mode=int(job.get("guidance_mode", 1)))
    output, intermediate = evidence_dir/"final.mkv", evidence_dir/"intermediate_video_only.mkv"
    diagnostics, progress = {}, []
    mux_record = {"argv": None, "intermediate_sha256": None, "intermediate_bytes": None}
    original = gtx.run_copy_pass
    original_loader = gtx._load
    fusion_path = Path(fusion_dll).resolve() if fusion_dll else None
    if fusion_path and not fusion_path.is_file():
        raise FileNotFoundError(f"fusion DLL override not found: {fusion_path}")
    fusion_hash = sha256_file(fusion_path) if fusion_path else None
    dll_directory_cookie = None
    if fusion_path:
        dll_directory_cookie = bundle_paths.add_dll_directory(gtx.NATIVE_DIR)

        def load_with_fusion_override(name):
            if name == "gpu_multiscale_fusion_bridge":
                return ctypes.CDLL(str(fusion_path))
            return original_loader(name)

        gtx._load = load_with_fusion_override

    def capture(ffmpeg_arg, encoded, original_source, transcode, out, with_audio=True, **kwargs):
        shutil.copy2(encoded, intermediate)
        mux_record["intermediate_sha256"] = sha256_file(intermediate)
        mux_record["intermediate_bytes"] = intermediate.stat().st_size
        command = original(ffmpeg_arg, encoded, original_source, transcode, out,
                           with_audio, **kwargs)
        mux_record["argv"] = [str(x) for x in command]
        return command

    gtx.run_copy_pass = capture
    started = time.time()
    error, result = None, None
    try:
        result = pipeline.export_gpu_transfer_stream(
            video=str(fixture), live=live, frames=frames, fps=fps, cq=cq,
            with_audio=with_audio, out_path=str(output), transcode_settings=ts,
            job_settings=job, flow_provider=provider,
            progress=lambda current,total,message: progress.append(current),
            diagnostics_out=diagnostics)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        gtx.run_copy_pass = original
        gtx._load = original_loader
        if dll_directory_cookie is not None:
            try:
                dll_directory_cookie.close()
            except Exception:
                pass
        error = _close_owned_runner_resources(live, service, error)
    run = {"started_unix": started, "elapsed_seconds": time.time()-started,
           "result": result, "error": error, "progress": progress,
           "output_exists": output.is_file(), "mux": mux_record}
    module_hashes = {}
    for name in ("pipeline", "gpu_transfer_export", "gpu_hdr_transfer_export",
                 "dlss_gpu_engine", "hdr_gpu_engine", "hdr_fusion_engine",
                 "optical_flow", "post_control", "temporal_control"):
        path = ROOT/f"{name}.py"
        if path.is_file():
            module_hashes[name] = {"path":str(path),"sha256":sha256_file(path)}
    library_names = {"nvdec_bridge", "nvenc_bridge", "nv12_rgba_bridge", "dlss_gpu_host"}
    if int(job.get("guidance_mode", 1)) == 1:
        library_names.update(("nvof_bridge", "nvof_d3d12_interop"))
    if any(float(scale) < 1.0 for scale in scales):
        library_names.update(("gpu_lanczos_bridge", "gpu_residual_bridge"))
    if mode != "single":
        library_names.update(("gpu_low_residual_bridge", "gpu_multiscale_fusion_bridge"))
    if case == "hdr":
        library_names.add("gpu_hdr_scrgb_bridge")
    if job.get("post_controls"):
        library_names.add("gpu_post_control_bridge")
    temporal_settings = job.get("temporal_controls") or {}
    if temporal_settings.get("scene_detection") or temporal_settings.get("mode", "off") != "off":
        library_names.add("gpu_temporal_bridge")
    library_hashes = {}
    for name in sorted(library_names):
        path = Path(gtx.NATIVE_DIR)/(name+".dll")
        if path.is_file():
            library_hashes[name] = {"path":str(path),"sha256":sha256_file(path)}
    if fusion_path:
        library_hashes["gpu_multiscale_fusion_bridge"] = {
            "path":str(fusion_path),"sha256":fusion_hash,"override":True}
    record = {"case": case, "mode": mode, "frames": frames, "cq": cq,
              "with_audio": with_audio, "fixture": str(fixture),
              "fixture_sha256": sha256_file(fixture), "ffmpeg": ffmpeg,
              "ffprobe": ffprobe, "job_settings": job, "transcode_settings": ts,
              "source_probe": source, "working_dims": dims, "run": run,
              "diagnostics": diagnostics, "python_modules":module_hashes,
              "native_library_inputs":library_hashes,
              "performance": "NOT_CHARACTERIZED"}
    record["fusion_dll_override"] = ({"path": str(fusion_path),
                                      "sha256": fusion_hash} if fusion_path else None)
    record["compare_video_evidence"] = (str(Path(compare_video_evidence).resolve())
                                        if compare_video_evidence else None)
    (evidence_dir/"settings_and_inputs.json").write_text(
        json.dumps(record, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    (evidence_dir/"diagnostics.json").write_text(
        json.dumps({"run":run,"diagnostics":diagnostics}, indent=2,
                   default=str, ensure_ascii=False), encoding="utf-8")
    if error or not output.is_file() or not intermediate.is_file():
        record["verdict"] = "export failed or no copy-pass intermediate captured"
        (evidence_dir/"gate_log.json").write_text(
            json.dumps(record, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        print(f"[gate] {record['verdict']}: {error}")
        return 1
    return offline_recheck(evidence_dir, record)


def offline_recheck(evidence_dir, record=None):
    if record is None:
        record = json.loads((evidence_dir/"settings_and_inputs.json").read_text(encoding="utf-8"))
        record["diagnostics"] = json.loads((evidence_dir/"diagnostics.json").read_text(
            encoding="utf-8")).get("diagnostics", {})
    fixture = Path(record["fixture"])
    intermediate, final = evidence_dir/"intermediate_video_only.mkv", evidence_dir/"final.mkv"
    source, inter, out = (probe_artifact(record["ffprobe"], p)
                          for p in (fixture, intermediate, final))
    seis = {}
    if record["case"] == "hdr":
        seis = {label: probe_hdr_sei(record["ffmpeg"], path, evidence_dir, label)
                for label, path in (("source",fixture),("intermediate",intermediate),
                                    ("final",final))}
    checks = evaluate_artifacts(source, inter, out, int(record["frames"]),
                                record["case"], record.get("run",{}).get("mux",{}).get("argv"),
                                seis.get("source"), seis.get("intermediate"), seis.get("final"),
                                record.get("transcode_settings"),
                                bool(record.get("with_audio", True)))
    control_checks = evaluate_control_diagnostics(record.get("diagnostics"),
                                                  record.get("job_settings"),
                                                  int(record["frames"]))
    checks.update(control_checks)
    if record["mode"] != "single":
        checks.update(evaluate_multiscale_diagnostics(
            record.get("diagnostics"), record.get("job_settings"),
            int(record["frames"]), record["case"], record["mode"],
            record.get("working_dims")))
    checks.update(evaluate_hdr_solver_diagnostics(record.get("diagnostics"),
        record.get("job_settings"), int(record["frames"]), record["mode"])
        if record["case"] == "hdr" else {})
    baseline_comparison = None
    compare_dir = record.get("compare_video_evidence")
    if compare_dir:
        base_path = Path(compare_dir) / "gate_log.json"
        baseline = json.loads(base_path.read_text(encoding="utf-8"))
        baseline_facts = baseline.get("offline_recheck", {}).get("artifact_facts", {})
        base_video = baseline_facts.get("final", {}).get("video_packets", [])
        current_hashes = [p.get("sha256") for p in out.get("video_packets", [])]
        baseline_hashes = [p.get("sha256") for p in base_video]
        baseline_diag = baseline.get("diagnostics") or {}
        current_diag = record.get("diagnostics") or {}
        def solver_rows(diagnostics):
            result = []
            for item in diagnostics.get("hdr_frame_reports", []) if isinstance(diagnostics, dict) else []:
                solver = item.get("solver") or item
                report = solver.get("report") or {}
                result.append({"iterations": solver.get("iterations",
                                item.get("hdr_solver_iterations")),
                               "converged": solver.get("converged",
                                item.get("hdr_solver_converged")),
                               "relative_residual": report.get("relative_residual",
                                   solver.get("relative_residual")),
                               "initial_residual_norm": report.get("initial_residual_norm"),
                               "final_residual_norm": report.get("final_residual_norm")})
            return result
        old_solver, new_solver = solver_rows(baseline_diag), solver_rows(current_diag)
        setup_fields = ("case", "mode", "frames", "cq", "with_audio", "working_dims",
                        "fixture_sha256", "job_settings", "transcode_settings")
        setup_differences = {key: {"other": json_shape(baseline.get(key)),
                                   "this": json_shape(record.get(key))}
                             for key in setup_fields
                             if json_shape(baseline.get(key)) != json_shape(record.get(key))}
        same_setup = not setup_differences
        decoded_comparison = None
        if record.get("case") == "hdr":
            decoded_comparison = compare_decoded_hdr_video(
                record["ffmpeg"], Path(compare_dir)/"final.mkv",
                evidence_dir/"final.mkv", evidence_dir,
                baseline_facts.get("final", {}), out, int(record["frames"]))
        solver_pairs = list(zip(old_solver, new_solver))
        solver_deltas = [abs(int(old["iterations"])-int(new["iterations"]))
                         for old,new in solver_pairs
                         if isinstance(old.get("iterations"), int) and
                            isinstance(new.get("iterations"), int)]
        solvers_comparable = (
            len(old_solver) == len(new_solver) == int(record.get("frames", 0)) and
            len(solver_deltas) == len(solver_pairs) and
            all(old.get("converged") is True and new.get("converged") is True and
                old.get("converged") == new.get("converged")
                for old,new in solver_pairs) and
            all(delta <= 2 for delta in solver_deltas))
        baseline_comparison = {
            "baseline_evidence": str(compare_dir), "same_case_mode_frames_fixture_settings": same_setup,
            "setup_differences": setup_differences,
            "final_video_packet_payloads_equal": current_hashes == baseline_hashes,
            "current_packet_count": len(current_hashes),
            "baseline_packet_count": len(baseline_hashes),
            "current_solver_frames": new_solver, "baseline_solver_frames": old_solver,
            "solver_iteration_deltas": solver_deltas,
            "solver_stats_comparable": solvers_comparable,
            "decoded_frame_comparison": decoded_comparison,
            "bitexact_picture_match_required": False}
        checks["baseline comparison uses matching fixture/settings"] = (
            same_setup, json.dumps({"same_setup":same_setup,
                                   "differences":setup_differences}, default=str))
        if record.get("case") == "hdr":
            checks["baseline decoded-frame difference stays within diagnostic bounds"] = (
                same_setup and isinstance(decoded_comparison, dict) and
                decoded_comparison.get("within_diagnostic_bounds") is True,
                json.dumps(decoded_comparison, default=str))
            checks["baseline solver convergence and iteration counts are comparable"] = (
                same_setup and solvers_comparable,
                json.dumps({"iterations_delta":solver_deltas,
                            "comparable":solvers_comparable,
                            "baseline":old_solver,"current":new_solver}, default=str))
    diag = record.get("diagnostics") or {}
    diagnostics = {k:v for k,v in diag.items()
                   if "post_control" in k or "temporal_control" in k} if isinstance(diag,dict) else {}
    record["offline_recheck"] = {"export_driven":False,
                                 "post_temporal_diagnostics":diagnostics,
                                 "control_diagnostic_checks":control_checks,
                                 "artifact_facts": {"source":source,"intermediate":inter,"final":out},
                                 "sei":seis,
                                 "baseline_comparison":baseline_comparison}
    record["checks"] = {k:{"passed":bool(v[0]),"detail":v[1]}
                        for k,v in checks.items()}
    failed = [k for k,v in checks.items() if not v[0]]
    record["verdict"] = "all checks passed" if not failed else f"{len(failed)} of {len(checks)} failed"
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (evidence_dir/"gate_log.json").write_text(
        json.dumps(record, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    for name,(passed,detail) in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(f"[gate] {len(checks)-len(failed)}/{len(checks)} checks; "
          f"{'PASS' if not failed else 'FAIL'}; performance NOT_CHARACTERIZED")
    return int(bool(failed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("sdr","hdr"))
    parser.add_argument("mode", choices=("single","dual","triple"))
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--settings-json", type=Path)
    parser.add_argument("--fusion-dll", type=Path,
                        help="load this fusion DLL for this process without replacing native DLLs")
    parser.add_argument("--compare-video-evidence", type=Path,
                        help="prior evidence directory; records payload equality and solver stats")
    parser.add_argument("--recheck", action="store_true")
    args = parser.parse_args(argv)
    out = args.evidence_dir.resolve()
    bench = (ROOT/"bench_out").resolve()
    if not out.is_relative_to(bench):
        raise SystemExit(f"evidence directory must be inside {bench}")
    if args.recheck:
        return offline_recheck(out)
    if out.exists():
        raise SystemExit("evidence directory must be fresh; use --recheck to inspect existing evidence")
    if not args.fixture.is_file():
        raise SystemExit(f"fixture not found: {args.fixture}")
    overrides = {}
    if args.settings_json:
        overrides = json.loads(args.settings_json.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict) or set(overrides)-{
            "job_settings","transcode_settings","frames","cq","with_audio"}:
            raise ValueError("settings JSON has unsupported root keys")
    out.mkdir(parents=True)
    return run_hardware(args.case,args.mode,args.fixture.resolve(),out,overrides,
                        args.fusion_dll, args.compare_video_evidence)


if __name__ == "__main__":
    raise SystemExit(main())
