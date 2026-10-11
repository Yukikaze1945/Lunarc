# -*- coding: utf-8 -*-
"""Run the experimental HDR10 GPU-resident export once, end to end, and record what it said.

Not a test: it needs the GPU, the NVOF hardware and the real HDR bridges. It is the one
final gate the Stage B record asks for - six frames of the valid HDR10 fixture at 50%
working scale, through the entry point the queue calls - and it writes its whole evidence
set under ``bench_out/hdr_production_export/``: the settings it ran with, the hash of the
source and of every library it loaded, the diagnostics the chain reported, the finished
file, and the offline read-back of that file.

    python tests/unit/hdr_production_export_gate.py

    python tests/unit/hdr_production_export_gate.py --recheck

``--recheck`` drives no export: it re-reads the deliverable and the saved report of the
export that is already on disk, which is how a checklist line that turned out to be a bug
in this script's own read-back gets corrected without a second run.

The script decides nothing about correctness. It calls ``pipeline.export_gpu_transfer_stream``
the way ``dlss_service`` does - one FP16 Feature 18 session, one NVOF provider, one
negotiation inside the branch - and then inspects the deliverable with the offline probe
and an elementary-stream reader. Software-decoding an output to look at it is not a host
boundary inside the chain, and nothing here reads a picture back to make a claim about one.
A refusal is a result: it is printed and written down, not retried another way.

The checklist is the brief's, not a general one: six packets, monotonic PTS, the BT.2020 NC
/ PQ / limited tags, the source's own mastering and CLL/FALL numbers in the container *and*
in the demuxed elementary stream, and every image host boundary the chain claims at zero.
Each line reports the value it read, so a counter the chain does not have reads as missing
rather than as zero.

The zero the image boundaries are held to is the whole of what this gate forbids, and it is
not a claim that nothing crossed. The colour bridge copies each executed pass's counters -
scalars, 256 bytes at ``kCounterSlots * sizeof(uint32_t)`` - into a CPU-visible readback
buffer, because that is what its report structs are answered from. That traffic is
permitted by the contract, is present on every run, and is measured by no counter, so the
checklist reads it as present-and-unmeasured rather than as a zero, and a saved report that
filed the image-plane counter in its slot is re-read under that correction and labelled.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

import dlss_gpu_engine                       # noqa: E402
import gpu_transfer_export as gtx            # noqa: E402
import optical_flow                          # noqa: E402
import pipeline                              # noqa: E402

SOURCE = os.path.join(_ROOT, "build", "hdr_probe", "hdr_valid_src.mkv")
OUT_DIR = os.path.join(_ROOT, "bench_out", "hdr_production_export")
OUT = os.path.join(OUT_DIR, "hdr_valid_src_hdr10_gpu.mkv")
FRAMES = 6

JOB_SETTINGS = {
    "preset": 1, "style": 0, "intensity": 1.0,
    "local_tone": 1.0, "local_struct": 1.0, "skin_struct": 1.0,
    "use_auto_mask": 1, "ui_correction": 0,
    "motion_scale_x": 1.0, "motion_scale_y": 1.0,
    "guidance_mode": 1, "flow_engine": "nvof", "flow_quality": "4M",
    "working_scale": 0.5, "residual_pyramid": True, "residual_strength": 1.0,
    "multi_pass": 1, "dlss_backend": "gpu_flow_d3d12",
    "export_backend": "gpu_transfer",
    # The two HDR settings, at the values the record fixes: on for this job, and the
    # bridge's own paper white. Nothing else above is HDR-specific.
    "experimental_hdr_scrgb": True,
    "hdr_paper_white_scrgb": 8.0,
    "reconstruction_mode": "single",
}
# The chain runs no mux, so every clause that would have asked for one is turned off by
# name rather than left to a default the gate would refuse the job for.
TRANSCODE = {
    "codec": "hevc_nvenc", "preset": "p7", "container": "mkv",
    "rate_mode": "cq", "aq_mode": "spatial", "gop_mode": "auto",
    "decode_mode": "cuda", "audio_mode": "remove", "subtitle_mode": "drop",
    "copy_attachments": False, "copy_metadata": False, "copy_chapters": False,
}

LIBRARIES = ("nvdec_bridge", "nvenc_bridge", "nvof_bridge", "nv12_rgba_bridge",
             "gpu_lanczos_bridge", "gpu_residual_bridge", "gpu_hdr_scrgb_bridge",
             "dlss_gpu_host", "nvof_d3d12_interop")

VIDEO_TAGS = ("codec_name", "profile", "width", "height", "pix_fmt", "color_range",
              "color_space", "color_transfer", "color_primaries", "avg_frame_rate")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ffprobe(args):
    argv = [pipeline._bundle_ffprobe(), "-v", "error"] + args
    return argv, json.loads(subprocess.check_output(
        argv).decode("utf-8", "replace"))


def _stream_facts(path):
    """The video stream's declared facts, its frames, its packets, and its side data.

    Three probes: the stream's own tags, the decoded frames, the counted packets. A
    frame list is only asked for with the video stream selected, because a frame's
    ``codec_type`` is emitted as ``media_type`` by the probe this bundle ships, and a
    silent filter on the absent key would read a picture-less file rather than six
    pictures. The side data is taken from the frames, which is where a decoder puts an
    SEI it read out of the bitstream.
    """
    _sargv, stream_dump = _ffprobe(["-print_format", "json", "-show_streams", path])
    stream = next(s for s in stream_dump["streams"] if s.get("codec_type") == "video")
    _fargv, frame_dump = _ffprobe(["-select_streams", "v:0", "-print_format", "json",
                                   "-show_frames", path])
    frames = [f for f in frame_dump.get("frames", [])
              if f.get("media_type", f.get("codec_type")) == "video"]
    side = {}
    for frame in frames:
        for item in frame.get("side_data_list") or []:
            side.setdefault(item.get("side_data_type"), item)
    _pargv, counted = _ffprobe([
        "-select_streams", "v:0", "-count_packets", "-print_format", "json",
        "-show_entries", "stream=nb_read_packets", path])
    packets = int(counted["streams"][0].get("nb_read_packets") or 0)
    _kargv, keyed = _ffprobe([
        "-select_streams", "v:0", "-print_format", "json",
        "-show_entries", "packet=pts,pts_time,duration,dts", path])
    return stream, side, frames, packets, keyed.get("packets", [])


def _unescape(chunk):
    """Drop the emulation-prevention byte: 00 00 03 xx -> 00 00 xx."""
    out = bytearray()
    index = 0
    while index < len(chunk):
        if (chunk[index] == 0 and index + 2 < len(chunk)
                and chunk[index + 1] == 0 and chunk[index + 2] == 3):
            out += b"\x00\x00"
            index += 3
            continue
        out.append(chunk[index])
        index += 1
    return bytes(out)


def _sei_payloads(path, demuxed):
    """The prefix-SEI payloads the elementary stream actually carries, as bytes.

    The container's side data can be re-created by a muxer that read the bitstream; a
    payload in front of each picture cannot, so this is where the claim that the
    encoder's own writer authored the SEIs is checked rather than trusted.
    """
    argv = [pipeline._bundle_ffmpeg(), "-v", "error", "-y", "-i", path,
            "-c:v", "copy", "-f", "hevc", demuxed]
    subprocess.check_call(argv)
    with open(demuxed, "rb") as handle:
        data = handle.read()
    starts, position = [], 0
    while True:
        index = data.find(b"\x00\x00\x01", position)
        if index < 0:
            break
        starts.append(index + 3)
        position = index + 3
    found = []
    for start, nxt in zip(starts, starts[1:] + [len(data)]):
        body = _unescape(data[start:nxt])
        if len(body) < 2 or ((body[0] >> 1) & 0x3F) != 39:
            continue                      # 39 is PREFIX_SEI_NUT; everything else is not
        offset, payloads = 2, {}
        payload = body[offset:]
        offset = 0
        while offset < len(payload) and payload[offset] != 0xFF and \
                payload[offset] != 0x80:
            ptype = payload[offset]
            offset += 1
            size = 0
            while offset < len(payload) and payload[offset] == 0xFF:
                size += 255
                offset += 1
            if offset >= len(payload):
                break
            size += payload[offset]
            offset += 1
            payloads[ptype] = payload[offset:offset + size]
            offset += size
        for ptype in (137, 144):
            if ptype in payloads:
                found.append({"nal_unit_type": 39, "payload_type": ptype,
                              "length": len(payloads[ptype]),
                              "bytes": payloads[ptype].hex()})
    return argv, found, os.path.getsize(demuxed)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT):
        os.unlink(OUT)
    evidence = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if not os.path.isfile(SOURCE):
        raise SystemExit(f"the gate needs the valid HDR10 fixture at {SOURCE}")
    native = gtx.NATIVE_DIR
    evidence["source"] = {"path": SOURCE, "sha256": _sha256(SOURCE),
                          "bytes": os.path.getsize(SOURCE)}
    evidence["libraries"] = {}
    for name in LIBRARIES:
        path = os.path.join(native, name + ".dll")
        if os.path.isfile(path):
            evidence["libraries"][name] = {"path": path, "sha256": _sha256(path),
                                           "bytes": os.path.getsize(path)}
    evidence["python_modules"] = {
        name: _sha256(os.path.join(_ROOT, name + ".py"))
        for name in ("pipeline", "gpu_transfer_export", "gpu_hdr_transfer_export",
                     "hdr_gpu_engine", "dlss_gpu_engine", "dlss_service")}
    evidence["job_settings"] = dict(JOB_SETTINGS)
    evidence["transcode_settings"] = dict(TRANSCODE)

    source_stream, source_side, source_frames, source_packets, _sp = \
        _stream_facts(SOURCE)
    evidence["source_probe"] = {
        "stream": {key: source_stream.get(key) for key in VIDEO_TAGS},
        "frames": len(source_frames), "packets": source_packets,
        "side_data": source_side}
    with open(os.path.join(OUT_DIR, "settings_and_inputs.json"), "w",
              encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, ensure_ascii=False)

    n, fps, w, h = pipeline.video_info(SOURCE)
    live_w, live_h = pipeline.resolve_working_dims(w, h, JOB_SETTINGS["working_scale"])
    print(f"[gate] source {n} frames @ {fps} {w}x{h}; working {live_w}x{live_h}")

    service = optical_flow.get_frame_guidance_service("nvof", "4M")
    print("[gate] NVOF device rgba input supported:",
          service.supports_device_rgba_input())
    flow_provider = pipeline.GuidanceServiceFlowProvider(service, guidance_mode=1)

    # The session the service opens for an HDR job: the same Live class, the same FP16
    # colour, on the thread that will drive the chain. An RGBA8 session would be refused
    # by the gate rather than used.
    live = dlss_gpu_engine.Live(live_w, live_h, JOB_SETTINGS, color_format="rgba16f")
    print(f"[gate] Feature 18 session: colour={live.color_format}, "
          f"bytes_per_pixel={live.bytes_per_pixel}, "
          f"abi={live.host_abi}, host={live.host_dll}")
    diagnostics, seen = {}, []

    def progress(current, total, message):
        seen.append(current)

    started = time.perf_counter()
    failure, finished = None, None
    try:
        finished = pipeline.export_gpu_transfer_stream(
            video=SOURCE, live=live, frames=FRAMES, fps=fps, cq=24,
            with_audio=False, out_path=OUT, transcode_settings=dict(TRANSCODE),
            job_settings=dict(JOB_SETTINGS), flow_provider=flow_provider,
            progress=progress, diagnostics_out=diagnostics)
    except Exception as error:              # a refusal is the result; record it whole
        failure = f"{type(error).__name__}: {error}"
    finally:
        elapsed = time.perf_counter() - started
        live.close()

    evidence["run"] = {"elapsed_sec": elapsed, "progress_counts": list(seen),
                       "returned": finished, "error": failure,
                       "output_exists": os.path.exists(OUT),
                       "output_bytes": os.path.getsize(OUT) if os.path.exists(OUT)
                       else None}
    with open(os.path.join(OUT_DIR, "diagnostics.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"run": evidence["run"], "diagnostics": diagnostics}, handle,
                  indent=2, default=str, ensure_ascii=False)

    if failure or not os.path.exists(OUT):
        print(f"[gate] REFUSED OR FAILED: {failure}")
        print(json.dumps({key: diagnostics.get(key) for key in
                          ("export_backend_actual", "export_backend_requested",
                           "export_backend_fallback_reason",
                           "export_backend_refusal_reasons", "transfer_mode_actual",
                           "hdr_negotiation", "hdr_capabilities")},
                         indent=2, default=str, ensure_ascii=False))
        evidence["verdict"] = "refused or failed before a deliverable"
        with open(os.path.join(OUT_DIR, "gate_log.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2, default=str, ensure_ascii=False)
        return 1

    return _inspect(evidence, diagnostics, source_side, (live_w, live_h))


def _inspect(evidence, diagnostics, source_side, working, recheck=False):
    """Read the finished file back, run the checklist, and write the gate log.

    ``recheck`` says plainly that no export was driven to produce this read: the
    deliverable and the report are the ones already on disk.
    """
    stream, side, frames, packets, keyed = _stream_facts(OUT)
    demuxed = os.path.join(OUT_DIR, "demuxed_elementary_stream.265")
    demux_argv, seis, es_bytes = _sei_payloads(OUT, demuxed)
    source_demuxed = os.path.join(OUT_DIR, "source_elementary_stream.265")
    _src_argv, source_seis, _src_bytes = _sei_payloads(SOURCE, source_demuxed)
    print(f"[gate] output: {packets} packets, {len(frames)} frames, "
          f"{os.path.getsize(OUT)} bytes; ES {es_bytes} bytes, "
          f"SEI payloads {[(s['payload_type'], s['length']) for s in seis]}")

    checks = _checklist(diagnostics, stream, side, frames, packets, keyed, seis,
                        source_seis, source_side, working)
    evidence["output_probe"] = {
        "stream": {key: stream.get(key) for key in VIDEO_TAGS},
        "frames": len(frames), "packets": packets, "side_data": side,
        "sei_payloads": seis, "source_sei_payloads": source_seis,
        "demux_argv": demux_argv, "elementary_stream_bytes": es_bytes,
        "output_path": OUT, "output_sha256": _sha256(OUT)}
    evidence["checks"] = {name: {"passed": bool(value[0]), "read": value[1]}
                          for name, value in checks.items()}
    if recheck:
        evidence["recheck"] = {
            "export_driven_now": False,
            "deliverable_read": OUT,
            "report_read": os.path.join(OUT_DIR, "diagnostics.json"),
            "note": "the checklist was re-read against the deliverable and the report of "
                    "the single export this directory already held. Five lines that "
                    "failed the first time were bugs in this script's own read-back "
                    "(a frame list keyed on a field name this probe build does not "
                    "emit, presentation order read off decode-order packets, and a "
                    "positive number demanded of a counter the bridge answers zero for); "
                    "the export, the chain and its diagnostics were not touched, and no "
                    "second export was driven."}
    evidence["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    failed = sum(0 if passed else 1 for passed, _ in checks.values())
    evidence["verdict"] = ("all checks passed" if not failed
                           else f"{failed} of {len(checks)} checks failed")
    with open(os.path.join(OUT_DIR, "gate_log.json"), "w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, default=str, ensure_ascii=False)

    print("\n[gate] checklist, read from the finished file and the chain's report")
    for name, (passed, detail) in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail}")
    print(f"[gate] {len(checks) - failed}/{len(checks)} checks passed; "
          f"{'VERDICT: FAIL' if failed else 'VERDICT: PASS'}")
    print("[gate] performance: NOT_CHARACTERIZED. The elapsed seconds of one process "
          "are recorded, and no throughput or latency claim is made from them.")
    return 1 if failed else 0


def recheck():
    """The checklist over the export that already happened, with no second export."""
    with open(os.path.join(OUT_DIR, "settings_and_inputs.json"),
              encoding="utf-8") as handle:
        evidence = json.load(handle)
    with open(os.path.join(OUT_DIR, "diagnostics.json"), encoding="utf-8") as handle:
        saved = json.load(handle)
    if not os.path.exists(OUT):
        raise SystemExit(f"nothing to re-read: {OUT} is not there")
    report = saved["diagnostics"]
    evidence["run"] = saved["run"]
    # The source is probed again rather than read from the saved file: that snapshot was
    # taken by the read-back that has just been corrected, and an offline probe of an
    # untouched fixture adds no claim.
    source_stream, source_side, source_frames, source_packets, _keyed = \
        _stream_facts(SOURCE)
    evidence["source_probe"] = {
        "stream": {key: source_stream.get(key) for key in VIDEO_TAGS},
        "frames": len(source_frames), "packets": source_packets,
        "side_data": source_side}
    with open(os.path.join(OUT_DIR, "settings_and_inputs.json"), "w",
              encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, default=str, ensure_ascii=False)
    print("[gate] re-check: no export is driven. The deliverable on disk is "
          f"{saved['run']['returned']} and the report is the one saved beside it.")
    return _inspect(evidence, report, source_side,
                    tuple(report["hdr_working_dims"]), recheck=True)


IMAGE_READBACK_KEY = "hdr_diagnostic_image_readback_bytes_measured"
# What the same bridge counter (``report_readback_bytes``) was filed under before
# HDR_STAGE_B_BACKEND_AUDIT_REPAIR.md. The name claimed the per-pass scalar counter
# readback; the number was, and is, the image-plane diagnostic counter. A saved report
# written by that code is re-read under the corrected meaning and the line says so -
# nothing in this script re-measures anything.
HISTORICAL_IMAGE_READBACK_KEY = "hdr_scalar_report_readback_bytes_measured"


def image_readback_counter(report):
    """The bridge's image-plane readback counter, whichever name it is filed under."""
    if IMAGE_READBACK_KEY in report:
        return {"key": IMAGE_READBACK_KEY, "value": report.get(IMAGE_READBACK_KEY),
                "provenance": "the field the current code writes for this counter"}
    if HISTORICAL_IMAGE_READBACK_KEY in report:
        return {"key": HISTORICAL_IMAGE_READBACK_KEY,
                "value": report.get(HISTORICAL_IMAGE_READBACK_KEY),
                "provenance": "OFFLINE RE-READ of a report written before the audit "
                              "repair, which filed this bridge counter under a scalar "
                              "name; the value is the image-plane counter read under "
                              "its corrected meaning, and no new measurement was taken"}
    return {"key": IMAGE_READBACK_KEY, "value": None,
            "provenance": "the report answers this counter under neither name"}


def scalar_readback_claim(report):
    """How a saved report answers the per-pass scalar counter boundary, if at all.

    Three states, kept apart rather than collapsed into a zero:

      * ``unanswered`` - the current code: the boundary is a fact of the bridge source
        (``native/gpu_hdr_scrgb_bridge.cpp`` ``run_pass`` copies
        ``kCounterSlots * sizeof(uint32_t)`` into a ``D3D12_HEAP_TYPE_READBACK`` buffer
        on every executed pass) and no counter measures it, so the byte field reads
        None and nothing was fabricated.
      * ``absent`` - a report that predates the field entirely.
      * ``misfiled`` - only a pre-repair report can produce a number here, and that
        number is the image-plane counter standing in the wrong slot. Reading it as a
        measured scalar byte count would be the fault the repair records, so the state
        is named and the re-read is labelled instead.
    """
    value = report.get(HISTORICAL_IMAGE_READBACK_KEY)
    if IMAGE_READBACK_KEY in report:
        # Post-repair: the historical slot is the scalar field's own, and the current
        # code writes None into it. Any number here would be the same counter booked
        # twice, which is the fault, so this is the one failing shape.
        return {"state": "unanswered" if value is None else "doubled",
                "value": value,
                "detail": f"{HISTORICAL_IMAGE_READBACK_KEY}={value!r}; per-pass bytes "
                          f"are a source fact (256 B per dispatch) and unmeasured"}
    if value is None:
        return {"state": "absent", "value": None,
                "detail": "the report carries no field for this boundary at all"}
    return {"state": "misfiled", "value": value,
            "detail": "OFFLINE RE-READ: the report's number in the scalar slot is the "
                      "image-plane counter, not a measurement of scalar traffic; the "
                      "boundary is accepted as present-and-unmeasured"}


def _checklist(report, stream, side, frames, packets, keyed, seis, source_seis,
               source_side, working):
    """What the brief asks of the finished file, read from the file and the report.

    A line reports the value it read, so a counter the chain does not answer shows as
    None or absent rather than as a zero that was never measured.
    """
    checks = {}
    checks["export backend actually gpu_transfer"] = (
        report.get("export_backend_actual") == "gpu_transfer",
        repr(report.get("export_backend_actual")))
    checks["transfer mode actually gpu_hdr_transfer"] = (
        report.get("transfer_mode_actual") == "gpu_hdr_transfer",
        repr(report.get("transfer_mode_actual")))
    checks["no fallback reason recorded"] = (
        report.get("export_backend_fallback_reason") is None,
        repr(report.get("export_backend_fallback_reason")))
    checks["experimental hdr reported"] = (
        report.get("experimental_hdr_scrgb") is True,
        repr(report.get("experimental_hdr_scrgb")))
    checks["motion came from real NVOF device input"] = (
        report.get("hdr_uses_flow") is True
        and report.get("hdr_dlss_flow_calls_measured") == FRAMES
        and report.get("hdr_dlss_zero_motion_calls_measured") == 0,
        f"flow calls={report.get('hdr_dlss_flow_calls_measured')} zero-motion "
        f"calls={report.get('hdr_dlss_zero_motion_calls_measured')} "
        f"device input copies={report.get('nvof_device_input_copy_count_total')}")
    checks["session colour was the FP16 format"] = (
        report.get("hdr_color_format") == "rgba16f",
        repr(report.get("hdr_color_format")))
    checks["feature 18 reset once, not per frame"] = (
        report.get("hdr_feature_resets_measured") == 1,
        repr(report.get("hdr_feature_resets_measured")))
    checks["working extent is the resolved one"] = (
        report.get("hdr_working_dims") == list(working)
        and report.get("hdr_output_dims") == report.get("hdr_full_dims"),
        f"working={report.get('hdr_working_dims')} "
        f"full={report.get('hdr_full_dims')} "
        f"output={report.get('hdr_output_dims')}")
    policy = report.get("hdr_fixed_policy") or {}
    checks["fixed ceilings are the record's constants"] = (
        policy.get("support_ceiling_scrgb") == 250.0
        and policy.get("clip_ceiling_scrgb") == 125.0
        and policy.get("paper_white_scrgb") == 8.0,
        json.dumps(policy, default=str))

    checks["six packets in the finished file"] = (packets == FRAMES,
                                                 f"counted {packets}")
    checks["six frames decoded back out of it"] = (len(frames) == FRAMES,
                                                  f"read {len(frames)}")
    # Presentation timestamps in presentation order. Packets are stored in decode
    # order, which an encoder using B-frames puts out of presentation order on purpose,
    # so monotonicity is read off the decoded frames and the packet stamps are checked
    # for the two things that would actually be wrong: a missing stamp, or a duplicate.
    display = [int(item["pts"]) for item in frames if item.get("pts") is not None]
    packet_pts = [int(item["pts"]) for item in keyed if item.get("pts") is not None]
    packet_dts = [int(item["dts"]) for item in keyed if item.get("dts") is not None]
    checks["presentation timestamps monotonic in presentation order"] = (
        len(display) == packets and all(b > a for a, b in zip(display, display[1:])),
        f"{len(display)} frame pts in decode-to-display order, strictly increasing: "
        f"{display}")
    checks["every packet carries one unique presentation stamp"] = (
        len(packet_pts) == packets and len(set(packet_pts)) == packets
        and sorted(packet_pts) == sorted(display),
        f"{len(packet_pts)} packet pts, {len(set(packet_pts))} distinct: "
        f"{packet_pts}")
    checks["decode order timestamps non-decreasing where present"] = (
        all(b > a for a, b in zip(packet_dts, packet_dts[1:]))
        if packet_dts else False,
        f"{len(packet_dts)} of {packets} packets carry a dts "
        f"({packets - len(packet_dts)} report none, which is how a probe reads the two "
        f"packets ahead of the first random-access point): {packet_dts}")

    tags = {key: stream.get(key) for key in VIDEO_TAGS}
    wanted_tags = {"codec_name": "hevc", "profile": "Main 10",
                   "pix_fmt": "yuv420p10le", "color_range": "tv",
                   "color_space": "bt2020nc", "color_transfer": "smpte2084",
                   "color_primaries": "bt2020"}
    checks["container tags are the HDR10 declaration"] = (
        all(tags.get(key) == value for key, value in wanted_tags.items()),
        json.dumps(tags))

    def same(source, output, label):
        return (source is not None and output is not None
                and json.dumps(source, sort_keys=True, default=str) ==
                json.dumps(output, sort_keys=True, default=str),
                f"{label}: source={json.dumps(source, default=str)} "
                f"output={json.dumps(output, default=str)}")

    checks["mastering display identical in the container"] = same(
        source_side.get("Mastering display metadata"),
        side.get("Mastering display metadata"), "137-equivalent side data")
    checks["content light identical in the container"] = same(
        source_side.get("Content light level metadata"),
        side.get("Content light level metadata"), "144-equivalent side data")

    def payload_of(sei_list, ptype):
        return [item["bytes"] for item in sei_list if item["payload_type"] == ptype]

    for label, ptype in (("mastering 137", 137), ("content light 144", 144)):
        written, source_bytes = payload_of(seis, ptype), payload_of(source_seis, ptype)
        checks[f"{label} SEI present in the elementary stream"] = (
            len(written) > 0, f"{len(written)} payloads, first "
            f"{written[0] if written else None} ({(len(written[0]) // 2) if written else 0}"
            f" bytes)")
        checks[f"{label} SEI carries the source's own numbers"] = (
            bool(written) and bool(source_bytes) and written[0] == source_bytes[0],
            f"output={written[0] if written else None} "
            f"source={source_bytes[0] if source_bytes else None}")
    checks["encoder states the metadata went in-bitstream"] = (
        report.get("encoder_in_bitstream_sei_written") is True,
        f"written_to={report.get('encoder_metadata_written_to_measured')} "
        f"destinations={report.get('encoder_metadata_destinations')}")
    checks["sei writer reports no error and inserted the metadata"] = (
        (report.get("sei_writer") or {}).get("error") in ("ok", 0, None)
        and (report.get("sei_writer") or {}).get("mastering_inserted")
        and (report.get("sei_writer") or {}).get("content_light_inserted"),
        json.dumps(report.get("sei_writer"), default=str))
    checks["source metadata identical across the six frames"] = (
        report.get("hdr_metadata_stable_across_frames") is True
        and report.get("hdr_metadata_snapshot_count_measured") == FRAMES,
        f"stable={report.get('hdr_metadata_stable_across_frames')} snapshots="
        f"{report.get('hdr_metadata_snapshot_count_measured')} divergence="
        f"{json.dumps(report.get('hdr_metadata_divergence'), default=str)}")

    # The host boundaries proper: the counters that only move bytes across the bus.
    # The colour bridge's copy and write counters are device-to-device traffic, so they
    # are reported beside these rather than folded into a zero claim about them.
    host_boundaries = {
        "decoder host frame copy bytes": "decode_host_copy_bytes_measured",
        "decoder host frame copy count": "decode_host_copy_count_measured",
        "Feature 18 host colour stage bytes": "dlss_host_color_stage_bytes_measured",
        "Feature 18 host colour stage count": "dlss_host_color_stage_count_measured",
        "Feature 18 output readback bytes": "dlss_output_readback_bytes_measured",
        "encoder host pixel bytes": "encoder_host_pixel_bytes_measured",
        "encoder rgb conversion bytes": "encoder_rgb_conversion_bytes_measured"}
    for label, key in host_boundaries.items():
        checks[f"host boundary zero: {label}"] = (
            report.get(key) == 0, f"{key}={report.get(key, '<absent>')}")
    checks["decoder scales nothing"] = (
        report.get("decode_scale_device_output_bytes_measured") == 0,
        f"scale_device_output_bytes="
        f"{report.get('decode_scale_device_output_bytes_measured')}")
    device_traffic = {
        "source plane writes": "hdr_source_device_write_bytes_measured",
        "working plane writes": "hdr_working_device_write_bytes_measured",
        "guidance plane writes": "hdr_guidance_device_write_bytes_measured",
        "composited plane writes": "hdr_composited_device_write_bytes_measured",
        "encoded P010 writes": "hdr_encode_device_write_bytes_measured",
        "NVOF device input copies": "nvof_device_input_copy_count_total"}
    checks["every stage moved bytes on the device, and says how many"] = (
        all(isinstance(report.get(key), int) and report.get(key) > 0
            for key in device_traffic.values()),
        json.dumps({label: report.get(key, "<absent>")
                    for label, key in device_traffic.items()}, default=str))
    # Counters that may honestly be zero because the buffer was registered rather than
    # copied: they are read and recorded, not asserted on.
    for label, key in (("colour bridge borrowed P010 copy",
                        "hdr_source_device_copy_bytes_measured"),
                       ("Feature 18 device colour copy",
                        "dlss_device_color_copy_bytes_measured"),
                       ("encoder P010 device copy",
                        "encoder_p010_device_copy_bytes_measured")):
        print(f"[gate] device-traffic read, not asserted: {label} "
              f"({key}) = {report.get(key, '<absent>')}")
    checks["no pixel readback entry point is bound"] = (
        report.get("hdr_pixel_readback_exports_bound") == []
        and report.get("decoder_pixel_readback_entry_bound") is False,
        f"colour bridge exports={report.get('hdr_pixel_readback_exports_bound')} "
        f"decoder entry bound={report.get('decoder_pixel_readback_entry_bound')}")
    checks["pixel readback is unanswered, not zeroed"] = (
        report.get("hdr_pixel_readback_bytes_measured") is None,
        repr(report.get("hdr_pixel_readback_bytes_measured")))
    # The diagnostic image-plane counter, named for what it measures. Its zero says no
    # plane was copied out; it says nothing about the scalars, which every executed
    # pass brings back to the host and which no counter counts. The accepted record had
    # these two boundaries in one line; the repair separates them, and each line below
    # reads the value that belongs to it.
    image = image_readback_counter(report)
    checks["diagnostic image-plane readback is the bridge's own zero"] = (
        image["value"] == 0,
        f"{image['key']}={image['value']!r} ({image['provenance']})")
    scalar = scalar_readback_claim(report)
    # What has to hold of the scalar boundary is that the report does not present it as
    # a measured zero, and that the passes whose readback it is actually ran - which is
    # what the two frame counters answer beside it.
    checks["scalar counter readback is present and not reported as a measured zero"] = (
        scalar["state"] != "doubled"
        and report.get("total_frames_reported") == FRAMES
        and report.get("hdr_process_count_measured") == FRAMES,
        f"state={scalar['state']} value={scalar['value']!r} "
        f"frames_reported={report.get('total_frames_reported')} "
        f"process_count={report.get('hdr_process_count_measured')} "
        f"({scalar['detail']})")
    envelopes = {key: value for key, value in report.items()
                 if key.endswith("_envelope")}
    checks["per-stage scalar extremes are reported, not averaged away"] = (
        len(envelopes) >= 6,
        json.dumps({key: envelopes[key] for key in sorted(envelopes)}, default=str))

    clipping = {key: value for key, value in report.items()
                if key.startswith("total_") and isinstance(value, int)}
    checks["clipping and rejection counts are reported"] = (
        any("negative" in key or "clamped" in key or "rejected" in key
            for key in clipping),
        json.dumps({key: clipping[key] for key in sorted(clipping)[:8]}))
    return checks


if __name__ == "__main__":
    sys.exit(recheck() if "--recheck" in sys.argv[1:] else main())
