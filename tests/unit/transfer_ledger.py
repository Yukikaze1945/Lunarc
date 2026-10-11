"""The export path's transfer ledger: every buffer a frame touches, and what moved it.

Two kinds of number appear here and they must never be confused:

* a payload size is arithmetic on a width, a height and a bytes-per-pixel figure. It says
  how big a buffer is. It says nothing about a bus.
* a transfer byte count says bytes crossed a boundary, and it only exists when something
  measured it - the NVOF bridge's own counters, or the return value of the read or write
  that did the moving.

A row whose transfer cannot be observed carries measurement "unavailable" and no number.
The DLSS host wrapper is the standing example: it is handed host pointers of known size,
reports no counters, and its source is not in this repository, so its internal GPU
traffic is unknown and stays labelled unknown.

Rows are written for the production downscaled export path at a given source size and
working scale, in stage order. `status` is only filled in once a phase has evidence for
it; until then it reads "not_yet_evaluated", which is what keeps a planned change from
being reported as a finished one.

Two orders run through those stages and they are not the same audit, so a ledger is built
for one of them: `transfer_mode` is "legacy" while the pipeline converts the working plane
twice for the DLSS host and once back for the composite, and "reduced" when it does not.
Switching mode changes which producer is named on a row and what crosses PCIe; it never
removes a row. A conversion that no longer happens is still a fact worth reporting, and the
only way to tell it apart from a conversion nobody measured is for the row to stay and read
a measured 0.
"""

import json
import os

SOURCE_W, SOURCE_H = 3840, 2160
WORKING_SCALE = 0.5
WORKING_W, WORKING_H = 1920, 1080

# Stage order the audit follows, matching the order a frame is touched in.
STAGES = ("decode", "working_color", "nvof", "dlss", "residual",
          "output_resize", "encoder")

# The two transfer orders ProductionExportCore negotiates between. A ledger built for one
# must not be used to describe a run of the other.
LEGACY = "legacy"
REDUCED = "reduced"
TRANSFER_MODES = (LEGACY, REDUCED)

TRANSFER_MODE_MEANING = {
    LEGACY: "the working plane is converted to RGBA for the DLSS host, NVOF is handed a "
            "second CPU-reordered copy of it, and the host's RGBA output is converted back "
            "to BGR before the residual composite reads it",
    REDUCED: "one filter pass writes both the BGR plane and the RGBA plane, the BGR plane "
             "goes to the NVOF bridge as it stands and is reordered on the GPU, and the "
             "residual composite reads the host's RGBA output where the host wrote it",
}

FORMAT_BPP = {"bgr24": 3, "rgba8": 4, "nv12": 1.5, "flow_f32x2": 8, "depth_f32": 4}

NOT_EVALUATED = "not_yet_evaluated"
MEASUREMENTS = ("measured", "estimated", "logical", "unavailable", "not_reached")


def payload(width, height, fmt):
    """Bytes a buffer of that format and size holds. Arithmetic, not a measurement."""
    return int(round(width * height * FORMAT_BPP[fmt]))


# The 4K / 50% payloads the phase is specified against. Listed so a reader can check any
# number in a row against arithmetic instead of trusting a table.
PAYLOAD_REFERENCE = {
    "4k_bgr24": payload(SOURCE_W, SOURCE_H, "bgr24"),
    "4k_nv12": payload(SOURCE_W, SOURCE_H, "nv12"),
    "working_bgr24": payload(WORKING_W, WORKING_H, "bgr24"),
    "working_rgba8": payload(WORKING_W, WORKING_H, "rgba8"),
    "working_flow_f32x2": payload(WORKING_W, WORKING_H, "flow_f32x2"),
    "working_depth_f32": payload(WORKING_W, WORKING_H, "depth_f32"),
}

FFMPEG_CHILD = "ffmpeg_child"
PYTHON_MAIN = "python_main"


def _row(name, stage, producer, consumer, width, height, fmt, owner_process, location,
         allocation_owner, direction, lifetime, allocation, measurement,
         measured_field=None, measured_calls=None, note="", status=NOT_EVALUATED):
    """One buffer in the ledger. See the module docstring for the payload/transfer rule."""
    return {
        "name": name,
        "stage": stage,
        "producer": producer,
        "consumer": consumer,
        "width": width,
        "height": height,
        "format": fmt,
        "bytes_per_pixel": FORMAT_BPP.get(fmt),
        "payload_bytes": payload(width, height, fmt) if fmt in FORMAT_BPP else None,
        "owner_process": owner_process,
        "location": location,
        "allocation_owner": allocation_owner,
        "transfer_direction": direction,
        "transfer_bytes": None,
        "measured_field": measured_field,
        "cpu_call_count": None,
        "measured_calls": measured_calls,
        "lifetime": lifetime,
        "allocation": allocation,
        "measurement": measurement,
        "note": note,
        "status": status,
    }


def ledger_rows(source_w=SOURCE_W, source_h=SOURCE_H, working_w=None, working_h=None,
                output_w=None, output_h=None, decode_actual="cuda",
                transfer_mode=LEGACY):
    """Every buffer of one frame of the production export path, in the order it touches them.

    The rows describe the path as the code wires it: which function writes the buffer,
    which one reads it next, and which boundary it crosses in between. `transfer_mode`
    selects which of the two orders that is.
    """
    if transfer_mode not in TRANSFER_MODES:
        raise ValueError(f"unknown transfer_mode {transfer_mode!r}; a ledger describes one "
                         f"of {TRANSFER_MODES} and must not be built for anything else")
    working_w = working_w or source_w
    working_h = working_h or source_h
    output_w = output_w or source_w
    output_h = output_h or source_h
    gpu_decoded = decode_actual == "cuda"
    reduced = transfer_mode == REDUCED
    rows = [
        # --- decode -------------------------------------------------------------
        _row("decoder_rawvideo", "decode",
             "ffmpeg child (NVDEC + scale_cuda + hwdownload)" if gpu_decoded
             else "ffmpeg child (software decode)",
             "pipeline._read_exactly", source_w, source_h,
             "nv12" if gpu_decoded else "bgr24", FFMPEG_CHILD, "cpu",
             "pipeline._read_exactly allocates a bytearray per read",
             "ffmpeg_child -> python_main (stdout pipe)", "one frame", "allocated_per_frame",
             "measured", "decoder_pipe_bytes",
             note="A pipe is a host byte stream: it cannot carry a device pointer or a "
                  "GPU resource handle, so whatever the decoder did on the GPU, the frame "
                  "arrives here as bytes."),
        _row("decoder_gpu_to_cpu_download", "decode", "ffmpeg hwdownload filter",
             "ffmpeg rawvideo muxer", source_w, source_h, "nv12", FFMPEG_CHILD,
             "gpu_to_cpu", "ffmpeg-internal CUDA buffers",
             "gpu -> cpu (inside the ffmpeg child)", "one frame", "ffmpeg_internal",
             "estimated" if gpu_decoded else "not_reached",
             note="FFmpeg exposes no transfer counter, so this row's payload is the size "
                  "of the copy that must have happened, not a measured figure."
             if gpu_decoded else
             "The decoder ran without a hardware download, so there is no such copy to "
             "count - and nothing measured that either."),
        _row("source_full_bgr", "decode",
             "cv2.cvtColor(YUV2BGR_NV12) at the decode site" if gpu_decoded
             else "the decoded BGR24 frame",
             "working-colour prepare and the residual composite", source_w, source_h,
             "bgr24", PYTHON_MAIN, "cpu",
             "cv2 allocates the destination per call" if gpu_decoded
             else "a view of the read buffer, allocated per read",
             "cpu -> cpu (conversion output write)" if gpu_decoded
             else "no write: the frame is used where it was read",
             "one frame", "allocated_per_frame",
             "logical" if gpu_decoded else "not_reached",
             note="The 4K frame the residual adds its detail back to."),

        # --- working colour -----------------------------------------------------
        _row("working_bgr", "working_color",
             "residual backend prepare_working_color_rgba (the fused pass)" if reduced
             else "residual backend prepare_working_color",
             "NVOF host input and the residual's low-resolution term" if reduced
             else "the RGBA conversion below, and the residual's low-resolution term",
             working_w, working_h, "bgr24", PYTHON_MAIN, "cpu",
             "FrameWorkspace.down_cur_bgr (once per job)",
             "cpu -> cpu (Lanczos output write)", "job", "reused", "logical",
             note="Written by the native bridge's Lanczos pass."
             + (" The reduced order hands this plane to the NVOF bridge unchanged, so no "
                "second copy of it is made on the CPU." if reduced else "")),
        _row("working_rgba_for_dlss", "working_color",
             "the fused Lanczos pass, as the second plane it publishes" if reduced
             else "cv2.cvtColor(BGR2RGBA) at the dlss_input site",
             "DLSS host colour input", working_w, working_h, "rgba8",
             PYTHON_MAIN, "cpu", "FrameWorkspace.live_rgba (once per job)",
             "cpu -> cpu (channel-order write)" if not reduced
             else "cpu -> cpu (second write of the same filter pass)",
             "job", "reused", "measured",
             measured_field="cpu_bgr_to_rgba@dlss_input_output_bytes",
             measured_calls="cpu_bgr_to_rgba@dlss_input_calls",
             note="The buffer the DLSS host is handed exists in both orders and is written "
                  "once per frame in both." + (
                      " What the reduced order deletes is the separate channel-order "
                      "conversion, so this row's call count is a measured 0 while the plane "
                      "itself is still there: the seam counts conversions, not writes."
                      if reduced else
                      " Counted at the conversion seam, at this site only: the total a frame "
                      "makes across both sites is a different number and is not what this row "
                      "claims. The bytes are the buffer this conversion wrote, which is its "
                      "payload, not a bus transfer.")),
        _row("working_rgba_for_nvof", "working_color",
             "no CPU copy: the BGR plane above goes to the bridge as it stands" if reduced
             else "cv2.cvtColor(BGR2RGBA) at the nvof_input site",
             "NVOF bridge host input" if reduced else "NVOF bridge host input (rgba8)",
             working_w, working_h, "rgba8", PYTHON_MAIN, "cpu",
             "no allocation: the reduced order never builds this buffer" if reduced
             else "NvOFHardwareEngine._cur_abgr_buf (once per session)",
             "no write" if reduced else "cpu -> cpu (channel-order write)",
             "job" if reduced else "session", "reused", "measured",
             measured_field="cpu_bgr_to_rgba@nvof_input_output_bytes",
             measured_calls="cpu_bgr_to_rgba@nvof_input_calls",
             note="A second, separate RGBA copy of the same working frame: the flow engine "
                  "converts for itself rather than consuming the buffer above. A temporal "
                  "hint reset makes it convert the reference frame too, which is why the "
                  "steady-state median is the number that matters."
             if not reduced else
             "The row stays because the copy's absence is the claim being reported, and a "
             "row that vanished could not carry it. The counter is the cv2 seam's own, so "
             "0 here means the conversion call site is gone, not that nobody looked."),

        # --- NVOF ---------------------------------------------------------------
        _row("nvof_input_upload", "nvof", "cuMemcpy2D host -> device", "nvOFExecute",
             working_w, working_h, "bgr24" if reduced else "rgba8", PYTHON_MAIN,
             "host_to_device",
             "nvof_bridge session BGR staging (allocated once per session)" if reduced
             else "nvof_bridge session input surface (allocated once per session)",
             "cpu -> gpu", "session", "reused", "measured", "nvof_h2d_bytes_measured",
             note="Native counter. Steady state uploads once; a temporal-hint reset "
                  "uploads the current and the reference frame."
             + (" Three bytes per pixel reach the GPU and the channel order is fixed by a "
                "byte-map kernel on it, which is where the alpha this row no longer counts "
                "would have been written." if reduced else
                " Four bytes per pixel, because the bridge of this era takes RGBA; the "
                "reduced order uploads the same frame as BGR24.")),
        _row("nvof_flow_download", "nvof", "cuMemcpy2D device -> host", "DLSS motion input",
             working_w, working_h, "flow_f32x2", PYTHON_MAIN, "device_to_host",
             "NvOFHardwareEngine._flow_buf (once per session)", "gpu -> cpu", "session",
             "reused", "measured", "nvof_d2h_bytes_measured",
             note="Native counter. The flow reaches DLSS as a host pointer, so this copy "
                  "is on the path in every mode."),

        # --- DLSS ---------------------------------------------------------------
        _row("dlss_colour_input", "dlss", "working_rgba_for_dlss", "dlssnr_host.dll",
             working_w, working_h, "rgba8", PYTHON_MAIN, "host_pointer_handed_to_wrapper",
             "FrameWorkspace.live_rgba", "cpu -> wrapper (host pointer)", "job", "reused",
             "logical", measured_field="dlss_input_color_bytes_logical",
             note="What the wrapper does with the pointer is not observable from the "
                  "public ABI; this is the size of what it was handed."),
        _row("dlss_motion_input", "dlss", "nvof_flow_download", "dlssnr_host.dll",
             working_w, working_h, "flow_f32x2", PYTHON_MAIN,
             "host_pointer_handed_to_wrapper", "NvOFHardwareEngine._flow_buf",
             "cpu -> wrapper (host pointer)", "session", "reused", "logical",
             measured_field="dlss_input_flow_bytes_logical"),
        _row("dlss_depth_input", "dlss", "constant zero plane", "dlssnr_host.dll", working_w,
             working_h, "depth_f32", PYTHON_MAIN, "host_pointer_handed_to_wrapper",
             "Live._zero_depth", "cpu -> wrapper (host pointer)", "session", "reused",
             "logical", measured_field="dlss_input_depth_bytes_logical",
             note="The closed legacy host DLL still requires a depth pointer; it always receives the same zero plane."),
        _row("dlss_internal_gpu_io", "dlss", "dlssnr_host.dll internals", "unknown", None,
             None, "unknown", PYTHON_MAIN, "unknown", "inside the wrapper", "unknown",
             "unknown", "unknown", "unavailable",
             measured_field="dlss_gpu_transfer_bytes_measured",
             note="No public export reports a transfer and no host source exists here, so "
                  "no number - including zero - belongs in this row."),
        _row("dlss_output_rgba", "dlss", "dlssnr_host.dll output write",
             "residual reconstruction", working_w, working_h, "rgba8", PYTHON_MAIN, "cpu",
             "FrameWorkspace.dlss_out_rgba (once per job)", "wrapper -> host buffer", "job",
             "reused", "logical", measured_field="dlss_output_bytes_logical"),

        # --- residual -----------------------------------------------------------
        _row("dlss_output_bgr", "residual",
             "no CPU copy: reconstruct_from_rgba reads the row above in place" if reduced
             else "cv2.cvtColor(RGBA2BGR) at the residual_input site",
             "residual reconstruction input", working_w, working_h, "bgr24",
             PYTHON_MAIN, "cpu",
             "no allocation: FrameWorkspace.dlss_out_bgr is not reserved" if reduced
             else "FrameWorkspace.dlss_out_bgr (once per job)",
             "no write" if reduced else "cpu -> cpu (channel-order write)",
             "job", "reused", "measured",
             measured_field="cpu_rgba_to_bgr@residual_input_output_bytes",
             measured_calls="cpu_rgba_to_bgr@residual_input_calls",
             note="The second tracked conversion: it exists only to hand the residual a "
                  "BGR-ordered buffer, because the residual reads BGR channels."
             if not reduced else
             "The four-channel store is read where it was written and the fourth channel is "
             "stepped over, so there is no three-channel copy to count. The payload column "
             "is the size of the buffer the legacy order used to build here, kept so the "
             "saving is arithmetic a reader can check."),
        _row("recon_full_bgr", "residual",
             "residual backend reconstruct (signed residual + Catmull-Rom + composite)",
             "output resize / encoder pipe", source_w, source_h, "bgr24", PYTHON_MAIN, "cpu",
             "FrameWorkspace.recon_bgr (once per job)",
             "cpu -> cpu (reconstruction write)", "job", "reused", "logical"),

        # --- output resize ------------------------------------------------------
        _row("output_resize", "output_resize", "cv2.resize to the target size",
             "encoder pipe", output_w, output_h, "bgr24", PYTHON_MAIN, "cpu",
             "FrameWorkspace.dst_bgr (once per job)", "cpu -> cpu (resize write)", "job",
             "reused",
             "not_reached" if (output_w, output_h) == (source_w, source_h) else "logical",
             note="The target equals the source in this configuration, so the pipeline "
                  "aliases the reconstruction buffer and no resize runs."
             if (output_w, output_h) == (source_w, source_h) else ""),

        # --- encoder ------------------------------------------------------------
        _row("encoder_rawvideo", "encoder", "the frame the pipeline finished",
             "ffmpeg child (NVENC) reading its stdin", output_w, output_h, "bgr24",
             PYTHON_MAIN, "cpu", "written straight out of the buffer above",
             "python_main -> ffmpeg_child (stdin pipe)", "one frame", "allocated_per_frame",
             "measured", "encoder_pipe_bytes",
             note="The write call's own return value, summed per frame - not a width times "
                  "a height times three."),
        _row("encoder_gpu_upload", "encoder",
             "ffmpeg's own upload of the rawvideo it read", "NVENC", output_w, output_h,
             "bgr24", FFMPEG_CHILD, "cpu_to_gpu", "ffmpeg-internal",
             "cpu -> gpu (inside the ffmpeg child)", "one frame", "ffmpeg_internal",
             "unavailable", measured_field="encoder_gpu_upload_bytes_measured",
             note="FFmpeg uploads what it reads and reports nothing about it; the pipe row "
                  "above is the measured part."),
    ]
    return rows


def apply_measurements(rows, measured):
    """Fill in what the run measured, leaving the rest visibly empty.

    `measured` maps per-frame field names to numbers (or to None when the run could not
    measure them). A row whose field is absent keeps transfer_bytes None and flips its
    measurement to "unavailable", so the ledger reports the gap instead of papering over
    it with an estimate.
    """
    for row in rows:
        field = row.get("measured_field")
        if field and row["measurement"] in ("measured", "logical"):
            value = (measured or {}).get(field)
            if isinstance(value, (int, float)):
                row["transfer_bytes"] = int(value)
            elif row["measurement"] == "measured":
                row["measurement"] = "unavailable"
                row["note"] = (row["note"] + " | the field this row reads was not "
                               "reported by the run that produced this ledger").lstrip(" |")
        calls_field = row.get("measured_calls")
        if calls_field:
            value = (measured or {}).get(calls_field)
            if isinstance(value, (int, float)):
                row["cpu_call_count"] = int(value)
    return rows


def build_ledger(measured=None, source_w=SOURCE_W, source_h=SOURCE_H, working_w=None,
                 working_h=None, output_w=None, output_h=None, decode_actual="cuda",
                 transfer_mode=LEGACY, extra=None):
    rows = ledger_rows(source_w=source_w, source_h=source_h, working_w=working_w,
                       working_h=working_h, output_w=output_w, output_h=output_h,
                       decode_actual=decode_actual, transfer_mode=transfer_mode)
    apply_measurements(rows, measured or {})
    ledger = {
        "schema": 3,
        "source_size": [source_w, source_h],
        "working_size": [working_w or source_w, working_h or source_h],
        "output_size": [output_w or source_w, output_h or source_h],
        "decode_actual": decode_actual,
        "transfer_mode": transfer_mode,
        "transfer_mode_meaning": TRANSFER_MODE_MEANING,
        "stages_in_frame_order": list(STAGES),
        "payload_reference": PAYLOAD_REFERENCE,
        "measurement_rules": {
            "measured": "a counter, or the return value of the read or write that did the "
                        "moving, from the run itself",
            "estimated": "arithmetic on a format, standing in for a copy nothing counted",
            "logical": "the size of a buffer a stage was handed, or wrote; not a bus "
                       "transfer",
            "unavailable": "nothing observes this, and no number - including 0 - is claimed",
            "not_reached": "the branch did not run in this configuration",
        },
        "status_values": ["ELIMINATED_PROVEN", "REDUCED_PROVEN", "BLOCKED_PROVEN",
                          NOT_EVALUATED],
        "rows": rows,
    }
    if extra:
        ledger.update(extra)
    return ledger


def render_markdown(ledger):
    """The same table a person reads, in frame order, with each number's basis."""
    lines = [
        f"# Transfer ledger - {ledger['source_size'][0]}x{ledger['source_size'][1]} source, "
        f"{ledger['working_size'][0]}x{ledger['working_size'][1]} working, "
        f"decode `{ledger['decode_actual']}`, "
        f"transfer order `{ledger['transfer_mode']}`",
        "",
        f"Generated by `{ledger.get('generated_by', 'tests/unit/transfer_ledger.py')}` on "
        f"code under test `{ledger.get('code_under_test', 'unknown')}`.",
        "",
    ]
    if ledger.get("summary"):
        lines += [ledger["summary"], ""]
    lines += [
        "A payload size is arithmetic on a buffer. A transfer byte count is a measurement. "
        "A row whose measurement is `unavailable` has no number on purpose, and `0` in a "
        "transfer column always means measured-and-really-zero.",
        "",
        f"This ledger describes the `{ledger['transfer_mode']}` order: "
        f"{TRANSFER_MODE_MEANING[ledger['transfer_mode']]}.",
        "",
    ]
    by_stage = {}
    for row in ledger["rows"]:
        by_stage.setdefault(row["stage"], []).append(row)
    for stage in ledger["stages_in_frame_order"]:
        rows = by_stage.get(stage, [])
        if not rows:
            continue
        lines += [f"## {stage}", ""]
        lines.append("| buffer | shape | format | bpp | payload | transfer | calls | basis "
                     "| direction | process | allocation | status |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in rows:
            dims = (f"{row['width']}x{row['height']}" if row["width"] else "-")

            def _num(value):
                return "-" if value is None else f"{value:,}"
            lines.append(
                f"| `{row['name']}` | {dims} | {row['format']} | "
                f"{row['bytes_per_pixel'] or '-'} | {_num(row['payload_bytes'])} | "
                f"{_num(row['transfer_bytes'])} | {_num(row['cpu_call_count'])} | "
                f"{row['measurement']} | {row['transfer_direction']} | "
                f"{row['owner_process']} | {row['allocation']} | {row['status']} |")
        for row in rows:
            lines += ["", f"- `{row['name']}` - producer: {row['producer']}; consumer: "
                      f"{row['consumer']}; lifetime: {row['lifetime']}; allocation owner: "
                      f"{row['allocation_owner']}."]
            if row["measured_field"]:
                lines.append(f"  Reads the per-frame field `{row['measured_field']}`.")
            if row["measured_calls"]:
                lines.append(f"  Reads the per-frame field `{row['measured_calls']}`.")
            if row["note"]:
                lines.append(f"  {row['note']}")
            lines.append(f"  Status: `{row['status']}`.")
        lines.append("")
    lines += ["## Payload reference", ""]
    for name, value in sorted(ledger["payload_reference"].items()):
        lines.append(f"- `{name}`: {value:,} bytes")
    lines += ["", "## Measurement rules", ""]
    for name in MEASUREMENTS:
        lines.append(f"- `{name}`: {ledger['measurement_rules'][name]}")
    lines += ["", "Statuses a row may carry: " +
              ", ".join(f"`{s}`" for s in ledger["status_values"]) + ".", ""]
    return "\n".join(lines)


def write_ledger(path, ledger):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_markdown(path, ledger):
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(ledger))
    return path
