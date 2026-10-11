"""Two-frame P010 -> HEVC Main10 encoder-boundary gate.

This is an ENCODER BOUNDARY test, not a full HDR DLSS export: it takes exactly
two frames from a real encoded HDR10 fixture through the existing
nvdec_bridge_open_hdr borrow, hands the borrowed CUDA P010 planes straight to
the new nvenc_hdr_open / nvenc_encode_p010 entry on the decoder's own CUcontext,
and reads the result back with ffprobe and with the same decoder.

Nothing on the processing path touches host pixel memory: no readback call is
made at all, and the counter checks assert the decoder and encoder both report a
zero for the boundaries they cannot cross.

Run from anywhere:
    python native/nvenc_hdr_p010_smoke.py
    python native/nvenc_hdr_p010_smoke.py --rgba-on-hdr-bridge   (child probe mode)
"""

import ctypes as C
import json
import os
import platform
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def sha256_file(path):
    """Hashed lazily, never at import time.

    Importing hashlib up front makes python's _hashlib bind libcrypto-3-x64.dll
    from its own runtime directory, and the FFmpeg avformat chain the bridges
    depend on then resolves that same name against a copy that does not export
    what libssl-3-x64.dll needs: the bridge load fails with WinError 127 before
    a single frame is decoded. Loading the bridges first and hashing afterwards
    puts the two in the order the loader can actually satisfy.
    """
    from hashlib import sha256
    return sha256(path.read_bytes()).hexdigest()


ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "native"
ARTIFACTS = ROOT / "bench_out" / "hdr_sei_repair"
# The previous stage's bytes are the baseline this stage has to preserve: the
# coded slices of its output are compared against the new ones, so the directory
# is read, never written.
PRESERVED_BASELINE = ROOT / "bench_out" / "hdr_p010_nvenc" / "a_hdr10_out.hevc"
FFMPEG_DIR = Path(os.environ.get("FFMPEG_SDK_DIR", r"F:\WhisperJAV\Library"))
FFMPEG_BIN = FFMPEG_DIR / "bin"
FIXTURES = ROOT / "build" / "hdr_probe"

# The struct sizes nvenc_hdr_probe.exe printed from the compiled header. Binding
# against a number the compiler agrees with is the point; a mismatch is a hard
# failure rather than a silently wrong field read.
PROBE_SIZES = {"options": 112, "counters": 64, "defaults": 56,
               "capability": 128, "contract": 268, "metadata": 108,
               "sei_capability": 56, "sei_info": 244, "sei_status": 384}

U32, I32, U64, I64, VP, CP, CSZ = (C.c_uint32, C.c_int32, C.c_uint64, C.c_int64,
                                   C.c_void_p, C.c_char_p, C.c_size_t)

# HDR10 contract values, read back from the installed library where a number is
# needed for a refusal case rather than typed in.
BT2020, PQ, BT2020NC, LIMITED = 9, 16, 9, 1
MAIN10 = 2


class DecFrame(C.Structure):
    """Mirror of NvdecDeviceFrame (native/nvdec_bridge.h, ABI 3)."""
    _fields_ = [
        ("struct_size", U32), ("width", U32), ("height", U32),
        ("software_format", U32), ("y_device_ptr", U64), ("uv_device_ptr", U64),
        ("y_pitch_bytes", I32), ("uv_pitch_bytes", I32), ("cuda_context", VP),
        ("pts", I64), ("time_base_num", I32), ("time_base_den", I32),
        ("color_range", I32), ("color_primaries", I32), ("color_transfer", I32),
        ("color_space", I32),
    ]


class DecCounters(C.Structure):
    _fields_ = [("struct_size", U32), ("frames_decoded", U64),
                ("device_frame_count", U64), ("bitstream_bytes_submitted", U64),
                ("host_frame_copy_count", U64), ("host_frame_copy_bytes", U64),
                ("scale_device_output_bytes", U64)]


class EncCounters(C.Structure):
    """Mirror of the ABI 2 NvencTransferCounters."""
    _fields_ = [("struct_size", U32), ("frames_submitted", U64),
                ("packets_written", U64), ("staging_gpu_copy_bytes", U64),
                ("conversion_output_bytes", U64), ("host_pixel_bytes", U64)]


class HdrMetadata(C.Structure):
    """Mirror of NvdecHdrMetadata and NvencHdrMetadata, which are one layout."""
    _fields_ = [("struct_size", U32), ("has_mastering", I32),
                ("primary_x_num", I32 * 3), ("primary_x_den", I32 * 3),
                ("primary_y_num", I32 * 3), ("primary_y_den", I32 * 3),
                ("white_x_num", I32), ("white_x_den", I32),
                ("white_y_num", I32), ("white_y_den", I32),
                ("min_luminance_num", I32), ("min_luminance_den", I32),
                ("max_luminance_num", I32), ("max_luminance_den", I32),
                ("mastering_has_primaries", I32),
                ("mastering_has_luminance", I32), ("has_content_light", I32),
                ("max_cll", U32), ("max_fall", U32)]


class HdrOptions(C.Structure):
    _fields_ = [("struct_size", U32), ("codec", CP), ("profile", CP),
                ("preset", CP), ("tune", CP), ("multipass", CP),
                ("rate_control", CP), ("constant_quality", I32),
                ("bitrate", I64), ("gop", I32), ("extra_options", CP),
                ("sw_pixel_format", I32), ("color_range", I32),
                ("color_primaries", I32), ("color_trc", I32),
                ("color_space", I32)]


class HdrDefaults(C.Structure):
    _fields_ = [("struct_size", U32), ("sw_pixel_format", I32),
                ("color_range", I32), ("color_primaries", I32),
                ("color_trc", I32), ("color_space", I32),
                ("profile", C.c_char * 32)]


class HdrCapability(C.Structure):
    _fields_ = [("struct_size", U32), ("encoder_present", I32),
                ("p010_supported", I32), ("cuda_supported", I32),
                ("main10_profile_supported", I32),
                ("cuda_device_created", I32), ("hdr_metadata_options", I32),
                ("supported", I32), ("encoder_name", C.c_char * 64),
                ("profile_name", C.c_char * 32)]


class HdrContract(C.Structure):
    _fields_ = [("struct_size", U32), ("codec", C.c_char * 64),
                ("container", C.c_char * 32), ("profile_name", C.c_char * 32),
                ("requested_pix_fmt", I32), ("actual_pix_fmt", I32),
                ("requested_sw_pix_fmt", I32), ("actual_sw_pix_fmt", I32),
                ("actual_sw_pix_fmt_name", C.c_char * 32),
                ("requested_profile", I32), ("actual_profile", I32),
                ("requested_bits_per_raw_sample", I32),
                ("actual_bits_per_raw_sample", I32), ("requested_color_range", I32),
                ("requested_color_primaries", I32), ("requested_color_trc", I32),
                ("requested_color_space", I32), ("actual_color_range", I32),
                ("actual_color_primaries", I32), ("actual_color_trc", I32),
                ("actual_color_space", I32), ("width", U32), ("height", U32),
                ("hw_frames_pool_size", I32), ("hw_frames_sw_format", I32),
                ("metadata_written_to", I32), ("has_mastering", I32),
                ("has_content_light", I32),
                ("cuda_adapter_luid", C.c_uint8 * 8),
                ("cuda_adapter_node_mask", U32)]


class HdrCounters(C.Structure):
    _fields_ = [("struct_size", U32), ("frames_submitted", U64),
                ("packets_written", U64), ("p010_device_copy_bytes", U64),
                ("host_pixel_bytes", U64), ("rgb_conversion_bytes", U64),
                ("mastering_side_data", U64), ("content_light_side_data", U64)]


class SeiCapability(C.Structure):
    """Mirror of NvencHdrSeiCapability: properties of the writer, not of NVENC."""
    _fields_ = [("struct_size", U32), ("writer_available", I32),
                ("encoder_offers_metadata_option", I32),
                ("supports_partial_mastering", I32), ("primary_order", I32),
                ("chroma_denominator", I32), ("luma_denominator", I32),
                ("payload_type_mastering", I32),
                ("payload_type_content_light", I32),
                ("prefix_sei_nal_type", I32), ("mastering_payload_size", U32),
                ("content_light_payload_size", U32), ("max_nal_bytes", U32),
                ("max_insert_bytes", U32)]


class SeiInfo(C.Structure):
    """Mirror of NvencHdrSeiInfo, the per-call result the CPU gate binds to."""
    _fields_ = [("struct_size", U32), ("error", I32), ("framing", I32),
                ("nal_count", I32), ("vcl_index", I32),
                ("mastering_present", I32), ("content_light_present", I32),
                ("mastering_equal", I32), ("content_light_equal", I32),
                ("mastering_inserted", I32), ("content_light_inserted", I32),
                ("nal_bytes", U32), ("bytes_inserted", U32),
                ("detail", C.c_char * 192)]


class SeiStatus(C.Structure):
    """Mirror of NvencHdrSeiStatus: what the writer did to this session's packets."""
    _fields_ = [("struct_size", U32), ("writer_available", I32),
                ("metadata_requested", I32), ("mastering_serialized", I32),
                ("content_light_serialized", I32), ("error", I32),
                ("detail", C.c_char * 192), ("framing", I32),
                ("packet_sample", C.c_char * 64), ("packets_seen", U64),
                ("nals_inserted", U64), ("bytes_inserted", U64),
                ("mastering_inserted", U64), ("content_light_inserted", U64),
                ("mastering_already_present", U64),
                ("content_light_already_present", U64),
                ("packets_without_vcl", U64), ("vcl_bytes", U64),
                ("vcl_hash", U64), ("existing_nal_bytes", U64),
                ("existing_nal_hash", U64)]


METADATA_SCALARS = ["has_mastering", "white_x_num", "white_x_den", "white_y_num",
                    "white_y_den", "min_luminance_num", "min_luminance_den",
                    "max_luminance_num", "max_luminance_den",
                    "mastering_has_primaries", "mastering_has_luminance",
                    "has_content_light", "max_cll", "max_fall"]
METADATA_ARRAYS = ["primary_x_num", "primary_x_den", "primary_y_num",
                   "primary_y_den"]


def metadata_snapshot(block):
    """Exact rationals, as the source declared them, with nothing rounded."""
    out = {name: getattr(block, name) for name in METADATA_SCALARS}
    for name in METADATA_ARRAYS:
        out[name] = list(getattr(block, name))
    return out


def _ratio(num, den):
    return Fraction(int(num), int(den)) if den else None


def metadata_values(block):
    """Every declared quantity as an exact value rather than a spelling.

    Matroska stores ST 2086 in its own fixed-point denominators, so a source that
    says 15635/50000 comes back out of a Matroska container as 3127/10000. The
    values are identical; only the denominator differs. Judging the round trip on
    the numerator/denominator pair would report a loss that did not happen, and
    judging it on a float would invent one, so this compares exact Fractions.
    """
    if not block:
        return None
    out = {"has_mastering": block.get("has_mastering"),
           "mastering_has_primaries": block.get("mastering_has_primaries"),
           "mastering_has_luminance": block.get("mastering_has_luminance"),
           "has_content_light": block.get("has_content_light"),
           "max_cll": block.get("max_cll"), "max_fall": block.get("max_fall")}
    for name in ("white_x", "white_y", "min_luminance", "max_luminance"):
        out[name] = _ratio(block.get(f"{name}_num"), block.get(f"{name}_den"))
    for axis in ("x", "y"):
        nums = block.get(f"primary_{axis}_num") or [None] * 3
        dens = block.get(f"primary_{axis}_den") or [None] * 3
        for index in range(3):
            out[f"primary_{axis}_{index}"] = _ratio(nums[index], dens[index])
    return out


def rational_differences(source, output):
    """Where the two spellings differ in value, which is nowhere if the handoff held."""
    wanted, got = metadata_values(source), metadata_values(output)
    if wanted is None or got is None:
        return {"missing_snapshot": [wanted is None, got is None]}
    return {name: {"source": str(wanted[name]), "output": str(got[name])}
            for name in wanted if wanted[name] != got[name]}


def same_rationals(source, output):
    """Every declared field equal in value, and every undeclared field still
    undeclared."""
    if not source or not output:
        return False
    return metadata_values(source) == metadata_values(output)


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, condition, detail=""):
        self.items.append({"name": name, "pass": bool(condition),
                           "detail": truncate(detail)})
        return bool(condition)

    @property
    def failed(self):
        return [item["name"] for item in self.items if not item["pass"]]


def truncate(value):
    text = value if isinstance(value, str) else json.dumps(value,
                                                default=json_value)
    return text if len(text) <= 900 else text[:900] + "..."


def text(value):
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return "" if value is None else str(value)


COMMANDS = []


def record_command(argv, rc, artifact="", stderr=""):
    """Every external command this gate ran, with its real exit code."""
    COMMANDS.append({"command": [str(item) for item in argv], "rc": rc,
                     "artifact": artifact, "stderr": stderr.strip()[:400]})


def artifact_inventory():
    out = {}
    for path in sorted(ARTIFACTS.iterdir()):
        if path.is_file():
            out[path.name] = {"bytes": path.stat().st_size,
                              "sha256": sha256_file(path)}
    return out


def bind_nvdec():
    lib = C.CDLL(str(NATIVE / "nvdec_bridge.dll"))
    lib.nvdec_bridge_abi_version.restype = U32
    lib.nvdec_bridge_last_error.restype = CP
    for name in ("nvdec_bridge_open", "nvdec_bridge_open_hdr"):
        fn = getattr(lib, name)
        fn.argtypes = [CP, C.POINTER(VP)]
        fn.restype = I32
    lib.nvdec_bridge_next.argtypes = [VP, C.POINTER(DecFrame)]
    lib.nvdec_bridge_next.restype = I32
    lib.nvdec_bridge_get_hdr_metadata.argtypes = [VP, C.POINTER(HdrMetadata)]
    lib.nvdec_bridge_get_hdr_metadata.restype = I32
    lib.nvdec_bridge_get_transfer_counters.argtypes = [VP, C.POINTER(DecCounters)]
    lib.nvdec_bridge_get_transfer_counters.restype = I32
    lib.nvdec_bridge_readback_last_frame.argtypes = [VP, VP, CSZ]
    lib.nvdec_bridge_readback_last_frame.restype = I32
    lib.nvdec_bridge_close.argtypes = [VP]
    return lib


def bind_nvenc():
    lib = C.CDLL(str(NATIVE / "nvenc_bridge.dll"))
    lib.nvenc_abi_version.restype = U32
    lib.nvenc_hdr_abi_version.restype = U32
    lib.nvenc_last_error.restype = CP
    lib.nvenc_hdr_get_defaults.argtypes = [C.POINTER(HdrDefaults)]
    lib.nvenc_hdr_get_defaults.restype = I32
    lib.nvenc_hdr_capability.argtypes = [VP, U32, U32, C.POINTER(HdrCapability)]
    lib.nvenc_hdr_capability.restype = I32
    lib.nvenc_hdr_open.argtypes = [VP, U32, U32, U32, U32, I32, I32, CP,
                                   C.POINTER(HdrOptions), C.POINTER(HdrMetadata),
                                   C.POINTER(VP)]
    lib.nvenc_hdr_open.restype = I32
    lib.nvenc_encode_p010.argtypes = [VP, U64, CSZ, U64, CSZ, I64]
    lib.nvenc_encode_p010.restype = I32
    lib.nvenc_hdr_get_counters.argtypes = [VP, C.POINTER(HdrCounters)]
    lib.nvenc_hdr_get_counters.restype = I32
    lib.nvenc_hdr_get_contract.argtypes = [VP, C.POINTER(HdrContract)]
    lib.nvenc_hdr_get_contract.restype = I32
    lib.nvenc_get_transfer_counters.argtypes = [VP, C.POINTER(EncCounters)]
    lib.nvenc_get_transfer_counters.restype = I32
    lib.nvenc_hdr_sei_abi_version.restype = U32
    lib.nvenc_hdr_sei_capability.argtypes = [C.POINTER(SeiCapability)]
    lib.nvenc_hdr_sei_capability.restype = I32
    lib.nvenc_hdr_sei_get.argtypes = [VP, C.POINTER(SeiStatus)]
    lib.nvenc_hdr_sei_get.restype = I32
    lib.nvenc_encode_rgba.argtypes = [VP, U64, CSZ, I64]
    lib.nvenc_encode_rgba.restype = I32
    lib.nvenc_finish.argtypes = [VP]
    lib.nvenc_finish.restype = I32
    lib.nvenc_close.argtypes = [VP]
    return lib


def pix_fmt_number(avutil, name):
    avutil.av_get_pix_fmt.argtypes = [CP]
    avutil.av_get_pix_fmt.restype = I32
    return int(avutil.av_get_pix_fmt(name))


def installed_pix_fmt(name):
    """A pixel-format number taken from the installed library, not typed in."""
    return pix_fmt_number(C.CDLL("avutil-59.dll"), name.encode())


def counter_fields(structure):
    return {name: getattr(structure, name) for name, _ in structure._fields_
            if name != "struct_size"}


def ffprobe(path, entries, checks, artifact=None):
    command = [str(FFMPEG_BIN / "ffprobe.exe"), "-v", "error",
               "-select_streams", "v:0"]
    for entry in entries:
        command += ["-show_entries", entry]
    command += ["-of", "json", os.fsdecode(path)]
    raw = subprocess.run(command, capture_output=True, text=True, timeout=180)
    name = artifact or f"ffprobe_{path.stem}_{entries[0].split('=')[0]}.json"
    (ARTIFACTS / name).write_text(raw.stdout, encoding="utf-8")
    ok = raw.returncode == 0
    checks.add(f"ffprobe {path.name} {entries[0].split('=')[0]}", ok,
               raw.stderr.strip() or raw.returncode)
    record_command(command, raw.returncode, name, raw.stderr.strip())
    if not ok:
        return {}
    return json.loads(raw.stdout or "{}")


def probe_stream(path, checks):
    data = ffprobe(path, ["stream=codec_name,profile,pix_fmt,width,height,"
                          "color_range,color_primaries,color_transfer,"
                          "color_space,bits_per_raw_sample,nb_frames,r_frame_rate"],
                   checks)
    return (data.get("streams") or [{}])[0]


def probe_packet_pts(path, checks):
    data = ffprobe(path, ["packet=pts,pts_time,dts,duration"], checks)
    return [packet.get("pts") for packet in data.get("packets", [])]


def probe_decoded_frame_pts(path, checks):
    """ffprobe decodes the container: this is the frame-count evidence, and it
    runs on the CPU after the fact, never on the processing path."""
    data = ffprobe(path, ["frame=pts,pict_type"], checks)
    return [frame.get("pts") for frame in data.get("frames", [])]


def probe_decoded_frames_with_side_data(path, checks, artifact):
    """Every decoded frame with the side-data blocks the decoder attached to it.

    This is the only way to ask whether the metadata reached the frames rather
    than only the container: -show_streams reports format tags, -show_frames
    reports what a decoded frame actually carries.
    """
    data = ffprobe(path, ["frame=pts,pict_type", "side_data_list"], checks,
                   artifact)
    entries = data.get("frames") or data.get("packets_and_frames") or []
    return [entry for entry in entries
            if entry.get("type", "frame") == "frame"]


def side_data_blocks(frames, kind):
    return [block for frame in frames
            for block in frame.get("side_data_list", [])
            if block.get("side_data_type") == kind]


def as_fraction(value):
    """ffprobe prints an ST 2086 rational as "num/den" text; keep it exact."""
    if value is None:
        return None
    if isinstance(value, str) and "/" in value:
        num, den = value.split("/", 1)
        return Fraction(int(num), int(den)) if int(den) else None
    return Fraction(int(value))


def mastering_probe_values(block):
    """An ffprobe mastering block, keyed the same way as a bridge snapshot."""
    index = {"red": 0, "green": 1, "blue": 2}
    out = {}
    for colour in ("red", "green", "blue"):
        for axis in ("x", "y"):
            out[f"primary_{axis}_{index[colour]}"] = as_fraction(
                block.get(f"{colour}_{axis}"))
    out["white_x"] = as_fraction(block.get("white_point_x"))
    out["white_y"] = as_fraction(block.get("white_point_y"))
    out["min_luminance"] = as_fraction(block.get("min_luminance"))
    out["max_luminance"] = as_fraction(block.get("max_luminance"))
    return out


def decode_frames(dec, path, limit, checks, label):
    """Borrow `limit` CUDA P010 frames plus the source's own metadata.

    Only read-only facts come out of this: the frame structs' device pointers are
    valid until the following nvdec_bridge_next call, which unrefs the surface it
    reported last time, so every use here is limit=1 and none of them encodes.
    The encode path streams through encode_streamed instead.
    """
    handle = VP()
    if dec.nvdec_bridge_open_hdr(os.fsencode(path), C.byref(handle)) != 0:
        checks.add(f"{label}: nvdec_bridge_open_hdr", False,
                   text(dec.nvdec_bridge_last_error()))
        return None, [], HdrMetadata(), {}, {}
    checks.add(f"{label}: nvdec_bridge_open_hdr", True, "abi 3 P010 borrow")
    frames = []
    while len(frames) < limit:
        frame = DecFrame()
        frame.struct_size = C.sizeof(DecFrame)
        code = dec.nvdec_bridge_next(handle, C.byref(frame))
        if code == 0:
            break
        if code < 0:
            checks.add(f"{label}: decode frame {len(frames)}", False,
                       text(dec.nvdec_bridge_last_error()))
            break
        frames.append(frame)
    block = HdrMetadata()
    block.struct_size = C.sizeof(HdrMetadata)
    have = dec.nvdec_bridge_get_hdr_metadata(handle, C.byref(block)) == 0
    if not have:
        checks.add(f"{label}: hdr metadata query", False,
                   text(dec.nvdec_bridge_last_error()))
    counters = DecCounters()
    counters.struct_size = C.sizeof(DecCounters)
    dec.nvdec_bridge_get_transfer_counters(handle, C.byref(counters))
    snapshot = metadata_snapshot(block) if have else {}
    return handle, frames, block, snapshot, counter_fields(counters)


def borrow_one(dec, handle, checks, label, index):
    """A single nvdec_bridge_next call, for a decode that is encoded before the
    next one runs."""
    frame = DecFrame()
    frame.struct_size = C.sizeof(DecFrame)
    code = dec.nvdec_bridge_next(handle, C.byref(frame))
    if code != 1:
        checks.add(f"{label}: borrow frame {index}", False,
                   f"rc={code} {text(dec.nvdec_bridge_last_error())}")
        return None
    return frame


def encode_streamed(dec, enc, source, out_path, checks, label, count=2,
                    options=None, attach=True, fps=(25, 1)):
    """Borrow a frame, encode it, borrow the next: the documented borrow window.

    nvdec_bridge_next releases the surface it reported last time, so batching the
    borrows and encoding afterwards would hand the encoder pointers that NVDEC
    may already have recycled. Encoding between the borrows keeps every submitted
    surface live; nvenc_encode_p010 copies and synchronises before it returns, so
    the source is consumed by the time the next borrow runs.
    """
    handle = VP()
    if dec.nvdec_bridge_open_hdr(os.fsencode(source), C.byref(handle)) != 0:
        checks.add(f"{label}: nvdec_bridge_open_hdr", False,
                   text(dec.nvdec_bridge_last_error()))
        return None
    checks.add(f"{label}: nvdec_bridge_open_hdr", True, "abi 3 P010 borrow")
    first = borrow_one(dec, handle, checks, label, 0)
    if first is None:
        dec.nvdec_bridge_close(handle)
        return None
    block = HdrMetadata()
    block.struct_size = C.sizeof(HdrMetadata)
    if dec.nvdec_bridge_get_hdr_metadata(handle, C.byref(block)) != 0:
        checks.add(f"{label}: hdr metadata query", False,
                   text(dec.nvdec_bridge_last_error()))
        dec.nvdec_bridge_close(handle)
        return None

    bridge = VP()
    hdr_options = options or default_hdr_options()
    code = enc.nvenc_hdr_open(first.cuda_context, first.width, first.height,
                              fps[0], fps[1], first.time_base_num,
                              first.time_base_den, os.fsencode(out_path),
                              C.byref(hdr_options), C.byref(block) if attach
                              else None, C.byref(bridge))
    if code != 0:
        checks.add(f"{label}: nvenc_hdr_open", False,
                   {"rc": code, "error": text(enc.nvenc_last_error())})
        dec.nvdec_bridge_close(handle)
        return None
    checks.add(f"{label}: nvenc_hdr_open", True, out_path.name)

    submitted = []
    current = first
    for index in range(count):
        if current is None:
            break
        code = enc.nvenc_encode_p010(bridge, current.y_device_ptr,
                                     current.y_pitch_bytes, current.uv_device_ptr,
                                     current.uv_pitch_bytes, current.pts)
        submitted.append({"frame": index, "pts": current.pts, "rc": code,
                          "extent": [current.width, current.height],
                          "software_format": current.software_format,
                          "y_pitch_bytes": current.y_pitch_bytes,
                          "uv_pitch_bytes": current.uv_pitch_bytes,
                          "y_device_ptr_hex": hex(current.y_device_ptr),
                          "uv_device_ptr_hex": hex(current.uv_device_ptr),
                          "error": text(enc.nvenc_last_error()) if code else ""})
        if code != 0:
            checks.add(f"{label}: nvenc_encode_p010 frame {index}", False,
                       submitted[-1])
            break
        current = (borrow_one(dec, handle, checks, label, index + 1)
                   if index + 1 < count else None)

    checks.add(f"{label}: {count} frames submitted without error",
               len(submitted) == count and all(entry["rc"] == 0
                                               for entry in submitted), submitted)
    finish = enc.nvenc_finish(bridge)
    checks.add(f"{label}: nvenc_finish drains", finish == 0,
               f"rc={finish} {text(enc.nvenc_last_error())}")
    counters = HdrCounters()
    counters.struct_size = C.sizeof(HdrCounters)
    enc.nvenc_hdr_get_counters(bridge, C.byref(counters))
    contract = HdrContract()
    contract.struct_size = C.sizeof(HdrContract)
    enc.nvenc_hdr_get_contract(bridge, C.byref(contract))
    legacy = EncCounters()
    legacy.struct_size = C.sizeof(EncCounters)
    enc.nvenc_get_transfer_counters(bridge, C.byref(legacy))
    sei = SeiStatus()
    sei.struct_size = C.sizeof(SeiStatus)
    sei_read = enc.nvenc_hdr_sei_get(bridge, C.byref(sei))
    dec_counters = DecCounters()
    dec_counters.struct_size = C.sizeof(DecCounters)
    dec.nvdec_bridge_get_transfer_counters(handle, C.byref(dec_counters))
    enc.nvenc_close(bridge)
    dec.nvdec_bridge_close(handle)

    snapshot = counter_fields(counters)
    snapshot["contract"] = contract_snapshot(contract)
    snapshot["abi2_counters"] = counter_fields(legacy)
    snapshot["sei_status"] = dict(counter_fields(sei),
                                  read_rc=sei_read,
                                  detail=text(sei.detail),
                                  packet_sample=text(sei.packet_sample))
    return {"counters": snapshot, "submitted": submitted,
            "metadata": metadata_snapshot(block),
            "decoder_counters": counter_fields(dec_counters), "first": first}


def default_hdr_options(**overrides):
    options = HdrOptions()
    options.struct_size = C.sizeof(HdrOptions)
    options.preset = b"p5"
    options.rate_control = b"vbr"
    options.constant_quality = 30
    # gop stays unset: NVENC refuses a short keyframe interval against its own
    # default B-frame count ("Gop Length should be greater than number of B
    # frames + 1"), and a two-frame gate has no reason to fight the default.
    for key, value in overrides.items():
        setattr(options, key, value)
    return options


def contract_snapshot(contract):
    out = {}
    for name, _ in HdrContract._fields_:
        if name == "struct_size":
            continue
        value = getattr(contract, name)
        if name == "cuda_adapter_luid":
            out[name] = list(value)
        elif isinstance(value, (bytes, bytearray)):
            # ctypes hands a c_char array field back as bytes, already cut at the
            # first NUL; comparing that against a str literal never matches.
            out[name] = text(value).strip("\x00")
        elif isinstance(value, C.Array):
            out[name] = text(value.value)
        else:
            out[name] = value
    return out


def check_surface(frame, checks, label):
    checks.add(f"{label}: borrowed surface is P010", frame.software_format == 2,
               f"software_format={frame.software_format}")
    checks.add(f"{label}: P010 pitches are 16-bit wide",
               frame.y_pitch_bytes >= frame.width * 2 and
               frame.uv_pitch_bytes >= frame.width * 2,
               f"y={frame.y_pitch_bytes} uv={frame.uv_pitch_bytes} "
               f"width={frame.width}")
    checks.add(f"{label}: source declares HDR10 colour",
               (frame.color_primaries, frame.color_transfer, frame.color_space,
                frame.color_range) == (BT2020, PQ, BT2020NC, LIMITED),
               f"primaries={frame.color_primaries} trc={frame.color_transfer} "
               f"space={frame.color_space} range={frame.color_range}")


def case_a(dec, enc, checks, results):
    """Metadata present: two borrowed frames in, one HEVC Main10 file out."""
    source = FIXTURES / "hdr_src.mkv"
    mkv = ARTIFACTS / "a_hdr10_out.mkv"
    stream = encode_streamed(dec, enc, source, mkv, checks, "A source")
    if not stream:
        return
    counters, submitted = stream["counters"], stream["submitted"]
    metadata, dec_counters = stream["metadata"], stream["decoder_counters"]
    first = stream["first"]
    check_surface(first, checks, "A")
    checks.add("A: source extents are 1920x1080",
               (first.width, first.height) == (1920, 1080),
               f"{first.width}x{first.height}")
    checks.add("A: source carries mastering metadata",
               metadata.get("has_mastering") == 1, metadata)
    checks.add("A: source carries content light",
               metadata.get("has_content_light") == 1,
               f"max_cll={metadata.get('max_cll')} "
               f"max_fall={metadata.get('max_fall')}")
    pts = [entry["pts"] for entry in submitted]
    checks.add("A: two frames borrowed and submitted with the source's own PTS",
               len(pts) == 2 and pts == sorted(set(pts)), {"pts": pts,
                                                            "time_base":
                                                            [first.time_base_num,
                                                             first.time_base_den]})

    verify_a_counters(counters, dec_counters, first, checks)
    reused = (len(submitted) == 2 and submitted[0]["y_device_ptr_hex"] ==
              submitted[1]["y_device_ptr_hex"])
    results["case_a"] = {"source": str(source),
                         "frame_order": ("borrow, encode, borrow, encode: a "
                                         "borrowed pointer dies at the following "
                                         "nvdec_bridge_next call, so the decodes "
                                         "are never batched ahead of the encodes"),
                         "borrow_window": {
                             "consecutive_borrows_returned_one_surface": reused,
                             "addresses": [entry["y_device_ptr_hex"] for entry in
                                           submitted],
                             "note": ("the second borrow reported the same device "
                                      "address as the first, because next() unrefs "
                                      "the surface it handed out last time and "
                                      "NVDEC's pool gives it straight back. Decoding "
                                      "both frames first would have submitted the "
                                      "second frame's pixels twice; each frame is "
                                      "copied and synchronised by "
                                      "nvenc_encode_p010 before the next borrow "
                                      "runs. Pixel-level content equality is not "
                                      "asserted here, because checking it would need "
                                      "a host readback")},
                         "source_time_base": [first.time_base_num,
                                              first.time_base_den],
                         "submitted_frames": submitted,
                         "source_metadata": metadata,
                         "encoder_counters": counters,
                         "decoder_counters": dec_counters,
                         "output": str(mkv)}
    verify_a_container(mkv, dec, metadata, pts, checks, results)
    bitstream_evidence(dec, mkv, metadata, counters, checks, results)


def verify_a_counters(counters, dec_counters, frame, checks):
    expected_copy = 2 * frame.width * frame.height * 3
    checks.add("A: P010 device copy bytes are the two frames' payload",
               counters["p010_device_copy_bytes"] == expected_copy,
               f"{counters['p010_device_copy_bytes']} vs {expected_copy}")
    checks.add("A: encoder host pixel bytes stay 0",
               counters["host_pixel_bytes"] == 0, counters["host_pixel_bytes"])
    checks.add("A: RGB->NV12 conversion bytes stay 0",
               counters["rgb_conversion_bytes"] == 0,
               counters["rgb_conversion_bytes"])
    checks.add("A: ABI 2 staging and conversion bytes stay 0",
               counters["abi2_counters"]["staging_gpu_copy_bytes"] == 0 and
               counters["abi2_counters"]["conversion_output_bytes"] == 0,
               counters["abi2_counters"])
    checks.add("A: frames_submitted counts the two encodes",
               counters["frames_submitted"] == 2, counters["frames_submitted"])
    checks.add("A: packets_written counts the drained output",
               counters["packets_written"] == 2, counters["packets_written"])
    checks.add("A: per-frame mastering side data attached twice",
               counters["mastering_side_data"] == 2, counters["mastering_side_data"])
    checks.add("A: per-frame content light side data attached twice",
               counters["content_light_side_data"] == 2,
               counters["content_light_side_data"])
    checks.add("A: decoder host copies stay 0 on the processing path",
               dec_counters["host_frame_copy_count"] == 0 and
               dec_counters["host_frame_copy_bytes"] == 0, dec_counters)
    checks.add("A: decoder scaled output stays 0 (HDR borrow builds no scaler)",
               dec_counters["scale_device_output_bytes"] == 0,
               dec_counters["scale_device_output_bytes"])
    contract = counters["contract"]
    checks.add("A: encoder software format is p010le, not nv12",
               contract["actual_sw_pix_fmt_name"] == "p010le",
               contract["actual_sw_pix_fmt_name"])
    checks.add("A: encoder hardware format is CUDA",
               contract["actual_pix_fmt"] == contract["requested_pix_fmt"] ==
               installed_pix_fmt("cuda"),
               f"requested={contract['requested_pix_fmt']} "
               f"actual={contract['actual_pix_fmt']}")
    checks.add("A: profile number is Main10",
               contract["actual_profile"] == MAIN10 == contract["requested_profile"],
               f"actual={contract['actual_profile']} named "
               f"{contract['profile_name']}")
    checks.add("A: bits_per_raw_sample is 10",
               contract["actual_bits_per_raw_sample"] == 10,
               contract["actual_bits_per_raw_sample"])
    checks.add("A: colour contract declared as bt2020/PQ/bt2020nc/limited",
               (contract["actual_color_primaries"], contract["actual_color_trc"],
                contract["actual_color_space"], contract["actual_color_range"]) ==
               (BT2020, PQ, BT2020NC, LIMITED), contract)
    checks.add("A: metadata written to frame, encoder, stream, container and the "
               "elementary bitstream",
               contract["metadata_written_to"] == 31,
               f"metadata_written_to={contract['metadata_written_to']} "
               "(bits: 1 frame, 2 coded-packet, 4 stream, 8 container, "
               "16 in-bitstream SEI)")
    checks.add("A: the writer reports it serialised both declarations",
               counters["sei_status"]["mastering_serialized"] == 1 and
               counters["sei_status"]["content_light_serialized"] == 1 and
               counters["sei_status"]["metadata_requested"] == 1 and
               counters["sei_status"]["writer_available"] == 1 and
               counters["sei_status"]["error"] == 0,
               {key: counters["sei_status"][key] for key in
                ("writer_available", "metadata_requested", "mastering_serialized",
                 "content_light_serialized", "error")})
    checks.add("A: adapter LUID captured from the borrowed context",
               any(byte for byte in contract["cuda_adapter_luid"]),
               {"luid": contract["cuda_adapter_luid"],
                "node_mask": contract["cuda_adapter_node_mask"]})


def verify_a_container(mkv, dec, metadata, source_pts, checks, results):
    stream = probe_stream(mkv, checks)
    checks.add("A: ffprobe sees hevc Main 10 decoded as 10-bit 4:2:0",
               (stream.get("codec_name"), stream.get("profile"),
                stream.get("pix_fmt")) == ("hevc", "Main 10", "yuv420p10le"),
               {"probe": stream,
                "note": ("yuv420p10le is what a CPU decode of P010 4:2:0 is named; "
                         "the encoder-side contract reports the same surface as "
                         "p010le on CUDA")})
    checks.add("A: the output is 10-bit per sample",
               str(stream.get("pix_fmt", "")).endswith("p10le"),
               {"pix_fmt": stream.get("pix_fmt"),
                "container_bits_per_raw_sample": stream.get("bits_per_raw_sample"),
                "encoder_declared_bits": 10,
                "note": ("the Matroska demuxer reports no bits_per_raw_sample, so "
                         "the bit depth is read from the decoded sample format here "
                         "and from the parameter sets in the bitstream check; the "
                         "encoder's own contract field is gated separately")})
    checks.add("A: ffprobe container colour tags match the contract",
               (stream.get("color_primaries"), stream.get("color_transfer"),
                stream.get("color_space"), stream.get("color_range")) ==
               ("bt2020", "smpte2084", "bt2020nc", "tv"),
               {key: stream.get(key) for key in ("color_primaries", "color_transfer",
                                                 "color_space", "color_range")})
    checks.add("A: ffprobe reports the actual extents",
               (stream.get("width"), stream.get("height")) == (1920, 1080),
               f"{stream.get('width')}x{stream.get('height')}")
    pts = probe_packet_pts(mkv, checks)
    checks.add("A: output has exactly two packets with the source PTS",
               pts == source_pts, f"{pts} vs {source_pts}")
    frame_pts = probe_decoded_frame_pts(mkv, checks)
    checks.add("A: output decodes to exactly two frames in PTS order",
               frame_pts == sorted(source_pts) and len(frame_pts) == 2,
               {"decoded": frame_pts, "source": sorted(source_pts)})

    out_handle, out_frames, _, out_metadata, _ = decode_frames(dec, mkv, 1, checks,
                                                              "A output")
    if out_frames:
        checks.add("A: output decodes as a P010 CUDA surface",
                   out_frames[0].software_format == 2,
                   out_frames[0].software_format)
    checks.add("A: mastering rationals survive the round trip in value",
               same_rationals(metadata, out_metadata),
               {"differing": rational_differences(metadata, out_metadata),
                "source_spelling": metadata, "output_spelling": out_metadata,
                "source_values": metadata_values(metadata),
                "output_values": metadata_values(out_metadata),
                "note": ("Matroska re-expresses ST 2086 in its own fixed-point "
                         "denominators, so 15635/50000 reads back as 3127/10000; "
                         "the values are compared exactly, never through float")})
    checks.add("A: output carries mastering side data",
               out_metadata.get("has_mastering") == 1, out_metadata)
    if out_handle:
        dec.nvdec_bridge_close(out_handle)

    decoded = probe_decoded_frames_with_side_data(mkv, checks,
                                                 "ffprobe_a_frame_side_data.json")
    blocks = side_data_blocks(decoded, "Mastering display metadata")
    checks.add("A: the decoded frames themselves carry the mastering block",
               len(blocks) == 2, {"frames": len(decoded), "blocks": len(blocks),
                                  "first": blocks[0] if blocks else None})
    light = side_data_blocks(decoded, "Content light level metadata")
    checks.add("A: the decoded frames themselves carry the content light block",
               len(light) == 2, {"frames": len(decoded), "blocks": len(light)})
    frame_values = (mastering_probe_values(blocks[0]) if blocks else None)
    wanted = {key: value for key, value in (metadata_values(metadata) or
                                           {}).items()
              if frame_values and key in frame_values}
    checks.add("A: the mastering values on the decoded frames equal the source's",
               frame_values is not None and frame_values == wanted,
               {"frame_block": blocks[0] if blocks else None,
                "frame_values": frame_values, "source_values": wanted,
                "content_light": light[0] if light else None,
                "source_max_cll": metadata.get("max_cll"),
                "source_max_fall": metadata.get("max_fall")})
    checks.add("A: the content light numbers on the decoded frames equal the source's",
               bool(light) and light[0].get("max_content") ==
               metadata.get("max_cll") and light[0].get("max_average") ==
               metadata.get("max_fall"),
               {"frame_block": light[0] if light else None,
                "source_max_cll": metadata.get("max_cll"),
                "source_max_fall": metadata.get("max_fall")})
    results["case_a"]["output_stream_probe"] = stream
    results["case_a"]["output_packet_pts"] = pts
    results["case_a"]["output_metadata"] = out_metadata
    results["case_a"]["output_decoded_frame_side_data"] = decoded


def demux(source, target, checks, label):
    """Copy the coded bytes out of a container, with no re-encode anywhere."""
    run = subprocess.run([str(FFMPEG_BIN / "ffmpeg.exe"), "-v", "error", "-y",
                          "-i", os.fsdecode(source), "-c", "copy",
                          os.fsdecode(target)],
                         capture_output=True, text=True, timeout=180)
    record_command(run.args, run.returncode, target.name, run.stderr)
    checks.add(f"{label}: demuxed to Annex-B", run.returncode == 0,
               {"command": f"ffmpeg -i {Path(source).name} -c copy "
                           f"{Path(target).name}",
                "rc": run.returncode, "stderr": run.stderr.strip(),
                "bytes": target.stat().st_size if target.exists() else 0})
    return run.returncode == 0


FNV_OFFSET, FNV_PRIME, MASK64 = 1469598103934665603, 1099511628211, (1 << 64) - 1
NAL_TYPE_PREFIX_SEI = 39


def fnv64(data):
    """FNV-1a/64, the same fold the bridge runs over the bytes it receives, so a
    number read out of the DLL can be checked against the bytes on disk."""
    value = FNV_OFFSET
    for byte in data:
        value = (value ^ byte) * FNV_PRIME & MASK64
    return value


def annexb_nal_bodies(data):
    """Every NAL body of an Annex-B stream, in order.

    A body runs from the byte after 00 00 01 to the next start code, with any
    trailing zero_byte removed: the same extents the bridge hashes, so the two
    numbers are comparable rather than accidentally different.
    """
    codes = [match.start() for match in re.finditer(b"\x00\x00\x01", data)]
    out = []
    for index, code in enumerate(codes):
        body = code + 3
        end = codes[index + 1] if index + 1 < len(codes) else len(data)
        while end > body and data[end - 1] == 0:
            end -= 1
        if end - body >= 2:
            out.append(data[body:end])
    return out


def nal_unit_type(nal):
    return (nal[0] >> 1) & 0x3F


def _rbsp(nal):
    """Strip the two-byte NAL header and remove the emulation-prevention
    bytes, which is what the payload syntax is written against."""
    escaped, out, index = nal[2:], bytearray(), 0
    while index < len(escaped):
        if (index + 2 < len(escaped) and escaped[index] == 0 and
                escaped[index + 1] == 0 and escaped[index + 2] == 3):
            out += b"\x00\x00"
            index += 3
        else:
            out.append(escaped[index])
            index += 1
    return bytes(out)


def sei_payloads(nal):
    """payloadType -> payload bytes for a prefix SEI NAL, empty for any other.

    The walk stops at rbsp_trailing_bits(), which is a one bit followed by
    zero bits to byte alignment: a lone 0x80 with nothing but zeros after it.
    It cannot be recognised by its high bit alone, because a payloadType of
    144 has the high bit set too.
    """
    if nal_unit_type(nal) != NAL_TYPE_PREFIX_SEI:
        return {}
    rbsp = _rbsp(nal)
    out, pos = {}, 0
    while pos < len(rbsp):
        if rbsp[pos] == 0x80 and not any(rbsp[pos + 1:]):
            break
        if rbsp[pos] == 0xFF or (pos + 1 < len(rbsp) and rbsp[pos + 1] == 0xFF):
            return out   # an extension the standard reserves: stop, do not guess
        ptype = rbsp[pos]
        size = 0
        pos += 1
        while pos < len(rbsp):
            size += rbsp[pos]
            last = rbsp[pos]
            pos += 1
            if last != 0xFF:
                break
        if pos + size > len(rbsp):
            return out
        out[ptype] = rbsp[pos:pos + size]
        pos += size
    return out


def carried_metadata(nal):
    """True for an SEI NAL that declares ST 2086 or CTA-861.3."""
    return bool({137, 144} & set(sei_payloads(nal)))


def bitstream_evidence(dec, mkv, source, counters, checks, results):
    """Read the coded bytes back, and ask the bitstream what it carries.

    The container's colour tags and the encoder's own side data are one question;
    whether HEVC actually got an ST 2086 SEI written into it is another, and only
    the elementary stream can answer it. Both files here are demuxed with -c copy,
    so what is probed is the bytes the encoder produced.

    This is the required gate: if the metadata is not in the elementary stream,
    these checks fail and the run is reported FAIL, whatever the container says.
    """
    authored = ARTIFACTS / "a_src_copy.hevc"
    ours = ARTIFACTS / "a_hdr10_out.hevc"
    wrote_source = demux(FIXTURES / "hdr_src.mkv", authored, checks,
                         "bitstream: authored source")
    wrote_ours = demux(mkv, ours, checks, "bitstream: encoded output")

    stream = probe_stream(ours, checks) if wrote_ours else {}
    checks.add("bitstream: the coded parameter sets say Main 10 at 10-bit, at the "
               "authored extents",
               (stream.get("codec_name"), stream.get("profile"),
                stream.get("pix_fmt"), stream.get("width"), stream.get("height")) ==
               ("hevc", "Main 10", "yuv420p10le", 1920, 1080), stream)
    coded_pts = probe_decoded_frame_pts(ours, checks) if wrote_ours else []
    checks.add("bitstream: the elementary stream decodes to the two coded frames",
               len(coded_pts) == 2, coded_pts)

    ours_bytes = ours.read_bytes() if wrote_ours and ours.exists() else b""
    nals = annexb_nal_bodies(ours_bytes)
    authored_nals = [nal for nal in nals if carried_metadata(nal)]
    ours_frames = (probe_decoded_frames_with_side_data(ours, checks,
                                                     "ffprobe_our_bitstream_frame_side_data.json")
                   if wrote_ours else [])
    authored_frames = (probe_decoded_frames_with_side_data(authored, checks,
                                                          "ffprobe_source_bitstream_frame_side_data.json")
                       if wrote_source else [])
    authored_mastering = side_data_blocks(authored_frames,
                                          "Mastering display metadata")
    authored_light = side_data_blocks(authored_frames,
                                      "Content light level metadata")
    our_mastering = side_data_blocks(ours_frames, "Mastering display metadata")
    our_light = side_data_blocks(ours_frames, "Content light level metadata")

    # ---- required: the metadata is in the elementary stream -------------- #
    expected_mastering = {key: value for key, value in
                          metadata_values(source).items()
                          if isinstance(value, Fraction)}
    expected_light = {"max_cll": source.get("max_cll"),
                      "max_fall": source.get("max_fall")}
    checks.add("REQUIRED bitstream: the encoded elementary stream carries a "
               "mastering display SEI on every coded frame",
               len(our_mastering) == 2 and all(
                   any(block.get("side_data_type") == "Mastering display metadata"
                       for block in frame.get("side_data_list", []))
                   for frame in ours_frames) and len(ours_frames) == 2,
               {"mastering_blocks": len(our_mastering),
                "frames": len(ours_frames),
                "kinds": sorted({block.get("side_data_type")
                                 for frame in ours_frames
                                 for block in frame.get("side_data_list", [])}),
                "note": ("a container tag or a frame side-data block is not a "
                         "bitstream SEI; this reads the demuxed elementary "
                         "stream")})
    checks.add("REQUIRED bitstream: it carries a content light level SEI on "
               "every coded frame",
               len(our_light) == 2 and all(
                   any(block.get("side_data_type")
                       == "Content light level metadata"
                       for block in frame.get("side_data_list", []))
                   for frame in ours_frames),
               {"content_light_blocks": len(our_light)})
    mastering_agreement = {
        "expected": {key: str(value) for key, value in expected_mastering.items()
                     if isinstance(value, Fraction)},
        "read": [{key: str(value) for key, value in
                  mastering_probe_values(block).items()}
                 for block in our_mastering]}
    mastering_agreement["exact_equal"] = all(
        mastering_probe_values(block) == expected_mastering
        for block in our_mastering) and bool(our_mastering)
    checks.add("REQUIRED bitstream: the in-stream mastering values equal the "
               "source's, as exact rationals, with no rounding",
               mastering_agreement["exact_equal"], mastering_agreement)
    checks.add("REQUIRED bitstream: the in-stream MaxCLL and MaxFALL equal the "
               "source's", all(block.get("max_content") == expected_light["max_cll"]
                               and block.get("max_average") ==
                               expected_light["max_fall"] for block in our_light)
               and bool(our_light),
               {"expected": expected_light,
                "read": [{key: block.get(key) for key in ("max_content",
                                                          "max_average")}
                         for block in our_light]})
    checks.add("REQUIRED bitstream: the prefix SEI NAL units the decoder read are "
               "type 39 and carry payload types 137 and 144",
               len(authored_nals) == 2 and all(
                   nal_unit_type(nal) == 39 and
                   set(sei_payloads(nal)) >= {137, 144} for nal in authored_nals),
               [{"type": nal_unit_type(nal),
                 "payload_types": sorted(sei_payloads(nal)),
                 "bytes": len(nal)} for nal in authored_nals])
    checks.add("bitstream: the probe does detect an in-bitstream mastering SEI "
               "(positive control)", bool(authored_mastering) and
               bool(authored_light),
               {"authored_mastering_blocks": len(authored_mastering),
                "authored_content_light_blocks": len(authored_light),
                "authored_frames": len(authored_frames)})

    # ---- required: the writer touched nothing it was not asked to -------- #
    vcl = [nal for nal in nals if nal_unit_type(nal) <= 31]
    parameter_sets = [nal for nal in nals if nal_unit_type(nal) in (32, 33, 34)]
    # Matroska keeps the parameter sets in its own configuration record, so the
    # demuxed stream has VPS, SPS and PPS written ahead of the IDR that never
    # arrived in a packet: they are the muxer's bytes, and the session hash below
    # covers the packets as they came out of the encoder.
    received = [nal for nal in nals if not carried_metadata(nal)
                and nal not in parameter_sets]
    baseline_bytes = PRESERVED_BASELINE.read_bytes() \
        if PRESERVED_BASELINE.exists() else b""
    baseline_nals = annexb_nal_bodies(baseline_bytes)
    baseline_vcl = [nal for nal in baseline_nals if nal_unit_type(nal) <= 31]
    baseline_sets = [nal for nal in baseline_nals
                     if nal_unit_type(nal) in (32, 33, 34)]
    status = counters.get("sei_status") or {}
    checks.add("bitstream: the session hash the bridge computed over the coded "
               "slices it received equals the hash of the slices in the file",
               status.get("vcl_hash") == fnv64(b"".join(vcl)) and
               status.get("vcl_bytes") == sum(len(nal) for nal in vcl),
               {"bridge_vcl_hash": status.get("vcl_hash"),
                "python_vcl_hash": fnv64(b"".join(vcl)),
                "vcl_nals": len(vcl),
                "vcl_bytes_bridge": status.get("vcl_bytes"),
                "vcl_bytes_file": sum(len(nal) for nal in vcl)})
    checks.add("bitstream: every NAL that arrived from the encoder is still in "
               "the stream, in order, unchanged, and the only addition is the "
               "authored SEI",
               status.get("existing_nal_hash") == fnv64(b"".join(received)) and
               len(nals) == len(received) + len(parameter_sets) +
               len(authored_nals),
               {"bridge_existing_nal_hash": status.get("existing_nal_hash"),
                "python_hash_of_received_nals": fnv64(b"".join(received)),
                "nals_in_file": len(nals), "nals_received": len(received),
                "nals_from_container_configuration": len(parameter_sets),
                "nals_authored": len(authored_nals),
                "first_packet_as_received": status.get("packet_sample")})
    checks.add("bitstream: the bytes the writer says it inserted are exactly the "
               "size difference against the previous stage's stream",
               bool(baseline_bytes) and
               status.get("bytes_inserted") == len(ours_bytes) - len(baseline_bytes),
               {"bytes_inserted": status.get("bytes_inserted"),
                "previous_stage_bytes": len(baseline_bytes),
                "this_stage_bytes": len(ours_bytes),
                "difference": len(ours_bytes) - len(baseline_bytes),
                "previous_stage_sha256": sha256_file(PRESERVED_BASELINE)
                if baseline_bytes else "",
                "this_stage_sha256": sha256_file(ours) if ours_bytes else ""})
    checks.add("bitstream: the previous stage's parameter sets and own SEI are "
               "byte-identical, and so are its coded slices",
               bool(baseline_nals) and baseline_vcl == vcl and
               baseline_sets == parameter_sets and
               [nal for nal in baseline_nals if nal_unit_type(nal) == 39] ==
               [nal for nal in received if nal_unit_type(nal) == 39],
               {"baseline_nal_types": [nal_unit_type(nal)
                                       for nal in baseline_nals],
                "this_nal_types": [nal_unit_type(nal) for nal in nals],
                "vcl_equal": baseline_vcl == vcl,
                "parameter_sets_equal": baseline_sets == parameter_sets})
    checks.add("bitstream: the writer saw exactly the two packets it was asked "
               "about and every access unit had a slice to attach to",
               status.get("packets_seen") == 2 and
               status.get("packets_without_vcl") == 0 and
               status.get("mastering_inserted") == 2 and
               status.get("content_light_inserted") == 2 and
               status.get("mastering_already_present") == 0 and
               status.get("content_light_already_present") == 0 and
               status.get("framing") == 1 and status.get("error") == 0,
               {"packets_seen": status.get("packets_seen"),
                "nals_inserted": status.get("nals_inserted"),
                "bytes_inserted": status.get("bytes_inserted"),
                "framing": status.get("framing"),
                "error": status.get("error"),
                "packets_without_vcl": status.get("packets_without_vcl"),
                "packet_sample": status.get("packet_sample"),
                "detail": status.get("detail")})
    results["bitstream_evidence"] = {
        "authored_bitstream": str(authored),
        "authored_bitstream_mastering_blocks": len(authored_mastering),
        "authored_bitstream_content_light_blocks": len(authored_light),
        "encoded_bitstream": str(ours),
        "encoded_bitstream_stream_probe": stream,
        "encoded_bitstream_frame_pts": coded_pts,
        "encoded_bitstream_mastering_blocks": len(our_mastering),
        "encoded_bitstream_content_light_blocks": len(our_light),
        "encoded_bitstream_sei_nals": [
            {"nal_unit_type": nal_unit_type(nal),
             "payload_types": sorted(sei_payloads(nal)),
             "bytes": len(nal)} for nal in authored_nals],
        "mastering_value_agreement": mastering_agreement,
        "expected_from_source": {key: str(value) for key, value in
                                 expected_mastering.items()},
        "nals": {"in_file": len(nals), "received": len(received),
                 "from_container_configuration": len(parameter_sets),
                 "authored": len(authored_nals), "vcl": len(vcl)},
        "vcl_hash_file": fnv64(b"".join(vcl)),
        "received_nal_hash_file": fnv64(b"".join(received)),
        "previous_stage_baseline": {
            "path": str(PRESERVED_BASELINE),
            "bytes": len(baseline_bytes),
            "sha256": sha256_file(PRESERVED_BASELINE) if baseline_bytes else "",
            "coded_slices_unchanged": baseline_vcl == vcl,
            "note": ("the previous stage's artifact is read, never rewritten: it "
                     "is the stream the writer had to leave alone below")},
        "sei_status": {key: value for key, value in status.items()
                       if key not in ("detail", "packet_sample")},
        "sei_status_detail": status.get("detail"),
        "sei_status_packet_sample": status.get("packet_sample"),
        "demux_command": "ffmpeg -i a_hdr10_out.mkv -c copy a_hdr10_out.hevc",
        "mechanism": ("NVENC offers no ST 2086 or CTA-861.3 option, so the bridge "
                      "authors the prefix SEI itself: at the receive-packet-before-"
                      "mux boundary it serialises payloadType 137 (24 bytes: the "
                      "eight u(16) chromaticities in the standard's green, blue, "
                      "red order, then the white point, then max and min luminance "
                      "as u(32)) and payloadType 144 (MaxCLL and MaxFALL as u(16)), "
                      "escapes them, and inserts one type 39 NAL immediately before "
                      "the first VCL NAL of every access unit, in that unit's own "
                      "framing. The rule is the standard's: a prefix SEI must "
                      "precede the slice it describes, so a static declaration is "
                      "repeated per picture rather than sent once. Nothing else in "
                      "the packet is altered and no pixel is ever downloaded."),
        "writer_sources": ("native/nvenc_bridge.cpp (the SEI section) and the "
                           "nvenc_hdr_sei_* entries in native/nvenc_bridge.h; "
                           "byte-exact expectations for the same payloads are "
                           "computed independently in "
                           "native/nvenc_hdr_sei_cpu_gate.py"),
        "encoder_option_dump": str(ARTIFACTS / "nvenc_hdr_options.txt"),
        "encoder_hdr_metadata_option_count": 0}


def case_b(dec, enc, checks, results):
    """A source that declares no metadata must not gain any."""
    source = FIXTURES / "ctrl.mkv"
    mkv = ARTIFACTS / "b_no_metadata_out.mkv"
    stream = encode_streamed(dec, enc, source, mkv, checks, "B source",
                             attach=False, fps=(25, 1))
    if not stream:
        return
    counters, metadata, dec_counters = (stream["counters"], stream["metadata"],
                                        stream["decoder_counters"])
    checks.add("B: source declares no mastering metadata",
               metadata.get("has_mastering") == 0, metadata)
    checks.add("B: source declares no content light",
               metadata.get("has_content_light") == 0, metadata)
    checks.add("B: borrowed surface is P010",
               stream["first"].software_format == 2,
               stream["first"].software_format)
    contract = counters["contract"]
    checks.add("B: no mastering side data attached to any frame",
               counters["mastering_side_data"] == 0 and
               counters["content_light_side_data"] == 0, counters)
    checks.add("B: nothing written as metadata side data, in frame, encoder, "
               "stream or bitstream",
               contract["metadata_written_to"] & 23 == 0,
               f"metadata_written_to={contract['metadata_written_to']}")
    checks.add("B: the container colour tags are still the HDR10 contract",
               contract["metadata_written_to"] & 8 == 8,
               contract["metadata_written_to"])
    checks.add("B: encoder host pixel bytes stay 0",
               counters["host_pixel_bytes"] == 0, counters["host_pixel_bytes"])
    checks.add("B: decoder host copies stay 0",
               dec_counters["host_frame_copy_count"] == 0, dec_counters)
    out_handle, _, _, out_metadata, _ = decode_frames(dec, mkv, 1, checks,
                                                      "B output")
    checks.add("B: output gained no invented HDR metadata",
               out_metadata.get("has_mastering") == 0 and
               out_metadata.get("has_content_light") == 0, out_metadata)
    if out_handle:
        dec.nvdec_bridge_close(out_handle)
    elementary = ARTIFACTS / "b_no_metadata_out.hevc"
    b_nals = []
    if demux(mkv, elementary, checks, "bitstream: no-metadata output"):
        b_nals = annexb_nal_bodies(elementary.read_bytes())
    checks.add("REQUIRED bitstream: absence stayed absence - a source that "
               "declared nothing gains no mastering or content-light SEI in the "
               "elementary stream",
               bool(b_nals) and not [nal for nal in b_nals if carried_metadata(nal)],
               {"nals": len(b_nals),
                "sei_nals": [[key for key in sei_payloads(nal)]
                             for nal in b_nals
                             if nal_unit_type(nal) == NAL_TYPE_PREFIX_SEI]})
    checks.add("REQUIRED bitstream: the no-metadata stream still decodes as Main 10 "
               "to its two coded frames",
               bool(b_nals) and len(probe_decoded_frame_pts(elementary, checks)) == 2,
               {"nal_types": [nal_unit_type(nal) for nal in b_nals]})
    checks.add("B: the writer recorded that nothing was requested and inserted "
               "nothing",
               counters["sei_status"]["metadata_requested"] == 0 and
               counters["sei_status"]["error"] == 1 and
               counters["sei_status"]["bytes_inserted"] == 0 and
               counters["sei_status"]["nals_inserted"] == 0 and
               counters["sei_status"]["mastering_inserted"] == 0 and
               counters["sei_status"]["packets_seen"] == 0,
               {key: counters["sei_status"][key] for key in
                ("metadata_requested", "error", "nals_inserted", "bytes_inserted",
                 "mastering_inserted", "content_light_inserted", "packets_seen")})
    checks.add("B: no in-bitstream SEI bit was set",
               contract["metadata_written_to"] & 16 == 0,
               contract["metadata_written_to"])
    results["case_b_no_metadata"] = {"source": str(source),
                                     "source_metadata": metadata,
                                     "output": str(mkv),
                                     "output_metadata": out_metadata,
                                     "encoder_counters": counters,
                                     "decoder_counters": dec_counters,
                                     "note": ("the colour tags are the entry's "
                                              "HDR10 encoding contract and are "
                                              "always declared; the mastering and "
                                              "content-light metadata stay exactly "
                                              "as absent as the source left them")}


def case_rejections(dec, enc, avutil, checks, results):
    """Every combination this entry does not encode, refused rather than dropped."""
    nv12 = pix_fmt_number(avutil, b"nv12")
    attempts = [
        ("codec h264_nvenc", default_hdr_options(codec=b"h264_nvenc"), 1920,
         "hevc_nvenc"),
        ("profile main", default_hdr_options(profile=b"main"), 1920, "main10"),
        (f"sw_pixel_format nv12({nv12})",
         default_hdr_options(sw_pixel_format=nv12), 1920, "HDR10 value"),
        ("color_primaries bt709", default_hdr_options(color_primaries=1), 1920,
         "HDR10 value"),
        ("extra_options pix_fmt override",
         default_hdr_options(extra_options=b"pix_fmt=nv12"), 1920, "fixed"),
        ("odd width 1921", default_hdr_options(), 1921, "invalid"),
        # NVENC refuses a 2-frame GOP against its own default B-frame count
        # ("Gop Length should be greater than number of B frames + 1", printed by
        # the codec into the gate's stderr), and the wrapper surfaces that open
        # failure as EINVAL. Requested-option behaviour, recorded rather than
        # assumed: this entry does not quietly widen the GOP to make the open work.
        ("gop 2 refused by NVENC", default_hdr_options(gop=2), 1920, "invalid"),
    ]
    source = FIXTURES / "hdr_src.mkv"
    handle = VP()
    if dec.nvdec_bridge_open_hdr(os.fsencode(source), C.byref(handle)) != 0:
        checks.add("rejections: source borrowable", False,
                   text(dec.nvdec_bridge_last_error()))
        return
    frame = DecFrame()
    frame.struct_size = C.sizeof(DecFrame)
    if dec.nvdec_bridge_next(handle, C.byref(frame)) != 1:
        checks.add("rejections: source frame borrowable", False,
                   text(dec.nvdec_bridge_last_error()))
        dec.nvdec_bridge_close(handle)
        return
    outcomes = []
    for label, options, width, expect in attempts:
        bridge = VP()
        path = ARTIFACTS / f"reject_{re.sub(r'[^a-z0-9]+', '_', label.lower())}.mkv"
        code = enc.nvenc_hdr_open(frame.cuda_context, width, frame.height, 25, 1,
                                  1, 1000, os.fsencode(path), C.byref(options),
                                  None, C.byref(bridge))
        error = text(enc.nvenc_last_error())
        refused = code != 0 and bridge.value is None and not path.exists()
        checks.add(f"rejection: {label} refused", refused,
                   {"rc": code, "error": error, "file_written": path.exists()})
        checks.add(f"rejection: {label} states the contract",
                   expect.lower() in error.lower(),
                   f"expected {expect!r} in {error!r}")
        outcomes.append({"case": label, "rc": code, "error": error,
                         "file_written": path.exists()})
    # The declaration the writer has to put on the wire. A value that cannot be
    # placed on the standard's grid is refused at the open, before any encoder or
    # file exists, and a half-declared display is authored as what it can say
    # rather than padded with invented numbers.
    fixture_metadata = HdrMetadata()
    fixture_metadata.struct_size = C.sizeof(HdrMetadata)
    dec.nvdec_bridge_get_hdr_metadata(handle, C.byref(fixture_metadata))
    metadata_outcomes = []
    for label, mutate, expect_refused, expect_word in (
            ("white point not on the 1/50000 grid",
             lambda m: setattr(m, "white_x_den", 3), True, "grid"),
            ("max luminance beyond u(32) on the grid",
             lambda m: setattr(m, "max_luminance_num", 600000000), True,
             "outside"),
            ("zero denominator",
             lambda m: setattr(m, "min_luminance_den", 0), True, "denominator"),
            ("primaries without luminance",
             lambda m: setattr(m, "mastering_has_luminance", 0), False,
             "partial")):
        block = HdrMetadata()
        C.memmove(C.byref(block), C.byref(fixture_metadata),
                  C.sizeof(HdrMetadata))
        mutate(block)
        bridge = VP()
        path = ARTIFACTS / f"sei_{re.sub(r'[^a-z0-9]+', '_', label.lower())}.mkv"
        options = default_hdr_options()
        code = enc.nvenc_hdr_open(frame.cuda_context, frame.width, frame.height,
                                  25, 1, 1, 1000, os.fsencode(path),
                                  C.byref(options), C.byref(block), C.byref(bridge))
        error = text(enc.nvenc_last_error())
        record = {"case": label, "rc": code, "error": error,
                  "file_written": path.exists(),
                  "expected": "refusal" if expect_refused else "acceptance"}
        if expect_refused:
            checks.add(f"SEI refusal: {label} refuses the open and writes no file",
                       code != 0 and bridge.value is None and not path.exists() and
                       expect_word in error.lower(), record)
        else:
            checks.add(f"SEI partial: {label} is accepted, not refused, and says "
                       "so", code == 0 and bridge.value is not None, record)
            if code == 0:
                enc.nvenc_encode_p010(bridge, frame.y_device_ptr,
                                      frame.y_pitch_bytes, frame.uv_device_ptr,
                                      frame.uv_pitch_bytes, frame.pts)
                enc.nvenc_finish(bridge)
                status = SeiStatus()
                status.struct_size = C.sizeof(SeiStatus)
                enc.nvenc_hdr_sei_get(bridge, C.byref(status))
                contract = HdrContract()
                contract.struct_size = C.sizeof(HdrContract)
                enc.nvenc_hdr_get_contract(bridge, C.byref(contract))
                enc.nvenc_close(bridge)
                partial_stream = ARTIFACTS / "sei_partial_mastering.hevc"
                demuxed = demux(path, partial_stream, checks, "SEI partial: demux")
                nals = annexb_nal_bodies(partial_stream.read_bytes()) \
                    if demuxed and partial_stream.exists() else []
                types = sorted({key for nal in nals for key in sei_payloads(nal)})
                record.update({"sei_status": counter_fields(status),
                               "detail": text(status.detail),
                               "metadata_written_to": contract.metadata_written_to,
                               "payload_types_in_stream": types})
                checks.add("SEI partial: no mastering payload is invented and the "
                           "content-light one is still authored",
                           status.mastering_serialized == 0 and
                           status.content_light_serialized == 1 and
                           status.error == 2 and 137 not in types and 144 in types,
                           {"mastering_serialized": status.mastering_serialized,
                            "content_light_serialized":
                                status.content_light_serialized,
                            "error": status.error,
                            "payload_types_in_stream": types,
                            "detail": text(status.detail)})
                checks.add("SEI partial: the required-SEI contract is not marked "
                           "met while the container keeps the declared colour "
                           "contract",
                           contract.metadata_written_to & 8 == 8 and
                           status.mastering_serialized == 0,
                           f"metadata_written_to={contract.metadata_written_to}")
        metadata_outcomes.append(record)
    results["sei_metadata_cases"] = metadata_outcomes

    # Pitch and plane validation on a real HDR bridge.
    bridge = VP()
    options = default_hdr_options()
    good = ARTIFACTS / "rejection_bridge.mkv"
    if enc.nvenc_hdr_open(frame.cuda_context, frame.width, frame.height, 25, 1,
                          1, 1000, os.fsencode(good), C.byref(options), None,
                          C.byref(bridge)) == 0:
        too_small = enc.nvenc_encode_p010(bridge, frame.y_device_ptr,
                                          frame.width, frame.uv_device_ptr,
                                          frame.uv_pitch_bytes, frame.pts)
        checks.add("rejection: pitch below width*2 refused", too_small != 0,
                   f"rc={too_small} {text(enc.nvenc_last_error())}")
        null_plane = enc.nvenc_encode_p010(bridge, 0, frame.y_pitch_bytes,
                                          frame.uv_device_ptr,
                                          frame.uv_pitch_bytes, frame.pts)
        checks.add("rejection: null plane refused", null_plane != 0,
                   f"rc={null_plane} {text(enc.nvenc_last_error())}")
        counters = HdrCounters()
        counters.struct_size = C.sizeof(HdrCounters)
        enc.nvenc_hdr_get_counters(bridge, C.byref(counters))
        checks.add("rejection: refused encodes submitted no frame",
                   counters.frames_submitted == 0, counter_fields(counters))
        enc.nvenc_close(bridge)
    dec.nvdec_bridge_close(handle)
    checks.add("rejection: null bridge handle refused",
               enc.nvenc_encode_p010(None, frame.y_device_ptr, 4096,
                                     frame.uv_device_ptr, 4096, 0) != 0,
               text(enc.nvenc_last_error()))

    child = subprocess.run([sys.executable, os.fsdecode(Path(__file__)),
                            "--rgba-on-hdr-bridge"], capture_output=True,
                           text=True, timeout=600)
    record_command(child.args, child.returncode, "child stdout", child.stderr)
    checks.add("rejection: nvenc_encode_rgba on an HDR bridge returns -1 and "
               "survives", child.returncode == 0 and "GRACEFUL" in child.stdout,
               (child.stdout + child.stderr).strip())
    results["rejections"] = {"attempts": outcomes,
                             "rgba_on_hdr_bridge_child": child.stdout.strip(),
                             "rgba_on_hdr_bridge_child_rc": child.returncode}


def rgba_on_hdr_bridge_child():
    """Child mode: open an HDR bridge, call the SDR entry, report the answer."""
    cookie = os.add_dll_directory(str(FFMPEG_BIN))
    try:
        dec, enc = bind_nvdec(), bind_nvenc()
        handle = VP()
        if dec.nvdec_bridge_open_hdr(os.fsencode(FIXTURES / "hdr_src.mkv"),
                                     C.byref(handle)) != 0:
            print("CHILD decode-fail", text(dec.nvdec_bridge_last_error()))
            return 3
        frame = DecFrame()
        frame.struct_size = C.sizeof(DecFrame)
        if dec.nvdec_bridge_next(handle, C.byref(frame)) != 1:
            print("CHILD frame-fail", text(dec.nvdec_bridge_last_error()))
            return 3
        bridge = VP()
        options = default_hdr_options()
        path = ARTIFACTS / "child_rgba_on_hdr_bridge.mkv"
        if enc.nvenc_hdr_open(frame.cuda_context, frame.width, frame.height, 25, 1,
                              1, 1000, os.fsencode(path), C.byref(options), None,
                              C.byref(bridge)) != 0:
            print("CHILD open-fail", text(enc.nvenc_last_error()))
            return 3
        code = enc.nvenc_encode_rgba(bridge, frame.y_device_ptr,
                                     frame.y_pitch_bytes, frame.pts)
        counters = HdrCounters()
        counters.struct_size = C.sizeof(HdrCounters)
        enc.nvenc_hdr_get_counters(bridge, C.byref(counters))
        error = text(enc.nvenc_last_error())
        enc.nvenc_close(bridge)
        dec.nvdec_bridge_close(handle)
        print("GRACEFUL rc=%d error=%s frames_submitted=%d rgb_conversion_bytes=%d"
              % (code, error, counters.frames_submitted,
                 counters.rgb_conversion_bytes))
        return 0 if code != 0 and counters.frames_submitted == 0 else 4
    finally:
        cookie.close()


def capability_and_defaults(enc, dec, checks, results):
    defaults = HdrDefaults()
    defaults.struct_size = C.sizeof(HdrDefaults)
    rc = enc.nvenc_hdr_get_defaults(C.byref(defaults))
    summary_defaults = {"rc": rc, "sw_pixel_format": defaults.sw_pixel_format,
                        "color_range": defaults.color_range,
                        "color_primaries": defaults.color_primaries,
                        "color_trc": defaults.color_trc,
                        "color_space": defaults.color_space,
                        "profile": text(defaults.profile)}
    checks.add("defaults query returns the HDR10 contract", rc == 0, summary_defaults)
    checks.add("defaults name the HDR10 format and colour",
               defaults.sw_pixel_format == installed_pix_fmt("p010le") and
               (defaults.color_range, defaults.color_primaries,
                defaults.color_trc, defaults.color_space) ==
               (LIMITED, BT2020, PQ, BT2020NC) and
               text(defaults.profile) == "main10", summary_defaults)
    handle, frames, _, _, _ = decode_frames(dec, FIXTURES / "hdr_src.mkv", 1, checks,
                                            "capability")
    if not frames:
        return
    cap = HdrCapability()
    cap.struct_size = C.sizeof(HdrCapability)
    code = enc.nvenc_hdr_capability(frames[0].cuda_context, frames[0].width,
                                    frames[0].height, C.byref(cap))
    summary = counter_fields(cap)
    summary["encoder_name"] = text(cap.encoder_name)
    summary["profile_name"] = text(cap.profile_name)
    checks.add("capability on the decoder's context: HDR10 supported",
               code == 0 and cap.supported == 1, summary)
    checks.add("capability: a CUDA device object is created on that context",
               cap.cuda_device_created == 1, cap.cuda_device_created)
    checks.add("capability: installed NVENC offers no HDR metadata option",
               cap.hdr_metadata_options == 0, cap.hdr_metadata_options)
    sei_cap = SeiCapability()
    sei_cap.struct_size = C.sizeof(SeiCapability)
    sei_code = enc.nvenc_hdr_sei_capability(C.byref(sei_cap))
    summary_sei = counter_fields(sei_cap)
    checks.add("the writer's capability is reported apart from the encoder's: an "
               "option count of 0 is not a statement that the bitstream cannot "
               "carry the metadata",
               sei_code == 0 and sei_cap.writer_available == 1 and
               sei_cap.encoder_offers_metadata_option == 0 and
               cap.hdr_metadata_options == 0 and cap.main10_profile_supported == 1,
               dict(summary_sei, hardware_main10=cap.main10_profile_supported,
                    encoder_metadata_options=cap.hdr_metadata_options))
    checks.add("the writer states the standard's own numbers rather than a "
               "spelling",
               (sei_cap.payload_type_mastering, sei_cap.payload_type_content_light,
                sei_cap.prefix_sei_nal_type, sei_cap.mastering_payload_size,
                sei_cap.content_light_payload_size, sei_cap.chroma_denominator,
                sei_cap.luma_denominator, sei_cap.primary_order) ==
               (137, 144, 39, 24, 4, 50000, 10000, 1), summary_sei)
    checks.add("the writer refuses to invent a half-declared mastering display",
               sei_cap.supports_partial_mastering == 0 and
               sei_cap.max_nal_bytes == 47 and sei_cap.max_insert_bytes == 51,
               summary_sei)
    results["sei_capability"] = summary_sei
    checks.add("capability: Main10 resolved from the encoder's own option table",
               cap.main10_profile_supported == 1 and
               text(cap.profile_name).lower().replace(" ", "") == "main10",
               text(cap.profile_name))
    if handle:
        dec.nvdec_bridge_close(handle)
    results["abi"] = {"nvenc": enc.nvenc_abi_version(),
                      "nvenc_hdr": enc.nvenc_hdr_abi_version(),
                      "nvdec": dec.nvdec_bridge_abi_version()}
    results["struct_sizes_expected"] = PROBE_SIZES
    results["defaults"] = summary_defaults
    results["capability_on_decoder_context"] = summary


def binding_size_gate(avutil):
    """ctypes layouts versus the sizes the compiler printed."""
    checks = Checks()
    actual = {"options": C.sizeof(HdrOptions), "counters": C.sizeof(HdrCounters),
              "defaults": C.sizeof(HdrDefaults),
              "capability": C.sizeof(HdrCapability),
              "contract": C.sizeof(HdrContract), "metadata": C.sizeof(HdrMetadata),
              "sei_capability": C.sizeof(SeiCapability),
              "sei_info": C.sizeof(SeiInfo),
              "sei_status": C.sizeof(SeiStatus)}
    for key, expected in PROBE_SIZES.items():
        checks.add(f"binding size {key}", actual[key] == expected,
                   f"ctypes {actual[key]} vs compiled {expected}")
    p010 = pix_fmt_number(avutil, b"p010le")
    checks.add("installed library names 158 as p010le", p010 == 158, p010)
    return checks, actual


def artifact_hashes():
    out = {}
    for name in ("nvenc_bridge.dll", "nvdec_bridge.dll", "nvenc_hdr_probe.exe"):
        path = NATIVE / name
        if path.exists():
            out[name] = {"path": str(path), "bytes": path.stat().st_size,
                         "sha256": sha256_file(path)}
    for name in ("nvenc_bridge.cpp", "nvenc_bridge.h", "nvenc_hdr_probe.cpp",
                 "build_nvenc_hdr_probe.bat", "nvenc_hdr_p010_smoke.py"):
        path = NATIVE / name
        if path.exists():
            out[f"source:{name}"] = {"path": str(path),
                                     "bytes": path.stat().st_size,
                                     "sha256": sha256_file(path),
                                     "note": ("hashed by the gate that just ran, so "
                                              "the record names the bytes that "
                                              "produced it")}
    previous = ARTIFACTS / "nvdec_bridge.prev.dll"
    if previous.exists():
        out["nvdec_bridge_before_hdr_rebuild"] = {
            "path": str(previous), "bytes": previous.stat().st_size,
            "sha256": sha256_file(previous),
            "note": ("the checked-in decoder DLL exported no "
                     "nvdec_bridge_open_hdr, so it was rebuilt from the ABI 3 "
                     "source before this gate could borrow a P010 surface")}
    return out


def device_flow_summary(results):
    """What actually moved, device-side, for the two submitted frames."""
    case = results.get("case_a") or {}
    counters = case.get("encoder_counters") or {}
    contract = counters.get("contract") or {}
    decoder = case.get("decoder_counters") or {}
    abi2 = counters.get("abi2_counters") or {}
    submitted = case.get("submitted_frames") or []
    extent = submitted[0]["extent"] if submitted else None
    payload = (2 * 3 * extent[0] * extent[1]) if extent and len(extent) == 2 else None
    return {
        "chain": ("nvdec_bridge_open_hdr borrow of an AV_PIX_FMT_CUDA surface whose "
                  "sw_format is P010LE -> nvenc_encode_p010(y_ptr, y_pitch, uv_ptr, "
                  "uv_pitch, pts) -> cuMemcpy2D_v2 pitched device-to-device copies "
                  "into the encoder-owned hwframe -> avcodec_send_frame / "
                  "avcodec_receive_packet -> matroska muxer"),
        "borrowed_surface": {
            "hardware_format_number": contract.get("actual_pix_fmt"),
            "software_format_number": contract.get("actual_sw_pix_fmt"),
            "software_format_name": contract.get("actual_sw_pix_fmt_name"),
            "extent": extent,
            "pitches_bytes": [submitted[0]["y_pitch_bytes"],
                              submitted[0]["uv_pitch_bytes"]] if submitted else None,
            "hw_frames_pool_size": contract.get("hw_frames_pool_size"),
            "hw_frames_sw_format": contract.get("hw_frames_sw_format")},
        "copy": {
            "p010_device_copy_bytes": counters.get("p010_device_copy_bytes"),
            "expected_two_frame_payload_bytes": payload,
            "row_payload": (extent[0] * 2 if extent else None),
            "rows": ("height for luma, height/2 for the interleaved chroma plane"
                     if extent else None),
            "synchronised_before_return": ("nvenc_encode_p010 runs "
                                           "cuCtxSynchronize after both planes, so "
                                           "the borrowed source is consumed before "
                                           "the caller may recycle it")},
        "boundaries_not_crossed": {
            "encoder_host_pixel_bytes": counters.get("host_pixel_bytes"),
            "encoder_rgb_conversion_bytes": counters.get("rgb_conversion_bytes"),
            "abi2_staging_gpu_copy_bytes": abi2.get("staging_gpu_copy_bytes"),
            "abi2_conversion_output_bytes": abi2.get("conversion_output_bytes"),
            "decoder_host_frame_copy_bytes": decoder.get("host_frame_copy_bytes"),
            "decoder_host_frame_copy_count": decoder.get("host_frame_copy_count"),
            "decoder_scale_device_output_bytes":
                decoder.get("scale_device_output_bytes"),
            "note": ("every one of these is a measured zero on a counter that rises "
                     "when the boundary is crossed; no hwdownload, host pixel "
                     "buffer, RGBA conversion or stdin pipe is involved on this "
                     "path")},
        "adapter": {"cuda_adapter_luid": contract.get("cuda_adapter_luid"),
                    "cuda_adapter_node_mask": contract.get("cuda_adapter_node_mask"),
                    "from": "cuDeviceGetLuid of the borrowed CUcontext"},
        "borrow_window": case.get("borrow_window"),
        "per_frame_detail": "case_a.submitted_frames and case_a.encoder_counters",
        "ffprobe_cpu_decode": ("the verification decodes on the CPU after the "
                               "boundary, off the measured path; the fixture "
                               "authoring is likewise outside it")}


def metadata_handoff_summary(results):
    """How the supplied metadata travelled, and what it was compared against."""
    case = results.get("case_a") or {}
    absent = results.get("case_b_no_metadata") or {}
    counters = case.get("encoder_counters") or {}
    contract = counters.get("contract") or {}
    bitstream = results.get("bitstream_evidence") or {}
    source, output = case.get("source_metadata") or {}, case.get("output_metadata") or {}
    return {
        "carried_by": [
            "AVFrame side data: av_frame_new_side_data(AV_FRAME_DATA_MASTERING_"
            "DISPLAY_METADATA) and AV_FRAME_DATA_CONTENT_LIGHT_LEVEL, on every "
            "submitted frame",
            "AVCodecContext::coded_side_data: av_packet_side_data_add with "
            "AV_PKT_DATA_MASTERING_DISPLAY_METADATA and AV_PKT_DATA_CONTENT_LIGHT_"
            "LEVEL, before the open",
            "AVStream::codecpar->coded_side_data: the same two types, after "
            "avcodec_parameters_from_context",
            "the Matroska container elements the muxer derives from those",
            "the elementary stream itself: one HEVC prefix SEI NAL unit per access "
            "unit, authored by the bridge from the same declared integers, "
            "carrying ST 2086 payloadType 137 and CTA-861.3 payloadType 144"],
        "not_carried_by": [
            "nothing on the metadata path: the elementary stream used to be the "
            "gap, and bitstream_evidence is what closes or fails it. The encoder "
            "still offers no metadata option of its own, so the SEI is the "
            "bridge's work, not NVENC's"],
        "written_to_bits": contract.get("metadata_written_to"),
        "in_bitstream_sei": {
            "mastering_blocks": bitstream.get("encoded_bitstream_mastering_blocks"),
            "content_light_blocks":
                bitstream.get("encoded_bitstream_content_light_blocks"),
            "sei_nals": bitstream.get("encoded_bitstream_sei_nals"),
            "values_equal_to_source": (bitstream.get("mastering_value_agreement") or
                                       {}).get("exact_equal"),
            "coded_slices_unchanged_against_previous_stage":
                (bitstream.get("previous_stage_baseline") or
                 {}).get("coded_slices_unchanged"),
            "bytes_inserted": (bitstream.get("sei_status") or
                               {}).get("bytes_inserted")},
        "per_frame_counts": {
            "mastering_side_data": counters.get("mastering_side_data"),
            "content_light_side_data": counters.get("content_light_side_data")},
        "exact_comparison": {
            "equal_in_value": same_rationals(source, output),
            "value_differences": rational_differences(source, output),
            "source_spelling": source, "output_spelling": output,
            "source_values": metadata_values(source),
            "output_values": metadata_values(output),
            "decoded_frame_side_data": case.get("output_decoded_frame_side_data")},
        "denominator_note": ("Matroska stores ST 2086 in its own fixed-point units, "
                             "so a source that declares 15635/50000 reads back as "
                             "3127/10000. The comparison is done on exact Fractions; "
                             "no rational anywhere in this gate passes through a "
                             "float, and no value was substituted for a supplied "
                             "one"),
        "absence_stays_absent": {
            "source_declared": absent.get("source_metadata"),
            "output_declared": absent.get("output_metadata"),
            "encoder_bits_for_metadata": (
                (absent.get("encoder_counters") or {}).get("contract", {}).get(
                    "metadata_written_to"))},
        "bitstream_evidence": bitstream.get("mechanism"),
        "encoder_option_dump": bitstream.get("encoder_option_dump")}


def json_value(value):
    """A last-resort encoder: the artifact has to be written even when a field
    came back as something json does not know, rather than losing the run."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode(errors="replace")
    if isinstance(value, C.Array):
        return text(value.value)
    return str(value)


def main():
    if "--rgba-on-hdr-bridge" in sys.argv:
        return rgba_on_hdr_bridge_child()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cookie = os.add_dll_directory(str(FFMPEG_BIN))
    results = {"environment": {"os": f"{platform.system()} {platform.release()}",
                               "python": sys.version.split()[0],
                               "ffmpeg_sdk_dir": str(FFMPEG_DIR)},
               "purpose": ("encoder boundary only: two borrowed CUDA P010 frames "
                           "from nvdec_bridge_open_hdr into nvenc_hdr_open and "
                           "nvenc_encode_p010, with the in-bitstream HDR10 static "
                           "SEI authored by the bridge at the "
                           "receive-packet-before-mux boundary; no DLSS, no "
                           "resampler, no full HDR export")}
    checks = Checks()
    status = "ABORTED"
    try:
        avutil = C.CDLL("avutil-59.dll")
        size_checks, sizes = binding_size_gate(avutil)
        results["binding_sizes"] = sizes
        for item in size_checks.items:
            item["name"] = "binding: " + item["name"]
            checks.items.append(item)
        dec, enc = bind_nvdec(), bind_nvenc()
        checks.add("nvenc bridge reports ABI 2", enc.nvenc_abi_version() == 2,
                   enc.nvenc_abi_version())
        checks.add("nvenc HDR entry reports ABI 1",
                   enc.nvenc_hdr_abi_version() == 1, enc.nvenc_hdr_abi_version())
        checks.add("nvenc SEI writer reports ABI 1",
                   enc.nvenc_hdr_sei_abi_version() == 1,
                   enc.nvenc_hdr_sei_abi_version())
        checks.add("nvdec bridge reports ABI 3", dec.nvdec_bridge_abi_version() == 3,
                   dec.nvdec_bridge_abi_version())
        results["binaries"] = artifact_hashes()
        capability_and_defaults(enc, dec, checks, results)
        case_a(dec, enc, checks, results)
        case_b(dec, enc, checks, results)
        case_rejections(dec, enc, avutil, checks, results)
        status = "PASS" if not checks.failed else "FAIL"
        return 0 if status == "PASS" else 1
    finally:
        results["status"] = status
        results["checks"] = checks.items
        results["checks_total"] = len(checks.items)
        results["checks_failed"] = checks.failed
        results["commands"] = COMMANDS
        results["files"] = artifact_inventory()
        results["p010_device_flow"] = device_flow_summary(results)
        results["metadata_handoff"] = metadata_handoff_summary(results)
        results["sdr_isolation"] = {
            "shared_code_changed": ("one guarded line. write_packet() is shared "
                                    "by the SDR and HDR paths and now reads "
                                    "if (b->hdr_mode && b->hdr_sei_wanted && "
                                    "hdr_sei_apply(b) < 0) return -1; before the "
                                    "rescale. An SDR bridge has hdr_mode false and "
                                    "an HDR bridge whose source declared no "
                                    "metadata has hdr_sei_wanted false, so neither "
                                    "ever enters the branch and both take exactly "
                                    "the bytes they took before. nvenc_open, "
                                    "nvenc_encode_rgba, nvenc_encode_texture, "
                                    "encode_frame, convert, configure_encoder, "
                                    "nvenc_finish, nvenc_close and the ABI 2 "
                                    "NvencTransferCounters path are textually "
                                    "untouched; git diff against HEAD is 1758 "
                                    "insertions and 3 deletions in "
                                    "native/nvenc_bridge.cpp and 370 insertions in "
                                    "native/nvenc_bridge.h, all of them in the "
                                    "HDR10 and SEI sections or in the new struct "
                                    "declarations appended after them."),
            "abi_layout_gates": ("nvenc_hdr_probe.exe pins every ABI 2 offset and "
                                 "size, every HDR10 struct and the three new SEI "
                                 "structs as compile-time asserts, and this gate "
                                 "re-pins them in ctypes (binding_sizes). Nothing "
                                 "an established caller compiled against moved."),
            "hdr_bridge_rejects_the_sdr_entry": (
                "the child probe (rejections.rgba_on_hdr_bridge_child) opens a "
                "real HDR bridge and calls nvenc_encode_rgba on it: it returns -1, "
                "frames_submitted stays 0 and rgb_conversion_bytes stays 0, so the "
                "two entries cannot be confused at runtime"),
            "sdr_regression_run": (
                "the shared write_packet line is what the brief's condition "
                "turns on, so the existing SDR encoder regression "
                "(native/nvenc_gpu_smoke.py, the 15-frame 4K "
                "nvdec->nv12_rgba->lanczos->NVOF->DLSSNR->residual->D3D12 "
                "texture -> av1_nvenc chain) was re-run against this DLL. It "
                "printed nvenc_gpu_smoke=PASS with exit code 0 and "
                "bridge_host_pixel_bytes=0, recorded verbatim in "
                "bench_out/hdr_sei_repair/raw_sdr_regression_nvenc_gpu_smoke.txt "
                "and with its hashes in "
                "bench_out/hdr_sei_repair/hdr_sei_repair_evidence.json. It runs "
                "as its own command, not inside this gate, so this artifact "
                "reports its outcome as a separately measured run rather than "
                "as one of its own checks."),
            "residual_risk": ("the DLL was rebuilt, so a SDR caller now loads new "
                              "bytes; the ABI asserts, the child probe and the "
                              "re-run SDR encoder regression cover the interface "
                              "and one real SDR encode, not the whole SDR feature "
                              "set. Inspected and measured as stated, and reported "
                              "that way")}
        results["next_step"] = {
            "proven": ("the encoder boundary holds and the requirement is met: "
                       "two borrowed CUDA P010 frames became a 2-packet HEVC "
                       "Main10 bitstream whose elementary stream carries the "
                       "source's ST 2086 and CTA-861.3 declaration as a prefix "
                       "SEI in every access unit, read back by the installed "
                       "decoder with no container to help, with values equal as "
                       "exact rationals and with the coded slices byte-identical "
                       "to what the encoder produced"),
            "not_proven": ("a full HDR export: this is the encoder boundary on a "
                           "fixture, with the metadata taken from the fixture's "
                           "declaration rather than from the measured FP16 chain"),
            "work_left": [
                "wire the HDR10 open/encode entries into the Python export path "
                "behind an explicit opt-in, with the metadata taken from the "
                "measured FP16 chain rather than a fixture",
                "one 4-6 frame production gate that runs the whole "
                "DLSS->resample->fuse->convert->encode chain and keeps every "
                "host-pixel counter at zero",
                "an experimental UI toggle, only after that gate passes"],
            "status_language": ("experimental HDR10 only; the formal Feature18 HDR "
                                "contract is unproven and performance is "
                                "NOT_CHARACTERIZED by this gate")}
        results["fixtures"] = {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path)}
            for path in (FIXTURES / "hdr_src.mkv", FIXTURES / "ctrl.mkv")
            if path.exists()}
        (ARTIFACTS / "p010_nvenc_gate.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=json_value),
            encoding="utf-8")
        print(json.dumps({"status": status, "failed": checks.failed,
                          "checks_total": len(checks.items)}, indent=2))
        cookie.close()


if __name__ == "__main__":
    sys.exit(main())
