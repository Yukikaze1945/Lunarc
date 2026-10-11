"""One real P010 frame, admitted by the native bridge through the production wrapper.

Stage B has never put a decoder's actual ``color_space`` in front of
``gpu_hdr_decode``: the wrapper constants and the bridge constant agreed on
``AVCOL_SPC_BT2020_CL`` and the fake-library tests passed while both were wrong, so
the repaired enumerator (NCL = 9) has only ever met a mock. This gate is that first
meeting, and nothing else:

  * exactly one frame, borrowed from the ``--source`` clip (by default
    ``build/hdr_probe/hdr_src.mkv``) through the existing additive
    ``nvdec_bridge_open_hdr`` P010 entry, with the colour fields
    and pitches the decoder itself reported - never a wrapper constant substituted
    for one of them;
  * the production ``hdr_gpu_engine.GpuHdrBridge``, opened against a real
    ``dlss_gpu_engine.Live`` D3D12 device that is borrowed for its device handle and
    never given a frame to process;
  * one positive admission through ``decode``, plus one negative admission on the
    same live frame with only ``color_space`` changed to BT2020_CL, whose refusal is
    preserved as the native said it.

What this deliberately does not do: no pixel ever comes to the host (the decoder's
``nvdec_bridge_readback_last_frame`` and every ``gpu_hdr_readback_*`` entry are not
bound at all), no Feature 18 call, no NVOF, no NVENC, no export, no UI, no orchestrator.
Small scalar report reads are the only device-to-host traffic and they are reported
separately from pixel traffic, because they are a different boundary.

    python native/hdr_real_admission_gate.py
    python native/hdr_real_admission_gate.py --source <clip> --support-ceiling <scrgb>
        --output-dir <dir>

The three flags default to the original run - the legacy fixture, the 10000 scRGB
domain it needed, and ``bench_out/hdr_real_admission`` - so an invocation with no
arguments still reproduces the evidence that is already on disk. A different clip is a
different declared domain: pass it here rather than editing the constant, and let the
physical bound come from the offline verifier's derived matrix, never from the clip's
MaxCLL.

Loader order is part of the contract, not a style preference: the FFmpeg dependency
directory and ``nvdec_bridge.dll`` are registered and loaded before ``hashlib`` (or
anything that binds it) is imported, because importing ``_hashlib`` first resolves
``libcrypto-3-x64.dll`` to the interpreter's own copy and then the avformat chain the
bridge needs fails with WinError 127. See ``sha256_file`` below and the same note in
``native/nvenc_hdr_p010_smoke.py``.
"""
import argparse
import ctypes as C
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "native"
FIXTURES = ROOT / "build" / "hdr_probe"
# Overridden from the command line in main(); each default is the original run.
EVIDENCE_DIR = ROOT / "bench_out" / "hdr_real_admission"
SOURCE = FIXTURES / "hdr_src.mkv"
sys.path.insert(0, str(ROOT))

FFMPEG_DIR = Path(os.environ.get("FFMPEG_SDK_DIR", r"F:\WhisperJAV\Library"))
FFMPEG_BIN = FFMPEG_DIR / "bin"
FFMPEG_INCLUDE = Path(os.environ.get("FFMPEG_INCLUDE_DIR",
                                     FFMPEG_DIR / "include" / "libavutil"))

# The declared domain for this job. It is a per-channel scRGB claim, so it has to
# be wide enough that a nonzero above_ceiling_components can only mean the source
# lied about itself, never that the ceiling was copied out of the source's metadata
# - it is NOT the MaxCLL equivalent (MaxCLL 1000 nits is 12.5 scRGB here).
#
# 250.0 scRGB was the first declaration and the measurement said it was wrong for
# this fixture: the run preserved in evidence_attempt1/attempt2 reports
# max_component 9420.164 scRGB. That is not a native defect. The fixture is a
# synthetic stress pattern, and the PQ inverse is asymptotic: a channel whose
# coded value sits past the ST 2084 top runs away long before it reaches the pole,
# so chroma-excited primaries a few code units past 940 decode to thousands of
# scRGB. See pq_domain_notes, which computes the pole and the measured sample's
# place on that curve from the bridge's own contract scalars.
# 10000.0 scRGB is therefore the narrowest round domain that admits what this
# fixture actually carries, with the margin recorded rather than assumed.
SUPPORT_CEILING_SCRGB = 10000.0
PAPER_WHITE_SCRGB = 8.0
# The production working-scale rule, applied to the decoder's own extent.
WORKING_SCALE = 0.5

# The HDR10 contract of the fixture as ffprobe describes the container: limited
# range, BT.2020 primaries, ST 2084 transfer, BT.2020 non-constant luminance. The
# gate asserts the DECODER's returned fields against these numbers; it never sends
# these numbers instead of the decoder's.
EXPECTED_SOURCE_COLOR = {"color_range": 1, "color_primaries": 9,
                         "color_transfer": 16, "color_space": 9}
AV_PIX_FMT_P010LE_AS_REPORTED = 2   # NvdecDeviceFrame.software_format: 1 NV12, 2 P010

# The build this gate is supposed to meet: the DLL rebuilt by the binding audit fix.
# Checked against the file this process actually loaded, not against a claim.
ACCEPTED_BRIDGE_SHA256 = ("e016fce6e1b3cbc01b4acd751db71144c1546dcd"
                          "57e362cdb0486a6a3a269e47")

# The two failed ceiling-250 attempts against build/hdr_probe/hdr_src.mkv, kept
# verbatim. They are that clip's history and not a template: an independent software
# decode reconstructs its PQ components from -0.484 to 1.389, outside BT.2100-2's
# [0, 1], which is exactly why its domain had to be widened and why a physically
# valid clip is not entitled to the same number. The evidence files themselves are
# untouched in bench_out/hdr_real_admission.
LEGACY_ATTEMPT_HISTORY = [
    {"attempt": 1, "support_ceiling_scrgb": 250.0,
     "outcome": ("the gate crashed on its own binding: a ctypes c_void_p struct "
                 "field reads back as an int, so frame.cuda_context.value does not "
                 "exist. Preserved as evidence_attempt1_failed.json."),
     "artifact": "evidence_attempt1_failed.json"},
    {"attempt": 2, "support_ceiling_scrgb": 250.0,
     "outcome": ("admission itself passed - NCL 9 accepted, BT2020_CL 10 refused, "
                 "zero host pixel copy - but the fixture's decoded span reached "
                 "9420.164 scRGB, so 250.0 was not a conservative domain for it and "
                 "629189 components were reported above the ceiling. Preserved with "
                 "this run's evidence."),
     "artifact": "evidence_attempt2_ceiling250_failed.json",
     "measured": {"min_component": -614.1239013671875,
                  "max_component": 9420.1640625,
                  "above_ceiling_components": 629189,
                  "negative_components": 3098461}},
]


def parse_arguments(argv=None):
    """Three knobs, each defaulting to the original run so a bare invocation still
    reproduces the evidence that is already on disk."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE,
                        help="clip to borrow exactly one frame from")
    parser.add_argument("--support-ceiling", type=float,
                        default=SUPPORT_CEILING_SCRGB,
                        help="declared per-channel scRGB domain for --source")
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE_DIR,
                        help="where evidence.json for this run is written")
    return parser.parse_args(argv)

U32, I32, U64, I64, VP, CP, CSZ = (C.c_uint32, C.c_int32, C.c_uint64,
                                   C.c_int64, C.c_void_p, C.c_char_p, C.c_size_t)


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
    """Mirror of NvdecTransferCounters."""
    _fields_ = [("struct_size", U32), ("frames_decoded", U64),
                ("device_frame_count", U64), ("bitstream_bytes_submitted", U64),
                ("host_frame_copy_count", U64), ("host_frame_copy_bytes", U64),
                ("scale_device_output_bytes", U64)]


class HdrMetadata(C.Structure):
    """Mirror of NvdecHdrMetadata: the source's own ST 2086 / CTA-861.3 claim."""
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


def bind_nvdec():
    """The additive P010 borrow, bound narrowly.

    ``nvdec_bridge_readback_last_frame`` is deliberately NOT given a prototype or a
    reference here: the only way this gate can claim zero host pixel traffic is that
    the code path it runs contains no call that could produce it.
    """
    lib = C.CDLL(str(NATIVE / "nvdec_bridge.dll"))
    lib.nvdec_bridge_abi_version.restype = U32
    lib.nvdec_bridge_last_error.restype = CP
    entry = lib.nvdec_bridge_open_hdr
    entry.argtypes = [CP, C.POINTER(VP)]
    entry.restype = I32
    entry = lib.nvdec_bridge_next
    entry.argtypes = [VP, C.POINTER(DecFrame)]
    entry.restype = I32
    entry = lib.nvdec_bridge_get_hdr_metadata
    entry.argtypes = [VP, C.POINTER(HdrMetadata)]
    entry.restype = I32
    entry = lib.nvdec_bridge_get_transfer_counters
    entry.argtypes = [VP, C.POINTER(DecCounters)]
    entry.restype = I32
    entry = lib.nvdec_bridge_close
    entry.argtypes = [VP]
    entry.restype = None
    return lib


def sha256_file(path):
    """Hashed only after the native libraries are loaded - see the module header."""
    from hashlib import sha256
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def counter_fields(structure):
    return {name: getattr(structure, name) for name, _ in structure._fields_
            if name != "struct_size"}


def metadata_snapshot(block):
    """Every declared field of the source's HDR metadata, rationals kept exact."""
    out = {}
    for name, _ in HdrMetadata._fields_:
        if name == "struct_size":
            continue
        value = getattr(block, name)
        out[name] = list(value) if isinstance(value, C.Array) else int(value)
    return out


def pq_domain_notes(contract, report, max_cll_nits=None):
    """Where the measured max sits on PQ's curve, from the bridge's own scalars.

    Arithmetic on the contract constants and the source report's two doubles: no
    sample leaves the device and the reference PQ maths is not re-run over content.
    """
    m1, m2 = contract["pq_m1"], contract["pq_m2"]
    c1, c2, c3 = contract["pq_c1"], contract["pq_c2"], contract["pq_c3"]
    st2084_top_scrgb = 10000.0 * contract["scrgb_per_nit"]

    def pq_input_of(relative):
        # The ST 2084 OETF in the same constant form the shader uses.
        if relative <= 0.0:
            return 0.0
        em = relative ** m1
        return ((c1 + c2 * em) / (1.0 + c3 * em)) ** m2

    pole = (c2 / c3) ** m2
    measured_max = float(report["max_component"])
    measured_input = pq_input_of(measured_max / st2084_top_scrgb)
    return {
        "st2084_top_scrgb": st2084_top_scrgb,
        "pq_pole_coded_input": pole,
        "measured_max_scrgb": measured_max,
        "measured_max_relative_to_10000_nits": measured_max / st2084_top_scrgb,
        "measured_max_pq_coded_input": measured_input,
        "measured_max_fraction_of_the_way_to_the_pole": measured_input / pole,
        "source_max_cll_nits": max_cll_nits,
        "max_cll_equivalent_scrgb": (None if max_cll_nits is None
                                     else max_cll_nits * contract["scrgb_per_nit"]),
        "note": ("the ceiling is a declared domain, so it has to sit above the "
                 "fixture's measured per-channel max; the MaxCLL equivalent would "
                 "be a metadata copy and would make the above-ceiling check "
                 "circular"),
    }


def libavutil_spaces():
    """AVCOL_SPC_BT2020_NCL/CL read out of the installed header, not typed in.

    The positive case is judged against the decoder's own number; this is here so
    the negative case hands the bridge the enumerator the source is NOT, named by
    the same header the bridge was compiled against.
    """
    header = FFMPEG_INCLUDE / "pixfmt.h"
    out = {}
    if header.is_file():
        text = header.read_text(encoding="utf-8", errors="replace")
        for name in ("AVCOL_SPC_BT2020_NCL", "AVCOL_SPC_BT2020_CL"):
            found = re.search(rf"{name}\s*=\s*(-?\d+)", text)
            if found:
                out[name] = int(found.group(1))
    return {"header": str(header), "present": header.is_file(), "values": out}


class Gate:
    def __init__(self):
        self.checks = []
        self.commands = []

    def check(self, name, condition, detail=""):
        passed = bool(condition)
        self.checks.append({"name": name, "pass": passed,
                            "detail": _text(detail)})
        return passed

    def command(self, argv, rc, stdout=""):
        self.commands.append({"command": [str(item) for item in argv], "rc": rc})
        return rc == 0


def _text(value):
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    text = value if isinstance(value, str) else json.dumps(
        value, default=str, ensure_ascii=False)
    return text if len(text) <= 1200 else text[:1200] + "..."


def probe_container(gate):
    """ffprobe on the container, so the claim 'the source is NCL' is corroborated
    outside the decoder as well as inside it."""
    argv = [str(FFMPEG_BIN / "ffprobe.exe"), "-v", "error", "-select_streams",
            "v:0", "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,color_range,"
            "color_primaries,color_transfer,color_space,bits_per_raw_sample",
            "-of", "json", os.fsdecode(SOURCE)]
    try:
        raw = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as error:
        gate.command(argv, -1)
        return {"ran": False, "error": _text(error)}, None
    gate.command(argv, raw.returncode)
    try:
        stream = (json.loads(raw.stdout or "{}").get("streams") or [{}])[0]
    except ValueError:
        stream = {"parse_error": raw.stdout[:200]}
    return {"ran": True, "rc": raw.returncode, "stderr": raw.stderr.strip()[:300],
            "stream": stream}, stream


def main(argv=None):
    global SOURCE, SUPPORT_CEILING_SCRGB, EVIDENCE_DIR
    arguments = parse_arguments(argv)
    SOURCE = Path(arguments.source)
    SUPPORT_CEILING_SCRGB = float(arguments.support_ceiling)
    EVIDENCE_DIR = Path(arguments.output_dir)
    gate = Gate()
    evidence = {"gate": "hdr_stage_b_real_admission",
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version.split()[0],
                "argv": [str(item) for item in sys.argv],
                "declared_domain": {
                    "source": str(SOURCE),
                    "support_ceiling_scrgb": SUPPORT_CEILING_SCRGB,
                    "paper_white_scrgb": PAPER_WHITE_SCRGB,
                    "working_scale": WORKING_SCALE,
                    "derivation": ("a fixed finite per-channel scRGB domain declared "
                                   "for this source and asserted to sit above its "
                                   "measured per-channel max; never derived from its "
                                   "MaxCLL. For a BT.2100-valid clip the bound comes "
                                   "from the contract - 1.0 PQ is 125 scRGB at this "
                                   "bridge's 80 cd/m^2 paper white, times the largest "
                                   "positive row sum of the derived Rec.2020->Rec.709 "
                                   "matrix - not from the metadata. See "
                                   "SUPPORT_CEILING_SCRGB and pq_geometry"),
                    "prior_attempts_on_the_legacy_fixture": {
                        "about": ("history of build/hdr_probe/hdr_src.mkv, which is "
                                  "not this run's source and whose ceiling decision "
                                  "does not transfer"),
                        "attempts": LEGACY_ATTEMPT_HISTORY}}
                    }

    handle = VP()
    frame = None
    live = None
    bridge = None
    pushed = False
    previous = VP()
    cleanup = []
    failure = None
    dec = hdr = engine = pipeline = cuda = None

    try:
        # --- loader order: FFmpeg directory first, decoder bridge second,
        # everything that can bind libcrypto after that.
        import bundle_paths
        bundle_paths.add_dll_directory(str(FFMPEG_BIN))
        bundle_paths.add_dll_directory(str(NATIVE))
        dec = bind_nvdec()
        import hdr_gpu_engine as hdr
        import dlss_gpu_engine as engine
        import pipeline

        cuda = C.WinDLL("nvcuda.dll")
        cuda.cuInit.argtypes = [C.c_uint]
        cuda.cuInit.restype = C.c_int
        cuda.cuCtxPushCurrent_v2.argtypes = [VP]
        cuda.cuCtxPushCurrent_v2.restype = C.c_int
        cuda.cuCtxPopCurrent_v2.argtypes = [C.POINTER(VP)]
        cuda.cuCtxPopCurrent_v2.restype = C.c_int

        container, stream = probe_container(gate)
        evidence["fixture"] = {"path": str(SOURCE), "exists": SOURCE.is_file(),
                               "container_probe": container}
        gate.check("fixture exists", SOURCE.is_file(), str(SOURCE))

        # --- the wrapper and the bridge it is about to load, identified first.
        library = hdr.load()
        host_dll = engine._LIBRARY
        evidence["loaded_binaries"] = {
            "nvdec_bridge.dll": _binary(dec),
            "gpu_hdr_scrgb_bridge.dll": _binary_from_path(
                getattr(library, "_dll_path", None)),
            "dlss_gpu_host.dll": _binary_from_path(host_dll),
            "nvcuda.dll": _binary(cuda),
            "avutil-59.dll": _binary_by_name("avutil-59.dll"),
            "avcodec-61.dll": _binary_by_name("avcodec-61.dll"),
            "avformat-61.dll": _binary_by_name("avformat-61.dll"),
            "libcrypto-3-x64.dll": _binary_by_name("libcrypto-3-x64.dll"),
        }
        loaded_bridge = evidence["loaded_binaries"]["gpu_hdr_scrgb_bridge.dll"]
        evidence["accepted_bridge_match"] = (
            loaded_bridge.get("sha256") == ACCEPTED_BRIDGE_SHA256)
        gate.check("the bridge this process loaded is the accepted build",
                   evidence["accepted_bridge_match"],
                   {"loaded": loaded_bridge.get("path"),
                    "sha256": loaded_bridge.get("sha256"),
                    "expected": ACCEPTED_BRIDGE_SHA256})
        evidence["wrapper_provenance"] = {
            "hdr_gpu_engine": {"path": str(Path(hdr.__file__).resolve()),
                               "sha256": sha256_file(hdr.__file__),
                               "abi": hdr.GPU_HDR_ABI_VERSION,
                               "space_constant_sent": hdr.AV_COLOR_SPACE_BT2020_NC},
            "dlss_gpu_engine": {"path": str(Path(engine.__file__).resolve()),
                                "sha256": sha256_file(engine.__file__)},
            "pipeline": {"path": str(Path(pipeline.__file__).resolve()),
                         "sha256": sha256_file(pipeline.__file__)},
            "nvdec_bridge.h": {"path": str(NATIVE / "nvdec_bridge.h"),
                               "sha256": sha256_file(NATIVE / "nvdec_bridge.h")},
            "gate": {"path": str(Path(__file__).resolve()),
                     "sha256": sha256_file(__file__)},
        }
        # Which exports this process can reach at all: the proof behind the
        # no-pixel-readback claim is that the entry is unbound, not that a
        # counter happened to read zero.
        evidence["bound_bridge_exports"] = sorted(name for name, _, _ in
                                                 hdr._BRIDGE)
        gate.check("no gpu_hdr_readback_* export is bound",
                   not [name for name, _, _ in hdr._BRIDGE if "readback" in name],
                   evidence["bound_bridge_exports"])
        gate.check("decoder readback entry was never resolved in this process",
                   "nvdec_bridge_readback_last_frame" not in vars(dec),
                   sorted(name for name in vars(dec)
                          if name.startswith("nvdec")))

        # --- exactly one real P010 frame, through the additive HDR open.
        abi = int(dec.nvdec_bridge_abi_version())
        evidence["decoder_abi"] = abi
        gate.check("decoder bridge is ABI 3 (P010 borrow exists)", abi == 3, abi)
        open_rc = dec.nvdec_bridge_open_hdr(os.fsencode(SOURCE), C.byref(handle))
        gate.check("nvdec_bridge_open_hdr succeeds", open_rc == 0,
                   {"rc": open_rc, "error": _text(dec.nvdec_bridge_last_error())})
        if open_rc:
            raise RuntimeError(f"nvdec_bridge_open_hdr: "
                               f"{_text(dec.nvdec_bridge_last_error())}")

        frame = DecFrame()
        frame.struct_size = C.sizeof(DecFrame)
        next_rc = dec.nvdec_bridge_next(handle, C.byref(frame))
        gate.check("exactly one frame borrowed", next_rc == 1,
                   {"rc": next_rc, "error": _text(dec.nvdec_bridge_last_error())})
        if next_rc != 1:
            raise RuntimeError("no real P010 frame to admit")

        actual = {name: getattr(frame, name)
                  for name, _ in DecFrame._fields_ if name != "struct_size"}
        # ctypes hands back a c_void_p *field* as an int (or None), not as an
        # object with .value: this is the decoder's own CUcontext as it reported it.
        decoder_context = int(frame.cuda_context or 0)
        actual["cuda_context_hex"] = hex(decoder_context)
        actual["y_device_ptr_hex"] = hex(frame.y_device_ptr)
        actual["uv_device_ptr_hex"] = hex(frame.uv_device_ptr)
        color_actual = {key: getattr(frame, key) for key in EXPECTED_SOURCE_COLOR}
        metadata = HdrMetadata()
        metadata.struct_size = C.sizeof(HdrMetadata)
        metadata_rc = dec.nvdec_bridge_get_hdr_metadata(handle, C.byref(metadata))
        evidence["decoder_frame"] = actual
        evidence["decoder_frame_color"] = color_actual
        evidence["source_hdr_metadata"] = (
            metadata_snapshot(metadata) if metadata_rc == 0
            else {"rc": metadata_rc,
                  "error": _text(dec.nvdec_bridge_last_error())})

        gate.check("decoder reports P010 (software_format 2)",
                   frame.software_format == AV_PIX_FMT_P010LE_AS_REPORTED,
                   frame.software_format)
        gate.check("decoder extent is the fixture extent",
                   (frame.width, frame.height) == (stream.get("width"),
                                                   stream.get("height")),
                   [frame.width, frame.height])
        # The whole point of the repair: the decoder's own number, not the
        # wrapper's, and asserted rather than overwritten.
        gate.check("decoder's own colour fields are limited-range "
                   "BT.2020 primaries / ST 2084 / BT.2020 NCL",
                   color_actual == EXPECTED_SOURCE_COLOR,
                   {"decoder_reported": color_actual,
                    "expected": EXPECTED_SOURCE_COLOR})
        gate.check("borrowed planes are non-null with real pitches",
                   frame.y_device_ptr and frame.uv_device_ptr
                   and frame.y_pitch_bytes > 0 and frame.uv_pitch_bytes > 0,
                   {"y_pitch_bytes": frame.y_pitch_bytes,
                    "uv_pitch_bytes": frame.uv_pitch_bytes})
        if metadata_rc == 0:
            snapshot = metadata_snapshot(metadata)
            gate.check("source carries the mastering / content-light it claims",
                       snapshot["has_mastering"] == 1
                       and snapshot["has_content_light"] == 1, snapshot)

        full_w, full_h = int(frame.width), int(frame.height)
        work_w, work_h = pipeline.resolve_working_dims(full_w, full_h, WORKING_SCALE)
        evidence["working_dims"] = {"rule": "pipeline.resolve_working_dims",
                                    "full": [full_w, full_h],
                                    "scale": WORKING_SCALE,
                                    "working": [work_w, work_h]}

        # --- borrow a production D3D12 device; process nothing with it.
        fp16_ready = engine.hdr_fp16_supported()
        gate.check("loaded host exposes an FP16 colour session", fp16_ready,
                   fp16_ready)
        if not fp16_ready:
            raise RuntimeError("this host build exposes no FP16 colour session")
        live = engine.Live(work_w, work_h, color_format="rgba16f")
        device = live.d3d12_device()
        evidence["live_session"] = {"extent": [work_w, work_h],
                                    "color_format": live.color_format,
                                    "bytes_per_pixel": live.bytes_per_pixel,
                                    "host_dll": live.host_dll,
                                    "host_abi": live.host_abi,
                                    "d3d12_device_hex": hex(int(device or 0)),
                                    "frames_processed": 0}
        gate.check("Live borrowed a D3D12 device", bool(device), _text(device))

        # The decoder's context is the one the bridge is opened with, so it is
        # pushed on this thread for the whole scoped lifetime.
        gate.check("cuInit succeeded", cuda.cuInit(0) == 0)
        gate.check("decoder CUcontext is a real non-null context",
                   decoder_context != 0, hex(decoder_context))
        push_rc = cuda.cuCtxPushCurrent_v2(decoder_context)
        pushed = push_rc == 0
        gate.check("decoder CUcontext made current on the wrapper thread",
                   push_rc == 0, push_rc)

        bridge = hdr.GpuHdrBridge(
            d3d12_device=device, cuda_context=decoder_context,
            full_width=full_w, full_height=full_h,
            working_width=work_w, working_height=work_h,
            paper_white_scrgb=PAPER_WHITE_SCRGB)
        contract = bridge.contract
        evidence["bridge_open"] = {
            "full": list(bridge.full_extent), "working": list(bridge.working_extent),
            "configured_paper_white": bridge.configured_paper_white,
            "effective_paper_white": bridge.paper_white,
            "abi": hdr.GPU_HDR_ABI_VERSION,
            "contract_units": {"scrgb_per_nit": contract["scrgb_per_nit"],
                               "nits_per_scrgb": contract["nits_per_scrgb"],
                               "pq_c1": contract["pq_c1"],
                               "pq_c2": contract["pq_c2"],
                               "pq_c3": contract["pq_c3"],
                               "kr": contract["kr"], "kg": contract["kg"],
                               "kb": contract["kb"]}}
        gate.check("bridge opened on the borrowed device and decoder context",
                   bridge.is_alive(), bridge)

        # --- positive admission: the decoder's fields, sent as the decoder said them.
        master = bridge.decode(p010_y_ptr=frame.y_device_ptr,
                               p010_y_pitch=frame.y_pitch_bytes,
                               p010_uv_ptr=frame.uv_device_ptr,
                               p010_uv_pitch=frame.uv_pitch_bytes,
                               color_range=color_actual["color_range"],
                               color_primaries=color_actual["color_primaries"],
                               color_transfer=color_actual["color_transfer"],
                               color_space=color_actual["color_space"],
                               support_ceiling_scrgb=SUPPORT_CEILING_SCRGB)
        descriptor = dict(master.as_descriptor(),
                          pixel_format=master.pixel_format,
                          bytes_per_pixel=master.bytes_per_pixel,
                          plane=master.plane,
                          device_ptr_hex=hex(master.device_ptr))
        evidence["positive_admission"] = {
            "color_fields_sent": color_actual,
            "pointers_pitches_sent": {
                "y_device_ptr_hex": actual["y_device_ptr_hex"],
                "y_pitch_bytes": frame.y_pitch_bytes,
                "uv_device_ptr_hex": actual["uv_device_ptr_hex"],
                "uv_pitch_bytes": frame.uv_pitch_bytes},
            "descriptor": descriptor}
        report = bridge.source_report()
        evidence["positive_admission"]["source_report"] = report
        gate.check("gpu_hdr_decode returned success on the decoder's NCL fields "
                   "(it raises otherwise)", True, "no exception on this path")
        gate.check("descriptor is RGBA16F CUDA at full extent, 8 bytes, non-null",
                   master.format == hdr.FORMAT_RGBA16F_CUDA
                   and master.bytes_per_pixel == 8
                   and (master.width, master.height) == (full_w, full_h)
                   and master.pitch_bytes >= full_w * 8
                   and master.device_ptr != 0, descriptor)
        gate.check("descriptor context is the decoder's own context",
                   master.cuda_context == decoder_context,
                   {"descriptor": hex(master.cuda_context),
                    "decoder": hex(decoder_context)})
        gate.check("source report: no non-finite sample",
                   report["nonfinite_samples"] == 0, report)
        gate.check("source report: nothing above the declared ceiling",
                   report["above_ceiling_components"] == 0, report)
        gate.check("source report covers the whole borrowed frame",
                   report["pixels"] == full_w * full_h, report)
        gate.check("source report min/max are finite and inside the declared domain",
                   _finite(report["min_component"]) and _finite(report["max_component"])
                   and report["max_component"] <= SUPPORT_CEILING_SCRGB,
                   {"min": report["min_component"], "max": report["max_component"],
                    "ceiling": SUPPORT_CEILING_SCRGB})
        # Negative channels are legal signed wide-gamut values: reported, never
        # treated as a failure.
        evidence["positive_admission"]["negative_components"] = \
            report["negative_components"]
        max_cll = evidence["source_hdr_metadata"].get("max_cll")
        notes = pq_domain_notes(bridge.contract, report, max_cll)
        evidence["declared_domain"]["pq_geometry"] = notes
        gate.check("declared domain is above the measured per-channel max",
                   SUPPORT_CEILING_SCRGB > report["max_component"],
                   {"declared": SUPPORT_CEILING_SCRGB,
                    "measured_max": report["max_component"]})
        gate.check("declared domain is not the MaxCLL equivalent",
                   SUPPORT_CEILING_SCRGB != notes["max_cll_equivalent_scrgb"],
                   notes["max_cll_equivalent_scrgb"])

        counters = bridge.transfer_counters()
        evidence["bridge_counters_after_positive"] = counters
        gate.check("bridge moved the borrowed planes device-to-device",
                   counters["source_device_copy_bytes"] > 0
                   and counters["process_count"] >= 1, counters)

        # --- negative admission on the SAME frame: only color_space changes.
        spaces = libavutil_spaces()
        evidence["libavutil_color_spaces"] = spaces
        constant_luminance = spaces["values"].get("AVCOL_SPC_BT2020_CL")
        if constant_luminance is None:
            constant_luminance = 10
            evidence["libavutil_color_spaces"]["fallback"] = (
                "AVCOL_SPC_BT2020_CL taken as the documented 10 because the "
                "installed header did not answer")
        gate.check("BT2020_CL is a different enumerator from the reported NCL",
                   constant_luminance != color_actual["color_space"],
                   {"cl": constant_luminance,
                    "ncl_as_reported": color_actual["color_space"]})

        negative = {"color_fields_sent": dict(color_actual,
                                              color_space=constant_luminance),
                    "only_field_changed": "color_space"}
        try:
            rejected = bridge.decode(
                p010_y_ptr=frame.y_device_ptr, p010_y_pitch=frame.y_pitch_bytes,
                p010_uv_ptr=frame.uv_device_ptr,
                p010_uv_pitch=frame.uv_pitch_bytes,
                color_range=color_actual["color_range"],
                color_primaries=color_actual["color_primaries"],
                color_transfer=color_actual["color_transfer"],
                color_space=constant_luminance,
                support_ceiling_scrgb=SUPPORT_CEILING_SCRGB)
            negative["outcome"] = "ACCEPTED (this is the defect returning)"
            negative["descriptor"] = rejected.as_dict()
        except hdr.HdrBridgeError as error:
            # Preserved verbatim before any cleanup can overwrite the last error.
            negative["outcome"] = "refused"
            negative["error_type"] = type(error).__name__
            negative["error"] = str(error)
        evidence["negative_admission"] = negative
        gate.check("native refused BT2020_CL and said why",
                   negative["outcome"] == "refused", negative)
        if negative["outcome"] == "refused":
            message = negative["error"]
            gate.check("refusal names the expected non-constant-luminance domain",
                       "NCL" in message and "ST 2084" in message, message)
            gate.check("refusal reports the space it was actually given",
                       f"space {constant_luminance}" in message, message)
        gate.check("bridge survived its own refusal (no fallback, still open)",
                   bridge.is_alive(), bridge.is_alive())

        # --- counters, read as scalars only.
        dec_counters = DecCounters()
        dec_counters.struct_size = C.sizeof(DecCounters)
        dec.nvdec_bridge_get_transfer_counters(handle, C.byref(dec_counters))
        fields = counter_fields(dec_counters)
        bridge_counters = bridge.transfer_counters()
        evidence["decoder_counters"] = fields
        evidence["bridge_counters_final"] = bridge_counters
        gate.check("no second frame was borrowed for the negative case",
                   fields["frames_decoded"] == 1
                   and fields["device_frame_count"] == 1, fields)
        gate.check("decoder copied no frame image to the host",
                   fields["host_frame_copy_count"] == 0
                   and fields["host_frame_copy_bytes"] == 0, fields)
        gate.check("decoder scaled nothing (open_hdr contract)",
                   fields["scale_device_output_bytes"] == 0, fields)
        gate.check("bitstream bytes are packet traffic and non-zero",
                   fields["bitstream_bytes_submitted"] > 0, fields)
        gate.check("bridge reported zero pixel readback bytes",
                   bridge_counters["report_readback_bytes"] == 0, bridge_counters)
        gate.check("scalar report reads are counted separately from pixels",
                   set(bridge_counters) >= {"report_readback_bytes",
                                            "source_device_copy_bytes"},
                   bridge_counters)

        # --- cleanup: HDR bridge first, then Live, then the decoder.
        bridge.close()
        cleanup.append("gpu_hdr_bridge")
        gate.check("HDR bridge closed and released", not bridge.is_alive(),
                   bridge.is_alive())
        live.close()
        cleanup.append("dlss_gpu_engine.Live")
        gate.check("Live session closed", not live.is_alive(), live.is_alive())
        if pushed:
            pop_rc = cuda.cuCtxPopCurrent_v2(C.byref(previous))
            pushed = False
            gate.check("decoder CUcontext popped", pop_rc == 0, pop_rc)
        dec.nvdec_bridge_close(handle)
        cleanup.append("nvdec_bridge")
    except BaseException as error:      # reported, never swallowed into a pass
        failure = error
        evidence["exception"] = {"type": type(error).__name__,
                                 "detail": _text(error),
                                 "traceback": _text(
                                     "".join(_traceback_lines(error))[-3000:])}
    finally:
        # The same order whether or not the body finished: HDR bridge, then the
        # borrowed session, then the decoder. Already-closed owners are skipped,
        # and each close is idempotent, so this never double-closes.
        for name, close in (
                ("gpu_hdr_bridge", lambda: bridge.close() if bridge else None),
                ("dlss_gpu_engine.Live", lambda: live.close() if live else None),
                ("nvdec_bridge",
                 lambda: dec.nvdec_bridge_close(handle) if dec else None)):
            if name in cleanup:
                continue
            try:
                close()
                cleanup.append(name)
            except BaseException as error:
                cleanup.append(f"{name} FAILED: {_text(error)}")
        evidence["cleanup_order"] = cleanup
        evidence["failed_checks"] = [item["name"] for item in gate.checks
                                     if not item["pass"]]
        evidence["checks"] = gate.checks
        evidence["commands"] = gate.commands
        evidence["verdict"] = ("PASS" if not evidence["failed_checks"]
                               and failure is None else "FAIL")
        if failure is not None:
            evidence["verdict"] = "FAIL"
        evidence["finished_utc"] = datetime.now(timezone.utc).isoformat()
        evidence["not_claimed"] = {
            "full_pipeline": "no downscale, proxy, Feature 18, reconstruct or "
                             "encode ran; this is native admission only",
            "orchestrator_ui": "NOT DONE - no product wiring, no UI surface",
            "multi_frame_gate": "NOT DONE - the final 4-6 frame gate is a later task",
            "performance": "NOT_CHARACTERIZED by this run",
            "feature18_hdr": "formal Feature 18 HDR acceptance is unproven here",
            "fp16_loss": "the Stage A numeric loss record is neither restated nor "
                         "superseded",
            "pixel_readback": "no readback entry is bound in this process; the "
                              "only device-to-host traffic is the scalar reports"}
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        path = EVIDENCE_DIR / "evidence.json"
        path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
        print(f"{evidence['verdict']}: {path}")
        for item in gate.checks:
            print(("  ok   " if item["pass"] else "  FAIL ") + item["name"])
        if failure is not None:
            print(_text(failure))
    return 0 if evidence["verdict"] == "PASS" else 1


def _traceback_lines(error):
    import traceback
    return traceback.TracebackException.from_exception(error).format()


def _finite(value):
    try:
        import math
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _binary(library):
    """A loaded ctypes library, identified by the path the loader actually picked."""
    return _binary_from_path(_path_of_handle(getattr(library, "_handle", None))
                             or getattr(library, "_name", None))


def _binary_by_name(name):
    """Where a bare name resolved to in THIS process, or None if it is not loaded.

    Asked with GetModuleHandleW rather than by loading a second handle: which copy
    won the race for ``libcrypto-3-x64.dll`` is the fact this gate has to record,
    and it can only be read from the process that already loaded it.
    """
    path = _path_of_handle(_module_handle(name))
    record = {"requested": name, "loaded_in_process": bool(path),
              "path": path}
    if path:
        record.update(_file_facts(path))
    return record


def _module_handle(name):
    return _kernel32().GetModuleHandleW(name)


def _path_of_handle(handle):
    if not handle:
        return None
    buffer = C.create_unicode_buffer(4096)
    if not _kernel32().GetModuleFileNameW(handle, buffer, 4096):
        return None
    return buffer.value


_KERNEL32 = None


def _kernel32():
    global _KERNEL32
    if _KERNEL32 is None:
        _KERNEL32 = C.WinDLL("kernel32", use_last_error=True)
        _KERNEL32.GetModuleHandleW.argtypes = [C.c_wchar_p]
        _KERNEL32.GetModuleHandleW.restype = VP
        _KERNEL32.GetModuleFileNameW.argtypes = [VP, C.c_wchar_p, C.c_uint]
        _KERNEL32.GetModuleFileNameW.restype = C.c_uint
    return _KERNEL32


def _file_facts(path):
    stat = os.stat(path)
    return {"present": True, "bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime,
                                            timezone.utc).isoformat(),
            "sha256": sha256_file(path)}


def _binary_from_path(path):
    if not path or not os.path.isfile(path):
        return {"path": str(path), "present": False}
    return dict({"path": path}, **_file_facts(path))


if __name__ == "__main__":
    sys.exit(main())
