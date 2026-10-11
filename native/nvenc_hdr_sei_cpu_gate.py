"""CPU-only gate for the HDR10 in-bitstream metadata writer.

Nothing here touches a GPU, a decoder or an encoder: the writer's public
entries take compressed HEVC bytes and return compressed HEVC bytes, so every
expectation below is computed in this file, from ITU-T H.265 Annex D and Annex
B, independently of the C++ that implements the same rules. Two implementations
of the arithmetic agreeing is the point; one restating the other would prove
nothing.

The real-packet case reads the elementary stream the previous stage captured
from NVENC (bench_out/hdr_p010_nvenc/a_hdr10_out.hevc), splits it into the two
access units it actually contains, authors into each, and then compares every
original NAL byte - parameter sets, the encoder's own payloadType 1 prefix SEI
and both slices - against the input. The result is probed with the installed
ffprobe, which reads the SEI we wrote with no container metadata to help it.

Run from anywhere:
    python native/nvenc_hdr_sei_cpu_gate.py
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


def sha256(data):
    """Hashed lazily, never at import time.

    Importing hashlib up front makes python's _hashlib bind libcrypto-3-x64.dll
    from its own runtime directory, and the FFmpeg avformat chain the bridges
    depend on then resolves that same name against a copy that does not export
    what libssl-3-x64.dll needs: the bridge load fails with WinError 127 before
    a single byte is parsed. Loading the bridges first and hashing afterwards
    puts the two in the order the loader can actually satisfy.
    """
    from hashlib import sha256 as algorithm
    return algorithm(data)


ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "native"
ARTIFACTS = Path(os.environ.get("HDR_SEI_GATE_DIR", str(ROOT / "bench_out" / "hdr_sei_repair")))
BASELINE = ROOT / "bench_out" / "hdr_p010_nvenc" / "a_hdr10_out.hevc"
FFMPEG_DIR = Path(os.environ.get("FFMPEG_SDK_DIR", r"F:\WhisperJAV\Library"))
FFMPEG_BIN = FFMPEG_DIR / "bin"

U32, I32, U64, VP, CP = (C.c_uint32, C.c_int32, C.c_uint64, C.c_void_p,
                         C.c_char_p)

# From native/nvenc_bridge.h. The compiled numbers are re-pinned against the
# probe's own report in the native gate; these are the writer's ABI constants.
SEI_ABI = 1
FRAMING_ANNEXB4, FRAMING_ANNEXB3, FRAMING_LP4 = 1, 2, 3
OK, NOTHING_TO_DO, BAD_RATIONAL, OUT_OF_RANGE = 0, 1, 3, 4
CONFLICT, FRAMING_UNSUPPORTED, NO_VCL, MALFORMED, CAPACITY = 5, 6, 7, 8, 9

NAL_TYPE_PREFIX_SEI = 39
PAYLOAD_MASTERING, PAYLOAD_LIGHT = 137, 144
CHROMA_DEN, LUMA_DEN = 50000, 10000
FNV_OFFSET, FNV_PRIME, MASK64 = 1469598103934665603, 1099511628211, (1 << 64) - 1

# AVMasteringDisplayMetadata and NvencHdrMetadata store the primaries red,
# green, blue. The wire is ordered by the syntax loop index c, which H.265 and
# ST 2086 both define as green, blue, red, so wire c reads struct index:
WIRE_TO_STRUCT = {0: 1, 1: 2, 2: 0}


class HdrMetadata(C.Structure):
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


class SeiCapability(C.Structure):
    _fields_ = [("struct_size", U32), ("writer_available", I32),
                ("encoder_offers_metadata_option", I32),
                ("supports_partial_mastering", I32), ("primary_order", I32),
                ("chroma_denominator", I32), ("luma_denominator", I32),
                ("payload_type_mastering", I32),
                ("payload_type_content_light", I32), ("prefix_sei_nal_type", I32),
                ("mastering_payload_size", U32),
                ("content_light_payload_size", U32), ("max_nal_bytes", U32),
                ("max_insert_bytes", U32)]


class SeiInfo(C.Structure):
    _fields_ = [("struct_size", U32), ("error", I32), ("framing", I32),
                ("nal_count", I32), ("vcl_index", I32),
                ("mastering_present", I32), ("content_light_present", I32),
                ("mastering_equal", I32), ("content_light_equal", I32),
                ("mastering_inserted", I32), ("content_light_inserted", I32),
                ("nal_bytes", U32), ("bytes_inserted", U32),
                ("detail", C.c_char * 192)]


def fields(structure):
    return {name: getattr(structure, name) for name, _ in structure._fields_
            if name != "struct_size"}


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
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).hex()
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= 900 else text[:900] + "..."


def note(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


# --------------------------------------------------------------------- #
# The expectations, computed here from the standard.
# --------------------------------------------------------------------- #

def grid(num, den, scale, low, high):
    """The value on the wire grid, or None when the declaration cannot be placed
    on it. Validate before nearest-unit quantization; no float anywhere."""
    if den <= 0 or num < 0:
        return None
    value = Fraction(int(num), int(den)) * scale
    if not low <= value <= high:
        return None
    whole, remainder = divmod(value.numerator, value.denominator)
    return whole + int(2 * remainder >= value.denominator)


def declared_ints(md):
    """The ten mastering integers in wire order: for c = 0..2 the x and y of
    green, blue and red, then white point x and y, then max luminance, then min."""
    ints = []
    for c in range(3):
        index = WIRE_TO_STRUCT[c]
        ints.append(grid(md.primary_x_num[index], md.primary_x_den[index],
                         CHROMA_DEN, 5, 37000))
        ints.append(grid(md.primary_y_num[index], md.primary_y_den[index],
                         CHROMA_DEN, 5, 42000))
    ints.append(grid(md.white_x_num, md.white_x_den, CHROMA_DEN, 5, 37000))
    ints.append(grid(md.white_y_num, md.white_y_den, CHROMA_DEN, 5, 42000))
    ints.append(grid(md.max_luminance_num, md.max_luminance_den, LUMA_DEN,
                     50000, 100000000))
    ints.append(grid(md.min_luminance_num, md.min_luminance_den, LUMA_DEN,
                     0, 50000))
    return ints


def mastering_complete(md):
    return bool(md.has_mastering and md.mastering_has_primaries and
                md.mastering_has_luminance)


def expected_mastering_payload(md):
    """The 24 payload bytes of D.2.27: eight u(16) (three primaries as x,y in
    green, blue, red order, then the white point) then max and min luminance as
    u(32), in that order."""
    ints = declared_ints(md)
    if any(value is None for value in ints) or ints[9] >= ints[8]:
        return None
    body = b"".join(int.to_bytes(value, 2, "big") for value in ints[:8])
    body += int.to_bytes(ints[8], 4, "big") + int.to_bytes(ints[9], 4, "big")
    assert len(body) == 24, len(body)
    return body


def expected_light_payload(md):
    if md.max_cll > 0xFFFF or md.max_fall > 0xFFFF:
        return None
    return (int.to_bytes(md.max_cll, 2, "big") +
            int.to_bytes(md.max_fall, 2, "big"))


def expected_rbsp(md):
    rbsp = b""
    if mastering_complete(md):
        payload = expected_mastering_payload(md)
        if payload is None:
            return None
        rbsp += bytes([PAYLOAD_MASTERING, 24]) + payload
    if md.has_content_light:
        payload = expected_light_payload(md)
        if payload is None:
            return None
        rbsp += bytes([PAYLOAD_LIGHT, 4]) + payload
    if not rbsp:
        return None
    return rbsp + b"\x80"          # rbsp_stop_one_bit, then alignment zero bits


def escape(rbsp):
    """Annex B.1.3: an emulation_prevention_three_byte before a 0x00..0x03 that
    would otherwise follow two zero bytes."""
    out, zeros = bytearray(), 0
    for byte in rbsp:
        if zeros >= 2 and byte <= 0x03:
            out.append(0x03)
            zeros = 0
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
    return bytes(out)


def expected_nal(md):
    rbsp = expected_rbsp(md)
    if rbsp is None:
        return None
    # forbidden_zero_bit 0, nal_unit_type 39, nuh_layer_id 0, tid_plus1 1
    return bytes([0x40 | (NAL_TYPE_PREFIX_SEI << 1), 0x01]) + escape(rbsp)


def unescape(data):
    out, zeros, index = bytearray(), 0, 0
    while index < len(data):
        byte = data[index]
        if (zeros >= 2 and byte == 0x03 and index + 1 < len(data) and
                data[index + 1] <= 0x03):
            index += 1                     # the escape is not data
            byte = data[index]
        out.append(byte)
        zeros = zeros + 1 if byte == 0 else 0
        index += 1
    return bytes(out)


def nal_units(data, lp4=False):
    """(framing_bytes, framing_start, nal) for every NAL unit of a packet.

    The 0x01 that closes a start code is the third byte of the 00 00 01 triple,
    so the NAL body begins three bytes after the triple whether the code was
    three or four bytes long; reading it as code-length-plus-three is how an
    Annex-B stream gets misparsed into the wrong NAL types.
    """
    if lp4:
        units, cursor = [], 0
        while cursor + 4 <= len(data):
            length = int.from_bytes(data[cursor:cursor + 4], "big")
            body = cursor + 4
            units.append((4, cursor, data[body:body + length]))
            cursor = body + length
        return units
    codes = [m.start() for m in re.finditer(rb"\x00\x00\x01", data)]
    units = []
    for index, position in enumerate(codes):
        width = 4 if position >= 1 and data[position - 1] == 0 else 3
        body = position + 3
        end = codes[index + 1] if index + 1 < len(codes) else len(data)
        while end > body and data[end - 1] == 0:
            end -= 1               # trailing_zero_8bits are not part of the NAL
        units.append((width, body - width, data[body:end]))
    return units


def nal_type(nal):
    return (nal[0] >> 1) & 0x3F


def is_vcl(nal):
    return nal_type(nal) <= 31


def read_sei(nal):
    """payloadType -> payload bytes for every message of one prefix SEI NAL."""
    body = unescape(nal[2:])
    messages, cursor = {}, 0
    while cursor < len(body):
        payload_type = 0
        while cursor < len(body) and body[cursor] == 0xFF:
            payload_type += 255
            cursor += 1
        if cursor >= len(body):
            break
        payload_type += body[cursor]
        cursor += 1
        payload_size = 0
        while cursor < len(body) and body[cursor] == 0xFF:
            payload_size += 255
            cursor += 1
        if cursor >= len(body):
            break
        payload_size += body[cursor]
        cursor += 1
        messages[payload_type] = body[cursor:cursor + payload_size]
        cursor += payload_size
    return messages


def fnv(data):
    value = FNV_OFFSET
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & MASK64
    return value


def access_units(data):
    """Split an Annex-B elementary stream into access units.

    The rule is the one this stream satisfies: a NAL that follows a VCL NAL
    begins a new AU. The caller asserts that each piece holds exactly one VCL
    NAL, so a wrong split is caught rather than trusted.
    """
    units = nal_units(data)
    chunks, start = [], 0
    for index in range(1, len(units)):
        if is_vcl(units[index - 1][2]):
            chunks.append(data[units[start][1]:units[index][1]])
            start = index
    chunks.append(data[units[start][1]:])
    return chunks


def probe_frames_with_side_data(path):
    """ffprobe's own decode of a file, with every side-data block it printed.

    The two -show_entries groups are load-bearing: naming side_data_list as its
    own group is what makes ffprobe print the values instead of empty objects,
    and a filtered frame list comes back under packets_and_frames, so both keys
    are read here rather than guessed at.
    """
    command = [str(FFMPEG_BIN / "ffprobe.exe"), "-v", "error",
               "-select_streams", "v:0", "-show_entries",
               "frame=pict_type", "-show_entries", "side_data_list",
               "-of", "json", os.fsdecode(path)]
    raw = subprocess.run(command, capture_output=True, text=True, timeout=180)
    data = json.loads(raw.stdout or "{}")
    entries = data.get("frames") or data.get("packets_and_frames") or []
    frames = [item for item in entries if item.get("type", "frame") == "frame"]
    blocks = [block for frame in frames for block in frame.get("side_data_list", [])]
    return raw.returncode, frames, blocks, command


# --------------------------------------------------------------------- #
# Metadata fixtures. These are the values the project's own HDR10 fixture
# declares, as its decoder reported them (bench_out/hdr_p010_nvenc,
# case_a.source_metadata), so the byte expectation is checked against a real
# declaration rather than a convenient one.
# --------------------------------------------------------------------- #

def metadata(**over):
    md = HdrMetadata()
    md.struct_size = C.sizeof(HdrMetadata)
    md.has_mastering = 1
    md.mastering_has_primaries = 1
    md.mastering_has_luminance = 1
    md.primary_x_num = (34000, 13250, 26500)
    md.primary_x_den = (50000, 50000, 50000)
    md.primary_y_num = (16000, 34500, 10000)
    md.primary_y_den = (50000, 50000, 50000)
    md.white_x_num, md.white_x_den = 15635, 50000
    md.white_y_num, md.white_y_den = 16450, 50000
    md.max_luminance_num, md.max_luminance_den = 10000000, 10000
    md.min_luminance_num, md.min_luminance_den = 1000, 10000
    md.has_content_light = 1
    md.max_cll, md.max_fall = 1000, 400
    for name, value in over.items():
        if isinstance(value, (list, tuple)):
            getattr(md, name)[:] = value
        else:
            setattr(md, name, value)
    return md


def bind():
    lib = C.CDLL(str(NATIVE / "nvenc_bridge.dll"))
    lib.nvenc_last_error.restype = CP
    lib.nvenc_hdr_sei_abi_version.restype = U32
    lib.nvenc_hdr_sei_capability.argtypes = [C.POINTER(SeiCapability)]
    lib.nvenc_hdr_sei_capability.restype = I32
    lib.nvenc_hdr_sei_encode_nal.argtypes = [C.POINTER(HdrMetadata),
                                             C.POINTER(C.c_uint8), U32,
                                             C.POINTER(U32), C.POINTER(SeiInfo)]
    lib.nvenc_hdr_sei_encode_nal.restype = I32
    lib.nvenc_hdr_sei_apply_packet.argtypes = [C.POINTER(HdrMetadata),
                                               C.POINTER(C.c_uint8), U32,
                                               C.POINTER(C.c_uint8), U32,
                                               C.POINTER(U32), C.POINTER(SeiInfo)]
    lib.nvenc_hdr_sei_apply_packet.restype = I32
    return lib


def encode_nal(lib, md, capacity=256):
    out = (C.c_uint8 * capacity)()
    size = U32(0)
    info = SeiInfo()
    info.struct_size = C.sizeof(SeiInfo)
    rc = lib.nvenc_hdr_sei_encode_nal(C.byref(md), out, capacity, C.byref(size),
                                      C.byref(info))
    return rc, bytes(out[:size.value]), size.value, info


def apply_packet(lib, md, packet, capacity=None):
    capacity = capacity or max(len(packet) * 2 + 512, 4096)
    out = (C.c_uint8 * capacity)()
    size = U32(0)
    info = SeiInfo()
    info.struct_size = C.sizeof(SeiInfo)
    src = (C.c_uint8 * len(packet)).from_buffer_copy(packet) if packet else None
    rc = lib.nvenc_hdr_sei_apply_packet(C.byref(md), src, len(packet), out,
                                        capacity, C.byref(size), C.byref(info))
    return rc, bytes(out[:size.value]), info


def synthetic_annexb():
    """One access unit in the shape the encoder really produces: its own
    payloadType 1 prefix SEI, the three parameter sets, then one slice."""
    def framed(nal):
        return b"\x00\x00\x00\x01" + nal
    sei = b"\x4e\x01" + b"\x01\x03\x00\x00\x0a" + b"\x80"
    vps = b"\x40\x01\x0c\x01\xff\xff\x02\x20"
    sps = b"\x42\x01\x01\x02\x20\x00\x00\x03"
    pps = b"\x44\x01\xc1\x91\x4f\x03\x32\x44"
    slice_ = (b"\x26\x01\xaf\xe0\x56\xf7" +
              b"\x00\x00\x03\x00\x00\x03\x00\x00\x03\x00\x00\x03\x00\x55")
    return framed(sei) + framed(vps) + framed(sps) + framed(pps) + framed(slice_)


def synthetic_lp4():
    nals = [b"\x4e\x01\x01\x03\x00\x00\x0a\x80",
            b"\x40\x01\x0c\x01\xff\xff\x02\x20",
            b"\x26\x01\xaf\xe0\x56\xf7" +
            b"\x00\x00\x03\x00\x00\x03\x00\x00\x03\x00\x00\x03\x00\x55"]
    return b"".join(int.to_bytes(len(nal), 4, "big") + nal for nal in nals)


def read_back_messages(data):
    """Every SEI message of an Annex-B stream, merged across NAL units."""
    found = {}
    for _, _, nal in nal_units(data):
        if nal_type(nal) == NAL_TYPE_PREFIX_SEI:
            found.update(read_sei(nal))
    return found


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    os.add_dll_directory(str(FFMPEG_BIN))
    checks = Checks()
    results = {"environment": {"os": f"{platform.system()} {platform.release()}",
                               "python": sys.version.split()[0]},
               "purpose": ("CPU-only serializer and framing gate for the HDR10 "
                           "in-bitstream writer: expectations computed here from "
                           "ITU-T H.265 Annex D and Annex B, compared against the "
                           "implemented entries, then against real NVENC bytes")}
    lib = bind()
    status = "ABORTED"
    try:
        abi = lib.nvenc_hdr_sei_abi_version()
        checks.add("writer ABI is 1", abi == SEI_ABI, abi)

        cap = SeiCapability()
        cap.struct_size = C.sizeof(SeiCapability)
        checks.add("capability entry succeeds",
                   lib.nvenc_hdr_sei_capability(C.byref(cap)) == 0,
                   note(lib.nvenc_last_error()))
        values = fields(cap)
        results["capability"] = values
        checks.add("capability separates this writer from the encoder's options",
                   cap.writer_available == 1 and
                   cap.encoder_offers_metadata_option == 0, values)
        checks.add("capability states the standard's own numbers",
                   (cap.payload_type_mastering, cap.payload_type_content_light,
                    cap.prefix_sei_nal_type, cap.mastering_payload_size,
                    cap.content_light_payload_size, cap.chroma_denominator,
                    cap.luma_denominator, cap.primary_order) ==
                   (137, 144, 39, 24, 4, 50000, 10000, 1), values)

        # ---- the full declaration -------------------------------------- #
        md = metadata()
        rc, nal, size, info = encode_nal(lib, md)
        want = expected_nal(md)
        want_ints = declared_ints(md)
        results["full_declaration"] = {
            "rc": rc, "nal_bytes": size, "nal_hex": nal.hex(),
            "expected_hex": (want or b"").hex(), "info": fields(info),
            "wire_integers_expected": want_ints,
            "declared_rationals": {
                "primary_x": list(md.primary_x_num), "primary_x_den":
                list(md.primary_x_den), "primary_y": list(md.primary_y_num),
                "white": [md.white_x_num, md.white_y_num],
                "max_luminance": [md.max_luminance_num, md.max_luminance_den],
                "min_luminance": [md.min_luminance_num, md.min_luminance_den],
                "max_cll": md.max_cll, "max_fall": md.max_fall}}
        checks.add("serialising the fixture's declaration succeeds",
                   rc == 0 and info.error == OK, fields(info))
        checks.add("the NAL the writer built is the byte sequence the standard "
                   "builds", nal == want,
                   {"got": nal.hex(), "expected": (want or b"").hex()})
        checks.add("the NAL header is nal_unit_type 39, nuh_layer_id 0, "
                   "nuh_temporal_id_plus1 1",
                   len(nal) >= 2 and nal_type(nal) == 39 and
                   ((nal[0] & 1) << 5 | (nal[1] >> 3)) == 0 and (nal[1] & 7) == 1,
                   nal[:2].hex())
        messages = read_sei(nal) if nal else {}
        checks.add("the payload types and sizes on the wire are 137/24 and 144/4",
                   sorted(messages) == [137, 144] and len(messages[137]) == 24 and
                   len(messages[144]) == 4,
                   {str(key): len(value) for key, value in messages.items()})

        # Primary order, proved value by value against the declaration.
        wire_u16 = [int.from_bytes(messages[137][i * 2:i * 2 + 2], "big")
                    for i in range(8)]
        checks.add("the primaries reach the wire in green, blue, red order",
                   wire_u16 == want_ints[:8],
                   {"wire": wire_u16, "green_blue_red_expected": want_ints[:8],
                    "struct_order_red_first_x": [
                        grid(md.primary_x_num[i], md.primary_x_den[i], CHROMA_DEN,
                             0, 0xFFFFFFFF) for i in range(3)]})
        got_max = int.from_bytes(messages[137][16:20], "big")
        got_min = int.from_bytes(messages[137][20:24], "big")
        checks.add("max luminance precedes min luminance and both are u(32)",
                   (got_max, got_min) == (want_ints[8], want_ints[9]) and
                   got_max > got_min,
                   {"max_hex": messages[137][16:20].hex(),
                    "min_hex": messages[137][20:24].hex(), "max": got_max,
                    "min": got_min})
        checks.add("MaxCLL and MaxFALL are the declared 16-bit integers",
                   (int.from_bytes(messages[144][0:2], "big"),
                    int.from_bytes(messages[144][2:4], "big")) == (1000, 400),
                   messages[144].hex())

        # Values, not spellings: the wire grid back to exact Fractions.
        read_values = {
            "green_x": Fraction(wire_u16[0], CHROMA_DEN),
            "green_y": Fraction(wire_u16[1], CHROMA_DEN),
            "blue_x": Fraction(wire_u16[2], CHROMA_DEN),
            "blue_y": Fraction(wire_u16[3], CHROMA_DEN),
            "red_x": Fraction(wire_u16[4], CHROMA_DEN),
            "red_y": Fraction(wire_u16[5], CHROMA_DEN),
            "white_x": Fraction(wire_u16[6], CHROMA_DEN),
            "white_y": Fraction(wire_u16[7], CHROMA_DEN),
            "max_luminance": Fraction(got_max, LUMA_DEN),
            "min_luminance": Fraction(got_min, LUMA_DEN),
            "max_cll": Fraction(int.from_bytes(messages[144][0:2], "big")),
            "max_fall": Fraction(int.from_bytes(messages[144][2:4], "big"))}
        declared_values = {
            "green_x": Fraction(13250, 50000), "green_y": Fraction(34500, 50000),
            "blue_x": Fraction(26500, 50000), "blue_y": Fraction(10000, 50000),
            "red_x": Fraction(34000, 50000), "red_y": Fraction(16000, 50000),
            "white_x": Fraction(15635, 50000), "white_y": Fraction(16450, 50000),
            "max_luminance": Fraction(10000000, 10000),
            "min_luminance": Fraction(1000, 10000),
            "max_cll": Fraction(1000), "max_fall": Fraction(400)}
        checks.add("every authored value equals the declared value exactly",
                   read_values == declared_values,
                   {"read": {k: str(v) for k, v in read_values.items()},
                    "declared": {k: str(v) for k, v in declared_values.items()}})
        results["full_declaration"]["values_read_from_wire"] = {
            k: str(v) for k, v in read_values.items()}

        # ---- emulation prevention really happens ----------------------- #
        rbsp = expected_rbsp(md)
        escaped_rbsp = escape(rbsp)
        checks.add("the fixture's own payload needs an escape byte and the writer "
                   "inserts one",
                   len(escaped_rbsp) > len(rbsp) and nal[2:] == escaped_rbsp,
                   {"rbsp": rbsp.hex(), "escaped": escaped_rbsp.hex(),
                    "inserted": len(escaped_rbsp) - len(rbsp)})
        checks.add("unescaping the writer's NAL returns the payload unchanged",
                   unescape(nal[2:]) == rbsp, nal[2:].hex())
        # 00 00 03 is the escape sequence itself, so it is legal here; only the
        # three values that would read as a start code may not appear.
        checks.add("no start-code-shaped sequence survives in the NAL body",
                   all(b"\x00\x00" + bytes([last]) not in nal[2:]
                       for last in range(3)), nal[2:].hex())

        # Container Float representations must serialize to nearest wire units.
        # Pin known integers independently of the reference implementation.
        float_md = metadata(
            primary_x_num=(11374617, 2055209, 9259681),
            primary_x_den=(16777216, 8388608, 67108864),
            primary_y_num=(675241, 11794383, 3489661),
            primary_y_den=(2097152, 16777216, 67108864),
            white_x_num=1311475, white_x_den=4194304,
            white_y_num=11038737, white_y_den=33554432,
            max_luminance_num=1000, max_luminance_den=1,
            min_luminance_num=0, min_luminance_den=1)
        quantization = []
        for label, case, expected_units in (
                ("Bulgaria Matroska float metadata", float_md,
                 [12250, 35150, 6899, 2600, 33899, 16099, 15634, 16449,
                  10000000, 0]),
                ("half-unit tie rounds up", metadata(white_x_num=31271, white_x_den=100000),
                 [13250, 34500, 26500, 10000, 34000, 16000, 15636, 16450,
                  10000000, 1000]),
                ("below half-unit rounds down", metadata(white_x_num=312709, white_x_den=1000000),
                 [13250, 34500, 26500, 10000, 34000, 16000, 15635, 16450,
                  10000000, 1000])):
            rc, case_nal, _, info = encode_nal(lib, case)
            payload = read_sei(case_nal).get(137, b"") if rc == 0 else b""
            expected_payload = b"".join(v.to_bytes(2, "big") for v in expected_units[:8]) + \
                b"".join(v.to_bytes(4, "big") for v in expected_units[8:])
            checks.add(label, rc == 0 and info.error == OK and payload == expected_payload,
                       {"rc": rc, "payload": payload.hex(), "expected": expected_payload.hex()})
            quantization.append({"case": label, "payload": payload.hex(),
                                 "expected_units": expected_units})
        results["container_quantization"] = quantization

        # ---- refusals -------------------------------------------------- #
        refusal_cases = [
            ("zero denominator", metadata(white_x_den=0), BAD_RATIONAL),
            ("negative numerator", metadata(white_y_num=-1), BAD_RATIONAL),
            ("non-grid value outside range",
             metadata(primary_x_den=(3, 50000, 50000)), OUT_OF_RANGE),
            ("an exact grid value below the accepted primary range",
             metadata(primary_x_num=(1, 13250, 26500)), OUT_OF_RANGE),
            ("primary above the accepted x range",
             metadata(primary_x_num=(40000, 13250, 26500)), OUT_OF_RANGE),
            ("just below range must not round into it",
             metadata(white_x_num=99, white_x_den=1000000), OUT_OF_RANGE),
            ("just above range must not round into it",
             metadata(white_x_num=740001, white_x_den=1000000), OUT_OF_RANGE),
            ("min luminance equal to max", metadata(min_luminance_num=10000000),
             OUT_OF_RANGE),
            ("max luminance beyond u(32) on the grid",
             metadata(max_luminance_num=600000000), OUT_OF_RANGE),
            ("MaxCLL above 16 bits", metadata(max_cll=70000), OUT_OF_RANGE),
        ]
        serialisation = []
        for label, case, error in refusal_cases:
            rc, nal_case, size, info = encode_nal(lib, case)
            record = {"case": label, "rc": rc, "error": info.error,
                      "nal_bytes": size, "detail": note(info.detail),
                      "python_refuses_the_same_way":
                          expected_rbsp(case) is None}
            serialisation.append(record)
            checks.add(f"refusal: {label} refused with error {error}",
                       rc == -1 and info.error == error and size == 0 and
                       record["python_refuses_the_same_way"], record)
        # A declaration the payload cannot express half-way is not padded with
        # invented numbers: the mastering message is simply absent.
        for label, case in (("primaries without luminance",
                             metadata(mastering_has_luminance=0)),
                            ("luminance without primaries",
                             metadata(mastering_has_primaries=0)),
                            ("mastering half-declared and no content light",
                             metadata(mastering_has_luminance=0,
                                      has_content_light=0))):
            rc, nal_case, size, info = encode_nal(lib, case)
            types = sorted(read_sei(nal_case)) if size else []
            record = {"case": label, "rc": rc, "error": info.error,
                      "nal_bytes": size, "payload_types": types,
                      "detail": note(info.detail)}
            serialisation.append(record)
            checks.add(f"no invented fields: {label} authors no mastering "
                       "payload", rc == 0 and 137 not in types, record)
        rc, nal_none, size, info = encode_nal(lib, metadata(has_mastering=0,
                                                            has_content_light=0))
        checks.add("a source that declared nothing gains nothing: no NAL is built",
                   rc == 0 and size == 0 and info.error == NOTHING_TO_DO,
                   {"rc": rc, "bytes": size, "error": info.error,
                    "detail": note(info.detail)})
        rc, nal_partial, size, info = encode_nal(lib, metadata(
            has_mastering=0, has_content_light=0, mastering_has_primaries=1))
        checks.add("has_mastering off means no mastering however the sub-flags "
                   "are set", rc == 0 and size == 0,
                   {"rc": rc, "bytes": size, "detail": note(info.detail)})
        results["serialisation_cases"] = serialisation

        # Content light declared on its own is still authored.
        light_only = metadata(mastering_has_primaries=0,
                              mastering_has_luminance=0)
        rc, nal_light, size, info = encode_nal(lib, light_only)
        checks.add("content light alone is authored as its own payload, with no "
                   "mastering invented beside it",
                   rc == 0 and sorted(read_sei(nal_light)) == [144] and
                   read_sei(nal_light)[144] == b"\x03\xe8\x01\x90",
                   {"rc": rc, "bytes": size, "hex": nal_light.hex()})

        # ---- Annex-B packet framing ------------------------------------ #
        annexb = synthetic_annexb()
        rc, out, info = apply_packet(lib, md, annexb)
        before = nal_units(annexb)
        after = nal_units(out)
        results["packet_annexb"] = {
            "rc": rc, "in_bytes": len(annexb), "out_bytes": len(out),
            "framing": info.framing, "info": fields(info),
            "types_before": [nal_type(nal) for _, _, nal in before],
            "types_after": [nal_type(nal) for _, _, nal in after],
            "out_hex_head": out[:96].hex()}
        checks.add("Annex-B packet handled", rc == 0 and info.error == OK,
                   fields(info))
        checks.add("the framing the writer reports is the framing the packet has",
                   info.framing == FRAMING_ANNEXB4, info.framing)
        checks.add("exactly one NAL unit was added to the packet",
                   len(after) == len(before) + 1,
                   {"before": len(before), "after": len(after)})
        vcl_before = [nal for _, _, nal in before if is_vcl(nal)]
        vcl_after = [nal for _, _, nal in after if is_vcl(nal)]
        checks.add("every original VCL NAL byte is unchanged after insertion",
                   vcl_before == vcl_after and
                   fnv(b"".join(vcl_before)) == fnv(b"".join(vcl_after)),
                   {"sha_before": sha256(b"".join(vcl_before)).hexdigest()[:16],
                    "sha_after": sha256(b"".join(vcl_after)).hexdigest()[:16],
                    "fnv": fnv(b"".join(vcl_before))})
        others_before = [nal for _, _, nal in before]
        others_after = [nal for _, _, nal in after if not read_sei(nal).get(137)]
        checks.add("every pre-existing NAL - parameter sets and the encoder's own "
                   "payloadType 1 SEI - is byte-identical and in the same order",
                   others_before == others_after,
                   {"before": [len(nal) for nal in others_before],
                    "after": [len(nal) for nal in others_after]})
        inserted_at = next(index for index, (_, _, nal) in enumerate(after)
                           if read_sei(nal).get(137))
        first_vcl = next(index for index, (_, _, nal) in enumerate(after)
                         if is_vcl(nal))
        checks.add("the prefix SEI sits after the parameter sets and before the "
                   "first VCL NAL", 0 < inserted_at < first_vcl and
                   info.vcl_index == first_vcl - 1,
                   {"inserted": inserted_at, "first_vcl": first_vcl,
                    "reported_vcl_index": info.vcl_index,
                    "types": [nal_type(nal) for _, _, nal in after]})
        messages_after = read_back_messages(out)
        checks.add("the packet now carries 137 and 144 with the declared values",
                   messages_after.get(137) == expected_mastering_payload(md) and
                   messages_after.get(144) == expected_light_payload(md),
                   {"types": sorted(messages_after)})
        checks.add("the bytes the writer inserted are the NAL plus one start code",
                   len(out) - len(annexb) == info.bytes_inserted ==
                   info.nal_bytes + 4,
                   {"delta": len(out) - len(annexb), "info": info.bytes_inserted,
                    "nal_bytes": info.nal_bytes})

        # Applying the same declaration again must not double it.
        rc, out2, info2 = apply_packet(lib, md, out)
        checks.add("re-applying an equal declaration inserts nothing",
                   rc == 0 and out2 == out and info2.mastering_present == 1 and
                   info2.mastering_equal == 1 and info2.content_light_present == 1
                   and info2.content_light_equal == 1 and
                   info2.bytes_inserted == 0,
                   {"rc": rc, "same_bytes": out2 == out, "info": fields(info2)})

        # A contradicting declaration is refused, not merged.
        rc, out3, info3 = apply_packet(lib, metadata(max_cll=4000), out)
        checks.add("a conflicting content-light declaration already in the AU is "
                   "refused, never overwritten or duplicated",
                   rc == -1 and info3.error == CONFLICT and out3 == b"",
                   {"rc": rc, "error": info3.error, "detail": note(info3.detail)})
        rc, out4, info4 = apply_packet(lib, metadata(primary_x_num=(13250, 26500,
                                                                   34000)), out)
        checks.add("a conflicting mastering declaration already in the AU is "
                   "refused", rc == -1 and info4.error == CONFLICT,
                   {"rc": rc, "error": info4.error,
                    "detail": note(info4.detail)})

        # Length-prefixed framing keeps its own framing.
        lp4 = synthetic_lp4()
        rc, out_lp4, info_lp4 = apply_packet(lib, md, lp4)
        units_lp4 = nal_units(out_lp4, lp4=True)
        checks.add("a length-prefixed packet is rewritten as a length-prefixed "
                   "packet", rc == 0 and info_lp4.framing == FRAMING_LP4 and
                   len(units_lp4) == len(nal_units(lp4, lp4=True)) + 1,
                   {"rc": rc, "framing": info_lp4.framing,
                    "nals": len(units_lp4), "detail": note(info_lp4.detail)})
        checks.add("the length-prefixed rewrite carries the authored SEI and "
                   "leaves the slice bytes alone",
                   any(read_sei(nal).get(137) == expected_mastering_payload(md)
                       for _, _, nal in units_lp4) and
                   [nal for _, _, nal in nal_units(lp4, True) if is_vcl(nal)] ==
                   [nal for _, _, nal in units_lp4 if is_vcl(nal)],
                   {"types": [nal_type(nal) for _, _, nal in units_lp4]})

        # A packet with no slice cannot be an access unit, so it is refused.
        no_vcl = b"\x00\x00\x00\x01" + b"\x40\x01\x0c\x01\xff\xff\x02\x20"
        rc, _, info_nv = apply_packet(lib, md, no_vcl)
        checks.add("a packet with no VCL NAL is refused rather than guessed at",
                   rc == -1 and info_nv.error == NO_VCL,
                   {"rc": rc, "error": info_nv.error,
                    "detail": note(info_nv.detail)})

        # Garbage is refused as an unknown framing, not passed through.
        rc, _, info_g = apply_packet(lib, md, bytes(range(64)))
        checks.add("a packet that is neither Annex-B nor length-prefixed is "
                   "refused", rc == -1 and info_g.error == FRAMING_UNSUPPORTED,
                   {"rc": rc, "error": info_g.error,
                    "detail": note(info_g.detail)})

        rc, _, info_z = apply_packet(lib, md, b"")
        checks.add("an empty packet is refused as malformed", rc == -1 and
                   info_z.error == MALFORMED,
                   {"rc": rc, "error": info_z.error,
                    "detail": note(info_z.detail)})

        rc, _, info_c = apply_packet(lib, md, annexb,
                                     capacity=len(annexb) + 1)
        checks.add("an output buffer that cannot hold the result is refused",
                   rc == -1 and info_c.error == CAPACITY,
                   {"rc": rc, "error": info_c.error,
                    "detail": note(info_c.detail)})

        # ---- the real encoder bytes ------------------------------------ #
        baseline = BASELINE.read_bytes()
        units_real = nal_units(baseline)
        types_real = [nal_type(nal) for _, _, nal in units_real]
        results["baseline"] = {"path": str(BASELINE), "bytes": len(baseline),
                               "sha256": sha256(baseline).hexdigest(),
                               "nal_types": types_real}
        checks.add("the captured NVENC stream parses as Annex-B with two VCL NALs",
                   len(units_real) == 7 and
                   sum(1 for _, _, nal in units_real if is_vcl(nal)) == 2,
                   types_real)
        aus = access_units(baseline)
        checks.add("the captured stream splits into the two access units it holds",
                   len(aus) == 2 and all(
                       sum(1 for _, _, nal in nal_units(unit) if is_vcl(nal)) == 1
                       for unit in aus),
                   [len(unit) for unit in aus])
        written = b""
        per_unit = []
        for unit in aus:
            rc, out_unit, info_unit = apply_packet(lib, md, unit)
            per_unit.append({"in_bytes": len(unit), "out_bytes": len(out_unit),
                             "rc": rc, "framing": info_unit.framing,
                             "nal_bytes": info_unit.nal_bytes,
                             "bytes_inserted": info_unit.bytes_inserted,
                             "nal_count": info_unit.nal_count,
                             "vcl_index": info_unit.vcl_index})
            if rc != 0:
                break
            written += out_unit
        vcl_in = [nal for _, _, nal in units_real if is_vcl(nal)]
        vcl_out = [nal for _, _, nal in nal_units(written) if is_vcl(nal)] if written else []
        other_in = [nal for _, _, nal in units_real
                    if not read_sei(nal).get(137)]
        other_out = [nal for _, _, nal in nal_units(written)
                     if not read_sei(nal).get(137)] if written else []
        results["real_packets"] = {
            "per_access_unit": per_unit, "in_bytes": len(baseline),
            "out_bytes": len(written),
            "out_sha256": sha256(written).hexdigest() if written else "",
            "vcl_hash_in": fnv(b"".join(vcl_in)),
            "vcl_hash_out": fnv(b"".join(vcl_out)) if written else 0,
            "vcl_bytes": sum(len(nal) for nal in vcl_in),
            "inserted_total": sum(item["bytes_inserted"] for item in per_unit)}
        checks.add("every real access unit was rewritten",
                   all(item["rc"] == 0 for item in per_unit) and bool(written),
                   per_unit)
        checks.add("the coded slices of the real encoder output are byte-identical "
                   "after authoring", vcl_in == vcl_out,
                   {"lengths_before": [len(n) for n in vcl_in],
                    "lengths_after": [len(n) for n in vcl_out],
                    "fnv_before": fnv(b"".join(vcl_in)),
                    "fnv_after": fnv(b"".join(vcl_out)) if written else 0})
        checks.add("the encoder's own parameter sets and payloadType 1 SEI are "
                   "byte-identical after authoring", other_in == other_out,
                   {"before": [len(n) for n in other_in],
                    "after": [len(n) for n in other_out]})
        after_types = [nal_type(nal) for _, _, nal in nal_units(written)]
        checks.add("one mastering and one content-light payload per access unit "
                   "are now in the elementary stream",
                   after_types.count(NAL_TYPE_PREFIX_SEI) == 4 and
                   sum(1 for _, _, nal in nal_units(written)
                       if read_sei(nal).get(137) ==
                       expected_mastering_payload(md)) == 2 and
                   sum(1 for _, _, nal in nal_units(written)
                       if read_sei(nal).get(144) ==
                       expected_light_payload(md)) == 2,
                   {"nal_types": after_types})
        target = ARTIFACTS / "sei_cpu_authored.hevc"
        target.write_bytes(written)
        checks.add("the authored elementary stream was written for a decoder to "
                   "judge", target.stat().st_size == len(written), str(target))

        # The same bytes, untouched, must still read as no metadata at all -
        # the absence this gate is about is the baseline's, not a fixture.
        control_rc, control_frames, control_blocks, control_command = \
            probe_frames_with_side_data(BASELINE)
        results["baseline_probe"] = {
            "rc": control_rc, "command": [str(item) for item in control_command],
            "frames": len(control_frames),
            "kinds": sorted({item.get("side_data_type") for item in control_blocks})}
        checks.add("the captured NVENC stream really did carry no SEI metadata "
                   "before this writer ran",
                   control_rc == 0 and len(control_frames) == 2 and
                   not [item for item in control_blocks
                        if item.get("side_data_type", "")
                        .startswith(("Mastering", "Content light"))],
                   results["baseline_probe"])

        # An independent decoder, with no container to help it.
        probe_rc, frames, blocks, command = probe_frames_with_side_data(target)
        (ARTIFACTS / "ffprobe_sei_cpu_authored.json").write_text(
            json.dumps({"command": [str(item) for item in command],
                        "rc": probe_rc, "frames": frames}, indent=2),
            encoding="utf-8")
        mastering = [item for item in blocks
                     if item.get("side_data_type") == "Mastering display metadata"]
        light = [item for item in blocks
                 if item.get("side_data_type") == "Content light level metadata"]
        def fractional(block):
            out = {}
            for key in ("red_x", "red_y", "green_x", "green_y", "blue_x",
                        "blue_y", "white_point_x", "white_point_y",
                        "max_luminance", "min_luminance"):
                value = block.get(key)
                if value is not None:
                    num, den = str(value).split("/")
                    out[key] = Fraction(int(num), int(den))
            return out
        read_fractional = fractional(mastering[0]) if mastering else {}
        expected_read = {
            "red_x": Fraction(34000, 50000), "red_y": Fraction(16000, 50000),
            "green_x": Fraction(13250, 50000), "green_y": Fraction(34500, 50000),
            "blue_x": Fraction(26500, 50000), "blue_y": Fraction(10000, 50000),
            "white_point_x": Fraction(15635, 50000),
            "white_point_y": Fraction(16450, 50000),
            "max_luminance": Fraction(10000000, 10000),
            "min_luminance": Fraction(1000, 10000)}
        results["decoder_read_back"] = {
            "frames": len(frames), "mastering_blocks": len(mastering),
            "content_light_blocks": len(light),
            "read": {key: str(value) for key, value in read_fractional.items()},
            "expected": {key: str(value) for key, value in expected_read.items()},
            "content_light": light[:1], "ffprobe_rc": probe_rc}
        checks.add("the installed decoder reads the authored SEI from the "
                   "standalone stream",
                   probe_rc == 0 and len(mastering) == 2 and len(light) == 2,
                   {"rc": probe_rc, "mastering": len(mastering),
                    "light": len(light), "frames": len(frames)})
        checks.add("every decoded frame carries both authored payloads, so each "
                   "access unit stands on its own",
                   all(any(item.get("side_data_type") == "Mastering display metadata"
                           for item in frame.get("side_data_list", [])) and
                       any(item.get("side_data_type")
                           == "Content light level metadata"
                           for item in frame.get("side_data_list", []))
                       for frame in frames) and len(frames) == 2,
                   [[item.get("side_data_type")
                     for item in frame.get("side_data_list", [])]
                    for frame in frames])
        checks.add("the decoded mastering values equal the declared ones, as exact "
                   "values", read_fractional == expected_read and
                   all(fractional(item) == expected_read for item in mastering),
                   results["decoder_read_back"])
        checks.add("the decoded MaxCLL and MaxFALL equal the declared ones",
                   bool(light) and light[0].get("max_content") == 1000 and
                   light[0].get("max_average") == 400, light[:1])

        status = "PASS" if not checks.failed else "FAIL"
        return 0 if status == "PASS" else 1
    finally:
        results["status"] = status
        results["checks"] = checks.items
        results["checks_total"] = len(checks.items)
        results["checks_failed"] = checks.failed
        results["writer_sources"] = {
            "header": str(NATIVE / "nvenc_bridge.h"),
            "implementation": str(NATIVE / "nvenc_bridge.cpp"),
            "dll_sha256": sha256((NATIVE / "nvenc_bridge.dll").read_bytes()).hexdigest()}
        (ARTIFACTS / "sei_cpu_gate.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str),
            encoding="utf-8")
        print(json.dumps({"status": status, "failed": checks.failed,
                          "checks_total": len(checks.items)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
