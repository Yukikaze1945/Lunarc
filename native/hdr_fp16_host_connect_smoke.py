"""Three-frame HDR FP16 device chain: the current HDR bridge, a production
Feature 18 session opened with R16G16B16A16_FLOAT colour, and real NVOF motion
estimated from the bridge's own RGBA8 guidance surface.

What the run is meant to establish, and nothing more:

  * the FP16 proxy plane leaves the HDR bridge as device memory, is transported
    by the CUDA/D3D12 row bridge as a width * 8 byte row, and arrives at Feature
    18 without one byte of host staging (``host_color_stage_count == 0``);
  * the Feature 18 result stays a D3D12 texture and is handed to the bridge's
    reconstruction by pointer and fence, so nothing is read back to the host on
    the processing path (``output_readback_count == 0`` at the snapshot);
  * the 8-bit surface in the chain is the guidance plane, which only NVOF reads
    and which never carries the picture's colour volume;
  * the transport is asked for the *whole* row, so the tail half of every FP16
    row moves with it - proved by the labelled row-bridge diagnostic below,
    which is a transport test and not a claim about what the network thought.

Pixel readbacks happen only after the counters have been snapshotted, and are
reported separately from the chain that produced them.

    python native/hdr_fp16_host_connect_smoke.py bench_out/hdr_fp16_host_connect/chain.json
"""
import ctypes as C
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NATIVE = ROOT / "native"
sys.path.insert(0, str(ROOT))

import dlss_gpu_engine as engine
import pipeline

FULL_W, FULL_H = 512, 512          # the job's master extent
MASTER_SCALE = 0.5                 # the production rule's working scale
# Feature 18 and NVOF both run here, and here is where the plan has to land:
# resolve_working_dims is the same function the export job sizes its working
# plane with, so a 50% job of a 512x512 master is 256x256 by rule, not by a
# number this test picked.
WORK_W, WORK_H = pipeline.resolve_working_dims(FULL_W, FULL_H, MASTER_SCALE)
FRAMES = 3
NVOF_GRID, NVOF_PERF = 4, 10       # the configuration the SDR baseline proves
CEILING_SCRGB = 12.5               # 1000 cd/m2 at this bridge's 1.0 == 80 cd/m2
RESIDUAL_STRENGTH = 1.0
SCRGB_PER_RELATIVE = 125.0         # 10000 cd/m2 / 80 cd/m2
LUMA_BLACK, LUMA_SPAN = 64, 876    # limited-range 10-bit luma
CHROMA_NEUTRAL, CHROMA_SPAN = 512, 896
# P010 code 680 is the brightest luma this fixture writes. With the chroma
# deviation below it, no channel of the decoded master reaches the declared
# ceiling, so above_ceiling_components is a measurement of a clean source and
# not a number the fixture pushed the bridge into.
LUMA_CODE_TOP = 680
CHROMA_DEVIATION = 4

# The same baked options the SDR baseline runs, so the two differ in colour
# format alone.
SETTINGS = {"preset": 0, "style": 0, "intensity": 1.0, "local_tone": 1.0,
            "local_struct": 1.0, "skin_struct": 0.0, "use_auto_mask": 0,
            "ui_correction": 0, "motion_scale_x": 1.0, "motion_scale_y": 1.0}

DXGI_R16G16B16A16_FLOAT = 10
DXGI_R8G8B8A8_UNORM = 28

# Mirror of GPU_HDR_MAX_TAPS in native/gpu_hdr_scrgb_bridge.h: the widest tap row
# the bridge will upload for a plan.
GPU_HDR_MAX_TAPS = 24

# The bridge's own plane numbering, for the diagnostic readbacks.
PLANE_WORKING_MASTER, PLANE_PROXY, PLANE_GUIDANCE, PLANE_COMPOSITED = 3, 4, 5, 7


class TapTable(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("taps", C.c_uint32),
        ("vertical_src_len", C.c_uint32), ("vertical_dst_len", C.c_uint32),
        ("vertical_indices", C.POINTER(C.c_int32)),
        ("vertical_weights", C.POINTER(C.c_float)),
        ("horizontal_src_len", C.c_uint32), ("horizontal_dst_len", C.c_uint32),
        ("horizontal_indices", C.POINTER(C.c_int32)),
        ("horizontal_weights", C.POINTER(C.c_float))]


class Contract(C.Structure):
    _fields_ = [("struct_size", C.c_uint32),
        ("scrgb_per_nit", C.c_double), ("nits_per_scrgb", C.c_double),
        ("to_scrgb", C.c_double * 9), ("to_bt2020", C.c_double * 9),
        ("pq_m1", C.c_double), ("pq_m2", C.c_double), ("pq_c1", C.c_double),
        ("pq_c2", C.c_double), ("pq_c3", C.c_double),
        ("kr", C.c_double), ("kg", C.c_double), ("kb", C.c_double),
        ("proxy_knee", C.c_double), ("proxy_shoulder_span", C.c_double),
        ("proxy_shoulder_k", C.c_double),
        ("proxy_saturation_relative", C.c_double),
        ("proxy_actionable_relative", C.c_double),
        ("max_stops", C.c_double), ("min_slope_per_stop", C.c_double)]

    def as_dict(self):
        return {name: getattr(self, name) for name, _ in self._fields_
                if name != "struct_size"}

    def scalars(self):
        return {name: getattr(self, name) for name, _ in self._fields_
                if name not in ("struct_size", "to_scrgb", "to_bt2020")}


class SourceReport(C.Structure):
    _fields_ = [("struct_size", C.c_uint32),
        ("support_ceiling_scrgb", C.c_double),
        ("paper_white_scrgb", C.c_double), ("nonfinite_samples", C.c_uint32),
        ("negative_components", C.c_uint32),
        ("above_ceiling_components", C.c_uint32),
        ("min_component", C.c_double), ("max_component", C.c_double),
        ("pixels", C.c_uint32)]


class ProxyReport(C.Structure):
    _fields_ = [("struct_size", C.c_uint32),
        ("negative_components", C.c_uint32), ("pinned_components", C.c_uint32),
        ("dead_zone_components", C.c_uint32),
        ("above_actionable_components", C.c_uint32),
        ("above_saturation_components", C.c_uint32),
        ("min_code", C.c_double), ("max_code", C.c_double),
        ("pixels", C.c_uint32)]


class ReconstructReport(C.Structure):
    _fields_ = [("struct_size", C.c_uint32),
        ("applied_components", C.c_uint32),
        ("gain_cap_components", C.c_uint32),
        ("rejected_pinned_components", C.c_uint32),
        ("rejected_dead_components", C.c_uint32),
        ("rejected_negative_components", C.c_uint32),
        ("nonfinite_components", C.c_uint32),
        ("max_abs_stops", C.c_double), ("max_abs_residual_scrgb", C.c_double),
        ("composite_out_of_domain_components", C.c_uint32),
        ("composite_nonfinite_components", C.c_uint32),
        ("composite_min_component", C.c_double),
        ("composite_max_component", C.c_double), ("pixels", C.c_uint32)]


class BridgeCounters(C.Structure):
    _fields_ = [("struct_size", C.c_uint32)] + [
        (name, C.c_uint64) for name in (
            "process_count", "source_device_copy_bytes",
            "source_device_write_bytes", "working_device_write_bytes",
            "guidance_device_write_bytes", "composited_device_write_bytes",
            "encode_device_write_bytes", "report_readback_bytes")]


class ColorLayout(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("width", C.c_uint32),
        ("height", C.c_uint32), ("row_pitch", C.c_uint32),
        ("bytes_per_row", C.c_uint32), ("pixel_bytes", C.c_uint32),
        ("plane_bytes", C.c_uint64)]


def _report(struct, getter, bridge):
    value = struct(C.sizeof(struct))
    if getter(bridge, C.byref(value)):
        raise RuntimeError(f"{getter.__name__}: {bridge_last_error()}")
    return {name: getattr(value, name) for name, _ in struct._fields_
            if name != "struct_size"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(buffer):
    return hashlib.sha256(bytes(buffer)).hexdigest()


cuda = C.WinDLL("nvcuda.dll")
nvof = C.WinDLL(str(NATIVE / "nvof_bridge.dll"))
interop = C.WinDLL(str(NATIVE / "nvof_d3d12_interop.dll"))
hdr = C.WinDLL(str(NATIVE / "gpu_hdr_scrgb_bridge.dll"))


def _prototype(library, name, argtypes, restype):
    entry = getattr(library, name)
    entry.argtypes = argtypes
    entry.restype = restype
    return entry


for _name, _types in (
        ("cuCtxPushCurrent_v2", [C.c_void_p]),
        ("cuCtxPopCurrent_v2", [C.POINTER(C.c_void_p)]),
        ("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t]),
        ("cuMemFree_v2", [C.c_uint64]),
        ("cuMemcpyHtoD_v2", [C.c_uint64, C.c_void_p, C.c_size_t]),
        ("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t])):
    _prototype(cuda, _name, _types, C.c_int)

_prototype(nvof, "nvof_bridge_create", [C.c_uint32, C.c_uint32, C.c_uint32,
    C.c_uint32, C.POINTER(C.c_void_p)], C.c_int32)
_prototype(nvof, "nvof_bridge_execute_rgba_device_sequential", [C.c_void_p,
    C.c_uint64, C.c_uint32, C.c_void_p, C.c_int32, C.c_void_p], C.c_int32)
_prototype(nvof, "nvof_bridge_get_output_deviceptr", [C.c_void_p,
    C.POINTER(C.c_uint64), C.POINTER(C.c_size_t)], C.c_int32)
_prototype(nvof, "nvof_bridge_get_session_info", [C.c_void_p,
    C.POINTER(C.c_uint32), C.POINTER(C.c_void_p), C.POINTER(C.c_uint32)],
    C.c_int32)
_prototype(nvof, "nvof_bridge_destroy", [C.c_void_p], None)
nvof.nvof_bridge_get_last_error.restype = C.c_char_p

_prototype(interop, "cuda_d3d12_color_row_capabilities", [], C.c_uint32)
_prototype(interop, "cuda_d3d12_color_create2", [C.c_void_p, C.c_void_p,
    C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.c_int32,
    C.POINTER(C.c_void_p)], C.c_int32)
_prototype(interop, "cuda_d3d12_color_get_layout", [C.c_void_p,
    C.POINTER(ColorLayout)], C.c_int32)
_prototype(interop, "cuda_d3d12_color_copy", [C.c_void_p, C.c_uint64,
    C.c_size_t, C.POINTER(C.c_uint64)], C.c_int32)
_prototype(interop, "cuda_d3d12_color_get_device_pointer", [C.c_void_p,
    C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)], C.c_int32)
_prototype(interop, "cuda_d3d12_color_destroy", [C.c_void_p], None)
interop.nvof_d3d12_last_error.restype = C.c_char_p

_prototype(hdr, "gpu_hdr_abi_version", [], C.c_uint32)
_prototype(hdr, "gpu_hdr_fill_contract", [C.POINTER(Contract)], None)
_prototype(hdr, "gpu_hdr_open", [C.c_void_p, C.c_void_p, C.c_uint32,
    C.c_uint32, C.c_uint32, C.c_uint32, C.POINTER(TapTable),
    C.POINTER(TapTable), C.c_double, C.POINTER(C.c_void_p)], C.c_int32)
_prototype(hdr, "gpu_hdr_paper_white", [C.c_void_p], C.c_double)
_prototype(hdr, "gpu_hdr_decode", [C.c_void_p, C.c_uint64, C.c_size_t,
    C.c_uint64, C.c_size_t, C.c_int32, C.c_int32, C.c_int32, C.c_int32,
    C.c_double, C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)], C.c_int32)
_prototype(hdr, "gpu_hdr_downscale", [C.c_void_p, C.POINTER(C.c_uint64),
    C.POINTER(C.c_uint32)], C.c_int32)
_prototype(hdr, "gpu_hdr_proxy", [C.c_void_p, C.POINTER(C.c_uint64),
    C.POINTER(C.c_uint32), C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)],
    C.c_int32)
_prototype(hdr, "gpu_hdr_reconstruct", [C.c_void_p, C.c_void_p, C.c_void_p,
    C.c_uint64, C.c_double, C.POINTER(C.c_uint64), C.POINTER(C.c_uint32)],
    C.c_int32)
_prototype(hdr, "gpu_hdr_plane_pitch", [C.c_void_p, C.c_int32], C.c_uint32)
_prototype(hdr, "gpu_hdr_plane_bytes", [C.c_void_p, C.c_int32], C.c_uint64)
_prototype(hdr, "gpu_hdr_readback_scrgb", [C.c_void_p, C.c_int32, C.c_void_p,
    C.c_size_t], C.c_int32)
_prototype(hdr, "gpu_hdr_get_transfer_counters", [C.c_void_p,
    C.POINTER(BridgeCounters)], C.c_int32)
_prototype(hdr, "gpu_hdr_get_source_report", [C.c_void_p,
    C.POINTER(SourceReport)], C.c_int32)
_prototype(hdr, "gpu_hdr_get_proxy_report", [C.c_void_p,
    C.POINTER(ProxyReport)], C.c_int32)
_prototype(hdr, "gpu_hdr_get_reconstruct_report", [C.c_void_p,
    C.POINTER(ReconstructReport)], C.c_int32)
_prototype(hdr, "gpu_hdr_close", [C.c_void_p], None)
hdr.gpu_hdr_last_error.restype = C.c_char_p


def bridge_last_error():
    detail = hdr.gpu_hdr_last_error()
    return detail.decode("utf-8", "replace") if detail else ""


def interop_last_error():
    detail = interop.nvof_d3d12_last_error()
    return detail.decode("utf-8", "replace") if detail else ""


def cuda_error(step, code):
    raise RuntimeError(f"{step} failed (CUDA {int(code)})")


def fill_contract():
    contract = Contract()
    contract.struct_size = C.sizeof(Contract)
    hdr.gpu_hdr_fill_contract(C.byref(contract))
    return contract


def pq_eotf(relative_linear_of_code, contract):
    """ST 2084 code (0..1 domain) -> relative linear, as the contract writes it."""
    root = np.maximum(np.asarray(relative_linear_of_code, np.float64),
                      0.0) ** (1.0 / contract.pq_m2)
    return (np.maximum(root - contract.pq_c1, 0.0)
            / (contract.pq_c2 - contract.pq_c3 * root)) ** (1.0 / contract.pq_m1)


def fixture(index):
    """A deterministic P010 frame whose *right half* is the high-key half.

    The halves differ on purpose: a copy that moved only width * 4 bytes of an
    FP16 row would leave the tail half of every plane stale, and a fixture whose
    tail is indistinguishable from its head would not show that.
    """
    row, column = np.indices((FULL_H, FULL_W))
    left = column < FULL_W // 2
    codes = np.where(left,
                     LUMA_BLACK + (row + index * 7) % 96,
                     600 + (column * 3 + row + index * 11)
                     % (LUMA_CODE_TOP - 600 + 1)).astype(np.uint16)
    y = (codes << 6).astype("<u2")
    chroma_row, chroma_column = np.indices((FULL_H // 2, FULL_W // 2))
    cb = (CHROMA_NEUTRAL +
          (chroma_row + chroma_column + index) % (2 * CHROMA_DEVIATION + 1)
          - CHROMA_DEVIATION)
    cr = (CHROMA_NEUTRAL -
          (chroma_row * 3 + chroma_column * 5 + index * 2)
          % (2 * CHROMA_DEVIATION + 1) + CHROMA_DEVIATION)
    uv = np.empty((FULL_H // 2, FULL_W), dtype="<u2")
    uv[:, 0::2] = cb << 6
    uv[:, 1::2] = cr << 6
    return y, uv


def tap_plan(vertical, horizontal):
    """Two production AxisPlans, serialized the way the bridge reads them.

    The bridge's row is ``[dst_len][taps]`` with ONE stride shared by both axes,
    and it reads exactly ``taps`` columns per row (native/gpu_hdr_scrgb_bridge.cpp
    ``upload_taps``), so the narrower axis is padded out to the wider one with
    weight 0.0 at an index that is already clamp-to-edge resolved. Nothing is ever
    truncated: a plan wider than the table stride is an error, not a crop.

    The plans themselves come from ``pipeline._build_lanczos2_axis_plan`` (colour
    down, scale-dependent window) and ``pipeline._build_catmull_rom_axis_plan``
    (residual up, fixed 4-tap window) - the same CPU reference the export job
    builds, so this chain reconstructs with the production filter contract rather
    than an invented one.
    """
    taps = max(vertical.max_taps, horizontal.max_taps)
    if taps > GPU_HDR_MAX_TAPS:
        raise RuntimeError(f"the {vertical.kernel}/{horizontal.kernel} plan needs "
                           f"{taps} tap columns, the bridge accepts at most "
                           f"{GPU_HDR_MAX_TAPS}")

    def axis(plan):
        if plan.max_taps > taps:
            raise RuntimeError(f"{plan.kernel} {plan.src_len}->{plan.dst_len} "
                               f"carries {plan.max_taps} columns and the table "
                               f"stride is {taps}; refusing to drop taps")
        indices = np.zeros((plan.dst_len, taps), np.int32)
        weights = np.zeros((plan.dst_len, taps), np.float32)
        indices[:, :plan.max_taps] = plan.indices
        weights[:, :plan.max_taps] = plan.weights
        if indices.min() < 0 or indices.max() >= plan.src_len:
            raise RuntimeError(f"{plan.kernel} plan index leaves [0, "
                               f"{plan.src_len})")
        return indices, weights

    vertical_indices, vertical_weights = axis(vertical)
    horizontal_indices, horizontal_weights = axis(horizontal)
    table = TapTable()
    table.struct_size = C.sizeof(TapTable)
    table.taps = taps
    table.vertical_src_len, table.vertical_dst_len = vertical.src_len, vertical.dst_len
    table.vertical_indices = vertical_indices.ctypes.data_as(C.POINTER(C.c_int32))
    table.vertical_weights = vertical_weights.ctypes.data_as(C.POINTER(C.c_float))
    table.horizontal_src_len, table.horizontal_dst_len = horizontal.src_len, horizontal.dst_len
    table.horizontal_indices = horizontal_indices.ctypes.data_as(C.POINTER(C.c_int32))
    table.horizontal_weights = horizontal_weights.ctypes.data_as(C.POINTER(C.c_float))
    record = {
        "table_taps": taps,
        "vertical": {
            "kernel": vertical.kernel, "src_len": vertical.src_len,
            "dst_len": vertical.dst_len, "plan_max_taps": vertical.max_taps,
            "real_taps_min": int(vertical.tap_counts.min()),
            "real_taps_max": int(vertical.tap_counts.max()),
            "indices_sha256": sha256_bytes(vertical_indices.tobytes()),
            "weights_sha256": sha256_bytes(vertical_weights.tobytes()),
            "row_weight_sum_min": float(vertical_weights.sum(axis=1).min()),
            "row_weight_sum_max": float(vertical_weights.sum(axis=1).max()),
        },
        "horizontal": {
            "kernel": horizontal.kernel, "src_len": horizontal.src_len,
            "dst_len": horizontal.dst_len, "plan_max_taps": horizontal.max_taps,
            "real_taps_min": int(horizontal.tap_counts.min()),
            "real_taps_max": int(horizontal.tap_counts.max()),
            "indices_sha256": sha256_bytes(horizontal_indices.tobytes()),
            "weights_sha256": sha256_bytes(horizontal_weights.tobytes()),
            "row_weight_sum_min": float(horizontal_weights.sum(axis=1).min()),
            "row_weight_sum_max": float(horizontal_weights.sum(axis=1).max()),
        },
    }
    # The arrays are the table's only backing store: a caller that let them go
    # would leave the bridge reading freed memory at the next launch.
    return (table, (vertical_indices, vertical_weights, horizontal_indices,
                    horizontal_weights), record)


def row_tail_transport_diagnostic(device, context):
    """Labelled transport test - not part of the frame chain.

    A width * 8 row is copied out of a source whose pitch is deliberately wider
    than width * 8, into the bridge's own 256-byte-aligned plane, and read back
    whole. The assertions are about the second half of every row and about every
    row after the first, which are exactly what an RGBA8-shaped copy loses.
    """
    width, height, pixel_bytes = 256, 128, 8
    payload_bytes = width * pixel_bytes          # 2048 of real row
    source_pitch = payload_bytes + 256           # non-tightly-packed source
    destination_pitch = _aligned(payload_bytes, 256)
    half = payload_bytes // 2                    # the tail this test is about

    source = np.zeros((height, source_pitch // 2), np.uint16)
    halfwords = np.arange(width * 4, dtype=np.uint16)
    for row in range(height):
        source[row, :width * 4] = ((row * 7) + halfwords) & 0xFFFF
    source = np.ascontiguousarray(source)
    source_bytes = np.frombuffer(source.tobytes(), np.uint8).reshape(
        height, source_pitch)

    bridge = C.c_void_p()
    code = interop.cuda_d3d12_color_create2(device, context, width, height,
        destination_pitch, pixel_bytes, 0, C.byref(bridge))
    if code:
        raise RuntimeError(f"cuda_d3d12_color_create2: {interop_last_error()!r}")
    try:
        layout = ColorLayout(C.sizeof(ColorLayout))
        if interop.cuda_d3d12_color_get_layout(bridge, C.byref(layout)):
            raise RuntimeError(f"cuda_d3d12_color_get_layout: "
                               f"{interop_last_error()!r}")

        refusals = []
        for label, probe_width, probe_pitch, probe_bytes in (
                ("row_pitch shorter than width * pixel_bytes", width, 1024, 8),
                ("row_pitch not a 256-byte multiple", width, 2100, 8),
                ("pixel_bytes other than 4 or 8", width, destination_pitch, 5),
                ("width * pixel_bytes overflowing uint32", 1 << 29,
                 destination_pitch, 8)):
            probe = C.c_void_p()
            refused = bool(interop.cuda_d3d12_color_create2(device, context,
                probe_width, height, probe_pitch, probe_bytes, 0,
                C.byref(probe))) and probe.value is None
            refusals.append({"case": label,
                "width": probe_width, "row_pitch": probe_pitch,
                "pixel_bytes": probe_bytes, "refused": refused,
                "error": "" if refused else interop_last_error()})
            if not refused and probe.value:
                interop.cuda_d3d12_color_destroy(probe)

        destination = C.c_uint64()
        reported_pitch = C.c_uint32()
        if interop.cuda_d3d12_color_get_device_pointer(bridge,
                C.byref(destination), C.byref(reported_pitch)):
            raise RuntimeError(f"cuda_d3d12_color_get_device_pointer: "
                               f"{interop_last_error()!r}")
        plane_bytes = layout.plane_bytes
        # Wipe the destination first, so a tail the copy did not move reads back
        # as nothing rather than as somebody else's pixels.
        zeros = C.create_string_buffer(plane_bytes)
        if cuda.cuMemcpyHtoD_v2(destination.value, zeros, plane_bytes):
            cuda_error("cuMemcpyHtoD(wipe destination)", 1)

        source_memory = C.c_uint64()
        if cuda.cuMemAlloc_v2(C.byref(source_memory), source.nbytes):
            cuda_error("cuMemAlloc(row-tail source)", 1)
        try:
            if cuda.cuMemcpyHtoD_v2(source_memory.value, source.ctypes.data,
                    source.nbytes):
                cuda_error("cuMemcpyHtoD(row-tail source)", 1)
            fence_value = C.c_uint64()
            if interop.cuda_d3d12_color_copy(bridge, source_memory.value,
                    source_pitch, C.byref(fence_value)):
                raise RuntimeError(f"cuda_d3d12_color_copy: "
                                   f"{interop_last_error()!r}")
            readback = C.create_string_buffer(plane_bytes)
            if cuda.cuMemcpyDtoH_v2(readback, destination.value, plane_bytes):
                cuda_error("cuMemcpyDtoH(row-tail destination)", 1)
            moved = np.frombuffer(readback.raw, np.uint8).reshape(
                height, reported_pitch.value)
            expected = source_bytes[:, :destination_pitch]
            tail_source = source_bytes[:, half:payload_bytes]
            tail_moved = moved[:, half:destination_pitch]
            head_source = source_bytes[:, :half]
        finally:
            cuda.cuMemFree_v2(source_memory.value)
    finally:
        interop.cuda_d3d12_color_destroy(bridge)

    return {
        "purpose": ("labelled transport diagnostic: a width * 8 row copied from "
                    "a wider-than-payload source pitch and read back whole"),
        "on_frame_path": False,
        "width": width, "height": height, "pixel_bytes": pixel_bytes,
        "source_pitch_bytes": source_pitch,
        "bridge_row_pitch_bytes": int(reported_pitch.value),
        "bridge_bytes_per_row": int(layout.bytes_per_row),
        "bridge_plane_bytes": int(plane_bytes),
        "copy_fence_value": int(fence_value.value),
        "rows_whole_row_matches_source": int(
            (moved == expected).all(axis=1).sum()),
        "rows_tail_half_matches_source": int(
            (tail_moved == tail_source).all(axis=1).sum()),
        "rows_tail_half_not_left_wiped": int(
            np.any(tail_moved != 0, axis=1).sum()),
        "rows_whose_tail_is_their_head": int(
            (tail_moved == head_source).all(axis=1).sum()),
        "create2_refusals": refusals,
    }


def _aligned(value, multiple):
    return (int(value) + multiple - 1) // multiple * multiple


def _main_inner(argv, resources):
    if not engine.hdr_fp16_supported():
        raise RuntimeError("this host build exposes no FP16 colour session")

    # The host owns the D3D12 device the whole chain has to share, so it opens
    # first and the HDR bridge is opened against the device it reports.
    host = engine.Live(WORK_W, WORK_H, SETTINGS, color_format="rgba16f")
    resources["host"] = host
    device = host.d3d12_device()
    if not device:
        raise RuntimeError("the host reported no D3D12 device")

    session = C.c_void_p()
    resources["session"] = session
    if nvof.nvof_bridge_create(WORK_W, WORK_H, NVOF_GRID, NVOF_PERF,
            C.byref(session)):
        raise RuntimeError(f"nvof_bridge_create: "
                           f"{nvof.nvof_bridge_get_last_error()!r}")
    device_index = C.c_uint32()
    context = C.c_void_p()
    owner = C.c_uint32()
    if nvof.nvof_bridge_get_session_info(session, C.byref(device_index),
            C.byref(context), C.byref(owner)):
        raise RuntimeError(f"nvof_bridge_get_session_info: "
                           f"{nvof.nvof_bridge_get_last_error()!r}")

    contract = fill_contract()
    paper_white = CEILING_SCRGB / contract.proxy_actionable_relative
    # The decoded ceiling this fixture can ask for: the brightest luma code,
    # widened by the largest chroma excursion it writes and by the matrix row
    # that amplifies it most. The source report's own max_component is the
    # measurement; this is only the reason the ceiling is a safe declaration.
    chroma_gain = 2.0 * (1.0 - min(contract.kr, contract.kb))
    top_encoded = ((LUMA_CODE_TOP - LUMA_BLACK) / LUMA_SPAN
                   + chroma_gain * CHROMA_DEVIATION / CHROMA_SPAN)
    predicted_max_scrgb = float(pq_eotf(top_encoded, contract)
                                * SCRGB_PER_RELATIVE)
    down_plan, down_keep, down_record = tap_plan(
        pipeline._build_lanczos2_axis_plan(FULL_H, WORK_H),
        pipeline._build_lanczos2_axis_plan(FULL_W, WORK_W))
    up_plan, up_keep, up_record = tap_plan(
        pipeline._build_catmull_rom_axis_plan(WORK_H, FULL_H),
        pipeline._build_catmull_rom_axis_plan(WORK_W, FULL_W))
    plan_built_before = pipeline.filter_plan_build_count

    bridge = C.c_void_p()
    resources["bridge"] = bridge
    code = hdr.gpu_hdr_open(C.c_void_p(device), context, FULL_W, FULL_H,
        WORK_W, WORK_H, C.byref(down_plan), C.byref(up_plan), paper_white,
        C.byref(bridge))
    if code:
        raise RuntimeError(f"gpu_hdr_open: {bridge_last_error()!r}")
    resources["keep"] = (down_keep, up_keep)

    y_pitch, uv_pitch = FULL_W * 2, FULL_W * 2
    y_bytes, uv_bytes = y_pitch * FULL_H, uv_pitch * (FULL_H // 2)
    frames = []
    # Counted from the calls themselves, not inferred from how many iterations
    # finished: the report claims a call count, so it has to have watched for it.
    feature18_calls = 0
    nvof_calls = 0
    previous = C.c_void_p()
    if cuda.cuCtxPushCurrent_v2(context):
        raise RuntimeError("cuCtxPushCurrent failed")
    try:
        source_y = C.c_uint64()
        source_uv = C.c_uint64()
        if cuda.cuMemAlloc_v2(C.byref(source_y), y_bytes):
            cuda_error("cuMemAlloc(P010 luma)", 1)
        try:
            if cuda.cuMemAlloc_v2(C.byref(source_uv), uv_bytes):
                cuda_error("cuMemAlloc(P010 chroma)", 1)
            try:
                for index in range(FRAMES):
                    y, uv = fixture(index)
                    predicted = (pq_eotf(
                        (LUMA_CODE_TOP - LUMA_BLACK) / LUMA_SPAN, contract)
                        * SCRGB_PER_RELATIVE)
                    predicted_max_scrgb = float(predicted)
                    if cuda.cuMemcpyHtoD_v2(source_y.value, y.ctypes.data,
                            y.nbytes):
                        cuda_error(f"cuMemcpyHtoD(luma {index})", 1)
                    if cuda.cuMemcpyHtoD_v2(source_uv.value, uv.ctypes.data,
                            uv.nbytes):
                        cuda_error(f"cuMemcpyHtoD(chroma {index})", 1)
                    reset = index == 0

                    master_ptr, master_pitch = C.c_uint64(), C.c_uint32()
                    if hdr.gpu_hdr_decode(bridge, source_y.value, y_pitch,
                            source_uv.value, uv_pitch, 1, 9, 16, 10,
                            CEILING_SCRGB, C.byref(master_ptr),
                            C.byref(master_pitch)):
                        raise RuntimeError(f"decode {index}: "
                                           f"{bridge_last_error()!r}")
                    working_ptr, working_pitch = C.c_uint64(), C.c_uint32()
                    if hdr.gpu_hdr_downscale(bridge, C.byref(working_ptr),
                            C.byref(working_pitch)):
                        raise RuntimeError(f"downscale {index}: "
                                           f"{bridge_last_error()!r}")
                    proxy_ptr, proxy_pitch = C.c_uint64(), C.c_uint32()
                    guidance_ptr, guidance_pitch = C.c_uint64(), C.c_uint32()
                    if hdr.gpu_hdr_proxy(bridge, C.byref(proxy_ptr),
                            C.byref(proxy_pitch), C.byref(guidance_ptr),
                            C.byref(guidance_pitch)):
                        raise RuntimeError(f"proxy {index}: "
                                           f"{bridge_last_error()!r}")

                    # NVOF reads the 8-bit guidance surface and nothing else;
                    # the FP16 proxy never enters it.
                    if nvof.nvof_bridge_execute_rgba_device_sequential(
                            session, guidance_ptr.value, guidance_pitch.value,
                            context, int(reset), None):
                        raise RuntimeError(f"nvof execute {index}: "
                            f"{nvof.nvof_bridge_get_last_error()!r}")
                    flow_ptr, flow_pitch = C.c_uint64(), C.c_size_t()
                    if nvof.nvof_bridge_get_output_deviceptr(session,
                            C.byref(flow_ptr), C.byref(flow_pitch)):
                        raise RuntimeError(f"flow pointer {index}: "
                            f"{nvof.nvof_bridge_get_last_error()!r}")
                    nvof_calls += 1

                    color = {"format": "rgba16f_cuda", "width": WORK_W,
                        "height": WORK_H, "cuda_context": context,
                        "device_ptr": proxy_ptr.value,
                        "pitch_bytes": int(proxy_pitch.value)}
                    flow = {"format": "float32x2", "width": WORK_W,
                        "height": WORK_H, "cuda_context": context,
                        "device_ptr": flow_ptr.value,
                        "pitch_bytes": int(flow_pitch.value)}
                    output = host.process_device_color_flow_resource(
                        color, flow, reset=reset)
                    feature18_calls += 1

                    composited_ptr, composited_pitch = C.c_uint64(), C.c_uint32()
                    if hdr.gpu_hdr_reconstruct(bridge, output["d3d12_resource"],
                            output["d3d12_fence"], output["fence_value"],
                            RESIDUAL_STRENGTH, C.byref(composited_ptr),
                            C.byref(composited_pitch)):
                        raise RuntimeError(f"reconstruct {index}: "
                                           f"{bridge_last_error()!r}")

                    frames.append({
                        "frame_index": index,
                        "reset": int(reset),
                        "source_luma_sha256": sha256_bytes(y.tobytes()),
                        "master_extent": [FULL_W, FULL_H],
                        "master_pitch_bytes": int(master_pitch.value),
                        "working_master_pitch_bytes": int(working_pitch.value),
                        "proxy_pitch_bytes": int(proxy_pitch.value),
                        "guidance_pitch_bytes": int(guidance_pitch.value),
                        "flow_pitch_bytes": int(flow_pitch.value),
                        "feature18_dxgi_format": int(output["dxgi_format"]),
                        "feature18_output_extent": [int(output["width"]),
                                                    int(output["height"])],
                        "feature18_fence_value": int(output["fence_value"]),
                        "feature18_calls_so_far": feature18_calls,
                        "nvof_calls_so_far": nvof_calls,
                        "color_info_after_frame": host.color_info(),
                        "composited_pitch_bytes": int(composited_pitch.value),
                    })

                # Snapshot first: every number below has to describe the chain
                # as it ran, before a diagnostic byte of readback touched it.
                counters_before = host.transfer_counters()
                bridge_counters = BridgeCounters(C.sizeof(BridgeCounters))
                if hdr.gpu_hdr_get_transfer_counters(bridge,
                        C.byref(bridge_counters)):
                    raise RuntimeError(f"bridge counters: "
                                       f"{bridge_last_error()!r}")
                source_report = _report(SourceReport,
                    hdr.gpu_hdr_get_source_report, bridge)
                proxy_report = _report(ProxyReport,
                    hdr.gpu_hdr_get_proxy_report, bridge)
                reconstruct_report = _report(ReconstructReport,
                    hdr.gpu_hdr_get_reconstruct_report, bridge)

                # --- diagnostics, deliberately after the snapshot ---
                feature18_output = host.readback_output()
                proxy_plane = np.empty((WORK_H, WORK_W, 4), np.float16)
                if hdr.gpu_hdr_readback_scrgb(bridge, PLANE_PROXY,
                        proxy_plane.ctypes.data, proxy_plane.strides[0]):
                    raise RuntimeError(f"proxy readback: "
                                       f"{bridge_last_error()!r}")
                composited = np.empty((FULL_H, FULL_W, 4), np.float16)
                if hdr.gpu_hdr_readback_scrgb(bridge, PLANE_COMPOSITED,
                        composited.ctypes.data, composited.strides[0]):
                    raise RuntimeError(f"composited readback: "
                                       f"{bridge_last_error()!r}")
                guidance_plane = np.empty((WORK_H, WORK_W, 4), np.uint8)
                if hdr.gpu_hdr_readback_scrgb(bridge, PLANE_GUIDANCE,
                        guidance_plane.ctypes.data, guidance_plane.strides[0]):
                    raise RuntimeError(f"guidance readback: "
                                       f"{bridge_last_error()!r}")
                counters_after = host.transfer_counters()
                after_readback_counters = BridgeCounters(C.sizeof(BridgeCounters))
                if hdr.gpu_hdr_get_transfer_counters(bridge,
                    C.byref(after_readback_counters)):
                    raise RuntimeError(f"bridge counters after readback: "
                                       f"{bridge_last_error()!r}")
                finite = {
                    "feature18_output_all_finite": bool(
                        np.all(np.isfinite(feature18_output))),
                    "proxy_plane_all_finite": bool(
                        np.all(np.isfinite(proxy_plane))),
                    "composited_all_finite": bool(np.all(np.isfinite(composited))),
                    "composited_min_component": float(np.min(composited)),
                    "composited_max_component": float(np.max(composited)),
                }
            finally:
                cuda.cuMemFree_v2(source_uv.value)
        finally:
            cuda.cuMemFree_v2(source_y.value)
    finally:
        cuda.cuCtxPopCurrent_v2(C.byref(previous))

    tail_diagnostic = row_tail_transport_diagnostic(device, context)

    # Snapshot the session's own numbers HERE, while it is still open. The host
    # raises on colour info once closed, so a report that read the session after
    # release would fail (or, worse, describe a session that no longer exists).
    # Nothing below this point touches `host`, `bridge` or `session`: the release
    # order is enforced by the fact that this function no longer needs them.
    final_colour_info = host.color_info()
    session_identity = {
        "session_id": host.session_id,
        "color_format": host.color_format,
        "bytes_per_pixel": host.bytes_per_pixel,
        "host_abi": host.host_abi,
        "alive_before_release": host.is_alive(),
    }

    def row_spans(plane, pitch, bytes_per_row):
        """Does the second half of every row hold the transport's payload?"""
        tail = plane[:, plane.shape[1] // 2:]
        return {
            "rows": int(plane.shape[0]),
            "row_bytes": bytes_per_row,
            "row_pitch": pitch,
            "tail_nonzero_rows": int(np.any(tail != 0, axis=(1, 2)).sum()),
            "tail_sha256": sha256_bytes(tail.tobytes()),
            "plane_sha256": sha256_bytes(plane.tobytes()),
        }

    return {
        "kind": "hdr_fp16_device_chain_stage_b",
        "build_identity": {
            "dlss_gpu_host_dll": str(NATIVE / "dlss_gpu_host.dll"),
            "dlss_gpu_host_dll_sha256": sha256_file(NATIVE / "dlss_gpu_host.dll"),
            "nvof_d3d12_interop_dll_sha256":
                sha256_file(NATIVE / "nvof_d3d12_interop.dll"),
            "nvof_bridge_dll_sha256": sha256_file(NATIVE / "nvof_bridge.dll"),
            "gpu_hdr_scrgb_bridge_dll_sha256":
                sha256_file(NATIVE / "gpu_hdr_scrgb_bridge.dll"),
            "hdr_bridge_abi": int(hdr.gpu_hdr_abi_version()),
            "row_bridge_capabilities": int(
                interop.cuda_d3d12_color_row_capabilities()),
            "engine_capabilities": engine._load().color_capabilities,
        },
        "session": {
            **session_identity,
            "extent": [WORK_W, WORK_H],
            "d3d12_device_shared_with_bridge": bool(device),
            "cuda_context_owner": "nvof_bridge session",
        },
        "job_constants": {
            "full_extent": [FULL_W, FULL_H],
            "working_extent": [WORK_W, WORK_H],
            "working_extent_rule": ("pipeline.resolve_working_dims(512, 512, "
                                    f"{MASTER_SCALE})"),
            "support_ceiling_scrgb": CEILING_SCRGB,
            "paper_white_scrgb": paper_white,
            "paper_white_rule": "ceiling / contract.proxy_actionable_relative",
            "residual_strength": RESIDUAL_STRENGTH,
            "plan_contract": ("colour down = pipeline._build_lanczos2_axis_plan, "
                              "residual up = pipeline._build_catmull_rom_axis_plan"
                              " - the production ResamplerPlan pairing, not a "
                              "substitute resampler"),
            "downscale_plan": down_record,
            "reconstruct_plan": up_record,
            "settings": dict(SETTINGS),
            "predicted_max_source_scrgb": predicted_max_scrgb,
        },
        "filter_plan_builds_during_chain": (
            pipeline.filter_plan_build_count - plan_built_before),
        "frames_run": frames,
        "reset_per_frame": [int(frame["reset"]) for frame in frames],
        "feature18_call_count": feature18_calls,
        "nvof_call_count": nvof_calls,
        "counters_before_any_pixel_readback": counters_before,
        "bridge_counters_before_any_pixel_readback": {
            name: getattr(bridge_counters, name) for name, _ in
            BridgeCounters._fields_ if name != "struct_size"},
        "source_report": source_report,
        "proxy_report": proxy_report,
        "reconstruct_report": reconstruct_report,
        "diagnostic_readbacks_after_snapshot": {
            "feature18_output_sha256": sha256_bytes(feature18_output.tobytes()),
            "feature18_output_dtype": str(feature18_output.dtype),
            "feature18_output_max_scrgb": float(np.max(feature18_output)),
            "feature18_output_row": row_spans(feature18_output,
                feature18_output.strides[0], WORK_W * 8),
            "proxy_row": row_spans(proxy_plane, proxy_plane.strides[0],
                WORK_W * 8),
            "composited_row": row_spans(composited, composited.strides[0],
                FULL_W * 8),
            "guidance_format": "rgba8",
            "guidance_bytes_per_pixel": 4,
            "guidance_sha256": sha256_bytes(guidance_plane.tobytes()),
            "host_counters_after": counters_after,
            "bridge_report_readback_bytes_after": int(
                after_readback_counters.report_readback_bytes),
            "finite_component_checks": finite,
        },
        "row_tail_transport_diagnostic": tail_diagnostic,
        "colour_info_final": final_colour_info,
        "not_established": [
            "no P010 encode or NVENC export ran in this job",
            "no picture-quality or performance claim",
            "the chain is a transport and format check on 3 frames, not a run",
            ("the formal colour contract is not proven by this gate; the FP16 "
             "session is an experimental opt-in mode"),
        ],
    }


def _refusals():
    """The interface's own no's, each one checked rather than assumed.

    This runs only after the chain has released the NGX core, and it does not
    loosen the one-active-core gate to make a call work: the SDR session joins the
    FP16 anchor through the existing ``open_layer`` entry point, which is the only
    way a second Feature session may exist while the first is still open.
    """
    records = []
    flow = {"format": "float32x2", "width": WORK_W, "height": WORK_H,
        "cuda_context": C.c_void_p(1), "device_ptr": 1,
        "pitch_bytes": WORK_W * 8}

    try:
        engine.Live(64, 64, SETTINGS, color_format="rgba32i")
        records.append({"case": "unknown colour name", "refused": False})
    except ValueError as error:
        records.append({"case": "unknown colour name", "refused": True,
                        "error": str(error)})

    session = engine.Live(WORK_W, WORK_H, SETTINGS, color_format="rgba16f")
    try:
        for label, color in (
                ("rgba8 descriptor on an FP16 session",
                 {"format": "rgba8_cuda", "width": WORK_W, "height": WORK_H,
                     "cuda_context": C.c_void_p(1), "device_ptr": 1,
                     "pitch_bytes": WORK_W * 4}),
                ("FP16 session with a width*4 pitch",
                 {"format": "rgba16f_cuda", "width": WORK_W, "height": WORK_H,
                     "cuda_context": C.c_void_p(1), "device_ptr": 1,
                     "pitch_bytes": WORK_W * 4})):
            try:
                session.process_device_color_flow_resource(color, dict(flow))
                records.append({"case": label, "refused": False})
            except ValueError as error:
                records.append({"case": label, "refused": True,
                                "error": str(error)})
    finally:
        session.close()

    anchor = engine.Live(WORK_W, WORK_H, SETTINGS, color_format="rgba16f")
    layer = None
    try:
        try:
            stray = engine.Live(WORK_W, WORK_H, SETTINGS,
                                color_format="rgba16f")
            stray.close()
            records.append({
                "case": "second independent core while one is active",
                "refused": False})
        except engine.DLSSInferenceError as error:
            records.append({
                "case": "second independent core while one is active",
                "refused": True, "error": str(error)})

        layer = engine.Live(WORK_W, WORK_H, SETTINGS, shared_with=anchor)
        layer_evidence = {
            "how": "engine.Live(..., shared_with=anchor) - the open_layer path",
            "anchor_session_id": anchor.session_id,
            "layer_session_id": layer.session_id,
            "anchor_dxgi_format": int(anchor.color_info()["dxgi_format"]),
            "layer_dxgi_format": int(layer.color_info()["dxgi_format"]),
            "anchor_bytes_per_pixel": anchor.bytes_per_pixel,
            "layer_bytes_per_pixel": layer.bytes_per_pixel,
        }
        layer.close()
        # The owner of the core outlives the layer that borrowed it.
        layer_evidence.update({
            "anchor_alive_after_layer_close": anchor.is_alive(),
            "anchor_dxgi_after_layer_close": int(
                anchor.color_info()["dxgi_format"]),
        })
    finally:
        if layer is not None:
            layer.close()
        anchor.close()

    reopened = engine.Live(WORK_W, WORK_H, SETTINGS, color_format="rgba16f")
    reopen_evidence = {
        "opened_after_every_close": True,
        "session_id": reopened.session_id,
        "dxgi_format": int(reopened.color_info()["dxgi_format"]),
        "bytes_per_pixel": reopened.bytes_per_pixel,
    }
    reopened.close()
    records.append({
        "case": ("what this API cannot tell: an 8-bit colour pointer given a "
                 "width*8 padded pitch"),
        "refused": None,
        "note": ("row length alone cannot distinguish a raw RGBA8 pointer from an "
                 "FP16 one; the C entry points are a trusted typed contract and "
                 "the descriptor format string is what the Python wrapper checks. "
                 "This is a documented limit, not a probe this gate ran."),
    })
    return {"refusals": records,
            "layered_multi_session": layer_evidence,
            "reopened_after_close": reopen_evidence}


def release(resources):
    """Release consumers before owners, and only what is still held.

    The HDR bridge borrows the host's ID3D12Device and the NVOF session's
    CUcontext, so it goes first; the host session owns the device; the NVOF
    session that provided the CUDA context goes last. Every path here - success
    or exception - runs it.
    """
    bridge = resources.get("bridge")
    host = resources.get("host")
    session = resources.get("session")
    if bridge is not None and bridge.value:
        hdr.gpu_hdr_close(bridge)
        bridge.value = None
    if host is not None:
        host.close()
    if session is not None and session.value:
        nvof.nvof_bridge_destroy(session)
        session.value = None
    resources.pop("keep", None)


def _check(checks, name, expected, actual):
    checks.append({"check": name, "expected": expected, "actual": actual,
                   "passed": bool(actual == expected)})


def _all(checks, name, values):
    checks.append({"check": name, "expected": True, "actual": bool(values),
                   "passed": bool(all(values))})


def evaluate(report):
    """Decide every claim this report makes, instead of describing them.

    A check that came back False or is not known is a FAIL, never a PASS, and the
    caller exits non-zero on any unmet check.
    """
    checks = []
    frames = report["frames_run"]
    colour = report["colour_info_final"]
    counters = report["counters_before_any_pixel_readback"]
    bridge_counters = report["bridge_counters_before_any_pixel_readback"]
    diagnostics = report["diagnostic_readbacks_after_snapshot"]
    tail = report["row_tail_transport_diagnostic"]
    source = report["source_report"]
    proxy = report["proxy_report"]
    reconstruct = report["reconstruct_report"]
    down = report["job_constants"]["downscale_plan"]
    up = report["job_constants"]["reconstruct_plan"]
    layer = report["layered_multi_session"]
    reopen = report["reopened_after_close"]
    rows = tail["height"]

    _check(checks, "frames_run", FRAMES, len(frames))
    _check(checks, "reset_per_frame", [1, 0, 0], report["reset_per_frame"])
    _check(checks, "feature18_call_count", FRAMES, report["feature18_call_count"])
    _check(checks, "nvof_call_count", FRAMES, report["nvof_call_count"])
    _check(checks, "no_colour_re_planning_per_frame", 0,
           report["filter_plan_builds_during_chain"])

    # Colour, formats and extents the host itself reported.
    _check(checks, "session_color_format", "rgba16f", report["session"]["color_format"])
    _check(checks, "session_bytes_per_pixel", 8, report["session"]["bytes_per_pixel"])
    _check(checks, "colour_info_dxgi_format", DXGI_R16G16B16A16_FLOAT,
           int(colour["dxgi_format"]))
    _check(checks, "colour_info_row_bytes", WORK_W * 8, int(colour["row_bytes"]))
    _check(checks, "colour_info_bridge_bytes_per_row", WORK_W * 8,
           int(colour["bridge_bytes_per_row"]))
    _all(checks, "every_frame_output_is_RGBA16F",
         [frame["feature18_dxgi_format"] == DXGI_R16G16B16A16_FLOAT
          for frame in frames])
    _all(checks, "every_frame_output_extent",
         [frame["feature18_output_extent"] == [WORK_W, WORK_H] for frame in frames])
    _all(checks, "fp16_rows_are_eight_bytes_wide",
         [frame["proxy_pitch_bytes"] >= WORK_W * 8 and
          frame["master_pitch_bytes"] >= FULL_W * 8 and
          frame["composited_pitch_bytes"] >= FULL_W * 8 for frame in frames])
    _all(checks, "guidance_stays_eight_bit",
         [frame["guidance_pitch_bytes"] == _aligned(WORK_W * 4, 256)
          for frame in frames])
    _check(checks, "guidance_bytes_per_pixel", 4,
           diagnostics["guidance_bytes_per_pixel"])

    # The chain's zero-host-colour claim, snapshotted before any pixel readback.
    _check(checks, "host_color_stage_count", 0, counters["host_color_stage_count"])
    _check(checks, "output_readback_count_at_snapshot", 0,
           counters["output_readback_count"])
    _check(checks, "device_color_copy_count", FRAMES,
           counters["device_color_copy_count"])
    _check(checks, "bridge_process_count", FRAMES, bridge_counters["process_count"])
    _check(checks, "readback_counted_only_after_snapshot", True,
           diagnostics["host_counters_after"]["output_readback_count"] > 0)

    # The production resampling contract, measured off the plans actually handed
    # to the bridge rather than off a constant in this file.
    _check(checks, "downscale_vertical_kernel", "lanczos2", down["vertical"]["kernel"])
    _check(checks, "downscale_horizontal_kernel", "lanczos2",
           down["horizontal"]["kernel"])
    _check(checks, "downscale_table_taps",
           max(pipeline.lanczos2_axis_taps(FULL_H, WORK_H),
               pipeline.lanczos2_axis_taps(FULL_W, WORK_W)), down["table_taps"])
    _check(checks, "reconstruct_vertical_kernel", "catmull-rom",
           up["vertical"]["kernel"])
    _check(checks, "reconstruct_horizontal_kernel", "catmull-rom",
           up["horizontal"]["kernel"])
    _check(checks, "reconstruct_table_taps", pipeline.CATMULL_ROM_TAPS,
           up["table_taps"])
    _all(checks, "plan_rows_are_weight_normalised",
         [abs(record["row_weight_sum_min"] - 1.0) < 1e-6 and
          abs(record["row_weight_sum_max"] - 1.0) < 1e-6
          for record in (down["vertical"], down["horizontal"], up["vertical"],
                         up["horizontal"])])

    # Row-tail transport: the second half of every FP16 row.
    _check(checks, "row_tail_whole_rows_moved", rows,
           tail["rows_whole_row_matches_source"])
    _check(checks, "row_tail_tail_half_moved", rows,
           tail["rows_tail_half_matches_source"])
    _check(checks, "row_tail_not_left_wiped", rows,
           tail["rows_tail_half_not_left_wiped"])
    _check(checks, "row_tail_is_not_a_copy_of_the_head", 0,
           tail["rows_whose_tail_is_their_head"])
    _check(checks, "row_bridge_reports_eight_byte_rows", 2048,
           tail["bridge_bytes_per_row"])
    _all(checks, "create2_refusals_held",
         [case["refused"] for case in tail["create2_refusals"]])

    # Refusals, layering and reopen - each one a result, not an assumption.
    _all(checks, "refusals_held",
         [case["refused"] is True for case in report["refusals"]
          if case["refused"] is not None])
    _check(checks, "layer_is_a_separate_session", True,
           layer["anchor_session_id"] != layer["layer_session_id"])
    _check(checks, "anchor_keeps_rgba16f", DXGI_R16G16B16A16_FLOAT,
           layer["anchor_dxgi_format"])
    _check(checks, "layer_session_is_rgba8", DXGI_R8G8B8A8_UNORM,
           layer["layer_dxgi_format"])
    _check(checks, "anchor_survives_layer_close", True,
           layer["anchor_alive_after_layer_close"])
    _check(checks, "reopen_after_final_close", True,
           reopen["opened_after_every_close"])
    _check(checks, "reopened_session_is_rgba16f", DXGI_R16G16B16A16_FLOAT,
           reopen["dxgi_format"])

    # Finite values: measured, not assumed.
    _check(checks, "source_nonfinite_samples", 0, source["nonfinite_samples"])
    _check(checks, "source_above_ceiling_components", 0,
           source["above_ceiling_components"])
    _check(checks, "proxy_negative_components_within_report", True,
           proxy["negative_components"] <= proxy["pixels"] * 4)
    _check(checks, "reconstruct_nonfinite_components", 0,
           reconstruct["nonfinite_components"])
    _check(checks, "composite_nonfinite_components", 0,
           reconstruct["composite_nonfinite_components"])
    _check(checks, "feature18_output_finite", True,
           diagnostics["finite_component_checks"]["feature18_output_all_finite"])
    _check(checks, "proxy_plane_finite", True,
           diagnostics["finite_component_checks"]["proxy_plane_all_finite"])
    _check(checks, "composited_finite", True,
           diagnostics["finite_component_checks"]["composited_all_finite"])
    return checks


def main(argv):
    resources = {}
    report = None
    try:
        report = _main_inner(argv, resources)
    finally:
        release(resources)
    if report is None:
        raise RuntimeError("the chain did not complete; see the failure report")
    # The core is released, so a fresh independent session is openable here and
    # the multi-session evidence means what it says.
    report.update(_refusals())
    report["assertions"] = evaluate(report)
    failed = [item["check"] for item in report["assertions"] if not item["passed"]]
    report["failed_checks"] = failed
    report["status"] = "PASS" if not failed else "FAIL"
    report["status_note"] = ("no HDR10 export claimed; this gate checks transport, "
                             "formats, counters and session lifetime on 3 frames")
    return report


if __name__ == "__main__":
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    def emit(payload):
        text = json.dumps(payload, indent=2)
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
        print(text)

    try:
        result = main(sys.argv)
    except Exception as error:  # noqa: BLE001 - a gate that cannot run still emits why
        # A gate that could not run still reports why, and still exits non-zero.
        emit({
            "kind": "hdr_fp16_device_chain_stage_b",
            "status": "FAIL",
            "failure": {"type": type(error).__name__, "message": str(error),
                        "traceback": traceback.format_exc()},
            "assertions": [{"check": "the chain completed", "expected": True,
                            "actual": False, "passed": False}],
            "failed_checks": ["the chain completed"],
            "build_identity": {
                "dlss_gpu_host_dll_sha256":
                    sha256_file(NATIVE / "dlss_gpu_host.dll"),
                "nvof_d3d12_interop_dll_sha256":
                    sha256_file(NATIVE / "nvof_d3d12_interop.dll"),
                "nvof_bridge_dll_sha256": sha256_file(NATIVE / "nvof_bridge.dll"),
                "gpu_hdr_scrgb_bridge_dll_sha256":
                    sha256_file(NATIVE / "gpu_hdr_scrgb_bridge.dll")},
        })
        raise SystemExit(1)
    emit(result)
    if result["status"] != "PASS":
        print(f"FAIL: {len(result['failed_checks'])} of "
              f"{len(result['assertions'])} checks did not hold: "
              + ", ".join(result["failed_checks"]), file=sys.stderr)
        raise SystemExit(1)
    print(f"PASS: {len(result['assertions'])} checks hold; no HDR10 export "
          f"claimed and no performance claim made.")
