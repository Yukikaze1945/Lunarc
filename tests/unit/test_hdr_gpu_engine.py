"""Focused CPU/contract tests for the production HDR bridge wrapper.

Every test drives an injected fake ABI: no DLL is loaded, no GPU is touched, no
pixel is read back and no native gate is rerun. The point is the wrapper's own
promises - that its ctypes mirrors match the header, that it refuses an ABI it
does not speak, that descriptors carry the real format and the borrowed context,
that RGBA8 guidance is never dressed as FP16, and that the owner closes exactly
once and refuses use after close or from the wrong thread.

Struct sizes are asserted against the x64 layout the compiled C++ ``struct_size``
guards in native/gpu_hdr_scrgb_bridge.cpp require, so a mirror that drifts from
the header fails here instead of at a real open.
"""
import ctypes
import os
import pathlib
import re
import threading

import numpy as np
import pytest

import hdr_gpu_engine as hdr
import pipeline


# The ABI-1 layout the compiled bridge enforces on every struct it fills.
EXPECTED_SIZES = {
    "GpuHdrTapTable": 56,
    "GpuHdrContract": 288,
    "GpuHdrSourceReport": 64,
    "GpuHdrProxyReport": 48,
    "GpuHdrReconstructReport": 80,
    "GpuHdrEncodeReport": 72,
    "GpuHdrTransferCounters": 72,
    "GpuHdrMultiscaleCounters": 72,
}


class FakeLibrary:
    """A stand-in for gpu_hdr_scrgb_bridge.dll: records calls, fills out-params,
    and can be told to fail an ABI check, an open, or any stage.

    The multiscale group is present by default and can be taken away symbol by
    symbol with ``missing``, or given a version of its own with ``multiscale_abi``
    (0 for a bridge that answers nothing recognisable), because the wrapper has to
    ask for it rather than infer it from ABI 1."""

    def __init__(self, abi=1, last_error=b"synthetic bridge fault",
                 paper_white=3.0, missing=None, open_code=0,
                 fail_symbol=None, set_handle_on_failed_open=False,
                 handle=0xA11CE, multiscale_abi=1):
        self._abi = abi
        self._last_error = last_error
        self._paper_white = paper_white
        self._missing = set(missing or ())
        self._open_code = open_code
        self._fail_symbol = fail_symbol
        self._set_handle_on_failed_open = set_handle_on_failed_open
        self._multiscale_abi = multiscale_abi
        self.calls = []
        self.close_count = 0
        # 0 simulates a bridge that reports success and leaves the out-slot null.
        self._handle = handle
        # Recognisable per-stage device addresses the wrapper must hand back.
        self._outs = {
            "gpu_hdr_decode": (0x1000, 8192),
            "gpu_hdr_downscale": (0x2000, 4096),
            "gpu_hdr_proxy": (0x3000, 4096, 0x4000, 1024),
            "gpu_hdr_reconstruct": (0x5000, 8192),
            "gpu_hdr_encode": (0x6000, 8192, 0x7000, 4096),
            "gpu_hdr_import_master": (0x1000, 8192),
            "gpu_hdr_observe": (0x9000, 4096),
            "gpu_hdr_set_composite": (0xA000, 8192),
        }
        self._report_fields = {
            "gpu_hdr_get_source_report": {
                "support_ceiling_scrgb": 12.5, "paper_white_scrgb": 3.0,
                "nonfinite_samples": 0, "above_ceiling_components": 0,
                "min_component": 0.0, "max_component": 9.0, "pixels": 65536},
            "gpu_hdr_get_proxy_report": {
                "negative_components": 4, "pinned_components": 0,
                "pixels": 65536},
            "gpu_hdr_get_reconstruct_report": {
                "applied_components": 10, "nonfinite_components": 0,
                "composite_nonfinite_components": 0, "pixels": 65536},
            "gpu_hdr_get_encode_report": {
                "clip_ceiling_scrgb": 125.0, "clamped_high_components": 1,
                "clamped_low_components": 0, "max_code_y": 1023.0},
        }
        self._counter_fields = {
            "process_count": 3, "source_device_copy_bytes": 1024,
            "composited_device_write_bytes": 2048, "encode_device_write_bytes": 0,
            "report_readback_bytes": 256}
        self._contract_fields = {
            "scrgb_per_nit": 0.0125, "nits_per_scrgb": 80.0,
            "proxy_actionable_relative": 2.5, "max_stops": 4.0}
        # 512x512 full and 256x256 working, the extents _bridge() opens with.
        self._multiscale_fields = {
            "master_provenance": 2, "master_import_count": 2,
            "master_import_device_bytes": 2 * 8192 * 512,
            "observe_count": 2, "observe_copy_bytes": 2 * 4096 * 256,
            "observe_residual_bytes": 2 * 4096 * 256,
            "composite_import_count": 1, "composite_import_bytes": 8192 * 512,
            "pass_dispatches": 2}

    def __getattr__(self, name):
        if name in self._missing:
            raise AttributeError(name)
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args):
            return self._dispatch(name, args)
        return call

    def _dispatch(self, symbol, args):
        self.calls.append((symbol, args))
        if symbol == "gpu_hdr_abi_version":
            return self._abi
        if symbol == "gpu_hdr_multiscale_abi_version":
            return self._multiscale_abi
        if symbol == "gpu_hdr_last_error":
            return self._last_error
        if symbol == "gpu_hdr_paper_white":
            return self._paper_white
        if symbol == "gpu_hdr_close":
            self.close_count += 1
            return None
        code = 1 if symbol == self._fail_symbol else 0
        if symbol == "gpu_hdr_fill_contract":
            self._fill_struct(args[0]._obj, self._contract_fields)
            return None
        if symbol in ("gpu_hdr_open", "gpu_hdr_open_purpose"):
            out_handle = args[-1]._obj
            if self._open_code == 0 or self._set_handle_on_failed_open:
                out_handle.value = self._handle
            return self._open_code
        if code:
            return code
        if symbol == "gpu_hdr_decode":
            self._set(args[10], args[11], self._outs[symbol])
        elif symbol == "gpu_hdr_downscale":
            self._set(args[1], args[2], self._outs[symbol])
        elif symbol == "gpu_hdr_reconstruct":
            self._set(args[5], args[6], self._outs[symbol])
        elif symbol == "gpu_hdr_proxy":
            for position, value in zip((1, 2, 3, 4), self._outs[symbol]):
                args[position]._obj.value = value
        elif symbol == "gpu_hdr_encode":
            for position, value in zip((2, 3, 4, 5), self._outs[symbol]):
                args[position]._obj.value = value
        elif symbol in self._report_fields:
            self._fill_struct(args[1]._obj, self._report_fields[symbol])
        elif symbol == "gpu_hdr_get_transfer_counters":
            self._fill_struct(args[1]._obj, self._counter_fields)
        elif symbol == "gpu_hdr_import_master":
            self._set(args[5], args[6], self._outs[symbol])
        elif symbol in ("gpu_hdr_observe", "gpu_hdr_set_composite"):
            self._set(args[4], args[5], self._outs[symbol])
        elif symbol == "gpu_hdr_get_multiscale_counters":
            self._fill_struct(args[1]._obj, self._multiscale_fields)
        return 0

    @staticmethod
    def _set(ptr_byref, pitch_byref, values):
        # byref() gives a CArgObject; the real bridge writes through _obj.value.
        ptr_byref._obj.value, pitch_byref._obj.value = values

    @staticmethod
    def _fill_struct(struct, fields):
        for name, value in fields.items():
            setattr(struct, name, value)


def _ptr_addr(pointer):
    """The integer address a ctypes POINTER field holds, to compare by value."""
    return ctypes.cast(pointer, ctypes.c_void_p).value


def _bridge(fake=None, **overrides):
    kwargs = dict(d3d12_device=0xBEEF, cuda_context=ctypes.c_void_p(0x9999),
                  full_width=512, full_height=512, working_width=256,
                  working_height=256, paper_white_scrgb=3.0)
    kwargs.update(overrides)
    return hdr.GpuHdrBridge(library=fake or FakeLibrary(), **kwargs)


# ---------------------------------------------------------------------------
# layout mirrors: the struct must be the header's, not an approximation of it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,size", sorted(EXPECTED_SIZES.items()))
def test_struct_size_matches_the_compiled_abi(name, size):
    struct = getattr(hdr, name)
    assert ctypes.sizeof(struct) == size


def test_field_order_mirrors_the_header_field_for_field():
    """A reordered field keeps the size and still corrupts the struct_size guard;
    the names in order are what bind the mirror to the header."""
    assert [f[0] for f in hdr.GpuHdrTapTable._fields_] == [
        "struct_size", "taps", "vertical_src_len", "vertical_dst_len",
        "vertical_indices", "vertical_weights", "horizontal_src_len",
        "horizontal_dst_len", "horizontal_indices", "horizontal_weights"]
    assert [f[0] for f in hdr.GpuHdrContract._fields_][:4] == [
        "struct_size", "scrgb_per_nit", "nits_per_scrgb", "to_scrgb"]
    assert [f[0] for f in hdr.GpuHdrTransferCounters._fields_] == [
        "struct_size", "process_count", "source_device_copy_bytes",
        "source_device_write_bytes", "working_device_write_bytes",
        "guidance_device_write_bytes", "composited_device_write_bytes",
        "encode_device_write_bytes", "report_readback_bytes"]


def test_fill_contract_sends_the_mirror_its_own_size():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    contract = bridge.contract
    assert contract["proxy_actionable_relative"] == 2.5
    assert contract["nits_per_scrgb"] == 80.0
    sent = [a for s, a in fake.calls if s == "gpu_hdr_fill_contract"][0]
    assert a_struct_size(sent) == ctypes.sizeof(hdr.GpuHdrContract)


def a_struct_size(args):
    return args[0]._obj.struct_size


def test_report_getters_send_the_mirror_size_and_return_plain_scalars():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    report = bridge.source_report()
    # struct_size is stripped; only the ABI's own fields survive, JSON-safe.
    assert "struct_size" not in report
    assert report["support_ceiling_scrgb"] == 12.5
    assert report["pixels"] == 65536
    import json
    json.dumps(report)  # serializable, no numpy or ctypes objects leak out


def test_transfer_counters_expose_only_real_fields():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    counters = bridge.transfer_counters()
    assert counters["process_count"] == 3
    # An un-set counter is a genuine field the ABI carries, present as zero, not a
    # fabricated number for something the bridge never measures.
    assert counters["encode_device_write_bytes"] == 0
    # The ABI carries exactly one readback field, and it is the counter for image
    # planes copied out by the diagnostic gpu_hdr_readback_* entries. Nothing in this
    # struct measures the per-pass counter readback, so no consumer may read a zero
    # here as that scalar boundary having gone uncrossed.
    assert [name for name in counters if "readback" in name] == [
        "report_readback_bytes"]
    assert set(counters) == set(
        f[0] for f in hdr.GpuHdrTransferCounters._fields_ if f[0] != "struct_size")


# ---------------------------------------------------------------------------
# ABI / export gate: refuse a bridge this wrapper does not speak
# ---------------------------------------------------------------------------

def test_abi_mismatch_is_refused_before_open():
    fake = FakeLibrary(abi=2)
    with pytest.raises(hdr.HdrBridgeError, match="ABI mismatch"):
        _bridge(fake)
    assert not any(s == "gpu_hdr_open" for s, _ in fake.calls)


def test_missing_export_is_refused_naming_the_symbol():
    fake = FakeLibrary(missing={"gpu_hdr_get_encode_report"})
    with pytest.raises(hdr.HdrBridgeError, match="gpu_hdr_get_encode_report"):
        _bridge(fake)


def test_pixel_readback_entries_are_never_bound():
    """The wrapper must not offer a host-side image path it does not want used."""
    bound = {name for name, _, _ in hdr._BRIDGE}
    assert not any("readback" in name for name in bound)
    assert "gpu_hdr_plane_pitch" not in bound
    assert "gpu_hdr_plane_bytes" not in bound


# ---------------------------------------------------------------------------
# fixed configuration: paper white at open, ceilings per call, never derived
# ---------------------------------------------------------------------------

def test_paper_white_is_fixed_at_open_and_read_back():
    fake = FakeLibrary(paper_white=3.0)
    bridge = _bridge(fake, paper_white_scrgb=3.0)
    assert bridge.configured_paper_white == 3.0
    assert bridge.paper_white == 3.0
    open_args = [a for s, a in fake.calls if s == "gpu_hdr_open"][0]
    assert open_args[8] == 3.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), 0.5, "x"])
def test_paper_white_below_one_or_non_finite_is_refused(bad):
    with pytest.raises(hdr.HdrBridgeError, match="paper_white_scrgb"):
        _bridge(paper_white_scrgb=bad)


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf")])
def test_support_ceiling_must_be_finite_positive(bad):
    bridge = _bridge()
    with pytest.raises(hdr.HdrBridgeError, match="support_ceiling_scrgb"):
        bridge.decode(p010_y_ptr=1, p010_y_pitch=8, p010_uv_ptr=2,
                      p010_uv_pitch=8, color_range=1, color_primaries=9,
                      color_transfer=16,
                      color_space=hdr.AV_COLOR_SPACE_BT2020_NC,
                      support_ceiling_scrgb=bad)


@pytest.mark.parametrize("bad", [0.0, -125.0, float("nan"), float("inf")])
def test_clip_ceiling_must_be_finite_positive(bad):
    bridge = _bridge()
    with pytest.raises(hdr.HdrBridgeError, match="clip_ceiling_scrgb"):
        bridge.encode(clip_ceiling_scrgb=bad)


# ---------------------------------------------------------------------------
# tap plans: reuse the CPU reference, refuse invalid ones, never truncate
# ---------------------------------------------------------------------------

def test_plans_come_from_the_production_cpu_reference():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    down, up = bridge._downscale_plan, bridge._reconstruct_plan
    # downscale: full -> working; reconstruct: working -> full.
    assert down.vertical_src_len == 512 and down.vertical_dst_len == 256
    assert down.horizontal_src_len == 512 and down.horizontal_dst_len == 256
    assert up.vertical_src_len == 256 and up.vertical_dst_len == 512
    # Lanczos-2 of 512->256 is the 8-tap window the reference builds; Catmull-Rom
    # is the fixed 4-tap upscale. No invented resampler shows up here.
    assert down.taps == max(pipeline.lanczos2_axis_taps(512, 256),
                            pipeline.lanczos2_axis_taps(512, 256))
    assert up.taps == pipeline.CATMULL_ROM_TAPS
    assert down.struct_size == ctypes.sizeof(hdr.GpuHdrTapTable)


def test_tap_arrays_are_kept_alive_by_the_owner():
    bridge = _bridge()
    keep = bridge._plan_keepalive
    assert isinstance(keep, tuple) and len(keep) == 2
    vi, vw, hi, hw = keep[0]
    assert vi.dtype == np.int32 and vw.dtype == np.float32
    # The struct's indices point into the very arrays the owner holds; if the
    # arrays were dropped, the bridge would read freed memory on the next launch.
    assert _ptr_addr(bridge._downscale_plan.vertical_indices) == vi.ctypes.data
    assert _ptr_addr(bridge._downscale_plan.horizontal_weights) == hw.ctypes.data


def test_a_wider_plan_than_the_bridge_walks_is_refused_not_truncated(monkeypatch):
    # Force a Lanczos window that exceeds GPU_HDR_MAX_TAPS without touching the
    # reference: fabricate two AxisPlans whose max_taps is one past the cap.
    wide = pipeline.AxisPlan(src_len=512, dst_len=256, kernel="lanczos2",
        indices=np.zeros((256, hdr.GPU_HDR_MAX_TAPS + 1), np.int32),
        weights=np.zeros((256, hdr.GPU_HDR_MAX_TAPS + 1), np.float32),
        tap_counts=np.zeros(256, np.int32))
    with pytest.raises(hdr.HdrBridgeError, match="at most 24"):
        hdr.build_tap_table(wide, wide)


def test_an_index_outside_the_source_is_refused():
    def plan(indices):
        return pipeline.AxisPlan(src_len=8, dst_len=2, kernel="test",
            indices=indices, weights=np.zeros((2, 2), np.float32),
            tap_counts=np.full(2, 2, np.int32))
    bad = np.array([[0, 8], [0, 1]], np.int32)  # index 8 is outside [0, 8)
    with pytest.raises(hdr.HdrBridgeError, match="leaves"):
        hdr.build_tap_table(plan(bad), plan(np.zeros((2, 2), np.int32)))
    negative = np.array([[-1, 0], [0, 1]], np.int32)
    with pytest.raises(hdr.HdrBridgeError, match="leaves"):
        hdr.build_tap_table(plan(negative), plan(np.zeros((2, 2), np.int32)))


def test_working_extent_larger_than_full_is_refused():
    with pytest.raises(hdr.HdrBridgeError, match="must not exceed"):
        _bridge(working_width=600, working_height=256)


@pytest.mark.parametrize("bad", [0, -4])
def test_non_positive_extents_are_refused(bad):
    with pytest.raises(hdr.HdrBridgeError):
        _bridge(full_width=bad)


# ---------------------------------------------------------------------------
# descriptors: typed, correct format, borrowed context, right extents
# ---------------------------------------------------------------------------

def test_decode_downscale_proxy_reconstruct_descriptors():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    master = bridge.decode(p010_y_ptr=0x111, p010_y_pitch=1024,
        p010_uv_ptr=0x222, p010_uv_pitch=1024, color_range=1,
        color_primaries=9, color_transfer=16,
        color_space=hdr.AV_COLOR_SPACE_BT2020_NC, support_ceiling_scrgb=12.5)
    assert (master.width, master.height) == (512, 512)
    assert master.bytes_per_pixel == 8 and master.format == "rgba16f_cuda"
    assert master.device_ptr == 0x1000 and master.pitch_bytes == 8192
    assert master.cuda_context == 0x9999  # the borrowed context, as an int

    working = bridge.downscale()
    assert (working.width, working.height) == (256, 256)

    proxy, guidance = bridge.proxy()
    assert (proxy.width, proxy.height) == (256, 256)

    composite = bridge.reconstruct(
        {"d3d12_resource": 0x33, "d3d12_fence": 0x44, "fence_value": 7,
         "dxgi_format": hdr.DXGI_R16G16B16A16_FLOAT, "width": 256, "height": 256},
        residual_strength=1.0)
    assert (composite.width, composite.height) == (512, 512)
    assert composite.bytes_per_pixel == 8


def test_proxy_returns_a_separate_fp16_proxy_and_rgba8_guidance():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    proxy, guidance = bridge.proxy()
    assert proxy.format == "rgba16f_cuda" and proxy.bytes_per_pixel == 8
    assert guidance.format == "rgba8_cuda" and guidance.bytes_per_pixel == 4
    assert guidance.device_ptr == 0x4000 and guidance.pitch_bytes == 1024
    # The FP16 colour descriptor is the shape Live.process_device_* accepts; the
    # guidance descriptor is the one NVOF accepts. They are not interchangeable.
    assert proxy.as_descriptor() == {
        "format": "rgba16f_cuda", "width": 256, "height": 256,
        "cuda_context": 0x9999, "device_ptr": 0x3000, "pitch_bytes": 4096}
    assert guidance.as_descriptor()["format"] == "rgba8_cuda"


def test_reconstruct_refuses_a_non_fp16_feature18_output():
    bridge = _bridge()
    with pytest.raises(hdr.HdrBridgeError, match="R16G16B16A16_FLOAT"):
        bridge.reconstruct({"d3d12_resource": 1, "d3d12_fence": 2,
            "fence_value": 3, "dxgi_format": 28, "width": 256, "height": 256})


def test_reconstruct_refuses_a_wrong_extent_feature18_output():
    bridge = _bridge()
    with pytest.raises(hdr.HdrBridgeError, match="does not match"):
        bridge.reconstruct({"d3d12_resource": 1, "d3d12_fence": 2,
            "fence_value": 3, "dxgi_format": hdr.DXGI_R16G16B16A16_FLOAT,
            "width": 512, "height": 512})


def test_reconstruct_accepts_raw_texture_fence_pair():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    out = bridge.reconstruct(0x33, fence=0x44, fence_value=9,
                             residual_strength=0.0)
    assert out.device_ptr == 0x5000
    args = [a for s, a in fake.calls if s == "gpu_hdr_reconstruct"][0]
    assert args[4] == 0.0  # residual strength passed through, not defaulted up


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_reconstruct_refuses_bad_residual_strength(bad):
    bridge = _bridge()
    with pytest.raises(hdr.HdrBridgeError, match="residual_strength"):
        bridge.reconstruct(0x33, fence=0x44, fence_value=1,
                           residual_strength=bad)


def test_encode_returns_p010_planes_at_full_size():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    out = bridge.encode(clip_ceiling_scrgb=125.0)
    assert isinstance(out, hdr.P010Output)
    assert out.y.format == "p010_cuda" and out.y.bytes_per_pixel == 2
    assert (out.y.width, out.y.height) == (512, 512)
    assert (out.uv.width, out.uv.height) == (256, 256)  # 4:2:0 chroma
    assert out.y.device_ptr == 0x6000 and out.uv.device_ptr == 0x7000


def test_encode_chroma_bpp_is_the_interleaved_pair_not_one_component():
    """P010 is semi-planar: the chroma plane is half-size in each axis and holds
    one interleaved Cb,Cr pair per entry, so a row carries 2 bytes per component
    and therefore the same bytes per row as the luma plane. Claiming 2 bytes per
    entry would size the buffer at half of what the bridge writes."""
    bridge = _bridge(FakeLibrary())
    out = bridge.encode(clip_ceiling_scrgb=125.0)
    full_width, full_height = bridge.full_extent
    assert out.uv.bytes_per_pixel == 4
    assert "chroma pairs" in out.uv.pixel_format
    assert "Cb/Cr" in out.uv.pixel_format
    assert out.uv.height == full_height // 2
    for plane in (out.y, out.uv):
        assert plane.width * plane.bytes_per_pixel == full_width * 2


@pytest.mark.parametrize("input_range", [hdr.AV_COLOR_RANGE_LIMITED, hdr.AV_COLOR_RANGE_FULL])
def test_decode_passes_the_native_color_enums_through(input_range):
    fake = FakeLibrary()
    bridge = _bridge(fake)
    bridge.decode(p010_y_ptr=1, p010_y_pitch=8, p010_uv_ptr=2, p010_uv_pitch=8,
        color_range=input_range,
        color_primaries=hdr.AV_COLOR_PRIMARIES_BT2020,
        color_transfer=hdr.AV_COLOR_TRANSFER_ST2084,
        color_space=hdr.AV_COLOR_SPACE_BT2020_NC, support_ceiling_scrgb=12.5)
    args = [a for s, a in fake.calls if s == "gpu_hdr_decode"][0]
    assert (args[5], args[6], args[7], args[8]) == (
        input_range, hdr.AV_COLOR_PRIMARIES_BT2020,
        hdr.AV_COLOR_TRANSFER_ST2084, hdr.AV_COLOR_SPACE_BT2020_NC)


# ---------------------------------------------------------------------------
# enum binding: the numbers come from the headers, not from memory
# ---------------------------------------------------------------------------

_NATIVE_SOURCE = (pathlib.Path(__file__).resolve().parents[2]
                  / "native" / "gpu_hdr_scrgb_bridge.cpp")

# The installed FFmpeg this build links against; FFMPEG_INCLUDE_DIR overrides it
# for a machine whose SDK lives elsewhere.
_PIXFMT_CANDIDATES = [
    os.environ.get("FFMPEG_INCLUDE_DIR"),
    str(pathlib.Path(os.environ["FFMPEG_SDK_DIR"]) / "include/libavutil/pixfmt.h")
    if os.environ.get("FFMPEG_SDK_DIR") else None,
]


def _constexpr_int(source_text, name):
    match = re.search(rf"constexpr int32_t {name} = (-?\d+);", source_text)
    assert match, f"native source no longer declares constexpr {name}"
    return int(match.group(1))


def _enum_value(header_text, enumerator):
    match = re.search(rf"^\s*{enumerator}\s*=\s*(\d+),", header_text,
                      re.MULTILINE)
    assert match, f"pixfmt.h no longer declares {enumerator} explicitly"
    return int(match.group(1))


def test_color_enums_match_the_bridge_constants_they_are_checked_against():
    """The wrapper must send exactly what gpu_hdr_decode compares against, or the
    bridge refuses a valid HDR10 source. Read off the native source rather than
    restated here, so the pair cannot drift apart silently."""
    native = _NATIVE_SOURCE.read_text(encoding="utf-8")
    assert hdr.AV_COLOR_RANGE_LIMITED == _constexpr_int(native, "kRangeLimited")
    assert hdr.AV_COLOR_PRIMARIES_BT2020 == _constexpr_int(
        native, "kPrimariesBt2020")
    assert hdr.AV_COLOR_TRANSFER_ST2084 == _constexpr_int(native, "kTransferPq")
    assert hdr.AV_COLOR_SPACE_BT2020_NC == _constexpr_int(
        native, "kSpaceBt2020Ncl")


def test_bt2020_ncl_is_the_libavutil_value_not_the_constant_luminance_one():
    """AVCOL_SPC_BT2020_NCL is 9 and 10 is AVCOL_SPC_BT2020_CL. A CL source would
    reach the NCL-only decode maths; the header, not a remembered number, decides
    what this wrapper sends."""
    header = next((pathlib.Path(p) for p in _PIXFMT_CANDIDATES
                   if p and pathlib.Path(p).is_file()), None)
    if header is None:
        pytest.skip("installed libavutil/pixfmt.h not on this machine")
    text = header.read_text(encoding="utf-8", errors="replace")
    ncl, cl = _enum_value(text, "AVCOL_SPC_BT2020_NCL"), _enum_value(
        text, "AVCOL_SPC_BT2020_CL")
    assert ncl != cl
    assert hdr.AV_COLOR_SPACE_BT2020_NC == ncl
    assert hdr.AV_COLOR_SPACE_BT2020_NC != cl
    # The bridge refuses the constant-luminance space it has no maths for.
    native = _NATIVE_SOURCE.read_text(encoding="utf-8")
    assert _constexpr_int(native, "kSpaceBt2020Ncl") == ncl


def test_dxgi_fp16_format_keeps_its_own_ten():
    """The NCL fix is a colour-space number; DXGI_FORMAT_R16G16B16A16_FLOAT is a
    different enum that also happens to be 10."""
    assert hdr.DXGI_R16G16B16A16_FLOAT == 10
    assert hdr.AV_COLOR_SPACE_BT2020_NC != hdr.DXGI_R16G16B16A16_FLOAT


# ---------------------------------------------------------------------------
# failure handling: propagate the native message, clean up on a failed open
# ---------------------------------------------------------------------------

def test_stage_failure_propagates_the_native_message():
    fake = FakeLibrary(fail_symbol="gpu_hdr_proxy",
                       last_error=b"proxy ceiling violated")
    bridge = _bridge(fake)
    with pytest.raises(hdr.HdrBridgeError, match="proxy ceiling violated"):
        bridge.proxy()


def test_open_failure_raises_with_the_bridge_message():
    fake = FakeLibrary(open_code=7, last_error=b"half grid does not bracket "
                                               b"paper white")
    with pytest.raises(hdr.HdrBridgeError, match="gpu_hdr_open failed .*7"):
        _bridge(fake)


def test_failed_open_closes_a_partially_created_bridge():
    fake = FakeLibrary(open_code=9, set_handle_on_failed_open=True)
    with pytest.raises(hdr.HdrBridgeError):
        _bridge(fake)
    assert fake.close_count == 1


def test_successful_open_with_a_null_handle_is_refused_not_adopted():
    """rc 0 is not proof of a usable bridge: a null out-slot means there is no
    handle to drive, and pretending otherwise would pass null to every stage."""
    fake = FakeLibrary(handle=0)
    with pytest.raises(hdr.HdrBridgeError, match="without a bridge handle"):
        _bridge(fake)
    assert fake.close_count == 0  # nothing was acquired, so nothing to close
    assert not any(s == "gpu_hdr_paper_white" for s, _ in fake.calls)


@pytest.mark.parametrize("reported", [float("nan"), float("inf"), 2.5])
def test_a_changed_or_non_finite_effective_paper_white_refuses_the_open(
        reported):
    """The bridge reports the paper white it applies; if that is not what was
    asked for, the chain would run on a different anchor than every number the
    wrapper returns, so the open is refused and the handle handed back."""
    fake = FakeLibrary(paper_white=reported)
    with pytest.raises(hdr.HdrBridgeError, match="paper white"):
        _bridge(fake, paper_white_scrgb=3.0)
    assert fake.close_count == 1
    # The requested value is what went into gpu_hdr_open, unmutated.
    open_args = [a for s, a in fake.calls if s == "gpu_hdr_open"][0]
    assert open_args[8] == 3.0


def test_a_non_numeric_paper_white_report_refuses_the_open():
    fake = FakeLibrary(paper_white="x")
    with pytest.raises(hdr.HdrBridgeError, match="did not report a number"):
        _bridge(fake)
    assert fake.close_count == 1


def test_report_getter_failure_propagates():
    fake = FakeLibrary(fail_symbol="gpu_hdr_get_source_report",
                       last_error=b"no decoded frame yet")
    bridge = _bridge(fake)
    with pytest.raises(hdr.HdrBridgeError, match="no decoded frame yet"):
        bridge.source_report()


# ---------------------------------------------------------------------------
# owner lifetime: idempotent close, no use after close, exception cleanup
# ---------------------------------------------------------------------------

def test_close_is_idempotent_and_frees_the_handle_once():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    assert bridge.is_alive()
    bridge.close()
    bridge.close()
    assert fake.close_count == 1
    assert not bridge.is_alive()


def test_use_after_close_is_refused():
    bridge = _bridge()
    bridge.close()
    with pytest.raises(hdr.HdrBridgeError, match="closed"):
        bridge.downscale()


def test_context_manager_closes_on_success_and_on_exception():
    fake = FakeLibrary()
    with _bridge(fake) as bridge:
        assert bridge.is_alive()
    assert fake.close_count == 1
    assert not bridge.is_alive()

    fake2 = FakeLibrary()
    with pytest.raises(RuntimeError, match="boom"):
        with _bridge(fake2):
            raise RuntimeError("boom")
    assert fake2.close_count == 1


def test_stage_calls_do_not_run_native_when_closed():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    bridge.close()
    before = len(fake.calls)
    for op in (lambda: bridge.downscale(),
               lambda: bridge.proxy(),
               lambda: bridge.encode(clip_ceiling_scrgb=125.0),
               lambda: bridge.transfer_counters()):
        with pytest.raises(hdr.HdrBridgeError):
            op()
    # No native entry was reached after close: only _require_open ran.
    assert len(fake.calls) == before


def test_cross_thread_use_is_refused():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    errors = []

    def worker():
        try:
            bridge.downscale()
        except hdr.HdrBridgeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors and "thread" in errors[0]
    bridge.close()


def test_close_from_another_thread_is_refused_before_the_native_call():
    """Closing tears down the bridge's device allocations through the CUDA context
    that is current on the owning thread, so a foreign close must not reach
    gpu_hdr_close - and a bridge that is already closed must stay a no-op."""
    fake = FakeLibrary()
    bridge = _bridge(fake)
    errors = []

    def worker():
        try:
            bridge.close()
        except hdr.HdrBridgeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert errors and "thread" in errors[0]
    assert fake.close_count == 0 and bridge.is_alive()

    bridge.close()
    assert fake.close_count == 1 and not bridge.is_alive()

    other = threading.Thread(target=bridge.close)
    other.start()
    other.join()
    assert fake.close_count == 1


def test_borrowed_device_and_context_are_not_destroyed():
    """Closing the bridge must never call anything that frees the device or the
    context it borrowed - those belong to the caller."""
    fake = FakeLibrary()
    bridge = _bridge(fake)
    bridge.close()
    symbols = {s for s, _ in fake.calls}
    assert "gpu_hdr_close" in symbols
    # The wrapper has no notion of a device/context free at all.
    assert not any("free" in s or "release_device" in s for s in symbols)


# ---------------------------------------------------------------------------
# the multiscale foundation: an optional group, probed by name and version
# ---------------------------------------------------------------------------

_NATIVE_HEADER = (pathlib.Path(__file__).resolve().parents[2]
                  / "native" / "gpu_hdr_scrgb_bridge.h")

# The only scalar types the bridge's counters are built from; a new one has to be
# taught here, so an unexpected field type fails the mirror test loudly.
_C_UINT_SIZES = {"uint32_t": 4, "uint64_t": 8}


def _header_struct_fields(text, struct_name):
    """[(field, byte size)] of one header typedef, in declaration order, with the
    comments stripped. The header, not this file, is the ABI."""
    match = re.search(rf"typedef struct {struct_name}\s*\{{(.*?)\}}\s*"
                      rf"{struct_name};", text, re.DOTALL)
    assert match, f"the header no longer declares typedef {struct_name}"
    body = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.DOTALL)
    fields = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        declared = re.fullmatch(r"(uint32_t|uint64_t)\s+(\w+);", line)
        assert declared, f"header field {line!r} is not a plain uint32/uint64"
        fields.append((declared.group(2), _C_UINT_SIZES[declared.group(1)]))
    return fields


def _natural_size(fields):
    """The x64 size of a field list laid out with natural alignment."""
    offset = alignment = 0
    for _, size in fields:
        alignment = max(alignment, size)
        offset = (offset + size - 1) // size * size + size
    return (offset + alignment - 1) // alignment * alignment


def _function_body(native, entry):
    """The code of one extern "C" entry in the .cpp, up to the brace that closes
    it, with every comment removed. Enough to say what a stage actually calls,
    without a parser.

    Comments go because prose is not behaviour: an entry that only *mentions* a
    resampler in a sentence must not pass an assertion that it runs no resampler,
    and the same sentence must not fail one. What is checked is the code."""
    start = native.index(f"int32_t {entry}(")
    body = native[start:native.index("\n}\n", start)]
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", body)


def _master_plane(**overrides):
    """The descriptor an upstream decode hands over for this bridge's full extent:
    512x512 RGBA16F, pitch 8192, in the bridge's borrowed context 0x9999."""
    plane = {"format": hdr.FORMAT_RGBA16F_CUDA,
             "pixel_format": "R16G16B16A16_FLOAT (scRGB)",
             "width": 512, "height": 512, "cuda_context": 0x9999,
             "device_ptr": 0x77000, "pitch_bytes": 8192, "bytes_per_pixel": 8}
    plane.update(overrides)
    return plane


def test_multiscale_counters_mirror_is_the_headers_field_for_field():
    fields = _header_struct_fields(
        _NATIVE_HEADER.read_text(encoding="utf-8"), "GpuHdrMultiscaleCounters")
    assert [name for name, _ in fields] == [
        name for name, _ in hdr.GpuHdrMultiscaleCounters._fields_]
    assert ctypes.sizeof(hdr.GpuHdrMultiscaleCounters) == _natural_size(fields)


def test_the_existing_transfer_counters_layout_did_not_change():
    """The multiscale traffic went into a struct of its own precisely because
    consumers read this layout; a field added here is a corrupt report there."""
    fields = _header_struct_fields(
        _NATIVE_HEADER.read_text(encoding="utf-8"), "GpuHdrTransferCounters")
    assert [name for name, _ in fields] == [
        name for name, _ in hdr.GpuHdrTransferCounters._fields_]
    assert _natural_size(fields) == EXPECTED_SIZES["GpuHdrTransferCounters"]


def test_python_binds_exactly_the_multiscale_entries_the_header_declares():
    text = _NATIVE_HEADER.read_text(encoding="utf-8")
    section = text.split("--- Multiscale foundation", 1)
    assert len(section) == 2, "the header has no multiscale section"
    declared = re.findall(r"GPU_HDR_API\s+\w+\s+(gpu_hdr_\w+)\s*\(",
                          section[1].split("GPU_HDR_API void gpu_hdr_close")[0])
    assert declared == list(hdr.MULTISCALE_SYMBOLS)


def test_the_group_has_its_own_version_and_abi_1_is_still_abi_1():
    text = _NATIVE_HEADER.read_text(encoding="utf-8")
    multiscale = re.search(
        r"#define GPU_HDR_MULTISCALE_ABI_VERSION\s+(\d+)u", text)
    abi = re.search(r"#define GPU_HDR_ABI_VERSION\s+(\d+)u", text)
    assert multiscale, "the multiscale group lost its own version"
    assert int(multiscale.group(1)) == hdr.GPU_HDR_MULTISCALE_ABI_VERSION
    assert int(abi.group(1)) == hdr.GPU_HDR_ABI_VERSION


def test_new_group_is_not_part_of_the_required_abi_set():
    """A bridge that speaks ABI 1 without these entries must still open, so nothing
    the load gate insists on may name one of them."""
    required = {name for name, _, _ in hdr._BRIDGE}
    assert required == set(hdr._REQUIRED_EXPORTS)
    assert not required & set(hdr.MULTISCALE_SYMBOLS)


def test_probe_names_the_missing_symbols_instead_of_guessing_from_abi():
    fake = FakeLibrary(missing=hdr.MULTISCALE_SYMBOLS)
    capability = hdr.probe_multiscale(fake)
    assert capability["supported"] is False
    assert capability["missing"] == list(hdr.MULTISCALE_SYMBOLS)
    assert "does not export" in capability["reason"]
    # Absence is answered without reaching a native call at all.
    assert fake.calls == []


def test_a_single_layer_bridge_without_the_group_still_runs_the_whole_chain():
    fake = FakeLibrary(missing=hdr.MULTISCALE_SYMBOLS)
    bridge = _bridge(fake)
    assert bridge.has_multiscale is False
    bridge.decode(p010_y_ptr=1, p010_y_pitch=1024, p010_uv_ptr=2,
                  p010_uv_pitch=1024, color_range=1, color_primaries=9,
                  color_transfer=16, color_space=hdr.AV_COLOR_SPACE_BT2020_NC,
                  support_ceiling_scrgb=12.5)
    bridge.downscale()
    bridge.proxy()
    bridge.reconstruct({"d3d12_resource": 0x33, "d3d12_fence": 0x44,
                        "fence_value": 7,
                        "dxgi_format": hdr.DXGI_R16G16B16A16_FLOAT,
                        "width": 256, "height": 256}, residual_strength=1.0)
    bridge.encode(clip_ceiling_scrgb=125.0)
    assert bridge.transfer_counters()["process_count"] == 3
    for op in (lambda: bridge.import_master(_master_plane()),
               lambda: bridge.observe(0x33, fence=0x44, fence_value=1),
               lambda: bridge.set_composite(0x33, fence=0x44, fence_value=1),
               lambda: bridge.multiscale_counters()):
        with pytest.raises(hdr.HdrBridgeError, match="multiscale"):
            op()


def test_a_partial_group_is_reported_missing_not_driven_half_way():
    fake = FakeLibrary(missing={"gpu_hdr_observe"})
    bridge = _bridge(fake)
    assert bridge.has_multiscale is False
    with pytest.raises(hdr.HdrBridgeError, match="gpu_hdr_observe"):
        bridge.observe(0x33, fence=0x44, fence_value=1)
    assert not any(s == "gpu_hdr_observe" for s, _ in fake.calls)


@pytest.mark.parametrize("reported", [0, 2, -1])
def test_a_multiscale_version_this_wrapper_does_not_speak_is_refused(reported):
    fake = FakeLibrary(multiscale_abi=reported)
    bridge = _bridge(fake)
    assert bridge.has_multiscale is False
    assert str(reported) in bridge.multiscale_capability["reason"]
    with pytest.raises(hdr.HdrBridgeError, match="reports version"):
        bridge.observe(0x33, fence=0x44, fence_value=1)


def test_the_capability_probe_is_answered_once_per_bridge():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    assert bridge.has_multiscale is True
    bridge.multiscale_capability
    bridge.has_multiscale
    asked = [1 for s, _ in fake.calls if s == "gpu_hdr_multiscale_abi_version"]
    assert len(asked) == 1


# ---------------------------------------------------------------------------
# import_master: a borrowed pristine FP16 master, copied and nothing else
# ---------------------------------------------------------------------------

def test_import_master_copies_the_declared_plane_into_this_bridges_master():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    plane = hdr.DevicePlane(
        fmt=hdr.FORMAT_RGBA16F_CUDA, pixel_format="R16G16B16A16_FLOAT (scRGB)",
        width=512, height=512, cuda_context=0x9999, device_ptr=0x77000,
        pitch_bytes=8192, bytes_per_pixel=8, plane=hdr.PLANE_MASTER_FULL)
    out = bridge.import_master(plane)
    args = [a for s, a in fake.calls if s == "gpu_hdr_import_master"][0]
    assert args[1].value == 0x77000  # the borrowed pointer, unaltered
    assert (args[2], args[3], args[4]) == (512, 512, 8192)
    assert out.plane == hdr.PLANE_MASTER_FULL
    assert (out.width, out.height) == (512, 512)
    assert out.bytes_per_pixel == 8 and out.format == "rgba16f_cuda"
    assert out.cuda_context == 0x9999
    # The shape a downstream FP16 consumer already takes.
    assert out.as_descriptor() == {"format": "rgba16f_cuda", "width": 512,
                                   "height": 512, "cuda_context": 0x9999,
                                   "device_ptr": 0x1000, "pitch_bytes": 8192}


@pytest.mark.parametrize("broken, reason", [
    ({"format": hdr.FORMAT_RGBA8_CUDA}, "must be rgba16f_cuda"),
    ({"format": hdr.FORMAT_P010_CUDA}, "must be rgba16f_cuda"),
    ({"pixel_format": "R8G8B8A8_UNORM (motion-only)"}, "not 16-bit float"),
    ({"bytes_per_pixel": 4}, "bytes per sample"),
    ({"device_ptr": 0}, "no device pointer"),
    ({"width": 256}, "does not match the full extent"),
    ({"height": 258}, "does not match the full extent"),
    ({"width": None}, "must declare its extent"),
    ({"pitch_bytes": 4088}, "under the 4096 bytes"),
    ({"pitch_bytes": None}, "must declare its row pitch"),
    ({"pitch_bytes": 4100}, "not a multiple of 16"),
    ({"cuda_context": 0x1234}, "lives in CUDA context"),
])
def test_import_master_refuses_a_plane_that_is_not_a_pristine_fp16_master(
        broken, reason):
    """Refused before the native entry, from the declaration alone: a bare address
    cannot be checked for format, so a lie about it must not reach a copy that would
    store 8-bit or integer samples as HDR master colour."""
    fake = FakeLibrary()
    bridge = _bridge(fake)
    with pytest.raises(hdr.HdrBridgeError, match=reason):
        bridge.import_master(_master_plane(**broken))
    assert not any(s == "gpu_hdr_import_master" for s, _ in fake.calls)


def test_import_master_propagates_the_bridges_own_refusal():
    fake = FakeLibrary(fail_symbol="gpu_hdr_import_master",
                       last_error=b"the imported master is 256 x 256")
    bridge = _bridge(fake)
    with pytest.raises(hdr.HdrBridgeError, match="imported master is 256"):
        bridge.import_master(_master_plane(width=512))


# ---------------------------------------------------------------------------
# observe: the signed working residual, with no composite
# ---------------------------------------------------------------------------

def test_observe_returns_the_signed_working_residual_plane():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    residual = bridge.observe({"d3d12_resource": 0x33, "d3d12_fence": 0x44,
                               "fence_value": 7,
                               "dxgi_format": hdr.DXGI_R16G16B16A16_FLOAT,
                               "width": 256, "height": 256})
    args = [a for s, a in fake.calls if s == "gpu_hdr_observe"][0]
    assert args[1].value == 0x33 and args[2].value == 0x44
    assert args[3].value == 7
    # Six arguments and no residual strength: the observation is the raw signed
    # residual, and weighting it is the fused decision.
    assert len(args) == 6
    assert (residual.width, residual.height) == (256, 256)
    assert residual.plane == hdr.PLANE_WORKING_RESIDUAL
    assert residual.format == hdr.FORMAT_RGBA16F_CUDA
    assert residual.bytes_per_pixel == 8
    assert residual.device_ptr == 0x9000 and residual.pitch_bytes == 4096
    # The units are stated by the descriptor rather than assumed by its reader.
    assert "signed" in residual.pixel_format
    assert "alpha 0.0" in residual.pixel_format


def test_observe_is_not_implemented_by_running_the_old_reconstruction():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    bridge.observe(0x33, fence=0x44, fence_value=5)
    symbols = [s for s, _ in fake.calls]
    assert "gpu_hdr_observe" in symbols
    assert "gpu_hdr_reconstruct" not in symbols


def test_observe_refuses_an_8bit_or_wrong_extent_feature18_output():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    for descriptor in (
            {"d3d12_resource": 1, "d3d12_fence": 2, "fence_value": 3,
             "dxgi_format": 28, "width": 256, "height": 256},
            {"d3d12_resource": 1, "d3d12_fence": 2, "fence_value": 3,
             "dxgi_format": hdr.DXGI_R16G16B16A16_FLOAT,
             "width": 512, "height": 512}):
        with pytest.raises(hdr.HdrBridgeError):
            bridge.observe(descriptor)
    assert not any(s == "gpu_hdr_observe" for s, _ in fake.calls)


# ---------------------------------------------------------------------------
# set_composite: a full FP16 composite in, for the existing encode to read
# ---------------------------------------------------------------------------

def test_set_composite_imports_the_full_extent_texture():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    out = bridge.set_composite({"d3d12_resource": 0x55, "d3d12_fence": 0x66,
                                "fence_value": 11,
                                "dxgi_format": hdr.DXGI_R16G16B16A16_FLOAT,
                                "width": 512, "height": 512})
    args = [a for s, a in fake.calls if s == "gpu_hdr_set_composite"][0]
    assert args[1].value == 0x55 and args[2].value == 0x66
    assert args[3].value == 11
    assert (out.width, out.height) == (512, 512)
    assert out.plane == hdr.PLANE_COMPOSITED
    assert out.format == hdr.FORMAT_RGBA16F_CUDA and out.bytes_per_pixel == 8
    assert out.device_ptr == 0xA000


def test_set_composite_refuses_a_working_extent_or_8bit_composite():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    with pytest.raises(hdr.HdrBridgeError, match="full extent"):
        bridge.set_composite({"d3d12_resource": 1, "d3d12_fence": 2,
                              "fence_value": 3,
                              "dxgi_format": hdr.DXGI_R16G16B16A16_FLOAT,
                              "width": 256, "height": 256})
    with pytest.raises(hdr.HdrBridgeError, match="never reads an 8-bit colour"):
        bridge.set_composite({"d3d12_resource": 1, "d3d12_fence": 2,
                              "fence_value": 3, "dxgi_format": 28,
                              "width": 512, "height": 512})
    assert not any(s == "gpu_hdr_set_composite" for s, _ in fake.calls)


@pytest.mark.parametrize("dropped", ["d3d12_fence", "fence_value"])
def test_set_composite_refuses_a_producer_with_nothing_to_wait_on(dropped):
    """The solver writes the composite asynchronously on its own queue, so a
    composite with no fence would be read while it is still being made. The wrapper
    refuses it and the native entry refuses it too: neither assumes completion."""
    fake = FakeLibrary()
    bridge = _bridge(fake)
    descriptor = {"d3d12_resource": 1, "d3d12_fence": 2, "fence_value": 3,
                  "dxgi_format": hdr.DXGI_R16G16B16A16_FLOAT,
                  "width": 512, "height": 512}
    descriptor.pop(dropped)
    with pytest.raises(hdr.HdrBridgeError, match="fence"):
        bridge.set_composite(descriptor)
    assert not any(s == "gpu_hdr_set_composite" for s, _ in fake.calls)


def test_set_composite_refuses_a_fence_declared_null():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    with pytest.raises(hdr.HdrBridgeError, match="producer fence and value"):
        bridge.set_composite({"d3d12_resource": 1, "d3d12_fence": None,
                              "fence_value": None,
                              "dxgi_format": hdr.DXGI_R16G16B16A16_FLOAT,
                              "width": 512, "height": 512})
    assert not any(s == "gpu_hdr_set_composite" for s, _ in fake.calls)


def test_the_old_producer_fence_rule_survives_and_only_the_new_entry_tightens_it():
    """Nothing about how the single-layer chain hands over Feature 18's output
    changed here: a descriptor brings its own fence, a bare pointer still has to
    name one, and the native copy still does not *insist* on a producer fence for
    that output. Requiring one is the new composite entry's own rule, because the
    solver writes it asynchronously, and it is not retrofitted onto the old path."""
    fake = FakeLibrary()
    bridge = _bridge(fake)
    assert bridge.reconstruct({"d3d12_resource": 0x33, "d3d12_fence": 0x44,
                               "fence_value": 7}).device_ptr == 0x5000
    assert bridge.observe({"d3d12_resource": 0x33, "d3d12_fence": 0x44,
                           "fence_value": 7}).device_ptr == 0x9000
    for entry in ("reconstruct", "observe"):
        with pytest.raises(hdr.HdrBridgeError,
                           match="needs its fence and fence_value"):
            getattr(bridge, entry)(0x33)
    # The bare-pointer refusal is the wrapper's long-standing one, and it is the
    # same words for both entries: the new path did not invent a stricter host.
    native = _NATIVE_SOURCE.read_text(encoding="utf-8")
    feature18 = re.search(r"kFeature18Copy = \{(.*?)\};", native,
                          re.DOTALL).group(1)
    composite = re.search(r"kCompositeCopy = \{(.*?)\};", native,
                          re.DOTALL).group(1)
    assert feature18.rstrip().endswith("false"), (
        "the single-layer Feature 18 copy gained a fence requirement")
    assert composite.rstrip().endswith("true"), (
        "the fused composite may be read while its producer is still writing")


# ---------------------------------------------------------------------------
# multiscale counters: attribution, and no claim about the scalar boundary
# ---------------------------------------------------------------------------

def test_multiscale_counters_report_only_fields_the_abi_carries():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    counters = bridge.multiscale_counters()
    assert set(counters) == {name for name, _ in
                             hdr.GpuHdrMultiscaleCounters._fields_
                             if name != "struct_size"}
    assert counters["master_provenance"] == hdr.MASTER_FROM_IMPORT
    assert counters["pass_dispatches"] == 2
    # No field here pretends to measure the per-pass counter readback, and none
    # stands in for a picture readback the new paths never do.
    assert not any("readback" in name for name in counters)
    sent = [a for s, a in fake.calls if s == "gpu_hdr_get_multiscale_counters"][0]
    assert sent[1]._obj.struct_size == ctypes.sizeof(
        hdr.GpuHdrMultiscaleCounters)


# ---------------------------------------------------------------------------
# lifetime and the native contract the new entries keep
# ---------------------------------------------------------------------------

def test_multiscale_entries_refuse_a_closed_bridge_before_native():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    bridge.close()
    before = len(fake.calls)
    for op in (lambda: bridge.import_master(_master_plane()),
               lambda: bridge.observe(0x33, fence=0x44, fence_value=1),
               lambda: bridge.set_composite(0x33, fence=0x44, fence_value=1),
               lambda: bridge.multiscale_counters()):
        with pytest.raises(hdr.HdrBridgeError, match="closed"):
            op()
    assert len(fake.calls) == before


def test_multiscale_entries_are_refused_from_another_thread():
    fake = FakeLibrary()
    bridge = _bridge(fake)
    errors = []

    def worker():
        try:
            bridge.observe(0x33, fence=0x44, fence_value=1)
        except hdr.HdrBridgeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors and "thread" in errors[0]
    assert not any(s == "gpu_hdr_observe" for s, _ in fake.calls)
    bridge.close()


def test_observe_and_reconstruct_share_one_residual_pass():
    """The observation's whole claim is that it is the residual the single-layer
    path already computes, so one helper must pack those constants and no second
    copy of the reconstruction may exist."""
    native = _NATIVE_SOURCE.read_text(encoding="utf-8")
    assert native.count("run_pass(b, kPassResidual,") == 1
    assert native.count("r.put(float(kMinSlopePerStop));") == 1
    assert "run_residual_pass(b)" in _function_body(native, "gpu_hdr_observe")
    assert "reconstruct_picture(" in _function_body(native, "gpu_hdr_reconstruct")
    assert "run_residual_pass(b, preserve_detail)" in _function_body(native, "reconstruct_picture")


def test_observe_adds_no_composite_and_no_strength():
    body = _function_body(_NATIVE_SOURCE.read_text(encoding="utf-8"),
                          "gpu_hdr_observe")
    assert "kPassComposite" not in body
    assert "residual_strength" not in body
    assert "kPlaneResidual].mapped" in body


def test_the_single_layer_reconstruction_keeps_its_order_and_its_counts():
    body = _function_body(_NATIVE_SOURCE.read_text(encoding="utf-8"),
                          "reconstruct_picture")
    sequence = ["copy_neural(b, dlss_texture", "run_residual_pass(b, preserve_detail)",
                "run_pass(b, kPassComposite", "composited_device_write_bytes"]
    positions = [body.index(step) for step in sequence]
    assert positions == sorted(positions), "the old chain was reordered"
    # Its residual constants now live in the shared helper, not in its own body.
    assert "kMinSlopePerStop" not in body


@pytest.mark.parametrize('purpose,number', [('guidance', 1), ('decode', 2), ('encode', 3)])
def test_resource_purpose_is_additive_and_explicit(purpose, number):
    fake = FakeLibrary()
    bridge = _bridge(fake, purpose=purpose)
    opens = [(name, args) for name, args in fake.calls if 'open' in name]
    assert opens[0][0] == 'gpu_hdr_open_purpose' and opens[0][1][-2] == number
    bridge.close()


def test_old_library_still_supports_full_but_refuses_trimmed_purpose():
    fake = FakeLibrary(missing={'gpu_hdr_open_purpose'})
    bridge = _bridge(fake)
    bridge.close()
    with pytest.raises(hdr.HdrBridgeError, match='rebuild'):
        _bridge(fake, purpose='decode')


def test_motion_only_guidance_writes_one_packed_rgba8_pixel():
    native = _NATIVE_SOURCE.read_text(encoding='utf-8')
    proxy = native.split('const char* proxy_body =', 1)[1].split(')hlsl";', 1)[0]
    assert 'Guidance.Store4' not in proxy
    assert 'value | (value << 8u) | (value << 16u) | 0xff000000u' in proxy
    assert 'Guidance.Store(id.y * GuidancePitch + id.x * 4u, packed_guidance)' in proxy


def test_import_master_is_a_copy_and_nothing_else():
    """No transfer function, no clamp, no resample, no 8-bit step: the entry that
    takes an extra layer's main colour may only move bytes."""
    body = _function_body(_NATIVE_SOURCE.read_text(encoding="utf-8"),
                          "gpu_hdr_import_master")
    assert "copy_device_to_device" in body
    for stage in ("run_pass", "kPassDecode", "pq_", "P010", "clamp"):
        assert stage not in body, f"import_master does something to {stage}"
    # The borrowed plane's pitch is the source's own; the bridge's plane pitch is
    # what the copy writes into, so a padded row cannot land in the wrong column.
    assert "source_pitch" in body and "row_bytes" in body


def test_the_new_entries_keep_their_own_counters_apart_from_the_old_struct():
    native = _NATIVE_SOURCE.read_text(encoding="utf-8")
    assert native.count("GpuHdrTransferCounters") >= 3
    import_re = _function_body(native, "gpu_hdr_import_master")
    assert "multiscale.master_import_count" in import_re
    assert "source_device_copy_bytes" not in import_re, (
        "the borrowed P010 counter must not be re-used for an FP16 import")
    assert "source_device_write_bytes" in import_re
