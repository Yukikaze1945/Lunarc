"""Stage A: isolated HDR bounded-proxy A/B feasibility gate for Feature 18.

Diagnostic only. It drives the separately built FP16 probe DLL in
``build/fp16_probe/`` and never touches the production GPU Transfer path, the
packaged host, the UI or any export setting. Nothing here enables HDR.

Two arms, same extent, same DLSS settings, same zero-motion flat input:

* A - the raw linear scRGB FP16 value goes straight into Feature 18.
* B - a bounded display proxy derived from that linear value goes into Feature
  18 (``native/hdr_proxy_math.py``), while the pristine linear HDR master stays
  outside the network and is carried alongside for reconstruction.

The proxy configuration, above all ``paperWhite``, is fixed for the whole life
of an opened Feature 18 session. Three sessions are run with three explicit
constants; there is no per-frame adaptation anywhere.

The host array this script uploads is a *test rig*, not a colour path: a real
export must apply the proxy per texel on the GPU and keep the HDR master
device-resident. Running here only proves the numbers behave as predicted
before any of that is built.
"""
from __future__ import annotations

import ctypes as C
import datetime
import hashlib
import json
import os
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hdr_proxy_math as proxy
import hdr_proxy_slope_oracle as oracle

ROOT = Path(__file__).resolve().parent.parent
PROBE_DLL = ROOT / "build" / "fp16_probe" / "dlss_gpu_fp16_probe.dll"
HOST_SRC = ROOT / "native" / "dlss_gpu_host.cpp"
MATH_SRC = Path(__file__).with_name("hdr_proxy_math.py")
ORACLE_SRC = Path(__file__).with_name("hdr_proxy_slope_oracle.py")
ARTIFACT = ROOT / "build" / "fp16_probe" / "hdr_proxy_ab_artifact.json"

REVISION = "A2_corrected_slope_and_saturation"
# Hashes of the binary and the single translation unit that built it, as
# recorded by the run this revision replaces.
RECORDED_HOST_SOURCE_SHA256 = (
    "b0862a9a9828361d479b1b2acee48f918fec32cda4fca87cd2b9eac7f09f6fa6")
RECORDED_DLL_SHA256 = (
    "d934dad00f6425d5bf7485cff86f667b7383cc85157a404b260973eca8fe92b2")

# The record this revision replaces. Its SHA-256 is written into the new
# artifact so the two can never be confused for one another.
SUPERSEDED = {
    "artifact": "build/fp16_probe/hdr_proxy_ab_artifact_run4_stale_slope.json",
    "artifact_sha256":
        "c93bf1d661c290fe977727e9b3b40fa2596a66eab500f754e67ed0f7be0ba40c",
    "stdout": "scratch/hdr_proxy_ab_stdout_run4_stale_slope.json",
    "stdout_sha256":
        "f8b6cae945311daf907e17697fd97150d3ed2f5b231834d6453af4fb55bde362",
    "selfcheck": "scratch/hdr_proxy_math_selfcheck_run4_stale.json",
    "selfcheck_sha256":
        "5f03c69f36763bcb6b373fe1f181bc925558efb2b2d2e7622a5be2299d990805",
}

WIDTH, HEIGHT = 640, 360
FRAMES = 4  # index 0 is the reset call, 1..3 are successive zero-motion frames

SCALARS = (0.1, 0.5, 1.0, 1.5, 2.0, 4.0)
COLORED = (("red_highlight", (4.0, 0.5, 0.25)),
           ("mixed_above", (1.5, 2.0, 0.5)),
           ("blue_above", (0.2, 0.8, 2.0)),
           ("green_extreme", (0.5, 4.0, 1.5)),
           ("sub_and_super", (0.1, 1.0, 1.5)))

# The highlight levels the gate must keep distinguishable, above the ceiling
# that arm A is expected to collapse onto.
HIGHLIGHT_SCALARS = ("scalar_1.5", "scalar_2.0", "scalar_4.0")

# 1.0 is the starting scRGB paper white. 4.0 and 8.0 are the additional fixed
# values declared before measuring, because a 1.0 white puts the test levels
# 1.5/2.0/4.0 at or above the proxy's encoded-domain saturation point
# (1.80873 relative whites, measured against the half grid: the first half
# level that pins is 1.8095703125).
PAPER_WHITES = (1.0, 4.0, 8.0)

CENTER = (slice(HEIGHT // 4, 3 * HEIGHT // 4),
          slice(WIDTH // 4, 3 * WIDTH // 4))


class Options(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("preset", C.c_int32),
                ("style", C.c_int32), ("intensity", C.c_float),
                ("local_tone", C.c_float), ("local_structure", C.c_float),
                ("skin_structure", C.c_float), ("auto_mask", C.c_int32),
                ("ui_correction", C.c_int32), ("motion_scale_x", C.c_float),
                ("motion_scale_y", C.c_float)]


API = {
    "dlss_gpu_host_abi_version": (C.c_uint32, []),
    "dlss_gpu_host_open": (C.c_int32,
                           [C.c_uint32, C.c_uint32, C.c_wchar_p,
                            C.POINTER(Options), C.POINTER(C.c_void_p)]),
    "dlss_gpu_host_session_id": (C.c_uint64, [C.c_void_p]),
    "dlss_gpu_host_process_zero_motion": (
        C.c_int32, [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int32,
                    C.c_void_p, C.c_size_t]),
    "dlss_gpu_host_probe_readback_input": (
        C.c_int32, [C.c_void_p, C.c_void_p, C.c_size_t]),
    "dlss_gpu_host_close": (None, [C.c_void_p]),
    "dlss_gpu_host_last_error": (C.c_char_p, []),
}


def bind(lib):
    for name, (restype, argtypes) in API.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    if lib.dlss_gpu_host_abi_version() != 3:
        raise RuntimeError("unexpected host ABI version")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def surface(rgb) -> np.ndarray:
    """Flat zero-motion field: every texel is the same half RGBA quadruple."""
    out = np.empty((HEIGHT, WIDTH, 4), dtype=np.float16)
    out[..., :3] = np.asarray(rgb, dtype=np.float16)
    out[..., 3] = np.float16(1.0)
    return out


def stats(block) -> dict:
    """Per-channel median/min/max over a float32 view of half samples."""
    flat = block.reshape(-1, 3)
    return {"median": [float(np.median(flat[:, c])) for c in range(3)],
            "min": [float(flat[:, c].min()) for c in range(3)],
            "max": [float(flat[:, c].max()) for c in range(3)]}


def run_frame(lib, handle, rgb, reset) -> dict:
    src = surface(rgb)
    dst = np.empty_like(src)
    pre = np.empty_like(src)
    code = lib.dlss_gpu_host_process_zero_motion(
        handle, src.ctypes.data, src.strides[0], 1 if reset else 0,
        dst.ctypes.data, dst.strides[0])
    if code:
        raise RuntimeError(f"evaluate: {code} {lib.dlss_gpu_host_last_error()!r}")
    code = lib.dlss_gpu_host_probe_readback_input(
        handle, pre.ctypes.data, pre.strides[0])
    if code:
        raise RuntimeError(f"readback: {code} "
                           f"{lib.dlss_gpu_host_last_error()!r}")
    out = dst[..., :3].astype(np.float32)
    landed = pre[..., :3].astype(np.float32)
    frame = {
        "reset": bool(reset),
        "submitted_half": [float(x) for x in src[0, 0, :3]],
        "pre_ngx_half": [float(x) for x in pre[0, 0, :3]],
        "pre_ngx_exact": bool(np.array_equal(src, pre)),
        "pre_ngx_min": float(landed.min()),
        "pre_ngx_max": float(landed.max()),
        "output_center": stats(out[CENTER]),
        "output_finite": bool(np.isfinite(out).all()),
        "output_full_min": float(out.min()),
        "output_full_max": float(out.max()),
    }
    return frame


def median_luma(frame) -> float:
    return float(np.dot(np.array(frame["output_center"]["median"],
                                dtype=np.float32), proxy.LUMA_WEIGHTS))


def sample_summary(frames) -> dict:
    """Steady level plus the measured frame-to-frame band of one flat patch.

    The band is the noise floor of this configuration: two highlight levels
    only count as distinguished when their steady levels are further apart
    than it.
    """
    luma = [median_luma(frame) for frame in frames]
    steady = luma[-1]
    return {"steady_median_luma": steady,
            "reset_median_luma": luma[0],
            "across_frame_band": float(max(luma) - min(luma)),
            "clipped_at_ceiling": bool(steady >= 1.0 - 1e-6)}


def cluster_count(points) -> int:
    """1-D clusters of (level, band) merged when a gap does not clear the band."""
    ordered = sorted(points)
    count = 1
    for (previous, previous_band), (current, current_band) in pairwise(ordered):
        if current - previous > max(previous_band, current_band):
            count += 1
    return count


def levels_of(samples, names) -> list:
    return [(samples[name]["summary"]["steady_median_luma"],
             samples[name]["summary"]["across_frame_band"])
            for name in names if name in samples]


def reconstruct_frames(sample, frames, paper_white, input_proxy):
    """Apply both reconstructors to the real Feature 18 output of one arm."""
    master = np.asarray(sample, dtype=np.float32)
    anchor = np.asarray(input_proxy, dtype=np.float32)
    rows = {}
    for index, frame in enumerate(frames):
        neural = np.array(frame["output_center"]["median"], dtype=np.float32)
        entry = {}
        for method in (proxy.REC_LIN, proxy.REC_LUMA):
            result = proxy.reconstruct(master, anchor, neural, paper_white,
                                       method)
            hdr = result["hdr"].astype(np.float32)
            entry[method] = {
                "hdr": [float(x) for x in np.atleast_1d(hdr)],
                "hdr_min": float(np.min(hdr)), "hdr_max": float(np.max(hdr)),
                "finite": bool(np.all(np.isfinite(hdr))),
                "headroom_above_one": bool(np.max(hdr) > 1.0),
                "stops": [float(x) for x in np.atleast_1d(result["stops"])],
                "gain_cap_hit": bool(np.any(result["gain_cap_hit"])),
                "rejected_code_noise": bool(np.any(
                    result["rejected_code_noise"])),
                "matches_master": bool(np.array_equal(
                    hdr, proxy.to_half(master))),
                "deviation_from_master": [
                    float(h) - float(mm) for h, mm in zip(
                        np.atleast_1d(hdr),
                        np.atleast_1d(proxy.to_half(master)))],
            }
            # The two numbers the audited defect corrupted: the code the network
            # moved, and the measured slope it is divided by. Recorded for the
            # local-linear arm so its dispositions can be re-derived from the
            # artifact alone; the luminance arm has no per-channel code or
            # slope by construction and records its ratio instead.
            for key in ("delta_code", "slope_per_stop"):
                if key in result:
                    entry[method][key] = [
                        float(x) for x in np.atleast_1d(result[key])]
            if "ratio" in result:
                entry[method]["ratio"] = float(
                    np.atleast_1d(result["ratio"])[0])
        rows[f"frame_{index}"] = entry
    # Measured identity: feed the GPU's own pre-NGX code back as the neural
    # side, so the gate runs on data that travelled through D3D12 and NGX.
    identity = {}
    for method in (proxy.REC_LIN, proxy.REC_LUMA):
        half = proxy.to_half(master)
        hdr = proxy.reconstruct(master, anchor, anchor, paper_white,
                                method)["hdr"].astype(np.float32)
        identity[method] = {
            "max_absolute_error": float(np.max(np.abs(
                hdr - half.astype(np.float32)))),
            "bit_exact": bool(np.array_equal(hdr, half)),
        }
    rows["identity_measured_with_gpu_input_code"] = identity
    return rows


def slope_oracle_deviation(paper_white, scenes) -> float:
    """Largest disagreement between the measured slope and an independent one.

    The identity gate cannot see a wrong slope: with a zero delta, zero divided
    by any slope at all is still zero, which is precisely how the
    double-normalization defect survived a green gate. This checks the number
    a real correction is divided by, against ``hdr_proxy_slope_oracle``, which
    reimplements the proxy chain in float64 without importing the module under
    test.

    ``proxy_code`` normalizes once, so scene ``paperWhite * x`` at any session
    constant is scene ``x`` at 1.0; measuring both and comparing them catches a
    second normalization directly rather than by proxy.
    """
    worst = 0.0
    for scene in scenes:
        values = np.atleast_1d(np.asarray(scene, dtype=np.float32))
        measured = np.atleast_1d(proxy.code_slope_per_stop(
            values, paper_white)).astype(np.float64)
        expected = np.array(
            [oracle.oracle_slope_per_stop(float(v), paper_white)
             for v in values], dtype=np.float64)
        worst = max(worst, float(np.max(np.abs(measured - expected))))
        for other in PAPER_WHITES:
            scaled = np.atleast_1d(proxy.code_slope_per_stop(
                values * np.float32(other / paper_white),
                other)).astype(np.float64)
            worst = max(worst, float(np.max(np.abs(scaled - measured))))
    return worst


def reconstruction_stats(frame_rows) -> dict:
    """Slope and correction spread over one session's recorded frames."""
    slopes, stops, deviations = [], [], []
    plateau_rows = 0
    for row in frame_rows:
        found = [float(s) for s in row[proxy.REC_LIN].get(
            "slope_per_stop", [])]
        slopes.extend(found)
        plateau_rows += int(any(s == 0.0 for s in found))
        for method in (proxy.REC_LIN, proxy.REC_LUMA):
            stops.extend(float(s) for s in row[method]["stops"])
            deviations.extend(float(d) for d in row[method][
                "deviation_from_master"])
    nonzero = [s for s in slopes if s != 0.0]
    return {
        "channels_measured": len(slopes),
        "rows_with_a_pinned_channel": plateau_rows,
        "min_nonzero_slope_per_stop": min(nonzero) if nonzero else None,
        "max_slope_per_stop": max(slopes) if slopes else None,
        "max_abs_stops": max((abs(s) for s in stops), default=0.0),
        "max_abs_deviation_from_master": max(
            (abs(d) for d in deviations), default=0.0),
    }


def gate_checks(sessions) -> dict:
    """The stated stage-A gate, evaluated from the measurements only.

    Distinguishability is judged two ways and both must hold: the proxy must
    hand Feature 18 different input codes for different highlight levels (a
    collision here is information the network can never get back), and the
    steady outputs must land in different clusters once the measured
    frame-to-frame band of the same configuration is taken into account.
    """
    out = {}
    for paper in sorted(sessions):
        arm = sessions[paper]["arms"]
        raw, bounded = arm["A_raw_linear"]["samples"], \
            arm["B_bounded_proxy"]["samples"]
        bounded_frames = [rec for sample in bounded.values()
                          for rec in sample["frames"]]
        all_frames = [rec for sample in raw.values() for rec in sample["frames"]]
        frame_rows = [row for sample in bounded.values()
                      for key, row in sample["reconstruction"].items()
                      if key.startswith("frame_")]
        measured = [sample["reconstruction"]
                    ["identity_measured_with_gpu_input_code"]
                    for sample in bounded.values()]
        highlight_names = HIGHLIGHT_SCALARS + tuple(
            f"color_{name}" for name, _ in COLORED)
        raw_levels = levels_of(raw, HIGHLIGHT_SCALARS)
        proxy_levels = levels_of(bounded, HIGHLIGHT_SCALARS)
        proxy_codes = [bounded[name]["proxy_code"][0]
                       for name in HIGHLIGHT_SCALARS]
        all_codes = [sample["proxy_code"] for sample in bounded.values()]
        out[str(paper)] = {
            "proxy_input_bounded_le_one": max(
                rec["pre_ngx_max"] for rec in bounded_frames) <= 1.0,
            "pre_ngx_matches_submitted": all(rec["pre_ngx_exact"]
                                             for rec in bounded_frames),
            "all_feature18_output_finite": all(
                rec["output_finite"] for rec in all_frames + bounded_frames),
            "raw_highlight_levels_clipped_at_ceiling": all(
                raw[name]["summary"]["clipped_at_ceiling"]
                for name in HIGHLIGHT_SCALARS),
            "raw_highlight_clusters": cluster_count(raw_levels),
            "proxy_highlight_distinct_input_codes": len(set(proxy_codes)),
            "proxy_highlight_input_codes_collide": len(set(proxy_codes))
            < len(proxy_codes),
            "proxy_highlight_clusters": cluster_count(proxy_levels),
            "proxy_highlight_clipped_at_ceiling": any(
                bounded[name]["summary"]["clipped_at_ceiling"]
                for name in HIGHLIGHT_SCALARS),
            "proxy_highlight_not_single_clipped_value": (
                len(set(proxy_codes)) == len(proxy_codes)
                and cluster_count(proxy_levels) == len(proxy_codes)
                and not any(bounded[name]["summary"]["clipped_at_ceiling"]
                            for name in HIGHLIGHT_SCALARS)),
            "proxy_distinct_input_codes_all_samples": len(
                {tuple(code) for code in all_codes}),
            "proxy_all_samples_total": len(all_codes),
            "proxy_colored_clusters": cluster_count(
                levels_of(bounded, tuple(f"color_{name}" for name, _ in COLORED))),
            "proxy_highlight_and_color_level_clusters": cluster_count(
                levels_of(bounded, highlight_names)),
            "identity_reconstruction_bit_exact":
                sessions[paper]["identity"]["local_linear_stops"]["bit_exact"]
                and sessions[paper]["identity"]["luminance_ratio"]["bit_exact"],
            "measured_identity_bit_exact": all(
                row[method]["bit_exact"] for row in measured
                for method in (proxy.REC_LIN, proxy.REC_LUMA)),
            "measured_identity_max_absolute_error": max(
                row[method]["max_absolute_error"] for row in measured
                for method in (proxy.REC_LIN, proxy.REC_LUMA)),
            "reconstruction_all_finite": all(
                row[method]["finite"] for row in frame_rows
                for method in (proxy.REC_LIN, proxy.REC_LUMA)),
            "reconstruction_headroom_above_one": any(
                row[method]["headroom_above_one"] for row in frame_rows
                for method in (proxy.REC_LIN, proxy.REC_LUMA)),
            "reconstruction_max_hdr": max(
                row[method]["hdr_max"] for row in frame_rows
                for method in (proxy.REC_LIN, proxy.REC_LUMA)),
            "reconstruction_gain_cap_hits": sum(
                int(row[method]["gain_cap_hit"]) for row in frame_rows
                for method in (proxy.REC_LIN, proxy.REC_LUMA)),
            "reconstruction_rejected_code_noise": sum(
                int(row[method]["rejected_code_noise"]) for row in frame_rows
                for method in (proxy.REC_LIN, proxy.REC_LUMA)),
            "reset_and_three_continuous_frames_ran": all(
                len(sample["frames"]) == FRAMES
                and [f["reset"] for f in sample["frames"]] ==
                [True, False, False, False]
                for sample in list(raw.values()) + list(bounded.values())),
            "temporal_state_moves_output": any(
                sample["summary"]["steady_median_luma"]
                != sample["summary"]["reset_median_luma"]
                for sample in bounded.values()),
            "proxy_saturation_scene":
                sessions[paper]["proxy_saturation_scene"],
            "saturation_relative_module":
                sessions[paper]["proxy_saturation_relative"],
            "saturation_relative_oracle": oracle.saturation_relative(),
            # The saturation claim is checked against the whole half grid in
            # both directions, not asserted by a formula.
            "saturation_verified_by_enumeration": (
                sessions[paper]["grid"][
                    "levels_at_or_above_saturation_not_pinned"] == 0
                and sessions[paper]["grid"][
                    "levels_below_saturation_pinned_to_one"] == 0
                and sessions[paper]["grid"][
                    "codes_used_by_scene_above_saturation"] == 1.0),
            # A wrong slope is invisible to the identity gate and decides every
            # real correction, so the gate has to look at it directly.
            "slope_oracle_max_abs_deviation": slope_oracle_deviation(
                float(paper),
                [np.asarray(sample["master"], dtype=np.float32)
                 for sample in bounded.values()]),
            "code_slope_matches_independent_oracle": slope_oracle_deviation(
                float(paper),
                [np.asarray(sample["master"], dtype=np.float32)
                 for sample in bounded.values()]) == 0.0,
            "reconstruction_stats": reconstruction_stats(frame_rows),
            "levels": {"raw_highlight": raw_levels,
                       "proxy_highlight": proxy_levels},
        }
    return out


def numeric_self_checks() -> dict:
    """Pure-number checks the GPU arms depend on but cannot themselves see.

    Run over the whole positive finite half grid rather than only the eleven
    gate scenes, because both audited defects survived the recorded gate: the
    identity test divides a zero delta by whatever slope it is handed, and the
    saturation formula was never compared to the grid it claims to describe.
    """
    grid = oracle.enumerate_half_grid()
    rows = {}
    for paper in PAPER_WHITES:
        measured = np.atleast_1d(
            proxy.code_slope_per_stop(grid, paper)).astype(np.float64)
        expected = np.array([oracle.oracle_slope_per_stop(float(v), paper)
                             for v in grid], dtype=np.float64)
        codes = proxy.proxy_code(grid, paper).astype(np.float64)
        oracle_codes = np.array([oracle.oracle_code(float(v), paper)
                                 for v in grid], dtype=np.float64)
        rows[str(paper)] = {
            "levels": int(grid.size),
            "slope_levels_disagreeing_with_oracle": int(
                (measured != expected).sum()),
            "slope_max_abs_deviation_vs_oracle": float(np.max(np.abs(
                measured - expected))),
            "code_levels_disagreeing_with_oracle": int(
                (codes != oracle_codes).sum()),
            "code_max_abs_deviation_vs_oracle": float(np.max(np.abs(
                codes - oracle_codes))),
        }
    return {
        "half_grid": rows,
        "relative_saturation_point_module": proxy.saturation_relative(),
        "relative_saturation_point_oracle": oracle.saturation_relative(),
        "stale_linear_domain_saturation": float(
            proxy.KNEE + np.log(1.0 / (2.0 * proxy.HALF_ULP_AT_ONE))
            / proxy.SHOULDER_K),
        "identity_gate_is_blind_to_slope": True,
    }


def main() -> int:
    lib_path = str(PROBE_DLL)
    if not os.path.exists(lib_path):
        print(f"missing probe build: {lib_path}", file=sys.stderr)
        return 2
    with os.add_dll_directory(str(ROOT / "native")):
        lib = C.CDLL(lib_path)
    bind(lib)

    samples = {}
    for value in SCALARS:
        samples[f"scalar_{value}"] = (value, value, value)
    for name, rgb in COLORED:
        samples[f"color_{name}"] = rgb

    options = Options(C.sizeof(Options), 0, 0, 1, 1, 1, 0, 0, 0, 1, 1)
    sessions = {}
    identity_scenes = [np.float32(v) for v in SCALARS] + \
        [np.array(rgb, dtype=np.float32) for _, rgb in COLORED]

    for paper in PAPER_WHITES:
        handle = C.c_void_p()
        code = lib.dlss_gpu_host_open(WIDTH, HEIGHT,
                                      str(ROOT / "nvngx_dlssnr.dll"),
                                      C.byref(options), C.byref(handle))
        if code:
            raise RuntimeError(f"open: {code} "
                               f"{lib.dlss_gpu_host_last_error()!r}")
        session = {"paper_white": paper,
                   "session_id": lib.dlss_gpu_host_session_id(handle),
                   "proxy_saturation_relative": proxy.saturation_relative(),
                   "proxy_saturation_scene":
                       proxy.saturation_relative() * paper,
                   "arms": {}, "identity": proxy.identity_gate(
                       identity_scenes, paper),
                   "grid": proxy.grid_analysis(paper)}
        try:
            for arm, bounded in (("A_raw_linear", False),
                                 ("B_bounded_proxy", True)):
                arms = {}
                for name, rgb in samples.items():
                    master = np.asarray(rgb, dtype=np.float32)
                    upload = proxy.proxy_code(master, paper) if bounded \
                        else proxy.to_half(master)
                    frames = [run_frame(lib, handle, upload, index == 0)
                              for index in range(FRAMES)]
                    for frame in frames:
                        frame["session_id"] = session["session_id"]
                    row = {"master": [float(x) for x in master],
                           "proxy_code": [float(x) for x in
                                          np.atleast_1d(upload)],
                           "frames": frames,
                           "summary": sample_summary(frames)}
                    if bounded:
                        row["reconstruction"] = reconstruct_frames(
                            master, frames, paper, frames[-1]["pre_ngx_half"])
                    else:
                        # Arm A has no proxy domain, so a proxy-domain
                        # correction is undefined: its network output is the
                        # clamped raw signal itself, not a code to compare.
                        row["reconstruction"] = {}
                        row["reconstruction_note"] = "arm A has no proxy"
                    arms[name] = row
                session["arms"][arm] = {"samples": arms}
        finally:
            lib.dlss_gpu_host_close(handle)
        sessions[paper] = session

    checks = gate_checks(sessions)
    for paper, row in checks.items():
        # temporal_state_moves_output is evidence, not a criterion: a flat
        # zero-motion field can converge to one value without the temporal
        # path having failed to run.
        row["pass"] = bool(row["proxy_input_bounded_le_one"]
                           and row["pre_ngx_matches_submitted"]
                           and row["all_feature18_output_finite"]
                           and row["proxy_highlight_not_single_clipped_value"]
                           and row["identity_reconstruction_bit_exact"]
                           and row["measured_identity_bit_exact"]
                           and row["reconstruction_all_finite"]
                           and row["reset_and_three_continuous_frames_ran"]
                           and row["saturation_verified_by_enumeration"]
                           and row["slope_oracle_max_abs_deviation"] == 0.0
                           and row["reconstruction_headroom_above_one"])
    checks["pass_any_session"] = any(row["pass"] for row in checks.values())

    artifact = {
        "stage": "A_bounded_proxy_ab_feasibility_gate",
        "revision": REVISION,
        "recorded_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "not_production_color_path": True,
        "production_hdr_enabled": False,
        "performance": "NOT_CHARACTERIZED",
        "probe_dll": lib_path,
        "probe_dll_sha256": sha256(PROBE_DLL),
        "probe_dll_sha256_before_this_build":
            "3236064a34d14db3afd57c02837155db52639b988def0daf54d6f8e87e18a7a1",
        # The probe is the same binary the previous record measured: none of
        # the sources that build it changed, only the Python math did. Stated
        # as a measured comparison rather than a claim.
        "probe_source_set": {
            "native/dlss_gpu_host.cpp": sha256(HOST_SRC),
            "native/dlss_gpu_host.h": sha256(ROOT / "native"
                                             / "dlss_gpu_host.h"),
            "native/nvof_d3d12_interop.h": sha256(ROOT / "native"
                                                  / "nvof_d3d12_interop.h"),
            "native/build_dlss_gpu_fp16_probe.bat": sha256(
                ROOT / "native" / "build_dlss_gpu_fp16_probe.bat"),
        },
        "probe_rebuilt_for_this_revision": False,
        "probe_unchanged_since_recorded_build": bool(
            sha256(HOST_SRC) == RECORDED_HOST_SOURCE_SHA256
            and sha256(PROBE_DLL) == RECORDED_DLL_SHA256),
        "host_source_sha256": sha256(HOST_SRC),
        "proxy_module_sha256": sha256(MATH_SRC),
        "oracle_module_sha256": sha256(ORACLE_SRC),
        "gate_script_sha256": sha256(Path(__file__)),
        "supersedes": SUPERSEDED,
        "corrected_defects": [
            "code_slope_per_stop divided by paper_white twice: it normalized "
            "the scene and passed the normalized value to proxy_code, which "
            "normalizes again, measuring the slope of a scene paperWhite times "
            "darker than asked. The identity gate could not see it because a "
            "zero delta divides to zero at any slope.",
            "saturation_relative solved the shoulder's linear value for a "
            "half-grid step, but rounding happens after the sRGB encode. The "
            "encoded boundary 1 - 2**-12, decoded back, puts the ceiling at "
            "1.80873 relative whites instead of 1.9511.",
            "a channel whose code is pinned at the bounded ceiling reported a "
            "nonzero central-difference slope, because its lower arm can still "
            "move over the cliff; that reading measured the cliff, not a "
            "signal, and now reports zero.",
        ],
        "feature": 18,
        "color_format": "DXGI_FORMAT_R16G16B16A16_FLOAT",
        "extent": [WIDTH, HEIGHT],
        "settings": {"preset": 0, "style": 0, "intensity": 1.0,
                     "local_tone": 1.0, "local_structure": 1.0,
                     "skin_structure": 0.0, "auto_mask": 0,
                     "ui_correction": 0, "motion_scale": [1.0, 1.0],
                     "generic_dlss_exposure_keys": [1.0, 1.0],
                     "guidance": "resident zero motion texture"},
        "proxy": {"knee": proxy.KNEE, "shoulder_span": proxy.SHOULDER_SPAN,
                  "shoulder_k": proxy.SHOULDER_K,
                  "paper_white_floor": proxy.PAPER_FLOOR,
                  "per_frame_adaptation": False,
                  "max_stops_reconstruction": proxy.MAX_STOPS,
                  "min_slope_per_stop": proxy.MIN_SLOPE_PER_STOP,
                  "half_ulp_at_one": proxy.HALF_ULP_AT_ONE,
                  "slope_stop_step": proxy.SLOPE_STOP_STEP,
                  "relative_saturation_point": proxy.saturation_relative(),
                  "encoded_rounding_boundary": 1.0 - 2.0 ** -12},
        "numeric_self_checks": numeric_self_checks(),
        "api_signatures": {name: {"restype": getattr(restype, "__name__",
                                                     str(restype)),
                                  "argtypes": [getattr(a, "__name__", str(a))
                                               for a in argtypes]}
                           for name, (restype, argtypes) in API.items()},
        "sessions": {str(paper): sessions[paper] for paper in PAPER_WHITES},
        "gate": checks,
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

    print(json.dumps({"artifact": str(ARTIFACT),
                      "revision": REVISION,
                      "recorded_utc": artifact["recorded_utc"],
                      "probe_dll_sha256": artifact["probe_dll_sha256"],
                      "probe_unchanged_since_recorded_build":
                          artifact["probe_unchanged_since_recorded_build"],
                      "gate": checks}, indent=2, sort_keys=True))
    return 0 if checks["pass_any_session"] else 1


if __name__ == "__main__":
    sys.exit(main())
