# -*- coding: utf-8 -*-
"""Stage A2 gate: one synthetic HDR frame through the real typed fusion bridge.

Not a pytest test. It needs a D3D12 device and a CUDA context on the same
adapter, the freshly built probe, and the two bridges under test. It builds a
64x48 scRGB fixture whose master carries components above 1.0 and below 0.0,
runs the typed solve on the device, and judges the picture the device actually
wrote against the float32 reference in multiscale_fusion.py - a reference that
builds its own Lanczos-2 plans and its own explicit transpose, so the two agree
by computation rather than by assertion.

    python tests/unit/hdr_multiscale_fusion_a2_gate.py
    python tests/unit/hdr_multiscale_fusion_a2_gate.py --archive <fusion dll>

What it measures, in order:

* the fused store is R16G16B16A16_FLOAT at the full extent, left in COPY_SOURCE
  behind a real producer fence, and holds no Inf and no value the half grid
  cannot hold while reporting success;
* headroom above paper white and out-of-gamut negatives survive the solve;
* the device solution matches the CPU reference within the budget the two
  formats allow, and only within it;
* the objective and the data term both drop from the Catmull baseline;
* the counted out-of-range fallback still counts, still stores the pristine
  master sample, and still reports success;
* a real gpu_hdr bridge takes the fused texture behind the solver's own fence
  and its composited plane is the fused samples bit for bit;
* the A1 observation ordering refuses a proxy that belongs to a superseded
  working plane, and reports the pass count with its bytes;
* a budget that cannot converge and a source plane the grid cannot hold both
  fail outright and hand back no texture;
* the legacy SDR entry's byte composite is identical between this build and the
  archived pre-A2 build of the same bridge.

Thresholds, and where each number comes from - fixed before the run, not after:
the dominant error is the binary16 store, so a stored component gets one half ulp
at the reference value plus 1e-3. That slack is the float32 CG: its residual is
allowed to be 1e-4 of an initial norm measured on a field of magnitude at most
0.4, at residual_strength 1.0, so 1e-3 is two orders of magnitude clear of it.
The correction read back out of the store gets one half ulp of the *stored*
sample plus 2e-3, for the same reason in the other direction: it is a difference
of two halves, so the store's rounding of a bright master is in it before the
solver's slack is, and a flat absolute bound cannot hold at 9.4 scRGB where one
ulp is already 7.8e-3. The store gets ulp+1e-3,
and the adjoint pairing gets 1e-3 relative against float64 - two transposed
float32 gathers over at most 24 taps, 3072 pixels.

The limitation this gate does not remove: the transpose the device applies is
verified transitively (the device solution matches a reference built from the
explicit transpose, and the tables are checked to be the pipeline's own padded
arrays) rather than by a direct adjoint dispatch on the GPU. Performance is
NOT_CHARACTERIZED: one 64x48 frame measures correctness and nothing else.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import gpu_transfer_export as gtx           # noqa: E402
import hdr_fusion_engine as hfe             # noqa: E402
import multiscale_fusion as msf             # noqa: E402
import pipeline                             # noqa: E402

NATIVE = os.path.join(ROOT, "native")
PROBE = os.path.join(ROOT, "build", "hdr_a2",
                     "gpu_multiscale_fusion_a2_test.exe")
FUSION_DLL = os.path.join(NATIVE, "gpu_multiscale_fusion_bridge.dll")
HDR_DLL = os.path.join(NATIVE, "gpu_hdr_scrgb_bridge.dll")
ARCHIVE_FUSION = os.path.join(ROOT, "bench_out", "hdr_a2_evidence", "pre_a2",
                              "gpu_multiscale_fusion_bridge.dll")

FULL_W, FULL_H = 64, 48
LAYERS = (((32, 24), 1.0), ((16, 12), 0.5))
BETA = 0.05
STRENGTH = 1.0
RELTOL = 1e-4
MAX_ITER = 64
SUPPORT_CEILING_SCRGB = 12.5

STORE_SLACK = 1.0e-3
CORRECTION_SLACK = 2.0e-3
ADJOINT_SLACK = 1.0e-3
ORDERING_PASS_DISPATCHES = 7   # 2 downscale + proxy + proxy + 2 downscale + proxy
COUNTER_BYTES_PER_PASS = 64 * 4     # kCounterSlots * sizeof(uint32_t)
# The SDR store's format, from the bridge's own report. hdr_gpu_engine publishes
# only the scRGB one, because only the scRGB one is a contract there.
DXGI_R8G8B8A8_UNORM = 28


class GateFailure(AssertionError):
    """A measured number outside the budget this gate declared before the run."""


def check(ok, what, detail=""):
    print(f"[{' ok ' if ok else 'FAIL'}] {what}"
          + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        raise GateFailure(what)
    return True


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def half_ulp(value):
    """The spacing of the binary16 grid at |value|: the store's own resolution."""
    magnitude = abs(float(value))
    if magnitude == 0.0 or not math.isfinite(magnitude):
        return 2.0 ** -24
    exponent = math.floor(math.log2(magnitude))
    if exponent < -14:
        return 2.0 ** -24
    return 2.0 ** (exponent - 10)


# --- the fixture -----------------------------------------------------------

def _table_axis(view, prefix):
    """One axis of a bridge tap table, read back from the pointers it carries.

    The fixture writes exactly these bytes, so the table the kernel walks and the
    table the reference is checked against are one array, not two builds of the
    same idea.
    """
    taps = int(view.taps)
    dst_len = int(getattr(view, prefix + "_dst_len"))
    count = dst_len * taps
    arrays = []
    for suffix, ctype, dtype in (("_indices", ctypes.c_int32, np.int32),
                                 ("_weights", ctypes.c_float, np.float32)):
        address = ctypes.cast(getattr(view, prefix + suffix),
                              ctypes.c_void_p).value
        if not address:
            raise GateFailure(f"the tap table carries no {prefix}{suffix} array")
        buffer = (ctype * count).from_address(address)
        arrays.append(np.frombuffer(buffer, dtype=dtype, count=count)
                      .reshape(dst_len, taps).copy())
    return (int(getattr(view, prefix + "_src_len")), dst_len, taps,
            arrays[0], arrays[1])


def _padded(plan, columns):
    """A pipeline.AxisPlan padded the way gpu_transfer_export.tap_tables pads it."""
    indices = np.zeros((plan.dst_len, columns), np.int32)
    weights = np.zeros((plan.dst_len, columns), np.float32)
    indices[:, :plan.max_taps] = plan.indices
    weights[:, :plan.max_taps] = plan.weights
    return indices, weights


def _write_axis(handle, axis):
    _, dst_len, taps, indices, weights = axis
    handle.write(struct.pack("<III", axis[0], dst_len, taps))
    handle.write(indices.astype("<i4").tobytes(order="C"))
    handle.write(weights.astype("<f4").tobytes(order="C"))


def _duplicate_taps(indices):
    """Repeated clamp-to-edge indices beyond the first use of each sample."""
    counts = np.bincount(indices.ravel())
    return int(np.sum(counts[counts > 1]) - np.count_nonzero(counts > 1))


def build_fixture(fixture_dir):
    """The synthetic frame, its signed residuals, and the pipeline's own plans."""
    os.makedirs(fixture_dir, exist_ok=True)
    yy, xx = np.mgrid[0:FULL_H, 0:FULL_W].astype(np.float32)
    u, v = xx / float(FULL_W - 1), yy / float(FULL_H - 1)

    # The correction the solver is asked to recover: signed, smooth, and small
    # enough that master + correction stays inside the declared support ceiling.
    r_true = np.empty((FULL_H, FULL_W, 3), np.float32)
    for channel in range(3):
        r_true[..., channel] = (
            0.18 * np.sin((xx + 7.0 * channel) * 0.31)
            * np.cos((yy - 3.0 * channel) * 0.22)
            + 0.07 * np.sin(xx * 0.13 + yy * 0.11 + channel))

    master = np.empty((FULL_H, FULL_W, 3), np.float32)
    master[..., 0] = 0.30 + 1.35 * u - 0.55 * v           # headroom above 1.0
    master[..., 1] = 0.20 - 0.80 * u + 0.10 * v           # out of gamut
    # A bright ramp, capped at 2.2 rather than run to the top of the declared
    # ceiling. The reason is a measurement, not a colour choice: the correction
    # this fixture asks for is ~0.25 and the store holds master + correction, so
    # reading r back out of an FP16 picture subtracts two numbers that were each
    # rounded at the *master's* spacing. At 9.4 scRGB that spacing is 7.8e-3 - four
    # percent of the correction itself - and the objective computed from the store
    # readback then measures the store rather than the solve. At 2.2 the spacing is
    # 9.8e-4 and the drop the solver produced is visible through the picture, which
    # is what this gate is for. The headroom and the out-of-gamut negatives are
    # both still there, and the counted out-of-range fallback drives the same
    # samples far past the grid by its own choice of strength.
    master[..., 2] = -0.20 + 2.40 * (0.45 * u + 0.55 * v)
    master16 = master.astype(np.float16).astype(np.float32)
    check(float(np.max(np.abs(master16))) <= SUPPORT_CEILING_SCRGB,
          "the master stays inside the declared support ceiling",
          f"max abs {float(np.max(np.abs(master16))):.6g}")
    check(float(np.max(master16)) > 1.0 and float(np.min(master16)) < 0.0,
          "the master carries headroom above paper white and negatives",
          f"max {float(np.max(master16)):.6g} min {float(np.min(master16)):.6g}")

    down_plans, up_plans, tables = [], [], []
    for width, height in (layer[0] for layer in LAYERS):
        plans = gtx.TapPlans(FULL_W, FULL_H, width, height)
        vertical = pipeline._build_lanczos2_axis_plan(FULL_H, height)
        horizontal = pipeline._build_lanczos2_axis_plan(FULL_W, width)
        down_plans.append((vertical, horizontal))
        table = (_table_axis(plans.working_color, "vertical"),
                 _table_axis(plans.working_color, "horizontal"))
        for name, axis, plan in zip(("vertical", "horizontal"), table,
                                    (vertical, horizontal)):
            expect = _padded(plan, axis[2])
            check(np.array_equal(axis[3], expect[0])
                  and np.array_equal(axis[4], expect[1]),
                  f"layer {width}x{height}'s {name} table is the pipeline plan "
                  f"padded, not a second build of it")
        tables.append(table)
        if width == LAYERS[0][0][0] and height == LAYERS[0][0][1]:
            up_plans.append((pipeline._build_catmull_rom_axis_plan(height, FULL_H),
                             pipeline._build_catmull_rom_axis_plan(width, FULL_W)))
            baseline_table = (_table_axis(plans.upscale, "vertical"),
                              _table_axis(plans.upscale, "horizontal"))
            for name, axis, plan in zip(("vertical", "horizontal"),
                                        baseline_table, up_plans[-1]):
                expect = _padded(plan, axis[2])
                check(np.array_equal(axis[3], expect[0])
                      and np.array_equal(axis[4], expect[1]),
                      f"the Catmull baseline {name} table is the pipeline plan "
                      f"padded, not a second build of it")

    def to_low(residual, plans_pair, width, height):
        vertical, horizontal = plans_pair
        return pipeline._apply_axis_plan(
            pipeline._apply_axis_plan(residual, vertical, axis=0),
            horizontal, axis=1)

    observations = []
    for index, ((width, height), _) in enumerate(LAYERS):
        low = to_low(r_true, down_plans[index], width, height)
        observations.append(low.astype(np.float16).astype(np.float32))

    # The baseline the solver starts from: the Catmull upscale of layer 0's own
    # residual, which is what kCatmull reads out of observed[0].
    vertical, horizontal = up_plans[0]
    baseline_full = pipeline._apply_axis_plan(
        pipeline._apply_axis_plan(observations[0], vertical, axis=0),
        horizontal, axis=1).astype(np.float32)

    with open(os.path.join(fixture_dir, "plans.bin"), "wb") as handle:
        handle.write(struct.pack("<I", len(LAYERS)))
        for index, ((width, height), weight) in enumerate(LAYERS):
            handle.write(struct.pack("<II", width, height))
            handle.write(struct.pack("<f", float(np.float32(weight))))
            _write_axis(handle, tables[index][0])
            _write_axis(handle, tables[index][1])
        _write_axis(handle, baseline_table[0])
        _write_axis(handle, baseline_table[1])

    def write_halves(name, values, width, height):
        rgba = np.zeros((height, width, 4), np.float16)
        rgba[..., :3] = values.astype(np.float16)
        rgba[..., 3] = np.float16(1.0)
        rgba.tofile(os.path.join(fixture_dir, name))

    write_halves("master_half.bin", master16, FULL_W, FULL_H)
    for index, residual in enumerate(observations):
        height, width = residual.shape[:2]
        write_halves(f"residual{index}_half.bin", residual, width, height)

    # The legacy SDR control needs the same extents and the same plans. Its
    # contract is not the HDR one: the source plane is RGBA8 bytes and each
    # observation is an R32G32B32A32_FLOAT *signed residual in byte units* -- the
    # entry refuses a picture where it wants that, so the fixture writes exactly
    # what gpu_fusion_process borrows. The numbers are the same scene the halves
    # above carry, scaled into the 0..255 domain; what is compared is one build's
    # bytes against another's, not a colour claim.
    sdr_master = np.empty((FULL_H, FULL_W, 4), np.uint8)
    sdr_master[..., 0] = (40 + 170 * u).astype(np.uint8)
    sdr_master[..., 1] = (60 + 120 * v).astype(np.uint8)
    sdr_master[..., 2] = (30 + 200 * (u * v)).astype(np.uint8)
    sdr_master[..., 3] = 255
    sdr_master.tofile(os.path.join(fixture_dir, "sdr_master_rgba8.bin"))
    for index, residual in enumerate(observations):
        height, width = residual.shape[:2]
        signed = np.zeros((height, width, 4), np.float32)
        signed[..., :3] = (residual.astype(np.float32) * 255.0)
        signed.tofile(os.path.join(fixture_dir, f"residual{index}_f32.bin"))

    with open(os.path.join(fixture_dir, "manifest.txt"), "w",
              encoding="utf-8", newline="\n") as handle:
        for key, value in (("full_width", FULL_W), ("full_height", FULL_H),
                           ("beta", BETA), ("residual_strength", STRENGTH),
                           ("relative_tolerance", RELTOL),
                           ("max_iterations", MAX_ITER),
                           ("support_ceiling_scrgb", SUPPORT_CEILING_SCRGB)):
            handle.write(f"{key}={value}\n")

    duplicates = {"layer0_vertical": _duplicate_taps(tables[0][0][3]),
                  "layer0_horizontal": _duplicate_taps(tables[0][1][3]),
                  "layer1_vertical": _duplicate_taps(tables[1][0][3]),
                  "layer1_horizontal": _duplicate_taps(tables[1][1][3]),
                  "baseline_vertical": _duplicate_taps(baseline_table[0][3])}
    return {"master16": master16, "observations": observations,
            "baseline": baseline_full, "duplicate_edge_taps": duplicates}


# --- running the probe -----------------------------------------------------

def run_probe(out_dir, mode, fusion_dll, fixture_dir):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(PROBE):
        raise GateFailure(f"{PROBE} is not built; run "
                          "native/build_gpu_multiscale_fusion_a2_test.bat")
    completed = subprocess.run([PROBE, fixture_dir, out_dir, fusion_dll, HDR_DLL,
                                mode], cwd=NATIVE, capture_output=True, text=True,
                               timeout=900)
    with open(os.path.join(out_dir, "stdout.txt"), "w", encoding="utf-8",
              newline="\n") as handle:
        handle.write(completed.stdout)
        handle.write("\n--- stderr ---\n")
        handle.write(completed.stderr)
    print("\n".join(line for line in completed.stdout.splitlines()
                    if line.startswith(("[FAIL", "       cannot", "       the"))
                    or line.startswith("[ ok ]")))
    summary = _parse_summary(os.path.join(out_dir, "summary.txt"))
    if completed.returncode != 0:
        raise GateFailure(f"the probe exited {completed.returncode} in mode "
                          f"{mode}: " + "; ".join(
                              line for line in completed.stdout.splitlines()
                              if line.startswith("[FAIL]"))[:700])
    return summary, out_dir


def _parse_summary(path):
    if not os.path.exists(path):
        raise GateFailure(f"{path} was never written")
    values = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if " = " in line:
                key, value = line.rstrip("\n").split(" = ", 1)
                values[key] = value
    return values


def number(summary, key):
    if key not in summary:
        raise GateFailure(f"the probe reported no {key}")
    try:
        return float(summary[key])
    except ValueError:
        raise GateFailure(f"{key} = {summary[key]!r} is not a number")


def integer(summary, key):
    return int(number(summary, key))


def read_plane(path, height, pitch, width, bytes_per_pixel):
    """A harvested plane: rows at the reported pitch, useful samples first.

    An 8-byte row is four binary16 components, so the half plane is *reinterpreted*
    rather than converted: astype would turn the 0x3C00 bit pattern of a stored
    1.0 into the number 15360.0 and every comparison downstream would be nonsense.
    The float16 that comes back holds exactly what the texture held."""
    raw = np.fromfile(path, dtype=np.uint8)
    expected = int(pitch) * height
    if raw.size != expected:
        raise GateFailure(f"{path} is {raw.size} bytes, the plane needs {expected}")
    rows = np.ascontiguousarray(
        raw.reshape(height, pitch)[:, :width * bytes_per_pixel])
    if bytes_per_pixel == 8:
        return rows.view(np.float16).reshape(height, width, 4)
    return rows.reshape(height, width, 4)


# --- the reference ---------------------------------------------------------

def reference_solve(fixture):
    observations = [
        msf.LanczosObservation.from_extents(FULL_W, FULL_H, residual, weight)
        for (residual, (_, weight)) in zip(fixture["observations"], LAYERS)]
    solution, metrics = msf.solve_reference(
        observations, fixture["baseline"], BETA, max_iterations=MAX_ITER,
        relative_tolerance=RELTOL)
    return observations, solution, metrics


def data_term(observations, vector):
    total = 0.0
    for item in observations:
        difference = item.down(np.asarray(vector, np.float32)) - item.residual
        total += item.weight * float(np.sum(difference.astype(np.float64) ** 2))
    return total


def adjoint_pairing(observations, seed=0xA2):
    """<Dv, y> against <v, D^T y>: the identity the CG depends on."""
    rng = np.random.default_rng(seed)
    report = {}
    for index, item in enumerate(observations):
        v = rng.standard_normal(
            (item.y.src_len, item.x.src_len, 3)).astype(np.float32)
        y = rng.standard_normal(
            (item.y.dst_len, item.x.dst_len, 3)).astype(np.float32)
        left = float(np.sum(item.down(v).astype(np.float64)
                            * y.astype(np.float64)))
        right = float(np.sum(v.astype(np.float64)
                             * item.adjoint(y).astype(np.float64)))
        scale = max(abs(left), abs(right), 1.0e-12)
        report[f"layer{index}"] = {"down_inner": left, "adjoint_inner": right,
                                   "relative_error": abs(left - right) / scale}
    return report


# --- the gate --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage A2 typed HDR fusion gate")
    parser.add_argument("--archive", default=ARCHIVE_FUSION,
                        help="the pre-A2 fusion bridge, for the SDR control")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.out or os.path.join(
        ROOT, "bench_out", "hdr_a2_evidence", time.strftime("a2-%Y%m%d-%H%M%S")))
    os.makedirs(run_dir, exist_ok=True)
    fixture_dir = os.path.join(run_dir, "fixture")
    evidence = {"run": os.path.relpath(run_dir, ROOT), "performance":
                "NOT_CHARACTERIZED", "thresholds": {
        "stored_component": "one binary16 ulp at the reference value + "
                            f"{STORE_SLACK}",
        "recovered_correction": f"one binary16 ulp of the stored sample + "
                                f"{CORRECTION_SLACK}",
        "adjoint_pairing": f"{ADJOINT_SLACK} relative"}}

    for name, path in (("probe", PROBE), ("fusion_dll", FUSION_DLL),
                       ("hdr_dll", HDR_DLL), ("archive_fusion_dll", args.archive)):
        if not os.path.exists(path):
            raise GateFailure(f"{name} is missing at {path}")
        evidence[name] = {"path": os.path.relpath(path, ROOT),
                          "bytes": os.path.getsize(path),
                          "sha256": sha256(path)}

    fixture = build_fixture(fixture_dir)
    evidence["fixture"] = {"full": [FULL_W, FULL_H],
                           "layers": [[list(size), weight]
                                      for size, weight in LAYERS],
                           "beta": BETA, "residual_strength": STRENGTH,
                           "relative_tolerance": RELTOL,
                           "max_iterations": MAX_ITER,
                           "duplicate_edge_taps": fixture["duplicate_edge_taps"]}
    check(sum(fixture["duplicate_edge_taps"].values()) > 0,
          "the plans carry repeated clamp-to-edge taps the transpose must keep")

    print("== the typed solve, on the device ==")
    typed, typed_dir = run_probe(os.path.join(run_dir, "typed"), "typed",
                                 FUSION_DLL, fixture_dir)
    evidence["typed_summary"] = typed
    check(integer(typed, "fusion_abi") == hfe.GPU_FUSION_ABI_VERSION,
          "the bridge answers the ABI the wrapper mirrors",
          f"abi {typed.get('fusion_abi')}")
    check(integer(typed, "fusion_hdr_abi") == hfe.GPU_FUSION_HDR_ABI_VERSION,
          "the typed HDR group is present and versioned",
          f"hdr abi {typed.get('fusion_hdr_abi')}")
    check(typed.get("fusion_typed_group") == "present",
          "the bridge under test carries the typed entries")
    check(integer(typed, "output_format") == hfe.DXGI_R16G16B16A16_FLOAT,
          "the fused store is R16G16B16A16_FLOAT",
          f"format {typed.get('output_format')}")
    check(integer(typed, "output_state") == hfe.D3D12_COPY_SOURCE,
          "the fused store is left in COPY_SOURCE",
          f"state {integer(typed, 'output_state'):#x}")
    check(integer(typed, "output_width") == FULL_W
          and integer(typed, "output_height") == FULL_H,
          "the fused store is the full extent",
          f"{typed.get('output_width')}x{typed.get('output_height')}")
    check(integer(typed, "converged") == 1
          and 0 < integer(typed, "iterations") <= MAX_ITER,
          "the solve converged inside the budget",
          f"iterations {typed.get('iterations')}")
    check(integer(typed, "fused_non_finite_components") == 0,
          "the store holds no Inf and no NaN while reporting success")
    check(integer(typed, "fused_components_above_one") > 0
          and integer(typed, "fused_components_below_zero") > 0,
          "headroom above 1.0 and out-of-gamut negatives survived the solve",
          f"{typed.get('fused_components_above_one')} above one, "
          f"{typed.get('fused_components_below_zero')} below zero")
    check(integer(typed, "unrepresentable_components") == 0,
          "the ordinary solve needed no fallback")

    pitch = integer(typed, "fused_row_pitch")
    stored16 = read_plane(os.path.join(typed_dir, "fused.bin"), FULL_H, pitch,
                         FULL_W, 8)
    stored = stored16.astype(np.float32)
    evidence["store"] = {"row_pitch": pitch,
                         "harness_bytes_downloaded": int(pitch) * FULL_H,
                         "download_is_harness_side": True,
                         "max_abs_component": float(np.max(np.abs(stored[..., :3]))),
                         "ulp_at_max": half_ulp(float(np.max(np.abs(
                             stored[..., :3]))))}
    evidence["counters"] = {
        key: integer(typed, key) for key in
        ("counter_process_count", "counter_source_device_copy_bytes",
         "counter_observation_device_copy_bytes", "counter_scalar_readback_bytes",
         "counter_unrepresentable_components")}
    check(integer(typed, "counter_scalar_readback_bytes") > 0,
          "the host reductions are metered as scalars",
          f"{typed.get('counter_scalar_readback_bytes')} bytes")
    check(integer(typed, "counter_process_count") == 1,
          "the counters count the one frame the bridge solved")

    print("== the same problem, on the CPU reference ==")
    observations, solution, metrics = reference_solve(fixture)
    evidence["reference"] = {key: float(value) for key, value in metrics.items()}
    print(f"       reference iterations {metrics['iterations']} relative "
          f"{metrics['relative_residual']:.6g}; device iterations "
          f"{typed.get('iterations')} relative "
          f"{number(typed, 'relative_residual'):.6g}")

    master16 = fixture["master16"]
    correction = (stored[..., :3] - master16).astype(np.float32)
    difference = np.abs(correction - solution)
    # The budget below is per component, not flat: the correction the device
    # reports is the difference of two binary16 samples, so it carries the store's
    # own rounding of a bright master along with the solver's float32 slack. One
    # ulp is already 7.8e-3 where the master reaches 9.4 and 2.4e-4 where it is
    # 0.3, which is why the budget moves with the sample and the part that does
    # not come from the grid is the declared CORRECTION_SLACK.
    budget = np.vectorize(half_ulp)(stored[..., :3]) + CORRECTION_SLACK
    ratio = float(np.max(difference / budget))
    evidence["reference_agreement"] = {
        "max_abs_correction_error": float(np.max(difference)),
        "worst_ratio_against_the_declared_budget": ratio,
        "ulp_at_the_brightest_stored_sample": float(np.max(
            np.vectorize(half_ulp)(stored[..., :3])))}
    check(ratio <= 1.0,
          "the device correction matches the CPU reference within one half ulp "
          "of the stored sample plus the float32 slack",
          f"worst ratio {ratio:.6g}, max abs {float(np.max(difference)):.6g}")

    expected = (master16 + STRENGTH * solution).astype(np.float32)
    budget = np.vectorize(half_ulp)(expected) + STORE_SLACK
    error = np.abs(stored[..., :3] - expected)
    ratio = float(np.max(error / budget))
    evidence["store"]["max_abs_error_vs_reference_picture"] = float(
        np.max(error))
    evidence["store"]["worst_ratio_against_one_ulp_plus_slack"] = ratio
    check(ratio <= 1.0,
          "every stored component is within one half ulp of the reference plus "
          "the declared slack", f"worst ratio {ratio:.6g}, max abs error "
          f"{float(np.max(error)):.6g}")

    print("== the objective and the data term ==")
    baseline = fixture["baseline"]
    objective = {"baseline": msf.consistency_objective(baseline, observations,
                                                       baseline, BETA),
                 "device": msf.consistency_objective(correction, observations,
                                                     baseline, BETA)}
    objective["data_baseline"] = data_term(observations, baseline)
    objective["data_device"] = data_term(observations, correction)
    # The reference's own value at the same point, so the two drops can be read
    # against each other: the device is judged against the baseline, and the
    # distance from the reference is the store's, reported in reference_agreement.
    objective["solution"] = msf.consistency_objective(solution, observations,
                                                      baseline, BETA)
    objective["data_solution"] = data_term(observations, solution)
    evidence["objective"] = objective
    check(objective["device"] < objective["baseline"],
          "the objective drops from the Catmull baseline",
          f"{objective['baseline']:.6g} -> {objective['device']:.6g}")
    check(objective["data_device"] < objective["data_baseline"],
          "the data term drops from the Catmull baseline",
          f"{objective['data_baseline']:.6g} -> {objective['data_device']:.6g}")
    print(f"       baseline {objective['baseline']:.6g}; device "
          f"{objective['device']:.6g}; CPU reference "
          f"{objective['solution']:.6g}")

    print("== the transpose the solver relies on ==")
    pairing = adjoint_pairing(observations)
    evidence["adjoint_pairing"] = pairing
    for layer, values in sorted(pairing.items()):
        check(values["relative_error"] <= ADJOINT_SLACK,
              f"{layer}: <Dv,y> matches <v,D^T y>",
              f"{values['down_inner']:.6g} against {values['adjoint_inner']:.6g}, "
              f"relative {values['relative_error']:.3g}")

    print("== the fused texture through a real HDR bridge ==")
    check(integer(typed, "composited_set_code") == 0,
          "gpu_hdr_set_composite accepted the fused texture and the fence")
    composited16 = read_plane(os.path.join(typed_dir, "composited.bin"), FULL_H,
                              integer(typed, "composited_pitch"), FULL_W, 8)
    evidence["handoff"] = {"composited_pitch": integer(typed, "composited_pitch"),
                           "bytes_downloaded_by_the_harness":
                               integer(typed, "composited_pitch") * FULL_H}
    check(np.array_equal(composited16[..., :3], stored16[..., :3]),
          "the bridge's composited plane is the fused samples, bit for bit")

    print("== the A1 observation ordering, on live state ==")
    check(integer(typed, "order_refusal_names_proxy") == 1,
          "an observation whose proxy belongs to a superseded working plane is "
          "refused, naming gpu_hdr_proxy",
          typed.get("order_observe_after_stale_proxy_message", ""))
    check(integer(typed, "order_current_refusal_names_proxy") == 0,
          "the same call after a current proxy gets past provenance",
          typed.get("order_observe_after_current_proxy_message", ""))
    check(integer(typed, "pass_dispatches")
          == ORDERING_PASS_DISPATCHES,
          "every dispatched pass is counted once, in one place",
          f"{typed.get('pass_dispatches')} against {ORDERING_PASS_DISPATCHES}")
    check(integer(typed, "scalar_counter_readback_bytes")
          == ORDERING_PASS_DISPATCHES * COUNTER_BYTES_PER_PASS,
          "the per-pass counter readback is reported in bytes",
          f"{typed.get('scalar_counter_readback_bytes')} bytes")

    print("== the counted out-of-range fallback ==")
    fell_back = integer(typed, "fallback_unrepresentable_components")
    kept = integer(typed, "fallback_pixels_holding_master")
    evidence["fallback"] = {"stage_code": integer(typed, "fallback_stage_code"),
                            "unrepresentable_components": fell_back,
                            "pixels_holding_master": kept,
                            "pixels": integer(typed, "fallback_pixels")}
    check(integer(typed, "fallback_stage_code") == 0,
          "a picture that fell back to the master still reports success")
    check(fell_back > 0 and fell_back % 3 == 0,
          "the store counted the components it could not hold, three a pixel",
          f"{fell_back} components")
    check(fell_back // 3 <= kept,
          "every counted fallback pixel holds the pristine master sample",
          f"{fell_back // 3} counted against {kept} identical")
    check(integer(typed, "fallback_non_finite_components") == 0
          and integer(typed, "fallback_out_of_range_components") == 0,
          "the fallback stores nothing the half grid cannot hold")

    print("== the failure gates ==")
    for key, what in (("non_finite_master_stage_code",
                       "a master the grid cannot hold fails outright"),
                      ("budget_stage_code",
                       "a one-iteration budget fails at the iteration stage")):
        code = integer(typed, key)
        check(code != 0, what, f"stage {code}")
        if key == "budget_stage_code":
            check(code == 12, "the starved budget reports the iteration stage",
                  f"stage {code}")
    check(integer(typed, "budget_texture_null") == 1
          and integer(typed, "non_finite_master_texture_null") == 1,
          "neither failure handed back a texture")
    check(integer(typed, "budget_retry_code") != 0
          and integer(typed, "non_finite_master_retry_code") != 0,
          "a failed bridge refuses the next frame instead of falling back")

    print("== the legacy SDR control, both builds ==")
    new_sdr, new_dir = run_probe(os.path.join(run_dir, "sdr_new"), "sdr",
                                 FUSION_DLL, fixture_dir)
    old_sdr, old_dir = run_probe(os.path.join(run_dir, "sdr_archive"), "sdr",
                                 args.archive, fixture_dir)
    evidence["sdr_new"] = new_sdr
    evidence["sdr_archive"] = old_sdr
    check(old_sdr.get("fusion_typed_group") == "absent"
          and integer(old_sdr, "fusion_hdr_abi") == 0,
          "the archived build carries no typed group and loads as a capability "
          "answer, not a failure", old_sdr.get("fusion_typed_group", "?"))
    check(integer(new_sdr, "sdr_output_format") == DXGI_R8G8B8A8_UNORM,
          "the SDR entry still composites bytes",
          f"format {new_sdr.get('sdr_output_format')}")
    check(integer(new_sdr, "sdr_harvested") == 1
          and integer(old_sdr, "sdr_harvested") == 1,
          "both builds harvested a composite")
    new_bytes = open(os.path.join(new_dir, "sdr_composite.bin"), "rb").read()
    old_bytes = open(os.path.join(old_dir, "sdr_composite.bin"), "rb").read()
    check(len(new_bytes) > 0 and new_bytes == old_bytes,
          "the untouched SDR entry writes byte-identical pictures on both builds",
          f"{len(new_bytes)} bytes against {len(old_bytes)}")
    for key in ("sdr_iterations", "sdr_initial_residual_norm",
               "sdr_final_residual_norm", "sdr_relative_residual",
               "sdr_converged", "sdr_scalar_readback_bytes",
               "sdr_full_device_copy_bytes", "sdr_process_count"):
        check(number(new_sdr, key) == number(old_sdr, key),
              f"the SDR report is unchanged on {key}",
              f"{new_sdr.get(key)} against {old_sdr.get(key)}")

    with open(os.path.join(run_dir, "verdict.json"), "w", encoding="utf-8",
              newline="\n") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True, default=str)
    print(f"\nverdict: PASS   evidence: {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GateFailure, msf.FusionError) as error:
        print(f"\nverdict: FAIL   {error}")
        sys.exit(1)
