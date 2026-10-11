"""Verify the authored HDR10 fixture in the physical domain, on the CPU only.

`native/make_hdr10_valid_fixture.py` wrote `build/hdr_probe/hdr_valid_src.mkv` from a
scene it knew in cd/m^2 and checked the samples before they were coded. This script is
the other half: it never runs the generator, it software-decodes the first frame of the
clip that is actually on disk with a decoder that is not the product's, and asks the
same physical questions of the bytes that come back -

  * are the decoded 10-bit codes the codes that were authored (lossless coding has to
    answer "exactly", so any non-zero code error is a defect, not a tolerance);
  * do the decoded codes land on a plane layout whose byte count matches the declared
    yuv420p10le geometry, with nothing above the 10-bit field;
  * do the limited-range BT.2020 non-constant-luminance inverse and the ST 2084 EOTF,
    re-derived here from the ITU-R primaries rather than copied from the manifest,
    reconstruct PQ components inside BT.2100-2's [0, 1] - with no clamp anywhere in
    this file, so an excursion is reported as the excursion it is;
  * and are the declared mastering / MaxCLL / MaxFALL numbers consistent with what the
    frame's own pixels reconstruct, rather than with what the container claims.

The PQ inverse here is deliberately unclamped: `(root - c1) / (c2 - c3 * root)` with
`root = E' ** (1/m2)`. For a coded component in [0, 1] the denominator cannot reach zero
(the pole sits at E' = 1.9921, far above BT.2100's top), and the numerator is negative
only below E' ~ 4e-21, so nothing in this fixture is near either edge and no sample is
quietly repaired. A negative or over-1 component would surface as NaN, a range
violation, or both.

Nothing here touches the GPU: no DLL is loaded, no device is created, no image is read
back. The native admission of the same clip is a separate gate
(`native/hdr_real_admission_gate.py --source ... --support-ceiling ...`); its
``evidence.json`` can be handed to this file with ``--native-evidence`` so the two
domains are compared per component - read as data, never run.

    python native/verify_hdr10_valid_fixture.py
    python native/verify_hdr10_valid_fixture.py --native-evidence <gate evidence.json>
        --output <consolidated evidence.json>
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "build" / "hdr_probe" / "hdr_valid_src.mkv"
REFERENCE_DIR = ROOT / "bench_out" / "hdr_valid_fixture"
AUTHORED_PLANES = REFERENCE_DIR / "authored_f0.yuv"
SCENE_MANIFEST = REFERENCE_DIR / "authored_scene.json"
DECODED_PLANES = REFERENCE_DIR / "decoded_f0.yuv420p10le.raw"
FFMPEG = os.environ.get("FFMPEG_BIN", r"F:\WhisperJAV\Library\bin\ffmpeg.exe")

WIDTH, HEIGHT, FRAMES, FPS = 1920, 1080, 6, 25
Y_BYTES = WIDTH * HEIGHT * 2
C_BYTES = (WIDTH // 2) * (HEIGHT // 2) * 2
FRAME_BYTES = Y_BYTES + 2 * C_BYTES

PQ_M1, PQ_M2 = 2610.0 / 16384.0, (2523.0 / 4096.0) * 128.0
PQ_C1, PQ_C2, PQ_C3 = 3424.0 / 4096.0, (2413.0 / 4096.0) * 32.0, (2392.0 / 4096.0) * 32.0
PQ_PEAK_NITS, REFERENCE_WHITE_NITS = 10000.0, 80.0
SCRGB_PER_RELATIVE = PQ_PEAK_NITS / REFERENCE_WHITE_NITS

Y_OFFSET, Y_SCALE, C_OFFSET, C_SCALE = 64.0, 876.0, 512.0, 896.0
Y_TOP, C_TOP = 940.0, 960.0
CODE_FIELD_TOP = (1 << 10) - 1

D65 = np.array([0.3127, 0.3290])
BT709_PRIMARIES = np.array([0.640, 0.330, 0.300, 0.600, 0.150, 0.060])
BT2020_PRIMARIES = np.array([0.708, 0.292, 0.170, 0.795, 0.131, 0.046])


def pq_eotf(coded):
    """ST 2084 E' -> linear relative to 10000 cd/m^2. Unclamped, by design."""
    root = np.asarray(coded, dtype=np.float64) ** (1.0 / PQ_M2)
    return ((root - PQ_C1) / (PQ_C2 - PQ_C3 * root)) ** (1.0 / PQ_M1)


def _rgb_to_xyz(primaries, white=D65):
    xyz = np.empty((3, 3), dtype=np.float64)
    for index in range(3):
        x, y = primaries[2 * index], primaries[2 * index + 1]
        xyz[:, index] = np.array([x / y, 1.0, (1.0 - x - y) / y])
    target = np.array([white[0] / white[1], 1.0,
                       (1.0 - white[0] - white[1]) / white[1]])
    return xyz * np.linalg.solve(xyz, target)


# Derived here, from the published primaries - the manifest's copy is compared to
# this, not used by it.
TO_SCRGB = np.linalg.inv(_rgb_to_xyz(BT709_PRIMARIES)) @ _rgb_to_xyz(BT2020_PRIMARIES)
XYZ_2020 = _rgb_to_xyz(BT2020_PRIMARIES)
KR, KG, KB = (float(XYZ_2020[1, i]) for i in range(3))
CR_GAIN, CB_GAIN = 2.0 * (1.0 - KR), 2.0 * (1.0 - KB)
CG_CR, CG_CB = 2.0 * KR * (1.0 - KR) / KG, 2.0 * KB * (1.0 - KB) / KG
POLE_PQ_INPUT = (PQ_C2 / PQ_C3) ** PQ_M2
POSITIVE_ROW_SUMS = np.maximum(TO_SCRGB, 0.0).sum(axis=1)
LEGAL_PEAK_SCRGB = float(POSITIVE_ROW_SUMS.max()) * SCRGB_PER_RELATIVE


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, condition, detail=""):
        passed = bool(condition)
        self.items.append({"name": name, "pass": passed, "detail": _text(detail)})
        return passed

    @property
    def failed(self):
        return [item["name"] for item in self.items if not item["pass"]]


def _text(value):
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    text = value if isinstance(value, str) else json.dumps(
        value, default=str, ensure_ascii=False)
    return text if len(text) <= 900 else text[:900] + "..."


def run(argv):
    done = subprocess.run([str(item) for item in argv], capture_output=True, text=True)
    return done


def software_decode(target, out_path):
    """One frame, native software hevc decoder, explicit planar yuv420p10le."""
    argv = [FFMPEG, "-hide_banner", "-y", "-loglevel", "error",
            "-hwaccel", "none", "-c:v", "hevc", "-i", str(target),
            "-frames:v", "1", "-pix_fmt", "yuv420p10le", "-f", "rawvideo",
            str(out_path)]
    done = run(argv)
    return {"argv": [str(item) for item in argv], "rc": done.returncode,
            "stderr": done.stderr.strip()[:400],
            "decoder": "ffmpeg native hevc (software), -hwaccel none",
            "output_format": ("yuv420p10le PLANAR: Y plane, Cb plane, Cr plane; "
                              "10-bit code in the low 10 bits of each little-endian "
                              "uint16; no padding on either stride"),
            "bytes": (out_path.stat().st_size if out_path.is_file() else None)}


def ffprobe(target, frames):
    argv = [str(Path(FFMPEG).with_name("ffprobe.exe")), "-v", "error",
            "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,profile,level,pix_fmt,width,height,r_frame_rate,"
            "color_range,color_primaries,color_transfer,color_space,"
            "bits_per_raw_sample", "-show_frames", "-read_intervals", f"%+{frames}",
            "-of", "json", str(target)]
    done = run(argv)
    if done.returncode:
        return {"ran": False, "rc": done.returncode, "stderr": done.stderr[:300]}, {}
    data = json.loads(done.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    side = {}
    for frame in data.get("frames") or []:
        for item in frame.get("side_data_list") or []:
            side.setdefault(str(item.get("side_data_type")), item)
    return {"ran": True, "rc": done.returncode, "stream": stream}, side


def count_frames(target):
    """Matroska carries no frame count in the stream header, so the frames have to
    be read to be counted rather than quoted from a container field."""
    argv = [str(Path(FFMPEG).with_name("ffprobe.exe")), "-v", "error",
            "-select_streams", "v:0", "-count_frames", "-show_entries",
            "stream=nb_read_frames,nb_read_frames_nodrops", "-of",
            "default=nw=1:nk=1", str(target)]
    done = run(argv)
    values = [line.strip() for line in (done.stdout or "").splitlines() if line.strip()]
    return {"argv": [str(item) for item in argv], "rc": done.returncode,
            "nb_read_frames": values[0] if values else None,
            "nb_read_frames_nodrops": values[1] if len(values) > 1 else None}


def planes_of(blob):
    y = np.frombuffer(blob, dtype="<u2", count=WIDTH * HEIGHT, offset=0)
    cb = np.frombuffer(blob, dtype="<u2", count=(WIDTH // 2) * (HEIGHT // 2),
                       offset=Y_BYTES)
    cr = np.frombuffer(blob, dtype="<u2", count=(WIDTH // 2) * (HEIGHT // 2),
                       offset=Y_BYTES + C_BYTES)
    return y, cb, cr


def reconstruct(y_code, cb_code, cr_code, phase_y=0, phase_x=0):
    """Decoded codes -> PQ -> linear relative -> scRGB, with the chroma box the
    product reads (each 4:2:0 sample replicated to its four pixels), or a chosen
    neighbouring box for the worst-case sweep."""
    rows = np.clip(np.arange(HEIGHT // 2) + phase_y, 0, HEIGHT // 2 - 1)
    columns = np.clip(np.arange(WIDTH // 2) + phase_x, 0, WIDTH // 2 - 1)

    def upsample(plane):
        return np.repeat(np.repeat(plane.reshape(HEIGHT // 2, WIDTH // 2)
                                   [np.ix_(rows, columns)], 2, 0), 2, 1)

    y_prime = (y_code.reshape(HEIGHT, WIDTH).astype(np.float64) - Y_OFFSET) / Y_SCALE
    cb_prime = (upsample(cb_code).astype(np.float64) - C_OFFSET) / C_SCALE
    cr_prime = (upsample(cr_code).astype(np.float64) - C_OFFSET) / C_SCALE
    encoded = np.stack([y_prime + CR_GAIN * cr_prime,
                        y_prime - CG_CB * cb_prime - CG_CR * cr_prime,
                        y_prime + CB_GAIN * cb_prime], axis=-1)
    relative = pq_eotf(encoded)
    return encoded, relative


def nits_of_neutral_luma(code):
    """Nits a single luma code reconstructs to with neutral chroma - the achromatic
    case, and the only honest way to talk about 'one code step' in cd/m^2."""
    return float(pq_eotf(np.array([(float(code) - Y_OFFSET) / Y_SCALE]))[0]
                 * PQ_PEAK_NITS)


def extents(values):
    finite = np.isfinite(values)
    held = values[finite]
    return {"min": float(held.min()), "max": float(held.max()),
            "nonfinite": int((~finite).sum())}


def component_counts(values):
    """Per-component tallies, so these numbers can be set against the native source
    report's own per-component counters rather than only against its min and max."""
    held = values[np.isfinite(values)]
    return {"negative_components": int(np.count_nonzero(held < 0.0)),
            "above_one_components": int(np.count_nonzero(held > 1.0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--authored", type=Path, default=AUTHORED_PLANES)
    parser.add_argument("--manifest", type=Path, default=SCENE_MANIFEST)
    parser.add_argument("--output", type=Path,
                        default=REFERENCE_DIR / "physical_check.json")
    parser.add_argument("--native-evidence", type=Path, default=None,
                        help="an admission gate evidence.json for the same clip, to "
                             "compare the device's per-component counters against")
    arguments = parser.parse_args()
    checks, evidence = Checks(), {}
    started = datetime.now(timezone.utc).isoformat()

    for path in (arguments.fixture, arguments.authored, arguments.manifest):
        checks.add(f"input exists: {path.name}", path.is_file(), str(path))
    if checks.failed:
        raise SystemExit("missing input; nothing was decoded")

    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))

    # --- 1. independent software decode of the frame that is on disk.
    decode = software_decode(arguments.fixture, DECODED_PLANES)
    evidence["software_decode"] = decode
    checks.add("software decode exits 0 and writes a frame",
               decode["rc"] == 0 and decode["bytes"] == FRAME_BYTES,
               {"rc": decode["rc"], "bytes": decode["bytes"],
                "expected": FRAME_BYTES})

    blob = DECODED_PLANES.read_bytes()
    y_code, cb_code, cr_code = planes_of(blob)
    evidence["plane_layout"] = {
        "declared": "yuv420p10le planar", "frame_bytes": len(blob),
        "y": {"offset": 0, "bytes": Y_BYTES, "samples": int(y_code.size),
              "stride_bytes": WIDTH * 2, "rows": HEIGHT},
        "cb": {"offset": Y_BYTES, "bytes": C_BYTES, "samples": int(cb_code.size),
               "stride_bytes": (WIDTH // 2) * 2, "rows": HEIGHT // 2},
        "cr": {"offset": Y_BYTES + C_BYTES, "bytes": C_BYTES,
               "samples": int(cr_code.size),
               "stride_bytes": (WIDTH // 2) * 2, "rows": HEIGHT // 2},
        "sum_of_plane_bytes_equals_frame": Y_BYTES + 2 * C_BYTES == FRAME_BYTES,
        "no_padding": len(blob) == Y_BYTES + 2 * C_BYTES}
    checks.add("raw byte count is exactly the declared planar geometry",
               len(blob) == FRAME_BYTES == Y_BYTES + 2 * C_BYTES,
               evidence["plane_layout"]["cr"])
    checks.add("no sample uses a bit above the 10-bit field",
               int(y_code.max()) <= CODE_FIELD_TOP
               and int(cb_code.max()) <= CODE_FIELD_TOP
               and int(cr_code.max()) <= CODE_FIELD_TOP,
               [int(y_code.max()), int(cb_code.max()), int(cr_code.max())])
    checks.add("decoded codes sit inside the limited range (Y 64..940, C 64..960)",
               y_code.min() >= Y_OFFSET and y_code.max() <= Y_TOP
               and cb_code.min() >= 64 and cb_code.max() <= C_TOP
               and cr_code.min() >= 64 and cr_code.max() <= C_TOP,
               {"y": [int(y_code.min()), int(y_code.max())],
                "cb": [int(cb_code.min()), int(cb_code.max())],
                "cr": [int(cr_code.min()), int(cr_code.max())]})

    # --- 2. input-vs-decoded code error against what the generator authored.
    authored_blob = arguments.authored.read_bytes()
    a_y, a_cb, a_cr = planes_of(authored_blob)
    per_plane = {}
    for name, decoded, authored in (("y", y_code, a_y), ("cb", cb_code, a_cb),
                                    ("cr", cr_code, a_cr)):
        delta = decoded.astype(np.int32) - authored.astype(np.int32)
        per_plane[name] = {"samples": int(delta.size),
                           "mismatching_samples": int(np.count_nonzero(delta)),
                           "max_abs_code_error": int(np.abs(delta).max()),
                           "mean_abs_code_error": float(np.abs(delta).mean())}
    evidence["code_vs_authored"] = per_plane
    checks.add("the coded bitstream preserved every authored code (lossless)",
               all(item["mismatching_samples"] == 0 for item in per_plane.values()),
               per_plane)

    # --- 3. independent reconstruction, own phase and the 8 neighbouring box phases.
    own_encoded, own_relative = reconstruct(y_code, cb_code, cr_code)
    own_scrgb = own_relative @ TO_SCRGB.T * SCRGB_PER_RELATIVE
    measured = {"pq": dict(extents(own_encoded),
                           **component_counts(own_encoded)),
                "relative": extents(own_relative),
                "nits": extents(own_relative * PQ_PEAK_NITS),
                "scrgb": dict(extents(own_scrgb),
                              negative_components=int(np.count_nonzero(
                                  own_scrgb < 0.0)))}
    pq_outside_unit = (measured["pq"]["negative_components"]
                       + measured["pq"]["above_one_components"])
    scrgb_negative = measured["scrgb"]["negative_components"]
    worst = {key: [np.inf, -np.inf] for key in ("pq", "relative", "nits", "scrgb")}
    nonfinite_across_phases = 0
    for phase_y in (-1, 0, 1):
        for phase_x in (-1, 0, 1):
            encoded, relative = reconstruct(y_code, cb_code, cr_code,
                                            phase_y, phase_x)
            scrgb = relative @ TO_SCRGB.T * SCRGB_PER_RELATIVE
            for key, values in (("pq", encoded), ("relative", relative),
                                ("nits", relative * PQ_PEAK_NITS), ("scrgb", scrgb)):
                item = extents(values)
                worst[key][0] = min(worst[key][0], item["min"])
                worst[key][1] = max(worst[key][1], item["max"])
                nonfinite_across_phases += item["nonfinite"]
    evidence["decoded_reconstruction"] = {
        "own_phase": measured,
        "pq_components_outside_unit_interval": pq_outside_unit,
        "scrgb_negative_components": scrgb_negative,
        "worst_case_chroma_phases": {key: {"min": lo, "max": hi}
                                     for key, (lo, hi) in worst.items()},
        "chroma_phases_checked": 9,
        "nonfinite_across_phases": nonfinite_across_phases,
        "pq_pole_coded_input": POLE_PQ_INPUT,
        "note": ("own_phase is the geometry the bridge's decode shader reads "
                 "(native/gpu_hdr_scrgb_bridge.cpp:653-684 replicates each chroma "
                 "sample to its own box); the sweep pairs every luma sample with the "
                 "chroma of its neighbours, which is what a different siting or an "
                 "interpolating upsampler would produce")}
    # The manifest's own-phase figures were computed before encoding, from the codes
    # the generator wrote. Agreement here means the clip on disk reconstructs the way
    # it was authored, in every domain - nits and scRGB included.
    authored_record = manifest["per_frame_checks"][0]
    deltas = {}
    for key in ("pq", "relative", "nits", "scrgb"):
        record = authored_record["own_phase"][key]
        deltas[key] = {"decoded": [measured[key]["min"], measured[key]["max"]],
                       "authored_record": [record["min"], record["max"]],
                       "max_abs_delta": max(abs(measured[key]["min"] - record["min"]),
                                            abs(measured[key]["max"] - record["max"]))}
    authored_worst = authored_record["worst_case_chroma_phases"]
    worst_deltas = {key: max(abs(worst[key][0] - authored_worst[key]["min"]),
                             abs(worst[key][1] - authored_worst[key]["max"]))
                    for key in ("pq", "relative", "nits", "scrgb")}
    evidence["decoded_vs_authored_reconstruction"] = {
        "own_phase": deltas, "worst_case_chroma_phases_max_abs_delta": worst_deltas}
    checks.add("the decoded frame reconstructs the authored frame's own-phase ranges",
               all(item["max_abs_delta"] < 1e-9 for item in deltas.values()), deltas)
    checks.add("and the same under the neighbouring-box chroma sweep",
               all(value < 1e-9 for value in worst_deltas.values()), worst_deltas)
    checks.add("no decoded PQ component sits outside [0, 1] counted per component, "
               "not only by range",
               measured["pq"]["negative_components"] == 0
               and measured["pq"]["above_one_components"] == 0,
               {"pq_components_outside_unit_interval": pq_outside_unit,
                "scrgb_negative_components": scrgb_negative,
                "note": ("the scRGB negative count is the figure the native source "
                         "report's negative_components is set against in the "
                         "consolidated evidence")})
    checks.add("every reconstructed sample is finite (nothing clamped or repaired)",
               nonfinite_across_phases == 0
               and measured["pq"]["nonfinite"] == 0
               and measured["scrgb"]["nonfinite"] == 0,
               {"across_phases": nonfinite_across_phases})
    checks.add("decoded PQ components are inside BT.2100-2's [0, 1] at the product's "
               "own chroma phase",
               measured["pq"]["min"] >= 0.0 and measured["pq"]["max"] <= 1.0,
               measured["pq"])
    checks.add("decoded PQ components are inside [0, 1] under every chroma phase",
               worst["pq"][0] >= 0.0 and worst["pq"][1] <= 1.0, worst["pq"])
    checks.add("decoded normalised linear stays inside [0, 1] (1.0 == 10000 nits)",
               worst["relative"][0] >= 0.0 and worst["relative"][1] <= 1.0,
               worst["relative"])
    checks.add("decoded per-channel scRGB peak stays under ST 2084's own top in "
               "these units (1.0 PQ == 10000 nits == 125 scRGB)",
               worst["scrgb"][1] <= SCRGB_PER_RELATIVE,
               {"measured_max_scrgb": worst["scrgb"][1],
                "st2084_top_scrgb": SCRGB_PER_RELATIVE,
                "note": ("125 scRGB is what an in-Rec.709-gamut sample can reach at "
                         "the top of PQ; out-of-gamut saturated patches can exceed "
                         "it by the primaries conversion, which the derived legal "
                         "content bound below is the check for")})

    # --- 4. the manifest's own numbers, re-derived rather than trusted.
    manifest_constants = manifest["constants"]
    drift = max(abs(TO_SCRGB - np.array(manifest["to_scrgb_matrix"])).max(),
                abs(KR - manifest_constants["kr"]), abs(KG - manifest_constants["kg"]),
                abs(KB - manifest_constants["kb"]))
    evidence["contract_agreement"] = {
        "kr_kg_kb_derived_here": [KR, KG, KB],
        "kr_kg_kb_in_manifest": [manifest_constants["kr"], manifest_constants["kg"],
                                 manifest_constants["kb"]],
        "to_scrgb_max_abs_delta": float(drift),
        "positive_row_sums_derived_here": POSITIVE_ROW_SUMS.tolist(),
        "legal_peak_scrgb_after_primary_conversion": LEGAL_PEAK_SCRGB,
        "manifest_legal_peak_scrgb":
            manifest["physical_bound"]["valid_peak_scrgb_after_primary_conversion"]}
    checks.add("this file's independently derived matrix and Kr/Kg/Kb agree with the "
               "manifest's", drift < 1e-12, evidence["contract_agreement"])
    checks.add("measured peak respects the derived legal-content scRGB bound "
               "(not a metadata copy)",
               worst["scrgb"][1] <= LEGAL_PEAK_SCRGB,
               {"measured_max_scrgb": worst["scrgb"][1], "derived_bound":
                LEGAL_PEAK_SCRGB})

    # --- 5. declared metadata against the scene the pixels reconstruct.
    linear_luma = own_relative @ np.array([KR, KG, KB])
    mean_nits = float(linear_luma.mean() * PQ_PEAK_NITS)
    peak_pixel_nits = float((own_relative.max(axis=-1) * PQ_PEAK_NITS).max())
    declared = manifest["declared_metadata"]
    # How far one luma code step moves nits at the brightest code in the frame: the
    # only tolerance in which a MaxCLL claim written in nits can be compared with a
    # peak that has to live on the 10-bit code grid.
    peak_code = int(y_code.max())
    neutral_peak_nits = nits_of_neutral_luma(peak_code)
    one_code_step_nits = nits_of_neutral_luma(peak_code + 1) - neutral_peak_nits
    evidence["measured_light_levels"] = {
        "frame_mean_nits": mean_nits,
        "frame_peak_pixel_nits": peak_pixel_nits,
        "declared_max_cll_nits": declared["max_cll_nits"],
        "declared_max_fall_nits": declared["max_fall_nits"],
        "declared_mastering_max_nits": declared["mastering_monitor_max_nits"],
        "brightest_luma_code": peak_code,
        "nits_at_brightest_luma_code_read_neutral": neutral_peak_nits,
        "one_luma_code_step_at_that_level_nits": one_code_step_nits,
        "max_cll_minus_decoded_peak_nits": neutral_peak_nits - declared["max_cll_nits"]}
    checks.add("the frame's peak is carried by luma, not by a chroma excursion "
               "(the brightest code read as neutral reproduces the measured peak)",
               abs(neutral_peak_nits - peak_pixel_nits) < 1e-6,
               {"neutral_read_of_peak_code": neutral_peak_nits,
                "measured_peak_pixel_nits": peak_pixel_nits})
    checks.add("measured peak is under the declared mastering-display maximum",
               peak_pixel_nits <= declared["mastering_monitor_max_nits"],
               evidence["measured_light_levels"])
    checks.add("the declared MaxCLL is the authored peak and differs from the "
               "decoded peak by less than one luma code step (quantisation on the "
               "10-bit grid, not an unbounded metadata claim)",
               abs(neutral_peak_nits - declared["max_cll_nits"]) < one_code_step_nits,
               {"declared_max_cll_nits": declared["max_cll_nits"],
                "decoded_peak_nits": neutral_peak_nits,
                "difference_nits": neutral_peak_nits - declared["max_cll_nits"],
                "one_code_step_nits": one_code_step_nits})
    checks.add("measured frame mean is under the declared MaxFALL",
               mean_nits <= declared["max_fall_nits"], mean_nits)

    # --- 6. container claims, read outside the product.
    probe, side = ffprobe(arguments.fixture, FRAMES)
    stream = probe.get("stream") or {}
    counted = count_frames(arguments.fixture)
    evidence["container_probe"] = {"probe": probe, "frame_count": counted,
                                   "frame_side_data": side}
    checks.add("container is Main 10 / yuv420p10le at the authored geometry",
               stream.get("profile") == "Main 10"
               and stream.get("pix_fmt") == "yuv420p10le"
               and (stream.get("width"), stream.get("height")) == (WIDTH, HEIGHT),
               {key: stream.get(key) for key in ("profile", "bits_per_raw_sample",
                                                 "pix_fmt", "width", "height",
                                                 "level")})
    checks.add("container carries the HDR10 tagging the scene was authored with",
               stream.get("color_primaries") == "bt2020"
               and stream.get("color_transfer") == "smpte2084"
               and stream.get("color_space") == "bt2020nc"
               and stream.get("color_range") == "tv",
               {key: stream.get(key) for key in ("color_primaries", "color_transfer",
                                                 "color_space", "color_range")})
    checks.add(f"container decodes to exactly {FRAMES} frames at {FPS} fps",
               str(counted["nb_read_frames"]) == str(FRAMES)
               and stream.get("r_frame_rate") == f"{FPS}/1",
               {"nb_read_frames": counted["nb_read_frames"],
                "r_frame_rate": stream.get("r_frame_rate")})
    mastering = next((item for key, item in side.items()
                      if "mastering" in key.lower()), {})
    content_light = next((item for key, item in side.items()
                          if "light level" in key.lower()), {})
    evidence["side_data"] = {"mastering": mastering, "content_light": content_light}
    checks.add("container carries mastering-display and content-light side data",
               bool(mastering) and bool(content_light),
               {"mastering_keys": sorted(mastering), "content_light": content_light})

    def rational(text):
        num, _, den = str(text).partition("/")
        try:
            return float(num) / float(den or 1)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    claimed = [rational(mastering.get("red_x")), rational(mastering.get("red_y")),
               rational(mastering.get("green_x")), rational(mastering.get("green_y")),
               rational(mastering.get("blue_x")), rational(mastering.get("blue_y"))]
    claimed_white = [rational(mastering.get("white_point_x")),
                     rational(mastering.get("white_point_y"))]
    evidence["mastering_display_decoded"] = {
        "primaries_cie_xy": claimed, "white_point_cie_xy": claimed_white,
        "authored_primaries_cie_xy": BT2020_PRIMARIES.tolist(),
        "authored_white_point_cie_xy": D65.tolist(),
        "max_luminance_nits": rational(mastering.get("max_luminance")),
        "min_luminance_nits": rational(mastering.get("min_luminance"))}
    checks.add("the mastering primaries and white point in the container are the "
               "BT.2020 / D65 values the scene was authored for",
               all(a is not None and abs(a - b) < 1e-9 for a, b in
                   zip(claimed, BT2020_PRIMARIES.tolist()))
               and all(a is not None and abs(a - b) < 1e-9 for a, b in
                       zip(claimed_white, D65.tolist())),
               evidence["mastering_display_decoded"])
    if content_light:
        checks.add("container MaxCLL / MaxFALL are the authored peak and a measured "
                   "frame mean",
                   float(content_light.get("max_content", -1))
                   == declared["max_cll_nits"]
                   and float(content_light.get("max_average", -1))
                   == declared["max_fall_nits"],
                   {"max_content": content_light.get("max_content"),
                    "max_average": content_light.get("max_average"),
                    "declared_max_cll_nits": declared["max_cll_nits"],
                    "declared_max_fall_nits": declared["max_fall_nits"],
                    "measured_frame_mean_nits": mean_nits})

    # --- 7. the device's own numbers, if its evidence was handed over.
    if arguments.native_evidence:
        native = json.loads(arguments.native_evidence.read_text(encoding="utf-8"))
        report = native["positive_admission"]["source_report"]
        evidence["native_admission"] = native
        cross = {
            "native_evidence": str(arguments.native_evidence),
            "native_verdict": native.get("verdict"),
            "native_declared_ceiling_scrgb": report["support_ceiling_scrgb"],
            "native_min_component_scrgb": report["min_component"],
            "native_max_component_scrgb": report["max_component"],
            "native_negative_components": report["negative_components"],
            "native_above_ceiling_components": report["above_ceiling_components"],
            "native_nonfinite_samples": report["nonfinite_samples"],
            "native_pixels": report["pixels"],
            "cpu_min_scrgb": measured["scrgb"]["min"],
            "cpu_max_scrgb": measured["scrgb"]["max"],
            "cpu_negative_components": measured["scrgb"]["negative_components"],
            "cpu_pixels": WIDTH * HEIGHT,
            "min_abs_agreement": abs(report["min_component"]
                                     - measured["scrgb"]["min"]),
            "max_abs_agreement": abs(report["max_component"]
                                     - measured["scrgb"]["max"]),
            "component_count_delta": abs(report["negative_components"]
                                         - measured["scrgb"]["negative_components"])}
        evidence["native_cross_check"] = cross
        checks.add("the device and this CPU reference count the same negative scRGB "
                   "components on the same frame",
                   cross["component_count_delta"] == 0, cross)
        checks.add("the device's min/max scRGB agree with the CPU reference within "
                   "the FP16 output's own resolution",
                   cross["min_abs_agreement"] < 1e-3
                   and cross["max_abs_agreement"] < 1e-3, cross)
        checks.add("the device reports no non-finite and no above-ceiling component "
                   "on a clip this file proved physically valid",
                   report["nonfinite_samples"] == 0
                   and report["above_ceiling_components"] == 0, cross)
        checks.add("the declared ceiling in the native run is not the MaxCLL "
                   "equivalent and is above the measured peak",
                   report["support_ceiling_scrgb"] > report["max_component"]
                   and report["support_ceiling_scrgb"]
                   != native["declared_domain"]["pq_geometry"]
                          ["max_cll_equivalent_scrgb"],
                   {"ceiling": report["support_ceiling_scrgb"],
                    "measured_max": report["max_component"],
                    "max_cll_equivalent_scrgb":
                        native["declared_domain"]["pq_geometry"]
                            ["max_cll_equivalent_scrgb"]})

    # --- 8. the file that was decoded is the file the generator recorded.
    digest = sha256(blob).hexdigest()
    fixture_sha = sha256(arguments.fixture.read_bytes()).hexdigest()
    evidence["identities"] = {
        "fixture": {"path": str(arguments.fixture), "bytes":
                    arguments.fixture.stat().st_size, "sha256": fixture_sha,
                    "generator_recorded_sha256": manifest["fixture"]["sha256"]},
        "decoded_frame": {"path": str(DECODED_PLANES), "bytes": len(blob),
                          "sha256": digest},
        "authored_frame": {"path": str(arguments.authored),
                           "bytes": len(authored_blob),
                           "sha256": sha256(authored_blob).hexdigest(),
                           "generator_recorded_sha256":
                               manifest["reference_planes"]["sha256"]}}
    checks.add("decoded frame bytes are the authored frame bytes",
               digest == evidence["identities"]["authored_frame"]["sha256"], digest)
    checks.add("the fixture on disk is the one the generator recorded",
               fixture_sha == manifest["fixture"]["sha256"],
               {"loaded": fixture_sha,
                "recorded": manifest["fixture"]["sha256"],
                "note": ("Matroska writes a DateUTC element, so a regenerated "
                         "fixture hashes differently by design; equality here means "
                         "this file, not a rerun")})

    evidence["artifact_class"] = ("offline CPU physical-domain verification of an "
                                  "existing fixture - software decode plus a CPU "
                                  "reference of the same transfer; not a production "
                                  "host image transfer, no GPU, no readback, no "
                                  "neural model")
    evidence["checks"] = checks.items
    evidence["failed_checks"] = checks.failed
    evidence["verdict"] = "PASS" if not checks.failed else "FAIL"
    evidence["started_utc"], evidence["finished_utc"] = started, \
        datetime.now(timezone.utc).isoformat()
    evidence["python"], evidence["numpy"] = sys.version.split()[0], np.__version__
    evidence["not_claimed"] = {
        "native_admission": ("nothing in the physical checks below says anything "
                             "about the bridge; with --native-evidence the gate's own "
                             "record is embedded under native_admission and compared "
                             "per component under native_cross_check, read as data - "
                             "this file never runs it"),
        "gpu": "no DLL loaded, no device created, no image read back",
        "generator_rerun": "the fixture and the authored planes were read, never written",
        "export": ("the six-frame clip is the source for a later production export; no "
                   "export, orchestrator or UI wiring ran here"),
        "performance": "NOT_CHARACTERIZED by this run"}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False,
                                           default=str), encoding="utf-8")

    print(f"{evidence['verdict']}: {arguments.output}")
    for item in checks.items:
        print(("  ok   " if item["pass"] else "  FAIL ") + item["name"])
    for key in ("pq", "relative", "nits", "scrgb"):
        own, worst_item = measured[key], worst[key]
        print(f"  decoded {key:8s} own [{own['min']:.6f}, {own['max']:.6f}] "
              f"worst [{worst_item[0]:.6f}, {worst_item[1]:.6f}]")
    print("  code errors: "
          + ", ".join(f"{name} {item['mismatching_samples']}"
                      for name, item in per_plane.items()))
    if arguments.native_evidence:
        print(f"  native vs cpu reference: negative components "
              f"{cross['native_negative_components']} vs "
              f"{cross['cpu_negative_components']} (delta "
              f"{cross['component_count_delta']}), max scRGB "
              f"{cross['native_max_component_scrgb']:.6f} vs "
              f"{cross['cpu_max_scrgb']:.6f} (delta "
              f"{cross['max_abs_agreement']:.2e})")
    return 0 if evidence["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
