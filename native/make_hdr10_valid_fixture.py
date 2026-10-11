"""Author a physically valid HDR10 fixture, in nits, on the CPU.

`build/hdr_probe/hdr_src.mkv` is tagged HDR10 but is not an HDR10-colour-domain
sample: an independent software decode of its first frame reconstructs PQ RGB
components from -0.484 to 1.389, and ITU-R BT.2100-2 defines the PQ component E'
as being in [0, 1] with the normalised linear result in [0, 1] (1.0 == 10000
cd/m^2, which is 125 scRGB against this bridge's 80 cd/m^2 reference white). A
declared ceiling of 10000 scRGB admits that fixture without verifying anything
about it.

This script authors a fixture instead of tagging one. The scene is known linear
BT.2020 RGB in cd/m^2 before a single code exists, goes through the ST 2084 OETF
and the BT.2020 non-constant-luminance matrix, is quantised to 10-bit
limited-range 4:2:0, and is checked in that quantised form - all before the
encoder is asked for anything. The encoder is libx265 lossless Main 10, so the
coded stream carries exactly the samples validated here.

Nothing in this file is production image transfer: the authored planes are offline
reference test data for `native/verify_hdr10_valid_fixture.py`. No GPU, no neural
model, no NVOF, no NVENC, no product pipeline, no clamping.

Geometry rule. The bridge replicates each 4:2:0 chroma sample to the four pixels
of its box (`native/gpu_hdr_scrgb_bridge.cpp:653-684`), so with every patch
boundary on an even coordinate a chroma box never straddles two patches and the
sample the product reads is exactly the sample that was authored. The checks go
one better and pair every luma sample with the chroma of the *neighbouring* boxes
too (the +/-1 box phases another siting or an interpolating upsampler could
produce), so the fixture is in range under either reading. That is what bounds the
chroma amplitudes: a colour box may sit beside surround luma, and
Y'_surround + 2(1-Kr)Cr' has to stay under 1.0.

    python native/make_hdr10_valid_fixture.py
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "build" / "hdr_probe"
OLD_FIXTURE = FIXTURES / "hdr_src.mkv"
OUT_FIXTURE = FIXTURES / "hdr_valid_src.mkv"
REFERENCE_DIR = ROOT / "bench_out" / "hdr_valid_fixture"
FFMPEG = os.environ.get("FFMPEG_BIN", r"F:\WhisperJAV\Library\bin\ffmpeg.exe")

WIDTH, HEIGHT, FRAMES, FPS = 1920, 1080, 6, 25

# ST 2084 as ITU-R BT.2100-2 publishes it, in the same rational form the bridge
# compiles into its shader (native/gpu_hdr_scrgb_bridge.cpp:143-152).
PQ_M1 = 2610.0 / 16384.0
PQ_M2 = (2523.0 / 4096.0) * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = (2413.0 / 4096.0) * 32.0
PQ_C3 = (2392.0 / 4096.0) * 32.0
PQ_PEAK_NITS = 10000.0
REFERENCE_WHITE_NITS = 80.0
SCRGB_PER_RELATIVE = PQ_PEAK_NITS / REFERENCE_WHITE_NITS          # 125.0

# ITU-R BT.2100 limited-range 10-bit quantisation: luma codes [64, 940], colour
# differences [64, 960] about 512. The divisors are the bridge's own (876, 896).
Y_OFFSET, Y_SCALE = 64.0, 876.0
C_OFFSET, C_SCALE = 512.0, 896.0
Y_TOP, C_TOP = 940.0, 960.0

D65 = np.array([0.3127, 0.3290])
BT709_PRIMARIES = np.array([0.640, 0.330, 0.300, 0.600, 0.150, 0.060])
BT2020_PRIMARIES = np.array([0.708, 0.292, 0.170, 0.795, 0.131, 0.046])

# The scene, in cd/m^2. Every authored component sits between these numbers, so
# PQ sits between pq_oetf(SCENE_FLOOR_NITS) and pq_oetf(SCENE_PEAK_NITS) with room
# on both sides, and a declared support ceiling is never what keeps it honest.
SCENE_FLOOR_NITS = 12.0
SCENE_PEAK_NITS = 900.0
SURROUND_NITS = 203.0
GRAY_STAIR_NITS = (12.0, 25.0, 50.0, 100.0, 203.0, 400.0, 650.0, 900.0)
COLOUR_DOMINANT_NITS = (700.0, 660.0, 620.0, 700.0, 660.0, 620.0)
COLOUR_MINOR_NITS = 60.0
RAMP_SHIFT_PER_FRAME = 8                 # pixels, four chroma boxes
MASTERING_MONITOR_MAX_NITS = 1000.0
MASTERING_MONITOR_MIN_NITS = 0.005
MAX_CLL_NITS = 900                       # the authored peak, so the claim is measured
MAX_FALL_NITS = 250                      # above the authored frame mean, below peak


def pq_oetf(nits):
    """cd/m^2 -> ST 2084 component E' (BT.2100-2, normalised input form)."""
    relative = np.maximum(np.asarray(nits, dtype=np.float64) / PQ_PEAK_NITS, 0.0)
    em = relative ** PQ_M1
    return ((PQ_C1 + PQ_C2 * em) / (1.0 + PQ_C3 * em)) ** PQ_M2


def pq_eotf(coded):
    """ST 2084 component E' -> linear relative to 10000 cd/m^2."""
    coded = np.asarray(coded, dtype=np.float64)
    root = np.maximum(coded, 0.0) ** (1.0 / PQ_M2)
    return ((np.maximum(root - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * root))
            ** (1.0 / PQ_M1))


def _rgb_to_xyz(primaries, white=D65):
    """Unit-luminance primary columns scaled onto the shared white point."""
    xyz = np.empty((3, 3), dtype=np.float64)
    for index in range(3):
        x, y = primaries[2 * index], primaries[2 * index + 1]
        xyz[:, index] = np.array([x / y, 1.0, (1.0 - x - y) / y])
    target = np.array([white[0] / white[1], 1.0,
                       (1.0 - white[0] - white[1]) / white[1]])
    return xyz * np.linalg.solve(xyz, target)


def derive_contract():
    """Rec.2020 -> scRGB and Kr/Kg/Kb, derived the way the bridge derives them:
    the Y row of Rec.2020's own RGB->XYZ is the non-constant-luminance row."""
    xyz_709 = _rgb_to_xyz(BT709_PRIMARIES)
    xyz_2020 = _rgb_to_xyz(BT2020_PRIMARIES)
    return (np.linalg.inv(xyz_709) @ xyz_2020,
            float(xyz_2020[1, 0]), float(xyz_2020[1, 1]), float(xyz_2020[1, 2]))


TO_SCRGB, KR, KG, KB = derive_contract()
CR_GAIN = 2.0 * (1.0 - KR)
CB_GAIN = 2.0 * (1.0 - KB)
CG_CR = 2.0 * KR * (1.0 - KR) / KG
CG_CB = 2.0 * KB * (1.0 - KB) / KG
# Worst PQ-component drift a round-to-nearest limited-range code can introduce.
QUANTISATION_DRIFT_BOUND = 1.0 / Y_SCALE + max(CR_GAIN, CB_GAIN) / C_SCALE


def _rect(x, y, width, height):
    if x % 2 or y % 2 or width % 2 or height % 2:
        raise ValueError(f"patch {x},{y},{width},{height} is not 2x2 aligned")
    return slice(y, y + height), slice(x, x + width)


def build_scene():
    patches = []
    for index, nits in enumerate(GRAY_STAIR_NITS):
        patches.append({"name": f"grey_stair_{index}", "kind": "grey",
                        "rect": [64, 64 + index * 112, 128, 96],
                        "nits_by_frame": [[nits, nits, nits]] * FRAMES})
    patches.append({"name": "luma_ramp", "kind": "ramp",
                    "rect": [224, 64, 256, 880]})
    layout = [("colour_red", (1, 0, 0)), ("colour_green", (0, 1, 0)),
              ("colour_blue", (0, 0, 1)), ("colour_cyan", (0, 1, 1)),
              ("colour_magenta", (1, 0, 1)), ("colour_yellow", (1, 1, 0))]
    positions = [(544, 200), (960, 200), (1376, 200),
                 (544, 616), (960, 616), (1376, 616)]
    for (name, mask), (x, y) in zip(layout, positions):
        patches.append({"name": name, "kind": "colour", "rect": [x, y, 192, 192],
                        "mask": list(mask),
                        "nits_by_frame": [[dominant if bit else COLOUR_MINOR_NITS
                                           for bit in mask]
                                          for dominant in COLOUR_DOMINANT_NITS]})
    return patches


def author_frame(patches, frame):
    """Known linear BT.2020 RGB in cd/m^2 for one frame."""
    nits = np.full((HEIGHT, WIDTH, 3), SURROUND_NITS, dtype=np.float64)
    for patch in patches:
        rows, columns = _rect(*patch["rect"])
        if patch["kind"] == "ramp":
            x, y, width, height = patch["rect"]
            step = ((np.arange(width, dtype=np.float64)
                     + RAMP_SHIFT_PER_FRAME * frame) % width) / (width - 1)
            ramp = SCENE_FLOOR_NITS * (SCENE_PEAK_NITS / SCENE_FLOOR_NITS) ** step
            nits[rows, columns, :] = ramp[None, :, None]
        else:
            nits[rows, columns, :] = np.array(patch["nits_by_frame"][frame],
                                              dtype=np.float64)
    return nits


def author_codes(nits):
    """nits -> PQ -> BT.2020 NCL -> limited-range 10-bit 4:2:0 codes, unclamped."""
    coded = pq_oetf(nits)
    y_prime = KR * coded[:, :, 0] + KG * coded[:, :, 1] + KB * coded[:, :, 2]
    cb_prime = (coded[:, :, 2] - y_prime) / CB_GAIN
    cr_prime = (coded[:, :, 0] - y_prime) / CR_GAIN

    def boxed(values):
        blocks = values.reshape(HEIGHT // 2, 2, WIDTH // 2, 2)
        own = blocks[:, 0:1, :, 0:1]          # each box's own top-left cell
        drift = float(np.max(np.abs(blocks - own)))
        if drift > 1e-9:
            raise RuntimeError("a chroma box straddles two patches (drift "
                               f"{drift}); the patch edges are not 2x2 aligned")
        return blocks[:, 0, :, 0]

    y_code = np.rint(y_prime * Y_SCALE + Y_OFFSET).astype(np.uint16)
    cb_code = np.rint(boxed(cb_prime) * C_SCALE + C_OFFSET).astype(np.uint16)
    cr_code = np.rint(boxed(cr_prime) * C_SCALE + C_OFFSET).astype(np.uint16)
    return coded, y_code, cb_code, cr_code


def reconstruct(y_code, cb_code, cr_code, phase_y=0, phase_x=0):
    """Codes -> PQ -> linear relative -> scRGB, as the bridge's decode pass does
    it, reading the chroma box at the requested neighbour phase."""
    rows = np.clip(np.arange(HEIGHT // 2) + phase_y, 0, HEIGHT // 2 - 1)
    columns = np.clip(np.arange(WIDTH // 2) + phase_x, 0, WIDTH // 2 - 1)
    cb = cb_code[np.ix_(rows, columns)]
    cr = cr_code[np.ix_(rows, columns)]

    def upsample(plane):
        return np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1)

    y_prime = (y_code.astype(np.float64) - Y_OFFSET) / Y_SCALE
    cb_prime = (upsample(cb).astype(np.float64) - C_OFFSET) / C_SCALE
    cr_prime = (upsample(cr).astype(np.float64) - C_OFFSET) / C_SCALE
    encoded = np.stack([y_prime + CR_GAIN * cr_prime,
                        y_prime - CG_CB * cb_prime - CG_CR * cr_prime,
                        y_prime + CB_GAIN * cb_prime], axis=-1)
    relative = pq_eotf(encoded)
    return encoded, relative, relative @ TO_SCRGB.T * SCRGB_PER_RELATIVE


def _finite_range(values):
    finite = np.isfinite(values)
    held = values[finite]
    return float(held.min()), float(held.max()), int((~finite).sum())


def check_frame(coded, y_code, cb_code, cr_code):
    """Every physical claim this fixture has to satisfy, measured on the samples
    that will actually be coded. Nothing here clamps: a violation stops the run.

    Two sets of numbers are reported, because they answer different questions.
    `own_phase` is the chroma geometry the product reads (each pixel with its own
    box's chroma), so it is what the native source report can be compared against.
    `worst_case_phases` pairs every luma sample with the chroma of the boxes around
    its own, which is the range a different chroma siting or an interpolating
    upsampler could produce; it is bounded too, and it is where the colour
    adjacency shows up.
    """
    authored_low, authored_high, authored_bad = _finite_range(coded)
    keys = ("pq", "relative", "nits", "scrgb")
    bounds = {key: [np.inf, -np.inf] for key in keys}
    own = {key: None for key in keys}
    nonfinite = authored_bad
    drift = 0.0
    for phase_y in (-1, 0, 1):
        for phase_x in (-1, 0, 1):
            encoded, relative, scrgb = reconstruct(y_code, cb_code, cr_code,
                                                   phase_y, phase_x)
            measured = {"pq": encoded, "relative": relative, "scrgb": scrgb,
                        "nits": relative * PQ_PEAK_NITS}
            for key, values in measured.items():
                minimum, maximum, bad = _finite_range(values)
                nonfinite += bad
                bounds[key][0] = min(bounds[key][0], minimum)
                bounds[key][1] = max(bounds[key][1], maximum)
                if (phase_y, phase_x) == (0, 0):
                    own[key] = [minimum, maximum]
                    drift = float(np.abs(encoded - coded).max())
    problems = []
    if nonfinite:
        problems.append(f"{nonfinite} non-finite reconstructed samples")
    if bounds["pq"][0] < 0.0:
        problems.append(f"PQ component below the BT.2100 floor 0: {bounds['pq'][0]}")
    if bounds["pq"][1] > 1.0:
        problems.append(f"PQ component above the BT.2100 top 1: {bounds['pq'][1]}")
    if bounds["relative"][1] > 1.0:
        problems.append(f"normalised linear above 1: {bounds['relative'][1]}")
    if drift > QUANTISATION_DRIFT_BOUND:
        problems.append(f"PQ drift {drift} exceeds the code step bound "
                        f"{QUANTISATION_DRIFT_BOUND}")
    if not (y_code.min() >= Y_OFFSET and y_code.max() <= Y_TOP
            and cb_code.min() >= 64 and cb_code.max() <= C_TOP
            and cr_code.min() >= 64 and cr_code.max() <= C_TOP):
        problems.append(f"codes outside limited range: Y [{y_code.min()}, "
                        f"{y_code.max()}] U [{cb_code.min()}, {cb_code.max()}] "
                        f"V [{cr_code.min()}, {cr_code.max()}]")
    if problems:
        raise RuntimeError("fixture is not physically valid: " + "; ".join(problems))
    return {"authored_pq_min": authored_low, "authored_pq_max": authored_high,
            "own_phase": {key: {"min": own[key][0], "max": own[key][1]}
                          for key in keys},
            "worst_case_chroma_phases": {key: {"min": bounds[key][0],
                                               "max": bounds[key][1]} for key in keys},
            "nonfinite_samples": nonfinite,
            "quantised_pq_max_abs_error": drift,
            "quantised_pq_drift_bound": QUANTISATION_DRIFT_BOUND,
            "chroma_phases_checked": 9,
            "code_ranges": {"y": [int(y_code.min()), int(y_code.max())],
                            "cb": [int(cb_code.min()), int(cb_code.max())],
                            "cr": [int(cr_code.min()), int(cr_code.max())]}}


def planes_to_bytes(y_code, cb_code, cr_code):
    return b"".join([y_code.tobytes(), cb_code.tobytes(), cr_code.tobytes()])


def x265_params():
    def point(xy):
        return "({},{})".format(int(round(xy[0] * 50000)), int(round(xy[1] * 50000)))
    master = ("master-display=G{}B{}R{}WP{}L({},{})".format(
        point(BT2020_PRIMARIES[2:4]), point(BT2020_PRIMARIES[4:6]),
        point(BT2020_PRIMARIES[0:2]), point(D65),
        int(round(MASTERING_MONITOR_MAX_NITS * 10000)),
        int(round(MASTERING_MONITOR_MIN_NITS * 10000))))
    return ":".join(["lossless=1", "colorprim=bt2020", "transfer=smpte2084",
                     "colormatrix=bt2020nc", "range=limited", master,
                     f"max-cll={MAX_CLL_NITS},{MAX_FALL_NITS}",
                     "repeat-headers=1", "pools=1", "frame-threads=1",
                     "log-level=info"])


def encode(raw_path, target, log_path, elementary=False):
    argv = [FFMPEG, "-hide_banner", "-y", "-loglevel", "info",
            "-fflags", "+bitexact", "-flags:v", "+bitexact",
            "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
            "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", str(raw_path),
            "-c:v", "libx265", "-preset", "medium", "-pix_fmt", "yuv420p10le",
            "-frames:v", str(FRAMES), "-x265-params", x265_params()]
    if elementary:
        argv += ["-f", "hevc"]
    argv.append(str(target))
    done = subprocess.run(argv, capture_output=True, text=True)
    log_path.write_text(done.stderr or "", encoding="utf-8", errors="replace")
    if done.returncode:
        raise RuntimeError(f"ffmpeg exited {done.returncode} for {target}; "
                           f"log kept at {log_path}")
    lines = [line.strip() for line in (done.stderr or "").splitlines()
             if any(word in line for word in ("Rate Control", "Main 10", "encoded",
                                              "bit-depth", "Time to encode"))]
    return {"argv": [str(item) for item in argv], "rc": done.returncode,
            "report_lines": lines[:8]}


def probe(target):
    ffprobe = Path(FFMPEG).with_name("ffprobe.exe")
    argv = [str(ffprobe), "-v", "error",
            "-select_streams", "v:0", "-show_entries",
            "stream=codec_name,profile,level,pix_fmt,width,height,r_frame_rate,"
            "nb_frames,color_range,color_primaries,color_transfer,color_space,"
            "bits_per_raw_sample", "-show_frames", "-read_intervals", "%+" + str(FRAMES),
            "-of", "json", str(target)]
    done = subprocess.run(argv, capture_output=True, text=True)
    if done.returncode:
        return {"ran": False, "rc": done.returncode, "stderr": done.stderr[:400]}
    data = json.loads(done.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    side = []
    for frame in data.get("frames") or []:
        for item in frame.get("side_data_list") or []:
            if item not in side:
                side.append(item)
    return {"ran": True, "rc": done.returncode, "stream": stream,
            "frame_side_data": side, "argv": [str(item) for item in argv]}


def sha256(path):
    from hashlib import sha256
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replace", action="store_true",
                        help="overwrite an existing hdr_valid_src.mkv")
    arguments = parser.parse_args()
    if OUT_FIXTURE.exists() and not arguments.replace:
        raise SystemExit(f"refusing to overwrite an existing fixture: {OUT_FIXTURE} "
                         "(use --replace)")
    if OUT_FIXTURE.resolve() == OLD_FIXTURE.resolve():
        raise SystemExit("the old fixture path must never be the output")
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    patches = build_scene()
    frames, checks = [], []
    reference = None
    for frame in range(FRAMES):
        nits = author_frame(patches, frame)
        coded, y_code, cb_code, cr_code = author_codes(nits)
        checks.append(check_frame(coded, y_code, cb_code, cr_code))
        frames.append(planes_to_bytes(y_code, cb_code, cr_code))
        if frame == 0:
            reference = (nits.copy(), coded.copy(), y_code.copy(),
                         cb_code.copy(), cr_code.copy())

    raw_path = REFERENCE_DIR / "_encoder_input_all6.yuv"
    raw_path.write_bytes(b"".join(frames))
    staged = REFERENCE_DIR / "_staged.mkv"
    twin = REFERENCE_DIR / "_staged_twin.mkv"
    es_arm_a, es_arm_b = REFERENCE_DIR / "_stream_a.265", REFERENCE_DIR / "_stream_b.265"
    first = encode(raw_path, staged, REFERENCE_DIR / "generator_encode_log.txt")
    encode(raw_path, twin, REFERENCE_DIR / "_twin_encode_log.txt")
    encode(raw_path, es_arm_a, REFERENCE_DIR / "_stream_a_log.txt", elementary=True)
    encode(raw_path, es_arm_b, REFERENCE_DIR / "_stream_b_log.txt", elementary=True)
    staged_sha, twin_sha = sha256(staged), sha256(twin)
    stream_sha_a, stream_sha_b = sha256(es_arm_a), sha256(es_arm_b)
    differing = sum(1 for x, y in zip(staged.read_bytes(), twin.read_bytes())
                    if x != y)
    staged.replace(OUT_FIXTURE)
    for leftover in (twin, es_arm_a, es_arm_b, raw_path,
                     REFERENCE_DIR / "_twin_encode_log.txt",
                     REFERENCE_DIR / "_stream_a_log.txt",
                     REFERENCE_DIR / "_stream_b_log.txt"):
        leftover.unlink()

    nits, coded, y_code, cb_code, cr_code = reference
    authored_path = REFERENCE_DIR / "authored_f0.yuv"
    authored_path.write_bytes(planes_to_bytes(y_code, cb_code, cr_code))

    patch_table = []
    for patch in patches:
        entry = dict(patch)
        if patch["kind"] != "ramp":
            rows, columns = _rect(*patch["rect"])
            block = coded[rows, columns]
            entry["pq_by_frame_0"] = [[float(block[..., 0].min()),
                                       float(block[..., 0].max())],
                                      [float(block[..., 1].min()),
                                       float(block[..., 1].max())],
                                      [float(block[..., 2].min()),
                                       float(block[..., 2].max())]]
        patch_table.append(entry)

    positive_row_sums = np.maximum(TO_SCRGB, 0.0).sum(axis=1)
    record = {
        "artifact_class": ("offline CPU fixture authoring - reference test data, "
                           "not a production host image transfer"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {"path": str(Path(__file__).resolve()),
                      "sha256": sha256(Path(__file__).resolve()),
                      "numpy": np.__version__, "python": sys.version.split()[0]},
        "constants": {"pq_m1": PQ_M1, "pq_m2": PQ_M2, "pq_c1": PQ_C1,
                      "pq_c2": PQ_C2, "pq_c3": PQ_C3,
                      "scrgb_per_nit": 1.0 / REFERENCE_WHITE_NITS,
                      "scrgb_per_relative": SCRGB_PER_RELATIVE,
                      "kr": KR, "kg": KG, "kb": KB, "cr_gain": CR_GAIN,
                      "cb_gain": CB_GAIN, "cg_cr": CG_CR, "cg_cb": CG_CB,
                      "y_offset": Y_OFFSET, "y_scale": Y_SCALE,
                      "c_offset": C_OFFSET, "c_scale": C_SCALE,
                      "reference_white_nits": REFERENCE_WHITE_NITS,
                      "pq_peak_nits": PQ_PEAK_NITS},
        "to_scrgb_matrix": TO_SCRGB.tolist(),
        "physical_bound": {
            "valid_linear2020_max_scrgb": SCRGB_PER_RELATIVE,
            "rec2020_to_709_positive_row_sums": positive_row_sums.tolist(),
            "max_positive_row_sum": float(positive_row_sums.max()),
            "valid_peak_scrgb_after_primary_conversion":
                float(positive_row_sums.max() * SCRGB_PER_RELATIVE),
            "note": ("1.0 PQ is 10000 cd/m^2 and this bridge's scRGB 1.0 is 80 "
                     "cd/m^2, so a BT.2100-valid sample cannot exceed 125 scRGB "
                     "before the primaries conversion; the largest positive row "
                     "sum of the derived Rec.2020->Rec.709 matrix bounds what it "
                     "can exceed that by. This is a bound on legal content, not "
                     "a metadata copy and not a MaxCLL equivalent.")},
        "scene": {"width": WIDTH, "height": HEIGHT, "frames": FRAMES, "fps": FPS,
                  "surround_nits": SURROUND_NITS, "floor_nits": SCENE_FLOOR_NITS,
                  "peak_nits": SCENE_PEAK_NITS,
                  "colour_dominant_nits_by_frame": list(COLOUR_DOMINANT_NITS),
                  "colour_minor_nits": COLOUR_MINOR_NITS,
                  "ramp_shift_pixels_per_frame": RAMP_SHIFT_PER_FRAME,
                  "patches": patch_table},
        "declared_metadata": {
            "color_primaries": "bt2020", "color_transfer": "smpte2084",
            "color_space": "bt2020nc", "color_range": "tv (limited)",
            "pix_fmt": "yuv420p10le", "profile": "Main 10",
            "mastering_primaries_cie_xy": BT2020_PRIMARIES.tolist(),
            "mastering_white_point_cie_xy": D65.tolist(),
            "mastering_monitor_max_nits": MASTERING_MONITOR_MAX_NITS,
            "mastering_monitor_min_nits": MASTERING_MONITOR_MIN_NITS,
            "max_cll_nits": MAX_CLL_NITS, "max_fall_nits": MAX_FALL_NITS,
            "max_cll_is_authored_peak": MAX_CLL_NITS == SCENE_PEAK_NITS},
        "per_frame_checks": checks,
        "fixture": {"path": str(OUT_FIXTURE), "sha256": sha256(OUT_FIXTURE),
                    "bytes": OUT_FIXTURE.stat().st_size,
                    "container_probe": probe(OUT_FIXTURE)},
        "encode": first,
        "reproducible": {
            "coded_bitstream_deterministic": bool(stream_sha_a == stream_sha_b),
            "elementary_stream_sha256_arms": [stream_sha_a, stream_sha_b],
            "container_deterministic": bool(staged_sha == twin_sha),
            "container_bytes_differing_between_arms": differing,
            "fixture_sha256": staged_sha,
            "twin_container_sha256": twin_sha,
            "note": ("four arms over the same authored planes: two Matroska and two "
                     "raw-elementary-stream. libx265 lossless with pools=1 and "
                     "frame-threads=1 produced byte-identical coded samples, so the "
                     "bitstream is deterministic; Matroska also writes a DateUTC "
                     "element this build cannot be asked to omit, which is the only "
                     "difference between the two container arms and the reason the "
                     "fixture file hash is per-run.")},
        "reference_planes": {"path": str(authored_path),
                             "sha256": sha256(authored_path),
                             "bytes": authored_path.stat().st_size,
                             "layout": ("frame 0 only, yuv420p10le planar, 10-bit "
                                        "codes in the low bits of each uint16")},
        "old_fixture_untouched": {"path": str(OLD_FIXTURE),
                                  "exists": OLD_FIXTURE.is_file(),
                                  "sha256": (sha256(OLD_FIXTURE)
                                             if OLD_FIXTURE.is_file() else None)},
    }
    (REFERENCE_DIR / "authored_scene.json").write_text(
        json.dumps(record, indent=2, default=float), encoding="utf-8")

    print(f"fixture   : {OUT_FIXTURE} "
          f"({OUT_FIXTURE.stat().st_size} bytes, sha256 {record['fixture']['sha256']})")
    print(f"reference : {authored_path}")
    print(f"scene json: {REFERENCE_DIR / 'authored_scene.json'}")
    print(f"bitstream deterministic: "
          f"{record['reproducible']['coded_bitstream_deterministic']} "
          f"(container arms differ in "
          f"{record['reproducible']['container_bytes_differing_between_arms']} bytes)")
    for index, check in enumerate(checks):
        own, worst = check["own_phase"], check["worst_case_chroma_phases"]
        print(f"frame {index}: authored PQ [{check['authored_pq_min']:.6f}, "
              f"{check['authored_pq_max']:.6f}] own-phase PQ "
              f"[{own['pq']['min']:.6f}, {own['pq']['max']:.6f}] nits "
              f"[{own['nits']['min']:.2f}, {own['nits']['max']:.2f}] scRGB "
              f"[{own['scrgb']['min']:.4f}, {own['scrgb']['max']:.4f}] | worst "
              f"chroma phase PQ [{worst['pq']['min']:.6f}, {worst['pq']['max']:.6f}] "
              f"nits {worst['nits']['max']:.2f} | nonfinite "
              f"{check['nonfinite_samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
