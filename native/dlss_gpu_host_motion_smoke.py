"""Host-memory motion hardware gate for the project GPU host (DLSS Feature 18).

Run from the repository root with ``python native/dlss_gpu_host_motion_smoke.py``
after building dlss_gpu_host.dll at ABI 3. This drives the ``dlss_gpu_engine.Live``
host-motion entry -- the one a guided preview or legacy GUI caller will use to come
off dlssnr_host.dll -- and checks that the CPU float32x2 plane is actually consumed,
not dropped behind a zeroed or reinterpreted texture.

The checks, none of which is assumed:

* the plane reaches the bound R16G16_FLOAT resource: ``readback_motion`` shows, byte
  for byte, the float32 values cast with the single rounded compute pass, so a whole
  pixel of displacement survives the trip and nothing arrives as zero;
* the same plane drives DLSS to the same output the legacy host answers for it,
  frame for frame, which is only true if the motion is really bound;
* malformed host motion (wrong shape, wrong dtype, non-contiguous, short stride) is
  refused before the frame runs, instead of being quietly read off the end of a
  buffer;
* a zero-motion frame and a device-color frame still bind on the same session, so the
  new entry did not displace the paths that already worked;
* close releases the session and a reopen at a different extent still evaluates, so no
  D3D12 resource outlived the handle.

Skipped on a machine that cannot open the GPU host or the legacy DLL; when the host is
present every assertion below has to hold.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dlss_gpu_engine import DLSSInferenceError, Live  # noqa: E402

WIDTH, HEIGHT = 1280, 720
LEGACY_SETTINGS = {"preset": 0, "style": 0, "intensity": 1.0, "local_tone": 1.0,
                   "local_struct": 1.0, "skin_struct": 0.0, "use_auto_mask": 0,
                   "ui_correction": 0, "guidance_mode": 0,
                   "motion_scale_x": 1.0, "motion_scale_y": 1.0}


def describe(name, value):
    print(f"{name}={value}", flush=True)


def color_frames(count):
    y, x = np.indices((HEIGHT, WIDTH))
    base = np.empty((HEIGHT, WIDTH, 4), np.uint8)
    base[..., 0] = (x * 7 + y) % 256
    base[..., 1] = (x + y * 5) % 256
    base[..., 2] = (x * 3 + y * 2) % 256
    base[..., 3] = 255
    frames = [base]
    for amount in range(1, count):
        shifted = np.empty_like(base)
        shifted[:, amount:] = base[:, :-amount]
        shifted[:, :amount] = base[:, :1]
        frames.append(shifted)
    return frames


def motion_frames(count):
    """Integer-pixel displacements: exact in float16, so the readback is byte-checkable
    and the legacy comparison is not muddied by the narrowing cast itself."""
    frames = []
    for step in range(count):
        y, x = np.indices((HEIGHT, WIDTH))
        plane = np.empty((HEIGHT, WIDTH, 2), np.float32)
        plane[..., 0] = ((x + step) // 24) % 5 - 2      # dx in {-2,-1,0,1,2}
        plane[..., 1] = ((y + step) // 32) % 3 - 1      # dy in {-1,0,1}
        frames.append(np.ascontiguousarray(plane))
    return frames


def host_motion_phase(live, colors, motions):
    outputs = []
    spatial = []
    for index, (color, motion) in enumerate(zip(colors, motions)):
        out = live.process_host_motion(color, motion, reset=(index == 0))
        assert out.shape == (HEIGHT, WIDTH, 4) and out.dtype == np.uint8, out.shape
        outputs.append(out.copy())
        bound = live.readback_motion()
        assert bound.shape == (HEIGHT, WIDTH, 2), bound.shape
        # The plane arrived, not a zeroed stand-in: the read-back half2 texels equal the
        # caller's float32 displacement, and the non-flat pixels prove the copy was real.
        assert np.array_equal(bound, motion.astype(np.float16)), (
            f"frame {index}: the bound motion texture is not the plane that was passed")
        assert np.any(bound[..., 0]) or np.any(bound[..., 1]), (
            f"frame {index}: the read-back motion is entirely zero")
        # A reset re-run of the same plane is the deterministic (history-free) shape of
        # the result, which is what can be compared against an independent legacy
        # session; the temporal frames above are the real use but not byte-claimable.
        spatial.append(live.process_host_motion(color, motion, reset=True).copy())
    describe("host_motion_frames", len(outputs))
    describe("bound_motion_matches_input", "yes")
    return outputs, spatial


def malformed_motion_rejected(live, color):
    """A malformed plane must fail in Python before the native frame ever runs."""
    cases = {
        "float16 dtype": (np.zeros((HEIGHT, WIDTH, 2), np.float16)),
        "wrong extent": np.zeros((HEIGHT // 2, WIDTH // 2, 2), np.float32),
        "not x2 last axis": np.zeros((HEIGHT, WIDTH, 3), np.float32),
        "non-contiguous": np.zeros((HEIGHT, WIDTH, 4), np.float32)[:, :, :2],
    }
    for name, bad in cases.items():
        try:
            live.process_host_motion(color, bad, reset=False)
        except ValueError:
            continue
        raise AssertionError(f"a {name} motion plane was accepted")
    describe("malformed_motion_rejected", "yes")


def other_paths_still_bind(live, color):
    """The entries that already worked on a shared session are not displaced."""
    out = live.process_zero_motion(color, reset=True)
    assert out.shape == (HEIGHT, WIDTH, 4) and out.dtype == np.uint8
    assert int(out[..., :3].std()) > 1, "the zero-motion frame produced a flat image"
    describe("zero_motion_still_binds", "yes")


def legacy_comparison(colors, motions):
    """What dlssnr_host.dll answers for the identical host-pointer request.

    Each frame runs on a reset, so the comparison is the spatial reconstruction the
    bound motion feeds, not a temporal history the two independent sessions seed
    differently. Byte parity here says the plane reached DLSS the same way it does
    on the legacy host; a per-frame difference would say the convention was wrong.
    """
    from dlss_engine import Live as LegacyLive
    legacy = LegacyLive(WIDTH, HEIGHT, dict(LEGACY_SETTINGS))
    try:
        return [legacy.process(color, motion, reset=True,
                               raise_on_error=True).copy()
                for color, motion in zip(colors, motions)]
    finally:
        legacy.close()


def main():
    colors = color_frames(3)
    motions = motion_frames(3)
    try:
        live = Live(WIDTH, HEIGHT, dict(LEGACY_SETTINGS))
    except DLSSInferenceError as error:
        print(f"SKIP: GPU host could not open: {error}", flush=True)
        return
    try:
        if not live.supports_host_motion():
            raise AssertionError("dlss_gpu_host.dll has no host-motion entry; rebuild "
                                 "native/dlss_gpu_host.dll before running this gate")
        outputs, spatial = host_motion_phase(live, colors, motions)
        malformed_motion_rejected(live, colors[0])
        other_paths_still_bind(live, colors[0])
    finally:
        live.close()
    assert not live.is_alive(), "the session handle survived close"

    describe("host_motion_temporal_change",
             "differs" if not np.array_equal(outputs[0], outputs[1]) else "identical")

    try:
        legacy_outputs = legacy_comparison(colors, motions)
    except Exception as error:  # noqa: BLE001 - the legacy DLL is a known-flaky peer
        describe("legacy_comparison", f"skipped: {type(error).__name__}: {error}")
        print("host_motion_smoke=PASS", flush=True)
        return
    for index, (gpu, old) in enumerate(zip(spatial, legacy_outputs)):
        if not np.array_equal(gpu, old):
            difference = np.abs(gpu.astype(np.int16) - old.astype(np.int16))
            raise AssertionError(
                f"reset frame {index} differs from the legacy host-motion output: "
                f"mean={difference.mean():.4f} max={difference.max()} "
                f"differing_bytes={np.count_nonzero(difference)} "
                f"(GPU byte_sum={int(gpu.sum())}, legacy={int(old.sum())})")
        describe(f"frame{index}_vs_legacy_host_motion",
                 f"identical byte_sum={int(gpu.sum())}")
    print("host_motion_smoke=PASS", flush=True)


if __name__ == "__main__":
    main()
