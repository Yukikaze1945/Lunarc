#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_final_production.py — regression tests for the phases that do not need a GPU.

1. NVOF wrapper counters (Phase B)            - skipped when NVOF is unavailable
2. Magpie reference residual math (Phase C)   - compared against resampler_oracle
3. Working resolution dimension rules         - resolve_working_dims behaviour

The tiny per-pixel numeric gates live in test_resampler_numeric.py. Anything here
that used to assert about source code text or dictionary values it wrote itself has
been removed: those assertions could not fail for a broken pipeline.
"""
import math
import os
import sys
import unittest

import numpy as np

_UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_UNIT_DIR))
for _p in (_ROOT, _UNIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline  # noqa: E402
from resampler_oracle import (  # noqa: E402
    catmull_rom,
    checkerboard,
    deterministic_random,
    impulse,
    quantized,
    scalar_catmull_rom_upsample,
    scalar_composite,
    scalar_lanczos2_downsample,
    signed_residual,
)
import optical_flow  # noqa: E402
from optical_flow import get_optical_flow_engine, get_frame_guidance_service  # noqa: E402


class TestNVOFWrapperFixes(unittest.TestCase):
    """Gate 4: the three-frame NVOF 4M counter regression, nothing else.

    A capability probe that reports NVOF available turns every later exception into
    a failure. These tests used to wrap their whole body in ``except Exception:
    skipTest()``, which is exactly how a counter regression stays invisible.
    """

    def setUp(self):
        self.probe = optical_flow.probe_nvof_capabilities()
        if not self.probe.get("available"):
            self.skipTest(f"NVOF unavailable per capability probe: {self.probe.get('error')}")

    def test_nvof_wrapper_exposes_native_history(self):
        engine = get_optical_flow_engine("nvof", quality="4M")
        self.assertTrue(engine.maintains_native_history,
                        "NVOF must report it keeps its own previous-frame history")

    def test_nvof_wrapper_forwards_counter_delta(self):
        engine = get_optical_flow_engine("nvof", quality="4M")
        delta = engine.get_counters_delta()
        for key in ("nvof_execute_count_delta", "nvof_h2d_copy_count_delta",
                    "nvof_h2d_bytes_delta", "nvof_d2h_copy_count_delta",
                    "nvof_d2h_bytes_delta"):
            self.assertIn(key, delta, f"Counter delta must include {key}")

    def test_nvof_three_frame_steady_state_counters(self):
        guidance = get_frame_guidance_service("nvof", quality="4M")
        h, w = 1080, 1920
        yy, xx = np.mgrid[0:h, 0:w]
        base = ((xx // 3 + yy // 3) % 200).astype(np.uint8)
        base = np.repeat(base[:, :, None], 3, axis=2).copy()

        timings = []
        try:
            for i in range(3):
                frame = np.roll(base, 6 * i, axis=1)
                _flow, timing = guidance.get_guidance(
                    frame_id=i, cur_bgr=frame, prev_bgr=None,
                    guidance_mode=1, is_discontinuous=(i == 0),
                )
                timings.append(timing)
        finally:
            guidance.close()

        labels = ("A", "B", "C")
        keys = ("nvof_execute_count_delta", "nvof_h2d_copy_count_delta", "nvof_h2d_bytes_delta",
                "nvof_d2h_copy_count_delta", "nvof_d2h_bytes_delta", "cpu_previous_frame_copy_bytes")
        for label, timing in zip(labels, timings):
            print(f"NVOF frame {label}: " + " ".join(f"{k}={timing.get(k)}" for k in keys), flush=True)

        for label, timing in zip(labels[1:], timings[1:]):
            self.assertTrue(timing["nvof_native_history"],
                            f"frame {label}: flow engine is not keeping GPU history, so the "
                            f"zero CPU copy below would only mean nothing was measured")
            self.assertEqual(timing["cpu_previous_frame_copy_bytes"], 0,
                             f"frame {label}: steady state must copy no previous frame on the CPU")
            self.assertGreater(timing["nvof_h2d_bytes_delta"], 0, f"frame {label}: no H2D traffic")
            self.assertGreater(timing["nvof_d2h_bytes_delta"], 0, f"frame {label}: no D2H traffic")
            self.assertEqual(timing["nvof_h2d_copy_count_delta"], 1,
                             f"frame {label}: steady state must upload exactly once, got "
                             f"{timing['nvof_h2d_copy_count_delta']}")
            self.assertEqual(timing["nvof_d2h_copy_count_delta"], 1,
                             f"frame {label}: steady state must download exactly once, got "
                             f"{timing['nvof_d2h_copy_count_delta']}")
            self.assertGreater(timing["nvof_execute_count_delta"], 0, f"frame {label}: NVOF did not execute")


class TestMagpieReferenceResidual(unittest.TestCase):
    """Phase C: Verify true Lanczos-2 and Catmull-Rom implementations."""
    
    def _assert_downsample_matches_oracle(self, src, dst_w, dst_h):
        got = pipeline.downsample_lanczos2(src, (dst_w, dst_h))
        expected = quantized(scalar_lanczos2_downsample(src, dst_w, dst_h))
        diff = np.abs(got.astype(np.float64) - expected.astype(np.float64))
        self.assertLessEqual(diff.max(), 1.0, f"max diff {diff.max()}")
        self.assertLess(diff.mean(), 0.05, f"mean diff {diff.mean()}")

    def test_lanczos2_native_matches_scalar_reference(self):
        """100x100 -> 50x50 per-pixel agreement, not just a non-zero max."""
        for maker in (impulse, checkerboard, deterministic_random):
            self._assert_downsample_matches_oracle(maker(100, 100), 50, 50)

    def test_catmull_rom_native_matches_scalar_reference(self):
        """Mid-size plan rows must equal the reference kernel, negative lobes included."""
        src_len, dst_len = 50, 100
        plan = pipeline._build_catmull_rom_axis_plan(src_len, dst_len)
        self.assertEqual(plan.indices.shape[0], dst_len)
        self.assertTrue((plan.tap_counts == 4).all(), plan.tap_counts)
        for d in (0, 1, 25, 50, 74, 99):
            position = (d + 0.5) * src_len / float(dst_len) - 0.5
            center = math.floor(position)
            ref = [catmull_rom(position - (center + o)) for o in (-1, 0, 1, 2)]
            total = sum(ref)
            ref = [w / total for w in ref]
            for o in range(4):
                self.assertAlmostEqual(float(plan.weights[d, o]), ref[o], places=5,
                                       msg=f"dest {d} tap {o}")
            expected_idx = [min(max(center + o, 0), src_len - 1) for o in (-1, 0, 1, 2)]
            self.assertEqual(plan.indices[d].tolist(), expected_idx, f"dest {d}")

    def test_catmull_rom_residual_reconstruction_matches_scalar_reference(self):
        """40x40 working -> 100x100 source, signed residual and composite."""
        low_w = low_h = 40
        high_w = high_h = 100
        backend = pipeline.PythonReferenceBackend(high_w, high_h, low_w, low_h)
        for kind in ("positive", "negative", "mixed"):
            rng = np.random.default_rng(5)
            source = rng.integers(0, 256, size=(high_h, high_w, 3), dtype=np.uint8)
            down_cur = rng.integers(0, 256, size=(low_h, low_w, 3), dtype=np.uint8)
            delta = signed_residual(low_h, low_w, kind).astype(np.int32)
            dlss_out = np.clip(down_cur.astype(np.int32) + delta, 0, 255).astype(np.uint8)

            out = np.zeros((high_h, high_w, 3), dtype=np.uint8)
            got = backend.reconstruct(source, down_cur, dlss_out, 1.0, out)
            residual_full = scalar_catmull_rom_upsample(
                dlss_out.astype(np.float64) - down_cur.astype(np.float64), high_w, high_h)
            expected = scalar_composite(source, residual_full, 1.0)
            diff = np.abs(got.astype(np.float64) - expected.astype(np.float64))
            self.assertLessEqual(diff.max(), 1.0, f"{kind}: max diff {diff.max()}")
            self.assertLess(diff.mean(), 0.05, f"{kind}: mean diff {diff.mean()}")
            if kind in ("negative", "mixed"):
                self.assertTrue((backend.context.res_diff_high_f32 < 0.0).any(),
                                "signed residual lost its negative part")

    def test_signed_residual_positive_negative(self):
        """A positive residual brightens and a negative one darkens the composite."""
        low_w = low_h = 20
        high_w = high_h = 40
        backend = pipeline.PythonReferenceBackend(high_w, high_h, low_w, low_h)
        source = np.full((high_h, high_w, 3), 128, dtype=np.uint8)
        down_cur = np.full((low_h, low_w, 3), 100, dtype=np.uint8)

        up = np.zeros((high_h, high_w, 3), dtype=np.uint8)
        backend.reconstruct(source, down_cur, np.full((low_h, low_w, 3), 140, np.uint8), 1.0, up)
        down = np.zeros((high_h, high_w, 3), dtype=np.uint8)
        backend.reconstruct(source, down_cur, np.full((low_h, low_w, 3), 60, np.uint8), 1.0, down)

        self.assertTrue((up > source).all(), "positive residual must brighten")
        self.assertTrue((down < source).all(), "negative residual must darken")
        self.assertAlmostEqual(float(up.mean()), 128.0 + 40.0, delta=1.0)
        self.assertAlmostEqual(float(down.mean()), 128.0 - 40.0, delta=1.0)

    def test_residual_strength_zero_one_one_point_five(self):
        """Strength scales the signed residual before the composite; results must differ."""
        low_w = low_h = 20
        high_w = high_h = 40
        backend = pipeline.PythonReferenceBackend(high_w, high_h, low_w, low_h)
        source = np.full((high_h, high_w, 3), 128, dtype=np.uint8)
        down_cur = np.full((low_h, low_w, 3), 100, dtype=np.uint8)
        dlss_out = np.full((low_h, low_w, 3), 140, dtype=np.uint8)

        results = {}
        for strength in (0.0, 1.0, 1.5):
            out = np.zeros((high_h, high_w, 3), dtype=np.uint8)
            backend.reconstruct(source, down_cur, dlss_out, strength, out)
            results[strength] = out.copy()

        residual_full = scalar_catmull_rom_upsample(
            dlss_out.astype(np.float64) - down_cur.astype(np.float64), high_w, high_h)
        for strength in (0.0, 1.0, 1.5):
            expected = scalar_composite(source, residual_full, strength)
            self.assertTrue(np.array_equal(results[strength], expected),
                            f"strength {strength} did not match the oracle")
        self.assertFalse(np.array_equal(results[0.0], results[1.0]))
        self.assertFalse(np.array_equal(results[1.0], results[1.5]))

    def test_working_scale_33(self):
        """Verify 33% working scale computes correct dimensions."""
        src_w, src_h = 3840, 2160
        ww, wh = pipeline.resolve_working_dims(src_w, src_h, 0.33)
        
        expected_w = round(3840 * 0.33 / 8) * 8
        expected_h = round(2160 * 0.33 / 8) * 8
        
        self.assertEqual(ww, expected_w)
        self.assertEqual(wh, expected_h)
    
    def test_working_scale_63(self):
        """Verify 63% working scale computes correct dimensions."""
        src_w, src_h = 3840, 2160
        ww, wh = pipeline.resolve_working_dims(src_w, src_h, 0.63)
        
        expected_w = round(3840 * 0.63 / 8) * 8
        expected_h = round(2160 * 0.63 / 8) * 8
        
        self.assertEqual(ww, expected_w)
        self.assertEqual(wh, expected_h)
    
    def test_working_scale_83(self):
        """Verify 83% working scale computes correct dimensions."""
        src_w, src_h = 3840, 2160
        ww, wh = pipeline.resolve_working_dims(src_w, src_h, 0.83)
        
        expected_w = round(3840 * 0.83 / 8) * 8
        expected_h = round(2160 * 0.83 / 8) * 8
        
        self.assertEqual(ww, expected_w)
        self.assertEqual(wh, expected_h)


if __name__ == "__main__":
    unittest.main(verbosity=2)
