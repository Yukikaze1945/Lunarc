import numpy as np
import pytest

from multiscale_fusion import (
    FusionError, LanczosObservation, consistency_objective, solve_reference,
)
from gpu_transfer_export import multi_scale_config, _frame_timestamp_discontinuity


def test_lanczos_adjoint_and_consistency_solve():
    rng = np.random.default_rng(2103)
    truth = rng.normal(size=(32, 48, 3)).astype(np.float32)
    probe = rng.normal(size=truth.shape).astype(np.float32)
    observations = []
    for height, width in ((16, 24), (24, 36)):
        blank = np.zeros((height, width, 3), dtype=np.float32)
        plan = LanczosObservation.from_extents(48, 32, blank, 1.0)
        low = rng.normal(size=blank.shape).astype(np.float32)
        lhs = float(np.sum(plan.down(probe).astype(np.float64) * low))
        rhs = float(np.sum(probe.astype(np.float64) * plan.adjoint(low)))
        assert lhs == pytest.approx(rhs, rel=2e-6, abs=2e-5)
        observations.append(LanczosObservation(plan.y, plan.x,
                                                plan.down(truth), 1.0))

    baseline = np.zeros_like(truth)
    before = sum(np.sum((item.down(baseline) - item.residual) ** 2)
                 for item in observations)
    result, metrics = solve_reference(
        observations, baseline, 0.05, max_iterations=120,
        relative_tolerance=1e-5)
    after = sum(np.sum((item.down(result) - item.residual) ** 2)
                for item in observations)
    assert after < before
    assert consistency_objective(result, observations, baseline, 0.05) < \
        consistency_objective(baseline, observations, baseline, 0.05)
    assert metrics["relative_residual"] <= 1e-5


def test_unconverged_solver_reports_failure():
    rng = np.random.default_rng(32)
    blank = np.zeros((16, 24, 3), dtype=np.float32)
    plan = LanczosObservation.from_extents(48, 32, blank, 1.0)
    item = LanczosObservation(plan.y, plan.x,
                              rng.normal(size=blank.shape).astype(np.float32), 1.0)
    with pytest.raises(FusionError, match="did not converge"):
        solve_reference([item], np.zeros((32, 48, 3), dtype=np.float32),
                        0.05, max_iterations=1, relative_tolerance=1e-10)


def test_layer_vector_uses_resolved_dimensions_and_rejects_duplicates():
    config = multi_scale_config({"working_scale": 0.4,
        "reconstruction_mode": "triple",
        "layer_scales": [0.4, 0.67, 1.0],
        "layer_weights": [1, 2, 1],
        "fusion_mode": "data_consistency_experimental"}, 1280, 720)
    assert config["dims"] == [(512, 288), (856, 480), (1280, 720)]
    assert config["weights"] == [1.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="distinct"):
        multi_scale_config({"working_scale": 0.5,
            "reconstruction_mode": "dual", "layer_scales": [0.5, 0.5]},
            1280, 720)


def test_single_mode_ignores_saved_experimental_layer_values():
    config = multi_scale_config({"working_scale": 0.4,
        "reconstruction_mode": "single", "layer_scales": [0.5, 0.75],
        "layer_weights": [1, 2],
        "fusion_mode": "data_consistency_experimental"}, 1280, 720)
    assert config["scales"] == [0.4]
    assert config["dims"] == [(512, 288)]
    assert config["fusion"] == "catmull_baseline"


def test_timestamp_gap_resets_all_layers_on_same_frame():
    class Frame:
        time_base_num = 1
        time_base_den = 1000

        def __init__(self, pts):
            self.pts = pts

    assert _frame_timestamp_discontinuity(None, Frame(0), (24, 1))
    assert not _frame_timestamp_discontinuity((0, 1, 1000), Frame(42), (24, 1))
    assert _frame_timestamp_discontinuity((42, 1, 1000), Frame(125), (24, 1))
    assert _frame_timestamp_discontinuity((125, 1, 1000), Frame(100), (24, 1))
