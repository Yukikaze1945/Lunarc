import numpy as np
import pytest

from multiscale_fusion import (
    FusionError, LanczosObservation, jacobi_diagonal, solve_optimized_reference,
)


def matrix(item):
    def axis(plan):
        out = np.zeros((plan.dst_len, plan.src_len))
        for row in range(plan.dst_len):
            np.add.at(out[row], plan.indices[row], plan.weights[row])
        return out
    return np.kron(axis(item.y), axis(item.x))


def problem():
    rng = np.random.default_rng(2811)
    baseline = rng.normal(size=(9, 13, 3))*3
    observations = [LanczosObservation.from_extents(13, 9,
        rng.normal(size=(h, w, 3)).astype(np.float32)*5, weight)
        for h, w, weight in ((4, 6, .7), (6, 9, 1.8))]
    beta = .17
    h = beta*np.eye(9*13)
    b = beta*baseline.reshape(-1, 3)
    for item in observations:
        d = matrix(item)
        h += item.weight*d.T@d
        b += item.weight*d.T@item.residual.reshape(-1, 3)
    return observations, baseline, beta, h, b


@pytest.mark.parametrize('space', ['primal', 'observation', 'auto'])
@pytest.mark.parametrize('preconditioner', ['none', 'jacobi'])
def test_spaces_preserve_coupled_objective_and_true_residual(space, preconditioner):
    obs, baseline, beta, h, b = problem()
    result, metrics = solve_optimized_reference(obs, baseline, beta, space=space,
        preconditioner=preconditioner, relative_tolerance=1e-8, max_iterations=160)
    expected = np.linalg.solve(h, b).reshape(baseline.shape)
    np.testing.assert_allclose(result, expected, rtol=2e-7, atol=2e-7)
    truth = np.linalg.norm(b-h@result.reshape(-1, 3))
    gauge = np.linalg.norm(b-h@baseline.reshape(-1, 3))
    assert truth/gauge <= 1e-8
    assert metrics['true_residual_norm'] == pytest.approx(truth, rel=2e-6)
    if space in ('observation', 'auto'):
        assert metrics['vector_elements'] < baseline.size


def test_diagonals_merge_boundary_duplicates_before_squaring():
    obs, baseline, beta, h, _ = problem()
    actual = jacobi_diagonal(obs, beta).reshape(baseline.shape)[..., 0].reshape(-1)
    np.testing.assert_allclose(actual, np.diag(h), rtol=2e-15, atol=2e-15)
    a = np.vstack([np.sqrt(item.weight)*matrix(item) for item in obs])
    actual = jacobi_diagonal(obs, beta, space='observation').reshape(-1, 3)[:, 0]
    np.testing.assert_allclose(actual, beta+np.sum(a*a, axis=1), rtol=2e-15, atol=2e-15)


def test_warm_start_changes_guess_but_keeps_current_baseline_objective():
    obs, baseline, beta, h, b = problem()
    expected = np.linalg.solve(h, b).reshape(baseline.shape)
    guess = expected+.001
    result, metrics = solve_optimized_reference(obs, baseline, beta,
        initial_guess=guess, preconditioner='jacobi', relative_tolerance=1e-8)
    np.testing.assert_allclose(result, expected, rtol=3e-7, atol=3e-7)
    assert metrics['space'] == 'primal'
    assert metrics['reference_residual_norm'] == pytest.approx(
        np.linalg.norm(b-h@baseline.reshape(-1, 3)))
    with pytest.raises(ValueError, match='primal'):
        solve_optimized_reference(obs, baseline, beta, initial_guess=guess, space='observation')


def test_budget_and_nonfinite_values_are_errors():
    obs, baseline, beta, _, _ = problem()
    with pytest.raises(FusionError, match='did not converge'):
        solve_optimized_reference(obs, baseline, beta, max_iterations=1, relative_tolerance=1e-12)
    with pytest.raises(ValueError):
        solve_optimized_reference(obs, baseline*np.nan, beta)


def test_auto_uses_full_space_when_observations_exceed_full_extent():
    rng = np.random.default_rng(4)
    baseline = np.zeros((9, 13, 3))
    obs = [LanczosObservation.from_extents(13, 9,
        rng.normal(size=(h, w, 3)).astype(np.float32), 1.)
        for h, w in ((8, 12), (7, 10))]
    _, metrics = solve_optimized_reference(obs, baseline, .1)
    assert metrics['space'] == 'primal'
