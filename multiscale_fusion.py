"""Experimental, matrix-free reference for multi-scale residual consistency.

This is a numerical reference for the GPU implementation, not an export path:
it operates on host arrays and must never be called for production frames.
The operator is the *linear filter* in the production Lanczos-2 plan. Its
RGBA8 output quantization is deliberately outside D: quantization has no
linear adjoint, while the signed residual observation is represented in pixel
units after that quantization.
"""

from dataclasses import dataclass

import numpy as np

import pipeline


class FusionError(RuntimeError):
    """An invalid objective or an unconverged numerical solve."""


@dataclass(frozen=True)
class LanczosObservation:
    y: pipeline.AxisPlan
    x: pipeline.AxisPlan
    residual: np.ndarray
    weight: float

    @classmethod
    def from_extents(cls, full_width, full_height, residual, weight):
        if residual.ndim != 3 or residual.shape[2] != 3:
            raise ValueError("signed residual must be HxWx3")
        low_height, low_width = residual.shape[:2]
        if not (0 < low_width <= full_width and 0 < low_height <= full_height):
            raise ValueError("observation extent must fit the full-resolution image")
        return cls(
            pipeline._build_lanczos2_axis_plan(full_height, low_height),
            pipeline._build_lanczos2_axis_plan(full_width, low_width),
            np.asarray(residual, dtype=np.float32), float(weight),
        )

    def down(self, full):
        if full.shape != (self.y.src_len, self.x.src_len, 3):
            raise ValueError("full residual extent differs from the Lanczos plans")
        vertical = pipeline._apply_axis_plan(np.asarray(full, dtype=np.float32),
                                             self.y, axis=0)
        return pipeline._apply_axis_plan(vertical, self.x, axis=1)

    def adjoint(self, low):
        if low.shape != (self.y.dst_len, self.x.dst_len, 3):
            raise ValueError("observation extent differs from the Lanczos plans")
        # D = X Y, hence D^T = Y^T X^T. Scatter every plan tap, including
        # repeated edge indices; a regular resize is not the transpose.
        horizontal = _axis_adjoint(np.asarray(low, dtype=np.float32), self.x, 1)
        return _axis_adjoint(horizontal, self.y, 0)


def _axis_adjoint(values, plan, axis):
    if values.shape[axis] != plan.dst_len:
        raise ValueError("adjoint input extent differs from its axis plan")
    shape = list(values.shape)
    shape[axis] = plan.src_len
    result = np.zeros(shape, dtype=np.float32)
    # np.add.at accumulates duplicated clamped border taps, exactly as the
    # transpose of the forward gather requires.
    for tap in range(plan.max_taps):
        if axis == 0:
            np.add.at(result, plan.indices[:, tap],
                      values * plan.weights[:, tap, None, None])
        else:
            np.add.at(result.transpose(1, 0, 2), plan.indices[:, tap],
                      values.transpose(1, 0, 2) *
                      plan.weights[:, tap, None, None])
    return result


def consistency_objective(full, observations, baseline, beta):
    total = float(beta) * float(np.sum((full - baseline).astype(np.float64) ** 2))
    for item in observations:
        total += item.weight * float(np.sum(
            (item.down(full) - item.residual).astype(np.float64) ** 2))
    return total


def solve_reference(observations, baseline, beta, *, max_iterations,
                    relative_tolerance):
    """Solve sum(w||D r-r_i||²)+beta||r-r0||² with explicit CG failure.

    The returned metrics are data-consistency and solver evidence only; they
    make no claim about perceptual quality or runtime performance.
    """
    if not observations or not np.isfinite(beta) or beta <= 0:
        raise ValueError("at least one observation and beta > 0 are required")
    if max_iterations < 1 or not (0 < relative_tolerance < 1):
        raise ValueError("max_iterations >= 1 and 0 < tolerance < 1 required")
    if any(not np.isfinite(item.weight) or item.weight <= 0
           for item in observations):
        raise ValueError("every layer weight must be finite and positive")
    baseline = np.asarray(baseline, dtype=np.float32)
    if baseline.ndim != 3 or baseline.shape[2] != 3 or not np.isfinite(baseline).all():
        raise ValueError("baseline must be finite HxWx3")
    if any(item.y.src_len != baseline.shape[0] or
           item.x.src_len != baseline.shape[1] or
           not np.isfinite(item.residual).all() for item in observations):
        raise ValueError("observation and baseline extents or values disagree")

    def h(vector):
        result = np.float32(beta) * vector
        for item in observations:
            result = result + np.float32(item.weight) * item.adjoint(item.down(vector))
        return result

    rhs = np.float32(beta) * baseline
    for item in observations:
        rhs = rhs + np.float32(item.weight) * item.adjoint(item.residual)
    result = baseline.copy()
    residual = rhs - h(result)
    direction = residual.copy()
    square = float(np.sum(residual.astype(np.float64) ** 2))
    initial = square ** 0.5
    if initial == 0:
        return result, {"iterations": 0, "relative_residual": 0.0}
    for iteration in range(1, max_iterations + 1):
        h_direction = h(direction)
        divisor = float(np.sum(direction.astype(np.float64) *
                               h_direction.astype(np.float64)))
        if not np.isfinite(divisor) or divisor <= 0:
            raise FusionError(f"CG lost positive definiteness at iteration {iteration}")
        step = square / divisor
        result = result + np.float32(step) * direction
        residual = residual - np.float32(step) * h_direction
        new_square = float(np.sum(residual.astype(np.float64) ** 2))
        relative = new_square ** 0.5 / initial
        if relative <= relative_tolerance:
            return result, {"iterations": iteration,
                            "relative_residual": relative}
        direction = residual + np.float32(new_square / square) * direction
        square = new_square
    raise FusionError(f"CG did not converge in {max_iterations} iterations; "
                      f"relative residual={relative:.6g}, "
                      f"required<={relative_tolerance:.6g}")
