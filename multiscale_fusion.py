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


def _axis_diagonal(plan, *, transpose):
    """diag(Y^T Y) or diag(Y Y^T), merging repeated clamped columns first."""
    result = np.zeros(plan.src_len if transpose else plan.dst_len, dtype=np.float64)
    for row in range(plan.dst_len):
        merged = {}
        for index, weight in zip(plan.indices[row], plan.weights[row]):
            merged[int(index)] = merged.get(int(index), 0.) + float(weight)
        for index, weight in merged.items():
            result[index if transpose else row] += weight * weight
    return result


def jacobi_diagonal(observations, beta, *, space="primal"):
    if space == "observation":
        # Cross-layer blocks are off-diagonal; each diagonal block is w_i D_i D_i^T.
        return np.concatenate([np.repeat((beta + item.weight * np.outer(
            _axis_diagonal(item.y, transpose=False),
            _axis_diagonal(item.x, transpose=False))).reshape(-1), 3)
            for item in observations])
    if space != "primal":
        raise ValueError("unknown solve space")
    result = np.full((observations[0].y.src_len, observations[0].x.src_len),
                     beta, dtype=np.float64)
    for item in observations:
        result += item.weight * np.outer(_axis_diagonal(item.y, transpose=True),
                                         _axis_diagonal(item.x, transpose=True))
    return np.repeat(result[..., None], 3, axis=2).reshape(-1)


def _filter64(values, plan, axis, transpose=False):
    values = np.asarray(values, dtype=np.float64)
    moved = np.moveaxis(values, axis, 0)
    if transpose:
        result = np.zeros((plan.src_len,) + moved.shape[1:], dtype=np.float64)
        for tap in range(plan.max_taps):
            np.add.at(result, plan.indices[:, tap],
                      moved * plan.weights[:, tap, None, None])
    else:
        result = np.zeros((plan.dst_len,) + moved.shape[1:], dtype=np.float64)
        for tap in range(plan.max_taps):
            result += moved[plan.indices[:, tap]] * plan.weights[:, tap, None, None]
    return np.moveaxis(result, 0, axis)


def solve_optimized_reference(observations, baseline, beta, *, max_iterations=128,
                              relative_tolerance=1e-7, space="auto",
                              preconditioner="none", initial_guess=None):
    """Numerical experiment, never used to process production host pictures.

    All variants certify ||b-Hx|| / ||b-Hr0||, including warm initial guesses.
    The baseline in the objective always remains the current frame's r0.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    if (not observations or not np.isfinite(beta) or beta <= 0 or
            baseline.ndim != 3 or baseline.shape[2] != 3 or
            not np.isfinite(baseline).all() or max_iterations < 1 or
            not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1):
        raise ValueError("invalid fusion objective or solver settings")
    if any(item.weight <= 0 or not np.isfinite(item.weight) or
           not np.isfinite(item.residual).all() or
           (item.y.src_len, item.x.src_len) != baseline.shape[:2]
           for item in observations):
        raise ValueError("invalid observation")
    shapes = [item.residual.shape for item in observations]
    sizes = [int(np.prod(shape)) for shape in shapes]
    offsets = np.cumsum([0] + sizes)
    if space == "auto":
        space = "observation" if sum(sizes) < baseline.size and initial_guess is None else "primal"
    if space not in ("observation", "primal") or preconditioner not in ("none", "jacobi"):
        raise ValueError("unknown solver space or preconditioner")

    def down(item, vector):
        return _filter64(_filter64(vector, item.y, 0), item.x, 1)

    def adjoint(item, vector):
        return _filter64(_filter64(vector, item.x, 1, True), item.y, 0, True)

    def a(vector):
        image = vector.reshape(baseline.shape)
        return np.concatenate([(np.sqrt(item.weight)*down(item, image)).reshape(-1)
                               for item in observations])

    def at(vector):
        result = np.zeros_like(baseline)
        for i, item in enumerate(observations):
            result += np.sqrt(item.weight)*adjoint(item,
                vector[offsets[i]:offsets[i+1]].reshape(shapes[i]))
        return result.reshape(-1)

    r0 = baseline.reshape(-1)
    e = np.concatenate([(np.sqrt(item.weight)*(item.residual-down(item, baseline))).reshape(-1)
                        for item in observations])
    correction_rhs = at(e)
    reference_norm = float(np.linalg.norm(correction_rhs))
    if reference_norm == 0:
        return baseline.copy(), dict(space=space, iterations=0, relative_residual=0.,
                                     true_residual_norm=0., reference_residual_norm=0.)
    if initial_guess is not None:
        guess = np.asarray(initial_guess, dtype=np.float64)
        if guess.shape != baseline.shape or not np.isfinite(guess).all():
            raise ValueError("invalid initial guess")
        shift = guess.reshape(-1)-r0
    else:
        shift = np.zeros_like(r0)
    if space == "observation":
        if initial_guess is not None:
            raise ValueError("arbitrary warm guesses require primal space")
        rhs = e
        def operator(v): return beta*v+a(at(v))
        def reconstruct(v): return r0+at(v)
    else:
        rhs = correction_rhs-beta*shift-at(a(shift))
        def operator(v): return beta*v+at(a(v))
        def reconstruct(v): return r0+shift+v
    value = np.zeros_like(rhs)
    residual = rhs.copy()
    diagonal = jacobi_diagonal(observations, beta, space=space) if preconditioner == "jacobi" else None
    z = residual/diagonal if diagonal is not None else residual.copy()
    direction = z.copy()
    rho = float(np.dot(residual, z))
    if not np.isfinite(rho) or rho < 0:
        raise FusionError("invalid initial preconditioned norm")
    for iteration in range(max_iterations+1):
        image = reconstruct(value)
        true_residual = correction_rhs-beta*(image-r0)-at(a(image-r0))
        true_norm = float(np.linalg.norm(true_residual))
        relative = true_norm/reference_norm
        if np.isfinite(relative) and relative <= relative_tolerance:
            return image.reshape(baseline.shape), dict(space=space, iterations=iteration,
                relative_residual=relative, true_residual_norm=true_norm,
                reference_residual_norm=reference_norm,
                vector_elements=value.size, preconditioner=preconditioner)
        if iteration == max_iterations:
            break
        applied = operator(direction)
        denominator = float(np.dot(direction, applied))
        if not np.isfinite(denominator) or denominator <= 0 or rho <= 0:
            raise FusionError("CG lost positive definiteness")
        step = rho/denominator
        value += step*direction
        residual -= step*applied
        z = residual/diagonal if diagonal is not None else residual.copy()
        next_rho = float(np.dot(residual, z))
        if not np.isfinite(next_rho) or next_rho < 0:
            raise FusionError("invalid preconditioned norm")
        direction = z+(next_rho/rho)*direction
        rho = next_rho
    raise FusionError(f"CG did not converge: true relative residual={relative:.8g}")
