"""Linear solvers for the interior Dirichlet Poisson system.

Four methods are provided for the same system, which is what makes the cost
comparison in the benchmarks meaningful.

``direct``
    Sparse LU of the five-point Laplacian.  Exact up to rounding and the right
    choice when the same operator is reused for many right-hand sides, since the
    factorization is computed once.  Its memory grows faster than the grid
    because of fill-in, which is what rules it out at high resolution.

``cg``
    Conjugate gradients with Jacobi preconditioning, matrix free.  Memory is
    proportional to the grid, but the iteration count grows with the condition
    number of the Laplacian, which scales as the number of pixels; the work is
    therefore superlinear.

``multigrid``
    V-cycles from :mod:`gradient_domain.multigrid`.  The iteration count is
    independent of the grid size, so the total work is linear.

``mgcg``
    Conjugate gradients preconditioned by one V-cycle.  It keeps the linear
    behaviour of multigrid and recovers the cases where the plain cycle
    converges slowly, which here means grids whose size does not coarsen
    exactly.  It is the default for that reason.

All methods solve the positive definite system ``-L u = -b`` rather than
``L u = b``; the Laplacian as written is negative definite, and conjugate
gradients requires positive definiteness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla
from scipy.fft import dctn, idctn

from .multigrid import MultigridOptions, v_cycle
from .operators import laplacian, sparse_laplacian

__all__ = ["SolverReport", "SOLVERS", "solve_system", "solve_neumann"]


@dataclass
class SolverReport:
    """Cost and accuracy of one solve.

    Attributes
    ----------
    method : str
    iterations : int
        Iterations or cycles; zero for the direct method.
    relative_residual : float
        Euclidean norm of the residual over the norm of the right-hand side.
    seconds : float
    converged : bool
    """

    method: str
    iterations: int
    relative_residual: float
    seconds: float
    converged: bool

    def __str__(self) -> str:
        return (
            f"{self.method}: {self.iterations} iterations, "
            f"residual {self.relative_residual:.2e}, {self.seconds * 1e3:.0f} ms"
        )


def _relative_residual(u: np.ndarray, b: np.ndarray, spacing: float) -> float:
    scale = float(np.linalg.norm(b))
    if scale == 0.0:
        return 0.0
    return float(np.linalg.norm(b - laplacian(u, spacing))) / scale


def _solve_direct(
    b: np.ndarray, spacing: float, tolerance: float, max_iterations: int
) -> tuple[np.ndarray, int, bool]:
    operator = sparse_laplacian(b.shape, spacing).tocsc()
    solution = spla.splu(operator).solve(b.reshape(-1))
    return solution.reshape(b.shape), 0, True


def _solve_cg(
    b: np.ndarray, spacing: float, tolerance: float, max_iterations: int
) -> tuple[np.ndarray, int, bool]:
    # Jacobi preconditioning on a constant-coefficient operator is a scalar, so
    # it changes nothing but the step scale; it is kept because it makes the
    # residual comparable with the preconditioned variants.
    diagonal = 4.0 / (spacing * spacing)
    rhs = -b
    u = np.zeros_like(b)
    residual = rhs.copy()
    direction = residual / diagonal
    delta = float(np.sum(residual * direction))
    target = tolerance * float(np.linalg.norm(rhs))

    for iteration in range(1, max_iterations + 1):
        operator_direction = -laplacian(direction, spacing)
        step = delta / float(np.sum(direction * operator_direction))
        u += step * direction
        residual -= step * operator_direction
        if float(np.linalg.norm(residual)) <= target:
            return u, iteration, True
        preconditioned = residual / diagonal
        new_delta = float(np.sum(residual * preconditioned))
        direction = preconditioned + (new_delta / delta) * direction
        delta = new_delta
    return u, max_iterations, False


def _solve_multigrid(
    b: np.ndarray, spacing: float, tolerance: float, max_iterations: int
) -> tuple[np.ndarray, int, bool]:
    options = MultigridOptions(max_cycles=max_iterations, tolerance=tolerance)
    u = np.zeros_like(b)
    scale = float(np.linalg.norm(b)) or 1.0
    for cycle in range(1, max_iterations + 1):
        u = v_cycle(u, b, spacing, options)
        if float(np.linalg.norm(b - laplacian(u, spacing))) / scale < tolerance:
            return u, cycle, True
    return u, max_iterations, False


def _solve_mgcg(
    b: np.ndarray, spacing: float, tolerance: float, max_iterations: int
) -> tuple[np.ndarray, int, bool]:
    options = MultigridOptions(pre_smoothing=1, post_smoothing=1)

    def precondition(residual: np.ndarray) -> np.ndarray:
        return v_cycle(np.zeros_like(residual), -residual, spacing, options)

    rhs = -b
    u = np.zeros_like(b)
    residual = rhs.copy()
    direction = precondition(residual)
    delta = float(np.sum(residual * direction))
    target = tolerance * float(np.linalg.norm(rhs))

    for iteration in range(1, max_iterations + 1):
        operator_direction = -laplacian(direction, spacing)
        step = delta / float(np.sum(direction * operator_direction))
        u += step * direction
        residual -= step * operator_direction
        if float(np.linalg.norm(residual)) <= target:
            return u, iteration, True
        preconditioned = precondition(residual)
        new_delta = float(np.sum(residual * preconditioned))
        direction = preconditioned + (new_delta / delta) * direction
        delta = new_delta
    return u, max_iterations, False


SOLVERS = {
    "direct": _solve_direct,
    "cg": _solve_cg,
    "multigrid": _solve_multigrid,
    "mgcg": _solve_mgcg,
}


def solve_system(
    b: np.ndarray,
    method: str = "mgcg",
    spacing: float = 1.0,
    tolerance: float = 1e-8,
    max_iterations: int = 2000,
) -> tuple[np.ndarray, SolverReport]:
    """Solve ``laplacian(u) = b`` with homogeneous Dirichlet conditions.

    Parameters
    ----------
    b : ndarray, shape (m, n)
        Right-hand side on the interior unknowns.
    method : {'direct', 'cg', 'multigrid', 'mgcg'}
    spacing : float
        Grid spacing.
    tolerance : float
        Target relative residual, ignored by the direct method.
    max_iterations : int
        Iteration or cycle budget for the iterative methods.

    Returns
    -------
    u : ndarray, shape (m, n)
    report : SolverReport

    Raises
    ------
    KeyError
        If ``method`` is not one of the supported names.
    """
    if method not in SOLVERS:
        raise KeyError(f"unknown solver {method!r}; choose from {sorted(SOLVERS)}")

    b = np.asarray(b, dtype=float)
    if not np.any(b):
        return np.zeros_like(b), SolverReport(
            method=method, iterations=0, relative_residual=0.0, seconds=0.0, converged=True
        )

    started = time.perf_counter()
    u, iterations, converged = SOLVERS[method](b, spacing, tolerance, max_iterations)
    elapsed = time.perf_counter() - started

    return u, SolverReport(
        method=method,
        iterations=iterations,
        relative_residual=_relative_residual(u, b, spacing),
        seconds=elapsed,
        converged=converged,
    )


def solve_neumann(b: np.ndarray) -> np.ndarray:
    """Solve ``laplacian(u) = b`` with homogeneous Neumann conditions.

    Reflecting the domain across its borders turns the five-point Laplacian into
    a circulant operator that the discrete cosine transform diagonalizes, with
    eigenvalues ``2 cos(pi i / m) + 2 cos(pi j / n) - 4``.  The solve is then one
    forward transform, a division and one inverse transform, which is exact and
    costs ``O(N log N)`` with no iteration at all.

    Neumann conditions leave the solution undetermined up to an additive
    constant, which shows as a zero eigenvalue; that coefficient is set to zero,
    which selects the solution with zero mean.  This is the right boundary
    condition when the guidance field is defined on the whole image and no part
    of it should be pinned, as in tone mapping.

    Parameters
    ----------
    b : ndarray, shape (m, n)
        Right-hand side, with the compatibility condition ``sum(b) == 0``
        implied; any constant component is discarded.

    Returns
    -------
    ndarray, shape (m, n)
        Zero-mean solution.
    """
    b = np.asarray(b, dtype=float)
    rows, columns = b.shape
    eigenvalues = (
        2.0 * np.cos(np.pi * np.arange(rows) / rows)[:, None]
        + 2.0 * np.cos(np.pi * np.arange(columns) / columns)[None, :]
        - 4.0
    )
    eigenvalues[0, 0] = 1.0

    transformed = dctn(b, type=2, norm="ortho") / eigenvalues
    transformed[0, 0] = 0.0
    return idctn(transformed, type=2, norm="ortho")
