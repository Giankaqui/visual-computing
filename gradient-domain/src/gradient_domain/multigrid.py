"""Geometric multigrid for the Dirichlet Poisson problem.

A relaxation scheme such as Gauss-Seidel removes the high-frequency part of the
error in a handful of sweeps and then stalls: the low-frequency part is what
makes plain relaxation converge in ``O(N)`` sweeps on an ``N``-pixel grid.  The
observation behind multigrid is that error which is smooth on a fine grid is not
smooth on a grid twice as coarse, so it can be relaxed there instead, at a
quarter of the cost.

One V-cycle applies that recursively: smooth, restrict the residual, solve the
coarse problem approximately by the same procedure, interpolate the correction
back, smooth again.  The cost of a cycle is a small constant times the cost of a
fine-grid sweep, because the grids shrink geometrically, and the error is reduced
by a factor that does not depend on the grid size.  The total work to reach a
fixed tolerance is therefore proportional to the number of pixels, which is
optimal.

Grid transfers are the bilinear interpolation ``P`` and its scaled adjoint
``P^T / 4``, which is the full-weighting restriction.  Using an adjoint pair
matters: it is what makes the coarse-grid correction a projection in the energy
norm, and a mismatched pair degrades the convergence factor or destroys it.

Coarsening halves the number of interior points as ``m -> (m - 1) // 2``, so the
coarse points sit exactly on odd fine points.  The relation is exact when the
size is odd and approximate otherwise, which costs a little convergence on
arbitrary image sizes.  :func:`solve` reports the observed convergence factor,
and :mod:`gradient_domain.solvers` offers the cycle as a preconditioner for
conjugate gradients, which is insensitive to that loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from .operators import laplacian

__all__ = ["MultigridOptions", "MultigridReport", "v_cycle", "solve"]


@dataclass
class MultigridOptions:
    """Configuration of the solver.

    Attributes
    ----------
    pre_smoothing, post_smoothing : int
        Gauss-Seidel sweeps before restriction and after interpolation.
    coarsest_size : int
        Grids with fewer points than this along either axis are solved directly.
    coarsest_sweeps : int
        Sweeps used on the coarsest grid, which is small enough that relaxation
        alone converges.
    max_cycles : int
        Upper bound on V-cycles.
    tolerance : float
        Target relative residual in the Euclidean norm.
    """

    pre_smoothing: int = 2
    post_smoothing: int = 2
    coarsest_size: int = 8
    coarsest_sweeps: int = 60
    max_cycles: int = 60
    tolerance: float = 1e-8


@dataclass
class MultigridReport:
    """Convergence record of a solve.

    Attributes
    ----------
    residuals : list of float
        Relative residual after each cycle, starting with the initial one.
    cycles : int
        V-cycles performed.
    converged : bool
    """

    residuals: list[float]
    cycles: int
    converged: bool

    @property
    def convergence_factor(self) -> float:
        """Geometric mean of the per-cycle residual reduction.

        Values around 0.1 are what a correctly assembled V-cycle achieves on the
        Poisson problem; values close to 1 mean the coarse-grid correction is
        not helping.
        """
        if len(self.residuals) < 2 or self.residuals[0] <= 0:
            return float("nan")
        ratio = self.residuals[-1] / self.residuals[0]
        return float(ratio ** (1.0 / (len(self.residuals) - 1)))

    def __str__(self) -> str:
        return (
            f"{self.cycles} cycles, relative residual {self.residuals[-1]:.2e}, "
            f"convergence factor {self.convergence_factor:.3f}"
        )


@lru_cache(maxsize=64)
def _prolongation_1d(fine: int, coarse: int) -> sp.csr_matrix:
    """Linear interpolation from ``coarse`` to ``fine`` points along one axis.

    Coarse point ``I`` lies on fine point ``2 I + 1``; the fine points between
    them take the average of their two neighbours, and points outside the coarse
    range see the homogeneous Dirichlet value of zero.
    """
    rows, cols, values = [], [], []
    for index in range(fine):
        if index % 2 == 1:
            source = (index - 1) // 2
            if 0 <= source < coarse:
                rows.append(index)
                cols.append(source)
                values.append(1.0)
        else:
            for source in (index // 2 - 1, index // 2):
                if 0 <= source < coarse:
                    rows.append(index)
                    cols.append(source)
                    values.append(0.5)
    return sp.csr_matrix((values, (rows, cols)), shape=(fine, coarse))


@lru_cache(maxsize=64)
def _parity_mask(shape: tuple[int, int], parity: int) -> np.ndarray:
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    return (rows + cols) % 2 == parity


def _coarse_shape(shape: tuple[int, int]) -> tuple[int, int]:
    return ((shape[0] - 1) // 2, (shape[1] - 1) // 2)


def _smooth(u: np.ndarray, b: np.ndarray, spacing: float, sweeps: int) -> np.ndarray:
    """Red-black Gauss-Seidel sweeps on ``laplacian(u) = b``.

    The two colours are updated in sequence, so the second half of each sweep
    already sees the new values; that is what makes the scheme a Gauss-Seidel
    rather than a Jacobi iteration, and it roughly doubles the smoothing rate.
    Within a colour the updates are independent, which is what allows the whole
    half-sweep to be one array expression.
    """
    squared = spacing * spacing
    for _ in range(sweeps):
        for parity in (0, 1):
            padded = np.pad(u, 1)
            neighbours = (
                padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
            )
            updated = 0.25 * (neighbours - squared * b)
            u = np.where(_parity_mask(u.shape, parity), updated, u)
    return u


def _restrict(residual: np.ndarray) -> np.ndarray:
    """Full-weighting restriction, the scaled adjoint of :func:`_prolong`."""
    coarse_rows, coarse_cols = _coarse_shape(residual.shape)
    rows = _prolongation_1d(residual.shape[0], coarse_rows)
    cols = _prolongation_1d(residual.shape[1], coarse_cols)
    return (rows.T @ residual @ cols) * 0.25


def _prolong(correction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear interpolation of a coarse-grid correction."""
    rows = _prolongation_1d(shape[0], correction.shape[0])
    cols = _prolongation_1d(shape[1], correction.shape[1])
    return rows @ correction @ cols.T


def v_cycle(
    u: np.ndarray, b: np.ndarray, spacing: float, options: MultigridOptions
) -> np.ndarray:
    """Apply one V-cycle to the system ``laplacian(u) = b``.

    Parameters
    ----------
    u : ndarray, shape (m, n)
        Current iterate; zeros are a valid starting point.
    b : ndarray, shape (m, n)
        Right-hand side on the interior, with Dirichlet values already folded in.
    spacing : float
        Grid spacing at this level.
    options : MultigridOptions

    Returns
    -------
    ndarray
        Improved iterate, same shape as ``u``.
    """
    coarse_shape = _coarse_shape(u.shape)
    if min(u.shape) <= options.coarsest_size or min(coarse_shape) < 2:
        return _smooth(u, b, spacing, options.coarsest_sweeps)

    u = _smooth(u, b, spacing, options.pre_smoothing)
    residual = b - laplacian(u, spacing)
    coarse_correction = v_cycle(
        np.zeros(coarse_shape), _restrict(residual), 2.0 * spacing, options
    )
    u = u + _prolong(coarse_correction, u.shape)
    return _smooth(u, b, spacing, options.post_smoothing)


def solve(
    b: np.ndarray,
    spacing: float = 1.0,
    options: MultigridOptions | None = None,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, MultigridReport]:
    """Solve ``laplacian(u) = b`` by repeated V-cycles.

    Parameters
    ----------
    b : ndarray, shape (m, n)
        Right-hand side on the interior unknowns.
    spacing : float
        Grid spacing of the finest level.
    options : MultigridOptions or None
    initial : ndarray or None
        Starting iterate; zeros when omitted.

    Returns
    -------
    u : ndarray, shape (m, n)
    report : MultigridReport
    """
    options = options or MultigridOptions()
    b = np.asarray(b, dtype=float)
    u = np.zeros_like(b) if initial is None else np.array(initial, dtype=float, copy=True)

    scale = float(np.linalg.norm(b))
    if scale == 0.0:
        return u, MultigridReport(residuals=[0.0], cycles=0, converged=True)

    residuals = [float(np.linalg.norm(b - laplacian(u, spacing))) / scale]
    for cycle in range(1, options.max_cycles + 1):
        u = v_cycle(u, b, spacing, options)
        residuals.append(float(np.linalg.norm(b - laplacian(u, spacing))) / scale)
        if residuals[-1] < options.tolerance:
            return u, MultigridReport(residuals=residuals, cycles=cycle, converged=True)
    return u, MultigridReport(residuals=residuals, cycles=options.max_cycles, converged=False)
