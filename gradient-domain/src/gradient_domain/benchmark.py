"""Cost comparison of the Poisson solvers.

The benchmark solves the same manufactured problem at growing resolutions and
records time and iteration count.  A manufactured solution is used rather than a
real editing problem so the exact answer is known and every method can be held
to the same residual, which is the only way the timings mean anything.

What the numbers are expected to show: the direct factorization is competitive
at small sizes and then loses to fill-in; conjugate gradients needs a number of
iterations that grows with the grid, because the condition number of the
Laplacian does; and the multigrid variants need a number of cycles that does not,
so their total work is proportional to the number of pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .operators import laplacian
from .solvers import solve_system

__all__ = ["BenchmarkRecord", "solver_scaling", "manufactured_problem"]


@dataclass
class BenchmarkRecord:
    """One measurement.

    Attributes
    ----------
    method : str
    shape : tuple of int
    unknowns : int
    seconds : float
    iterations : int
    relative_residual : float
    max_error : float
        Largest absolute difference from the known solution.
    """

    method: str
    shape: tuple[int, int]
    unknowns: int
    seconds: float
    iterations: int
    relative_residual: float
    max_error: float

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "shape": list(self.shape),
            "unknowns": self.unknowns,
            "seconds": self.seconds,
            "iterations": self.iterations,
            "relative_residual": self.relative_residual,
            "max_error": self.max_error,
        }


def manufactured_problem(
    shape: tuple[int, int], seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Build a right-hand side whose exact solution is known.

    The solution mixes a few smooth modes with white noise, so it contains
    energy across the whole spectrum.  A purely smooth solution would flatter
    multigrid and a purely rough one would flatter relaxation.

    Parameters
    ----------
    shape : tuple of int
    seed : int

    Returns
    -------
    b : ndarray, shape (m, n)
        Right-hand side with homogeneous Dirichlet conditions folded in.
    solution : ndarray, shape (m, n)
    """
    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]] / max(shape)
    solution = (
        np.sin(3.0 * np.pi * columns) * np.sin(2.0 * np.pi * rows)
        + 0.4 * np.cos(11.0 * np.pi * columns) * np.sin(7.0 * np.pi * rows)
        + 0.25 * rng.standard_normal(shape)
    )
    solution[0] = solution[-1] = 0.0
    solution[:, 0] = solution[:, -1] = 0.0
    return laplacian(solution), solution


def solver_scaling(
    sizes: list[int],
    methods: list[str] | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 5000,
    direct_limit: int = 600,
    seed: int = 0,
) -> dict[str, list[BenchmarkRecord]]:
    """Time every solver on square grids of the given sizes.

    Parameters
    ----------
    sizes : list of int
        Side lengths of the square grids.
    methods : list of str or None
        Defaults to all four solvers.
    tolerance : float
        Target relative residual.
    max_iterations : int
    direct_limit : int
        Grids larger than this are skipped for the direct method, whose
        factorization becomes the dominant cost in both time and memory.
    seed : int

    Returns
    -------
    dict
        Maps a method name to its records, ordered by size.
    """
    methods = methods or ["direct", "cg", "multigrid", "mgcg"]
    results: dict[str, list[BenchmarkRecord]] = {method: [] for method in methods}

    for size in sizes:
        b, solution = manufactured_problem((size, size), seed=seed)
        for method in methods:
            if method == "direct" and size > direct_limit:
                continue
            estimate, report = solve_system(
                b, method=method, tolerance=tolerance, max_iterations=max_iterations
            )
            results[method].append(
                BenchmarkRecord(
                    method=method,
                    shape=(size, size),
                    unknowns=size * size,
                    seconds=report.seconds,
                    iterations=report.iterations,
                    relative_residual=report.relative_residual,
                    max_error=float(np.abs(estimate - solution).max()),
                )
            )
    return results


@dataclass
class BenchmarkTable:
    """Formatted view of a scaling run."""

    results: dict[str, list[BenchmarkRecord]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the records as a Markdown table ordered by size then method."""
        rows = [record for records in self.results.values() for record in records]
        rows.sort(key=lambda record: (record.unknowns, record.method))

        lines = [
            "| Grid | Unknowns | Method | Iterations | Seconds | Max error |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
        for record in rows:
            grid = f"{record.shape[0]} x {record.shape[1]}"
            iterations = "-" if record.iterations == 0 else str(record.iterations)
            lines.append(
                f"| {grid} | {record.unknowns} | {record.method} | {iterations} | "
                f"{record.seconds:.3f} | {record.max_error:.2e} |"
            )
        return "\n".join(lines)
