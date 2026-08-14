"""Checks on the four Dirichlet solvers and on the spectral Neumann solver.

Every method solves the same system, so the strongest statement is that they all
land on the same answer.  The multigrid tests additionally pin down what makes it
worth having: a convergence factor that does not depend on the grid size.
"""

from __future__ import annotations

import numpy as np
import pytest

from gradient_domain.benchmark import manufactured_problem
from gradient_domain.multigrid import (
    MultigridOptions,
    _prolong,
    _restrict,
    solve,
)
from gradient_domain.operators import laplacian
from gradient_domain.solvers import solve_neumann, solve_system


@pytest.mark.parametrize("method", ["direct", "cg", "multigrid", "mgcg"])
def test_every_solver_recovers_the_manufactured_solution(method: str) -> None:
    b, expected = manufactured_problem((63, 63), seed=3)
    estimate, report = solve_system(b, method=method, tolerance=1e-10)

    assert report.converged
    assert report.relative_residual < 1e-8
    assert np.abs(estimate - expected).max() < 1e-6


def test_solvers_agree_with_each_other() -> None:
    b, _ = manufactured_problem((47, 61), seed=4)
    reference, _ = solve_system(b, method="direct")
    for method in ("cg", "multigrid", "mgcg"):
        estimate, _ = solve_system(b, method=method, tolerance=1e-11)
        assert np.abs(estimate - reference).max() < 1e-6


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(KeyError):
        solve_system(np.zeros((8, 8)), method="jacobi")


def test_zero_right_hand_side_gives_zero() -> None:
    estimate, report = solve_system(np.zeros((16, 16)), method="mgcg")
    assert np.allclose(estimate, 0.0)
    assert report.relative_residual == 0.0


@pytest.mark.parametrize("size", [31, 63, 127, 255])
def test_multigrid_convergence_does_not_depend_on_the_grid(size: int) -> None:
    b, _ = manufactured_problem((size, size), seed=5)
    _, report = solve(b, options=MultigridOptions(tolerance=1e-10))

    assert report.converged
    assert report.cycles <= 10
    assert report.convergence_factor < 0.1


def test_restriction_is_the_scaled_adjoint_of_prolongation() -> None:
    rng = np.random.default_rng(6)
    fine_shape = (31, 25)
    coarse_shape = ((fine_shape[0] - 1) // 2, (fine_shape[1] - 1) // 2)

    fine = rng.standard_normal(fine_shape)
    coarse = rng.standard_normal(coarse_shape)

    left = float(np.sum(_restrict(fine) * coarse))
    right = 0.25 * float(np.sum(fine * _prolong(coarse, fine_shape)))
    assert left == pytest.approx(right, rel=1e-12)


def test_prolongation_reproduces_a_constant_in_the_interior() -> None:
    coarse = np.ones((7, 7))
    fine = _prolong(coarse, (15, 15))
    # Interpolation sees zero outside the coarse grid, so only the interior,
    # where every stencil is complete, reaches the constant.
    assert np.allclose(fine[1:-1, 1:-1], 1.0)


def test_one_cycle_reduces_the_residual_by_an_order_of_magnitude() -> None:
    b, _ = manufactured_problem((127, 127), seed=7)
    _, report = solve(b, options=MultigridOptions(max_cycles=1, tolerance=0.0))
    assert report.residuals[1] < 0.1 * report.residuals[0]


def test_neumann_solver_is_exact() -> None:
    rng = np.random.default_rng(8)
    for shape in ((64, 64), (97, 131)):
        expected = rng.standard_normal(shape)
        expected -= expected.mean()

        padded = np.pad(expected, 1, mode="edge")
        b = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
            - 4.0 * expected
        )
        assert np.abs(solve_neumann(b) - expected).max() < 1e-10


def test_neumann_solution_has_zero_mean() -> None:
    rng = np.random.default_rng(9)
    b = rng.standard_normal((32, 40))
    b -= b.mean()
    assert abs(float(solve_neumann(b).mean())) < 1e-12


def test_residual_matches_the_operator() -> None:
    b, _ = manufactured_problem((33, 33), seed=10)
    estimate, report = solve_system(b, method="mgcg", tolerance=1e-9)
    residual = float(np.linalg.norm(b - laplacian(estimate))) / float(np.linalg.norm(b))
    assert residual == pytest.approx(report.relative_residual, rel=1e-9)
