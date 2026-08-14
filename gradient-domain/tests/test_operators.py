"""Checks that the discrete operators form a consistent adjoint pair.

If the gradient and the divergence are not adjoint, the Poisson equation being
solved is no longer the normal equation of the least-squares problem the
applications are posed as, and the solvers converge to the wrong image without
any visible sign of failure.  The tests below pin that relation down.
"""

from __future__ import annotations

import numpy as np
import pytest

from gradient_domain.operators import (
    divergence,
    fold_boundary,
    gradient,
    laplacian,
    sparse_laplacian,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(11)


def test_divergence_is_the_negative_adjoint_of_the_gradient(rng: np.random.Generator) -> None:
    u = rng.standard_normal((23, 31))
    vx, vy = rng.standard_normal((23, 31)), rng.standard_normal((23, 31))

    gx, gy = gradient(u)
    left = float(np.sum(gx * vx) + np.sum(gy * vy))
    right = -float(np.sum(u * divergence(vx, vy)))
    assert left == pytest.approx(right, rel=1e-12)


def test_gradient_of_a_constant_vanishes() -> None:
    gx, gy = gradient(np.full((9, 12), 3.5))
    assert np.allclose(gx, 0.0)
    assert np.allclose(gy, 0.0)


def test_gradient_reproduces_a_linear_ramp() -> None:
    rows, columns = np.mgrid[0:8, 0:10]
    gx, gy = gradient(2.0 * columns + 3.0 * rows)
    assert np.allclose(gx[:, :-1], 2.0)
    assert np.allclose(gy[:-1, :], 3.0)


def test_composition_is_the_neumann_laplacian(rng: np.random.Generator) -> None:
    u = rng.standard_normal((17, 21))
    padded = np.pad(u, 1, mode="edge")
    expected = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * u
    )
    assert np.allclose(divergence(*gradient(u)), expected)


def test_dense_and_sparse_laplacians_agree(rng: np.random.Generator) -> None:
    u = rng.standard_normal((13, 19))
    for spacing in (1.0, 0.5, 2.0):
        expected = laplacian(u, spacing)
        actual = (sparse_laplacian(u.shape, spacing) @ u.reshape(-1)).reshape(u.shape)
        assert np.allclose(actual, expected)


def test_laplacian_is_symmetric_negative_definite() -> None:
    operator = sparse_laplacian((7, 9)).toarray()
    assert np.allclose(operator, operator.T)
    assert np.linalg.eigvalsh(operator).max() < 0.0


def test_folding_the_boundary_reproduces_the_full_stencil(rng: np.random.Generator) -> None:
    full = rng.standard_normal((14, 18))
    interior = full[1:-1, 1:-1]

    padded_result = laplacian(full)[1:-1, 1:-1]
    folded = laplacian(interior) - fold_boundary(np.zeros_like(interior), full)
    assert np.allclose(folded, padded_result)


def test_folding_scales_with_the_grid_spacing() -> None:
    boundary = np.ones((5, 5))
    folded = fold_boundary(np.zeros((3, 3)), boundary, spacing=0.5)
    # Corners lose two neighbours, edges one, the centre none; each contributes
    # one over the squared spacing.
    assert folded[0, 0] == pytest.approx(-8.0)
    assert folded[0, 1] == pytest.approx(-4.0)
    assert folded[1, 1] == pytest.approx(0.0)
