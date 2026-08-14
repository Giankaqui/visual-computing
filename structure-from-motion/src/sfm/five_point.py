"""Minimal solver for the essential matrix from five point correspondences.

Five correspondences between calibrated views leave a four-dimensional null
space of the linear epipolar constraint, so the essential matrix is written as
``E = x*E1 + y*E2 + z*E3 + E4``.  The remaining unknowns are pinned down by the
two algebraic properties of an essential matrix,

.. math::

    \\det(E) = 0, \\qquad 2 E E^\\top E - \\operatorname{tr}(E E^\\top) E = 0,

which expand to ten cubic polynomials in ``(x, y, z)``.  Written in the twenty
monomials of degree at most three, the system is a ``10 x 20`` matrix whose
leading ``10 x 10`` block (the degree-three monomials) is generically
invertible.  Eliminating that block expresses every cubic monomial in the
quotient-ring basis ``{x^2, xy, y^2, xz, yz, z^2, x, y, z, 1}``, which is exactly
what is needed to build the matrix of multiplication by ``x`` in the quotient
ring.  Its eigenvectors are the basis monomials evaluated at the solutions, so
the up to ten roots are read off directly (Stewenius, Engels and Nister, 2006).

Compared with the eight-point algorithm this halves the RANSAC sample size,
which for a 50 percent inlier ratio cuts the expected number of hypotheses by
roughly two orders of magnitude, and it enforces the essential-matrix
constraints exactly instead of projecting a general matrix onto them afterwards.
"""

from __future__ import annotations

from itertools import product

import numpy as np

__all__ = ["five_point_essential"]

Monomial = tuple[int, int, int]
Polynomial = dict[Monomial, float]

_DEGREE_THREE: list[Monomial] = [
    (3, 0, 0), (2, 1, 0), (1, 2, 0), (0, 3, 0), (2, 0, 1),
    (1, 1, 1), (0, 2, 1), (1, 0, 2), (0, 1, 2), (0, 0, 3),
]
_QUOTIENT_BASIS: list[Monomial] = [
    (2, 0, 0), (1, 1, 0), (0, 2, 0), (1, 0, 1), (0, 1, 1),
    (0, 0, 2), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0),
]
_MONOMIAL_INDEX = {m: i for i, m in enumerate(_DEGREE_THREE + _QUOTIENT_BASIS)}

# Row k of the multiplication-by-x matrix expresses ``x * basis[k]``.  For the
# first six basis monomials the product has degree three and is substituted from
# the eliminated block; the remaining four stay inside the basis.
_ACTION_FROM_CUBIC = [(0, (3, 0, 0)), (1, (2, 1, 0)), (2, (1, 2, 0)),
                      (3, (2, 0, 1)), (4, (1, 1, 1)), (5, (1, 0, 2))]
_ACTION_FROM_BASIS = [(6, (2, 0, 0)), (7, (1, 1, 0)), (8, (1, 0, 1)), (9, (1, 0, 0))]


def _poly_mul(a: Polynomial, b: Polynomial) -> Polynomial:
    out: Polynomial = {}
    for (i1, j1, k1), c1 in a.items():
        for (i2, j2, k2), c2 in b.items():
            key = (i1 + i2, j1 + j2, k1 + k2)
            out[key] = out.get(key, 0.0) + c1 * c2
    return out


def _poly_add(*polys: Polynomial) -> Polynomial:
    out: Polynomial = {}
    for poly in polys:
        for key, coeff in poly.items():
            out[key] = out.get(key, 0.0) + coeff
    return out


def _poly_scale(a: Polynomial, factor: float) -> Polynomial:
    return {key: coeff * factor for key, coeff in a.items()}


def _symbolic_essential(basis: np.ndarray) -> list[list[Polynomial]]:
    """Write ``E = x*B0 + y*B1 + z*B2 + B3`` with polynomial entries."""
    variables: list[Monomial] = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
    return [
        [{variables[k]: float(basis[k, r, c]) for k in range(4)} for c in range(3)]
        for r in range(3)
    ]


def _constraint_polynomials(E: list[list[Polynomial]]) -> list[Polynomial]:
    """Return the ten cubics enforcing that ``E`` is an essential matrix."""
    EEt = [
        [_poly_add(*(_poly_mul(E[r][k], E[c][k]) for k in range(3))) for c in range(3)]
        for r in range(3)
    ]
    trace = _poly_add(*(EEt[i][i] for i in range(3)))

    equations: list[Polynomial] = []
    for r, c in product(range(3), repeat=2):
        singular_values = _poly_add(*(_poly_mul(EEt[r][k], E[k][c]) for k in range(3)))
        equations.append(
            _poly_add(
                _poly_scale(singular_values, 2.0),
                _poly_scale(_poly_mul(trace, E[r][c]), -1.0),
            )
        )

    determinant = _poly_add(
        _poly_mul(E[0][0], _poly_add(
            _poly_mul(E[1][1], E[2][2]), _poly_scale(_poly_mul(E[1][2], E[2][1]), -1.0))),
        _poly_scale(_poly_mul(E[0][1], _poly_add(
            _poly_mul(E[1][0], E[2][2]), _poly_scale(_poly_mul(E[1][2], E[2][0]), -1.0))), -1.0),
        _poly_mul(E[0][2], _poly_add(
            _poly_mul(E[1][0], E[2][1]), _poly_scale(_poly_mul(E[1][1], E[2][0]), -1.0))),
    )
    equations.append(determinant)
    return equations


def _coefficient_matrix(equations: list[Polynomial]) -> np.ndarray:
    matrix = np.zeros((len(equations), 20), dtype=float)
    for row, equation in enumerate(equations):
        for monomial, coeff in equation.items():
            matrix[row, _MONOMIAL_INDEX[monomial]] = coeff
    return matrix


def _epipolar_constraint_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Rows of ``x2^T E x1 = 0`` with ``E`` flattened in row-major order."""
    x1 = np.hstack([points1, np.ones((len(points1), 1))])
    x2 = np.hstack([points2, np.ones((len(points2), 1))])
    return (x2[:, :, None] * x1[:, None, :]).reshape(len(points1), 9)


def five_point_essential(points1: np.ndarray, points2: np.ndarray) -> list[np.ndarray]:
    """Solve for the essential matrices consistent with five correspondences.

    Parameters
    ----------
    points1, points2 : ndarray, shape (5, 2)
        Normalized image coordinates, that is pixel coordinates premultiplied by
        the inverse of the intrinsic matrix.  More than five points are accepted,
        in which case the null space is taken from the smallest singular values.

    Returns
    -------
    list of ndarray, shape (3, 3)
        Between zero and ten real solutions, each scaled to unit Frobenius norm.
        The list is empty when the constraint system is degenerate, which happens
        for coplanar-and-collinear configurations or repeated points.
    """
    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)
    if len(points1) < 5 or len(points1) != len(points2):
        return []

    _, _, Vt = np.linalg.svd(_epipolar_constraint_matrix(points1, points2))
    basis = Vt[-4:].reshape(4, 3, 3)

    coefficients = _coefficient_matrix(_constraint_polynomials(_symbolic_essential(basis)))
    cubic_block, basis_block = coefficients[:, :10], coefficients[:, 10:]
    if np.linalg.matrix_rank(cubic_block, tol=1e-10) < 10:
        return []
    reduced = np.linalg.solve(cubic_block, basis_block)

    action = np.zeros((10, 10), dtype=float)
    for row, monomial in _ACTION_FROM_CUBIC:
        action[row] = -reduced[_DEGREE_THREE.index(monomial)]
    for row, monomial in _ACTION_FROM_BASIS:
        action[row, _QUOTIENT_BASIS.index(monomial)] = 1.0

    eigenvalues, eigenvectors = np.linalg.eig(action)

    solutions: list[np.ndarray] = []
    for index in range(len(eigenvalues)):
        if abs(eigenvalues[index].imag) > 1e-8:
            continue
        vector = eigenvectors[:, index]
        if abs(vector[9]) < 1e-12:
            continue
        coordinates = (vector[[6, 7, 8]] / vector[9]).real
        E = np.tensordot(coordinates, basis[:3], axes=1) + basis[3]
        norm = np.linalg.norm(E)
        if norm > 1e-12:
            solutions.append(E / norm)
    return solutions
