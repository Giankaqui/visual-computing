"""Discrete differential operators on a regular grid.

Every application in this package reduces to the same statement: given a target
gradient field ``v`` that is not necessarily integrable, find the image whose
gradient is closest to it in the least-squares sense.  The normal equations of
that problem are the Poisson equation

.. math::

    \\nabla^2 u = \\nabla \\cdot v,

so the operators below are the only ones needed: a forward-difference gradient
to build guidance fields, its negative adjoint as the divergence, and the
five-point Laplacian that results from composing them.

Using the adjoint pair rather than two independently chosen stencils is what
makes the discrete problem symmetric.  A symmetric system is what lets conjugate
gradients and multigrid apply at all, and it is why the divergence uses backward
differences when the gradient uses forward ones.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["gradient", "divergence", "laplacian", "sparse_laplacian", "fold_boundary"]


def gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward differences along both axes.

    The last row and column are set to zero, which is the Neumann condition that
    makes the operator's adjoint exactly :func:`divergence`.

    Parameters
    ----------
    image : ndarray, shape (h, w) or (h, w, c)

    Returns
    -------
    gx, gy : ndarray
        Horizontal and vertical derivatives, same shape as the input.
    """
    gx = np.zeros_like(image, dtype=float)
    gy = np.zeros_like(image, dtype=float)
    gx[:, :-1] = image[:, 1:] - image[:, :-1]
    gy[:-1, :] = image[1:, :] - image[:-1, :]
    return gx, gy


def divergence(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Negative adjoint of :func:`gradient`.

    The last row and column of the input are ignored, because the gradient never
    writes there and the adjoint must therefore not read from there.  Making the
    relation exact for arbitrary fields, rather than only for fields that came
    out of :func:`gradient`, is what keeps the composition symmetric when a
    guidance field is built by an operation that does not preserve that
    structure.

    Parameters
    ----------
    gx, gy : ndarray, shape (h, w) or (h, w, c)

    Returns
    -------
    ndarray
        Same shape as the inputs; ``divergence(*gradient(u))`` is the five-point
        Laplacian of ``u`` under homogeneous Neumann conditions.
    """
    dx = np.zeros_like(gx, dtype=float)
    dy = np.zeros_like(gy, dtype=float)
    if gx.shape[1] > 1:
        dx[:, 0] = gx[:, 0]
        dx[:, 1:-1] = gx[:, 1:-1] - gx[:, :-2]
        dx[:, -1] = -gx[:, -2]
    if gy.shape[0] > 1:
        dy[0, :] = gy[0, :]
        dy[1:-1, :] = gy[1:-1, :] - gy[:-2, :]
        dy[-1, :] = -gy[-2, :]
    return dx + dy


def laplacian(u: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Five-point Laplacian with homogeneous Dirichlet conditions.

    Values outside the array are treated as zero, so this is the operator of the
    interior system produced by :func:`fold_boundary`.

    Parameters
    ----------
    u : ndarray, shape (m, n) or (m, n, c)
    spacing : float
        Grid spacing ``h``; the operator scales as ``1 / h ** 2``.

    Returns
    -------
    ndarray
        Same shape as ``u``.
    """
    padded = np.pad(u, ((1, 1), (1, 1)) + ((0, 0),) * (u.ndim - 2))
    neighbours = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    )
    return (neighbours - 4.0 * u) / (spacing * spacing)


def sparse_laplacian(shape: tuple[int, int], spacing: float = 1.0) -> sp.csr_matrix:
    """Assemble the five-point Laplacian of :func:`laplacian` as a sparse matrix.

    Parameters
    ----------
    shape : tuple of int
        Interior grid size ``(m, n)``.
    spacing : float

    Returns
    -------
    scipy.sparse.csr_matrix, shape (m * n, m * n)
        Symmetric negative definite, in row-major (C) ordering.
    """
    rows, cols = shape

    def second_difference(size: int) -> sp.dia_matrix:
        return sp.diags(
            [np.ones(size - 1), np.full(size, -2.0), np.ones(size - 1)], [-1, 0, 1], format="csr"
        )

    identity_rows = sp.identity(rows, format="csr")
    identity_cols = sp.identity(cols, format="csr")
    operator = sp.kron(second_difference(rows), identity_cols) + sp.kron(
        identity_rows, second_difference(cols)
    )
    return (operator / (spacing * spacing)).tocsr()


def fold_boundary(
    right_hand_side: np.ndarray, boundary: np.ndarray, spacing: float = 1.0
) -> np.ndarray:
    """Move known Dirichlet values from the operator to the right-hand side.

    The unknowns are the interior of ``boundary``; its one-pixel ring holds the
    prescribed values.  Every interior pixel adjacent to the ring loses one
    neighbour from the stencil, and the corresponding term moves to the
    right-hand side with the opposite sign.

    Parameters
    ----------
    right_hand_side : ndarray, shape (h - 2, w - 2) or (h - 2, w - 2, c)
        Divergence of the guidance field on the interior.
    boundary : ndarray, shape (h, w) or (h, w, c)
        Image whose border supplies the boundary values.
    spacing : float
        Grid spacing, matching the one used by the operator.

    Returns
    -------
    ndarray
        Same shape as ``right_hand_side``.
    """
    folded = np.array(right_hand_side, dtype=float, copy=True)
    weight = 1.0 / (spacing * spacing)
    folded[0] -= weight * boundary[0, 1:-1]
    folded[-1] -= weight * boundary[-1, 1:-1]
    folded[:, 0] -= weight * boundary[1:-1, 0]
    folded[:, -1] -= weight * boundary[1:-1, -1]
    return folded
