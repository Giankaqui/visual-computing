"""Rotation utilities for the SO(3) manifold.

Bundle adjustment updates rotations multiplicatively: a pose stored as a matrix
``R`` is perturbed by a small axis-angle increment ``delta`` applied on the left,
``R <- exp(skew(delta)) R``.  Keeping the increment local avoids the
singularities of a global three-parameter representation and makes the Jacobian
of a rotated point a single cross-product matrix (see :func:`rotate_point_jacobian`).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "skew",
    "exp_so3",
    "log_so3",
    "project_to_so3",
    "rotate_point_jacobian",
    "random_rotation",
]

_EPS = 1e-12


def skew(v: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix (or batch of them) of ``v``.

    Parameters
    ----------
    v : ndarray, shape (..., 3)

    Returns
    -------
    ndarray, shape (..., 3, 3)
        Matrices ``S`` such that ``S @ w == np.cross(v, w)``.
    """
    v = np.asarray(v, dtype=float)
    zero = np.zeros(v.shape[:-1], dtype=float)
    return np.stack(
        [
            zero, -v[..., 2], v[..., 1],
            v[..., 2], zero, -v[..., 0],
            -v[..., 1], v[..., 0], zero,
        ],
        axis=-1,
    ).reshape(v.shape[:-1] + (3, 3))


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """Exponential map from an axis-angle vector to a rotation matrix.

    Uses the Rodrigues formula, falling back to the second-order Taylor
    expansion when the rotation angle approaches zero.

    Parameters
    ----------
    omega : ndarray, shape (3,)
        Rotation axis scaled by the rotation angle in radians.

    Returns
    -------
    ndarray, shape (3, 3)
    """
    omega = np.asarray(omega, dtype=float).reshape(3)
    theta = float(np.linalg.norm(omega))
    K = skew(omega)
    if theta < 1e-8:
        return np.eye(3) + K + 0.5 * (K @ K)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithmic map from a rotation matrix to an axis-angle vector.

    The branch for angles near ``pi`` reconstructs the axis from the symmetric
    part ``R + R.T``, where the antisymmetric part vanishes and is therefore
    uninformative.

    Parameters
    ----------
    R : ndarray, shape (3, 3)

    Returns
    -------
    ndarray, shape (3,)
        Vector with norm in ``[0, pi]``.
    """
    R = np.asarray(R, dtype=float)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-8:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - theta < 1e-4:
        # Near pi the axis is the unit eigenvector of R for eigenvalue 1; it is
        # read off the diagonal of (R + I) / 2, whose columns are all parallel
        # to the axis.  The most reliable column is the one with largest norm.
        M = 0.5 * (R + np.eye(3))
        axis = M[:, int(np.argmax(np.diag(M)))]
        axis = axis / max(np.linalg.norm(axis), _EPS)
        # Resolve the sign ambiguity with the residual antisymmetric part.
        residual = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(axis, residual) < 0:
            axis = -axis
        return axis * theta
    factor = theta / (2.0 * np.sin(theta))
    return factor * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Return the rotation matrix closest to ``M`` in Frobenius norm.

    Solves the orthogonal Procrustes problem ``min_R ||R - M||_F`` subject to
    ``R.T @ R = I`` and ``det(R) = 1``.

    Parameters
    ----------
    M : ndarray, shape (3, 3)

    Returns
    -------
    ndarray, shape (3, 3)
    """
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return U @ D @ Vt


def rotate_point_jacobian(rotated_point: np.ndarray) -> np.ndarray:
    """Jacobian of ``exp(skew(delta)) @ p`` with respect to ``delta`` at zero.

    Differentiating the first-order expansion ``(I + skew(delta)) p`` gives
    ``-skew(p)``, evaluated at the already-rotated point.

    Parameters
    ----------
    rotated_point : ndarray, shape (..., 3)
        The point after the current rotation has been applied.

    Returns
    -------
    ndarray, shape (..., 3, 3)
    """
    return -skew(rotated_point)


def random_rotation(rng: np.random.Generator, max_angle: float = np.pi) -> np.ndarray:
    """Sample a rotation with a uniformly distributed axis and bounded angle."""
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), _EPS)
    return exp_so3(axis * rng.uniform(-max_angle, max_angle))
