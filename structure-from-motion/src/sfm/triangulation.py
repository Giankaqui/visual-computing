"""Triangulation of 3D points from calibrated observations.

All functions take projection matrices and image points expressed in normalized
coordinates, so a projection matrix is simply ``[R | t]``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["triangulate_dlt", "triangulate_multiview", "refine_point"]


def triangulate_dlt(
    projection1: np.ndarray,
    projection2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    """Two-view linear triangulation.

    Each observation contributes two rows of the cross-product constraint
    ``x x (P X) = 0``; the homogeneous solution is the right singular vector of
    the resulting ``4 x 4`` system for the smallest singular value.  The
    minimized quantity is algebraic rather than geometric, so the result is a
    starting point for :func:`refine_point` or for bundle adjustment.

    Parameters
    ----------
    projection1, projection2 : ndarray, shape (3, 4)
    points1, points2 : ndarray, shape (n, 2)
        Normalized image coordinates.

    Returns
    -------
    ndarray, shape (n, 3)
        Points whose homogeneous coordinate vanishes, that is intersections at
        infinity, are returned as ``inf`` and should be discarded by the caller.
    """
    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)
    n = len(points1)

    A = np.empty((n, 4, 4), dtype=float)
    A[:, 0] = points1[:, 0, None] * projection1[2] - projection1[0]
    A[:, 1] = points1[:, 1, None] * projection1[2] - projection1[1]
    A[:, 2] = points2[:, 0, None] * projection2[2] - projection2[0]
    A[:, 3] = points2[:, 1, None] * projection2[2] - projection2[1]

    row_norms = np.maximum(np.linalg.norm(A, axis=2, keepdims=True), 1e-12)
    _, _, Vt = np.linalg.svd(A / row_norms)
    homogeneous = Vt[:, -1, :]

    scale = homogeneous[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        points3d = homogeneous[:, :3] / scale[:, None]
    points3d[np.abs(scale) < 1e-12] = np.inf
    return points3d


def triangulate_multiview(
    projections: np.ndarray, observations: np.ndarray
) -> np.ndarray:
    """Linear triangulation of a single point seen in ``m`` views.

    Parameters
    ----------
    projections : ndarray, shape (m, 3, 4)
    observations : ndarray, shape (m, 2)
        Normalized image coordinates.

    Returns
    -------
    ndarray, shape (3,)
    """
    projections = np.asarray(projections, dtype=float)
    observations = np.asarray(observations, dtype=float)

    A = np.empty((2 * len(projections), 4), dtype=float)
    A[0::2] = observations[:, 0, None] * projections[:, 2] - projections[:, 0]
    A[1::2] = observations[:, 1, None] * projections[:, 2] - projections[:, 1]
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)

    _, _, Vt = np.linalg.svd(A)
    homogeneous = Vt[-1]
    if abs(homogeneous[3]) < 1e-12:
        return np.full(3, np.inf)
    return homogeneous[:3] / homogeneous[3]


def refine_point(
    point3d: np.ndarray,
    projections: np.ndarray,
    observations: np.ndarray,
    iterations: int = 10,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Minimize reprojection error for one point by Gauss-Newton iteration.

    The normal equations are a ``3 x 3`` system, so each step costs almost
    nothing and the refinement is worth running before the point enters bundle
    adjustment: the linear estimate can be biased by several pixels when the
    parallax is small.

    Parameters
    ----------
    point3d : ndarray, shape (3,)
        Initial estimate, typically from :func:`triangulate_multiview`.
    projections : ndarray, shape (m, 3, 4)
    observations : ndarray, shape (m, 2)
        Normalized image coordinates.
    iterations : int
        Maximum Gauss-Newton steps.
    tolerance : float
        Stop when the squared step length falls below this value.

    Returns
    -------
    ndarray, shape (3,)
        The input is returned unchanged if the point falls behind a camera at
        any point during the iteration.
    """
    point = np.asarray(point3d, dtype=float).copy()
    if not np.isfinite(point).all():
        return point

    R = projections[:, :, :3]
    t = projections[:, :, 3]

    for _ in range(iterations):
        camera_points = np.einsum("mij,j->mi", R, point) + t
        depth = camera_points[:, 2]
        if np.any(depth <= 1e-9):
            return np.asarray(point3d, dtype=float)

        predicted = camera_points[:, :2] / depth[:, None]
        residual = (predicted - observations).reshape(-1)

        # d(x/z, y/z) / d(camera point), stacked over views.
        d_projection = np.zeros((len(R), 2, 3), dtype=float)
        d_projection[:, 0, 0] = 1.0 / depth
        d_projection[:, 1, 1] = 1.0 / depth
        d_projection[:, 0, 2] = -camera_points[:, 0] / depth**2
        d_projection[:, 1, 2] = -camera_points[:, 1] / depth**2

        J = np.einsum("mij,mjk->mik", d_projection, R).reshape(-1, 3)
        normal_matrix = J.T @ J
        gradient = J.T @ residual
        try:
            step = np.linalg.solve(normal_matrix + 1e-12 * np.eye(3), gradient)
        except np.linalg.LinAlgError:
            break
        point -= step
        if float(step @ step) < tolerance:
            break
    return point
