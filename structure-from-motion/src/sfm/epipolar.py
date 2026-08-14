"""Two-view epipolar geometry: estimation, error metrics and pose recovery."""

from __future__ import annotations

import numpy as np

from .camera import PinholeCamera, Pose
from .five_point import five_point_essential
from .ransac import RansacOptions, RansacResult, ransac
from .rotations import project_to_so3
from .triangulation import triangulate_dlt

__all__ = [
    "normalize_for_conditioning",
    "eight_point_essential",
    "sampson_distance",
    "decompose_essential",
    "recover_pose",
    "estimate_essential",
    "triangulation_angles",
]


def normalize_for_conditioning(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply Hartley's isotropic normalization.

    Translating the centroid to the origin and scaling the mean radius to
    ``sqrt(2)`` brings the entries of the linear system to a comparable
    magnitude; without it the least-squares solution of the eight-point
    algorithm is dominated by rounding error (Hartley, 1997).

    Parameters
    ----------
    points : ndarray, shape (n, 2)

    Returns
    -------
    normalized : ndarray, shape (n, 2)
    transform : ndarray, shape (3, 3)
        Similarity ``T`` with ``normalized = T @ homogeneous(points)``.
    """
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_radius = float(np.sqrt((centred**2).sum(axis=1)).mean())
    scale = np.sqrt(2.0) / max(mean_radius, 1e-12)
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]]
    )
    return centred * scale, transform


def eight_point_essential(points1: np.ndarray, points2: np.ndarray) -> np.ndarray | None:
    """Estimate an essential matrix from at least eight normalized correspondences.

    The linear solution is projected onto the essential manifold by forcing the
    two largest singular values to their mean and the smallest to zero, which is
    the closest essential matrix in Frobenius norm.

    Parameters
    ----------
    points1, points2 : ndarray, shape (n, 2)
        Normalized image coordinates.

    Returns
    -------
    ndarray of shape (3, 3), or None if the system is rank deficient.
    """
    if len(points1) < 8:
        return None
    x1, T1 = normalize_for_conditioning(points1)
    x2, T2 = normalize_for_conditioning(points2)

    h1 = np.hstack([x1, np.ones((len(x1), 1))])
    h2 = np.hstack([x2, np.ones((len(x2), 1))])
    A = (h2[:, :, None] * h1[:, None, :]).reshape(len(x1), 9)

    _, singular_values, Vt = np.linalg.svd(A)
    # The solution is unique only when the null space is one-dimensional.  With
    # noisy measurements the smallest singular value is never exactly zero, so
    # degeneracy is detected from the second smallest one instead.
    if singular_values[-2] < 1e-10 * singular_values[0]:
        return None
    E = Vt[-1].reshape(3, 3)
    E = T2.T @ E @ T1

    U, s, Vt_e = np.linalg.svd(E)
    mean_scale = 0.5 * (s[0] + s[1])
    E = U @ np.diag([mean_scale, mean_scale, 0.0]) @ Vt_e
    norm = np.linalg.norm(E)
    return E / norm if norm > 1e-12 else None


def sampson_distance(E: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """First-order approximation of the geometric epipolar error.

    The algebraic residual ``x2^T E x1`` is divided by the norm of its gradient
    with respect to the four image coordinates, which turns it into a distance
    in the units of the input coordinates.

    Parameters
    ----------
    E : ndarray, shape (3, 3)
    points1, points2 : ndarray, shape (n, 2)

    Returns
    -------
    ndarray, shape (n,)
    """
    h1 = np.hstack([points1, np.ones((len(points1), 1))])
    h2 = np.hstack([points2, np.ones((len(points2), 1))])
    Ex1 = h1 @ E.T
    Etx2 = h2 @ E
    algebraic = np.einsum("ij,ij->i", h2, Ex1)
    gradient_norm_squared = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return np.abs(algebraic) / np.sqrt(np.maximum(gradient_norm_squared, 1e-24))


def decompose_essential(E: np.ndarray) -> list[Pose]:
    """Return the four poses compatible with an essential matrix.

    An essential matrix determines the relative pose up to a twofold rotation
    ambiguity and the sign of the baseline; only one of the four combinations
    places points in front of both cameras (Hartley and Zisserman, section 9.6).

    Parameters
    ----------
    E : ndarray, shape (3, 3)

    Returns
    -------
    list of Pose, length 4
        Poses of the second camera relative to the first, with unit baseline.
    """
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1] *= -1
    W = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R1 = project_to_so3(U @ W @ Vt)
    R2 = project_to_so3(U @ W.T @ Vt)
    t = U[:, 2]
    return [Pose(R1, t), Pose(R1, -t), Pose(R2, t), Pose(R2, -t)]


def recover_pose(
    E: np.ndarray, points1: np.ndarray, points2: np.ndarray, min_depth: float = 1e-6
) -> tuple[Pose, np.ndarray]:
    """Select the pose that triangulates the most points in front of both cameras.

    Parameters
    ----------
    E : ndarray, shape (3, 3)
    points1, points2 : ndarray, shape (n, 2)
        Normalized coordinates of inlier correspondences.
    min_depth : float
        Points closer than this to either camera plane are treated as invalid;
        they carry no depth information and are usually intersections at infinity.

    Returns
    -------
    pose : Pose
        Second camera relative to the first, with unit baseline.
    valid : ndarray of bool, shape (n,)
        Correspondences that satisfy the cheirality constraint for ``pose``.
    """
    identity = Pose()
    best_pose, best_valid, best_count = identity, np.zeros(len(points1), dtype=bool), -1
    for candidate in decompose_essential(E):
        points3d = triangulate_dlt(identity.matrix, candidate.matrix, points1, points2)
        depth1 = points3d[:, 2]
        depth2 = candidate.transform(points3d)[:, 2]
        valid = (depth1 > min_depth) & (depth2 > min_depth) & np.isfinite(points3d).all(axis=1)
        count = int(valid.sum())
        if count > best_count:
            best_pose, best_valid, best_count = candidate, valid, count
    return best_pose, best_valid


def estimate_essential(
    points1: np.ndarray,
    points2: np.ndarray,
    camera1: PinholeCamera,
    camera2: PinholeCamera,
    pixel_threshold: float = 1.5,
    options: RansacOptions | None = None,
) -> RansacResult:
    """Robustly estimate the essential matrix between two calibrated views.

    Hypotheses come from the five-point solver, so a minimal sample is five
    correspondences.  Residuals are Sampson distances converted to pixels by the
    mean focal length, which keeps the threshold interpretable regardless of the
    normalization.

    Parameters
    ----------
    points1, points2 : ndarray, shape (n, 2)
        Matched pixel coordinates.
    camera1, camera2 : PinholeCamera
    pixel_threshold : float
        Inlier threshold in pixels.
    options : RansacOptions or None
        Overrides the defaults; ``threshold`` is always taken from
        ``pixel_threshold``.

    Returns
    -------
    RansacResult
        ``model`` is the essential matrix in normalized coordinates.
    """
    normalized1 = camera1.normalize(points1)
    normalized2 = camera2.normalize(points2)
    mean_focal = 0.25 * (camera1.fx + camera1.fy + camera2.fx + camera2.fy)

    base = options or RansacOptions(threshold=pixel_threshold)
    settings = RansacOptions(
        threshold=pixel_threshold / mean_focal,
        confidence=base.confidence,
        max_iterations=base.max_iterations,
        min_iterations=base.min_iterations,
        local_optimization=base.local_optimization,
        seed=base.seed,
    )

    def fit(indices: np.ndarray) -> list[np.ndarray]:
        sample1, sample2 = normalized1[indices], normalized2[indices]
        if len(indices) == 5:
            return five_point_essential(sample1, sample2)
        model = eight_point_essential(sample1, sample2)
        return [model] if model is not None else []

    def residuals(model: np.ndarray) -> np.ndarray:
        return sampson_distance(model, normalized1, normalized2)

    return ransac(len(points1), 5, fit, residuals, settings)


def triangulation_angles(
    points3d: np.ndarray, center1: np.ndarray, center2: np.ndarray
) -> np.ndarray:
    """Angle in degrees subtended at each point by two camera centres.

    Small parallax angles make depth ill conditioned, so this is the standard
    criterion for accepting a triangulated point or for choosing an
    initialization pair.

    Parameters
    ----------
    points3d : ndarray, shape (n, 3)
    center1, center2 : ndarray, shape (3,)

    Returns
    -------
    ndarray, shape (n,)
    """
    ray1 = points3d - center1
    ray2 = points3d - center2
    ray1 /= np.maximum(np.linalg.norm(ray1, axis=1, keepdims=True), 1e-12)
    ray2 /= np.maximum(np.linalg.norm(ray2, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(np.einsum("ij,ij->i", ray1, ray2), -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))
