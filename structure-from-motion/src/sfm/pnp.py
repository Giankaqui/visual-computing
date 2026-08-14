"""Absolute pose estimation from 2D-3D correspondences.

The hypothesis generator is the direct linear transform on six correspondences.
A three-point solver would shrink the RANSAC sample and therefore the number of
hypotheses, but during incremental reconstruction the 2D-3D inlier ratio is
already high, since the correspondences come from tracks that survived two-view
verification.  At an inlier ratio of 0.7 the adaptive bound is under 90 samples
for six-point sampling, which is not the bottleneck of the pipeline.  Accuracy
is recovered by :func:`refine_pose`, which runs a robust nonlinear fit on the
full inlier set.
"""

from __future__ import annotations

import numpy as np

from .camera import PinholeCamera, Pose, project_points
from .ransac import RansacOptions, RansacResult, ransac
from .rotations import exp_so3, project_to_so3, rotate_point_jacobian

__all__ = ["pnp_dlt", "estimate_pose_ransac", "refine_pose"]


def _isotropic_normalization(points: np.ndarray, target_distance: float) -> np.ndarray:
    """Return the similarity that centres ``points`` and scales their mean radius.

    Parameters
    ----------
    points : ndarray, shape (n, d)
    target_distance : float
        Mean distance to the origin after the transform, conventionally
        ``sqrt(d)``.

    Returns
    -------
    ndarray, shape (d + 1, d + 1)
        Homogeneous transform to apply to the points.
    """
    dimension = points.shape[1]
    centroid = points.mean(axis=0)
    mean_radius = float(np.linalg.norm(points - centroid, axis=1).mean())
    scale = target_distance / max(mean_radius, 1e-12)

    transform = np.eye(dimension + 1)
    transform[:dimension, :dimension] *= scale
    transform[:dimension, dimension] = -scale * centroid
    return transform


def pnp_dlt(points3d: np.ndarray, normalized: np.ndarray) -> Pose | None:
    """Solve for a camera pose from at least six 2D-3D correspondences.

    Each correspondence gives two rows of ``x x ([R|t] X) = 0``.  Both point sets
    are centred and isotropically scaled before the system is assembled, which is
    what keeps the algebraic least-squares solution close to the geometric one
    (Hartley, 1997); without it the entries of the design matrix span several
    orders of magnitude and the smallest singular vector is dominated by
    rounding error.

    The twelve entries of the projection matrix are then recovered as that
    singular vector and projected back onto ``SE(3)``: the scale is the mean
    singular value of the leading ``3 x 3`` block and the sign is fixed by
    requiring a positive determinant, which is equivalent to placing the points
    in front of the camera.

    Parameters
    ----------
    points3d : ndarray, shape (n, 3)
    normalized : ndarray, shape (n, 2)
        Normalized image coordinates of the same points.

    Returns
    -------
    Pose or None
        ``None`` when the system has no isolated solution, which happens for
        fewer than six points or when the configuration is degenerate, for
        instance when every point lies on a plane through the camera centre.
    """
    points3d = np.asarray(points3d, dtype=float)
    normalized = np.asarray(normalized, dtype=float)
    n = len(points3d)
    if n < 6:
        return None

    world_transform = _isotropic_normalization(points3d, np.sqrt(3.0))
    image_transform = _isotropic_normalization(normalized, np.sqrt(2.0))
    homogeneous = np.hstack([points3d, np.ones((n, 1))]) @ world_transform.T
    image = np.hstack([normalized, np.ones((n, 1))]) @ image_transform.T

    A = np.zeros((2 * n, 12), dtype=float)
    A[0::2, 0:4] = homogeneous
    A[0::2, 8:12] = -image[:, 0, None] * homogeneous
    A[1::2, 4:8] = homogeneous
    A[1::2, 8:12] = -image[:, 1, None] * homogeneous

    _, singular_values, Vt = np.linalg.svd(A)
    # A one-dimensional null space is what makes the solution unique.  Comparing
    # the two smallest singular values detects the degenerate configurations;
    # the smallest one alone is only zero for noise-free data.
    if singular_values[-2] < 1e-8 * singular_values[0]:
        return None

    P = np.linalg.inv(image_transform) @ Vt[-1].reshape(3, 4) @ world_transform
    if np.linalg.det(P[:, :3]) < 0:
        P = -P

    U, block_singular_values, Vt_block = np.linalg.svd(P[:, :3])
    depth_scale = float(block_singular_values.mean())
    if depth_scale < 1e-12:
        return None
    R = project_to_so3(U @ Vt_block)
    return Pose(R=R, t=P[:, 3] / depth_scale)


def estimate_pose_ransac(
    points3d: np.ndarray,
    pixels: np.ndarray,
    camera: PinholeCamera,
    pixel_threshold: float = 4.0,
    options: RansacOptions | None = None,
) -> RansacResult:
    """Robustly estimate a camera pose from 2D-3D correspondences.

    Parameters
    ----------
    points3d : ndarray, shape (n, 3)
    pixels : ndarray, shape (n, 2)
    camera : PinholeCamera
    pixel_threshold : float
        Reprojection error above which a correspondence is an outlier.
    options : RansacOptions or None

    Returns
    -------
    RansacResult
        ``model`` is a ``(3, 4)`` array holding ``[R | t]``, which is turned back
        into a pose with ``Pose(model[:, :3], model[:, 3])``.
    """
    points3d = np.asarray(points3d, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    normalized = camera.normalize(pixels)

    base = options or RansacOptions(threshold=pixel_threshold)
    settings = RansacOptions(
        threshold=pixel_threshold,
        confidence=base.confidence,
        max_iterations=base.max_iterations,
        min_iterations=min(base.min_iterations, 30),
        local_optimization=base.local_optimization,
        seed=base.seed,
    )

    def fit(indices: np.ndarray) -> list[np.ndarray]:
        pose = pnp_dlt(points3d[indices], normalized[indices])
        return [pose.matrix] if pose is not None else []

    def residuals(model: np.ndarray) -> np.ndarray:
        pose = Pose(R=model[:, :3], t=model[:, 3])
        projected, depths = project_points(points3d, pose, camera)
        errors = np.linalg.norm(projected - pixels, axis=1)
        return np.where(depths > 1e-6, errors, np.inf)

    return ransac(len(points3d), 6, fit, residuals, settings)


def refine_pose(
    pose: Pose,
    points3d: np.ndarray,
    pixels: np.ndarray,
    camera: PinholeCamera,
    huber_delta: float = 4.0,
    iterations: int = 30,
) -> Pose:
    """Refine a pose by robust Levenberg-Marquardt on reprojection error.

    The rotation increment is applied on the left of the current rotation, so the
    Jacobian of a rotated point with respect to the increment is the
    cross-product matrix of the rotated point and no chart singularity is ever
    approached.  Outliers are down-weighted with iteratively reweighted least
    squares using the Huber loss.

    Parameters
    ----------
    pose : Pose
        Initial estimate, typically from :func:`estimate_pose_ransac`.
    points3d : ndarray, shape (n, 3)
    pixels : ndarray, shape (n, 2)
    camera : PinholeCamera
    huber_delta : float
        Residual norm in pixels beyond which the loss becomes linear.
    iterations : int
        Maximum Levenberg-Marquardt steps.

    Returns
    -------
    Pose
    """
    points3d = np.asarray(points3d, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    current = pose.copy()
    damping = 1e-4

    def cost_and_terms(candidate: Pose):
        camera_points = candidate.transform(points3d)
        depth = camera_points[:, 2]
        valid = depth > 1e-6
        if not np.any(valid):
            return np.inf, None, None

        safe_depth = np.where(valid, depth, 1.0)
        projected = camera.denormalize(camera_points[:, :2] / safe_depth[:, None])
        residual = projected - pixels
        residual[~valid] = 0.0

        norms = np.linalg.norm(residual, axis=1)
        weights = np.where(norms > huber_delta, huber_delta / np.maximum(norms, 1e-12), 1.0)
        weights[~valid] = 0.0
        cost = float((weights * norms**2).sum())
        return cost, (camera_points, safe_depth, residual, weights), valid

    cost, terms, _ = cost_and_terms(current)
    if terms is None:
        return current

    improvement = np.inf
    for _ in range(iterations):
        camera_points, depth, residual, weights = terms

        d_projection = np.zeros((len(points3d), 2, 3), dtype=float)
        d_projection[:, 0, 0] = camera.fx / depth
        d_projection[:, 1, 1] = camera.fy / depth
        d_projection[:, 0, 2] = -camera.fx * camera_points[:, 0] / depth**2
        d_projection[:, 1, 2] = -camera.fy * camera_points[:, 1] / depth**2

        rotated = points3d @ current.R.T
        J = np.concatenate(
            [
                np.einsum("nij,njk->nik", d_projection, rotate_point_jacobian(rotated)),
                d_projection,
            ],
            axis=2,
        ).reshape(-1, 6)

        weight_per_residual = np.repeat(weights, 2)[:, None]
        normal_matrix = J.T @ (weight_per_residual * J)
        gradient = J.T @ (weight_per_residual[:, 0] * residual.reshape(-1))

        for _ in range(10):
            try:
                step = np.linalg.solve(
                    normal_matrix + damping * np.diag(np.maximum(np.diag(normal_matrix), 1e-9)),
                    -gradient,
                )
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = Pose(R=exp_so3(step[:3]) @ current.R, t=current.t + step[3:])
            new_cost, new_terms, _ = cost_and_terms(candidate)
            if new_cost < cost:
                improvement = cost - new_cost
                current, cost, terms = candidate, new_cost, new_terms
                damping = max(damping * 0.3, 1e-10)
                break
            damping *= 10.0
        else:
            break

        if improvement < 1e-9 * max(cost, 1.0):
            break

    return current
