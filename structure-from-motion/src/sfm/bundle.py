"""Sparse bundle adjustment by Levenberg-Marquardt with the Schur complement.

Bundle adjustment minimizes reprojection error over camera poses and 3D points
simultaneously.  The normal equations have a characteristic block structure

.. math::

    \\begin{pmatrix} U & W \\\\ W^\\top & V \\end{pmatrix}
    \\begin{pmatrix} \\delta_c \\\\ \\delta_p \\end{pmatrix}
    = \\begin{pmatrix} g_c \\\\ g_p \\end{pmatrix},

where ``U`` is block diagonal with one ``6 x 6`` block per camera, ``V`` is block
diagonal with one ``3 x 3`` block per point, and ``W`` has one ``6 x 3`` block per
observation.  Because ``V`` is trivially invertible, eliminating the points gives
the reduced camera system ``(U - W V^{-1} W^\\top) \\delta_c = g_c - W V^{-1} g_p``
whose size depends only on the number of cameras.  For a reconstruction with a
few hundred cameras and a hundred thousand points this turns an intractable
dense solve into a sparse system of a few thousand unknowns (Brown, 1958;
Triggs et al., 2000).

Three details matter for behaviour on real data:

* Rotations are updated multiplicatively, ``R <- exp(skew(delta)) R``, so the
  parameterization is a local chart around the current estimate and never
  degenerates the way a global three-parameter representation does near a
  rotation of pi.
* Residuals are reweighted with the Huber loss, so a handful of surviving
  mismatches bend the solution by a bounded amount instead of dominating it.
* The seven-parameter gauge freedom of a free reconstruction is removed by
  holding at least one pose constant; the remaining scale freedom is absorbed by
  the Levenberg-Marquardt damping term, which keeps the reduced system positive
  definite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .camera import PinholeCamera, Pose
from .rotations import exp_so3, rotate_point_jacobian

__all__ = ["BundleProblem", "BundleOptions", "BundleReport", "adjust"]


@dataclass
class BundleProblem:
    """Inputs and outputs of a bundle adjustment run.

    The pose and point arrays are modified in place by :func:`adjust`.

    Attributes
    ----------
    poses : list of Pose
        World-to-camera transforms, one per view.
    points : ndarray, shape (n_points, 3)
    cameras : list of PinholeCamera
        Intrinsics for each view; intrinsics are held fixed.
    camera_indices, point_indices : ndarray, shape (n_observations,)
        Index arrays linking each observation to a view and a point.
    observations : ndarray, shape (n_observations, 2)
        Measured pixel coordinates.
    constant_poses : set of int
        Views excluded from the optimization.  At least one view must be
        constant, otherwise the problem is gauge degenerate.
    """

    poses: list[Pose]
    points: np.ndarray
    cameras: list[PinholeCamera]
    camera_indices: np.ndarray
    point_indices: np.ndarray
    observations: np.ndarray
    constant_poses: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        self.camera_indices = np.asarray(self.camera_indices, dtype=np.int64).reshape(-1)
        self.point_indices = np.asarray(self.point_indices, dtype=np.int64).reshape(-1)
        self.observations = np.asarray(self.observations, dtype=float).reshape(-1, 2)
        if not (len(self.camera_indices) == len(self.point_indices) == len(self.observations)):
            raise ValueError("observation index arrays must have matching lengths")

    @property
    def num_observations(self) -> int:
        return len(self.observations)

    def intrinsics_per_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return focal lengths and principal points broadcast to observations."""
        focal = np.array([[c.fx, c.fy] for c in self.cameras], dtype=float)
        principal = np.array([[c.cx, c.cy] for c in self.cameras], dtype=float)
        return focal[self.camera_indices], principal[self.camera_indices]


@dataclass
class BundleOptions:
    """Levenberg-Marquardt settings.

    Attributes
    ----------
    max_iterations : int
        Maximum accepted steps.
    huber_delta : float
        Residual norm in pixels beyond which the loss grows linearly.  Set to
        ``inf`` for a plain least-squares fit.
    initial_damping : float
        Starting value of the Levenberg-Marquardt parameter, scaled by the
        diagonal of the normal matrix.
    function_tolerance : float
        Stop when the relative cost decrease falls below this value.
    step_tolerance : float
        Stop when the largest parameter update falls below this value.
    max_inner_iterations : int
        Damping increases allowed per outer iteration before giving up.
    verbose : bool
        Print the cost at every accepted step.
    """

    max_iterations: int = 50
    huber_delta: float = 2.0
    initial_damping: float = 1e-4
    function_tolerance: float = 1e-8
    step_tolerance: float = 1e-10
    max_inner_iterations: int = 12
    verbose: bool = False


@dataclass
class BundleReport:
    """Summary of an optimization run.

    Attributes
    ----------
    initial_cost, final_cost : float
        Robustified sum of squared reprojection errors.
    initial_rmse, final_rmse : float
        Root mean squared reprojection error in pixels over all observations.
    iterations : int
        Accepted steps.
    message : str
        Termination reason.
    """

    initial_cost: float
    final_cost: float
    initial_rmse: float
    final_rmse: float
    iterations: int
    message: str

    def __str__(self) -> str:
        return (
            f"bundle adjustment: rmse {self.initial_rmse:.3f} -> {self.final_rmse:.3f} px "
            f"in {self.iterations} iterations ({self.message})"
        )


def _residuals_and_weights(
    problem: BundleProblem,
    poses: list[Pose],
    points: np.ndarray,
    huber_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate reprojection residuals and Huber weights for every observation."""
    rotations = np.stack([p.R for p in poses])[problem.camera_indices]
    translations = np.stack([p.t for p in poses])[problem.camera_indices]
    world_points = points[problem.point_indices]

    camera_points = np.einsum("nij,nj->ni", rotations, world_points) + translations
    depth = camera_points[:, 2]
    valid = depth > 1e-6
    safe_depth = np.where(valid, depth, 1.0)

    focal, principal = problem.intrinsics_per_observation()
    projected = focal * (camera_points[:, :2] / safe_depth[:, None]) + principal
    residual = np.where(valid[:, None], projected - problem.observations, 0.0)

    norms = np.linalg.norm(residual, axis=1)
    if np.isfinite(huber_delta):
        weights = np.where(norms > huber_delta, huber_delta / np.maximum(norms, 1e-12), 1.0)
    else:
        weights = np.ones_like(norms)
    weights = np.where(valid, weights, 0.0)
    return residual, weights, camera_points, safe_depth, valid


def _cost(residual: np.ndarray, weights: np.ndarray) -> float:
    return float((weights * (residual**2).sum(axis=1)).sum())


def _rmse(residual: np.ndarray, valid: np.ndarray) -> float:
    if not np.any(valid):
        return float("inf")
    return float(np.sqrt((residual[valid] ** 2).sum() / valid.sum()))


def _block_bincount(values: np.ndarray, indices: np.ndarray, size: int) -> np.ndarray:
    """Sum ``values`` of shape ``(n, a, b)`` into ``size`` groups given by ``indices``."""
    n, a, b = values.shape
    flat = values.reshape(n, a * b)
    summed = np.stack(
        [np.bincount(indices, weights=flat[:, k], minlength=size) for k in range(a * b)],
        axis=1,
    )
    return summed.reshape(size, a, b)


def _sparse_from_blocks(
    blocks: np.ndarray, row_index: np.ndarray, col_index: np.ndarray, shape: tuple[int, int]
) -> sp.csr_matrix:
    """Assemble blocks of shape ``(n, a, b)`` into a sparse matrix, summing duplicates."""
    n, a, b = blocks.shape
    rows = (a * row_index[:, None, None] + np.arange(a)[None, :, None]).repeat(b, axis=2)
    cols = (b * col_index[:, None, None] + np.arange(b)[None, None, :]).repeat(a, axis=1)
    return sp.coo_matrix(
        (blocks.reshape(-1), (rows.reshape(-1), cols.reshape(-1))), shape=shape
    ).tocsr()


def _damped(blocks: np.ndarray, damping: float) -> np.ndarray:
    """Add the Levenberg-Marquardt term, scaled by each block's own diagonal.

    Scaling by the diagonal rather than by the identity makes the step invariant
    to the units of the parameters, which matters here because rotation
    increments are in radians while translations are in scene units.
    """
    size = blocks.shape[1]
    diagonal = np.maximum(np.diagonal(blocks, axis1=1, axis2=2), 1e-12)
    return blocks + damping * (diagonal[:, :, None] * np.eye(size))


def adjust(problem: BundleProblem, options: BundleOptions | None = None) -> BundleReport:
    """Optimize poses and points in place, minimizing robust reprojection error.

    Parameters
    ----------
    problem : BundleProblem
        Modified in place; ``poses`` and ``points`` hold the solution on return.
    options : BundleOptions or None

    Returns
    -------
    BundleReport

    Raises
    ------
    ValueError
        If no pose is held constant, which leaves the problem gauge degenerate.
    """
    options = options or BundleOptions()
    if not problem.constant_poses:
        raise ValueError("at least one pose must be held constant to fix the gauge")

    num_poses = len(problem.poses)
    num_points = len(problem.points)
    free_poses = [i for i in range(num_poses) if i not in problem.constant_poses]
    num_free = len(free_poses)

    free_index = np.full(num_poses, -1, dtype=np.int64)
    free_index[free_poses] = np.arange(num_free)
    observation_slot = free_index[problem.camera_indices]
    optimizes_pose = observation_slot >= 0
    pose_rows = np.where(optimizes_pose, observation_slot, 0)

    poses = [p.copy() for p in problem.poses]
    points = problem.points.copy()
    focal, _ = problem.intrinsics_per_observation()

    residual, weights, camera_points, depth, valid = _residuals_and_weights(
        problem, poses, points, options.huber_delta
    )
    cost = _cost(residual, weights)
    initial_cost, initial_rmse = cost, _rmse(residual, valid)

    damping = options.initial_damping
    message = "reached iteration limit"
    iteration = 0

    while iteration < options.max_iterations:
        d_pixel = np.zeros((problem.num_observations, 2, 3), dtype=float)
        d_pixel[:, 0, 0] = focal[:, 0] / depth
        d_pixel[:, 1, 1] = focal[:, 1] / depth
        d_pixel[:, 0, 2] = -focal[:, 0] * camera_points[:, 0] / depth**2
        d_pixel[:, 1, 2] = -focal[:, 1] * camera_points[:, 1] / depth**2
        d_pixel[~valid] = 0.0

        rotations = np.stack([p.R for p in poses])[problem.camera_indices]
        rotated = np.einsum("nij,nj->ni", rotations, points[problem.point_indices])

        jacobian_point = np.einsum("nij,njk->nik", d_pixel, rotations)
        jacobian_pose = np.concatenate(
            [np.einsum("nij,njk->nik", d_pixel, rotate_point_jacobian(rotated)), d_pixel], axis=2
        )
        jacobian_pose[~optimizes_pose] = 0.0

        weighted = weights[:, None, None]
        V_blocks = _block_bincount(
            weighted * np.einsum("nik,nil->nkl", jacobian_point, jacobian_point),
            problem.point_indices,
            num_points,
        )
        gradient_point = -_block_bincount(
            (weights[:, None] * np.einsum("nik,ni->nk", jacobian_point, residual))[:, :, None],
            problem.point_indices,
            num_points,
        ).reshape(-1)

        if num_free:
            U_blocks = _block_bincount(
                weighted * np.einsum("nik,nil->nkl", jacobian_pose, jacobian_pose),
                pose_rows,
                num_free,
            )
            gradient_pose = -_block_bincount(
                (weights[:, None] * np.einsum("nik,ni->nk", jacobian_pose, residual))[:, :, None],
                pose_rows,
                num_free,
            ).reshape(-1)
            W_blocks = weighted * np.einsum("nik,nil->nkl", jacobian_pose, jacobian_point)
            block_shape = (6 * num_free, 3 * num_points)
            W_sparse = _sparse_from_blocks(
                W_blocks, pose_rows, problem.point_indices, block_shape
            )

        accepted = False
        converged = ""
        for _ in range(options.max_inner_iterations):
            V_inverse = np.linalg.inv(_damped(V_blocks, damping))

            if num_free:
                Y_sparse = _sparse_from_blocks(
                    np.einsum("nij,njk->nik", W_blocks, V_inverse[problem.point_indices]),
                    pose_rows,
                    problem.point_indices,
                    block_shape,
                )
                camera_block_index = np.arange(num_free)
                reduced = (
                    _sparse_from_blocks(
                        _damped(U_blocks, damping),
                        camera_block_index,
                        camera_block_index,
                        (6 * num_free, 6 * num_free),
                    )
                    - Y_sparse @ W_sparse.T
                ).tocsc()
                try:
                    step_pose = spla.spsolve(reduced, gradient_pose - Y_sparse @ gradient_point)
                except (RuntimeError, ValueError, np.linalg.LinAlgError):
                    damping *= 10.0
                    continue
                if not np.all(np.isfinite(step_pose)):
                    damping *= 10.0
                    continue
                point_rhs = gradient_point - W_sparse.T @ step_pose
            else:
                step_pose = np.zeros(0)
                point_rhs = gradient_point

            step_point = np.einsum(
                "nij,nj->ni", V_inverse, point_rhs.reshape(num_points, 3)
            )

            candidate_poses = [p.copy() for p in poses]
            for slot, pose_index in enumerate(free_poses):
                increment = step_pose[6 * slot : 6 * slot + 6]
                candidate_poses[pose_index] = Pose(
                    R=exp_so3(increment[:3]) @ poses[pose_index].R,
                    t=poses[pose_index].t + increment[3:],
                )
            candidate_points = points + step_point

            candidate_terms = _residuals_and_weights(
                problem, candidate_poses, candidate_points, options.huber_delta
            )
            candidate_cost = _cost(candidate_terms[0], candidate_terms[1])
            if candidate_cost >= cost:
                damping *= 10.0
                continue

            relative_decrease = (cost - candidate_cost) / max(cost, 1e-30)
            step_size = max(
                float(np.abs(step_pose).max(initial=0.0)),
                float(np.abs(step_point).max(initial=0.0)),
            )
            poses, points, cost = candidate_poses, candidate_points, candidate_cost
            residual, weights, camera_points, depth, valid = candidate_terms
            damping = max(damping * 0.3, 1e-12)
            accepted = True
            iteration += 1

            if options.verbose:
                print(
                    f"  iter {iteration:3d}  cost {cost:.6e}  "
                    f"rmse {_rmse(residual, valid):.4f} px  lambda {damping:.2e}"
                )
            if relative_decrease < options.function_tolerance:
                converged = "converged on cost"
            elif step_size < options.step_tolerance:
                converged = "converged on step size"
            break

        if not accepted:
            message = "damping saturated"
            break
        if converged:
            message = converged
            break

    problem.poses[:] = poses
    problem.points[...] = points
    return BundleReport(
        initial_cost=initial_cost,
        final_cost=cost,
        initial_rmse=initial_rmse,
        final_rmse=_rmse(residual, valid),
        iterations=iteration,
        message=message,
    )
