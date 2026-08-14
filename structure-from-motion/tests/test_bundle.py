"""Bundle adjustment checks, including a comparison against an independent solver.

The Schur-complement step is the part most likely to hide a sign or indexing
error, and its symptom would be slow convergence rather than an exception.  The
reference test therefore optimizes the same problem with
``scipy.optimize.least_squares``, which uses a completely different algorithm
(trust-region reflective with a numerically differentiated sparse Jacobian), and
requires the two final costs to agree.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from sfm.bundle import BundleOptions, BundleProblem, adjust
from sfm.camera import Pose
from sfm.metrics import align_similarity
from sfm.rotations import exp_so3, log_so3
from sfm.synthetic import make_scene


def _build_problem(scene, perturbation: float, rng: np.random.Generator) -> BundleProblem:
    poses = [
        Pose(
            R=exp_so3(rng.normal(scale=perturbation, size=3)) @ pose.R,
            t=pose.t + rng.normal(scale=2.0 * perturbation, size=3),
        )
        for pose in scene.poses
    ]
    poses[0] = scene.poses[0].copy()
    return BundleProblem(
        poses=poses,
        points=scene.points + rng.normal(scale=5.0 * perturbation, size=scene.points.shape),
        cameras=[scene.camera] * len(poses),
        camera_indices=scene.camera_indices,
        point_indices=scene.point_indices,
        observations=scene.observations,
        constant_poses={0},
    )


def _residual_vector(problem: BundleProblem, parameters: np.ndarray) -> np.ndarray:
    """Reprojection residuals under a global axis-angle parameterization."""
    num_poses = len(problem.poses)
    rotations = parameters[: 3 * num_poses].reshape(num_poses, 3)
    translations = parameters[3 * num_poses : 6 * num_poses].reshape(num_poses, 3)
    points = parameters[6 * num_poses :].reshape(-1, 3)

    matrices = np.stack([exp_so3(vector) for vector in rotations])[problem.camera_indices]
    camera_points = (
        np.einsum("nij,nj->ni", matrices, points[problem.point_indices])
        + translations[problem.camera_indices]
    )
    depth = np.where(camera_points[:, 2] > 1e-6, camera_points[:, 2], 1e-6)
    focal, principal = problem.intrinsics_per_observation()
    projected = focal * (camera_points[:, :2] / depth[:, None]) + principal
    return (projected - problem.observations).reshape(-1)


def _jacobian_sparsity(problem: BundleProblem) -> lil_matrix:
    num_poses, num_points = len(problem.poses), len(problem.points)
    pattern = lil_matrix((2 * problem.num_observations, 6 * num_poses + 3 * num_points), dtype=int)
    rows = np.arange(problem.num_observations)
    for offset in range(2):
        for column in range(3):
            pattern[2 * rows + offset, 3 * problem.camera_indices + column] = 1
            pattern[2 * rows + offset, 3 * num_poses + 3 * problem.camera_indices + column] = 1
            pattern[2 * rows + offset, 6 * num_poses + 3 * problem.point_indices + column] = 1
    return pattern


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(4)


def test_requires_a_constant_pose(rng: np.random.Generator) -> None:
    scene = make_scene(num_points=40, num_views=3, outlier_fraction=0.0, seed=1)
    problem = _build_problem(scene, 0.0, rng)
    problem.constant_poses = set()
    with pytest.raises(ValueError):
        adjust(problem)


def test_recovers_ground_truth_from_perturbed_initialization(rng: np.random.Generator) -> None:
    scene = make_scene(num_points=200, num_views=6, noise_pixels=0.0, outlier_fraction=0.0, seed=2)
    problem = _build_problem(scene, 0.01, rng)

    report = adjust(problem, BundleOptions(max_iterations=80, huber_delta=np.inf))

    assert report.final_rmse < 1e-3
    rotation_errors = [
        np.rad2deg(np.linalg.norm(log_so3(estimated.R @ truth.R.T)))
        for estimated, truth in zip(problem.poses, scene.poses, strict=False)
    ]
    assert max(rotation_errors) < 1e-3

    # Holding one pose constant removes six of the seven gauge degrees of
    # freedom; the overall scale stays free, so the structure is only recovered
    # up to a similarity and has to be aligned before being compared.
    aligned = align_similarity(problem.points, scene.points).apply(problem.points)
    assert np.abs(aligned - scene.points).max() < 1e-5


def test_huber_loss_limits_the_influence_of_outliers(rng: np.random.Generator) -> None:
    scene = make_scene(
        num_points=200, num_views=5, noise_pixels=0.3, outlier_fraction=0.15, seed=5
    )
    squared = _build_problem(scene, 0.008, rng)
    robust = _build_problem(scene, 0.008, rng)

    adjust(squared, BundleOptions(max_iterations=60, huber_delta=np.inf))
    adjust(robust, BundleOptions(max_iterations=60, huber_delta=1.5))

    def worst_rotation_error(problem: BundleProblem) -> float:
        return max(
            np.rad2deg(np.linalg.norm(log_so3(estimated.R @ truth.R.T)))
            for estimated, truth in zip(problem.poses, scene.poses, strict=False)
        )

    assert worst_rotation_error(robust) < worst_rotation_error(squared)


def test_matches_an_independent_least_squares_solver(rng: np.random.Generator) -> None:
    scene = make_scene(num_points=60, num_views=4, noise_pixels=0.4, outlier_fraction=0.0, seed=6)
    problem = _build_problem(scene, 0.01, rng)
    reference = BundleProblem(
        poses=[pose.copy() for pose in problem.poses],
        points=problem.points.copy(),
        cameras=problem.cameras,
        camera_indices=problem.camera_indices,
        point_indices=problem.point_indices,
        observations=problem.observations,
        constant_poses={0},
    )

    report = adjust(problem, BundleOptions(max_iterations=100, huber_delta=np.inf))

    initial = np.concatenate(
        [
            np.concatenate([log_so3(pose.R) for pose in reference.poses]),
            np.concatenate([pose.t for pose in reference.poses]),
            reference.points.reshape(-1),
        ]
    )
    solution = least_squares(
        lambda parameters: _residual_vector(reference, parameters),
        initial,
        jac_sparsity=_jacobian_sparsity(reference),
        method="trf",
        xtol=1e-14,
        ftol=1e-14,
        max_nfev=200,
    )

    schur_cost = 0.5 * report.final_cost
    assert schur_cost <= solution.cost * 1.02
