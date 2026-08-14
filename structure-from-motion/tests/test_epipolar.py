import numpy as np
import pytest

from sfm.camera import Pose
from sfm.epipolar import (
    decompose_essential,
    eight_point_essential,
    estimate_essential,
    recover_pose,
    sampson_distance,
    triangulation_angles,
)
from sfm.five_point import five_point_essential
from sfm.rotations import log_so3, random_rotation, skew
from sfm.synthetic import make_scene


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(7)


def _random_two_view(rng: np.random.Generator, num_points: int):
    """Return an exact essential matrix and matching normalized coordinates."""
    R = random_rotation(rng, max_angle=0.6)
    t = rng.normal(size=3)
    t /= np.linalg.norm(t)

    points = rng.normal(scale=1.5, size=(num_points, 3)) + np.array([0.0, 0.0, 8.0])
    first = points[:, :2] / points[:, 2:]
    transformed = points @ R.T + t
    second = transformed[:, :2] / transformed[:, 2:]

    essential = skew(t) @ R
    return essential / np.linalg.norm(essential), first, second, Pose(R=R, t=t)


def _closest_distance(candidates: list[np.ndarray], truth: np.ndarray) -> float:
    return min(
        min(np.linalg.norm(E - truth), np.linalg.norm(E + truth)) for E in candidates
    )


def test_five_point_recovers_exact_solution(rng: np.random.Generator) -> None:
    for _ in range(40):
        truth, first, second, _ = _random_two_view(rng, 5)
        solutions = five_point_essential(first, second)
        assert solutions
        assert _closest_distance(solutions, truth) < 1e-7


def test_five_point_solutions_satisfy_the_epipolar_constraint(rng: np.random.Generator) -> None:
    _, first, second, _ = _random_two_view(rng, 5)
    for E in five_point_essential(first, second):
        assert np.abs(sampson_distance(E, first, second)).max() < 1e-8


def test_five_point_solutions_are_essential_matrices(rng: np.random.Generator) -> None:
    _, first, second, _ = _random_two_view(rng, 5)
    for E in five_point_essential(first, second):
        singular_values = np.linalg.svd(E, compute_uv=False)
        assert np.isclose(singular_values[0], singular_values[1], atol=1e-6)
        assert singular_values[2] < 1e-7


def test_five_point_rejects_degenerate_input() -> None:
    repeated = np.tile(np.array([[0.1, 0.2]]), (5, 1))
    assert five_point_essential(repeated, repeated) == []


def test_eight_point_recovers_exact_solution(rng: np.random.Generator) -> None:
    truth, first, second, _ = _random_two_view(rng, 40)
    estimated = eight_point_essential(first, second)
    assert estimated is not None
    assert _closest_distance([estimated], truth) < 1e-8


def test_eight_point_needs_eight_points(rng: np.random.Generator) -> None:
    _, first, second, _ = _random_two_view(rng, 7)
    assert eight_point_essential(first, second) is None


def test_decompose_essential_contains_the_true_pose(rng: np.random.Generator) -> None:
    truth, _, _, pose = _random_two_view(rng, 8)
    rotation_errors = [
        np.linalg.norm(log_so3(candidate.R @ pose.R.T)) for candidate in decompose_essential(truth)
    ]
    assert min(rotation_errors) < 1e-8


def test_recover_pose_selects_the_cheiral_solution(rng: np.random.Generator) -> None:
    truth, first, second, pose = _random_two_view(rng, 60)
    recovered, valid = recover_pose(truth, first, second)
    assert valid.all()
    assert np.rad2deg(np.linalg.norm(log_so3(recovered.R @ pose.R.T))) < 1e-6
    assert np.allclose(recovered.t, pose.t, atol=1e-8)


def test_estimate_essential_is_robust_to_outliers() -> None:
    scene = make_scene(num_points=400, num_views=2, outlier_fraction=0.3, seed=11)
    ids_a, pixels_a = scene.observations_for_view(0)
    ids_b, pixels_b = scene.observations_for_view(1)
    shared, index_a, index_b = np.intersect1d(ids_a, ids_b, return_indices=True)
    assert len(shared) > 100

    result = estimate_essential(
        pixels_a[index_a], pixels_b[index_b], scene.camera, scene.camera, pixel_threshold=1.5
    )
    assert result.success
    # Each observation is corrupted independently, so a correspondence survives
    # only when both of its endpoints are clean: roughly 0.7 * 0.7 of the pairs.
    assert result.num_inliers > 0.45 * len(shared)

    relative = Pose(
        R=scene.poses[1].R @ scene.poses[0].R.T,
        t=scene.poses[1].t - scene.poses[1].R @ scene.poses[0].R.T @ scene.poses[0].t,
    )
    pose, _ = recover_pose(
        result.model,
        scene.camera.normalize(pixels_a[index_a][result.inliers]),
        scene.camera.normalize(pixels_b[index_b][result.inliers]),
    )
    assert np.rad2deg(np.linalg.norm(log_so3(pose.R @ relative.R.T))) < 1.0
    baseline = relative.t / np.linalg.norm(relative.t)
    assert np.rad2deg(np.arccos(np.clip(pose.t @ baseline, -1.0, 1.0))) < 2.0


def test_triangulation_angle_is_symmetric_and_bounded() -> None:
    points = np.array([[0.0, 0.0, 0.0]])
    left, right = np.array([-1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0])
    forward = triangulation_angles(points, left, right)
    backward = triangulation_angles(points, right, left)
    assert np.allclose(forward, backward)
    assert np.isclose(forward[0], 90.0)
