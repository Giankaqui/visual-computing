import numpy as np
import pytest

from sfm.camera import Pose
from sfm.rotations import random_rotation
from sfm.triangulation import refine_point, triangulate_dlt, triangulate_multiview


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(99)


def _views(rng: np.random.Generator, count: int) -> list[Pose]:
    poses = [Pose()]
    for _ in range(count - 1):
        poses.append(Pose(R=random_rotation(rng, 0.3), t=rng.normal(scale=0.8, size=3)))
    return poses


def _observe(poses: list[Pose], points: np.ndarray) -> np.ndarray:
    stacked = []
    for pose in poses:
        camera_points = pose.transform(points)
        stacked.append(camera_points[:, :2] / camera_points[:, 2:])
    return np.stack(stacked)


def test_two_view_triangulation_is_exact_without_noise(rng: np.random.Generator) -> None:
    poses = _views(rng, 2)
    points = rng.normal(scale=1.0, size=(50, 3)) + np.array([0.0, 0.0, 6.0])
    observations = _observe(poses, points)

    estimated = triangulate_dlt(poses[0].matrix, poses[1].matrix, observations[0], observations[1])
    assert np.allclose(estimated, points, atol=1e-9)


def test_parallel_rays_are_flagged_as_infinite() -> None:
    first, second = Pose(), Pose(t=np.array([1.0, 0.0, 0.0]))
    identical = np.array([[0.0, 0.0]])
    estimated = triangulate_dlt(first.matrix, second.matrix, identical, identical)
    assert not np.isfinite(estimated).all()


def test_multiview_triangulation_is_exact_without_noise(rng: np.random.Generator) -> None:
    poses = _views(rng, 5)
    point = np.array([0.3, -0.4, 7.0])
    observations = _observe(poses, point[None])[:, 0]
    projections = np.stack([pose.matrix for pose in poses])

    assert np.allclose(triangulate_multiview(projections, observations), point, atol=1e-9)


def test_refinement_reduces_error_under_noise(rng: np.random.Generator) -> None:
    poses = _views(rng, 4)
    point = np.array([-0.2, 0.5, 9.0])
    projections = np.stack([pose.matrix for pose in poses])
    observations = _observe(poses, point[None])[:, 0]
    observations = observations + rng.normal(scale=2e-3, size=observations.shape)

    linear = triangulate_multiview(projections, observations)
    refined = refine_point(linear, projections, observations)

    def reprojection_cost(candidate: np.ndarray) -> float:
        predicted = np.stack(
            [
                (pose.transform(candidate[None])[0, :2] / pose.transform(candidate[None])[0, 2])
                for pose in poses
            ]
        )
        return float(((predicted - observations) ** 2).sum())

    assert reprojection_cost(refined) <= reprojection_cost(linear) + 1e-15


def test_refinement_rejects_points_behind_the_camera(rng: np.random.Generator) -> None:
    poses = _views(rng, 3)
    projections = np.stack([pose.matrix for pose in poses])
    observations = np.zeros((3, 2))
    behind = np.array([0.0, 0.0, -5.0])
    assert np.allclose(refine_point(behind, projections, observations), behind)
