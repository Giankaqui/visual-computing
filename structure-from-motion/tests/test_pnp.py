import numpy as np
import pytest

from sfm.camera import PinholeCamera, Pose, project_points
from sfm.pnp import estimate_pose_ransac, pnp_dlt, refine_pose
from sfm.rotations import log_so3, random_rotation


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(31)


@pytest.fixture
def camera() -> PinholeCamera:
    return PinholeCamera.from_fov(800, 600, 60.0)


def _scene(rng: np.random.Generator, camera: PinholeCamera, count: int):
    pose = Pose(
        R=random_rotation(rng, 0.5), t=rng.normal(scale=0.4, size=3) + np.array([0, 0, 7.0])
    )
    points = rng.normal(scale=1.6, size=(count, 3))
    pixels, depths = project_points(points, pose, camera)
    keep = depths > 0.5
    return pose, points[keep], pixels[keep]


def test_dlt_is_exact_without_noise(rng: np.random.Generator, camera: PinholeCamera) -> None:
    pose, points, pixels = _scene(rng, camera, 40)
    estimated = pnp_dlt(points, camera.normalize(pixels))

    assert estimated is not None
    assert np.rad2deg(np.linalg.norm(log_so3(estimated.R @ pose.R.T))) < 1e-6
    assert np.allclose(estimated.t, pose.t, atol=1e-8)


def test_dlt_needs_six_points(rng: np.random.Generator, camera: PinholeCamera) -> None:
    pose, points, pixels = _scene(rng, camera, 5)
    assert pnp_dlt(points[:5], camera.normalize(pixels[:5])) is None


def test_ransac_rejects_outliers(rng: np.random.Generator, camera: PinholeCamera) -> None:
    pose, points, pixels = _scene(rng, camera, 300)
    pixels = pixels + rng.normal(scale=0.5, size=pixels.shape)

    outliers = rng.random(len(pixels)) < 0.35
    pixels[outliers] = rng.uniform(0, [camera.width, camera.height], size=(int(outliers.sum()), 2))

    result = estimate_pose_ransac(points, pixels, camera, pixel_threshold=4.0)
    assert result.success
    assert result.inliers[~outliers].mean() > 0.9
    assert result.inliers[outliers].mean() < 0.05

    estimated = Pose(R=result.model[:, :3], t=result.model[:, 3])
    assert np.rad2deg(np.linalg.norm(log_so3(estimated.R @ pose.R.T))) < 1.0


def test_refinement_improves_a_noisy_pose(rng: np.random.Generator, camera: PinholeCamera) -> None:
    pose, points, pixels = _scene(rng, camera, 120)
    pixels = pixels + rng.normal(scale=0.7, size=pixels.shape)

    perturbed = Pose(
        R=random_rotation(np.random.default_rng(2), 0.02) @ pose.R,
        t=pose.t + np.array([0.05, -0.04, 0.06]),
    )
    refined = refine_pose(perturbed, points, pixels, camera)

    def rotation_error(candidate: Pose) -> float:
        return float(np.rad2deg(np.linalg.norm(log_so3(candidate.R @ pose.R.T))))

    assert rotation_error(refined) < 0.2 * rotation_error(perturbed)
    assert np.linalg.norm(refined.center - pose.center) < 0.02


def test_refinement_is_stable_when_all_points_are_behind(camera: PinholeCamera) -> None:
    pose = Pose(t=np.array([0.0, 0.0, 5.0]))
    behind = np.array([[0.0, 0.0, -10.0], [1.0, 1.0, -12.0]])
    pixels = np.zeros((2, 2))
    assert np.allclose(refine_pose(pose, behind, pixels, camera).R, pose.R)
