"""Checks on the EWA projection.

The strongest test compares the analytic screen-space covariance against the
empirical covariance of points sampled from the 3D Gaussian and projected
individually.  That is an independent computation of the same quantity, and it
fails if the Jacobian, the rotation or the ordering of the triple product is
wrong.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gsplat.cameras import Camera, look_at
from gsplat.projection import LOW_PASS_VARIANCE, project


@pytest.fixture
def camera() -> Camera:
    R, t = look_at(np.array([0.0, 0.0, -6.0]), np.zeros(3), np.array([0.0, -1.0, 0.0]))
    return Camera.from_fov(R, t, width=256, height=192, fov_x_degrees=50.0)


def _covariance_from(scales: np.ndarray, rotation: np.ndarray) -> torch.Tensor:
    M = torch.as_tensor(rotation @ np.diag(scales), dtype=torch.float64)
    return (M @ M.T)[None]


def test_isotropic_gaussian_projects_to_a_circle(camera: Camera) -> None:
    means = torch.zeros((1, 3), dtype=torch.float64)
    sigma = 0.05
    covariances = _covariance_from(np.full(3, sigma), np.eye(3))

    projected = project(means, covariances, camera)

    assert bool(projected.visible.all())
    assert torch.allclose(
        projected.means2d[0],
        torch.tensor([camera.cx, camera.cy], dtype=torch.float64),
        atol=1e-9,
    )
    # A sphere at the optical axis at distance d has screen-space standard
    # deviation f * sigma / d along both axes, plus the low-pass term.
    expected = (camera.fx * sigma / 6.0) ** 2 + LOW_PASS_VARIANCE
    conic = projected.conics[0]
    assert conic[1].abs() < 1e-9
    assert float(1.0 / conic[0]) == pytest.approx(expected, rel=1e-6)
    assert float(1.0 / conic[2]) == pytest.approx(expected, rel=1e-6)


def test_matches_the_empirical_covariance_of_projected_samples(camera: Camera) -> None:
    generator = np.random.default_rng(0)
    rotation = np.linalg.qr(generator.normal(size=(3, 3)))[0]
    rotation *= np.sign(np.linalg.det(rotation))
    scales = np.array([0.04, 0.02, 0.012])
    mean = np.array([0.35, -0.2, 0.4])

    covariance = _covariance_from(scales, rotation)
    projected = project(
        torch.as_tensor(mean, dtype=torch.float64)[None], covariance, camera
    )

    samples = mean + generator.normal(size=(400_000, 3)) @ (rotation @ np.diag(scales)).T
    camera_points = samples @ camera.R.T + camera.t
    pixels = np.stack(
        [
            camera.fx * camera_points[:, 0] / camera_points[:, 2] + camera.cx,
            camera.fy * camera_points[:, 1] / camera_points[:, 2] + camera.cy,
        ],
        axis=1,
    )
    empirical = np.cov(pixels.T)

    conic = projected.conics[0].numpy()
    analytic = np.linalg.inv(np.array([[conic[0], conic[1]], [conic[1], conic[2]]]))
    analytic -= LOW_PASS_VARIANCE * np.eye(2)

    assert np.allclose(analytic, empirical, rtol=0.02, atol=1e-3)


def test_culls_primitives_behind_and_outside_the_frustum(camera: Camera) -> None:
    means = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -12.0], [40.0, 0.0, 0.0]], dtype=torch.float64
    )
    covariances = _covariance_from(np.full(3, 0.02), np.eye(3)).repeat(3, 1, 1)

    projected = project(means, covariances, camera)

    assert projected.visible.tolist() == [True, False, False]
    assert len(projected) == 1


def test_radius_covers_three_standard_deviations(camera: Camera) -> None:
    means = torch.zeros((1, 3), dtype=torch.float64)
    covariances = _covariance_from(np.array([0.08, 0.02, 0.02]), np.eye(3))

    projected = project(means, covariances, camera)

    conic = projected.conics[0].numpy()
    covariance2d = np.linalg.inv(np.array([[conic[0], conic[1]], [conic[1], conic[2]]]))
    largest = np.linalg.eigvalsh(covariance2d).max()
    assert float(projected.radii[0]) == pytest.approx(3.0 * np.sqrt(largest), rel=1e-6)


def test_returns_empty_result_when_nothing_is_visible(camera: Camera) -> None:
    means = torch.tensor([[0.0, 0.0, -50.0]], dtype=torch.float64)
    projected = project(means, _covariance_from(np.full(3, 0.01), np.eye(3)), camera)

    assert len(projected) == 0
    assert not bool(projected.visible.any())
