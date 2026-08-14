import numpy as np
import pytest

from sfm.rotations import (
    exp_so3,
    log_so3,
    project_to_so3,
    random_rotation,
    rotate_point_jacobian,
    skew,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260813)


def test_skew_matches_cross_product(rng: np.random.Generator) -> None:
    a, b = rng.normal(size=3), rng.normal(size=3)
    assert np.allclose(skew(a) @ b, np.cross(a, b))


def test_exp_produces_rotations(rng: np.random.Generator) -> None:
    for _ in range(50):
        R = exp_so3(rng.normal(scale=2.0, size=3))
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_exp_log_roundtrip(rng: np.random.Generator) -> None:
    for _ in range(300):
        R = random_rotation(rng)
        assert np.allclose(exp_so3(log_so3(R)), R, atol=1e-9)


@pytest.mark.parametrize("angle", [0.0, 1e-9, np.pi - 1e-6, np.pi])
def test_log_handles_boundary_angles(angle: float, rng: np.random.Generator) -> None:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    R = exp_so3(axis * angle)
    assert np.allclose(exp_so3(log_so3(R)), R, atol=1e-7)


def test_rotation_jacobian_matches_finite_differences(rng: np.random.Generator) -> None:
    R = random_rotation(rng)
    point = rng.normal(size=3)
    rotated = R @ point

    step = 1e-6
    numerical = np.stack(
        [
            (exp_so3(step * axis) @ rotated - exp_so3(-step * axis) @ rotated) / (2 * step)
            for axis in np.eye(3)
        ],
        axis=1,
    )
    assert np.allclose(rotate_point_jacobian(rotated), numerical, atol=1e-8)


def test_project_to_so3_is_identity_on_rotations(rng: np.random.Generator) -> None:
    R = random_rotation(rng)
    assert np.allclose(project_to_so3(R), R, atol=1e-12)


def test_project_to_so3_removes_reflections(rng: np.random.Generator) -> None:
    reflection = random_rotation(rng) @ np.diag([1.0, 1.0, -1.0])
    projected = project_to_so3(reflection)
    assert np.isclose(np.linalg.det(projected), 1.0)
    assert np.allclose(projected.T @ projected, np.eye(3), atol=1e-12)
