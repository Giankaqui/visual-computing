import numpy as np
import pytest

from sfm.ransac import RansacOptions, ransac


def _line_problem(rng: np.random.Generator, outlier_fraction: float, count: int = 400):
    """Points on a line plus uniformly distributed gross outliers."""
    direction = np.array([2.0, 1.0])
    direction /= np.linalg.norm(direction)
    origin = np.array([-1.0, 3.0])

    parameters = rng.uniform(-10, 10, size=count)
    points = origin + parameters[:, None] * direction
    points += rng.normal(scale=0.02, size=points.shape)

    outliers = rng.random(count) < outlier_fraction
    points[outliers] = rng.uniform(-15, 15, size=(int(outliers.sum()), 2))
    return points, outliers


def _fit_line(points: np.ndarray) -> np.ndarray:
    """Total least squares line as ``(a, b, c)`` with ``a^2 + b^2 = 1``."""
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    normal = Vt[-1]
    return np.array([normal[0], normal[1], -normal @ centroid])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(17)


def test_recovers_the_model_under_heavy_contamination(rng: np.random.Generator) -> None:
    points, outliers = _line_problem(rng, outlier_fraction=0.6)

    result = ransac(
        len(points),
        2,
        lambda indices: _fit_line(points[indices]),
        lambda model: np.abs(points @ model[:2] + model[2]),
        RansacOptions(threshold=0.1, seed=3),
    )

    assert result.success
    assert result.inliers[~outliers].mean() > 0.95
    assert result.inliers[outliers].mean() < 0.05


def test_stops_early_when_the_data_is_clean(rng: np.random.Generator) -> None:
    points, _ = _line_problem(rng, outlier_fraction=0.0)
    options = RansacOptions(threshold=0.1, min_iterations=5, max_iterations=5000, seed=3)

    result = ransac(
        len(points),
        2,
        lambda indices: _fit_line(points[indices]),
        lambda model: np.abs(points @ model[:2] + model[2]),
        options,
    )

    assert result.iterations <= 20
    assert result.inliers.mean() > 0.99


def test_local_optimization_lowers_the_cost(rng: np.random.Generator) -> None:
    points, _ = _line_problem(rng, outlier_fraction=0.4)

    def run(local_optimization: bool):
        return ransac(
            len(points),
            2,
            lambda indices: _fit_line(points[indices]),
            lambda model: np.abs(points @ model[:2] + model[2]),
            RansacOptions(
                threshold=0.1,
                seed=3,
                min_iterations=30,
                max_iterations=30,
                local_optimization=local_optimization,
            ),
        )

    assert run(True).score <= run(False).score


def test_reports_failure_when_there_is_not_enough_data() -> None:
    result = ransac(
        1, 2, lambda indices: np.zeros(3), lambda model: np.zeros(1), RansacOptions(threshold=1.0)
    )
    assert not result.success
    assert result.num_inliers == 0


def test_is_reproducible_for_a_fixed_seed(rng: np.random.Generator) -> None:
    points, _ = _line_problem(rng, outlier_fraction=0.3)
    options = RansacOptions(threshold=0.1, seed=42, max_iterations=200)
    arguments = (
        len(points),
        2,
        lambda indices: _fit_line(points[indices]),
        lambda model: np.abs(points @ model[:2] + model[2]),
        options,
    )
    first, second = ransac(*arguments), ransac(*arguments)
    assert np.array_equal(first.inliers, second.inliers)
    assert first.iterations == second.iterations
