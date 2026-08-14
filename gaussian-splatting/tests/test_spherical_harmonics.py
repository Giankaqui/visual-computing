"""Checks that the hard-coded basis really is the orthonormal real one."""

from __future__ import annotations

import math

import pytest
import torch

from gsplat import spherical_harmonics as sh


def _fibonacci_sphere(count: int) -> torch.Tensor:
    """Nearly uniform directions on the unit sphere.

    A spiral lattice integrates smooth functions on the sphere far more
    accurately than the same number of random samples, which is what makes an
    orthonormality check practical at this sample count.
    """
    index = torch.arange(count, dtype=torch.float64) + 0.5
    z = 1.0 - 2.0 * index / count
    radius = torch.sqrt((1.0 - z * z).clamp_min(0.0))
    azimuth = index * math.pi * (1.0 + math.sqrt(5.0))
    return torch.stack([radius * torch.cos(azimuth), radius * torch.sin(azimuth), z], dim=1)


def _basis_values(directions: torch.Tensor) -> torch.Tensor:
    """Evaluate every basis function by expanding one coefficient at a time."""
    count = sh.num_coefficients(sh.MAX_DEGREE)
    values = []
    for index in range(count):
        coefficients = torch.zeros((len(directions), count, 1), dtype=torch.float64)
        coefficients[:, index, 0] = 1.0
        values.append(sh.evaluate(coefficients, directions, sh.MAX_DEGREE)[:, 0])
    return torch.stack(values, dim=1)


def test_basis_is_orthonormal() -> None:
    directions = _fibonacci_sphere(200_000)
    values = _basis_values(directions)

    # The quadrature weight of a uniform lattice is the sphere area over the
    # sample count, so the inner product is the mean scaled by 4 pi.
    gram = (values.T @ values) * (4.0 * math.pi / len(directions))
    assert torch.allclose(gram, torch.eye(gram.shape[0], dtype=torch.float64), atol=2e-3)


def test_degree_zero_is_constant() -> None:
    directions = _fibonacci_sphere(64)
    coefficients = torch.zeros((len(directions), 16, 1), dtype=torch.float64)
    coefficients[:, 0, 0] = 1.0
    values = sh.evaluate(coefficients, directions, 0)
    assert torch.allclose(values, values[0].expand_as(values))


def test_truncation_ignores_higher_bands() -> None:
    torch.manual_seed(0)
    directions = _fibonacci_sphere(32)
    coefficients = torch.randn((len(directions), 16, 3), dtype=torch.float64)

    truncated = coefficients.clone()
    truncated[:, sh.num_coefficients(1) :] = 0.0
    assert torch.allclose(
        sh.evaluate(coefficients, directions, 1), sh.evaluate(truncated, directions, 1)
    )


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_coefficient_count(degree: int) -> None:
    assert sh.num_coefficients(degree) == (degree + 1) ** 2


def test_rejects_out_of_range_degree() -> None:
    with pytest.raises(ValueError):
        sh.num_coefficients(4)


def test_rejects_too_few_coefficients() -> None:
    directions = _fibonacci_sphere(8)
    with pytest.raises(ValueError):
        sh.evaluate(torch.zeros((8, 4, 3), dtype=torch.float64), directions, 2)


def test_colour_roundtrip() -> None:
    rgb = torch.rand((32, 3), dtype=torch.float64)
    assert torch.allclose(sh.dc_to_rgb(sh.rgb_to_dc(rgb)), rgb)
