"""Checks on the tile-based rasterizer.

The compositing law, the depth ordering and the tiling are tested separately,
and the whole path is checked against numerical differentiation.  Gradients are
where a hand-written rasterizer usually goes wrong, and a finite-difference
comparison catches sign errors and misplaced detaches that a forward-only test
cannot.
"""

from __future__ import annotations

import pytest
import torch

from gsplat.projection import ProjectedGaussians
from gsplat.rasterizer import rasterize


def _isotropic(
    centres: torch.Tensor, sigma: float, depths: torch.Tensor, sigma_cutoff: float = 3.0
) -> ProjectedGaussians:
    count = len(centres)
    inverse = 1.0 / sigma**2
    conics = torch.tensor([[inverse, 0.0, inverse]], dtype=centres.dtype).repeat(count, 1)
    return ProjectedGaussians(
        means2d=centres,
        conics=conics,
        depths=depths.to(centres.dtype),
        radii=torch.full((count,), sigma_cutoff * sigma, dtype=centres.dtype),
        visible=torch.ones(count, dtype=torch.bool),
    )


def test_single_gaussian_matches_the_analytic_profile() -> None:
    size, sigma, opacity = 32, 3.0, 0.6
    centre = torch.tensor([[16.0, 16.0]], dtype=torch.float64)
    projected = _isotropic(centre, sigma, torch.tensor([1.0], dtype=torch.float64))

    output = rasterize(
        projected,
        colors=torch.ones((1, 3), dtype=torch.float64),
        opacities=torch.tensor([opacity], dtype=torch.float64),
        image_height=size,
        image_width=size,
        tile_size=8,
    )

    grid = torch.arange(size, dtype=torch.float64) + 0.5
    y, x = torch.meshgrid(grid, grid, indexing="ij")
    expected = opacity * torch.exp(-0.5 * ((x - 16.0) ** 2 + (y - 16.0) ** 2) / sigma**2)
    assert torch.allclose(output.alpha, expected, atol=1e-9)


def test_front_primitive_occludes_the_one_behind() -> None:
    centres = torch.tensor([[8.0, 8.0], [8.0, 8.0]], dtype=torch.float64)
    projected = _isotropic(centres, 2.0, torch.tensor([1.0, 5.0], dtype=torch.float64))
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)

    output = rasterize(
        projected,
        colors=colors,
        opacities=torch.tensor([0.95, 0.95], dtype=torch.float64),
        image_height=16,
        image_width=16,
        tile_size=8,
    )

    centre_pixel = output.image[8, 8]
    assert centre_pixel[0] > 4.0 * centre_pixel[2]

    swapped = rasterize(
        _isotropic(centres, 2.0, torch.tensor([5.0, 1.0], dtype=torch.float64)),
        colors=colors,
        opacities=torch.tensor([0.95, 0.95], dtype=torch.float64),
        image_height=16,
        image_width=16,
        tile_size=8,
    )
    assert swapped.image[8, 8, 2] > 4.0 * swapped.image[8, 8, 0]


def test_background_shows_through_where_nothing_is_drawn() -> None:
    projected = _isotropic(
        torch.tensor([[2.0, 2.0]], dtype=torch.float64), 0.8, torch.tensor([1.0])
    )
    background = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

    output = rasterize(
        projected,
        colors=torch.ones((1, 3), dtype=torch.float64),
        opacities=torch.tensor([0.9], dtype=torch.float64),
        image_height=32,
        image_width=32,
        tile_size=8,
        background=background,
    )

    assert torch.allclose(output.image[30, 30], background, atol=1e-9)
    assert float(output.alpha[30, 30]) < 1e-9


def test_empty_input_returns_the_background() -> None:
    empty = ProjectedGaussians(
        means2d=torch.zeros((0, 2)),
        conics=torch.zeros((0, 3)),
        depths=torch.zeros(0),
        radii=torch.zeros(0),
        visible=torch.zeros(4, dtype=torch.bool),
    )
    background = torch.tensor([0.4, 0.5, 0.6])
    output = rasterize(empty, torch.zeros((0, 3)), torch.zeros(0), 8, 8, background=background)

    assert torch.allclose(output.image, background.expand(8, 8, 3))
    assert float(output.alpha.max()) == 0.0


@pytest.mark.parametrize("tile_size", [4, 8, 16])
def test_tiling_does_not_change_the_image(tile_size: int) -> None:
    generator = torch.Generator().manual_seed(3)
    count = 40
    centres = torch.rand((count, 2), generator=generator, dtype=torch.float64) * 40.0
    depths = torch.rand(count, generator=generator, dtype=torch.float64) + 1.0
    # A primitive only reaches the tiles its radius overlaps, so a cutoff wide
    # enough to make the discarded tail negligible is what isolates the tiling
    # logic from the support truncation it deliberately performs.
    projected = _isotropic(centres, 2.5, depths, sigma_cutoff=8.0)
    colors = torch.rand((count, 3), generator=generator, dtype=torch.float64)
    opacities = torch.rand(count, generator=generator, dtype=torch.float64) * 0.5

    reference = rasterize(projected, colors, opacities, 40, 40, tile_size=16, max_per_tile=4096)
    candidate = rasterize(
        projected, colors, opacities, 40, 40, tile_size=tile_size, max_per_tile=4096
    )
    assert torch.allclose(reference.image, candidate.image, atol=1e-9)


def test_chunking_does_not_change_the_image() -> None:
    generator = torch.Generator().manual_seed(5)
    count = 30
    centres = torch.rand((count, 2), generator=generator, dtype=torch.float64) * 32.0
    depths = torch.rand(count, generator=generator, dtype=torch.float64) + 1.0
    projected = _isotropic(centres, 2.0, depths)
    colors = torch.rand((count, 3), generator=generator, dtype=torch.float64)
    opacities = torch.rand(count, generator=generator, dtype=torch.float64)

    single = rasterize(projected, colors, opacities, 32, 32, element_budget=10**9)
    chunked = rasterize(projected, colors, opacities, 32, 32, element_budget=1)
    assert torch.allclose(single.image, chunked.image, atol=1e-12)


def test_reports_truncation_when_the_cap_binds() -> None:
    count = 24
    centres = torch.full((count, 2), 8.0, dtype=torch.float64)
    depths = torch.arange(1, count + 1, dtype=torch.float64)
    projected = _isotropic(centres, 4.0, depths)
    colors = torch.ones((count, 3), dtype=torch.float64)
    opacities = torch.full((count,), 0.05, dtype=torch.float64)

    capped = rasterize(projected, colors, opacities, 16, 16, tile_size=16, max_per_tile=4)
    full = rasterize(projected, colors, opacities, 16, 16, tile_size=16, max_per_tile=count)

    assert capped.saturated_tiles == 1
    assert capped.unspent_transmittance > 0.5
    assert full.saturated_tiles == 0
    assert float(full.alpha.max()) > float(capped.alpha.max())


def test_gradients_match_finite_differences() -> None:
    dtype = torch.float64
    centres = torch.tensor([[3.1, 3.4], [4.6, 4.2], [2.7, 5.1]], dtype=dtype, requires_grad=True)
    conics = torch.tensor(
        [[0.32, 0.05, 0.27], [0.21, -0.04, 0.31], [0.28, 0.02, 0.24]],
        dtype=dtype,
        requires_grad=True,
    )
    colors = torch.tensor(
        [[0.8, 0.2, 0.1], [0.1, 0.7, 0.3], [0.2, 0.3, 0.9]], dtype=dtype, requires_grad=True
    )
    opacities = torch.tensor([0.35, 0.45, 0.25], dtype=dtype, requires_grad=True)
    depths = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    radii = torch.full((3,), 6.0, dtype=dtype)

    def rendered(
        centres: torch.Tensor,
        conics: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
    ) -> torch.Tensor:
        projected = ProjectedGaussians(
            means2d=centres,
            conics=conics,
            depths=depths,
            radii=radii,
            visible=torch.ones(3, dtype=torch.bool),
        )
        # One tile keeps the discrete tile assignment constant under the
        # perturbations gradcheck applies, so only the smooth part is compared.
        return rasterize(
            projected, colors, opacities, 8, 8, tile_size=8, max_per_tile=8, use_checkpoint=False
        ).image

    assert torch.autograd.gradcheck(
        rendered, (centres, conics, colors, opacities), eps=1e-6, atol=1e-7, rtol=1e-5
    )


def test_checkpointing_produces_the_same_gradients() -> None:
    dtype = torch.float64

    def run(use_checkpoint: bool) -> torch.Tensor:
        centres = torch.tensor(
            [[6.0, 6.5], [9.5, 7.0], [7.5, 10.0]], dtype=dtype, requires_grad=True
        )
        projected = ProjectedGaussians(
            means2d=centres,
            conics=torch.tensor(
                [[0.2, 0.0, 0.2], [0.15, 0.02, 0.18], [0.25, -0.03, 0.22]], dtype=dtype
            ),
            depths=torch.tensor([1.0, 2.0, 3.0], dtype=dtype),
            radii=torch.full((3,), 7.0, dtype=dtype),
            visible=torch.ones(3, dtype=torch.bool),
        )
        colors = torch.full((3, 3), 0.5, dtype=dtype)
        opacities = torch.tensor([0.4, 0.5, 0.6], dtype=dtype)
        output = rasterize(
            projected,
            colors,
            opacities,
            16,
            16,
            tile_size=8,
            element_budget=1 if use_checkpoint else 10**9,
            use_checkpoint=use_checkpoint,
        )
        output.image.pow(2).sum().backward()
        return centres.grad

    assert torch.allclose(run(True), run(False), atol=1e-12)
