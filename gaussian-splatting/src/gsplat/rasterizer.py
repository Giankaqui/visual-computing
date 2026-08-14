"""Tile-based differentiable rasterization of screen-space Gaussians.

Compositing every primitive against every pixel costs ``O(H W N)`` and is out of
reach for a scene with a hundred thousand primitives.  The image is therefore
split into square tiles, each primitive is assigned to the tiles its support
overlaps, and the tiles are composited independently.  A primitive
touches only a handful of tiles, so the work becomes proportional to the total
screen area covered rather than to the product of image size and primitive count.

Within a tile the primitives are sorted by depth and blended front to back with

.. math::

    C = \\sum_i c_i \\alpha_i \\prod_{j<i} (1 - \\alpha_j).

The transmittance product looks sequential, but it is a cumulative product along
the depth axis, so a whole tile evaluates as a handful of batched tensor
operations and PyTorch differentiates it without a custom backward pass.  That is
the trade this implementation makes deliberately: a CUDA kernel with a
hand-written backward is faster, and this is readable, portable to CPU and MPS,
and correct by construction.

The dense intermediate has one entry per (tile, primitive, pixel) triple, so its
size is the number of pixels times the occupancy of the busiest tile.  Three
mechanisms keep that bounded.

Primitives whose opacity is below one level of an eight-bit channel never enter a
tile list, since they cannot change the image.

The number of primitives composited by a single tile is capped after the depth
sort, so the nearest ones are kept.  Once accumulated transmittance is negligible
the remaining primitives cannot change the pixel, which is why the cap is usually
free.  When it is not, the artefact is distinctive and easy to misread as a
training failure: neighbouring tiles truncate different numbers of primitives, so
the image acquires visible rectangular steps.  :class:`RenderOutput` therefore
reports how many tiles hit the cap and how much transmittance was still unspent
when they did, which turns the assumption into a measurement.

Tiles are composited in chunks sized to a fixed element budget, and under gradient
each chunk is wrapped in activation checkpointing.  Peak memory becomes a function
of the budget instead of the image size, at the cost of recomputing the forward
pass of each chunk during the backward pass.

Smaller tiles are not obviously worse.  The dense work is the pixel count times
the busiest tile's occupancy, and halving the tile side roughly halves that
occupancy while leaving the pixel count alone; what grows instead is the number
of primitive-tile pairs to sort.  Eight pixels is where the two effects balance
for the scenes here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from .projection import ProjectedGaussians

__all__ = ["RenderOutput", "rasterize"]

MAX_ALPHA = 0.99
DEFAULT_ELEMENT_BUDGET = 24_000_000


@dataclass
class RenderOutput:
    """Result of rasterizing one view.

    Attributes
    ----------
    image : Tensor, shape (h, w, 3)
        Rendered colour, already composited over the background.
    alpha : Tensor, shape (h, w)
        Accumulated opacity, one minus the transmittance that reached the
        background.
    depth : Tensor, shape (h, w)
        Alpha-weighted mean depth, useful for inspection rather than training.
    visible_count : int
        Primitives that survived frustum culling.
    saturated_tiles : int
        Tiles whose primitive list was truncated by the per-tile cap.
    unspent_transmittance : float
        Mean transmittance still available at the truncation point, averaged
        over the pixels of the saturated tiles.  Values near zero mean the
        discarded primitives were fully occluded and the cap cost nothing.
    """

    image: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    visible_count: int
    saturated_tiles: int
    unspent_transmittance: float


def _tile_assignments(
    means2d: torch.Tensor,
    radii: torch.Tensor,
    tiles_x: int,
    tiles_y: int,
    tile_size: int,
    contributing: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand each primitive into the tiles its support overlaps.

    Parameters
    ----------
    means2d, radii : Tensor
        Screen-space centres and support radii.
    tiles_x, tiles_y, tile_size : int
    contributing : Tensor of bool, optional
        Primitives to consider; the rest produce no entries at all, which is
        cheaper than letting them through and multiplying by a zero opacity.

    Returns
    -------
    primitive_index : Tensor, shape (p,)
    tile_index : Tensor, shape (p,)
        Flattened tile identifiers, one entry per primitive-tile incidence.
    """
    device = means2d.device
    low = ((means2d - radii[:, None]) / tile_size).floor().long()
    high = ((means2d + radii[:, None]) / tile_size).floor().long()

    x0 = low[:, 0].clamp(0, tiles_x - 1)
    x1 = high[:, 0].clamp(0, tiles_x - 1)
    y0 = low[:, 1].clamp(0, tiles_y - 1)
    y1 = high[:, 1].clamp(0, tiles_y - 1)

    overlaps = (high[:, 0] >= 0) & (low[:, 0] < tiles_x) & (high[:, 1] >= 0) & (low[:, 1] < tiles_y)
    if contributing is not None:
        overlaps &= contributing
    span_x = torch.where(overlaps, x1 - x0 + 1, torch.zeros_like(x0))
    span_y = torch.where(overlaps, y1 - y0 + 1, torch.zeros_like(y0))
    counts = (span_x * span_y).clamp_min(0)

    total = int(counts.sum().item())
    if total == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty

    primitive_index = torch.repeat_interleave(torch.arange(len(means2d), device=device), counts)
    offsets = torch.cumsum(counts, dim=0) - counts
    local = torch.arange(total, device=device) - offsets[primitive_index]
    width = span_x[primitive_index]
    tile_x = x0[primitive_index] + local % width
    tile_y = y0[primitive_index] + local // width
    return primitive_index, tile_y * tiles_x + tile_x


def _tile_primitive_table(
    primitive_index: torch.Tensor,
    tile_index: torch.Tensor,
    depths: torch.Tensor,
    num_tiles: int,
    max_per_tile: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a dense ``(num_tiles, k)`` table of primitives sorted front to back.

    Empty slots hold ``-1``.  The width ``k`` adapts to the busiest tile, capped
    at ``max_per_tile``, so early training with few primitives does not pay for
    the worst case.

    Returns
    -------
    table : Tensor of int64, shape (num_tiles, k)
    counts : Tensor of int64, shape (num_tiles,)
        Number of primitives that reached each tile before truncation.
    """
    device = primitive_index.device
    by_depth = torch.argsort(depths[primitive_index])
    grouped = torch.sort(tile_index[by_depth], stable=True)
    order = by_depth[grouped.indices]
    sorted_tiles = grouped.values

    counts = torch.bincount(tile_index, minlength=num_tiles)
    offsets = torch.cumsum(counts, dim=0) - counts
    rank = torch.arange(len(order), device=device) - offsets[sorted_tiles]

    width = int(min(int(counts.max().item()), max_per_tile))
    table = torch.full((num_tiles, max(width, 1)), -1, dtype=torch.long, device=device)
    keep = rank < width
    table[sorted_tiles[keep], rank[keep]] = primitive_index[order[keep]]
    return table, counts


def _tile_pixel_centers(
    tiles_x: int, tiles_y: int, tile_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Pixel centres of every tile, shape ``(num_tiles, tile_size ** 2, 2)``."""
    within = torch.arange(tile_size, device=device, dtype=dtype) + 0.5
    offset_y, offset_x = torch.meshgrid(within, within, indexing="ij")
    offsets = torch.stack([offset_x.reshape(-1), offset_y.reshape(-1)], dim=1)

    tile_ids = torch.arange(tiles_x * tiles_y, device=device)
    origins = torch.stack(
        [(tile_ids % tiles_x) * tile_size, (tile_ids // tiles_x) * tile_size], dim=1
    ).to(dtype)
    return origins[:, None, :] + offsets[None, :, :]


def _composite(
    centres: torch.Tensor,
    conics: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    pixels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Blend one chunk of tiles front to back.

    Parameters
    ----------
    centres : Tensor, shape (t, k, 2)
    conics : Tensor, shape (t, k, 3)
    colors : Tensor, shape (t, k, 3)
    opacities : Tensor, shape (t, k)
        Already zeroed for empty slots, which makes them contribute nothing and
        keeps their gradient at zero.
    depths : Tensor, shape (t, k)
    pixels : Tensor, shape (t, p, 2)

    Returns
    -------
    image : Tensor, shape (t, p, 3)
    alpha : Tensor, shape (t, p)
    depth : Tensor, shape (t, p)
    residual : Tensor, shape (t, p)
        Transmittance left after the last composited primitive.
    """
    delta = pixels[:, None, :, :] - centres[:, :, None, :]
    power = -0.5 * (
        conics[:, :, None, 0] * delta[..., 0] ** 2
        + 2.0 * conics[:, :, None, 1] * delta[..., 0] * delta[..., 1]
        + conics[:, :, None, 2] * delta[..., 1] ** 2
    )
    alpha = (opacities[:, :, None] * torch.exp(power.clamp(max=0.0))).clamp(max=MAX_ALPHA)

    transmittance = torch.cumprod(1.0 - alpha, dim=1)
    exclusive = torch.cat([torch.ones_like(transmittance[:, :1]), transmittance[:, :-1]], dim=1)
    weight = alpha * exclusive

    return (
        torch.einsum("tkp,tkc->tpc", weight, colors),
        weight.sum(dim=1),
        torch.einsum("tkp,tk->tp", weight, depths),
        transmittance[:, -1],
    )


def _chunk_size(num_tiles: int, width: int, pixels_per_tile: int, budget: int) -> int:
    """Largest tile chunk whose dense intermediate stays within ``budget`` elements."""
    per_tile = max(width * pixels_per_tile, 1)
    return max(1, min(num_tiles, budget // per_tile))


def rasterize(
    projected: ProjectedGaussians,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    image_height: int,
    image_width: int,
    tile_size: int = 8,
    max_per_tile: int = 2048,
    background: torch.Tensor | None = None,
    element_budget: int = DEFAULT_ELEMENT_BUDGET,
    use_checkpoint: bool = True,
    min_alpha: float = 1.0 / 255.0,
) -> RenderOutput:
    """Composite projected Gaussians into an image.

    Parameters
    ----------
    projected : ProjectedGaussians
        Output of :func:`gsplat.projection.project`.
    colors : Tensor, shape (m, 3)
        Colour of each visible primitive, in the same order as ``projected``.
    opacities : Tensor, shape (m,)
    image_height, image_width : int
    tile_size : int
        Side of a square tile in pixels.
    max_per_tile : int
        Maximum primitives composited per tile, applied after the depth sort.
    background : Tensor, shape (3,), optional
        Colour behind the scene; black when omitted.
    element_budget : int
        Target size of the dense per-chunk intermediate, in tensor elements.
        Tiles are processed in chunks that respect it.
    use_checkpoint : bool
        Recompute each chunk during the backward pass instead of storing its
        activations.  Roughly a third more compute for a large reduction in peak
        memory; it has no effect when gradients are not required.
    min_alpha : float
        Primitives whose peak opacity is below this never reach a tile list.
        One level of an eight-bit channel is the natural threshold: below it the
        primitive cannot change the rendered image.

    Returns
    -------
    RenderOutput
    """
    device = colors.device
    dtype = colors.dtype
    if background is None:
        background = torch.zeros(3, device=device, dtype=dtype)

    tiles_x = (image_width + tile_size - 1) // tile_size
    tiles_y = (image_height + tile_size - 1) // tile_size
    num_tiles = tiles_x * tiles_y

    def empty_output(visible: int) -> RenderOutput:
        image = background.expand(image_height, image_width, 3).clone()
        zeros = torch.zeros((image_height, image_width), device=device, dtype=dtype)
        return RenderOutput(image, zeros, zeros.clone(), visible, 0, 0.0)

    if len(projected) == 0:
        return empty_output(0)

    primitive_index, tile_index = _tile_assignments(
        projected.means2d.detach(),
        projected.radii.detach(),
        tiles_x,
        tiles_y,
        tile_size,
        contributing=opacities.detach() >= min_alpha,
    )
    if len(primitive_index) == 0:
        return empty_output(len(projected))

    table, counts = _tile_primitive_table(
        primitive_index, tile_index, projected.depths.detach(), num_tiles, max_per_tile
    )
    occupied = table >= 0
    gathered = table.clamp_min(0)
    width = table.shape[1]

    pixels = _tile_pixel_centers(tiles_x, tiles_y, tile_size, device, dtype)
    zero = torch.zeros((), device=device, dtype=dtype)

    chunk = _chunk_size(num_tiles, width, tile_size * tile_size, element_budget)
    differentiable = use_checkpoint and torch.is_grad_enabled() and colors.requires_grad

    images, alphas, depths, residuals = [], [], [], []
    for start in range(0, num_tiles, chunk):
        stop = min(start + chunk, num_tiles)
        index = gathered[start:stop]
        arguments = (
            projected.means2d[index],
            projected.conics[index],
            colors[index],
            torch.where(occupied[start:stop], opacities[index], zero),
            projected.depths[index],
            pixels[start:stop],
        )
        parts = (
            checkpoint(_composite, *arguments, use_reentrant=False)
            if differentiable
            else _composite(*arguments)
        )
        images.append(parts[0])
        alphas.append(parts[1])
        depths.append(parts[2])
        residuals.append(parts[3])

    tile_image = torch.cat(images, dim=0)
    tile_alpha = torch.cat(alphas, dim=0)
    tile_depth = torch.cat(depths, dim=0)
    tile_residual = torch.cat(residuals, dim=0)

    padded_height, padded_width = tiles_y * tile_size, tiles_x * tile_size

    def to_image(values: torch.Tensor) -> torch.Tensor:
        channels = values.shape[-1]
        grid = values.reshape(tiles_y, tiles_x, tile_size, tile_size, channels)
        return grid.permute(0, 2, 1, 3, 4).reshape(padded_height, padded_width, channels)[
            :image_height, :image_width
        ]

    alpha_image = to_image(tile_alpha[..., None])
    saturated = counts > width
    unspent = (
        float(tile_residual.detach()[saturated].mean()) if bool(saturated.any()) else 0.0
    )
    return RenderOutput(
        image=to_image(tile_image) + (1.0 - alpha_image) * background,
        alpha=alpha_image[..., 0],
        depth=to_image(tile_depth[..., None])[..., 0],
        visible_count=len(projected),
        saturated_tiles=int(saturated.sum()),
        unspent_transmittance=unspent,
    )
