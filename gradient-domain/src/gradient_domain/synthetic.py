"""Procedural test images.

Gradient-domain methods are easiest to judge on images with the properties they
are designed for: a destination with slowly varying illumination, a source lit
differently from it, texture that is separable from structure, and a radiance map
whose dynamic range genuinely exceeds a display.  Generating them procedurally
keeps the demonstrations reproducible and the repository free of binary assets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "value_noise",
    "CompositingExample",
    "make_compositing_example",
    "make_texture_example",
    "make_radiance_map",
]


def _bilinear_resize(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.linspace(0, grid.shape[0] - 1, shape[0])
    columns = np.linspace(0, grid.shape[1] - 1, shape[1])
    row0, col0 = np.floor(rows).astype(int), np.floor(columns).astype(int)
    row1 = np.minimum(row0 + 1, grid.shape[0] - 1)
    col1 = np.minimum(col0 + 1, grid.shape[1] - 1)
    row_weight, col_weight = (rows - row0)[:, None], (columns - col0)[None, :]

    top = grid[row0][:, col0] * (1 - col_weight) + grid[row0][:, col1] * col_weight
    bottom = grid[row1][:, col0] * (1 - col_weight) + grid[row1][:, col1] * col_weight
    return top * (1 - row_weight) + bottom * row_weight


def value_noise(
    shape: tuple[int, int],
    rng: np.random.Generator,
    octaves: int = 5,
    base: int = 4,
    persistence: float = 0.5,
) -> np.ndarray:
    """Fractal value noise in ``[0, 1]``.

    Each octave is a random grid at twice the previous frequency, interpolated up
    and added with half the amplitude, which gives the roughly ``1 / f`` spectrum
    that natural texture has.

    Parameters
    ----------
    shape : tuple of int
    rng : numpy.random.Generator
    octaves : int
        Number of frequency bands.
    base : int
        Resolution of the coarsest band.
    persistence : float
        Amplitude ratio between consecutive bands.

    Returns
    -------
    ndarray, shape (h, w)
    """
    total = np.zeros(shape)
    amplitude, normalization = 1.0, 0.0
    for octave in range(octaves):
        resolution = base * 2**octave
        grid = rng.random((min(resolution, shape[0]), min(resolution, shape[1])))
        total += amplitude * _bilinear_resize(grid, shape)
        normalization += amplitude
        amplitude *= persistence
    return total / normalization


@dataclass
class CompositingExample:
    """A destination image, a source patch and the selection to transfer.

    Attributes
    ----------
    target : ndarray, shape (h, w, 3)
    source : ndarray, shape (sh, sw, 3)
    mask : ndarray of bool, shape (sh, sw)
    offset : tuple of int
        Where the patch's top-left corner sits in the target.
    """

    target: np.ndarray
    source: np.ndarray
    mask: np.ndarray
    offset: tuple[int, int]

    def naive_composite(self) -> np.ndarray:
        """Copy the selected pixels straight into the target, seam and all."""
        row, column = self.offset
        height, width = self.source.shape[:2]
        result = self.target.copy()
        window = result[row : row + height, column : column + width]
        result[row : row + height, column : column + width] = np.where(
            self.mask[..., None], self.source, window
        )
        return result


def _sky(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A dusk gradient with soft cloud banding."""
    height, width = shape
    vertical = np.linspace(0.0, 1.0, height)[:, None]
    zenith = np.array([0.16, 0.28, 0.52])
    horizon = np.array([0.94, 0.72, 0.46])
    sky = zenith + (horizon - zenith) * (vertical[..., None] ** 2.2)

    clouds = value_noise(shape, rng, octaves=5, base=3)
    banding = np.clip((clouds - 0.45) * 2.6, 0.0, 1.0) * (1.0 - vertical) ** 0.5
    return sky * np.ones((height, width, 1)) + banding[..., None] * np.array(
        [0.22, 0.18, 0.12]
    )


def make_compositing_example(
    shape: tuple[int, int] = (384, 576), seed: int = 0
) -> CompositingExample:
    """Build a dusk landscape and a balloon photographed under a brighter sky.

    The two images differ by a global illumination offset and a colour cast,
    which is exactly the situation where copying pixels fails and copying
    gradients works: the offset lives entirely in the boundary condition and is
    therefore replaced by the destination's own.
    """
    rng = np.random.default_rng(seed)
    height, width = shape

    target = _sky(shape, rng)
    horizon = int(0.62 * height)
    rows, _ = np.mgrid[0:height, 0:width]

    ridge = 0.5 + 0.5 * value_noise((1, width), rng, octaves=4, base=3)[0]
    hills = rows >= (horizon - (18 + 26 * ridge).astype(int))[None, :]
    haze = np.linspace(0.0, 1.0, height)[:, None, None]
    hill_color = np.broadcast_to(
        np.array([0.20, 0.22, 0.30]) * (1.0 - 0.5 * haze)
        + np.array([0.30, 0.26, 0.28]) * haze,
        (height, width, 3),
    )
    target[hills] = hill_color[hills]

    texture = value_noise(shape, rng, octaves=6, base=6)
    ground_color = np.array([0.30, 0.26, 0.18]) + 0.28 * texture[..., None] * np.array(
        [0.6, 0.5, 0.35]
    )
    depth = np.clip((rows - horizon) / max(height - horizon, 1), 0.0, 1.0)
    ground = rows >= horizon
    target[ground] = (ground_color * (0.55 + 0.45 * depth[..., None]))[ground]

    patch_size = max(int(0.30 * min(shape)), 24)
    grid = np.linspace(-1.0, 1.0, patch_size)
    px, py = np.meshgrid(grid, grid)
    radius = np.sqrt(px**2 + py**2)

    # The selection is a disc noticeably larger than the balloon, so the seam
    # runs through the source's own sky rather than across the object.  That is
    # the condition under which gradient-domain compositing behaves: the
    # boundary must lie where source and destination are plausibly similar,
    # because everything that differs between them is absorbed there.
    balloon_radius = 0.58
    mask = radius <= 0.94

    # The balloon is lit from the upper left and painted with radial stripes, so
    # it carries both a smooth shading ramp and hard chromatic edges.
    stripes = (np.floor((np.arctan2(py, px) + np.pi) / (np.pi / 5.0)).astype(int)) % 2
    base_colors = np.where(
        stripes[..., None] == 0, np.array([0.90, 0.24, 0.20]), np.array([0.96, 0.86, 0.30])
    )
    normalized = np.clip(radius / balloon_radius, 0.0, 1.0)
    normal_z = np.sqrt(np.clip(1.0 - normalized**2, 0.0, 1.0))
    shading = 0.35 + 0.65 * np.clip(
        (-0.5 * px / balloon_radius - 0.5 * py / balloon_radius + 0.7 * normal_z) / 1.2, 0.0, 1.0
    )
    source = base_colors * shading[..., None]

    # The patch was "photographed" against a bright noon sky, one stop brighter
    # and much cooler than the destination.
    source = source * 1.9 * np.array([0.92, 0.96, 1.06])
    background = np.array([0.62, 0.78, 0.98]) * (
        0.85 + 0.15 * value_noise((patch_size, patch_size), rng, octaves=3, base=2)[..., None]
    )
    source = np.where(radius[..., None] <= balloon_radius, source, background)

    offset = (int(0.14 * height), int(0.16 * width))
    return CompositingExample(
        target=np.clip(target, 0.0, 1.0),
        source=np.clip(source, 0.0, 1.0),
        mask=mask,
        offset=offset,
    )


def make_texture_example(
    shape: tuple[int, int] = (320, 320), seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """A shape with strong texture, and the edge map that defines its structure.

    Returns
    -------
    image : ndarray, shape (h, w, 3)
    edges : ndarray of float, shape (h, w)
        One on the edges that should survive flattening, zero elsewhere.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width]

    grain = value_noise(shape, rng, octaves=6, base=8)
    fine = 0.5 + 0.5 * np.sin(rows * 0.55) * np.sin(columns * 0.47)
    texture = 0.6 * grain + 0.4 * fine

    centre = np.array([height * 0.5, width * 0.5])
    radius = np.sqrt((rows - centre[0]) ** 2 + (columns - centre[1]) ** 2)
    disc = radius < 0.32 * min(height, width)
    band = np.abs(columns - width * 0.5) < 0.09 * width

    image = np.empty((height, width, 3))
    image[...] = (np.array([0.24, 0.30, 0.42]) * (0.7 + 0.6 * texture[..., None]))
    image[disc] = (np.array([0.86, 0.52, 0.22]) * (0.7 + 0.6 * texture[..., None]))[disc]
    image[disc & band] = (
        np.array([0.30, 0.62, 0.48]) * (0.7 + 0.6 * texture[..., None])
    )[disc & band]

    # Structural edges are the boundaries between the three regions, dilated by
    # one pixel so that the gradient straddling each boundary is preserved.
    labels = disc.astype(int) + (disc & band).astype(int)
    padded = np.pad(labels, 1, mode="edge")
    edges = np.zeros(shape)
    for shift in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rolled = padded[
            1 + shift[0] : shape[0] + 1 + shift[0], 1 + shift[1] : shape[1] + 1 + shift[1]
        ]
        edges = np.maximum(edges, (rolled != labels).astype(float))
    return np.clip(image, 0.0, 1.0), edges


def make_radiance_map(
    shape: tuple[int, int] = (320, 448), seed: int = 2
) -> np.ndarray:
    """An interior with a bright window, spanning about five orders of magnitude.

    The window is four decades brighter than the wall it is set into, and the
    wall itself carries texture that is invisible under any global tone curve
    that keeps the window from clipping.

    Returns
    -------
    ndarray, shape (h, w, 3)
        Linear radiance, strictly positive.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width]

    plaster = 0.6 + 0.8 * value_noise(shape, rng, octaves=6, base=5)
    radiance = 0.9 * plaster[..., None] * np.array([0.55, 0.50, 0.44])

    # A wooden floor in the lower fifth, darker still.
    floor = rows > 0.80 * height
    planks = 0.5 + 0.5 * np.sin(columns * 0.35 + 3.0 * value_noise(shape, rng, octaves=3, base=4))
    radiance[floor] = (0.25 * planks[..., None] * np.array([0.42, 0.28, 0.18]))[floor]

    # A framed window onto a sunlit exterior.
    window = (
        (columns > 0.52 * width)
        & (columns < 0.88 * width)
        & (rows > 0.14 * height)
        & (rows < 0.62 * height)
    )
    frame = (
        (columns > 0.50 * width)
        & (columns < 0.90 * width)
        & (rows > 0.12 * height)
        & (rows < 0.64 * height)
    ) & ~window
    radiance[frame] = (0.30 * np.array([0.30, 0.22, 0.16]))[None, :]

    exterior_sky = np.array([0.55, 0.72, 1.00]) * 9.0e3
    exterior_ground = np.array([0.72, 0.78, 0.42]) * 2.2e3
    skyline = rows < 0.44 * height
    outside = np.where(
        skyline[..., None],
        exterior_sky * (0.75 + 0.5 * value_noise(shape, rng, octaves=4, base=3)[..., None]),
        exterior_ground * (0.6 + 0.8 * value_noise(shape, rng, octaves=5, base=6)[..., None]),
    )
    radiance[window] = outside[window]

    # A mullion crossing the window, and a small lamp in the dark corner.
    mullion = window & (np.abs(columns - 0.70 * width) < 0.006 * width)
    radiance[mullion] = (5.0 * np.array([0.30, 0.24, 0.18]))[None, :]

    lamp_radius = np.sqrt((rows - 0.74 * height) ** 2 + (columns - 0.16 * width) ** 2)
    glow = np.exp(-((lamp_radius / (0.05 * width)) ** 2))
    radiance += (60.0 * glow)[..., None] * np.array([1.0, 0.86, 0.62])

    # A small ambient term stands in for interreflection.  Without it the
    # darkest pixels are numerically zero, which would make the dynamic range a
    # property of the floating point format rather than of the scene.
    return radiance + 0.02
