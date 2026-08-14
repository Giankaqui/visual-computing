"""Gradient-domain tone mapping of high dynamic range images.

A radiance map can span five orders of magnitude, and a display covers two.
Scaling the values globally destroys either the highlights or the shadows.  The
observation of Fattal, Lischinski and Werman (2002) is that dynamic range lives
in the *large* gradients of the log-radiance image, while detail lives in the
small ones, so attenuating gradients by a factor that decreases with their
magnitude compresses the range and leaves detail alone.

The attenuation has to act at the right scale.  A large edge in the image is not
a single large gradient at full resolution; it is a ramp spread over many pixels,
each of whose gradients is moderate.  The factor is therefore computed on a
Gaussian pyramid and propagated from coarse to fine, so that a pixel sitting on
a large-scale edge is attenuated even when its own local gradient is small.

The attenuated field is not the gradient of any image, so it has to be
integrated in the least-squares sense, which is again a Poisson equation.  Here
the natural boundary condition is Neumann: nothing in the image should be pinned
to a prescribed value, and only differences matter.  That makes the cosine
transform solver in :mod:`gradient_domain.solvers` exact and immediate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import divergence, gradient
from .solvers import solve_neumann

__all__ = ["ToneMapConfig", "attenuation_field", "tone_map"]


@dataclass
class ToneMapConfig:
    """Parameters of the attenuation.

    Attributes
    ----------
    alpha : float
        Gradient magnitude left unchanged, expressed as a multiple of the mean
        gradient magnitude of the level.  Gradients below it are amplified and
        gradients above it are attenuated.
    beta : float
        Exponent in ``(0, 1]``.  The factor is ``(g / alpha) ** (beta - 1)``, so
        ``beta = 1`` leaves every gradient untouched and smaller values attenuate
        the large ones more aggressively.  The original work recommends 0.8 to
        0.9, which is a deliberately mild setting: over-attenuating produces the
        washed-out, low-contrast look that gives tone mapping a bad name.
    saturation : float
        Exponent applied to the chromatic ratios.  Compressing luminance without
        this leaves colours looking oversaturated, because the ratios that
        produced them were tuned to a much larger luminance range.
    levels : int
        Pyramid levels used to propagate the attenuation; the pyramid stops
        early when a level would fall below eight pixels.
    epsilon : float
        Floor on gradient magnitudes, which keeps flat regions from being
        amplified without bound.
    """

    alpha: float = 0.1
    beta: float = 0.85
    saturation: float = 0.5
    levels: int = 8
    epsilon: float = 1e-4


def _downsample(image: np.ndarray) -> np.ndarray:
    """Blur with a binomial kernel and drop every other sample."""
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    padded = np.pad(image, ((0, 0), (2, 2)), mode="edge")
    blurred = sum(weight * padded[:, i : i + image.shape[1]] for i, weight in enumerate(kernel))
    padded = np.pad(blurred, ((2, 2), (0, 0)), mode="edge")
    blurred = sum(weight * padded[i : i + image.shape[0], :] for i, weight in enumerate(kernel))
    return blurred[::2, ::2]


def _upsample(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear interpolation back to a finer grid."""
    rows = np.linspace(0, image.shape[0] - 1, shape[0])
    columns = np.linspace(0, image.shape[1] - 1, shape[1])
    row0 = np.floor(rows).astype(int)
    col0 = np.floor(columns).astype(int)
    row1 = np.minimum(row0 + 1, image.shape[0] - 1)
    col1 = np.minimum(col0 + 1, image.shape[1] - 1)
    row_weight = (rows - row0)[:, None]
    col_weight = (columns - col0)[None, :]

    top = image[row0][:, col0] * (1 - col_weight) + image[row0][:, col1] * col_weight
    bottom = image[row1][:, col0] * (1 - col_weight) + image[row1][:, col1] * col_weight
    return top * (1 - row_weight) + bottom * row_weight


def _central_gradient(image: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Central differences with replicated borders."""
    padded = np.pad(image, 1, mode="edge")
    gx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * spacing)
    gy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * spacing)
    return gx, gy


def attenuation_field(log_luminance: np.ndarray, config: ToneMapConfig) -> np.ndarray:
    """Build the multiscale gradient attenuation factor.

    Parameters
    ----------
    log_luminance : ndarray, shape (h, w)
        Logarithm of the luminance channel.
    config : ToneMapConfig

    Returns
    -------
    ndarray, shape (h, w)
        Per-pixel factor to apply to the gradient field.
    """
    pyramid = [log_luminance]
    while len(pyramid) < config.levels and min(pyramid[-1].shape) > 16:
        pyramid.append(_downsample(pyramid[-1]))

    factor: np.ndarray | None = None
    for level in reversed(range(len(pyramid))):
        gx, gy = _central_gradient(pyramid[level], spacing=2.0**level)
        magnitude = np.sqrt(gx * gx + gy * gy)
        scale = config.alpha * max(float(magnitude.mean()), config.epsilon)
        local = (scale / np.maximum(magnitude, config.epsilon)) * (
            np.maximum(magnitude, config.epsilon) / scale
        ) ** config.beta
        factor = local if factor is None else _upsample(factor, local.shape) * local
    return factor


def tone_map(
    radiance: np.ndarray, config: ToneMapConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compress a radiance map to a displayable image.

    Parameters
    ----------
    radiance : ndarray, shape (h, w, 3)
        Linear radiance with an arbitrary scale; only ratios matter.
    config : ToneMapConfig or None

    Returns
    -------
    image : ndarray of float, shape (h, w, 3)
        Display-referred values in ``[0, 1]``.
    factor : ndarray, shape (h, w)
        The attenuation field that was applied, returned for inspection.

    Raises
    ------
    ValueError
        If the input is not a three-channel image.
    """
    config = config or ToneMapConfig()
    radiance = np.asarray(radiance, dtype=float)
    if radiance.ndim != 3 or radiance.shape[2] != 3:
        raise ValueError("tone mapping expects a three-channel radiance map")

    luminance = radiance @ np.array([0.2126, 0.7152, 0.0722])
    luminance = np.maximum(luminance, np.finfo(float).tiny)
    log_luminance = np.log(luminance)

    factor = attenuation_field(log_luminance, config)
    # The magnitudes that drive the attenuation come from central differences,
    # which are unbiased, but the field that gets integrated uses the forward
    # differences whose exact adjoint is the divergence in `operators`.  Mixing
    # the two stencils here would leave a residual the solver cannot remove.
    gx, gy = gradient(log_luminance)
    compressed = solve_neumann(divergence(gx * factor, gy * factor))

    # The solution is defined up to a constant; anchoring the brightest pixel
    # puts the result in a predictable range before the final normalization.
    result = np.exp(compressed - compressed.max())
    ratios = radiance / luminance[..., None]
    image = (ratios ** config.saturation) * result[..., None]

    peak = float(np.percentile(image, 99.5))
    return np.clip(image / max(peak, 1e-12), 0.0, 1.0), factor
