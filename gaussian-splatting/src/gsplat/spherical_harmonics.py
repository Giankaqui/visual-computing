"""Real spherical harmonics up to degree three.

View-dependent colour is stored as coefficients of the real spherical harmonic
basis evaluated along the viewing direction.  Degree zero is a constant, so the
first coefficient behaves like a diffuse albedo, and the higher bands add the
angular variation that makes specular highlights and Fresnel-like rim effects
possible without a shading model.

The basis is the standard real form, orthonormal on the unit sphere with respect
to the ordinary surface measure.  Values are hard-coded rather than evaluated
through associated Legendre recurrences: with only sixteen terms the closed form
is faster, allocation-free, and directly differentiable.
"""

from __future__ import annotations

import torch

__all__ = ["MAX_DEGREE", "num_coefficients", "evaluate", "rgb_to_dc", "dc_to_rgb"]

MAX_DEGREE = 3

_C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def num_coefficients(degree: int) -> int:
    """Number of coefficients of a basis truncated at ``degree``."""
    if not 0 <= degree <= MAX_DEGREE:
        raise ValueError(f"degree must be in [0, {MAX_DEGREE}], got {degree}")
    return (degree + 1) ** 2


def evaluate(coefficients: torch.Tensor, directions: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate a spherical harmonic expansion.

    Parameters
    ----------
    coefficients : Tensor, shape (n, k, c)
        Coefficients per primitive, where ``k`` is at least
        ``num_coefficients(degree)`` and ``c`` is the number of channels.
    directions : Tensor, shape (n, 3)
        Unit viewing directions.
    degree : int
        Bands to include; coefficients beyond it are ignored, which is what makes
        the progressive band schedule during training a no-op on the tensor
        layout.

    Returns
    -------
    Tensor, shape (n, c)
    """
    if coefficients.shape[1] < num_coefficients(degree):
        raise ValueError(
            f"degree {degree} needs {num_coefficients(degree)} coefficients, "
            f"got {coefficients.shape[1]}"
        )

    value = _C0 * coefficients[:, 0]
    if degree == 0:
        return value

    x, y, z = directions[:, 0:1], directions[:, 1:2], directions[:, 2:3]
    value = (
        value
        - _C1 * y * coefficients[:, 1]
        + _C1 * z * coefficients[:, 2]
        - _C1 * x * coefficients[:, 3]
    )
    if degree == 1:
        return value

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    value = (
        value
        + _C2[0] * xy * coefficients[:, 4]
        + _C2[1] * yz * coefficients[:, 5]
        + _C2[2] * (2.0 * zz - xx - yy) * coefficients[:, 6]
        + _C2[3] * xz * coefficients[:, 7]
        + _C2[4] * (xx - yy) * coefficients[:, 8]
    )
    if degree == 2:
        return value

    return (
        value
        + _C3[0] * y * (3.0 * xx - yy) * coefficients[:, 9]
        + _C3[1] * xy * z * coefficients[:, 10]
        + _C3[2] * y * (4.0 * zz - xx - yy) * coefficients[:, 11]
        + _C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coefficients[:, 12]
        + _C3[4] * x * (4.0 * zz - xx - yy) * coefficients[:, 13]
        + _C3[5] * z * (xx - yy) * coefficients[:, 14]
        + _C3[6] * x * (xx - 3.0 * yy) * coefficients[:, 15]
    )


def rgb_to_dc(rgb: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB into the degree-zero coefficient.

    Colour is reconstructed as ``sh(direction) + 0.5`` so that a zero expansion
    is mid grey and the residual bands are centred; this inverts that mapping.
    """
    return (rgb - 0.5) / _C0


def dc_to_rgb(dc: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`rgb_to_dc`."""
    return _C0 * dc + 0.5
