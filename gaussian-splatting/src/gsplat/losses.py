"""Photometric losses and image quality metrics.

The training objective mixes an L1 term with a structural dissimilarity term.
L1 alone converges to a blurry optimum because it is indifferent to where the
error sits as long as its magnitude is the same, while SSIM compares local means,
variances and covariances and therefore penalizes exactly the loss of local
contrast that blur produces.  The mixing weight follows the original 3D Gaussian
splatting work, which puts most of the weight on L1 and uses SSIM as a
regularizer rather than as the primary signal.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional

__all__ = ["l1_loss", "ssim", "photometric_loss", "psnr"]

_C1 = 0.01**2
_C2 = 0.03**2


def _gaussian_window(size: int, sigma: float, device: torch.device, dtype: torch.dtype):
    """Return a separable 1D Gaussian kernel normalized to unit sum."""
    coordinates = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(coordinates**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _to_nchw(image: torch.Tensor) -> torch.Tensor:
    """Accept ``(h, w, c)`` or ``(n, c, h, w)`` and return the latter."""
    if image.dim() == 3:
        return image.permute(2, 0, 1)[None]
    if image.dim() == 4:
        return image
    raise ValueError(f"expected an image with 3 or 4 dimensions, got {image.dim()}")


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between two images."""
    return (prediction - target).abs().mean()


def ssim(
    prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5
) -> torch.Tensor:
    """Mean structural similarity over an image.

    Local statistics are estimated with a Gaussian window applied separably,
    which is the formulation of Wang et al. (2004).  Padding replicates the
    border so that the metric is not depressed by a dark frame around the image.

    Parameters
    ----------
    prediction, target : Tensor
        Images shaped ``(h, w, c)`` or ``(n, c, h, w)`` with values in ``[0, 1]``.
    window_size : int
        Side of the square window in pixels.
    sigma : float
        Standard deviation of the window.

    Returns
    -------
    Tensor
        Scalar in ``[-1, 1]``; one for identical images.
    """
    x, y = _to_nchw(prediction), _to_nchw(target)
    channels = x.shape[1]
    kernel = _gaussian_window(window_size, sigma, x.device, x.dtype)
    kernel_x = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    kernel_y = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padding = window_size // 2

    def blur(image: torch.Tensor) -> torch.Tensor:
        padded = functional.pad(image, (padding, padding, padding, padding), mode="replicate")
        horizontal = functional.conv2d(padded, kernel_x, groups=channels)
        return functional.conv2d(horizontal, kernel_y, groups=channels)

    mu_x, mu_y = blur(x), blur(y)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_xx = blur(x * x) - mu_xx
    sigma_yy = blur(y * y) - mu_yy
    sigma_xy = blur(x * y) - mu_xy

    numerator = (2 * mu_xy + _C1) * (2 * sigma_xy + _C2)
    denominator = (mu_xx + mu_yy + _C1) * (sigma_xx + sigma_yy + _C2)
    return (numerator / denominator).mean()


def photometric_loss(
    prediction: torch.Tensor, target: torch.Tensor, ssim_weight: float = 0.2
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine L1 and structural dissimilarity.

    Parameters
    ----------
    prediction, target : Tensor, shape (h, w, 3)
    ssim_weight : float
        Weight of the ``1 - SSIM`` term.

    Returns
    -------
    loss : Tensor
        Scalar to backpropagate.
    components : dict
        The two terms, detached, for logging.
    """
    absolute = l1_loss(prediction, target)
    structural = ssim(prediction, target)
    loss = (1.0 - ssim_weight) * absolute + ssim_weight * (1.0 - structural)
    return loss, {"l1": float(absolute.detach()), "ssim": float(structural.detach())}


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in decibels, for images in ``[0, 1]``."""
    mse = float(((prediction - target) ** 2).mean().detach())
    return float("inf") if mse <= 0 else -10.0 * float(torch.log10(torch.tensor(mse)))
