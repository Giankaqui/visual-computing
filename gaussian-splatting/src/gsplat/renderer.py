"""One-call rendering of a Gaussian model into a camera."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cameras import Camera
from .gaussians import GaussianModel
from .projection import ProjectedGaussians, project
from .rasterizer import RenderOutput, rasterize

__all__ = ["RenderResult", "render"]


@dataclass
class RenderResult:
    """A rendered view together with the intermediates density control needs.

    Attributes
    ----------
    output : RenderOutput
        Image, alpha and depth.
    projected : ProjectedGaussians
        Screen-space state of the visible primitives.  ``projected.means2d``
        keeps its gradient, which is the signal that tells the density control
        where the reconstruction is under-parameterized.
    """

    output: RenderOutput
    projected: ProjectedGaussians

    @property
    def image(self) -> torch.Tensor:
        return self.output.image


def render(
    model: GaussianModel,
    camera: Camera,
    background: torch.Tensor | None = None,
    tile_size: int = 8,
    max_per_tile: int = 2048,
    use_checkpoint: bool = True,
    retain_screen_gradients: bool = False,
) -> RenderResult:
    """Render a model from one viewpoint.

    Parameters
    ----------
    model : GaussianModel
    camera : Camera
    background : Tensor, shape (3,), optional
    tile_size : int
        Side of a rasterization tile in pixels.
    max_per_tile : int
        Maximum primitives composited per tile.
    use_checkpoint : bool
        Recompute the compositing chunks during the backward pass instead of
        storing their activations.
    retain_screen_gradients : bool
        Keep the gradient of the projected centres after the backward pass.  It
        is a non-leaf tensor, so PyTorch discards its gradient unless asked;
        training turns this on and inference leaves it off.

    Returns
    -------
    RenderResult
    """
    projected = project(model.means, model.covariances(), camera)
    if retain_screen_gradients and projected.means2d.requires_grad:
        projected.means2d.retain_grad()

    visible = projected.visible
    camera_center = torch.as_tensor(
        camera.center, dtype=model.means.dtype, device=model.means.device
    )
    colors = model.colors(camera_center)[visible]
    opacities = model.opacities[visible]

    output = rasterize(
        projected,
        colors,
        opacities,
        image_height=camera.height,
        image_width=camera.width,
        tile_size=tile_size,
        max_per_tile=max_per_tile,
        background=background,
        use_checkpoint=use_checkpoint,
    )
    return RenderResult(output=output, projected=projected)
