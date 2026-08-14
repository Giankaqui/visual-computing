"""Projection of 3D Gaussians to screen-space conics.

A perspective projection is not affine, so the image of a Gaussian is not a
Gaussian.  The EWA splatting approximation (Zwicker et al., 2001) linearizes the
projection at each primitive's centre and pushes the covariance through that
linear map:

.. math::

    \\Sigma_{2D} = J W \\Sigma W^\\top J^\\top,

where ``W`` is the world-to-camera rotation and ``J`` is the Jacobian of the
perspective divide.  The approximation is accurate while a primitive subtends a
small angle, which is exactly the regime a splatting representation operates in;
it degrades for primitives that fill a large part of the frame, and the density
control is what keeps those from persisting.

Two corrections are applied to the projected covariance.  A low-pass term adds a
fraction of a pixel of variance along both axes, which prevents primitives
smaller than a pixel from disappearing between samples and from producing the
aliasing that would otherwise dominate at low resolution.  The Jacobian is
evaluated on ray directions clamped to the field of view, so primitives just
outside the frustum do not receive the extreme derivatives of the perspective
divide near the image plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cameras import Camera

__all__ = ["ProjectedGaussians", "project"]

LOW_PASS_VARIANCE = 0.3


@dataclass
class ProjectedGaussians:
    """Screen-space representation of the visible primitives.

    Attributes
    ----------
    means2d : Tensor, shape (m, 2)
        Projected centres in pixels.  Gradients with respect to this tensor
        drive the density control, so the trainer retains them.
    conics : Tensor, shape (m, 3)
        Upper triangle ``(a, b, c)`` of the inverse 2D covariance, so that the
        exponent of the splat is ``-0.5 (a dx^2 + 2 b dx dy + c dy^2)``.
    depths : Tensor, shape (m,)
        Distance along the optical axis, used to sort front to back.
    radii : Tensor, shape (m,)
        Screen-space radius covering three standard deviations.
    visible : Tensor of bool, shape (n,)
        Mask selecting which of the input primitives survived culling.
    """

    means2d: torch.Tensor
    conics: torch.Tensor
    depths: torch.Tensor
    radii: torch.Tensor
    visible: torch.Tensor

    def __len__(self) -> int:
        return self.means2d.shape[0]


def _perspective_jacobian(
    camera_points: torch.Tensor, fx: float, fy: float, limit_x: float, limit_y: float
) -> torch.Tensor:
    """Jacobian of the perspective divide, evaluated on clamped ray directions.

    Parameters
    ----------
    camera_points : Tensor, shape (m, 3)
    fx, fy : float
        Focal lengths in pixels.
    limit_x, limit_y : float
        Maximum absolute value of ``x / z`` and ``y / z`` that is still inside
        the frustum, with margin.

    Returns
    -------
    Tensor, shape (m, 2, 3)
    """
    depth = camera_points[:, 2].clamp_min(1e-6)
    x = torch.clamp(camera_points[:, 0] / depth, -limit_x, limit_x) * depth
    y = torch.clamp(camera_points[:, 1] / depth, -limit_y, limit_y) * depth

    zero = torch.zeros_like(depth)
    return torch.stack(
        [
            fx / depth, zero, -fx * x / depth**2,
            zero, fy / depth, -fy * y / depth**2,
        ],
        dim=1,
    ).reshape(-1, 2, 3)


def project(
    means: torch.Tensor,
    covariances: torch.Tensor,
    camera: Camera,
    near: float = 0.01,
    frustum_margin: float = 1.3,
) -> ProjectedGaussians:
    """Project 3D Gaussians into a camera and cull the ones that cannot contribute.

    Parameters
    ----------
    means : Tensor, shape (n, 3)
        Primitive centres in world coordinates.
    covariances : Tensor, shape (n, 3, 3)
        World-space covariances.
    camera : Camera
    near : float
        Primitives closer than this to the image plane are dropped; the
        projection is unbounded as the depth goes to zero.
    frustum_margin : float
        Multiplier on the half field of view used both for culling and for
        clamping the Jacobian.  Values above one keep primitives whose centre is
        outside the frame but whose support is not.

    Returns
    -------
    ProjectedGaussians
    """
    device = means.device
    R, t = camera.to_tensors(device=device, dtype=means.dtype)

    camera_points = means @ R.T + t
    depths = camera_points[:, 2]

    limit_x = frustum_margin * max(camera.cx, camera.width - camera.cx) / camera.fx
    limit_y = frustum_margin * max(camera.cy, camera.height - camera.cy) / camera.fy
    inside = (
        (depths > near)
        & (camera_points[:, 0].abs() < limit_x * depths.clamp_min(1e-6))
        & (camera_points[:, 1].abs() < limit_y * depths.clamp_min(1e-6))
    )
    if not bool(inside.any()):
        empty = torch.zeros((0,), device=device, dtype=means.dtype)
        return ProjectedGaussians(
            means2d=empty.reshape(0, 2),
            conics=empty.reshape(0, 3),
            depths=empty,
            radii=empty,
            visible=inside,
        )

    camera_points = camera_points[inside]
    depths = camera_points[:, 2]
    means2d = torch.stack(
        [
            camera.fx * camera_points[:, 0] / depths + camera.cx,
            camera.fy * camera_points[:, 1] / depths + camera.cy,
        ],
        dim=1,
    )

    J = _perspective_jacobian(camera_points, camera.fx, camera.fy, limit_x, limit_y)
    transform = J @ R
    covariance2d = transform @ covariances[inside] @ transform.transpose(1, 2)

    a = covariance2d[:, 0, 0] + LOW_PASS_VARIANCE
    b = covariance2d[:, 0, 1]
    c = covariance2d[:, 1, 1] + LOW_PASS_VARIANCE
    determinant = (a * c - b * b).clamp_min(1e-9)
    conics = torch.stack([c / determinant, -b / determinant, a / determinant], dim=1)

    # Radius of the circle that contains the three-sigma ellipse: three standard
    # deviations along the major axis, whose square is the larger eigenvalue of
    # the 2x2 covariance.
    half_trace = 0.5 * (a + c)
    offset = torch.sqrt((0.5 * (a - c)) ** 2 + b * b)
    radii = 3.0 * torch.sqrt((half_trace + offset).clamp_min(1e-9))

    return ProjectedGaussians(
        means2d=means2d, conics=conics, depths=depths, radii=radii, visible=inside
    )
