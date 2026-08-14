"""The scene representation: a set of anisotropic 3D Gaussians.

Each primitive carries a mean, an anisotropic covariance, an opacity and a
spherical harmonic expansion for view-dependent colour.  Every quantity that has
a constrained range is stored through an unconstrained parameterization, so
gradient descent never has to be projected back onto a feasible set:

===================  ==========================  =========================
Quantity             Stored as                   Recovered by
===================  ==========================  =========================
scale                logarithm                   ``exp``
opacity              logit                       ``sigmoid``
rotation             unnormalized quaternion     normalize, then to matrix
covariance           scale and rotation          ``R S S^T R^T``
===================  ==========================  =========================

Factoring the covariance as ``R S S^T R^T`` rather than storing six free
coefficients is what keeps it positive semidefinite throughout optimization; an
unconstrained symmetric matrix drifts indefinite within a few hundred steps and
the projection to screen space then produces a conic with no interior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import spherical_harmonics as sh

__all__ = ["GaussianModel", "InitializationConfig"]


@dataclass
class InitializationConfig:
    """Settings for building a model from a sparse point cloud.

    Attributes
    ----------
    sh_degree : int
        Highest spherical harmonic band the model can represent.
    initial_opacity : float
        Opacity assigned to every primitive.  Starting well below one lets the
        optimizer discover which primitives matter instead of having to erode an
        opaque cloud.
    neighbours : int
        Number of nearest neighbours whose mean distance sets the initial scale.
    max_initial_scale : float
        Upper bound on the initial scale, as a fraction of the scene extent.
        Isolated points would otherwise start as scene-sized blobs.
    """

    sh_degree: int = 3
    initial_opacity: float = 0.1
    neighbours: int = 3
    max_initial_scale: float = 0.02


def quaternion_to_rotation(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices.

    Parameters
    ----------
    quaternions : Tensor, shape (n, 4)
        Quaternions in ``(w, x, y, z)`` order; they are normalized internally, so
        the optimizer can move them freely.

    Returns
    -------
    Tensor, shape (n, 3, 3)
    """
    q = quaternions / quaternions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=1,
    ).reshape(-1, 3, 3)


def _mean_neighbour_distance(points: np.ndarray, neighbours: int) -> np.ndarray:
    """Mean distance from each point to its ``k`` nearest neighbours."""
    from scipy.spatial import cKDTree

    if len(points) <= neighbours:
        extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        return np.full(len(points), max(extent, 1.0) * 0.05)
    distances, _ = cKDTree(points).query(points, k=neighbours + 1)
    return distances[:, 1:].mean(axis=1)


class GaussianModel(nn.Module):
    """A differentiable set of 3D Gaussians.

    Attributes
    ----------
    means : Parameter, shape (n, 3)
    log_scales : Parameter, shape (n, 3)
    quaternions : Parameter, shape (n, 4)
    opacity_logits : Parameter, shape (n,)
    sh_dc : Parameter, shape (n, 1, 3)
        Degree-zero coefficient, the view-independent part of the colour.
    sh_rest : Parameter, shape (n, k - 1, 3)
        Higher bands, kept in a separate tensor because they need their own
        learning rate and are activated progressively during training.
    sh_degree : int
        Highest band the tensors can hold.
    active_sh_degree : int
        Highest band currently used; raised by the training schedule.
    """

    PARAMETER_NAMES = ("means", "log_scales", "quaternions", "opacity_logits", "sh_dc", "sh_rest")

    def __init__(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        quaternions: torch.Tensor,
        opacity_logits: torch.Tensor,
        sh_dc: torch.Tensor,
        sh_rest: torch.Tensor,
        sh_degree: int = 3,
    ) -> None:
        super().__init__()
        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(log_scales)
        self.quaternions = nn.Parameter(quaternions)
        self.opacity_logits = nn.Parameter(opacity_logits)
        self.sh_dc = nn.Parameter(sh_dc)
        self.sh_rest = nn.Parameter(sh_rest)
        self.sh_degree = sh_degree
        self.active_sh_degree = 0

    def __len__(self) -> int:
        return self.means.shape[0]

    @property
    def scales(self) -> torch.Tensor:
        """Standard deviations along the principal axes, shape ``(n, 3)``."""
        return torch.exp(self.log_scales)

    @property
    def opacities(self) -> torch.Tensor:
        """Opacity of each primitive in ``(0, 1)``, shape ``(n,)``."""
        return torch.sigmoid(self.opacity_logits)

    @property
    def rotations(self) -> torch.Tensor:
        """Rotation matrices of the principal axes, shape ``(n, 3, 3)``."""
        return quaternion_to_rotation(self.quaternions)

    def covariances(self) -> torch.Tensor:
        """World-space covariance matrices, shape ``(n, 3, 3)``.

        Formed as ``M M^T`` with ``M = R S``, which is positive semidefinite by
        construction and costs one batched product instead of two.
        """
        M = self.rotations * self.scales[:, None, :]
        return M @ M.transpose(1, 2)

    def colors(self, camera_center: torch.Tensor) -> torch.Tensor:
        """Evaluate view-dependent colour, shape ``(n, 3)``.

        Parameters
        ----------
        camera_center : Tensor, shape (3,)
            Camera position in world coordinates.

        Returns
        -------
        Tensor, shape (n, 3)
            Non-negative linear RGB.  Clamping at zero rather than passing the
            expansion through a sigmoid keeps the mapping linear where it is
            already valid, which matters because the loss is computed in the
            same linear space.
        """
        directions = self.means.detach() - camera_center
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
        coefficients = torch.cat([self.sh_dc, self.sh_rest], dim=1)
        return (sh.evaluate(coefficients, directions, self.active_sh_degree) + 0.5).clamp_min(0.0)

    def raise_sh_degree(self) -> None:
        """Activate one more spherical harmonic band, up to the stored maximum."""
        self.active_sh_degree = min(self.active_sh_degree + 1, self.sh_degree)

    def parameter_dict(self) -> dict[str, nn.Parameter]:
        """Return the optimizable tensors keyed by name."""
        return {name: getattr(self, name) for name in self.PARAMETER_NAMES}

    def extent(self) -> float:
        """Radius of the bounding sphere of the primitive centres."""
        centres = self.means.detach()
        return float((centres - centres.mean(dim=0)).norm(dim=1).max().item())

    @classmethod
    def from_point_cloud(
        cls,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        config: InitializationConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> GaussianModel:
        """Initialize one Gaussian per input point.

        Scales start at the mean distance to the nearest neighbours, so a dense
        region begins with small primitives and a sparse one with large
        primitives.  That is a far better starting point than a global constant:
        the density control described in :mod:`gsplat.densify` can split and
        clone, but it cannot cheaply recover from an initialization that hides
        the whole scene behind a few oversized blobs.

        Parameters
        ----------
        points : ndarray, shape (n, 3)
        colors : ndarray, shape (n, 3), optional
            Linear RGB in ``[0, 1]``; mid grey when omitted.
        config : InitializationConfig or None
        device : torch.device or str

        Returns
        -------
        GaussianModel
        """
        config = config or InitializationConfig()
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            raise ValueError("cannot initialize from an empty point cloud")
        if colors is None:
            colors = np.full((len(points), 3), 0.5, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32).reshape(-1, 3)

        extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        radius = _mean_neighbour_distance(points, config.neighbours)
        radius = np.clip(radius, 1e-6, max(config.max_initial_scale * extent, 1e-6))

        coefficient_count = sh.num_coefficients(config.sh_degree)
        opacity = float(np.log(config.initial_opacity / (1.0 - config.initial_opacity)))

        quaternions = np.zeros((len(points), 4), dtype=np.float32)
        quaternions[:, 0] = 1.0

        def to_tensor(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array, dtype=torch.float32, device=device)

        model = cls(
            means=to_tensor(points),
            log_scales=to_tensor(np.log(radius)[:, None].repeat(3, axis=1)),
            quaternions=to_tensor(quaternions),
            opacity_logits=to_tensor(np.full(len(points), opacity, dtype=np.float32)),
            sh_dc=sh.rgb_to_dc(to_tensor(colors))[:, None, :],
            sh_rest=torch.zeros(
                (len(points), coefficient_count - 1, 3), dtype=torch.float32, device=device
            ),
            sh_degree=config.sh_degree,
        )
        return model.to(device)

    @classmethod
    def random(
        cls,
        count: int,
        center: np.ndarray,
        radius: float,
        config: InitializationConfig | None = None,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> GaussianModel:
        """Initialize points uniformly inside a ball, for runs without a prior."""
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = np.asarray(center) + directions * radius * rng.random((count, 1)) ** (1 / 3)
        return cls.from_point_cloud(points, None, config, device)

    def save(self, path: str | Path) -> None:
        """Write the parameters and the band schedule to a compressed archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in self.parameter_dict().items()
        }
        np.savez_compressed(
            path, sh_degree=self.sh_degree, active_sh_degree=self.active_sh_degree, **arrays
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> GaussianModel:
        """Read a model written by :meth:`save`."""
        archive = np.load(path)
        model = cls(
            **{
                name: torch.as_tensor(archive[name], dtype=torch.float32, device=device)
                for name in cls.PARAMETER_NAMES
            },
            sh_degree=int(archive["sh_degree"]),
        )
        model.active_sh_degree = int(archive["active_sh_degree"])
        return model.to(device)
