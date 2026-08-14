"""Pinhole cameras for rendering and training.

The convention matches the structure-from-motion project in this repository:
``x_camera = R @ x_world + t`` with the optical axis along ``+z``, so a
``cameras.json`` written by that pipeline loads without any change of basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

__all__ = ["Camera", "look_at", "orbit_cameras"]


@dataclass
class Camera:
    """A calibrated view.

    Attributes
    ----------
    R : ndarray, shape (3, 3)
        World-to-camera rotation.
    t : ndarray, shape (3,)
        World-to-camera translation.
    fx, fy, cx, cy : float
        Intrinsics in pixels.
    width, height : int
        Image resolution.
    name : str
        Identifier used when writing renders to disk.
    """

    R: np.ndarray
    t: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    name: str = ""

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    @classmethod
    def from_fov(
        cls,
        R: np.ndarray,
        t: np.ndarray,
        width: int,
        height: int,
        fov_x_degrees: float,
        name: str = "",
    ) -> Camera:
        """Build a camera with square pixels from a horizontal field of view."""
        focal = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_x_degrees))
        return cls(
            R=R,
            t=t,
            fx=focal,
            fy=focal,
            cx=0.5 * width,
            cy=0.5 * height,
            width=width,
            height=height,
            name=name,
        )

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.R.T @ self.t

    def rescaled(self, factor: float) -> Camera:
        """Return the same view at a different resolution.

        Training at reduced resolution and evaluating at full resolution is the
        usual way to keep early iterations cheap, and every intrinsic parameter
        scales linearly with the image size.
        """
        return Camera(
            R=self.R,
            t=self.t,
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=self.cx * factor,
            cy=self.cy * factor,
            width=max(int(round(self.width * factor)), 1),
            height=max(int(round(self.height * factor)), 1),
            name=self.name,
        )

    def to_tensors(
        self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the rotation and translation as tensors on ``device``."""
        return (
            torch.as_tensor(self.R, dtype=dtype, device=device),
            torch.as_tensor(self.t, dtype=dtype, device=device),
        )


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a world-to-camera transform pointing from ``eye`` towards ``target``.

    Parameters
    ----------
    eye, target, up : ndarray, shape (3,)

    Returns
    -------
    R : ndarray, shape (3, 3)
    t : ndarray, shape (3,)
    """
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    norm = np.linalg.norm(right)
    if norm < 1e-8:
        raise ValueError("the up vector is parallel to the viewing direction")
    right /= norm
    true_up = np.cross(forward, right)
    R = np.stack([right, true_up, forward])
    return R, -R @ np.asarray(eye, dtype=float)


def orbit_cameras(
    count: int,
    radius: float,
    target: np.ndarray,
    width: int,
    height: int,
    fov_x_degrees: float = 50.0,
    elevation_degrees: float = 20.0,
    elevation_amplitude: float = 12.0,
    turns: float = 1.0,
    phase: float = 0.0,
) -> list[Camera]:
    """Place cameras on an orbit around a target.

    The elevation oscillates around its nominal value so that the viewpoints
    cover a band rather than a circle; a single circle leaves the vertical
    extent of every Gaussian unconstrained, and the model responds by growing
    thin sheets that look correct only from the training band.

    Parameters
    ----------
    count : int
        Number of cameras.
    radius : float
        Orbit radius.
    target : ndarray, shape (3,)
        Point the cameras look at.
    width, height : int
        Image resolution.
    fov_x_degrees : float
        Horizontal field of view.
    elevation_degrees : float
        Mean elevation above the target.
    elevation_amplitude : float
        Half-range of the elevation oscillation.
    turns : float
        Number of full revolutions covered by the sequence.
    phase : float
        Offset in revolutions, used to interleave a test orbit between the
        training views.

    Returns
    -------
    list of Camera
    """
    target = np.asarray(target, dtype=float)
    cameras: list[Camera] = []
    for index in range(count):
        fraction = (index / max(count, 1) + phase) * turns
        azimuth = 2.0 * np.pi * fraction
        elevation = np.deg2rad(
            elevation_degrees + elevation_amplitude * np.sin(3.0 * azimuth)
        )
        offset = radius * np.array(
            [
                np.cos(elevation) * np.sin(azimuth),
                -np.sin(elevation),
                np.cos(elevation) * np.cos(azimuth),
            ]
        )
        R, t = look_at(target + offset, target, np.array([0.0, -1.0, 0.0]))
        cameras.append(
            Camera.from_fov(R, t, width, height, fov_x_degrees, name=f"view_{index:03d}")
        )
    return cameras
