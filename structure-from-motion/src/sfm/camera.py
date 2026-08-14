"""Pinhole camera model and rigid poses.

The world-to-camera convention used throughout the package is ``x_cam = R @ x_world + t``
with the camera looking down its positive z axis, which matches OpenCV and COLMAP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["PinholeCamera", "Pose", "project_points"]


@dataclass(frozen=True)
class PinholeCamera:
    """Intrinsic calibration of a pinhole camera.

    Attributes
    ----------
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point in pixels.
    width, height : int
        Image resolution, used for visibility tests.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, fov_x_degrees: float) -> PinholeCamera:
        """Build a camera with a square pixel aspect from a horizontal field of view."""
        f = 0.5 * width / np.tan(0.5 * np.deg2rad(fov_x_degrees))
        return cls(fx=f, fy=f, cx=0.5 * width, cy=0.5 * height, width=width, height=height)

    @classmethod
    def guess_from_image(cls, width: int, height: int) -> PinholeCamera:
        """Fallback calibration when no EXIF data is available.

        Assumes a 55 degree horizontal field of view, roughly a 35 mm-equivalent
        focal length of 35 mm.  Bundle adjustment cannot recover from a badly
        wrong initial focal length, so this is a starting point for casual
        captures rather than a substitute for calibration.
        """
        return cls.from_fov(width, height, 55.0)

    @property
    def matrix(self) -> np.ndarray:
        """Return the 3x3 intrinsic matrix ``K``."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float
        )

    @property
    def inverse_matrix(self) -> np.ndarray:
        """Return ``K^-1`` in closed form."""
        return np.array(
            [
                [1.0 / self.fx, 0.0, -self.cx / self.fx],
                [0.0, 1.0 / self.fy, -self.cy / self.fy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def normalize(self, pixels: np.ndarray) -> np.ndarray:
        """Map pixel coordinates to normalized image coordinates.

        Parameters
        ----------
        pixels : ndarray, shape (n, 2)

        Returns
        -------
        ndarray, shape (n, 2)
            Coordinates on the ``z = 1`` plane of the camera frame.
        """
        pixels = np.asarray(pixels, dtype=float)
        return np.stack(
            [(pixels[:, 0] - self.cx) / self.fx, (pixels[:, 1] - self.cy) / self.fy], axis=1
        )

    def denormalize(self, normalized: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`normalize`."""
        normalized = np.asarray(normalized, dtype=float)
        return np.stack(
            [normalized[:, 0] * self.fx + self.cx, normalized[:, 1] * self.fy + self.cy], axis=1
        )

    def contains(self, pixels: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of pixels inside the image rectangle."""
        pixels = np.asarray(pixels, dtype=float)
        return (
            (pixels[:, 0] >= -margin)
            & (pixels[:, 0] < self.width + margin)
            & (pixels[:, 1] >= -margin)
            & (pixels[:, 1] < self.height + margin)
        )


@dataclass
class Pose:
    """A rigid world-to-camera transform.

    Attributes
    ----------
    R : ndarray, shape (3, 3)
        Rotation from world to camera coordinates.
    t : ndarray, shape (3,)
        Translation from world to camera coordinates.
    """

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates, ``-R.T @ t``."""
        return -self.R.T @ self.t

    @property
    def matrix(self) -> np.ndarray:
        """Return the 3x4 extrinsic matrix ``[R | t]``."""
        return np.hstack([self.R, self.t[:, None]])

    @property
    def viewing_direction(self) -> np.ndarray:
        """Unit vector along the optical axis, in world coordinates."""
        return self.R.T @ np.array([0.0, 0.0, 1.0])

    def transform(self, points_world: np.ndarray) -> np.ndarray:
        """Map world points to the camera frame."""
        points_world = np.asarray(points_world, dtype=float)
        return points_world @ self.R.T + self.t

    def inverse(self) -> Pose:
        """Return the camera-to-world transform."""
        return Pose(R=self.R.T, t=-self.R.T @ self.t)

    def compose(self, other: Pose) -> Pose:
        """Return the pose mapping ``other``'s source frame to this one's target."""
        return Pose(R=self.R @ other.R, t=self.R @ other.t + self.t)

    def copy(self) -> Pose:
        return Pose(R=self.R.copy(), t=self.t.copy())


def project_points(
    points_world: np.ndarray, pose: Pose, camera: PinholeCamera
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into an image.

    Parameters
    ----------
    points_world : ndarray, shape (n, 3)
    pose : Pose
    camera : PinholeCamera

    Returns
    -------
    pixels : ndarray, shape (n, 2)
        Projections; entries behind the camera are meaningless and flagged below.
    depths : ndarray, shape (n,)
        Depth along the optical axis, negative for points behind the camera.
    """
    cam_points = pose.transform(points_world)
    depths = cam_points[:, 2]
    safe_z = np.where(np.abs(depths) < 1e-12, 1e-12, depths)
    normalized = cam_points[:, :2] / safe_z[:, None]
    return camera.denormalize(normalized), depths
