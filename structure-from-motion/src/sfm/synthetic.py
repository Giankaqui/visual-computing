"""Synthetic multi-view data with known ground truth.

The generator produces the same objects the real pipeline works with, so the
geometric solvers, bundle adjustment and the incremental loop can all be
exercised without any image data and with an exact reference to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import PinholeCamera, Pose, project_points
from .rotations import exp_so3

__all__ = ["SyntheticScene", "make_scene", "to_matching_problem"]


@dataclass
class SyntheticScene:
    """A ground-truth reconstruction and its noisy observations.

    Attributes
    ----------
    camera : PinholeCamera
        Shared intrinsics for all views.
    poses : list of Pose
        Ground-truth world-to-camera transforms.
    points : ndarray, shape (n_points, 3)
        Ground-truth structure.
    camera_indices, point_indices : ndarray, shape (n_observations,)
    observations : ndarray, shape (n_observations, 2)
        Pixel measurements including noise and, for a fraction of entries,
        gross errors.
    is_outlier : ndarray of bool, shape (n_observations,)
        Marks the observations that were corrupted.
    """

    camera: PinholeCamera
    poses: list[Pose]
    points: np.ndarray
    camera_indices: np.ndarray
    point_indices: np.ndarray
    observations: np.ndarray
    is_outlier: np.ndarray

    def observations_for_view(self, view: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the point indices and pixels observed by one view."""
        mask = self.camera_indices == view
        return self.point_indices[mask], self.observations[mask]


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> Pose:
    """Build a world-to-camera pose whose optical axis points from eye to target."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    R = np.stack([right, true_up, forward])
    return Pose(R=R, t=-R @ eye)


def make_scene(
    num_points: int = 300,
    num_views: int = 8,
    image_size: tuple[int, int] = (960, 720),
    fov_degrees: float = 55.0,
    arc_degrees: float = 90.0,
    radius: float = 6.0,
    noise_pixels: float = 0.5,
    outlier_fraction: float = 0.05,
    outlier_pixels: float = 40.0,
    seed: int = 0,
) -> SyntheticScene:
    """Generate a scene of points on three orthogonal walls viewed along an arc.

    Points on mutually orthogonal planes avoid the degenerate configurations that
    make the essential matrix ambiguous, and a camera arc guarantees the parallax
    that triangulation needs.

    Parameters
    ----------
    num_points : int
        Number of 3D points.
    num_views : int
        Number of cameras placed on the arc.
    image_size : tuple of int
        Image width and height in pixels.
    fov_degrees : float
        Horizontal field of view.
    arc_degrees : float
        Angular extent of the camera arc around the scene centre.
    radius : float
        Distance from the scene centre to each camera.
    noise_pixels : float
        Standard deviation of isotropic Gaussian measurement noise.
    outlier_fraction : float
        Fraction of observations displaced by a gross error.
    outlier_pixels : float
        Standard deviation of the gross errors.
    seed : int
        Seed for the random generator.

    Returns
    -------
    SyntheticScene
    """
    rng = np.random.default_rng(seed)
    width, height = image_size
    camera = PinholeCamera.from_fov(width, height, fov_degrees)

    per_wall = num_points // 3
    remainder = num_points - 2 * per_wall

    def uniform(count: int, low: float, high: float) -> np.ndarray:
        return rng.uniform(low, high, size=count)

    back = np.stack(
        [uniform(per_wall, -2, 2), uniform(per_wall, -1.5, 1.5), np.full(per_wall, 2.0)], axis=1
    )
    left = np.stack(
        [np.full(per_wall, -2.0), uniform(per_wall, -1.5, 1.5), uniform(per_wall, -1, 2)], axis=1
    )
    floor = np.stack(
        [uniform(remainder, -2, 2), np.full(remainder, 1.5), uniform(remainder, -1, 2)], axis=1
    )
    points = np.vstack([back, left, floor])
    points += rng.normal(scale=0.03, size=points.shape)

    centre = points.mean(axis=0)
    angles = np.deg2rad(np.linspace(-arc_degrees / 2, arc_degrees / 2, num_views))
    poses: list[Pose] = []
    for angle in angles:
        eye = centre + radius * np.array([np.sin(angle), 0.0, -np.cos(angle)])
        eye[1] += 0.4 * np.sin(2.0 * angle)
        pose = _look_at(eye, centre, np.array([0.0, -1.0, 0.0]))
        jitter = exp_so3(rng.normal(scale=0.01, size=3))
        poses.append(Pose(R=jitter @ pose.R, t=jitter @ pose.t))

    camera_indices: list[np.ndarray] = []
    point_indices: list[np.ndarray] = []
    pixels: list[np.ndarray] = []
    for view, pose in enumerate(poses):
        projected, depths = project_points(points, pose, camera)
        visible = (depths > 0.1) & camera.contains(projected)
        indices = np.flatnonzero(visible)
        camera_indices.append(np.full(len(indices), view, dtype=np.int64))
        point_indices.append(indices.astype(np.int64))
        pixels.append(projected[visible])

    camera_index = np.concatenate(camera_indices)
    point_index = np.concatenate(point_indices)
    measurement = np.concatenate(pixels)
    measurement += rng.normal(scale=noise_pixels, size=measurement.shape)

    is_outlier = rng.random(len(measurement)) < outlier_fraction
    measurement[is_outlier] += rng.normal(scale=outlier_pixels, size=(int(is_outlier.sum()), 2))

    return SyntheticScene(
        camera=camera,
        poses=poses,
        points=points,
        camera_indices=camera_index,
        point_indices=point_index,
        observations=measurement,
        is_outlier=is_outlier,
    )


def to_matching_problem(
    scene: SyntheticScene, mismatch_fraction: float = 0.08, seed: int = 0
) -> tuple[list[np.ndarray], dict[tuple[int, int], np.ndarray], list[PinholeCamera]]:
    """Convert a scene into the inputs of :func:`sfm.reconstruction.reconstruct`.

    Observations become per-view features and shared points become putative
    matches, so the full incremental pipeline can be exercised without images.
    A fraction of the matches is rewired to the wrong feature, which reproduces
    the failure mode that geometric verification exists to catch.

    Parameters
    ----------
    scene : SyntheticScene
    mismatch_fraction : float
        Fraction of putative matches whose second index is randomized.
    seed : int

    Returns
    -------
    keypoints : list of ndarray
        Pixel coordinates per view.
    matches : dict
        Putative matches keyed by ``(i, j)`` with ``i < j``.
    cameras : list of PinholeCamera
    """
    rng = np.random.default_rng(seed)
    num_views = len(scene.poses)

    keypoints: list[np.ndarray] = []
    point_to_feature: list[dict[int, int]] = []
    for view in range(num_views):
        point_ids, pixels = scene.observations_for_view(view)
        keypoints.append(pixels)
        point_to_feature.append({int(p): i for i, p in enumerate(point_ids)})

    matches: dict[tuple[int, int], np.ndarray] = {}
    for view_a in range(num_views):
        for view_b in range(view_a + 1, num_views):
            shared = sorted(set(point_to_feature[view_a]) & set(point_to_feature[view_b]))
            if len(shared) < 8:
                continue
            pairs = np.array(
                [[point_to_feature[view_a][p], point_to_feature[view_b][p]] for p in shared],
                dtype=np.int64,
            )
            corrupted = rng.random(len(pairs)) < mismatch_fraction
            pairs[corrupted, 1] = rng.integers(0, len(keypoints[view_b]), size=int(corrupted.sum()))
            matches[(view_a, view_b)] = pairs

    return keypoints, matches, [scene.camera] * num_views
