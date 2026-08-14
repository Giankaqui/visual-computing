"""Evaluation of a reconstruction against ground truth.

A reconstruction from images alone is determined only up to a similarity
transform: the world frame, the orientation and the overall scale are free.
Comparing against ground truth therefore requires estimating that transform
first, which is the closed-form problem solved by Umeyama (1991).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import Pose
from .rotations import log_so3, project_to_so3

__all__ = ["Similarity", "align_similarity", "PoseErrors", "compare_poses"]


@dataclass
class Similarity:
    """A scaled rigid transform ``y = scale * R @ x + t``."""

    scale: float
    R: np.ndarray
    t: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform an array of points of shape ``(n, 3)``."""
        return self.scale * (np.asarray(points, dtype=float) @ self.R.T) + self.t

    def apply_pose(self, pose: Pose) -> Pose:
        """Transform a world-to-camera pose into the target frame.

        Substituting ``x = R^T (y - t) / s`` into ``R_c x + t_c`` and multiplying
        by ``s``, which leaves the projected ray unchanged, gives the transformed
        pose ``R_c R^T`` and ``s t_c - R_c R^T t``.  Equivalently, the camera
        centre transforms like any other point.
        """
        rotation = pose.R @ self.R.T
        return Pose(R=rotation, t=self.scale * pose.t - rotation @ self.t)

    def inverse(self) -> Similarity:
        inverse_rotation = self.R.T
        return Similarity(
            scale=1.0 / self.scale,
            R=inverse_rotation,
            t=-inverse_rotation @ self.t / self.scale,
        )


def align_similarity(source: np.ndarray, target: np.ndarray) -> Similarity:
    """Estimate the similarity that maps ``source`` onto ``target``.

    Implements the closed-form least-squares solution: the rotation comes from
    the singular value decomposition of the cross-covariance, with a reflection
    guard, and the scale is the ratio between the explained variance and the
    variance of the source.

    Parameters
    ----------
    source, target : ndarray, shape (n, 3)
        Corresponding points, in the same order.

    Returns
    -------
    Similarity

    Raises
    ------
    ValueError
        If fewer than three correspondences are supplied.
    """
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    if len(source) < 3 or len(source) != len(target):
        raise ValueError("at least three matching points are required")

    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centred, target_centred = source - source_mean, target - target_mean

    covariance = target_centred.T @ source_centred / len(source)
    U, singular_values, Vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        correction[2, 2] = -1.0
        singular_values = singular_values.copy()
        singular_values[2] *= -1.0

    R = project_to_so3(U @ correction @ Vt)
    variance = float((source_centred**2).sum() / len(source))
    scale = float(singular_values.sum() / variance) if variance > 1e-18 else 1.0
    return Similarity(scale=scale, R=R, t=target_mean - scale * R @ source_mean)


@dataclass
class PoseErrors:
    """Per-view discrepancy between an aligned reconstruction and ground truth.

    Attributes
    ----------
    rotation_degrees : ndarray, shape (n,)
        Geodesic angle between the estimated and true rotations.
    center_distance : ndarray, shape (n,)
        Distance between camera centres, in ground-truth units.
    """

    rotation_degrees: np.ndarray
    center_distance: np.ndarray

    def summary(self) -> str:
        return (
            f"rotation error median {np.median(self.rotation_degrees):.4f} deg "
            f"max {self.rotation_degrees.max():.4f} deg, "
            f"centre error median {np.median(self.center_distance):.5f} "
            f"max {self.center_distance.max():.5f}"
        )


def compare_poses(
    estimated: dict[int, Pose], ground_truth: list[Pose], transform: Similarity
) -> PoseErrors:
    """Measure pose errors after mapping the estimate into the ground-truth frame.

    Parameters
    ----------
    estimated : dict
        View index mapped to the estimated world-to-camera pose.
    ground_truth : list of Pose
        True poses indexed by view.
    transform : Similarity
        Transform taking estimated world coordinates to ground-truth ones,
        typically obtained from :func:`align_similarity` on camera centres.

    Returns
    -------
    PoseErrors
    """
    rotation_errors: list[float] = []
    center_errors: list[float] = []
    for view, pose in sorted(estimated.items()):
        aligned = transform.apply_pose(pose)
        reference = ground_truth[view]
        rotation_errors.append(
            float(np.rad2deg(np.linalg.norm(log_so3(aligned.R @ reference.R.T))))
        )
        center_errors.append(float(np.linalg.norm(aligned.center - reference.center)))
    return PoseErrors(np.array(rotation_errors), np.array(center_errors))
