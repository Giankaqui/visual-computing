"""Incremental structure from motion from calibrated images.

The package implements the classical pipeline end to end: feature extraction and
matching, robust two-view geometry with a five-point minimal solver, linear and
nonlinear triangulation, absolute pose estimation, track construction, and
sparse bundle adjustment with the Schur complement.
"""

from .bundle import BundleOptions, BundleProblem, BundleReport, adjust
from .camera import PinholeCamera, Pose, project_points
from .epipolar import decompose_essential, estimate_essential, recover_pose, sampson_distance
from .five_point import five_point_essential
from .pnp import estimate_pose_ransac, pnp_dlt, refine_pose
from .ransac import RansacOptions, RansacResult, ransac
from .reconstruction import Reconstruction, ReconstructionOptions, reconstruct
from .synthetic import SyntheticScene, make_scene
from .triangulation import refine_point, triangulate_dlt, triangulate_multiview

__all__ = [
    "BundleOptions",
    "BundleProblem",
    "BundleReport",
    "PinholeCamera",
    "Pose",
    "RansacOptions",
    "RansacResult",
    "Reconstruction",
    "ReconstructionOptions",
    "SyntheticScene",
    "adjust",
    "decompose_essential",
    "estimate_essential",
    "estimate_pose_ransac",
    "five_point_essential",
    "make_scene",
    "pnp_dlt",
    "project_points",
    "ransac",
    "reconstruct",
    "recover_pose",
    "refine_point",
    "refine_pose",
    "sampson_distance",
    "triangulate_dlt",
    "triangulate_multiview",
]

__version__ = "0.1.0"
