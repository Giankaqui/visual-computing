"""Differentiable 3D Gaussian splatting in pure PyTorch.

The package implements the full pipeline: an EWA projection of anisotropic 3D
Gaussians to screen-space conics, a tile-based rasterizer that composites them
front to back and differentiates without a custom backward pass, view-dependent
colour through spherical harmonics, and the adaptive density control that
decides how many primitives the scene needs.
"""

from .cameras import Camera, look_at, orbit_cameras
from .datasets import Dataset, View, load_dataset, save_views, synthetic_dataset
from .densify import DensityConfig, DensityController
from .gaussians import GaussianModel, InitializationConfig
from .losses import l1_loss, photometric_loss, psnr, ssim
from .projection import ProjectedGaussians, project
from .rasterizer import RenderOutput, rasterize
from .renderer import RenderResult, render
from .scenes import SceneDefinition, Sphere, default_scene, render_scene
from .trainer import EvaluationResult, Trainer, TrainingConfig, TrainingHistory

__all__ = [
    "Camera",
    "Dataset",
    "DensityConfig",
    "DensityController",
    "EvaluationResult",
    "GaussianModel",
    "InitializationConfig",
    "ProjectedGaussians",
    "RenderOutput",
    "RenderResult",
    "SceneDefinition",
    "Sphere",
    "Trainer",
    "TrainingConfig",
    "TrainingHistory",
    "View",
    "default_scene",
    "l1_loss",
    "load_dataset",
    "look_at",
    "orbit_cameras",
    "photometric_loss",
    "project",
    "psnr",
    "rasterize",
    "render",
    "render_scene",
    "save_views",
    "ssim",
    "synthetic_dataset",
]

__version__ = "0.1.0"
