"""Figures for inspecting a trained model.

Rendering uses the Agg backend so the module works headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .cameras import Camera  # noqa: E402
from .datasets import View  # noqa: E402
from .gaussians import GaussianModel  # noqa: E402
from .losses import psnr  # noqa: E402
from .renderer import render  # noqa: E402
from .trainer import TrainingHistory  # noqa: E402

__all__ = ["comparison_figure", "training_curves", "turntable_strip"]

_ACCENT = "#0f4c81"
_HIGHLIGHT = "#d1495b"


def _to_numpy(image: torch.Tensor) -> np.ndarray:
    return image.detach().clamp(0.0, 1.0).cpu().numpy()


@torch.no_grad()
def comparison_figure(
    model: GaussianModel,
    views: list[View],
    path: str | Path,
    background: torch.Tensor | None = None,
    title: str | None = None,
) -> Path:
    """Lay out reference, render and absolute error for a set of views.

    The error row is scaled to the maximum error in the figure rather than to
    each panel, so panels remain comparable with one another.

    Parameters
    ----------
    model : GaussianModel
    views : list of View
    path : str or Path
    background : Tensor, shape (3,), optional
    title : str or None

    Returns
    -------
    Path
    """
    if not views:
        raise ValueError("at least one view is required")

    references, renders, errors, peaks = [], [], [], []
    for view in views:
        prediction = render(model, view.camera, background=background).image
        references.append(_to_numpy(view.image))
        renders.append(_to_numpy(prediction))
        errors.append(np.abs(references[-1] - renders[-1]).mean(axis=2))
        peaks.append(psnr(prediction.clamp(0.0, 1.0), view.image))

    scale = max(float(np.max(error)) for error in errors)
    figure, axes = plt.subplots(
        3, len(views), figsize=(2.9 * len(views), 8.0), squeeze=False
    )
    for column, view in enumerate(views):
        axes[0][column].imshow(references[column])
        axes[0][column].set_title(view.camera.name or f"view {column}", fontsize=10)
        axes[1][column].imshow(renders[column])
        axes[1][column].set_title(f"{peaks[column]:.2f} dB", fontsize=10)
        image = axes[2][column].imshow(errors[column], cmap="magma", vmin=0.0, vmax=scale)
        for row in range(3):
            axes[row][column].set_axis_off()

    for row, label in enumerate(("reference", "render", "absolute error")):
        axes[row][0].set_ylabel(label)
        axes[row][0].text(
            -0.06, 0.5, label, rotation=90, va="center", ha="right",
            transform=axes[row][0].transAxes, fontsize=11,
        )
    figure.colorbar(image, ax=axes[2], fraction=0.025, pad=0.01)
    figure.suptitle(title or f"{len(model)} primitives, mean {np.mean(peaks):.2f} dB")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path


def training_curves(history: TrainingHistory, path: str | Path) -> Path:
    """Plot the loss, the primitive count and the held-out quality.

    Parameters
    ----------
    history : TrainingHistory
    path : str or Path

    Returns
    -------
    Path
    """
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))

    window = max(len(history.loss) // 100, 1)
    smoothed = np.convolve(history.loss, np.ones(window) / window, mode="valid")
    axes[0].plot(history.iterations[: len(smoothed)], smoothed, color=_ACCENT)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("training loss")
    axes[0].set_title("photometric loss")

    axes[1].plot(history.iterations, history.primitives, color=_ACCENT)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("primitives")
    axes[1].set_title("adaptive density control")

    if history.test_iterations:
        axes[2].plot(history.test_iterations, history.test_psnr, marker="o",
                     color=_HIGHLIGHT, label="psnr")
        axes[2].set_ylabel("test psnr (dB)")
        twin = axes[2].twinx()
        twin.plot(history.test_iterations, history.test_ssim, marker="s",
                  color=_ACCENT, label="ssim")
        twin.set_ylabel("test ssim")
        handles = axes[2].get_lines() + twin.get_lines()
        axes[2].legend(handles, [line.get_label() for line in handles], loc="lower right")
    axes[2].set_xlabel("iteration")
    axes[2].set_title("held-out views")

    for axis in axes:
        axis.spines[["top"]].set_visible(False)
    figure.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


@torch.no_grad()
def turntable_strip(
    model: GaussianModel,
    cameras: list[Camera],
    path: str | Path,
    background: torch.Tensor | None = None,
    show_depth: bool = True,
) -> Path:
    """Render a strip of novel views, optionally with their depth maps.

    Parameters
    ----------
    model : GaussianModel
    cameras : list of Camera
    path : str or Path
    background : Tensor, shape (3,), optional
    show_depth : bool
        Add a second row with the alpha-weighted depth.

    Returns
    -------
    Path
    """
    if not cameras:
        raise ValueError("at least one camera is required")

    rows = 2 if show_depth else 1
    figure, axes = plt.subplots(
        rows, len(cameras), figsize=(2.7 * len(cameras), 2.4 * rows), squeeze=False
    )
    depths = []
    for column, camera in enumerate(cameras):
        result = render(model, camera, background=background)
        axes[0][column].imshow(_to_numpy(result.image))
        axes[0][column].set_axis_off()
        depths.append(result.output.depth.detach().cpu().numpy())

    if show_depth:
        covered = np.concatenate([depth[depth > 0].reshape(-1) for depth in depths])
        low, high = np.percentile(covered, [2, 98]) if covered.size else (0.0, 1.0)
        for column, depth in enumerate(depths):
            axes[1][column].imshow(np.where(depth > 0, depth, np.nan),
                                   cmap="viridis_r", vmin=low, vmax=high)
            axes[1][column].set_axis_off()

    figure.tight_layout(pad=0.4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
