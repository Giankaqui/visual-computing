"""Interactive panel for the trained Gaussian splatting model.

Every render here is a viewpoint the model was never trained on, and the same
camera is also ray traced analytically, so the two panes are a direct
side-by-side of the representation against the truth it was fitted to. Moving
the sliders between the training viewpoints is where a splatting model shows its
failure modes: silhouettes soften, and geometry that only ever appeared at a
grazing angle turns into streaks.
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from gsplat.cameras import Camera, look_at
from gsplat.gaussians import GaussianModel
from gsplat.losses import psnr, ssim
from gsplat.renderer import render
from gsplat.scenes import default_scene, render_scene

from common import as_display_image, metric_table, spherical_position

CHECKPOINT = Path(__file__).resolve().parent.parent / "gaussian-splatting" / "docs" / "model.npz"

_scene = default_scene()
_model: GaussianModel | None = None
_device = "mps" if torch.backends.mps.is_available() else "cpu"


def _load_model() -> GaussianModel | None:
    """Load the checkpoint once, keeping it resident between renders."""
    global _model
    if _model is None and CHECKPOINT.exists():
        _model = GaussianModel.load(CHECKPOINT, device=_device)
        _model.active_sh_degree = _model.sh_degree
    return _model


def _camera(azimuth: float, elevation: float, distance: float, width: int) -> Camera:
    eye = spherical_position(_scene.center, distance, azimuth, elevation)
    R, t = look_at(eye, _scene.center, np.array([0.0, -1.0, 0.0]))
    return Camera.from_fov(R, t, width, int(round(width * 0.75)), 50.0)


def _depth_to_image(depth: np.ndarray) -> np.ndarray:
    """Map a depth buffer to a viewable image, ignoring uncovered pixels."""
    covered = depth > 0
    if not covered.any():
        return np.zeros(depth.shape + (3,))
    low, high = np.percentile(depth[covered], [2, 98])
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    shaded = np.where(covered, 1.0 - normalized, 0.0)
    return np.stack([shaded * 0.35, shaded * 0.75, shaded], axis=2)


def _render(azimuth: float, elevation: float, distance: float, width: int, show_depth: bool):
    """Render one viewpoint from the model and from the analytic scene."""
    model = _load_model()
    if model is None:
        message = (
            f"**No checkpoint found at `{CHECKPOINT}`.**\n\n"
            "Train one first:\n\n"
            "```bash\ngsplat train --scene synthetic --iterations 3500 "
            "--width 200 --height 150 --output out/\n```"
        )
        return None, None, message

    camera = _camera(azimuth, elevation, distance, int(width))
    background = torch.as_tensor(_scene.background, dtype=torch.float32, device=_device)

    started = time.perf_counter()
    with torch.no_grad():
        result = render(model, camera, background=background)
    render_seconds = time.perf_counter() - started

    started = time.perf_counter()
    reference = render_scene(camera, _scene)
    trace_seconds = time.perf_counter() - started

    prediction = result.image.clamp(0.0, 1.0).cpu()
    target = torch.as_tensor(reference)
    left = (
        _depth_to_image(result.output.depth.cpu().numpy())
        if show_depth
        else prediction.numpy()
    )

    summary = metric_table(
        [
            ("PSNR against the ray tracer", f"{psnr(prediction, target):.2f} dB"),
            ("SSIM", f"{float(ssim(prediction, target)):.4f}"),
            ("Primitives in the model", f"{len(model):,}"),
            ("Visible after culling", f"{result.output.visible_count:,}"),
            ("Splatting", f"{1000 * render_seconds:.0f} ms on {_device}"),
            ("Ray tracing the same view", f"{1000 * trace_seconds:.0f} ms on cpu"),
            ("Tiles hitting the cap", f"{result.output.saturated_tiles}"),
        ]
    )
    return as_display_image(left), as_display_image(reference), summary


def build() -> None:
    """Add the panel to the enclosing Gradio layout."""
    available = CHECKPOINT.exists()
    gr.Markdown(
        "Move the camera anywhere on the orbit. The left pane is the fitted "
        "Gaussian model, the right pane is the same view ray traced analytically. "
        "Neither viewpoint was in the training set, so the difference between the "
        "panes is generalization, not reconstruction error."
        + ("" if available else "\n\n**No trained checkpoint is present.**")
    )
    defaults = (25.0, 22.0, 4.6, 280, False)
    # Rendering once here gives the panel something to show the moment the page
    # opens, which is cheaper and simpler than wiring a load event.
    initial_render, initial_reference, initial_summary = _render(*defaults)

    with gr.Row():
        with gr.Column(scale=1):
            azimuth = gr.Slider(-180, 180, value=defaults[0], step=1, label="Azimuth (degrees)")
            elevation = gr.Slider(-5, 60, value=defaults[1], step=1, label="Elevation (degrees)")
            distance = gr.Slider(2.5, 9.0, value=defaults[2], step=0.1, label="Distance")
            width = gr.Slider(160, 480, value=defaults[3], step=40, label="Render width (pixels)")
            show_depth = gr.Checkbox(value=defaults[4], label="Show depth instead of colour")
            summary = gr.Markdown(value=initial_summary)
        with gr.Column(scale=2):
            with gr.Row():
                rendered = gr.Image(
                    value=initial_render, label="Gaussian splatting", height=340
                )
                reference = gr.Image(
                    value=initial_reference, label="Ray traced reference", height=340
                )

    controls = [azimuth, elevation, distance, width, show_depth]
    outputs = [rendered, reference, summary]
    for control in controls:
        control.change(_render, inputs=controls, outputs=outputs, show_progress="minimal")
