"""Panel interactivo del modelo de Gaussian splatting entrenado.

Cada render es un punto de vista con el que el modelo nunca se entrenó, y la
misma cámara se traza además por rayos, así que los dos paneles son una
comparación directa entre la representación y la verdad a la que se ajustó.
Mover los sliders entre los puntos de vista de entrenamiento es donde un modelo
de splatting enseña sus modos de fallo: las siluetas se ablandan y la geometría
que solo apareció en ángulo rasante se convierte en estrías.
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
    """Carga el checkpoint una sola vez y lo mantiene residente entre renders."""
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
    """Convierte un buffer de profundidad en algo visible, ignorando los píxeles vacíos."""
    covered = depth > 0
    if not covered.any():
        return np.zeros(depth.shape + (3,))
    low, high = np.percentile(depth[covered], [2, 98])
    normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
    shaded = np.where(covered, 1.0 - normalized, 0.0)
    return np.stack([shaded * 0.35, shaded * 0.75, shaded], axis=2)


def _render(azimuth: float, elevation: float, distance: float, width: int, show_depth: bool):
    """Renderiza un punto de vista desde el modelo y desde la escena analítica."""
    model = _load_model()
    if model is None:
        message = (
            f"**No se ha encontrado ningún checkpoint en `{CHECKPOINT}`.**\n\n"
            "Entrena uno primero:\n\n"
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
            ("PSNR contra el trazador de rayos", f"{psnr(prediction, target):.2f} dB"),
            ("SSIM", f"{float(ssim(prediction, target)):.4f}"),
            ("Primitivas del modelo", f"{len(model):,}"),
            ("Visibles tras el culling", f"{result.output.visible_count:,}"),
            ("Splatting", f"{1000 * render_seconds:.0f} ms en {_device}"),
            ("Trazar la misma vista", f"{1000 * trace_seconds:.0f} ms en cpu"),
            ("Tiles que tocan el tope", f"{result.output.saturated_tiles}"),
        ]
    )
    return as_display_image(left), as_display_image(reference), summary


def build() -> None:
    """Añade el panel al layout de Gradio que lo envuelve."""
    available = CHECKPOINT.exists()
    gr.Markdown(
        "Mueve la cámara a cualquier punto de la órbita. El panel de la izquierda es "
        "el modelo gaussiano ajustado, el de la derecha es la misma vista trazada por "
        "rayos de forma analítica. Ninguno de estos puntos de vista estuvo en el "
        "conjunto de entrenamiento, así que la diferencia entre ambos es "
        "generalización, no error de ajuste."
        + ("" if available else "\n\n**No hay ningún checkpoint entrenado.**")
    )
    defaults = (25.0, 22.0, 4.6, 280, False)
    # Renderizar una vez aquí da al panel algo que enseñar en cuanto se abre la
    # página, lo que es más simple y barato que cablear un evento de carga.
    initial_render, initial_reference, initial_summary = _render(*defaults)

    with gr.Row():
        with gr.Column(scale=1):
            azimuth = gr.Slider(-180, 180, value=defaults[0], step=1, label="Azimut (grados)")
            elevation = gr.Slider(-5, 60, value=defaults[1], step=1, label="Elevación (grados)")
            distance = gr.Slider(2.5, 9.0, value=defaults[2], step=0.1, label="Distancia")
            width = gr.Slider(160, 480, value=defaults[3], step=40, label="Ancho del render (px)")
            show_depth = gr.Checkbox(value=defaults[4], label="Ver profundidad en vez de color")
            summary = gr.Markdown(value=initial_summary)
        with gr.Column(scale=2):
            with gr.Row():
                rendered = gr.Image(
                    value=initial_render, label="Gaussian splatting", height=340
                )
                reference = gr.Image(
                    value=initial_reference, label="Referencia trazada por rayos", height=340
                )

    controls = [azimuth, elevation, distance, width, show_depth]
    outputs = [rendered, reference, summary]
    for control in controls:
        control.change(_render, inputs=controls, outputs=outputs, show_progress="minimal")
