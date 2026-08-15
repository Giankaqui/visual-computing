"""Panel interactivo del pipeline de structure from motion.

Los controles exponen las dos cosas que deciden si una reconstrucción sale bien:
cuánto ruido corrompe cada medida y qué fracción de las correspondencias es
directamente errónea. Correr la misma escena con varios ajustes es la forma más
rápida de ver dónde dejan de aguantar los estimadores robustos.
"""

from __future__ import annotations

import time

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from sfm.camera import PinholeCamera, Pose
from sfm.metrics import align_similarity, compare_poses
from sfm.reconstruction import Reconstruction, ReconstructionOptions
from sfm.reconstruction import reconstruct as run_reconstruction
from sfm.synthetic import make_scene, to_matching_problem

from common import metric_table

_POINT_COLOR = "#0f4c81"
_CAMERA_COLOR = "#d1495b"


def _to_display_frame(points: np.ndarray) -> np.ndarray:
    """Lleva coordenadas de mundo a un sistema de dibujo con el eje vertical arriba."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)


def _frustum_polyline(pose: Pose, camera: PinholeCamera, scale: float) -> np.ndarray:
    """Una sola polilínea que recorre las esquinas del frustum y el centro de la cámara."""
    corners_pixel = np.array(
        [[0.0, 0.0], [camera.width, 0.0], [camera.width, camera.height], [0.0, camera.height]]
    )
    rays = np.hstack([camera.normalize(corners_pixel), np.ones((4, 1))]) * scale
    vertices = np.vstack([rays @ pose.R + pose.center, pose.center])
    return _to_display_frame(vertices[[4, 0, 1, 4, 2, 3, 4, 0, 3, 2, 1]])


def _scene_figure(model: Reconstruction) -> go.Figure:
    """Construye una vista 3D de la reconstrucción que se puede orbitar."""
    points, _ = model.point_cloud()
    display = _to_display_frame(points)
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=display[:, 0], y=display[:, 1], z=display[:, 2],
            mode="markers",
            marker={"size": 1.6, "color": display[:, 2], "colorscale": "Viridis", "opacity": 0.9},
            name=f"{len(points)} puntos",
            hoverinfo="skip",
        )
    )

    trajectory = _to_display_frame(model.camera_centers())
    for index, view in enumerate(model.registration_order):
        polyline = _frustum_polyline(model.poses[view], model.cameras[view], 0.05 * extent)
        figure.add_trace(
            go.Scatter3d(
                x=polyline[:, 0], y=polyline[:, 1], z=polyline[:, 2],
                mode="lines",
                line={"color": _CAMERA_COLOR, "width": 3},
                name=f"vista {view}",
                showlegend=index == 0,
                hoverinfo="name",
            )
        )
    figure.add_trace(
        go.Scatter3d(
            x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
            mode="lines",
            line={"color": _CAMERA_COLOR, "width": 2, "dash": "dot"},
            name="trayectoria",
            hoverinfo="skip",
        )
    )

    span = float(np.max(display.max(axis=0) - display.min(axis=0))) * 0.6
    centre = 0.5 * (display.max(axis=0) + display.min(axis=0))
    axis_range = [
        [centre[axis] - span, centre[axis] + span] for axis in range(3)
    ]
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        height=520,
        scene={
            "xaxis": {"title": "x", "range": axis_range[0]},
            "yaxis": {"title": "z", "range": axis_range[1]},
            "zaxis": {"title": "-y", "range": axis_range[2]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 0.9}},
        },
        legend={"orientation": "h", "y": 0.02},
    )
    return figure


def _error_figure(errors: np.ndarray) -> go.Figure:
    """Histograma del error de reproyección con la mediana marcada."""
    figure = go.Figure(
        go.Histogram(x=errors, nbinsx=60, marker={"color": _POINT_COLOR}, name="observaciones")
    )
    figure.add_vline(
        x=float(np.median(errors)),
        line={"color": _CAMERA_COLOR, "dash": "dash"},
        annotation_text=f"mediana {np.median(errors):.2f} px",
    )
    figure.update_layout(
        margin={"l": 40, "r": 10, "t": 30, "b": 40},
        height=260,
        xaxis_title="error de reproyección (px)",
        yaxis_title="observaciones",
        showlegend=False,
    )
    return figure


def _run(views: int, points: int, noise: float, outliers: float, seed: int):
    """Genera una escena, la reconstruye y mide el resultado contra la verdad conocida."""
    scene = make_scene(
        num_points=int(points),
        num_views=int(views),
        noise_pixels=float(noise),
        outlier_fraction=float(outliers),
        seed=int(seed),
    )
    keypoints, matches, cameras = to_matching_problem(scene, seed=int(seed))

    started = time.perf_counter()
    model = run_reconstruction(
        keypoints,
        matches,
        cameras,
        options=ReconstructionOptions(verbose=False, seed=int(seed)),
    )
    elapsed = time.perf_counter() - started

    if model.num_registered < 2 or not model.points:
        message = (
            "**La reconstrucción ha fallado.** Ningún par de vistas ha superado la "
            "verificación geométrica con estos ajustes, así que no había con qué "
            "sembrar el modelo. Baja el ruido o la fracción de outliers."
        )
        return message, None, None

    registered = sorted(model.poses)
    transform = align_similarity(
        np.array([model.poses[view].center for view in registered]),
        np.array([scene.poses[view].center for view in registered]),
    )
    pose_errors = compare_poses(model.poses, scene.poses, transform)

    feature_to_point = {
        (view, feature): int(point_id)
        for view in range(len(scene.poses))
        for feature, point_id in enumerate(scene.observations_for_view(view)[0])
    }
    track_ids = sorted(model.points)
    aligned = transform.apply(np.array([model.points[t] for t in track_ids]))
    reference = np.array(
        [scene.points[feature_to_point[model.tracks.tracks[t][0]]] for t in track_ids]
    )
    structure_error = float(np.median(np.linalg.norm(aligned - reference, axis=1)))
    diameter = float(np.linalg.norm(scene.points.max(axis=0) - scene.points.min(axis=0)))

    errors = model.reprojection_errors()
    summary = metric_table(
        [
            ("Vistas registradas", f"{model.num_registered} / {len(scene.poses)}"),
            ("Puntos triangulados", f"{len(model.points)}"),
            ("Observaciones por punto", f"{len(errors) / max(len(model.points), 1):.2f}"),
            ("RMSE de reproyección", f"{np.sqrt((errors ** 2).mean()):.3f} px"),
            ("Error de rotación, mediana", f"{np.median(pose_errors.rotation_degrees):.4f}&deg;"),
            ("Error de rotación, peor caso", f"{pose_errors.rotation_degrees.max():.4f}&deg;"),
            ("Error de centro, mediana", f"{np.median(pose_errors.center_distance):.5f}"),
            (
                "Error de estructura, mediana",
                f"{100 * structure_error / diameter:.3f}% del diámetro",
            ),
            ("Tiempo de reloj", f"{elapsed:.1f} s"),
        ]
    )
    note = (
        "\nQue el RMSE quede cerca del ruido inyectado es el resultado sano: significa "
        "que el estimador reproduce el error de medida en vez de absorberlo en el "
        "modelo. Los errores de pose y estructura se miden después de alinear la "
        "reconstrucción con la verdad mediante una similitud, porque una "
        "reconstrucción hecha solo con imágenes queda determinada salvo por una.\n"
    )
    return summary + "\n" + note, _scene_figure(model), _error_figure(errors)


def build() -> None:
    """Añade el panel al layout de Gradio que lo envuelve."""
    gr.Markdown(
        "Reconstruye una escena sintética de la que se conoce la verdad. El ruido de "
        "medida y la fracción de correspondencias erróneas los pones tú, y ese es el "
        "objetivo: enseñar dónde dejan de aguantar el solver de cinco puntos, los "
        "estimadores robustos y el bundle adjustment."
    )
    with gr.Row():
        with gr.Column(scale=1):
            views = gr.Slider(4, 20, value=12, step=1, label="Cámaras")
            points = gr.Slider(200, 1500, value=800, step=50, label="Puntos 3D")
            noise = gr.Slider(
                0.0, 4.0, value=0.5, step=0.1, label="Ruido de medida (píxeles)"
            )
            outliers = gr.Slider(
                0.0, 0.5, value=0.10, step=0.01, label="Fracción de outliers groseros"
            )
            seed = gr.Slider(0, 100, value=7, step=1, label="Semilla aleatoria")
            run = gr.Button("Reconstruir", variant="primary")
            summary = gr.Markdown()
        with gr.Column(scale=2):
            scene_plot = gr.Plot(label="Modelo disperso y cámaras (arrastra para orbitar)")
            error_plot = gr.Plot(label="Error de reproyección")

    run.click(
        _run,
        inputs=[views, points, noise, outliers, seed],
        outputs=[summary, scene_plot, error_plot],
    )
