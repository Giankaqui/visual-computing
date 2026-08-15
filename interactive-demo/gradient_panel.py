"""Panel interactivo de las operaciones en el dominio del gradiente.

Cada resolución tarda decenas de milisegundos, así que los controles actualizan
el resultado directamente en vez de esconderlo detrás de un botón. Esa
respuesta inmediata es el objetivo: los parámetros de estos operadores cuestan
de razonar en abstracto y se vuelven obvios en cuanto puedes barrerlos.
"""

from __future__ import annotations

import time

import gradio as gr
import numpy as np
from gradient_domain.hdr import ToneMapConfig, tone_map
from gradient_domain.poisson import illumination_change, seamless_clone, texture_flatten
from gradient_domain.synthetic import (
    make_compositing_example,
    make_radiance_map,
    make_texture_example,
)

from common import as_display_image, metric_table

_example = make_compositing_example()
_texture, _edges = make_texture_example()
_radiance = make_radiance_map()

_LUMINANCE = np.array([0.2126, 0.7152, 0.0722])


def _clone(mode: str, domain: str, method: str):
    started = time.perf_counter()
    result, reports = seamless_clone(
        _example.source,
        _example.target,
        _example.mask,
        _example.offset,
        mode=mode,
        domain=domain,
        method=method,
    )
    elapsed = time.perf_counter() - started

    report = reports[0] if reports else None
    summary = metric_table(
        [
            ("Solver", report.method if report else "-"),
            (
                "Iteraciones",
                "-" if not report or report.iterations == 0 else str(report.iterations),
            ),
            (
                "Residuo relativo",
                f"{report.relative_residual:.1e}" if report else "-",
            ),
            ("Tiempo total, tres canales", f"{1000 * elapsed:.0f} ms"),
        ]
    )
    note = {
        "import": "Los gradientes de la fuente se transfieren tal cual.",
        "mixed": (
            "Gana el mayor de los dos gradientes, componente a componente, así que la "
            "estructura del destino sobrevive por debajo de una región de fuente plana."
        ),
        "average": "Los dos campos de gradiente se promedian, lo que produce un fantasma.",
    }[mode]
    domain_note = {
        "mask": (
            "Las incógnitas son solo los píxeles seleccionados; fuera de ellos el "
            "destino se preserva exactamente."
        ),
        "rectangle": (
            "Las incógnitas son un rectángulo alrededor de la selección, que es lo que "
            "multigrid puede acelerar. Fuera de la selección el resultado difiere del "
            "destino en una pequeña corrección armónica."
        ),
    }[domain]
    return as_display_image(result), f"{summary}\n\n{note} {domain_note}"


def _tonemap(alpha: float, beta: float, saturation: float, exposure: float):
    started = time.perf_counter()
    mapped, factor = tone_map(
        _radiance, ToneMapConfig(alpha=alpha, beta=beta, saturation=saturation)
    )
    elapsed = time.perf_counter() - started

    reference = float(np.percentile(_radiance, 99.5))
    fixed = np.clip(exposure * _radiance / reference, 0.0, 1.0) ** (1 / 2.2)

    def decades(image: np.ndarray) -> float:
        luminance = np.maximum(image @ _LUMINANCE, 1e-8)
        low, high = np.percentile(luminance, [1, 99])
        return float(np.log10(max(high, 1e-8) / max(low, 1e-8)))

    summary = metric_table(
        [
            ("Rango dinámico de entrada", f"{decades(_radiance):.2f} décadas"),
            ("Tras el tone mapping", f"{decades(mapped):.2f} décadas"),
            ("Con exposición fija", f"{decades(fixed):.2f} décadas"),
            ("Rango del factor de atenuación", f"{factor.min():.2f} a {factor.max():.2f}"),
            ("Tiempo de resolución", f"{1000 * elapsed:.0f} ms"),
        ]
    )
    note = (
        "El exponente va al revés de lo que uno esperaría: `beta = 1` deja todos los "
        "gradientes intactos y los valores más pequeños comprimen más. Bájalo y aparece "
        "el interior; súbelo a uno y el resultado colapsa sobre la exposición fija de "
        "al lado."
    )
    return as_display_image(mapped), as_display_image(fixed), f"{summary}\n\n{note}"


def _flatten(strength: float, method: str):
    started = time.perf_counter()
    result, reports = texture_flatten(_texture, _edges * strength, method=method)
    elapsed = time.perf_counter() - started

    def variation(image: np.ndarray) -> float:
        interior = ~_edges.astype(bool)
        gradient = np.abs(np.diff(image, axis=1)).mean(axis=2)
        return float(gradient[interior[:, :-1]].mean())

    report = reports[0] if reports else None
    summary = metric_table(
        [
            ("Solver", report.method if report else "-"),
            (
                "Iteraciones",
                "-" if not report or report.iterations == 0 else str(report.iterations),
            ),
            ("Energía de textura, entrada", f"{variation(_texture):.5f}"),
            ("Energía de textura, salida", f"{variation(result):.5f}"),
            ("Tiempo de resolución", f"{1000 * elapsed:.0f} ms"),
        ]
    )
    note = (
        "El campo guía es el gradiente de la imagen multiplicado por el mapa de bordes. "
        "A intensidad máxima solo sobreviven las fronteras entre regiones y todo lo que "
        "hay entre ellas queda tan plano como esas fronteras permiten; a cero el "
        "resultado es una constante."
    )
    return as_display_image(result), f"{summary}\n\n{note}"


def _relight(alpha: float, beta: float, radius: float):
    height, width = _example.target.shape[:2]
    rows, columns = np.mgrid[0:height, 0:width]
    mask = ((rows - 0.84 * height) ** 2 + (columns - 0.5 * width) ** 2) < (radius * width) ** 2

    started = time.perf_counter()
    result, reports = illumination_change(_example.target, mask, alpha=alpha, beta=beta)
    elapsed = time.perf_counter() - started

    changed = float(np.abs(result - _example.target).mean(axis=2)[mask].mean())
    summary = metric_table(
        [
            ("Píxeles seleccionados", f"{int(mask.sum()):,}"),
            ("Cambio medio dentro", f"{changed:.4f}"),
            ("Fuera de la selección", "intacto por construcción"),
            ("Tiempo de resolución", f"{1000 * elapsed:.0f} ms"),
        ]
    )
    note = (
        "Las magnitudes de gradiente dentro de la selección se remapean con "
        "`(alpha / |grad f|) ** beta`, lo que amplifica las pequeñas y atenúa las "
        "grandes. Aquí `beta = 0` es la identidad, el convenio opuesto al del tone "
        "mapper de arriba; cada uno sigue el de su artículo original."
    )
    return as_display_image(result), f"{summary}\n\n{note}"


def build() -> None:
    """Añade el panel al layout de Gradio que lo envuelve."""
    gr.Markdown(
        "Cuatro operaciones, un solo cálculo: elegir un campo de gradiente objetivo y "
        "reconstruir la imagen cuyo gradiente más se le parece. Cada resolución tarda "
        "decenas de milisegundos, así que los resultados siguen a los sliders."
    )

    # Resolver una vez al construir deja todos los paneles llenos antes de la
    # primera interacción.
    initial_clone = _clone("import", "mask", "mgcg")
    initial_tone = _tonemap(0.12, 0.88, 0.55, 60.0)
    initial_flatten = _flatten(1.0, "mgcg")
    initial_relight = _relight(0.05, 0.5, 0.20)

    with gr.Tab("Clonado sin costura"):
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    ["import", "mixed", "average"],
                    value="import",
                    label="Combinación de gradientes",
                )
                domain = gr.Radio(["mask", "rectangle"], value="mask", label="Dominio")
                method = gr.Dropdown(
                    ["mgcg", "multigrid", "cg", "direct"],
                    value="mgcg",
                    label="Solver (solo con dominio rectángulo)",
                )
                clone_summary = gr.Markdown(value=initial_clone[1])
            with gr.Column(scale=2):
                with gr.Row():
                    gr.Image(
                        value=as_display_image(_example.naive_composite()),
                        label="Copiando píxeles",
                        height=280,
                    )
                    clone_out = gr.Image(
                        value=initial_clone[0], label="Copiando gradientes", height=280
                    )
        controls = [mode, domain, method]
        for control in controls:
            control.change(_clone, inputs=controls, outputs=[clone_out, clone_summary])

    with gr.Tab("Tone mapping"):
        with gr.Row():
            with gr.Column(scale=1):
                alpha = gr.Slider(0.02, 0.5, value=0.12, step=0.01, label="Escala de atenuación")
                beta = gr.Slider(
                    0.3, 1.0, value=0.88, step=0.01, label="Exponente (1 = sin cambio)"
                )
                saturation = gr.Slider(0.2, 1.0, value=0.55, step=0.05, label="Saturación de color")
                exposure = gr.Slider(
                    0.5, 200.0, value=60.0, step=0.5, label="Exposición fija de comparación"
                )
                tone_summary = gr.Markdown(value=initial_tone[2])
            with gr.Column(scale=2):
                with gr.Row():
                    tone_out = gr.Image(
                        value=initial_tone[0],
                        label="Tone mapping en el dominio del gradiente",
                        height=300,
                    )
                    exposure_out = gr.Image(
                        value=initial_tone[1], label="Exposición única, gamma 2.2", height=300
                    )
        controls = [alpha, beta, saturation, exposure]
        for control in controls:
            control.change(
                _tonemap, inputs=controls, outputs=[tone_out, exposure_out, tone_summary]
            )

    with gr.Tab("Aplanado de textura"):
        with gr.Row():
            with gr.Column(scale=1):
                strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Retención de bordes")
                flatten_method = gr.Dropdown(
                    ["mgcg", "multigrid", "cg", "direct"], value="mgcg", label="Solver"
                )
                flatten_summary = gr.Markdown(value=initial_flatten[1])
            with gr.Column(scale=2):
                with gr.Row():
                    gr.Image(value=as_display_image(_texture), label="Entrada", height=300)
                    flatten_out = gr.Image(
                        value=initial_flatten[0], label="Aplanada", height=300
                    )
        controls = [strength, flatten_method]
        for control in controls:
            control.change(_flatten, inputs=controls, outputs=[flatten_out, flatten_summary])

    with gr.Tab("Contraste local"):
        with gr.Row():
            with gr.Column(scale=1):
                relight_alpha = gr.Slider(
                    0.01, 0.3, value=0.05, step=0.01, label="Magnitud de gradiente sin tocar"
                )
                relight_beta = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05, label="Exponente (0 = sin cambio)"
                )
                relight_radius = gr.Slider(
                    0.05, 0.35, value=0.20, step=0.01, label="Radio de la selección"
                )
                relight_summary = gr.Markdown(value=initial_relight[1])
            with gr.Column(scale=2):
                with gr.Row():
                    gr.Image(
                        value=as_display_image(_example.target), label="Entrada", height=280
                    )
                    relight_out = gr.Image(
                        value=initial_relight[0], label="Con contraste local", height=280
                    )
        controls = [relight_alpha, relight_beta, relight_radius]
        for control in controls:
            control.change(_relight, inputs=controls, outputs=[relight_out, relight_summary])
