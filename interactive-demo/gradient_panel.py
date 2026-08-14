"""Interactive panel for the gradient-domain operations.

Each solve takes tens of milliseconds, so the controls update the result
directly rather than behind a button. That responsiveness is the point: the
parameters of these operators are hard to reason about in the abstract and
obvious the moment you can sweep them.
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
            ("Iterations", "-" if not report or report.iterations == 0 else str(report.iterations)),
            (
                "Relative residual",
                f"{report.relative_residual:.1e}" if report else "-",
            ),
            ("Total time, three channels", f"{1000 * elapsed:.0f} ms"),
        ]
    )
    note = {
        "import": "The source gradients are transferred as they are.",
        "mixed": (
            "Whichever of the source and destination gradients is larger wins, per "
            "component, so structure in the destination survives underneath a flat "
            "source region."
        ),
        "average": "The two gradient fields are averaged, which is a ghosting effect.",
    }[mode]
    domain_note = {
        "mask": (
            "Unknowns are the selected pixels only; the destination is preserved "
            "exactly outside."
        ),
        "rectangle": (
            "Unknowns are a rectangle around the selection, which is what multigrid can "
            "accelerate. Outside the selection the answer differs from the destination "
            "by a small harmonic correction."
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
            ("Input dynamic range", f"{decades(_radiance):.2f} decades"),
            ("After tone mapping", f"{decades(mapped):.2f} decades"),
            ("At the fixed exposure", f"{decades(fixed):.2f} decades"),
            ("Attenuation factor range", f"{factor.min():.2f} to {factor.max():.2f}"),
            ("Solve time", f"{1000 * elapsed:.0f} ms"),
        ]
    )
    note = (
        "The exponent runs backwards from what you might expect: `beta = 1` leaves "
        "every gradient untouched and smaller values compress harder. Drag it down and "
        "the interior appears; drag it to one and the result collapses onto the fixed "
        "exposure beside it."
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
            ("Iterations", "-" if not report or report.iterations == 0 else str(report.iterations)),
            ("Texture energy, input", f"{variation(_texture):.5f}"),
            ("Texture energy, output", f"{variation(result):.5f}"),
            ("Solve time", f"{1000 * elapsed:.0f} ms"),
        ]
    )
    note = (
        "The guidance field is the image gradient multiplied by the edge map. At full "
        "strength only the region boundaries survive and everything between them is "
        "forced as flat as those boundaries allow; at zero the result is a constant."
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
            ("Selected pixels", f"{int(mask.sum()):,}"),
            ("Mean change inside", f"{changed:.4f}"),
            ("Outside the selection", "unchanged by construction"),
            ("Solve time", f"{1000 * elapsed:.0f} ms"),
        ]
    )
    note = (
        "Gradient magnitudes inside the selection are remapped by `(alpha / |grad f|) ** beta`, "
        "which amplifies the small ones and attenuates the large ones. Here `beta = 0` is the "
        "identity, the opposite convention to the tone mapper above; both follow their "
        "original papers."
    )
    return as_display_image(result), f"{summary}\n\n{note}"


def build() -> None:
    """Add the panel to the enclosing Gradio layout."""
    gr.Markdown(
        "Four operations, one computation: choose a target gradient field, then "
        "reconstruct the image whose gradient is closest to it. Each solve takes "
        "tens of milliseconds, so the results follow the sliders directly."
    )

    # Solving once at build time fills every pane before the first interaction.
    initial_clone = _clone("import", "mask", "mgcg")
    initial_tone = _tonemap(0.12, 0.88, 0.55, 60.0)
    initial_flatten = _flatten(1.0, "mgcg")
    initial_relight = _relight(0.05, 0.5, 0.20)

    with gr.Tab("Seamless cloning"):
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    ["import", "mixed", "average"], value="import", label="Gradient combination"
                )
                domain = gr.Radio(["mask", "rectangle"], value="mask", label="Domain")
                method = gr.Dropdown(
                    ["mgcg", "multigrid", "cg", "direct"],
                    value="mgcg",
                    label="Solver (rectangle domain only)",
                )
                clone_summary = gr.Markdown(value=initial_clone[1])
            with gr.Column(scale=2):
                with gr.Row():
                    gr.Image(
                        value=as_display_image(_example.naive_composite()),
                        label="Copying pixels",
                        height=280,
                    )
                    clone_out = gr.Image(
                        value=initial_clone[0], label="Copying gradients", height=280
                    )
        controls = [mode, domain, method]
        for control in controls:
            control.change(_clone, inputs=controls, outputs=[clone_out, clone_summary])

    with gr.Tab("Tone mapping"):
        with gr.Row():
            with gr.Column(scale=1):
                alpha = gr.Slider(0.02, 0.5, value=0.12, step=0.01, label="Attenuation scale")
                beta = gr.Slider(
                    0.3, 1.0, value=0.88, step=0.01, label="Exponent (1 = no change)"
                )
                saturation = gr.Slider(0.2, 1.0, value=0.55, step=0.05, label="Colour saturation")
                exposure = gr.Slider(
                    0.5, 200.0, value=60.0, step=0.5, label="Fixed exposure for comparison"
                )
                tone_summary = gr.Markdown(value=initial_tone[2])
            with gr.Column(scale=2):
                with gr.Row():
                    tone_out = gr.Image(
                        value=initial_tone[0], label="Gradient-domain tone map", height=300
                    )
                    exposure_out = gr.Image(
                        value=initial_tone[1], label="Single exposure, gamma 2.2", height=300
                    )
        controls = [alpha, beta, saturation, exposure]
        for control in controls:
            control.change(
                _tonemap, inputs=controls, outputs=[tone_out, exposure_out, tone_summary]
            )

    with gr.Tab("Texture flattening"):
        with gr.Row():
            with gr.Column(scale=1):
                strength = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Edge retention")
                flatten_method = gr.Dropdown(
                    ["mgcg", "multigrid", "cg", "direct"], value="mgcg", label="Solver"
                )
                flatten_summary = gr.Markdown(value=initial_flatten[1])
            with gr.Column(scale=2):
                with gr.Row():
                    gr.Image(value=as_display_image(_texture), label="Input", height=300)
                    flatten_out = gr.Image(
                        value=initial_flatten[0], label="Flattened", height=300
                    )
        controls = [strength, flatten_method]
        for control in controls:
            control.change(_flatten, inputs=controls, outputs=[flatten_out, flatten_summary])

    with gr.Tab("Local contrast"):
        with gr.Row():
            with gr.Column(scale=1):
                relight_alpha = gr.Slider(
                    0.01, 0.3, value=0.05, step=0.01, label="Gradient magnitude kept unchanged"
                )
                relight_beta = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05, label="Exponent (0 = no change)"
                )
                relight_radius = gr.Slider(
                    0.05, 0.35, value=0.20, step=0.01, label="Selection radius"
                )
                relight_summary = gr.Markdown(value=initial_relight[1])
            with gr.Column(scale=2):
                with gr.Row():
                    gr.Image(
                        value=as_display_image(_example.target), label="Input", height=280
                    )
                    relight_out = gr.Image(
                        value=initial_relight[0], label="After local contrast", height=280
                    )
        controls = [relight_alpha, relight_beta, relight_radius]
        for control in controls:
            control.change(_relight, inputs=controls, outputs=[relight_out, relight_summary])
