"""Interfaz de navegador para los tres proyectos.

Los paneles llaman al mismo código de librería que usan las herramientas de línea
de comandos; aquí no se reimplementa ningún algoritmo ni se reproduce un
resultado cacheado. Los sliders que disparan un cálculo de decenas de
milisegundos actualizan directamente, y el que tarda segundos vive detrás de un
botón.

Se lanza con ``python app.py`` y se abre en la dirección que imprime.
"""

from __future__ import annotations

import argparse

import gradio as gr

import gradient_panel
import reconstruction_panel
import splatting_panel

THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

INTRODUCTION = """
# Proyectos de Computación Visual

Tres proyectos, cada uno movido aquí por el mismo código que corre la línea de comandos.

| Pestaña | Qué estás cambiando |
| --- | --- |
| Structure from motion | El ruido de medida y la fracción de correspondencias erróneas |
| Gaussian splatting | El punto de vista, contra una referencia trazada por rayos |
| Dominio del gradiente | Campos guía, solvers y exponentes de atenuación |
"""


def build_interface() -> gr.Blocks:
    """Monta los tres paneles en una sola página."""
    with gr.Blocks(title="Proyectos de Computación Visual") as interface:
        gr.Markdown(INTRODUCTION)
        with gr.Tab("Structure from motion"):
            reconstruction_panel.build()
        with gr.Tab("Gaussian splatting"):
            splatting_panel.build()
        with gr.Tab("Dominio del gradiente"):
            gradient_panel.build()
    return interface


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860, help="puerto en el que servir")
    parser.add_argument(
        "--share", action="store_true", help="expone una URL pública temporal a través de Gradio"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="no abrir una ventana del navegador al arrancar"
    )
    args = parser.parse_args()

    build_interface().launch(
        theme=THEME,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
