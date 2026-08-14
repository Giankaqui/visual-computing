"""A browser interface for the three projects.

The panels call the same library code the command line tools do; nothing here
reimplements an algorithm or replays a cached result. Sliders that drive a
computation of a few tens of milliseconds update directly, and the one that
takes seconds sits behind a button.

Run with ``python app.py`` and open the address it prints.
"""

from __future__ import annotations

import argparse

import gradio as gr

import gradient_panel
import reconstruction_panel
import splatting_panel

THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

INTRODUCTION = """
# Visual Computing Projects

Three projects, each driven here by the same code the command line runs.

| Tab | What you are changing |
| --- | --- |
| Structure from motion | Measurement noise and the fraction of wrong correspondences |
| Gaussian splatting | The viewpoint, against a ray-traced reference |
| Gradient domain | Guidance fields, solvers and attenuation exponents |
"""


def build_interface() -> gr.Blocks:
    """Assemble the three panels into one page."""
    with gr.Blocks(title="Visual Computing Projects") as interface:
        gr.Markdown(INTRODUCTION)
        with gr.Tab("Structure from motion"):
            reconstruction_panel.build()
        with gr.Tab("Gaussian splatting"):
            splatting_panel.build()
        with gr.Tab("Gradient domain"):
            gradient_panel.build()
    return interface


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860, help="port to serve on")
    parser.add_argument(
        "--share", action="store_true", help="expose a temporary public URL through Gradio"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window on start"
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
