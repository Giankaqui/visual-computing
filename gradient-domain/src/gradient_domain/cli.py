"""Command line interface.

Each subcommand runs one demonstration on procedurally generated data and writes
a figure; ``benchmark`` measures how the solvers scale, and ``demo`` runs
everything into a single directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .benchmark import BenchmarkTable, solver_scaling
from .hdr import ToneMapConfig, tone_map
from .poisson import illumination_change, seamless_clone, texture_flatten
from .synthetic import make_compositing_example, make_radiance_map, make_texture_example
from .visualize import panel_figure, solver_scaling_figure, tone_mapping_figure


def _run_clone(args: argparse.Namespace) -> int:
    example = make_compositing_example(seed=args.seed)
    panels = [("naive paste", example.naive_composite())]

    for mode in ("import", "mixed"):
        result, reports = seamless_clone(
            example.source, example.target, example.mask, example.offset,
            mode=mode, domain=args.domain, method=args.method,
        )
        panels.append((f"{mode} gradients, {args.domain} domain", result))
        print(f"{mode}: {reports[0]}")

    rectangle, _ = seamless_clone(
        example.source, example.target, example.mask, example.offset,
        mode="import", domain="rectangle", method=args.method,
    )
    masked, _ = seamless_clone(
        example.source, example.target, example.mask, example.offset,
        mode="import", domain="mask",
    )
    difference = np.abs(rectangle - masked)
    print(
        f"the two formulations differ by {difference.mean():.2e} on average "
        f"and {difference.max():.2e} at worst"
    )

    path = panel_figure(panels, args.output / "seamless_cloning.png")
    print(f"wrote {path}")
    return 0


def _run_flatten(args: argparse.Namespace) -> int:
    image, edges = make_texture_example(seed=args.seed)
    flattened, reports = texture_flatten(image, edges, method=args.method)
    print(reports[0])

    path = panel_figure(
        [
            ("input", image),
            ("retained edges", np.repeat(edges[..., None], 3, axis=2)),
            ("flattened", flattened),
        ],
        args.output / "texture_flattening.png",
    )
    print(f"wrote {path}")
    return 0


def _run_relight(args: argparse.Namespace) -> int:
    example = make_compositing_example(seed=args.seed)
    height, width = example.target.shape[:2]
    rows, columns = np.mgrid[0:height, 0:width]
    # The selection stays inside the shadowed foreground.  Crossing a structural
    # edge such as the horizon would attenuate that edge along with the texture,
    # which is a real effect of the operator and a poor demonstration of it.
    mask = ((rows - 0.84 * height) ** 2 + (columns - 0.5 * width) ** 2) < (0.20 * width) ** 2

    enhanced, reports = illumination_change(
        example.target, mask, alpha=args.alpha, beta=args.beta
    )
    print(reports[0])

    detail = np.abs(enhanced - example.target).mean(axis=2)
    print(f"mean change inside the selection: {detail[mask].mean():.4f}")

    path = panel_figure(
        [
            ("input", example.target),
            ("selection", np.repeat(mask[..., None].astype(float), 3, axis=2)),
            (f"local contrast, alpha={args.alpha}, beta={args.beta}", enhanced),
        ],
        args.output / "illumination_change.png",
    )
    print(f"wrote {path}")
    return 0


def _run_tonemap(args: argparse.Namespace) -> int:
    radiance = make_radiance_map(seed=args.seed)
    luminance = radiance @ np.array([0.2126, 0.7152, 0.0722])
    decades = np.log10(luminance.max() / luminance.min())
    print(f"radiance spans {decades:.2f} decades")

    mapped, factor = tone_map(
        radiance, ToneMapConfig(alpha=args.alpha, beta=args.beta, saturation=args.saturation)
    )
    path = tone_mapping_figure(radiance, mapped, factor, args.output / "tone_mapping.png")
    print(f"wrote {path}")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    results = solver_scaling(args.sizes, tolerance=args.tolerance, seed=args.seed)
    table = BenchmarkTable(results)
    print(table.to_markdown())

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(
        json.dumps(
            {name: [record.as_dict() for record in records] for name, records in results.items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output / "benchmark.md").write_text(table.to_markdown() + "\n", encoding="utf-8")

    path = solver_scaling_figure(
        {
            name: [record.as_dict() for record in records]
            for name, records in results.items()
            if records
        },
        args.output / "solver_scaling.png",
    )
    print(f"wrote {path}")
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    _run_clone(args)
    _run_flatten(args)
    # Local range compression wants a much gentler exponent than tone mapping,
    # so it does not inherit the shared alpha and beta of the demo command.
    _run_relight(
        argparse.Namespace(output=args.output, seed=args.seed, alpha=0.05, beta=0.5)
    )
    _run_tonemap(args)
    return _run_benchmark(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gradient-domain",
        description="Gradient-domain image processing with multigrid Poisson solvers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", type=Path, default=Path("outputs"),
                        help="directory for the figures (default: %(default)s)")
    common.add_argument("--seed", type=int, default=0, help="random seed (default: %(default)s)")

    clone = subparsers.add_parser("clone", parents=[common], help="seamless cloning")
    clone.add_argument("--domain", choices=("mask", "rectangle"), default="mask",
                       help="formulation to use (default: %(default)s)")
    clone.add_argument("--method", default="mgcg",
                       help="solver for the rectangle formulation (default: %(default)s)")
    clone.set_defaults(handler=_run_clone)

    flatten = subparsers.add_parser("flatten", parents=[common], help="texture flattening")
    flatten.add_argument("--method", default="mgcg", help="solver name (default: %(default)s)")
    flatten.set_defaults(handler=_run_flatten)

    relight = subparsers.add_parser("relight", parents=[common],
                                    help="local dynamic range compression")
    relight.add_argument("--alpha", type=float, default=0.05,
                         help="gradient magnitude left unchanged (default: %(default)s)")
    relight.add_argument("--beta", type=float, default=0.5,
                         help="remapping exponent; 0 leaves the image unchanged "
                              "(default: %(default)s)")
    relight.set_defaults(handler=_run_relight)

    tonemap = subparsers.add_parser(
        "tonemap", parents=[common], help="high dynamic range tone mapping"
    )
    tonemap.add_argument("--alpha", type=float, default=0.12,
                         help="attenuation scale (default: %(default)s)")
    tonemap.add_argument("--beta", type=float, default=0.88,
                         help="attenuation exponent; 1 leaves gradients unchanged and "
                              "smaller values compress more (default: %(default)s)")
    tonemap.add_argument("--saturation", type=float, default=0.55,
                         help="colour saturation exponent (default: %(default)s)")
    tonemap.set_defaults(handler=_run_tonemap)

    benchmark = subparsers.add_parser("benchmark", parents=[common], help="solver scaling study")
    benchmark.add_argument("--sizes", type=int, nargs="+",
                           default=[63, 127, 255, 511, 1023],
                           help="square grid sizes (default: %(default)s)")
    benchmark.add_argument("--tolerance", type=float, default=1e-8,
                           help="target relative residual (default: %(default)s)")
    benchmark.set_defaults(handler=_run_benchmark)

    demo = subparsers.add_parser("demo", parents=[common], help="run every demonstration")
    demo.add_argument("--method", default="mgcg", help="solver name (default: %(default)s)")
    demo.add_argument("--domain", choices=("mask", "rectangle"), default="mask",
                      help="formulation for cloning (default: %(default)s)")
    demo.add_argument("--alpha", type=float, default=0.12, help="attenuation scale")
    demo.add_argument("--beta", type=float, default=0.88, help="attenuation exponent")
    demo.add_argument("--saturation", type=float, default=0.55, help="colour saturation exponent")
    demo.add_argument("--sizes", type=int, nargs="+", default=[63, 127, 255, 511],
                      help="square grid sizes for the benchmark (default: %(default)s)")
    demo.add_argument("--tolerance", type=float, default=1e-8, help="target relative residual")
    demo.set_defaults(handler=_run_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
