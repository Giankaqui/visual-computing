"""Command line interface.

``train`` fits a model to either the procedural scene or a reconstruction
produced by the structure-from-motion project, ``render`` produces novel views
from a checkpoint, and ``export`` writes the procedural scene as images and
cameras so that it can be fed back through a reconstruction pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cameras import orbit_cameras
from .datasets import Dataset, load_dataset, save_views, synthetic_dataset
from .gaussians import GaussianModel, InitializationConfig
from .scenes import default_scene
from .trainer import Trainer, TrainingConfig
from .visualize import comparison_figure, training_curves, turntable_strip


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_dataset(args: argparse.Namespace, device: str) -> Dataset:
    if args.scene == "synthetic":
        return synthetic_dataset(
            num_train=args.train_views,
            num_test=args.test_views,
            width=args.width,
            height=args.height,
            device=device,
        )
    return load_dataset(args.scene, images=args.images, device=device)


def _build_model(args: argparse.Namespace, dataset: Dataset, device: str) -> GaussianModel:
    config = InitializationConfig(sh_degree=args.sh_degree)
    if args.init == "points" and dataset.point_cloud is not None:
        points, colors = dataset.point_cloud
        return GaussianModel.from_point_cloud(points, colors, config, device=device)

    centres = np.stack([view.camera.center for view in dataset.train])
    return GaussianModel.random(
        args.random_points,
        center=centres.mean(axis=0),
        radius=args.random_radius * dataset.extent,
        config=config,
        seed=args.seed,
        device=device,
    )


def _run_train(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)

    dataset = _build_dataset(args, device)
    print(dataset.describe())

    model = _build_model(args, dataset, device)
    print(f"initialized {len(model)} primitives on {device}")

    config = TrainingConfig(
        iterations=args.iterations,
        ssim_weight=args.ssim_weight,
        evaluate_every=args.evaluate_every,
        log_every=args.log_every,
        seed=args.seed,
        device=device,
    )
    config.density.stop_iteration = min(config.density.stop_iteration, int(0.6 * args.iterations))
    if args.no_densify:
        config.density.start_iteration = args.iterations + 1

    trainer = Trainer(model, dataset, config)
    history = trainer.train(verbose=not args.quiet)

    train_quality = trainer.evaluate(dataset.train)
    test_quality = trainer.evaluate(dataset.test)
    print(f"train {train_quality}")
    print(f"test  {test_quality}")
    print(f"{len(trainer.model)} primitives in {history.seconds:.0f} s")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trainer.model.save(output / "model.npz")
    training_curves(history, output / "training.png")
    comparison_figure(
        trainer.model,
        dataset.test[: args.figure_views] or dataset.train[: args.figure_views],
        output / "comparison.png",
        background=dataset.background,
    )

    scene = default_scene()
    centres = np.stack([view.camera.center for view in dataset.train])
    reference = dataset.train[0].camera
    novel = orbit_cameras(
        args.figure_views,
        radius=float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).mean()),
        target=scene.center if args.scene == "synthetic" else centres.mean(axis=0),
        width=reference.width,
        height=reference.height,
        fov_x_degrees=2.0 * np.degrees(np.arctan(0.5 * reference.width / reference.fx)),
        phase=0.13,
    )
    turntable_strip(trainer.model, novel, output / "turntable.png", background=dataset.background)

    summary = {
        "device": device,
        "iterations": args.iterations,
        "primitives": len(trainer.model),
        "seconds": history.seconds,
        "train": {"psnr": train_quality.psnr, "ssim": train_quality.ssim},
        "test": {"psnr": test_quality.psnr, "ssim": test_quality.ssim},
        "test_psnr_curve": dict(zip(history.test_iterations, history.test_psnr, strict=True)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}/model.npz and three figures")
    return 0


def _run_render(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    model = GaussianModel.load(args.checkpoint, device=device)
    scene = default_scene()
    cameras = orbit_cameras(
        args.views,
        radius=args.radius,
        target=scene.center,
        width=args.width,
        height=args.height,
    )
    background = torch.as_tensor(scene.background, dtype=torch.float32, device=device)
    path = turntable_strip(model, cameras, args.output, background=background)
    print(f"wrote {path}")
    return 0


def _run_export(args: argparse.Namespace) -> int:
    dataset = synthetic_dataset(
        num_train=args.views, num_test=0, width=args.width, height=args.height, device="cpu"
    )
    directory = save_views(dataset.train, args.output)
    print(f"wrote {args.views} images and cameras.json to {directory}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gsplat", description="Differentiable 3D Gaussian splatting in pure PyTorch."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="fit a model to a scene")
    train.add_argument("--scene", default="synthetic",
                       help="'synthetic' or a directory holding cameras.json "
                            "(default: %(default)s)")
    train.add_argument("--images", default=None, help="directory holding the images of a scene")
    train.add_argument("--output", type=Path, required=True, help="directory for the results")
    train.add_argument("--iterations", type=int, default=7000,
                       help="optimization steps (default: %(default)s)")
    train.add_argument("--train-views", type=int, default=40,
                       help="synthetic training views (default: %(default)s)")
    train.add_argument("--test-views", type=int, default=8,
                       help="synthetic held-out views (default: %(default)s)")
    train.add_argument("--width", type=int, default=160, help="image width (default: %(default)s)")
    train.add_argument("--height", type=int, default=120,
                       help="image height (default: %(default)s)")
    train.add_argument("--init", choices=("points", "random"), default="points",
                       help="initialize from the point cloud when available (default: %(default)s)")
    train.add_argument("--random-points", type=int, default=20000,
                       help="primitives for random initialization (default: %(default)s)")
    train.add_argument("--random-radius", type=float, default=0.55,
                       help="radius of the initialization ball, in scene extents "
                            "(default: %(default)s)")
    train.add_argument("--sh-degree", type=int, default=3, choices=(0, 1, 2, 3),
                       help="highest spherical harmonic band (default: %(default)s)")
    train.add_argument("--ssim-weight", type=float, default=0.2,
                       help="weight of the structural term (default: %(default)s)")
    train.add_argument("--no-densify", action="store_true",
                       help="disable adaptive density control")
    train.add_argument("--evaluate-every", type=int, default=1000,
                       help="iterations between test evaluations (default: %(default)s)")
    train.add_argument("--log-every", type=int, default=250,
                       help="iterations between progress lines (default: %(default)s)")
    train.add_argument("--figure-views", type=int, default=4,
                       help="views shown in the figures (default: %(default)s)")
    train.add_argument("--device", default="auto", help="torch device (default: %(default)s)")
    train.add_argument("--seed", type=int, default=0, help="random seed (default: %(default)s)")
    train.add_argument("--quiet", action="store_true", help="suppress per-step progress")
    train.set_defaults(handler=_run_train)

    render_parser = subparsers.add_parser("render", help="render novel views from a checkpoint")
    render_parser.add_argument("checkpoint", type=Path, help="model.npz written by train")
    render_parser.add_argument("--output", type=Path, required=True, help="destination PNG")
    render_parser.add_argument("--views", type=int, default=6,
                               help="number of novel views (default: %(default)s)")
    render_parser.add_argument("--radius", type=float, default=4.6,
                               help="orbit radius (default: %(default)s)")
    render_parser.add_argument("--width", type=int, default=320,
                               help="image width (default: %(default)s)")
    render_parser.add_argument("--height", type=int, default=240,
                               help="image height (default: %(default)s)")
    render_parser.add_argument("--device", default="auto",
                               help="torch device (default: %(default)s)")
    render_parser.set_defaults(handler=_run_render)

    export = subparsers.add_parser(
        "export", help="write the procedural scene as images and cameras"
    )
    export.add_argument("--output", type=Path, required=True, help="destination directory")
    export.add_argument("--views", type=int, default=24,
                        help="number of views to render (default: %(default)s)")
    export.add_argument("--width", type=int, default=480, help="image width (default: %(default)s)")
    export.add_argument("--height", type=int, default=360,
                        help="image height (default: %(default)s)")
    export.set_defaults(handler=_run_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
