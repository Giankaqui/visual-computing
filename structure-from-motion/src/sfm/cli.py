"""Command line interface.

Two entry points are provided: ``reconstruct`` runs the pipeline on a directory
of images, and ``demo`` runs it on a synthetic scene and reports the error
against ground truth, which is the fastest way to check that an installation
behaves as expected.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .camera import PinholeCamera
from .features import detect_and_describe, list_images, load_image, match_all_pairs
from .io import export_reconstruction
from .metrics import align_similarity, compare_poses
from .reconstruction import ReconstructionOptions, reconstruct
from .synthetic import make_scene, to_matching_problem
from .visualize import plot_reconstruction


def _options_from_args(args: argparse.Namespace) -> ReconstructionOptions:
    return ReconstructionOptions(
        essential_threshold=args.essential_threshold,
        pnp_threshold=args.pnp_threshold,
        min_triangulation_angle=args.min_angle,
        max_reprojection_error=args.max_error,
        seed=args.seed,
        verbose=not args.quiet,
    )


def _run_reconstruct(args: argparse.Namespace) -> int:
    paths = list_images(args.images)
    if len(paths) < 2:
        print(f"need at least two images in {args.images}")
        return 1

    print(f"extracting features from {len(paths)} images")
    started = time.perf_counter()
    features = []
    for path in paths:
        image = load_image(path)
        feature_set = detect_and_describe(image, max_features=args.max_features)
        features.append(feature_set)
        print(f"  {path.name:<40} {len(feature_set):>6} keypoints")

    cameras = [
        PinholeCamera.from_fov(f.size[0], f.size[1], args.fov)
        if args.fov is not None
        else PinholeCamera.guess_from_image(f.size[0], f.size[1])
        for f in features
    ]

    print("matching")
    matches = match_all_pairs(features, ratio=args.ratio, verbose=not args.quiet)
    if not matches:
        print("no image pair produced enough matches")
        return 1

    model = reconstruct(
        keypoints=[f.keypoints for f in features],
        matches=matches,
        cameras=cameras,
        colors=[f.colors for f in features],
        options=_options_from_args(args),
    )
    if not model.points:
        print("reconstruction failed: no points were triangulated")
        return 1

    written = export_reconstruction(args.output, model, [p.name for p in paths])
    figure = plot_reconstruction(model, Path(args.output) / "reconstruction.png")
    print(f"wrote {written['points']}, {written['cameras']} and {figure}")
    print(f"total time {time.perf_counter() - started:.1f} s")
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    scene = make_scene(
        num_points=args.points,
        num_views=args.views,
        noise_pixels=args.noise,
        outlier_fraction=args.outliers,
        seed=args.seed,
    )
    keypoints, matches, cameras = to_matching_problem(scene, seed=args.seed)

    started = time.perf_counter()
    model = reconstruct(
        keypoints=keypoints,
        matches=matches,
        cameras=cameras,
        options=_options_from_args(args),
    )
    elapsed = time.perf_counter() - started
    if model.num_registered < 2:
        print("reconstruction failed")
        return 1

    estimated_centers = np.array([model.poses[v].center for v in sorted(model.poses)])
    true_centers = np.array([scene.poses[v].center for v in sorted(model.poses)])
    transform = align_similarity(estimated_centers, true_centers)
    errors = compare_poses(model.poses, scene.poses, transform)

    feature_to_scene_point = {
        (view, feature): int(point_id)
        for view in range(len(scene.poses))
        for feature, point_id in enumerate(scene.observations_for_view(view)[0])
    }
    track_ids = sorted(model.points)
    aligned_points = transform.apply(np.array([model.points[t] for t in track_ids]))
    reference_points = np.array(
        [scene.points[feature_to_scene_point[model.tracks.tracks[t][0]]] for t in track_ids]
    )
    point_errors = np.linalg.norm(aligned_points - reference_points, axis=1)
    scene_diameter = float(np.linalg.norm(scene.points.max(axis=0) - scene.points.min(axis=0)))

    print()
    print(model.summary())
    print(errors.summary())
    print(
        f"structure error median {np.median(point_errors):.5f} "
        f"({100 * np.median(point_errors) / scene_diameter:.3f} percent of the "
        f"{scene_diameter:.2f} scene diameter), {elapsed:.1f} s"
    )

    if args.output:
        written = export_reconstruction(args.output, model)
        figure = plot_reconstruction(model, Path(args.output) / "reconstruction.png")
        print(f"wrote {written['points']}, {written['cameras']} and {figure}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sfm", description="Incremental structure from motion from calibrated images."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--essential-threshold", type=float, default=1.5,
                        help="Sampson inlier threshold in pixels (default: %(default)s)")
    common.add_argument("--pnp-threshold", type=float, default=4.0,
                        help="reprojection inlier threshold for pose estimation "
                             "(default: %(default)s)")
    common.add_argument("--min-angle", type=float, default=2.0,
                        help="minimum triangulation angle in degrees (default: %(default)s)")
    common.add_argument("--max-error", type=float, default=4.0,
                        help="maximum reprojection error kept after bundle adjustment "
                             "(default: %(default)s)")
    common.add_argument("--seed", type=int, default=0, help="random seed (default: %(default)s)")
    common.add_argument("--quiet", action="store_true", help="suppress per-step progress")

    run = subparsers.add_parser(
        "reconstruct", parents=[common], help="reconstruct a folder of images"
    )
    run.add_argument("images", type=Path, help="directory containing the input images")
    run.add_argument("--output", type=Path, required=True, help="directory for the results")
    run.add_argument("--fov", type=float, default=None,
                     help="horizontal field of view in degrees; guessed at 55 when omitted")
    run.add_argument("--max-features", type=int, default=8000,
                     help="maximum SIFT keypoints per image (default: %(default)s)")
    run.add_argument("--ratio", type=float, default=0.8,
                     help="Lowe ratio for descriptor matching (default: %(default)s)")
    run.set_defaults(handler=_run_reconstruct)

    demo = subparsers.add_parser("demo", parents=[common], help="run on a synthetic scene")
    demo.add_argument("--views", type=int, default=10,
                      help="number of cameras (default: %(default)s)")
    demo.add_argument("--points", type=int, default=600,
                      help="number of 3D points (default: %(default)s)")
    demo.add_argument("--noise", type=float, default=0.5,
                      help="measurement noise in pixels (default: %(default)s)")
    demo.add_argument("--outliers", type=float, default=0.05,
                      help="fraction of corrupted observations (default: %(default)s)")
    demo.add_argument("--output", type=Path, default=None,
                      help="optional directory for the results")
    demo.set_defaults(handler=_run_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
