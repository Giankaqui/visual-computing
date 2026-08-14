"""Incremental reconstruction: seed from two views, then register the rest.

The loop follows the structure that became standard with Bundler and COLMAP.
A well-conditioned pair seeds the map; every further view is registered by
absolute pose estimation against already-triangulated points, new points are
triangulated as soon as two registered views see them, and bundle adjustment is
interleaved so that drift is corrected before it compounds.

Two scheduling choices keep the cost manageable.  Local bundle adjustment after
each registration optimizes only a window of recent views, which is where the
error has just been introduced.  Global bundle adjustment runs when the model
has grown by a fixed ratio, so its cost is amortized logarithmically in the
number of views instead of being paid at every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bundle import BundleOptions, BundleProblem, BundleReport, adjust
from .camera import PinholeCamera, Pose, project_points
from .epipolar import estimate_essential, recover_pose, triangulation_angles
from .pnp import estimate_pose_ransac, refine_pose
from .ransac import RansacOptions
from .tracks import TrackGraph, build_tracks
from .triangulation import refine_point, triangulate_dlt, triangulate_multiview

__all__ = [
    "ReconstructionOptions",
    "Reconstruction",
    "TwoViewGeometry",
    "verify_pairs",
    "reconstruct",
]


@dataclass
class ReconstructionOptions:
    """Thresholds and scheduling parameters for the incremental loop.

    Attributes
    ----------
    essential_threshold : float
        Sampson inlier threshold in pixels for two-view verification.
    pnp_threshold : float
        Reprojection inlier threshold in pixels for absolute pose estimation.
    min_pair_inliers : int
        Pairs with fewer verified matches are dropped before track building.
    min_track_length : int
        Shortest track kept by the track builder.
    min_triangulation_angle : float
        Minimum parallax in degrees for a point to be accepted.
    max_reprojection_error : float
        Points whose error exceeds this in any view are removed after each
        bundle adjustment.
    min_registration_inliers : int
        Minimum number of 2D-3D inliers required to register a view.
    local_window : int
        Number of most recently registered views optimized by local bundle
        adjustment.
    global_bundle_ratio : float
        Global bundle adjustment runs whenever the number of registered views
        exceeds the count at the previous global run by this factor.
    huber_delta : float
        Huber threshold in pixels used inside bundle adjustment.
    seed : int
        Seed shared by all robust estimators, so runs are reproducible.
    verbose : bool
        Print progress for each registration.
    """

    essential_threshold: float = 1.5
    pnp_threshold: float = 4.0
    min_pair_inliers: int = 30
    min_track_length: int = 2
    min_triangulation_angle: float = 2.0
    max_reprojection_error: float = 4.0
    min_registration_inliers: int = 15
    local_window: int = 6
    global_bundle_ratio: float = 1.25
    huber_delta: float = 2.0
    seed: int = 0
    verbose: bool = True


@dataclass
class TwoViewGeometry:
    """Verified geometry of an image pair.

    Attributes
    ----------
    matches : ndarray, shape (m, 2)
        Feature indices that survived the essential-matrix test.
    essential : ndarray, shape (3, 3)
    relative_pose : Pose
        Second view relative to the first, with unit baseline.
    median_angle : float
        Median triangulation angle in degrees over the inliers, a proxy for how
        well conditioned the pair is for initialization.
    """

    matches: np.ndarray
    essential: np.ndarray
    relative_pose: Pose
    median_angle: float

    @property
    def num_inliers(self) -> int:
        return len(self.matches)


@dataclass
class Reconstruction:
    """A partial or complete sparse model.

    Attributes
    ----------
    cameras : list of PinholeCamera
    keypoints : list of ndarray
        Pixel coordinates of the features of each image.
    tracks : TrackGraph
    poses : dict
        Registered views mapped to their world-to-camera transform.
    points : dict
        Track indices mapped to their triangulated position.
    colors : dict
        Track indices mapped to an RGB triple, when colours were supplied.
    registration_order : list of int
        Views in the order they entered the model.
    """

    cameras: list[PinholeCamera]
    keypoints: list[np.ndarray]
    tracks: TrackGraph
    poses: dict[int, Pose] = field(default_factory=dict)
    points: dict[int, np.ndarray] = field(default_factory=dict)
    colors: dict[int, np.ndarray] = field(default_factory=dict)
    registration_order: list[int] = field(default_factory=list)
    track_features: list[dict[int, int]] = field(default_factory=list)

    @property
    def num_registered(self) -> int:
        return len(self.poses)

    def point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the triangulated points and their colours.

        Returns
        -------
        points : ndarray, shape (n, 3)
        colors : ndarray of uint8, shape (n, 3)
            Mid grey is used for tracks without an associated colour.
        """
        track_ids = sorted(self.points)
        points = np.array([self.points[t] for t in track_ids], dtype=float).reshape(-1, 3)
        colors = np.array(
            [self.colors.get(t, np.full(3, 128)) for t in track_ids], dtype=np.uint8
        ).reshape(-1, 3)
        return points, colors

    def camera_centers(self) -> np.ndarray:
        """Return the centres of the registered cameras in registration order."""
        return np.array([self.poses[v].center for v in self.registration_order], dtype=float)

    def reprojection_errors(self) -> np.ndarray:
        """Return the reprojection error of every observation in the model."""
        errors: list[float] = []
        for track_id, point in self.points.items():
            for view, feature in self.track_features[track_id].items():
                if view not in self.poses:
                    continue
                projected, depth = project_points(point[None], self.poses[view], self.cameras[view])
                if depth[0] <= 0:
                    continue
                errors.append(float(np.linalg.norm(projected[0] - self.keypoints[view][feature])))
        return np.array(errors, dtype=float)

    def summary(self) -> str:
        errors = self.reprojection_errors()
        rmse = float(np.sqrt((errors**2).mean())) if len(errors) else float("nan")
        observations_per_point = len(errors) / max(len(self.points), 1)
        return (
            f"{self.num_registered}/{len(self.cameras)} views registered, "
            f"{len(self.points)} points, {len(errors)} observations "
            f"({observations_per_point:.2f} per point), rmse {rmse:.3f} px"
        )


def verify_pairs(
    keypoints: list[np.ndarray],
    matches: dict[tuple[int, int], np.ndarray],
    cameras: list[PinholeCamera],
    options: ReconstructionOptions,
) -> dict[tuple[int, int], TwoViewGeometry]:
    """Run the essential-matrix test on every candidate image pair.

    Parameters
    ----------
    keypoints : list of ndarray
        Pixel coordinates per image.
    matches : dict
        Raw putative matches keyed by ``(i, j)`` with ``i < j``.
    cameras : list of PinholeCamera
    options : ReconstructionOptions

    Returns
    -------
    dict
        Verified geometries, keyed the same way as the input.
    """
    verified: dict[tuple[int, int], TwoViewGeometry] = {}
    for (view_a, view_b), pair_matches in matches.items():
        pair_matches = np.asarray(pair_matches, dtype=np.int64).reshape(-1, 2)
        if len(pair_matches) < 8:
            continue
        points_a = keypoints[view_a][pair_matches[:, 0]]
        points_b = keypoints[view_b][pair_matches[:, 1]]

        result = estimate_essential(
            points_a,
            points_b,
            cameras[view_a],
            cameras[view_b],
            pixel_threshold=options.essential_threshold,
            options=RansacOptions(threshold=options.essential_threshold, seed=options.seed),
        )
        if not result.success or result.num_inliers < options.min_pair_inliers:
            continue

        inlier_matches = pair_matches[result.inliers]
        normalized_a = cameras[view_a].normalize(points_a[result.inliers])
        normalized_b = cameras[view_b].normalize(points_b[result.inliers])
        pose, cheiral = recover_pose(result.model, normalized_a, normalized_b)
        if cheiral.sum() < options.min_pair_inliers:
            continue

        points3d = triangulate_dlt(
            Pose().matrix, pose.matrix, normalized_a[cheiral], normalized_b[cheiral]
        )
        angles = triangulation_angles(points3d, np.zeros(3), pose.center)
        verified[(view_a, view_b)] = TwoViewGeometry(
            matches=inlier_matches[cheiral],
            essential=result.model,
            relative_pose=pose,
            median_angle=float(np.median(angles)),
        )
    return verified


def _choose_initial_pair(
    geometries: dict[tuple[int, int], TwoViewGeometry], options: ReconstructionOptions
) -> tuple[int, int] | None:
    """Pick the pair with the best trade-off between support and parallax.

    A pair with many matches but a short baseline triangulates badly, and a pair
    with a wide baseline but few matches leaves the model unable to grow.  The
    score saturates the parallax term at four times the acceptance threshold, so
    beyond that only the number of inliers matters.
    """
    best_pair, best_score = None, -np.inf
    saturation = 4.0 * options.min_triangulation_angle
    for pair, geometry in geometries.items():
        if geometry.median_angle < options.min_triangulation_angle:
            continue
        score = geometry.num_inliers * min(geometry.median_angle / saturation, 1.0)
        if score > best_score:
            best_pair, best_score = pair, score
    return best_pair


def _triangulate_track(
    reconstruction: Reconstruction, track_id: int, options: ReconstructionOptions
) -> np.ndarray | None:
    """Triangulate one track from all registered views that observe it."""
    views = [v for v in reconstruction.track_features[track_id] if v in reconstruction.poses]
    if len(views) < 2:
        return None

    projections = np.stack([reconstruction.poses[v].matrix for v in views])
    observations = np.stack(
        [
            reconstruction.cameras[v].normalize(
                reconstruction.keypoints[v][reconstruction.track_features[track_id][v]][None]
            )[0]
            for v in views
        ]
    )
    point = triangulate_multiview(projections, observations)
    if not np.isfinite(point).all():
        return None
    point = refine_point(point, projections, observations)

    centers = np.stack([reconstruction.poses[v].center for v in views])
    angles = np.array(
        [
            triangulation_angles(point[None], centers[i], centers[j])[0]
            for i in range(len(views))
            for j in range(i + 1, len(views))
        ]
    )
    if angles.max(initial=0.0) < options.min_triangulation_angle:
        return None

    for view in views:
        projected, depth = project_points(
            point[None], reconstruction.poses[view], reconstruction.cameras[view]
        )
        if depth[0] <= 0:
            return None
        feature = reconstruction.track_features[track_id][view]
        error = np.linalg.norm(projected[0] - reconstruction.keypoints[view][feature])
        if error > options.max_reprojection_error:
            return None
    return point


def _build_problem(
    reconstruction: Reconstruction, free_views: set[int] | None = None
) -> tuple[BundleProblem, list[int], list[int]]:
    """Assemble a bundle adjustment problem from the current model.

    Parameters
    ----------
    reconstruction : Reconstruction
    free_views : set of int or None
        Views allowed to move.  ``None`` frees every registered view except the
        one that entered the model first, which anchors the world frame.

    Returns
    -------
    problem : BundleProblem
    views : list of int
        Model view indices in the order used by the problem.
    track_ids : list of int
        Track indices in the order used by the problem.
    """
    views = sorted(reconstruction.poses)
    view_slot = {view: slot for slot, view in enumerate(views)}
    track_ids = sorted(reconstruction.points)
    track_slot = {track: slot for slot, track in enumerate(track_ids)}

    camera_indices: list[int] = []
    point_indices: list[int] = []
    observations: list[np.ndarray] = []
    for track_id in track_ids:
        for view, feature in reconstruction.track_features[track_id].items():
            if view not in view_slot:
                continue
            camera_indices.append(view_slot[view])
            point_indices.append(track_slot[track_id])
            observations.append(reconstruction.keypoints[view][feature])

    anchor = reconstruction.registration_order[0]
    if free_views is None:
        constant = {view_slot[anchor]}
    else:
        constant = {view_slot[v] for v in views if v not in free_views}
        constant.add(view_slot[anchor])

    problem = BundleProblem(
        poses=[reconstruction.poses[v].copy() for v in views],
        points=np.array([reconstruction.points[t] for t in track_ids], dtype=float),
        cameras=[reconstruction.cameras[v] for v in views],
        camera_indices=np.array(camera_indices, dtype=np.int64),
        point_indices=np.array(point_indices, dtype=np.int64),
        observations=np.array(observations, dtype=float).reshape(-1, 2),
        constant_poses=constant,
    )
    return problem, views, track_ids


def _apply_solution(
    reconstruction: Reconstruction,
    problem: BundleProblem,
    views: list[int],
    track_ids: list[int],
) -> None:
    for view, pose in zip(views, problem.poses, strict=False):
        reconstruction.poses[view] = pose
    for track_id, point in zip(track_ids, problem.points, strict=False):
        reconstruction.points[track_id] = point


def _run_bundle(
    reconstruction: Reconstruction,
    options: ReconstructionOptions,
    free_views: set[int] | None,
    max_iterations: int,
) -> BundleReport | None:
    if len(reconstruction.points) < 3 or reconstruction.num_registered < 2:
        return None
    problem, views, track_ids = _build_problem(reconstruction, free_views)
    report = adjust(
        problem,
        BundleOptions(max_iterations=max_iterations, huber_delta=options.huber_delta),
    )
    _apply_solution(reconstruction, problem, views, track_ids)
    return report


def _filter_points(reconstruction: Reconstruction, options: ReconstructionOptions) -> int:
    """Drop points that no longer satisfy the error and parallax constraints."""
    removed = 0
    for track_id in list(reconstruction.points):
        point = reconstruction.points[track_id]
        views = [v for v in reconstruction.track_features[track_id] if v in reconstruction.poses]
        if len(views) < 2 or not np.isfinite(point).all():
            del reconstruction.points[track_id]
            removed += 1
            continue

        centers = np.stack([reconstruction.poses[v].center for v in views])
        best_angle = max(
            triangulation_angles(point[None], centers[i], centers[j])[0]
            for i in range(len(views))
            for j in range(i + 1, len(views))
        )
        errors = []
        behind = False
        for view in views:
            projected, depth = project_points(
                point[None], reconstruction.poses[view], reconstruction.cameras[view]
            )
            behind |= depth[0] <= 0
            feature = reconstruction.track_features[track_id][view]
            errors.append(np.linalg.norm(projected[0] - reconstruction.keypoints[view][feature]))

        if (
            behind
            or best_angle < options.min_triangulation_angle
            or max(errors) > options.max_reprojection_error
        ):
            del reconstruction.points[track_id]
            removed += 1
    return removed


def _next_view(
    reconstruction: Reconstruction, excluded: set[int]
) -> tuple[int, list[int]] | None:
    """Return the unregistered view that sees the most triangulated points.

    Views whose registration already failed are excluded, so a view that cannot
    be placed does not stall the loop.  It stays a candidate for nothing else:
    once the model has grown, a second attempt would need a different failure
    reason to be worth the cost, and in practice the shared tracks do not
    improve enough to change the outcome.
    """
    best_view, best_tracks = None, []
    for view in range(len(reconstruction.cameras)):
        if view in reconstruction.poses or view in excluded:
            continue
        visible = [
            t for t in reconstruction.tracks.observations[view] if t in reconstruction.points
        ]
        if len(visible) > len(best_tracks):
            best_view, best_tracks = view, visible
    return (best_view, best_tracks) if best_view is not None else None


def reconstruct(
    keypoints: list[np.ndarray],
    matches: dict[tuple[int, int], np.ndarray],
    cameras: list[PinholeCamera],
    colors: list[np.ndarray] | None = None,
    options: ReconstructionOptions | None = None,
) -> Reconstruction:
    """Run incremental structure from motion.

    Parameters
    ----------
    keypoints : list of ndarray
        Pixel coordinates of the features of each image, shape ``(n_i, 2)``.
    matches : dict
        Putative matches keyed by ``(i, j)`` with ``i < j``; each value is an
        array of feature index pairs.  Matches are verified internally.
    cameras : list of PinholeCamera
        Intrinsics per image.
    colors : list of ndarray or None
        Optional per-feature RGB values used to colour the output point cloud.
    options : ReconstructionOptions or None

    Returns
    -------
    Reconstruction
        The largest model that could be grown from the best initial pair.  If no
        pair passes verification the model is returned empty.
    """
    options = options or ReconstructionOptions()
    geometries = verify_pairs(keypoints, matches, cameras, options)
    if options.verbose:
        print(f"verified {len(geometries)}/{len(matches)} pairs")
    if not geometries:
        return Reconstruction(cameras=cameras, keypoints=keypoints, tracks=TrackGraph())

    graph = build_tracks(
        len(cameras),
        {pair: geometry.matches for pair, geometry in geometries.items()},
        min_length=options.min_track_length,
    )
    reconstruction = Reconstruction(
        cameras=cameras,
        keypoints=keypoints,
        tracks=graph,
        track_features=[dict(track) for track in graph.tracks],
    )
    if options.verbose:
        print(f"built {len(graph.tracks)} tracks, lengths {graph.track_length_histogram()}")

    initial_pair = _choose_initial_pair(geometries, options)
    if initial_pair is None:
        return reconstruction

    view_a, view_b = initial_pair
    geometry = geometries[initial_pair]
    reconstruction.poses[view_a] = Pose()
    reconstruction.poses[view_b] = geometry.relative_pose
    reconstruction.registration_order.extend([view_a, view_b])
    if options.verbose:
        print(
            f"seeded from views {view_a} and {view_b}: {geometry.num_inliers} inliers, "
            f"median parallax {geometry.median_angle:.1f} deg"
        )

    for track_id in set(graph.observations[view_a]) & set(graph.observations[view_b]):
        point = _triangulate_track(reconstruction, track_id, options)
        if point is not None:
            reconstruction.points[track_id] = point
    _run_bundle(reconstruction, options, None, max_iterations=30)
    _filter_points(reconstruction, options)

    last_global = reconstruction.num_registered
    failed_views: set[int] = set()
    while True:
        candidate = _next_view(reconstruction, failed_views)
        if candidate is None:
            break
        view, visible_tracks = candidate
        if len(visible_tracks) < options.min_registration_inliers:
            break

        points3d = np.array([reconstruction.points[t] for t in visible_tracks], dtype=float)
        pixels = np.array(
            [
                reconstruction.keypoints[view][reconstruction.track_features[t][view]]
                for t in visible_tracks
            ],
            dtype=float,
        )
        result = estimate_pose_ransac(
            points3d,
            pixels,
            cameras[view],
            pixel_threshold=options.pnp_threshold,
            options=RansacOptions(threshold=options.pnp_threshold, seed=options.seed),
        )
        if not result.success or result.num_inliers < options.min_registration_inliers:
            if options.verbose:
                print(f"view {view}: registration failed ({result.num_inliers} inliers)")
            failed_views.add(view)
            continue

        pose = refine_pose(
            Pose(R=result.model[:, :3], t=result.model[:, 3]),
            points3d[result.inliers],
            pixels[result.inliers],
            cameras[view],
            huber_delta=options.pnp_threshold,
        )
        reconstruction.poses[view] = pose
        reconstruction.registration_order.append(view)

        new_points = 0
        for track_id in graph.observations[view]:
            if track_id in reconstruction.points:
                continue
            point = _triangulate_track(reconstruction, track_id, options)
            if point is not None:
                reconstruction.points[track_id] = point
                new_points += 1

        window = set(reconstruction.registration_order[-options.local_window :])
        _run_bundle(reconstruction, options, window, max_iterations=15)
        removed = _filter_points(reconstruction, options)

        if reconstruction.num_registered >= options.global_bundle_ratio * last_global:
            _run_bundle(reconstruction, options, None, max_iterations=40)
            removed += _filter_points(reconstruction, options)
            last_global = reconstruction.num_registered

        if options.verbose:
            print(
                f"view {view}: {result.num_inliers}/{len(visible_tracks)} pose inliers, "
                f"+{new_points} points, -{removed} filtered, {len(reconstruction.points)} total"
            )

    _run_bundle(reconstruction, options, None, max_iterations=100)
    _filter_points(reconstruction, options)

    if colors is not None:
        for track_id in reconstruction.points:
            samples = [
                colors[view][feature]
                for view, feature in reconstruction.track_features[track_id].items()
                if view in reconstruction.poses
            ]
            if samples:
                reconstruction.colors[track_id] = np.mean(samples, axis=0).astype(np.uint8)

    if options.verbose:
        print(reconstruction.summary())
    return reconstruction
