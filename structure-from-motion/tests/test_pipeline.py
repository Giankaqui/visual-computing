"""End-to-end and serialization checks."""

from __future__ import annotations

import numpy as np
import pytest

from sfm.camera import PinholeCamera, Pose
from sfm.io import read_cameras, read_ply, write_cameras, write_ply
from sfm.metrics import align_similarity, compare_poses
from sfm.reconstruction import ReconstructionOptions, reconstruct
from sfm.rotations import random_rotation
from sfm.synthetic import make_scene, to_matching_problem
from sfm.tracks import build_tracks


def test_tracks_merge_transitive_matches() -> None:
    matches = {(0, 1): np.array([[0, 5]]), (1, 2): np.array([[5, 9]])}
    graph = build_tracks(3, matches)

    assert len(graph.tracks) == 1
    assert graph.tracks[0] == [(0, 0), (1, 5), (2, 9)]
    assert graph.feature_to_track[(2, 9)] == 0
    assert graph.observations[1] == [0]


def test_tracks_reject_components_with_two_features_in_one_image() -> None:
    matches = {(0, 1): np.array([[0, 5], [0, 6]])}
    assert build_tracks(2, matches).tracks == []


def test_tracks_respect_the_minimum_length() -> None:
    matches = {(0, 1): np.array([[0, 5]]), (1, 2): np.array([[7, 9]])}
    assert len(build_tracks(3, matches, min_length=2).tracks) == 2
    assert build_tracks(3, matches, min_length=3).tracks == []


def test_ply_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(size=(50, 3))
    colors = rng.integers(0, 256, size=(50, 3), dtype=np.uint8)

    path = tmp_path / "cloud.ply"
    write_ply(path, points, colors)
    restored_points, restored_colors = read_ply(path)

    assert np.allclose(restored_points, points, atol=1e-6)
    assert np.array_equal(restored_colors, colors)


def test_camera_file_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(1)
    cameras = [PinholeCamera.from_fov(640, 480, 50.0) for _ in range(3)]
    poses = {index: Pose(R=random_rotation(rng), t=rng.normal(size=3)) for index in range(3)}

    path = tmp_path / "cameras.json"
    write_cameras(path, cameras, poses, [f"frame_{i}.jpg" for i in range(3)])
    restored_cameras, restored_poses, names = read_cameras(path)

    assert names == ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg"]
    for index in range(3):
        assert np.allclose(restored_poses[index].R, poses[index].R)
        assert np.allclose(restored_poses[index].t, poses[index].t)
        assert restored_cameras[index] == cameras[index]


def test_similarity_alignment_recovers_a_known_transform() -> None:
    rng = np.random.default_rng(8)
    source = rng.normal(size=(40, 3))
    rotation, scale, translation = random_rotation(rng), 2.7, np.array([1.0, -3.0, 0.5])
    target = scale * (source @ rotation.T) + translation

    estimated = align_similarity(source, target)

    assert np.isclose(estimated.scale, scale, rtol=1e-9)
    assert np.allclose(estimated.R, rotation, atol=1e-9)
    assert np.allclose(estimated.t, translation, atol=1e-9)
    assert np.allclose(estimated.apply(source), target, atol=1e-9)


def test_similarity_inverse_is_consistent() -> None:
    rng = np.random.default_rng(9)
    points = rng.normal(size=(20, 3))
    transform = align_similarity(points, 1.5 * (points @ random_rotation(rng).T) + 2.0)
    assert np.allclose(transform.inverse().apply(transform.apply(points)), points, atol=1e-9)


@pytest.mark.slow
def test_incremental_reconstruction_recovers_the_synthetic_scene() -> None:
    scene = make_scene(
        num_points=400, num_views=7, noise_pixels=0.5, outlier_fraction=0.05, seed=13
    )
    keypoints, matches, cameras = to_matching_problem(scene, mismatch_fraction=0.1, seed=13)

    model = reconstruct(
        keypoints, matches, cameras, options=ReconstructionOptions(verbose=False, seed=13)
    )

    assert model.num_registered == len(scene.poses)
    assert len(model.points) > 0.8 * len(scene.points)

    errors = model.reprojection_errors()
    assert np.sqrt((errors**2).mean()) < 1.0

    registered = sorted(model.poses)
    transform = align_similarity(
        np.array([model.poses[view].center for view in registered]),
        np.array([scene.poses[view].center for view in registered]),
    )
    pose_errors = compare_poses(model.poses, scene.poses, transform)
    assert pose_errors.rotation_degrees.max() < 0.5
    assert pose_errors.center_distance.max() < 0.05
