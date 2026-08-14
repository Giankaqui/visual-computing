"""Losses, dataset round trips and an end-to-end training run."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gsplat.cameras import Camera, look_at, orbit_cameras
from gsplat.datasets import load_dataset, save_views, synthetic_dataset
from gsplat.gaussians import GaussianModel
from gsplat.losses import l1_loss, photometric_loss, psnr, ssim
from gsplat.scenes import default_scene, render_scene
from gsplat.trainer import Trainer, TrainingConfig


@pytest.fixture(scope="module")
def dataset():
    return synthetic_dataset(num_train=12, num_test=3, width=64, height=48)


def test_ssim_of_an_image_with_itself_is_one() -> None:
    image = torch.rand((32, 40, 3), dtype=torch.float64)
    assert float(ssim(image, image)) == pytest.approx(1.0, abs=1e-9)


def test_ssim_decreases_with_noise() -> None:
    generator = torch.Generator().manual_seed(0)
    image = torch.rand((48, 48, 3), generator=generator)
    scores = [
        float(ssim(image + level * torch.randn(image.shape, generator=generator), image))
        for level in (0.02, 0.1, 0.3)
    ]
    assert scores[0] > scores[1] > scores[2]


def test_psnr_matches_its_definition() -> None:
    target = torch.zeros((8, 8, 3))
    prediction = torch.full((8, 8, 3), 0.1)
    assert psnr(prediction, target) == pytest.approx(20.0, abs=1e-6)


def test_photometric_loss_mixes_both_terms() -> None:
    generator = torch.Generator().manual_seed(1)
    prediction = torch.rand((24, 24, 3), generator=generator)
    target = torch.rand((24, 24, 3), generator=generator)

    loss, parts = photometric_loss(prediction, target, ssim_weight=0.2)
    expected = 0.8 * parts["l1"] + 0.2 * (1.0 - parts["ssim"])
    assert float(loss) == pytest.approx(expected, rel=1e-6)
    assert parts["l1"] == pytest.approx(float(l1_loss(prediction, target)), rel=1e-6)


def test_rendered_scene_is_deterministic() -> None:
    R, t = look_at(np.array([0.0, -1.0, -4.0]), np.zeros(3), np.array([0.0, -1.0, 0.0]))
    camera = Camera.from_fov(R, t, 48, 36, 50.0)
    first = render_scene(camera, default_scene())
    second = render_scene(camera, default_scene())
    assert np.array_equal(first, second)
    assert first.min() >= 0.0 and first.max() <= 1.0


def test_orbit_cameras_look_at_the_target() -> None:
    target = np.array([0.5, -0.2, 0.3])
    for camera in orbit_cameras(8, radius=3.0, target=target, width=32, height=24):
        direction = target - camera.center
        direction /= np.linalg.norm(direction)
        optical_axis = camera.R.T @ np.array([0.0, 0.0, 1.0])
        assert float(direction @ optical_axis) == pytest.approx(1.0, abs=1e-9)


def test_dataset_roundtrips_through_disk(dataset, tmp_path) -> None:
    save_views(dataset.train, tmp_path)
    restored = load_dataset(tmp_path, test_every=0)

    assert len(restored.train) == len(dataset.train)
    for original, loaded in zip(restored.train, dataset.train, strict=True):
        assert np.allclose(original.camera.R, loaded.camera.R)
        assert np.allclose(original.camera.t, loaded.camera.t)
        # Images round trip through 8-bit PNG, so half a quantization step is
        # the tightest tolerance that can hold.
        assert float((original.image - loaded.image).abs().max()) <= 1.0 / 255.0


def test_missing_camera_file_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path)


def test_camera_rescaling_preserves_the_field_of_view() -> None:
    R, t = look_at(np.array([0.0, 0.0, -3.0]), np.zeros(3), np.array([0.0, -1.0, 0.0]))
    camera = Camera.from_fov(R, t, 200, 100, 60.0)
    half = camera.rescaled(0.5)
    assert half.width == 100 and half.height == 50
    assert half.fx / half.width == pytest.approx(camera.fx / camera.width)


@pytest.mark.slow
def test_training_improves_held_out_quality(dataset) -> None:
    scene = default_scene()
    model = GaussianModel.random(4000, center=scene.center, radius=2.5, seed=0)
    config = TrainingConfig(iterations=600, evaluate_every=0, log_every=10**9, seed=0)
    config.density.start_iteration = 200
    config.density.interval = 100
    config.density.stop_iteration = 500

    trainer = Trainer(model, dataset, config)
    before = trainer.evaluate(dataset.test)
    history = trainer.train(verbose=False)
    after = trainer.evaluate(dataset.test)

    assert after.psnr > before.psnr + 2.0
    assert after.ssim > before.ssim
    assert np.mean(history.loss[-50:]) < np.mean(history.loss[:50])
    assert len(trainer.model) != 4000
