import numpy as np
import pytest
import torch

from gsplat.gaussians import GaussianModel, InitializationConfig, quaternion_to_rotation


@pytest.fixture
def model() -> GaussianModel:
    generator = np.random.default_rng(0)
    return GaussianModel.from_point_cloud(
        generator.normal(size=(200, 3)), generator.random((200, 3))
    )


def test_quaternions_map_to_rotations() -> None:
    generator = torch.Generator().manual_seed(1)
    quaternions = torch.randn((64, 4), generator=generator) * 3.0
    rotations = quaternion_to_rotation(quaternions)

    identity = torch.eye(3).expand_as(rotations)
    assert torch.allclose(rotations @ rotations.transpose(1, 2), identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotations), torch.ones(64), atol=1e-5)


def test_quaternion_scale_does_not_change_the_rotation() -> None:
    quaternion = torch.tensor([[0.3, -0.7, 0.2, 0.5]])
    assert torch.allclose(
        quaternion_to_rotation(quaternion), quaternion_to_rotation(4.0 * quaternion), atol=1e-6
    )


def test_covariance_is_symmetric_positive_semidefinite(model: GaussianModel) -> None:
    covariances = model.covariances()
    assert torch.allclose(covariances, covariances.transpose(1, 2), atol=1e-6)
    assert float(torch.linalg.eigvalsh(covariances).detach().min()) > -1e-6


def test_covariance_matches_the_explicit_factorization(model: GaussianModel) -> None:
    expected = (
        model.rotations
        @ torch.diag_embed(model.scales**2)
        @ model.rotations.transpose(1, 2)
    )
    assert torch.allclose(model.covariances(), expected, atol=1e-6)


def test_initial_scales_follow_local_density() -> None:
    dense = np.random.default_rng(0).normal(scale=0.05, size=(400, 3))
    sparse = np.random.default_rng(0).normal(scale=1.0, size=(400, 3))

    dense_scale = GaussianModel.from_point_cloud(dense).scales.detach().mean()
    sparse_scale = GaussianModel.from_point_cloud(sparse).scales.detach().mean()
    assert float(dense_scale) < float(sparse_scale)


def test_initial_scales_are_capped_by_the_scene_extent() -> None:
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    config = InitializationConfig(max_initial_scale=0.02)
    model = GaussianModel.from_point_cloud(points, config=config)
    assert float(model.scales.detach().max()) <= 0.02 * 10.0 + 1e-6


def test_initial_colours_survive_the_encoding() -> None:
    colors = np.array([[0.2, 0.4, 0.9], [0.7, 0.1, 0.3]])
    model = GaussianModel.from_point_cloud(np.zeros((2, 3)), colors)
    model.active_sh_degree = 0
    rendered = model.colors(torch.tensor([0.0, 0.0, 5.0]))
    assert torch.allclose(rendered, torch.as_tensor(colors, dtype=torch.float32), atol=1e-6)


def test_view_dependent_colour_changes_with_the_viewpoint(model: GaussianModel) -> None:
    model.active_sh_degree = 3
    with torch.no_grad():
        model.sh_rest.normal_(std=0.5)
    front = model.colors(torch.tensor([0.0, 0.0, 5.0]))
    side = model.colors(torch.tensor([5.0, 0.0, 0.0]))
    assert float((front - side).detach().abs().max()) > 1e-3


def test_raising_the_degree_stops_at_the_maximum() -> None:
    model = GaussianModel.from_point_cloud(np.zeros((4, 3)))
    for _ in range(10):
        model.raise_sh_degree()
    assert model.active_sh_degree == model.sh_degree


def test_save_and_load_roundtrip(model: GaussianModel, tmp_path) -> None:
    model.active_sh_degree = 2
    path = tmp_path / "model.npz"
    model.save(path)
    restored = GaussianModel.load(path)

    assert restored.active_sh_degree == 2
    assert restored.sh_degree == model.sh_degree
    for name, parameter in model.parameter_dict().items():
        assert torch.allclose(getattr(restored, name), parameter, atol=0)


def test_rejects_an_empty_point_cloud() -> None:
    with pytest.raises(ValueError):
        GaussianModel.from_point_cloud(np.zeros((0, 3)))
