"""Checks on adaptive density control and on the optimizer surgery it requires.

The subtle failure mode is not the selection logic but the bookkeeping: when a
parameter tensor is replaced, Adam's moment estimates have to be reindexed the
same way.  If they are not, the tests below see either a shape error on the next
step or a survivor whose moments belong to a different primitive.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gsplat.cameras import Camera, look_at
from gsplat.densify import DensityConfig, DensityController, append, prune
from gsplat.gaussians import GaussianModel


@pytest.fixture
def setup() -> tuple[GaussianModel, torch.optim.Optimizer]:
    generator = np.random.default_rng(0)
    model = GaussianModel.from_point_cloud(
        generator.normal(size=(50, 3)), generator.random((50, 3))
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter], "lr": 1e-3, "name": name}
            for name, parameter in model.parameter_dict().items()
        ]
    )
    return model, optimizer


def _take_a_step(model: GaussianModel, optimizer: torch.optim.Optimizer) -> None:
    loss = sum(parameter.square().sum() for parameter in model.parameter_dict().values())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def test_pruning_keeps_the_selected_rows(setup) -> None:
    model, optimizer = setup
    _take_a_step(model, optimizer)
    original = model.means.detach().clone()

    keep = torch.zeros(len(model), dtype=torch.bool)
    keep[::2] = True
    prune(model, optimizer, keep)

    assert len(model) == int(keep.sum())
    assert torch.allclose(model.means.detach(), original[keep])
    _take_a_step(model, optimizer)


def test_pruning_reindexes_the_optimizer_state(setup) -> None:
    model, optimizer = setup
    _take_a_step(model, optimizer)

    group = next(g for g in optimizer.param_groups if g["name"] == "means")
    moments = optimizer.state[group["params"][0]]["exp_avg"].clone()

    keep = torch.zeros(len(model), dtype=torch.bool)
    keep[10:20] = True
    prune(model, optimizer, keep)

    group = next(g for g in optimizer.param_groups if g["name"] == "means")
    assert torch.allclose(optimizer.state[group["params"][0]]["exp_avg"], moments[keep])


def test_appending_zeroes_the_moments_of_new_rows(setup) -> None:
    model, optimizer = setup
    _take_a_step(model, optimizer)

    additions = {
        name: parameter.detach()[:5].clone() for name, parameter in model.parameter_dict().items()
    }
    append(model, optimizer, additions)

    group = next(g for g in optimizer.param_groups if g["name"] == "means")
    moments = optimizer.state[group["params"][0]]["exp_avg"]
    assert len(model) == 55
    assert float(moments[-5:].abs().max()) == 0.0
    assert float(moments[:50].abs().max()) > 0.0
    _take_a_step(model, optimizer)


def test_opacity_reset_lowers_every_opacity(setup) -> None:
    model, optimizer = setup
    with torch.no_grad():
        model.opacity_logits.fill_(3.0)
    controller = DensityController(model, extent=1.0)

    controller.reset_opacity(model, optimizer)

    assert float(model.opacities.detach().max()) <= DensityConfig().reset_opacity_value + 1e-6
    _take_a_step(model, optimizer)


def test_large_gradients_add_primitives(setup) -> None:
    model, optimizer = setup
    config = DensityConfig(gradient_threshold=0.0, dense_percent=1.0, min_opacity=0.0)
    controller = DensityController(model, extent=10.0, config=config)
    controller._gradient_sum.fill_(1.0)
    controller._observations.fill_(1.0)

    before = len(model)
    stats = controller.step(model, optimizer, iteration=1000)

    assert stats.cloned == before
    assert stats.split == 0
    assert len(model) == 2 * before
    _take_a_step(model, optimizer)


def test_large_primitives_are_split_and_shrunk(setup) -> None:
    model, optimizer = setup
    with torch.no_grad():
        model.log_scales.fill_(float(np.log(0.5)))
    config = DensityConfig(
        gradient_threshold=0.0, dense_percent=0.01, min_opacity=0.0, max_world_scale=10.0
    )
    controller = DensityController(model, extent=1.0, config=config)
    controller._gradient_sum.fill_(1.0)
    controller._observations.fill_(1.0)

    before = len(model)
    stats = controller.step(model, optimizer, iteration=1000)

    assert stats.split == before
    assert stats.pruned == before
    assert len(model) == config.split_children * before
    assert float(model.scales.detach().max()) < 0.5
    _take_a_step(model, optimizer)


def test_transparent_primitives_are_pruned(setup) -> None:
    model, optimizer = setup
    with torch.no_grad():
        model.opacity_logits[:20] = -20.0
    config = DensityConfig(gradient_threshold=1e9, min_opacity=1e-3, max_world_scale=1e9)
    controller = DensityController(model, extent=1.0, config=config)

    stats = controller.step(model, optimizer, iteration=100)

    assert stats.pruned == 20
    assert len(model) == 30


def test_accumulated_gradients_are_resolution_independent() -> None:
    from gsplat.renderer import render

    generator = np.random.default_rng(2)
    model = GaussianModel.from_point_cloud(
        generator.normal(scale=0.4, size=(64, 3)), generator.random((64, 3))
    )
    R, t = look_at(np.array([0.0, 0.0, -4.0]), np.zeros(3), np.array([0.0, -1.0, 0.0]))

    means = []
    for width, height in ((64, 48), (256, 192)):
        camera = Camera.from_fov(R, t, width, height, 50.0)
        controller = DensityController(model, extent=1.0)
        result = render(model, camera, retain_screen_gradients=True)
        model.zero_grad(set_to_none=True)
        result.image.square().mean().backward()
        controller.accumulate(model, result, camera)
        means.append(float(controller._mean_gradient().mean()))

    assert means[0] == pytest.approx(means[1], rel=0.5)
