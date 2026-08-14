"""Adaptive density control.

Gradient descent can move, resize and recolour primitives, but it cannot change
how many there are, and the number needed is not known in advance.  The control
loop from Kerbl et al. (2023) infers it from the optimization itself: where the
reconstruction is under-parameterized, the projected centre of a primitive
receives a large and persistent gradient, because a single splat is being pulled
towards several image features at once.

Primitives with a large accumulated screen-space gradient are handled in one of
two ways.  A small one is *cloned*, which adds capacity where geometry is
missing.  A large one is *split* into children drawn from its own distribution
with scales divided by a constant, which adds resolution where a single splat
covers detail it cannot represent.  Primitives that have become transparent are
removed, and opacity is periodically pushed back towards zero so that splats
which are no longer justified by the data have to re-earn their opacity instead
of lingering as floaters near the cameras.

Every operation changes the number of rows in the parameter tensors, so the
optimizer state has to be edited in step with them.  Dropping the state would
restart Adam's moment estimates for every surviving primitive and visibly stall
training after each densification; the helpers below slice and pad the moments
so that survivors keep their history and newcomers start from zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .cameras import Camera
from .gaussians import GaussianModel
from .renderer import RenderResult

__all__ = ["DensityConfig", "DensityStats", "DensityController"]


@dataclass
class DensityConfig:
    """Thresholds and schedule of the density control.

    Attributes
    ----------
    gradient_threshold : float
        Mean screen-space gradient, in normalized device coordinates, above
        which a primitive is densified.
    dense_percent : float
        Fraction of the scene extent separating clone from split.
    split_children : int
        Children produced when splitting.
    split_scale_divisor : float
        Factor by which children shrink; the original work uses 1.6, which is
        close to the ratio that keeps the summed density of the children equal
        to the parent's.
    min_opacity : float
        Primitives below this opacity are removed.
    max_world_scale : float
        Upper bound on a primitive's largest standard deviation, as a fraction
        of the scene extent.
    max_screen_radius : float
        Upper bound on the screen radius in pixels, enforced only after
        ``screen_pruning_from`` iterations so that early large splats can still
        cover the image.
    screen_pruning_from : int
    max_primitives : int
        Ceiling on the model size.  The gradient threshold alone does not bound
        it: the threshold is a property of the image resolution and the scene,
        so a setting that produces a reasonable model at one megapixel produces
        a wildly over-parameterized one at a tenth of that.  Growth stops at the
        ceiling while pruning continues, which lets the model keep improving by
        replacing primitives rather than adding them.
    start_iteration, stop_iteration : int
        Densification window.
    interval : int
        Iterations between densification passes.
    opacity_reset_interval : int
        Iterations between opacity resets; zero disables them.
    reset_opacity_value : float
        Opacity every primitive is clamped to at a reset.
    """

    gradient_threshold: float = 4.0e-4
    dense_percent: float = 0.01
    split_children: int = 2
    split_scale_divisor: float = 1.6
    min_opacity: float = 5.0e-3
    max_world_scale: float = 0.1
    max_screen_radius: float = 60.0
    screen_pruning_from: int = 3000
    max_primitives: int = 60_000
    start_iteration: int = 500
    stop_iteration: int = 12000
    interval: int = 100
    opacity_reset_interval: int = 3000
    reset_opacity_value: float = 0.01


@dataclass
class DensityStats:
    """Outcome of one densification pass."""

    cloned: int = 0
    split: int = 0
    pruned: int = 0
    total: int = 0
    events: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"+{self.cloned} cloned, +{self.split} split, "
            f"-{self.pruned} pruned, {self.total} total"
        )


def _group_for(optimizer: torch.optim.Optimizer, name: str) -> dict:
    for group in optimizer.param_groups:
        if group.get("name") == name:
            return group
    raise KeyError(f"the optimizer has no parameter group named {name!r}")


def _swap_parameter(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    name: str,
    value: torch.Tensor,
    state_transform,
) -> None:
    """Replace one parameter tensor and carry its Adam moments across.

    Parameters
    ----------
    model : GaussianModel
    optimizer : torch.optim.Optimizer
    name : str
        Parameter and parameter-group name.
    value : Tensor
        New tensor; it becomes a leaf parameter.
    state_transform : callable
        ``state_transform(moment) -> moment`` applied to ``exp_avg`` and
        ``exp_avg_sq``, mirroring whatever reindexing ``value`` encodes.
    """
    group = _group_for(optimizer, name)
    old = group["params"][0]
    state = optimizer.state.pop(old, None)

    new = nn.Parameter(value.contiguous().detach().clone())
    group["params"][0] = new
    setattr(model, name, new)

    if state is not None:
        optimizer.state[new] = {
            key: state_transform(moment) if key in ("exp_avg", "exp_avg_sq") else moment
            for key, moment in state.items()
        }


def prune(model: GaussianModel, optimizer: torch.optim.Optimizer, keep: torch.Tensor) -> None:
    """Keep the primitives selected by the boolean mask ``keep``."""
    for name, parameter in model.parameter_dict().items():
        _swap_parameter(model, optimizer, name, parameter[keep], lambda moment: moment[keep])


def append(
    model: GaussianModel, optimizer: torch.optim.Optimizer, additions: dict[str, torch.Tensor]
) -> None:
    """Append new primitives, giving them zeroed optimizer moments."""
    for name, parameter in model.parameter_dict().items():
        extra = additions[name]
        combined = torch.cat([parameter.detach(), extra], dim=0)

        def pad(moment: torch.Tensor, extra: torch.Tensor = extra) -> torch.Tensor:
            return torch.cat([moment, torch.zeros_like(extra)], dim=0)

        _swap_parameter(model, optimizer, name, combined, pad)


class DensityController:
    """Accumulates visibility statistics and applies the density operations.

    Parameters
    ----------
    model : GaussianModel
    extent : float
        Radius of the scene, used to turn the relative thresholds into absolute
        ones.
    config : DensityConfig or None
    """

    def __init__(
        self, model: GaussianModel, extent: float, config: DensityConfig | None = None
    ) -> None:
        self.config = config or DensityConfig()
        self.extent = float(extent)
        self._reset_statistics(model)

    def _reset_statistics(self, model: GaussianModel) -> None:
        device = model.means.device
        count = len(model)
        self._gradient_sum = torch.zeros(count, device=device)
        self._observations = torch.zeros(count, device=device)
        self._max_screen_radius = torch.zeros(count, device=device)

    def accumulate(self, model: GaussianModel, result: RenderResult, camera: Camera) -> None:
        """Record screen-space gradients and radii from one rendered view.

        The gradient is rescaled to normalized device coordinates so the
        threshold does not have to be retuned when the training resolution
        changes.
        """
        gradient = result.projected.means2d.grad
        if gradient is None:
            return
        if len(self._gradient_sum) != len(model):
            self._reset_statistics(model)

        visible = result.projected.visible
        scale = torch.tensor(
            [0.5 * camera.width, 0.5 * camera.height],
            device=gradient.device,
            dtype=gradient.dtype,
        )
        self._gradient_sum[visible] += (gradient * scale).norm(dim=1)
        self._observations[visible] += 1.0
        self._max_screen_radius[visible] = torch.maximum(
            self._max_screen_radius[visible], result.projected.radii.detach()
        )

    def _mean_gradient(self) -> torch.Tensor:
        return self._gradient_sum / self._observations.clamp_min(1.0)

    def _clone(
        self, model: GaussianModel, selected: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach()[selected].clone()
            for name, parameter in model.parameter_dict().items()
        }

    def _split(
        self, model: GaussianModel, selected: torch.Tensor, generator: torch.Generator | None
    ) -> dict[str, torch.Tensor]:
        children = self.config.split_children
        scales = model.scales.detach()[selected]
        rotations = model.rotations.detach()[selected]

        offsets = torch.randn(
            (children, *scales.shape), device=scales.device, dtype=scales.dtype, generator=generator
        )
        displacement = torch.einsum(
            "nij,knj->kni", rotations, offsets * scales[None]
        ).reshape(-1, 3)

        def repeat(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach()[selected].repeat(
                children, *([1] * (tensor.dim() - 1))
            )

        parameters = model.parameter_dict()
        return {
            "means": repeat(parameters["means"]) + displacement,
            "log_scales": repeat(parameters["log_scales"])
            - torch.log(torch.tensor(self.config.split_scale_divisor, device=scales.device)),
            "quaternions": repeat(parameters["quaternions"]),
            "opacity_logits": repeat(parameters["opacity_logits"]),
            "sh_dc": repeat(parameters["sh_dc"]),
            "sh_rest": repeat(parameters["sh_rest"]),
        }

    def step(
        self,
        model: GaussianModel,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        generator: torch.Generator | None = None,
    ) -> DensityStats:
        """Run one densification and pruning pass.

        Parameters
        ----------
        model : GaussianModel
        optimizer : torch.optim.Optimizer
            Must have one parameter group per model parameter, each carrying a
            ``name`` entry.
        iteration : int
            Current training iteration, used by the pruning schedule.
        generator : torch.Generator or None
            Source of randomness for splitting, so runs are reproducible.

        Returns
        -------
        DensityStats
        """
        config = self.config
        gradients = self._mean_gradient()
        scales = model.scales.detach()
        largest_scale = scales.max(dim=1).values

        needs_density = gradients >= config.gradient_threshold
        if len(model) >= config.max_primitives:
            needs_density = torch.zeros_like(needs_density)
        small = largest_scale <= config.dense_percent * self.extent
        clone_mask = needs_density & small
        split_mask = needs_density & ~small

        stats = DensityStats(cloned=int(clone_mask.sum()), split=int(split_mask.sum()))

        additions: dict[str, torch.Tensor] = {}
        if stats.cloned:
            additions = self._clone(model, clone_mask)
        if stats.split:
            children = self._split(model, split_mask, generator)
            additions = (
                children
                if not additions
                else {name: torch.cat([additions[name], children[name]]) for name in children}
            )
        if additions:
            append(model, optimizer, additions)

        # A split primitive is replaced by its children, so the parent is
        # removed in the same pass; the mask has to be padded for the rows that
        # were just appended.
        added = next(iter(additions.values())).shape[0] if additions else 0
        remove = torch.cat(
            [split_mask, torch.zeros(added, dtype=torch.bool, device=split_mask.device)]
        )
        remove |= model.opacities.detach() < config.min_opacity
        remove |= model.scales.detach().max(dim=1).values > config.max_world_scale * self.extent
        if iteration >= config.screen_pruning_from:
            padded_radius = torch.cat(
                [
                    self._max_screen_radius,
                    torch.zeros(added, device=self._max_screen_radius.device),
                ]
            )
            remove |= padded_radius > config.max_screen_radius

        if bool(remove.any()) and int((~remove).sum()) > 0:
            stats.pruned = int(remove.sum())
            prune(model, optimizer, ~remove)

        stats.total = len(model)
        self._reset_statistics(model)
        return stats

    def reset_opacity(self, model: GaussianModel, optimizer: torch.optim.Optimizer) -> None:
        """Clamp every opacity down to the configured floor.

        Resetting periodically is what stops the model from parking
        near-opaque primitives right in front of the training cameras: those
        explain a view cheaply, survive because nothing pushes back on them, and
        ruin every other viewpoint.  After a reset they have to be re-inflated by
        the data, and the ones that were only fitting a single view are pruned
        instead.
        """
        floor = self.config.reset_opacity_value
        logit = torch.log(torch.tensor(floor / (1.0 - floor), device=model.means.device))
        clamped = torch.minimum(model.opacity_logits.detach(), logit)
        _swap_parameter(model, optimizer, "opacity_logits", clamped, torch.zeros_like)
