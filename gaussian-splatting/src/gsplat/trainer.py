"""The optimization loop.

Every iteration renders one training view, compares it against the reference and
takes an Adam step.  Three schedules run alongside it and matter as much as the
loss itself.

The position learning rate decays exponentially by three orders of magnitude.
Positions are the only parameters whose useful step size depends on the size of
the scene, so their rate is scaled by the scene extent; keeping it high for too
long lets primitives drift through surfaces they had already found.

Spherical harmonic bands are activated one at a time.  Fitting all sixteen
coefficients from the start lets the model explain a view-dependent effect with
angular colour variation before the geometry that actually causes it exists, and
the resulting local optimum is hard to leave.

Density control runs on its own interval inside a fixed window, with periodic
opacity resets.  It changes the number of parameters, so it is interleaved with
the optimizer rather than layered on top of it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .cameras import Camera
from .datasets import Dataset, View
from .densify import DensityConfig, DensityController
from .gaussians import GaussianModel
from .losses import photometric_loss, psnr, ssim
from .renderer import render

__all__ = ["TrainingConfig", "TrainingHistory", "EvaluationResult", "Trainer"]


@dataclass
class TrainingConfig:
    """Hyperparameters of a training run.

    Attributes
    ----------
    iterations : int
        Total optimization steps.
    position_lr_init, position_lr_final : float
        Learning rate for the primitive centres, before and after the
        exponential decay; both are multiplied by the scene extent.
    sh_dc_lr, sh_rest_lr, opacity_lr, scale_lr, rotation_lr : float
        Constant learning rates for the remaining parameter groups.  The higher
        bands learn twenty times slower than the constant one, which keeps the
        model from explaining texture with angular variation.
    ssim_weight : float
        Weight of the structural term in the photometric loss.
    sh_interval : int
        Iterations between spherical harmonic band activations.
    density : DensityConfig
        Settings of the adaptive density control.
    tile_size, max_per_tile : int
        Rasterization parameters.
    use_checkpoint : bool
        Trade compute for memory inside the rasterizer.
    evaluate_every : int
        Iterations between test-set evaluations; zero disables them.
    log_every : int
        Iterations between progress lines.
    seed : int
        Seed for view shuffling and for splitting.
    device : str
        Torch device.
    """

    iterations: int = 7000
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    sh_dc_lr: float = 2.5e-3
    sh_rest_lr: float = 1.25e-4
    opacity_lr: float = 5.0e-2
    scale_lr: float = 5.0e-3
    rotation_lr: float = 1.0e-3
    ssim_weight: float = 0.2
    sh_interval: int = 1000
    density: DensityConfig = field(default_factory=DensityConfig)
    tile_size: int = 8
    max_per_tile: int = 2048
    use_checkpoint: bool = True
    evaluate_every: int = 1000
    log_every: int = 250
    seed: int = 0
    device: str = "cpu"


@dataclass
class EvaluationResult:
    """Aggregate image quality over a set of views."""

    psnr: float
    ssim: float
    count: int

    def __str__(self) -> str:
        return f"psnr {self.psnr:.2f} dB, ssim {self.ssim:.4f} over {self.count} views"


@dataclass
class TrainingHistory:
    """Per-iteration record of the run, for plots and regression checks."""

    iterations: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    primitives: list[int] = field(default_factory=list)
    test_iterations: list[int] = field(default_factory=list)
    test_psnr: list[float] = field(default_factory=list)
    test_ssim: list[float] = field(default_factory=list)
    seconds: float = 0.0


def _exponential_schedule(step: int, total: int, initial: float, final: float) -> float:
    """Interpolate geometrically between two learning rates."""
    if total <= 1:
        return final
    fraction = min(max(step / (total - 1), 0.0), 1.0)
    return float(np.exp(np.log(initial) * (1.0 - fraction) + np.log(final) * fraction))


class Trainer:
    """Fits a :class:`~gsplat.gaussians.GaussianModel` to a dataset.

    Parameters
    ----------
    model : GaussianModel
    dataset : Dataset
    config : TrainingConfig or None
    """

    def __init__(
        self, model: GaussianModel, dataset: Dataset, config: TrainingConfig | None = None
    ) -> None:
        self.config = config or TrainingConfig()
        self.model = model.to(self.config.device)
        self.dataset = dataset
        self.extent = dataset.extent
        self.background = dataset.background.to(self.config.device)

        self.optimizer = torch.optim.Adam(self._parameter_groups(), eps=1e-15)
        self.density = DensityController(self.model, self.extent, self.config.density)
        self.generator = torch.Generator(device=self.model.means.device)
        self.generator.manual_seed(self.config.seed)
        self._rng = np.random.default_rng(self.config.seed)
        self._warned_about_truncation = False

    def _parameter_groups(self) -> list[dict]:
        config = self.config
        rates = {
            "means": config.position_lr_init * self.extent,
            "log_scales": config.scale_lr,
            "quaternions": config.rotation_lr,
            "opacity_logits": config.opacity_lr,
            "sh_dc": config.sh_dc_lr,
            "sh_rest": config.sh_rest_lr,
        }
        return [
            {"params": [parameter], "lr": rates[name], "name": name}
            for name, parameter in self.model.parameter_dict().items()
        ]

    def _update_position_lr(self, iteration: int) -> None:
        rate = _exponential_schedule(
            iteration,
            self.config.iterations,
            self.config.position_lr_init * self.extent,
            self.config.position_lr_final * self.extent,
        )
        for group in self.optimizer.param_groups:
            if group["name"] == "means":
                group["lr"] = rate

    def render_view(self, camera: Camera, retain_screen_gradients: bool = False):
        """Render the current model from one camera."""
        return render(
            self.model,
            camera,
            background=self.background,
            tile_size=self.config.tile_size,
            max_per_tile=self.config.max_per_tile,
            use_checkpoint=self.config.use_checkpoint,
            retain_screen_gradients=retain_screen_gradients,
        )

    @torch.no_grad()
    def evaluate(self, views: list[View]) -> EvaluationResult:
        """Measure PSNR and SSIM over a set of views."""
        if not views:
            return EvaluationResult(psnr=float("nan"), ssim=float("nan"), count=0)
        peak, structural = [], []
        for view in views:
            prediction = self.render_view(view.camera).image.clamp(0.0, 1.0)
            peak.append(psnr(prediction, view.image))
            structural.append(float(ssim(prediction, view.image)))
        return EvaluationResult(
            psnr=float(np.mean(peak)), ssim=float(np.mean(structural)), count=len(views)
        )

    def train(self, verbose: bool = True) -> TrainingHistory:
        """Run the optimization loop.

        Returns
        -------
        TrainingHistory
        """
        config = self.config
        history = TrainingHistory()
        order: list[int] = []
        started = time.perf_counter()

        for iteration in range(1, config.iterations + 1):
            if not order:
                order = list(self._rng.permutation(len(self.dataset.train)))
            view = self.dataset.train[order.pop()]

            self._update_position_lr(iteration)
            if iteration % config.sh_interval == 0:
                self.model.raise_sh_degree()

            result = self.render_view(view.camera, retain_screen_gradients=True)
            loss, _ = photometric_loss(result.image, view.image, config.ssim_weight)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            in_window = (
                config.density.start_iteration <= iteration <= config.density.stop_iteration
            )
            if in_window:
                self.density.accumulate(self.model, result, view.camera)

            self.optimizer.step()

            if in_window and iteration % config.density.interval == 0:
                stats = self.density.step(self.model, self.optimizer, iteration, self.generator)
                if verbose:
                    print(f"  iter {iteration:6d}  density: {stats}")
            # A reset late in the densification window has no time to recover:
            # every primitive is pushed to near-transparent and the passes that
            # would re-inflate the useful ones never run.
            if (
                config.density.opacity_reset_interval
                and iteration % config.density.opacity_reset_interval == 0
                and iteration < 0.7 * config.density.stop_iteration
            ):
                self.density.reset_opacity(self.model, self.optimizer)

            history.iterations.append(iteration)
            history.loss.append(float(loss.detach()))
            history.primitives.append(len(self.model))

            # A binding per-tile cap shows up as rectangular steps in the render
            # rather than as an error, so it is reported the first time it costs
            # anything measurable.
            if not self._warned_about_truncation and result.output.unspent_transmittance > 0.01:
                self._warned_about_truncation = True
                print(
                    f"  iter {iteration:6d}  warning: {result.output.saturated_tiles} tiles hit "
                    f"the per-tile cap with "
                    f"{result.output.unspent_transmittance:.1%} transmittance unspent; "
                    f"raise max_per_tile or lower tile_size"
                )

            if verbose and iteration % config.log_every == 0:
                recent = float(np.mean(history.loss[-config.log_every :]))
                print(
                    f"  iter {iteration:6d}  loss {recent:.5f}  "
                    f"{len(self.model)} primitives  "
                    f"{iteration / (time.perf_counter() - started):.1f} it/s"
                )
            if config.evaluate_every and iteration % config.evaluate_every == 0:
                evaluation = self.evaluate(self.dataset.test)
                history.test_iterations.append(iteration)
                history.test_psnr.append(evaluation.psnr)
                history.test_ssim.append(evaluation.ssim)
                if verbose:
                    print(f"  iter {iteration:6d}  test {evaluation}")

        history.seconds = time.perf_counter() - started
        return history
