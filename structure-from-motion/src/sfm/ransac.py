"""Generic robust estimator with adaptive stopping and local optimization.

The estimator is model agnostic: callers supply a function that fits a model to a
minimal sample and a function that scores every datum against a model.  Two
refinements over textbook RANSAC are implemented.

MSAC scoring (Torr and Zisserman, 2000) replaces the inlier count with the
truncated squared error ``sum(min(e_i^2, tau^2))``, which discriminates between
hypotheses that share an inlier set but differ in how well they explain it.

Local optimization (Chum et al., 2003) re-fits the model on the current inlier
set whenever the best score improves.  A minimal sample is a poor least-squares
estimate, and one non-minimal refit typically recovers most of the accuracy that
a final global refit would provide, while also enlarging the inlier set and
therefore tightening the adaptive iteration bound.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["RansacResult", "RansacOptions", "ransac"]

ModelFitter = Callable[[np.ndarray], Sequence[np.ndarray] | np.ndarray | None]
ModelScorer = Callable[[np.ndarray], np.ndarray]


@dataclass
class RansacOptions:
    """Configuration for :func:`ransac`.

    Attributes
    ----------
    threshold : float
        Inlier threshold in the units returned by the error function.
    confidence : float
        Target probability that at least one all-inlier sample is drawn.
    max_iterations : int
        Hard cap on hypotheses, applied on top of the adaptive bound.
    min_iterations : int
        Lower bound, so that an optimistic early inlier ratio cannot stop the
        search after one or two hypotheses.
    local_optimization : bool
        Enable the inlier refit described in the module docstring.
    seed : int or None
        Seed for the internal random generator; ``None`` draws from the OS.
    """

    threshold: float
    confidence: float = 0.9999
    max_iterations: int = 5000
    min_iterations: int = 50
    local_optimization: bool = True
    seed: int | None = 0


@dataclass
class RansacResult:
    """Outcome of a robust fit.

    Attributes
    ----------
    model : ndarray or None
        Best model found, or ``None`` if no hypothesis produced enough inliers.
    inliers : ndarray of bool, shape (n,)
        Mask of data indices consistent with ``model``.
    score : float
        MSAC cost of ``model``; lower is better.
    iterations : int
        Number of hypotheses evaluated before the adaptive bound was met.
    """

    model: np.ndarray | None
    inliers: np.ndarray
    score: float
    iterations: int

    @property
    def num_inliers(self) -> int:
        return int(self.inliers.sum())

    @property
    def success(self) -> bool:
        return self.model is not None


def _required_iterations(inlier_ratio: float, sample_size: int, confidence: float) -> float:
    """Number of samples needed to draw an all-inlier subset with probability ``confidence``."""
    if inlier_ratio <= 0.0:
        return np.inf
    all_inlier_probability = inlier_ratio**sample_size
    if all_inlier_probability >= 1.0:
        return 1.0
    denominator = np.log1p(-all_inlier_probability)
    if denominator >= -np.finfo(float).tiny:
        return np.inf
    return np.log1p(-confidence) / denominator


def _as_model_list(models: Sequence[np.ndarray] | np.ndarray | None) -> list[np.ndarray]:
    """Normalize a fitter's return value to a list of candidate models."""
    if models is None:
        return []
    if isinstance(models, np.ndarray):
        return [models]
    return [m for m in models if m is not None]


def ransac(
    num_data: int,
    sample_size: int,
    fit: ModelFitter,
    residuals: ModelScorer,
    options: RansacOptions,
) -> RansacResult:
    """Fit a model robustly to data containing outliers.

    Parameters
    ----------
    num_data : int
        Number of data points; samples are drawn from ``range(num_data)``.
    sample_size : int
        Size of a minimal sample for the model at hand.
    fit : callable
        ``fit(indices) -> model | list[model] | None``.  Multiple candidates are
        allowed because minimal solvers such as the five-point algorithm return
        several roots.
    residuals : callable
        ``residuals(model) -> ndarray of shape (num_data,)`` with non-negative
        errors in the same units as ``options.threshold``.
    options : RansacOptions

    Returns
    -------
    RansacResult
    """
    if num_data < sample_size:
        return RansacResult(None, np.zeros(num_data, dtype=bool), np.inf, 0)

    rng = np.random.default_rng(options.seed)
    tau_squared = options.threshold**2
    indices = np.arange(num_data)

    best_model: np.ndarray | None = None
    best_inliers = np.zeros(num_data, dtype=bool)
    best_score = np.inf

    def evaluate(model: np.ndarray) -> tuple[float, np.ndarray]:
        errors = np.asarray(residuals(model), dtype=float)
        squared = np.where(np.isfinite(errors), errors, np.inf) ** 2
        inliers = squared < tau_squared
        cost = float(np.minimum(squared, tau_squared).sum())
        return cost, inliers

    iteration = 0
    required = float(options.max_iterations)
    while iteration < min(required, options.max_iterations) or iteration < options.min_iterations:
        iteration += 1
        sample = rng.choice(indices, size=sample_size, replace=False)
        for candidate in _as_model_list(fit(sample)):
            score, inliers = evaluate(candidate)
            if score >= best_score:
                continue
            best_model, best_inliers, best_score = candidate, inliers, score

            if options.local_optimization and inliers.sum() > sample_size:
                for refined in _as_model_list(fit(indices[inliers])):
                    refined_score, refined_inliers = evaluate(refined)
                    if refined_score < best_score:
                        best_model = refined
                        best_inliers = refined_inliers
                        best_score = refined_score

            required = _required_iterations(
                best_inliers.mean(), sample_size, options.confidence
            )

        if iteration >= options.max_iterations:
            break

    return RansacResult(best_model, best_inliers, best_score, iteration)
