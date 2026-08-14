"""Figures for the demonstrations and the solver benchmark.

Rendering uses the Agg backend so the module works headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["panel_figure", "solver_scaling_figure", "tone_mapping_figure"]

_MARKERS = ("o", "s", "^", "D")


def _show(axis, image: np.ndarray, title: str) -> None:
    if image.ndim == 3:
        axis.imshow(np.clip(image, 0.0, 1.0))
    else:
        axis.imshow(image, cmap="gray")
    axis.set_title(title, fontsize=11)
    axis.set_axis_off()


def panel_figure(
    panels: list[tuple[str, np.ndarray]], path: str | Path, columns: int | None = None
) -> Path:
    """Lay out labelled images in a grid.

    Parameters
    ----------
    panels : list of tuple
        ``(title, image)`` pairs; images may be grayscale or RGB.
    path : str or Path
    columns : int or None
        Grid width; defaults to the number of panels, capped at four.

    Returns
    -------
    Path
    """
    if not panels:
        raise ValueError("at least one panel is required")
    columns = columns or min(len(panels), 4)
    rows = (len(panels) + columns - 1) // columns

    height, width = panels[0][1].shape[:2]
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 4.2 * rows * height / width), squeeze=False
    )
    for index, (title, image) in enumerate(panels):
        _show(axes[index // columns][index % columns], image, title)
    for index in range(len(panels), rows * columns):
        axes[index // columns][index % columns].set_axis_off()

    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path


def solver_scaling_figure(results: dict[str, list[dict]], path: str | Path) -> Path:
    """Plot solver time and iteration count against the number of unknowns.

    Reference slopes for linear and quadratic growth are drawn so the measured
    curves can be read against them directly.

    Parameters
    ----------
    results : dict
        Maps a solver name to a list of records with keys ``unknowns``,
        ``seconds`` and ``iterations``.
    path : str or Path

    Returns
    -------
    Path
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for index, (name, records) in enumerate(sorted(results.items())):
        unknowns = np.array([record["unknowns"] for record in records], dtype=float)
        seconds = np.array([record["seconds"] for record in records], dtype=float)
        iterations = np.array([record["iterations"] for record in records], dtype=float)
        marker = _MARKERS[index % len(_MARKERS)]
        axes[0].loglog(unknowns, seconds, marker=marker, label=name)
        if iterations.max() > 0:
            axes[1].semilogx(unknowns, iterations, marker=marker, label=name)

    reference = np.array(
        [record["unknowns"] for record in next(iter(results.values()))], dtype=float
    )
    anchor = min(
        records[0]["seconds"] for records in results.values() if records
    ) / reference[0]
    axes[0].loglog(reference, anchor * reference, "--", color="0.5", linewidth=1, label="linear")
    axes[0].loglog(
        reference,
        anchor * reference[0] * (reference / reference[0]) ** 1.5,
        ":",
        color="0.5",
        linewidth=1,
        label="N^1.5",
    )

    axes[0].set_xlabel("unknowns")
    axes[0].set_ylabel("seconds")
    axes[0].set_title("time to a relative residual of 1e-8")
    axes[0].legend()
    axes[1].set_xlabel("unknowns")
    axes[1].set_ylabel("iterations")
    axes[1].set_title("iterations to converge")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def tone_mapping_figure(
    radiance: np.ndarray,
    mapped: np.ndarray,
    factor: np.ndarray,
    path: str | Path,
    exposures: tuple[float, ...] = (1.0, 60.0),
) -> Path:
    """Compare a tone-mapped image against fixed exposures of the same radiance.

    Two exposures are shown because that is the honest baseline: any single
    global curve has to choose between the interior and the window, and showing
    both makes the choice visible.

    Parameters
    ----------
    radiance : ndarray, shape (h, w, 3)
    mapped : ndarray, shape (h, w, 3)
    factor : ndarray, shape (h, w)
        Attenuation field, displayed on a logarithmic scale.
    path : str or Path
    exposures : tuple of float
        Multipliers applied before the gamma curve, relative to a normalization
        at the 99.5th percentile.  The defaults bracket the choice: one holds the
        highlights and loses the interior, the other does the opposite.

    Returns
    -------
    Path
    """
    reference = float(np.percentile(radiance, 99.5))
    panels = [
        (f"exposure {stop:g}x, gamma 2.2", np.clip(stop * radiance / reference, 0, 1) ** (1 / 2.2))
        for stop in exposures
    ]

    figure, axes = plt.subplots(1, len(panels) + 2, figsize=(4.6 * (len(panels) + 2), 4.0))
    for axis, (title, image) in zip(axes, panels, strict=False):
        _show(axis, image, title)
    _show(axes[len(panels)], mapped, "gradient-domain tone map")

    image = axes[-1].imshow(np.log10(np.maximum(factor, 1e-6)), cmap="magma")
    axes[-1].set_title("log10 attenuation factor", fontsize=11)
    axes[-1].set_axis_off()
    figure.colorbar(image, ax=axes[-1], fraction=0.046)

    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path
