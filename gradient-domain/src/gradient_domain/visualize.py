"""Figuras para las demostraciones y para el benchmark de solvers.

El dibujado usa el backend Agg para que el módulo funcione sin pantalla.
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
    """Coloca imágenes etiquetadas en una rejilla.

    Parameters
    ----------
    panels : list of tuple
        Pares ``(título, imagen)``; las imágenes pueden ser en gris o RGB.
    path : str o Path
    columns : int o None
        Ancho de la rejilla; por defecto el número de paneles, con tope en cuatro.

    Returns
    -------
    Path
    """
    if not panels:
        raise ValueError("hace falta al menos un panel")
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
    """Dibuja tiempo y número de iteraciones frente al número de incógnitas.

    Se trazan pendientes de referencia para crecimiento lineal y cuadrático, de
    modo que las curvas medidas se puedan leer directamente contra ellas.

    Parameters
    ----------
    results : dict
        Asocia cada nombre de solver con una lista de registros con claves
        ``unknowns``, ``seconds`` e ``iterations``.
    path : str o Path

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
    axes[0].loglog(reference, anchor * reference, "--", color="0.5", linewidth=1, label="lineal")
    axes[0].loglog(
        reference,
        anchor * reference[0] * (reference / reference[0]) ** 1.5,
        ":",
        color="0.5",
        linewidth=1,
        label="N^1.5",
    )

    axes[0].set_xlabel("incógnitas")
    axes[0].set_ylabel("segundos")
    axes[0].set_title("tiempo hasta un residuo relativo de 1e-8")
    axes[0].legend()
    axes[1].set_xlabel("incógnitas")
    axes[1].set_ylabel("iteraciones")
    axes[1].set_title("iteraciones hasta converger")
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
    """Compara una imagen con mapeo tonal contra exposiciones fijas de la misma radiancia.

    Se muestran dos exposiciones porque esa es la comparación honesta: cualquier
    curva global tiene que elegir entre el interior y la ventana, y enseñar las
    dos hace visible esa elección.

    Parameters
    ----------
    radiance : ndarray, shape (h, w, 3)
    mapped : ndarray, shape (h, w, 3)
    factor : ndarray, shape (h, w)
        Campo de atenuación, mostrado en escala logarítmica.
    path : str o Path
    exposures : tuple of float
        Multiplicadores aplicados antes de la curva gamma, relativos a una
        normalización en el percentil 99.5.  Los valores por defecto acotan la
        elección: uno conserva las altas luces y pierde el interior, el otro hace
        lo contrario.

    Returns
    -------
    Path
    """
    reference = float(np.percentile(radiance, 99.5))
    panels = [
        (
            f"exposición {stop:g}x, gamma 2.2",
            np.clip(stop * radiance / reference, 0, 1) ** (1 / 2.2),
        )
        for stop in exposures
    ]

    figure, axes = plt.subplots(1, len(panels) + 2, figsize=(4.6 * (len(panels) + 2), 4.0))
    for axis, (title, image) in zip(axes, panels, strict=False):
        _show(axis, image, title)
    _show(axes[len(panels)], mapped, "mapeo tonal en el dominio del gradiente")

    image = axes[-1].imshow(np.log10(np.maximum(factor, 1e-6)), cmap="magma")
    axes[-1].set_title("log10 del factor de atenuación", fontsize=11)
    axes[-1].set_axis_off()
    figure.colorbar(image, ax=axes[-1], fraction=0.046)

    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return path
