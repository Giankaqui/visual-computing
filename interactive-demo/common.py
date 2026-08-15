"""Utilidades compartidas por los paneles de la demostración."""

from __future__ import annotations

import numpy as np

__all__ = ["as_display_image", "metric_table", "spherical_position"]


def as_display_image(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen en coma flotante en ``[0, 1]`` a los 8 bits que espera el navegador.

    Los valores se recortan en lugar de reescalarse, de modo que un render que se
    pasa de rango se ve como saturación en vez de normalizarse en silencio.

    Parameters
    ----------
    image : ndarray, shape (h, w) o (h, w, 3)

    Returns
    -------
    ndarray de uint8
    """
    array = np.asarray(image, dtype=float)
    return (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def metric_table(rows: list[tuple[str, str]], title: str | None = None) -> str:
    """Compone pares nombre-valor como una tabla Markdown.

    Parameters
    ----------
    rows : list of tuple
        Pares ``(etiqueta, valor)``; los valores ya vienen formateados.
    title : str o None

    Returns
    -------
    str
    """
    lines = [f"**{title}**", ""] if title else []
    lines += ["| | |", "| --- | ---: |"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    return "\n".join(lines)


def spherical_position(
    target: np.ndarray, radius: float, azimuth_degrees: float, elevation_degrees: float
) -> np.ndarray:
    """Punto sobre una esfera alrededor de ``target``.

    La elevación se mide hacia arriba desde el plano horizontal, lo que en este
    sistema de coordenadas significa hacia ``y`` negativa.

    Parameters
    ----------
    target : ndarray, shape (3,)
    radius : float
    azimuth_degrees, elevation_degrees : float

    Returns
    -------
    ndarray, shape (3,)
    """
    azimuth = np.deg2rad(azimuth_degrees)
    elevation = np.deg2rad(elevation_degrees)
    offset = radius * np.array(
        [
            np.cos(elevation) * np.sin(azimuth),
            -np.sin(elevation),
            np.cos(elevation) * np.cos(azimuth),
        ]
    )
    return np.asarray(target, dtype=float) + offset
