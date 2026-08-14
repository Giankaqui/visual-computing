"""Helpers shared by the demonstration panels."""

from __future__ import annotations

import numpy as np

__all__ = ["as_display_image", "metric_table", "DISPLAY_FRAME_NOTE"]

DISPLAY_FRAME_NOTE = (
    "Plotted as `(x, z, -y)`: the camera convention has `+y` pointing down in "
    "the image, so the raw axes would show the scene upside down."
)


def as_display_image(image: np.ndarray) -> np.ndarray:
    """Convert a float image in ``[0, 1]`` to the 8-bit array a browser expects.

    Values are clipped rather than rescaled, so a render that overshoots is
    visible as clipping instead of being silently normalized away.

    Parameters
    ----------
    image : ndarray, shape (h, w) or (h, w, 3)

    Returns
    -------
    ndarray of uint8
    """
    array = np.asarray(image, dtype=float)
    return (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def metric_table(rows: list[tuple[str, str]], title: str | None = None) -> str:
    """Render name-value pairs as a Markdown table.

    Parameters
    ----------
    rows : list of tuple
        ``(label, value)`` pairs; values are already formatted.
    title : str or None

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
    """Point on a sphere around ``target``.

    Elevation is measured upwards from the horizontal plane, which in this
    coordinate system means towards negative ``y``.

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
