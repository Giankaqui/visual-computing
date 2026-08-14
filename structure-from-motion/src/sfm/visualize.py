"""Figures summarizing a reconstruction.

Rendering uses the Agg backend so the module works over SSH and inside CI.

Matplotlib draws its third axis as the vertical one, while the camera convention
used here has ``+y`` pointing down in the image and therefore roughly downwards
in the world.  Points are mapped to ``(x, z, -y)`` before plotting so that the
default viewpoint shows the scene the right way up.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .camera import PinholeCamera, Pose  # noqa: E402

__all__ = ["plot_reconstruction"]

_STRUCTURE_COLOR = "#0f4c81"
_CAMERA_COLOR = "#d1495b"


def _to_display_frame(points: np.ndarray) -> np.ndarray:
    """Map world coordinates to the plotting frame."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)


def _frustum_polyline(pose: Pose, camera: PinholeCamera, scale: float) -> np.ndarray:
    """Return a single polyline through the frustum corners and the camera centre.

    Drawing one polyline per camera instead of eight separate segments keeps the
    number of Matplotlib artists proportional to the number of views, which
    matters for scenes with hundreds of cameras.
    """
    corners_pixel = np.array(
        [[0.0, 0.0], [camera.width, 0.0], [camera.width, camera.height], [0.0, camera.height]]
    )
    rays = np.hstack([camera.normalize(corners_pixel), np.ones((4, 1))]) * scale
    corners = rays @ pose.R + pose.center
    order = [4, 0, 1, 4, 2, 3, 4, 0, 3, 2, 1]
    vertices = np.vstack([corners, pose.center])
    return _to_display_frame(vertices[order])


def _equalize_axes(axes, points: np.ndarray) -> None:
    """Give the three axes a common scale, which Matplotlib does not do for 3D."""
    span = float(np.max(points.max(axis=0) - points.min(axis=0))) * 0.55
    centre = 0.5 * (points.max(axis=0) + points.min(axis=0))
    axes.set_xlim(centre[0] - span, centre[0] + span)
    axes.set_ylim(centre[1] - span, centre[1] + span)
    axes.set_zlim(centre[2] - span, centre[2] + span)
    axes.set_box_aspect((1.0, 1.0, 1.0))


def plot_reconstruction(
    reconstruction,
    path: str | Path,
    title: str | None = None,
    max_points: int = 20000,
    elevation: float = 22.0,
    azimuth: float = -63.0,
) -> Path:
    """Render the point cloud, the camera trajectory and the error distribution.

    Parameters
    ----------
    reconstruction : sfm.reconstruction.Reconstruction
    path : str or Path
        Destination PNG.
    title : str or None
        Figure title; defaults to the reconstruction summary.
    max_points : int
        Points are subsampled above this count to keep the figure light.
    elevation, azimuth : float
        Viewing angles in degrees, in the display frame described in the module
        docstring.

    Returns
    -------
    Path
        The path that was written.

    Raises
    ------
    ValueError
        If the reconstruction contains no points.
    """
    points, colors = reconstruction.point_cloud()
    if len(points) == 0:
        raise ValueError("nothing to plot: the reconstruction has no points")
    if len(points) > max_points:
        selection = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[selection], colors[selection]

    display_points = _to_display_frame(points)
    if colors.std(axis=0).max() < 1e-6:
        # No photometric information is available, as in the synthetic demo.
        # Height gives the eye something to latch onto instead of a flat blob.
        height = display_points[:, 2]
        point_colors = plt.get_cmap("viridis")(
            (height - height.min()) / max(np.ptp(height), 1e-9)
        )
    else:
        point_colors = colors / 255.0

    figure = plt.figure(figsize=(13.5, 5.6))
    axes3d = figure.add_subplot(1, 2, 1, projection="3d")
    axes3d.scatter(
        display_points[:, 0],
        display_points[:, 1],
        display_points[:, 2],
        c=point_colors,
        s=2.5,
        linewidths=0,
        alpha=0.9,
        depthshade=False,
    )

    centers = _to_display_frame(reconstruction.camera_centers())
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    for view in reconstruction.registration_order:
        polyline = _frustum_polyline(
            reconstruction.poses[view], reconstruction.cameras[view], 0.045 * extent
        )
        axes3d.plot(polyline[:, 0], polyline[:, 1], polyline[:, 2],
                    color=_CAMERA_COLOR, linewidth=0.9)
    axes3d.plot(centers[:, 0], centers[:, 1], centers[:, 2],
                color=_CAMERA_COLOR, linewidth=1.4, alpha=0.55)

    _equalize_axes(axes3d, np.vstack([display_points, centers]))
    axes3d.view_init(elev=elevation, azim=azimuth)
    axes3d.set_xlabel("x")
    axes3d.set_ylabel("z")
    axes3d.set_zlabel("-y")
    axes3d.set_title(f"{len(reconstruction.points)} points, "
                     f"{reconstruction.num_registered} cameras")

    errors = reconstruction.reprojection_errors()
    axes2d = figure.add_subplot(1, 2, 2)
    axes2d.hist(errors, bins=60, color=_STRUCTURE_COLOR, alpha=0.85)
    axes2d.axvline(float(np.median(errors)), color=_CAMERA_COLOR, linestyle="--",
                   label=f"median {np.median(errors):.2f} px")
    axes2d.set_xlabel("reprojection error (px)")
    axes2d.set_ylabel("observations")
    axes2d.set_title("reprojection error distribution")
    axes2d.legend()
    axes2d.spines[["top", "right"]].set_visible(False)

    figure.suptitle(title or reconstruction.summary())
    figure.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
