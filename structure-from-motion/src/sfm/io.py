"""Serialization of reconstructions.

Two artefacts are written.  A binary PLY holds the coloured point cloud and
opens in MeshLab, CloudCompare or Blender without conversion.  A JSON sidecar
holds the calibrated poses, which is what a downstream renderer needs; the
schema is deliberately flat and self-describing rather than a re-implementation
of the COLMAP binary format.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .camera import PinholeCamera, Pose

__all__ = ["write_ply", "read_ply", "write_cameras", "read_cameras", "export_reconstruction"]

_PLY_HEADER = """ply
format binary_little_endian 1.0
element vertex {count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""

_VERTEX_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
)


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write a coloured point cloud as binary little-endian PLY.

    Parameters
    ----------
    path : str or Path
    points : ndarray, shape (n, 3)
    colors : ndarray of uint8, shape (n, 3), optional
        Defaults to mid grey.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if colors is None:
        colors = np.full((len(points), 3), 128, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(colors) != len(points):
        raise ValueError("points and colors must have the same length")

    vertices = np.empty(len(points), dtype=_VERTEX_DTYPE)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(_PLY_HEADER.format(count=len(points)).encode("ascii"))
        stream.write(vertices.tobytes())


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a point cloud written by :func:`write_ply`.

    Only the binary little-endian layout produced by this module is supported,
    which keeps the reader short and makes malformed input fail loudly.

    Returns
    -------
    points : ndarray, shape (n, 3)
    colors : ndarray of uint8, shape (n, 3)
    """
    with Path(path).open("rb") as stream:
        header_lines: list[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("truncated PLY header")
            header_lines.append(line.decode("ascii").strip())
            if header_lines[-1] == "end_header":
                break
        if "format binary_little_endian 1.0" not in header_lines:
            raise ValueError("only binary little-endian PLY files are supported")
        count = next(
            int(line.split()[2]) for line in header_lines if line.startswith("element vertex")
        )
        vertices = np.frombuffer(stream.read(count * _VERTEX_DTYPE.itemsize), dtype=_VERTEX_DTYPE)

    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(float)
    colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1)
    return points, colors


def write_cameras(
    path: str | Path,
    cameras: list[PinholeCamera],
    poses: dict[int, Pose],
    image_names: list[str] | None = None,
) -> None:
    """Write intrinsics and world-to-camera poses of the registered views.

    Parameters
    ----------
    path : str or Path
    cameras : list of PinholeCamera
        Intrinsics indexed by view.
    poses : dict
        Registered views mapped to their pose; unregistered views are skipped.
    image_names : list of str or None
        File names indexed by view, stored for traceability.
    """
    entries = []
    for view in sorted(poses):
        camera, pose = cameras[view], poses[view]
        entries.append(
            {
                "id": int(view),
                "image": image_names[view] if image_names else f"view_{view:04d}",
                "width": int(camera.width),
                "height": int(camera.height),
                "fx": float(camera.fx),
                "fy": float(camera.fy),
                "cx": float(camera.cx),
                "cy": float(camera.cy),
                "rotation": pose.R.tolist(),
                "translation": pose.t.tolist(),
            }
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "convention": "world-to-camera, x_cam = R @ x_world + t, camera looks along +z",
        "views": entries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_cameras(path: str | Path) -> tuple[list[PinholeCamera], list[Pose], list[str]]:
    """Read a camera file written by :func:`write_cameras`.

    Returns
    -------
    cameras : list of PinholeCamera
    poses : list of Pose
    names : list of str
        All three lists are ordered by the stored view id.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cameras: list[PinholeCamera] = []
    poses: list[Pose] = []
    names: list[str] = []
    for entry in payload["views"]:
        cameras.append(
            PinholeCamera(
                fx=entry["fx"],
                fy=entry["fy"],
                cx=entry["cx"],
                cy=entry["cy"],
                width=entry["width"],
                height=entry["height"],
            )
        )
        poses.append(Pose(R=np.array(entry["rotation"]), t=np.array(entry["translation"])))
        names.append(entry["image"])
    return cameras, poses, names


def export_reconstruction(
    directory: str | Path, reconstruction, image_names: list[str] | None = None
) -> dict[str, Path]:
    """Write the point cloud and cameras of a reconstruction into a directory.

    Parameters
    ----------
    directory : str or Path
    reconstruction : sfm.reconstruction.Reconstruction
    image_names : list of str or None

    Returns
    -------
    dict
        Maps ``"points"`` and ``"cameras"`` to the paths that were written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    points, colors = reconstruction.point_cloud()

    point_path = directory / "points.ply"
    camera_path = directory / "cameras.json"
    write_ply(point_path, points, colors)
    write_cameras(camera_path, reconstruction.cameras, reconstruction.poses, image_names)
    return {"points": point_path, "cameras": camera_path}
