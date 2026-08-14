"""Training and evaluation views, from the procedural scene or from disk.

A dataset carries the calibrated cameras, the reference images, a background
colour and, optionally, a sparse point cloud to initialize from.  The two
loaders differ only in where those come from, so the trainer never learns
whether it is fitting synthetic renders or the output of a real reconstruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .cameras import Camera, orbit_cameras
from .scenes import SceneDefinition, default_scene, render_scene

__all__ = ["View", "Dataset", "synthetic_dataset", "load_dataset", "save_views"]

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass
class View:
    """A camera paired with its reference image.

    Attributes
    ----------
    camera : Camera
    image : Tensor, shape (h, w, 3)
        Linear RGB in ``[0, 1]``.
    """

    camera: Camera
    image: torch.Tensor


@dataclass
class Dataset:
    """A split collection of views.

    Attributes
    ----------
    train, test : list of View
    background : Tensor, shape (3,)
        Colour composited behind the model.
    extent : float
        Radius of the sphere bounding the camera centres, used to scale the
        position learning rate and the density thresholds so that the same
        configuration works for scenes of any size.
    point_cloud : tuple of ndarray or None
        Positions and colours to initialize from, when available.
    """

    train: list[View]
    test: list[View]
    background: torch.Tensor
    extent: float
    point_cloud: tuple[np.ndarray, np.ndarray] | None = None

    def describe(self) -> str:
        camera = self.train[0].camera
        source = "point cloud" if self.point_cloud is not None else "no point cloud"
        return (
            f"{len(self.train)} training and {len(self.test)} test views at "
            f"{camera.width}x{camera.height}, extent {self.extent:.2f}, {source}"
        )


def _camera_extent(cameras: list[Camera]) -> float:
    centres = np.stack([camera.center for camera in cameras])
    return float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max() * 1.1)


def synthetic_dataset(
    num_train: int = 40,
    num_test: int = 8,
    width: int = 160,
    height: int = 120,
    radius: float = 4.6,
    fov_x_degrees: float = 50.0,
    scene: SceneDefinition | None = None,
    device: torch.device | str = "cpu",
) -> Dataset:
    """Ray trace an orbit of training views and an interleaved test orbit.

    The test cameras are offset by half a step along the same orbit, so they are
    novel views in the sense that matters here: nearby viewpoints are seen during
    training, but never these exact ones.

    Parameters
    ----------
    num_train, num_test : int
    width, height : int
        Resolution of the rendered images.
    radius : float
        Orbit radius.
    fov_x_degrees : float
    scene : SceneDefinition or None
    device : torch.device or str

    Returns
    -------
    Dataset
    """
    scene = scene or default_scene()
    target = scene.center

    def build(count: int, phase: float, prefix: str) -> list[View]:
        views = []
        for camera in orbit_cameras(
            count, radius, target, width, height, fov_x_degrees, phase=phase
        ):
            image = torch.as_tensor(render_scene(camera, scene), dtype=torch.float32, device=device)
            camera.name = f"{prefix}_{len(views):03d}"
            views.append(View(camera=camera, image=image))
        return views

    train = build(num_train, 0.0, "train")
    test = build(num_test, 0.5 / max(num_train, 1), "test")
    return Dataset(
        train=train,
        test=test,
        background=torch.as_tensor(scene.background, dtype=torch.float32, device=device),
        extent=_camera_extent([view.camera for view in train]),
    )


def _read_image(path: Path, device: torch.device | str) -> torch.Tensor:
    from PIL import Image

    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.as_tensor(array, device=device)


def _read_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the binary PLY layout written by the structure-from-motion project."""
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    with path.open("rb") as stream:
        count = 0
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[2])
            if line == "end_header":
                break
        vertices = np.frombuffer(stream.read(count * dtype.itemsize), dtype=dtype)
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    colors = (
        np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.float32)
        / 255.0
    )
    return points, colors


def load_dataset(
    directory: str | Path,
    images: str | Path | None = None,
    test_every: int = 8,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: torch.device | str = "cpu",
) -> Dataset:
    """Load a reconstruction produced by the structure-from-motion project.

    The directory is expected to contain ``cameras.json`` and, optionally,
    ``points.ply``.  Images are looked up by the file name recorded for each
    view, first inside ``images`` and then next to the camera file.

    Parameters
    ----------
    directory : str or Path
        Directory holding ``cameras.json``.
    images : str or Path or None
        Where to find the images; defaults to ``directory / "images"`` and then
        to ``directory``.
    test_every : int
        One view out of this many is held out for evaluation.
    background : tuple of float
        Colour composited behind the model.
    device : torch.device or str

    Returns
    -------
    Dataset

    Raises
    ------
    FileNotFoundError
        If the camera file or any referenced image is missing.
    """
    directory = Path(directory)
    camera_file = directory / "cameras.json"
    if not camera_file.exists():
        raise FileNotFoundError(f"no cameras.json in {directory}")
    payload = json.loads(camera_file.read_text(encoding="utf-8"))

    search_paths = [Path(images)] if images else [directory / "images", directory]
    views: list[View] = []
    for entry in payload["views"]:
        candidates = [
            root / name
            for root in search_paths
            for name in (entry["image"], *(Path(entry["image"]).stem + s for s in _IMAGE_SUFFIXES))
        ]
        found = next((path for path in candidates if path.exists()), None)
        if found is None:
            raise FileNotFoundError(f"no image for view {entry['image']!r} under {search_paths}")

        image = _read_image(found, device)
        camera = Camera(
            R=np.array(entry["rotation"]),
            t=np.array(entry["translation"]),
            fx=entry["fx"],
            fy=entry["fy"],
            cx=entry["cx"],
            cy=entry["cy"],
            width=entry["width"],
            height=entry["height"],
            name=entry["image"],
        )
        if (camera.height, camera.width) != tuple(image.shape[:2]):
            factor = image.shape[1] / camera.width
            camera = camera.rescaled(factor)
        views.append(View(camera=camera, image=image))

    if not views:
        raise FileNotFoundError(f"{camera_file} contains no views")

    test = views[::test_every] if test_every > 0 else []
    train = [view for index, view in enumerate(views) if index % max(test_every, 1) != 0]
    if not train:
        train, test = views, []

    point_file = directory / "points.ply"
    cloud = _read_point_cloud(point_file) if point_file.exists() else None
    return Dataset(
        train=train,
        test=test,
        background=torch.as_tensor(background, dtype=torch.float32, device=device),
        extent=_camera_extent([view.camera for view in train]),
        point_cloud=cloud,
    )


def save_views(views: list[View], directory: str | Path) -> Path:
    """Write images and a ``cameras.json`` compatible with the loader.

    This is the bridge that lets the procedural scene be handed to the
    structure-from-motion pipeline, which then produces the poses and sparse
    cloud that :func:`load_dataset` reads back.
    """
    from PIL import Image

    directory = Path(directory)
    image_directory = directory / "images"
    image_directory.mkdir(parents=True, exist_ok=True)

    entries = []
    for index, view in enumerate(views):
        name = f"{view.camera.name or f'view_{index:03d}'}.png"
        array = (view.image.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        Image.fromarray(array).save(image_directory / name)
        entries.append(
            {
                "id": index,
                "image": name,
                "width": view.camera.width,
                "height": view.camera.height,
                "fx": view.camera.fx,
                "fy": view.camera.fy,
                "cx": view.camera.cx,
                "cy": view.camera.cy,
                "rotation": view.camera.R.tolist(),
                "translation": view.camera.t.tolist(),
            }
        )

    payload = {
        "convention": "world-to-camera, x_cam = R @ x_world + t, camera looks along +z",
        "views": entries,
    }
    (directory / "cameras.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return directory
