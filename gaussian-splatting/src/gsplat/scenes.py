"""A procedural scene rendered by ray tracing, used as reproducible training data.

Evaluating a novel-view method needs images whose camera poses are exactly
known, otherwise pose error and reconstruction error are impossible to separate.
Rendering the training views from an analytic scene removes that ambiguity and
keeps the repository free of downloaded datasets.

The scene is deliberately chosen to exercise the parts of the representation
that a simpler one would leave untested.  Specular highlights move across the
spheres as the camera orbits, which only a view-dependent colour model can
reproduce, so the spherical harmonic bands are doing real work rather than
decoration.  The floor carries aperiodic procedural detail near the
image plane, which is what forces the density control to split primitives.  Cast
shadows create sharp radiance discontinuities that are not aligned with any
surface, a common failure case for splatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cameras import Camera

__all__ = ["SurfaceTexture", "Sphere", "SceneDefinition", "default_scene", "render_scene"]


@dataclass
class SurfaceTexture:
    """A band-limited procedural texture defined on all of space.

    The texture is a sparse spectral synthesis: a sum of sinusoids with randomly
    oriented frequency vectors, amplitudes falling geometrically across octaves.
    Being a function of the 3D point rather than of a surface parameterization
    means it can be evaluated wherever a ray happens to land, with no unwrapping
    and no seams.

    Detail matters here beyond appearance.  Untextured surfaces produce almost no
    distinctive keypoints, so a scene of smooth spheres on a flat floor is close
    to unmatchable, and a periodic one is worse: every tile corner shares its
    descriptor with every other, and the ratio test correctly rejects them all.
    Aperiodic variation is what makes the rendered views usable as input to the
    structure-from-motion project in this repository.

    Attributes
    ----------
    frequencies : ndarray, shape (k, 3)
    phases : ndarray, shape (k,)
    amplitudes : ndarray, shape (k,)
    """

    frequencies: np.ndarray
    phases: np.ndarray
    amplitudes: np.ndarray

    @classmethod
    def random(
        cls,
        seed: int = 0,
        octaves: int = 6,
        waves_per_octave: int = 8,
        base_frequency: float = 4.0,
        lacunarity: float = 2.0,
        gain: float = 0.62,
    ) -> SurfaceTexture:
        """Sample a texture with a roughly ``1 / f`` spectrum."""
        rng = np.random.default_rng(seed)
        frequencies, phases, amplitudes = [], [], []
        for octave in range(octaves):
            scale = base_frequency * lacunarity**octave
            directions = rng.normal(size=(waves_per_octave, 3))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            frequencies.append(directions * scale)
            phases.append(rng.uniform(0.0, 2.0 * np.pi, size=waves_per_octave))
            amplitudes.append(np.full(waves_per_octave, gain**octave))
        return cls(
            frequencies=np.concatenate(frequencies),
            phases=np.concatenate(phases),
            amplitudes=np.concatenate(amplitudes),
        )

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the texture in ``[0, 1]`` at each point of shape ``(n, 3)``.

        The sum is normalized by its standard deviation rather than by the sum of
        the amplitudes.  Dividing by the sum would be the worst-case bound and,
        with dozens of independent phases, the field never approaches it: the
        result would concentrate around the mean and the texture would come out
        washed out, with too little contrast to produce keypoints.
        """
        if len(points) == 0:
            return np.zeros(0)
        waves = np.sin(points @ self.frequencies.T + self.phases)
        deviation = np.sqrt(0.5 * float((self.amplitudes**2).sum()))
        return 0.5 + 0.5 * np.clip((waves @ self.amplitudes) / (2.0 * deviation), -1.0, 1.0)


@dataclass
class Sphere:
    """An analytically shaded sphere.

    Attributes
    ----------
    center : ndarray, shape (3,)
    radius : float
    albedo : ndarray, shape (3,)
        Diffuse reflectance in ``[0, 1]``.
    specular : float
        Strength of the Blinn-Phong lobe.
    shininess : float
        Exponent of the lobe; larger values give a tighter highlight.
    """

    center: np.ndarray
    radius: float
    albedo: np.ndarray
    specular: float = 0.35
    shininess: float = 64.0


@dataclass
class SceneDefinition:
    """Everything needed to render the scene.

    Attributes
    ----------
    spheres : list of Sphere
    floor_height : float
        World ``y`` of the ground plane; ``+y`` points down.
    floor_radius : float
        The plane is clipped to this radius so the horizon does not extend to
        infinity, which would leave the model with unbounded geometry to fit.
    floor_colors : tuple of ndarray
        The two colours the floor pattern interpolates between.
    floor_pattern : SurfaceTexture
        Low-frequency field that decides where each floor colour appears.  A
        periodic checkerboard would look tidier and be actively harmful: every
        tile corner has the same descriptor as every other, so the ratio test
        correctly rejects the matches, and a reconstruction from these views
        fails for a reason that has nothing to do with the reconstruction.
    light_direction : ndarray, shape (3,)
        Direction the light travels; normalized on construction.
    light_color : ndarray, shape (3,)
    ambient : ndarray, shape (3,)
    background : ndarray, shape (3,)
    texture : SurfaceTexture
        Procedural detail modulating every albedo.
    texture_strength : float
        Fraction of the albedo the texture is allowed to swing.
    """

    spheres: list[Sphere]
    floor_height: float = 1.0
    floor_radius: float = 6.0
    floor_colors: tuple[np.ndarray, np.ndarray] = (
        np.array([0.78, 0.76, 0.72]),
        np.array([0.30, 0.32, 0.37]),
    )
    floor_pattern: SurfaceTexture = field(
        default_factory=lambda: SurfaceTexture.random(
            seed=17, octaves=3, waves_per_octave=5, base_frequency=1.1, lacunarity=2.1
        )
    )
    light_direction: np.ndarray = field(
        default_factory=lambda: np.array([0.45, 0.85, 0.28])
    )
    light_color: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.97, 0.92]))
    ambient: np.ndarray = field(default_factory=lambda: np.array([0.16, 0.18, 0.22]))
    background: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.06, 0.09]))
    texture: SurfaceTexture = field(default_factory=SurfaceTexture.random)
    texture_strength: float = 0.55

    def __post_init__(self) -> None:
        self.light_direction = np.asarray(self.light_direction, dtype=float)
        self.light_direction /= np.linalg.norm(self.light_direction)

    @property
    def center(self) -> np.ndarray:
        """Centroid of the spheres, a reasonable orbit target."""
        return np.mean([sphere.center for sphere in self.spheres], axis=0)


def default_scene() -> SceneDefinition:
    """Three spheres of different sizes and finishes on a mottled stone floor."""
    return SceneDefinition(
        spheres=[
            Sphere(np.array([-0.85, 0.35, 0.10]), 0.65, np.array([0.85, 0.24, 0.22]), 0.45, 96.0),
            Sphere(np.array([0.75, 0.55, -0.35]), 0.45, np.array([0.20, 0.45, 0.85]), 0.30, 48.0),
            Sphere(np.array([0.25, 0.75, 0.85]), 0.25, np.array([0.95, 0.80, 0.25]), 0.55, 160.0),
        ]
    )


def _sphere_intersections(
    origin: np.ndarray, directions: np.ndarray, sphere: Sphere
) -> np.ndarray:
    """Distance to the nearest intersection, or ``inf`` where the ray misses."""
    offset = origin - sphere.center
    b = directions @ offset
    c = float(offset @ offset) - sphere.radius**2
    discriminant = b * b - c

    distance = np.full(len(directions), np.inf)
    hit = discriminant > 0.0
    if not np.any(hit):
        return distance

    root = np.sqrt(discriminant[hit])
    near = -b[hit] - root
    far = -b[hit] + root
    chosen = np.where(near > 1e-4, near, np.where(far > 1e-4, far, np.inf))
    distance[hit] = chosen
    return distance


def _in_shadow(points: np.ndarray, scene: SceneDefinition) -> np.ndarray:
    """Test each surface point for occlusion along the light direction."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    direction = -scene.light_direction
    occluded = np.zeros(len(points), dtype=bool)
    for sphere in scene.spheres:
        offset = points - sphere.center
        b = offset @ direction
        c = (offset**2).sum(axis=1) - sphere.radius**2
        discriminant = b * b - c
        root = np.sqrt(np.maximum(discriminant, 0.0))
        near, far = -b - root, -b + root
        occluded |= (discriminant > 0.0) & ((near > 1e-3) | (far > 1e-3))
    return occluded


def _shade(
    points: np.ndarray,
    normals: np.ndarray,
    view_directions: np.ndarray,
    albedo: np.ndarray,
    specular: np.ndarray,
    shininess: np.ndarray,
    scene: SceneDefinition,
) -> np.ndarray:
    """Blinn-Phong shading with a single directional light and hard shadows."""
    to_light = -scene.light_direction
    lambert = np.clip(normals @ to_light, 0.0, None)
    lambert = np.where(_in_shadow(points, scene), 0.0, lambert)

    halfway = to_light - view_directions
    halfway /= np.linalg.norm(halfway, axis=1, keepdims=True).clip(1e-9)
    highlight = np.clip((normals * halfway).sum(axis=1), 0.0, None) ** shininess
    highlight = np.where(lambert > 0.0, highlight, 0.0)

    diffuse = albedo * (scene.ambient + lambert[:, None] * scene.light_color)
    return diffuse + (specular * highlight)[:, None] * scene.light_color


def render_scene(camera: Camera, scene: SceneDefinition | None = None) -> np.ndarray:
    """Ray trace one view.

    Parameters
    ----------
    camera : Camera
    scene : SceneDefinition or None

    Returns
    -------
    ndarray of float32, shape (height, width, 3)
        Linear RGB clipped to ``[0, 1]``.
    """
    scene = scene or default_scene()
    height, width = camera.height, camera.width

    column, row = np.meshgrid(np.arange(width), np.arange(height))
    directions_camera = np.stack(
        [
            (column.reshape(-1) + 0.5 - camera.cx) / camera.fx,
            (row.reshape(-1) + 0.5 - camera.cy) / camera.fy,
            np.ones(height * width),
        ],
        axis=1,
    )
    directions = directions_camera @ camera.R
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    origin = camera.center

    distance = np.full(height * width, np.inf)
    surface_id = np.full(height * width, -1, dtype=int)
    for index, sphere in enumerate(scene.spheres):
        candidate = _sphere_intersections(origin, directions, sphere)
        closer = candidate < distance
        distance[closer] = candidate[closer]
        surface_id[closer] = index

    with np.errstate(divide="ignore", invalid="ignore"):
        floor_distance = (scene.floor_height - origin[1]) / directions[:, 1]
    floor_point = origin + floor_distance[:, None] * directions
    floor_hit = (
        (floor_distance > 1e-4)
        & np.isfinite(floor_distance)
        & (np.linalg.norm(floor_point[:, [0, 2]], axis=1) < scene.floor_radius)
        & (floor_distance < distance)
    )
    distance[floor_hit] = floor_distance[floor_hit]
    surface_id[floor_hit] = len(scene.spheres)

    image = np.tile(scene.background, (height * width, 1))
    hit = surface_id >= 0
    if np.any(hit):
        points = origin + distance[hit, None] * directions[hit]
        identifiers = surface_id[hit]

        normals = np.zeros_like(points)
        albedo = np.zeros_like(points)
        specular = np.zeros(len(points))
        shininess = np.full(len(points), 32.0)

        for index, sphere in enumerate(scene.spheres):
            selected = identifiers == index
            if not np.any(selected):
                continue
            normal = points[selected] - sphere.center
            normals[selected] = normal / np.linalg.norm(normal, axis=1, keepdims=True)
            albedo[selected] = sphere.albedo
            specular[selected] = sphere.specular
            shininess[selected] = sphere.shininess

        on_floor = identifiers == len(scene.spheres)
        if np.any(on_floor):
            normals[on_floor] = np.array([0.0, -1.0, 0.0])
            pattern = np.clip(3.0 * (scene.floor_pattern(points[on_floor]) - 0.5) + 0.5, 0.0, 1.0)
            albedo[on_floor] = (
                scene.floor_colors[0] * pattern[:, None]
                + scene.floor_colors[1] * (1.0 - pattern[:, None])
            )
            specular[on_floor] = 0.05
            shininess[on_floor] = 16.0

        modulation = 1.0 - scene.texture_strength + 2.0 * scene.texture_strength * scene.texture(
            points
        )
        albedo = albedo * modulation[:, None]

        image[hit] = _shade(
            points, normals, directions[hit], albedo, specular, shininess, scene
        )

    return np.clip(image, 0.0, 1.0).reshape(height, width, 3).astype(np.float32)
