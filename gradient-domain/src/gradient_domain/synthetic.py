"""Imágenes de prueba procedurales.

Los métodos de dominio del gradiente se juzgan mejor sobre imágenes con las
propiedades para las que están pensados: un destino con iluminación que varía
despacio, una fuente iluminada de otra manera, textura separable de la
estructura, y un mapa de radiancia cuyo rango dinámico supere de verdad al de una
pantalla.  Generarlas proceduralmente mantiene las demostraciones reproducibles y
el repositorio libre de ficheros binarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "value_noise",
    "CompositingExample",
    "make_compositing_example",
    "make_texture_example",
    "make_radiance_map",
]


def _bilinear_resize(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.linspace(0, grid.shape[0] - 1, shape[0])
    columns = np.linspace(0, grid.shape[1] - 1, shape[1])
    row0, col0 = np.floor(rows).astype(int), np.floor(columns).astype(int)
    row1 = np.minimum(row0 + 1, grid.shape[0] - 1)
    col1 = np.minimum(col0 + 1, grid.shape[1] - 1)
    row_weight, col_weight = (rows - row0)[:, None], (columns - col0)[None, :]

    top = grid[row0][:, col0] * (1 - col_weight) + grid[row0][:, col1] * col_weight
    bottom = grid[row1][:, col0] * (1 - col_weight) + grid[row1][:, col1] * col_weight
    return top * (1 - row_weight) + bottom * row_weight


def value_noise(
    shape: tuple[int, int],
    rng: np.random.Generator,
    octaves: int = 5,
    base: int = 4,
    persistence: float = 0.5,
) -> np.ndarray:
    """Ruido de valor fractal en ``[0, 1]``.

    Cada octava es una malla aleatoria al doble de frecuencia que la anterior,
    interpolada hacia arriba y sumada con la mitad de amplitud, lo que da el
    espectro aproximadamente ``1 / f`` que tiene la textura natural.

    Parameters
    ----------
    shape : tuple of int
    rng : numpy.random.Generator
    octaves : int
        Número de bandas de frecuencia.
    base : int
        Resolución de la banda más gruesa.
    persistence : float
        Razón de amplitud entre bandas consecutivas.

    Returns
    -------
    ndarray, shape (h, w)
    """
    total = np.zeros(shape)
    amplitude, normalization = 1.0, 0.0
    for octave in range(octaves):
        resolution = base * 2**octave
        grid = rng.random((min(resolution, shape[0]), min(resolution, shape[1])))
        total += amplitude * _bilinear_resize(grid, shape)
        normalization += amplitude
        amplitude *= persistence
    return total / normalization


@dataclass
class CompositingExample:
    """Una imagen de destino, un recorte de fuente y la selección a transferir.

    Attributes
    ----------
    target : ndarray, shape (h, w, 3)
    source : ndarray, shape (sh, sw, 3)
    mask : ndarray of bool, shape (sh, sw)
    offset : tuple of int
        Dónde cae la esquina superior izquierda del recorte en el destino.
    """

    target: np.ndarray
    source: np.ndarray
    mask: np.ndarray
    offset: tuple[int, int]

    def naive_composite(self) -> np.ndarray:
        """Copia los píxeles seleccionados tal cual en el destino, con costura y todo."""
        row, column = self.offset
        height, width = self.source.shape[:2]
        result = self.target.copy()
        window = result[row : row + height, column : column + width]
        result[row : row + height, column : column + width] = np.where(
            self.mask[..., None], self.source, window
        )
        return result


def _sky(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Un degradado de atardecer con bandas suaves de nubes."""
    height, width = shape
    vertical = np.linspace(0.0, 1.0, height)[:, None]
    zenith = np.array([0.16, 0.28, 0.52])
    horizon = np.array([0.94, 0.72, 0.46])
    sky = zenith + (horizon - zenith) * (vertical[..., None] ** 2.2)

    clouds = value_noise(shape, rng, octaves=5, base=3)
    banding = np.clip((clouds - 0.45) * 2.6, 0.0, 1.0) * (1.0 - vertical) ** 0.5
    return sky * np.ones((height, width, 1)) + banding[..., None] * np.array(
        [0.22, 0.18, 0.12]
    )


def make_compositing_example(
    shape: tuple[int, int] = (384, 576), seed: int = 0
) -> CompositingExample:
    """Construye un paisaje de atardecer y un globo fotografiado bajo un cielo más claro.

    Las dos imágenes se diferencian en un desplazamiento global de iluminación y
    una dominante de color, que es exactamente la situación en la que copiar
    píxeles falla y copiar gradientes funciona: el desplazamiento vive por
    completo en la condición de contorno y por tanto lo sustituye el propio
    desplazamiento del destino.
    """
    rng = np.random.default_rng(seed)
    height, width = shape

    target = _sky(shape, rng)
    horizon = int(0.62 * height)
    rows, _ = np.mgrid[0:height, 0:width]

    ridge = 0.5 + 0.5 * value_noise((1, width), rng, octaves=4, base=3)[0]
    hills = rows >= (horizon - (18 + 26 * ridge).astype(int))[None, :]
    haze = np.linspace(0.0, 1.0, height)[:, None, None]
    hill_color = np.broadcast_to(
        np.array([0.20, 0.22, 0.30]) * (1.0 - 0.5 * haze)
        + np.array([0.30, 0.26, 0.28]) * haze,
        (height, width, 3),
    )
    target[hills] = hill_color[hills]

    texture = value_noise(shape, rng, octaves=6, base=6)
    ground_color = np.array([0.30, 0.26, 0.18]) + 0.28 * texture[..., None] * np.array(
        [0.6, 0.5, 0.35]
    )
    depth = np.clip((rows - horizon) / max(height - horizon, 1), 0.0, 1.0)
    ground = rows >= horizon
    target[ground] = (ground_color * (0.55 + 0.45 * depth[..., None]))[ground]

    patch_size = max(int(0.30 * min(shape)), 24)
    grid = np.linspace(-1.0, 1.0, patch_size)
    px, py = np.meshgrid(grid, grid)
    radius = np.sqrt(px**2 + py**2)

    # La selección es un disco bastante mayor que el globo, así que la costura
    # pasa por el propio cielo de la fuente y no por encima del objeto.  Esa es
    # la condición bajo la que la composición en el dominio del gradiente se
    # porta bien: la frontera tiene que caer donde fuente y destino sean
    # verosímilmente parecidos, porque todo lo que difiere entre ambos se absorbe
    # ahí.
    balloon_radius = 0.58
    mask = radius <= 0.94

    # El globo está iluminado desde arriba a la izquierda y pintado con franjas
    # radiales, así que lleva a la vez una rampa suave de sombreado y bordes
    # cromáticos duros.
    stripes = (np.floor((np.arctan2(py, px) + np.pi) / (np.pi / 5.0)).astype(int)) % 2
    base_colors = np.where(
        stripes[..., None] == 0, np.array([0.90, 0.24, 0.20]), np.array([0.96, 0.86, 0.30])
    )
    normalized = np.clip(radius / balloon_radius, 0.0, 1.0)
    normal_z = np.sqrt(np.clip(1.0 - normalized**2, 0.0, 1.0))
    shading = 0.35 + 0.65 * np.clip(
        (-0.5 * px / balloon_radius - 0.5 * py / balloon_radius + 0.7 * normal_z) / 1.2, 0.0, 1.0
    )
    source = base_colors * shading[..., None]

    # El recorte se "fotografió" contra un cielo de mediodía brillante, un paso
    # más luminoso y mucho más frío que el destino.
    source = source * 1.9 * np.array([0.92, 0.96, 1.06])
    background = np.array([0.62, 0.78, 0.98]) * (
        0.85 + 0.15 * value_noise((patch_size, patch_size), rng, octaves=3, base=2)[..., None]
    )
    source = np.where(radius[..., None] <= balloon_radius, source, background)

    offset = (int(0.14 * height), int(0.16 * width))
    return CompositingExample(
        target=np.clip(target, 0.0, 1.0),
        source=np.clip(source, 0.0, 1.0),
        mask=mask,
        offset=offset,
    )


def make_texture_example(
    shape: tuple[int, int] = (320, 320), seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Una forma con textura marcada, y el mapa de bordes que define su estructura.

    Returns
    -------
    image : ndarray, shape (h, w, 3)
    edges : ndarray of float, shape (h, w)
        Uno en los bordes que deben sobrevivir al aplanado, cero en el resto.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width]

    grain = value_noise(shape, rng, octaves=6, base=8)
    fine = 0.5 + 0.5 * np.sin(rows * 0.55) * np.sin(columns * 0.47)
    texture = 0.6 * grain + 0.4 * fine

    centre = np.array([height * 0.5, width * 0.5])
    radius = np.sqrt((rows - centre[0]) ** 2 + (columns - centre[1]) ** 2)
    disc = radius < 0.32 * min(height, width)
    band = np.abs(columns - width * 0.5) < 0.09 * width

    image = np.empty((height, width, 3))
    image[...] = (np.array([0.24, 0.30, 0.42]) * (0.7 + 0.6 * texture[..., None]))
    image[disc] = (np.array([0.86, 0.52, 0.22]) * (0.7 + 0.6 * texture[..., None]))[disc]
    image[disc & band] = (
        np.array([0.30, 0.62, 0.48]) * (0.7 + 0.6 * texture[..., None])
    )[disc & band]

    # Los bordes estructurales son las fronteras entre las tres regiones,
    # dilatadas un píxel para preservar el gradiente que cruza cada frontera.
    labels = disc.astype(int) + (disc & band).astype(int)
    padded = np.pad(labels, 1, mode="edge")
    edges = np.zeros(shape)
    for shift in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rolled = padded[
            1 + shift[0] : shape[0] + 1 + shift[0], 1 + shift[1] : shape[1] + 1 + shift[1]
        ]
        edges = np.maximum(edges, (rolled != labels).astype(float))
    return np.clip(image, 0.0, 1.0), edges


def make_radiance_map(
    shape: tuple[int, int] = (320, 448), seed: int = 2
) -> np.ndarray:
    """Un interior con una ventana luminosa, con unos cinco órdenes de magnitud.

    La ventana es cuatro décadas más brillante que la pared en la que está
    encajada, y la pared misma lleva una textura invisible bajo cualquier curva
    tonal global que evite que la ventana sature.

    Returns
    -------
    ndarray, shape (h, w, 3)
        Radiancia lineal, estrictamente positiva.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    rows, columns = np.mgrid[0:height, 0:width]

    plaster = 0.6 + 0.8 * value_noise(shape, rng, octaves=6, base=5)
    radiance = 0.9 * plaster[..., None] * np.array([0.55, 0.50, 0.44])

    # Un suelo de madera en el quinto inferior, todavía más oscuro.
    floor = rows > 0.80 * height
    planks = 0.5 + 0.5 * np.sin(columns * 0.35 + 3.0 * value_noise(shape, rng, octaves=3, base=4))
    radiance[floor] = (0.25 * planks[..., None] * np.array([0.42, 0.28, 0.18]))[floor]

    # Una ventana enmarcada hacia un exterior soleado.
    window = (
        (columns > 0.52 * width)
        & (columns < 0.88 * width)
        & (rows > 0.14 * height)
        & (rows < 0.62 * height)
    )
    frame = (
        (columns > 0.50 * width)
        & (columns < 0.90 * width)
        & (rows > 0.12 * height)
        & (rows < 0.64 * height)
    ) & ~window
    radiance[frame] = (0.30 * np.array([0.30, 0.22, 0.16]))[None, :]

    exterior_sky = np.array([0.55, 0.72, 1.00]) * 9.0e3
    exterior_ground = np.array([0.72, 0.78, 0.42]) * 2.2e3
    skyline = rows < 0.44 * height
    outside = np.where(
        skyline[..., None],
        exterior_sky * (0.75 + 0.5 * value_noise(shape, rng, octaves=4, base=3)[..., None]),
        exterior_ground * (0.6 + 0.8 * value_noise(shape, rng, octaves=5, base=6)[..., None]),
    )
    radiance[window] = outside[window]

    # Un parteluz cruzando la ventana y una lamparita en la esquina oscura.
    mullion = window & (np.abs(columns - 0.70 * width) < 0.006 * width)
    radiance[mullion] = (5.0 * np.array([0.30, 0.24, 0.18]))[None, :]

    lamp_radius = np.sqrt((rows - 0.74 * height) ** 2 + (columns - 0.16 * width) ** 2)
    glow = np.exp(-((lamp_radius / (0.05 * width)) ** 2))
    radiance += (60.0 * glow)[..., None] * np.array([1.0, 0.86, 0.62])

    # Un pequeño término ambiental hace de interreflexión.  Sin él los píxeles
    # más oscuros son numéricamente cero, lo que haría del rango dinámico una
    # propiedad del formato de coma flotante y no de la escena.
    return radiance + 0.02
