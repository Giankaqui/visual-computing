"""Mapeo tonal en el dominio del gradiente para imágenes de alto rango dinámico.

Un mapa de radiancia puede abarcar cinco órdenes de magnitud y una pantalla
cubre dos.  Escalar los valores globalmente destruye o las altas luces o las
sombras.  La observación de Fattal, Lischinski y Werman (2002) es que el rango
dinámico vive en los gradientes *grandes* de la imagen de log-radiancia mientras
que el detalle vive en los pequeños, así que atenuar los gradientes con un
factor que decrece con su magnitud comprime el rango y deja el detalle en paz.

La atenuación tiene que actuar a la escala correcta.  Un borde grande de la
imagen no es un único gradiente grande a resolución completa; es una rampa
repartida entre muchos píxeles, cada uno con un gradiente moderado.  Por eso el
factor se calcula sobre una pirámide gaussiana y se propaga de grueso a fino, de
modo que un píxel situado sobre un borde a gran escala se atenúa aunque su
gradiente local sea pequeño.

El campo atenuado no es el gradiente de ninguna imagen, así que hay que
integrarlo en el sentido de mínimos cuadrados, que es de nuevo una ecuación de
Poisson.  Aquí la condición de contorno natural es la de Neumann: nada en la
imagen debe quedar fijado a un valor prescrito, y solo importan las diferencias.
Eso hace que el solver por transformada del coseno de
:mod:`gradient_domain.solvers` sea exacto e inmediato.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import divergence, gradient
from .solvers import solve_neumann

__all__ = ["ToneMapConfig", "attenuation_field", "tone_map"]


@dataclass
class ToneMapConfig:
    """Parámetros de la atenuación.

    Attributes
    ----------
    alpha : float
        Magnitud de gradiente que se deja intacta, expresada como múltiplo de la
        magnitud media de gradiente del nivel.  Los gradientes por debajo se
        amplifican y los de por encima se atenúan.
    beta : float
        Exponente en ``(0, 1]``.  El factor es ``(g / alpha) ** (beta - 1)``, así
        que ``beta = 1`` deja todos los gradientes intactos y los valores más
        pequeños atenúan los grandes de forma más agresiva.  El artículo original
        recomienda entre 0.8 y 0.9, que es un ajuste deliberadamente suave:
        atenuar de más produce ese aspecto lavado y sin contraste que le ha dado
        mala fama al mapeo tonal.
    saturation : float
        Exponente aplicado a las razones cromáticas.  Comprimir la luminancia sin
        esto deja los colores sobresaturados, porque las razones que los
        produjeron estaban ajustadas a un rango de luminancia mucho mayor.
    levels : int
        Niveles de pirámide usados para propagar la atenuación; la pirámide para
        antes si un nivel bajaría de ocho píxeles.
    epsilon : float
        Suelo para las magnitudes de gradiente, que evita que las regiones planas
        se amplifiquen sin límite.
    """

    alpha: float = 0.1
    beta: float = 0.85
    saturation: float = 0.5
    levels: int = 8
    epsilon: float = 1e-4


def _downsample(image: np.ndarray) -> np.ndarray:
    """Difumina con un núcleo binomial y descarta una muestra de cada dos."""
    kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    padded = np.pad(image, ((0, 0), (2, 2)), mode="edge")
    blurred = sum(weight * padded[:, i : i + image.shape[1]] for i, weight in enumerate(kernel))
    padded = np.pad(blurred, ((2, 2), (0, 0)), mode="edge")
    blurred = sum(weight * padded[i : i + image.shape[0], :] for i, weight in enumerate(kernel))
    return blurred[::2, ::2]


def _upsample(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Interpolación bilineal de vuelta a una malla más fina."""
    rows = np.linspace(0, image.shape[0] - 1, shape[0])
    columns = np.linspace(0, image.shape[1] - 1, shape[1])
    row0 = np.floor(rows).astype(int)
    col0 = np.floor(columns).astype(int)
    row1 = np.minimum(row0 + 1, image.shape[0] - 1)
    col1 = np.minimum(col0 + 1, image.shape[1] - 1)
    row_weight = (rows - row0)[:, None]
    col_weight = (columns - col0)[None, :]

    top = image[row0][:, col0] * (1 - col_weight) + image[row0][:, col1] * col_weight
    bottom = image[row1][:, col0] * (1 - col_weight) + image[row1][:, col1] * col_weight
    return top * (1 - row_weight) + bottom * row_weight


def _central_gradient(image: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Diferencias centradas con los bordes replicados."""
    padded = np.pad(image, 1, mode="edge")
    gx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * spacing)
    gy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * spacing)
    return gx, gy


def attenuation_field(log_luminance: np.ndarray, config: ToneMapConfig) -> np.ndarray:
    """Construye el factor multiescala de atenuación de gradientes.

    Parameters
    ----------
    log_luminance : ndarray, shape (h, w)
        Logaritmo del canal de luminancia.
    config : ToneMapConfig

    Returns
    -------
    ndarray, shape (h, w)
        Factor por píxel que se aplica al campo de gradiente.
    """
    pyramid = [log_luminance]
    while len(pyramid) < config.levels and min(pyramid[-1].shape) > 16:
        pyramid.append(_downsample(pyramid[-1]))

    factor: np.ndarray | None = None
    for level in reversed(range(len(pyramid))):
        gx, gy = _central_gradient(pyramid[level], spacing=2.0**level)
        magnitude = np.sqrt(gx * gx + gy * gy)
        scale = config.alpha * max(float(magnitude.mean()), config.epsilon)
        local = (scale / np.maximum(magnitude, config.epsilon)) * (
            np.maximum(magnitude, config.epsilon) / scale
        ) ** config.beta
        factor = local if factor is None else _upsample(factor, local.shape) * local
    return factor


def tone_map(
    radiance: np.ndarray, config: ToneMapConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Comprime un mapa de radiancia hasta una imagen mostrable.

    Parameters
    ----------
    radiance : ndarray, shape (h, w, 3)
        Radiancia lineal con escala arbitraria; solo importan las razones.
    config : ToneMapConfig o None

    Returns
    -------
    image : ndarray of float, shape (h, w, 3)
        Valores referidos a pantalla en ``[0, 1]``.
    factor : ndarray, shape (h, w)
        El campo de atenuación que se aplicó, devuelto para poder inspeccionarlo.

    Raises
    ------
    ValueError
        Si la entrada no es una imagen de tres canales.
    """
    config = config or ToneMapConfig()
    radiance = np.asarray(radiance, dtype=float)
    if radiance.ndim != 3 or radiance.shape[2] != 3:
        raise ValueError("el mapeo tonal espera un mapa de radiancia de tres canales")

    luminance = radiance @ np.array([0.2126, 0.7152, 0.0722])
    luminance = np.maximum(luminance, np.finfo(float).tiny)
    log_luminance = np.log(luminance)

    factor = attenuation_field(log_luminance, config)
    # Las magnitudes que dirigen la atenuación salen de diferencias centradas,
    # que son insesgadas, pero el campo que se integra usa las diferencias hacia
    # delante cuyo adjunto exacto es la divergencia de `operators`.  Mezclar las
    # dos plantillas aquí dejaría un residuo que el solver no puede eliminar.
    gx, gy = gradient(log_luminance)
    compressed = solve_neumann(divergence(gx * factor, gy * factor))

    # La solución está definida salvo una constante; anclar el píxel más
    # brillante deja el resultado en un rango predecible antes de la
    # normalización final.
    result = np.exp(compressed - compressed.max())
    ratios = radiance / luminance[..., None]
    image = (ratios ** config.saturation) * result[..., None]

    peak = float(np.percentile(image, 99.5))
    return np.clip(image / max(peak, 1e-12), 0.0, 1.0), factor
