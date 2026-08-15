"""Edición de imágenes en el dominio del gradiente.

Todas las operaciones de aquí tienen la misma forma.  Se construye un campo guía
modificando los gradientes de una o varias imágenes de entrada, y la salida es la
imagen cuyo gradiente mejor encaja con ese campo sujeto a condiciones de
contorno.  Lo único que cambia entre aplicaciones es cómo se elige el campo guía.

Hay dos dominios soportados, y la diferencia merece entenderse.

``mask``
    Las incógnitas son los píxeles de dentro de la selección, y los píxeles justo
    de fuera aportan los valores de Dirichlet.  Es la formulación de Pérez,
    Gangnet y Blake (2003).  La salida coincide exactamente con el destino fuera
    de la selección, que suele ser lo que una herramienta de edición debería
    garantizar, pero el dominio es irregular, así que el sistema hay que montarlo
    y factorizarlo explícitamente.

``rectangle``
    Las incógnitas son un rectángulo alrededor de la selección, el campo guía es
    el gradiente de la fuente dentro y el del destino fuera, y el borde del
    rectángulo aporta los valores de contorno.  El residuo fuera de la selección
    es entonces una función armónica con valores de contorno nulos, así que es
    pequeño pero no exactamente cero.  A cambio el dominio es regular, que es lo
    que necesita el multigrid geométrico, y la resolución es lineal en el número
    de píxeles.

Ambas coinciden salvo esa corrección armónica, cosa que los tests verifican.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .operators import divergence, fold_boundary, gradient
from .solvers import SolverReport, solve_system

__all__ = [
    "GuidanceField",
    "solve_dirichlet",
    "solve_masked",
    "seamless_clone",
    "texture_flatten",
    "illumination_change",
]


@dataclass
class GuidanceField:
    """Un campo de gradiente objetivo.

    Attributes
    ----------
    gx, gy : ndarray, shape (h, w) o (h, w, c)
        Derivadas horizontal y vertical deseadas.
    """

    gx: np.ndarray
    gy: np.ndarray

    @classmethod
    def of(cls, image: np.ndarray) -> GuidanceField:
        """El campo que reproduce ``image`` exactamente."""
        return cls(*gradient(image))

    def select(self, mask: np.ndarray, other: GuidanceField) -> GuidanceField:
        """Toma este campo dentro de ``mask`` y ``other`` fuera de ella."""
        selector = mask[..., None] if self.gx.ndim == 3 else mask
        return GuidanceField(
            np.where(selector, self.gx, other.gx), np.where(selector, self.gy, other.gy)
        )

    def mix_by_magnitude(self, other: GuidanceField) -> GuidanceField:
        """Conserva, componente a componente, el campo de mayor magnitud.

        Es la variante de gradientes mezclados: deja que la estructura fuerte del
        destino sobreviva bajo una región de fuente comparativamente plana, que
        es lo que hace que pegar sobre un fondo con textura quede bien en vez de
        borrarlo.
        """
        return GuidanceField(
            np.where(np.abs(self.gx) >= np.abs(other.gx), self.gx, other.gx),
            np.where(np.abs(self.gy) >= np.abs(other.gy), self.gy, other.gy),
        )

    def scaled(self, factor: np.ndarray | float) -> GuidanceField:
        """Multiplica ambas componentes por un escalar o un factor por píxel."""
        return GuidanceField(self.gx * factor, self.gy * factor)


def _as_channels(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Devuelve una vista ``(h, w, c)`` y si la entrada era de un solo canal."""
    if image.ndim == 2:
        return image[..., None], True
    return image, False


def solve_dirichlet(
    field: GuidanceField,
    boundary: np.ndarray,
    method: str = "mgcg",
    tolerance: float = 1e-8,
    max_iterations: int = 2000,
) -> tuple[np.ndarray, list[SolverReport]]:
    """Integra un campo guía sobre un rectángulo con los bordes fijados.

    Parameters
    ----------
    field : GuidanceField
        Gradientes objetivo sobre el rectángulo completo.
    boundary : ndarray, shape (h, w) o (h, w, c)
        Imagen cuyo borde de un píxel aporta los valores de Dirichlet; su
        interior se ignora.
    method : str
        Nombre de un solver de :data:`gradient_domain.solvers.SOLVERS`.
    tolerance : float
    max_iterations : int

    Returns
    -------
    image : ndarray
        Misma forma que ``boundary``, e igual a ella en el borde.
    reports : list of SolverReport
        Una entrada por canal de color.
    """
    boundary_channels, was_gray = _as_channels(np.asarray(boundary, dtype=float))
    right_hand_side = divergence(field.gx, field.gy)
    if right_hand_side.ndim == 2:
        right_hand_side = right_hand_side[..., None]

    result = boundary_channels.copy()
    reports: list[SolverReport] = []
    for channel in range(boundary_channels.shape[2]):
        folded = fold_boundary(
            right_hand_side[1:-1, 1:-1, channel], boundary_channels[..., channel]
        )
        solution, report = solve_system(
            folded, method=method, tolerance=tolerance, max_iterations=max_iterations
        )
        result[1:-1, 1:-1, channel] = solution
        reports.append(report)

    return (result[..., 0] if was_gray else result), reports


def _neighbour_offsets() -> tuple[tuple[int, int], ...]:
    return ((-1, 0), (1, 0), (0, -1), (0, 1))


def solve_masked(
    field: GuidanceField, image: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, list[SolverReport]]:
    """Integra un campo guía sobre los píxeles que selecciona ``mask``.

    Los píxeles fuera de la máscara conservan su valor y actúan como datos de
    Dirichlet.  El sistema se monta y se factoriza una sola vez y luego se
    reutiliza para cada canal de color, que es de donde sale casi todo el ahorro
    en una imagen multicanal.

    Parameters
    ----------
    field : GuidanceField
        Gradientes objetivo, definidos sobre toda la imagen.
    image : ndarray, shape (h, w) o (h, w, c)
        Imagen de destino; aporta los valores de contorno.
    mask : ndarray of bool, shape (h, w)
        Píxeles que hay que resolver.

    Returns
    -------
    result : ndarray
        Misma forma que ``image``.
    reports : list of SolverReport
    """
    channels, was_gray = _as_channels(np.asarray(image, dtype=float))
    height, width = mask.shape
    interior = np.zeros_like(mask, dtype=bool)
    interior[1:-1, 1:-1] = mask[1:-1, 1:-1]
    if not interior.any():
        return image.copy(), []

    index = np.full(mask.shape, -1, dtype=np.int64)
    index[interior] = np.arange(int(interior.sum()))
    unknowns = int(interior.sum())

    rows = [np.arange(unknowns)]
    cols = [np.arange(unknowns)]
    values = [np.full(unknowns, -4.0)]
    for row_offset, column_offset in _neighbour_offsets():
        shifted = np.roll(np.roll(index, -row_offset, axis=0), -column_offset, axis=1)
        neighbour = shifted[interior]
        connected = neighbour >= 0
        rows.append(np.arange(unknowns)[connected])
        cols.append(neighbour[connected])
        values.append(np.ones(int(connected.sum())))

    operator = sp.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(unknowns, unknowns),
    ).tocsc()

    started = time.perf_counter()
    factorization = spla.splu(operator)
    factorization_seconds = time.perf_counter() - started

    right_hand_side = divergence(field.gx, field.gy)
    if right_hand_side.ndim == 2:
        right_hand_side = right_hand_side[..., None]

    result = channels.copy()
    reports: list[SolverReport] = []
    for channel in range(channels.shape[2]):
        b = right_hand_side[..., channel][interior].astype(float)
        for row_offset, column_offset in _neighbour_offsets():
            shifted_index = np.roll(np.roll(index, -row_offset, axis=0), -column_offset, axis=1)
            shifted_value = np.roll(
                np.roll(channels[..., channel], -row_offset, axis=0), -column_offset, axis=1
            )
            outside = shifted_index[interior] < 0
            b[outside] -= shifted_value[interior][outside]

        started = time.perf_counter()
        solution = factorization.solve(b)
        elapsed = time.perf_counter() - started

        result[..., channel][interior] = solution
        residual = float(np.linalg.norm(operator @ solution - b)) / max(
            float(np.linalg.norm(b)), 1e-30
        )
        reports.append(
            SolverReport(
                method="direct (masked)",
                iterations=0,
                relative_residual=residual,
                seconds=elapsed + factorization_seconds / channels.shape[2],
                converged=True,
            )
        )

    return (result[..., 0] if was_gray else result), reports


def _place(source: np.ndarray, mask: np.ndarray, target_shape, offset: tuple[int, int]):
    """Pega un recorte de fuente y su máscara en un lienzo del tamaño del destino."""
    row, column = offset
    height, width = source.shape[:2]
    if row < 0 or column < 0 or row + height > target_shape[0] or column + width > target_shape[1]:
        raise ValueError("el recorte de fuente no cabe en el destino con este desplazamiento")

    canvas = np.zeros(target_shape, dtype=float)
    canvas[row : row + height, column : column + width] = source
    placed_mask = np.zeros(target_shape[:2], dtype=bool)
    placed_mask[row : row + height, column : column + width] = mask
    return canvas, placed_mask


def seamless_clone(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    offset: tuple[int, int] = (0, 0),
    mode: str = "import",
    domain: str = "mask",
    method: str = "mgcg",
    margin: int = 24,
) -> tuple[np.ndarray, list[SolverReport]]:
    """Pega una región de ``source`` en ``target`` sin costura visible.

    Copiar píxeles transfiere el color absoluto de la fuente, que casi nunca
    encaja con el destino.  Copiar gradientes transfiere en cambio solo la
    variación relativa, y el nivel absoluto se recupera del destino a través de
    la condición de contorno, así que el inserto adopta la iluminación de su
    entorno.

    Parameters
    ----------
    source : ndarray, shape (sh, sw) o (sh, sw, c)
        Recorte que se inserta.
    target : ndarray, shape (h, w) o (h, w, c)
        Imagen de destino.
    mask : ndarray of bool, shape (sh, sw)
        Selección dentro del recorte de fuente.
    offset : tuple of int
        Posición de la esquina superior izquierda del recorte en el destino.
    mode : {'import', 'mixed', 'average'}
        Cómo se combinan los gradientes de fuente y destino dentro de la máscara.
    domain : {'mask', 'rectangle'}
        Qué formulación usar; ver el docstring del módulo.
    method : str
        Nombre del solver, usado solo por la formulación de rectángulo.
    margin : int
        Margen alrededor de la caja envolvente de la máscara para la formulación
        de rectángulo.

    Returns
    -------
    result : ndarray
        Misma forma que ``target``.
    reports : list of SolverReport

    Raises
    ------
    ValueError
        Si ``mode`` o ``domain`` son desconocidos, o si el recorte no cabe.
    """
    target = np.asarray(target, dtype=float)
    source = np.asarray(source, dtype=float)
    if source.ndim != target.ndim:
        raise ValueError("source y target deben tener el mismo número de dimensiones")

    placed, placed_mask = _place(source, mask, target.shape, offset)
    source_field = GuidanceField.of(placed)
    target_field = GuidanceField.of(target)

    if mode == "import":
        inside = source_field
    elif mode == "mixed":
        inside = source_field.mix_by_magnitude(target_field)
    elif mode == "average":
        inside = GuidanceField(
            0.5 * (source_field.gx + target_field.gx), 0.5 * (source_field.gy + target_field.gy)
        )
    else:
        raise ValueError(f"modo desconocido {mode!r}")

    field = inside.select(placed_mask, target_field)

    if domain == "mask":
        return solve_masked(field, target, placed_mask)
    if domain != "rectangle":
        raise ValueError(f"dominio desconocido {domain!r}")

    rows, columns = np.nonzero(placed_mask)
    top = max(int(rows.min()) - margin, 0)
    bottom = min(int(rows.max()) + margin + 1, target.shape[0])
    left = max(int(columns.min()) - margin, 0)
    right = min(int(columns.max()) + margin + 1, target.shape[1])
    window = (slice(top, bottom), slice(left, right))

    cropped_field = GuidanceField(field.gx[window], field.gy[window])
    patch, reports = solve_dirichlet(cropped_field, target[window], method=method)
    result = target.copy()
    result[window] = patch
    return result, reports


def texture_flatten(
    image: np.ndarray,
    edges: np.ndarray,
    method: str = "mgcg",
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[SolverReport]]:
    """Elimina la textura conservando los bordes que definen las formas.

    El campo guía es el gradiente de la imagen multiplicado por un indicador que
    vale uno sobre los bordes que se conservan y cero en el resto.  Integrar un
    campo que es nulo en una región obliga a esa región a quedar tan plana como
    permitan las condiciones de contorno, así que la textura desaparece y la
    estructura a gran escala se queda.

    Parameters
    ----------
    image : ndarray, shape (h, w) o (h, w, c)
    edges : ndarray, shape (h, w)
        Pesos en ``[0, 1]``; lo habitual es un mapa de bordes binario.
    method : str
        Nombre del solver.
    mask : ndarray of bool, shape (h, w), opcional
        Restringe el efecto a una región, resuelta con la formulación de máscara.

    Returns
    -------
    result : ndarray
    reports : list of SolverReport
    """
    image = np.asarray(image, dtype=float)
    weights = np.asarray(edges, dtype=float)
    if image.ndim == 3:
        weights = weights[..., None]

    field = GuidanceField.of(image).scaled(weights)
    if mask is not None:
        return solve_masked(field, image, mask)
    return solve_dirichlet(field, image, method=method)


def illumination_change(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.2,
    beta: float = 0.2,
    epsilon: float = 1e-4,
) -> tuple[np.ndarray, list[SolverReport]]:
    """Comprime el rango dinámico local dentro de una selección.

    La magnitud del gradiente se remapea con ``alpha ** beta * g ** (-beta)``,
    que atenúa los gradientes grandes más que los pequeños.  Aplicado dentro de
    una selección con los píxeles del entorno fijados, esto saca detalle de las
    sombras sin cambiar la iluminación global de la imagen.

    Parameters
    ----------
    image : ndarray, shape (h, w) o (h, w, c)
        Se esperan valores en ``[0, 1]``.
    mask : ndarray of bool, shape (h, w)
    alpha : float
        Magnitud de gradiente que se deja intacta; valores más pequeños oscurecen
        menos.
    beta : float
        Exponente de la atenuación, en ``[0, 1]``.  Ojo: el convenio es el
        opuesto al de :mod:`gradient_domain.hdr`; aquí cero deja la imagen
        intacta y uno la aplana a una magnitud de gradiente constante, siguiendo
        a Pérez et al. y no a Fattal et al.
    epsilon : float
        Suelo para la magnitud del gradiente, que evita que las regiones planas
        se amplifiquen sin límite.

    Returns
    -------
    result : ndarray
    reports : list of SolverReport
    """
    image = np.asarray(image, dtype=float)
    field = GuidanceField.of(image)
    magnitude = np.sqrt(field.gx**2 + field.gy**2)
    factor = (alpha ** beta) * np.maximum(magnitude, epsilon) ** (-beta)
    return solve_masked(field.scaled(factor), image, mask)
