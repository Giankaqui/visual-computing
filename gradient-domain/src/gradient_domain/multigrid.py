"""Multigrid geométrico para el problema de Poisson con condiciones de Dirichlet.

Un esquema de relajación como Gauss-Seidel elimina la parte de alta frecuencia
del error en unas pocas pasadas y luego se estanca: la parte de baja frecuencia
es lo que hace que la relajación a secas necesite ``O(N)`` pasadas en una malla
de ``N`` píxeles.  La observación detrás de multigrid es que un error suave en
una malla fina no es suave en una malla dos veces más gruesa, así que se puede
relajar allí en su lugar, a la cuarta parte del coste.

Un ciclo V aplica eso de forma recursiva: suavizar, restringir el residuo,
resolver aproximadamente el problema grueso con el mismo procedimiento,
interpolar la corrección de vuelta y volver a suavizar.  El coste de un ciclo es
una constante pequeña por el coste de una pasada en la malla fina, porque las
mallas encogen geométricamente, y el error se reduce en un factor que no depende
del tamaño de malla.  El trabajo total para alcanzar una tolerancia fija es por
tanto proporcional al número de píxeles, que es lo óptimo.

Las transferencias de malla son la interpolación bilineal ``P`` y su adjunta
escalada ``P^T / 4``, que es la restricción de ponderación completa.  Usar un par
adjunto importa: es lo que convierte la corrección en la malla gruesa en una
proyección en la norma de energía, y un par mal emparejado degrada el factor de
convergencia o lo destruye.

El engrosamiento reduce a la mitad el número de puntos interiores como
``m -> (m - 1) // 2``, de modo que los puntos gruesos caen exactamente sobre los
puntos finos impares.  La relación es exacta cuando el tamaño es impar y
aproximada en otro caso, lo que cuesta algo de convergencia con tamaños de imagen
arbitrarios.  :func:`solve` informa del factor de convergencia observado, y
:mod:`gradient_domain.solvers` ofrece el ciclo como precondicionador del
gradiente conjugado, que es insensible a esa pérdida.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from .operators import laplacian

__all__ = ["MultigridOptions", "MultigridReport", "v_cycle", "solve"]


@dataclass
class MultigridOptions:
    """Configuración del solver.

    Attributes
    ----------
    pre_smoothing, post_smoothing : int
        Pasadas de Gauss-Seidel antes de restringir y después de interpolar.
    coarsest_size : int
        Las mallas con menos puntos que este valor en cualquiera de los dos ejes
        se resuelven directamente.
    coarsest_sweeps : int
        Pasadas usadas en la malla más gruesa, que es lo bastante pequeña como
        para que la relajación sola converja.
    max_cycles : int
        Cota superior de ciclos V.
    tolerance : float
        Residuo relativo objetivo en norma euclídea.
    """

    pre_smoothing: int = 2
    post_smoothing: int = 2
    coarsest_size: int = 8
    coarsest_sweeps: int = 60
    max_cycles: int = 60
    tolerance: float = 1e-8


@dataclass
class MultigridReport:
    """Registro de convergencia de una resolución.

    Attributes
    ----------
    residuals : list of float
        Residuo relativo tras cada ciclo, empezando por el inicial.
    cycles : int
        Ciclos V realizados.
    converged : bool
    """

    residuals: list[float]
    cycles: int
    converged: bool

    @property
    def convergence_factor(self) -> float:
        """Media geométrica de la reducción de residuo por ciclo.

        Valores en torno a 0.1 son lo que consigue un ciclo V bien montado sobre
        el problema de Poisson; valores cercanos a 1 significan que la corrección
        en la malla gruesa no está ayudando.
        """
        if len(self.residuals) < 2 or self.residuals[0] <= 0:
            return float("nan")
        ratio = self.residuals[-1] / self.residuals[0]
        return float(ratio ** (1.0 / (len(self.residuals) - 1)))

    def __str__(self) -> str:
        return (
            f"{self.cycles} ciclos, residuo relativo {self.residuals[-1]:.2e}, "
            f"factor de convergencia {self.convergence_factor:.3f}"
        )


@lru_cache(maxsize=64)
def _prolongation_1d(fine: int, coarse: int) -> sp.csr_matrix:
    """Interpolación lineal de puntos ``coarse`` a puntos ``fine`` en un eje.

    El punto grueso ``I`` cae sobre el punto fino ``2 I + 1``; los puntos finos
    intermedios toman la media de sus dos vecinos, y los puntos fuera del rango
    grueso ven el valor de Dirichlet homogéneo, que es cero.
    """
    rows, cols, values = [], [], []
    for index in range(fine):
        if index % 2 == 1:
            source = (index - 1) // 2
            if 0 <= source < coarse:
                rows.append(index)
                cols.append(source)
                values.append(1.0)
        else:
            for source in (index // 2 - 1, index // 2):
                if 0 <= source < coarse:
                    rows.append(index)
                    cols.append(source)
                    values.append(0.5)
    return sp.csr_matrix((values, (rows, cols)), shape=(fine, coarse))


@lru_cache(maxsize=64)
def _parity_mask(shape: tuple[int, int], parity: int) -> np.ndarray:
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    return (rows + cols) % 2 == parity


def _coarse_shape(shape: tuple[int, int]) -> tuple[int, int]:
    return ((shape[0] - 1) // 2, (shape[1] - 1) // 2)


def _smooth(u: np.ndarray, b: np.ndarray, spacing: float, sweeps: int) -> np.ndarray:
    """Pasadas de Gauss-Seidel rojo-negro sobre ``laplacian(u) = b``.

    Los dos colores se actualizan en secuencia, así que la segunda mitad de cada
    pasada ya ve los valores nuevos; eso es lo que convierte el esquema en un
    Gauss-Seidel en vez de en una iteración de Jacobi, y aproximadamente duplica
    la tasa de suavizado.  Dentro de un color las actualizaciones son
    independientes, que es lo que permite escribir la media pasada entera como
    una sola expresión de arrays.
    """
    squared = spacing * spacing
    for _ in range(sweeps):
        for parity in (0, 1):
            padded = np.pad(u, 1)
            neighbours = (
                padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
            )
            updated = 0.25 * (neighbours - squared * b)
            u = np.where(_parity_mask(u.shape, parity), updated, u)
    return u


def _restrict(residual: np.ndarray) -> np.ndarray:
    """Restricción de ponderación completa, la adjunta escalada de :func:`_prolong`."""
    coarse_rows, coarse_cols = _coarse_shape(residual.shape)
    rows = _prolongation_1d(residual.shape[0], coarse_rows)
    cols = _prolongation_1d(residual.shape[1], coarse_cols)
    return (rows.T @ residual @ cols) * 0.25


def _prolong(correction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Interpolación bilineal de una corrección de la malla gruesa."""
    rows = _prolongation_1d(shape[0], correction.shape[0])
    cols = _prolongation_1d(shape[1], correction.shape[1])
    return rows @ correction @ cols.T


def v_cycle(
    u: np.ndarray, b: np.ndarray, spacing: float, options: MultigridOptions
) -> np.ndarray:
    """Aplica un ciclo V al sistema ``laplacian(u) = b``.

    Parameters
    ----------
    u : ndarray, shape (m, n)
        Iterado actual; los ceros son un punto de partida válido.
    b : ndarray, shape (m, n)
        Término independiente sobre el interior, con los valores de Dirichlet ya
        plegados dentro.
    spacing : float
        Paso de malla en este nivel.
    options : MultigridOptions

    Returns
    -------
    ndarray
        Iterado mejorado, con la misma forma que ``u``.
    """
    coarse_shape = _coarse_shape(u.shape)
    if min(u.shape) <= options.coarsest_size or min(coarse_shape) < 2:
        return _smooth(u, b, spacing, options.coarsest_sweeps)

    u = _smooth(u, b, spacing, options.pre_smoothing)
    residual = b - laplacian(u, spacing)
    coarse_correction = v_cycle(
        np.zeros(coarse_shape), _restrict(residual), 2.0 * spacing, options
    )
    u = u + _prolong(coarse_correction, u.shape)
    return _smooth(u, b, spacing, options.post_smoothing)


def solve(
    b: np.ndarray,
    spacing: float = 1.0,
    options: MultigridOptions | None = None,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, MultigridReport]:
    """Resuelve ``laplacian(u) = b`` mediante ciclos V repetidos.

    Parameters
    ----------
    b : ndarray, shape (m, n)
        Término independiente sobre las incógnitas interiores.
    spacing : float
        Paso de malla del nivel más fino.
    options : MultigridOptions o None
    initial : ndarray o None
        Iterado inicial; ceros si se omite.

    Returns
    -------
    u : ndarray, shape (m, n)
    report : MultigridReport
    """
    options = options or MultigridOptions()
    b = np.asarray(b, dtype=float)
    u = np.zeros_like(b) if initial is None else np.array(initial, dtype=float, copy=True)

    scale = float(np.linalg.norm(b))
    if scale == 0.0:
        return u, MultigridReport(residuals=[0.0], cycles=0, converged=True)

    residuals = [float(np.linalg.norm(b - laplacian(u, spacing))) / scale]
    for cycle in range(1, options.max_cycles + 1):
        u = v_cycle(u, b, spacing, options)
        residuals.append(float(np.linalg.norm(b - laplacian(u, spacing))) / scale)
        if residuals[-1] < options.tolerance:
            return u, MultigridReport(residuals=residuals, cycles=cycle, converged=True)
    return u, MultigridReport(residuals=residuals, cycles=options.max_cycles, converged=False)
