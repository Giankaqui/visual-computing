"""Solvers lineales para el sistema de Poisson interior con Dirichlet.

Se ofrecen cuatro métodos para el mismo sistema, que es lo que hace significativa
la comparación de costes de los benchmarks.

``direct``
    LU dispersa del laplaciano de cinco puntos.  Exacto salvo redondeo y la
    elección correcta cuando el mismo operador se reutiliza para muchos términos
    independientes, ya que la factorización se calcula una sola vez.  Su memoria
    crece más rápido que la malla por el llenado, que es lo que acaba
    descartándolo a alta resolución.

``cg``
    Gradiente conjugado con precondicionado de Jacobi, sin matriz.  La memoria es
    proporcional a la malla, pero el número de iteraciones crece con el número de
    condición del laplaciano, que escala con el número de píxeles; el trabajo es
    por tanto superlineal.

``multigrid``
    Ciclos V de :mod:`gradient_domain.multigrid`.  El número de iteraciones es
    independiente del tamaño de malla, así que el trabajo total es lineal.

``mgcg``
    Gradiente conjugado precondicionado con un ciclo V.  Mantiene el
    comportamiento lineal de multigrid y recupera los casos en los que el ciclo
    simple converge despacio, que aquí son las mallas cuyo tamaño no engrosa de
    forma exacta.  Por eso es el método por defecto.

Todos los métodos resuelven el sistema definido positivo ``-L u = -b`` en lugar
de ``L u = b``; el laplaciano tal como está escrito es definido negativo, y el
gradiente conjugado exige que sea definido positivo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla
from scipy.fft import dctn, idctn

from .multigrid import MultigridOptions, v_cycle
from .operators import laplacian, sparse_laplacian

__all__ = ["SolverReport", "SOLVERS", "solve_system", "solve_neumann"]


@dataclass
class SolverReport:
    """Coste y precisión de una resolución.

    Attributes
    ----------
    method : str
    iterations : int
        Iteraciones o ciclos; cero para el método directo.
    relative_residual : float
        Norma euclídea del residuo dividida por la norma del término
        independiente.
    seconds : float
    converged : bool
    """

    method: str
    iterations: int
    relative_residual: float
    seconds: float
    converged: bool

    def __str__(self) -> str:
        return (
            f"{self.method}: {self.iterations} iteraciones, "
            f"residuo {self.relative_residual:.2e}, {self.seconds * 1e3:.0f} ms"
        )


def _relative_residual(u: np.ndarray, b: np.ndarray, spacing: float) -> float:
    scale = float(np.linalg.norm(b))
    if scale == 0.0:
        return 0.0
    return float(np.linalg.norm(b - laplacian(u, spacing))) / scale


def _solve_direct(
    b: np.ndarray, spacing: float, tolerance: float, max_iterations: int
) -> tuple[np.ndarray, int, bool]:
    operator = sparse_laplacian(b.shape, spacing).tocsc()
    solution = spla.splu(operator).solve(b.reshape(-1))
    return solution.reshape(b.shape), 0, True


def _solve_cg(
    b: np.ndarray, spacing: float, tolerance: float, max_iterations: int
) -> tuple[np.ndarray, int, bool]:
    # El precondicionado de Jacobi sobre un operador de coeficientes constantes
    # es un escalar, así que solo cambia la escala del paso; se mantiene porque
    # hace comparable el residuo con el de las variantes precondicionadas.
    diagonal = 4.0 / (spacing * spacing)
    rhs = -b
    u = np.zeros_like(b)
    residual = rhs.copy()
    direction = residual / diagonal
    delta = float(np.sum(residual * direction))
    target = tolerance * float(np.linalg.norm(rhs))

    for iteration in range(1, max_iterations + 1):
        operator_direction = -laplacian(direction, spacing)
        step = delta / float(np.sum(direction * operator_direction))
        u += step * direction
        residual -= step * operator_direction
        if float(np.linalg.norm(residual)) <= target:
            return u, iteration, True
        preconditioned = residual / diagonal
        new_delta = float(np.sum(residual * preconditioned))
        direction = preconditioned + (new_delta / delta) * direction
        delta = new_delta
    return u, max_iterations, False


def _solve_multigrid(
    b: np.ndarray, spacing: float, tolerance: float, max_iterations: int
) -> tuple[np.ndarray, int, bool]:
    options = MultigridOptions(max_cycles=max_iterations, tolerance=tolerance)
    u = np.zeros_like(b)
    scale = float(np.linalg.norm(b)) or 1.0
    for cycle in range(1, max_iterations + 1):
        u = v_cycle(u, b, spacing, options)
        if float(np.linalg.norm(b - laplacian(u, spacing))) / scale < tolerance:
            return u, cycle, True
    return u, max_iterations, False


def _solve_mgcg(
    b: np.ndarray, spacing: float, tolerance: float, max_iterations: int
) -> tuple[np.ndarray, int, bool]:
    options = MultigridOptions(pre_smoothing=1, post_smoothing=1)

    def precondition(residual: np.ndarray) -> np.ndarray:
        return v_cycle(np.zeros_like(residual), -residual, spacing, options)

    rhs = -b
    u = np.zeros_like(b)
    residual = rhs.copy()
    direction = precondition(residual)
    delta = float(np.sum(residual * direction))
    target = tolerance * float(np.linalg.norm(rhs))

    for iteration in range(1, max_iterations + 1):
        operator_direction = -laplacian(direction, spacing)
        step = delta / float(np.sum(direction * operator_direction))
        u += step * direction
        residual -= step * operator_direction
        if float(np.linalg.norm(residual)) <= target:
            return u, iteration, True
        preconditioned = precondition(residual)
        new_delta = float(np.sum(residual * preconditioned))
        direction = preconditioned + (new_delta / delta) * direction
        delta = new_delta
    return u, max_iterations, False


SOLVERS = {
    "direct": _solve_direct,
    "cg": _solve_cg,
    "multigrid": _solve_multigrid,
    "mgcg": _solve_mgcg,
}


def solve_system(
    b: np.ndarray,
    method: str = "mgcg",
    spacing: float = 1.0,
    tolerance: float = 1e-8,
    max_iterations: int = 2000,
) -> tuple[np.ndarray, SolverReport]:
    """Resuelve ``laplacian(u) = b`` con condiciones de Dirichlet homogéneas.

    Parameters
    ----------
    b : ndarray, shape (m, n)
        Término independiente sobre las incógnitas interiores.
    method : {'direct', 'cg', 'multigrid', 'mgcg'}
    spacing : float
        Paso de malla.
    tolerance : float
        Residuo relativo objetivo; el método directo lo ignora.
    max_iterations : int
        Presupuesto de iteraciones o ciclos para los métodos iterativos.

    Returns
    -------
    u : ndarray, shape (m, n)
    report : SolverReport

    Raises
    ------
    KeyError
        Si ``method`` no es uno de los nombres soportados.
    """
    if method not in SOLVERS:
        raise KeyError(f"solver desconocido {method!r}; elige entre {sorted(SOLVERS)}")

    b = np.asarray(b, dtype=float)
    if not np.any(b):
        return np.zeros_like(b), SolverReport(
            method=method, iterations=0, relative_residual=0.0, seconds=0.0, converged=True
        )

    started = time.perf_counter()
    u, iterations, converged = SOLVERS[method](b, spacing, tolerance, max_iterations)
    elapsed = time.perf_counter() - started

    return u, SolverReport(
        method=method,
        iterations=iterations,
        relative_residual=_relative_residual(u, b, spacing),
        seconds=elapsed,
        converged=converged,
    )


def solve_neumann(b: np.ndarray) -> np.ndarray:
    """Resuelve ``laplacian(u) = b`` con condiciones de Neumann homogéneas.

    Reflejar el dominio a través de sus bordes convierte el laplaciano de cinco
    puntos en un operador circulante que la transformada discreta del coseno
    diagonaliza, con autovalores ``2 cos(pi i / m) + 2 cos(pi j / n) - 4``.  La
    resolución es entonces una transformada directa, una división y una
    transformada inversa, lo cual es exacto y cuesta ``O(N log N)`` sin iterar
    nada.

    Las condiciones de Neumann dejan la solución indeterminada salvo una
    constante aditiva, que aparece como un autovalor nulo; ese coeficiente se
    pone a cero, lo que selecciona la solución de media nula.  Esta es la
    condición de contorno correcta cuando el campo guía está definido sobre toda
    la imagen y ninguna parte de ella debe quedar fijada, como en el mapeo tonal.

    Parameters
    ----------
    b : ndarray, shape (m, n)
        Término independiente, con la condición de compatibilidad
        ``sum(b) == 0`` implícita; cualquier componente constante se descarta.

    Returns
    -------
    ndarray, shape (m, n)
        Solución de media nula.
    """
    b = np.asarray(b, dtype=float)
    rows, columns = b.shape
    eigenvalues = (
        2.0 * np.cos(np.pi * np.arange(rows) / rows)[:, None]
        + 2.0 * np.cos(np.pi * np.arange(columns) / columns)[None, :]
        - 4.0
    )
    eigenvalues[0, 0] = 1.0

    transformed = dctn(b, type=2, norm="ortho") / eigenvalues
    transformed[0, 0] = 0.0
    return idctn(transformed, type=2, norm="ortho")
