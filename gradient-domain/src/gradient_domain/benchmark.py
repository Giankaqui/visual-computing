"""Comparación de coste de los solvers de Poisson.

El benchmark resuelve el mismo problema manufacturado a resoluciones crecientes y
anota tiempo y número de iteraciones.  Se usa una solución manufacturada en lugar
de un problema de edición real para conocer la respuesta exacta y poder exigir a
todos los métodos el mismo residuo, que es la única forma de que los tiempos
signifiquen algo.

Lo que se espera que muestren los números: la factorización directa es
competitiva a tamaños pequeños y luego pierde por el llenado; el gradiente
conjugado necesita un número de iteraciones que crece con la malla, porque así lo
hace el número de condición del laplaciano; y las variantes multigrid necesitan
un número de ciclos que no crece, así que su trabajo total es proporcional al
número de píxeles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .operators import laplacian
from .solvers import solve_system

__all__ = ["BenchmarkRecord", "solver_scaling", "manufactured_problem"]


@dataclass
class BenchmarkRecord:
    """Una medición.

    Attributes
    ----------
    method : str
    shape : tuple of int
    unknowns : int
    seconds : float
    iterations : int
    relative_residual : float
    max_error : float
        Mayor diferencia absoluta respecto a la solución conocida.
    """

    method: str
    shape: tuple[int, int]
    unknowns: int
    seconds: float
    iterations: int
    relative_residual: float
    max_error: float

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "shape": list(self.shape),
            "unknowns": self.unknowns,
            "seconds": self.seconds,
            "iterations": self.iterations,
            "relative_residual": self.relative_residual,
            "max_error": self.max_error,
        }


def manufactured_problem(
    shape: tuple[int, int], seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Construye un término independiente cuya solución exacta se conoce.

    La solución mezcla unos pocos modos suaves con ruido blanco, así que contiene
    energía en todo el espectro.  Una solución puramente suave favorecería a
    multigrid y una puramente rugosa favorecería a la relajación.

    Parameters
    ----------
    shape : tuple of int
    seed : int

    Returns
    -------
    b : ndarray, shape (m, n)
        Término independiente con las condiciones de Dirichlet homogéneas ya
        plegadas dentro.
    solution : ndarray, shape (m, n)
    """
    rng = np.random.default_rng(seed)
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]] / max(shape)
    solution = (
        np.sin(3.0 * np.pi * columns) * np.sin(2.0 * np.pi * rows)
        + 0.4 * np.cos(11.0 * np.pi * columns) * np.sin(7.0 * np.pi * rows)
        + 0.25 * rng.standard_normal(shape)
    )
    solution[0] = solution[-1] = 0.0
    solution[:, 0] = solution[:, -1] = 0.0
    return laplacian(solution), solution


def solver_scaling(
    sizes: list[int],
    methods: list[str] | None = None,
    tolerance: float = 1e-8,
    max_iterations: int = 5000,
    direct_limit: int = 600,
    seed: int = 0,
) -> dict[str, list[BenchmarkRecord]]:
    """Cronometra cada solver sobre mallas cuadradas de los tamaños dados.

    Parameters
    ----------
    sizes : list of int
        Lados de las mallas cuadradas.
    methods : list of str o None
        Por defecto, los cuatro solvers.
    tolerance : float
        Residuo relativo objetivo.
    max_iterations : int
    direct_limit : int
        Las mallas mayores que esto se saltan para el método directo, cuya
        factorización pasa a dominar el coste tanto en tiempo como en memoria.
    seed : int

    Returns
    -------
    dict
        Asocia cada nombre de método con sus registros, ordenados por tamaño.
    """
    methods = methods or ["direct", "cg", "multigrid", "mgcg"]
    results: dict[str, list[BenchmarkRecord]] = {method: [] for method in methods}

    for size in sizes:
        b, solution = manufactured_problem((size, size), seed=seed)
        for method in methods:
            if method == "direct" and size > direct_limit:
                continue
            estimate, report = solve_system(
                b, method=method, tolerance=tolerance, max_iterations=max_iterations
            )
            results[method].append(
                BenchmarkRecord(
                    method=method,
                    shape=(size, size),
                    unknowns=size * size,
                    seconds=report.seconds,
                    iterations=report.iterations,
                    relative_residual=report.relative_residual,
                    max_error=float(np.abs(estimate - solution).max()),
                )
            )
    return results


@dataclass
class BenchmarkTable:
    """Vista formateada de un estudio de escalado."""

    results: dict[str, list[BenchmarkRecord]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Compone los registros como tabla Markdown ordenada por tamaño y método."""
        rows = [record for records in self.results.values() for record in records]
        rows.sort(key=lambda record: (record.unknowns, record.method))

        lines = [
            "| Malla | Incógnitas | Método | Iteraciones | Segundos | Error máx. |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
        for record in rows:
            grid = f"{record.shape[0]} x {record.shape[1]}"
            iterations = "-" if record.iterations == 0 else str(record.iterations)
            lines.append(
                f"| {grid} | {record.unknowns} | {record.method} | {iterations} | "
                f"{record.seconds:.3f} | {record.max_error:.2e} |"
            )
        return "\n".join(lines)
