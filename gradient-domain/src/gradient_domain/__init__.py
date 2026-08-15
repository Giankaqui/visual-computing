"""Procesamiento en el dominio del gradiente con solvers de Poisson multigrid.

El clonado sin costura, el aplanado de textura, la compresión de rango local y el
mapeo tonal de alto rango dinámico son todos el mismo cálculo: elegir un campo de
gradiente objetivo y encontrar la imagen cuyo gradiente más se le parece.  El
paquete implementa ese paso de integración de cuatro maneras, desde una
factorización dispersa directa hasta un ciclo V multigrid geométrico, y encima
las aplicaciones.
"""

from .benchmark import BenchmarkRecord, BenchmarkTable, manufactured_problem, solver_scaling
from .hdr import ToneMapConfig, attenuation_field, tone_map
from .multigrid import MultigridOptions, MultigridReport, solve, v_cycle
from .operators import divergence, fold_boundary, gradient, laplacian, sparse_laplacian
from .poisson import (
    GuidanceField,
    illumination_change,
    seamless_clone,
    solve_dirichlet,
    solve_masked,
    texture_flatten,
)
from .solvers import SOLVERS, SolverReport, solve_neumann, solve_system
from .synthetic import (
    CompositingExample,
    make_compositing_example,
    make_radiance_map,
    make_texture_example,
    value_noise,
)

__all__ = [
    "SOLVERS",
    "BenchmarkRecord",
    "BenchmarkTable",
    "CompositingExample",
    "GuidanceField",
    "MultigridOptions",
    "MultigridReport",
    "SolverReport",
    "ToneMapConfig",
    "attenuation_field",
    "divergence",
    "fold_boundary",
    "gradient",
    "illumination_change",
    "laplacian",
    "make_compositing_example",
    "make_radiance_map",
    "make_texture_example",
    "manufactured_problem",
    "seamless_clone",
    "solve",
    "solve_dirichlet",
    "solve_masked",
    "solve_neumann",
    "solve_system",
    "solver_scaling",
    "sparse_laplacian",
    "texture_flatten",
    "tone_map",
    "v_cycle",
    "value_noise",
]

__version__ = "0.1.0"
