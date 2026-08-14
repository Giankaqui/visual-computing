"""Gradient-domain image processing with multigrid Poisson solvers.

Seamless cloning, texture flattening, local range compression and high dynamic
range tone mapping are all the same computation: choose a target gradient field,
then find the image whose gradient is closest to it.  The package implements that
integration step four ways, from a sparse direct factorization to a geometric
multigrid V-cycle, and the applications on top of it.
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
