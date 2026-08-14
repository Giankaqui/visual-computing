# Gradient-Domain Image Processing

Seamless compositing, texture flattening, local contrast enhancement and high
dynamic range tone mapping, all built on the same computation: choose a target
gradient field, then reconstruct the image whose gradient is closest to it. The
integration step is implemented four ways, from a sparse direct factorization to
a geometric multigrid V-cycle written from scratch.

![Seamless cloning](docs/seamless_cloning.png)

## The one idea

Everything here minimizes

```
min_u  ∫ |∇u − v|²        subject to boundary conditions,
```

where `v` is a target gradient field that is not, in general, the gradient of any
image. The normal equations of that least-squares problem are the Poisson
equation `∇²u = ∇·v`, so the applications differ only in how `v` is chosen:

| Application | Guidance field | Boundary condition |
| --- | --- | --- |
| Seamless cloning | source gradients inside the selection | destination values around it |
| Mixed gradients | the larger of source and destination gradient | destination values |
| Texture flattening | image gradients masked by an edge map | the image itself |
| Local contrast | image gradients rescaled by `(α/‖∇f‖)^β` | the surrounding pixels |
| Tone mapping | log-radiance gradients attenuated multiscale | Neumann, nothing pinned |

Because the discrete gradient and divergence in
[operators.py](src/gradient_domain/operators.py) are an exact adjoint pair, the
resulting system is symmetric, which is what makes conjugate gradients and
multigrid applicable at all. The adjoint relation is checked directly in the
tests rather than assumed.

## Solvers

Four methods solve the same Dirichlet system, which is what makes the comparison
below meaningful.

**Direct.** Sparse LU of the five-point Laplacian. Exact, and the right choice
when one operator serves many right-hand sides, since the factorization is
computed once and reused across colour channels. Fill-in is what eventually
rules it out.

**Conjugate gradients.** Matrix free, memory proportional to the grid. The
iteration count grows with the condition number of the Laplacian, which grows
with the number of pixels, so the total work is superlinear.

**Multigrid.** Relaxation kills high-frequency error in a few sweeps and then
stalls, because the remaining error is smooth. Error that is smooth on a fine
grid is not smooth on a grid twice as coarse, so a V-cycle relaxes it there
instead, at a quarter of the cost, recursively. Grid transfers are bilinear
interpolation and its scaled adjoint, and the smoother is a red-black
Gauss-Seidel sweep written as two array expressions.

**Multigrid-preconditioned CG.** One V-cycle as the preconditioner. Coarsening
maps `m` interior points to `(m − 1) // 2`, which is exact only when the size is
odd; on arbitrary image sizes the plain cycle loses some of its convergence
factor, and wrapping it in conjugate gradients recovers it. This is the default.

A fifth solver handles the Neumann problem that tone mapping poses. Reflecting
the domain turns the Laplacian into an operator the discrete cosine transform
diagonalizes, so that solve is one transform, a division and one inverse
transform, exact and non-iterative.

## Scaling

Each solver runs on the same manufactured problem, driven to a relative residual
of `1e-8`. Timings are single-threaded on an Apple M-series CPU.

![Solver scaling](docs/solver_scaling.png)

| Grid | Unknowns | Method | Iterations | Seconds |
| --- | ---: | --- | ---: | ---: |
| 63 × 63 | 3 969 | direct | – | 0.006 |
| 63 × 63 | 3 969 | cg | 174 | 0.007 |
| 63 × 63 | 3 969 | mgcg | 7 | 0.017 |
| 255 × 255 | 65 025 | direct | – | 0.139 |
| 255 × 255 | 65 025 | cg | 507 | 0.167 |
| 255 × 255 | 65 025 | mgcg | 7 | 0.034 |
| 511 × 511 | 261 121 | direct | – | 0.892 |
| 511 × 511 | 261 121 | cg | 994 | 2.450 |
| 511 × 511 | 261 121 | mgcg | 7 | 0.104 |
| 1023 × 1023 | 1 046 529 | cg | 1 866 | 24.230 |
| 1023 × 1023 | 1 046 529 | mgcg | 7 | 0.459 |

The iteration counts are the point. Conjugate gradients needs 174, then 507,
then 1 866 iterations as the grid grows, roughly in proportion to the side
length, exactly as the condition number of the Laplacian predicts. The multigrid
variants need seven, at every size. At one megapixel that is a factor of 53 in
wall-clock time, and the gap widens with resolution.

Reproduce with

```bash
gradient-domain benchmark --sizes 63 127 255 511 1023
```

## Applications

### Seamless cloning

Copying pixels transfers the source's absolute colour, which rarely matches the
destination. Copying gradients transfers only the relative variation; the
absolute level comes from the destination through the boundary condition, so the
insert takes on the surrounding illumination. The balloon above is composited
from a patch photographed a stop brighter and much cooler than the dusk scene it
lands in, and none of that difference survives.

Two formulations are implemented, and the difference is worth knowing.

* **Mask domain.** Unknowns are the selected pixels; the ring around them
  supplies Dirichlet values. This is Pérez, Gangnet and Blake (2003). The
  destination is preserved exactly outside the selection, which is what an
  editing tool should guarantee, but the domain is irregular, so the system has
  to be assembled and factorized explicitly.
* **Rectangle domain.** Unknowns are a rectangle around the selection, with the
  source gradient inside and the destination gradient outside. The domain is
  regular, so multigrid applies. Outside the selection the answer differs from
  the destination by a harmonic function with zero boundary values: on this
  example, `3.5e-4` on average.

The tests assert both the exactness of the first and the closeness of the second.

The seam only disappears if it runs where source and destination are plausibly
similar. The selection here is a disc noticeably larger than the balloon, so the
boundary lies in the source's own sky. A selection cut tight around the object
puts the boundary across the object itself, and the illumination difference is
then absorbed into the object's colours rather than into its surroundings.

### Texture flattening

![Texture flattening](docs/texture_flattening.png)

Multiplying the image gradient by an edge indicator and integrating forces every
non-edge region to be as flat as its boundaries allow. Structure survives,
texture does not.

### Local contrast

![Local contrast](docs/illumination_change.png)

Remapping gradient magnitudes by `(α/‖∇f‖)^β` inside a selection amplifies small
gradients and attenuates large ones, which lifts detail out of a shadowed
region. The boundary condition guarantees the result meets the untouched image
continuously, so there is no edge at the selection.

### Tone mapping

![Tone mapping](docs/tone_mapping.png)

The synthetic radiance map spans 5.5 decades; a display covers about two. Any
global curve has to choose, which is what the first two panels show. The
gradient-domain method (Fattal, Lischinski and Werman, 2002) attenuates
gradients by a factor that decreases with their magnitude, so the large jumps
that carry dynamic range shrink and the small ones that carry detail do not.

The attenuation is computed on a Gaussian pyramid and propagated coarse to fine.
That matters: a large edge is not a single large gradient at full resolution, it
is a ramp spread over many pixels, each of them moderate. Computing the factor
at full resolution alone would leave those ramps untouched.

Note the sign convention, which is a common source of confusion. Here `β = 1` is
the identity and smaller values compress harder, following the original paper.
The exponent in the local-contrast operator above runs the other way, following
Pérez et al.; both are implemented under their own conventions and both are
documented where they are defined.

## Usage

```bash
pip install -e gradient-domain
gradient-domain demo --output outputs
```

Individual demonstrations:

```bash
gradient-domain clone     --output outputs --domain mask
gradient-domain flatten   --output outputs --method mgcg
gradient-domain relight   --output outputs --alpha 0.05 --beta 0.5
gradient-domain tonemap   --output outputs --beta 0.88
gradient-domain benchmark --output outputs --sizes 63 127 255 511 1023
```

All inputs are generated procedurally, so nothing has to be downloaded and every
figure in this README is reproducible from a clean checkout.

Using the library directly:

```python
import numpy as np
from gradient_domain import seamless_clone, make_compositing_example

example = make_compositing_example()
result, reports = seamless_clone(
    example.source, example.target, example.mask, example.offset,
    mode="mixed", domain="rectangle", method="mgcg",
)
print(reports[0])
```

## Layout

```
src/gradient_domain/
  operators.py   gradient, divergence, Laplacian, boundary folding
  multigrid.py   V-cycle, red-black Gauss-Seidel, grid transfers
  solvers.py     direct, CG, multigrid, MGCG, and the DCT Neumann solver
  poisson.py     guidance fields and the editing operations
  hdr.py         multiscale gradient attenuation and tone mapping
  synthetic.py   procedural images, including a 5.5-decade radiance map
  benchmark.py   the scaling study
  visualize.py   figures
  cli.py         command line interface
tests/           42 tests, about one second
```

## Scope

The multigrid solver handles rectangular Dirichlet domains. Irregular selections
go through the direct factorization instead, which is exact and fast enough at
the sizes an interactive selection produces, but does not scale the way the
V-cycle does. Extending multigrid to irregular domains needs the mask coarsened
alongside the grid, or an algebraic construction of the coarse operators; both
are real work and neither is here.

Colour channels are solved independently, which is standard and is why the
factorization is reused rather than recomputed. Nothing enforces consistency
between channels, so a guidance field that is inconsistent across them will
produce a hue shift rather than an error.

## References

* Pérez, Gangnet and Blake, *Poisson Image Editing*, SIGGRAPH 2003.
* Fattal, Lischinski and Werman, *Gradient Domain High Dynamic Range Compression*, SIGGRAPH 2002.
* Agarwala, *Efficient Gradient-Domain Compositing Using Quadtrees*, SIGGRAPH 2007.
* Bhat, Zitnick, Cohen and Curless, *GradientShop: A Gradient-Domain Optimization Framework for Image and Video Filtering*, TOG 2010.
* Briggs, Henson and McCormick, *A Multigrid Tutorial*, 2nd edition, SIAM 2000.
* Wang, Bovik, Sheikh and Simoncelli, *Image Quality Assessment: From Error Visibility to Structural Similarity*, TIP 2004.
