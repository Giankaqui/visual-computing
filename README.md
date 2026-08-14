# Visual Computing Projects

Three self-contained projects covering the path from photographs to a rendered
3D scene, and the gradient-domain machinery that image compositing and tone
mapping share with it. Each is a separate installable package with its own
tests, benchmarks and documentation.

| Project | What it does | Core technique |
| --- | --- | --- |
| [structure-from-motion](structure-from-motion) | Recovers camera poses and a sparse point cloud from images | Five-point minimal solver, Schur-complement bundle adjustment |
| [gaussian-splatting](gaussian-splatting) | Fits a 3D scene to posed images and renders novel views | EWA projection, differentiable tile rasterizer, adaptive density control |
| [gradient-domain](gradient-domain) | Composites, flattens and tone maps images | Poisson integration with a geometric multigrid solver |

An [interactive demo](interactive-demo) drives all three from the browser, with
the sliders wired to the same library code the command line calls.

```bash
python interactive-demo/app.py
```

The first two compose. Structure from motion writes `cameras.json` and
`points.ply`; the splatting trainer reads exactly those and initializes from
them, which is the same handover a production pipeline performs between a sparse
reconstructor and a renderer.

```bash
sfm reconstruct photos/ --output scene/
gsplat train --scene scene/ --images photos/ --output model/
```

That chain wants real photographs. The procedural scene bundled with the
splatting project is a novel-view benchmark rather than a feature-matching one,
and reconstructing it from its own renders registers only a fraction of the
views, for [reasons that are about the images](gaussian-splatting#scope) rather
than about the pipeline.

## What is implemented here rather than called

The point of these projects is the algorithms, so the parts that carry the ideas
are written out rather than delegated:

* the five-point relative pose solver, including the symbolic construction of
  the ten cubic constraints and the action-matrix eigenvalue solve;
* sparse bundle adjustment with the Schur complement, a local `SO(3)`
  parameterization and a robust loss;
* the EWA projection of anisotropic 3D Gaussians and a tile-based differentiable
  rasterizer, with activation checkpointing to bound its memory;
* the density control that decides how many primitives a scene needs, including
  the optimizer-state surgery that keeps Adam consistent across it;
* a geometric multigrid V-cycle with red-black Gauss-Seidel smoothing and
  adjoint grid transfers.

Third-party code is used where it is not the subject: OpenCV for image decoding
and SIFT description, SciPy for sparse factorization and transforms, PyTorch for
automatic differentiation and array operations.

## Running everything

Each project installs independently and needs no downloaded data; every
demonstration generates its own inputs.

```bash
python -m venv .venv && source .venv/bin/activate

pip install -e structure-from-motion && sfm demo --output out/sfm
pip install -e gaussian-splatting   && gsplat train --scene synthetic --output out/gsplat
pip install -e gradient-domain      && gradient-domain demo --output out/gradient
```

Tests, per project:

```bash
cd structure-from-motion && pytest -q
```

## Results at a glance

**Structure from motion.** Twelve views of an 800-point scene, half a pixel of
measurement noise and 15 percent gross outliers: every view registered, median
rotation error 0.007 degrees, structure error 0.04 percent of the scene
diameter. [Details](structure-from-motion#accuracy-on-the-synthetic-benchmark)

**Gradient domain.** At one megapixel, conjugate gradients needs 1 866
iterations to reach a relative residual of `1e-8` and multigrid-preconditioned
CG needs 7, a factor of 53 in wall-clock time.
[Details](gradient-domain#scaling)

**Gaussian splatting.** Forty views of a ray-traced scene at 200 x 150, starting
from ten thousand randomly placed primitives and no point cloud: 24.23 dB and
0.907 SSIM on held-out views, half a decibel below the training views.
[Details](gaussian-splatting#results)

## Conventions shared across the projects

Cameras are world-to-camera, `x_camera = R x_world + t`, with the optical axis
along `+z` and `+y` pointing down in the image; this matches OpenCV and COLMAP,
so poses move between the projects without a change of basis. Images are float
arrays in `[0, 1]` with channels last. Every randomized routine takes a seed and
is reproducible.

## Licence

MIT. See [LICENSE](LICENSE).
