# 3D Gaussian Splatting

A differentiable renderer that represents a scene as a set of anisotropic 3D
Gaussians and fits them to posed images by gradient descent. Written entirely in
PyTorch: the EWA projection, the tile-based rasterizer, the spherical harmonic
colour model and the adaptive density control are all here, and the backward
pass comes from autograd rather than from a hand-written CUDA kernel.

## How it works

A primitive is a 3D Gaussian with a mean, an anisotropic covariance, an opacity
and a spherical harmonic expansion for view-dependent colour. Rendering it is
four steps.

**Project.** A perspective projection is not affine, so the image of a Gaussian
is not a Gaussian. EWA splatting linearizes the projection at each centre and
pushes the covariance through that linear map, `Σ₂ᴅ = J W Σ Wᵀ Jᵀ`. The
approximation is good while a primitive subtends a small angle, which is the
regime the representation operates in.

**Bin.** Compositing every primitive against every pixel costs `O(H·W·N)`. The
image is split into tiles instead, and each primitive is assigned to the tiles
its support overlaps, so the work becomes proportional to covered screen area.

**Composite.** Within a tile the primitives are depth sorted and blended front to
back, `C = Σᵢ cᵢ αᵢ Πⱼ<ᵢ (1 − αⱼ)`. The transmittance product looks sequential,
but it is a cumulative product along the depth axis, so a tile evaluates as a
handful of batched tensor operations that autograd differentiates directly. The
gradients are verified against central differences in the tests.

**Densify.** Gradient descent can move, resize and recolour primitives but cannot
change how many there are. Where the model is under-parameterized, a primitive's
projected centre receives a large and persistent gradient, because one splat is
being pulled towards several image features at once. Small primitives with a
large accumulated gradient are cloned; large ones are split into children drawn
from their own distribution.

## Parameterization

Every constrained quantity is stored through an unconstrained parameterization,
so gradient descent never has to be projected back onto a feasible set.

| Quantity | Stored as | Recovered by |
| --- | --- | --- |
| scale | logarithm | `exp` |
| opacity | logit | `sigmoid` |
| rotation | unnormalized quaternion | normalize, then to a matrix |
| covariance | scale and rotation | `R S Sᵀ Rᵀ` |
| colour | spherical harmonic coefficients | evaluate along the view direction |

Factoring the covariance rather than storing six free coefficients is what keeps
it positive semidefinite throughout optimization; an unconstrained symmetric
matrix drifts indefinite within a few hundred steps, and the screen-space conic
then has no interior.

## Implementation notes worth reading

### The per-tile cap, and the artefact it causes

The dense compositing intermediate holds one entry per (tile, primitive, pixel)
triple, so its size is the pixel count times the occupancy of the busiest tile.
That occupancy is data dependent and unbounded, so it is capped after the depth
sort, keeping the nearest primitives.

The cap is usually free, because once transmittance is spent the remaining
primitives cannot change the pixel. When it is *not* free the failure is
distinctive and easy to misread as a training problem: neighbouring tiles
truncate different numbers of primitives, and the render acquires visible
rectangular steps. This is what it looks like, at a cap of 512 on a model whose
busiest tile held 1499 primitives:

| cap | saturated tiles | unspent transmittance | PSNR vs. untruncated |
| ---: | ---: | ---: | ---: |
| 512 | 87 of 130 | 0.199 | 21.3 dB |
| 1024 | 27 of 130 | 0.032 | 43.5 dB |
| 2048 | 0 | 0.000 | exact |

`RenderOutput` reports both quantities and the trainer prints a warning the first
time the unspent transmittance exceeds one percent, which turns an assumption
into a measurement.

Tile size is a related and non-obvious trade-off. Dense work is the pixel count
times the busiest tile's occupancy; halving the tile side roughly halves that
occupancy and leaves the pixel count alone, while the number of primitive-tile
pairs to sort grows. Eight pixels is where the two balance for these scenes:

| tile | tiles | primitive-tile pairs | busiest tile | dense elements | render |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 900 | 718 483 | 886 | 26.9 M | 328 ms |
| 8 | 475 | 220 320 | 1 096 | 33.3 M | 242 ms |
| 16 | 130 | 85 035 | 1 499 | 49.9 M | 315 ms |
| 32 | 35 | 40 061 | 2 755 | 98.7 M | 924 ms |

### Activation checkpointing

Tiles are composited in chunks sized to a fixed element budget, and under
gradient each chunk is wrapped in `torch.utils.checkpoint`. Peak memory then
depends on the budget rather than on the image size, at the cost of recomputing
each chunk's forward pass during the backward pass. At 400 × 300 with 40 000
primitives on CPU:

| checkpointing | peak resident memory |
| --- | ---: |
| off | 6.7 GB |
| on | 3.2 GB |

The tests assert that the gradients are identical either way.

### Optimizer surgery

Densification changes the number of rows in every parameter tensor, so Adam's
first and second moment estimates have to be reindexed in step with them.
Dropping the state instead would restart the moment estimates for every
surviving primitive and visibly stall training after each pass. The helpers in
[densify.py](src/gsplat/densify.py) slice the moments for survivors and pad
zeros for newcomers, and the tests check both directly.

### A ceiling on the model size

The screen-space gradient threshold does not bound the model. It is a property
of the image resolution and the scene, so a value that produces a reasonable
model at one megapixel produces a wildly over-parameterized one at a tenth of
that. On this benchmark at 160 × 120, the published threshold of `2e-4` grew the
model to 110 751 primitives and held-out PSNR *fell* from 15.1 dB at iteration
1 000 to 11.4 dB at 6 000, with a five-decibel gap to the training views: the
classic signature of memorizing them. Raising the threshold and adding an
explicit ceiling closes the gap to a few tenths of a decibel. Growth stops at the
ceiling while pruning continues, so the model keeps improving by replacing
primitives rather than by adding them.

## Results

The benchmark scene is three spheres of different sizes and finishes on a
mottled stone floor, ray traced analytically so the poses are exact. Specular
highlights sweep across the spheres as the camera orbits, which only a
view-dependent colour model can reproduce; the floor carries aperiodic detail at
several scales, which is what forces the density control to split; and the cast
shadows are radiance discontinuities not aligned with any surface.

Forty training views on an orbit, eight held-out views interleaved half a step
between them, at 200 × 150. The model starts from 10 000 primitives placed
uniformly in a ball, with no point cloud to initialize from.

Held-out quality, the primitive count and the figures below all come from a
single run of the command that follows; it writes `training.png`,
`comparison.png` and `turntable.png` next to the checkpoint.

Reproduce with

```bash
gsplat train --scene synthetic --iterations 6000 --width 200 --height 150 \
             --init random --random-points 10000 --output out/
```

## Usage

```bash
pip install -e gaussian-splatting
```

Fit the procedural scene:

```bash
gsplat train --scene synthetic --output out/ --device auto
```

Fit a reconstruction from the [structure-from-motion](../structure-from-motion)
project, initializing from its sparse point cloud:

```bash
sfm reconstruct photos/ --output scene/
gsplat train --scene scene/ --images photos/ --output out/
```

Render novel views from a checkpoint:

```bash
gsplat render out/model.npz --output turntable.png --views 6 --width 320
```

Write the procedural scene out as images and poses, which is what the
reconstruction pipeline consumes:

```bash
gsplat export --output scene/ --views 24
```

`--device auto` selects Metal on Apple silicon, CUDA where available, and CPU
otherwise. Metal is roughly twice as fast as CPU on this workload.

## Layout

```
src/gsplat/
  spherical_harmonics.py  real basis up to degree three
  cameras.py              pinhole cameras, look-at, orbits
  gaussians.py            the model and its parameterization
  projection.py           EWA projection to screen-space conics
  rasterizer.py           tile binning, depth sort, differentiable compositing
  renderer.py             model and camera to image, in one call
  losses.py               L1, differentiable SSIM, PSNR
  densify.py              adaptive density control and optimizer surgery
  trainer.py              the optimization loop and its schedules
  scenes.py               the analytic ray tracer and its procedural texture
  datasets.py             procedural and on-disk datasets
  visualize.py            comparison, curve and turntable figures
  cli.py                  command line interface
tests/                    55 tests, about half a minute
```

## Scope

The rasterizer is written for clarity and portability, not for speed. A CUDA
kernel with a hand-written backward pass is one to two orders of magnitude
faster, and that is the right choice for scenes of millions of primitives at
megapixel resolution. What this implementation buys instead is a compositing
pass that is a dozen lines of tensor algebra, runs unchanged on CPU, Metal and
CUDA, and is checked against numerical differentiation.

Camera poses are held fixed; joint refinement of poses and geometry is not
implemented. Primitives beyond the per-tile cap are dropped rather than composited
into a background approximation. Exposure and white balance are assumed constant
across views, which is true of rendered data and rarely true of a real capture.

The procedural scene is a novel-view benchmark, not a feature-matching one. Three
smooth specular spheres over a ground plane, viewed from a full orbit, give SIFT
very little to work with: the spheres' appearance is view dependent, the floor is
seen at grazing angles that change sharply between views, and the background is
empty. A reconstruction from these renders registers only a fraction of the
views, which says something about the images rather than about the pipeline. The
handover between the two projects is a file interface, exercised by the tests and
by the `--scene <directory>` path above; demonstrating the full chain end to end
wants a real capture.

## References

* Kerbl, Kopanas, Leimkühler and Drettakis, *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023.
* Zwicker, Pfister, van Baar and Gross, *EWA Volume Splatting*, IEEE Visualization 2001.
* Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi and Ng, *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*, ECCV 2020.
* Wang, Bovik, Sheikh and Simoncelli, *Image Quality Assessment: From Error Visibility to Structural Similarity*, TIP 2004.
* Chen and Wang, *A Survey on 3D Gaussian Splatting*, 2024.
