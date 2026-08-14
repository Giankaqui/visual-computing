# Interactive Demo

A browser interface for the three projects. Every panel calls the same library
code the command line tools do, so what you see is the real pipeline running on
your machine, not a recording of one.

## Running it

From the repository root, with the three projects installed:

```bash
pip install -e structure-from-motion -e gaussian-splatting -e gradient-domain
pip install -r interactive-demo/requirements.txt

python interactive-demo/app.py
```

It opens at `http://127.0.0.1:7860`. Add `--share` for a temporary public URL,
`--port` to move it, `--no-browser` to keep it from opening a window.

The Gaussian splatting panel needs a trained checkpoint at
`gaussian-splatting/docs/model.npz`. If it is missing, that panel says so and
tells you the command to produce one; the other two work regardless.

## What each panel does

**Structure from motion.** Generates a synthetic scene with known ground truth,
reconstructs it, and reports the error against that truth. The two controls that
matter are the measurement noise and the fraction of gross outliers: raise
either and you can watch the five-point solver, the robust estimators and the
bundle adjuster degrade in a specific order. The 3D view is orbitable, so the
camera trajectory and the recovered structure can be inspected from any angle.
A run takes a few seconds, so it sits behind a button.

**Gaussian splatting.** Renders the trained model from any viewpoint on the
orbit and ray traces the same camera analytically beside it. Neither viewpoint
was in the training set, so the difference between the two panes is
generalization rather than fit. Push the elevation to its extremes, where no
training camera ever went, and the representation starts to come apart in the
way splatting models do. Each render takes a fraction of a second, so the view
follows the sliders.

**Gradient domain.** Four operations across four sub-tabs: seamless cloning,
tone mapping, texture flattening and local contrast. Each solve is tens of
milliseconds, so the result follows the controls directly. Two things are worth
sweeping: the tone-mapping exponent, whose convention runs backwards from the
obvious one, and the choice of solver on the rectangle domain, where the
iteration counts of conjugate gradients and the multigrid variants are printed
side by side.

## Layout

```
interactive-demo/
  app.py                   assembles the three tabs
  common.py                image conversion, metric tables, camera placement
  reconstruction_panel.py  structure from motion, with an orbitable 3D view
  splatting_panel.py       novel views against a ray-traced reference
  gradient_panel.py        the four gradient-domain operations
```

## Scope

This is a demonstration surface, not part of any of the three packages: nothing
here is imported by them, and none of it is covered by their tests. It is
deliberately single-user and single-process, so two people pointing a browser at
the same instance will queue behind each other on the GPU.
