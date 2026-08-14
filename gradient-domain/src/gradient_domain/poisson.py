"""Gradient-domain image editing.

Every operation here has the same shape.  A guidance field is built by modifying
the gradients of one or more input images, and the output is the image whose
gradient best matches that field subject to boundary conditions.  What changes
between applications is only how the guidance field is chosen.

Two domains are supported, and the difference is worth understanding.

``mask``
    The unknowns are the pixels inside the selection, and the pixels just outside
    it supply Dirichlet values.  This is the formulation of Perez, Gangnet and
    Blake (2003).  The output equals the target exactly outside the selection,
    which is usually what an editing tool should guarantee, but the domain is
    irregular, so the system has to be assembled explicitly and factorized.

``rectangle``
    The unknowns are a rectangle around the selection, the guidance field is the
    source gradient inside it and the target gradient outside, and the rectangle
    border supplies the boundary values.  The residual outside the selection is
    then a harmonic function with zero boundary values, so it is small but not
    exactly zero.  In exchange the domain is regular, which is what geometric
    multigrid needs, and the solve is linear in the number of pixels.

The two agree to within that harmonic correction, which the tests verify.
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
    """A target gradient field.

    Attributes
    ----------
    gx, gy : ndarray, shape (h, w) or (h, w, c)
        Desired horizontal and vertical derivatives.
    """

    gx: np.ndarray
    gy: np.ndarray

    @classmethod
    def of(cls, image: np.ndarray) -> GuidanceField:
        """The field that reproduces ``image`` exactly."""
        return cls(*gradient(image))

    def select(self, mask: np.ndarray, other: GuidanceField) -> GuidanceField:
        """Take this field inside ``mask`` and ``other`` outside it."""
        selector = mask[..., None] if self.gx.ndim == 3 else mask
        return GuidanceField(
            np.where(selector, self.gx, other.gx), np.where(selector, self.gy, other.gy)
        )

    def mix_by_magnitude(self, other: GuidanceField) -> GuidanceField:
        """Keep, per component, whichever field has the larger magnitude.

        This is the mixed-gradient variant: it lets strong structure in the
        destination survive underneath a source region that is comparatively
        flat, which is what makes pasting onto a textured background look right
        instead of erasing it.
        """
        return GuidanceField(
            np.where(np.abs(self.gx) >= np.abs(other.gx), self.gx, other.gx),
            np.where(np.abs(self.gy) >= np.abs(other.gy), self.gy, other.gy),
        )

    def scaled(self, factor: np.ndarray | float) -> GuidanceField:
        """Multiply both components by a scalar or a per-pixel factor."""
        return GuidanceField(self.gx * factor, self.gy * factor)


def _as_channels(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a ``(h, w, c)`` view and whether the input was single channel."""
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
    """Integrate a guidance field on a rectangle with fixed borders.

    Parameters
    ----------
    field : GuidanceField
        Target gradients on the full rectangle.
    boundary : ndarray, shape (h, w) or (h, w, c)
        Image whose one-pixel border supplies the Dirichlet values; its interior
        is ignored.
    method : str
        Name of a solver in :data:`gradient_domain.solvers.SOLVERS`.
    tolerance : float
    max_iterations : int

    Returns
    -------
    image : ndarray
        Same shape as ``boundary``, equal to it on the border.
    reports : list of SolverReport
        One entry per colour channel.
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
    """Integrate a guidance field on the pixels selected by ``mask``.

    Pixels outside the mask keep their value and act as Dirichlet data.  The
    system is assembled once and factorized once, then reused for every colour
    channel, which is where most of the saving comes from on a multi-channel
    image.

    Parameters
    ----------
    field : GuidanceField
        Target gradients, defined on the whole image.
    image : ndarray, shape (h, w) or (h, w, c)
        Destination image; supplies the boundary values.
    mask : ndarray of bool, shape (h, w)
        Pixels to solve for.

    Returns
    -------
    result : ndarray
        Same shape as ``image``.
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
    """Paste a source patch and its mask into a target-sized canvas."""
    row, column = offset
    height, width = source.shape[:2]
    if row < 0 or column < 0 or row + height > target_shape[0] or column + width > target_shape[1]:
        raise ValueError("the source patch does not fit inside the target at this offset")

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
    """Paste a region of ``source`` into ``target`` without a visible seam.

    Copying pixels transfers the source's absolute colour, which almost never
    matches the destination.  Copying gradients instead transfers only the
    relative variation, and the absolute level is recovered from the destination
    through the boundary condition, so the insert takes on the surrounding
    illumination.

    Parameters
    ----------
    source : ndarray, shape (sh, sw) or (sh, sw, c)
        Patch to insert.
    target : ndarray, shape (h, w) or (h, w, c)
        Destination image.
    mask : ndarray of bool, shape (sh, sw)
        Selection inside the source patch.
    offset : tuple of int
        Position of the patch's top-left corner in the target.
    mode : {'import', 'mixed', 'average'}
        How the source and destination gradients are combined inside the mask.
    domain : {'mask', 'rectangle'}
        Which formulation to use; see the module docstring.
    method : str
        Solver name, used only by the rectangle formulation.
    margin : int
        Padding around the mask bounding box for the rectangle formulation.

    Returns
    -------
    result : ndarray
        Same shape as ``target``.
    reports : list of SolverReport

    Raises
    ------
    ValueError
        For an unknown ``mode`` or ``domain``, or if the patch does not fit.
    """
    target = np.asarray(target, dtype=float)
    source = np.asarray(source, dtype=float)
    if source.ndim != target.ndim:
        raise ValueError("source and target must have the same number of dimensions")

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
        raise ValueError(f"unknown mode {mode!r}")

    field = inside.select(placed_mask, target_field)

    if domain == "mask":
        return solve_masked(field, target, placed_mask)
    if domain != "rectangle":
        raise ValueError(f"unknown domain {domain!r}")

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
    """Remove texture while keeping the edges that define the shapes.

    The guidance field is the image gradient multiplied by an indicator that is
    one on the retained edges and zero elsewhere.  Integrating a field that is
    zero over a region forces that region to be as flat as the boundary
    conditions allow, so texture disappears and the large-scale structure stays.

    Parameters
    ----------
    image : ndarray, shape (h, w) or (h, w, c)
    edges : ndarray, shape (h, w)
        Weights in ``[0, 1]``; a binary edge map is the usual choice.
    method : str
        Solver name.
    mask : ndarray of bool, shape (h, w), optional
        Restricts the effect to a region, solved with the mask formulation.

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
    """Compress the local dynamic range inside a selection.

    The gradient magnitude is remapped by ``alpha ** beta * g ** (-beta)``, which
    attenuates large gradients more than small ones.  Applied inside a selection
    with the surrounding pixels fixed, this lifts detail out of shadows without
    changing the overall illumination of the image.

    Parameters
    ----------
    image : ndarray, shape (h, w) or (h, w, c)
        Values are expected in ``[0, 1]``.
    mask : ndarray of bool, shape (h, w)
    alpha : float
        Gradient magnitude that is left unchanged; smaller values darken less.
    beta : float
        Exponent of the attenuation, in ``[0, 1]``.  Note that the convention is
        the opposite of the one in :mod:`gradient_domain.hdr`: here zero leaves
        the image unchanged and one flattens it to a constant gradient
        magnitude, following Perez et al. rather than Fattal et al.
    epsilon : float
        Floor on the gradient magnitude, which keeps flat regions from being
        amplified without bound.

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
