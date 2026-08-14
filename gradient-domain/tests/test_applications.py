"""Checks on the editing operations and on tone mapping.

Each application has an invariant that has to hold regardless of the images
involved, which is what these tests exercise: cloning an image into itself must
change nothing, the masked formulation must leave the destination untouched
outside the selection, flattening with every edge retained must be the identity,
and tone mapping must compress the dynamic range without inverting it.
"""

from __future__ import annotations

import numpy as np
import pytest

from gradient_domain.hdr import ToneMapConfig, attenuation_field, tone_map
from gradient_domain.poisson import (
    GuidanceField,
    illumination_change,
    seamless_clone,
    solve_dirichlet,
    solve_masked,
    texture_flatten,
)
from gradient_domain.synthetic import (
    make_compositing_example,
    make_radiance_map,
    make_texture_example,
)


@pytest.fixture(scope="module")
def example():
    return make_compositing_example(shape=(160, 240), seed=0)


def _decades(image: np.ndarray) -> float:
    """Log ratio between the first and ninety-ninth luminance percentiles."""
    luminance = np.maximum(image @ np.array([0.2126, 0.7152, 0.0722]), 1e-8)
    low, high = np.percentile(luminance, [1, 99])
    return float(np.log10(max(high, 1e-8) / max(low, 1e-8)))


def _disc(shape: tuple[int, int], radius: float) -> np.ndarray:
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
    centre = (shape[0] / 2, shape[1] / 2)
    return (rows - centre[0]) ** 2 + (columns - centre[1]) ** 2 < radius**2


def test_integrating_an_image_reproduces_it() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((40, 52, 3))
    result, reports = solve_dirichlet(GuidanceField.of(image), image, method="direct")

    assert np.abs(result - image).max() < 1e-9
    assert len(reports) == 3


def test_cloning_an_image_into_itself_changes_nothing(example) -> None:
    mask = _disc(example.target.shape[:2], 30.0)
    field = GuidanceField.of(example.target)
    result, _ = solve_masked(field, example.target, mask)
    assert np.abs(result - example.target).max() < 1e-9


def test_masked_domain_leaves_the_destination_untouched(example) -> None:
    result, _ = seamless_clone(
        example.source, example.target, example.mask, example.offset, domain="mask"
    )
    placed = np.zeros(example.target.shape[:2], dtype=bool)
    row, column = example.offset
    height, width = example.source.shape[:2]
    placed[row : row + height, column : column + width] = example.mask

    assert np.array_equal(result[~placed], example.target[~placed])


def test_the_two_formulations_agree_up_to_a_harmonic_correction(example) -> None:
    masked, _ = seamless_clone(
        example.source, example.target, example.mask, example.offset, domain="mask"
    )
    rectangle, _ = seamless_clone(
        example.source, example.target, example.mask, example.offset,
        domain="rectangle", method="direct",
    )
    difference = np.abs(masked - rectangle)
    assert difference.mean() < 5e-3
    assert difference.max() < 0.2


def _boundary_ring(mask: np.ndarray) -> np.ndarray:
    """Pixels where the mask changes value, that is the seam itself."""
    height, width = mask.shape
    padded = np.pad(mask, 1)
    dilated = np.zeros_like(mask)
    eroded = np.ones_like(mask)
    for row_offset, column_offset in ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)):
        shifted = padded[
            1 + row_offset : 1 + row_offset + height, 1 + column_offset : 1 + column_offset + width
        ]
        dilated |= shifted
        eroded &= shifted
    return dilated & ~eroded


def test_cloning_removes_the_seam_that_copying_leaves(example) -> None:
    naive = example.naive_composite()
    blended, _ = seamless_clone(
        example.source, example.target, example.mask, example.offset, domain="mask"
    )

    row, column = example.offset
    height, width = example.source.shape[:2]
    placed = np.zeros(example.target.shape[:2], dtype=bool)
    placed[row : row + height, column : column + width] = example.mask
    ring = _boundary_ring(placed)

    def seam_energy(image: np.ndarray) -> float:
        gradient_magnitude = np.abs(np.diff(image, axis=0)).mean(axis=2)
        return float(gradient_magnitude[ring[:-1]].mean())

    assert seam_energy(blended) < 0.25 * seam_energy(naive)


@pytest.mark.parametrize("mode", ["import", "mixed", "average"])
def test_every_blend_mode_runs(example, mode: str) -> None:
    result, _ = seamless_clone(
        example.source, example.target, example.mask, example.offset, mode=mode, domain="mask"
    )
    assert result.shape == example.target.shape
    assert np.isfinite(result).all()


def test_unknown_mode_and_domain_are_rejected(example) -> None:
    with pytest.raises(ValueError):
        seamless_clone(example.source, example.target, example.mask, example.offset, mode="blend")
    with pytest.raises(ValueError):
        seamless_clone(
            example.source, example.target, example.mask, example.offset, domain="triangle"
        )


def test_patch_outside_the_target_is_rejected(example) -> None:
    with pytest.raises(ValueError):
        seamless_clone(
            example.source, example.target, example.mask, offset=(10_000, 0), domain="mask"
        )


def test_flattening_with_every_edge_retained_is_the_identity() -> None:
    image, _ = make_texture_example(shape=(96, 96), seed=1)
    result, _ = texture_flatten(image, np.ones(image.shape[:2]), method="direct")
    assert np.abs(result - image).max() < 1e-9


def test_flattening_reduces_texture_energy() -> None:
    image, edges = make_texture_example(shape=(128, 128), seed=1)
    flattened, _ = texture_flatten(image, edges, method="mgcg")

    def texture_energy(candidate: np.ndarray) -> float:
        interior = ~edges.astype(bool)
        variation = np.abs(np.diff(candidate, axis=1)).mean(axis=2)
        return float(variation[interior[:, :-1]].mean())

    assert texture_energy(flattened) < 0.2 * texture_energy(image)


def test_illumination_change_lifts_a_dark_selection() -> None:
    rows, columns = np.mgrid[0:96, 0:96]
    image = np.repeat((0.05 + 0.02 * np.sin(columns * 0.4))[..., None], 3, axis=2)
    mask = _disc((96, 96), 30.0)

    result, _ = illumination_change(image, mask, alpha=0.4, beta=0.6)
    interior = mask & ~_disc((96, 96), 24.0)
    assert result[interior].std() > image[interior].std()
    assert np.array_equal(result[~mask], image[~mask])


def test_tone_mapping_compresses_the_dynamic_range() -> None:
    radiance = make_radiance_map(shape=(128, 176), seed=2)
    mapped, factor = tone_map(radiance)

    assert mapped.shape == radiance.shape
    assert mapped.min() >= 0.0 and mapped.max() <= 1.0
    assert factor.shape == radiance.shape[:2]

    assert _decades(mapped) < 0.4 * _decades(radiance)


def test_smaller_beta_compresses_more() -> None:
    # The attenuation factor is (g / alpha) ** (beta - 1), so beta = 1 is the
    # identity and smaller values compress harder.
    radiance = make_radiance_map(shape=(96, 128), seed=2)
    gentle, _ = tone_map(radiance, ToneMapConfig(beta=0.95))
    strong, _ = tone_map(radiance, ToneMapConfig(beta=0.4))
    assert _decades(strong) < _decades(gentle)


def test_attenuation_leaves_a_flat_image_alone() -> None:
    factor = attenuation_field(np.zeros((64, 64)), ToneMapConfig())
    assert np.isfinite(factor).all()
    assert factor.min() > 0.0


def test_tone_mapping_rejects_a_grayscale_input() -> None:
    with pytest.raises(ValueError):
        tone_map(np.ones((16, 16)))
