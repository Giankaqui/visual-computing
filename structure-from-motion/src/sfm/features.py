"""Feature detection, description and matching.

Detection and description use the SIFT implementation shipped with OpenCV.
Matching is implemented here rather than delegated, because the pipeline needs
control over two properties that a black-box matcher does not expose: the exact
ratio-test statistic, and a mutual-consistency filter that runs before geometric
verification instead of after it.

The brute-force search is blocked over query descriptors so that the pairwise
distance matrix never exceeds a bounded amount of memory, which matters when a
pair of images contributes tens of thousands of descriptors each.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "FeatureSet",
    "detect_and_describe",
    "list_images",
    "load_image",
    "match_all_pairs",
    "match_descriptors",
]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class FeatureSet:
    """Features of a single image.

    Attributes
    ----------
    keypoints : ndarray, shape (n, 2)
        Subpixel keypoint locations in pixels.
    descriptors : ndarray of float32, shape (n, 128)
        L2-normalized SIFT descriptors.
    colors : ndarray of uint8, shape (n, 3)
        RGB sample at each keypoint, used to colour the output point cloud.
    size : tuple of int
        Image width and height.
    """

    keypoints: np.ndarray
    descriptors: np.ndarray
    colors: np.ndarray
    size: tuple[int, int]

    def __len__(self) -> int:
        return len(self.keypoints)


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as RGB uint8.

    Raises
    ------
    FileNotFoundError
        If the file cannot be decoded.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_and_describe(
    image: np.ndarray, max_features: int = 8000, contrast_threshold: float = 0.02
) -> FeatureSet:
    """Detect SIFT keypoints and compute their descriptors.

    Descriptors are L2 normalized so that squared Euclidean distances can be
    evaluated through inner products without a per-pair renormalization.

    Parameters
    ----------
    image : ndarray, shape (h, w, 3)
        RGB image.
    max_features : int
        Upper bound on retained keypoints, ordered by response.
    contrast_threshold : float
        SIFT contrast threshold; lower values keep more low-contrast features.

    Returns
    -------
    FeatureSet
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=contrast_threshold)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) == 0:
        empty = np.zeros((0, 2), dtype=float)
        return FeatureSet(empty, np.zeros((0, 128), np.float32), np.zeros((0, 3), np.uint8),
                          (image.shape[1], image.shape[0]))

    coordinates = np.array([kp.pt for kp in keypoints], dtype=float)
    descriptors = np.asarray(descriptors, dtype=np.float32)
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-8)

    rows = np.clip(np.round(coordinates[:, 1]).astype(int), 0, image.shape[0] - 1)
    cols = np.clip(np.round(coordinates[:, 0]).astype(int), 0, image.shape[1] - 1)
    return FeatureSet(
        keypoints=coordinates,
        descriptors=descriptors,
        colors=image[rows, cols],
        size=(image.shape[1], image.shape[0]),
    )


def _two_nearest(
    query: np.ndarray, database: np.ndarray, block_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices and squared distances of the two nearest neighbours."""
    indices = np.empty((len(query), 2), dtype=np.int64)
    distances = np.empty((len(query), 2), dtype=np.float32)
    database_norms = (database**2).sum(axis=1)

    for start in range(0, len(query), block_size):
        block = query[start : start + block_size]
        squared = (
            (block**2).sum(axis=1)[:, None] + database_norms[None, :] - 2.0 * block @ database.T
        )
        np.maximum(squared, 0.0, out=squared)
        nearest = np.argpartition(squared, 1, axis=1)[:, :2]
        block_distances = np.take_along_axis(squared, nearest, axis=1)
        order = np.argsort(block_distances, axis=1)
        indices[start : start + len(block)] = np.take_along_axis(nearest, order, axis=1)
        distances[start : start + len(block)] = np.take_along_axis(block_distances, order, axis=1)
    return indices, distances


def match_descriptors(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    ratio: float = 0.8,
    mutual: bool = True,
    block_size: int = 2048,
) -> np.ndarray:
    """Match two descriptor sets with Lowe's ratio test.

    A match is kept when the nearest neighbour is closer than ``ratio`` times the
    second nearest, which rejects descriptors whose neighbourhood is ambiguous.
    With ``mutual`` enabled the test is run in both directions and only
    reciprocal matches survive, which removes the many-to-one matches that
    repeated structure produces.

    Parameters
    ----------
    descriptors_a, descriptors_b : ndarray, shape (n, d)
    ratio : float
        Lowe's ratio on distances, not squared distances.
    mutual : bool
        Require reciprocal nearest neighbours.
    block_size : int
        Number of query descriptors handled per distance-matrix block.

    Returns
    -------
    ndarray of int64, shape (m, 2)
        Index pairs into the two descriptor sets.
    """
    if len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return np.zeros((0, 2), dtype=np.int64)

    squared_ratio = ratio**2
    forward_index, forward_distance = _two_nearest(descriptors_a, descriptors_b, block_size)
    keep = forward_distance[:, 0] < squared_ratio * np.maximum(forward_distance[:, 1], 1e-12)
    pairs = np.stack([np.flatnonzero(keep), forward_index[keep, 0]], axis=1)

    if mutual and len(pairs):
        backward_index, backward_distance = _two_nearest(descriptors_b, descriptors_a, block_size)
        consistent = backward_index[pairs[:, 1], 0] == pairs[:, 0]
        consistent &= backward_distance[pairs[:, 1], 0] < squared_ratio * np.maximum(
            backward_distance[pairs[:, 1], 1], 1e-12
        )
        pairs = pairs[consistent]
    return pairs.astype(np.int64)


def match_all_pairs(
    features: list[FeatureSet],
    ratio: float = 0.8,
    min_matches: int = 20,
    verbose: bool = False,
) -> dict[tuple[int, int], np.ndarray]:
    """Match every image pair exhaustively.

    Exhaustive matching is quadratic in the number of images and is the right
    choice for the unordered collections of a few dozen images this pipeline
    targets.  Larger collections need a vocabulary tree or a learned image
    retrieval step to shortlist pairs first.

    Parameters
    ----------
    features : list of FeatureSet
    ratio : float
        Lowe's ratio passed to :func:`match_descriptors`.
    min_matches : int
        Pairs with fewer matches are not reported.
    verbose : bool
        Print the number of matches per retained pair.

    Returns
    -------
    dict
        Maps ``(i, j)`` with ``i < j`` to an array of matched feature indices.
    """
    matches: dict[tuple[int, int], np.ndarray] = {}
    for view_a, view_b in combinations(range(len(features)), 2):
        pairs = match_descriptors(features[view_a].descriptors, features[view_b].descriptors, ratio)
        if len(pairs) >= min_matches:
            matches[(view_a, view_b)] = pairs
            if verbose:
                print(f"  {view_a:>3} - {view_b:<3} {len(pairs):>6} matches")
    return matches


def list_images(directory: str | Path) -> list[Path]:
    """Return the image files in a directory, sorted by name."""
    return sorted(
        path for path in Path(directory).iterdir() if path.suffix.lower() in _IMAGE_SUFFIXES
    )
