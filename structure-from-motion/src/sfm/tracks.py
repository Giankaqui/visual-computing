"""Feature tracks built from pairwise matches with a disjoint-set forest.

A track is a set of image features that are believed to be projections of the
same 3D point.  Pairwise matches induce a graph over features; its connected
components are candidate tracks.  Two failure modes are handled explicitly.

A component that contains two features from the same image is inconsistent,
because one 3D point projects to a single location per view.  Such components
usually arise from repeated structure and are discarded rather than trimmed,
since choosing which of the conflicting features to keep is exactly the decision
the matcher already got wrong.

Components of length two carry no redundancy and are kept, but the caller is
expected to require a larger track length before promoting a point to the
reconstruction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

__all__ = ["TrackGraph", "build_tracks"]

FeatureKey = tuple[int, int]


class _DisjointSet:
    """Union-find with path halving and union by size."""

    def __init__(self) -> None:
        self._parent: dict[FeatureKey, FeatureKey] = {}
        self._size: dict[FeatureKey, int] = {}

    def find(self, item: FeatureKey) -> FeatureKey:
        parent = self._parent.setdefault(item, item)
        self._size.setdefault(item, 1)
        while parent != item:
            grandparent = self._parent[parent]
            self._parent[item] = grandparent
            item, parent = parent, grandparent
        return item

    def union(self, left: FeatureKey, right: FeatureKey) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self._size[left_root] < self._size[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        self._size[left_root] += self._size[right_root]

    def components(self) -> dict[FeatureKey, list[FeatureKey]]:
        groups: dict[FeatureKey, list[FeatureKey]] = defaultdict(list)
        for item in self._parent:
            groups[self.find(item)].append(item)
        return groups


@dataclass
class TrackGraph:
    """Feature tracks and the lookup tables the reconstruction loop needs.

    Attributes
    ----------
    tracks : list of list of tuple
        Each track is a list of ``(image_index, feature_index)`` pairs sorted by
        image index.
    feature_to_track : dict
        Maps ``(image_index, feature_index)`` to its position in ``tracks``.
    observations : list of list of int
        For each image, the indices of the tracks it observes.
    """

    tracks: list[list[FeatureKey]] = field(default_factory=list)
    feature_to_track: dict[FeatureKey, int] = field(default_factory=dict)
    observations: list[list[int]] = field(default_factory=list)

    def track_length_histogram(self) -> dict[int, int]:
        """Return a mapping from track length to the number of such tracks."""
        histogram: dict[int, int] = defaultdict(int)
        for track in self.tracks:
            histogram[len(track)] += 1
        return dict(sorted(histogram.items()))


def build_tracks(
    num_images: int,
    pairwise_matches: dict[tuple[int, int], np.ndarray],
    min_length: int = 2,
) -> TrackGraph:
    """Group verified matches into tracks.

    Parameters
    ----------
    num_images : int
    pairwise_matches : dict
        Maps an image pair ``(i, j)`` with ``i < j`` to an array of shape
        ``(m, 2)`` holding indices of matched features in image ``i`` and ``j``.
    min_length : int
        Tracks shorter than this are dropped.

    Returns
    -------
    TrackGraph
    """
    forest = _DisjointSet()
    for (image_a, image_b), matches in pairwise_matches.items():
        for feature_a, feature_b in np.asarray(matches, dtype=np.int64):
            forest.union((image_a, int(feature_a)), (image_b, int(feature_b)))

    graph = TrackGraph(observations=[[] for _ in range(num_images)])
    for members in forest.components().values():
        images = [image for image, _ in members]
        if len(set(images)) != len(images) or len(members) < min_length:
            continue
        track_index = len(graph.tracks)
        members.sort()
        graph.tracks.append(members)
        for key in members:
            graph.feature_to_track[key] = track_index
            graph.observations[key[0]].append(track_index)
    return graph
