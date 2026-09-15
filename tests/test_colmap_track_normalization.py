from types import SimpleNamespace

import pytest

from wre.domain.observations import ObservationId
from wre.reconstruction.colmap_import import (
    ColmapReconstructionImportError,
    _canonicalize_colmap_track,
)


def test_colmap_track_normalization_keeps_one_deterministic_feature_per_observation() -> None:
    observations = {
        1: ObservationId("obs:a"),
        2: ObservationId("obs:b"),
    }
    native_track = (
        SimpleNamespace(image_id=2, point2D_idx=7),
        SimpleNamespace(image_id=1, point2D_idx=9),
        SimpleNamespace(image_id=1, point2D_idx=3),
        SimpleNamespace(image_id=2, point2D_idx=11),
    )

    normalized = _canonicalize_colmap_track(native_track, observations)

    assert tuple((item.observation_id, item.feature_index) for item in normalized) == (
        (ObservationId("obs:a"), 3),
        (ObservationId("obs:b"), 7),
    )


def test_colmap_track_normalization_rejects_unregistered_image() -> None:
    native_track = (SimpleNamespace(image_id=9, point2D_idx=3),)

    with pytest.raises(
        ColmapReconstructionImportError,
        match="outside registered model membership",
    ):
        _canonicalize_colmap_track(native_track, {1: ObservationId("obs:a")})
