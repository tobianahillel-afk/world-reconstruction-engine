from __future__ import annotations

from typing import cast

import pytest

from wre.domain import (
    CameraId,
    LocalFrameId,
    ObservationId,
    SpatialFragment,
    SpatialFragmentId,
)


def _observation(value: str) -> ObservationId:
    return ObservationId(f"obs:{value}")


def test_fragment_ids_are_typed_domains() -> None:
    assert SpatialFragmentId("shared:id") != ObservationId("shared:id")
    assert SpatialFragmentId("shared:id") != CameraId("shared:id")
    assert LocalFrameId("shared:id") != SpatialFragmentId("shared:id")

    with pytest.raises(ValueError, match="fragment_id"):
        SpatialFragmentId("bad fragment id")
    with pytest.raises(ValueError, match="local_frame_id"):
        LocalFrameId("bad frame id")


def test_spatial_fragment_canonicalizes_unordered_membership() -> None:
    fragment = SpatialFragment(
        fragment_id=SpatialFragmentId("fragment:001"),
        local_frame_id=LocalFrameId("frame:fragment-001"),
        observation_ids=(_observation("c"), _observation("a"), _observation("b")),
    )

    assert fragment.observation_ids == (
        _observation("a"),
        _observation("b"),
        _observation("c"),
    )
    assert fragment.observation_count == 3
    assert fragment.contains(_observation("b")) is True
    assert fragment.contains(_observation("other")) is False


def test_fragment_equality_does_not_depend_on_input_order() -> None:
    first = SpatialFragment(
        fragment_id=SpatialFragmentId("fragment:stable"),
        local_frame_id=LocalFrameId("frame:stable"),
        observation_ids=(_observation("2"), _observation("1")),
    )
    second = SpatialFragment(
        fragment_id=SpatialFragmentId("fragment:stable"),
        local_frame_id=LocalFrameId("frame:stable"),
        observation_ids=(_observation("1"), _observation("2")),
    )

    assert first == second


def test_fragment_requires_non_empty_unique_observation_membership() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        SpatialFragment(
            fragment_id=SpatialFragmentId("fragment:empty"),
            local_frame_id=LocalFrameId("frame:empty"),
            observation_ids=(),
        )

    duplicate = _observation("duplicate")
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        SpatialFragment(
            fragment_id=SpatialFragmentId("fragment:duplicate"),
            local_frame_id=LocalFrameId("frame:duplicate"),
            observation_ids=(duplicate, duplicate),
        )


def test_fragment_membership_container_must_be_immutable_tuple() -> None:
    mutable_members = [_observation("a")]
    invalid_members = cast(tuple[ObservationId, ...], mutable_members)

    with pytest.raises(ValueError, match="immutable tuple"):
        SpatialFragment(
            fragment_id=SpatialFragmentId("fragment:list"),
            local_frame_id=LocalFrameId("frame:list"),
            observation_ids=invalid_members,
        )
