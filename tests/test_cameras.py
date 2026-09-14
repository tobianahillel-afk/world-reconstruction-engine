from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from wre.domain import (
    Camera,
    CameraId,
    ImageDimensions,
    ObservationId,
    ObservationMetadata,
    RawMetadataEntry,
)


def test_camera_identity_is_explicit_and_solver_independent() -> None:
    camera = Camera(
        camera_id=CameraId("camera:phone-01"),
        manufacturer="Example Corp",
        model="Model X",
        serial_number="ABC-123",
    )

    assert str(camera.camera_id) == "camera:phone-01"
    assert camera.manufacturer == "Example Corp"
    assert camera.model == "Model X"
    assert camera.serial_number == "ABC-123"


def test_camera_id_is_distinct_from_observation_id() -> None:
    assert CameraId("shared:identifier") != ObservationId("shared:identifier")

    with pytest.raises(ValueError, match="camera_id"):
        CameraId("bad camera id")


def test_camera_descriptive_fields_may_be_unknown_but_not_blank() -> None:
    anonymous = Camera(camera_id=CameraId("camera:unknown"))
    assert anonymous.manufacturer is None

    with pytest.raises(ValueError, match="manufacturer"):
        Camera(camera_id=CameraId("camera:manufacturer"), manufacturer=" ")
    with pytest.raises(ValueError, match="model"):
        Camera(camera_id=CameraId("camera:model"), model=" ")
    with pytest.raises(ValueError, match="serial_number"):
        Camera(camera_id=CameraId("camera:serial"), serial_number=" ")


def test_image_dimensions_require_positive_integer_pixels() -> None:
    dimensions = ImageDimensions(width_px=4032, height_px=3024)
    assert dimensions.width_px == 4032
    assert dimensions.height_px == 3024

    with pytest.raises(ValueError, match="width_px"):
        ImageDimensions(width_px=0, height_px=10)
    with pytest.raises(ValueError, match="height_px"):
        ImageDimensions(width_px=10, height_px=-1)
    with pytest.raises(ValueError, match="width_px"):
        ImageDimensions(width_px=True, height_px=10)


def test_raw_metadata_preserves_uninterpreted_value() -> None:
    entry = RawMetadataEntry(
        namespace="exif",
        key="DateTimeOriginal",
        value="2026:09:14 21:00:00",
    )

    assert entry.value == "2026:09:14 21:00:00"

    empty_value = RawMetadataEntry(namespace="custom", key="EmptyTag", value="")
    assert empty_value.value == ""


def test_raw_metadata_requires_named_namespace_and_key() -> None:
    with pytest.raises(ValueError, match="namespace"):
        RawMetadataEntry(namespace=" ", key="tag", value="value")
    with pytest.raises(ValueError, match="key"):
        RawMetadataEntry(namespace="exif", key=" ", value="value")


def test_observation_metadata_links_without_rewriting_raw_observation() -> None:
    metadata = ObservationMetadata(
        observation_id=ObservationId("obs:image:0001"),
        camera_id=CameraId("camera:phone-01"),
        dimensions=ImageDimensions(width_px=4032, height_px=3024),
        raw_entries=(
            RawMetadataEntry(namespace="exif", key="Make", value="Example Corp"),
            RawMetadataEntry(namespace="exif", key="Model", value="Model X"),
        ),
    )

    assert metadata.observation_id == ObservationId("obs:image:0001")
    assert metadata.camera_id == CameraId("camera:phone-01")
    assert metadata.dimensions == ImageDimensions(width_px=4032, height_px=3024)
    assert [entry.key for entry in metadata.raw_entries] == ["Make", "Model"]


def test_observation_metadata_entries_are_immutable() -> None:
    mutable_entries = [RawMetadataEntry(namespace="exif", key="Make", value="Example")]
    invalid_entries = cast(tuple[RawMetadataEntry, ...], mutable_entries)
    with pytest.raises(ValueError, match="immutable tuple"):
        ObservationMetadata(
            observation_id=ObservationId("obs:image:list"),
            raw_entries=invalid_entries,
        )

    metadata = ObservationMetadata(observation_id=ObservationId("obs:image:frozen"))
    unsafe_metadata = cast(Any, metadata)
    with pytest.raises(FrozenInstanceError):
        unsafe_metadata.camera_id = CameraId("camera:other")
