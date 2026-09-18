from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from inspect import signature
from typing import Any, cast

import pytest

import wre.profiling.input_classification as input_classification_module
from wre.domain import (
    ImageDimensions,
    ObservationId,
    ObservationKind,
    ObservationMetadata,
    RawMetadataEntry,
)
from wre.profiling import (
    InputClass,
    InputClassEvidence,
    InputClassEvidenceKind,
    evaluate_input_classification,
)


def _metadata(
    *,
    width: int | None = None,
    height: int | None = None,
    entries: tuple[RawMetadataEntry, ...] = (),
) -> ObservationMetadata:
    dimensions = (
        None
        if width is None or height is None
        else ImageDimensions(width_px=width, height_px=height)
    )
    return ObservationMetadata(
        observation_id=ObservationId("obs:input-class"),
        dimensions=dimensions,
        raw_entries=entries,
    )


def _entry(key: str, value: str, namespace: str = "exif") -> RawMetadataEntry:
    return RawMetadataEntry(namespace=namespace, key=key, value=value)


def test_input_class_enums_are_exact_closed_vocabularies() -> None:
    assert tuple(item.value for item in InputClass) == (
        "still",
        "video",
        "equirectangular_360",
        "drone_aerial",
        "fisheye",
        "rolling_shutter",
    )
    assert tuple(item.value for item in InputClassEvidenceKind) == (
        "observed",
        "candidate",
    )


def test_input_class_evidence_is_typed_immutable_hashable_and_canonical() -> None:
    evidence = InputClassEvidence(
        input_class=InputClass.FISHEYE,
        evidence_kind=InputClassEvidenceKind.CANDIDATE,
        evidence_keys=("exif:Image Model", "exif:Lens Model"),
    )

    assert tuple(field.name for field in fields(InputClassEvidence)) == (
        "input_class",
        "evidence_kind",
        "evidence_keys",
    )
    assert hash(evidence) == hash(
        InputClassEvidence(
            input_class=InputClass.FISHEYE,
            evidence_kind=InputClassEvidenceKind.CANDIDATE,
            evidence_keys=("exif:Image Model", "exif:Lens Model"),
        )
    )
    with pytest.raises(FrozenInstanceError):
        evidence.input_class = InputClass.STILL  # type: ignore[misc]

    with pytest.raises(TypeError, match="input_class must be InputClass"):
        InputClassEvidence(
            input_class=cast(Any, "fisheye"),
            evidence_kind=InputClassEvidenceKind.CANDIDATE,
            evidence_keys=("exif:Lens Model",),
        )
    with pytest.raises(TypeError, match="evidence_kind"):
        InputClassEvidence(
            input_class=InputClass.FISHEYE,
            evidence_kind=cast(Any, "candidate"),
            evidence_keys=("exif:Lens Model",),
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        InputClassEvidence(
            input_class=InputClass.FISHEYE,
            evidence_kind=InputClassEvidenceKind.CANDIDATE,
            evidence_keys=cast(Any, ["exif:Lens Model"]),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        InputClassEvidence(
            input_class=InputClass.FISHEYE,
            evidence_kind=InputClassEvidenceKind.CANDIDATE,
            evidence_keys=(),
        )
    with pytest.raises(ValueError, match="non-blank"):
        InputClassEvidence(
            input_class=InputClass.FISHEYE,
            evidence_kind=InputClassEvidenceKind.CANDIDATE,
            evidence_keys=("",),
        )
    with pytest.raises(ValueError, match="unique"):
        InputClassEvidence(
            input_class=InputClass.FISHEYE,
            evidence_kind=InputClassEvidenceKind.CANDIDATE,
            evidence_keys=("exif:Lens Model", "exif:Lens Model"),
        )
    with pytest.raises(ValueError, match="canonical lexical order"):
        InputClassEvidence(
            input_class=InputClass.FISHEYE,
            evidence_kind=InputClassEvidenceKind.CANDIDATE,
            evidence_keys=("exif:Lens Model", "exif:Image Model"),
        )


@pytest.mark.parametrize(
    ("observation_kind", "expected_class"),
    [
        (ObservationKind.IMAGE, InputClass.STILL),
        (ObservationKind.VIDEO, InputClass.VIDEO),
        (ObservationKind.VIDEO_FRAME, InputClass.VIDEO),
    ],
)
def test_observation_kind_produces_exact_observed_media_class(
    observation_kind: ObservationKind,
    expected_class: InputClass,
) -> None:
    result = evaluate_input_classification(observation_kind)

    assert result == (
        InputClassEvidence(
            input_class=expected_class,
            evidence_kind=InputClassEvidenceKind.OBSERVED,
            evidence_keys=("observation.kind",),
        ),
    )


def test_evaluator_requires_typed_kind_and_optional_metadata() -> None:
    assert tuple(signature(evaluate_input_classification).parameters) == (
        "observation_kind",
        "metadata",
    )

    with pytest.raises(TypeError, match="observation_kind"):
        evaluate_input_classification(cast(Any, "image"))

    with pytest.raises(TypeError, match="metadata"):
        evaluate_input_classification(ObservationKind.IMAGE, cast(Any, {}))


def test_no_metadata_and_ordinary_phone_metadata_do_not_fabricate_specialists() -> None:
    ordinary = _metadata(
        width=4032,
        height=3024,
        entries=(
            _entry("GPS GPSLatitude", "[48, 51, 0]"),
            _entry("Image Make", "Apple"),
            _entry("Image Model", "iPhone 15 Pro"),
        ),
    )

    assert tuple(
        item.input_class
        for item in evaluate_input_classification(ObservationKind.IMAGE)
    ) == (InputClass.STILL,)
    assert tuple(
        item.input_class
        for item in evaluate_input_classification(ObservationKind.IMAGE, ordinary)
    ) == (InputClass.STILL,)


def test_exact_two_to_one_dimensions_emit_only_360_candidate() -> None:
    metadata = _metadata(width=4000, height=2000)

    result = evaluate_input_classification(ObservationKind.IMAGE, metadata)

    assert result == (
        InputClassEvidence(
            input_class=InputClass.STILL,
            evidence_kind=InputClassEvidenceKind.OBSERVED,
            evidence_keys=("observation.kind",),
        ),
        InputClassEvidence(
            input_class=InputClass.EQUIRECTANGULAR_360,
            evidence_kind=InputClassEvidenceKind.CANDIDATE,
            evidence_keys=("metadata.dimensions",),
        ),
    )
    assert evaluate_input_classification(
        ObservationKind.IMAGE,
        _metadata(width=4000, height=1999),
    ) == (
        InputClassEvidence(
            input_class=InputClass.STILL,
            evidence_kind=InputClassEvidenceKind.OBSERVED,
            evidence_keys=("observation.kind",),
        ),
    )


def test_explicit_fisheye_and_rolling_shutter_phrases_emit_candidates() -> None:
    metadata = _metadata(
        entries=(
            _entry("Lens Model", "Ultra Fisheye 8mm"),
            _entry("Camera Mode", "Rolling-Shutter Capture"),
        )
    )

    result = evaluate_input_classification(ObservationKind.IMAGE, metadata)

    assert tuple(item.input_class for item in result) == (
        InputClass.STILL,
        InputClass.FISHEYE,
        InputClass.ROLLING_SHUTTER,
    )
    assert result[1].evidence_keys == ("exif:Lens Model",)
    assert result[2].evidence_keys == ("exif:Camera Mode",)
    assert all(
        item.evidence_kind is InputClassEvidenceKind.CANDIDATE
        for item in result[1:]
    )


def test_generic_focal_length_and_exposure_time_do_not_imply_specialists() -> None:
    metadata = _metadata(
        entries=(
            _entry("EXIF FocalLength", "8"),
            _entry("EXIF ExposureTime", "1/30"),
            _entry("Lens Model", "Wide Angle Lens"),
        )
    )

    result = evaluate_input_classification(ObservationKind.IMAGE, metadata)

    assert tuple(item.input_class for item in result) == (InputClass.STILL,)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("Image Make", "DJI"),
        ("Image Make", "Autel Robotics"),
        ("Image Model", "Skydio 2+"),
        ("Image Make", "Parrot"),
    ],
)
def test_allowlisted_uav_make_or_model_emits_drone_candidate(
    key: str,
    value: str,
) -> None:
    result = evaluate_input_classification(
        ObservationKind.IMAGE,
        _metadata(entries=(_entry(key, value),)),
    )

    assert result[-1] == InputClassEvidence(
        input_class=InputClass.DRONE_AERIAL,
        evidence_kind=InputClassEvidenceKind.CANDIDATE,
        evidence_keys=(f"exif:{key}",),
    )


def test_gps_only_and_ordinary_camera_metadata_do_not_imply_drone_capture() -> None:
    metadata = _metadata(
        entries=(
            _entry("GPS GPSLatitude", "[48, 51, 0]"),
            _entry("GPS GPSAltitude", "120"),
            _entry("Image Make", "Canon"),
            _entry("Image Model", "EOS R5"),
        )
    )

    result = evaluate_input_classification(ObservationKind.IMAGE, metadata)

    assert tuple(item.input_class for item in result) == (InputClass.STILL,)


def test_specialist_candidates_can_coexist_in_fixed_canonical_order() -> None:
    metadata = _metadata(
        width=4096,
        height=2048,
        entries=(
            _entry("Image Make", "DJI"),
            _entry("Lens Model", "Fisheye"),
            _entry("Sensor Mode", "rolling_shutter"),
        ),
    )

    before = metadata
    result = evaluate_input_classification(ObservationKind.IMAGE, metadata)

    assert tuple(item.input_class for item in result) == (
        InputClass.STILL,
        InputClass.EQUIRECTANGULAR_360,
        InputClass.DRONE_AERIAL,
        InputClass.FISHEYE,
        InputClass.ROLLING_SHUTTER,
    )
    assert tuple(item.evidence_kind for item in result) == (
        InputClassEvidenceKind.OBSERVED,
        InputClassEvidenceKind.CANDIDATE,
        InputClassEvidenceKind.CANDIDATE,
        InputClassEvidenceKind.CANDIDATE,
        InputClassEvidenceKind.CANDIDATE,
    )
    assert metadata is before
    assert metadata == before


def test_public_surface_has_no_negative_truth_routing_or_solver_semantics() -> None:
    for name in (
        "is_360",
        "is_drone",
        "is_fisheye",
        "is_rolling_shutter",
        "confidence",
        "threshold",
        "route",
        "QualityDecision",
        "Camera",
        "camera_model",
        "calibrate",
        "solve_projection",
        "dynamic_likelihood",
        "sparse_coverage",
        "select_frames",
    ):
        assert name not in input_classification_module.__dict__
