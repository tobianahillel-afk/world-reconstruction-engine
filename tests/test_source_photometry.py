from __future__ import annotations

import inspect
import math
from dataclasses import FrozenInstanceError, fields

import pytest

import wre.domain.source_photometry as source_photometry_module
from wre.domain import ObservationId, ObservationMetadata, RawMetadataEntry
from wre.domain.source_photometry import (
    SourceColorMetadata,
    SourceExposureMetadata,
    SourcePhotometryInterpretationStatus,
    SourcePhotometryMetadata,
    SourceWhiteBalanceMetadata,
)

OBSERVATION_ID = ObservationId("obs:photometry:0001")


def _entry(key: str, value: str, namespace: str = "exif") -> RawMetadataEntry:
    return RawMetadataEntry(namespace=namespace, key=key, value=value)


COLOR_SPACE = _entry("EXIF ColorSpace", "1")
PRIMARIES = _entry("Video ColorPrimaries", "bt709", namespace="container")
TRANSFER = _entry("Video TransferCharacteristic", "bt709", namespace="container")
MATRIX = _entry("Video MatrixCoefficients", "bt709", namespace="container")
RANGE = _entry("Video ColorRange", "limited", namespace="container")
PROFILE = _entry("ICC ProfileName", "Display P3", namespace="icc")
ISO = _entry("EXIF ISOSpeedRatings", "100")
EXPOSURE_TIME = _entry("EXIF ExposureTime", "1/125")
F_NUMBER = _entry("EXIF FNumber", "28/10")
EXPOSURE_BIAS = _entry("EXIF ExposureBiasValue", "-1/3")
WB_MODE = _entry("EXIF WhiteBalance", "1")
WB_TEMP = _entry("EXIF ColorTemperature", "5200")


ALL_ENTRIES = tuple(
    sorted(
        (
            COLOR_SPACE,
            PRIMARIES,
            TRANSFER,
            MATRIX,
            RANGE,
            PROFILE,
            ISO,
            EXPOSURE_TIME,
            F_NUMBER,
            EXPOSURE_BIAS,
            WB_MODE,
            WB_TEMP,
        ),
        key=lambda entry: (entry.namespace, entry.key, entry.value),
    )
)


def _metadata(
    observation_id: ObservationId = OBSERVATION_ID,
    raw_entries: tuple[RawMetadataEntry, ...] = ALL_ENTRIES,
) -> ObservationMetadata:
    return ObservationMetadata(observation_id=observation_id, raw_entries=raw_entries)


def _absent_color() -> SourceColorMetadata:
    return SourceColorMetadata(status=SourcePhotometryInterpretationStatus.ABSENT)


def _absent_exposure() -> SourceExposureMetadata:
    return SourceExposureMetadata(status=SourcePhotometryInterpretationStatus.ABSENT)


def _absent_white_balance() -> SourceWhiteBalanceMetadata:
    return SourceWhiteBalanceMetadata(status=SourcePhotometryInterpretationStatus.ABSENT)


def _resolved_color() -> SourceColorMetadata:
    evidence = tuple(
        sorted(
            (COLOR_SPACE, PRIMARIES, TRANSFER, MATRIX, RANGE, PROFILE),
            key=lambda entry: (entry.namespace, entry.key, entry.value),
        )
    )
    return SourceColorMetadata(
        status=SourcePhotometryInterpretationStatus.RESOLVED,
        declared_color_space="EXIF:1",
        declared_primaries="bt709",
        declared_transfer_characteristic="bt709",
        declared_matrix_coefficients="bt709",
        declared_range="limited",
        profile_name="Display P3",
        evidence=evidence,
    )


def _resolved_exposure() -> SourceExposureMetadata:
    evidence = tuple(
        sorted(
            (ISO, EXPOSURE_TIME, F_NUMBER, EXPOSURE_BIAS),
            key=lambda entry: (entry.namespace, entry.key, entry.value),
        )
    )
    return SourceExposureMetadata(
        status=SourcePhotometryInterpretationStatus.RESOLVED,
        iso_speed=100.0,
        exposure_time_seconds=1.0 / 125.0,
        f_number=2.8,
        exposure_compensation_ev=-1.0 / 3.0,
        evidence=evidence,
    )


def _resolved_white_balance() -> SourceWhiteBalanceMetadata:
    evidence = tuple(
        sorted(
            (WB_MODE, WB_TEMP),
            key=lambda entry: (entry.namespace, entry.key, entry.value),
        )
    )
    return SourceWhiteBalanceMetadata(
        status=SourcePhotometryInterpretationStatus.RESOLVED,
        mode="manual",
        color_temperature_kelvin=5200.0,
        evidence=evidence,
    )


def _photometry(
    *,
    observation_id: ObservationId = OBSERVATION_ID,
    source_metadata: ObservationMetadata | None = None,
    color: SourceColorMetadata | None = None,
    exposure: SourceExposureMetadata | None = None,
    white_balance: SourceWhiteBalanceMetadata | None = None,
) -> SourcePhotometryMetadata:
    return SourcePhotometryMetadata(
        observation_id=observation_id,
        source_metadata=source_metadata or _metadata(observation_id),
        color=color or _absent_color(),
        exposure=exposure or _absent_exposure(),
        white_balance=white_balance or _absent_white_balance(),
    )


def test_contract_shapes_and_status_vocabulary_are_exact() -> None:
    assert tuple(item.value for item in SourcePhotometryInterpretationStatus) == (
        "absent",
        "unknown",
        "invalid",
        "resolved",
    )
    assert [field.name for field in fields(SourceColorMetadata)] == [
        "status",
        "declared_color_space",
        "declared_primaries",
        "declared_transfer_characteristic",
        "declared_matrix_coefficients",
        "declared_range",
        "profile_name",
        "evidence",
        "issue",
    ]
    assert [field.name for field in fields(SourceExposureMetadata)] == [
        "status",
        "iso_speed",
        "exposure_time_seconds",
        "f_number",
        "exposure_compensation_ev",
        "evidence",
        "issue",
    ]
    assert [field.name for field in fields(SourceWhiteBalanceMetadata)] == [
        "status",
        "mode",
        "color_temperature_kelvin",
        "evidence",
        "issue",
    ]
    assert [field.name for field in fields(SourcePhotometryMetadata)] == [
        "observation_id",
        "source_metadata",
        "color",
        "exposure",
        "white_balance",
    ]


@pytest.mark.parametrize(
    "factory",
    [_absent_color, _absent_exposure, _absent_white_balance],
)
def test_absent_state_carries_no_values_evidence_or_issue(factory: object) -> None:
    value = factory()  # type: ignore[operator]
    assert value.status is SourcePhotometryInterpretationStatus.ABSENT
    assert value.evidence == ()
    assert value.issue is None


@pytest.mark.parametrize(
    ("constructor", "kwargs"),
    [
        (
            SourceColorMetadata,
            {"declared_color_space": "sRGB"},
        ),
        (
            SourceExposureMetadata,
            {"iso_speed": 100.0},
        ),
        (
            SourceWhiteBalanceMetadata,
            {"mode": "auto"},
        ),
    ],
)
def test_absent_state_rejects_interpreted_values(
    constructor: object,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="absent state"):
        constructor(  # type: ignore[operator]
            status=SourcePhotometryInterpretationStatus.ABSENT,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("status", "issue"),
    [
        (SourcePhotometryInterpretationStatus.UNKNOWN, "unsupported vendor declaration"),
        (SourcePhotometryInterpretationStatus.INVALID, "malformed declaration"),
    ],
)
def test_unknown_and_invalid_color_keep_raw_evidence_without_guessing(
    status: SourcePhotometryInterpretationStatus,
    issue: str,
) -> None:
    value = SourceColorMetadata(
        status=status,
        evidence=(COLOR_SPACE,),
        issue=issue,
    )

    assert value.declared_color_space is None
    assert value.declared_primaries is None
    assert value.declared_transfer_characteristic is None
    assert value.declared_matrix_coefficients is None
    assert value.declared_range is None
    assert value.profile_name is None
    assert value.evidence == (COLOR_SPACE,)


@pytest.mark.parametrize(
    "status",
    [
        SourcePhotometryInterpretationStatus.UNKNOWN,
        SourcePhotometryInterpretationStatus.INVALID,
    ],
)
def test_unknown_or_invalid_state_requires_evidence_and_issue(
    status: SourcePhotometryInterpretationStatus,
) -> None:
    with pytest.raises(ValueError, match="requires evidence"):
        SourceExposureMetadata(status=status, issue="missing")

    with pytest.raises(ValueError, match="requires an issue"):
        SourceExposureMetadata(status=status, evidence=(ISO,))


def test_unknown_or_invalid_state_rejects_interpreted_values() -> None:
    with pytest.raises(ValueError, match="must not expose interpreted values"):
        SourceWhiteBalanceMetadata(
            status=SourcePhotometryInterpretationStatus.UNKNOWN,
            mode="auto",
            evidence=(WB_MODE,),
            issue="unsupported",
        )


def test_resolved_color_preserves_only_explicit_declarations() -> None:
    value = _resolved_color()

    assert value.declared_color_space == "EXIF:1"
    assert value.declared_primaries == "bt709"
    assert value.declared_transfer_characteristic == "bt709"
    assert value.declared_matrix_coefficients == "bt709"
    assert value.declared_range == "limited"
    assert value.profile_name == "Display P3"
    assert value.issue is None


def test_resolved_state_requires_value_and_evidence_and_rejects_issue() -> None:
    with pytest.raises(ValueError, match="requires evidence"):
        SourceColorMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            declared_color_space="sRGB",
        )

    with pytest.raises(ValueError, match="at least one interpreted value"):
        SourceColorMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            evidence=(COLOR_SPACE,),
        )

    with pytest.raises(ValueError, match="must not carry an issue"):
        SourceColorMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            declared_color_space="sRGB",
            evidence=(COLOR_SPACE,),
            issue="should not coexist",
        )


def test_resolved_exposure_keeps_capture_hints_without_normalization() -> None:
    value = _resolved_exposure()

    assert value.iso_speed == 100.0
    assert value.exposure_time_seconds == pytest.approx(0.008)
    assert value.f_number == 2.8
    assert value.exposure_compensation_ev == pytest.approx(-1.0 / 3.0)
    assert value.issue is None


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("iso_speed", 0.0, "greater than zero"),
        ("iso_speed", -100.0, "greater than zero"),
        ("exposure_time_seconds", 0.0, "greater than zero"),
        ("exposure_time_seconds", -1.0, "greater than zero"),
        ("f_number", 0.0, "greater than zero"),
        ("f_number", -2.8, "greater than zero"),
        ("exposure_compensation_ev", math.inf, "finite"),
        ("exposure_compensation_ev", math.nan, "finite"),
    ],
)
def test_exposure_numeric_domains_fail_closed(
    field_name: str,
    value: float,
    message: str,
) -> None:
    kwargs = {
        "status": SourcePhotometryInterpretationStatus.RESOLVED,
        "evidence": (ISO,),
        field_name: value,
    }
    with pytest.raises(ValueError, match=message):
        SourceExposureMetadata(**kwargs)  # type: ignore[arg-type]


def test_numeric_fields_reject_bool_and_non_float_values() -> None:
    with pytest.raises(TypeError, match="must be a float"):
        SourceExposureMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            iso_speed=100,  # type: ignore[arg-type]
            evidence=(ISO,),
        )

    with pytest.raises(TypeError, match="must be a float"):
        SourceWhiteBalanceMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            color_temperature_kelvin=True,  # type: ignore[arg-type]
            evidence=(WB_TEMP,),
        )


def test_resolved_white_balance_is_capture_metadata_only() -> None:
    value = _resolved_white_balance()

    assert value.mode == "manual"
    assert value.color_temperature_kelvin == 5200.0
    assert value.issue is None
    assert not hasattr(value, "red_gain")
    assert not hasattr(value, "blue_gain")
    assert not hasattr(value, "chromatic_adaptation")


@pytest.mark.parametrize("temperature", [0.0, -1.0, math.inf, math.nan])
def test_white_balance_temperature_must_be_positive_finite(temperature: float) -> None:
    with pytest.raises(ValueError):
        SourceWhiteBalanceMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            color_temperature_kelvin=temperature,
            evidence=(WB_TEMP,),
        )


def test_interpretation_families_are_independently_stateful() -> None:
    photometry = _photometry(
        color=_resolved_color(),
        exposure=SourceExposureMetadata(
            status=SourcePhotometryInterpretationStatus.UNKNOWN,
            evidence=(ISO,),
            issue="unsupported exposure encoding",
        ),
        white_balance=_absent_white_balance(),
    )

    assert photometry.color.status is SourcePhotometryInterpretationStatus.RESOLVED
    assert photometry.exposure.status is SourcePhotometryInterpretationStatus.UNKNOWN
    assert photometry.white_balance.status is SourcePhotometryInterpretationStatus.ABSENT


def test_photometry_is_bound_to_exact_observation_metadata_identity() -> None:
    foreign_id = ObservationId("obs:photometry:foreign")

    with pytest.raises(ValueError, match="must match"):
        _photometry(
            observation_id=OBSERVATION_ID,
            source_metadata=_metadata(foreign_id),
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("observation_id", "obs:not-typed", "ObservationId"),
        ("source_metadata", object(), "ObservationMetadata"),
        ("color", object(), "SourceColorMetadata"),
        ("exposure", object(), "SourceExposureMetadata"),
        ("white_balance", object(), "SourceWhiteBalanceMetadata"),
    ],
)
def test_top_level_members_are_strictly_typed(
    field_name: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "observation_id": OBSERVATION_ID,
        "source_metadata": _metadata(),
        "color": _absent_color(),
        "exposure": _absent_exposure(),
        "white_balance": _absent_white_balance(),
    }
    kwargs[field_name] = value

    with pytest.raises(TypeError, match=message):
        SourcePhotometryMetadata(**kwargs)  # type: ignore[arg-type]


def test_evidence_must_be_immutable_unique_and_canonically_ordered() -> None:
    with pytest.raises(TypeError, match="immutable tuple"):
        SourceColorMetadata(
            status=SourcePhotometryInterpretationStatus.UNKNOWN,
            evidence=[COLOR_SPACE],  # type: ignore[arg-type]
            issue="unsupported",
        )

    with pytest.raises(ValueError, match="duplicates"):
        SourceColorMetadata(
            status=SourcePhotometryInterpretationStatus.UNKNOWN,
            evidence=(COLOR_SPACE, COLOR_SPACE),
            issue="unsupported",
        )

    canonical = tuple(
        sorted(
            (COLOR_SPACE, PRIMARIES),
            key=lambda entry: (entry.namespace, entry.key, entry.value),
        )
    )
    reversed_evidence = tuple(reversed(canonical))
    assert reversed_evidence != canonical

    with pytest.raises(ValueError, match="canonical"):
        SourceColorMetadata(
            status=SourcePhotometryInterpretationStatus.UNKNOWN,
            evidence=reversed_evidence,
            issue="unsupported",
        )


def test_foreign_or_fabricated_evidence_fails_closed() -> None:
    foreign = _entry("EXIF ISO", "200")
    exposure = SourceExposureMetadata(
        status=SourcePhotometryInterpretationStatus.RESOLVED,
        iso_speed=200.0,
        evidence=(foreign,),
    )

    with pytest.raises(ValueError, match=r"source_metadata\.raw_entries"):
        _photometry(exposure=exposure)


def test_exact_raw_evidence_is_retained_without_rewriting_source_metadata() -> None:
    metadata = _metadata()
    before = metadata.raw_entries

    photometry = _photometry(
        source_metadata=metadata,
        color=_resolved_color(),
        exposure=_resolved_exposure(),
        white_balance=_resolved_white_balance(),
    )

    assert photometry.source_metadata is metadata
    assert photometry.source_metadata.raw_entries == before
    assert photometry.color.evidence == _resolved_color().evidence
    assert photometry.exposure.evidence == _resolved_exposure().evidence
    assert photometry.white_balance.evidence == _resolved_white_balance().evidence


def test_missing_metadata_never_creates_common_photometric_defaults() -> None:
    value = _photometry(source_metadata=_metadata(raw_entries=()))

    assert value.color == _absent_color()
    assert value.exposure == _absent_exposure()
    assert value.white_balance == _absent_white_balance()
    serialized_names = {field.name for field in fields(SourcePhotometryMetadata)}
    assert "decoded_color_space" not in serialized_names
    assert "working_color_space" not in serialized_names
    assert "output_color_space" not in serialized_names
    assert "normalized_exposure" not in serialized_names


def test_observation_identity_does_not_encode_still_or_video_photometry_semantics() -> None:
    still_id = ObservationId("obs:image:photometry")
    frame_id = ObservationId("obs:frame:photometry")

    still = _photometry(
        observation_id=still_id,
        source_metadata=_metadata(still_id, raw_entries=()),
    )
    frame = _photometry(
        observation_id=frame_id,
        source_metadata=_metadata(frame_id, raw_entries=()),
    )

    assert still.color == frame.color
    assert still.exposure == frame.exposure
    assert still.white_balance == frame.white_balance


def test_contracts_are_frozen() -> None:
    value = _photometry(
        color=_resolved_color(),
        exposure=_resolved_exposure(),
        white_balance=_resolved_white_balance(),
    )

    with pytest.raises(FrozenInstanceError):
        value.observation_id = ObservationId("obs:mutated")  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        value.color.declared_color_space = "mutated"  # type: ignore[misc]


def test_module_surface_is_pure_metadata_contract_only() -> None:
    source = inspect.getsource(source_photometry_module)

    forbidden_imports = (
        "pathlib",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "exifread",
        "ffmpeg",
        "PIL",
        "cv2",
        "colour",
        "lcms",
        "numpy",
        "torch",
    )
    for forbidden in forbidden_imports:
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source

    forbidden_api_tokens = (
        "AppearanceModel",
        "QualityDecision",
        "decode(",
        "normalize_exposure",
        "normalize_white_balance",
        "linearize",
        "tone_map",
        "icc_transform",
        "pixel_buffer",
    )
    for token in forbidden_api_tokens:
        assert token not in source
