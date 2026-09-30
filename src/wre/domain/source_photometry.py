from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from wre.domain.cameras import ObservationMetadata, RawMetadataEntry
from wre.domain.observations import ObservationId


def _optional_non_blank(value: str | None, context: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{context} must be non-blank when present")


def _canonical_evidence(
    value: tuple[RawMetadataEntry, ...],
    context: str,
) -> tuple[RawMetadataEntry, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{context} must be an immutable tuple")
    if any(not isinstance(entry, RawMetadataEntry) for entry in value):
        raise TypeError(f"{context} members must be RawMetadataEntry")
    identities = tuple((entry.namespace, entry.key, entry.value) for entry in value)
    if len(set(identities)) != len(identities):
        raise ValueError(f"{context} must not contain duplicates")
    if identities != tuple(sorted(identities)):
        raise ValueError(f"{context} must be in canonical namespace/key/value order")
    return value


def _validate_finite_float(
    value: float | None,
    context: str,
    *,
    positive: bool = False,
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, float):
        raise TypeError(f"{context} must be a float when present")
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{context} must be greater than zero")


class SourcePhotometryInterpretationStatus(StrEnum):
    """Interpretation state for one independent source-photometry evidence family."""

    ABSENT = "absent"
    UNKNOWN = "unknown"
    INVALID = "invalid"
    RESOLVED = "resolved"


def _validate_interpretation_state(
    *,
    status: SourcePhotometryInterpretationStatus,
    evidence: tuple[RawMetadataEntry, ...],
    issue: str | None,
    has_value: bool,
    context: str,
) -> None:
    if not isinstance(status, SourcePhotometryInterpretationStatus):
        raise TypeError(f"{context}.status must be SourcePhotometryInterpretationStatus")
    _optional_non_blank(issue, f"{context}.issue")

    if status is SourcePhotometryInterpretationStatus.ABSENT:
        if evidence or issue is not None or has_value:
            raise ValueError(f"{context} absent state must not carry evidence values or issue")
        return

    if status is SourcePhotometryInterpretationStatus.RESOLVED:
        if not evidence:
            raise ValueError(f"{context} resolved state requires evidence")
        if not has_value:
            raise ValueError(f"{context} resolved state requires at least one interpreted value")
        if issue is not None:
            raise ValueError(f"{context} resolved state must not carry an issue")
        return

    if not evidence:
        raise ValueError(f"{context} unknown or invalid state requires evidence")
    if has_value:
        raise ValueError(f"{context} unknown or invalid state must not expose interpreted values")
    if issue is None:
        raise ValueError(f"{context} unknown or invalid state requires an issue")


@dataclass(frozen=True, slots=True)
class SourceColorMetadata:
    """Declared source-color metadata without decoded/working/output color semantics."""

    status: SourcePhotometryInterpretationStatus
    declared_color_space: str | None = None
    declared_primaries: str | None = None
    declared_transfer_characteristic: str | None = None
    declared_matrix_coefficients: str | None = None
    declared_range: str | None = None
    profile_name: str | None = None
    evidence: tuple[RawMetadataEntry, ...] = ()
    issue: str | None = None

    def __post_init__(self) -> None:
        evidence = _canonical_evidence(self.evidence, "source_color.evidence")
        object.__setattr__(self, "evidence", evidence)

        for field_name in (
            "declared_color_space",
            "declared_primaries",
            "declared_transfer_characteristic",
            "declared_matrix_coefficients",
            "declared_range",
            "profile_name",
        ):
            _optional_non_blank(getattr(self, field_name), f"source_color.{field_name}")

        has_value = any(
            getattr(self, field_name) is not None
            for field_name in (
                "declared_color_space",
                "declared_primaries",
                "declared_transfer_characteristic",
                "declared_matrix_coefficients",
                "declared_range",
                "profile_name",
            )
        )
        _validate_interpretation_state(
            status=self.status,
            evidence=evidence,
            issue=self.issue,
            has_value=has_value,
            context="source_color",
        )


@dataclass(frozen=True, slots=True)
class SourceExposureMetadata:
    """Capture exposure hints retained separately from photometric normalization."""

    status: SourcePhotometryInterpretationStatus
    iso_speed: float | None = None
    exposure_time_seconds: float | None = None
    f_number: float | None = None
    exposure_compensation_ev: float | None = None
    evidence: tuple[RawMetadataEntry, ...] = ()
    issue: str | None = None

    def __post_init__(self) -> None:
        evidence = _canonical_evidence(self.evidence, "source_exposure.evidence")
        object.__setattr__(self, "evidence", evidence)

        _validate_finite_float(self.iso_speed, "source_exposure.iso_speed", positive=True)
        _validate_finite_float(
            self.exposure_time_seconds,
            "source_exposure.exposure_time_seconds",
            positive=True,
        )
        _validate_finite_float(self.f_number, "source_exposure.f_number", positive=True)
        _validate_finite_float(
            self.exposure_compensation_ev,
            "source_exposure.exposure_compensation_ev",
        )

        has_value = any(
            value is not None
            for value in (
                self.iso_speed,
                self.exposure_time_seconds,
                self.f_number,
                self.exposure_compensation_ev,
            )
        )
        _validate_interpretation_state(
            status=self.status,
            evidence=evidence,
            issue=self.issue,
            has_value=has_value,
            context="source_exposure",
        )


@dataclass(frozen=True, slots=True)
class SourceWhiteBalanceMetadata:
    """Capture white-balance hints without gains or chromatic adaptation."""

    status: SourcePhotometryInterpretationStatus
    mode: str | None = None
    color_temperature_kelvin: float | None = None
    evidence: tuple[RawMetadataEntry, ...] = ()
    issue: str | None = None

    def __post_init__(self) -> None:
        evidence = _canonical_evidence(self.evidence, "source_white_balance.evidence")
        object.__setattr__(self, "evidence", evidence)

        _optional_non_blank(self.mode, "source_white_balance.mode")
        _validate_finite_float(
            self.color_temperature_kelvin,
            "source_white_balance.color_temperature_kelvin",
            positive=True,
        )

        _validate_interpretation_state(
            status=self.status,
            evidence=evidence,
            issue=self.issue,
            has_value=self.mode is not None or self.color_temperature_kelvin is not None,
            context="source_white_balance",
        )


@dataclass(frozen=True, slots=True)
class SourcePhotometryMetadata:
    """Photometric source metadata bound to one exact immutable raw metadata record."""

    observation_id: ObservationId
    source_metadata: ObservationMetadata
    color: SourceColorMetadata
    exposure: SourceExposureMetadata
    white_balance: SourceWhiteBalanceMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, ObservationId):
            raise TypeError("source_photometry.observation_id must be ObservationId")
        if not isinstance(self.source_metadata, ObservationMetadata):
            raise TypeError("source_photometry.source_metadata must be ObservationMetadata")
        if not isinstance(self.color, SourceColorMetadata):
            raise TypeError("source_photometry.color must be SourceColorMetadata")
        if not isinstance(self.exposure, SourceExposureMetadata):
            raise TypeError("source_photometry.exposure must be SourceExposureMetadata")
        if not isinstance(self.white_balance, SourceWhiteBalanceMetadata):
            raise TypeError("source_photometry.white_balance must be SourceWhiteBalanceMetadata")
        if self.source_metadata.observation_id != self.observation_id:
            raise ValueError(
                "source_photometry observation_id must match source_metadata.observation_id"
            )

        source_entries = set(self.source_metadata.raw_entries)
        for context, evidence in (
            ("source_color", self.color.evidence),
            ("source_exposure", self.exposure.evidence),
            ("source_white_balance", self.white_balance.evidence),
        ):
            foreign = tuple(entry for entry in evidence if entry not in source_entries)
            if foreign:
                raise ValueError(
                    f"{context}.evidence must come from source_metadata.raw_entries"
                )
