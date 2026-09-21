from __future__ import annotations

import math
import re
from dataclasses import dataclass

from wre.domain.camera_solutions import CameraSolutionId
from wre.domain.cameras import ImageDimensions
from wre.domain.metrics import MetricVector
from wre.domain.observations import ObservationId

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LOWER_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")


def _require_token(
    value: object,
    pattern: re.Pattern[str],
    context: str,
    message: str,
) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{context} {message}")


def _require_finite_float(value: object, context: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{context} must be float")
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")


def _pixel_count(dimensions: ImageDimensions) -> int:
    return dimensions.width_px * dimensions.height_px


def _validate_depth_values(
    values: object,
    validity: tuple[bool, ...],
    pixel_count: int,
) -> None:
    if not isinstance(values, tuple):
        raise TypeError("depth_field.depth_values must be an immutable tuple")
    if len(values) != pixel_count:
        raise ValueError("depth_field.depth_values length must match image pixel count")

    for index, value in enumerate(values):
        _require_finite_float(value, "depth_field.depth_values member")
        if validity[index]:
            if value <= 0.0:
                raise ValueError("depth_field valid pixels must have strictly positive depth")
        elif value != 0.0:
            raise ValueError("depth_field invalid pixels must use canonical 0.0 depth")


def _validate_validity(value: object, pixel_count: int) -> tuple[bool, ...]:
    if not isinstance(value, tuple):
        raise TypeError("depth_field.validity must be an immutable tuple")
    if len(value) != pixel_count:
        raise ValueError("depth_field.validity length must match image pixel count")
    if any(type(member) is not bool for member in value):
        raise TypeError("depth_field.validity members must be bool")
    return value


def _validate_confidence(
    value: object,
    validity: tuple[bool, ...],
    pixel_count: int,
) -> None:
    if value is None:
        return
    if not isinstance(value, tuple):
        raise TypeError("depth_field.confidence must be None or an immutable tuple")
    if len(value) != pixel_count:
        raise ValueError("depth_field.confidence length must match image pixel count")

    for index, member in enumerate(value):
        _require_finite_float(member, "depth_field.confidence member")
        if member < 0.0 or member > 1.0:
            raise ValueError(
                "depth_field.confidence members must be in the inclusive range 0.0 to 1.0"
            )
        if not validity[index] and member != 0.0:
            raise ValueError("depth_field invalid pixels must use canonical 0.0 confidence")


@dataclass(frozen=True, slots=True, order=True)
class DepthFieldId:
    """Opaque identity for one canonical per-observation depth field."""

    value: str

    def __post_init__(self) -> None:
        _require_token(
            self.value,
            _OPAQUE_ID_RE,
            "depth_field_id",
            "must be 1-128 characters using letters, digits, '.', '_', ':' or '-'",
        )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class DepthValueConventionName:
    """Open solver-independent semantic token describing stored depth values."""

    value: str

    def __post_init__(self) -> None:
        _require_token(
            self.value,
            _LOWER_TOKEN_RE,
            "depth_value_convention",
            "must be a 1-128 character lowercase token using letters, digits, '.', '_', ':' or '-'",
        )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class DepthField:
    """Canonical per-observation depth samples with explicit support and confidence.

    Values are row-major over the supplied dimensions. The convention is descriptive only:
    this contract performs no depth-convention or unit conversion and makes no metric,
    world, Earth-alignment or geographic-placement claim.
    """

    depth_field_id: DepthFieldId
    observation_id: ObservationId
    camera_solution_id: CameraSolutionId
    dimensions: ImageDimensions
    depth_value_convention: DepthValueConventionName
    depth_values: tuple[float, ...]
    validity: tuple[bool, ...]
    confidence: tuple[float, ...] | None
    metrics: MetricVector

    def __post_init__(self) -> None:
        if not isinstance(self.depth_field_id, DepthFieldId):
            raise TypeError("depth_field.depth_field_id must be DepthFieldId")
        if not isinstance(self.observation_id, ObservationId):
            raise TypeError("depth_field.observation_id must be ObservationId")
        if not isinstance(self.camera_solution_id, CameraSolutionId):
            raise TypeError("depth_field.camera_solution_id must be CameraSolutionId")
        if not isinstance(self.dimensions, ImageDimensions):
            raise TypeError("depth_field.dimensions must be ImageDimensions")
        if not isinstance(self.depth_value_convention, DepthValueConventionName):
            raise TypeError(
                "depth_field.depth_value_convention must be DepthValueConventionName"
            )
        if not isinstance(self.metrics, MetricVector):
            raise TypeError("depth_field.metrics must be MetricVector")

        pixel_count = _pixel_count(self.dimensions)
        validity = _validate_validity(self.validity, pixel_count)
        _validate_depth_values(self.depth_values, validity, pixel_count)
        _validate_confidence(self.confidence, validity, pixel_count)
