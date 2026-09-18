from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from wre.domain.cameras import ObservationMetadata, RawMetadataEntry
from wre.domain.observations import ObservationKind


class InputClass(StrEnum):
    STILL = "still"
    VIDEO = "video"
    EQUIRECTANGULAR_360 = "equirectangular_360"
    DRONE_AERIAL = "drone_aerial"
    FISHEYE = "fisheye"
    ROLLING_SHUTTER = "rolling_shutter"


class InputClassEvidenceKind(StrEnum):
    OBSERVED = "observed"
    CANDIDATE = "candidate"


@dataclass(frozen=True, slots=True)
class InputClassEvidence:
    input_class: InputClass
    evidence_kind: InputClassEvidenceKind
    evidence_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.input_class, InputClass):
            raise TypeError("input_class_evidence.input_class must be InputClass")
        if not isinstance(self.evidence_kind, InputClassEvidenceKind):
            raise TypeError("input_class_evidence.evidence_kind must be InputClassEvidenceKind")
        if not isinstance(self.evidence_keys, tuple):
            raise TypeError("input_class_evidence.evidence_keys must be an immutable tuple")
        if not self.evidence_keys:
            raise ValueError("input_class_evidence.evidence_keys must not be empty")
        if any(not isinstance(key, str) or not key.strip() for key in self.evidence_keys):
            raise ValueError("input_class_evidence.evidence_keys must contain non-blank strings")
        if len(set(self.evidence_keys)) != len(self.evidence_keys):
            raise ValueError("input_class_evidence.evidence_keys must be unique")
        if self.evidence_keys != tuple(sorted(self.evidence_keys)):
            raise ValueError("input_class_evidence.evidence_keys must use canonical lexical order")


_INPUT_CLASS_ORDER = (
    InputClass.STILL,
    InputClass.VIDEO,
    InputClass.EQUIRECTANGULAR_360,
    InputClass.DRONE_AERIAL,
    InputClass.FISHEYE,
    InputClass.ROLLING_SHUTTER,
)
_UAV_FAMILY_PATTERNS = (
    re.compile(r"\bmavic\b"),
    re.compile(r"\bphantom\b"),
    re.compile(r"\binspire\b"),
    re.compile(r"\bmatrice\b"),
    re.compile(r"\bavata\b"),
    re.compile(r"\banafi\b"),
    re.compile(r"\bskydio\s+(?:2\+?|x2|x10)\b"),
    re.compile(r"\bautel(?:\s+robotics)?\s+evo\b"),
)
_EXIF_CAMERA_KEYS = frozenset({"image make", "image model"})


def _evidence_key(entry: RawMetadataEntry) -> str:
    return f"{entry.namespace}:{entry.key}"


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").replace("_", " ").split())


def _entry_contains_phrase(entry: RawMetadataEntry, phrase: str) -> bool:
    normalized_phrase = _normalized_text(phrase)
    return normalized_phrase in _normalized_text(
        entry.key
    ) or normalized_phrase in _normalized_text(entry.value)


def _drone_evidence_keys(metadata: ObservationMetadata) -> tuple[str, ...]:
    keys = {
        _evidence_key(entry)
        for entry in metadata.raw_entries
        if entry.namespace.casefold() == "exif"
        and entry.key.casefold() in _EXIF_CAMERA_KEYS
        and any(pattern.search(_normalized_text(entry.value)) for pattern in _UAV_FAMILY_PATTERNS)
    }
    return tuple(sorted(keys))


def _phrase_evidence_keys(
    metadata: ObservationMetadata,
    phrase: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _evidence_key(entry)
                for entry in metadata.raw_entries
                if _entry_contains_phrase(entry, phrase)
            }
        )
    )


def evaluate_input_classification(
    observation_kind: ObservationKind,
    metadata: ObservationMetadata | None = None,
) -> tuple[InputClassEvidence, ...]:
    """Return only positive observed or candidate input-class evidence."""

    if not isinstance(observation_kind, ObservationKind):
        raise TypeError("input_classification.observation_kind must be ObservationKind")
    if metadata is not None and not isinstance(metadata, ObservationMetadata):
        raise TypeError("input_classification.metadata must be ObservationMetadata or None")

    by_class: dict[InputClass, InputClassEvidence] = {}

    observed_class = (
        InputClass.STILL if observation_kind is ObservationKind.IMAGE else InputClass.VIDEO
    )
    by_class[observed_class] = InputClassEvidence(
        input_class=observed_class,
        evidence_kind=InputClassEvidenceKind.OBSERVED,
        evidence_keys=("observation.kind",),
    )

    if metadata is not None:
        if (
            metadata.dimensions is not None
            and metadata.dimensions.width_px == 2 * metadata.dimensions.height_px
        ):
            by_class[InputClass.EQUIRECTANGULAR_360] = InputClassEvidence(
                input_class=InputClass.EQUIRECTANGULAR_360,
                evidence_kind=InputClassEvidenceKind.CANDIDATE,
                evidence_keys=("metadata.dimensions",),
            )

        drone_keys = _drone_evidence_keys(metadata)
        if drone_keys:
            by_class[InputClass.DRONE_AERIAL] = InputClassEvidence(
                input_class=InputClass.DRONE_AERIAL,
                evidence_kind=InputClassEvidenceKind.CANDIDATE,
                evidence_keys=drone_keys,
            )

        fisheye_keys = _phrase_evidence_keys(metadata, "fisheye")
        if fisheye_keys:
            by_class[InputClass.FISHEYE] = InputClassEvidence(
                input_class=InputClass.FISHEYE,
                evidence_kind=InputClassEvidenceKind.CANDIDATE,
                evidence_keys=fisheye_keys,
            )

        rolling_shutter_keys = _phrase_evidence_keys(metadata, "rolling shutter")
        if rolling_shutter_keys:
            by_class[InputClass.ROLLING_SHUTTER] = InputClassEvidence(
                input_class=InputClass.ROLLING_SHUTTER,
                evidence_kind=InputClassEvidenceKind.CANDIDATE,
                evidence_keys=rolling_shutter_keys,
            )

    return tuple(
        by_class[input_class] for input_class in _INPUT_CLASS_ORDER if input_class in by_class
    )
