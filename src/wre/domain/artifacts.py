from __future__ import annotations

import re
from dataclasses import dataclass

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ARTIFACT_KIND_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")


def _require_token(value: object, pattern: re.Pattern[str], context: str, message: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{context} {message}")


@dataclass(frozen=True, slots=True, order=True)
class ArtifactId:
    value: str

    def __post_init__(self) -> None:
        _require_token(
            self.value,
            _OPAQUE_ID_RE,
            "artifact_id",
            "must be 1-128 characters using letters, digits, '.', '_', ':' or '-'",
        )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class ArtifactKind:
    value: str

    def __post_init__(self) -> None:
        _require_token(
            self.value,
            _ARTIFACT_KIND_RE,
            "artifact_kind",
            "must be a 1-128 character lowercase token using letters, digits, '.', '_', ':' or '-'",
        )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: ArtifactId
    artifact_kind: ArtifactKind

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, ArtifactId):
            raise TypeError("artifact_ref.artifact_id must be ArtifactId")
        if not isinstance(self.artifact_kind, ArtifactKind):
            raise TypeError("artifact_ref.artifact_kind must be ArtifactKind")
