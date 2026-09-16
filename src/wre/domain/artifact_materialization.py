from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath

from wre.domain.artifacts import ArtifactRef
from wre.domain.observations import Sha256Digest


def _validate_relative_path(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("artifact_materialization_entry.relative_path must be str")
    if not value:
        raise ValueError("artifact materialization relative path must not be empty")
    if "\x00" in value:
        raise ValueError("artifact materialization relative path must not contain NUL")
    if "\\" in value:
        raise ValueError("artifact materialization relative path must use POSIX separators")

    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("artifact materialization relative path must not be absolute")
    if PureWindowsPath(value).drive:
        raise ValueError("artifact materialization relative path must not contain a drive")

    parts = value.split("/")
    if any(part == "" for part in parts):
        raise ValueError("artifact materialization relative path must not contain empty segments")
    if any(part == "." for part in parts):
        raise ValueError("artifact materialization relative path must not contain dot segments")
    if any(part == ".." for part in parts):
        raise ValueError("artifact materialization relative path must not contain parent traversal")
    if str(path) != value:
        raise ValueError("artifact materialization relative path must be canonical POSIX form")


@dataclass(frozen=True, slots=True)
class ArtifactMaterializationEntry:
    """Expected immutable content identity for one local materialized output file."""

    relative_path: str
    sha256: Sha256Digest
    byte_length: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path)
        if not isinstance(self.sha256, Sha256Digest):
            raise TypeError("artifact_materialization_entry.sha256 must be Sha256Digest")
        if (
            isinstance(self.byte_length, bool)
            or not isinstance(self.byte_length, int)
            or self.byte_length < 0
        ):
            raise ValueError(
                "artifact_materialization_entry.byte_length must be a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class ArtifactMaterializationMetadata:
    """Immutable local-file materialization manifest for one exact artifact."""

    artifact_ref: ArtifactRef
    entries: tuple[ArtifactMaterializationEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("artifact_materialization_metadata.artifact_ref must be ArtifactRef")
        if not isinstance(self.entries, tuple):
            raise TypeError("artifact_materialization_metadata.entries must be an immutable tuple")
        if not self.entries:
            raise ValueError("artifact materialization metadata must contain at least one entry")
        if not all(isinstance(entry, ArtifactMaterializationEntry) for entry in self.entries):
            raise TypeError(
                "artifact_materialization_metadata.entries must contain "
                "ArtifactMaterializationEntry values"
            )

        paths = tuple(entry.relative_path for entry in self.entries)
        if len(set(paths)) != len(paths):
            raise ValueError("artifact materialization metadata paths must be unique")
        if paths != tuple(sorted(paths)):
            raise ValueError(
                "artifact materialization metadata entries must be sorted by relative path"
            )


class ArtifactMaterializationVerificationStatus(str, Enum):
    VERIFIED = "verified"
    MISSING = "missing"
    CORRUPTED = "corrupted"


@dataclass(frozen=True, slots=True)
class ArtifactMaterializationVerification:
    """Read-only verification outcome for one exact artifact materialization."""

    artifact_ref: ArtifactRef
    status: ArtifactMaterializationVerificationStatus

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("artifact_materialization_verification.artifact_ref must be ArtifactRef")
        if not isinstance(self.status, ArtifactMaterializationVerificationStatus):
            raise TypeError(
                "artifact_materialization_verification.status must be "
                "ArtifactMaterializationVerificationStatus"
            )
