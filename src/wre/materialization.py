from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from wre.domain.artifact_materialization import (
    ArtifactMaterializationMetadata,
    ArtifactMaterializationVerification,
    ArtifactMaterializationVerificationStatus,
)
from wre.domain.observations import Sha256Digest

_HASH_CHUNK_SIZE = 1024 * 1024


def _sha256_file(path: Path) -> Sha256Digest:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return Sha256Digest(digest.hexdigest())


def verify_local_artifact_materialization(
    metadata: ArtifactMaterializationMetadata,
    root: Path,
) -> ArtifactMaterializationVerification:
    """Verify declared local output files without mutating filesystem or metadata state."""

    if not isinstance(metadata, ArtifactMaterializationMetadata):
        raise TypeError("metadata must be ArtifactMaterializationMetadata")
    if not isinstance(root, Path):
        raise TypeError("root must be pathlib.Path")

    try:
        root_resolved = root.resolve(strict=False)
    except (OSError, RuntimeError):
        return ArtifactMaterializationVerification(
            artifact_ref=metadata.artifact_ref,
            status=ArtifactMaterializationVerificationStatus.CORRUPTED,
        )

    if root.exists() and not root.is_dir():
        return ArtifactMaterializationVerification(
            artifact_ref=metadata.artifact_ref,
            status=ArtifactMaterializationVerificationStatus.CORRUPTED,
        )

    missing = False
    corrupted = False

    for entry in metadata.entries:
        relative = PurePosixPath(entry.relative_path)
        candidate = root.joinpath(*relative.parts)
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            corrupted = True
            continue

        if not resolved.is_relative_to(root_resolved):
            corrupted = True
            continue
        if not resolved.exists():
            missing = True
            continue
        if not resolved.is_file():
            corrupted = True
            continue

        try:
            byte_length = resolved.stat().st_size
            if byte_length != entry.byte_length:
                corrupted = True
                continue
            if _sha256_file(resolved) != entry.sha256:
                corrupted = True
        except OSError:
            corrupted = True

    if corrupted:
        status = ArtifactMaterializationVerificationStatus.CORRUPTED
    elif missing:
        status = ArtifactMaterializationVerificationStatus.MISSING
    else:
        status = ArtifactMaterializationVerificationStatus.VERIFIED

    return ArtifactMaterializationVerification(
        artifact_ref=metadata.artifact_ref,
        status=status,
    )
