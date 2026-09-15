from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from wre.domain import ObservationId, SceneProjectId
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef


def test_artifact_id_is_distinct_typed_identity() -> None:
    artifact_id = ArtifactId("artifact:geometry-01")

    assert str(artifact_id) == "artifact:geometry-01"
    assert artifact_id == ArtifactId("artifact:geometry-01")
    assert hash(artifact_id) == hash(ArtifactId("artifact:geometry-01"))
    assert artifact_id != SceneProjectId("artifact:geometry-01")
    assert artifact_id != ObservationId("artifact:geometry-01")


def test_artifact_kind_is_open_but_lexically_strict() -> None:
    assert str(ArtifactKind("geometry.solution")) == "geometry.solution"
    assert str(ArtifactKind("4d.motion_field")) == "4d.motion_field"
    assert ArtifactKind("runtime:web") == ArtifactKind("runtime:web")


def test_artifact_ref_identifies_one_typed_artifact() -> None:
    artifact_ref = ArtifactRef(
        artifact_id=ArtifactId("artifact:surface-01"),
        artifact_kind=ArtifactKind("surface.model"),
    )

    assert artifact_ref == ArtifactRef(
        artifact_id=ArtifactId("artifact:surface-01"),
        artifact_kind=ArtifactKind("surface.model"),
    )
    assert hash(artifact_ref) == hash(
        ArtifactRef(
            artifact_id=ArtifactId("artifact:surface-01"),
            artifact_kind=ArtifactKind("surface.model"),
        )
    )
    assert artifact_ref != ArtifactRef(
        artifact_id=ArtifactId("artifact:surface-01"),
        artifact_kind=ArtifactKind("appearance.model"),
    )


def test_artifact_id_rejects_invalid_values() -> None:
    for value in ("", " ", "bad id", "/absolute/path", "x" * 129):
        with pytest.raises(ValueError, match="artifact_id"):
            ArtifactId(value)

    with pytest.raises(ValueError, match="artifact_id"):
        ArtifactId(cast(Any, 42))


def test_artifact_kind_rejects_invalid_values() -> None:
    for value in ("", " ", "Geometry.Solution", "/surface", "x" * 129):
        with pytest.raises(ValueError, match="artifact_kind"):
            ArtifactKind(value)

    with pytest.raises(ValueError, match="artifact_kind"):
        ArtifactKind(cast(Any, 42))


def test_artifact_ref_requires_typed_identity_and_kind() -> None:
    with pytest.raises(TypeError, match="artifact_ref.artifact_id"):
        ArtifactRef(
            artifact_id=cast(Any, "artifact:surface-01"),
            artifact_kind=ArtifactKind("surface.model"),
        )

    with pytest.raises(TypeError, match="artifact_ref.artifact_kind"):
        ArtifactRef(
            artifact_id=ArtifactId("artifact:surface-01"),
            artifact_kind=cast(Any, "surface.model"),
        )


def test_artifact_identity_contracts_are_immutable() -> None:
    artifact_id = ArtifactId("artifact:immutable")
    artifact_kind = ArtifactKind("geometry.solution")
    artifact_ref = ArtifactRef(artifact_id=artifact_id, artifact_kind=artifact_kind)

    with pytest.raises(FrozenInstanceError):
        artifact_id.value = "artifact:other"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        artifact_kind.value = "surface.model"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        artifact_ref.artifact_id = ArtifactId("artifact:other")  # type: ignore[misc]
