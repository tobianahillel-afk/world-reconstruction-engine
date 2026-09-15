from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from wre.domain import ObservationId, SceneProject, SceneProjectId


def test_scene_project_id_is_distinct_typed_identity() -> None:
    project_id = SceneProjectId("project:demo-01")

    assert str(project_id) == "project:demo-01"
    assert project_id == SceneProjectId("project:demo-01")
    assert hash(project_id) == hash(SceneProjectId("project:demo-01"))
    assert project_id != ObservationId("project:demo-01")


def test_scene_project_root_contains_only_explicit_project_identity() -> None:
    project_id = SceneProjectId("project:root")
    project = SceneProject(project_id=project_id)

    assert project.project_id is project_id
    assert project == SceneProject(project_id=SceneProjectId("project:root"))
    assert hash(project) == hash(SceneProject(project_id=SceneProjectId("project:root")))


def test_scene_project_id_rejects_invalid_values() -> None:
    for value in ("", " ", "bad id", "/absolute/path", "x" * 129):
        with pytest.raises(ValueError, match="scene_project_id"):
            SceneProjectId(value)


def test_scene_project_contract_is_immutable() -> None:
    project_id = SceneProjectId("project:immutable")
    project = SceneProject(project_id=project_id)

    with pytest.raises(FrozenInstanceError):
        project.project_id = SceneProjectId("project:other")  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        project_id.value = "project:other"  # type: ignore[misc]
