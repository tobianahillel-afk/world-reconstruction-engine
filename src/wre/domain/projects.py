from __future__ import annotations

import re
from dataclasses import dataclass

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True, slots=True, order=True)
class SceneProjectId:
    value: str

    def __post_init__(self) -> None:
        if not _OPAQUE_ID_RE.fullmatch(self.value):
            raise ValueError(
                "scene_project_id must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SceneProject:
    project_id: SceneProjectId
