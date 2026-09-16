from __future__ import annotations

import re
from dataclasses import dataclass

from wre.domain.artifacts import ArtifactKind

_ADAPTER_CAPABILITY_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")


@dataclass(frozen=True, slots=True, order=True)
class AdapterCapabilityName:
    """Open solver-independent semantic capability identity."""

    value: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.value, str)
            or _ADAPTER_CAPABILITY_NAME_RE.fullmatch(self.value) is None
        ):
            raise ValueError(
                "adapter_capability_name must be a 1-128 character lowercase token using "
                "letters, digits, '.', '_', ':' or '-'"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class AdapterCapabilityDescriptor:
    """Semantic artifact-kind contract for one adapter capability."""

    capability: AdapterCapabilityName
    input_kinds: frozenset[ArtifactKind]
    output_kinds: frozenset[ArtifactKind]

    def __post_init__(self) -> None:
        if not isinstance(self.capability, AdapterCapabilityName):
            raise TypeError("adapter_capability.capability must be AdapterCapabilityName")
        if not isinstance(self.input_kinds, frozenset):
            raise TypeError("adapter_capability.input_kinds must be an immutable frozenset")
        if not isinstance(self.output_kinds, frozenset):
            raise TypeError("adapter_capability.output_kinds must be an immutable frozenset")
        if any(not isinstance(kind, ArtifactKind) for kind in self.input_kinds):
            raise TypeError("adapter_capability.input_kinds members must be ArtifactKind")
        if any(not isinstance(kind, ArtifactKind) for kind in self.output_kinds):
            raise TypeError("adapter_capability.output_kinds members must be ArtifactKind")
        if not self.output_kinds:
            raise ValueError("adapter_capability.output_kinds must not be empty")
