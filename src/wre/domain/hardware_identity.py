from __future__ import annotations

from dataclasses import dataclass

from wre.domain.observations import Sha256Digest


@dataclass(frozen=True, slots=True)
class HardwareRuntimeIdentity:
    """Exact caller-supplied identity for hardware/runtime-sensitive computation."""

    sha256: Sha256Digest

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, Sha256Digest):
            raise TypeError("hardware_runtime.sha256 must be Sha256Digest")
