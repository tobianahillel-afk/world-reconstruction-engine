from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from wre.domain import HardwareRuntimeIdentity, Sha256Digest


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def test_hardware_runtime_identity_is_typed_immutable_and_hashable() -> None:
    first = HardwareRuntimeIdentity(sha256=_digest("a"))
    same = HardwareRuntimeIdentity(sha256=_digest("a"))
    other = HardwareRuntimeIdentity(sha256=_digest("b"))

    assert first == same
    assert hash(first) == hash(same)
    assert first != other

    with pytest.raises(FrozenInstanceError):
        first.sha256 = _digest("c")  # type: ignore[misc]


def test_hardware_runtime_identity_rejects_raw_digest_substitutes() -> None:
    with pytest.raises(TypeError, match=r"hardware_runtime\.sha256"):
        HardwareRuntimeIdentity(sha256=cast(Any, "a" * 64))
