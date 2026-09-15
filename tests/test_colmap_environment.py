from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from wre.reconstruction import (
    SUPPORTED_COLMAP_VERSION,
    SUPPORTED_PYCOLMAP_VERSION,
    ColmapEnvironmentError,
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)


@dataclass
class _FakePycolmap:
    __version__: object = SUPPORTED_PYCOLMAP_VERSION
    COLMAP_version: object = SUPPORTED_COLMAP_VERSION
    COLMAP_build: object = "Commit abc123 on 2026-09-01 without GPU support"
    __ceres_version__: object = "2.2.0"
    has_cuda: object = False


def test_valid_environment_identity_preserves_upstream_facts() -> None:
    identity = inspect_colmap_environment(_FakePycolmap())

    assert identity == ColmapEnvironmentIdentity(
        pycolmap_version="4.2.0",
        colmap_version="COLMAP 4.2.0",
        colmap_build="Commit abc123 on 2026-09-01 without GPU support",
        ceres_version="2.2.0",
        upstream_has_cuda=False,
    )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("__version__", "4.1.0", "unsupported PyCOLMAP version"),
        ("COLMAP_version", "COLMAP 4.1.0", "unsupported COLMAP version"),
        ("COLMAP_build", "", "COLMAP_build"),
        ("__ceres_version__", None, "__ceres_version__"),
        ("has_cuda", "false", "has_cuda"),
    ],
)
def test_incompatible_environment_is_rejected(
    attribute: str,
    value: object,
    message: str,
) -> None:
    module = _FakePycolmap()
    setattr(module, attribute, value)

    with pytest.raises(ColmapEnvironmentError, match=message):
        inspect_colmap_environment(module)


def test_missing_pycolmap_is_wrapped_as_explicit_environment_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing(_: str) -> object:
        raise ModuleNotFoundError("no pycolmap")

    monkeypatch.setattr("wre.reconstruction.colmap_environment.importlib.import_module", _missing)

    with pytest.raises(ColmapEnvironmentError, match="PyCOLMAP is unavailable"):
        inspect_colmap_environment()


def test_broken_pycolmap_backend_is_wrapped_as_explicit_environment_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken(_: str) -> object:
        raise RuntimeError("cannot import pycolmap backend")

    monkeypatch.setattr("wre.reconstruction.colmap_environment.importlib.import_module", _broken)

    with pytest.raises(ColmapEnvironmentError, match="PyCOLMAP is unavailable"):
        inspect_colmap_environment()


def test_real_pycolmap_420_environment_when_integration_lane_enabled() -> None:
    if os.environ.get("WRE_COLMAP_INTEGRATION") != "1":
        pytest.skip("real PyCOLMAP environment is exercised only in the COLMAP integration lane")

    identity = inspect_colmap_environment()

    assert identity.pycolmap_version == SUPPORTED_PYCOLMAP_VERSION
    assert identity.colmap_version == SUPPORTED_COLMAP_VERSION
    assert identity.colmap_build
    assert identity.ceres_version
    assert isinstance(identity.upstream_has_cuda, bool)
