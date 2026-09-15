from __future__ import annotations

import importlib
from dataclasses import dataclass

SUPPORTED_PYCOLMAP_VERSION = "4.2.0"
SUPPORTED_COLMAP_VERSION = "COLMAP 4.2.0"


class ColmapEnvironmentError(RuntimeError):
    """Raised when the external PyCOLMAP environment is missing or incompatible."""


@dataclass(frozen=True, slots=True)
class ColmapEnvironmentIdentity:
    """Immutable provenance facts reported by one compatible PyCOLMAP environment."""

    pycolmap_version: str
    colmap_version: str
    colmap_build: str
    ceres_version: str
    upstream_has_cuda: bool


def _required_text(module: object, attribute: str) -> str:
    value = getattr(module, attribute, None)
    if not isinstance(value, str) or not value.strip():
        raise ColmapEnvironmentError(f"pycolmap.{attribute} must be a non-empty string")
    return value


def _load_pycolmap() -> object:
    try:
        return importlib.import_module("pycolmap")
    except (ImportError, RuntimeError) as exc:
        raise ColmapEnvironmentError(
            "PyCOLMAP is unavailable; install the approved external pycolmap==4.2.0 environment"
        ) from exc


def inspect_colmap_environment(
    module: object | None = None,
) -> ColmapEnvironmentIdentity:
    """Validate and describe the exact COLMAP environment supported by L3."""

    pycolmap = module if module is not None else _load_pycolmap()
    binding_version = _required_text(pycolmap, "__version__")
    if binding_version != SUPPORTED_PYCOLMAP_VERSION:
        raise ColmapEnvironmentError(
            "unsupported PyCOLMAP version: "
            f"expected {SUPPORTED_PYCOLMAP_VERSION!r}, found {binding_version!r}"
        )

    colmap_version = _required_text(pycolmap, "COLMAP_version")
    if colmap_version != SUPPORTED_COLMAP_VERSION:
        raise ColmapEnvironmentError(
            "unsupported COLMAP version: "
            f"expected {SUPPORTED_COLMAP_VERSION!r}, found {colmap_version!r}"
        )

    colmap_build = _required_text(pycolmap, "COLMAP_build")
    ceres_version = _required_text(pycolmap, "__ceres_version__")
    upstream_has_cuda = getattr(pycolmap, "has_cuda", None)
    if not isinstance(upstream_has_cuda, bool):
        raise ColmapEnvironmentError("pycolmap.has_cuda must be a boolean")

    return ColmapEnvironmentIdentity(
        pycolmap_version=binding_version,
        colmap_version=colmap_version,
        colmap_build=colmap_build,
        ceres_version=ceres_version,
        upstream_has_cuda=upstream_has_cuda,
    )
