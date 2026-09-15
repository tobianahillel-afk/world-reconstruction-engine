from __future__ import annotations

from pathlib import Path

import pytest

import wre.reconstruction as reconstruction
import wre.reconstruction.colmap_verification as verification


def test_l34_geometric_verification_api_is_publicly_exported() -> None:
    assert (
        reconstruction.ColmapGeometricVerificationConfig
        is verification.ColmapGeometricVerificationConfig
    )
    assert (
        reconstruction.ColmapGeometricVerificationRequest
        is verification.ColmapGeometricVerificationRequest
    )
    assert (
        reconstruction.ColmapGeometricVerificationResult
        is verification.ColmapGeometricVerificationResult
    )
    assert reconstruction.ColmapPairGeometryEvidence is verification.ColmapPairGeometryEvidence
    assert reconstruction.verify_colmap_geometry is verification.verify_colmap_geometry


def test_atomic_publish_refuses_to_replace_competing_output(tmp_path: Path) -> None:
    working = tmp_path / "working.db"
    output = tmp_path / "verified.db"
    working.write_bytes(b"verified-artifact")
    output.write_bytes(b"competing-artifact")

    with pytest.raises(ValueError, match="must not already exist"):
        verification._publish_verified_database(working, output)

    assert working.read_bytes() == b"verified-artifact"
    assert output.read_bytes() == b"competing-artifact"


def test_atomic_publish_exposes_only_complete_working_artifact(tmp_path: Path) -> None:
    working = tmp_path / "working.db"
    output = tmp_path / "verified.db"
    working.write_bytes(b"verified-artifact")

    verification._publish_verified_database(working, output)

    assert working.read_bytes() == b"verified-artifact"
    assert output.read_bytes() == b"verified-artifact"
