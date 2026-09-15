from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

import wre.reconstruction as reconstruction
import wre.reconstruction.colmap_verification as verification
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRunId
from wre.reconstruction.colmap_environment import ColmapEnvironmentIdentity


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


def test_pair_evidence_rejects_non_integer_raw_match_count() -> None:
    with pytest.raises(ValueError, match="raw_match_count must be a non-negative integer"):
        verification.ColmapPairGeometryEvidence(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            image_name1="a.pgm",
            image_name2="b.pgm",
            raw_match_count=cast(Any, 1.5),
            configuration="UNDEFINED",
            inlier_matches=(),
            fundamental_matrix=None,
            essential_matrix=None,
            homography_matrix=None,
            relative_pose_matrix=None,
            triangulation_angle_rad=None,
            has_estimated_camera1=False,
            has_estimated_camera2=False,
        )


def test_result_rejects_non_integer_database_byte_length(tmp_path: Path) -> None:
    digest = Sha256Digest("0" * 64)
    with pytest.raises(ValueError, match="database_byte_length must be a positive integer"):
        verification.ColmapGeometricVerificationResult(
            provenance=DerivedArtifactProvenance(
                producing_run_id=ReconstructionRunId("run:test"),
                source_observation_ids=(ObservationId("obs:a"),),
            ),
            environment=ColmapEnvironmentIdentity(
                pycolmap_version="4.2.0",
                colmap_version="COLMAP 4.2.0",
                colmap_build="test build",
                ceres_version="2.2.0",
                upstream_has_cuda=False,
            ),
            configuration_sha256=digest,
            source_matching_database_sha256=digest,
            database_path=tmp_path / "verified.db",
            database_sha256=digest,
            database_byte_length=cast(Any, 1.5),
            geometries=(),
        )


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
