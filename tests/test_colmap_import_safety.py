from __future__ import annotations

import wre.domain as domain
import wre.domain.estimated_geometry as geometry
import wre.reconstruction as reconstruction
import wre.reconstruction.colmap_import as colmap_import


def test_l36_estimated_geometry_api_is_publicly_exported() -> None:
    assert domain.CameraCalibrationEstimate is geometry.CameraCalibrationEstimate
    assert domain.CameraPoseEstimate is geometry.CameraPoseEstimate
    assert domain.EstimatedPoint3DId is geometry.EstimatedPoint3DId
    assert domain.EstimatedTrackElement is geometry.EstimatedTrackElement
    assert domain.LocalScaleStatus is geometry.LocalScaleStatus
    assert domain.Point3DEstimate is geometry.Point3DEstimate
    assert domain.SparseReconstructionEstimate is geometry.SparseReconstructionEstimate


def test_l36_colmap_import_api_is_publicly_exported() -> None:
    assert reconstruction.COLMAP_IMPORTER_VERSION == colmap_import.COLMAP_IMPORTER_VERSION
    assert (
        reconstruction.ColmapReconstructionImportConfig
        is colmap_import.ColmapReconstructionImportConfig
    )
    assert (
        reconstruction.ColmapReconstructionImportRequest
        is colmap_import.ColmapReconstructionImportRequest
    )
    assert (
        reconstruction.ColmapReconstructionImportResult
        is colmap_import.ColmapReconstructionImportResult
    )
    assert reconstruction.ImportedColmapSparseModel is colmap_import.ImportedColmapSparseModel
    assert reconstruction.import_colmap_reconstruction is colmap_import.import_colmap_reconstruction
