from __future__ import annotations

import wre.reconstruction as reconstruction
import wre.reconstruction.colmap_reconstruction as incremental


def test_l35_incremental_reconstruction_api_is_publicly_exported() -> None:
    assert (
        reconstruction.ColmapIncrementalReconstructionConfig
        is incremental.ColmapIncrementalReconstructionConfig
    )
    assert (
        reconstruction.ColmapIncrementalReconstructionRequest
        is incremental.ColmapIncrementalReconstructionRequest
    )
    assert (
        reconstruction.ColmapIncrementalReconstructionResult
        is incremental.ColmapIncrementalReconstructionResult
    )
    assert reconstruction.ColmapReconstructionInput is incremental.ColmapReconstructionInput
    assert reconstruction.ColmapModelFileArtifact is incremental.ColmapModelFileArtifact
    assert reconstruction.ColmapSparseModelArtifact is incremental.ColmapSparseModelArtifact
    assert (
        reconstruction.reconstruct_colmap_incrementally
        is incremental.reconstruct_colmap_incrementally
    )
