from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.reconstruction.colmap_geometry_outcomes as outcome_module
from wre.domain.camera_solutions import (
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
)
from wre.domain.cameras import ImageDimensions
from wre.domain.fragments import LocalFrameId
from wre.domain.geometry_solutions import (
    GeometryScaleStatus,
    GeometrySolution,
    GeometrySolutionId,
)
from wre.domain.metrics import MetricVector
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.point_maps import PointMap, PointMapId
from wre.reconstruction.colmap_canonical_geometry import (
    COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
    CanonicalColmapSparseModel,
    colmap_sparse_model_content_identity,
)
from wre.reconstruction.colmap_geometry_outcomes import (
    ColmapGeometryOutcome,
    ColmapGeometryOutcomeState,
)
from wre.reconstruction.colmap_global_reconstruction import (
    COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
)
from wre.reconstruction.colmap_reconstruction import (
    ColmapModelFileArtifact,
    ColmapSparseModelArtifact,
)


def _native_model(index: int, digest_char: str) -> ColmapSparseModelArtifact:
    return ColmapSparseModelArtifact(
        model_index=index,
        relative_path=str(index),
        num_registered_images=1,
        num_points3d=1,
        files=(
            ColmapModelFileArtifact(
                relative_path="cameras.bin",
                sha256=Sha256Digest(digest_char * 64),
                byte_length=10,
            ),
            ColmapModelFileArtifact(
                relative_path="images.bin",
                sha256=Sha256Digest(digest_char * 64),
                byte_length=20,
            ),
            ColmapModelFileArtifact(
                relative_path="points3D.bin",
                sha256=Sha256Digest(digest_char * 64),
                byte_length=30,
            ),
        ),
    )


def _canonical_model(native: ColmapSparseModelArtifact) -> CanonicalColmapSparseModel:
    identity = colmap_sparse_model_content_identity(native)
    index = native.model_index
    local_frame = LocalFrameId(f"frame:colmap:{identity.value}")
    observation_id = ObservationId(f"obs:{index}")
    camera = CameraSolution(
        solution_id=CameraSolutionId(f"camera:test:{index}"),
        observation_id=observation_id,
        local_frame_id=local_frame,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(width_px=640, height_px=480),
        intrinsic_parameters=(500.0, 500.0, 320.0, 240.0),
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_xyz=(float(index), 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=MetricVector(observations=()),
    )
    point_map = PointMap(
        point_map_id=PointMapId(f"points:test:{index}"),
        local_frame_id=local_frame,
        source_observation_ids=(observation_id,),
        positions_xyz=((float(index), 1.0, 2.0),),
        confidence=None,
        metrics=MetricVector(observations=()),
    )
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:test:{index}"),
        local_frame_id=local_frame,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_ids=(camera.solution_id,),
        depth_field_ids=(),
        point_map_ids=(point_map.point_map_id,),
        metrics=MetricVector(observations=()),
    )
    return CanonicalColmapSparseModel(
        source_model_index=index,
        source_model_identity_sha256=identity,
        camera_solutions=(camera,),
        point_map=point_map,
        geometry_solution=geometry,
    )


def test_outcome_contract_has_exact_immutable_fields_and_state_vocabulary() -> None:
    assert tuple(field.name for field in fields(ColmapGeometryOutcome)) == (
        "source_adapter_id",
        "native_models",
        "canonical_models",
    )
    assert tuple(state.value for state in ColmapGeometryOutcomeState) == (
        "no_model",
        "single_model",
        "disconnected_models",
    )

    outcome = ColmapGeometryOutcome(
        source_adapter_id=COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
        native_models=(),
        canonical_models=(),
    )
    with pytest.raises(FrozenInstanceError):
        outcome.source_adapter_id = COLMAP_GLOBAL_CANONICAL_ADAPTER_ID  # type: ignore[misc]


@pytest.mark.parametrize(
    "source_adapter_id",
    [
        COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
        COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
    ],
)
def test_current_incremental_and_global_adapter_identities_are_accepted(
    source_adapter_id: str,
) -> None:
    outcome = ColmapGeometryOutcome(
        source_adapter_id=source_adapter_id,
        native_models=(),
        canonical_models=(),
    )

    assert outcome.source_adapter_id == source_adapter_id


@pytest.mark.parametrize(
    "source_adapter_id",
    [
        "colmap.precision_geometry",
        "colmap.sparse_sfm",
        "colmap.unknown_precision_geometry",
        "",
    ],
)
def test_unknown_legacy_or_ambiguous_source_adapter_identity_fails_closed(
    source_adapter_id: str,
) -> None:
    with pytest.raises(ValueError, match="current incremental or global"):
        ColmapGeometryOutcome(
            source_adapter_id=source_adapter_id,
            native_models=(),
            canonical_models=(),
        )


def test_zero_model_outcome_contains_no_geometry_payload_or_placeholder() -> None:
    outcome = ColmapGeometryOutcome(
        source_adapter_id=COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
        native_models=(),
        canonical_models=(),
    )

    assert outcome.state is ColmapGeometryOutcomeState.NO_MODEL
    assert outcome.model_count == 0
    assert outcome.has_geometry is False
    assert outcome.native_models == ()
    assert outcome.canonical_models == ()


def test_single_model_outcome_preserves_exact_canonical_payload_and_local_frame() -> None:
    native = _native_model(0, "a")
    canonical = _canonical_model(native)
    native_models = (native,)
    canonical_models = (canonical,)

    outcome = ColmapGeometryOutcome(
        source_adapter_id=COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
        native_models=native_models,
        canonical_models=canonical_models,
    )

    assert outcome.state is ColmapGeometryOutcomeState.SINGLE_MODEL
    assert outcome.model_count == 1
    assert outcome.has_geometry is True
    assert outcome.native_models is native_models
    assert outcome.canonical_models is canonical_models
    assert outcome.canonical_models[0] is canonical
    assert outcome.canonical_models[0].camera_solutions is canonical.camera_solutions
    assert outcome.canonical_models[0].point_map is canonical.point_map
    assert outcome.canonical_models[0].geometry_solution is canonical.geometry_solution
    assert canonical.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED


def test_disconnected_outcome_preserves_source_order_and_independent_local_frames() -> None:
    native_zero = _native_model(0, "a")
    native_one = _native_model(1, "b")
    canonical_zero = _canonical_model(native_zero)
    canonical_one = _canonical_model(native_one)
    native_models = (native_zero, native_one)
    canonical_models = (canonical_zero, canonical_one)

    outcome = ColmapGeometryOutcome(
        source_adapter_id=COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
        native_models=native_models,
        canonical_models=canonical_models,
    )

    assert outcome.state is ColmapGeometryOutcomeState.DISCONNECTED_MODELS
    assert outcome.model_count == 2
    assert outcome.native_models is native_models
    assert outcome.canonical_models is canonical_models
    assert outcome.canonical_models == (canonical_zero, canonical_one)
    assert outcome.canonical_models[0] is canonical_zero
    assert outcome.canonical_models[1] is canonical_one
    assert (
        canonical_zero.geometry_solution.local_frame_id
        != canonical_one.geometry_solution.local_frame_id
    )
    assert canonical_zero.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert canonical_one.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED


def test_mutable_native_or_canonical_collections_fail_closed_without_coercion() -> None:
    native = _native_model(0, "a")
    canonical = _canonical_model(native)

    with pytest.raises(TypeError, match="native_models must be an immutable tuple"):
        ColmapGeometryOutcome(
            source_adapter_id=COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
            native_models=cast(Any, [native]),
            canonical_models=(canonical,),
        )

    with pytest.raises(TypeError, match="canonical_models must be an immutable tuple"):
        ColmapGeometryOutcome(
            source_adapter_id=COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
            native_models=(native,),
            canonical_models=cast(Any, [canonical]),
        )


def test_duplicate_or_noncanonical_native_model_order_fails_closed() -> None:
    native_zero_a = _native_model(0, "a")
    native_zero_b = _native_model(0, "b")
    native_one = _native_model(1, "c")

    with pytest.raises(ValueError, match="unique model indices"):
        ColmapGeometryOutcome(
            source_adapter_id=COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
            native_models=(native_zero_a, native_zero_b),
            canonical_models=(_canonical_model(native_zero_a), _canonical_model(native_zero_b)),
        )

    with pytest.raises(ValueError, match="canonically ordered"):
        ColmapGeometryOutcome(
            source_adapter_id=COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
            native_models=(native_one, native_zero_a),
            canonical_models=(_canonical_model(native_one), _canonical_model(native_zero_a)),
        )


@pytest.mark.parametrize(
    ("native_models", "canonical_models", "message"),
    [
        (
            (_native_model(0, "a"),),
            (),
            "cardinality",
        ),
        (
            (),
            (_canonical_model(_native_model(0, "a")),),
            "cardinality",
        ),
        (
            (_native_model(0, "a"),),
            (_canonical_model(_native_model(1, "b")),),
            "model-index order",
        ),
        (
            (_native_model(0, "a"), _native_model(1, "b")),
            (
                _canonical_model(_native_model(1, "b")),
                _canonical_model(_native_model(0, "a")),
            ),
            "model-index order",
        ),
    ],
)
def test_missing_extra_foreign_or_reordered_canonical_models_fail_closed(
    native_models: tuple[ColmapSparseModelArtifact, ...],
    canonical_models: tuple[CanonicalColmapSparseModel, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ColmapGeometryOutcome(
            source_adapter_id=COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
            native_models=native_models,
            canonical_models=canonical_models,
        )


def test_canonical_content_identity_mismatch_fails_closed() -> None:
    audited_native = _native_model(0, "a")
    foreign_same_index = _native_model(0, "b")
    foreign_canonical = _canonical_model(foreign_same_index)

    with pytest.raises(ValueError, match="content identities"):
        ColmapGeometryOutcome(
            source_adapter_id=COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
            native_models=(audited_native,),
            canonical_models=(foreign_canonical,),
        )


def test_outcome_state_cannot_be_manually_labeled_contradicting_cardinality() -> None:
    constructor = cast(Any, ColmapGeometryOutcome)

    with pytest.raises(TypeError, match="unexpected keyword argument 'state'"):
        constructor(
            source_adapter_id=COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
            native_models=(),
            canonical_models=(),
            state=ColmapGeometryOutcomeState.SINGLE_MODEL,
        )


def test_outcome_module_owns_no_solver_filesystem_legacy_routing_or_later_layers() -> None:
    forbidden_names = {
        "Path",
        "sqlite3",
        "tempfile",
        "shutil",
        "pycolmap",
        "SparseReconstructionEstimate",
        "convert_sparse_reconstruction_estimate",
        "CameraId",
        "SurfaceModel",
        "SpatialFragment",
        "QualityDecision",
        "Router",
        "SQLiteLocalStore",
        "MasterScene",
        "RuntimeScene",
        "compare",
        "benchmark",
    }

    assert forbidden_names.isdisjoint(vars(outcome_module))
