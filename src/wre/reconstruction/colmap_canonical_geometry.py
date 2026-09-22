from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from wre.domain.adapter_capabilities import AdapterCapabilityDescriptor, AdapterCapabilityName
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
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_adapter import COLMAP_PRECISION_OUTPUT_KINDS
from wre.reconstruction.colmap_environment import (
    SUPPORTED_PYCOLMAP_VERSION,
    ColmapEnvironmentError,
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_evidence_artifacts import (
    GEOMETRIC_VERIFICATION_KIND,
    IMAGE_OBSERVATION_KIND,
)
from wre.reconstruction.colmap_features import ColmapFeatureExtractionResult
from wre.reconstruction.colmap_reconstruction import ColmapSparseModelArtifact

COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID = "colmap.incremental_precision_geometry"
COLMAP_INCREMENTAL_CANONICAL_CAPABILITY_NAME = AdapterCapabilityName(
    "geometry.precision_sfm.incremental"
)
COLMAP_INCREMENTAL_CANONICAL_CAPABILITY = AdapterCapabilityDescriptor(
    capability=COLMAP_INCREMENTAL_CANONICAL_CAPABILITY_NAME,
    input_kinds=frozenset({IMAGE_OBSERVATION_KIND, GEOMETRIC_VERIFICATION_KIND}),
    output_kinds=COLMAP_PRECISION_OUTPUT_KINDS,
)
COLMAP_INCREMENTAL_CANONICAL_PRODUCER_IMPLEMENTATION = "pycolmap.incremental_mapping"
COLMAP_INCREMENTAL_CANONICAL_PRODUCER_VERSION = SUPPORTED_PYCOLMAP_VERSION
COLMAP_INCREMENTAL_CANONICAL_DEPENDENCY_REF = "colmap"
COLMAP_INCREMENTAL_CANONICAL_MODEL = None
COLMAP_INCREMENTAL_CANONICAL_CHECKPOINT = None
COLMAP_INCREMENTAL_CANONICAL_ARTIFACT_KEY_HARDWARE_POLICY = "required"
COLMAP_INCREMENTAL_CANONICAL_SHIPPING_STATUS = "experimental"
COLMAP_INCREMENTAL_CANONICAL_REPRODUCIBILITY_NOTES = (
    "V2L12.3 executes the retained deterministic PyCOLMAP 4.2.0 incremental mapper "
    "and normalizes each audited native sparse model directly into canonical CameraSolution, "
    "PointMap and GeometrySolution values in one explicit unresolved-scale local frame. "
    "The retained legacy sparse estimate is regression evidence only, not the V2 output path."
)


class ColmapCanonicalGeometryError(RuntimeError):
    """Raised when audited native COLMAP geometry cannot be normalized safely."""


@dataclass(frozen=True, slots=True)
class CanonicalColmapSparseModel:
    """Canonical V2 geometry emitted from exactly one audited native COLMAP model."""

    source_model_index: int
    source_model_identity_sha256: Sha256Digest
    camera_solutions: tuple[CameraSolution, ...]
    point_map: PointMap
    geometry_solution: GeometrySolution

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_model_index, bool)
            or not isinstance(self.source_model_index, int)
            or self.source_model_index < 0
        ):
            raise ValueError("source_model_index must be a non-negative integer")
        if not isinstance(self.source_model_identity_sha256, Sha256Digest):
            raise TypeError("source_model_identity_sha256 must be Sha256Digest")
        if not isinstance(self.camera_solutions, tuple):
            raise TypeError("camera_solutions must be an immutable tuple")
        if not self.camera_solutions:
            raise ValueError("canonical COLMAP model must contain at least one CameraSolution")
        if any(not isinstance(item, CameraSolution) for item in self.camera_solutions):
            raise TypeError("camera_solutions members must be CameraSolution")
        camera_ids = tuple(item.solution_id.value for item in self.camera_solutions)
        if camera_ids != tuple(sorted(camera_ids)) or len(camera_ids) != len(set(camera_ids)):
            raise ValueError("camera_solutions must be unique and canonically ordered")
        if not isinstance(self.point_map, PointMap):
            raise TypeError("point_map must be PointMap")
        if not isinstance(self.geometry_solution, GeometrySolution):
            raise TypeError("geometry_solution must be GeometrySolution")

        local_frame = self.geometry_solution.local_frame_id
        if self.point_map.local_frame_id is not local_frame:
            raise ValueError("canonical COLMAP children must reuse the exact LocalFrameId object")
        if any(item.local_frame_id is not local_frame for item in self.camera_solutions):
            raise ValueError("canonical COLMAP children must reuse the exact LocalFrameId object")
        if self.geometry_solution.camera_solution_ids != tuple(
            item.solution_id for item in self.camera_solutions
        ):
            raise ValueError("geometry_solution must reference the exact emitted camera IDs")
        if self.geometry_solution.point_map_ids != (self.point_map.point_map_id,):
            raise ValueError("geometry_solution must reference the exact emitted PointMap")
        if self.geometry_solution.depth_field_ids:
            raise ValueError("classical sparse COLMAP geometry must not invent depth fields")
        if self.geometry_solution.scale_status is not GeometryScaleStatus.UNRESOLVED:
            raise ValueError("unanchored COLMAP sparse geometry must retain unresolved scale")


def _load_pycolmap() -> Any:
    try:
        return cast(Any, importlib.import_module("pycolmap"))
    except (ImportError, OSError, RuntimeError) as exc:
        raise ColmapEnvironmentError(
            "PyCOLMAP is unavailable; install the approved external pycolmap==4.2.0 environment"
        ) from exc


def colmap_sparse_model_content_identity(
    model: ColmapSparseModelArtifact,
) -> Sha256Digest:
    """Derive identity only from the audited source-model manifest."""

    if not isinstance(model, ColmapSparseModelArtifact):
        raise TypeError("model must be ColmapSparseModelArtifact")
    document = {
        "model_index": model.model_index,
        "files": [
            {
                "relative_path": item.relative_path,
                "sha256": item.sha256.value,
                "byte_length": item.byte_length,
            }
            for item in model.files
        ],
    }
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return Sha256Digest(hashlib.sha256(payload).hexdigest())


def _verified_native_model_path(
    output_path: Path,
    model: ColmapSparseModelArtifact,
) -> Path:
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be pathlib.Path")
    if not isinstance(model, ColmapSparseModelArtifact):
        raise TypeError("model must be ColmapSparseModelArtifact")

    root = output_path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ColmapCanonicalGeometryError("COLMAP reconstruction output must be a directory")

    candidate = root / model.relative_path
    if candidate.is_symlink():
        raise ColmapCanonicalGeometryError("COLMAP source model must not be a symbolic link")
    try:
        model_path = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ColmapCanonicalGeometryError("COLMAP source model directory is missing") from exc
    if model_path.parent != root or not model_path.is_dir():
        raise ColmapCanonicalGeometryError("COLMAP source model must be one real model directory")

    expected = {item.relative_path: item for item in model.files}
    actual: dict[str, Path] = {}
    for path in model_path.rglob("*"):
        if path.is_symlink():
            raise ColmapCanonicalGeometryError("COLMAP source model must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ColmapCanonicalGeometryError(
                "COLMAP source model contains a non-regular filesystem entry"
            )
        relative_path = path.relative_to(model_path).as_posix()
        if relative_path in actual:
            raise ColmapCanonicalGeometryError(
                "COLMAP source model contains duplicate relative file identity"
            )
        actual[relative_path] = path

    if set(actual) != set(expected):
        raise ColmapCanonicalGeometryError(
            "COLMAP source model file membership changed after mapper publication"
        )

    for relative_path, artifact in expected.items():
        digest = hash_file_content(actual[relative_path])
        if digest.sha256 != artifact.sha256 or digest.byte_length != artifact.byte_length:
            raise ColmapCanonicalGeometryError(
                f"COLMAP source model file changed after mapper publication: {relative_path}"
            )
    return model_path


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise ColmapCanonicalGeometryError(f"{context} must be finite")
    try:
        normalized = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ColmapCanonicalGeometryError(f"{context} must be finite") from exc
    if not math.isfinite(normalized):
        raise ColmapCanonicalGeometryError(f"{context} must be finite")
    return normalized


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ColmapCanonicalGeometryError(f"{context} must be a positive integer")
    return value


def _vector3(value: object, context: str) -> tuple[float, float, float]:
    try:
        items = tuple(cast(Any, value))
    except TypeError as exc:
        raise ColmapCanonicalGeometryError(f"{context} must contain three values") from exc
    if len(items) != 3:
        raise ColmapCanonicalGeometryError(f"{context} must contain three values")
    return (
        _finite_float(items[0], context),
        _finite_float(items[1], context),
        _finite_float(items[2], context),
    )


def _rotation_matrix(
    value: object,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    try:
        rows = tuple(cast(Any, value))
    except TypeError as exc:
        raise ColmapCanonicalGeometryError("COLMAP rotation must be a 3x3 matrix") from exc
    if len(rows) != 3:
        raise ColmapCanonicalGeometryError("COLMAP rotation must be a 3x3 matrix")
    return (
        _vector3(rows[0], "COLMAP rotation"),
        _vector3(rows[1], "COLMAP rotation"),
        _vector3(rows[2], "COLMAP rotation"),
    )


def _feature_name_map(features: ColmapFeatureExtractionResult) -> dict[str, ObservationId]:
    if not isinstance(features, ColmapFeatureExtractionResult):
        raise TypeError("features must be ColmapFeatureExtractionResult")
    mapping: dict[str, ObservationId] = {}
    for image in features.images:
        if image.image_name in mapping:
            raise ColmapCanonicalGeometryError(
                "COLMAP feature mapping contains duplicate image names"
            )
        mapping[image.image_name] = image.observation_id
    return mapping


def _empty_metrics() -> MetricVector:
    return MetricVector(observations=())


def _camera_solution_id(
    source_identity: Sha256Digest,
    image_id: int,
) -> CameraSolutionId:
    return CameraSolutionId(f"camera:colmap:{source_identity.value}:{image_id}")


def _point_map_id(source_identity: Sha256Digest) -> PointMapId:
    return PointMapId(f"points:colmap:{source_identity.value}")


def _geometry_solution_id(source_identity: Sha256Digest) -> GeometrySolutionId:
    return GeometrySolutionId(f"geometry:colmap:{source_identity.value}")


def canonicalize_colmap_sparse_model(
    *,
    output_path: Path,
    model_artifact: ColmapSparseModelArtifact,
    features: ColmapFeatureExtractionResult,
    expected_environment: ColmapEnvironmentIdentity,
    module: object | None = None,
) -> CanonicalColmapSparseModel:
    """Normalize one audited native COLMAP model directly into canonical V2 geometry."""

    if not isinstance(expected_environment, ColmapEnvironmentIdentity):
        raise TypeError("expected_environment must be ColmapEnvironmentIdentity")
    if (
        features.environment.pycolmap_version != expected_environment.pycolmap_version
        or features.environment.colmap_version != expected_environment.colmap_version
    ):
        raise ValueError("feature mapping and native model must use the same COLMAP format version")

    model_path = _verified_native_model_path(output_path, model_artifact)
    source_identity = colmap_sparse_model_content_identity(model_artifact)
    local_frame_id = LocalFrameId(f"frame:colmap:{source_identity.value}")

    pycolmap = cast(Any, module) if module is not None else _load_pycolmap()
    environment = inspect_colmap_environment(pycolmap)
    if environment != expected_environment:
        raise ColmapCanonicalGeometryError(
            "native COLMAP model must be read with the exact mapper PyCOLMAP environment"
        )

    try:
        reconstruction = pycolmap.Reconstruction(model_path)
    except Exception as exc:
        raise ColmapCanonicalGeometryError("PyCOLMAP could not read the audited model") from exc
    if not bool(reconstruction.is_valid()):
        raise ColmapCanonicalGeometryError("PyCOLMAP loaded an internally invalid model")

    try:
        registered_ids = sorted(int(value) for value in reconstruction.reg_image_ids())
        point_ids = sorted(int(value) for value in reconstruction.point3D_ids())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ColmapCanonicalGeometryError("COLMAP model IDs must be readable integers") from exc

    if len(registered_ids) != model_artifact.num_registered_images:
        raise ColmapCanonicalGeometryError(
            "registered-image count no longer matches the audited COLMAP model"
        )
    if len(point_ids) != model_artifact.num_points3d:
        raise ColmapCanonicalGeometryError(
            "3D-point count no longer matches the audited COLMAP model"
        )
    if not registered_ids:
        raise ColmapCanonicalGeometryError("an audited COLMAP model must register an image")
    if len(registered_ids) != len(set(registered_ids)):
        raise ColmapCanonicalGeometryError("COLMAP model contains duplicate registered image IDs")
    if len(point_ids) != len(set(point_ids)):
        raise ColmapCanonicalGeometryError("COLMAP model contains duplicate 3D point IDs")

    name_map = _feature_name_map(features)
    image_observations: dict[int, ObservationId] = {}
    camera_solutions: list[CameraSolution] = []

    for image_id in registered_ids:
        try:
            image = reconstruction.image(image_id)
            image_name = image.name
        except Exception as exc:
            raise ColmapCanonicalGeometryError(
                "COLMAP registered image is not readable"
            ) from exc
        if not isinstance(image_name, str):
            raise ColmapCanonicalGeometryError("COLMAP registered image name must be text")
        try:
            observation_id = name_map[image_name]
        except KeyError as exc:
            raise ColmapCanonicalGeometryError(
                f"registered COLMAP image has no observation mapping: {image_name}"
            ) from exc
        if observation_id in image_observations.values():
            raise ColmapCanonicalGeometryError(
                "one WRE observation appears more than once in a COLMAP model"
            )
        image_observations[image_id] = observation_id

        try:
            camera_id = int(image.camera_id)
            camera = reconstruction.camera(camera_id)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ColmapCanonicalGeometryError(
                "COLMAP registered image camera is not readable"
            ) from exc

        model_name = getattr(camera, "model_name", None)
        if not isinstance(model_name, str):
            raise ColmapCanonicalGeometryError("COLMAP camera model name must be text")
        try:
            projection_model = CameraProjectionModelName(model_name.lower())
        except ValueError as exc:
            raise ColmapCanonicalGeometryError(
                "COLMAP camera projection model is not a canonical token"
            ) from exc

        width = _positive_int(getattr(camera, "width", None), "COLMAP camera width")
        height = _positive_int(getattr(camera, "height", None), "COLMAP camera height")
        try:
            intrinsic_parameters = tuple(
                _finite_float(value, "COLMAP camera parameter") for value in camera.params
            )
        except (AttributeError, TypeError) as exc:
            raise ColmapCanonicalGeometryError(
                "COLMAP camera parameters must be an iterable sequence"
            ) from exc
        if not intrinsic_parameters:
            raise ColmapCanonicalGeometryError("COLMAP camera parameters must not be empty")

        try:
            transform = image.cam_from_world()
            rotation = _rotation_matrix(transform.rotation.matrix())
            translation = _vector3(transform.translation, "COLMAP pose translation")
        except ColmapCanonicalGeometryError:
            raise
        except Exception as exc:
            raise ColmapCanonicalGeometryError(
                "COLMAP registered image does not expose a valid camera-from-local pose"
            ) from exc

        try:
            camera_solution = CameraSolution(
                solution_id=_camera_solution_id(source_identity, image_id),
                observation_id=observation_id,
                local_frame_id=local_frame_id,
                projection_model=projection_model,
                dimensions=ImageDimensions(width_px=width, height_px=height),
                intrinsic_parameters=intrinsic_parameters,
                rotation_matrix=rotation,
                translation_xyz=translation,
                uncertainty_artifacts=(),
                metrics=_empty_metrics(),
            )
        except (TypeError, ValueError) as exc:
            raise ColmapCanonicalGeometryError(
                "COLMAP camera values do not satisfy the canonical CameraSolution contract"
            ) from exc
        camera_solutions.append(camera_solution)

    model_observations = tuple(
        sorted(image_observations.values(), key=lambda item: item.value)
    )
    positions: list[tuple[float, float, float]] = []
    for point_id in point_ids:
        try:
            point = reconstruction.point3D(point_id)
            position = _vector3(point.xyz, "COLMAP 3D point")
        except ColmapCanonicalGeometryError:
            raise
        except Exception as exc:
            raise ColmapCanonicalGeometryError("COLMAP 3D point is not readable") from exc
        positions.append(position)

    canonical_cameras = tuple(
        sorted(camera_solutions, key=lambda item: item.solution_id.value)
    )
    point_map = PointMap(
        point_map_id=_point_map_id(source_identity),
        local_frame_id=local_frame_id,
        source_observation_ids=model_observations,
        positions_xyz=tuple(positions),
        confidence=None,
        metrics=_empty_metrics(),
    )
    geometry_solution = GeometrySolution(
        geometry_solution_id=_geometry_solution_id(source_identity),
        local_frame_id=local_frame_id,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_ids=tuple(item.solution_id for item in canonical_cameras),
        depth_field_ids=(),
        point_map_ids=(point_map.point_map_id,),
        metrics=_empty_metrics(),
    )

    return CanonicalColmapSparseModel(
        source_model_index=model_artifact.model_index,
        source_model_identity_sha256=source_identity,
        camera_solutions=canonical_cameras,
        point_map=point_map,
        geometry_solution=geometry_solution,
    )
