from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from wre.domain.estimated_geometry import (
    CameraCalibrationEstimate,
    CameraCalibrationEstimateId,
    CameraPoseEstimate,
    EstimatedPoint3DId,
    EstimatedTrackElement,
    LocalScaleStatus,
    Point3DEstimate,
    SparseReconstructionEstimate,
    SparseReconstructionEstimateId,
)
from wre.domain.fragments import LocalFrameId
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRun
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_environment import (
    ColmapEnvironmentError,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_features import ColmapFeatureExtractionResult
from wre.reconstruction.colmap_reconstruction import (
    ColmapIncrementalReconstructionResult,
    ColmapSparseModelArtifact,
)

COLMAP_IMPORTER_VERSION = "1"
_IMPORTER_PRODUCER = "wre.colmap_reconstruction_importer"


class ColmapReconstructionImportError(RuntimeError):
    """Raised when solver-native COLMAP geometry cannot be imported safely."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapReconstructionImportConfig:
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("reconstruction import config schema_version must be 1")

    def canonical_document(self) -> dict[str, object]:
        return {"schema_version": self.schema_version}

    @property
    def sha256(self) -> Sha256Digest:
        payload = json.dumps(
            self.canonical_document(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return Sha256Digest(hashlib.sha256(payload).hexdigest())


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapReconstructionImportRequest:
    run: ReconstructionRun
    reconstruction: ColmapIncrementalReconstructionResult
    features: ColmapFeatureExtractionResult
    config: ColmapReconstructionImportConfig = field(
        default_factory=ColmapReconstructionImportConfig
    )

    def __post_init__(self) -> None:
        observation_ids = self.reconstruction.provenance.source_observation_ids
        if self.features.provenance.source_observation_ids != observation_ids:
            raise ValueError("L3.2 feature mapping and L3.5 reconstruction membership must match")
        if self.run.input_observation_ids != observation_ids:
            raise ValueError("ReconstructionRun inputs must exactly match L3.5 source observations")
        if self.run.producer.implementation != _IMPORTER_PRODUCER:
            raise ValueError(f"ReconstructionRun producer must be {_IMPORTER_PRODUCER!r}")
        if self.run.producer.version != COLMAP_IMPORTER_VERSION:
            raise ValueError(
                f"ReconstructionRun producer version must be {COLMAP_IMPORTER_VERSION!r}"
            )
        if self.run.configuration_sha256 != self.config.sha256:
            raise ValueError(
                "ReconstructionRun configuration SHA-256 must match the canonical import config"
            )
        if (
            self.features.environment.pycolmap_version
            != self.reconstruction.environment.pycolmap_version
            or self.features.environment.colmap_version
            != self.reconstruction.environment.colmap_version
        ):
            raise ValueError("L3.2 and L3.5 artifacts must use the same COLMAP format version")


@dataclass(frozen=True, slots=True)
class ImportedColmapSparseModel:
    source_model_index: int
    source_model_identity_sha256: Sha256Digest
    estimate: SparseReconstructionEstimate

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_model_index, bool)
            or not isinstance(self.source_model_index, int)
            or self.source_model_index < 0
        ):
            raise ValueError("source_model_index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ColmapReconstructionImportResult:
    provenance: DerivedArtifactProvenance
    configuration_sha256: Sha256Digest
    source_reconstruction_configuration_sha256: Sha256Digest
    models: tuple[ImportedColmapSparseModel, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.models, tuple):
            raise ValueError("models must be an immutable tuple")
        indices = tuple(item.source_model_index for item in self.models)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("imported COLMAP models must be unique and canonically ordered")

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def has_reconstruction(self) -> bool:
        return bool(self.models)


def _load_pycolmap() -> Any:
    try:
        return cast(Any, importlib.import_module("pycolmap"))
    except (ImportError, OSError, RuntimeError) as exc:
        raise ColmapEnvironmentError(
            "PyCOLMAP is unavailable; install the approved external pycolmap==4.2.0 environment"
        ) from exc


def _model_identity(model: ColmapSparseModelArtifact) -> Sha256Digest:
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


def _verify_native_model(
    reconstruction: ColmapIncrementalReconstructionResult,
    model: ColmapSparseModelArtifact,
) -> Path:
    model_path = (
        reconstruction.output_path / model.relative_path
    ).expanduser().resolve(strict=True)
    if not model_path.is_dir() or model_path.is_symlink():
        raise ColmapReconstructionImportError("COLMAP source model must be a real directory")

    expected = {item.relative_path: item for item in model.files}
    actual: dict[str, Path] = {}
    for path in model_path.rglob("*"):
        if path.is_symlink():
            raise ColmapReconstructionImportError("COLMAP source model must not contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ColmapReconstructionImportError(
                "COLMAP source model contains a non-regular filesystem entry"
            )
        actual[path.relative_to(model_path).as_posix()] = path

    if set(actual) != set(expected):
        raise ColmapReconstructionImportError(
            "COLMAP source model file membership changed after L3.5 publication"
        )
    for relative_path, artifact in expected.items():
        digest = hash_file_content(actual[relative_path])
        if digest.sha256 != artifact.sha256 or digest.byte_length != artifact.byte_length:
            raise ColmapReconstructionImportError(
                f"COLMAP source model file changed after L3.5 publication: {relative_path}"
            )
    return model_path


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise ColmapReconstructionImportError(f"{context} must be finite")
    try:
        normalized = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ColmapReconstructionImportError(f"{context} must be finite") from exc
    if not math.isfinite(normalized):
        raise ColmapReconstructionImportError(f"{context} must be finite")
    return normalized


def _vector3(value: object, context: str) -> tuple[float, float, float]:
    try:
        items = tuple(cast(Any, value))
    except TypeError as exc:
        raise ColmapReconstructionImportError(f"{context} must contain three values") from exc
    if len(items) != 3:
        raise ColmapReconstructionImportError(f"{context} must contain three values")
    return (
        _finite_float(items[0], context),
        _finite_float(items[1], context),
        _finite_float(items[2], context),
    )


def _rotation_matrix(value: object) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    try:
        rows = tuple(cast(Any, value))
    except TypeError as exc:
        raise ColmapReconstructionImportError("COLMAP rotation must be a 3x3 matrix") from exc
    if len(rows) != 3:
        raise ColmapReconstructionImportError("COLMAP rotation must be a 3x3 matrix")
    return (
        _vector3(rows[0], "COLMAP rotation"),
        _vector3(rows[1], "COLMAP rotation"),
        _vector3(rows[2], "COLMAP rotation"),
    )


def _feature_name_map(features: ColmapFeatureExtractionResult) -> dict[str, ObservationId]:
    mapping: dict[str, ObservationId] = {}
    for image in features.images:
        if image.image_name in mapping:
            raise ColmapReconstructionImportError(
                "L3.2 feature mapping contains duplicate image names"
            )
        mapping[image.image_name] = image.observation_id
    return mapping


def _import_model(
    *,
    run: ReconstructionRun,
    model_artifact: ColmapSparseModelArtifact,
    model_path: Path,
    name_map: dict[str, ObservationId],
    pycolmap: Any,
) -> ImportedColmapSparseModel:
    source_identity = _model_identity(model_artifact)
    identity_prefix = source_identity.value[:24]
    local_frame_id = LocalFrameId(f"frame:colmap:{identity_prefix}")
    estimate_id = SparseReconstructionEstimateId(f"sparse:colmap:{identity_prefix}")

    try:
        reconstruction = pycolmap.Reconstruction(model_path)
    except Exception as exc:
        raise ColmapReconstructionImportError("PyCOLMAP could not read the L3.5 model") from exc
    if not bool(reconstruction.is_valid()):
        raise ColmapReconstructionImportError("PyCOLMAP loaded an internally invalid L3.5 model")

    registered_ids = sorted(int(value) for value in reconstruction.reg_image_ids())
    if len(registered_ids) != model_artifact.num_registered_images:
        raise ColmapReconstructionImportError(
            "registered-image count no longer matches the audited L3.5 model"
        )
    point_ids = sorted(int(value) for value in reconstruction.point3D_ids())
    if len(point_ids) != model_artifact.num_points3d:
        raise ColmapReconstructionImportError(
            "3D-point count no longer matches the audited L3.5 model"
        )
    if not registered_ids:
        raise ColmapReconstructionImportError("a published L3.5 model must register an image")

    image_observations: dict[int, ObservationId] = {}
    camera_observations: dict[int, list[ObservationId]] = {}
    image_camera_ids: dict[int, int] = {}
    for image_id in registered_ids:
        image = reconstruction.image(image_id)
        image_name = str(image.name)
        try:
            observation_id = name_map[image_name]
        except KeyError as exc:
            raise ColmapReconstructionImportError(
                f"registered COLMAP image has no L3.2 observation mapping: {image_name}"
            ) from exc
        if observation_id in image_observations.values():
            raise ColmapReconstructionImportError(
                "one WRE observation appears more than once in a COLMAP model"
            )
        camera_id = int(image.camera_id)
        image_observations[image_id] = observation_id
        image_camera_ids[image_id] = camera_id
        camera_observations.setdefault(camera_id, []).append(observation_id)

    calibrations: list[CameraCalibrationEstimate] = []
    calibration_ids: dict[int, CameraCalibrationEstimateId] = {}
    for camera_id in sorted(camera_observations):
        camera = reconstruction.camera(camera_id)
        calibration_id = CameraCalibrationEstimateId(
            f"calibration:colmap:{identity_prefix}:{camera_id}"
        )
        calibration_ids[camera_id] = calibration_id
        try:
            parameters = tuple(
                _finite_float(value, "COLMAP camera parameter") for value in camera.params
            )
        except TypeError as exc:
            raise ColmapReconstructionImportError(
                "COLMAP camera parameters must be iterable"
            ) from exc
        source_observations = tuple(
            sorted(camera_observations[camera_id], key=lambda item: item.value)
        )
        calibrations.append(
            CameraCalibrationEstimate(
                calibration_id=calibration_id,
                projection_model=str(camera.model_name),
                width_px=int(camera.width),
                height_px=int(camera.height),
                parameters=parameters,
                has_prior_focal_length=bool(camera.has_prior_focal_length),
                provenance=DerivedArtifactProvenance(
                    producing_run_id=run.run_id,
                    source_observation_ids=source_observations,
                ),
            )
        )

    poses: list[CameraPoseEstimate] = []
    for image_id in registered_ids:
        image = reconstruction.image(image_id)
        transform = image.cam_from_world()
        try:
            rotation = _rotation_matrix(transform.rotation.matrix())
            translation = _vector3(transform.translation, "COLMAP pose translation")
        except AttributeError as exc:
            raise ColmapReconstructionImportError(
                "COLMAP registered image does not expose a camera-from-world rigid pose"
            ) from exc
        observation_id = image_observations[image_id]
        poses.append(
            CameraPoseEstimate(
                observation_id=observation_id,
                local_frame_id=local_frame_id,
                calibration_id=calibration_ids[image_camera_ids[image_id]],
                rotation_matrix=rotation,
                translation_xyz=translation,
                provenance=DerivedArtifactProvenance(
                    producing_run_id=run.run_id,
                    source_observation_ids=(observation_id,),
                ),
            )
        )

    points: list[Point3DEstimate] = []
    for point_id in point_ids:
        point = reconstruction.point3D(point_id)
        track: list[EstimatedTrackElement] = []
        try:
            track_elements = tuple(point.track.elements)
        except (AttributeError, TypeError) as exc:
            raise ColmapReconstructionImportError("COLMAP 3D point has no readable track") from exc
        for element in track_elements:
            image_id = int(element.image_id)
            try:
                observation_id = image_observations[image_id]
            except KeyError as exc:
                raise ColmapReconstructionImportError(
                    "COLMAP point track references an image outside registered model membership"
                ) from exc
            track.append(
                EstimatedTrackElement(
                    observation_id=observation_id,
                    feature_index=int(element.point2D_idx),
                )
            )
        if not track:
            raise ColmapReconstructionImportError("COLMAP 3D point track must not be empty")
        track_observations = tuple(
            sorted({item.observation_id for item in track}, key=lambda item: item.value)
        )
        has_error = bool(point.has_error())
        points.append(
            Point3DEstimate(
                point_id=EstimatedPoint3DId(f"point:colmap:{identity_prefix}:{point_id}"),
                local_frame_id=local_frame_id,
                position_xyz=_vector3(point.xyz, "COLMAP 3D point"),
                reprojection_error_px=(
                    _finite_float(point.error, "COLMAP reprojection error") if has_error else None
                ),
                track=tuple(track),
                provenance=DerivedArtifactProvenance(
                    producing_run_id=run.run_id,
                    source_observation_ids=track_observations,
                ),
            )
        )

    model_observations = tuple(
        sorted(image_observations.values(), key=lambda item: item.value)
    )
    estimate = SparseReconstructionEstimate(
        estimate_id=estimate_id,
        local_frame_id=local_frame_id,
        scale_status=LocalScaleStatus.UNRESOLVED,
        provenance=DerivedArtifactProvenance(
            producing_run_id=run.run_id,
            source_observation_ids=model_observations,
        ),
        camera_calibrations=tuple(calibrations),
        camera_poses=tuple(poses),
        points3d=tuple(points),
    )
    return ImportedColmapSparseModel(
        source_model_index=model_artifact.model_index,
        source_model_identity_sha256=source_identity,
        estimate=estimate,
    )


def import_colmap_reconstruction(
    request: ColmapReconstructionImportRequest,
    *,
    module: object | None = None,
) -> ColmapReconstructionImportResult:
    """Import audited solver-native sparse models into WRE estimated geometry."""

    provenance = DerivedArtifactProvenance(
        producing_run_id=request.run.run_id,
        source_observation_ids=request.run.input_observation_ids,
    )
    if not request.reconstruction.models:
        return ColmapReconstructionImportResult(
            provenance=provenance,
            configuration_sha256=request.config.sha256,
            source_reconstruction_configuration_sha256=(
                request.reconstruction.configuration_sha256
            ),
            models=(),
        )

    pycolmap = cast(Any, module) if module is not None else _load_pycolmap()
    environment = inspect_colmap_environment(pycolmap)
    if environment != request.reconstruction.environment:
        raise ColmapReconstructionImportError(
            "L3.6 must read native models with the exact L3.5 PyCOLMAP environment"
        )

    name_map = _feature_name_map(request.features)
    imported: list[ImportedColmapSparseModel] = []
    for model_artifact in request.reconstruction.models:
        model_path = _verify_native_model(request.reconstruction, model_artifact)
        imported.append(
            _import_model(
                run=request.run,
                model_artifact=model_artifact,
                model_path=model_path,
                name_map=name_map,
                pycolmap=pycolmap,
            )
        )

    imported.sort(key=lambda item: item.source_model_index)
    return ColmapReconstructionImportResult(
        provenance=provenance,
        configuration_sha256=request.config.sha256,
        source_reconstruction_configuration_sha256=request.reconstruction.configuration_sha256,
        models=tuple(imported),
    )
