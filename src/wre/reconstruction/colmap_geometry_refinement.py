from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from wre.domain.adapter_capabilities import AdapterCapabilityDescriptor, AdapterCapabilityName
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.geometry_solutions import GeometryScaleStatus
from wre.domain.observations import Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.reconstruction.colmap_adapter import COLMAP_PRECISION_OUTPUT_KINDS
from wre.reconstruction.colmap_canonical_geometry import (
    CanonicalColmapSparseModel,
    _verified_native_model_path,
    canonicalize_colmap_sparse_model,
    colmap_sparse_model_content_identity,
)
from wre.reconstruction.colmap_environment import (
    SUPPORTED_PYCOLMAP_VERSION,
    ColmapEnvironmentError,
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_features import ColmapFeatureExtractionResult
from wre.reconstruction.colmap_reconstruction import (
    ColmapSparseModelArtifact,
    _audit_model_files,
)
from wre.reconstruction.geometry_refinement import (
    GeometryRefinementAdapter,
    GeometryRefinementRequest,
    GeometryRefinementResult,
    build_geometry_refinement_result,
)
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_ADAPTER_ID = "colmap.bundle_adjustment_refinement"
COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_CAPABILITY_NAME = AdapterCapabilityName(
    "geometry.refinement.classical_bundle_adjustment"
)
COLMAP_NATIVE_SPARSE_MODEL_KIND = ArtifactKind("geometry.colmap_native_sparse_model")
COLMAP_GEOMETRY_SOLUTION_KIND = ArtifactKind("geometry.solution")
COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_CAPABILITY = AdapterCapabilityDescriptor(
    capability=COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_CAPABILITY_NAME,
    input_kinds=frozenset({COLMAP_NATIVE_SPARSE_MODEL_KIND, COLMAP_GEOMETRY_SOLUTION_KIND}),
    output_kinds=COLMAP_PRECISION_OUTPUT_KINDS,
)
COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_PRODUCER_IMPLEMENTATION = "pycolmap.bundle_adjustment"
COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_PRODUCER_VERSION = SUPPORTED_PYCOLMAP_VERSION
COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_DEPENDENCY_REF = "colmap"
COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_MODEL = None
COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_CHECKPOINT = None
COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_ARTIFACT_KEY_HARDWARE_POLICY = "required"
COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_SHIPPING_STATUS = "experimental"
COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_REPRODUCIBILITY_NOTES = (
    "V2L14.3 refines exactly one audited solver-native COLMAP sparse model through "
    "PyCOLMAP 4.2.0 bundle_adjustment using the Ceres backend on CPU with one solver "
    "thread, on a private copy only, then audits and normalizes the fresh result through "
    "the accepted V2L12 canonical COLMAP path. It does not synthesize solver state, "
    "resolve metric scale, align unrelated local frames, score candidates, or promote "
    "a default route."
)


class ColmapGeometryRefinementError(RuntimeError):
    """Raised when the classical COLMAP refinement path cannot remain auditable."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapBundleAdjustmentRefinementConfig:
    """Exact Ceres CPU bundle-adjustment baseline for V2L14.3."""

    schema_version: int = 1
    refine_focal_length: bool = True
    refine_principal_point: bool = False
    refine_extra_params: bool = True
    refine_rig_from_world: bool = True
    refine_sensor_from_rig: bool = True
    refine_points3D: bool = True
    min_track_length: int = 3
    print_summary: bool = False
    backend: str = "CERES"
    use_gpu: bool = False
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("COLMAP refinement config schema_version must be 1")
        if self.backend != "CERES":
            raise ValueError("V2L14.3 classical baseline requires the CERES backend")
        if self.use_gpu:
            raise ValueError("V2L14.3 classical baseline requires CPU Ceres bundle adjustment")
        if self.num_threads != 1:
            raise ValueError("V2L14.3 classical baseline requires one Ceres solver thread")
        if (
            isinstance(self.min_track_length, bool)
            or not isinstance(self.min_track_length, int)
            or self.min_track_length <= 0
        ):
            raise ValueError("min_track_length must be a positive integer")

        exact_flags = {
            "refine_focal_length": True,
            "refine_principal_point": False,
            "refine_extra_params": True,
            "refine_rig_from_world": True,
            "refine_sensor_from_rig": True,
            "refine_points3D": True,
            "print_summary": False,
        }
        for name, expected in exact_flags.items():
            if getattr(self, name) is not expected:
                raise ValueError(
                    f"V2L14.3 classical baseline requires {name}={expected!r}"
                )

    def canonical_document(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "min_track_length": self.min_track_length,
            "num_threads": self.num_threads,
            "print_summary": self.print_summary,
            "refine_extra_params": self.refine_extra_params,
            "refine_focal_length": self.refine_focal_length,
            "refine_points3D": self.refine_points3D,
            "refine_principal_point": self.refine_principal_point,
            "refine_rig_from_world": self.refine_rig_from_world,
            "refine_sensor_from_rig": self.refine_sensor_from_rig,
            "schema_version": self.schema_version,
            "use_gpu": self.use_gpu,
        }

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


def colmap_native_sparse_model_artifact_ref(
    model_artifact: ColmapSparseModelArtifact,
) -> ArtifactRef:
    """Return the exact supporting-evidence identity of one audited native sparse model."""

    identity = colmap_sparse_model_content_identity(model_artifact)
    return ArtifactRef(
        artifact_id=ArtifactId(f"artifact:colmap-native:{identity.value}"),
        artifact_kind=COLMAP_NATIVE_SPARSE_MODEL_KIND,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapGeometryRefinementSource:
    """Solver-native state explicitly bound to one canonical COLMAP hypothesis."""

    model_root: Path
    model_artifact: ColmapSparseModelArtifact
    features: ColmapFeatureExtractionResult
    expected_environment: ColmapEnvironmentIdentity
    artifact_ref: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.model_root, Path):
            raise TypeError("colmap_refinement_source.model_root must be pathlib.Path")
        if not isinstance(self.model_artifact, ColmapSparseModelArtifact):
            raise TypeError(
                "colmap_refinement_source.model_artifact must be ColmapSparseModelArtifact"
            )
        if not isinstance(self.features, ColmapFeatureExtractionResult):
            raise TypeError(
                "colmap_refinement_source.features must be ColmapFeatureExtractionResult"
            )
        if not isinstance(self.expected_environment, ColmapEnvironmentIdentity):
            raise TypeError(
                "colmap_refinement_source.expected_environment must be "
                "ColmapEnvironmentIdentity"
            )
        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("colmap_refinement_source.artifact_ref must be ArtifactRef")
        if self.features.environment != self.expected_environment:
            raise ValueError(
                "COLMAP refinement feature mapping and source model environment must match"
            )
        expected_ref = colmap_native_sparse_model_artifact_ref(self.model_artifact)
        if self.artifact_ref != expected_ref:
            raise ValueError(
                "COLMAP refinement source ArtifactRef must identify the exact audited "
                "native sparse model"
            )


def _load_pycolmap() -> Any:
    try:
        return cast(Any, importlib.import_module("pycolmap"))
    except (ImportError, OSError, RuntimeError) as exc:
        raise ColmapEnvironmentError(
            "PyCOLMAP is unavailable; install the approved external pycolmap==4.2.0 environment"
        ) from exc


def _configure_bundle_adjustment(
    pycolmap: Any,
    config: ColmapBundleAdjustmentRefinementConfig,
) -> Any:
    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = config.refine_focal_length
    options.refine_principal_point = config.refine_principal_point
    options.refine_extra_params = config.refine_extra_params
    options.refine_rig_from_world = config.refine_rig_from_world
    options.refine_sensor_from_rig = config.refine_sensor_from_rig
    options.refine_points3D = config.refine_points3D
    options.min_track_length = config.min_track_length
    options.print_summary = config.print_summary
    options.backend = pycolmap.BundleAdjustmentBackend.CERES
    options.ceres.use_gpu = config.use_gpu
    options.ceres.solver_options.num_threads = config.num_threads
    return options


def _validate_request_source_evidence(
    request: GeometryRefinementRequest,
    source: ColmapGeometryRefinementSource,
) -> None:
    same_id = tuple(
        item
        for item in request.supporting_artifacts
        if item.artifact_id == source.artifact_ref.artifact_id
    )
    if same_id != (source.artifact_ref,):
        raise ColmapGeometryRefinementError(
            "geometry refinement request must contain exactly the bound native COLMAP "
            "source ArtifactRef"
        )


def _validate_initialization_matches_source(
    initialization: GeometrySolutionCandidate,
    canonical_source: CanonicalColmapSparseModel,
) -> None:
    if initialization.depth_fields:
        raise ColmapGeometryRefinementError(
            "classical COLMAP bundle adjustment cannot reconstruct solver state from "
            "depth-bearing generic geometry"
        )
    if initialization.geometry_solution != canonical_source.geometry_solution:
        raise ColmapGeometryRefinementError(
            "refinement initialization GeometrySolution does not match the audited "
            "native COLMAP model"
        )
    if initialization.camera_solutions != canonical_source.camera_solutions:
        raise ColmapGeometryRefinementError(
            "refinement initialization cameras do not match the audited native COLMAP model"
        )
    if initialization.point_maps != (canonical_source.point_map,):
        raise ColmapGeometryRefinementError(
            "refinement initialization PointMap does not match the audited native COLMAP model"
        )


def _canonical_source_artifacts(
    initialization: GeometrySolutionCandidate,
    request: GeometryRefinementRequest,
) -> tuple[ArtifactRef, ...]:
    references = (*initialization.source_artifacts, *request.supporting_artifacts)
    by_identity = {
        (item.artifact_id.value, item.artifact_kind.value): item for item in references
    }
    return tuple(by_identity[key] for key in sorted(by_identity))


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ColmapGeometryRefinementError(f"{context} must be a non-negative integer")
    return value


def _audit_refined_model(
    reconstruction: Any,
    output_root: Path,
    model_index: int,
) -> ColmapSparseModelArtifact:
    model_path = output_root / str(model_index)
    files = _audit_model_files(model_path)
    return ColmapSparseModelArtifact(
        model_index=model_index,
        relative_path=str(model_index),
        num_registered_images=_nonnegative_int(
            reconstruction.num_reg_images(),
            "refined COLMAP registered-image count",
        ),
        num_points3d=_nonnegative_int(
            reconstruction.num_points3D(),
            "refined COLMAP point count",
        ),
        files=files,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapBundleAdjustmentRefinementAdapter(GeometryRefinementAdapter):
    """Concrete exact PyCOLMAP 4.2.0 classical refinement baseline."""

    source: ColmapGeometryRefinementSource
    output_root: Path
    config: ColmapBundleAdjustmentRefinementConfig = field(
        default_factory=ColmapBundleAdjustmentRefinementConfig
    )
    module: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, ColmapGeometryRefinementSource):
            raise TypeError("colmap_refinement.source must be ColmapGeometryRefinementSource")
        if not isinstance(self.output_root, Path):
            raise TypeError("colmap_refinement.output_root must be pathlib.Path")
        if not isinstance(self.config, ColmapBundleAdjustmentRefinementConfig):
            raise TypeError(
                "colmap_refinement.config must be ColmapBundleAdjustmentRefinementConfig"
            )

    def refine(self, request: GeometryRefinementRequest) -> GeometryRefinementResult:
        if not isinstance(request, GeometryRefinementRequest):
            raise TypeError("request must be GeometryRefinementRequest")

        _validate_request_source_evidence(request, self.source)
        source_model_path = _verified_native_model_path(
            self.source.model_root,
            self.source.model_artifact,
        )

        pycolmap = cast(Any, self.module) if self.module is not None else _load_pycolmap()
        environment = inspect_colmap_environment(pycolmap)
        if environment != self.source.expected_environment:
            raise ColmapGeometryRefinementError(
                "active PyCOLMAP environment does not match the bound source environment"
            )

        canonical_source = canonicalize_colmap_sparse_model(
            output_path=self.source.model_root,
            model_artifact=self.source.model_artifact,
            features=self.source.features,
            expected_environment=self.source.expected_environment,
            module=pycolmap,
        )
        _validate_initialization_matches_source(request.initialization, canonical_source)

        source_root = self.source.model_root.expanduser().resolve(strict=True)
        output_root = self.output_root.expanduser().resolve()
        if (
            output_root == source_root
            or output_root.is_relative_to(source_root)
            or source_root.is_relative_to(output_root)
        ):
            raise ValueError("refinement output_root must not overlap the retained source root")
        if output_root.exists():
            raise ValueError("refinement output_root must not already exist")
        output_root.parent.mkdir(parents=True, exist_ok=True)

        owns_output = False
        try:
            output_root.mkdir()
            owns_output = True
            with tempfile.TemporaryDirectory(
                prefix="wre-colmap-ba-",
                dir=output_root.parent,
            ) as working_dir_name:
                working_root = Path(working_dir_name)
                working_model_path = working_root / self.source.model_artifact.relative_path
                shutil.copytree(source_model_path, working_model_path)
                _verified_native_model_path(
                    working_root,
                    self.source.model_artifact,
                )

                reconstruction = pycolmap.Reconstruction(working_model_path)
                if not bool(reconstruction.is_valid()):
                    raise ColmapGeometryRefinementError(
                        "PyCOLMAP loaded an invalid private refinement reconstruction"
                    )

                options = _configure_bundle_adjustment(pycolmap, self.config)
                pycolmap.bundle_adjustment(reconstruction, options=options)

                published_model_path = (
                    output_root / self.source.model_artifact.relative_path
                )
                published_model_path.mkdir()
                reconstruction.write(published_model_path)
                refined_artifact = _audit_refined_model(
                    reconstruction,
                    output_root,
                    self.source.model_artifact.model_index,
                )

            refined_canonical = canonicalize_colmap_sparse_model(
                output_path=output_root,
                model_artifact=refined_artifact,
                features=self.source.features,
                expected_environment=self.source.expected_environment,
                module=pycolmap,
            )
            refined_candidate = GeometrySolutionCandidate(
                geometry_solution=refined_canonical.geometry_solution,
                camera_solutions=refined_canonical.camera_solutions,
                depth_fields=(),
                point_maps=(refined_canonical.point_map,),
                producer=ArtifactProducerIdentity(
                    producer=ProducerRef(
                        implementation=(
                            COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_PRODUCER_IMPLEMENTATION
                        ),
                        version=COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_PRODUCER_VERSION,
                        revision=environment.colmap_build,
                    ),
                    configuration=ConfigurationIdentity(sha256=self.config.sha256),
                ),
                source_artifacts=_canonical_source_artifacts(
                    request.initialization,
                    request,
                ),
            )
            if (
                refined_candidate.geometry_solution.scale_status
                is not GeometryScaleStatus.UNRESOLVED
            ):
                raise ColmapGeometryRefinementError(
                    "unanchored bundle adjustment must retain unresolved local scale"
                )
            if (
                refined_candidate.geometry_solution_id
                == request.initialization.geometry_solution_id
            ):
                raise ColmapGeometryRefinementError(
                    "bundle adjustment produced no distinct audited geometry hypothesis"
                )

            result = build_geometry_refinement_result(request, refined_candidate)
            _verified_native_model_path(
                self.source.model_root,
                self.source.model_artifact,
            )
        except Exception:
            if owns_output:
                shutil.rmtree(output_root, ignore_errors=True)
            _verified_native_model_path(
                self.source.model_root,
                self.source.model_artifact,
            )
            raise

        owns_output = False
        return result
