from __future__ import annotations

import hashlib
import inspect
import os
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

import wre.reconstruction.colmap_geometry_refinement as refinement_module
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.depth_fields import DepthField, DepthFieldId, DepthValueConventionName
from wre.domain.geometry_solutions import GeometryScaleStatus, GeometrySolution, GeometrySolutionId
from wre.domain.metrics import MetricVector
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import DerivedArtifactProvenance, ProducerRef, ReconstructionRunId
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_canonical_geometry import canonicalize_colmap_sparse_model
from wre.reconstruction.colmap_environment import (
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_features import (
    ColmapFeatureExtractionResult,
    ColmapImageFeatureSummary,
)
from wre.reconstruction.colmap_geometry_refinement import (
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_ADAPTER_ID,
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_ARTIFACT_KEY_HARDWARE_POLICY,
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_CAPABILITY,
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_CAPABILITY_NAME,
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_CHECKPOINT,
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_DEPENDENCY_REF,
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_MODEL,
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_PRODUCER_IMPLEMENTATION,
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_PRODUCER_VERSION,
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_REPRODUCIBILITY_NOTES,
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_SHIPPING_STATUS,
    COLMAP_NATIVE_SPARSE_MODEL_KIND,
    ColmapBundleAdjustmentRefinementAdapter,
    ColmapBundleAdjustmentRefinementConfig,
    ColmapGeometryRefinementError,
    ColmapGeometryRefinementSource,
    colmap_native_sparse_model_artifact_ref,
)
from wre.reconstruction.colmap_reconstruction import (
    ColmapModelFileArtifact,
    ColmapSparseModelArtifact,
)
from wre.reconstruction.geometry_refinement import GeometryRefinementRequest
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _ROOT / "registry" / "adapter-models.yaml"

_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


class _FakeCamera:
    model_name = "PINHOLE"
    width = 640
    height = 480
    params = (500.0, 510.0, 320.0, 240.0)


class _FakeRotation:
    @staticmethod
    def matrix() -> tuple[tuple[float, float, float], ...]:
        return _ROTATION


class _FakeTransform:
    def __init__(self, translation: tuple[float, float, float]) -> None:
        self.rotation = _FakeRotation()
        self.translation = translation


class _FakeImage:
    def __init__(self, name: str, translation: tuple[float, float, float]) -> None:
        self.name = name
        self.camera_id = 1
        self._translation = translation

    def cam_from_world(self) -> _FakeTransform:
        return _FakeTransform(self._translation)


class _FakePoint:
    def __init__(self, xyz: tuple[float, float, float]) -> None:
        self.xyz = xyz
        self.error = 0.1
        self.track = SimpleNamespace(elements=())


class _FakeReconstruction:
    def __init__(self, path: Path, *, force_refined: bool | None = None) -> None:
        self.path = Path(path)
        marker = (self.path / "points3D.bin").read_bytes()
        self.refined = b"refined" in marker if force_refined is None else force_refined

    @staticmethod
    def is_valid() -> bool:
        return True

    @staticmethod
    def reg_image_ids() -> list[int]:
        return [1, 2]

    @staticmethod
    def point3D_ids() -> list[int]:
        return [1, 2]

    @staticmethod
    def camera(camera_id: int) -> _FakeCamera:
        assert camera_id == 1
        return _FakeCamera()

    def image(self, image_id: int) -> _FakeImage:
        names = {1: "000000-a.png", 2: "000001-b.png"}
        translation = (
            (0.0, 0.0, 0.0)
            if image_id == 1
            else ((-1.1, 0.0, 0.0) if self.refined else (-1.0, 0.0, 0.0))
        )
        return _FakeImage(names[image_id], translation)

    def point3D(self, point_id: int) -> _FakePoint:
        if self.refined:
            positions = {1: (1.1, 2.0, 3.0), 2: (4.0, 5.1, 6.0)}
        else:
            positions = {1: (1.0, 2.0, 3.0), 2: (4.0, 5.0, 6.0)}
        return _FakePoint(positions[point_id])

    @staticmethod
    def num_reg_images() -> int:
        return 2

    @staticmethod
    def num_points3D() -> int:
        return 2

    def write(self, output_path: Path) -> None:
        path = Path(output_path)
        marker = b"refined" if self.refined else b"source"
        (path / "cameras.bin").write_bytes(b"camera-" + marker)
        (path / "images.bin").write_bytes(b"images-" + marker)
        (path / "points3D.bin").write_bytes(b"points-" + marker)


class _FakeSolverOptions:
    def __init__(self) -> None:
        self.num_threads = -1


class _FakeCeresOptions:
    def __init__(self) -> None:
        self.use_gpu = True
        self.solver_options = _FakeSolverOptions()


class _FakeBundleAdjustmentOptions:
    def __init__(self) -> None:
        self.refine_focal_length = False
        self.refine_principal_point = True
        self.refine_extra_params = False
        self.refine_rig_from_world = False
        self.refine_sensor_from_rig = False
        self.refine_points3D = False
        self.min_track_length = 0
        self.print_summary = True
        self.backend: object = None
        self.ceres = _FakeCeresOptions()


class _FakeBundleAdjustmentBackend:
    CERES = "CERES"
    CASPAR = "CASPAR"


class _FakePycolmap:
    __version__ = "4.2.0"
    COLMAP_version = "COLMAP 4.2.0"
    COLMAP_build = "Commit fake-v2l14-3 without GPU support"
    __ceres_version__ = "2.2.0"
    has_cuda = False
    BundleAdjustmentOptions = _FakeBundleAdjustmentOptions
    BundleAdjustmentBackend = _FakeBundleAdjustmentBackend

    def __init__(
        self,
        *,
        make_change: bool = True,
        fail_bundle_adjustment: bool = False,
        build: str | None = None,
    ) -> None:
        if build is not None:
            self.COLMAP_build = build
        self.make_change = make_change
        self.fail_bundle_adjustment = fail_bundle_adjustment
        self.reconstruction_reads: list[Path] = []
        self.bundle_adjustment_calls: list[
            tuple[_FakeReconstruction, _FakeBundleAdjustmentOptions]
        ] = []

    def Reconstruction(self, path: Path) -> _FakeReconstruction:
        self.reconstruction_reads.append(Path(path))
        return _FakeReconstruction(Path(path))

    def bundle_adjustment(
        self,
        reconstruction: _FakeReconstruction,
        *,
        options: _FakeBundleAdjustmentOptions,
    ) -> None:
        self.bundle_adjustment_calls.append((reconstruction, options))
        if self.fail_bundle_adjustment:
            raise RuntimeError("synthetic bundle adjustment failure")
        reconstruction.refined = self.make_change


def _environment() -> ColmapEnvironmentIdentity:
    return ColmapEnvironmentIdentity(
        pycolmap_version="4.2.0",
        colmap_version="COLMAP 4.2.0",
        colmap_build=_FakePycolmap.COLMAP_build,
        ceres_version="2.2.0",
        upstream_has_cuda=False,
    )


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.colmap.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _write_native_model(root: Path) -> ColmapSparseModelArtifact:
    model_path = root / "0"
    model_path.mkdir(parents=True)
    for name, content in (
        ("cameras.bin", b"camera-source"),
        ("images.bin", b"images-source"),
        ("points3D.bin", b"points-source"),
    ):
        (model_path / name).write_bytes(content)

    files = tuple(
        ColmapModelFileArtifact(
            relative_path=path.name,
            sha256=hash_file_content(path).sha256,
            byte_length=hash_file_content(path).byte_length,
        )
        for path in sorted(model_path.iterdir(), key=lambda item: item.name)
    )
    return ColmapSparseModelArtifact(
        model_index=0,
        relative_path="0",
        num_registered_images=2,
        num_points3d=2,
        files=files,
    )


def _features(
    tmp_path: Path,
    *,
    environment: ColmapEnvironmentIdentity | None = None,
) -> ColmapFeatureExtractionResult:
    database_path = tmp_path / "features.db"
    database_path.write_bytes(b"feature-evidence")
    digest = hash_file_content(database_path)
    observation_ids = (ObservationId("obs:a"), ObservationId("obs:b"))
    return ColmapFeatureExtractionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:features"),
            source_observation_ids=observation_ids,
        ),
        environment=environment or _environment(),
        configuration_sha256=Sha256Digest("1" * 64),
        database_path=database_path,
        database_sha256=digest.sha256,
        database_byte_length=digest.byte_length,
        images=(
            ColmapImageFeatureSummary(
                observation_id=observation_ids[0],
                image_name="000000-a.png",
                keypoint_rows=1,
                keypoint_cols=4,
                descriptor_rows=1,
                descriptor_cols=128,
            ),
            ColmapImageFeatureSummary(
                observation_id=observation_ids[1],
                image_name="000001-b.png",
                keypoint_rows=1,
                keypoint_cols=4,
                descriptor_rows=1,
                descriptor_cols=128,
            ),
        ),
    )


def _source_fixture(
    tmp_path: Path,
    *,
    module: _FakePycolmap | None = None,
) -> tuple[
    _FakePycolmap,
    ColmapGeometryRefinementSource,
    GeometrySolutionCandidate,
]:
    pycolmap = module or _FakePycolmap()
    source_root = tmp_path / "source"
    model_artifact = _write_native_model(source_root)
    features = _features(tmp_path)
    canonical = canonicalize_colmap_sparse_model(
        output_path=source_root,
        model_artifact=model_artifact,
        features=features,
        expected_environment=_environment(),
        module=pycolmap,
    )
    candidate = GeometrySolutionCandidate(
        geometry_solution=canonical.geometry_solution,
        camera_solutions=canonical.camera_solutions,
        depth_fields=(),
        point_maps=(canonical.point_map,),
        producer=_producer("initial"),
        source_artifacts=(_artifact("artifact:initial", "geometry.solution"),),
    )
    source = ColmapGeometryRefinementSource(
        model_root=source_root,
        model_artifact=model_artifact,
        features=features,
        expected_environment=_environment(),
        artifact_ref=colmap_native_sparse_model_artifact_ref(model_artifact),
    )
    return pycolmap, source, candidate


def _request(
    initialization: GeometrySolutionCandidate,
    source: ColmapGeometryRefinementSource,
) -> GeometryRefinementRequest:
    return GeometryRefinementRequest(
        initialization=initialization,
        supporting_artifacts=(source.artifact_ref,),
    )


def _native_bytes(source: ColmapGeometryRefinementSource) -> dict[str, bytes]:
    model_path = source.model_root / source.model_artifact.relative_path
    return {
        item.relative_path: (model_path / item.relative_path).read_bytes()
        for item in source.model_artifact.files
    }


def _entries_by_id() -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    entries = document["entries"]
    assert isinstance(entries, list)
    return {
        cast(str, entry["adapter_id"]): cast(dict[str, Any], entry)
        for entry in entries
        if isinstance(entry, dict)
    }


def test_config_is_exact_frozen_deterministic_ceres_cpu_baseline() -> None:
    config = ColmapBundleAdjustmentRefinementConfig()

    assert tuple(field.name for field in fields(config)) == (
        "schema_version",
        "refine_focal_length",
        "refine_principal_point",
        "refine_extra_params",
        "refine_rig_from_world",
        "refine_sensor_from_rig",
        "refine_points3D",
        "min_track_length",
        "print_summary",
        "backend",
        "use_gpu",
        "num_threads",
    )
    assert config.sha256 == ColmapBundleAdjustmentRefinementConfig().sha256
    assert config.backend == "CERES"
    assert config.use_gpu is False
    assert config.num_threads == 1
    assert config.refine_points3D is True

    with pytest.raises(FrozenInstanceError):
        config.use_gpu = True  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"backend": "CASPAR"}, "CERES"),
        ({"use_gpu": True}, "CPU Ceres"),
        ({"num_threads": 2}, "one Ceres solver thread"),
        ({"min_track_length": 0}, "positive integer"),
        ({"refine_focal_length": False}, "refine_focal_length"),
        ({"refine_principal_point": True}, "refine_principal_point"),
        ({"refine_extra_params": False}, "refine_extra_params"),
        ({"refine_rig_from_world": False}, "refine_rig_from_world"),
        ({"refine_sensor_from_rig": False}, "refine_sensor_from_rig"),
        ({"refine_points3D": False}, "refine_points3D"),
        ({"print_summary": True}, "print_summary"),
    ],
)
def test_config_rejects_unapproved_solver_modes(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ColmapBundleAdjustmentRefinementConfig(**kwargs)


def test_source_binding_requires_exact_environment_and_native_artifact_ref(
    tmp_path: Path,
) -> None:
    _, source, _ = _source_fixture(tmp_path)

    assert tuple(field.name for field in fields(source)) == (
        "model_root",
        "model_artifact",
        "features",
        "expected_environment",
        "artifact_ref",
    )
    assert source.artifact_ref == colmap_native_sparse_model_artifact_ref(
        source.model_artifact
    )
    assert source.artifact_ref.artifact_kind == COLMAP_NATIVE_SPARSE_MODEL_KIND

    with pytest.raises(ValueError, match="exact audited native"):
        replace(
            source,
            artifact_ref=_artifact(
                source.artifact_ref.artifact_id.value,
                "geometry.other",
            ),
        )
    with pytest.raises(ValueError, match="environment"):
        replace(
            source,
            expected_environment=replace(
                source.expected_environment,
                colmap_build="different",
            ),
        )


def test_request_must_name_bound_native_source_before_solver_execution(
    tmp_path: Path,
) -> None:
    module, source, initialization = _source_fixture(tmp_path)
    adapter = ColmapBundleAdjustmentRefinementAdapter(
        source=source,
        output_root=tmp_path / "refined",
        module=module,
    )

    with pytest.raises(ColmapGeometryRefinementError, match="bound native COLMAP"):
        adapter.refine(
            GeometryRefinementRequest(
                initialization=initialization,
                supporting_artifacts=(),
            )
        )
    assert module.bundle_adjustment_calls == []

    conflicting = ArtifactRef(
        artifact_id=source.artifact_ref.artifact_id,
        artifact_kind=ArtifactKind("geometry.other"),
    )
    with pytest.raises(ColmapGeometryRefinementError, match="bound native COLMAP"):
        adapter.refine(
            GeometryRefinementRequest(
                initialization=initialization,
                supporting_artifacts=(conflicting,),
            )
        )
    assert module.bundle_adjustment_calls == []


def test_source_tamper_and_environment_mismatch_fail_before_bundle_adjustment(
    tmp_path: Path,
) -> None:
    module, source, initialization = _source_fixture(tmp_path)
    request = _request(initialization, source)
    (source.model_root / "0" / "images.bin").write_bytes(b"tampered")

    with pytest.raises(Exception, match="changed after mapper publication"):
        ColmapBundleAdjustmentRefinementAdapter(
            source=source,
            output_root=tmp_path / "refined-tamper",
            module=module,
        ).refine(request)
    assert module.bundle_adjustment_calls == []

    clean_root = tmp_path / "clean"
    mismatch_module = _FakePycolmap(build="different-build")
    _, clean_source, clean_candidate = _source_fixture(
        clean_root,
        module=_FakePycolmap(),
    )
    with pytest.raises(ColmapGeometryRefinementError, match="environment"):
        ColmapBundleAdjustmentRefinementAdapter(
            source=clean_source,
            output_root=tmp_path / "refined-env",
            module=mismatch_module,
        ).refine(_request(clean_candidate, clean_source))
    assert mismatch_module.bundle_adjustment_calls == []


def test_source_symlink_fails_closed_before_bundle_adjustment(tmp_path: Path) -> None:
    module, source, initialization = _source_fixture(tmp_path)
    link = source.model_root / "0" / "unexpected.bin"
    link.symlink_to(source.model_root / "0" / "images.bin")

    with pytest.raises(Exception, match="symlink"):
        ColmapBundleAdjustmentRefinementAdapter(
            source=source,
            output_root=tmp_path / "refined",
            module=module,
        ).refine(_request(initialization, source))
    assert module.bundle_adjustment_calls == []


def test_initialization_must_exactly_match_bound_canonical_source(
    tmp_path: Path,
) -> None:
    module, source, initialization = _source_fixture(tmp_path)
    foreign_geometry = replace(
        initialization.geometry_solution,
        geometry_solution_id=GeometrySolutionId("geometry:foreign"),
    )
    foreign = replace(initialization, geometry_solution=foreign_geometry)

    with pytest.raises(ColmapGeometryRefinementError, match="GeometrySolution"):
        ColmapBundleAdjustmentRefinementAdapter(
            source=source,
            output_root=tmp_path / "refined",
            module=module,
        ).refine(_request(foreign, source))
    assert module.bundle_adjustment_calls == []


def test_depth_bearing_generic_candidate_cannot_reconstruct_native_ba_state(
    tmp_path: Path,
) -> None:
    module, source, initialization = _source_fixture(tmp_path)
    camera = initialization.camera_solutions[0]
    pixel_count = camera.dimensions.width_px * camera.dimensions.height_px
    depth = DepthField(
        depth_field_id=DepthFieldId("depth:generic"),
        observation_id=camera.observation_id,
        camera_solution_id=camera.solution_id,
        dimensions=camera.dimensions,
        depth_value_convention=DepthValueConventionName("relative-depth"),
        depth_values=tuple(1.0 for _ in range(pixel_count)),
        validity=tuple(True for _ in range(pixel_count)),
        confidence=None,
        metrics=MetricVector(observations=()),
    )
    generic_geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId("geometry:generic-depth"),
        local_frame_id=initialization.geometry_solution.local_frame_id,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_ids=tuple(item.solution_id for item in initialization.camera_solutions),
        depth_field_ids=(depth.depth_field_id,),
        point_map_ids=(),
        metrics=MetricVector(observations=()),
    )
    generic = GeometrySolutionCandidate(
        geometry_solution=generic_geometry,
        camera_solutions=initialization.camera_solutions,
        depth_fields=(depth,),
        point_maps=(),
        producer=_producer("generic"),
        source_artifacts=(),
    )

    with pytest.raises(ColmapGeometryRefinementError, match="depth-bearing"):
        ColmapBundleAdjustmentRefinementAdapter(
            source=source,
            output_root=tmp_path / "refined",
            module=module,
        ).refine(_request(generic, source))
    assert module.bundle_adjustment_calls == []


def test_success_uses_private_copy_one_ceres_cpu_ba_and_fresh_canonical_output(
    tmp_path: Path,
) -> None:
    module, source, initialization = _source_fixture(tmp_path)
    before = _native_bytes(source)
    request = _request(initialization, source)
    adapter = ColmapBundleAdjustmentRefinementAdapter(
        source=source,
        output_root=tmp_path / "refined",
        module=module,
    )

    result = adapter.refine(request)

    assert result.request is request
    assert result.request.initialization is initialization
    assert result.refined_candidate.geometry_solution_id != initialization.geometry_solution_id
    assert result.refined_candidate.depth_fields == ()
    assert len(result.refined_candidate.point_maps) == 1
    assert (
        result.refined_candidate.geometry_solution.scale_status
        is GeometryScaleStatus.UNRESOLVED
    )
    assert result.refined_candidate.producer.producer == ProducerRef(
        implementation=COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_PRODUCER_IMPLEMENTATION,
        version=COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_PRODUCER_VERSION,
        revision=_FakePycolmap.COLMAP_build,
    )
    assert (
        result.refined_candidate.producer.configuration.sha256
        == ColmapBundleAdjustmentRefinementConfig().sha256
    )
    assert source.artifact_ref in result.refined_candidate.source_artifacts
    assert initialization.source_artifacts[0] in result.refined_candidate.source_artifacts
    assert len(module.bundle_adjustment_calls) == 1

    _, options = module.bundle_adjustment_calls[0]
    assert options.backend == _FakeBundleAdjustmentBackend.CERES
    assert options.ceres.use_gpu is False
    assert options.ceres.solver_options.num_threads == 1
    assert options.refine_focal_length is True
    assert options.refine_principal_point is False
    assert options.refine_extra_params is True
    assert options.refine_rig_from_world is True
    assert options.refine_sensor_from_rig is True
    assert options.refine_points3D is True

    assert _native_bytes(source) == before
    assert (adapter.output_root / "0" / "points3D.bin").read_bytes() == b"points-refined"
    private_reads = [
        path
        for path in module.reconstruction_reads
        if path != source.model_root / source.model_artifact.relative_path
        and path != adapter.output_root / source.model_artifact.relative_path
    ]
    assert private_reads
    assert all(not path.exists() for path in private_reads)

    for attribute in (
        "transform",
        "alignment",
        "score",
        "quality",
        "winner",
        "decision",
    ):
        assert not hasattr(result, attribute)


def test_identical_ba_output_fails_instead_of_relabeling_and_cleans_output(
    tmp_path: Path,
) -> None:
    module = _FakePycolmap(make_change=False)
    module, source, initialization = _source_fixture(tmp_path, module=module)
    output_root = tmp_path / "refined"

    with pytest.raises(ColmapGeometryRefinementError, match="no distinct audited"):
        ColmapBundleAdjustmentRefinementAdapter(
            source=source,
            output_root=output_root,
            module=module,
        ).refine(_request(initialization, source))

    assert len(module.bundle_adjustment_calls) == 1
    assert not output_root.exists()


def test_solver_failure_preserves_source_and_cleans_owned_output(tmp_path: Path) -> None:
    module = _FakePycolmap(fail_bundle_adjustment=True)
    module, source, initialization = _source_fixture(tmp_path, module=module)
    before = _native_bytes(source)
    output_root = tmp_path / "refined"

    with pytest.raises(RuntimeError, match="synthetic bundle adjustment failure"):
        ColmapBundleAdjustmentRefinementAdapter(
            source=source,
            output_root=output_root,
            module=module,
        ).refine(_request(initialization, source))

    assert _native_bytes(source) == before
    assert not output_root.exists()


def test_existing_output_is_never_reused_or_deleted(tmp_path: Path) -> None:
    module, source, initialization = _source_fixture(tmp_path)
    output_root = tmp_path / "refined"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("owned elsewhere", encoding="utf-8")

    with pytest.raises(ValueError, match="must not already exist"):
        ColmapBundleAdjustmentRefinementAdapter(
            source=source,
            output_root=output_root,
            module=module,
        ).refine(_request(initialization, source))

    assert sentinel.read_text(encoding="utf-8") == "owned elsewhere"
    assert module.bundle_adjustment_calls == []


def test_adapter_exposes_only_refine_as_public_execution_operation(
    tmp_path: Path,
) -> None:
    module, source, _ = _source_fixture(tmp_path)
    adapter = ColmapBundleAdjustmentRefinementAdapter(
        source=source,
        output_root=tmp_path / "refined",
        module=module,
    )
    public_callables = {
        name
        for name, value in type(adapter).__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    assert public_callables == {"refine"}
    signature = inspect.signature(type(adapter).refine)
    assert tuple(signature.parameters) == ("self", "request")


def test_adapter_registry_contract_is_exact_experimental_colmap_420() -> None:
    entry = _entries_by_id()[COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_ADAPTER_ID]

    assert COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_CAPABILITY.capability is (
        COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_CAPABILITY_NAME
    )
    assert entry["capability"] == {
        "name": COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_CAPABILITY_NAME.value,
        "input_kinds": [
            "geometry.colmap_native_sparse_model",
            "geometry.solution",
        ],
        "output_kinds": [
            "geometry.camera_solution",
            "geometry.point_map",
            "geometry.solution",
        ],
    }
    assert entry["producer"] == {
        "implementation": COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_PRODUCER_IMPLEMENTATION,
        "version": COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_PRODUCER_VERSION,
        "revision": None,
    }
    assert entry["dependency_refs"] == [COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_DEPENDENCY_REF]
    assert entry["model"] is COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_MODEL
    assert entry["checkpoint"] is COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_CHECKPOINT
    assert (
        entry["artifact_key_hardware_policy"]
        == COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_ARTIFACT_KEY_HARDWARE_POLICY
    )
    assert entry["shipping_status"] == COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_SHIPPING_STATUS
    assert entry["license"]["review"] == "approved"
    assert entry["reproducibility_notes"] == (
        COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_REPRODUCIBILITY_NOTES
    )


def test_module_surface_has_no_learned_alignment_quality_or_later_layer_dependencies() -> None:
    forbidden = {
        "torch",
        "numpy",
        "Da3ExecutionResult",
        "QualityDecision",
        "QualityPolicy",
        "MetricDescriptor",
        "MetricObservation",
        "SE3",
        "Sim3",
        "GTSAM",
        "Open3D",
        "CASPAR",
        "Router",
        "Scheduler",
        "SurfaceModel",
        "MasterScene",
        "RuntimeScene",
    }
    assert forbidden.isdisjoint(vars(refinement_module))


@pytest.mark.skipif(
    os.environ.get("WRE_COLMAP_INTEGRATION") != "1",
    reason="real V2L14.3 refinement runs only in the dedicated COLMAP integration lane",
)
def test_real_pycolmap_bundle_adjustment_refines_audited_native_model(
    tmp_path: Path,
) -> None:
    pycolmap = pytest.importorskip("pycolmap")
    environment = inspect_colmap_environment(pycolmap)
    database_path = tmp_path / "synthetic.db"

    with pycolmap.Database.open(database_path) as database:
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs = 2
        options.num_cameras_per_rig = 1
        options.num_frames_per_rig = 6
        options.num_points3D = 80
        options.camera_width = 16
        options.camera_height = 16
        options.camera_params = [20.0, 8.0, 8.0, 0.01]
        options.camera_has_prior_focal_length = False
        reconstruction = pycolmap.synthesize_dataset(options, database)

    point_id = sorted(int(value) for value in reconstruction.point3D_ids())[0]
    point = reconstruction.point3D(point_id)
    xyz = point.xyz
    point.xyz = [
        float(xyz[0]) + 0.35,
        float(xyz[1]) - 0.20,
        float(xyz[2]) + 0.15,
    ]

    source_root = tmp_path / "source"
    source_model_path = source_root / "0"
    source_model_path.mkdir(parents=True)
    reconstruction.write(source_model_path)

    files = tuple(
        ColmapModelFileArtifact(
            relative_path=path.relative_to(source_model_path).as_posix(),
            sha256=hash_file_content(path).sha256,
            byte_length=hash_file_content(path).byte_length,
        )
        for path in sorted(source_model_path.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    )
    model_artifact = ColmapSparseModelArtifact(
        model_index=0,
        relative_path="0",
        num_registered_images=int(reconstruction.num_reg_images()),
        num_points3d=int(reconstruction.num_points3D()),
        files=files,
    )

    image_ids = sorted(int(value) for value in reconstruction.reg_image_ids())
    observations = tuple(
        ObservationId(f"obs:ba:{index:04d}") for index in range(len(image_ids))
    )
    image_names = tuple(str(reconstruction.image(image_id).name) for image_id in image_ids)
    database_digest = hash_file_content(database_path)
    features = ColmapFeatureExtractionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:ba-features"),
            source_observation_ids=observations,
        ),
        environment=environment,
        configuration_sha256=Sha256Digest("9" * 64),
        database_path=database_path,
        database_sha256=database_digest.sha256,
        database_byte_length=database_digest.byte_length,
        images=tuple(
            ColmapImageFeatureSummary(
                observation_id=observation_id,
                image_name=image_name,
                keypoint_rows=0,
                keypoint_cols=2,
                descriptor_rows=0,
                descriptor_cols=128,
            )
            for observation_id, image_name in zip(
                observations,
                image_names,
                strict=True,
            )
        ),
    )

    canonical = canonicalize_colmap_sparse_model(
        output_path=source_root,
        model_artifact=model_artifact,
        features=features,
        expected_environment=environment,
        module=pycolmap,
    )
    initialization = GeometrySolutionCandidate(
        geometry_solution=canonical.geometry_solution,
        camera_solutions=canonical.camera_solutions,
        depth_fields=(),
        point_maps=(canonical.point_map,),
        producer=_producer("real-initial"),
        source_artifacts=(),
    )
    source = ColmapGeometryRefinementSource(
        model_root=source_root,
        model_artifact=model_artifact,
        features=features,
        expected_environment=environment,
        artifact_ref=colmap_native_sparse_model_artifact_ref(model_artifact),
    )
    source_before = {
        item.relative_path: hash_file_content(
            source_root / model_artifact.relative_path / item.relative_path
        )
        for item in model_artifact.files
    }

    result = ColmapBundleAdjustmentRefinementAdapter(
        source=source,
        output_root=tmp_path / "refined",
        module=pycolmap,
    ).refine(_request(initialization, source))

    assert result.refined_candidate.geometry_solution_id != initialization.geometry_solution_id
    assert result.refined_candidate.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert result.refined_candidate.depth_fields == ()
    assert len(result.refined_candidate.point_maps) == 1
    assert result.refined_candidate.producer.producer.implementation == (
        "pycolmap.bundle_adjustment"
    )
    assert {
        item.relative_path: hash_file_content(
            source_root / model_artifact.relative_path / item.relative_path
        )
        for item in model_artifact.files
    } == source_before
