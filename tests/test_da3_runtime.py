from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any, cast

import pytest

import wre.reconstruction.da3_runtime as da3_runtime
from wre.domain import (
    ArtifactId,
    ArtifactKey,
    ArtifactKind,
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
    ArtifactProducerIdentity,
    ArtifactRef,
    ConfigurationIdentity,
    DecodedImageLevelDescriptor,
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
    DecodedImagePyramidManifest,
    DecodedImagePyramidSpec,
    GeometryScaleStatus,
    HardwareRuntimeIdentity,
    ImageDimensions,
    ObservationId,
    ObservationKind,
    ProducerRef,
    Sha256Digest,
)
from wre.ingestion.decoded_images import DecodedImagePyramidMaterializationResult
from wre.ingestion.hashing import FileContentHash, hash_file_content
from wre.reconstruction.da3_preview import Da3BaseObservationPrediction


def _decoded_input(
    tmp_path: Path,
    observation: str,
    *,
    rgb8: bytes = b"\x00\x7f\xff\xff\x7f\x00",
    artifact_key: str = "b",
) -> da3_runtime.Da3ImageInput:
    token = observation.replace(":", "-")
    root = tmp_path / token
    level_path = root / "levels" / "level-000000.rgb"
    level_path.parent.mkdir(parents=True, exist_ok=True)
    level_path.write_bytes(rgb8)
    content_hash = hash_file_content(level_path)
    artifact_ref = ArtifactRef(
        artifact_id=ArtifactId(f"artifact:{token}"),
        artifact_kind=ArtifactKind("media.decoded_image_pyramid"),
    )
    manifest = DecodedImagePyramidManifest(
        source_observation_id=ObservationId(observation),
        source_kind=ObservationKind.IMAGE,
        source_asset_sha256=Sha256Digest("a" * 64),
        pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
        orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
        spec=DecodedImagePyramidSpec(minimum_max_edge_px=2),
        levels=(
            DecodedImageLevelDescriptor(
                level_index=0,
                width_px=2,
                height_px=1,
                relative_path="levels/level-000000.rgb",
            ),
        ),
    )
    result = DecodedImagePyramidMaterializationResult(
        artifact_key=ArtifactKey(sha256=Sha256Digest(artifact_key * 64)),
        producer=ArtifactProducerIdentity(
            producer=ProducerRef(implementation="test.decoded", version="1"),
            configuration=ConfigurationIdentity(sha256=Sha256Digest("c" * 64)),
        ),
        manifest=manifest,
        materialization=ArtifactMaterializationMetadata(
            artifact_ref=artifact_ref,
            entries=(
                ArtifactMaterializationEntry(
                    relative_path="levels/level-000000.rgb",
                    sha256=content_hash.sha256,
                    byte_length=content_hash.byte_length,
                ),
            ),
        ),
    )
    return da3_runtime.Da3ImageInput(
        decoded_result=result,
        materialization_root=root,
    )


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "da3-source"
    package = root / "src" / "depth_anything_3"
    (package / "configs").mkdir(parents=True)
    (package / "configs" / "da3-base.yaml").write_text("fixture: true\n", encoding="utf-8")
    return root


def _checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"fixture-da3-checkpoint")
    return path


def _hardware(value: str = "d") -> HardwareRuntimeIdentity:
    return HardwareRuntimeIdentity(sha256=Sha256Digest(value * 64))


def _environment() -> da3_runtime.Da3EnvironmentIdentity:
    return da3_runtime.Da3EnvironmentIdentity(
        source_revision=da3_runtime.DA3_SOURCE_REVISION,
        python_version=da3_runtime.DA3_REFERENCE_PYTHON_VERSION,
        package_versions=da3_runtime.DA3_REFERENCE_PACKAGE_VERSIONS,
        device="cpu",
        precision="float32",
    )


def _prediction(observation: ObservationId, depth: float = 2.0) -> Da3BaseObservationPrediction:
    return Da3BaseObservationPrediction(
        observation_id=observation,
        dimensions=ImageDimensions(width_px=2, height_px=1),
        world_to_camera_rotation=(
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        world_to_camera_translation=(1.0, 2.0, 3.0),
        intrinsics=(
            (500.0, 0.0, 1.0),
            (0.0, 510.0, 0.5),
            (0.0, 0.0, 1.0),
        ),
        depth_values=(depth, depth + 1.0),
        validity=(True, True),
        confidence=None,
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def infer(
        self,
        *,
        source_root: Path,
        checkpoint_path: Path,
        images: tuple[object, ...],
        config: da3_runtime.Da3ReferenceConfig,
    ) -> tuple[
        da3_runtime.Da3EnvironmentIdentity,
        tuple[Da3BaseObservationPrediction, ...],
    ]:
        self.calls.append(
            {
                "source_root": source_root,
                "checkpoint_path": checkpoint_path,
                "images": images,
                "config": config,
            }
        )
        predictions = tuple(
            _prediction(cast(Any, image).observation_id, 2.0 + index)
            for index, image in enumerate(images)
        )
        return _environment(), predictions


def _allow_external_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_root: Path,
    checkpoint_path: Path,
) -> None:
    monkeypatch.setattr(da3_runtime, "_verify_source_root", lambda path: source_root.resolve())
    monkeypatch.setattr(da3_runtime, "_verify_checkpoint", lambda path: checkpoint_path.resolve())


def test_reference_config_is_exact_frozen_and_content_identified() -> None:
    config = da3_runtime.Da3ReferenceConfig()

    assert tuple(field.name for field in fields(da3_runtime.Da3ReferenceConfig)) == (
        "schema_version",
        "process_res",
        "process_res_method",
        "ref_view_strategy",
        "infer_gs",
        "use_ray_pose",
        "num_workers",
        "random_seed",
        "device",
        "precision",
    )
    assert config.canonical_document() == {
        "device": "cpu",
        "infer_gs": False,
        "num_workers": 1,
        "precision": "float32",
        "process_res": 504,
        "process_res_method": "upper_bound_resize",
        "random_seed": 0,
        "ref_view_strategy": "saddle_balanced",
        "schema_version": 1,
        "use_ray_pose": False,
    }
    assert config.configuration == da3_runtime.Da3ReferenceConfig().configuration
    with pytest.raises(FrozenInstanceError):
        config.device = "cuda"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"schema_version": True}, "schema_version"),
        ({"process_res": 518}, "process_res"),
        ({"process_res_method": "upper_bound_crop"}, "process_res_method"),
        ({"ref_view_strategy": "first"}, "ref_view_strategy"),
        ({"infer_gs": True}, "infer_gs"),
        ({"use_ray_pose": True}, "use_ray_pose"),
        ({"num_workers": 2}, "num_workers"),
        ({"random_seed": 1}, "random_seed"),
        ({"device": "cuda"}, "device"),
        ({"precision": "float16"}, "precision"),
    ],
)
def test_reference_config_rejects_other_execution_profiles(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        da3_runtime.Da3ReferenceConfig(**kwargs)


def test_exact_model_checkpoint_and_environment_identities() -> None:
    assert da3_runtime.DA3_MODEL.revision == da3_runtime.DA3_SOURCE_REVISION
    assert da3_runtime.DA3_SOURCE_PACKAGE_TREE_SHA == ("be87985cb286f0747d4fd9bfaec47f1b765457ea")
    assert da3_runtime.DA3_CHECKPOINT.identifier == "depth-anything/DA3-BASE/model.safetensors"
    assert da3_runtime.DA3_CHECKPOINT.sha256 == Sha256Digest(
        "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5"
    )
    environment = _environment()
    assert environment.python_version == da3_runtime.DA3_REFERENCE_PYTHON_VERSION
    assert environment.package_versions == da3_runtime.DA3_REFERENCE_PACKAGE_VERSIONS
    assert dict(environment.package_versions)["torch"] == "2.4.1+cpu"
    assert dict(environment.package_versions)["torchvision"] == "0.19.1+cpu"

    with pytest.raises(ValueError, match="package versions"):
        da3_runtime.Da3EnvironmentIdentity(
            source_revision=da3_runtime.DA3_SOURCE_REVISION,
            python_version=da3_runtime.DA3_REFERENCE_PYTHON_VERSION,
            package_versions=(("torch", "latest"),),
            device="cpu",
            precision="float32",
        )


def test_request_requires_canonical_unique_inputs_and_exact_identity(tmp_path: Path) -> None:
    input_a = _decoded_input(tmp_path, "obs:a")
    input_b = _decoded_input(tmp_path, "obs:b")
    source = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path)

    request = da3_runtime.Da3ExecutionRequest(
        inputs=(input_a, input_b),
        source_root=source,
        checkpoint_path=checkpoint,
        hardware_runtime=_hardware(),
    )
    assert request.model is da3_runtime.DA3_MODEL
    assert request.checkpoint is da3_runtime.DA3_CHECKPOINT

    with pytest.raises(ValueError, match="canonical"):
        da3_runtime.Da3ExecutionRequest(
            inputs=(input_b, input_a),
            source_root=source,
            checkpoint_path=checkpoint,
            hardware_runtime=_hardware(),
        )
    with pytest.raises(ValueError, match="repeat"):
        da3_runtime.Da3ExecutionRequest(
            inputs=(input_a, input_a),
            source_root=source,
            checkpoint_path=checkpoint,
            hardware_runtime=_hardware(),
        )


def test_source_root_requires_exact_clean_local_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_root(tmp_path)
    calls: list[tuple[str, ...]] = []

    def _git(root: Path, *args: str) -> str:
        calls.append(args)
        if args == ("rev-parse", "HEAD"):
            return da3_runtime.DA3_SOURCE_REVISION
        if args == ("rev-parse", "HEAD:src/depth_anything_3"):
            return da3_runtime.DA3_SOURCE_PACKAGE_TREE_SHA
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(da3_runtime, "_run_git", _git)
    assert da3_runtime._verify_source_root(source) == source.resolve()
    assert calls == [
        ("rev-parse", "HEAD"),
        ("rev-parse", "HEAD:src/depth_anything_3"),
        ("status", "--porcelain", "--untracked-files=all"),
    ]

    monkeypatch.setattr(da3_runtime, "_run_git", lambda root, *args: "0" * 40)
    with pytest.raises(da3_runtime.Da3RuntimeError, match="revision"):
        da3_runtime._verify_source_root(source)

    def _wrong_tree(root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return da3_runtime.DA3_SOURCE_REVISION
        if args == ("rev-parse", "HEAD:src/depth_anything_3"):
            return "0" * 40
        return ""

    monkeypatch.setattr(da3_runtime, "_run_git", _wrong_tree)
    with pytest.raises(da3_runtime.Da3RuntimeError, match="package tree"):
        da3_runtime._verify_source_root(source)


def test_source_root_rejects_dirty_and_uri_like_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_root(tmp_path)

    def _dirty(root: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return da3_runtime.DA3_SOURCE_REVISION
        if args == ("rev-parse", "HEAD:src/depth_anything_3"):
            return da3_runtime.DA3_SOURCE_PACKAGE_TREE_SHA
        return "?? src/depth_anything_3/shadow.py"

    monkeypatch.setattr(da3_runtime, "_run_git", _dirty)
    with pytest.raises(da3_runtime.Da3RuntimeError, match="clean"):
        da3_runtime._verify_source_root(source)
    with pytest.raises(da3_runtime.Da3RuntimeError, match="local"):
        da3_runtime._verify_source_root(Path("https://example.invalid/da3"))


def test_checkpoint_is_local_nonempty_and_exact_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    real_hash = da3_runtime.hash_file_content

    def _accepted_hash(path: Path) -> FileContentHash:
        if path.resolve() == checkpoint.resolve():
            return FileContentHash(
                sha256=da3_runtime.DA3_CHECKPOINT.sha256,
                byte_length=path.stat().st_size,
            )
        return real_hash(path)

    monkeypatch.setattr(da3_runtime, "hash_file_content", _accepted_hash)
    assert da3_runtime._verify_checkpoint(checkpoint) == checkpoint.resolve()

    monkeypatch.setattr(da3_runtime, "hash_file_content", real_hash)
    with pytest.raises(da3_runtime.Da3RuntimeError, match="SHA-256"):
        da3_runtime._verify_checkpoint(checkpoint)
    with pytest.raises(da3_runtime.Da3RuntimeError, match="local"):
        da3_runtime._verify_checkpoint(Path("https://example.invalid/model.safetensors"))


def test_verified_rgb_input_reads_only_canonical_materialization(tmp_path: Path) -> None:
    item = _decoded_input(tmp_path, "obs:a")
    verified = da3_runtime._verified_rgb_image(item)

    assert verified.observation_id == ObservationId("obs:a")
    assert verified.width_px == 2
    assert verified.height_px == 1
    assert verified.rgb8 == b"\x00\x7f\xff\xff\x7f\x00"
    assert verified.artifact_key_sha256 == Sha256Digest("b" * 64)

    level = item.materialization_root / "levels" / "level-000000.rgb"
    level.write_bytes(b"corrupt")
    with pytest.raises(da3_runtime.Da3RuntimeError, match="materialization"):
        da3_runtime._verified_rgb_image(item)


def test_environment_inspection_requires_every_exact_reference_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = dict(da3_runtime.DA3_REFERENCE_PACKAGE_VERSIONS)
    monkeypatch.setattr(
        da3_runtime.importlib.metadata,
        "version",
        lambda name: expected[name],
    )
    monkeypatch.setattr(
        da3_runtime.sys,
        "version",
        da3_runtime.DA3_REFERENCE_PYTHON_VERSION + " (fixture)",
    )
    identity = da3_runtime.inspect_da3_reference_environment()
    assert identity.package_versions == da3_runtime.DA3_REFERENCE_PACKAGE_VERSIONS

    monkeypatch.setattr(
        da3_runtime.importlib.metadata,
        "version",
        lambda name: "999" if name == "torch" else expected[name],
    )
    with pytest.raises(da3_runtime.Da3RuntimeError, match="torch"):
        da3_runtime.inspect_da3_reference_environment()


def test_execute_uses_fake_runtime_then_existing_candidate_normalizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = (
        _decoded_input(tmp_path, "obs:a", artifact_key="b"),
        _decoded_input(tmp_path, "obs:b", artifact_key="c"),
    )
    source = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    _allow_external_fixture(
        monkeypatch,
        source_root=source,
        checkpoint_path=checkpoint,
    )
    runtime = _FakeRuntime()
    request = da3_runtime.Da3ExecutionRequest(
        inputs=inputs,
        source_root=source,
        checkpoint_path=checkpoint,
        hardware_runtime=_hardware(),
    )

    result = da3_runtime.execute_da3_base_preview(request, runtime=runtime)

    assert len(runtime.calls) == 1
    assert result.model is da3_runtime.DA3_MODEL
    assert result.checkpoint is da3_runtime.DA3_CHECKPOINT
    assert result.environment == _environment()
    assert result.hardware_runtime == _hardware()
    assert result.geometry.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert result.geometry.point_maps == ()
    assert tuple(item.observation_id for item in result.geometry.camera_solutions) == (
        ObservationId("obs:a"),
        ObservationId("obs:b"),
    )
    assert all(item.confidence is None for item in result.geometry.depth_fields)


def test_reproducibility_identity_changes_with_material_input_or_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    _allow_external_fixture(
        monkeypatch,
        source_root=source,
        checkpoint_path=checkpoint,
    )
    runtime = _FakeRuntime()

    request_a = da3_runtime.Da3ExecutionRequest(
        inputs=(_decoded_input(tmp_path, "obs:a", artifact_key="b"),),
        source_root=source,
        checkpoint_path=checkpoint,
        hardware_runtime=_hardware("d"),
    )
    request_b = da3_runtime.Da3ExecutionRequest(
        inputs=(_decoded_input(tmp_path / "other", "obs:a", artifact_key="e"),),
        source_root=source,
        checkpoint_path=checkpoint,
        hardware_runtime=_hardware("d"),
    )
    request_c = da3_runtime.Da3ExecutionRequest(
        inputs=request_a.inputs,
        source_root=source,
        checkpoint_path=checkpoint,
        hardware_runtime=_hardware("f"),
    )

    result_a1 = da3_runtime.execute_da3_base_preview(request_a, runtime=runtime)
    result_a2 = da3_runtime.execute_da3_base_preview(request_a, runtime=runtime)
    result_b = da3_runtime.execute_da3_base_preview(request_b, runtime=runtime)
    result_c = da3_runtime.execute_da3_base_preview(request_c, runtime=runtime)

    assert result_a1.normalization_identity == result_a2.normalization_identity
    assert result_a1.geometry == result_a2.geometry
    assert result_b.normalization_identity != result_a1.normalization_identity
    assert result_c.normalization_identity != result_a1.normalization_identity


def test_runtime_prediction_membership_must_exactly_match_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    _allow_external_fixture(
        monkeypatch,
        source_root=source,
        checkpoint_path=checkpoint,
    )
    request = da3_runtime.Da3ExecutionRequest(
        inputs=(_decoded_input(tmp_path, "obs:a"),),
        source_root=source,
        checkpoint_path=checkpoint,
        hardware_runtime=_hardware(),
    )

    class _WrongRuntime:
        def infer(
            self, **kwargs: object
        ) -> tuple[
            da3_runtime.Da3EnvironmentIdentity,
            tuple[Da3BaseObservationPrediction, ...],
        ]:
            return _environment(), (_prediction(ObservationId("obs:foreign")),)

    with pytest.raises(da3_runtime.Da3RuntimeError, match="observations"):
        da3_runtime.execute_da3_base_preview(request, runtime=_WrongRuntime())


def test_isolated_import_disables_bytecode_and_restores_process_state(tmp_path: Path) -> None:
    source = _source_root(tmp_path)
    original = da3_runtime.sys.dont_write_bytecode

    with da3_runtime._isolated_da3_import(source):
        assert da3_runtime.sys.dont_write_bytecode is True
        assert str(source / "src") == da3_runtime.sys.path[0]

    assert da3_runtime.sys.dont_write_bytecode is original
    assert str(source / "src") not in da3_runtime.sys.path


def test_module_import_surface_has_no_learned_runtime_objects() -> None:
    forbidden = {
        "torch",
        "numpy",
        "cv2",
        "PIL",
        "safetensors_torch",
        "DepthAnything3",
        "InputProcessor",
        "OutputProcessor",
    }
    assert forbidden.isdisjoint(vars(da3_runtime))
