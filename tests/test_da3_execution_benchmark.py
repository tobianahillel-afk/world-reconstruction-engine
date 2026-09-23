from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from wre.domain import (
    ArtifactId,
    ArtifactKey,
    ArtifactKind,
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
    ArtifactProducerIdentity,
    ArtifactRef,
    BenchmarkFixtureId,
    BenchmarkFixtureIdentity,
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
    ConfigurationIdentity,
    DecodedImageLevelDescriptor,
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
    DecodedImagePyramidManifest,
    DecodedImagePyramidSpec,
    HardwareRuntimeIdentity,
    ImageDimensions,
    LocalFrameId,
    MetricAggregation,
    MetricDirection,
    MetricObservation,
    MetricProvenance,
    MetricVector,
    ObservationId,
    ObservationKind,
    ProducerRef,
    QualityMode,
    Sha256Digest,
)
from wre.ingestion.decoded_images import DecodedImagePyramidMaterializationResult
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.da3_execution_benchmark import (
    DA3_CAMERA_METRIC_ABS_TOLERANCE,
    DA3_CUDA_EAGER_PROFILE,
    DA3_EXECUTION_END_TO_END_STAGE,
    DA3_REFERENCE_PROFILE,
    Da3ExecutionBenchmarkRequest,
    Da3ExecutionProfileAvailability,
    Da3ExecutionProfileId,
    Da3ExecutionProfileStatus,
    Da3ExecutionSampleKind,
    Da3GeometrySemantics,
    Da3ProfileExecutionObservation,
    Da3StageTiming,
    benchmark_da3_execution,
    geometry_semantics,
)
from wre.reconstruction.da3_runtime import (
    DA3_PROFILE_STAGE_NAMES,
    Da3ExecutionRequest,
    Da3ImageInput,
    execute_da3_base_preview,
)
from wre.reconstruction.feed_forward_camera_quality import (
    OBSERVATION_COVERAGE_DESCRIPTOR,
    FeedForwardCameraQualityRequest,
    evaluate_feed_forward_camera_quality,
)


def _hardware(char: str = "a") -> HardwareRuntimeIdentity:
    return HardwareRuntimeIdentity(sha256=Sha256Digest(char * 64))


def _producer(char: str = "b") -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="test.da3.profile",
            version="1",
            revision=f"profile:{char}",
        ),
        configuration=ConfigurationIdentity(sha256=Sha256Digest(char * 64)),
    )


def _artifact(identifier: str = "fixture") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind("geometry.camera_reference"),
    )


def _fixture() -> BenchmarkFixtureIdentity:
    return BenchmarkFixtureIdentity(
        fixture_id=BenchmarkFixtureId("fixture.da3-execution.v1"),
        sha256=Sha256Digest("c" * 64),
    )


def _quality(value: float = 1.0) -> MetricVector:
    observation = MetricObservation(
        descriptor=OBSERVATION_COVERAGE_DESCRIPTOR,
        value=value,
        provenance=MetricProvenance(
            evaluator=_producer("d"),
            input_artifacts=(_artifact(),),
        ),
    )
    return MetricVector(observations=(observation,))


def _semantics(*, projection: str = "pinhole") -> Da3GeometrySemantics:
    return Da3GeometrySemantics(
        observation_ids=(ObservationId("obs:a"),),
        projection_models=(projection,),
        image_dimensions=((504, 504),),
        depth_dimensions=((504, 504),),
        depth_valid_counts=(504 * 504,),
        depth_all_valid_samples_positive=(True,),
        scale_status="unresolved",
        point_map_count=0,
    )


def _timings(
    *,
    base: float = 0.01,
    include_gpu_transfer: bool = False,
) -> tuple[Da3StageTiming, ...]:
    stages = list(DA3_PROFILE_STAGE_NAMES)
    if include_gpu_transfer:
        raise AssertionError("the current accepted runtime stage vocabulary has no transfer stage")
    return (
        *(
            Da3StageTiming(stage=stage, elapsed_seconds=float(base + index * 0.001))
            for index, stage in enumerate(stages)
        ),
        Da3StageTiming(
            stage=DA3_EXECUTION_END_TO_END_STAGE,
            elapsed_seconds=float(base + 0.5),
        ),
    )


def _observation(
    profile: str,
    *,
    hardware: HardwareRuntimeIdentity | None = None,
    quality: float = 1.0,
    semantics: Da3GeometrySemantics | None = None,
    base: float = 0.01,
    gpu: bool = False,
) -> Da3ProfileExecutionObservation:
    return Da3ProfileExecutionObservation(
        profile=Da3ExecutionProfileId(profile),
        producer=_producer("b" if profile == DA3_REFERENCE_PROFILE else "e"),
        hardware=hardware if hardware is not None else _hardware(),
        normalization_identity=Sha256Digest("f" * 64),
        stage_timings=_timings(base=base),
        peak_process_rss_bytes=123_456_789,
        peak_gpu_allocated_bytes=10_000 if gpu else None,
        peak_gpu_reserved_bytes=20_000 if gpu else None,
        semantics=semantics if semantics is not None else _semantics(),
        camera_quality=_quality(quality),
    )


class _Executor:
    def __init__(
        self,
        *,
        candidate_status: Da3ExecutionProfileStatus = Da3ExecutionProfileStatus.UNAVAILABLE,
        candidate_reason: str = "cuda unavailable",
        candidate_quality: float = 1.0,
        candidate_semantics: Da3GeometrySemantics | None = None,
        candidate_hardware: HardwareRuntimeIdentity | None = None,
    ) -> None:
        self.candidate_status = candidate_status
        self.candidate_reason = candidate_reason
        self.candidate_quality = candidate_quality
        self.candidate_semantics = candidate_semantics
        self.candidate_hardware = candidate_hardware
        self.calls: list[tuple[str, str, int]] = []

    def probe(self, profile: Da3ExecutionProfileId) -> Da3ExecutionProfileAvailability:
        if profile.value == DA3_REFERENCE_PROFILE:
            return Da3ExecutionProfileAvailability(
                profile=profile,
                status=Da3ExecutionProfileStatus.AVAILABLE,
            )
        return Da3ExecutionProfileAvailability(
            profile=profile,
            status=self.candidate_status,
            reason=None
            if self.candidate_status is Da3ExecutionProfileStatus.AVAILABLE
            else self.candidate_reason,
        )

    def execute(
        self,
        profile: Da3ExecutionProfileId,
        *,
        kind: Da3ExecutionSampleKind,
        sample_index: int,
    ) -> Da3ProfileExecutionObservation:
        self.calls.append((profile.value, kind.value, sample_index))
        if profile.value == DA3_REFERENCE_PROFILE:
            return _observation(
                profile.value,
                base=0.01 + sample_index * 0.001,
            )
        return _observation(
            profile.value,
            hardware=self.candidate_hardware,
            quality=self.candidate_quality,
            semantics=self.candidate_semantics,
            base=0.02 + sample_index * 0.001,
            gpu=True,
        )


def _request(**overrides: Any) -> Da3ExecutionBenchmarkRequest:
    values: dict[str, Any] = {
        "fixture": _fixture(),
        "hardware": _hardware(),
        "input_artifacts": (_artifact(),),
        "steady_state_samples": 2,
    }
    values.update(overrides)
    return Da3ExecutionBenchmarkRequest(**values)


def test_profile_ids_availability_and_request_are_exact_immutable() -> None:
    reference = Da3ExecutionProfileId(DA3_REFERENCE_PROFILE)
    cuda = Da3ExecutionProfileId(DA3_CUDA_EAGER_PROFILE)
    request = _request()

    assert request.reference_profile == reference
    assert request.candidate_profiles == (cuda,)
    with pytest.raises(FrozenInstanceError):
        request.steady_state_samples = 3  # type: ignore[misc]
    with pytest.raises(ValueError, match="unsupported"):
        Da3ExecutionProfileId("da3.magic.fast")
    with pytest.raises(ValueError, match="reason"):
        Da3ExecutionProfileAvailability(
            profile=cuda,
            status=Da3ExecutionProfileStatus.UNAVAILABLE,
        )
    with pytest.raises(ValueError, match="must not carry"):
        Da3ExecutionProfileAvailability(
            profile=cuda,
            status=Da3ExecutionProfileStatus.AVAILABLE,
            reason="unexpected",
        )


def test_request_rejects_noncanonical_or_duplicate_inputs_and_profiles() -> None:
    first = _artifact("a")
    second = _artifact("b")
    cuda = Da3ExecutionProfileId(DA3_CUDA_EAGER_PROFILE)

    with pytest.raises(ValueError, match="canonical"):
        _request(input_artifacts=(second, first))
    with pytest.raises(ValueError, match="unique"):
        _request(input_artifacts=(first, first))
    with pytest.raises(ValueError, match="unique"):
        _request(candidate_profiles=(cuda, cuda))
    with pytest.raises(ValueError, match="positive"):
        _request(steady_state_samples=0)


def test_stage_timing_and_geometry_semantics_fail_closed() -> None:
    timing = Da3StageTiming(stage=DA3_EXECUTION_END_TO_END_STAGE, elapsed_seconds=1.0)
    assert timing.elapsed_seconds == 1.0

    with pytest.raises(ValueError, match="unsupported"):
        Da3StageTiming(stage="hidden_gpu_magic", elapsed_seconds=1.0)
    for value in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="finite and non-negative"):
            Da3StageTiming(stage=DA3_EXECUTION_END_TO_END_STAGE, elapsed_seconds=value)

    semantics = _semantics()
    assert semantics.point_map_count == 0
    with pytest.raises(ValueError, match="observation_ids"):
        replace(semantics, observation_ids=())
    with pytest.raises(ValueError, match="projection_models"):
        replace(semantics, projection_models=())


def test_unavailable_candidate_is_retained_without_hidden_substitution() -> None:
    executor = _Executor()
    result = benchmark_da3_execution(_request(), executor=executor)

    assert result.reference.profile == Da3ExecutionProfileId(DA3_REFERENCE_PROFILE)
    assert result.reference.benchmark_record.quality_mode is QualityMode.PREVIEW
    assert result.reference.benchmark_record.comparison_baseline is None
    assert len(result.reference.steady_samples) == 2
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.availability.status is Da3ExecutionProfileStatus.UNAVAILABLE
    assert candidate.availability.reason == "cuda unavailable"
    assert candidate.measured is None
    assert all(call[0] == DA3_REFERENCE_PROFILE for call in executor.calls)

    names = tuple(
        item.descriptor.name.value
        for item in result.reference.benchmark_record.metrics.observations
    )
    assert "geometry.camera.observation_coverage_ratio" in names
    assert "runtime.da3.end_to_end.cold_seconds" in names
    assert "runtime.da3.end_to_end.steady_median_seconds" in names
    assert "runtime.da3.peak_process_rss_bytes" in names
    assert not any("gpu" in name for name in names)
    assert result.reference.benchmark_record.performance_evidence[0].artifact_kind == ArtifactKind(
        "performance.execution_profile"
    )


def test_available_candidate_links_baseline_and_retains_gpu_memory_evidence() -> None:
    executor = _Executor(candidate_status=Da3ExecutionProfileStatus.AVAILABLE)
    result = benchmark_da3_execution(_request(steady_state_samples=1), executor=executor)

    candidate = result.candidates[0]
    assert candidate.availability.status is Da3ExecutionProfileStatus.AVAILABLE
    assert candidate.measured is not None
    assert candidate.measured.benchmark_record.comparison_baseline == (
        result.reference.benchmark_record.record_id
    )
    assert candidate.measured.benchmark_record.hardware == (
        result.reference.benchmark_record.hardware
    )
    names = {
        item.descriptor.name.value
        for item in candidate.measured.benchmark_record.metrics.observations
    }
    assert "runtime.da3.peak_gpu_allocated_bytes" in names
    assert "runtime.da3.peak_gpu_reserved_bytes" in names
    for attribute in ("winner", "rank", "default", "threshold", "decision", "route"):
        assert not hasattr(candidate.measured.benchmark_record, attribute)


def test_candidate_semantic_or_camera_quality_regression_becomes_failed_evidence() -> None:
    semantic_executor = _Executor(
        candidate_status=Da3ExecutionProfileStatus.AVAILABLE,
        candidate_semantics=_semantics(projection="foreign"),
    )
    semantic_result = benchmark_da3_execution(_request(), executor=semantic_executor)
    assert semantic_result.candidates[0].availability.status is Da3ExecutionProfileStatus.FAILED
    assert "geometry semantics" in (semantic_result.candidates[0].availability.reason or "")

    quality_executor = _Executor(
        candidate_status=Da3ExecutionProfileStatus.AVAILABLE,
        candidate_quality=1.0 - DA3_CAMERA_METRIC_ABS_TOLERANCE * 2.0,
    )
    quality_result = benchmark_da3_execution(_request(), executor=quality_executor)
    assert quality_result.candidates[0].availability.status is Da3ExecutionProfileStatus.FAILED
    assert "camera quality" in (quality_result.candidates[0].availability.reason or "")


def test_different_hardware_cannot_be_reported_as_controlled_candidate() -> None:
    executor = _Executor(
        candidate_status=Da3ExecutionProfileStatus.AVAILABLE,
        candidate_hardware=_hardware("9"),
    )
    result = benchmark_da3_execution(_request(), executor=executor)

    candidate = result.candidates[0]
    assert candidate.availability.status is Da3ExecutionProfileStatus.FAILED
    assert "hardware identity differs" in (candidate.availability.reason or "")
    assert candidate.measured is None


def test_reference_sample_semantics_and_quality_must_be_repeatable() -> None:
    class _ChangingExecutor(_Executor):
        def execute(
            self,
            profile: Da3ExecutionProfileId,
            *,
            kind: Da3ExecutionSampleKind,
            sample_index: int,
        ) -> Da3ProfileExecutionObservation:
            observation = super().execute(
                profile,
                kind=kind,
                sample_index=sample_index,
            )
            if profile.value == DA3_REFERENCE_PROFILE and sample_index == 1:
                return replace(observation, camera_quality=_quality(0.5))
            return observation

    with pytest.raises(ValueError, match="camera-quality"):
        benchmark_da3_execution(_request(), executor=_ChangingExecutor())


def test_performance_metric_medians_and_provenance_are_deterministic() -> None:
    request = _request(steady_state_samples=2)
    first = benchmark_da3_execution(request, executor=_Executor())
    second = benchmark_da3_execution(request, executor=_Executor())

    first_record = first.reference.benchmark_record
    second_record = second.reference.benchmark_record
    assert first_record == second_record
    metrics = {item.descriptor.name.value: item for item in first_record.metrics.observations}
    observed = metrics["runtime.da3.end_to_end.steady_median_seconds"]
    assert observed.value == pytest.approx((0.511 + 0.512) / 2.0)
    assert observed.descriptor.aggregation == MetricAggregation("median")
    assert observed.descriptor.direction is MetricDirection.LOWER_IS_BETTER
    assert observed.provenance.input_artifacts == request.input_artifacts


def _real_rgb_bytes() -> bytes:
    width = 504
    height = 504
    payload = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            index = (y * width + x) * 3
            payload[index] = (x * 7 + y * 3) % 256
            payload[index + 1] = (x * 2 + y * 11) % 256
            payload[index + 2] = (x * 13 + y * 5) % 256
    return bytes(payload)


def _real_decoded_input(tmp_path: Path) -> Da3ImageInput:
    rgb8 = _real_rgb_bytes()
    root = tmp_path / "decoded"
    level_path = root / "levels" / "level-000000.rgb"
    level_path.parent.mkdir(parents=True)
    level_path.write_bytes(rgb8)
    content_hash = hash_file_content(level_path)
    observation_id = ObservationId("obs:da3-benchmark")
    artifact_ref = ArtifactRef(
        artifact_id=ArtifactId("artifact:da3-benchmark"),
        artifact_kind=ArtifactKind("media.decoded_image_pyramid"),
    )
    result = DecodedImagePyramidMaterializationResult(
        artifact_key=ArtifactKey(
            sha256=Sha256Digest(hashlib.sha256(b"da3-benchmark-artifact").hexdigest())
        ),
        producer=_producer("1"),
        manifest=DecodedImagePyramidManifest(
            source_observation_id=observation_id,
            source_kind=ObservationKind.IMAGE,
            source_asset_sha256=content_hash.sha256,
            pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
            orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
            spec=DecodedImagePyramidSpec(minimum_max_edge_px=504),
            levels=(
                DecodedImageLevelDescriptor(
                    level_index=0,
                    width_px=504,
                    height_px=504,
                    relative_path="levels/level-000000.rgb",
                ),
            ),
        ),
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
    return Da3ImageInput(decoded_result=result, materialization_root=root)


def _real_reference_camera() -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId("camera:da3-benchmark-reference"),
        observation_id=ObservationId("obs:da3-benchmark"),
        local_frame_id=LocalFrameId("frame:da3-benchmark-reference"),
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(width_px=504, height_px=504),
        intrinsic_parameters=(400.0, 400.0, 252.0, 252.0),
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_xyz=(0.0, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=MetricVector(observations=()),
    )


@pytest.mark.skipif(
    os.environ.get("WRE_DA3_EXECUTION_BENCHMARK") != "1",
    reason="exact DA3 execution benchmark environment is not provisioned",
)
def test_real_da3_reference_execution_profile_benchmark(tmp_path: Path) -> None:
    source_root = Path(os.environ["WRE_DA3_SOURCE_ROOT"])
    checkpoint_path = Path(os.environ["WRE_DA3_CHECKPOINT_PATH"])
    output_path = Path(os.environ["WRE_DA3_EXECUTION_PROFILE_OUTPUT"])
    hardware = _hardware_from_environment()
    decoded = _real_decoded_input(tmp_path)
    input_artifact = decoded.decoded_result.materialization.artifact_ref
    reference_camera = _real_reference_camera()

    execution_request = Da3ExecutionRequest(
        inputs=(decoded,),
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        hardware_runtime=hardware,
    )

    class _RealExecutor:
        def probe(
            self,
            profile: Da3ExecutionProfileId,
        ) -> Da3ExecutionProfileAvailability:
            if profile.value == DA3_REFERENCE_PROFILE:
                return Da3ExecutionProfileAvailability(
                    profile=profile,
                    status=Da3ExecutionProfileStatus.AVAILABLE,
                )
            import torch

            assert torch.__version__ == "2.4.1+cpu"
            assert torch.cuda.is_available() is False
            return Da3ExecutionProfileAvailability(
                profile=profile,
                status=Da3ExecutionProfileStatus.UNAVAILABLE,
                reason="cuda_not_available_in_exact_cpu_reference_environment",
            )

        def execute(
            self,
            profile: Da3ExecutionProfileId,
            *,
            kind: Da3ExecutionSampleKind,
            sample_index: int,
        ) -> Da3ProfileExecutionObservation:
            assert profile.value == DA3_REFERENCE_PROFILE
            timings: list[Da3StageTiming] = []

            def _observe(stage: str, elapsed_seconds: float) -> None:
                timings.append(
                    Da3StageTiming(
                        stage=stage,
                        elapsed_seconds=elapsed_seconds,
                    )
                )

            started = time.perf_counter()
            result = execute_da3_base_preview(
                execution_request,
                stage_observer=_observe,
            )
            elapsed = float(time.perf_counter() - started)
            timings.append(
                Da3StageTiming(
                    stage=DA3_EXECUTION_END_TO_END_STAGE,
                    elapsed_seconds=elapsed,
                )
            )
            quality = evaluate_feed_forward_camera_quality(
                FeedForwardCameraQualityRequest(
                    candidate=result.geometry,
                    reference_cameras=(reference_camera,),
                    input_artifacts=(input_artifact,),
                )
            )
            return Da3ProfileExecutionObservation(
                profile=profile,
                producer=result.producer,
                hardware=result.hardware_runtime,
                normalization_identity=result.normalization_identity,
                stage_timings=tuple(timings),
                peak_process_rss_bytes=_peak_rss_bytes(),
                peak_gpu_allocated_bytes=None,
                peak_gpu_reserved_bytes=None,
                semantics=geometry_semantics(result),
                camera_quality=quality,
            )

    fixture_sha = Sha256Digest(hashlib.sha256(_real_rgb_bytes()).hexdigest())
    request = Da3ExecutionBenchmarkRequest(
        fixture=BenchmarkFixtureIdentity(
            fixture_id=BenchmarkFixtureId("fixture.da3-execution-504.v1"),
            sha256=fixture_sha,
        ),
        hardware=hardware,
        input_artifacts=(input_artifact,),
        steady_state_samples=1,
    )
    result = benchmark_da3_execution(request, executor=_RealExecutor())

    assert result.reference.benchmark_record.quality_mode is QualityMode.PREVIEW
    assert result.reference.cold_sample.observation.semantics.scale_status == "unresolved"
    assert result.reference.cold_sample.observation.semantics.point_map_count == 0
    assert result.reference.cold_sample.observation.semantics.depth_all_valid_samples_positive == (
        True,
    )
    assert result.candidates[0].availability.status is Da3ExecutionProfileStatus.UNAVAILABLE
    assert result.candidates[0].measured is None
    metric_names = {
        item.descriptor.name.value
        for item in result.reference.benchmark_record.metrics.observations
    }
    assert "geometry.camera.observation_coverage_ratio" in metric_names
    assert "geometry.camera.focal_relative_error_median" in metric_names
    assert "geometry.camera.principal_point_error_normalized_median" in metric_names
    assert "runtime.da3.end_to_end.cold_seconds" in metric_names
    assert "runtime.da3.end_to_end.steady_median_seconds" in metric_names
    assert "runtime.da3.model_execution.cold_seconds" in metric_names
    assert not any("gpu" in name for name in metric_names)

    output_path.write_text(
        json.dumps(_result_document(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hardware_from_environment() -> HardwareRuntimeIdentity:
    return HardwareRuntimeIdentity(
        sha256=Sha256Digest(os.environ["WRE_DA3_HARDWARE_RUNTIME_SHA256"])
    )


def _peak_rss_bytes() -> int:
    # GitHub's benchmark lane is Linux, where ru_maxrss is KiB.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert value > 0
    return value * 1024


def _result_document(result: Any) -> dict[str, Any]:
    reference = result.reference
    metrics = {
        item.descriptor.name.value: item.value
        for item in reference.benchmark_record.metrics.observations
    }
    return {
        "schema_version": 1,
        "reference_profile": reference.profile.value,
        "reference_benchmark_record_id": reference.benchmark_record.record_id.value,
        "fixture_id": reference.benchmark_record.fixture.fixture_id.value,
        "fixture_sha256": reference.benchmark_record.fixture.sha256.value,
        "hardware_sha256": reference.benchmark_record.hardware.sha256.value,
        "quality_mode": reference.benchmark_record.quality_mode.value,
        "metrics": metrics,
        "performance_evidence": [
            {
                "artifact_id": item.artifact_id.value,
                "artifact_kind": item.artifact_kind.value,
            }
            for item in reference.benchmark_record.performance_evidence
        ],
        "candidates": [
            {
                "profile": item.availability.profile.value,
                "status": item.availability.status.value,
                "reason": item.availability.reason,
                "measured": item.measured is not None,
            }
            for item in result.candidates
        ],
        "claims": {
            "accelerated_profile_validated": False,
            "speedup_claimed": False,
            "winner": None,
            "default_profile": None,
            "shipping_promotion": False,
        },
    }
