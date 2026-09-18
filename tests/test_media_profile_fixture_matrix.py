from __future__ import annotations

from dataclasses import fields

from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactProducerIdentity,
    ArtifactRef,
    ConfigurationIdentity,
    ImageDimensions,
    MediaProfile,
    MetricDirection,
    MetricProvenance,
    ObservationId,
    ObservationKind,
    ObservationMetadata,
    ProducerRef,
    RawMetadataEntry,
    Sha256Digest,
)
from wre.profiling import (
    InputClass,
    InputClassEvidenceKind,
    LumaRaster,
    ProfileSummaryInput,
    evaluate_image_quality,
    evaluate_input_classification,
    evaluate_profile_summary,
    evaluate_visual_similarity,
)


def _artifact_ref(artifact_id: str, artifact_kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=ArtifactKind(artifact_kind),
    )


def _provenance(revision: str) -> MetricProvenance:
    return MetricProvenance(
        evaluator=ArtifactProducerIdentity(
            producer=ProducerRef(
                implementation="wre.profiling.fixture_matrix",
                version="1.0.0",
                revision=revision,
            ),
            configuration=ConfigurationIdentity(sha256=Sha256Digest("d" * 64)),
        ),
        input_artifacts=(
            _artifact_ref("artifact:fixture:input", "media.fixture_input"),
        ),
    )


def _metadata(
    *,
    width: int | None = None,
    height: int | None = None,
    entries: tuple[RawMetadataEntry, ...] = (),
) -> ObservationMetadata:
    dimensions = (
        None
        if width is None or height is None
        else ImageDimensions(width_px=width, height_px=height)
    )
    return ObservationMetadata(
        observation_id=ObservationId("obs:fixture-matrix"),
        dimensions=dimensions,
        raw_entries=entries,
    )


def _entry(key: str, value: str) -> RawMetadataEntry:
    return RawMetadataEntry(namespace="exif", key=key, value=value)


def _metric_values(vector: object) -> dict[str, float]:
    observations = getattr(vector, "observations")
    return {
        observation.descriptor.name.value: observation.value
        for observation in observations
    }


def test_ordinary_still_composes_profile_quality_class_and_summary_without_truth_leakage() -> None:
    evidence = (
        _artifact_ref("artifact:fixture:metadata", "media.metadata"),
        _artifact_ref("artifact:fixture:quality", "media.quality"),
    )
    profile = MediaProfile(
        observation_id=ObservationId("obs:fixture:ordinary-still"),
        evidence_artifacts=evidence,
    )
    raster = LumaRaster(
        width=3,
        height=3,
        pixels=bytes(
            [
                0,
                64,
                128,
                32,
                96,
                160,
                64,
                128,
                255,
            ]
        ),
    )
    provenance = _provenance("ordinary-still")

    quality = evaluate_image_quality(raster, provenance)
    input_classes = evaluate_input_classification(ObservationKind.IMAGE)
    summary = evaluate_profile_summary(
        ProfileSummaryInput(
            observation_count=1,
            distinct_source_count=1,
        ),
        provenance,
    )

    assert profile.evidence_artifacts == evidence
    assert tuple(item.input_class for item in input_classes) == (InputClass.STILL,)
    assert input_classes[0].evidence_kind is InputClassEvidenceKind.OBSERVED
    assert tuple(item.descriptor.name.value for item in quality.observations) == (
        "media.exposure.black_clip_fraction",
        "media.exposure.mean_luma",
        "media.exposure.white_clip_fraction",
        "media.sharpness.laplacian_variance",
    )
    assert tuple(item.descriptor.name.value for item in summary.observations) == (
        "media.collection.distinct_source_count",
        "media.collection.observation_count",
    )
    assert all(item.provenance is provenance for item in quality.observations)
    assert all(item.provenance is provenance for item in summary.observations)


def test_specialist_hints_coexist_only_as_candidates_beside_observed_kind() -> None:
    metadata = _metadata(
        width=4096,
        height=2048,
        entries=(
            _entry("Image Model", "DJI Mavic 3 Pro"),
            _entry("Lens Model", "Fisheye"),
            _entry("Sensor Mode", "rolling_shutter"),
        ),
    )

    result = evaluate_input_classification(ObservationKind.IMAGE, metadata)

    assert tuple(item.input_class for item in result) == (
        InputClass.STILL,
        InputClass.EQUIRECTANGULAR_360,
        InputClass.DRONE_AERIAL,
        InputClass.FISHEYE,
        InputClass.ROLLING_SHUTTER,
    )
    assert result[0].evidence_kind is InputClassEvidenceKind.OBSERVED
    assert all(
        item.evidence_kind is InputClassEvidenceKind.CANDIDATE
        for item in result[1:]
    )
    assert metadata.dimensions == ImageDimensions(width_px=4096, height_px=2048)
    assert metadata.raw_entries == (
        _entry("Image Model", "DJI Mavic 3 Pro"),
        _entry("Lens Model", "Fisheye"),
        _entry("Sensor Mode", "rolling_shutter"),
    )


def test_video_sequence_metrics_remain_informational_evidence_not_labels() -> None:
    provenance = _provenance("video-sequence")
    classes = evaluate_input_classification(ObservationKind.VIDEO_FRAME)
    summary = evaluate_profile_summary(
        ProfileSummaryInput(
            observation_count=12,
            distinct_source_count=2,
            video_duration_us=5_000_000,
            temporal_grid_luma_changes=(0.2, 0.8),
            coverage_grid_luma_distances=(0.1, 0.9),
        ),
        provenance,
    )
    values = _metric_values(summary)

    assert tuple(item.input_class for item in classes) == (InputClass.VIDEO,)
    assert classes[0].evidence_kind is InputClassEvidenceKind.OBSERVED
    assert values == {
        "media.collection.distinct_source_count": 2.0,
        "media.collection.observation_count": 12.0,
        "media.coverage.grid_luma_diversity_mean": 0.5,
        "media.sequence.duration_seconds": 5.0,
        "media.temporal.grid_luma_change_mean": 0.5,
    }
    assert all(
        item.descriptor.direction is MetricDirection.INFORMATIONAL
        for item in summary.observations
    )
    assert all(item.provenance is provenance for item in summary.observations)


def test_missing_optional_evidence_stays_absent_instead_of_becoming_false_or_zero() -> None:
    provenance = _provenance("missing-evidence")
    classes = evaluate_input_classification(ObservationKind.VIDEO)
    summary = evaluate_profile_summary(
        ProfileSummaryInput(
            observation_count=3,
            distinct_source_count=1,
        ),
        provenance,
    )

    assert tuple(item.input_class for item in classes) == (InputClass.VIDEO,)
    assert tuple(item.descriptor.name.value for item in summary.observations) == (
        "media.collection.distinct_source_count",
        "media.collection.observation_count",
    )
    assert "media.sequence.duration_seconds" not in _metric_values(summary)
    assert "media.temporal.grid_luma_change_mean" not in _metric_values(summary)
    assert "media.coverage.grid_luma_diversity_mean" not in _metric_values(summary)


def test_visual_similarity_remains_complementary_measurement_not_duplicate_decision() -> None:
    provenance = _provenance("visual-similarity")
    black = LumaRaster(width=8, height=8, pixels=bytes([0] * 64))
    white = LumaRaster(width=8, height=8, pixels=bytes([255] * 64))

    identical = _metric_values(
        evaluate_visual_similarity(black, black, provenance)
    )
    complementary = _metric_values(
        evaluate_visual_similarity(black, white, provenance)
    )

    assert identical == {
        "media.visual.average_hash_hamming_fraction": 0.0,
        "media.visual.grid_luma_mae": 0.0,
    }
    assert complementary == {
        "media.visual.average_hash_hamming_fraction": 0.0,
        "media.visual.grid_luma_mae": 1.0,
    }


def test_media_profile_remains_exact_evidence_link_boundary_across_matrix() -> None:
    assert tuple(field.name for field in fields(MediaProfile)) == (
        "observation_id",
        "evidence_artifacts",
    )

    profile = MediaProfile(
        observation_id=ObservationId("obs:fixture:boundary"),
        evidence_artifacts=(
            _artifact_ref("artifact:fixture:classes", "media.input_class_evidence"),
            _artifact_ref("artifact:fixture:metrics", "media.metric_vector"),
        ),
    )

    for attribute in (
        "metrics",
        "input_classes",
        "summary",
        "signals",
        "metadata",
        "quality_score",
        "confidence",
        "truth",
        "route",
        "selection",
        "solver",
    ):
        assert not hasattr(profile, attribute)


def test_composed_evaluators_do_not_mutate_inputs_or_create_cross_signal_decisions() -> None:
    provenance = _provenance("non-mutation")
    raster = LumaRaster(
        width=3,
        height=3,
        pixels=bytes([0, 32, 64, 96, 128, 160, 192, 224, 255]),
    )
    metadata = _metadata(
        width=4000,
        height=2000,
        entries=(_entry("Lens Model", "Fisheye"),),
    )
    summary_input = ProfileSummaryInput(
        observation_count=4,
        distinct_source_count=2,
        video_duration_us=1_000_000,
        temporal_grid_luma_changes=(0.25, 0.75),
        coverage_grid_luma_distances=(0.2, 0.6),
    )

    raster_before = raster
    metadata_before = metadata
    summary_before = summary_input

    quality = evaluate_image_quality(raster, provenance)
    similarity = evaluate_visual_similarity(raster, raster, provenance)
    classes = evaluate_input_classification(ObservationKind.IMAGE, metadata)
    summary = evaluate_profile_summary(summary_input, provenance)

    assert raster is raster_before
    assert metadata is metadata_before
    assert summary_input is summary_before
    assert tuple(item.value for item in similarity.observations) == (0.0, 0.0)
    assert all(item.provenance is provenance for item in quality.observations)
    assert all(item.provenance is provenance for item in similarity.observations)
    assert all(item.provenance is provenance for item in summary.observations)
    assert tuple(item.input_class for item in classes) == (
        InputClass.STILL,
        InputClass.EQUIRECTANGULAR_360,
        InputClass.FISHEYE,
    )
