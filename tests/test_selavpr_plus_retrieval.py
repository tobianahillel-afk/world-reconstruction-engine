from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import wre.retrieval.selavpr_plus as selavpr_module
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
    ModelIdentity,
    ObservationId,
    ObservationKind,
    PairCandidate,
    PairCandidateSourceId,
    ProducerRef,
    ReconstructionRun,
    ReconstructionRunId,
    Sha256Digest,
)
from wre.ingestion.decoded_images import DecodedImagePyramidMaterializationResult
from wre.ingestion.hashing import FileContentHash, hash_file_content
from wre.retrieval.selavpr_plus import (
    SELAVPR_PLUS_CHECKPOINT,
    SELAVPR_PLUS_CHECKPOINT_SHA256,
    SELAVPR_PLUS_FAISS_CPU_VERSION,
    SELAVPR_PLUS_MODEL,
    SELAVPR_PLUS_NUMPY_VERSION,
    SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
    SELAVPR_PLUS_RETRIEVAL_IMPLEMENTATION,
    SELAVPR_PLUS_RETRIEVAL_VERSION,
    SELAVPR_PLUS_SOURCE_REVISION,
    SELAVPR_PLUS_TORCH_VERSION,
    SELAVPR_PLUS_TQDM_VERSION,
    SelaVprPlusEnvironmentIdentity,
    SelaVprPlusImageInput,
    SelaVprPlusPair,
    SelaVprPlusPairCandidateAdapterInput,
    SelaVprPlusRetrievalConfig,
    SelaVprPlusRetrievalError,
    SelaVprPlusRetrievalRequest,
    SelaVprPlusRetrievalResult,
    adapt_selavpr_plus_retrieval_result,
    propose_selavpr_plus_pairs,
    retrieve_selavpr_plus_pairs,
)

NOW = datetime(2026, 9, 20, 0, 40, tzinfo=UTC)


def _unit_descriptor(index: int) -> tuple[float, ...]:
    return tuple(1.0 if position == index else 0.0 for position in range(2048))


def _environment() -> SelaVprPlusEnvironmentIdentity:
    return SelaVprPlusEnvironmentIdentity(
        source_revision=SELAVPR_PLUS_SOURCE_REVISION,
        python_version="3.12.test",
        torch_version=SELAVPR_PLUS_TORCH_VERSION,
        numpy_version=SELAVPR_PLUS_NUMPY_VERSION,
        faiss_cpu_version=SELAVPR_PLUS_FAISS_CPU_VERSION,
        tqdm_version=SELAVPR_PLUS_TQDM_VERSION,
        device="cpu",
    )


def _decoded_input(
    tmp_path: Path,
    observation_value: str,
    *,
    rgb8: bytes = b"\x00\x7f\xff\xff\x7f\x00",
) -> SelaVprPlusImageInput:
    token = observation_value.replace(":", "-")
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
        source_observation_id=ObservationId(observation_value),
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
        artifact_key=ArtifactKey(sha256=Sha256Digest("b" * 64)),
        producer=ArtifactProducerIdentity(
            producer=ProducerRef(
                implementation="test.decoded",
                version="1",
            ),
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
    return SelaVprPlusImageInput(
        decoded_result=result,
        materialization_root=root,
    )


def _source_root(tmp_path: Path, *, revision: str = SELAVPR_PLUS_SOURCE_REVISION) -> Path:
    root = tmp_path / "selavpr-source"
    for relative in selavpr_module._MODEL_REQUIRED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# exact-source-fixture\n", encoding="utf-8")
    (root / ".wre-selavpr-plus-revision").write_text(revision + "\n", encoding="utf-8")
    return root


def _checkpoint_path(tmp_path: Path) -> Path:
    path = tmp_path / "SelaVPRplusplus_base.pth"
    path.write_bytes(b"fixture-checkpoint")
    return path


def _run(
    inputs: tuple[SelaVprPlusImageInput, ...],
    config: SelaVprPlusRetrievalConfig,
    *,
    observation_ids: tuple[ObservationId, ...] | None = None,
) -> ReconstructionRun:
    ids = observation_ids or tuple(
        item.decoded_result.manifest.source_observation_id for item in inputs
    )
    return ReconstructionRun(
        run_id=ReconstructionRunId("run:selavpr-plus"),
        producer=ProducerRef(
            implementation=SELAVPR_PLUS_RETRIEVAL_IMPLEMENTATION,
            version=SELAVPR_PLUS_RETRIEVAL_VERSION,
            revision=SELAVPR_PLUS_SOURCE_REVISION,
        ),
        input_observation_ids=ids,
        started_at=NOW,
        configuration_sha256=config.sha256,
    )


def _request(
    tmp_path: Path,
    *,
    inputs: tuple[SelaVprPlusImageInput, ...] | None = None,
    config: SelaVprPlusRetrievalConfig | None = None,
) -> SelaVprPlusRetrievalRequest:
    actual_inputs = inputs or (
        _decoded_input(tmp_path, "obs:a"),
        _decoded_input(tmp_path, "obs:b"),
        _decoded_input(tmp_path, "obs:c"),
    )
    actual_config = config or SelaVprPlusRetrievalConfig(neighbors_per_observation=1)
    return SelaVprPlusRetrievalRequest(
        run=_run(actual_inputs, actual_config),
        inputs=actual_inputs,
        source_root=_source_root(tmp_path),
        checkpoint_path=_checkpoint_path(tmp_path),
        config=actual_config,
    )


def _allow_fixture_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_path: Path,
) -> None:
    real_hash = selavpr_module.hash_file_content
    resolved = checkpoint_path.resolve()

    def _hash(path: Path) -> FileContentHash:
        if path.resolve() == resolved:
            return FileContentHash(
                sha256=SELAVPR_PLUS_CHECKPOINT_SHA256,
                byte_length=path.stat().st_size,
            )
        return real_hash(path)

    monkeypatch.setattr(selavpr_module, "hash_file_content", _hash)


class _FakeRuntime:
    def __init__(self, descriptors: tuple[tuple[float, ...], ...]) -> None:
        self.descriptors = descriptors
        self.calls: list[dict[str, object]] = []

    def infer(
        self,
        *,
        source_root: Path,
        checkpoint_path: Path,
        images: tuple[object, ...],
        config: SelaVprPlusRetrievalConfig,
    ) -> tuple[SelaVprPlusEnvironmentIdentity, tuple[tuple[float, ...], ...]]:
        self.calls.append(
            {
                "source_root": source_root,
                "checkpoint_path": checkpoint_path,
                "images": images,
                "config": config,
            }
        )
        return _environment(), self.descriptors


def test_config_is_exact_immutable_and_content_identified() -> None:
    config = SelaVprPlusRetrievalConfig(neighbors_per_observation=7)

    assert tuple(field.name for field in fields(SelaVprPlusRetrievalConfig)) == (
        "schema_version",
        "image_height_px",
        "image_width_px",
        "descriptor_dimension",
        "neighbors_per_observation",
        "random_seed",
        "device",
    )
    assert config.canonical_document() == {
        "descriptor_dimension": 2048,
        "device": "cpu",
        "image_height_px": 322,
        "image_width_px": 322,
        "neighbors_per_observation": 7,
        "random_seed": 0,
        "schema_version": 1,
    }
    expected = hashlib.sha256(
        b'{"descriptor_dimension":2048,"device":"cpu","image_height_px":322,'
        b'"image_width_px":322,"neighbors_per_observation":7,"random_seed":0,'
        b'"schema_version":1}'
    ).hexdigest()
    assert config.sha256 == Sha256Digest(expected)
    with pytest.raises(FrozenInstanceError):
        config.device = "cuda"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"schema_version": True},
        {"schema_version": 0},
        {"image_height_px": 322.0},
        {"image_height_px": 321},
        {"image_width_px": 322.0},
        {"image_width_px": 321},
        {"descriptor_dimension": 2048.0},
        {"descriptor_dimension": 512},
        {"neighbors_per_observation": True},
        {"neighbors_per_observation": 0},
        {"neighbors_per_observation": -1},
        {"random_seed": True},
        {"random_seed": 1},
        {"device": "cuda"},
    ],
)
def test_config_rejects_non_reference_states(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SelaVprPlusRetrievalConfig(**cast(Any, kwargs))


def test_request_is_exact_and_requires_canonical_identity(tmp_path: Path) -> None:
    request = _request(tmp_path)

    assert tuple(field.name for field in fields(SelaVprPlusImageInput)) == (
        "decoded_result",
        "materialization_root",
    )
    assert tuple(field.name for field in fields(SelaVprPlusRetrievalRequest)) == (
        "run",
        "inputs",
        "source_root",
        "checkpoint_path",
        "model",
        "checkpoint",
        "config",
    )
    assert request.model == SELAVPR_PLUS_MODEL
    assert request.checkpoint == SELAVPR_PLUS_CHECKPOINT
    assert request.run.input_observation_ids == (
        ObservationId("obs:a"),
        ObservationId("obs:b"),
        ObservationId("obs:c"),
    )

    with pytest.raises(ValueError, match="canonical observation order"):
        SelaVprPlusRetrievalRequest(
            run=request.run,
            inputs=tuple(reversed(request.inputs)),
            source_root=request.source_root,
            checkpoint_path=request.checkpoint_path,
            config=request.config,
        )

    with pytest.raises(ValueError, match="exact reviewed model"):
        SelaVprPlusRetrievalRequest(
            run=request.run,
            inputs=request.inputs,
            source_root=request.source_root,
            checkpoint_path=request.checkpoint_path,
            model=ModelIdentity(name="other", version="1"),
            config=request.config,
        )


def test_request_rejects_run_membership_or_producer_mismatch(tmp_path: Path) -> None:
    request = _request(tmp_path)
    wrong_ids = request.run.input_observation_ids[:-1]

    with pytest.raises(ValueError, match="exactly match"):
        SelaVprPlusRetrievalRequest(
            run=_run(request.inputs, request.config, observation_ids=wrong_ids),
            inputs=request.inputs,
            source_root=request.source_root,
            checkpoint_path=request.checkpoint_path,
            config=request.config,
        )

    wrong_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:wrong"),
        producer=ProducerRef(
            implementation="wrong",
            version=SELAVPR_PLUS_RETRIEVAL_VERSION,
            revision=SELAVPR_PLUS_SOURCE_REVISION,
        ),
        input_observation_ids=request.run.input_observation_ids,
        started_at=NOW,
        configuration_sha256=request.config.sha256,
    )
    with pytest.raises(ValueError, match="producer"):
        SelaVprPlusRetrievalRequest(
            run=wrong_run,
            inputs=request.inputs,
            source_root=request.source_root,
            checkpoint_path=request.checkpoint_path,
            config=request.config,
        )


def test_source_root_requires_exact_local_revision_and_model_files(tmp_path: Path) -> None:
    valid = _source_root(tmp_path)
    assert selavpr_module._verify_source_root(valid) == valid.resolve()

    wrong = tmp_path / "wrong-source"
    for relative in selavpr_module._MODEL_REQUIRED_PATHS:
        path = wrong / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    (wrong / ".wre-selavpr-plus-revision").write_text("0" * 40, encoding="utf-8")
    with pytest.raises(SelaVprPlusRetrievalError, match="revision"):
        selavpr_module._verify_source_root(wrong)

    missing = tmp_path / "missing-source"
    missing.mkdir()
    (missing / ".wre-selavpr-plus-revision").write_text(
        SELAVPR_PLUS_SOURCE_REVISION,
        encoding="utf-8",
    )
    with pytest.raises(SelaVprPlusRetrievalError, match="missing required file"):
        selavpr_module._verify_source_root(missing)

    with pytest.raises(SelaVprPlusRetrievalError, match="explicit local path"):
        selavpr_module._verify_source_root(Path("https://example.invalid/model"))


def test_checkpoint_must_be_local_regular_file_with_exact_sha(tmp_path: Path) -> None:
    checkpoint = _checkpoint_path(tmp_path)

    with pytest.raises(SelaVprPlusRetrievalError, match="SHA-256"):
        selavpr_module._verify_checkpoint(checkpoint, SELAVPR_PLUS_CHECKPOINT)

    with pytest.raises(SelaVprPlusRetrievalError, match="explicit local path"):
        selavpr_module._verify_checkpoint(
            Path("https://example.invalid/checkpoint.pth"),
            SELAVPR_PLUS_CHECKPOINT,
        )


def test_safe_checkpoint_loader_is_weights_only_cpu_and_strips_known_prefix() -> None:
    calls: list[dict[str, object]] = []
    sentinel = object()

    class _Torch:
        @staticmethod
        def load(path: Path, **kwargs: object) -> object:
            calls.append({"path": path, **kwargs})
            return {"model_state_dict": {"module.weight": sentinel}}

    path = Path("/tmp/checkpoint.pth")
    state = selavpr_module._safe_checkpoint_state(_Torch, path)

    assert calls == [
        {
            "path": path,
            "map_location": "cpu",
            "weights_only": True,
        }
    ]
    assert state == {"weight": sentinel}


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"model_state_dict": {}},
        {"model_state_dict": {"module.a": object(), "b": object()}},
        {"model_state_dict": {1: object()}},
    ],
)
def test_safe_checkpoint_loader_rejects_unexpected_payloads(payload: object) -> None:
    class _Torch:
        @staticmethod
        def load(path: Path, **kwargs: object) -> object:
            del path, kwargs
            return payload

    with pytest.raises(SelaVprPlusRetrievalError):
        selavpr_module._safe_checkpoint_state(_Torch, Path("/tmp/checkpoint.pth"))


def test_safe_checkpoint_loader_rejects_runtime_without_weights_only() -> None:
    class _Torch:
        @staticmethod
        def load(path: Path, **kwargs: object) -> object:
            del path, kwargs
            raise TypeError("weights_only unsupported")

    with pytest.raises(SelaVprPlusRetrievalError, match="weights-only"):
        selavpr_module._safe_checkpoint_state(_Torch, Path("/tmp/checkpoint.pth"))


def test_verified_rgb_input_uses_only_declared_level_zero_bytes(tmp_path: Path) -> None:
    item = _decoded_input(
        tmp_path,
        "obs:a",
        rgb8=b"\x00\x7f\xff\xff\x7f\x00",
    )

    image = selavpr_module._verified_rgb_image(item)

    assert image.observation_id == ObservationId("obs:a")
    assert (image.width_px, image.height_px) == (2, 1)
    assert image.rgb8 == b"\x00\x7f\xff\xff\x7f\x00"


def test_corrupted_or_wrong_length_decoded_materialization_fails_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    _allow_fixture_checkpoint(monkeypatch, request.checkpoint_path)
    runtime = _FakeRuntime(
        (
            _unit_descriptor(0),
            _unit_descriptor(1),
            _unit_descriptor(2),
        )
    )
    level_path = (
        request.inputs[0].materialization_root
        / request.inputs[0].decoded_result.manifest.levels[0].relative_path
    )
    level_path.write_bytes(b"corrupted")

    with pytest.raises(SelaVprPlusRetrievalError, match="not verified"):
        retrieve_selavpr_plus_pairs(request, runtime=runtime)

    assert runtime.calls == []

    short_input = _decoded_input(
        tmp_path / "short",
        "obs:a",
        rgb8=b"\x00\x01\x02",
    )
    other = _decoded_input(tmp_path / "short", "obs:b")
    short_inputs = (short_input, other)
    short_config = SelaVprPlusRetrievalConfig(neighbors_per_observation=1)
    short_request = SelaVprPlusRetrievalRequest(
        run=_run(short_inputs, short_config),
        inputs=short_inputs,
        source_root=_source_root(tmp_path / "short"),
        checkpoint_path=_checkpoint_path(tmp_path / "short"),
        config=short_config,
    )
    _allow_fixture_checkpoint(monkeypatch, short_request.checkpoint_path)

    with pytest.raises(SelaVprPlusRetrievalError, match="invalid RGB8 byte length"):
        retrieve_selavpr_plus_pairs(short_request, runtime=runtime)


def test_wrong_pixel_contract_fails_before_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    _allow_fixture_checkpoint(monkeypatch, request.checkpoint_path)
    runtime = _FakeRuntime(
        (
            _unit_descriptor(0),
            _unit_descriptor(1),
            _unit_descriptor(2),
        )
    )
    manifest = request.inputs[0].decoded_result.manifest
    object.__setattr__(manifest, "pixel_layout", cast(Any, "rgb16"))

    with pytest.raises(SelaVprPlusRetrievalError, match="packed RGB8"):
        retrieve_selavpr_plus_pairs(request, runtime=runtime)

    assert runtime.calls == []


def test_exact_cosine_top_k_is_deterministic_bounded_and_order_independent() -> None:
    ids = (
        ObservationId("obs:a"),
        ObservationId("obs:b"),
        ObservationId("obs:c"),
    )
    descriptors = (
        _unit_descriptor(0),
        _unit_descriptor(0),
        _unit_descriptor(1),
    )

    result = propose_selavpr_plus_pairs(
        ids,
        descriptors,
        neighbors_per_observation=1,
    )
    shuffled = propose_selavpr_plus_pairs(
        (ids[2], ids[0], ids[1]),
        (descriptors[2], descriptors[0], descriptors[1]),
        neighbors_per_observation=1,
    )

    assert result == shuffled
    assert tuple(
        (pair.observation_id1.value, pair.observation_id2.value, pair.cosine_similarity)
        for pair in result
    ) == (
        ("obs:a", "obs:b", 1.0),
        ("obs:a", "obs:c", 0.0),
    )


def test_pair_proposal_rejects_invalid_descriptors_and_neighbors() -> None:
    ids = (ObservationId("obs:a"), ObservationId("obs:b"))

    with pytest.raises(SelaVprPlusRetrievalError, match="L2-normalized"):
        propose_selavpr_plus_pairs(
            ids,
            (tuple(0.5 for _ in range(2)), tuple(0.5 for _ in range(2))),
            neighbors_per_observation=1,
        )
    with pytest.raises(SelaVprPlusRetrievalError, match="finite"):
        propose_selavpr_plus_pairs(
            ids,
            ((1.0, 0.0), (math.nan, 0.0)),
            neighbors_per_observation=1,
        )
    with pytest.raises(ValueError, match="positive integer"):
        propose_selavpr_plus_pairs(
            ids,
            ((1.0, 0.0), (0.0, 1.0)),
            neighbors_per_observation=0,
        )


def test_fake_runtime_retrieval_preserves_exact_evidence_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    _allow_fixture_checkpoint(monkeypatch, request.checkpoint_path)
    runtime = _FakeRuntime(
        (
            _unit_descriptor(0),
            _unit_descriptor(0),
            _unit_descriptor(1),
        )
    )

    result = retrieve_selavpr_plus_pairs(request, runtime=runtime)

    assert len(runtime.calls) == 1
    call = runtime.calls[0]
    images = cast(tuple[Any, ...], call["images"])
    assert tuple(image.observation_id for image in images) == request.run.input_observation_ids
    assert tuple(image.rgb8 for image in images) == (
        b"\x00\x7f\xff\xff\x7f\x00",
        b"\x00\x7f\xff\xff\x7f\x00",
        b"\x00\x7f\xff\xff\x7f\x00",
    )
    assert result.provenance.source_observation_ids == request.run.input_observation_ids
    assert result.model == SELAVPR_PLUS_MODEL
    assert result.checkpoint == SELAVPR_PLUS_CHECKPOINT
    assert result.configuration_sha256 == request.config.sha256
    assert result.environment == _environment()
    assert tuple(
        (pair.observation_id1.value, pair.observation_id2.value) for pair in result.pairs
    ) == (("obs:a", "obs:b"), ("obs:a", "obs:c"))


@pytest.mark.parametrize(
    "descriptors",
    [
        (_unit_descriptor(0),),
        (
            tuple(0.0 for _ in range(2048)),
            _unit_descriptor(1),
            _unit_descriptor(2),
        ),
        (
            tuple(math.nan if index == 0 else 0.0 for index in range(2048)),
            _unit_descriptor(1),
            _unit_descriptor(2),
        ),
        (
            tuple(1.0 if index == 0 else 0.0 for index in range(2047)),
            _unit_descriptor(1),
            _unit_descriptor(2),
        ),
    ],
)
def test_runtime_descriptor_failures_are_fail_closed_before_pair_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    descriptors: tuple[tuple[float, ...], ...],
) -> None:
    request = _request(tmp_path)
    _allow_fixture_checkpoint(monkeypatch, request.checkpoint_path)

    with pytest.raises(SelaVprPlusRetrievalError):
        retrieve_selavpr_plus_pairs(request, runtime=_FakeRuntime(descriptors))


def test_result_contract_is_source_specific_and_rejects_foreign_pairs() -> None:
    provenance = selavpr_module.DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId("run:result"),
        source_observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
    )
    pair = SelaVprPlusPair(
        observation_id1=ObservationId("obs:a"),
        observation_id2=ObservationId("obs:b"),
        cosine_similarity=0.25,
    )
    result = SelaVprPlusRetrievalResult(
        provenance=provenance,
        model=SELAVPR_PLUS_MODEL,
        checkpoint=SELAVPR_PLUS_CHECKPOINT,
        configuration_sha256=Sha256Digest("d" * 64),
        environment=_environment(),
        pairs=(pair,),
    )

    assert tuple(field.name for field in fields(SelaVprPlusPair)) == (
        "observation_id1",
        "observation_id2",
        "cosine_similarity",
    )
    assert tuple(field.name for field in fields(SelaVprPlusRetrievalResult)) == (
        "provenance",
        "model",
        "checkpoint",
        "configuration_sha256",
        "environment",
        "pairs",
    )
    assert result.pairs == (pair,)
    for forbidden in (
        "scene_identity",
        "geolocation",
        "capture_time",
        "geometry",
        "route",
        "rank",
    ):
        assert forbidden not in SelaVprPlusPair.__dataclass_fields__

    with pytest.raises(ValueError, match="provenance observations"):
        SelaVprPlusRetrievalResult(
            provenance=provenance,
            model=SELAVPR_PLUS_MODEL,
            checkpoint=SELAVPR_PLUS_CHECKPOINT,
            configuration_sha256=Sha256Digest("d" * 64),
            environment=_environment(),
            pairs=(
                SelaVprPlusPair(
                    observation_id1=ObservationId("obs:a"),
                    observation_id2=ObservationId("obs:c"),
                    cosine_similarity=0.0,
                ),
            ),
        )


def test_pair_candidate_adaptation_is_pure_and_does_not_leak_similarity() -> None:
    provenance = selavpr_module.DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId("run:adapter"),
        source_observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
    )
    result = SelaVprPlusRetrievalResult(
        provenance=provenance,
        model=SELAVPR_PLUS_MODEL,
        checkpoint=SELAVPR_PLUS_CHECKPOINT,
        configuration_sha256=Sha256Digest("e" * 64),
        environment=_environment(),
        pairs=(
            SelaVprPlusPair(
                observation_id1=ObservationId("obs:a"),
                observation_id2=ObservationId("obs:b"),
                cosine_similarity=0.75,
            ),
        ),
    )
    first_ref = ArtifactRef(
        artifact_id=ArtifactId("artifact:selavpr-first"),
        artifact_kind=ArtifactKind("retrieval.learned"),
    )
    second_ref = ArtifactRef(
        artifact_id=ArtifactId("artifact:selavpr-second"),
        artifact_kind=ArtifactKind("retrieval.learned"),
    )

    first = adapt_selavpr_plus_retrieval_result(
        SelaVprPlusPairCandidateAdapterInput(
            result=result,
            evidence_ref=first_ref,
        )
    )
    repeated = adapt_selavpr_plus_retrieval_result(
        SelaVprPlusPairCandidateAdapterInput(
            result=result,
            evidence_ref=first_ref,
        )
    )
    changed = adapt_selavpr_plus_retrieval_result(
        SelaVprPlusPairCandidateAdapterInput(
            result=result,
            evidence_ref=second_ref,
        )
    )

    assert SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID == PairCandidateSourceId("selavpr_plus")
    assert first == repeated
    assert len(first) == 1
    assert first[0].sources[0].evidence_refs == (first_ref,)
    assert changed[0].sources[0].evidence_refs == (second_ref,)
    assert tuple(field.name for field in fields(PairCandidate)) == (
        "observation_id1",
        "observation_id2",
        "sources",
    )
    assert "cosine_similarity" not in PairCandidate.__dataclass_fields__
    assert "checkpoint" not in PairCandidate.__dataclass_fields__
    assert "model" not in PairCandidate.__dataclass_fields__


def test_core_module_does_not_import_or_download_optional_ml_runtime() -> None:
    for forbidden in (
        "torch",
        "torchvision",
        "faiss",
        "numpy",
        "requests",
        "urllib",
        "huggingface_hub",
    ):
        assert forbidden not in selavpr_module.__dict__

    source = Path(selavpr_module.__file__).read_text(encoding="utf-8")
    assert "torch.hub" not in source
    assert "load_state_dict_from_url" not in source
    assert "huggingface_hub" not in source
    assert "requests.get" not in source
    assert "urllib.request" not in source
    assert "subprocess" not in source
    assert "git clone" not in source
