from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from wre.domain import (
    ArtifactProducerIdentity,
    CheckpointIdentity,
    ConfigurationIdentity,
    ModelIdentity,
    ObservationId,
    ProducerRef,
    ReconstructionRun,
    ReconstructionRunId,
    Sha256Digest,
)


def _producer() -> ProducerRef:
    return ProducerRef(
        implementation="wre.reconstruction.colmap",
        version="4.2.0",
        revision="adapter:1",
    )


def _configuration() -> ConfigurationIdentity:
    return ConfigurationIdentity(sha256=Sha256Digest("a" * 64))


def test_native_artifact_producer_reuses_retained_producer_ref() -> None:
    identity = ArtifactProducerIdentity(
        producer=_producer(),
        configuration=_configuration(),
    )

    assert identity.producer == _producer()
    assert identity.configuration == _configuration()
    assert identity.model is None
    assert identity.checkpoint is None
    assert hash(identity) == hash(
        ArtifactProducerIdentity(
            producer=_producer(),
            configuration=_configuration(),
        )
    )


def test_learned_artifact_producer_has_exact_model_and_checkpoint_identity() -> None:
    model = ModelIdentity(name="example.model", version="2.1", revision="commit:abc123")
    checkpoint = CheckpointIdentity(
        identifier="weights:quality-v2",
        sha256=Sha256Digest("b" * 64),
    )
    identity = ArtifactProducerIdentity(
        producer=ProducerRef(implementation="wre.adapters.example", version="1.4.0"),
        configuration=_configuration(),
        model=model,
        checkpoint=checkpoint,
    )

    assert identity.model == model
    assert identity.checkpoint == checkpoint
    assert identity == ArtifactProducerIdentity(
        producer=ProducerRef(implementation="wre.adapters.example", version="1.4.0"),
        configuration=ConfigurationIdentity(sha256=Sha256Digest("a" * 64)),
        model=ModelIdentity(name="example.model", version="2.1", revision="commit:abc123"),
        checkpoint=CheckpointIdentity(
            identifier="weights:quality-v2",
            sha256=Sha256Digest("b" * 64),
        ),
    )


def test_model_identity_rejects_blank_fields() -> None:
    with pytest.raises(ValueError, match=r"model\.name"):
        ModelIdentity(name=" ", version="1")
    with pytest.raises(ValueError, match=r"model\.version"):
        ModelIdentity(name="model", version=" ")
    with pytest.raises(ValueError, match=r"model\.revision"):
        ModelIdentity(name="model", version="1", revision=" ")


def test_checkpoint_and_configuration_require_typed_digests() -> None:
    with pytest.raises(TypeError, match=r"checkpoint\.sha256"):
        CheckpointIdentity(
            identifier="weights:1",
            sha256=cast(Any, "not-a-digest"),
        )
    with pytest.raises(TypeError, match=r"configuration\.sha256"):
        ConfigurationIdentity(sha256=cast(Any, "not-a-digest"))
    with pytest.raises(ValueError, match=r"checkpoint\.identifier"):
        CheckpointIdentity(identifier=" ", sha256=Sha256Digest("c" * 64))


def test_checkpoint_without_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="checkpoint requires model identity"):
        ArtifactProducerIdentity(
            producer=_producer(),
            configuration=_configuration(),
            checkpoint=CheckpointIdentity(
                identifier="weights:1",
                sha256=Sha256Digest("d" * 64),
            ),
        )


def test_artifact_producer_rejects_untyped_members() -> None:
    with pytest.raises(TypeError, match=r"artifact_producer\.producer"):
        ArtifactProducerIdentity(
            producer=cast(Any, "wre.test"),
            configuration=_configuration(),
        )
    with pytest.raises(TypeError, match=r"artifact_producer\.configuration"):
        ArtifactProducerIdentity(
            producer=_producer(),
            configuration=cast(Any, Sha256Digest("e" * 64)),
        )
    with pytest.raises(TypeError, match=r"artifact_producer\.model"):
        ArtifactProducerIdentity(
            producer=_producer(),
            configuration=_configuration(),
            model=cast(Any, "model"),
        )
    with pytest.raises(TypeError, match=r"artifact_producer\.checkpoint"):
        ArtifactProducerIdentity(
            producer=_producer(),
            configuration=_configuration(),
            model=ModelIdentity(name="model", version="1"),
            checkpoint=cast(Any, "weights"),
        )


def test_producer_identity_objects_are_immutable() -> None:
    model = ModelIdentity(name="model", version="1")
    checkpoint = CheckpointIdentity(
        identifier="weights:1",
        sha256=Sha256Digest("f" * 64),
    )
    configuration = _configuration()
    identity = ArtifactProducerIdentity(
        producer=_producer(),
        configuration=configuration,
        model=model,
        checkpoint=checkpoint,
    )

    with pytest.raises(FrozenInstanceError):
        model.version = "2"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        checkpoint.identifier = "weights:2"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        configuration.sha256 = Sha256Digest("0" * 64)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        identity.model = None  # type: ignore[misc]


def test_retained_reconstruction_run_behavior_is_unchanged() -> None:
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:producer-regression"),
        producer=_producer(),
        input_observation_ids=(ObservationId("obs:one"),),
        started_at=datetime(2026, 9, 15, 21, 0, tzinfo=UTC),
        configuration_sha256=Sha256Digest("1" * 64),
    )

    assert run.producer == _producer()
    assert run.configuration_sha256 == Sha256Digest("1" * 64)
