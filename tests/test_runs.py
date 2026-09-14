from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from wre.domain import (
    DerivedArtifactProvenance,
    ObservationId,
    ProducerRef,
    ReconstructionRun,
    ReconstructionRunId,
    Sha256Digest,
    SpatialFragmentId,
)

START = datetime(2026, 9, 14, 19, 0, tzinfo=UTC)


def _observation(value: str) -> ObservationId:
    return ObservationId(f"obs:{value}")


def _producer() -> ProducerRef:
    return ProducerRef(
        implementation="wre.reconstruction.colmap",
        version="4.2.0",
        revision="adapter:1",
    )


def test_reconstruction_run_id_is_a_distinct_typed_domain() -> None:
    assert ReconstructionRunId("shared:id") != ObservationId("shared:id")
    assert ReconstructionRunId("shared:id") != SpatialFragmentId("shared:id")

    with pytest.raises(ValueError, match="reconstruction_run_id"):
        ReconstructionRunId("bad run id")


def test_producer_requires_explicit_implementation_and_version() -> None:
    producer = _producer()
    assert producer.implementation == "wre.reconstruction.colmap"
    assert producer.version == "4.2.0"
    assert producer.revision == "adapter:1"

    with pytest.raises(ValueError, match=r"producer\.implementation"):
        ProducerRef(implementation=" ", version="1")
    with pytest.raises(ValueError, match=r"producer\.version"):
        ProducerRef(implementation="wre.test", version=" ")
    with pytest.raises(ValueError, match=r"producer\.revision"):
        ProducerRef(implementation="wre.test", version="1", revision=" ")


def test_reconstruction_run_canonicalizes_inputs_and_keeps_exact_producer() -> None:
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:001"),
        producer=_producer(),
        input_observation_ids=(_observation("c"), _observation("a"), _observation("b")),
        started_at=START,
        completed_at=START + timedelta(seconds=12),
        configuration_sha256=Sha256Digest("A" * 64),
    )

    assert run.input_observation_ids == (
        _observation("a"),
        _observation("b"),
        _observation("c"),
    )
    assert run.producer == _producer()
    assert run.configuration_sha256 == Sha256Digest("a" * 64)


def test_reconstruction_run_rejects_invalid_input_membership() -> None:
    with pytest.raises(ValueError, match="at least one observation"):
        ReconstructionRun(
            run_id=ReconstructionRunId("run:empty"),
            producer=_producer(),
            input_observation_ids=(),
            started_at=START,
        )

    duplicate = _observation("same")
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        ReconstructionRun(
            run_id=ReconstructionRunId("run:duplicate"),
            producer=_producer(),
            input_observation_ids=(duplicate, duplicate),
            started_at=START,
        )

    invalid_inputs = cast(tuple[ObservationId, ...], [_observation("list")])
    with pytest.raises(ValueError, match="immutable tuple"):
        ReconstructionRun(
            run_id=ReconstructionRunId("run:list"),
            producer=_producer(),
            input_observation_ids=invalid_inputs,
            started_at=START,
        )


def test_reconstruction_run_requires_aware_monotonic_times() -> None:
    naive = datetime(2026, 9, 14, 19, 0)

    with pytest.raises(ValueError, match="started_at must be timezone-aware"):
        ReconstructionRun(
            run_id=ReconstructionRunId("run:naive-start"),
            producer=_producer(),
            input_observation_ids=(_observation("a"),),
            started_at=naive,
        )

    with pytest.raises(ValueError, match="completed_at must be timezone-aware"):
        ReconstructionRun(
            run_id=ReconstructionRunId("run:naive-end"),
            producer=_producer(),
            input_observation_ids=(_observation("a"),),
            started_at=START,
            completed_at=naive,
        )

    with pytest.raises(ValueError, match="earlier than started_at"):
        ReconstructionRun(
            run_id=ReconstructionRunId("run:backwards"),
            producer=_producer(),
            input_observation_ids=(_observation("a"),),
            started_at=START,
            completed_at=START - timedelta(microseconds=1),
        )


def test_reconstruction_run_rejects_untyped_configuration_digest() -> None:
    invalid_digest = cast(Sha256Digest, "not-a-digest")

    with pytest.raises(ValueError, match="configuration_sha256"):
        ReconstructionRun(
            run_id=ReconstructionRunId("run:bad-config"),
            producer=_producer(),
            input_observation_ids=(_observation("a"),),
            started_at=START,
            configuration_sha256=invalid_digest,
        )


def test_derived_artifact_provenance_is_deterministic_and_non_empty() -> None:
    provenance = DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId("run:001"),
        source_observation_ids=(_observation("2"), _observation("1")),
    )

    assert provenance.producing_run_id == ReconstructionRunId("run:001")
    assert provenance.source_observation_ids == (_observation("1"), _observation("2"))

    with pytest.raises(ValueError, match="at least one observation"):
        DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:empty-provenance"),
            source_observation_ids=(),
        )

    duplicate = _observation("duplicate")
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:duplicate-provenance"),
            source_observation_ids=(duplicate, duplicate),
        )
