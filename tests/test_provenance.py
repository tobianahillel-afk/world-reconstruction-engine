from __future__ import annotations

import pytest

from wre.domain import (
    DerivedArtifactProvenance,
    ObservationId,
    ProvenanceClass,
    ReconstructionRunId,
)


def test_provenance_class_has_exact_closed_member_set() -> None:
    assert list(ProvenanceClass) == [
        ProvenanceClass.OBSERVED_RECONSTRUCTED,
        ProvenanceClass.INFERRED,
        ProvenanceClass.GENERATED,
    ]
    assert [member.name for member in ProvenanceClass] == [
        "OBSERVED_RECONSTRUCTED",
        "INFERRED",
        "GENERATED",
    ]
    assert [member.value for member in ProvenanceClass] == [
        "observed_reconstructed",
        "inferred",
        "generated",
    ]


def test_provenance_class_constructs_from_stable_wire_tokens() -> None:
    assert ProvenanceClass("observed_reconstructed") is ProvenanceClass.OBSERVED_RECONSTRUCTED
    assert ProvenanceClass("inferred") is ProvenanceClass.INFERRED
    assert ProvenanceClass("generated") is ProvenanceClass.GENERATED
    assert str(ProvenanceClass.GENERATED) == "generated"
    assert hash(ProvenanceClass.INFERRED) == hash(ProvenanceClass("inferred"))


@pytest.mark.parametrize("value", ["unknown", "mixed", "verified", "synthetic", "solver_specific"])
def test_provenance_class_rejects_unknown_or_speculative_values(value: str) -> None:
    with pytest.raises(ValueError):
        ProvenanceClass(value)


def test_retained_derived_artifact_provenance_lineage_is_unchanged() -> None:
    provenance = DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId("run:provenance-regression"),
        source_observation_ids=(ObservationId("obs:b"), ObservationId("obs:a")),
    )

    assert provenance.producing_run_id == ReconstructionRunId("run:provenance-regression")
    assert provenance.source_observation_ids == (ObservationId("obs:a"), ObservationId("obs:b"))
