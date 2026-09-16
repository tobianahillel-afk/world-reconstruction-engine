from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from wre.domain.adapter_capabilities import AdapterCapabilityDescriptor, AdapterCapabilityName
from wre.domain.artifacts import ArtifactKind
from wre.domain.observations import Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef


def _kind(value: str) -> ArtifactKind:
    return ArtifactKind(value)


def test_adapter_capability_name_is_open_immutable_and_deterministic() -> None:
    first = AdapterCapabilityName("geometry.sfm")
    second = AdapterCapabilityName("geometry.sfm")
    specialist = AdapterCapabilityName("appearance.gaussian-refine")

    assert str(first) == "geometry.sfm"
    assert first == second
    assert hash(first) == hash(second)
    assert specialist != first
    with pytest.raises(FrozenInstanceError):
        first.value = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "UPPERCASE",
        " leading",
        "trailing ",
        ".leading-dot",
        "contains/slash",
        "contains space",
        "a" * 129,
    ],
)
def test_adapter_capability_name_rejects_invalid_tokens(value: str) -> None:
    with pytest.raises(ValueError, match="adapter_capability_name"):
        AdapterCapabilityName(value)


def test_descriptor_supports_single_and_multiple_semantic_kinds() -> None:
    descriptor = AdapterCapabilityDescriptor(
        capability=AdapterCapabilityName("geometry.sfm"),
        input_kinds=frozenset({_kind("media.image"), _kind("feature.matches")}),
        output_kinds=frozenset({_kind("camera.solution"), _kind("geometry.solution")}),
    )

    assert descriptor.input_kinds == frozenset({_kind("media.image"), _kind("feature.matches")})
    assert descriptor.output_kinds == frozenset(
        {_kind("camera.solution"), _kind("geometry.solution")}
    )


def test_descriptor_set_semantics_are_order_independent_and_hashable() -> None:
    first = AdapterCapabilityDescriptor(
        capability=AdapterCapabilityName("retrieval.visual"),
        input_kinds=frozenset((_kind("media.image"), _kind("media.profile"))),
        output_kinds=frozenset((_kind("retrieval.index"), _kind("pair.candidate"))),
    )
    second = AdapterCapabilityDescriptor(
        capability=AdapterCapabilityName("retrieval.visual"),
        input_kinds=frozenset((_kind("media.profile"), _kind("media.image"))),
        output_kinds=frozenset((_kind("pair.candidate"), _kind("retrieval.index"))),
    )

    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1


def test_descriptor_allows_explicit_empty_input_set() -> None:
    descriptor = AdapterCapabilityDescriptor(
        capability=AdapterCapabilityName("source.bootstrap"),
        input_kinds=frozenset(),
        output_kinds=frozenset({_kind("media.image")}),
    )

    assert descriptor.input_kinds == frozenset()


def test_descriptor_allows_overlapping_input_and_output_kinds() -> None:
    geometry = _kind("geometry.solution")
    descriptor = AdapterCapabilityDescriptor(
        capability=AdapterCapabilityName("geometry.refine"),
        input_kinds=frozenset({geometry}),
        output_kinds=frozenset({geometry}),
    )

    assert descriptor.input_kinds == descriptor.output_kinds == frozenset({geometry})


def test_descriptor_rejects_empty_output_set() -> None:
    with pytest.raises(ValueError, match="output_kinds must not be empty"):
        AdapterCapabilityDescriptor(
            capability=AdapterCapabilityName("invalid.no-output"),
            input_kinds=frozenset(),
            output_kinds=frozenset(),
        )


def test_descriptor_rejects_untyped_capability_and_mutable_collections() -> None:
    capability = AdapterCapabilityName("geometry.sfm")
    image = _kind("media.image")
    geometry = _kind("geometry.solution")

    with pytest.raises(TypeError, match="capability must be AdapterCapabilityName"):
        AdapterCapabilityDescriptor(
            capability=cast(Any, "geometry.sfm"),
            input_kinds=frozenset({image}),
            output_kinds=frozenset({geometry}),
        )
    with pytest.raises(TypeError, match="input_kinds must be an immutable frozenset"):
        AdapterCapabilityDescriptor(
            capability=capability,
            input_kinds=cast(Any, {image}),
            output_kinds=frozenset({geometry}),
        )
    with pytest.raises(TypeError, match="output_kinds must be an immutable frozenset"):
        AdapterCapabilityDescriptor(
            capability=capability,
            input_kinds=frozenset({image}),
            output_kinds=cast(Any, [geometry]),
        )


def test_descriptor_rejects_raw_string_and_wrong_kind_members() -> None:
    capability = AdapterCapabilityName("geometry.sfm")

    with pytest.raises(TypeError, match="input_kinds members must be ArtifactKind"):
        AdapterCapabilityDescriptor(
            capability=capability,
            input_kinds=cast(Any, frozenset({"media.image"})),
            output_kinds=frozenset({_kind("geometry.solution")}),
        )
    with pytest.raises(TypeError, match="output_kinds members must be ArtifactKind"):
        AdapterCapabilityDescriptor(
            capability=capability,
            input_kinds=frozenset({_kind("media.image")}),
            output_kinds=cast(Any, frozenset({1})),
        )


def test_descriptor_is_immutable() -> None:
    descriptor = AdapterCapabilityDescriptor(
        capability=AdapterCapabilityName("geometry.sfm"),
        input_kinds=frozenset({_kind("media.image")}),
        output_kinds=frozenset({_kind("geometry.solution")}),
    )

    with pytest.raises(FrozenInstanceError):
        descriptor.capability = AdapterCapabilityName("other")  # type: ignore[misc]


def test_existing_artifact_and_producer_identity_semantics_remain_independent() -> None:
    artifact_kind = ArtifactKind("geometry.solution")
    producer = ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="wre.reconstruction.colmap",
            version="4.2.0",
            revision="adapter:1",
        ),
        configuration=ConfigurationIdentity(sha256=Sha256Digest("a" * 64)),
    )
    descriptor = AdapterCapabilityDescriptor(
        capability=AdapterCapabilityName("geometry.sfm"),
        input_kinds=frozenset({_kind("media.image")}),
        output_kinds=frozenset({artifact_kind}),
    )

    assert str(artifact_kind) == "geometry.solution"
    assert producer.producer.implementation == "wre.reconstruction.colmap"
    assert descriptor.capability != cast(Any, artifact_kind)
    assert not hasattr(descriptor, "producer")
    assert not hasattr(descriptor, "model")
    assert not hasattr(descriptor, "hardware")
