from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from wre.domain import (
    ArtifactInputFingerprint,
    ArtifactKey,
    ArtifactKeyMaterial,
    ArtifactKind,
    ArtifactProducerIdentity,
    CheckpointIdentity,
    ConfigurationIdentity,
    ModelIdentity,
    ProducerRef,
    Sha256Digest,
    canonical_artifact_key_bytes,
    derive_artifact_key,
)


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _learned_producer(
    *,
    implementation: str = "wre.adapters.example",
    producer_version: str = "1.2.3",
    producer_revision: str | None = "commit:abc",
    configuration_character: str = "3",
    model_name: str = "example.model",
    model_version: str = "4.5",
    model_revision: str | None = "commit:def",
    checkpoint_identifier: str = "weights:quality",
    checkpoint_character: str = "4",
) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=implementation,
            version=producer_version,
            revision=producer_revision,
        ),
        configuration=ConfigurationIdentity(
            sha256=_digest(configuration_character),
        ),
        model=ModelIdentity(
            name=model_name,
            version=model_version,
            revision=model_revision,
        ),
        checkpoint=CheckpointIdentity(
            identifier=checkpoint_identifier,
            sha256=_digest(checkpoint_character),
        ),
    )


def _input_fingerprints() -> tuple[ArtifactInputFingerprint, ...]:
    return (
        ArtifactInputFingerprint(
            artifact_kind=ArtifactKind("media.image"),
            sha256=_digest("1"),
        ),
        ArtifactInputFingerprint(
            artifact_kind=ArtifactKind("feature.matches"),
            sha256=_digest("2"),
        ),
    )


def _material(
    *,
    output_kind: str = "geometry.solution",
    input_fingerprints: tuple[ArtifactInputFingerprint, ...] | None = None,
    producer: ArtifactProducerIdentity | None = None,
) -> ArtifactKeyMaterial:
    return ArtifactKeyMaterial(
        output_kind=ArtifactKind(output_kind),
        input_fingerprints=(
            _input_fingerprints() if input_fingerprints is None else input_fingerprints
        ),
        producer=_learned_producer() if producer is None else producer,
    )


def test_canonical_artifact_key_has_stable_golden_vector() -> None:
    material = _material()
    expected = (
        '{"checkpoint":{"identifier":"weights:quality","sha256":"'
        + "4" * 64
        + '"},"configuration_sha256":"'
        + "3" * 64
        + '","domain":"wre.artifact-key","inputs":'
        + '[{"artifact_kind":"media.image","sha256":"'
        + "1" * 64
        + '"},{"artifact_kind":"feature.matches","sha256":"'
        + "2" * 64
        + '"}],"model":{"name":"example.model","revision":"commit:def",'
        + '"version":"4.5"},"output_kind":"geometry.solution","producer":'
        + '{"implementation":"wre.adapters.example","revision":"commit:abc",'
        + '"version":"1.2.3"},"schema_version":1}'
    )

    canonical = canonical_artifact_key_bytes(material)
    key = derive_artifact_key(material)

    assert canonical == expected.encode("utf-8")
    assert key == ArtifactKey(
        sha256=Sha256Digest(
            "553de78370551d9ebbd843bad4f7f0b286bc4f5c18bf339aa6ac04c11a384111"
        )
    )
    assert str(key) == (
        "artifact-key:v1:"
        "553de78370551d9ebbd843bad4f7f0b286bc4f5c18bf339aa6ac04c11a384111"
    )


def test_key_material_excludes_logical_and_epistemic_metadata() -> None:
    canonical = canonical_artifact_key_bytes(_material())

    assert b"artifact_id" not in canonical
    assert b"scene_project" not in canonical
    assert b"provenance" not in canonical
    assert b"timestamp" not in canonical
    assert b"path" not in canonical


def test_native_producer_has_explicit_null_model_and_checkpoint() -> None:
    producer = ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="wre.reconstruction.colmap",
            version="4.2.0",
            revision="adapter:1",
        ),
        configuration=ConfigurationIdentity(sha256=_digest("a")),
    )
    canonical = canonical_artifact_key_bytes(_material(producer=producer))

    assert b'"model":null' in canonical
    assert b'"checkpoint":null' in canonical
    assert derive_artifact_key(_material(producer=producer)) == derive_artifact_key(
        _material(producer=producer)
    )


def test_every_owned_computation_identity_field_changes_the_key() -> None:
    base = _material()
    inputs = _input_fingerprints()
    first, second = inputs
    variants = [
        _material(output_kind="surface.model"),
        _material(
            input_fingerprints=(
                ArtifactInputFingerprint(
                    artifact_kind=ArtifactKind("media.frame"),
                    sha256=first.sha256,
                ),
                second,
            )
        ),
        _material(
            input_fingerprints=(
                ArtifactInputFingerprint(
                    artifact_kind=first.artifact_kind,
                    sha256=_digest("5"),
                ),
                second,
            )
        ),
        _material(input_fingerprints=(second, first)),
        _material(producer=_learned_producer(implementation="wre.adapters.other")),
        _material(producer=_learned_producer(producer_version="1.2.4")),
        _material(producer=_learned_producer(producer_revision="commit:xyz")),
        _material(producer=_learned_producer(configuration_character="6")),
        _material(producer=_learned_producer(model_name="other.model")),
        _material(producer=_learned_producer(model_version="4.6")),
        _material(producer=_learned_producer(model_revision="commit:ghi")),
        _material(producer=_learned_producer(checkpoint_identifier="weights:other")),
        _material(producer=_learned_producer(checkpoint_character="7")),
    ]

    base_key = derive_artifact_key(base)
    variant_keys = [derive_artifact_key(variant) for variant in variants]

    assert all(key != base_key for key in variant_keys)
    assert len(set(variant_keys)) == len(variant_keys)


def test_input_order_and_duplicates_are_preserved_without_hidden_normalization() -> None:
    first, second = _input_fingerprints()
    ordered = _material(input_fingerprints=(first, second))
    reversed_order = _material(input_fingerprints=(second, first))
    duplicate = _material(input_fingerprints=(first, first))
    single = _material(input_fingerprints=(first,))

    assert ordered.input_fingerprints == (first, second)
    assert duplicate.input_fingerprints == (first, first)
    assert derive_artifact_key(ordered) != derive_artifact_key(reversed_order)
    assert derive_artifact_key(duplicate) != derive_artifact_key(single)


def test_artifact_key_contracts_reject_untyped_members() -> None:
    with pytest.raises(TypeError, match=r"artifact_input\.artifact_kind"):
        ArtifactInputFingerprint(
            artifact_kind=cast(Any, "media.image"),
            sha256=_digest("1"),
        )
    with pytest.raises(TypeError, match=r"artifact_input\.sha256"):
        ArtifactInputFingerprint(
            artifact_kind=ArtifactKind("media.image"),
            sha256=cast(Any, "1" * 64),
        )
    with pytest.raises(TypeError, match=r"artifact_key_material\.output_kind"):
        ArtifactKeyMaterial(
            output_kind=cast(Any, "geometry.solution"),
            input_fingerprints=_input_fingerprints(),
            producer=_learned_producer(),
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        ArtifactKeyMaterial(
            output_kind=ArtifactKind("geometry.solution"),
            input_fingerprints=cast(Any, list(_input_fingerprints())),
            producer=_learned_producer(),
        )
    with pytest.raises(TypeError, match="ArtifactInputFingerprint values"):
        ArtifactKeyMaterial(
            output_kind=ArtifactKind("geometry.solution"),
            input_fingerprints=cast(Any, ("raw",)),
            producer=_learned_producer(),
        )
    with pytest.raises(TypeError, match=r"artifact_key_material\.producer"):
        ArtifactKeyMaterial(
            output_kind=ArtifactKind("geometry.solution"),
            input_fingerprints=_input_fingerprints(),
            producer=cast(Any, "producer"),
        )
    with pytest.raises(TypeError, match=r"artifact_key\.sha256"):
        ArtifactKey(sha256=cast(Any, "8" * 64))
    with pytest.raises(TypeError, match="material must be ArtifactKeyMaterial"):
        canonical_artifact_key_bytes(cast(Any, {}))


def test_artifact_key_contracts_are_immutable_and_distinct_from_raw_digest() -> None:
    fingerprint = _input_fingerprints()[0]
    material = _material()
    key = derive_artifact_key(material)

    with pytest.raises(FrozenInstanceError):
        fingerprint.sha256 = _digest("9")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        material.output_kind = ArtifactKind("surface.model")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        key.sha256 = _digest("0")  # type: ignore[misc]

    assert key != key.sha256
    assert hash(key) == hash(ArtifactKey(sha256=key.sha256))
