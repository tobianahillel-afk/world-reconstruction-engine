from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.routing.inputs as router_inputs_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    FailureCategory,
    MediaProfile,
    ObservationId,
    QualityMode,
)
from wre.routing import RouteQualityRequest, RouteResourceBudget, RouterInputSnapshot


def _budget(
    *,
    cpu_threads: int = 8,
    ram_bytes: int = 16_000_000_000,
    gpu_count: int = 0,
    gpu_vram_bytes: int = 0,
    scratch_storage_bytes: int = 100_000_000_000,
) -> RouteResourceBudget:
    return RouteResourceBudget(
        cpu_threads=cpu_threads,
        ram_bytes=ram_bytes,
        gpu_count=gpu_count,
        gpu_vram_bytes=gpu_vram_bytes,
        scratch_storage_bytes=scratch_storage_bytes,
    )


def _profile(observation_id: str) -> MediaProfile:
    return MediaProfile(observation_id=ObservationId(observation_id))


def _artifact(artifact_id: str, artifact_kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=ArtifactKind(artifact_kind),
    )


def _snapshot(
    *,
    quality: RouteQualityRequest | None = None,
    media_profiles: tuple[MediaProfile, ...] | None = None,
    resource_budget: RouteResourceBudget | None = None,
    existing_artifacts: tuple[ArtifactRef, ...] = (),
    prior_failures: tuple[FailureCategory, ...] = (),
) -> RouterInputSnapshot:
    return RouterInputSnapshot(
        quality=quality or RouteQualityRequest(QualityMode.QUALITY),
        media_profiles=media_profiles or (_profile("obs:a"),),
        resource_budget=resource_budget or _budget(),
        existing_artifacts=existing_artifacts,
        prior_failures=prior_failures,
    )


def test_route_resource_budget_is_exact_frozen_contract() -> None:
    budget = _budget()

    assert tuple(field.name for field in fields(RouteResourceBudget)) == (
        "cpu_threads",
        "ram_bytes",
        "gpu_count",
        "gpu_vram_bytes",
        "scratch_storage_bytes",
    )
    with pytest.raises(FrozenInstanceError):
        budget.cpu_threads = 4  # type: ignore[misc]


def test_route_resource_budget_accepts_cpu_only_and_gpu_budgets() -> None:
    cpu_only = _budget()
    gpu = _budget(gpu_count=2, gpu_vram_bytes=48_000_000_000)

    assert cpu_only.gpu_count == 0
    assert cpu_only.gpu_vram_bytes == 0
    assert gpu.gpu_count == 2
    assert gpu.gpu_vram_bytes == 48_000_000_000


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("cpu_threads", 0, "positive integer"),
        ("cpu_threads", True, "positive integer"),
        ("ram_bytes", 0, "positive integer"),
        ("ram_bytes", True, "positive integer"),
        ("gpu_count", -1, "non-negative integer"),
        ("gpu_count", True, "non-negative integer"),
        ("gpu_vram_bytes", -1, "non-negative integer"),
        ("gpu_vram_bytes", True, "non-negative integer"),
        ("scratch_storage_bytes", -1, "non-negative integer"),
        ("scratch_storage_bytes", True, "non-negative integer"),
    ],
)
def test_route_resource_budget_rejects_invalid_integer_values(
    field_name: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, Any] = {
        "cpu_threads": 8,
        "ram_bytes": 16_000_000_000,
        "gpu_count": 0,
        "gpu_vram_bytes": 0,
        "scratch_storage_bytes": 100_000_000_000,
    }
    values[field_name] = value

    with pytest.raises(ValueError, match=message):
        RouteResourceBudget(**values)


def test_route_resource_budget_rejects_vram_without_gpu() -> None:
    with pytest.raises(ValueError, match="must be zero when gpu_count is zero"):
        _budget(gpu_count=0, gpu_vram_bytes=1)


def test_router_input_snapshot_is_exact_frozen_contract_and_retains_objects() -> None:
    quality = RouteQualityRequest(QualityMode.MASTER)
    profiles = (_profile("obs:a"), _profile("obs:b"))
    budget = _budget(gpu_count=1, gpu_vram_bytes=24_000_000_000)
    artifacts = (_artifact("a", "media.profile"),)
    failures = (FailureCategory.TIMEOUT,)

    snapshot = _snapshot(
        quality=quality,
        media_profiles=profiles,
        resource_budget=budget,
        existing_artifacts=artifacts,
        prior_failures=failures,
    )

    assert tuple(field.name for field in fields(RouterInputSnapshot)) == (
        "quality",
        "media_profiles",
        "resource_budget",
        "existing_artifacts",
        "prior_failures",
    )
    assert snapshot.quality is quality
    assert snapshot.media_profiles is profiles
    assert snapshot.resource_budget is budget
    assert snapshot.existing_artifacts is artifacts
    assert snapshot.prior_failures is failures

    with pytest.raises(FrozenInstanceError):
        snapshot.quality = RouteQualityRequest(QualityMode.FAST)  # type: ignore[misc]


def test_router_input_snapshot_rejects_wrong_core_types() -> None:
    with pytest.raises(TypeError, match="quality must be RouteQualityRequest"):
        _snapshot(quality=cast(Any, QualityMode.FAST))
    with pytest.raises(TypeError, match="resource_budget must be RouteResourceBudget"):
        _snapshot(resource_budget=cast(Any, "budget"))


def test_router_input_snapshot_requires_nonempty_canonical_unique_profiles() -> None:
    profile_a = _profile("obs:a")
    profile_b = _profile("obs:b")

    with pytest.raises(TypeError, match="immutable tuple"):
        _snapshot(media_profiles=cast(Any, [profile_a]))
    with pytest.raises(ValueError, match="must not be empty"):
        RouterInputSnapshot(
            quality=RouteQualityRequest(QualityMode.FAST),
            media_profiles=(),
            resource_budget=_budget(),
            existing_artifacts=(),
            prior_failures=(),
        )
    with pytest.raises(TypeError, match="members must be MediaProfile"):
        _snapshot(media_profiles=cast(Any, ("obs:a",)))
    with pytest.raises(ValueError, match="unique ObservationId"):
        _snapshot(media_profiles=(profile_a, profile_a))
    with pytest.raises(ValueError, match="canonical ObservationId order"):
        _snapshot(media_profiles=(profile_b, profile_a))


def test_existing_artifacts_accept_empty_and_canonical_populated_values() -> None:
    assert _snapshot().existing_artifacts == ()

    refs = (
        _artifact("a", "media.profile"),
        _artifact("b", "scene.cluster"),
    )
    snapshot = _snapshot(existing_artifacts=refs)

    assert snapshot.existing_artifacts is refs


def test_existing_artifacts_fail_closed_on_collection_identity_and_order_errors() -> None:
    ref_a = _artifact("a", "media.profile")
    ref_b = _artifact("b", "scene.cluster")

    with pytest.raises(TypeError, match="immutable tuple"):
        _snapshot(existing_artifacts=cast(Any, [ref_a]))
    with pytest.raises(TypeError, match="members must be ArtifactRef"):
        _snapshot(existing_artifacts=cast(Any, ("a",)))
    with pytest.raises(ValueError, match="must be unique"):
        _snapshot(existing_artifacts=(ref_a, ref_a))
    with pytest.raises(ValueError, match="canonical ArtifactId/ArtifactKind order"):
        _snapshot(existing_artifacts=(ref_b, ref_a))
    with pytest.raises(ValueError, match="conflicting ArtifactKind"):
        _snapshot(
            existing_artifacts=(
                _artifact("same", "kind.a"),
                _artifact("same", "kind.b"),
            )
        )


def test_prior_failures_accept_empty_and_canonical_unique_values() -> None:
    assert _snapshot().prior_failures == ()

    failures = (
        FailureCategory.CAMERA_AMBIGUITY,
        FailureCategory.TIMEOUT,
    )
    snapshot = _snapshot(prior_failures=failures)

    assert snapshot.prior_failures is failures


def test_prior_failures_fail_closed_on_collection_type_duplicates_and_order() -> None:
    with pytest.raises(TypeError, match="immutable tuple"):
        _snapshot(prior_failures=cast(Any, [FailureCategory.TIMEOUT]))
    with pytest.raises(TypeError, match="members must be FailureCategory"):
        _snapshot(prior_failures=cast(Any, ("timeout",)))
    with pytest.raises(ValueError, match="must be unique"):
        _snapshot(
            prior_failures=(
                FailureCategory.TIMEOUT,
                FailureCategory.TIMEOUT,
            )
        )
    with pytest.raises(ValueError, match="canonical FailureCategory order"):
        _snapshot(
            prior_failures=(
                FailureCategory.TIMEOUT,
                FailureCategory.CAMERA_AMBIGUITY,
            )
        )


def test_router_input_snapshot_has_no_selection_fallback_or_execution_surface() -> None:
    snapshot = _snapshot()

    for attribute in (
        "route",
        "route_graph",
        "route_node",
        "adapter",
        "adapter_id",
        "capability",
        "score",
        "rank",
        "winner",
        "reason",
        "fallback",
        "retry",
        "escalation",
        "attempt",
        "schedule",
        "placement",
        "timestamp",
        "seed",
        "environment",
        "metadata",
    ):
        assert not hasattr(snapshot, attribute)


def test_router_inputs_module_has_no_io_persistence_or_external_execution_surface() -> None:
    forbidden_symbols = {
        "Path",
        "subprocess",
        "socket",
        "requests",
        "sqlite3",
        "FFmpegToolchain",
        "HardwareRuntimeIdentity",
        "AdapterCapabilityDescriptor",
    }

    assert forbidden_symbols.isdisjoint(router_inputs_module.__dict__)
