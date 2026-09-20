from __future__ import annotations

from dataclasses import dataclass

from wre.domain import ArtifactRef, FailureCategory, MediaProfile
from wre.routing.quality_mode import RouteQualityRequest


def _require_positive_int(value: object, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")


def _require_non_negative_int(value: object, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class RouteResourceBudget:
    """Explicit caller-supplied resource ceiling for deterministic routing input."""

    cpu_threads: int
    ram_bytes: int
    gpu_count: int
    gpu_vram_bytes: int
    scratch_storage_bytes: int

    def __post_init__(self) -> None:
        _require_positive_int(self.cpu_threads, "route_resource_budget.cpu_threads")
        _require_positive_int(self.ram_bytes, "route_resource_budget.ram_bytes")
        _require_non_negative_int(self.gpu_count, "route_resource_budget.gpu_count")
        _require_non_negative_int(
            self.gpu_vram_bytes,
            "route_resource_budget.gpu_vram_bytes",
        )
        _require_non_negative_int(
            self.scratch_storage_bytes,
            "route_resource_budget.scratch_storage_bytes",
        )
        if self.gpu_count == 0 and self.gpu_vram_bytes != 0:
            raise ValueError(
                "route_resource_budget.gpu_vram_bytes must be zero when gpu_count is zero"
            )


@dataclass(frozen=True, slots=True)
class RouterInputSnapshot:
    """Validated immutable inputs available to later deterministic route selection."""

    quality: RouteQualityRequest
    media_profiles: tuple[MediaProfile, ...]
    resource_budget: RouteResourceBudget
    existing_artifacts: tuple[ArtifactRef, ...]
    prior_failures: tuple[FailureCategory, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.quality, RouteQualityRequest):
            raise TypeError("router_input.quality must be RouteQualityRequest")
        if not isinstance(self.media_profiles, tuple):
            raise TypeError("router_input.media_profiles must be an immutable tuple")
        if not self.media_profiles:
            raise ValueError("router_input.media_profiles must not be empty")
        if any(not isinstance(profile, MediaProfile) for profile in self.media_profiles):
            raise TypeError("router_input.media_profiles members must be MediaProfile")

        observation_ids = tuple(profile.observation_id.value for profile in self.media_profiles)
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("router_input.media_profiles must use unique ObservationId values")
        if observation_ids != tuple(sorted(observation_ids)):
            raise ValueError(
                "router_input.media_profiles must use canonical ObservationId order"
            )

        if not isinstance(self.resource_budget, RouteResourceBudget):
            raise TypeError("router_input.resource_budget must be RouteResourceBudget")

        if not isinstance(self.existing_artifacts, tuple):
            raise TypeError("router_input.existing_artifacts must be an immutable tuple")
        if any(not isinstance(ref, ArtifactRef) for ref in self.existing_artifacts):
            raise TypeError("router_input.existing_artifacts members must be ArtifactRef")

        artifact_keys = tuple(
            (ref.artifact_id.value, ref.artifact_kind.value)
            for ref in self.existing_artifacts
        )
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ValueError("router_input.existing_artifacts must be unique")
        if artifact_keys != tuple(sorted(artifact_keys)):
            raise ValueError(
                "router_input.existing_artifacts must use canonical ArtifactId/ArtifactKind order"
            )

        kinds_by_id: dict[str, str] = {}
        for ref in self.existing_artifacts:
            previous = kinds_by_id.setdefault(
                ref.artifact_id.value,
                ref.artifact_kind.value,
            )
            if previous != ref.artifact_kind.value:
                raise ValueError(
                    "router_input.existing_artifacts must not declare conflicting "
                    "ArtifactKind values for one ArtifactId"
                )

        if not isinstance(self.prior_failures, tuple):
            raise TypeError("router_input.prior_failures must be an immutable tuple")
        if any(not isinstance(failure, FailureCategory) for failure in self.prior_failures):
            raise TypeError("router_input.prior_failures members must be FailureCategory")

        failure_values = tuple(failure.value for failure in self.prior_failures)
        if len(failure_values) != len(set(failure_values)):
            raise ValueError("router_input.prior_failures must be unique")
        if failure_values != tuple(sorted(failure_values)):
            raise ValueError(
                "router_input.prior_failures must use canonical FailureCategory order"
            )
