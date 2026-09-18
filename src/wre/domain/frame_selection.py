from __future__ import annotations

import re
from dataclasses import dataclass

from wre.domain.metrics import MetricName

_POLICY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True, slots=True, order=True)
class FrameSelectionPolicyId:
    """Opaque stable identity for one frame-selection policy family."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or _POLICY_ID_RE.fullmatch(self.value) is None:
            raise ValueError(
                "frame_selection_policy_id must be 1-128 characters using "
                "letters, digits, '.', '_', ':' or '-'"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class FrameSelectionPolicyRevision:
    """Positive integer revision for one frame-selection policy."""

    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int or self.value <= 0:
            raise ValueError("frame_selection_policy_revision must be a positive integer")


@dataclass(frozen=True, slots=True)
class FrameSelectionPolicy:
    """Immutable declaration of metric evidence used by a future frame selector."""

    policy_id: FrameSelectionPolicyId
    revision: FrameSelectionPolicyRevision
    required_metric_names: tuple[MetricName, ...] = ()
    optional_metric_names: tuple[MetricName, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, FrameSelectionPolicyId):
            raise TypeError("frame_selection_policy.policy_id must be FrameSelectionPolicyId")
        if not isinstance(self.revision, FrameSelectionPolicyRevision):
            raise TypeError(
                "frame_selection_policy.revision must be FrameSelectionPolicyRevision"
            )

        self._validate_metric_names(
            self.required_metric_names,
            "frame_selection_policy.required_metric_names",
        )
        self._validate_metric_names(
            self.optional_metric_names,
            "frame_selection_policy.optional_metric_names",
        )

        required = set(self.required_metric_names)
        optional = set(self.optional_metric_names)
        if required & optional:
            raise ValueError(
                "frame_selection_policy required and optional metric names must be disjoint"
            )

    @staticmethod
    def _validate_metric_names(
        values: object,
        context: str,
    ) -> None:
        if not isinstance(values, tuple):
            raise TypeError(f"{context} must be an immutable tuple")
        if any(not isinstance(value, MetricName) for value in values):
            raise TypeError(f"{context} members must be MetricName")

        names = tuple(value.value for value in values)
        if len(names) != len(set(names)):
            raise ValueError(f"{context} members must be unique")
        if names != tuple(sorted(names)):
            raise ValueError(f"{context} must use canonical MetricName order")
