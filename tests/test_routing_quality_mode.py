from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from enum import StrEnum
from typing import Any, cast

import pytest

import wre.domain as domain_module
import wre.routing as routing_module
import wre.routing.quality_mode as routing_quality_module
from wre.domain import ProvenanceClass, QualityDecision, QualityMode
from wre.routing import RouteQualityRequest


class ForeignMode(StrEnum):
    PREVIEW = "preview"


def test_routing_reuses_exact_canonical_quality_mode_object() -> None:
    assert routing_module.QualityMode is domain_module.QualityMode
    assert routing_quality_module.QualityMode is domain_module.QualityMode
    assert routing_module.QualityMode is QualityMode

    assert [(member.name, member.value) for member in routing_module.QualityMode] == [
        ("PREVIEW", "preview"),
        ("FAST", "fast"),
        ("QUALITY", "quality"),
        ("MASTER", "master"),
    ]


def test_route_quality_request_is_exact_frozen_contract() -> None:
    request = RouteQualityRequest(quality_mode=QualityMode.PREVIEW)

    assert tuple(field.name for field in fields(RouteQualityRequest)) == ("quality_mode",)
    assert request.quality_mode is QualityMode.PREVIEW

    with pytest.raises(FrozenInstanceError):
        request.quality_mode = QualityMode.MASTER  # type: ignore[misc]


@pytest.mark.parametrize("quality_mode", list(QualityMode))
def test_every_canonical_quality_mode_preserves_identity_and_wire_token(
    quality_mode: QualityMode,
) -> None:
    request = RouteQualityRequest(quality_mode=quality_mode)

    assert request.quality_mode is quality_mode
    assert str(request.quality_mode) == quality_mode.value


@pytest.mark.parametrize(
    "invalid",
    [
        "preview",
        1,
        True,
        QualityDecision.PASS,
        ForeignMode.PREVIEW,
    ],
)
def test_route_quality_request_rejects_noncanonical_mode_values(invalid: object) -> None:
    with pytest.raises(TypeError, match="must be QualityMode"):
        RouteQualityRequest(quality_mode=cast(Any, invalid))


def test_quality_mode_remains_distinct_from_decision_and_provenance() -> None:
    assert routing_module.QualityMode is not QualityDecision
    assert routing_module.QualityMode is not ProvenanceClass

    mode_values = {member.value for member in routing_module.QualityMode}
    decision_values = {member.value for member in QualityDecision}
    provenance_values = {member.value for member in ProvenanceClass}

    assert mode_values.isdisjoint(decision_values)
    assert mode_values.isdisjoint(provenance_values)


def test_route_quality_request_has_no_future_routing_or_truth_surface() -> None:
    request = RouteQualityRequest(quality_mode=QualityMode.QUALITY)

    for attribute in (
        "provenance",
        "provenance_class",
        "confidence",
        "score",
        "decision",
        "failure_category",
        "threshold",
        "adapter_capability",
        "route",
        "route_node",
        "route_graph",
        "profile",
        "budget",
        "artifacts",
        "fallback",
        "retry",
        "escalation",
        "reason",
        "metadata",
    ):
        assert not hasattr(request, attribute)


def test_routing_quality_mode_module_has_no_execution_or_persistence_surface() -> None:
    forbidden_symbols = {
        "Path",
        "subprocess",
        "socket",
        "requests",
        "sqlite3",
        "FFmpegToolchain",
        "AdapterCapabilityDescriptor",
        "QualityDecision",
        "ProvenanceClass",
    }

    assert forbidden_symbols.isdisjoint(routing_quality_module.__dict__)
