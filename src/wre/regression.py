from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any


class Comparator(StrEnum):
    EQ = "eq"
    LTE = "lte"
    GTE = "gte"


@dataclass(frozen=True)
class MetricExpectation:
    metric: str
    comparator: Comparator
    target: float
    abs_tolerance: float = 0.0
    unit: str | None = None


@dataclass(frozen=True)
class FixtureSpec:
    schema_version: int
    fixture_id: str
    kind: str
    lane: str
    deterministic: bool
    seed: int | None
    input_files: tuple[str, ...]
    expectations: tuple[MetricExpectation, ...]
    provenance: Mapping[str, str]
    metadata_path: Path

    def resolve_input(self, relative_path: str) -> Path:
        if relative_path not in self.input_files:
            raise ValueError(f"fixture input is not declared: {relative_path}")
        return self.metadata_path.parent / relative_path


@dataclass(frozen=True)
class MetricCheck:
    metric: str
    comparator: Comparator
    target: float
    observed: float
    abs_tolerance: float
    passed: bool
    unit: str | None = None


@dataclass(frozen=True)
class RegressionResult:
    schema_version: int
    fixture_id: str
    runner: str
    status: str
    metrics: Mapping[str, float]
    checks: tuple[MetricCheck, ...]

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fixture_id": self.fixture_id,
            "runner": self.runner,
            "status": self.status,
            "metrics": {name: self.metrics[name] for name in sorted(self.metrics)},
            "checks": [
                {
                    "metric": check.metric,
                    "comparator": check.comparator.value,
                    "target": check.target,
                    "observed": check.observed,
                    "abs_tolerance": check.abs_tolerance,
                    "passed": check.passed,
                    "unit": check.unit,
                }
                for check in self.checks
            ],
        }


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _non_negative_number(value: object, context: str) -> float:
    result = _finite_number(value, context)
    if result < 0:
        raise ValueError(f"{context} must be non-negative")
    return result


def _safe_relative_path(value: object, context: str) -> str:
    text = _string(value, context)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} must stay inside the fixture directory")
    return text


def load_fixture(path: Path) -> FixtureSpec:
    metadata_path = path.resolve()
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load fixture metadata {path}: {exc}") from exc

    data = _mapping(raw, "fixture")
    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported fixture schema_version: {schema_version!r}")

    deterministic = data.get("deterministic")
    if not isinstance(deterministic, bool):
        raise ValueError("fixture.deterministic must be a boolean")

    seed_value = data.get("seed")
    if seed_value is not None and (isinstance(seed_value, bool) or not isinstance(seed_value, int)):
        raise ValueError("fixture.seed must be an integer or null")

    input_values = data.get("input_files")
    if not isinstance(input_values, list) or not input_values:
        raise ValueError("fixture.input_files must be a non-empty list")
    input_files = tuple(
        _safe_relative_path(value, f"fixture.input_files[{index}]")
        for index, value in enumerate(input_values)
    )
    for relative_path in input_files:
        if not (metadata_path.parent / relative_path).is_file():
            raise ValueError(f"fixture input does not exist: {relative_path}")

    expectation_values = data.get("expectations")
    if not isinstance(expectation_values, list) or not expectation_values:
        raise ValueError("fixture.expectations must be a non-empty list")

    expectations: list[MetricExpectation] = []
    seen_metrics: set[str] = set()
    for index, value in enumerate(expectation_values):
        expectation = _mapping(value, f"fixture.expectations[{index}]")
        metric = _string(expectation.get("metric"), f"fixture.expectations[{index}].metric")
        if metric in seen_metrics:
            raise ValueError(f"duplicate fixture expectation metric: {metric}")
        seen_metrics.add(metric)
        try:
            comparator = Comparator(
                _string(
                    expectation.get("comparator"),
                    f"fixture.expectations[{index}].comparator",
                )
            )
        except ValueError as exc:
            raise ValueError(f"unsupported comparator for metric {metric}") from exc

        unit_value = expectation.get("unit")
        if unit_value is not None and not isinstance(unit_value, str):
            raise ValueError(f"fixture expectation unit for {metric} must be a string or null")

        expectations.append(
            MetricExpectation(
                metric=metric,
                comparator=comparator,
                target=_finite_number(
                    expectation.get("target"), f"fixture.expectations[{index}].target"
                ),
                abs_tolerance=_non_negative_number(
                    expectation.get("abs_tolerance", 0.0),
                    f"fixture.expectations[{index}].abs_tolerance",
                ),
                unit=unit_value,
            )
        )

    provenance_raw = _mapping(data.get("provenance", {}), "fixture.provenance")
    provenance: dict[str, str] = {}
    for key, value in provenance_raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("fixture.provenance keys and values must be strings")
        provenance[key] = value

    return FixtureSpec(
        schema_version=1,
        fixture_id=_string(data.get("fixture_id"), "fixture.fixture_id"),
        kind=_string(data.get("kind"), "fixture.kind"),
        lane=_string(data.get("lane"), "fixture.lane"),
        deterministic=deterministic,
        seed=seed_value,
        input_files=input_files,
        expectations=tuple(expectations),
        provenance=provenance,
        metadata_path=metadata_path,
    )


def _check_metric(expectation: MetricExpectation, observed: float) -> bool:
    if expectation.comparator is Comparator.EQ:
        return abs(observed - expectation.target) <= expectation.abs_tolerance
    if expectation.comparator is Comparator.LTE:
        return observed <= expectation.target + expectation.abs_tolerance
    return observed >= expectation.target - expectation.abs_tolerance


def evaluate_fixture(
    fixture: FixtureSpec,
    observed_metrics: Mapping[str, int | float],
    *,
    runner: str,
) -> RegressionResult:
    runner_name = _string(runner, "runner")
    metrics: dict[str, float] = {}
    for name, value in observed_metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("observed metric names must be non-empty strings")
        metrics[name] = _finite_number(value, f"observed metric {name}")

    checks: list[MetricCheck] = []
    for expectation in fixture.expectations:
        if expectation.metric not in metrics:
            raise ValueError(f"missing observed metric: {expectation.metric}")
        observed = metrics[expectation.metric]
        checks.append(
            MetricCheck(
                metric=expectation.metric,
                comparator=expectation.comparator,
                target=expectation.target,
                observed=observed,
                abs_tolerance=expectation.abs_tolerance,
                passed=_check_metric(expectation, observed),
                unit=expectation.unit,
            )
        )

    status = "pass" if all(check.passed for check in checks) else "fail"
    return RegressionResult(
        schema_version=1,
        fixture_id=fixture.fixture_id,
        runner=runner_name,
        status=status,
        metrics=metrics,
        checks=tuple(checks),
    )


def write_regression_result(path: Path, result: RegressionResult) -> None:
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
