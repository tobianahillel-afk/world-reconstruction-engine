from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from wre.regression import evaluate_fixture, load_fixture, write_regression_result

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "synthetic" / "tiny-smoke"
FIXTURE_PATH = FIXTURE_DIR / "fixture.json"


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _measure_tiny_points() -> dict[str, float]:
    fixture = load_fixture(FIXTURE_PATH)
    payload = _load_mapping(fixture.resolve_input("points.json"))
    raw_points = payload["points"]
    assert isinstance(raw_points, list)

    points: list[tuple[float, float, float]] = []
    for raw_point in raw_points:
        assert isinstance(raw_point, list)
        assert len(raw_point) == 3
        points.append((float(raw_point[0]), float(raw_point[1]), float(raw_point[2])))

    xs = sorted(point[0] for point in points)
    ys = [point[1] for point in points]
    spacings = [right - left for left, right in pairwise(xs)]

    return {
        "point_count": float(len(points)),
        "centroid_x": sum(xs) / len(xs),
        "max_abs_y": max(abs(value) for value in ys),
        "min_spacing_x": min(spacings),
    }


def test_tiny_synthetic_fixture_passes() -> None:
    fixture = load_fixture(FIXTURE_PATH)
    result = evaluate_fixture(fixture, _measure_tiny_points(), runner="pytest-tiny-smoke")

    assert fixture.fixture_id == "synthetic.tiny-smoke.v1"
    assert fixture.lane == "fast"
    assert fixture.deterministic is True
    assert fixture.seed == 0
    assert result.passed is True
    assert result.status == "pass"
    assert all(check.passed for check in result.checks)


def test_metric_regression_is_reported() -> None:
    fixture = load_fixture(FIXTURE_PATH)
    metrics = _measure_tiny_points()
    metrics["centroid_x"] = 9.0

    result = evaluate_fixture(fixture, metrics, runner="pytest-regression")

    assert result.passed is False
    assert result.status == "fail"
    failed = [check.metric for check in result.checks if not check.passed]
    assert failed == ["centroid_x"]


def test_missing_required_metric_is_invalid_runner_output() -> None:
    fixture = load_fixture(FIXTURE_PATH)
    metrics = _measure_tiny_points()
    del metrics["min_spacing_x"]

    with pytest.raises(ValueError, match="missing observed metric: min_spacing_x"):
        evaluate_fixture(fixture, metrics, runner="pytest-missing-metric")


def test_regression_result_serialization_is_stable(tmp_path: Path) -> None:
    fixture = load_fixture(FIXTURE_PATH)
    result = evaluate_fixture(fixture, _measure_tiny_points(), runner="pytest-stable-output")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_regression_result(first, result)
    write_regression_result(second, result)

    assert first.read_bytes() == second.read_bytes()
    payload = _load_mapping(first)
    assert list(payload["metrics"]) == sorted(payload["metrics"])


@pytest.mark.parametrize("escape_path", ["../outside.json", "..\\outside.json"])
def test_fixture_inputs_cannot_escape_fixture_directory(tmp_path: Path, escape_path: str) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    metadata = {
        "schema_version": 1,
        "fixture_id": "synthetic.escape-test.v1",
        "kind": "synthetic",
        "lane": "fast",
        "deterministic": True,
        "seed": 0,
        "input_files": [escape_path],
        "expectations": [
            {
                "metric": "count",
                "comparator": "eq",
                "target": 1,
                "abs_tolerance": 0,
                "unit": "count",
            }
        ],
        "provenance": {"source": "test"},
    }
    metadata_path = fixture_dir / "fixture.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="must stay inside the fixture directory"):
        load_fixture(metadata_path)


def test_contract_schemas_are_valid_json() -> None:
    fixture_schema = _load_mapping(ROOT / "registry" / "schemas" / "fixture.schema.json")
    result_schema = _load_mapping(ROOT / "registry" / "schemas" / "regression-result.schema.json")

    assert fixture_schema["title"] == "WRE Fixture Metadata"
    assert result_schema["title"] == "WRE Regression Result"
