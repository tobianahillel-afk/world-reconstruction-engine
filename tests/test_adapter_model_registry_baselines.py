from __future__ import annotations

from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

import yaml

from wre.ingestion import SUPPORTED_FFMPEG_VERSION
from wre.reconstruction import (
    COLMAP_IMPORTER_VERSION,
    SUPPORTED_COLMAP_VERSION,
    SUPPORTED_PYCOLMAP_VERSION,
)
from wre.repo_validation import validate_repository

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry" / "adapter-models.yaml"
DEPENDENCIES_PATH = ROOT / "registry" / "dependencies.yaml"

_EXPECTED_ADAPTER_IDS = (
    "colmap.sparse_sfm",
    "exifread.raw_exif",
    "ffmpeg.keyframe_extraction",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _entries_by_id() -> dict[str, dict[str, Any]]:
    registry = _load_mapping(REGISTRY_PATH)
    assert registry["schema_version"] == 1
    entries = registry["entries"]
    assert isinstance(entries, list)
    typed_entries: list[dict[str, Any]] = []
    for entry in entries:
        assert isinstance(entry, dict)
        typed_entries.append(entry)
    return {entry["adapter_id"]: entry for entry in typed_entries}


def test_registry_contains_exactly_the_three_retained_baselines() -> None:
    registry = _load_mapping(REGISTRY_PATH)
    entries = registry["entries"]
    assert isinstance(entries, list)
    adapter_ids = tuple(entry["adapter_id"] for entry in entries)

    assert adapter_ids == _EXPECTED_ADAPTER_IDS
    assert adapter_ids == tuple(sorted(adapter_ids))
    assert validate_repository(ROOT) == []


def test_baseline_versions_and_dependency_refs_match_retained_evidence() -> None:
    entries = _entries_by_id()
    dependencies = _load_mapping(DEPENDENCIES_PATH)["dependencies"]
    assert isinstance(dependencies, dict)

    colmap = entries["colmap.sparse_sfm"]
    assert colmap["producer"] == {
        "implementation": "wre.colmap_reconstruction_importer",
        "version": COLMAP_IMPORTER_VERSION,
        "revision": None,
    }
    assert colmap["dependency_refs"] == ["colmap"]
    assert dependencies["colmap"]["pinned_version"] == SUPPORTED_PYCOLMAP_VERSION == "4.2.0"
    assert SUPPORTED_COLMAP_VERSION == "COLMAP 4.2.0"

    exifread = entries["exifread.raw_exif"]
    assert exifread["producer"] == {
        "implementation": "wre.ingestion.exif.ExifMetadataIngestor",
        "version": "3.5.1",
        "revision": None,
    }
    assert exifread["dependency_refs"] == ["exifread"]
    assert dependencies["exifread"]["pinned_version"] == distribution_version("ExifRead") == "3.5.1"

    ffmpeg = entries["ffmpeg.keyframe_extraction"]
    assert ffmpeg["producer"] == {
        "implementation": "wre.ingestion.keyframes.LocalKeyframeExtractor",
        "version": SUPPORTED_FFMPEG_VERSION,
        "revision": None,
    }
    assert ffmpeg["dependency_refs"] == ["ffmpeg"]
    assert dependencies["ffmpeg"]["pinned_version"] == SUPPORTED_FFMPEG_VERSION


def test_baseline_capabilities_are_conservative_and_existing_only() -> None:
    entries = _entries_by_id()

    assert entries["colmap.sparse_sfm"]["capability"] == {
        "name": "geometry.sparse_sfm",
        "input_kinds": ["image.observation"],
        "output_kinds": ["geometry.sparse_reconstruction_estimate"],
    }
    assert entries["exifread.raw_exif"]["capability"] == {
        "name": "metadata.raw_exif",
        "input_kinds": ["image.observation"],
        "output_kinds": ["observation.metadata"],
    }
    assert entries["ffmpeg.keyframe_extraction"]["capability"] == {
        "name": "media.keyframe_extraction",
        "input_kinds": ["video.observation"],
        "output_kinds": ["video.frame_observation"],
    }

    for entry in entries.values():
        assert entry["model"] is None
        assert entry["checkpoint"] is None
        assert entry["failure_signals"] == []
        assert entry["metric_names"] == []
        assert entry["resume_mode"] == "unsupported"

    output_kinds = {
        output_kind
        for entry in entries.values()
        for output_kind in entry["capability"]["output_kinds"]
    }
    assert "camera.solution" not in output_kinds
    assert "geometry.solution" not in output_kinds


def test_baseline_shipping_license_and_hardware_policy_match_approved_evidence() -> None:
    entries = _entries_by_id()
    dependencies = _load_mapping(DEPENDENCIES_PATH)["dependencies"]
    assert isinstance(dependencies, dict)

    expected_hardware_policy = {
        "colmap.sparse_sfm": "required",
        "exifread.raw_exif": "omitted",
        "ffmpeg.keyframe_extraction": "omitted",
    }

    for adapter_id, entry in entries.items():
        dependency_ref = entry["dependency_refs"][0]
        dependency = dependencies[dependency_ref]
        assert dependency["license_review"] == "approved"
        assert entry["license"]["direct"] == dependency["license"]
        assert entry["license"]["review"] == "approved"
        assert entry["shipping_status"] == "approved"
        assert entry["artifact_key_hardware_policy"] == expected_hardware_policy[adapter_id]
        assert isinstance(entry["license"]["transitive_notes"], str)
        assert entry["license"]["transitive_notes"].strip()
        assert isinstance(entry["license"]["redistribution_notes"], str)
        assert entry["license"]["redistribution_notes"].strip()
        assert isinstance(entry["reproducibility_notes"], str)
        assert entry["reproducibility_notes"].strip()


def test_registry_does_not_promote_pending_candidates() -> None:
    entries = _entries_by_id()
    registered_dependencies = {
        dependency_ref
        for entry in entries.values()
        for dependency_ref in entry["dependency_refs"]
    }

    assert registered_dependencies == {"colmap", "exifread", "ffmpeg"}
    assert registered_dependencies.isdisjoint(
        {
            "alicevision",
            "gtsam",
            "hloc",
            "lightglue",
            "limap",
            "open3d",
            "openmvg",
            "opensfm",
            "teaserpp",
        }
    )
