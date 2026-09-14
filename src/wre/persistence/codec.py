from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

from wre.domain.cameras import (
    Camera,
    CameraId,
    ImageDimensions,
    ObservationMetadata,
    RawMetadataEntry,
)
from wre.domain.fragments import LocalFrameId, SpatialFragment, SpatialFragmentId
from wre.domain.metadata import (
    CaptureTimeInterpretation,
    CaptureTimeInterpretationStatus,
    GpsInterpretationStatus,
    GpsMetadataInterpretation,
    ObservationMetadataInterpretation,
)
from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    Observation,
    ObservationId,
    ObservationKind,
    Sha256Digest,
    SourceId,
    SourceRef,
    VideoFrameObservation,
)
from wre.domain.runs import (
    DerivedArtifactProvenance,
    ProducerRef,
    ReconstructionRun,
    ReconstructionRunId,
)

JsonObject = dict[str, object]


def canonical_json(payload: Mapping[str, object]) -> str:
    """Serialize a payload deterministically for persistence and comparison."""

    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def parse_json_object(payload_json: str) -> JsonObject:
    try:
        value = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise ValueError("payload_json is not valid JSON") from exc
    return _object(value, "payload")


def _object(value: object, context: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return cast(JsonObject, value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    return value


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context)


def _int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context} must be a number")
    return float(value)


def _optional_float(value: object, context: str) -> float | None:
    if value is None:
        return None
    return _float(value, context)


def _optional_object(value: object, context: str) -> JsonObject | None:
    if value is None:
        return None
    return _object(value, context)


def _object_list(value: object, context: str) -> list[JsonObject]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return [_object(item, f"{context}[]") for item in value]


def _string_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON array")
    return [_string(item, f"{context}[]") for item in value]


def _encode_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _decode_datetime(value: object, context: str) -> datetime:
    text = _string(value, context)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{context} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{context} must be timezone-aware")
    return parsed


def _encode_asset(asset: MediaAssetRef) -> JsonObject:
    return {
        "byte_length": asset.byte_length,
        "mime_type": asset.mime_type,
        "sha256": asset.sha256.value,
        "uri": asset.uri,
    }


def _decode_asset(payload: object, context: str) -> MediaAssetRef:
    data = _object(payload, context)
    return MediaAssetRef(
        uri=_string(data.get("uri"), f"{context}.uri"),
        sha256=Sha256Digest(_string(data.get("sha256"), f"{context}.sha256")),
        byte_length=_int(data.get("byte_length"), f"{context}.byte_length"),
        mime_type=_optional_string(data.get("mime_type"), f"{context}.mime_type"),
    )


def _encode_source(source: SourceRef) -> JsonObject:
    return {
        "locator": source.locator,
        "source_id": source.source_id.value,
    }


def _decode_source(payload: object) -> SourceRef:
    data = _object(payload, "source")
    return SourceRef(
        source_id=SourceId(_string(data.get("source_id"), "source.source_id")),
        locator=_optional_string(data.get("locator"), "source.locator"),
    )


def encode_observation(observation: Observation) -> JsonObject:
    captured_at = observation.captured_at
    payload: JsonObject = {
        "asset": _encode_asset(observation.asset),
        "captured_at": _encode_datetime(captured_at) if captured_at is not None else None,
        "kind": observation.kind.value,
        "observation_id": observation.observation_id.value,
        "received_at": _encode_datetime(observation.received_at),
        "source": _encode_source(observation.source),
    }
    if isinstance(observation, VideoFrameObservation):
        payload.update(
            {
                "frame_index": observation.frame_index,
                "frame_time_us": observation.frame_time_us,
                "video_asset": _encode_asset(observation.video_asset),
            }
        )
    return payload


def decode_observation(payload: Mapping[str, object]) -> Observation:
    data = dict(payload)
    kind = _string(data.get("kind"), "kind")
    observation_id = ObservationId(_string(data.get("observation_id"), "observation_id"))
    asset = _decode_asset(data.get("asset"), "asset")
    source = _decode_source(data.get("source"))
    received_at = _decode_datetime(data.get("received_at"), "received_at")
    captured_at_value = data.get("captured_at")
    captured_at = (
        _decode_datetime(captured_at_value, "captured_at")
        if captured_at_value is not None
        else None
    )
    if kind == ObservationKind.IMAGE.value:
        return ImageObservation(
            observation_id=observation_id,
            asset=asset,
            source=source,
            received_at=received_at,
            captured_at=captured_at,
        )
    if kind == ObservationKind.VIDEO_FRAME.value:
        return VideoFrameObservation(
            observation_id=observation_id,
            asset=asset,
            source=source,
            received_at=received_at,
            captured_at=captured_at,
            video_asset=_decode_asset(data.get("video_asset"), "video_asset"),
            frame_index=_int(data.get("frame_index"), "frame_index"),
            frame_time_us=_int(data.get("frame_time_us"), "frame_time_us"),
        )
    raise ValueError(f"unsupported observation kind: {kind!r}")


def encode_camera(camera: Camera) -> JsonObject:
    return {
        "camera_id": camera.camera_id.value,
        "manufacturer": camera.manufacturer,
        "model": camera.model,
        "serial_number": camera.serial_number,
    }


def decode_camera(payload: Mapping[str, object]) -> Camera:
    data = dict(payload)
    return Camera(
        camera_id=CameraId(_string(data.get("camera_id"), "camera_id")),
        manufacturer=_optional_string(data.get("manufacturer"), "manufacturer"),
        model=_optional_string(data.get("model"), "model"),
        serial_number=_optional_string(data.get("serial_number"), "serial_number"),
    )


def encode_observation_metadata(metadata: ObservationMetadata) -> JsonObject:
    dimensions: JsonObject | None = None
    if metadata.dimensions is not None:
        dimensions = {
            "height_px": metadata.dimensions.height_px,
            "width_px": metadata.dimensions.width_px,
        }
    return {
        "camera_id": metadata.camera_id.value if metadata.camera_id is not None else None,
        "dimensions": dimensions,
        "observation_id": metadata.observation_id.value,
        "raw_entries": [
            {"key": entry.key, "namespace": entry.namespace, "value": entry.value}
            for entry in metadata.raw_entries
        ],
    }


def decode_observation_metadata(payload: Mapping[str, object]) -> ObservationMetadata:
    data = dict(payload)
    camera_id_value = _optional_string(data.get("camera_id"), "camera_id")
    dimensions_payload = _optional_object(data.get("dimensions"), "dimensions")
    dimensions = None
    if dimensions_payload is not None:
        dimensions = ImageDimensions(
            width_px=_int(dimensions_payload.get("width_px"), "dimensions.width_px"),
            height_px=_int(dimensions_payload.get("height_px"), "dimensions.height_px"),
        )
    raw_entries = tuple(
        RawMetadataEntry(
            namespace=_string(entry.get("namespace"), "raw_entries[].namespace"),
            key=_string(entry.get("key"), "raw_entries[].key"),
            value=_string(entry.get("value"), "raw_entries[].value"),
        )
        for entry in _object_list(data.get("raw_entries"), "raw_entries")
    )
    return ObservationMetadata(
        observation_id=ObservationId(_string(data.get("observation_id"), "observation_id")),
        camera_id=CameraId(camera_id_value) if camera_id_value is not None else None,
        dimensions=dimensions,
        raw_entries=raw_entries,
    )


def encode_metadata_interpretation(
    interpretation: ObservationMetadataInterpretation,
) -> JsonObject:
    instant = interpretation.capture_time.instant
    return {
        "capture_time": {
            "evidence_keys": list(interpretation.capture_time.evidence_keys),
            "instant": _encode_datetime(instant) if instant is not None else None,
            "issue": interpretation.capture_time.issue,
            "raw_datetime": interpretation.capture_time.raw_datetime,
            "raw_offset": interpretation.capture_time.raw_offset,
            "status": interpretation.capture_time.status.value,
        },
        "gps": {
            "evidence_keys": list(interpretation.gps.evidence_keys),
            "issue": interpretation.gps.issue,
            "latitude_deg": interpretation.gps.latitude_deg,
            "longitude_deg": interpretation.gps.longitude_deg,
            "map_datum": interpretation.gps.map_datum,
            "status": interpretation.gps.status.value,
        },
        "observation_id": interpretation.observation_id.value,
    }


def decode_metadata_interpretation(
    payload: Mapping[str, object],
) -> ObservationMetadataInterpretation:
    data = dict(payload)
    gps_data = _object(data.get("gps"), "gps")
    capture_data = _object(data.get("capture_time"), "capture_time")
    instant_value = capture_data.get("instant")
    return ObservationMetadataInterpretation(
        observation_id=ObservationId(_string(data.get("observation_id"), "observation_id")),
        gps=GpsMetadataInterpretation(
            status=GpsInterpretationStatus(_string(gps_data.get("status"), "gps.status")),
            latitude_deg=_optional_float(gps_data.get("latitude_deg"), "gps.latitude_deg"),
            longitude_deg=_optional_float(gps_data.get("longitude_deg"), "gps.longitude_deg"),
            map_datum=_optional_string(gps_data.get("map_datum"), "gps.map_datum"),
            evidence_keys=tuple(_string_list(gps_data.get("evidence_keys"), "gps.evidence_keys")),
            issue=_optional_string(gps_data.get("issue"), "gps.issue"),
        ),
        capture_time=CaptureTimeInterpretation(
            status=CaptureTimeInterpretationStatus(
                _string(capture_data.get("status"), "capture_time.status")
            ),
            raw_datetime=_optional_string(
                capture_data.get("raw_datetime"), "capture_time.raw_datetime"
            ),
            raw_offset=_optional_string(capture_data.get("raw_offset"), "capture_time.raw_offset"),
            instant=(
                _decode_datetime(instant_value, "capture_time.instant")
                if instant_value is not None
                else None
            ),
            evidence_keys=tuple(
                _string_list(capture_data.get("evidence_keys"), "capture_time.evidence_keys")
            ),
            issue=_optional_string(capture_data.get("issue"), "capture_time.issue"),
        ),
    )


def encode_spatial_fragment(fragment: SpatialFragment) -> JsonObject:
    return {
        "fragment_id": fragment.fragment_id.value,
        "local_frame_id": fragment.local_frame_id.value,
        "observation_ids": [item.value for item in fragment.observation_ids],
    }


def decode_spatial_fragment(payload: Mapping[str, object]) -> SpatialFragment:
    data = dict(payload)
    return SpatialFragment(
        fragment_id=SpatialFragmentId(_string(data.get("fragment_id"), "fragment_id")),
        local_frame_id=LocalFrameId(_string(data.get("local_frame_id"), "local_frame_id")),
        observation_ids=tuple(
            ObservationId(value)
            for value in _string_list(data.get("observation_ids"), "observation_ids")
        ),
    )


def encode_reconstruction_run(run: ReconstructionRun) -> JsonObject:
    completed_at = run.completed_at
    configuration_sha256 = run.configuration_sha256
    return {
        "completed_at": _encode_datetime(completed_at) if completed_at is not None else None,
        "configuration_sha256": (
            configuration_sha256.value if configuration_sha256 is not None else None
        ),
        "input_observation_ids": [item.value for item in run.input_observation_ids],
        "producer": {
            "implementation": run.producer.implementation,
            "revision": run.producer.revision,
            "version": run.producer.version,
        },
        "run_id": run.run_id.value,
        "started_at": _encode_datetime(run.started_at),
    }


def decode_reconstruction_run(payload: Mapping[str, object]) -> ReconstructionRun:
    data = dict(payload)
    producer = _object(data.get("producer"), "producer")
    completed_at = data.get("completed_at")
    config_sha256 = _optional_string(data.get("configuration_sha256"), "configuration_sha256")
    return ReconstructionRun(
        run_id=ReconstructionRunId(_string(data.get("run_id"), "run_id")),
        producer=ProducerRef(
            implementation=_string(producer.get("implementation"), "producer.implementation"),
            version=_string(producer.get("version"), "producer.version"),
            revision=_optional_string(producer.get("revision"), "producer.revision"),
        ),
        input_observation_ids=tuple(
            ObservationId(value)
            for value in _string_list(data.get("input_observation_ids"), "input_observation_ids")
        ),
        started_at=_decode_datetime(data.get("started_at"), "started_at"),
        completed_at=(
            _decode_datetime(completed_at, "completed_at") if completed_at is not None else None
        ),
        configuration_sha256=(Sha256Digest(config_sha256) if config_sha256 is not None else None),
    )


def encode_derived_artifact_provenance(provenance: DerivedArtifactProvenance) -> JsonObject:
    return {
        "producing_run_id": provenance.producing_run_id.value,
        "source_observation_ids": [item.value for item in provenance.source_observation_ids],
    }


def decode_derived_artifact_provenance(
    payload: Mapping[str, object],
) -> DerivedArtifactProvenance:
    data = dict(payload)
    return DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId(
            _string(data.get("producing_run_id"), "producing_run_id")
        ),
        source_observation_ids=tuple(
            ObservationId(value)
            for value in _string_list(data.get("source_observation_ids"), "source_observation_ids")
        ),
    )
