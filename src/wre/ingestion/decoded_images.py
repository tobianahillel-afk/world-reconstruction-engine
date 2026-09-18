from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from wre.domain.artifact_keys import (
    ArtifactInputFingerprint,
    ArtifactKey,
    ArtifactKeyMaterial,
    derive_artifact_key,
)
from wre.domain.artifact_materialization import (
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
)
from wre.domain.artifacts import ArtifactKind, ArtifactRef
from wre.domain.decoded_images import (
    DECODED_IMAGE_PYRAMID_KIND,
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
    DecodedImagePyramidManifest,
    DecodedImagePyramidSpec,
    build_decoded_image_level_descriptors,
)
from wre.domain.observations import (
    ImageObservation,
    ObservationKind,
    Sha256Digest,
    VideoFrameObservation,
)
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.ingestion.hashing import hash_file_content
from wre.ingestion.keyframes import (
    FFmpegExecutionError,
    FFmpegToolchain,
    FFmpegToolchainIdentity,
    inspect_toolchain,
)

_OUTPUT_KIND = ArtifactKind(DECODED_IMAGE_PYRAMID_KIND)
_IMAGE_INPUT_KIND = ArtifactKind("image.observation")
_VIDEO_FRAME_INPUT_KIND = ArtifactKind("video.frame_observation")


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodedImagePyramidMaterializationRequest:
    source: ImageObservation | VideoFrameObservation
    source_path: Path
    output_root: Path
    artifact_ref: ArtifactRef
    spec: DecodedImagePyramidSpec

    def __post_init__(self) -> None:
        if not isinstance(self.source, (ImageObservation, VideoFrameObservation)):
            raise TypeError(
                "decoded_image_materialization.source must be ImageObservation "
                "or VideoFrameObservation"
            )
        if not isinstance(self.source_path, Path):
            raise TypeError("decoded_image_materialization.source_path must be Path")
        if not isinstance(self.output_root, Path):
            raise TypeError("decoded_image_materialization.output_root must be Path")
        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("decoded_image_materialization.artifact_ref must be ArtifactRef")
        if self.artifact_ref.artifact_kind != _OUTPUT_KIND:
            raise ValueError(
                f"decoded_image_materialization.artifact_ref kind must be {_OUTPUT_KIND.value}"
            )
        if not isinstance(self.spec, DecodedImagePyramidSpec):
            raise TypeError("decoded_image_materialization.spec must be DecodedImagePyramidSpec")


@dataclass(frozen=True, slots=True)
class DecodedImagePyramidMaterializationResult:
    artifact_key: ArtifactKey
    producer: ArtifactProducerIdentity
    manifest: DecodedImagePyramidManifest
    materialization: ArtifactMaterializationMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_key, ArtifactKey):
            raise TypeError("decoded_image_result.artifact_key must be ArtifactKey")
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("decoded_image_result.producer must be ArtifactProducerIdentity")
        if not isinstance(self.manifest, DecodedImagePyramidManifest):
            raise TypeError("decoded_image_result.manifest must be DecodedImagePyramidManifest")
        if not isinstance(self.materialization, ArtifactMaterializationMetadata):
            raise TypeError(
                "decoded_image_result.materialization must be ArtifactMaterializationMetadata"
            )


def _run_command(args: list[str], *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FFmpegExecutionError(f"failed to execute {args[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if len(stderr) > 2_000:
            stderr = stderr[-2_000:]
        raise FFmpegExecutionError(f"{args[0]!r} exited with code {completed.returncode}: {stderr}")
    return completed


def _probe_native_dimensions(
    source_path: Path,
    toolchain: FFmpegToolchain,
) -> tuple[int, int]:
    completed = _run_command(
        [
            toolchain.ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(source_path),
        ],
        timeout_seconds=toolchain.timeout_seconds,
    )
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegExecutionError("ffprobe image dimensions are not valid JSON") from exc
    streams = document.get("streams") if isinstance(document, dict) else None
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise FFmpegExecutionError("ffprobe returned no image stream dimensions")
    width = streams[0].get("width")
    height = streams[0].get("height")
    if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
        raise FFmpegExecutionError("ffprobe returned invalid image stream dimensions")
    return width, height


def _configuration_identity(spec: DecodedImagePyramidSpec) -> ConfigurationIdentity:
    payload = {
        "schema_version": 1,
        "pixel_layout": DecodedImagePixelLayout.RGB8_PACKED.value,
        "orientation_policy": DecodedImageOrientationPolicy.SOURCE_PIXELS.value,
        "minimum_max_edge_px": spec.minimum_max_edge_px,
        "halving": "floor_each_dimension",
        "reference_pixel_format": "rgb24",
        "reference_scaler": "area",
        "reference_autorotate": False,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ConfigurationIdentity(sha256=Sha256Digest(hashlib.sha256(encoded).hexdigest()))


def _reference_artifact_identity(
    source: ImageObservation | VideoFrameObservation,
    spec: DecodedImagePyramidSpec,
    toolchain_identity: FFmpegToolchainIdentity,
) -> tuple[ArtifactProducerIdentity, ArtifactKey]:
    if not isinstance(source, (ImageObservation, VideoFrameObservation)):
        raise TypeError("decoded_image_identity.source must be image or video-frame observation")
    if not isinstance(spec, DecodedImagePyramidSpec):
        raise TypeError("decoded_image_identity.spec must be DecodedImagePyramidSpec")
    if not isinstance(toolchain_identity, FFmpegToolchainIdentity):
        raise TypeError("decoded_image_identity.toolchain_identity must be FFmpegToolchainIdentity")

    producer = ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="wre.ingestion.decoded_images.ffmpeg_reference",
            version=toolchain_identity.canonical_version,
        ),
        configuration=_configuration_identity(spec),
    )
    input_kind = (
        _IMAGE_INPUT_KIND if source.kind is ObservationKind.IMAGE else _VIDEO_FRAME_INPUT_KIND
    )
    artifact_key = derive_artifact_key(
        ArtifactKeyMaterial(
            output_kind=_OUTPUT_KIND,
            input_fingerprints=(
                ArtifactInputFingerprint(
                    artifact_kind=input_kind,
                    sha256=source.asset.sha256,
                ),
            ),
            producer=producer,
        )
    )
    return producer, artifact_key


def _verified_source_path(
    source: ImageObservation | VideoFrameObservation,
    source_path: Path,
) -> Path:
    resolved = source_path.expanduser().resolve(strict=True)
    content_hash = hash_file_content(resolved)
    if (
        content_hash.sha256 != source.asset.sha256
        or content_hash.byte_length != source.asset.byte_length
    ):
        raise ValueError("decoded image source bytes do not match the observation asset")
    return resolved


class FFmpegDecodedImagePyramidMaterializer:
    """Portable software reference path for canonical decoded-image pyramids."""

    def __init__(self, toolchain: FFmpegToolchain) -> None:
        if not isinstance(toolchain, FFmpegToolchain):
            raise TypeError("decoded_image_materializer.toolchain must be FFmpegToolchain")
        self._toolchain = toolchain

    def materialize(
        self,
        request: DecodedImagePyramidMaterializationRequest,
    ) -> DecodedImagePyramidMaterializationResult:
        if not isinstance(request, DecodedImagePyramidMaterializationRequest):
            raise TypeError(
                "decoded_image_materializer.request must be "
                "DecodedImagePyramidMaterializationRequest"
            )

        source_path = _verified_source_path(request.source, request.source_path)
        toolchain_identity = inspect_toolchain(self._toolchain)
        producer, artifact_key = _reference_artifact_identity(
            request.source,
            request.spec,
            toolchain_identity,
        )
        native_width, native_height = _probe_native_dimensions(
            source_path,
            self._toolchain,
        )
        levels = build_decoded_image_level_descriptors(
            native_width,
            native_height,
            request.spec,
        )
        manifest = DecodedImagePyramidManifest(
            source_observation_id=request.source.observation_id,
            source_kind=request.source.kind,
            source_asset_sha256=request.source.asset.sha256,
            pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
            orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
            spec=request.spec,
            levels=levels,
        )

        output_root = request.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        entries: list[ArtifactMaterializationEntry] = []

        with tempfile.TemporaryDirectory(
            prefix="wre-decoded-pyramid-",
            dir=output_root,
        ) as temp_name:
            temp_root = Path(temp_name)
            for level in levels:
                temporary = temp_root / f"level-{level.level_index:06d}.rgb"
                _run_command(
                    [
                        self._toolchain.ffmpeg_path,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-y",
                        "-noautorotate",
                        "-i",
                        str(source_path),
                        "-map",
                        "0:v:0",
                        "-frames:v",
                        "1",
                        "-vf",
                        f"scale={level.width_px}:{level.height_px}:flags=area",
                        "-pix_fmt",
                        "rgb24",
                        "-f",
                        "rawvideo",
                        str(temporary),
                    ],
                    timeout_seconds=self._toolchain.timeout_seconds,
                )
                content_hash = hash_file_content(temporary)
                expected_length = level.width_px * level.height_px * 3
                if content_hash.byte_length != expected_length:
                    raise FFmpegExecutionError(
                        "decoded image level byte length does not match RGB8 dimensions"
                    )

                final_path = output_root.joinpath(*level.relative_path.split("/"))
                final_path.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(final_path)
                entries.append(
                    ArtifactMaterializationEntry(
                        relative_path=level.relative_path,
                        sha256=content_hash.sha256,
                        byte_length=content_hash.byte_length,
                    )
                )

        materialization = ArtifactMaterializationMetadata(
            artifact_ref=request.artifact_ref,
            entries=tuple(entries),
        )
        return DecodedImagePyramidMaterializationResult(
            artifact_key=artifact_key,
            producer=producer,
            manifest=manifest,
            materialization=materialization,
        )
