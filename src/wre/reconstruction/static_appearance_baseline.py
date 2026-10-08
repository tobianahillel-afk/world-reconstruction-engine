"""Fail-closed source-byte and camera preflight for the optional gsplat v1.5.3 route.

This is *not* a trainer or a photometric-compatibility approval. A later V2L18.3
step must independently audit native COLMAP camera/track semantics, multi-view
exposure consistency, the exact external environment and real GPU execution.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from wre.domain.artifacts import ArtifactRef
from wre.domain.observations import ObservationId, Sha256Digest
from wre.reconstruction.colmap_canonical_geometry import _verified_native_model_path
from wre.reconstruction.colmap_geometry_refinement import (
    colmap_native_sparse_model_artifact_ref,
)
from wre.reconstruction.colmap_reconstruction import (
    ColmapReconstructionInput,
    ColmapSparseModelArtifact,
)
from wre.reconstruction.static_appearance_candidate import StaticAppearanceCandidateRequest

GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION = "937e29912570c372bed6747a5c9bf85fed877bae"
GSPLAT_STATIC_APPEARANCE_TRAINER_GIT_BLOB = "6a30be737b5c9af53a140f64faf499d8d4d0933f"
GSPLAT_STATIC_APPEARANCE_REPRESENTATION = "gaussian.splat.ply"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_IMAGE_BYTES = 128 * 1024 * 1024
_MAX_IMAGE_PIXELS = 24_000_000


class GsplatStaticAppearancePreflightError(ValueError):
    """An exact source is incompatible or insufficiently proven for this baseline."""


@dataclass(frozen=True, slots=True, kw_only=True)
class GsplatStaticAppearancePreflightSource:
    """Caller-bound immutable reconstruction and source-image evidence."""

    image_root: Path
    images: tuple[ColmapReconstructionInput, ...]
    native_model_root: Path
    native_model_artifact: ColmapSparseModelArtifact
    native_model_ref: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.image_root, Path):
            raise TypeError("gsplat image_root must be pathlib.Path")
        if not isinstance(self.native_model_root, Path):
            raise TypeError("gsplat native_model_root must be pathlib.Path")
        if not isinstance(self.native_model_artifact, ColmapSparseModelArtifact):
            raise TypeError("gsplat native_model_artifact must be ColmapSparseModelArtifact")
        if not isinstance(self.native_model_ref, ArtifactRef):
            raise TypeError("gsplat native_model_ref must be ArtifactRef")
        if self.native_model_ref != colmap_native_sparse_model_artifact_ref(
            self.native_model_artifact
        ):
            raise GsplatStaticAppearancePreflightError(
                "native COLMAP ArtifactRef does not bind the audited model manifest"
            )
        if not isinstance(self.images, tuple) or not self.images:
            raise TypeError("gsplat images must be a non-empty immutable tuple")
        if any(not isinstance(item, ColmapReconstructionInput) for item in self.images):
            raise TypeError("gsplat images must contain ColmapReconstructionInput values")
        ids = tuple(item.observation.observation_id.value for item in self.images)
        names = tuple(item.image_name for item in self.images)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise GsplatStaticAppearancePreflightError(
                "gsplat input observations must be unique and sorted by ObservationId"
            )
        if len(names) != len(set(names)):
            raise GsplatStaticAppearancePreflightError(
                "gsplat input image names must be unique"
            )


@dataclass(frozen=True, slots=True)
class GsplatStaticAppearanceVerifiedImage:
    observation_id: ObservationId
    image_name: str
    sha256: Sha256Digest
    byte_length: int
    width_px: int
    height_px: int


@dataclass(frozen=True, slots=True)
class GsplatStaticAppearancePreflightEvidence:
    """Byte-level gate only; not native-pose, color-normalization or GPU proof."""

    source_geometry_id: str
    native_model_ref: ArtifactRef
    images: tuple[GsplatStaticAppearanceVerifiedImage, ...]


def _safe_root(root: Path, label: str) -> Path:
    if root.is_symlink():
        raise GsplatStaticAppearancePreflightError(f"{label} cannot be a symlink")
    try:
        resolved = root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GsplatStaticAppearancePreflightError(f"{label} is missing") from exc
    if not resolved.is_dir():
        raise GsplatStaticAppearancePreflightError(f"{label} must be a directory")
    return resolved


def _png_srgb_dimensions(data: bytes) -> tuple[int, int]:
    """Inspect and decode one bounded, 8-bit, noninterlaced sRGB RGB PNG.

    sRGB tagging proves pixel transfer semantics, not capture exposure equality.
    Unsupported PNG variants are deliberately deferred rather than coerced.
    """

    if len(data) > _MAX_IMAGE_BYTES or not data.startswith(_PNG_SIGNATURE):
        raise GsplatStaticAppearancePreflightError("image must be a bounded PNG")
    cursor = len(_PNG_SIGNATURE)
    seen_header = False
    seen_srgb = False
    seen_idat = False
    width = height = 0
    compressed = bytearray()
    while cursor < len(data):
        if cursor + 12 > len(data):
            raise GsplatStaticAppearancePreflightError("truncated PNG chunk")
        size = struct.unpack_from(">I", data, cursor)[0]
        kind = data[cursor + 4 : cursor + 8]
        cursor += 8
        if size > len(data) - cursor - 4:
            raise GsplatStaticAppearancePreflightError("truncated PNG payload")
        payload = data[cursor : cursor + size]
        digest = struct.unpack_from(">I", data, cursor + size)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != digest:
            raise GsplatStaticAppearancePreflightError("PNG chunk checksum mismatch")
        cursor += size + 4

        if not seen_header and kind != b"IHDR":
            raise GsplatStaticAppearancePreflightError("PNG must begin with IHDR")
        if kind == b"IHDR":
            if seen_header or size != 13:
                raise GsplatStaticAppearancePreflightError("PNG IHDR is invalid")
            width, height, depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if (
                width < 1
                or height < 1
                or width * height > _MAX_IMAGE_PIXELS
                or (depth, color_type, comp, filt, interlace) != (8, 2, 0, 0, 0)
            ):
                raise GsplatStaticAppearancePreflightError(
                    "baseline supports only bounded noninterlaced RGB8 PNG"
                )
            seen_header = True
        elif kind == b"sRGB":
            if seen_srgb or seen_idat or size != 1 or payload[0] > 3:
                raise GsplatStaticAppearancePreflightError("invalid sRGB PNG declaration")
            seen_srgb = True
        elif kind in {b"iCCP", b"gAMA", b"cHRM", b"acTL", b"PLTE", b"tRNS"}:
            raise GsplatStaticAppearancePreflightError(
                "ambiguous/unsupported PNG color or animation metadata"
            )
        elif kind == b"IDAT":
            if not seen_srgb:
                raise GsplatStaticAppearancePreflightError(
                    "PNG must declare an explicit sRGB transfer before pixels"
                )
            seen_idat = True
            compressed.extend(payload)
        elif kind == b"IEND":
            if size != 0 or not seen_idat or cursor != len(data):
                raise GsplatStaticAppearancePreflightError("invalid PNG end or trailing bytes")
            break
        elif kind[0] & 0x20 == 0:
            raise GsplatStaticAppearancePreflightError("unsupported critical PNG chunk")
    else:
        raise GsplatStaticAppearancePreflightError("PNG IEND is missing")

    # Check that the pixel stream is genuinely decodable, not just an intact IHDR.
    expected = height * (1 + 3 * width)
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(bytes(compressed), expected + 1)
        if len(raw) != expected or not decoder.eof or decoder.unused_data:
            raise GsplatStaticAppearancePreflightError("PNG raster size is invalid")
        if any(raw[row * (1 + 3 * width)] > 4 for row in range(height)):
            raise GsplatStaticAppearancePreflightError("PNG filter is invalid")
    except zlib.error as exc:
        raise GsplatStaticAppearancePreflightError("PNG raster is corrupt") from exc
    return width, height


def preflight_gsplat_static_appearance_inputs(
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
) -> GsplatStaticAppearancePreflightEvidence:
    """Reject image/source mismatches before invoking any optional trainer.

    This gate does not assert that a native COLMAP model matches canonical pose
    values or that cameras have common exposure. Those require separate evidence.
    """

    if not isinstance(request, StaticAppearanceCandidateRequest):
        raise TypeError("gsplat request must be StaticAppearanceCandidateRequest")
    if not isinstance(source, GsplatStaticAppearancePreflightSource):
        raise TypeError("gsplat source must be GsplatStaticAppearancePreflightSource")
    if request.output_representation.value != GSPLAT_STATIC_APPEARANCE_REPRESENTATION:
        raise GsplatStaticAppearancePreflightError("unsupported gsplat output representation")
    if source.native_model_ref not in request.supporting_artifacts:
        raise GsplatStaticAppearancePreflightError(
            "request must retain the native COLMAP model ArtifactRef"
        )
    if source.native_model_artifact.num_registered_images != len(
        request.source_geometry.camera_solutions
    ):
        raise GsplatStaticAppearancePreflightError(
            "native sparse registered image count and camera count disagree"
        )
    if source.native_model_artifact.num_points3d < 1:
        raise GsplatStaticAppearancePreflightError(
            "native COLMAP model must contain real sparse point evidence"
        )
    _verified_native_model_path(source.native_model_root, source.native_model_artifact)
    image_root = _safe_root(source.image_root, "gsplat image_root")

    cameras = request.source_geometry.camera_solutions
    camera_ids = tuple(camera.observation_id for camera in cameras)
    selected_ids = tuple(item.observation.observation_id for item in source.images)
    if camera_ids != request.source_observation_ids or selected_ids != camera_ids:
        raise GsplatStaticAppearancePreflightError(
            "source images must match every canonical CameraSolution and requested ObservationId"
        )

    verified: list[GsplatStaticAppearanceVerifiedImage] = []
    for item, camera in zip(source.images, cameras, strict=True):
        name = item.image_name
        relative = PurePosixPath(name)
        if (
            len(relative.parts) != 1
            or relative.name != name
            or not name.lower().endswith(".png")
        ):
            raise GsplatStaticAppearancePreflightError(
                "baseline requires a safe single-level PNG image filename"
            )
        if camera.projection_model.value != "pinhole" or len(
            camera.intrinsic_parameters
        ) != 4:
            raise GsplatStaticAppearancePreflightError(
                "gsplat baseline supports only native undistorted PINHOLE cameras"
            )
        fx, fy, _cx, _cy = camera.intrinsic_parameters
        if fx <= 0 or fy <= 0:
            raise GsplatStaticAppearancePreflightError("camera focal length is invalid")

        source_file = image_root / name
        if source_file.is_symlink() or item.source_path.is_symlink():
            raise GsplatStaticAppearancePreflightError("input image cannot be a symlink")
        try:
            actual = source_file.resolve(strict=True)
            bound = item.source_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise GsplatStaticAppearancePreflightError("source image is missing") from exc
        if actual.parent != image_root or bound != actual or not actual.is_file():
            raise GsplatStaticAppearancePreflightError(
                "source image is not the exact bound file inside image_root"
            )
        data = actual.read_bytes()
        digest = Sha256Digest(hashlib.sha256(data).hexdigest())
        asset = item.observation.asset
        if len(data) != asset.byte_length or digest != asset.sha256:
            raise GsplatStaticAppearancePreflightError(
                "source image size or SHA-256 differs from the original Observation"
            )
        width, height = _png_srgb_dimensions(data)
        if (
            width != camera.dimensions.width_px
            or height != camera.dimensions.height_px
        ):
            raise GsplatStaticAppearancePreflightError(
                "image dimensions disagree with WRE camera calibration; no implicit resize"
            )
        verified.append(
            GsplatStaticAppearanceVerifiedImage(
                observation_id=item.observation.observation_id,
                image_name=name,
                sha256=digest,
                byte_length=len(data),
                width_px=width,
                height_px=height,
            )
        )

    return GsplatStaticAppearancePreflightEvidence(
        source_geometry_id=request.source_geometry.geometry_solution_id.value,
        native_model_ref=source.native_model_ref,
        images=tuple(verified),
    )


__all__ = [
    "GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION",
    "GSPLAT_STATIC_APPEARANCE_TRAINER_GIT_BLOB",
    "GSPLAT_STATIC_APPEARANCE_REPRESENTATION",
    "GsplatStaticAppearancePreflightError",
    "GsplatStaticAppearancePreflightSource",
    "GsplatStaticAppearanceVerifiedImage",
    "GsplatStaticAppearancePreflightEvidence",
    "preflight_gsplat_static_appearance_inputs",
]
