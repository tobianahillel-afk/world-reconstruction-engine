"""Fail-closed source-byte and camera preflight for the optional gsplat v1.5.3 route.

This is *not* a trainer or a photometric-compatibility approval. A later V2L18.3
step must independently audit native COLMAP camera/track semantics, multi-view
exposure consistency, the exact external environment and real GPU execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from wre.domain.appearance import APPEARANCE_MODEL_ARTIFACT_KIND
from wre.domain.artifact_materialization import (
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
)
from wre.domain.artifacts import ArtifactRef
from wre.domain.color_conventions import (
    ColorConversionStatus,
    DecodedPixelColorEncoding,
)
from wre.domain.decoded_images import (
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
)
from wre.domain.observations import ObservationId, ObservationKind, Sha256Digest
from wre.domain.photometric_normalization import PhotometricNormalizationStatus
from wre.domain.photometric_validity import (
    PhotometricCompatibilityAssessment,
    PhotometricCompatibilityStatus,
    assess_photometric_compatibility,
)
from wre.reconstruction.colmap_canonical_geometry import (
    CanonicalColmapSparseModel,
    _verified_native_model_path,
    canonicalize_colmap_sparse_model,
    colmap_sparse_model_content_identity,
)
from wre.reconstruction.colmap_environment import ColmapEnvironmentIdentity
from wre.reconstruction.colmap_features import ColmapFeatureExtractionResult
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
            raise GsplatStaticAppearancePreflightError("gsplat input image names must be unique")


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
    by_observation = {camera.observation_id: camera for camera in cameras}
    camera_ids = set(by_observation)
    selected_ids = tuple(item.observation.observation_id for item in source.images)
    if (
        len(by_observation) != len(cameras)
        or set(selected_ids) != camera_ids
        or selected_ids != request.source_observation_ids
    ):
        raise GsplatStaticAppearancePreflightError(
            "source images must match every canonical CameraSolution and requested ObservationId"
        )

    verified: list[GsplatStaticAppearanceVerifiedImage] = []
    for item in source.images:
        camera = by_observation[item.observation.observation_id]
        name = item.image_name
        relative = PurePosixPath(name)
        if len(relative.parts) != 1 or relative.name != name or not name.lower().endswith(".png"):
            raise GsplatStaticAppearancePreflightError(
                "baseline requires a safe single-level PNG image filename"
            )
        if camera.projection_model.value != "pinhole" or len(camera.intrinsic_parameters) != 4:
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
        if actual.stat().st_size > _MAX_IMAGE_BYTES:
            raise GsplatStaticAppearancePreflightError("input PNG exceeds the bounded size")
        data = actual.read_bytes()
        digest = Sha256Digest(hashlib.sha256(data).hexdigest())
        asset = item.observation.asset
        if len(data) != asset.byte_length or digest != asset.sha256:
            raise GsplatStaticAppearancePreflightError(
                "source image size or SHA-256 differs from the original Observation"
            )
        width, height = _png_srgb_dimensions(data)
        if width != camera.dimensions.width_px or height != camera.dimensions.height_px:
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


def verify_gsplat_static_appearance_native_geometry(
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
    *,
    features: ColmapFeatureExtractionResult,
    expected_environment: ColmapEnvironmentIdentity,
    module: object | None = None,
) -> CanonicalColmapSparseModel:
    """Require independently decoded native COLMAP geometry to equal WRE input.

    This is a narrow first-baseline gate: only an unchanged canonical COLMAP
    source model is accepted. Refined/learned geometry needs separate audited
    native correspondence; the equality check must never coerce a frame or pose.
    No gsplat or CUDA package is loaded.
    """

    preflight_gsplat_static_appearance_inputs(request, source)
    if not isinstance(features, ColmapFeatureExtractionResult):
        raise TypeError("gsplat native geometry features must be ColmapFeatureExtractionResult")
    if not isinstance(expected_environment, ColmapEnvironmentIdentity):
        raise TypeError("gsplat native geometry environment must be ColmapEnvironmentIdentity")

    source_name_map = {item.observation.observation_id: item.image_name for item in source.images}
    feature_name_map = {item.observation_id: item.image_name for item in features.images}
    if source_name_map != feature_name_map:
        raise GsplatStaticAppearancePreflightError(
            "native COLMAP feature/image naming is not the exact source ObservationId mapping"
        )

    canonical = canonicalize_colmap_sparse_model(
        output_path=source.native_model_root,
        model_artifact=source.native_model_artifact,
        features=features,
        expected_environment=expected_environment,
        module=module,
    )
    if canonical.source_model_identity_sha256 != colmap_sparse_model_content_identity(
        source.native_model_artifact
    ):
        raise GsplatStaticAppearancePreflightError(
            "native COLMAP result does not match the verified sparse model identity"
        )
    geometry = request.source_geometry
    if (
        geometry.geometry_solution != canonical.geometry_solution
        or geometry.camera_solutions != canonical.camera_solutions
        or geometry.depth_fields
        or geometry.point_maps != (canonical.point_map,)
    ):
        raise GsplatStaticAppearancePreflightError(
            "native COLMAP camera poses, calibration, points, frame or scale "
            "do not match the exact source GeometrySolutionCandidate"
        )
    return canonical


def verify_gsplat_static_appearance_source_photometry(
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
    assessments: tuple[PhotometricCompatibilityAssessment, ...],
) -> tuple[Sha256Digest, ...]:
    """Require exact V2L17 source-color evidence for every *unchanged* training PNG.

    The pinned gsplat reference trainer reads the original PNG pixels, not V2L17
    normalized linear-light buffers. Consequently this gate permits only identity
    EV/RGB adjustments and a non-resized, source-orientation sRGB RGB8 declaration.
    It does not assert equal scene radiance or calibrate exposure across cameras.
    A later executable route must independently prove multi-view suitability.
    """

    verified = preflight_gsplat_static_appearance_inputs(request, source)
    if not isinstance(assessments, tuple):
        raise TypeError("gsplat photometric assessments must be an immutable tuple")
    if len(assessments) != len(verified.images):
        raise GsplatStaticAppearancePreflightError(
            "each gsplat source observation requires one photometric assessment"
        )

    identities: list[Sha256Digest] = []
    for image, assessment in zip(verified.images, assessments, strict=True):
        if not isinstance(assessment, PhotometricCompatibilityAssessment):
            raise TypeError("gsplat photometric evidence must be PhotometricCompatibilityAssessment")
        if (
            assessment != assess_photometric_compatibility(assessment.compatibility_input)
            or assessment.status is not PhotometricCompatibilityStatus.COMPATIBLE
        ):
            raise GsplatStaticAppearancePreflightError(
                "gsplat source photometry is unresolved, incompatible or inconsistent"
            )

        color = assessment.compatibility_input.color_assessment
        normalization = assessment.compatibility_input.normalization_assessment
        if (
            color.status is not ColorConversionStatus.READY
            or color.plan is None
            or normalization.status is not PhotometricNormalizationStatus.READY
            or normalization.plan is None
        ):
            raise GsplatStaticAppearancePreflightError(
                "gsplat source photometry requires ready V2L17 plans"
            )

        decoded = color.request
        manifest = decoded.decoded_manifest
        if (
            manifest.source_observation_id != image.observation_id
            or manifest.source_asset_sha256 != image.sha256
            or manifest.source_kind is not ObservationKind.IMAGE
        ):
            raise GsplatStaticAppearancePreflightError(
                "gsplat photometric evidence does not identify the exact source PNG"
            )
        if (
            manifest.orientation_policy is not DecodedImageOrientationPolicy.SOURCE_PIXELS
            or manifest.pixel_layout is not DecodedImagePixelLayout.RGB8_PACKED
            or decoded.decoded_encoding is not DecodedPixelColorEncoding.SRGB_FULL_RGB8
            or decoded.decoded_level_index != 0
        ):
            raise GsplatStaticAppearancePreflightError(
                "gsplat source requires unrotated, original-level sRGB RGB8 evidence"
            )
        level = manifest.levels[0]
        if (level.width_px, level.height_px) != (image.width_px, image.height_px):
            raise GsplatStaticAppearancePreflightError(
                "gsplat source photometry cannot silently resize or reorient PNG pixels"
            )
        factors = normalization.plan.request.factors
        if (
            factors.exposure_adjustment_ev != 0.0
            or factors.white_balance_rgb_gains != (1.0, 1.0, 1.0)
        ):
            raise GsplatStaticAppearancePreflightError(
                "gsplat reference trainer reads raw PNG; non-identity V2L17 "
                "normalization is not applied to those training bytes"
            )
        identities.append(assessment.identity)

    return tuple(identities)


@dataclass(frozen=True, slots=True, kw_only=True)
class GsplatStaticAppearanceTrainingProfile:
    """Audited single-GPU overrides for the exact upstream reference trainer.

    This is an immutable *configuration contract*, not a training runner or evidence
    that its external Torch/CUDA/import environment has been approved. The eventual
    optional runner must reject upstream source drift before constructing Config.
    """

    max_steps: int = 30_000
    test_every: int = 8
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("gsplat training profile schema_version must equal 1")
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("gsplat max_steps must be a positive integer")
        if type(self.test_every) is not int or self.test_every < 2:
            raise ValueError("gsplat test_every must be an integer >= 2")

    def reference_overrides(self) -> dict[str, object]:
        """Values for exact-source examples.simple_trainer.Config, not CLI guesses."""

        return {
            "app_opt": False,
            "camera_model": "pinhole",
            "ckpt": None,
            "compression": None,
            "data_factor": 1,
            "depth_loss": False,
            "disable_video": True,
            "disable_viewer": True,
            "eval_steps": [],
            "global_scale": 1.0,
            "init_type": "sfm",
            "max_steps": self.max_steps,
            "normalize_world_space": False,
            "patch_size": None,
            "ply_steps": [self.max_steps],
            "pose_noise": 0.0,
            "pose_opt": False,
            "random_bkgd": False,
            "save_ply": True,
            "save_steps": [],
            "steps_scaler": 1.0,
            "tb_every": 0,
            "tb_save_image": False,
            "test_every": self.test_every,
            "use_bilateral_grid": False,
            "use_fused_bilagrid": False,
            "with_eval3d": False,
            "with_ut": False,
        }

    def canonical_document(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trainer_source_revision": GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION,
            "trainer_git_blob_sha1": GSPLAT_STATIC_APPEARANCE_TRAINER_GIT_BLOB,
            "profile": "wre.gsplat.static_appearance.safe_single_gpu",
            "upstream_config_overrides": self.reference_overrides(),
        }

    @property
    def configuration_sha256(self) -> Sha256Digest:
        encoded = json.dumps(
            self.canonical_document(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        return Sha256Digest(hashlib.sha256(encoded).hexdigest())

    @property
    def expected_ply_relative_path(self) -> str:
        """Exact upstream export location at the last 0-based training step."""

        return f"ply/point_cloud_{self.max_steps - 1}.ply"


_MAX_GSPLAT_PLY_HEADER_BYTES = 16 * 1024
_MAX_GSPLAT_PLY_PROPERTIES = 256
_MAX_GSPLAT_PLY_VERTICES = 100_000_000


def materialize_verified_gsplat_ply(
    *,
    artifact_ref: ArtifactRef,
    output_root: Path,
    profile: GsplatStaticAppearanceTrainingProfile,
) -> ArtifactMaterializationMetadata:
    """Audit one real uncompressed gsplat exporter payload without importing Torch.

    The input is a private, already-produced training workspace. This function
    cannot claim a training run, check native cameras, or approve radiometry.
    It only validates the exact upstream PLY layout and binds its real bytes to
    the canonical WRE appearance artifact materialization contract.
    """

    if not isinstance(artifact_ref, ArtifactRef):
        raise TypeError("gsplat output artifact_ref must be ArtifactRef")
    if artifact_ref.artifact_kind != APPEARANCE_MODEL_ARTIFACT_KIND:
        raise GsplatStaticAppearancePreflightError(
            "gsplat output must use the canonical appearance.static ArtifactKind"
        )
    if not isinstance(profile, GsplatStaticAppearanceTrainingProfile):
        raise TypeError("gsplat output profile must be GsplatStaticAppearanceTrainingProfile")
    if not isinstance(output_root, Path):
        raise TypeError("gsplat output_root must be pathlib.Path")

    root = _safe_root(output_root, "gsplat output_root")
    ply_directory = root / "ply"
    if ply_directory.is_symlink() or not ply_directory.is_dir():
        raise GsplatStaticAppearancePreflightError(
            "gsplat output PLY directory must be a real directory"
        )
    file_path = root / profile.expected_ply_relative_path
    if file_path.is_symlink() or not file_path.is_file():
        raise GsplatStaticAppearancePreflightError("gsplat expected final PLY file is missing")
    if file_path.resolve(strict=True).parent != ply_directory.resolve(strict=True):
        raise GsplatStaticAppearancePreflightError("gsplat PLY escaped its output directory")

    digest = hashlib.sha256()
    with file_path.open("rb") as stream:

        def read_header_line() -> str:
            raw = stream.readline(_MAX_GSPLAT_PLY_HEADER_BYTES + 1)
            if (
                not raw
                or len(raw) > _MAX_GSPLAT_PLY_HEADER_BYTES
                or stream.tell() > _MAX_GSPLAT_PLY_HEADER_BYTES
                or not raw.endswith(b"\n")
            ):
                raise GsplatStaticAppearancePreflightError(
                    "gsplat PLY header is truncated or oversized"
                )
            digest.update(raw)
            try:
                return raw.decode("ascii").removesuffix("\n")
            except UnicodeDecodeError as exc:
                raise GsplatStaticAppearancePreflightError(
                    "gsplat PLY header must be ASCII"
                ) from exc

        if read_header_line() != "ply":
            raise GsplatStaticAppearancePreflightError("gsplat output is not PLY")
        if read_header_line() != "format binary_little_endian 1.0":
            raise GsplatStaticAppearancePreflightError(
                "gsplat output must be uncompressed little-endian binary PLY"
            )
        element_line = read_header_line()
        parts = element_line.split(" ")
        if (
            len(parts) != 3
            or parts[:2] != ["element", "vertex"]
            or not parts[2].isascii()
            or not parts[2].isdecimal()
            or parts[2].startswith("0")
        ):
            raise GsplatStaticAppearancePreflightError(
                "gsplat PLY must declare one non-empty vertex element"
            )
        count = int(parts[2])
        if count < 1 or count > _MAX_GSPLAT_PLY_VERTICES:
            raise GsplatStaticAppearancePreflightError(
                "gsplat PLY vertex count is outside the audited bounded profile"
            )

        properties: list[str] = []
        while True:
            line = read_header_line()
            if line == "end_header":
                break
            if not line.startswith("property float "):
                raise GsplatStaticAppearancePreflightError(
                    "gsplat PLY contains an unexpected element, type or header field"
                )
            properties.append(line[len("property float ") :])
            if len(properties) > _MAX_GSPLAT_PLY_PROPERTIES:
                raise GsplatStaticAppearancePreflightError(
                    "gsplat PLY declares too many float properties"
                )

        prefix = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
        suffix = ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        if len(properties) < len(prefix) + len(suffix):
            raise GsplatStaticAppearancePreflightError("gsplat PLY is missing Gaussian properties")
        remainder = properties[len(prefix) : -len(suffix)]
        if (
            properties[: len(prefix)] != prefix
            or properties[-len(suffix) :] != suffix
            or len(remainder) % 3 != 0
            or remainder != [f"f_rest_{index}" for index in range(len(remainder))]
        ):
            raise GsplatStaticAppearancePreflightError(
                "gsplat PLY property schema differs from the exact exporter"
            )

        header_length = stream.tell()
        stride = len(properties) * 4
        expected_size = header_length + count * stride
        size = file_path.stat().st_size
        if size != expected_size:
            raise GsplatStaticAppearancePreflightError(
                "gsplat PLY payload is truncated or contains unexpected bytes"
            )
        # Check the whole record stream, not just a header, and hash the same
        # bytes we have inspected. The bounded read limits transient memory.
        remaining = count * stride
        while remaining:
            chunk = stream.read(min(remaining, stride * max(1, 4096 // stride)))
            if not chunk or len(chunk) % stride:
                raise GsplatStaticAppearancePreflightError("gsplat PLY vertex data is incomplete")
            if any(not math.isfinite(value) for (value,) in struct.iter_unpack("<f", chunk)):
                raise GsplatStaticAppearancePreflightError(
                    "gsplat PLY contains non-finite Gaussian parameters"
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise GsplatStaticAppearancePreflightError("gsplat PLY contains trailing bytes")
    if file_path.stat().st_size != size:
        raise GsplatStaticAppearancePreflightError(
            "gsplat PLY changed during materialization verification"
        )
    return ArtifactMaterializationMetadata(
        artifact_ref=artifact_ref,
        entries=(
            ArtifactMaterializationEntry(
                relative_path=profile.expected_ply_relative_path,
                sha256=Sha256Digest(digest.hexdigest()),
                byte_length=size,
            ),
        ),
    )


__all__ = [
    "GSPLAT_STATIC_APPEARANCE_REPRESENTATION",
    "GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION",
    "GSPLAT_STATIC_APPEARANCE_TRAINER_GIT_BLOB",
    "GsplatStaticAppearancePreflightError",
    "GsplatStaticAppearancePreflightEvidence",
    "GsplatStaticAppearancePreflightSource",
    "GsplatStaticAppearanceTrainingProfile",
    "GsplatStaticAppearanceVerifiedImage",
    "materialize_verified_gsplat_ply",
    "preflight_gsplat_static_appearance_inputs",
    "verify_gsplat_static_appearance_native_geometry",
    "verify_gsplat_static_appearance_source_photometry",
]
