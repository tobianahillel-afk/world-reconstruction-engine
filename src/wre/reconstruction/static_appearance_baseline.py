"""Fail-closed source-byte and camera preflight for the optional gsplat v1.5.3 route.

This is *not* a trainer or a photometric-compatibility approval. A later V2L18.3
step must independently audit native COLMAP camera/track semantics, multi-view
exposure consistency, the exact external environment and real GPU execution.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from wre.domain.appearance import APPEARANCE_MODEL_ARTIFACT_KIND, AppearanceModel
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
from wre.domain.producer_identity import ArtifactProducerIdentity
from wre.domain.source_photometry import SourcePhotometryInterpretationStatus
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
from wre.reconstruction.static_appearance_candidate import (
    StaticAppearanceCandidateRequest,
    StaticAppearanceCandidateResult,
    build_static_appearance_candidate_result,
)

# Reviewed MIT fused-ssim source at rahul-goel/fused-ssim revision
# 328dc9836f513d00c4b5bc38fe30478b4435cbb5. This eagerly imported
# CUDA loss is mandatory for the selected upstream trainer. These are the
# executable/build source files plus license and README, not the full Git tree
# or proof of an installed CUDA extension or its binary/toolchain license.
# Reviewed nerfview v0.1.2 at nerfstudio-project/nerfview commit
# 4538024fe0d15fd1a0e4d760f3695fc44ca72787. These are the eager Python
# package/import and distribution metadata files, not a full Git/source-tree
# lock or approval of viser, numpy, jaxtyping or their transitive dependencies.
_GSPLAT_NERFVIEW_REVISION = "4538024fe0d15fd1a0e4d760f3695fc44ca72787"
_GSPLAT_NERFVIEW_SOURCE_FILES: tuple[tuple[str, int, str], ...] = (
    ("README.md", 5579, "44c91608517a0e87e3c13c52d09305f22f665273"),
    ("nerfview/__init__.py", 504, "619c0513faf897bf58ed1c4698f6b5fdde0cac14"),
    ("nerfview/_renderer.py", 6454, "ad17ca39fc00bf41b0b99b7b7b621ac9f7df805a"),
    ("nerfview/render_panel.py", 54931, "a81fa3ea9a468a94ebea6a8de931fbbce5d4"),
    ("nerfview/version.py", 22, "b3f4756216d06217d275110297a69bbb7ea74a59"),
    ("nerfview/viewer.py", 10564, "8cf32290872a670a351cf328249782c5cec4b550"),
    ("pyproject.toml", 849, "4fa1b90b2b1d38e17d464bb7d12cec839e1aa023"),
)


_GSPLAT_FUSED_SSIM_REVISION = "328dc9836f513d00c4b5bc38fe30478b4435cbb5"
_GSPLAT_FUSED_SSIM_SOURCE_FILES: tuple[tuple[str, int, str], ...] = (
    ("LICENSE", 1067, "541f73944912fde14ffb971f22ba516042b95ab7"),
    ("README.md", 3960, "b2cd9676ef94dbd0f6488bd08b71fc6b5a159310"),
    ("ext.cpp", 179, "3eeece12ad64d46e79c810ba7569325f96e35573"),
    ("fused_ssim/__init__.py", 1388, "776fed79e4ef93fb40ff768df7606ab2b0efa192"),
    ("setup.py", 2348, "47d2689243b49a9005e061edd3a95e30170d279b"),
    ("ssim.cu", 18431, "2df752d7fcedc008f34de3ed185a6dc507fa4476"),
    ("ssim.h", 508, "adb00543b71403bd5350b0734f1ced0bee8da2fe"),
)


GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION = "937e29912570c372bed6747a5c9bf85fed877bae"
GSPLAT_STATIC_APPEARANCE_TRAINER_GIT_BLOB = "6a30be737b5c9af53a140f64faf499d8d4d0933f"
GSPLAT_STATIC_APPEARANCE_SOURCE_TREE = "90c3f0b2352e6d2725bcba1ef0407168f922c0fa"
GSPLAT_STATIC_APPEARANCE_REPRESENTATION = "gaussian.splat.ply"
# The exact simple_trainer initializes SH of degree 3 by default:
# shN has ((degree + 1) ** 2 - 1) basis terms, each with 3 RGB coefficients.
# This is a payload-layout invariant, not evidence of appearance quality.
_GSPLAT_REFERENCE_SH_DEGREE = 3
_GSPLAT_REFERENCE_SH_REST_FLOATS = ((_GSPLAT_REFERENCE_SH_DEGREE + 1) ** 2 - 1) * 3
GSPLAT_REFERENCE_PYCOLMAP_REVISION = "cc7ea4b7301720ac29287dbe450952511b32125e"

# This is the pure-Python SceneManager reader pinned by the exact upstream
# examples/requirements.txt. It is *not* WRE's native PyCOLMAP 4.2.0 runtime.
# The pinned MIT source is a necessary input for an isolated external environment,
# not proof of native-model cross-read, pip transitive closure or GPU execution.
_GSPLAT_REFERENCE_PYCOLMAP_FILES: tuple[tuple[str, int, str], ...] = (
    ("LICENSE.txt", 1084, "5156d3d49e0c312561c59680658b6261f635abe3"),
    ("README.md", 490, "7f6769bb42537467eb155356012441e90b3c8920"),
    ("pycolmap/__init__.py", 178, "62e55b8d45b45f2255dc116c367e393f4c27e353"),
    ("pycolmap/camera.py", 9527, "29f2fcb7815a2ea66193520c02d5e0d4bf4b13f0"),
    ("pycolmap/database.py", 10081, "c11948d8ec464c567c1581e6dd588350efa4c7a5"),
    ("pycolmap/image.py", 944, "14efa32b0a91f116cbd7836b6480a60b10371196"),
    ("pycolmap/rotation.py", 11595, "f0b4e811620e9668e8a44b1fd15e0574a7307f6a"),
    (
        "pycolmap/scene_manager.py",
        26996,
        "352f051f71aad8bbad70c0b6cc63b83d3ab90ce5",
    ),
    ("pyproject.toml", 359, "610d833fc084bc48e114c36f52f2a9e3b99945a6"),
)


# Exact git-blob identities at the reviewed gsplat v1.5.3 source revision.
# These are the *reference trainer's entrypoints*, not proof that an arbitrary
# installed gsplat package, Python environment, or CUDA extension is approved.
_GSPLAT_REFERENCE_SOURCE_FILES: tuple[tuple[str, int, str], ...] = (
    ("LICENSE", 11345, "1dc520ba6aa1ff169e95250cf0398beb3757590a"),
    (
        "examples/datasets/colmap.py",
        18447,
        "6c21f2c663b60d9dc38471a9963a3b5d092e5ed2",
    ),
    (
        "examples/datasets/normalize.py",
        4650,
        "681623b311625065744813f565666e9a8154a33c",
    ),
    (
        "examples/datasets/traj.py",
        9447,
        "9e8a2d69dc969e78d5ea019465fe39c10618cf5c",
    ),
    (
        "examples/gsplat_viewer.py",
        9675,
        "e47d75a83f0423b1dab11dbbf75d28b37f6bda07",
    ),
    (
        "examples/requirements.txt",
        677,
        "ea0a940ea796e486aa68e8fc284ee01cb5a27662",
    ),
    ("examples/simple_trainer.py", 49728, GSPLAT_STATIC_APPEARANCE_TRAINER_GIT_BLOB),
    (
        "examples/utils.py",
        7519,
        "80f8e35f364aa884af00106b352631cc04b10d43",
    ),
    (
        "gsplat/version.py",
        22,
        "a06ff4e08777642c97011d6c993690dba5a7a02f",
    ),
    ("setup.py", 4601, "f152008e5f35604b38e0ec6c98e8e17c525c1bcb"),
)
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


def _git_blob_sha1(data: bytes) -> str:
    """Git's blob object identity; do not confuse it with a SHA-256 artifact digest."""

    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def verify_gsplat_static_appearance_reference_sources(
    source_root: Path,
) -> tuple[ArtifactMaterializationEntry, ...]:
    """Verify reviewed *trainer entrypoint* bytes in an offline local source tree.

    Check the exact upstream requirements/parser/helper files as well as the
    trainer before any future external runner imports them. This is NOT a
    full-tree, transitive-license, Python ABI, Torch/CUDA, or GPU execution
    approval. No source download, subprocess, import, or code execution occurs.
    """

    if not isinstance(source_root, Path):
        raise TypeError("gsplat reference source_root must be pathlib.Path")
    root = _safe_root(source_root, "gsplat reference source_root")
    entries: list[ArtifactMaterializationEntry] = []
    for relative, expected_size, expected_git_blob in _GSPLAT_REFERENCE_SOURCE_FILES:
        candidate = root
        for part in PurePosixPath(relative).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise GsplatStaticAppearancePreflightError(
                    f"gsplat reference source path contains symlink: {relative}"
                )
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise GsplatStaticAppearancePreflightError(
                f"gsplat reference source file missing: {relative}"
            ) from exc
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise GsplatStaticAppearancePreflightError(
                f"gsplat reference source path invalid: {relative}"
            )
        if resolved.stat().st_size != expected_size:
            raise GsplatStaticAppearancePreflightError(
                f"gsplat reference source byte length differs: {relative}"
            )
        data = resolved.read_bytes()
        if len(data) != expected_size or _git_blob_sha1(data) != expected_git_blob:
            raise GsplatStaticAppearancePreflightError(
                f"gsplat reference source Git blob differs: {relative}"
            )
        if resolved.stat().st_size != expected_size:
            raise GsplatStaticAppearancePreflightError(
                f"gsplat reference source changed during verification: {relative}"
            )
        entries.append(
            ArtifactMaterializationEntry(
                relative_path=relative,
                sha256=Sha256Digest(hashlib.sha256(data).hexdigest()),
                byte_length=len(data),
            )
        )
    return tuple(entries)


def verify_gsplat_reference_pycolmap_sources(
    source_root: Path,
) -> tuple[ArtifactMaterializationEntry, ...]:
    """Verify the exact MIT SceneManager fork's offline source closure.

    The official gsplat trainer imports SceneManager from this fork, not from
    the independent WRE PyCOLMAP 4.2.0 package. This checks source bytes only:
    it does not import/install the fork, prove native sparse-model compatibility,
    resolve numpy/scipy, or authorize a trainer/GPU execution environment.
    """

    if not isinstance(source_root, Path):
        raise TypeError("gsplat reference pycolmap source_root must be pathlib.Path")
    root = _safe_root(source_root, "gsplat reference pycolmap source_root")
    package = root / "pycolmap"
    if package.is_symlink() or not package.is_dir():
        raise GsplatStaticAppearancePreflightError(
            "reference pycolmap package must be a real directory"
        )
    expected = {
        relative
        for relative, _size, _git_sha in _GSPLAT_REFERENCE_PYCOLMAP_FILES
        if relative.startswith("pycolmap/")
    }
    actual: set[str] = set()
    for path in package.rglob("*"):
        if path.is_symlink() or not path.is_file():
            raise GsplatStaticAppearancePreflightError(
                "reference pycolmap contains an unexpected directory or symlink"
            )
        actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise GsplatStaticAppearancePreflightError(
            "reference pycolmap package source closure differs from the reviewed commit"
        )

    entries: list[ArtifactMaterializationEntry] = []
    for relative, expected_length, git_blob in _GSPLAT_REFERENCE_PYCOLMAP_FILES:
        candidate = root
        for part in PurePosixPath(relative).parts:
            candidate /= part
            if candidate.is_symlink():
                raise GsplatStaticAppearancePreflightError(
                    f"reference pycolmap source path contains symlink: {relative}"
                )
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise GsplatStaticAppearancePreflightError(
                f"reference pycolmap source file missing: {relative}"
            ) from exc
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise GsplatStaticAppearancePreflightError(
                f"reference pycolmap source path invalid: {relative}"
            )
        if resolved.stat().st_size != expected_length:
            raise GsplatStaticAppearancePreflightError(
                f"reference pycolmap source byte length differs: {relative}"
            )
        data = resolved.read_bytes()
        if len(data) != expected_length or _git_blob_sha1(data) != git_blob:
            raise GsplatStaticAppearancePreflightError(
                f"reference pycolmap source Git blob differs: {relative}"
            )
        if resolved.stat().st_size != expected_length:
            raise GsplatStaticAppearancePreflightError(
                f"reference pycolmap source changed during verification: {relative}"
            )
        entries.append(
            ArtifactMaterializationEntry(
                relative_path=relative,
                sha256=Sha256Digest(hashlib.sha256(data).hexdigest()),
                byte_length=len(data),
            )
        )
    return tuple(entries)


def verify_gsplat_fused_ssim_reference_sources(
    source_root: Path,
) -> tuple[ArtifactMaterializationEntry, ...]:
    """Verify the reviewed local fused-ssim import/CUDA-build sources without executing them.

    This does not approve a built wheel, compiled CUDA extension, Torch/CUDA ABI,
    complete transitive dependency closure or a real appearance training run.
    """

    if not isinstance(source_root, Path):
        raise TypeError("gsplat fused-ssim source_root must be pathlib.Path")
    root = _safe_root(source_root, "gsplat fused-ssim source_root")
    package = root / "fused_ssim"
    if package.is_symlink() or not package.is_dir():
        raise GsplatStaticAppearancePreflightError("fused-ssim package must be a real directory")
    expected_package = {"fused_ssim/__init__.py"}
    actual_package: set[str] = set()
    for path in package.rglob("*"):
        if path.is_symlink() or not path.is_file():
            raise GsplatStaticAppearancePreflightError(
                "fused-ssim package has an unexpected source path or symlink"
            )
        actual_package.add(path.relative_to(root).as_posix())
    if actual_package != expected_package:
        raise GsplatStaticAppearancePreflightError(
            "fused-ssim import package differs from the reviewed source"
        )

    reviewed_top_level = {
        relative
        for relative, _size, _blob in _GSPLAT_FUSED_SSIM_SOURCE_FILES
        if "/" not in relative
    }
    for path in root.iterdir():
        if path.is_symlink():
            raise GsplatStaticAppearancePreflightError("fused-ssim source root contains a symlink")
        if path.is_file() and path.suffix in {".py", ".cpp", ".cu", ".h"}:
            if path.name not in reviewed_top_level:
                raise GsplatStaticAppearancePreflightError(
                    "fused-ssim has an unexpected executable or build source file"
                )

    entries: list[ArtifactMaterializationEntry] = []
    for relative, expected_size, expected_blob in _GSPLAT_FUSED_SSIM_SOURCE_FILES:
        candidate = root
        for part in PurePosixPath(relative).parts:
            candidate /= part
            if candidate.is_symlink():
                raise GsplatStaticAppearancePreflightError(
                    f"fused-ssim source path contains symlink: {relative}"
                )
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise GsplatStaticAppearancePreflightError(
                f"fused-ssim source file missing: {relative}"
            ) from exc
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise GsplatStaticAppearancePreflightError(
                f"fused-ssim source file invalid: {relative}"
            )
        if resolved.stat().st_size != expected_size:
            raise GsplatStaticAppearancePreflightError(
                f"fused-ssim source byte length differs: {relative}"
            )
        data = resolved.read_bytes()
        if len(data) != expected_size or _git_blob_sha1(data) != expected_blob:
            raise GsplatStaticAppearancePreflightError(
                f"fused-ssim source Git blob differs: {relative}"
            )
        if resolved.stat().st_size != expected_size:
            raise GsplatStaticAppearancePreflightError(
                f"fused-ssim source changed during verification: {relative}"
            )
        entries.append(
            ArtifactMaterializationEntry(
                relative_path=relative,
                sha256=Sha256Digest(hashlib.sha256(data).hexdigest()),
                byte_length=len(data),
            )
        )
    return tuple(entries)


def verify_gsplat_nerfview_reference_sources(
    source_root: Path,
) -> tuple[ArtifactMaterializationEntry, ...]:
    """Verify pinned viewer import-source and packaging bytes without importing them.

    This is source-only evidence. It does not approve viewer execution, a
    dependency resolver, web server exposure or a trainer/GPU execution runtime.
    """

    if not isinstance(source_root, Path):
        raise TypeError("gsplat nerfview source_root must be pathlib.Path")
    root = _safe_root(source_root, "gsplat nerfview source_root")
    package = root / "nerfview"
    if package.is_symlink() or not package.is_dir():
        raise GsplatStaticAppearancePreflightError("nerfview package must be a real directory")
    expected = {
        relative
        for relative, _size, _blob in _GSPLAT_NERFVIEW_SOURCE_FILES
        if relative.startswith("nerfview/")
    }
    actual: set[str] = set()
    for path in package.rglob("*"):
        if path.is_symlink() or not path.is_file():
            raise GsplatStaticAppearancePreflightError(
                "nerfview contains an unexpected package path or symlink"
            )
        actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise GsplatStaticAppearancePreflightError(
            "nerfview import package differs from the reviewed source"
        )

    for path in root.iterdir():
        if path.is_symlink():
            raise GsplatStaticAppearancePreflightError("nerfview source root contains a symlink")

    entries: list[ArtifactMaterializationEntry] = []
    for relative, expected_size, expected_blob in _GSPLAT_NERFVIEW_SOURCE_FILES:
        candidate = root
        for part in PurePosixPath(relative).parts:
            candidate /= part
            if candidate.is_symlink():
                raise GsplatStaticAppearancePreflightError(
                    f"nerfview source path contains symlink: {relative}"
                )
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise GsplatStaticAppearancePreflightError(
                f"nerfview source file missing: {relative}"
            ) from exc
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise GsplatStaticAppearancePreflightError(f"nerfview source file invalid: {relative}")
        if resolved.stat().st_size != expected_size:
            raise GsplatStaticAppearancePreflightError(
                f"nerfview source byte length differs: {relative}"
            )
        data = resolved.read_bytes()
        if len(data) != expected_size or _git_blob_sha1(data) != expected_blob:
            raise GsplatStaticAppearancePreflightError(
                f"nerfview source Git blob differs: {relative}"
            )
        if resolved.stat().st_size != expected_size:
            raise GsplatStaticAppearancePreflightError(
                f"nerfview source changed during verification: {relative}"
            )
        entries.append(
            ArtifactMaterializationEntry(
                relative_path=relative,
                sha256=Sha256Digest(hashlib.sha256(data).hexdigest()),
                byte_length=len(data),
            )
        )
    return tuple(entries)


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
            raise TypeError(
                "gsplat photometric evidence must be PhotometricCompatibilityAssessment"
            )
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
        if factors.exposure_adjustment_ev != 0.0 or factors.white_balance_rgb_gains != (
            1.0,
            1.0,
            1.0,
        ):
            raise GsplatStaticAppearancePreflightError(
                "gsplat reference trainer reads raw PNG; non-identity V2L17 "
                "normalization is not applied to those training bytes"
            )
        identities.append(assessment.identity)

    return tuple(identities)


def verify_gsplat_static_appearance_matching_capture_metadata(
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
    assessments: tuple[PhotometricCompatibilityAssessment, ...],
) -> tuple[Sha256Digest, ...]:
    """Verify only matching, *declared* multi-view capture settings.

    This is a conservative necessary-evidence check before a separate
    scene-overlap/radiometric assessment. Matching EXIF-like capture settings
    cannot prove equal radiance, equal tone curves, static scene content, or
    suitable cross-view photometry. Missing/unknown values remain unresolved.
    The gsplat reference trainer still reads the original PNG pixels.
    """

    identities = verify_gsplat_static_appearance_source_photometry(request, source, assessments)
    if len(assessments) < 2:
        raise GsplatStaticAppearancePreflightError(
            "multi-view capture comparison requires at least two real observations"
        )

    declared_settings: set[tuple[float, float, float, float, str, float]] = set()
    for assessment in assessments:
        photometry = (
            assessment.compatibility_input.normalization_assessment.request.source_photometry
        )
        exposure = photometry.exposure
        white_balance = photometry.white_balance
        if (
            exposure.status is not SourcePhotometryInterpretationStatus.RESOLVED
            or white_balance.status is not SourcePhotometryInterpretationStatus.RESOLVED
            or exposure.iso_speed is None
            or exposure.exposure_time_seconds is None
            or exposure.f_number is None
            or exposure.exposure_compensation_ev is None
            or white_balance.mode is None
            or white_balance.color_temperature_kelvin is None
        ):
            raise GsplatStaticAppearancePreflightError(
                "multi-view capture exposure and white-balance metadata are incomplete "
                "or unresolved; no cross-observation equivalence may be inferred"
            )
        declared_settings.add(
            (
                exposure.iso_speed,
                exposure.exposure_time_seconds,
                exposure.f_number,
                exposure.exposure_compensation_ev,
                white_balance.mode,
                white_balance.color_temperature_kelvin,
            )
        )

    if len(declared_settings) != 1:
        raise GsplatStaticAppearancePreflightError(
            "multi-view capture settings differ; gsplat raw-PNG input cannot silently "
            "equalize exposure or white balance"
        )

    return identities


@dataclass(frozen=True, slots=True)
class GsplatStaticAppearanceSharedTrackPair:
    """A verified COLMAP sparse-track overlap, NOT photometric calibration."""

    left_observation_id: ObservationId
    right_observation_id: ObservationId
    shared_point_count: int


def verify_gsplat_static_appearance_shared_scene_tracks(
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
    assessments: tuple[PhotometricCompatibilityAssessment, ...],
    *,
    features: ColmapFeatureExtractionResult,
    expected_environment: ColmapEnvironmentIdentity,
    module: object,
) -> tuple[GsplatStaticAppearanceSharedTrackPair, ...]:
    """Require a connected graph of real multi-view COLMAP point tracks.

    A point counts only when the audited native model links the same 3D point
    to distinct registered source images through reciprocal Point2D IDs.
    Counts are *structural overlap evidence*, not proof of matching exposure,
    surface color, scene radiance, adequate baselines or training quality.
    Caller must still perform a separate radiometric/shared-scene suitability
    assessment and use an approved external runtime before executing gsplat.
    """

    verify_gsplat_static_appearance_matching_capture_metadata(request, source, assessments)
    if module is None:
        raise TypeError("shared-track verification requires an explicit PyCOLMAP module")
    verify_gsplat_static_appearance_native_geometry(
        request,
        source,
        features=features,
        expected_environment=expected_environment,
        module=module,
    )
    path = _verified_native_model_path(source.native_model_root, source.native_model_artifact)
    try:
        pycolmap = cast(Any, module)
        reconstruction = pycolmap.Reconstruction(path)
        if not bool(reconstruction.is_valid()):
            raise GsplatStaticAppearancePreflightError(
                "native COLMAP model is invalid during track verification"
            )
        registered = tuple(sorted(int(value) for value in reconstruction.reg_image_ids()))
        point_ids = tuple(sorted(int(value) for value in reconstruction.point3D_ids()))
        if (
            len(registered) != len(source.images)
            or len(point_ids) != source.native_model_artifact.num_points3d
        ):
            raise GsplatStaticAppearancePreflightError(
                "native COLMAP registered-image or point count changed"
            )
        native_images = {image_id: reconstruction.image(image_id) for image_id in registered}
        if len({image.name for image in native_images.values()}) != len(source.images):
            raise GsplatStaticAppearancePreflightError("native COLMAP image names are not unique")
        by_name = {item.image_name: item.observation.observation_id for item in source.images}
        if set(image.name for image in native_images.values()) != set(by_name):
            raise GsplatStaticAppearancePreflightError(
                "native COLMAP registered-image names differ from exact source images"
            )

        pair_counts: dict[tuple[int, int], int] = {}
        for point_id in point_ids:
            point = reconstruction.point3D(point_id)
            seen: set[int] = set()
            for element in point.track.elements:
                image_id = int(element.image_id)
                point2d_index = int(element.point2D_idx)
                if image_id not in native_images or image_id in seen:
                    raise GsplatStaticAppearancePreflightError(
                        "native COLMAP track references foreign or duplicate registered images"
                    )
                image = native_images[image_id]
                if point2d_index < 0 or point2d_index >= int(image.num_points2D()):
                    raise GsplatStaticAppearancePreflightError(
                        "native COLMAP track point2D index is invalid"
                    )
                point2d = image.point2D(point2d_index)
                if int(point2d.point3D_id) != point_id:
                    raise GsplatStaticAppearancePreflightError(
                        "native COLMAP track/Point2D identity is not reciprocal"
                    )
                seen.add(image_id)
            for left in sorted(seen):
                for right in sorted(seen):
                    if left < right:
                        pair_counts[(left, right)] = pair_counts.get((left, right), 0) + 1
    except GsplatStaticAppearancePreflightError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError, RuntimeError, OSError) as exc:
        raise GsplatStaticAppearancePreflightError(
            "native COLMAP point tracks cannot be audited"
        ) from exc

    # A connected graph is a *necessary* scene-overlap condition. Requiring
    # every pair to overlap would incorrectly reject valid multi-view sequences.
    neighbours: dict[int, set[int]] = {image_id: set() for image_id in registered}
    pairs: list[GsplatStaticAppearanceSharedTrackPair] = []
    for (left, right), count in pair_counts.items():
        neighbours[left].add(right)
        neighbours[right].add(left)
        left_id, right_id = (
            by_name[native_images[left].name],
            by_name[native_images[right].name],
        )
        if left_id.value > right_id.value:
            left_id, right_id = right_id, left_id
        pairs.append(
            GsplatStaticAppearanceSharedTrackPair(
                left_observation_id=left_id,
                right_observation_id=right_id,
                shared_point_count=count,
            )
        )
    visited = {registered[0]}
    pending = [registered[0]]
    while pending:
        current = pending.pop()
        for neighbour in neighbours[current] - visited:
            visited.add(neighbour)
            pending.append(neighbour)
    if len(visited) != len(registered):
        raise GsplatStaticAppearancePreflightError(
            "native COLMAP sparse tracks do not connect all source observations"
        )

    return tuple(
        sorted(
            pairs,
            key=lambda item: (item.left_observation_id.value, item.right_observation_id.value),
        )
    )


@dataclass(frozen=True, slots=True)
class GsplatStaticAppearanceSharedTrackPixelPair:
    """Observed sRGB8 pixel difference at reciprocal COLMAP tracks, not radiance truth.

    Differences are descriptive before photometric calibration, occlusion testing,
    outlier filtering or held-out rendering. They cannot approve GPU training.
    """

    left_observation_id: ObservationId
    right_observation_id: ObservationId
    left_png_sha256: Sha256Digest
    right_png_sha256: Sha256Digest
    shared_track_count: int
    mean_absolute_srgb_channel_delta: float


def _decode_audited_gsplat_png_rgb8(data: bytes) -> tuple[int, int, bytes]:
    """Decode the already-supported sRGB RGB8 PNG filters without optional imports.

    The existing source-byte gate verifies PNG chunk CRCs, all critical chunks,
    raster length, sRGB declaration and zlib integrity. This second read performs
    the five PNG inverse scanline filters before sampling *actual* source pixels.
    It does not apply exposure or color-response calibration.
    """

    width, height = _png_srgb_dimensions(data)
    cursor = len(_PNG_SIGNATURE)
    compressed = bytearray()
    while cursor < len(data):
        length = struct.unpack_from(">I", data, cursor)[0]
        kind = data[cursor + 4 : cursor + 8]
        cursor += 8
        if kind == b"IDAT":
            compressed.extend(data[cursor : cursor + length])
        cursor += length + 4

    stride = width * 3
    expected = height * (1 + stride)
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(bytes(compressed), expected + 1)
    except zlib.error as exc:
        raise GsplatStaticAppearancePreflightError("shared-track PNG raster is corrupt") from exc
    if len(raw) != expected or not decoder.eof or decoder.unused_data:
        raise GsplatStaticAppearancePreflightError("shared-track PNG raster is incomplete")

    raster = bytearray(height * stride)
    previous = bytearray(stride)
    offset = 0
    for row_index in range(height):
        filter_type = raw[offset]
        offset += 1
        encoded = raw[offset : offset + stride]
        offset += stride
        current = bytearray(stride)
        for position, byte in enumerate(encoded):
            left = current[position - 3] if position >= 3 else 0
            up = previous[position]
            upper_left = previous[position - 3] if position >= 3 else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = up
            elif filter_type == 3:
                predictor = (left + up) // 2
            else:
                base = left + up - upper_left
                a = abs(base - left)
                b = abs(base - up)
                c = abs(base - upper_left)
                predictor = left if a <= b and a <= c else up if b <= c else upper_left
            current[position] = (byte + predictor) & 0xFF
        raster[row_index * stride : (row_index + 1) * stride] = current
        previous = current
    return width, height, bytes(raster)


def inspect_gsplat_static_appearance_shared_track_pixels(
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
    assessments: tuple[PhotometricCompatibilityAssessment, ...],
    *,
    features: ColmapFeatureExtractionResult,
    expected_environment: ColmapEnvironmentIdentity,
    module: object,
) -> tuple[GsplatStaticAppearanceSharedTrackPixelPair, ...]:
    """Read original sRGB pixels at verified common 3D tracks for diagnostics only.

    This first repeats the strict existing source/photometry/capture/native-track
    gates. It then re-hashes the exact original PNG bytes, applies PNG inverse
    filters, and samples each reciprocal 2D track at its containing pixel. No
    RGB threshold, calibration, view-weight or quality-pass decision is inferred:
    equal observed pixels do not prove equal scene radiance, and unequal pixels
    can be caused by viewpoint, occlusion, lighting or camera-response effects.
    """

    pairs = verify_gsplat_static_appearance_shared_scene_tracks(
        request,
        source,
        assessments,
        features=features,
        expected_environment=expected_environment,
        module=module,
    )
    root = _safe_root(source.image_root, "gsplat image_root")
    images_by_name: dict[str, tuple[ObservationId, Sha256Digest, int, int, bytes]] = {}
    for item in source.images:
        path = root / item.image_name
        if path.is_symlink() or item.source_path.is_symlink():
            raise GsplatStaticAppearancePreflightError(
                "shared-track source image changed into a symlink"
            )
        try:
            resolved = path.resolve(strict=True)
            bound = item.source_path.resolve(strict=True)
            if not resolved.is_file() or resolved.parent != root or resolved != bound:
                raise GsplatStaticAppearancePreflightError(
                    "shared-track source image is not the exact original bound file"
                )
            data = resolved.read_bytes()
        except (OSError, RuntimeError) as exc:
            raise GsplatStaticAppearancePreflightError(
                "shared-track source image is inaccessible"
            ) from exc
        if (
            len(data) != item.observation.asset.byte_length
            or Sha256Digest(hashlib.sha256(data).hexdigest()) != item.observation.asset.sha256
        ):
            raise GsplatStaticAppearancePreflightError(
                "shared-track source PNG bytes differ from the original Observation"
            )
        width, height, pixels = _decode_audited_gsplat_png_rgb8(data)
        images_by_name[item.image_name] = (
            item.observation.observation_id,
            item.observation.asset.sha256,
            width,
            height,
            pixels,
        )

    path = _verified_native_model_path(source.native_model_root, source.native_model_artifact)
    try:
        native = cast(Any, module).Reconstruction(path)
        if not bool(native.is_valid()):
            raise GsplatStaticAppearancePreflightError(
                "native COLMAP model is invalid during source-pixel inspection"
            )
        image_ids = tuple(sorted(int(value) for value in native.reg_image_ids()))
        native_images = {image_id: native.image(image_id) for image_id in image_ids}
        if len(image_ids) != len(images_by_name) or {
            image.name for image in native_images.values()
        } != set(images_by_name):
            raise GsplatStaticAppearancePreflightError(
                "native COLMAP image identities differ during source-pixel inspection"
            )

        totals: dict[tuple[str, str], tuple[int, int]] = {}
        for point_id in sorted(int(value) for value in native.point3D_ids()):
            point = native.point3D(point_id)
            sampled: dict[str, tuple[int, int, int]] = {}
            for element in point.track.elements:
                image_id = int(element.image_id)
                index = int(element.point2D_idx)
                if image_id not in native_images:
                    raise GsplatStaticAppearancePreflightError(
                        "foreign native COLMAP image in shared-track pixel evidence"
                    )
                image = native_images[image_id]
                if index < 0 or index >= int(image.num_points2D()):
                    raise GsplatStaticAppearancePreflightError(
                        "invalid native COLMAP 2D index in shared-track pixel evidence"
                    )
                point2d = image.point2D(index)
                if int(point2d.point3D_id) != point_id:
                    raise GsplatStaticAppearancePreflightError(
                        "non-reciprocal native COLMAP track in shared-track pixel evidence"
                    )
                obs_id, _sha, width, height, pixels = images_by_name[image.name]
                xy = point2d.xy
                x, y = float(xy[0]), float(xy[1])
                if not math.isfinite(x) or not math.isfinite(y):
                    raise GsplatStaticAppearancePreflightError(
                        "native COLMAP track contains non-finite source-pixel coordinates"
                    )
                # COLMAP image coordinates refer to pixel centers (0.5, 0.5).
                px, py = math.floor(x), math.floor(y)
                if px < 0 or py < 0 or px >= width or py >= height:
                    raise GsplatStaticAppearancePreflightError(
                        "native COLMAP track lies outside the exact source image"
                    )
                offset = (py * width + px) * 3
                if obs_id.value in sampled:
                    raise GsplatStaticAppearancePreflightError(
                        "duplicate native COLMAP observation in shared-track pixel evidence"
                    )
                sampled[obs_id.value] = (
                    pixels[offset],
                    pixels[offset + 1],
                    pixels[offset + 2],
                )
            sorted_ids = sorted(sampled)
            for position, left in enumerate(sorted_ids):
                for right in sorted_ids[position + 1 :]:
                    delta = sum(
                        abs(a - b) for a, b in zip(sampled[left], sampled[right], strict=True)
                    )
                    count, total = totals.get((left, right), (0, 0))
                    totals[(left, right)] = count + 1, total + delta
    except GsplatStaticAppearancePreflightError:
        raise
    except (
        AttributeError,
        TypeError,
        IndexError,
        KeyError,
        ValueError,
        OverflowError,
        RuntimeError,
        OSError,
    ) as exc:
        raise GsplatStaticAppearancePreflightError(
            "native COLMAP source-pixel track evidence cannot be audited"
        ) from exc

    sha_by_id = {
        image.observation.observation_id.value: image.observation.asset.sha256
        for image in source.images
    }
    result: list[GsplatStaticAppearanceSharedTrackPixelPair] = []
    for pair in pairs:
        left, right = pair.left_observation_id.value, pair.right_observation_id.value
        count, total = totals.get((left, right), (0, 0))
        if count != pair.shared_point_count:
            raise GsplatStaticAppearancePreflightError(
                "native COLMAP shared-track counts changed during pixel inspection"
            )
        result.append(
            GsplatStaticAppearanceSharedTrackPixelPair(
                left_observation_id=pair.left_observation_id,
                right_observation_id=pair.right_observation_id,
                left_png_sha256=sha_by_id[left],
                right_png_sha256=sha_by_id[right],
                shared_track_count=count,
                mean_absolute_srgb_channel_delta=total / (count * 3 * 255),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class GsplatStaticAppearanceStagedDataset:
    """Private exact COLMAP/PNG trainer input tree, not a trained AppearanceModel."""

    dataset_root: Path
    source_geometry_id: str
    native_model_ref: ArtifactRef
    entries: tuple[ArtifactMaterializationEntry, ...]


def _copy_gsplat_source_exact(
    source_file: Path,
    output_file: Path,
    *,
    expected_sha256: Sha256Digest,
    expected_bytes: int,
) -> ArtifactMaterializationEntry:
    """Copy one immutable file in bounded chunks; verify source AND copied bytes."""

    if source_file.is_symlink() or not source_file.is_file():
        raise GsplatStaticAppearancePreflightError(
            "gsplat staging requires a regular source file without symlinks"
        )
    digest = hashlib.sha256()
    count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with source_file.open("rb") as source_stream, output_file.open("xb") as output_stream:
        while True:
            chunk = source_stream.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > expected_bytes:
                raise GsplatStaticAppearancePreflightError(
                    "gsplat staging source byte count exceeds the audited input"
                )
            digest.update(chunk)
            output_stream.write(chunk)
    actual_sha256 = Sha256Digest(digest.hexdigest())
    if count != expected_bytes or actual_sha256 != expected_sha256:
        raise GsplatStaticAppearancePreflightError(
            "gsplat staging source identity changed during copy"
        )

    # Re-hash the independently staged bytes, rather than assuming copy succeeded.
    staged_hash = hashlib.sha256()
    staged_size = 0
    with output_file.open("rb") as staged_stream:
        while chunk := staged_stream.read(1024 * 1024):
            staged_size += len(chunk)
            staged_hash.update(chunk)
    if staged_size != expected_bytes or staged_hash.hexdigest() != expected_sha256.value:
        raise GsplatStaticAppearancePreflightError(
            "gsplat staged file differs from the audited input"
        )
    if source_file.is_symlink() or not source_file.is_file():
        raise GsplatStaticAppearancePreflightError(
            "gsplat staging source disappeared or changed type"
        )
    return ArtifactMaterializationEntry(
        relative_path=output_file.name,
        sha256=actual_sha256,
        byte_length=count,
    )


def stage_verified_gsplat_static_appearance_dataset(
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
    *,
    dataset_root: Path,
) -> GsplatStaticAppearanceStagedDataset:
    """Stage verified PNG and unmodified native COLMAP files in a fresh private tree.

    The reference trainer expects data_dir/images and data_dir/sparse/0. This
    CPU-only stage preserves exact original PNG and binary model bytes without
    image resampling, pose transforms, hidden parser normalization or invoking
    gsplat. The final destination must not exist; failures remove only WRE's
    own temporary tree and never touch source images or solver-native files.
    """

    verified = preflight_gsplat_static_appearance_inputs(request, source)
    if not isinstance(dataset_root, Path):
        raise TypeError("gsplat dataset_root must be pathlib.Path")
    if dataset_root.is_symlink() or dataset_root.exists():
        raise GsplatStaticAppearancePreflightError(
            "gsplat staged dataset destination must not already exist"
        )
    parent = _safe_root(dataset_root.parent, "gsplat dataset parent")
    destination = parent / dataset_root.name
    images_root = _safe_root(source.image_root, "gsplat image_root")
    native_root = _safe_root(source.native_model_root, "gsplat native_model_root")
    if any(
        destination.is_relative_to(root) or root.is_relative_to(destination)
        for root in (images_root, native_root)
    ):
        raise GsplatStaticAppearancePreflightError(
            "gsplat staged dataset must not overlap source image or native-model roots"
        )
    native_model = _verified_native_model_path(
        source.native_model_root, source.native_model_artifact
    )
    expected_model_files = ("cameras.bin", "images.bin", "points3D.bin")
    if (
        tuple(item.relative_path for item in source.native_model_artifact.files)
        != expected_model_files
    ):
        raise GsplatStaticAppearancePreflightError(
            "gsplat staging requires the exact COLMAP binary cameras/images/points model"
        )

    temporary = Path(tempfile.mkdtemp(prefix=".wre-gsplat-dataset-", dir=parent))
    published = False
    try:
        entries: list[ArtifactMaterializationEntry] = []
        for item, expected in zip(source.images, verified.images, strict=True):
            if (
                item.image_name != expected.image_name
                or item.observation.observation_id != expected.observation_id
            ):
                raise GsplatStaticAppearancePreflightError(
                    "gsplat source observation identity changed before staging"
                )
            staged_file = temporary / "images" / item.image_name
            copied = _copy_gsplat_source_exact(
                item.source_path,
                staged_file,
                expected_sha256=expected.sha256,
                expected_bytes=expected.byte_length,
            )
            entries.append(
                ArtifactMaterializationEntry(
                    relative_path=f"images/{item.image_name}",
                    sha256=copied.sha256,
                    byte_length=copied.byte_length,
                )
            )

        for manifest in source.native_model_artifact.files:
            staged_file = temporary / "sparse" / "0" / manifest.relative_path
            copied = _copy_gsplat_source_exact(
                native_model / manifest.relative_path,
                staged_file,
                expected_sha256=manifest.sha256,
                expected_bytes=manifest.byte_length,
            )
            entries.append(
                ArtifactMaterializationEntry(
                    relative_path=f"sparse/0/{manifest.relative_path}",
                    sha256=copied.sha256,
                    byte_length=copied.byte_length,
                )
            )
        ordered = tuple(sorted(entries, key=lambda item: item.relative_path))
        if destination.exists() or destination.is_symlink():
            raise GsplatStaticAppearancePreflightError(
                "gsplat destination appeared while preparing private data"
            )
        temporary.rename(destination)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary)

    return GsplatStaticAppearanceStagedDataset(
        dataset_root=destination,
        source_geometry_id=verified.source_geometry_id,
        native_model_ref=verified.native_model_ref,
        entries=ordered,
    )


def verify_gsplat_static_appearance_staged_dataset(
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
    staged: GsplatStaticAppearanceStagedDataset,
) -> tuple[ArtifactMaterializationEntry, ...]:
    """Recheck the complete private dataset before handing it to an external trainer.

    A successful earlier staging operation is not proof that the on-disk tree is
    still intact. Verify the original input identities again, require exactly the
    canonical trainer file set and reject symlinks, unknown files, directories,
    changed payloads or mismatched request/geometry ancestry. No trainer import.
    """

    if not isinstance(staged, GsplatStaticAppearanceStagedDataset):
        raise TypeError("gsplat staged dataset must be GsplatStaticAppearanceStagedDataset")
    verified = preflight_gsplat_static_appearance_inputs(request, source)
    if (
        staged.source_geometry_id != verified.source_geometry_id
        or staged.native_model_ref != verified.native_model_ref
    ):
        raise GsplatStaticAppearancePreflightError(
            "gsplat staged dataset has different source geometry or native model ancestry"
        )
    if staged.dataset_root.is_symlink():
        raise GsplatStaticAppearancePreflightError(
            "gsplat staged dataset root must not be a symlink"
        )
    root = _safe_root(staged.dataset_root, "gsplat staged dataset")
    native_model = _verified_native_model_path(
        source.native_model_root, source.native_model_artifact
    )
    expected_native_names = ("cameras.bin", "images.bin", "points3D.bin")
    if tuple(item.relative_path for item in source.native_model_artifact.files) != (
        expected_native_names
    ):
        raise GsplatStaticAppearancePreflightError(
            "gsplat staged dataset requires exactly three audited COLMAP binary files"
        )

    expected_files: dict[str, tuple[Path, Sha256Digest, int]] = {}
    for item, image in zip(source.images, verified.images, strict=True):
        if (
            item.image_name != image.image_name
            or item.observation.observation_id != image.observation_id
        ):
            raise GsplatStaticAppearancePreflightError(
                "gsplat staged dataset source observation identities changed"
            )
        expected_files[f"images/{image.image_name}"] = (
            item.source_path,
            image.sha256,
            image.byte_length,
        )
    for item in source.native_model_artifact.files:
        expected_files[f"sparse/0/{item.relative_path}"] = (
            native_model / item.relative_path,
            item.sha256,
            item.byte_length,
        )
    expected_entries = tuple(
        ArtifactMaterializationEntry(
            relative_path=relative_path, sha256=digest, byte_length=byte_length
        )
        for relative_path, (_source, digest, byte_length) in sorted(expected_files.items())
    )
    if staged.entries != expected_entries:
        raise GsplatStaticAppearancePreflightError(
            "gsplat staged dataset manifest differs from exact original inputs"
        )

    allowed_directories = {"images", "sparse", "sparse/0"}
    present_files: set[str] = set()
    for path in root.rglob("*"):
        relative_path = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise GsplatStaticAppearancePreflightError("gsplat staged dataset contains a symlink")
        if path.is_dir():
            if relative_path not in allowed_directories:
                raise GsplatStaticAppearancePreflightError(
                    "gsplat staged dataset contains an unexpected directory"
                )
        elif path.is_file():
            if relative_path not in expected_files:
                raise GsplatStaticAppearancePreflightError(
                    "gsplat staged dataset contains an unexpected file"
                )
            present_files.add(relative_path)
        else:
            raise GsplatStaticAppearancePreflightError(
                "gsplat staged dataset contains an unsupported file type"
            )
    if present_files != set(expected_files):
        raise GsplatStaticAppearancePreflightError(
            "gsplat staged dataset is missing an audited input file"
        )

    for relative_path, (original, expected_digest, expected_size) in sorted(expected_files.items()):
        for path in (original, root / relative_path):
            if path.is_symlink() or not path.is_file():
                raise GsplatStaticAppearancePreflightError(
                    "gsplat staged dataset or its original source is not a regular file"
                )
            digest = hashlib.sha256()
            total = 0
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    total += len(chunk)
                    if total > expected_size:
                        raise GsplatStaticAppearancePreflightError(
                            "gsplat staged dataset file exceeds its audited size"
                        )
                    digest.update(chunk)
            if total != expected_size or digest.hexdigest() != expected_digest.value:
                raise GsplatStaticAppearancePreflightError(
                    "gsplat staged dataset or original source differs from audited SHA-256"
                )
    return expected_entries


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
            "sh_degree": _GSPLAT_REFERENCE_SH_DEGREE,
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
    The reviewed trainer has SH degree 3 (45 f_rest values); truncated or
    differently configured SH payloads are not accepted as this profile.
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
            or len(remainder) != _GSPLAT_REFERENCE_SH_REST_FLOATS
            or remainder != [f"f_rest_{index}" for index in range(_GSPLAT_REFERENCE_SH_REST_FLOATS)]
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


def assemble_verified_gsplat_static_appearance_result(
    *,
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
    staged: GsplatStaticAppearanceStagedDataset,
    artifact_ref: ArtifactRef,
    output_root: Path,
    profile: GsplatStaticAppearanceTrainingProfile,
    producer: ArtifactProducerIdentity,
) -> StaticAppearanceCandidateResult:
    """Retain already-produced exact gsplat PLY bytes under the WRE appearance contract.

    This is *output assembly*, not a trainer or a claim that a GPU execution,
    shared radiance calibration, or generalization benchmark passed. The optional
    external runner must independently establish those facts. Revalidating the
    private dataset and output here prevents a stale staging tree or a malformed
    export from masquerading as a canonically materialized AppearanceModel.
    """

    if not isinstance(request, StaticAppearanceCandidateRequest):
        raise TypeError("gsplat result request must be StaticAppearanceCandidateRequest")
    if not isinstance(profile, GsplatStaticAppearanceTrainingProfile):
        raise TypeError("gsplat result profile must be GsplatStaticAppearanceTrainingProfile")
    if not isinstance(producer, ArtifactProducerIdentity):
        raise TypeError("gsplat result producer must be ArtifactProducerIdentity")
    if producer.configuration.sha256 != profile.configuration_sha256:
        raise GsplatStaticAppearancePreflightError(
            "gsplat output producer configuration differs from the reviewed training profile"
        )
    if (
        producer.producer.implementation != "gsplat.examples.simple_trainer"
        or producer.producer.version != "1.5.3"
        or producer.model is None
        or producer.model.name != "gsplat"
        or producer.model.version != "1.5.3"
        or producer.model.revision != GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION
        or producer.checkpoint is not None
    ):
        raise GsplatStaticAppearancePreflightError(
            "gsplat output producer does not identify the exact checkpoint-free reference trainer"
        )
    if request.output_representation.value != GSPLAT_STATIC_APPEARANCE_REPRESENTATION:
        raise GsplatStaticAppearancePreflightError(
            "gsplat output representation differs from the reviewed PLY payload"
        )

    # Both checks inspect real local bytes; neither infers a training run.
    verify_gsplat_static_appearance_staged_dataset(request, source, staged)
    materialization = materialize_verified_gsplat_ply(
        artifact_ref=artifact_ref, output_root=output_root, profile=profile
    )

    source_refs = list(request.source_geometry.source_artifacts)
    if request.source_surface is not None:
        source_refs.append(request.source_surface.artifact_ref)
        source_refs.extend(request.source_surface.source_artifacts)
    source_refs.extend(request.supporting_artifacts)
    # Multiple canonical ancestry paths may legitimately refer to the same
    # exact artifact. The result boundary requires the canonical set union.
    distinct = {(item.artifact_id.value, item.artifact_kind.value): item for item in source_refs}
    ancestry = tuple(distinct[key] for key in sorted(distinct))
    geometry = request.source_geometry.geometry_solution
    candidate = AppearanceModel(
        artifact_ref=artifact_ref,
        source_geometry=request.source_geometry,
        source_surface=request.source_surface,
        source_observation_ids=request.source_observation_ids,
        representation=request.output_representation,
        local_frame_id=geometry.local_frame_id,
        scale_status=geometry.scale_status,
        producer=producer,
        source_artifacts=ancestry,
    )
    return build_static_appearance_candidate_result(request, candidate, materialization)


__all__ = [
    "GSPLAT_REFERENCE_PYCOLMAP_REVISION",
    "GSPLAT_STATIC_APPEARANCE_REPRESENTATION",
    "GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION",
    "GSPLAT_STATIC_APPEARANCE_SOURCE_TREE",
    "GSPLAT_STATIC_APPEARANCE_TRAINER_GIT_BLOB",
    "GsplatStaticAppearancePreflightError",
    "GsplatStaticAppearancePreflightEvidence",
    "GsplatStaticAppearancePreflightSource",
    "GsplatStaticAppearanceSharedTrackPair",
    "GsplatStaticAppearanceSharedTrackPixelPair",
    "GsplatStaticAppearanceStagedDataset",
    "GsplatStaticAppearanceTrainingProfile",
    "GsplatStaticAppearanceVerifiedImage",
    "assemble_verified_gsplat_static_appearance_result",
    "inspect_gsplat_static_appearance_shared_track_pixels",
    "materialize_verified_gsplat_ply",
    "preflight_gsplat_static_appearance_inputs",
    "stage_verified_gsplat_static_appearance_dataset",
    "verify_gsplat_reference_pycolmap_sources",
    "verify_gsplat_static_appearance_matching_capture_metadata",
    "verify_gsplat_static_appearance_native_geometry",
    "verify_gsplat_static_appearance_reference_sources",
    "verify_gsplat_static_appearance_shared_scene_tracks",
    "verify_gsplat_static_appearance_source_photometry",
    "verify_gsplat_static_appearance_staged_dataset",
]
