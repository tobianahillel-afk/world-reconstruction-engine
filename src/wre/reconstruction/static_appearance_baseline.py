"""Fail-closed source-byte and camera preflight for the optional gsplat v1.5.3 route.

This is *not* a trainer or a photometric-compatibility approval. A later V2L18.3
step must independently audit native COLMAP camera/track semantics, multi-view
exposure consistency, the exact external environment and real GPU execution.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
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
    ("nerfview/render_panel.py", 54931, "a81fa3ea9a468a94ebea6a8de4da9312ef5185d4"),
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
        try:
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
        except OSError as exc:
            raise GsplatStaticAppearancePreflightError(
                f"gsplat reference source disappeared during verification: {relative}"
            ) from exc
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


def build_gsplat_reference_trainer_path_overrides(
    *,
    profile: GsplatStaticAppearanceTrainingProfile,
    dataset_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Bind the exact trainer to explicit private input/output paths, without running it.

    The pinned Config defaults point to upstream sample data/results; no future
    runner may fall back to those paths. Source bytes must be checked separately.
    This builder neither creates the output directory nor validates dataset bytes.
    """

    if not isinstance(profile, GsplatStaticAppearanceTrainingProfile):
        raise TypeError("gsplat runtime profile must be GsplatStaticAppearanceTrainingProfile")
    if not isinstance(dataset_root, Path) or not isinstance(output_root, Path):
        raise TypeError("gsplat runtime paths must be pathlib.Path")
    if not dataset_root.is_absolute() or not output_root.is_absolute():
        raise GsplatStaticAppearancePreflightError(
            "gsplat runtime input and output paths must be absolute"
        )
    dataset = _safe_root(dataset_root, "gsplat runtime dataset_root")
    staged_directories = (
        dataset / "images",
        dataset / "sparse",
        dataset / "sparse" / "0",
    )
    if any(path.is_symlink() or not path.is_dir() for path in staged_directories):
        raise GsplatStaticAppearancePreflightError(
            "gsplat runtime dataset requires real staged images and COLMAP directories"
        )
    # The reference COLMAP parser opens all three binary model components.
    # A directory-only check can otherwise accept an empty staged fixture.
    model_root = dataset / "sparse" / "0"
    if any(
        (model_root / name).is_symlink() or not (model_root / name).is_file()
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    ):
        raise GsplatStaticAppearancePreflightError(
            "gsplat runtime dataset lacks real COLMAP binary model files"
        )
    try:
        has_png = any(
            path.suffix.lower() == ".png" and not path.is_symlink() and path.is_file()
            for path in (dataset / "images").iterdir()
        )
    except OSError as exc:
        raise GsplatStaticAppearancePreflightError(
            "gsplat runtime image directory could not be inspected"
        ) from exc
    if not has_png:
        raise GsplatStaticAppearancePreflightError(
            "gsplat runtime dataset has no regular PNG image"
        )
    if output_root.is_symlink() or output_root.exists():
        raise GsplatStaticAppearancePreflightError(
            "gsplat runtime output directory must not already exist"
        )
    output_parent = _safe_root(output_root.parent, "gsplat runtime output parent")
    destination = output_parent / output_root.name
    if destination.is_relative_to(dataset) or dataset.is_relative_to(destination):
        raise GsplatStaticAppearancePreflightError(
            "gsplat runtime output directory overlaps the staged dataset"
        )
    if destination.is_symlink() or destination.exists():
        raise GsplatStaticAppearancePreflightError(
            "gsplat runtime output directory changed before configuration"
        )
    result = profile.reference_overrides()
    result["data_dir"] = str(dataset)
    result["result_dir"] = str(destination)
    return result


# The external interpreter is an explicit, caller-owned CUDA environment.
# -I prevents user site/PYTHONPATH injection; only the pinned examples directory
# is prepended in the child. The parent never imports torch or CUDA modules.
_GSPLAT_TRAINER_CHILD = """import json
import sys
import torch

# Upstream constructs LPIPS/AlexNet even when evaluation is disabled.
# Require pretrained weights to be provisioned in this runtime's torch cache.
# torch.hub.load_state_dict_from_url uses this hook on cache misses.
def reject_implicit_download(*args, **kwargs):
    raise RuntimeError("gsplat LPIPS weights must be cached; implicit download forbidden")

torch.hub.download_url_to_file = reject_implicit_download

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("gsplat reference training requires exactly one visible CUDA GPU")
import gsplat
if gsplat.__version__ != "1.5.3":
    raise RuntimeError("gsplat runtime version differs from the pinned reference trainer")
sys.path.insert(0, sys.argv[1])
from simple_trainer import Config, main

cfg = Config(**json.load(sys.stdin))
main(0, 0, 1, cfg)
"""


def _verify_gsplat_reference_producer(
    profile: GsplatStaticAppearanceTrainingProfile, producer: ArtifactProducerIdentity
) -> None:
    """Check one exact source/profile identity before training and at output assembly."""

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
        or producer.producer.revision != GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION
        or producer.model is None
        or producer.model.name != "gsplat"
        or producer.model.version != "1.5.3"
        or producer.model.revision != GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION
        or producer.checkpoint is not None
    ):
        raise GsplatStaticAppearancePreflightError(
            "gsplat output producer does not identify the exact checkpoint-free reference trainer"
        )


def run_gsplat_reference_training(
    *,
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
    assessments: tuple[PhotometricCompatibilityAssessment, ...],
    staged: GsplatStaticAppearanceStagedDataset,
    profile: GsplatStaticAppearanceTrainingProfile,
    features: ColmapFeatureExtractionResult,
    expected_environment: ColmapEnvironmentIdentity,
    source_root: Path,
    python_executable: Path,
    output_root: Path,
    artifact_ref: ArtifactRef,
    producer: ArtifactProducerIdentity,
    cuda_device_index: int,
    timeout_seconds: int,
    lpips_alexnet_weights: Path | None = None,
    lpips_alexnet_sha256: str | None = None,
) -> StaticAppearanceCandidateResult:
    """Run the exact reference trainer on an EXPLICIT external CUDA environment.

    This optional runtime path never loads torch in WRE itself. It verifies the
    source checkout and full staged input first, launches the upstream Config
    and main in one non-shell subprocess, then assembles the actual produced
    PLY bytes into one canonical appearance candidate with source ancestry.
    The caller must provision/pin a compatible Python/Torch/CUDA environment;
    CPU-only CI exercises orchestration only, not GPU execution or training.
    V2L17 assessments are mandatory; matched settings alone do not prove
    scene-radiance equivalence or held-out rendering quality.
    """

    _verify_gsplat_reference_producer(profile, producer)
    # Training consumes the original sRGB PNGs: require V2L17 identity
    # and matching multi-view capture evidence before any CUDA process.
    verify_gsplat_static_appearance_matching_capture_metadata(request, source, assessments)
    if type(cuda_device_index) is not int or cuda_device_index < 0:
        raise ValueError("gsplat CUDA device index must be a nonnegative integer")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("gsplat training timeout_seconds must be positive")
    if not isinstance(python_executable, Path) or not python_executable.is_absolute():
        raise TypeError("gsplat runtime requires an absolute external Python executable path")
    if not python_executable.is_file():
        raise GsplatStaticAppearancePreflightError(
            "gsplat external Python executable does not exist"
        )
    # gsplat trains on the exact native COLMAP poses/points, not on an
    # independently refined WRE frame. Reject mismatches before CUDA launch.
    verify_gsplat_static_appearance_native_geometry(
        request,
        source,
        features=features,
        expected_environment=expected_environment,
    )
    verify_gsplat_static_appearance_staged_dataset(request, source, staged)
    source_directory = _safe_root(source_root, "gsplat pinned reference checkout")
    verify_gsplat_reference_trainer_config_fields(source_directory, profile)
    overrides = build_gsplat_reference_trainer_path_overrides(
        profile=profile, dataset_root=staged.dataset_root, output_root=output_root
    )
    if not isinstance(artifact_ref, ArtifactRef):
        raise TypeError("gsplat output artifact_ref must be ArtifactRef")
    if artifact_ref.artifact_kind != APPEARANCE_MODEL_ARTIFACT_KIND:
        raise GsplatStaticAppearancePreflightError(
            "gsplat external trainer output must be an appearance artifact"
        )
    # The pinned upstream Runner always constructs LPIPS/AlexNet. Without an
    # explicit offline checkpoint it cannot train; reject before creating any
    # output or attempting to launch the CUDA interpreter.
    if lpips_alexnet_weights is None or lpips_alexnet_sha256 is None:
        raise GsplatStaticAppearancePreflightError(
            "gsplat LPIPS offline weights require both a local file and SHA-256"
        )
    if not isinstance(lpips_alexnet_weights, Path) or not lpips_alexnet_weights.is_file():
        raise GsplatStaticAppearancePreflightError(
            "gsplat LPIPS pretrained weight file is unavailable"
        )
    if not isinstance(lpips_alexnet_sha256, str) or (
        len(lpips_alexnet_sha256) != 64
        or any(char not in "0123456789abcdef" for char in lpips_alexnet_sha256)
    ):
        raise GsplatStaticAppearancePreflightError(
            "gsplat LPIPS pretrained weight digest must be lowercase SHA-256"
        )
    # The private directory is created only after mandatory dependency inputs
    # have passed their inexpensive boundary checks.
    output_root.mkdir(mode=0o700)
    torch_home = output_root / "torch-cache"
    cache = torch_home / "hub" / "checkpoints"
    cache.mkdir(parents=True, mode=0o700)
    destination = cache / "alexnet-owt-7be5be79.pth"
    try:
        shutil.copyfile(lpips_alexnet_weights, destination)
        with destination.open("rb") as checked:
            digest = hashlib.file_digest(checked, "sha256").hexdigest()
    except OSError as exc:
        raise GsplatStaticAppearancePreflightError(
            "gsplat LPIPS offline weights could not be staged"
        ) from exc
    if digest != lpips_alexnet_sha256:
        destination.unlink(missing_ok=True)
        raise GsplatStaticAppearancePreflightError(
            "gsplat LPIPS offline pretrained weight SHA-256 mismatch"
        )
    log_path = output_root / "wre-reference-training.log"
    # The external interpreter executes third-party native code. Never
    # forward unrelated credentials (e.g. CI tokens or cloud keys).
    # Keep only the runtime paths used by Python/CUDA and temp/cache helpers.
    runtime_environment_names = (
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "TMP",
        "TEMP",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "XDG_CACHE_HOME",
        "TORCH_EXTENSIONS_DIR",
    )
    env = {name: os.environ[name] for name in runtime_environment_names if name in os.environ}
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device_index)
    if torch_home is not None:
        env["TORCH_HOME"] = str(torch_home)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    try:
        with log_path.open("xb") as log:
            completed = subprocess.run(
                [
                    str(python_executable),
                    "-I",
                    "-c",
                    _GSPLAT_TRAINER_CHILD,
                    str(source_directory / "examples"),
                ],
                input=json.dumps(overrides, allow_nan=False).encode("utf-8"),
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=source_directory / "examples",
                env=env,
                timeout=timeout_seconds,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise GsplatStaticAppearancePreflightError(
            "gsplat reference training timed out; inspect the private training log"
        ) from exc
    except OSError as exc:
        raise GsplatStaticAppearancePreflightError(
            "gsplat external Python process could not be launched"
        ) from exc
    if completed.returncode != 0:
        raise GsplatStaticAppearancePreflightError(
            "gsplat reference training failed; inspect the private training log"
        )
    return assemble_verified_gsplat_static_appearance_result(
        request=request,
        source=source,
        staged=staged,
        artifact_ref=artifact_ref,
        output_root=output_root,
        profile=profile,
        producer=producer,
    )


def reconstruct_gsplat_static_appearance(
    *,
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
    assessments: tuple[PhotometricCompatibilityAssessment, ...],
    dataset_root: Path,
    profile: GsplatStaticAppearanceTrainingProfile,
    features: ColmapFeatureExtractionResult,
    expected_environment: ColmapEnvironmentIdentity,
    source_root: Path,
    python_executable: Path,
    output_root: Path,
    artifact_ref: ArtifactRef,
    producer: ArtifactProducerIdentity,
    cuda_device_index: int,
    timeout_seconds: int,
    lpips_alexnet_weights: Path | None = None,
    lpips_alexnet_sha256: str | None = None,
) -> StaticAppearanceCandidateResult:
    """Stage real COLMAP/PNG files, execute the selected trainer, retain its 3D PLY.

    The only GPU operation is in the caller-supplied external environment. The
    WRE process stages/verifies exact file bytes and returns a canonical result;
    a CPU-only test may emulate the trainer but is not evidence of GPU training.
    """

    _verify_gsplat_reference_producer(profile, producer)
    if not isinstance(dataset_root, Path) or not dataset_root.is_absolute():
        raise GsplatStaticAppearancePreflightError(
            "gsplat complete reconstruction requires an absolute private dataset path"
        )
    staged = stage_verified_gsplat_static_appearance_dataset(
        request, source, dataset_root=dataset_root
    )
    return run_gsplat_reference_training(
        request=request,
        source=source,
        assessments=assessments,
        staged=staged,
        profile=profile,
        features=features,
        expected_environment=expected_environment,
        source_root=source_root,
        python_executable=python_executable,
        output_root=output_root,
        artifact_ref=artifact_ref,
        producer=producer,
        cuda_device_index=cuda_device_index,
        timeout_seconds=timeout_seconds,
        lpips_alexnet_weights=lpips_alexnet_weights,
        lpips_alexnet_sha256=lpips_alexnet_sha256,
    )


@dataclass(frozen=True, slots=True)
class GsplatStaticAppearanceAdapter:
    """One configured V2L18.2 specialist; no implicit runtime or GPU fallback.

    The caller supplies the reviewed source, native COLMAP evidence, private
    workspaces and the explicit optional CUDA environment. Every request goes
    through the existing complete stage -> trainer -> canonical result path.
    """

    source: GsplatStaticAppearancePreflightSource
    dataset_root: Path
    profile: GsplatStaticAppearanceTrainingProfile
    features: ColmapFeatureExtractionResult
    expected_environment: ColmapEnvironmentIdentity
    source_root: Path
    python_executable: Path
    output_root: Path
    artifact_ref: ArtifactRef
    producer: ArtifactProducerIdentity
    cuda_device_index: int
    timeout_seconds: int
    lpips_alexnet_weights: Path
    lpips_alexnet_sha256: str

    def derive_candidate(
        self, request: StaticAppearanceCandidateRequest
    ) -> StaticAppearanceCandidateResult:
        """Implement the existing solver-independent adapter boundary."""
        return reconstruct_gsplat_static_appearance(
            request=request,
            source=self.source,
            dataset_root=self.dataset_root,
            profile=self.profile,
            features=self.features,
            expected_environment=self.expected_environment,
            source_root=self.source_root,
            python_executable=self.python_executable,
            output_root=self.output_root,
            artifact_ref=self.artifact_ref,
            producer=self.producer,
            cuda_device_index=self.cuda_device_index,
            timeout_seconds=self.timeout_seconds,
            lpips_alexnet_weights=self.lpips_alexnet_weights,
            lpips_alexnet_sha256=self.lpips_alexnet_sha256,
        )


def verify_gsplat_reference_trainer_config_fields(
    source_root: Path,
    profile: GsplatStaticAppearanceTrainingProfile,
) -> tuple[str, ...]:
    """Check actual pinned trainer Config fields without importing or running it.

    This is a source-schema gate, not an ABI, CUDA or training execution check.
    It prevents a future external runner from silently dropping a misspelled
    WRE override when constructing the exact reference trainer's Config.
    """

    if not isinstance(source_root, Path):
        raise TypeError("gsplat trainer source_root must be pathlib.Path")
    if not isinstance(profile, GsplatStaticAppearanceTrainingProfile):
        raise TypeError("gsplat training profile must be GsplatStaticAppearanceTrainingProfile")
    verify_gsplat_static_appearance_reference_sources(source_root)
    source_file = (
        _safe_root(source_root, "gsplat trainer source_root") / "examples/simple_trainer.py"
    )
    try:
        payload = source_file.read_bytes()
    except OSError as exc:
        raise GsplatStaticAppearancePreflightError(
            "gsplat trainer source disappeared during Config schema verification"
        ) from exc
    trainer_pins = [
        (size, blob)
        for path, size, blob in _GSPLAT_REFERENCE_SOURCE_FILES
        if path == "examples/simple_trainer.py"
    ]
    if len(trainer_pins) != 1:
        raise GsplatStaticAppearancePreflightError(
            "gsplat trainer source registry must contain exactly one pinned Config source"
        )
    pinned_size, pinned_blob = trainer_pins[0]
    if len(payload) != pinned_size or _git_blob_sha1(payload) != pinned_blob:
        raise GsplatStaticAppearancePreflightError(
            "gsplat trainer source changed before Config schema verification"
        )
    try:
        parsed = ast.parse(payload, filename="examples/simple_trainer.py")
    except (SyntaxError, ValueError) as exc:
        raise GsplatStaticAppearancePreflightError(
            "gsplat trainer has an invalid Config source"
        ) from exc
    configs = [
        node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "Config"
    ]
    if len(configs) != 1 or not any(
        isinstance(dec, ast.Name) and dec.id == "dataclass" for dec in configs[0].decorator_list
    ):
        raise GsplatStaticAppearancePreflightError(
            "gsplat trainer does not declare the expected dataclass Config"
        )
    # An exact source pin is not an installed environment: also check the
    # trainer's declared direct integration edges before a future repin.
    required_imports = {
        "datasets.colmap": {"Parser", "Dataset"},
        "fused_ssim": {"fused_ssim"},
        "gsplat": {"export_splats"},
        "gsplat.rendering": {"rasterization"},
        "gsplat_viewer": {"GsplatViewer"},
        "nerfview": {"CameraState"},
    }
    imported: dict[str, set[str]] = {}
    for node in parsed.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            imported.setdefault(node.module, set()).update(alias.name for alias in node.names)
    if any(
        not required <= imported.get(module, set()) for module, required in required_imports.items()
    ):
        raise GsplatStaticAppearancePreflightError(
            "gsplat trainer is missing reviewed direct integration imports"
        )
    if not any(
        isinstance(node, ast.Import) and any(alias.name == "torch" for alias in node.names)
        for node in parsed.body
    ):
        raise GsplatStaticAppearancePreflightError(
            "gsplat trainer is missing the reviewed PyTorch import"
        )

    fields = {
        node.target.id
        for node in configs[0].body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    overrides = set(profile.reference_overrides())
    if not overrides <= fields:
        raise GsplatStaticAppearancePreflightError(
            "gsplat trainer Config lacks reviewed WRE override fields"
        )
    return tuple(sorted(overrides))


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


def export_verified_gsplat_rgb_centers_ply(
    *,
    artifact_ref: ArtifactRef,
    output_root: Path,
    profile: GsplatStaticAppearanceTrainingProfile,
    preview_path: Path,
) -> Sha256Digest:
    """Export viewable RGB Gaussian *centers*, not radiance or a rendered scene.

    Reuse the full source PLY audit, stream xyz and degree-zero SH color from
    the pinned gsplat exporter into a standard XYZ/RGB PLY. No Torch/CUDA or
    custom Gaussian renderer is required. The original splat remains intact.
    The return value hashes this distinct visualization aid, not the model.
    """

    if not isinstance(preview_path, Path) or not preview_path.is_absolute():
        raise TypeError("gsplat RGB preview_path must be an absolute pathlib.Path")
    verified = materialize_verified_gsplat_ply(
        artifact_ref=artifact_ref, output_root=output_root, profile=profile
    )
    root = _safe_root(output_root, "gsplat output_root")
    source_path = root / profile.expected_ply_relative_path
    parent = _safe_root(preview_path.parent, "gsplat RGB preview parent")
    destination = parent / preview_path.name
    if destination.suffix != ".ply" or destination.exists() or destination == source_path:
        raise GsplatStaticAppearancePreflightError(
            "gsplat RGB preview must use a new distinct .ply file"
        )

    source_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    source_stride = 59 * 4  # fixed audited degree-3 SH exporter layout
    created = False
    try:
        with source_path.open("rb") as source, destination.open("xb") as output:
            created = True
            header = bytearray()
            while not header.endswith(b"end_header\n"):
                line = source.readline(_MAX_GSPLAT_PLY_HEADER_BYTES + 1 - len(header))
                if not line or len(header) + len(line) > _MAX_GSPLAT_PLY_HEADER_BYTES:
                    raise GsplatStaticAppearancePreflightError(
                        "gsplat source PLY header changed during RGB export"
                    )
                header.extend(line)
            source_hash.update(header)
            try:
                count = int(header.splitlines()[2].split()[2])
            except (IndexError, ValueError) as exc:
                raise GsplatStaticAppearancePreflightError(
                    "gsplat source PLY vertex header changed during RGB export"
                ) from exc
            preview_header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {count}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                "end_header\n"
            ).encode("ascii")
            output.write(preview_header)
            output_hash.update(preview_header)

            remaining = count
            while remaining:
                batch_count = min(remaining, 1024)
                block = source.read(batch_count * source_stride)
                if len(block) != batch_count * source_stride:
                    raise GsplatStaticAppearancePreflightError(
                        "gsplat source PLY changed during RGB export"
                    )
                source_hash.update(block)
                rgb_block = bytearray()
                for record in struct.iter_unpack("<59f", block):
                    # Exactly gsplat/exporter.py sh2rgb(sh0) = sh0 * C0 + 0.5.
                    colors = tuple(
                        max(0, min(255, int((dc * 0.28209479177387814 + 0.5) * 255)))
                        for dc in record[3:6]
                    )
                    rgb_block.extend(struct.pack("<3f3B", *record[:3], *colors))
                output.write(rgb_block)
                output_hash.update(rgb_block)
                remaining -= batch_count
            if source.read(1):
                raise GsplatStaticAppearancePreflightError(
                    "gsplat source PLY gained unexpected bytes during RGB export"
                )
            if Sha256Digest(source_hash.hexdigest()) != verified.entries[0].sha256:
                raise GsplatStaticAppearancePreflightError(
                    "gsplat source PLY identity changed during RGB export"
                )
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise

    return Sha256Digest(output_hash.hexdigest())


def export_verified_gsplat_rgb_centers_glb(
    *,
    artifact_ref: ArtifactRef,
    output_root: Path,
    profile: GsplatStaticAppearanceTrainingProfile,
    preview_path: Path,
) -> Sha256Digest:
    """Export a standalone glTF 2.0 POINTS GLB from the verified Gaussian PLY.

    Reuses the existing SH0-to-RGB PLY exporter as the only color conversion.
    This visualizes Gaussian centers, not rendered splats or full radiance.
    """

    if not isinstance(preview_path, Path) or not preview_path.is_absolute():
        raise TypeError("gsplat GLB preview_path must be an absolute pathlib.Path")
    parent = _safe_root(preview_path.parent, "gsplat GLB preview parent")
    destination = parent / preview_path.name
    if destination.suffix != ".glb" or destination.exists() or destination.is_symlink():
        raise GsplatStaticAppearancePreflightError(
            "gsplat RGB preview requires a new .glb destination"
        )
    created = False
    digest = hashlib.sha256()
    try:
        with tempfile.TemporaryDirectory(prefix="wre-gsplat-glb-", dir=parent) as folder:
            rgb_ply = Path(folder) / "centers.ply"
            ply_hash = export_verified_gsplat_rgb_centers_ply(
                artifact_ref=artifact_ref,
                output_root=output_root,
                profile=profile,
                preview_path=rgb_ply,
            )
            with rgb_ply.open("rb") as source:
                header = bytearray()
                while not header.endswith(b"end_header\n"):
                    chunk = source.readline(256)
                    if not chunk or len(header) + len(chunk) > 1024:
                        raise GsplatStaticAppearancePreflightError(
                            "gsplat RGB center PLY has an invalid preview header"
                        )
                    header.extend(chunk)
                try:
                    count = int(header.splitlines()[2].split()[2])
                except (IndexError, ValueError) as exc:
                    raise GsplatStaticAppearancePreflightError(
                        "gsplat RGB preview has no valid vertex count"
                    ) from exc
                expected_header = (
                    "ply\nformat binary_little_endian 1.0\n"
                    f"element vertex {count}\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    "end_header\n"
                ).encode("ascii")
                if count <= 0 or header != expected_header:
                    raise GsplatStaticAppearancePreflightError(
                        "gsplat RGB preview has an unexpected PLY layout"
                    )
                ply_digest = hashlib.sha256(header)
                minimum = [math.inf] * 3
                maximum = [-math.inf] * 3
                remaining = count
                while remaining:
                    block_count = min(remaining, 1024)
                    data = source.read(block_count * 15)
                    if len(data) != block_count * 15:
                        raise GsplatStaticAppearancePreflightError(
                            "gsplat RGB preview is truncated"
                        )
                    ply_digest.update(data)
                    for record in struct.iter_unpack("<3f3B", data):
                        for axis in range(3):
                            minimum[axis] = min(minimum[axis], record[axis])
                            maximum[axis] = max(maximum[axis], record[axis])
                    remaining -= block_count
                if source.read(1) or Sha256Digest(ply_digest.hexdigest()) != ply_hash:
                    raise GsplatStaticAppearancePreflightError(
                        "gsplat RGB preview bytes changed before GLB conversion"
                    )
                # Single interleaved glTF bufferView: float32 XYZ + u8 RGB + padding.
                bin_length = count * 16
                model = {
                    "asset": {"version": "2.0", "generator": "WRE gsplat RGB centers"},
                    "scene": 0,
                    "scenes": [{"nodes": [0]}],
                    "nodes": [{"mesh": 0}],
                    "meshes": [
                        {
                            "primitives": [
                                {
                                    "attributes": {"POSITION": 0, "COLOR_0": 1},
                                    "mode": 0,
                                }
                            ]
                        }
                    ],
                    "buffers": [{"byteLength": bin_length}],
                    "bufferViews": [
                        {
                            "buffer": 0,
                            "byteOffset": 0,
                            "byteLength": bin_length,
                            "byteStride": 16,
                            "target": 34962,
                        }
                    ],
                    "accessors": [
                        {
                            "bufferView": 0,
                            "byteOffset": 0,
                            "componentType": 5126,
                            "count": count,
                            "type": "VEC3",
                            "min": minimum,
                            "max": maximum,
                        },
                        {
                            "bufferView": 0,
                            "byteOffset": 12,
                            "componentType": 5121,
                            "normalized": True,
                            "count": count,
                            "type": "VEC3",
                        },
                    ],
                }
                encoded = json.dumps(model, sort_keys=True, separators=(",", ":")).encode("utf-8")
                json_chunk = encoded + b" " * (-len(encoded) % 4)
                total_length = 12 + 8 + len(json_chunk) + 8 + bin_length
                if total_length >= 2**32:
                    raise GsplatStaticAppearancePreflightError(
                        "gsplat GLB point preview exceeds the format's 4 GiB limit"
                    )
                with destination.open("xb") as output:
                    created = True

                    def emit(data: bytes) -> None:
                        output.write(data)
                        digest.update(data)

                    emit(struct.pack("<4sII", b"glTF", 2, total_length))
                    emit(struct.pack("<I4s", len(json_chunk), b"JSON"))
                    emit(json_chunk)
                    emit(struct.pack("<I4s", bin_length, b"BIN\x00"))
                    source.seek(len(header))
                    remaining = count
                    while remaining:
                        block_count = min(remaining, 1024)
                        block = source.read(block_count * 15)
                        if len(block) != block_count * 15:
                            raise GsplatStaticAppearancePreflightError(
                                "gsplat RGB preview changed during GLB export"
                            )
                        packed = bytearray()
                        for record in struct.iter_unpack("<3f3B", block):
                            packed.extend(struct.pack("<3f4B", *record, 0))
                        emit(bytes(packed))
                        remaining -= block_count
        return Sha256Digest(digest.hexdigest())
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise


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
    _verify_gsplat_reference_producer(profile, producer)
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
    "GsplatStaticAppearanceAdapter",
    "GsplatStaticAppearancePreflightError",
    "GsplatStaticAppearancePreflightEvidence",
    "GsplatStaticAppearancePreflightSource",
    "GsplatStaticAppearanceSharedTrackPair",
    "GsplatStaticAppearanceSharedTrackPixelPair",
    "GsplatStaticAppearanceStagedDataset",
    "GsplatStaticAppearanceTrainingProfile",
    "GsplatStaticAppearanceVerifiedImage",
    "assemble_verified_gsplat_static_appearance_result",
    "export_verified_gsplat_rgb_centers_glb",
    "export_verified_gsplat_rgb_centers_ply",
    "inspect_gsplat_static_appearance_shared_track_pixels",
    "materialize_verified_gsplat_ply",
    "preflight_gsplat_static_appearance_inputs",
    "reconstruct_gsplat_static_appearance",
    "run_gsplat_reference_training",
    "stage_verified_gsplat_static_appearance_dataset",
    "verify_gsplat_reference_pycolmap_sources",
    "verify_gsplat_static_appearance_matching_capture_metadata",
    "verify_gsplat_static_appearance_native_geometry",
    "verify_gsplat_static_appearance_reference_sources",
    "verify_gsplat_static_appearance_shared_scene_tracks",
    "verify_gsplat_static_appearance_source_photometry",
    "verify_gsplat_static_appearance_staged_dataset",
]
