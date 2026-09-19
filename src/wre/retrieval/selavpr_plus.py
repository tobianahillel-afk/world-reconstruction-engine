from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Protocol, cast

from wre.domain.artifact_materialization import ArtifactMaterializationVerificationStatus
from wre.domain.artifacts import ArtifactRef
from wre.domain.decoded_images import (
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
)
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.pair_candidates import (
    PairCandidate,
    PairCandidateSource,
    PairCandidateSourceId,
    merge_pair_candidates,
)
from wre.domain.producer_identity import CheckpointIdentity, ModelIdentity
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRun
from wre.ingestion.decoded_images import DecodedImagePyramidMaterializationResult
from wre.ingestion.hashing import hash_file_content
from wre.materialization import verify_local_artifact_materialization

SELAVPR_PLUS_SOURCE_REVISION = "56bd921cbd3d53e9c5f91d0aafff147f95fb362a"
SELAVPR_PLUS_RETRIEVAL_IMPLEMENTATION = "wre.retrieval.selavpr_plus"
SELAVPR_PLUS_RETRIEVAL_VERSION = "1"
SELAVPR_PLUS_CHECKPOINT_IDENTIFIER = (
    "fenglu96/SelaVPRplusplus/SelaVPRplusplus_base.pth"
)
SELAVPR_PLUS_CHECKPOINT_SHA256 = Sha256Digest(
    "b048490dbd1c27dee67fce6faaec7bec267d19a044c85877af94b8588e596a62"
)
SELAVPR_PLUS_MODEL = ModelIdentity(
    name="selavpr_plus.base_gem_float",
    version="1",
    revision=SELAVPR_PLUS_SOURCE_REVISION,
)
SELAVPR_PLUS_CHECKPOINT = CheckpointIdentity(
    identifier=SELAVPR_PLUS_CHECKPOINT_IDENTIFIER,
    sha256=SELAVPR_PLUS_CHECKPOINT_SHA256,
)
SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID = PairCandidateSourceId("selavpr_plus")

SELAVPR_PLUS_TORCH_VERSION = "2.14.0"
SELAVPR_PLUS_NUMPY_VERSION = "2.5.3"
SELAVPR_PLUS_FAISS_CPU_VERSION = "1.15.1"
SELAVPR_PLUS_TQDM_VERSION = "4.68.2"

_MODEL_REQUIRED_PATHS = (
    "model/network.py",
    "model/vision_transformer.py",
    "model/aggregation.py",
)
_NETWORK_PREFIXES = ("http:", "https:", "ftp:", "s3:", "gs:")
_DESCRIPTOR_NORM_TOLERANCE = 1e-4


class SelaVprPlusRetrievalError(RuntimeError):
    """Raised when the learned retrieval candidate cannot be audited safely."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SelaVprPlusRetrievalConfig:
    """Canonical bounded configuration for the first SelaVPR++ reference path."""

    schema_version: int = 1
    image_height_px: int = 322
    image_width_px: int = 322
    descriptor_dimension: int = 2048
    neighbors_per_observation: int = 20
    random_seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
        ):
            raise ValueError("selavpr_plus schema_version must be integer 1")
        if type(self.image_height_px) is not int or self.image_height_px != 322:
            raise ValueError("selavpr_plus image_height_px must be integer 322")
        if type(self.image_width_px) is not int or self.image_width_px != 322:
            raise ValueError("selavpr_plus image_width_px must be integer 322")
        if type(self.descriptor_dimension) is not int or self.descriptor_dimension != 2048:
            raise ValueError("selavpr_plus descriptor_dimension must be integer 2048")
        if (
            isinstance(self.neighbors_per_observation, bool)
            or not isinstance(self.neighbors_per_observation, int)
            or self.neighbors_per_observation <= 0
        ):
            raise ValueError("neighbors_per_observation must be a positive integer")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed != 0
        ):
            raise ValueError("deterministic SelaVPR++ retrieval requires random_seed=0")
        if self.device != "cpu":
            raise ValueError("the first SelaVPR++ reference path requires device='cpu'")

    def canonical_document(self) -> dict[str, object]:
        return {
            "descriptor_dimension": self.descriptor_dimension,
            "device": self.device,
            "image_height_px": self.image_height_px,
            "image_width_px": self.image_width_px,
            "neighbors_per_observation": self.neighbors_per_observation,
            "random_seed": self.random_seed,
            "schema_version": self.schema_version,
        }

    @property
    def sha256(self) -> Sha256Digest:
        payload = json.dumps(
            self.canonical_document(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return Sha256Digest(hashlib.sha256(payload).hexdigest())


@dataclass(frozen=True, slots=True, kw_only=True)
class SelaVprPlusImageInput:
    decoded_result: DecodedImagePyramidMaterializationResult
    materialization_root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.decoded_result, DecodedImagePyramidMaterializationResult):
            raise TypeError("selavpr_plus image decoded_result has the wrong type")
        if not isinstance(self.materialization_root, Path):
            raise TypeError("selavpr_plus image materialization_root must be pathlib.Path")


@dataclass(frozen=True, slots=True, kw_only=True)
class SelaVprPlusRetrievalRequest:
    run: ReconstructionRun
    inputs: tuple[SelaVprPlusImageInput, ...]
    source_root: Path
    checkpoint_path: Path
    model: ModelIdentity = SELAVPR_PLUS_MODEL
    checkpoint: CheckpointIdentity = SELAVPR_PLUS_CHECKPOINT
    config: SelaVprPlusRetrievalConfig = field(default_factory=SelaVprPlusRetrievalConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.run, ReconstructionRun):
            raise TypeError("selavpr_plus request run must be ReconstructionRun")
        if not isinstance(self.inputs, tuple):
            raise TypeError("selavpr_plus request inputs must be an immutable tuple")
        if len(self.inputs) < 2:
            raise ValueError("selavpr_plus request requires at least two inputs")
        if any(not isinstance(item, SelaVprPlusImageInput) for item in self.inputs):
            raise TypeError("selavpr_plus request inputs contain an invalid member")
        if not isinstance(self.source_root, Path):
            raise TypeError("selavpr_plus source_root must be pathlib.Path")
        if not isinstance(self.checkpoint_path, Path):
            raise TypeError("selavpr_plus checkpoint_path must be pathlib.Path")
        if not isinstance(self.model, ModelIdentity):
            raise TypeError("selavpr_plus model must be ModelIdentity")
        if not isinstance(self.checkpoint, CheckpointIdentity):
            raise TypeError("selavpr_plus checkpoint must be CheckpointIdentity")
        if not isinstance(self.config, SelaVprPlusRetrievalConfig):
            raise TypeError("selavpr_plus config must be SelaVprPlusRetrievalConfig")
        if self.model != SELAVPR_PLUS_MODEL:
            raise ValueError("selavpr_plus request model must match the exact reviewed model")
        if self.checkpoint != SELAVPR_PLUS_CHECKPOINT:
            raise ValueError("selavpr_plus request checkpoint must match the exact reviewed checkpoint")

        observation_ids = tuple(
            item.decoded_result.manifest.source_observation_id for item in self.inputs
        )
        if observation_ids != tuple(sorted(observation_ids, key=lambda item: item.value)):
            raise ValueError("selavpr_plus request inputs must use canonical observation order")
        if len({item.value for item in observation_ids}) != len(observation_ids):
            raise ValueError("selavpr_plus request inputs must not repeat observations")
        if self.run.input_observation_ids != observation_ids:
            raise ValueError("ReconstructionRun inputs must exactly match SelaVPR++ inputs")
        if self.run.producer.implementation != SELAVPR_PLUS_RETRIEVAL_IMPLEMENTATION:
            raise ValueError(
                f"ReconstructionRun producer must be {SELAVPR_PLUS_RETRIEVAL_IMPLEMENTATION!r}"
            )
        if self.run.producer.version != SELAVPR_PLUS_RETRIEVAL_VERSION:
            raise ValueError(
                f"ReconstructionRun producer version must be {SELAVPR_PLUS_RETRIEVAL_VERSION!r}"
            )
        if self.run.producer.revision != SELAVPR_PLUS_SOURCE_REVISION:
            raise ValueError("ReconstructionRun producer revision must match SelaVPR++ source")
        if self.run.configuration_sha256 != self.config.sha256:
            raise ValueError(
                "ReconstructionRun configuration SHA-256 must match the SelaVPR++ config"
            )


@dataclass(frozen=True, slots=True)
class SelaVprPlusEnvironmentIdentity:
    source_revision: str
    python_version: str
    torch_version: str
    numpy_version: str
    faiss_cpu_version: str
    tqdm_version: str
    device: str

    def __post_init__(self) -> None:
        if self.source_revision != SELAVPR_PLUS_SOURCE_REVISION:
            raise ValueError("SelaVPR++ environment source revision is unsupported")
        for name in (
            "python_version",
            "torch_version",
            "numpy_version",
            "faiss_cpu_version",
            "tqdm_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"selavpr_plus environment {name} must be non-blank")
        if self.device != "cpu":
            raise ValueError("SelaVPR++ environment device must be cpu")


@dataclass(frozen=True, slots=True)
class SelaVprPlusPair:
    observation_id1: ObservationId
    observation_id2: ObservationId
    cosine_similarity: float

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id1, ObservationId):
            raise TypeError("selavpr_plus pair observation_id1 must be ObservationId")
        if not isinstance(self.observation_id2, ObservationId):
            raise TypeError("selavpr_plus pair observation_id2 must be ObservationId")
        if self.observation_id1.value >= self.observation_id2.value:
            raise ValueError(
                "selavpr_plus pair observation IDs must be distinct and canonically ordered"
            )
        if isinstance(self.cosine_similarity, bool) or not isinstance(
            self.cosine_similarity, (int, float)
        ):
            raise TypeError("selavpr_plus cosine_similarity must be a real number")
        similarity = float(self.cosine_similarity)
        if not math.isfinite(similarity):
            raise ValueError("selavpr_plus cosine_similarity must be finite")
        if similarity < -1.000001 or similarity > 1.000001:
            raise ValueError("selavpr_plus cosine_similarity must be a cosine value")
        object.__setattr__(self, "cosine_similarity", similarity)


@dataclass(frozen=True, slots=True)
class SelaVprPlusRetrievalResult:
    provenance: DerivedArtifactProvenance
    model: ModelIdentity
    checkpoint: CheckpointIdentity
    configuration_sha256: Sha256Digest
    environment: SelaVprPlusEnvironmentIdentity
    pairs: tuple[SelaVprPlusPair, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, DerivedArtifactProvenance):
            raise TypeError("selavpr_plus result provenance must be DerivedArtifactProvenance")
        if self.model != SELAVPR_PLUS_MODEL:
            raise ValueError("selavpr_plus result model must match the reviewed model")
        if self.checkpoint != SELAVPR_PLUS_CHECKPOINT:
            raise ValueError("selavpr_plus result checkpoint must match the reviewed checkpoint")
        if not isinstance(self.configuration_sha256, Sha256Digest):
            raise TypeError("selavpr_plus result configuration_sha256 must be Sha256Digest")
        if not isinstance(self.environment, SelaVprPlusEnvironmentIdentity):
            raise TypeError("selavpr_plus result environment has the wrong type")
        if not isinstance(self.pairs, tuple):
            raise TypeError("selavpr_plus result pairs must be an immutable tuple")
        if any(not isinstance(pair, SelaVprPlusPair) for pair in self.pairs):
            raise TypeError("selavpr_plus result pairs contain an invalid member")
        keys = tuple((pair.observation_id1.value, pair.observation_id2.value) for pair in self.pairs)
        if len(keys) != len(set(keys)):
            raise ValueError("selavpr_plus result pairs must be unique")
        if keys != tuple(sorted(keys)):
            raise ValueError("selavpr_plus result pairs must use canonical lexical order")
        provenance_ids = {item.value for item in self.provenance.source_observation_ids}
        if any(
            pair.observation_id1.value not in provenance_ids
            or pair.observation_id2.value not in provenance_ids
            for pair in self.pairs
        ):
            raise ValueError("selavpr_plus result pairs must belong to provenance observations")


@dataclass(frozen=True, slots=True)
class SelaVprPlusPairCandidateAdapterInput:
    result: SelaVprPlusRetrievalResult
    evidence_ref: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.result, SelaVprPlusRetrievalResult):
            raise TypeError("selavpr_plus adapter result has the wrong type")
        if not isinstance(self.evidence_ref, ArtifactRef):
            raise TypeError("selavpr_plus adapter evidence_ref must be ArtifactRef")


@dataclass(frozen=True, slots=True)
class _VerifiedRgbImage:
    observation_id: ObservationId
    width_px: int
    height_px: int
    rgb8: bytes


class SelaVprPlusRuntime(Protocol):
    def infer(
        self,
        *,
        source_root: Path,
        checkpoint_path: Path,
        images: tuple[_VerifiedRgbImage, ...],
        config: SelaVprPlusRetrievalConfig,
    ) -> tuple[SelaVprPlusEnvironmentIdentity, tuple[tuple[float, ...], ...]]: ...


def _uri_like(path: Path) -> bool:
    raw = str(path).strip().lower()
    return raw.startswith(_NETWORK_PREFIXES)


def _git_directory(source_root: Path) -> Path | None:
    dot_git = source_root / ".git"
    if dot_git.is_dir():
        return dot_git
    if not dot_git.is_file():
        return None
    try:
        marker = dot_git.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SelaVprPlusRetrievalError("cannot read SelaVPR++ .git marker") from exc
    prefix = "gitdir:"
    if not marker.lower().startswith(prefix):
        raise SelaVprPlusRetrievalError("SelaVPR++ .git marker is malformed")
    target = marker[len(prefix) :].strip()
    git_dir = Path(target)
    if not git_dir.is_absolute():
        git_dir = (source_root / git_dir).resolve()
    return git_dir


def _revision_from_git_dir(git_dir: Path) -> str:
    head = git_dir / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SelaVprPlusRetrievalError("cannot read SelaVPR++ git HEAD") from exc
    if not value.startswith("ref:"):
        return value
    ref = value[4:].strip()
    ref_path = git_dir / Path(ref)
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            sha, separator, name = line.partition(" ")
            if separator and name == ref:
                return sha
    raise SelaVprPlusRetrievalError("cannot resolve SelaVPR++ git HEAD ref")


def _verify_source_root(source_root: Path) -> Path:
    if _uri_like(source_root):
        raise SelaVprPlusRetrievalError("SelaVPR++ source must be an explicit local path")
    try:
        resolved = source_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SelaVprPlusRetrievalError("SelaVPR++ source root is unavailable") from exc
    if not resolved.is_dir():
        raise SelaVprPlusRetrievalError("SelaVPR++ source root must be a directory")

    git_dir = _git_directory(resolved)
    if git_dir is not None:
        revision = _revision_from_git_dir(git_dir)
    else:
        marker = resolved / ".wre-selavpr-plus-revision"
        if not marker.is_file():
            raise SelaVprPlusRetrievalError(
                "SelaVPR++ source root needs exact git HEAD or WRE revision marker"
            )
        try:
            revision = marker.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SelaVprPlusRetrievalError("cannot read SelaVPR++ revision marker") from exc
    if revision != SELAVPR_PLUS_SOURCE_REVISION:
        raise SelaVprPlusRetrievalError("SelaVPR++ source revision does not match reviewed code")

    for relative_path in _MODEL_REQUIRED_PATHS:
        candidate = resolved / relative_path
        if not candidate.is_file():
            raise SelaVprPlusRetrievalError(
                f"SelaVPR++ source is missing required file {relative_path!r}"
            )
    return resolved


def _verify_checkpoint(path: Path, checkpoint: CheckpointIdentity) -> Path:
    if _uri_like(path):
        raise SelaVprPlusRetrievalError("SelaVPR++ checkpoint must be an explicit local path")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SelaVprPlusRetrievalError("SelaVPR++ checkpoint is unavailable") from exc
    if not resolved.is_file():
        raise SelaVprPlusRetrievalError("SelaVPR++ checkpoint must be a regular file")
    content = hash_file_content(resolved)
    if content.sha256 != checkpoint.sha256:
        raise SelaVprPlusRetrievalError("SelaVPR++ checkpoint SHA-256 does not match")
    return resolved


def _verified_rgb_image(item: SelaVprPlusImageInput) -> _VerifiedRgbImage:
    result = item.decoded_result
    verification = verify_local_artifact_materialization(
        result.materialization,
        item.materialization_root,
    )
    if verification.status is not ArtifactMaterializationVerificationStatus.VERIFIED:
        raise SelaVprPlusRetrievalError(
            f"decoded-image materialization is {verification.status.value}, not verified"
        )
    manifest = result.manifest
    if manifest.pixel_layout is not DecodedImagePixelLayout.RGB8_PACKED:
        raise SelaVprPlusRetrievalError("SelaVPR++ requires packed RGB8 decoded pixels")
    if manifest.orientation_policy is not DecodedImageOrientationPolicy.SOURCE_PIXELS:
        raise SelaVprPlusRetrievalError("SelaVPR++ requires source-pixel orientation")
    level = manifest.levels[0]
    entries = {
        entry.relative_path: entry
        for entry in result.materialization.entries
    }
    entry = entries.get(level.relative_path)
    if entry is None:
        raise SelaVprPlusRetrievalError("decoded-image level zero is not materialized")
    expected_length = level.width_px * level.height_px * 3
    if entry.byte_length != expected_length:
        raise SelaVprPlusRetrievalError("decoded-image level zero has invalid RGB8 byte length")
    path = item.materialization_root.joinpath(*level.relative_path.split("/"))
    try:
        rgb8 = path.read_bytes()
    except OSError as exc:
        raise SelaVprPlusRetrievalError("cannot read verified decoded-image level zero") from exc
    if len(rgb8) != expected_length:
        raise SelaVprPlusRetrievalError("decoded-image level zero byte count changed after verify")
    return _VerifiedRgbImage(
        observation_id=manifest.source_observation_id,
        width_px=level.width_px,
        height_px=level.height_px,
        rgb8=rgb8,
    )


def _validate_descriptors(
    descriptors: tuple[tuple[float, ...], ...],
    *,
    expected_count: int,
    dimension: int,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(descriptors, tuple) or len(descriptors) != expected_count:
        raise SelaVprPlusRetrievalError("SelaVPR++ descriptor count does not match inputs")
    validated: list[tuple[float, ...]] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, tuple) or len(descriptor) != dimension:
            raise SelaVprPlusRetrievalError("SelaVPR++ descriptor dimension is invalid")
        values: list[float] = []
        for value in descriptor:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SelaVprPlusRetrievalError("SelaVPR++ descriptor values must be real numbers")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise SelaVprPlusRetrievalError("SelaVPR++ descriptors must be finite")
            values.append(numeric)
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 0.0:
            raise SelaVprPlusRetrievalError("SelaVPR++ descriptor norm must be positive")
        if abs(norm - 1.0) > _DESCRIPTOR_NORM_TOLERANCE:
            raise SelaVprPlusRetrievalError("SelaVPR++ descriptors must be L2-normalized")
        validated.append(tuple(values))
    return tuple(validated)


def propose_selavpr_plus_pairs(
    observation_ids: tuple[ObservationId, ...],
    descriptors: tuple[tuple[float, ...], ...],
    *,
    neighbors_per_observation: int,
) -> tuple[SelaVprPlusPair, ...]:
    """Pure exact cosine top-k proposal over already normalized descriptors."""

    if not isinstance(observation_ids, tuple):
        raise TypeError("observation_ids must be an immutable tuple")
    if len(observation_ids) < 2:
        raise ValueError("exact learned retrieval requires at least two observations")
    if any(not isinstance(item, ObservationId) for item in observation_ids):
        raise TypeError("observation_ids must contain only ObservationId values")
    if len({item.value for item in observation_ids}) != len(observation_ids):
        raise ValueError("observation_ids must not contain duplicates")
    if (
        isinstance(neighbors_per_observation, bool)
        or not isinstance(neighbors_per_observation, int)
        or neighbors_per_observation <= 0
    ):
        raise ValueError("neighbors_per_observation must be a positive integer")

    validated = _validate_descriptors(
        descriptors,
        expected_count=len(observation_ids),
        dimension=len(descriptors[0]) if descriptors else 0,
    )
    by_id = {
        observation_id.value: (observation_id, validated[index])
        for index, observation_id in enumerate(observation_ids)
    }
    ordered = tuple(by_id[key] for key in sorted(by_id))
    selected: dict[tuple[str, str], SelaVprPlusPair] = {}

    for source_index, (source_id, source_descriptor) in enumerate(ordered):
        neighbors: list[tuple[float, str, ObservationId]] = []
        for target_index, (target_id, target_descriptor) in enumerate(ordered):
            if source_index == target_index:
                continue
            similarity = sum(
                left * right
                for left, right in zip(source_descriptor, target_descriptor, strict=True)
            )
            if not math.isfinite(similarity):
                raise SelaVprPlusRetrievalError("cosine similarity is not finite")
            neighbors.append((similarity, target_id.value, target_id))
        neighbors.sort(key=lambda item: (-item[0], item[1]))
        for similarity, _, target_id in neighbors[:neighbors_per_observation]:
            first, second = sorted((source_id, target_id), key=lambda item: item.value)
            key = (first.value, second.value)
            candidate = SelaVprPlusPair(
                observation_id1=first,
                observation_id2=second,
                cosine_similarity=similarity,
            )
            previous = selected.get(key)
            if previous is None or candidate.cosine_similarity > previous.cosine_similarity:
                selected[key] = candidate

    return tuple(selected[key] for key in sorted(selected))


def _safe_checkpoint_state(torch_module: Any, checkpoint_path: Path) -> Mapping[str, Any]:
    try:
        payload = torch_module.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        raise SelaVprPlusRetrievalError(
            "installed torch does not support safe weights-only checkpoint loading"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SelaVprPlusRetrievalError("SelaVPR++ checkpoint payload must be a mapping")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise SelaVprPlusRetrievalError(
            "SelaVPR++ checkpoint must contain a non-empty model_state_dict mapping"
        )
    keys = tuple(state.keys())
    if any(not isinstance(key, str) or not key for key in keys):
        raise SelaVprPlusRetrievalError("SelaVPR++ state-dict keys must be non-empty strings")
    prefixed = tuple(key.startswith("module.") for key in cast(tuple[str, ...], keys))
    if any(prefixed) and not all(prefixed):
        raise SelaVprPlusRetrievalError("SelaVPR++ state dict has mixed module prefixes")
    if all(prefixed):
        return {cast(str, key)[len("module.") :]: value for key, value in state.items()}
    return cast(Mapping[str, Any], state)


@contextmanager
def _local_source_import(source_root: Path) -> Iterator[None]:
    conflicting = tuple(
        name for name in sys.modules if name == "model" or name.startswith("model.")
    )
    if conflicting:
        raise SelaVprPlusRetrievalError(
            "cannot load SelaVPR++ while another top-level model package is imported"
        )
    sys.path.insert(0, str(source_root))
    try:
        yield
    finally:
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)
        else:
            try:
                sys.path.remove(str(source_root))
            except ValueError:
                pass
        for name in tuple(sys.modules):
            if name == "model" or name.startswith("model."):
                sys.modules.pop(name, None)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise SelaVprPlusRetrievalError(
            f"required optional SelaVPR++ distribution {distribution!r} is unavailable"
        ) from exc


class _TorchSelaVprPlusRuntime:
    def _environment(self) -> SelaVprPlusEnvironmentIdentity:
        versions = {
            "torch": _package_version("torch"),
            "numpy": _package_version("numpy"),
            "faiss-cpu": _package_version("faiss-cpu"),
            "tqdm": _package_version("tqdm"),
        }
        expected = {
            "torch": SELAVPR_PLUS_TORCH_VERSION,
            "numpy": SELAVPR_PLUS_NUMPY_VERSION,
            "faiss-cpu": SELAVPR_PLUS_FAISS_CPU_VERSION,
            "tqdm": SELAVPR_PLUS_TQDM_VERSION,
        }
        if versions != expected:
            raise SelaVprPlusRetrievalError(
                "optional SelaVPR++ environment versions do not match the reviewed reference"
            )
        return SelaVprPlusEnvironmentIdentity(
            source_revision=SELAVPR_PLUS_SOURCE_REVISION,
            python_version=platform.python_version(),
            torch_version=versions["torch"],
            numpy_version=versions["numpy"],
            faiss_cpu_version=versions["faiss-cpu"],
            tqdm_version=versions["tqdm"],
            device="cpu",
        )

    def infer(
        self,
        *,
        source_root: Path,
        checkpoint_path: Path,
        images: tuple[_VerifiedRgbImage, ...],
        config: SelaVprPlusRetrievalConfig,
    ) -> tuple[SelaVprPlusEnvironmentIdentity, tuple[tuple[float, ...], ...]]:
        environment = self._environment()
        torch = importlib.import_module("torch")
        previous_threads = torch.get_num_threads()
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        descriptors: list[tuple[float, ...]] = []

        with torch.random.fork_rng(devices=[], enabled=True):
            try:
                torch.manual_seed(config.random_seed)
                torch.set_num_threads(1)
                torch.use_deterministic_algorithms(True)

                with _local_source_import(source_root):
                    network = importlib.import_module("model.network")
                    attention = importlib.import_module("model.dinov2.attention")
                    if bool(getattr(attention, "XFORMERS_AVAILABLE", False)):
                        raise SelaVprPlusRetrievalError(
                            "the CPU reference path requires xFormers to be absent"
                        )
                    model = network.GeoLocalizationNet(
                        SimpleNamespace(
                            backbone="dinov2-base",
                            aggregation="gem",
                            hashing=False,
                            rerank=False,
                            resume=True,
                            foundation_model_path=None,
                        )
                    )
                    state = _safe_checkpoint_state(torch, checkpoint_path)
                    try:
                        model.load_state_dict(state, strict=True)
                    except Exception as exc:
                        raise SelaVprPlusRetrievalError(
                            "SelaVPR++ checkpoint is incompatible with the reviewed model"
                        ) from exc
                    model = model.to("cpu")
                    model.eval()

                    mean = torch.tensor(
                        [0.485, 0.456, 0.406],
                        dtype=torch.float32,
                    ).view(1, 3, 1, 1)
                    std = torch.tensor(
                        [0.229, 0.224, 0.225],
                        dtype=torch.float32,
                    ).view(1, 3, 1, 1)
                    with torch.no_grad():
                        for image in images:
                            tensor = torch.frombuffer(
                                memoryview(image.rgb8),
                                dtype=torch.uint8,
                            ).clone()
                            tensor = tensor.reshape(image.height_px, image.width_px, 3)
                            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
                            tensor = tensor.div(255.0)
                            tensor = tensor.sub(mean).div(std)
                            tensor = torch.nn.functional.interpolate(
                                tensor,
                                size=(config.image_height_px, config.image_width_px),
                                mode="bilinear",
                                align_corners=False,
                                antialias=True,
                            )
                            output = model(tensor)
                            if getattr(output, "shape", None) != (
                                1,
                                config.descriptor_dimension,
                            ):
                                raise SelaVprPlusRetrievalError(
                                    "SelaVPR++ model returned an unexpected descriptor shape"
                                )
                            row = output[0].detach().to("cpu").tolist()
                            descriptors.append(tuple(float(value) for value in row))
            finally:
                torch.set_num_threads(previous_threads)
                torch.use_deterministic_algorithms(
                    previous_deterministic,
                    warn_only=previous_warn_only,
                )

        return environment, tuple(descriptors)


def retrieve_selavpr_plus_pairs(
    request: SelaVprPlusRetrievalRequest,
    *,
    runtime: SelaVprPlusRuntime | None = None,
) -> SelaVprPlusRetrievalResult:
    """Run the bounded learned retrieval candidate over verified decoded artifacts."""

    if not isinstance(request, SelaVprPlusRetrievalRequest):
        raise TypeError("request must be SelaVprPlusRetrievalRequest")

    source_root = _verify_source_root(request.source_root)
    checkpoint_path = _verify_checkpoint(request.checkpoint_path, request.checkpoint)
    images = tuple(_verified_rgb_image(item) for item in request.inputs)
    runtime_impl = runtime if runtime is not None else _TorchSelaVprPlusRuntime()
    environment, raw_descriptors = runtime_impl.infer(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        images=images,
        config=request.config,
    )
    if not isinstance(environment, SelaVprPlusEnvironmentIdentity):
        raise SelaVprPlusRetrievalError("SelaVPR++ runtime returned invalid environment identity")
    descriptors = _validate_descriptors(
        raw_descriptors,
        expected_count=len(images),
        dimension=request.config.descriptor_dimension,
    )
    pairs = propose_selavpr_plus_pairs(
        tuple(image.observation_id for image in images),
        descriptors,
        neighbors_per_observation=request.config.neighbors_per_observation,
    )
    return SelaVprPlusRetrievalResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=request.run.run_id,
            source_observation_ids=request.run.input_observation_ids,
        ),
        model=request.model,
        checkpoint=request.checkpoint,
        configuration_sha256=request.config.sha256,
        environment=environment,
        pairs=pairs,
    )


def adapt_selavpr_plus_retrieval_result(
    adapter_input: SelaVprPlusPairCandidateAdapterInput,
) -> tuple[PairCandidate, ...]:
    """Adapt source-specific learned proposals without leaking scores into PairCandidate."""

    if not isinstance(adapter_input, SelaVprPlusPairCandidateAdapterInput):
        raise TypeError("adapter_input must be SelaVprPlusPairCandidateAdapterInput")

    candidates = tuple(
        PairCandidate(
            observation_id1=pair.observation_id1,
            observation_id2=pair.observation_id2,
            sources=(
                PairCandidateSource(
                    source_id=SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
                    evidence_refs=(adapter_input.evidence_ref,),
                ),
            ),
        )
        for pair in adapter_input.result.pairs
    )
    return merge_pair_candidates(candidates)
