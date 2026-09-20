"""Scene-identity evidence adapters behind stable WRE contracts."""

from wre.scene_identity.clustering import (
    SceneClusteringConflictError,
    SceneRelationBatch,
    VerifiedSceneClusteringInput,
    cluster_verified_scene_relation_batches,
    cluster_verified_scene_relations,
)
from wre.scene_identity.colmap_adapter import (
    ColmapSceneRelationAdapterInput,
    adapt_colmap_scene_relations,
)

__all__ = [
    "ColmapSceneRelationAdapterInput",
    "SceneClusteringConflictError",
    "SceneRelationBatch",
    "VerifiedSceneClusteringInput",
    "adapt_colmap_scene_relations",
    "cluster_verified_scene_relation_batches",
    "cluster_verified_scene_relations",
]
