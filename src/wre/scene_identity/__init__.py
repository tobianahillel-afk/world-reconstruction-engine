"""Scene-identity evidence adapters behind stable WRE contracts."""

from wre.scene_identity.colmap_adapter import (
    ColmapSceneRelationAdapterInput,
    adapt_colmap_scene_relations,
)

__all__ = [
    "ColmapSceneRelationAdapterInput",
    "adapt_colmap_scene_relations",
]
