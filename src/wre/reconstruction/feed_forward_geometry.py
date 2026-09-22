from __future__ import annotations

from dataclasses import dataclass

from wre.domain.adapter_capabilities import AdapterCapabilityDescriptor, AdapterCapabilityName
from wre.domain.artifacts import ArtifactKind, ArtifactRef
from wre.domain.camera_solutions import CameraSolution
from wre.domain.depth_fields import DepthField
from wre.domain.geometry_solutions import GeometrySolution
from wre.domain.point_maps import PointMap

FEED_FORWARD_GEOMETRY_CAPABILITY_NAME = AdapterCapabilityName("geometry.feed_forward")
FEED_FORWARD_GEOMETRY_INPUT_KIND = ArtifactKind("media.decoded_image_pyramid")
FEED_FORWARD_GEOMETRY_OUTPUT_KINDS = frozenset(
    {
        ArtifactKind("geometry.camera_solution"),
        ArtifactKind("geometry.depth_field"),
        ArtifactKind("geometry.point_map"),
        ArtifactKind("geometry.solution"),
    }
)
FEED_FORWARD_GEOMETRY_CAPABILITY = AdapterCapabilityDescriptor(
    capability=FEED_FORWARD_GEOMETRY_CAPABILITY_NAME,
    input_kinds=frozenset({FEED_FORWARD_GEOMETRY_INPUT_KIND}),
    output_kinds=FEED_FORWARD_GEOMETRY_OUTPUT_KINDS,
)


def normalize_feed_forward_geometry_inputs(
    inputs: tuple[ArtifactRef, ...],
) -> tuple[ArtifactRef, ...]:
    """Validate canonical decoded-image inputs without reading or rewriting them."""

    if not isinstance(inputs, tuple):
        raise TypeError("feed-forward geometry inputs must be an immutable tuple")
    if not inputs:
        raise ValueError("feed-forward geometry inputs must contain at least one artifact")
    if any(not isinstance(item, ArtifactRef) for item in inputs):
        raise TypeError("feed-forward geometry inputs must contain only ArtifactRef values")
    if any(item.artifact_kind != FEED_FORWARD_GEOMETRY_INPUT_KIND for item in inputs):
        raise ValueError(
            "feed-forward geometry inputs must all have kind media.decoded_image_pyramid"
        )
    if len(inputs) != len(set(inputs)):
        raise ValueError("feed-forward geometry inputs cannot contain duplicate ArtifactRef values")

    canonical = tuple(
        sorted(
            inputs,
            key=lambda item: (item.artifact_kind.value, item.artifact_id.value),
        )
    )
    if inputs != canonical:
        raise ValueError("feed-forward geometry inputs must be in canonical ArtifactRef order")

    return inputs


@dataclass(frozen=True, slots=True)
class FeedForwardGeometryResult:
    """Canonical geometry values emitted by one feed-forward adapter execution.

    This aggregate validates ownership and referential integrity only. It performs
    no solver execution, coordinate conversion, scale inference, unit conversion,
    alignment, quality decision, routing, fallback, or product-layer promotion.
    """

    camera_solutions: tuple[CameraSolution, ...]
    depth_fields: tuple[DepthField, ...]
    point_maps: tuple[PointMap, ...]
    geometry_solution: GeometrySolution

    def __post_init__(self) -> None:
        if not isinstance(self.camera_solutions, tuple):
            raise TypeError("feed_forward_geometry.camera_solutions must be an immutable tuple")
        if not self.camera_solutions:
            raise ValueError("feed_forward_geometry requires at least one CameraSolution")
        if any(not isinstance(item, CameraSolution) for item in self.camera_solutions):
            raise TypeError(
                "feed_forward_geometry.camera_solutions members must be CameraSolution"
            )

        camera_observation_values = tuple(
            item.observation_id.value for item in self.camera_solutions
        )
        if len(camera_observation_values) != len(set(camera_observation_values)):
            raise ValueError("feed_forward_geometry camera observations must be unique")
        if camera_observation_values != tuple(sorted(camera_observation_values)):
            raise ValueError(
                "feed_forward_geometry camera_solutions must be in canonical ObservationId order"
            )
        camera_solution_values = tuple(item.solution_id.value for item in self.camera_solutions)
        if len(camera_solution_values) != len(set(camera_solution_values)):
            raise ValueError("feed_forward_geometry camera solution IDs must be unique")

        if not isinstance(self.depth_fields, tuple):
            raise TypeError("feed_forward_geometry.depth_fields must be an immutable tuple")
        if any(not isinstance(item, DepthField) for item in self.depth_fields):
            raise TypeError("feed_forward_geometry.depth_fields members must be DepthField")
        depth_observation_values = tuple(item.observation_id.value for item in self.depth_fields)
        if len(depth_observation_values) != len(set(depth_observation_values)):
            raise ValueError("feed_forward_geometry depth observations must be unique")
        if depth_observation_values != tuple(sorted(depth_observation_values)):
            raise ValueError(
                "feed_forward_geometry depth_fields must be in canonical ObservationId order"
            )
        depth_id_values = tuple(item.depth_field_id.value for item in self.depth_fields)
        if len(depth_id_values) != len(set(depth_id_values)):
            raise ValueError("feed_forward_geometry depth field IDs must be unique")

        if not isinstance(self.point_maps, tuple):
            raise TypeError("feed_forward_geometry.point_maps must be an immutable tuple")
        if any(not isinstance(item, PointMap) for item in self.point_maps):
            raise TypeError("feed_forward_geometry.point_maps members must be PointMap")
        point_map_values = tuple(item.point_map_id.value for item in self.point_maps)
        if len(point_map_values) != len(set(point_map_values)):
            raise ValueError("feed_forward_geometry point map IDs must be unique")
        if point_map_values != tuple(sorted(point_map_values)):
            raise ValueError(
                "feed_forward_geometry point_maps must be in canonical PointMapId order"
            )

        if not self.depth_fields and not self.point_maps:
            raise ValueError(
                "feed_forward_geometry requires at least one DepthField or PointMap"
            )
        if not isinstance(self.geometry_solution, GeometrySolution):
            raise TypeError("feed_forward_geometry.geometry_solution must be GeometrySolution")

        local_frame = self.geometry_solution.local_frame_id
        if any(item.local_frame_id != local_frame for item in self.camera_solutions):
            raise ValueError(
                "feed_forward_geometry cameras and GeometrySolution must share one LocalFrameId"
            )
        if any(item.local_frame_id != local_frame for item in self.point_maps):
            raise ValueError(
                "feed_forward_geometry PointMaps and GeometrySolution must share one LocalFrameId"
            )

        cameras_by_observation = {
            item.observation_id: item for item in self.camera_solutions
        }
        for depth in self.depth_fields:
            camera = cameras_by_observation.get(depth.observation_id)
            if camera is None:
                raise ValueError(
                    "feed_forward_geometry DepthField observation must have an emitted CameraSolution"
                )
            if depth.camera_solution_id != camera.solution_id:
                raise ValueError(
                    "feed_forward_geometry DepthField must reference its emitted CameraSolution"
                )
            if depth.dimensions != camera.dimensions:
                raise ValueError(
                    "feed_forward_geometry DepthField dimensions must match its CameraSolution"
                )

        camera_observations = set(cameras_by_observation)
        for point_map in self.point_maps:
            if not set(point_map.source_observation_ids).issubset(camera_observations):
                raise ValueError(
                    "feed_forward_geometry PointMap support must stay inside camera membership"
                )

        expected_camera_ids = tuple(
            sorted((item.solution_id for item in self.camera_solutions), key=lambda item: item.value)
        )
        expected_depth_ids = tuple(
            sorted((item.depth_field_id for item in self.depth_fields), key=lambda item: item.value)
        )
        expected_point_map_ids = tuple(
            sorted((item.point_map_id for item in self.point_maps), key=lambda item: item.value)
        )

        if self.geometry_solution.camera_solution_ids != expected_camera_ids:
            raise ValueError(
                "feed_forward_geometry GeometrySolution must reference exactly all emitted cameras"
            )
        if self.geometry_solution.depth_field_ids != expected_depth_ids:
            raise ValueError(
                "feed_forward_geometry GeometrySolution must reference exactly all emitted depths"
            )
        if self.geometry_solution.point_map_ids != expected_point_map_ids:
            raise ValueError(
                "feed_forward_geometry GeometrySolution must reference exactly all emitted point maps"
            )
