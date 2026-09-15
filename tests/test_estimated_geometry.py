from __future__ import annotations

from wre.domain.estimated_geometry import (
    CameraCalibrationEstimate,
    CameraCalibrationEstimateId,
    CameraPoseEstimate,
    EstimatedPoint3DId,
    EstimatedTrackElement,
    LocalScaleStatus,
    Point3DEstimate,
    SparseReconstructionEstimate,
    SparseReconstructionEstimateId,
)
from wre.domain.fragments import LocalFrameId
from wre.domain.observations import ObservationId
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRunId


def _provenance(*values: str) -> DerivedArtifactProvenance:
    return DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId("run:l36"),
        source_observation_ids=tuple(ObservationId(value) for value in values),
    )


def _calibration(*values: str) -> CameraCalibrationEstimate:
    return CameraCalibrationEstimate(
        calibration_id=CameraCalibrationEstimateId("calibration:test"),
        projection_model="SIMPLE_RADIAL",
        width_px=640,
        height_px=480,
        parameters=(500.0, 320.0, 240.0, 0.01),
        has_prior_focal_length=False,
        provenance=_provenance(*values),
    )


def _pose(observation_id: str) -> CameraPoseEstimate:
    observation = ObservationId(observation_id)
    return CameraPoseEstimate(
        observation_id=observation,
        local_frame_id=LocalFrameId("frame:test"),
        calibration_id=CameraCalibrationEstimateId("calibration:test"),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation_xyz=(0.0, 0.0, 0.0),
        provenance=_provenance(observation_id),
    )


def test_sparse_estimate_keeps_local_scale_explicit_and_canonical() -> None:
    point = Point3DEstimate(
        point_id=EstimatedPoint3DId("point:test"),
        local_frame_id=LocalFrameId("frame:test"),
        position_xyz=(1.0, 2.0, 3.0),
        reprojection_error_px=0.25,
        track=(
            EstimatedTrackElement(ObservationId("obs:b"), 4),
            EstimatedTrackElement(ObservationId("obs:a"), 2),
        ),
        provenance=_provenance("obs:a", "obs:b"),
    )
    estimate = SparseReconstructionEstimate(
        estimate_id=SparseReconstructionEstimateId("sparse:test"),
        local_frame_id=LocalFrameId("frame:test"),
        scale_status=LocalScaleStatus.UNRESOLVED,
        provenance=_provenance("obs:a", "obs:b"),
        camera_calibrations=(_calibration("obs:a", "obs:b"),),
        camera_poses=(_pose("obs:b"), _pose("obs:a")),
        points3d=(point,),
    )

    assert estimate.scale_status is LocalScaleStatus.UNRESOLVED
    assert estimate.observation_ids == (ObservationId("obs:a"), ObservationId("obs:b"))
    assert tuple(item.observation_id for item in estimate.camera_poses) == estimate.observation_ids
    assert tuple(item.observation_id for item in estimate.points3d[0].track) == (
        ObservationId("obs:a"),
        ObservationId("obs:b"),
    )


def test_camera_pose_rejects_non_rotation_matrix() -> None:
    try:
        CameraPoseEstimate(
            observation_id=ObservationId("obs:a"),
            local_frame_id=LocalFrameId("frame:test"),
            calibration_id=CameraCalibrationEstimateId("calibration:test"),
            rotation_matrix=((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            translation_xyz=(0.0, 0.0, 0.0),
            provenance=_provenance("obs:a"),
        )
    except ValueError as exc:
        assert "unit length" in str(exc)
    else:
        raise AssertionError("non-rotation matrix must be rejected")


def test_point_track_rejects_two_features_from_same_observation() -> None:
    try:
        Point3DEstimate(
            point_id=EstimatedPoint3DId("point:test"),
            local_frame_id=LocalFrameId("frame:test"),
            position_xyz=(1.0, 2.0, 3.0),
            reprojection_error_px=None,
            track=(
                EstimatedTrackElement(ObservationId("obs:a"), 2),
                EstimatedTrackElement(ObservationId("obs:a"), 3),
            ),
            provenance=_provenance("obs:a"),
        )
    except ValueError as exc:
        assert "multiple features" in str(exc)
    else:
        raise AssertionError("duplicate observation in point track must be rejected")


def test_sparse_estimate_rejects_unreferenced_calibration() -> None:
    extra = CameraCalibrationEstimate(
        calibration_id=CameraCalibrationEstimateId("calibration:extra"),
        projection_model="SIMPLE_RADIAL",
        width_px=640,
        height_px=480,
        parameters=(500.0, 320.0, 240.0, 0.01),
        has_prior_focal_length=False,
        provenance=_provenance("obs:a"),
    )
    try:
        SparseReconstructionEstimate(
            estimate_id=SparseReconstructionEstimateId("sparse:test"),
            local_frame_id=LocalFrameId("frame:test"),
            scale_status=LocalScaleStatus.UNRESOLVED,
            provenance=_provenance("obs:a"),
            camera_calibrations=(_calibration("obs:a"), extra),
            camera_poses=(_pose("obs:a"),),
            points3d=(),
        )
    except ValueError as exc:
        assert "exactly match pose references" in str(exc)
    else:
        raise AssertionError("unreferenced calibration must be rejected")
