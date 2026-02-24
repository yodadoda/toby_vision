import cv2

from .config import PERSON_CLASS_NAME, VisionConfig
from .detection import Detection
from .tracking import TargetState


def get_detection_style(detection: Detection, target_state: TargetState | None, config: VisionConfig,
    ) -> tuple[tuple[int, int, int], int]:

    is_person = detection.class_name == PERSON_CLASS_NAME
    color = config.person_box_color if is_person else config.non_person_box_color

    is_locked_target = (
        target_state is not None and detection.bbox_xyxy == target_state.bbox_xyxy
    )
    thickness = config.locked_box_thickness if is_locked_target else config.box_thickness
    return color, thickness


def draw_overlay(
    frame, detections: list[Detection], target_state: TargetState | None, config: VisionConfig, ):
    """
    Render detections, deadzone, frame center, and current target annotations.
    """

    annotated = frame.copy()
    frame_center = (annotated.shape[1] // 2, annotated.shape[0] // 2)

    _draw_deadzone(annotated, frame_center, config)
    _draw_frame_center(annotated, frame_center, config)

    for detection in detections:
        color, thickness = get_detection_style(detection, target_state, config)
        x1, y1, x2, y2 = detection.bbox_xyxy

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"{detection.class_name} {detection.confidence:.2f}"
        label_y = max(y1 - 10, 20)
        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.label_font_scale,
            color,
            config.label_thickness,
            cv2.LINE_AA,
        )

    telemetry_line = "No person target"
    if target_state is not None:
        tx, ty = target_state.lock_point
        cv2.circle(
            annotated,
            (int(tx), int(ty)),
            config.target_point_radius,
            config.target_point_color,
            -1,
        )
        cv2.line(
            annotated,
            frame_center,
            (int(tx), int(ty)),
            config.target_point_color,
            1,
        )

        status = "CENTERED" if target_state.inside_deadzone else "OFF-CENTER"
        telemetry_line = (
            f"dx={target_state.error_x:+.1f} dy={target_state.error_y:+.1f} "
            f"dist={target_state.distance_px:.1f}px {status}"
        )

    cv2.putText(
        annotated,
        telemetry_line,
        (12, max(22, annotated.shape[0] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        config.target_point_color,
        2,
        cv2.LINE_AA,
    )

    return annotated


def _draw_deadzone(frame, frame_center: tuple[int, int], config: VisionConfig) -> None:
    x, y = frame_center
    top_left = (x - config.deadzone_x_px, y - config.deadzone_y_px)
    bottom_right = (x + config.deadzone_x_px, y + config.deadzone_y_px)
    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        config.deadzone_color,
        config.deadzone_thickness,
    )


def _draw_frame_center(frame, frame_center: tuple[int, int], config: VisionConfig) -> None:
    cv2.circle(
        frame,
        frame_center,
        config.frame_center_radius,
        config.frame_center_color,
        -1,
    )
