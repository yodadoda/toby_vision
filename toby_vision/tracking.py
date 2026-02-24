from dataclasses import dataclass
import math

from .config import PERSON_CLASS_NAME
from .detection import Detection


@dataclass(frozen=True)
class TargetState:
    lock_point: tuple[int, int]
    bbox_xyxy: tuple[int, int, int, int]
    error_x: float
    error_y: float
    distance_px: float
    inside_deadzone: bool


def compute_lock_point(bbox_xyxy: tuple[int, int, int, int]) -> tuple[int, int]:
    """
    Approximate face center using top-third center of the person box.
    """
    x1, y1, x2, y2 = bbox_xyxy
    center_x = int(round((x1 + x2) / 2.0))
    center_y = int(round(y1 + 0.33 * (y2 - y1)))
    
    return center_x, center_y


def compute_tracking_error(
    lock_point: tuple[int, int], frame_center: tuple[int, int],
    deadzone_x: int, deadzone_y: int,) -> tuple[float, float, float, bool]:
    
    """
    Compute signed pixel error and deadzone state.
    """
    error_x = float(lock_point[0] - frame_center[0])
    error_y = float(lock_point[1] - frame_center[1])

    distance_px = math.hypot(error_x, error_y)
    inside_deadzone = abs(error_x) <= deadzone_x and abs(error_y) <= deadzone_y

    return error_x, error_y, distance_px, inside_deadzone


def select_target_person(
    detections: list[Detection], frame_center: tuple[int, int],
    previous_target: TargetState | None, lock_persistence_radius_px: float,) -> Detection | None:
    
    """
    Select person to track, favoring lock persistence and center proximity.
    """
    person_detections = [d for d in detections if d.class_name == PERSON_CLASS_NAME]
    if not person_detections:
        return None

    if previous_target is not None:
        closest_to_previous = min(
            person_detections,
            key=lambda det: _distance(compute_lock_point(det.bbox_xyxy), previous_target.lock_point),
        )
        previous_distance = _distance(
            compute_lock_point(closest_to_previous.bbox_xyxy),
            previous_target.lock_point,
        )
        if previous_distance <= lock_persistence_radius_px:
            return closest_to_previous

    return min(
        person_detections,
        key=lambda det: _distance(compute_lock_point(det.bbox_xyxy), frame_center),
    )


def build_target_state(
    selected_person: Detection | None, frame_center: tuple[int, int],
    deadzone_x: int, deadzone_y: int,) -> TargetState | None:
    
    if selected_person is None:
        return None

    lock_point = compute_lock_point(selected_person.bbox_xyxy)
    error_x, error_y, distance_px, inside_deadzone = compute_tracking_error(
        lock_point,
        frame_center,
        deadzone_x,
        deadzone_y,
    )

    return TargetState(
        lock_point=lock_point,
        bbox_xyxy=selected_person.bbox_xyxy,
        error_x=error_x,
        error_y=error_y,
        distance_px=distance_px,
        inside_deadzone=inside_deadzone,
    )


def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
