from dataclasses import dataclass
from pathlib import Path


PERSON_CLASS_NAME = "person"


@dataclass
class VisionConfig:
    # Model + inference settings
    model_path: str = "yolo26n.pt"
    confidence_threshold: float = 0.5
    frame_rate: float = 10
    temperature: float = 0.2  # Reserved for future brain integration.

    # Camera settings
    camera_index: int = 0
    max_camera_scan_index: int = 8
    window_name: str = "TOBY Vision"
    mirror_camera: bool = True

    # Tracking settings
    deadzone_x_px: int = 30
    deadzone_y_px: int = 30
    lock_persistence_radius_px: float = 80.0

    # Rendering settings (BGR for OpenCV)
    non_person_box_color: tuple[int, int, int] = (0, 0, 0)
    person_box_color: tuple[int, int, int] = (0, 0, 255)
    target_point_color: tuple[int, int, int] = (255, 255, 255)
    frame_center_color: tuple[int, int, int] = (255, 255, 255)
    deadzone_color: tuple[int, int, int] = (255, 255, 255)

    box_thickness: int = 2
    locked_box_thickness: int = 4
    label_font_scale: float = 0.6
    label_thickness: int = 2
    target_point_radius: int = 5
    frame_center_radius: int = 4
    deadzone_thickness: int = 1


def resolve_model_path(model_path: str) -> str:

    """
    Resolve relative model paths from project root.
    """
    path = Path(model_path)
    
    if path.is_absolute():
        return str(path)
    return str(Path(__file__).resolve().parent.parent / path)
