from dataclasses import dataclass
import cv2

@dataclass(frozen=True)
class CameraInfo:
    index: int
    width: int
    height: int


def list_available_cameras(max_index: int) -> list[CameraInfo]:
    """
    Probe indexes 0. max_index and return available camera devices.
    """
    cameras: list[CameraInfo] = []

    for index in range(max_index + 1):
        probe = cv2.VideoCapture(index)
        if not probe.isOpened():
            probe.release()
            continue

        ret, _ = probe.read()
        if ret:
            width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append(CameraInfo(index=index, width=width, height=height))
        probe.release()

    return cameras


def prompt_for_camera(cameras: list[CameraInfo], default_index: int) -> int:
    """
    Prompt user to select one of the discovered cameras.
    """
    print("\nAvailable cameras:")

    for camera in cameras:
        print(f"  {camera.index}: {camera.width}x{camera.height}")

    valid_indexes = {camera.index for camera in cameras}
    if default_index not in valid_indexes:
        default_index = cameras[0].index

    while True:
        raw = input(f"\nSelect camera index [{default_index}]: ").strip()
        if raw == "":
            return default_index

        try:
            selected = int(raw)
        except ValueError:
            print("Invalid input. Enter a number shown above.")
            continue

        if selected in valid_indexes:
            return selected

        print(f"Invalid camera index. Choose one of: {sorted(valid_indexes)}")


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """
    Create and configure an OpenCV capture object.
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    return cap
