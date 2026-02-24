import time
import cv2

from .camera import list_available_cameras, open_camera, prompt_for_camera
from .config import VisionConfig
from .detection import YoloDetector
from .overlay import draw_overlay
from .tracking import build_target_state, select_target_person


def format_tracking_output(target_state) -> str:
    if target_state is None:
        return "No person target"

    status = "CENTERED" if target_state.inside_deadzone else "OFF-CENTER"
    return (
        f"dx={target_state.error_x:+.1f} dy={target_state.error_y:+.1f} "
        f"dist={target_state.distance_px:.1f}px {status}"
    )


def run_vision_loop(config: VisionConfig) -> None:
    cameras = list_available_cameras(config.max_camera_scan_index)
    if not cameras:
        raise RuntimeError("No cameras detected. Check camera connection/permissions.")

    config.camera_index = prompt_for_camera(cameras, config.camera_index)

    detector = YoloDetector(config)
    cap = open_camera(config.camera_index)

    frame_interval = 1.0 / max(config.frame_rate, 0.1)
    previous_target = None

    print(
        "\nPress Q in the camera window (or Ctrl+C) to stop.\n"
        f"Camera={config.camera_index}, FPS={config.frame_rate}, "
        f"Conf={config.confidence_threshold}, Temp={config.temperature}, "
        f"Mirror={config.mirror_camera}"
    )

    try:
        while True:
            loop_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("Camera error.")
                break
            if config.mirror_camera:
                frame = cv2.flip(frame, 1)

            detections = detector.detect(frame)
            frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

            selected_person = select_target_person(
                detections=detections,
                frame_center=frame_center,
                previous_target=previous_target,
                lock_persistence_radius_px=config.lock_persistence_radius_px,
            )
            target_state = build_target_state(
                selected_person=selected_person,
                frame_center=frame_center,
                deadzone_x=config.deadzone_x_px,
                deadzone_y=config.deadzone_y_px,
            )
            previous_target = target_state

            annotated = draw_overlay(frame, detections, target_state, config)
            cv2.imshow(config.window_name, annotated)

            print(format_tracking_output(target_state))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    config = VisionConfig()
    run_vision_loop(config)
