import unittest

from toby_vision.config import VisionConfig
from toby_vision.detection import Detection
from toby_vision.overlay import get_detection_style
from toby_vision.tracking import (
    TargetState,
    build_target_state,
    compute_lock_point,
    compute_tracking_error,
    select_target_person,
)


class TrackingTests(unittest.TestCase):
    def test_compute_lock_point_top_third(self):
        bbox = (100, 100, 200, 400)
        self.assertEqual(compute_lock_point(bbox), (150, 199))

    def test_select_target_person_nearest_center(self):
        frame_center = (320, 240)
        near_center_person = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox_xyxy=(280, 100, 360, 380),
        )
        far_person = Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox_xyxy=(430, 100, 520, 380),
        )
        non_person = Detection(
            class_id=1,
            class_name="bicycle",
            confidence=0.8,
            bbox_xyxy=(10, 10, 80, 80),
        )

        selected = select_target_person(
            detections=[non_person, far_person, near_center_person],
            frame_center=frame_center,
            previous_target=None,
            lock_persistence_radius_px=80.0,
        )

        self.assertEqual(selected, near_center_person)

    def test_select_target_person_prefers_previous_lock_when_close(self):
        frame_center = (320, 240)
        previous_target = TargetState(
            lock_point=(460, 180),
            bbox_xyxy=(420, 80, 500, 380),
            error_x=140.0,
            error_y=-60.0,
            distance_px=152.32,
            inside_deadzone=False,
        )
        near_previous = Detection(
            class_id=0,
            class_name="person",
            confidence=0.7,
            bbox_xyxy=(430, 90, 510, 390),
        )
        near_center = Detection(
            class_id=0,
            class_name="person",
            confidence=0.8,
            bbox_xyxy=(290, 100, 350, 350),
        )

        selected = select_target_person(
            detections=[near_center, near_previous],
            frame_center=frame_center,
            previous_target=previous_target,
            lock_persistence_radius_px=80.0,
        )

        self.assertEqual(selected, near_previous)

    def test_compute_tracking_error_deadzone(self):
        frame_center = (320, 240)

        boundary = compute_tracking_error(
            lock_point=(350, 210),
            frame_center=frame_center,
            deadzone_x=30,
            deadzone_y=30,
        )
        self.assertEqual(boundary[0], 30.0)
        self.assertEqual(boundary[1], -30.0)
        self.assertTrue(boundary[3])

        outside = compute_tracking_error(
            lock_point=(351, 240),
            frame_center=frame_center,
            deadzone_x=30,
            deadzone_y=30,
        )
        self.assertFalse(outside[3])

    def test_no_person_returns_none_target(self):
        frame_center = (320, 240)
        detections = [
            Detection(
                class_id=2,
                class_name="car",
                confidence=0.77,
                bbox_xyxy=(100, 100, 180, 180),
            )
        ]

        selected = select_target_person(
            detections=detections,
            frame_center=frame_center,
            previous_target=None,
            lock_persistence_radius_px=80.0,
        )
        self.assertIsNone(selected)

        target_state = build_target_state(
            selected_person=selected,
            frame_center=frame_center,
            deadzone_x=30,
            deadzone_y=30,
        )
        self.assertIsNone(target_state)

    def test_color_mapping_person_red_non_person_black(self):
        config = VisionConfig()

        person = Detection(
            class_id=0,
            class_name="person",
            confidence=0.91,
            bbox_xyxy=(20, 20, 80, 200),
        )
        cup = Detection(
            class_id=41,
            class_name="cup",
            confidence=0.66,
            bbox_xyxy=(200, 200, 260, 300),
        )

        target_state = TargetState(
            lock_point=(50, 79),
            bbox_xyxy=person.bbox_xyxy,
            error_x=5.0,
            error_y=-8.0,
            distance_px=9.43,
            inside_deadzone=True,
        )

        person_color, person_thickness = get_detection_style(person, target_state, config)
        cup_color, cup_thickness = get_detection_style(cup, target_state, config)

        self.assertEqual(person_color, (0, 0, 255))
        self.assertEqual(cup_color, (0, 0, 0))
        self.assertEqual(person_thickness, config.locked_box_thickness)
        self.assertEqual(cup_thickness, config.box_thickness)


if __name__ == "__main__":
    unittest.main()
