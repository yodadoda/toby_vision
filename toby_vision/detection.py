from dataclasses import dataclass
from typing import Iterable
from .config import VisionConfig, resolve_model_path


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


class YoloDetector:
    def __init__(self, config: VisionConfig):
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(resolve_model_path(config.model_path))
        self.names = self.model.names

    def detect(self, frame) -> list[Detection]:
        """
        Run one YOLO pass and return normalized detections.
        """
        results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: list[Detection] = []
        for box in boxes:
            x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = self._get_class_name(class_id)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )

        return detections

    def _get_class_name(self, class_id: int) -> str:
        if isinstance(self.names, dict):
            return str(self.names.get(class_id, class_id))
        if isinstance(self.names, Iterable):
            names_list = list(self.names)
            if 0 <= class_id < len(names_list):
                return str(names_list[class_id])
            
        return str(class_id)
