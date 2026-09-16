import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from ultralytics import YOLO

DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.7
DEFAULT_RELEVANT_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
)


class YOLODetectorError(RuntimeError):
    pass


def _parse_float(name: str, default: float) -> float:
    value = os.getenv(name, str(default)).strip()
    try:
        parsed = float(value)
    except ValueError as exc:
        raise YOLODetectorError(f"{name} must be a number.") from exc
    if not 0.0 <= parsed <= 1.0:
        raise YOLODetectorError(f"{name} must be between 0 and 1.")
    return parsed


def _resolve_device() -> str:
    configured = os.getenv("YOLO_DEVICE", "auto").strip().lower()
    if configured in {"", "auto"}:
        return "0" if torch.cuda.is_available() else "cpu"
    return configured


def _resolve_classes() -> set[str] | None:
    configured = os.getenv("YOLO_CLASSES")
    if configured is None:
        return set(DEFAULT_RELEVANT_CLASSES)
    class_names = {item.strip().lower() for item in configured.split(",") if item.strip()}
    return class_names or None


class YOLODetector:
    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        device: str | None = None,
        class_names: set[str] | None = None,
    ):
        load_dotenv(Path(__file__).resolve().with_name(".env"))
        self.model_path = str(model_path or os.getenv("YOLO_MODEL", DEFAULT_MODEL)).strip()
        self.confidence = confidence if confidence is not None else _parse_float("YOLO_CONFIDENCE", DEFAULT_CONFIDENCE)
        self.iou = iou if iou is not None else _parse_float("YOLO_IOU", DEFAULT_IOU)
        self.device = device or _resolve_device()
        self.class_names = _resolve_classes() if class_names is None else class_names
        try:
            self.model = YOLO(self.model_path)
        except Exception as exc:
            raise YOLODetectorError(
                f"Could not load YOLO model '{self.model_path}': {type(exc).__name__}."
            ) from exc

    def detect(self, image_path: str | Path) -> list[dict[str, Any]]:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise YOLODetectorError(f"Image does not exist: {image_path}")
        try:
            results = self.model.predict(
                source=str(image_path),
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise YOLODetectorError(
                f"YOLO inference failed for '{image_path}': {type(exc).__name__}."
            ) from exc
        if not results:
            return []

        result = results[0]
        names = result.names
        detections = []
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            try:
                class_id = int(class_id)
                class_name = str(names[class_id])
                if self.class_names is not None and class_name.lower() not in self.class_names:
                    continue
                if len(box) != 4:
                    raise ValueError("bounding box has an invalid shape")
                x1, y1, x2, y2 = (float(value) for value in box)
                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(confidence),
                        "bbox": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    }
                )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise YOLODetectorError("YOLO returned malformed detection data.") from exc
        return detections
