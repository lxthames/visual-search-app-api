# app/services/detection/yolo_v11_detection.py
from __future__ import annotations

import base64
from functools import lru_cache
from typing import Any, Dict, List

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import supervision as sv

from app.core.config import settings


@lru_cache(maxsize=1)
def get_yolo_v11_model() -> YOLO:
    """
    Load YOLO v11 model once.
    Expects settings.YOLO_V11_MODEL_PATH; defaults to 'yolo11n.pt' if not set.
    """
    model_path = getattr(settings, "YOLO_V11_MODEL_PATH", None) or "yolo11n.pt"
    return YOLO(model_path)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR."""
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_base64_png(image_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image as base64 PNG string."""
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode annotated image as PNG")
    png_bytes = buffer.tobytes()
    return base64.b64encode(png_bytes).decode("utf-8")


def run_yolo_v11_detection(
    image: Image.Image,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Dict[str, Any]:
    """
    Run YOLO v11 on a single image.

    Returns:
        {
          "detections": [
             {
               "bbox": [x1, y1, x2, y2],
               "confidence": float,
               "class_id": int,
               "class_name": str,
             },
             ...
          ],
          "annotated_image_base64": "<base64 PNG>",
        }
    """
    model = get_yolo_v11_model()

    img_bgr = pil_to_bgr(image)

    results = model.predict(
        source=img_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    if not results:
        return {"detections": [], "annotated_image_base64": ""}

    result = results[0]

    # Convert Ultralytics result to supervision.Detections
    detections = sv.Detections.from_ultralytics(result)

    # Prepare labels "class_name conf"
    labels: List[str] = []
    for i in range(len(detections)):
        class_id = int(detections.class_id[i])
        class_name = result.names.get(class_id, str(class_id))
        confidence = float(detections.confidence[i])
        labels.append(f"{class_name} {confidence:.2f}")

    # Annotate image
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
    )
    annotated_bgr = box_annotator.annotate(
        scene=img_bgr.copy(),
        detections=detections,
        labels=labels,
    )

    annotated_b64 = bgr_to_base64_png(annotated_bgr)

    # Build detections list
    det_list: List[Dict[str, Any]] = []
    xyxy = detections.xyxy

    for i in range(len(detections)):
        x1, y1, x2, y2 = xyxy[i].tolist()
        class_id = int(detections.class_id[i])
        confidence = float(detections.confidence[i])
        class_name = result.names.get(class_id, str(class_id))

        det_list.append(
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
            }
        )

    return {
        "detections": det_list,
        "annotated_image_base64": annotated_b64,
    }
