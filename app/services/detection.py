from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection
import supervision as sv

from app.core.config import settings

_processor: GroundingDinoProcessor | None = None
_model: GroundingDinoForObjectDetection | None = None
_device: str | None = None


def _get_device() -> str:
    if settings.DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return settings.DEVICE


def get_dino() -> Tuple[GroundingDinoProcessor, GroundingDinoForObjectDetection, str]:
    global _processor, _model, _device
    if _processor is None or _model is None or _device is None:
        _device = _get_device()
        _processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        _model = GroundingDinoForObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(_device)
    return _processor, _model, _device


def run_detection(
    image: Image.Image,
    prompt: str,
    box_thresh: float = 0.3,
) -> sv.Detections:
    processor, model, device = get_dino()

    if not prompt.endswith("."):
        prompt = prompt + "."

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    width, height = image.size

    postprocessed_outputs = processor.image_processor.post_process_object_detection(
        outputs,
        target_sizes=[(height, width)],
        threshold=box_thresh,
    )
    result = postprocessed_outputs[0]

    if len(result["boxes"]) == 0:
        return sv.Detections.empty()

    detections = sv.Detections(
        xyxy=result["boxes"].cpu().numpy(),
        confidence=result["scores"].cpu().numpy(),
        class_id=result["labels"].cpu().numpy().astype(int),
    )

    labels_str = [f"Object_ID_{cls_id}" for cls_id in detections.class_id]
    detections.data["class_name"] = np.array(labels_str)

    return detections
