from typing import Tuple
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
    if getattr(settings, "DEVICE", "auto") == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return settings.DEVICE


def get_dino() -> Tuple[GroundingDinoProcessor, GroundingDinoForObjectDetection, str]:
    global _processor, _model, _device
    if _processor is None or _model is None or _device is None:
        _device = _get_device()
        print("🦕 Loading GroundingDINO on device:", _device)
        _processor = GroundingDinoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        _model = GroundingDinoForObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(_device)
        _model.eval()
    return _processor, _model, _device


def run_detection(
    image: Image.Image,
    prompt: str,
    box_thresh: float = 0.3,
) -> sv.Detections:
    processor, model, device = get_dino()

    print(f"🔍 Detecting objects for prompt={repr(prompt)}, box_thresh={box_thresh}")
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
        print("⚠️ No boxes after post-processing")
        return sv.Detections.empty()

    boxes = result["boxes"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    labels = result["labels"].cpu().numpy().astype(int)

    print(f"✅ {len(boxes)} boxes found. Scores: {scores}")

    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=labels,
    )

    labels_str = [f"Object_ID_{cls_id}" for cls_id in detections.class_id]
    detections.data["class_name"] = np.array(labels_str)

    return detections
