from __future__ import annotations

from typing import List, Optional, Dict, Any
from pathlib import Path
from PIL import Image
import json

from app.core.config import settings

SHELF_IMAGE: Optional[Image.Image] = None
SHELF_IMAGE_ID: Optional[str] = None
SHELF_BBOXES: List[List[int]] = []
SHELF_IMAGE_PATH: Optional[Path] = None


def set_shelf(
    image: Image.Image,
    image_id: str,
    bboxes: List[List[int]],
    persist: bool = True,
) -> None:
    """Set current shelf in memory, and optionally persist it to disk."""
    global SHELF_IMAGE, SHELF_IMAGE_ID, SHELF_BBOXES, SHELF_IMAGE_PATH

    SHELF_IMAGE = image
    SHELF_IMAGE_ID = image_id
    SHELF_BBOXES = bboxes

    if not persist:
        return

    # Ensure shelf dir exists
    settings.SHELF_DIR.mkdir(parents=True, exist_ok=True)

    # Save shelf image
    img_path = settings.SHELF_DIR / f"{image_id}.png"
    image.save(img_path)
    SHELF_IMAGE_PATH = img_path

    # Save state JSON
    state_data: Dict[str, Any] = {
        "image_id": image_id,
        "image_path": str(img_path),
        "bboxes": bboxes,
    }

    settings.SHELF_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with settings.SHELF_STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(state_data, f, ensure_ascii=False, indent=2)


def load_shelf_from_disk() -> None:
    """Load last shelf image and bboxes from disk, if available."""
    global SHELF_IMAGE, SHELF_IMAGE_ID, SHELF_BBOXES, SHELF_IMAGE_PATH

    if not settings.SHELF_STATE_PATH.exists():
        return

    try:
        with settings.SHELF_STATE_PATH.open("r", encoding="utf-8") as f:
            state_data = json.load(f)

        image_id = state_data.get("image_id")
        image_path_str = state_data.get("image_path")
        bboxes = state_data.get("bboxes", [])

        if not image_id or not image_path_str:
            return

        img_path = Path(image_path_str)
        if not img_path.exists():
            return

        image = Image.open(img_path).convert("RGB")

        SHELF_IMAGE = image
        SHELF_IMAGE_ID = image_id
        SHELF_BBOXES = bboxes
        SHELF_IMAGE_PATH = img_path
    except Exception:
        # If anything goes wrong, we just don't restore
        SHELF_IMAGE = None
        SHELF_IMAGE_ID = None
        SHELF_BBOXES = []
        SHELF_IMAGE_PATH = None
