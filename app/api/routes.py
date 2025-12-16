from __future__ import annotations

from typing import List, Dict, Optional, Any
from collections import defaultdict
import io
import uuid
import zipfile
import base64
from datetime import datetime
import tempfile
from pathlib import Path

from pymongo.errors import OperationFailure
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from PIL import Image, ImageOps
import cv2
import numpy as np
import pymongo

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from ultralytics import YOLO

from app.core.config import settings
from app.services.vectorizer import get_vectorizer
from app.services.datastore import get_datastore

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


# ===========================
# BBox normalization / schema helpers (CRITICAL)
# ===========================

def get_bbox_from_record(d: Dict[str, Any]) -> Optional[List[int]]:
    """
    Extract bbox from multiple possible schemas.
    Priority: d["bbox"] -> d["metadata"]["bbox"] -> d["bbox_list"].
    """
    if not isinstance(d, dict):
        return None
    bbox = d.get("bbox")
    if bbox is None:
        bbox = (d.get("metadata") or {}).get("bbox")
    if bbox is None:
        bbox = d.get("bbox_list")
    return bbox


def normalize_bbox_no_clamp(bbox: List[Any]) -> Optional[List[int]]:
    """
    Normalize bbox without image-size clamping.
    - Converts to ints (rounded).
    - Ensures x1 < x2 and y1 < y2 (swap if needed).
    - Rejects degenerate boxes.
    Convention used by this file: [x1, y1, x2, y2) (x2,y2 exclusive).
    """
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(c))) for c in bbox]
    except (ValueError, TypeError):
        return None

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def validate_and_clamp_bbox(bbox: List[Any], img_width: int, img_height: int) -> Optional[List[int]]:
    """
    Validate and clamp bbox to image bounds.
    Convention: [x1, y1, x2, y2) exclusive.
    - x1 in [0, W-1], y1 in [0, H-1]
    - x2 in [x1+1, W], y2 in [y1+1, H]
    """
    norm = normalize_bbox_no_clamp(bbox)
    if norm is None:
        return None

    x1, y1, x2, y2 = norm

    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(x1 + 1, min(x2, img_width))
    y2 = max(y1 + 1, min(y2, img_height))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    IoU for bboxes in [x1,y1,x2,y2) exclusive convention.
    """
    b1 = normalize_bbox_no_clamp(bbox1)
    b2 = normalize_bbox_no_clamp(bbox2)
    if b1 is None or b2 is None:
        return 0.0

    x1_1, y1_1, x2_1, y2_1 = b1
    x1_2, y1_2, x2_2, y2_2 = b2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    iw = max(0, x2_i - x1_i)
    ih = max(0, y2_i - y1_i)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
    area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0

    return inter / union


# ===========================
# Models
# ===========================

class IndexShelfResponse(BaseModel):
    image_id: str
    num_detections: int
    num_indexed: int


class MatchResult(BaseModel):
    shelf_id: str
    bbox: List[int]  # internal convention: [x1,y1,x2,y2) exclusive
    score: float


class SearchResponse(BaseModel):
    matches: List[MatchResult]


class SingleQueryImageOut(BaseModel):
    id: str
    modelName: str
    skuDescription: Optional[str] = None
    skuId: Optional[str] = None
    tenantId: Optional[str] = None
    clientId: Optional[str] = None
    categoryId: Optional[str] = None
    brandId: Optional[str] = None
    image_base64: str


# ===========================
# Helper function for parallel processing
# ===========================

def extract_embedding_only(
    crop: Image.Image,
    bbox_list: List[int],
    label: str,
    conf: float,
    vectorizer,
) -> Optional[Dict[str, Any]]:
    """
    Extract embedding from a single crop (no database save).
    Ensures crop is valid-sized RGB to prevent processor/embedding failures.
    """
    try:
        # Hard guard against tiny/degenerate crops (prevents 1px height crops -> processor errors)
        w, h = crop.size
        if w < 5 or h < 5:
            return None

        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        vector = vectorizer.get_image_embedding(crop)
        crop_id = str(uuid.uuid4())

        # Canonical schema: bbox at top-level.
        # Keep metadata.bbox and bbox_list for compatibility.
        return {
            "crop_id": crop_id,
            "vector": vector,
            "bbox": bbox_list,
            "metadata": {
                "bbox": bbox_list,
                "label": label,
                "confidence": conf,
            },
            "bbox_list": bbox_list,
        }
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None


# ===========================
# Mongo: single query images
# ===========================

def get_single_query_images_collection():
    if not settings.MONGO_URI:
        raise HTTPException(status_code=500, detail="MongoDB URI is not configured")

    try:
        mongo_client = pymongo.MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=2000,
        )
        mongo_client.server_info()

        if settings.MONGO_DB_NAME:
            mongo_db = mongo_client[settings.MONGO_DB_NAME]
        else:
            try:
                mongo_db = mongo_client.get_default_database()
            except Exception:
                mongo_db = mongo_client["cstore-ai"]

        if settings.MONGO_SINGLE_QUERY_COLLECTION_NAME_TEST:
            coll_name = settings.MONGO_SINGLE_QUERY_COLLECTION_NAME_TEST
            print(f"Using TEST single query images collection: {coll_name}")
        else:
            coll_name = settings.MONGO_SINGLE_QUERY_COLLECTION_NAME
            print(f"Using PRODUCTION single query images collection: {coll_name}")

        return mongo_db[coll_name]
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")


@lru_cache(maxsize=1)
def get_yolo_v11_model() -> YOLO:
    model_path = "models/yolo/yolo_v11_best.pt"
    return YOLO(model_path)


# ===========================
# Index shelf with YOLO
# ===========================

@router.post("/index-shelf-yolo", response_model=IndexShelfResponse)
async def index_shelf_yolo(
    file: UploadFile = File(...),
    box_thresh: float = 0.2,
):
    if not (0.0 <= box_thresh <= 1.0):
        raise HTTPException(status_code=400, detail="box_thresh must be between 0.0 and 1.0")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # --- Load image once with PIL, normalize EXIF orientation, then save that normalized pixel data to disk for YOLO ---
    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    original_width, original_height = pil_img.size
    print(f"Normalized image size (W×H): {original_width}x{original_height}")

    try:
        suffix = Path(file.filename).suffix or ".jpg"
    except Exception:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    # Save normalized image to tmp path so YOLO reads the same orientation/pixels
    try:
        pil_img.save(tmp_path)
    except Exception:
        pil_img.save(tmp_path, format="PNG")

    model = get_yolo_v11_model()

    results = model.predict(
        source=tmp_path,
        imgsz=640,
        conf=box_thresh,
        save=False,
        show_labels=False,
        show_conf=False,
        verbose=False,
    )

    if not results:
        raise HTTPException(status_code=500, detail="YOLO v11 returned no results")

    result = results[0]
    boxes = result.boxes

    print("YOLO result:", result)
    print("Number of boxes:", 0 if boxes is None else len(boxes))

    # Ultralytics orig_shape is (H, W)
    try:
        yolo_h, yolo_w = result.orig_shape
        print(f"YOLO orig_shape (H×W): {yolo_h}x{yolo_w}")
        if (yolo_w, yolo_h) != (original_width, original_height):
            print(
                "WARNING: YOLO orig_shape does not match PIL normalized size. "
                "This typically indicates an orientation/pipeline mismatch."
            )
    except Exception:
        pass

    image_id = str(uuid.uuid4())

    # If no detections, still save shelf image
    if boxes is None or len(boxes) == 0:
        shelves_dir = settings.SHELF_DIR
        shelves_dir.mkdir(parents=True, exist_ok=True)
        out_path = shelves_dir / f"{image_id}.png"
        pil_img.save(out_path, format="PNG", optimize=False)
        return IndexShelfResponse(image_id=image_id, num_detections=0, num_indexed=0)

    vectorizer = get_vectorizer()
    datastore = get_datastore()
    names = result.names  # {class_id: class_name}

    crop_tasks = []
    for i in range(len(boxes)):
        box = boxes[i]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = names.get(class_id, str(class_id))

        # Round (not truncate) to reduce systematic bias
        bbox_list = [
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
        ]

        # Clamp to bounds using exclusive semantics
        vb = validate_and_clamp_bbox(bbox_list, original_width, original_height)
        if vb is None:
            continue
        bbox_list = vb

        crop = pil_img.crop((bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3]))
        crop_tasks.append((crop, bbox_list, label, conf))

    objects_to_save: List[Dict[str, Any]] = []
    max_workers = min(len(crop_tasks), 8) if crop_tasks else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_embedding_only, crop, bbox_list, label, conf, vectorizer)
            for crop, bbox_list, label, conf in crop_tasks
        ]
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                objects_to_save.append(r)

    if objects_to_save:
        datastore.batch_save_objects(image_id, objects_to_save)

    num_indexed = len(objects_to_save)

    shelves_dir = settings.SHELF_DIR
    shelves_dir.mkdir(parents=True, exist_ok=True)
    out_path = shelves_dir / f"{image_id}.png"
    pil_img.save(out_path, format="PNG", optimize=False)

    # Verify saved image size
    try:
        saved_image = Image.open(out_path)
        saved_width, saved_height = saved_image.size
        if (saved_width, saved_height) != (original_width, original_height):
            print(
                f"WARNING: Saved image size mismatch! "
                f"Original: {original_width}x{original_height}, Saved: {saved_width}x{saved_height}"
            )
        else:
            print(f"Image saved successfully at original size: {saved_width}x{saved_height}")
    except Exception:
        pass

    return IndexShelfResponse(
        image_id=image_id,
        num_detections=len(boxes),
        num_indexed=num_indexed,
    )


# ===========================
# Upload / list query images
# ===========================

@router.post("/ModelTraining")
async def upload_single_query_image(
    modelName: str = Form(...),
    skuDescription: str = Form(...),
    skuId: str = Form(...),
    tenantId: str = Form(...),
    clientId: str | None = Form(None),
    categoryId: str | None = Form(None),
    brandId: str | None = Form(None),
    image: UploadFile = File(...),
):
    if not settings.MONGO_URI:
        raise HTTPException(status_code=500, detail="MongoDB URI is not configured")

    collection = get_single_query_images_collection()

    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are allowed")

    MAX_SIZE = 5 * 1024 * 1024
    file_bytes = await image.read()
    if len(file_bytes) > MAX_SIZE:
        raise HTTPException(status_code=413, detail="Image file too large")

    image_b64 = base64.b64encode(file_bytes).decode("utf-8")

    doc_id = str(uuid.uuid4())
    document = {
        "_id": doc_id,
        "modelName": modelName,
        "skuDescription": skuDescription,
        "skuId": skuId,
        "tenantId": tenantId,
        "clientId": clientId,
        "categoryId": categoryId,
        "brandId": brandId,
        "filename": image.filename,
        "content_type": image.content_type,
        "image_base64": image_b64,
        "created_at": datetime.utcnow(),
    }

    try:
        result = collection.insert_one(document)
    except OperationFailure as e:
        errmsg = getattr(e, "details", {}).get("errmsg", str(e))
        raise HTTPException(status_code=500, detail=f"MongoDB write error: {errmsg}")

    return {"id": str(result.inserted_id), "status": "stored"}


@router.get("/AvailableModel", response_model=List[SingleQueryImageOut])
async def list_query_images(
    modelName: str,
    skuId: str | None = None,
    tenantId: str | None = None,
    skuDescription: str | None = None,
    clientId: str | None = None,
    categoryId: str | None = None,
    brandId: str | None = None,
):
    collection = get_single_query_images_collection()

    query: Dict[str, str] = {"modelName": modelName}
    if skuId:
        query["skuId"] = skuId
    if tenantId:
        query["tenantId"] = tenantId
    if skuDescription:
        query["skuDescription"] = skuDescription
    if clientId:
        query["clientId"] = clientId
    if categoryId:
        query["categoryId"] = categoryId
    if brandId:
        query["brandId"] = brandId

    try:
        docs = list(collection.find(query))
    except OperationFailure as e:
        errmsg = getattr(e, "details", {}).get("errmsg", str(e))
        raise HTTPException(status_code=500, detail=f"MongoDB read error: {errmsg}")

    results: List[SingleQueryImageOut] = []
    for doc in docs:
        results.append(
            SingleQueryImageOut(
                id=str(doc.get("_id")),
                modelName=doc.get("modelName", ""),
                skuDescription=doc.get("skuDescription"),
                skuId=doc.get("skuId"),
                tenantId=doc.get("tenantId"),
                clientId=doc.get("clientId"),
                categoryId=doc.get("categoryId"),
                brandId=doc.get("brandId"),
                image_base64=doc.get("image_base64", ""),
            )
        )
    return results


# ===========================
# NMS
# ===========================

def apply_nms(matches: List[MatchResult], iou_threshold: float = 0.5) -> List[MatchResult]:
    """
    Remove overlapping boxes per shelf; keep the best (lowest) score.
    Uses normalized bbox ordering for stability.
    """
    if not matches:
        return []

    valid: List[MatchResult] = []
    for m in matches:
        nb = normalize_bbox_no_clamp(m.bbox)
        if nb is None:
            continue
        valid.append(MatchResult(shelf_id=m.shelf_id, bbox=nb, score=m.score))

    if not valid:
        return []

    valid.sort(key=lambda m: m.score)
    kept: List[MatchResult] = []

    while valid:
        best = valid.pop(0)
        kept.append(best)

        remaining: List[MatchResult] = []
        for m in valid:
            if calculate_iou(best.bbox, m.bbox) < iou_threshold:
                remaining.append(m)
        valid = remaining

    return kept

def crop_to_content(img: Image.Image) -> Image.Image:
    """
    Crops away transparent or near-white background (common in packshots)
    to reduce domain gap between packshot queries and shelf crops.
    """
    img = ImageOps.exif_transpose(img)

    # If alpha exists, crop by alpha mask
    if img.mode in ("RGBA", "LA"):
        alpha = np.array(img.split()[-1])
        ys, xs = np.where(alpha > 0)
        if len(xs) > 0:
            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1
            return img.crop((x1, y1, x2, y2)).convert("RGB")

    # Fallback: crop anything not near-white
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    mask = np.any(arr < 245, axis=2)  # pixels not almost-white
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return rgb

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return rgb.crop((x1, y1, x2, y2))


def label_crop(img: Image.Image) -> Image.Image:
    """
    Heuristic label-focused crop (often most discriminative for packaged goods).
    """
    w, h = img.size
    return img.crop((int(w * 0.15), int(h * 0.35), int(w * 0.85), int(h * 0.90)))


def make_query_views(img: Image.Image) -> List[Image.Image]:
    """
    Multi-view queries to improve recall:
    - content crop
    - label crop
    - rotations (for camera/shelf rotations)
    """
    base = crop_to_content(img).convert("RGB")
    views = [base, label_crop(base)]
    for angle in (90, 180, 270):
        r = base.rotate(angle, expand=True)
        views.append(r)
        views.append(label_crop(r))
    return views


def _orb_good_match_count(query_rgb: np.ndarray, cand_rgb: np.ndarray) -> int:
    """
    ORB verifier: returns count of "good" matches (higher is better).
    Used to rerank top candidates after vector retrieval.
    """
    # Guard against tiny crops
    if query_rgb.shape[0] < 20 or query_rgb.shape[1] < 20:
        return 0
    if cand_rgb.shape[0] < 20 or cand_rgb.shape[1] < 20:
        return 0

    qg = cv2.cvtColor(query_rgb, cv2.COLOR_RGB2GRAY)
    cg = cv2.cvtColor(cand_rgb, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=800)
    kp1, des1 = orb.detectAndCompute(qg, None)
    kp2, des2 = orb.detectAndCompute(cg, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0

    # Lower distance = better. Threshold is heuristic; tune if needed.
    good = [m for m in matches if m.distance < 60]
    return len(good)

# ===========================
# Search & annotate
# ===========================

@router.get("/search-visual-by-available")
async def search_visual_by_available(
    modelName: str,
    skuId: str | None = None,
    tenantId: str | None = None,
    skuDescription: str | None = None,
    clientId: str | None = None,
    categoryId: str | None = None,
    brandId: str | None = None,
    as_zip: bool = False,
    max_results_per_shelf: int = 50,
    nms_iou_threshold: float = 0.5,
    orb_rerank: bool = True,
    orb_max_per_shelf: int = 25,   # only rerank top N per shelf for speed
):
    collection = get_single_query_images_collection()

    query: Dict[str, str] = {"modelName": modelName}
    if skuId:
        query["skuId"] = skuId
    if tenantId:
        query["tenantId"] = tenantId
    if skuDescription:
        query["skuDescription"] = skuDescription
    if clientId:
        query["clientId"] = clientId
    if categoryId:
        query["categoryId"] = categoryId
    if brandId:
        query["brandId"] = brandId

    docs = list(collection.find(query))
    if not docs:
        raise HTTPException(status_code=404, detail="No matching AvailableModel entries found")

    vectorizer = get_vectorizer()
    datastore = get_datastore()

    # Query label references for ORB reranking (from all query docs)
    query_label_refs_rgb: List[np.ndarray] = []

    # Deduplicate by crop_id and keep best (lowest) distance across all query views/docs
    best_by_crop: Dict[str, MatchResult] = {}

    # With multi-view querying, cap results per view to control total work
    PER_VIEW_LIMIT = 120

    for doc in docs:
        image_b64 = doc.get("image_base64")
        if not image_b64:
            continue

        try:
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes))
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as e:
            print(f"Skipping broken query image: {e}")
            continue

        # Multi-view query images
        views = make_query_views(img)

        # ORB reference: label crop of content-cropped query
        try:
            ref = label_crop(crop_to_content(img)).convert("RGB")
            query_label_refs_rgb.append(np.array(ref))
        except Exception:
            pass

        for view in views:
            try:
                query_vector = vectorizer.get_image_embedding(view)
            except Exception as e:
                print(f"Skipping view embedding failure: {e}")
                continue

            results = datastore.query_similar(query_vector, n_results=PER_VIEW_LIMIT)

            for res in results:
                data = res.get("data") or {}

                crop_id = data.get("_id")
                if not crop_id:
                    continue

                parent_id = data.get("parent_image_id")
                if parent_id is None:
                    continue

                score = res.get("score")
                if score is None:
                    continue
                score = float(score)  # L2 distance (lower is better)

                bbox_raw = get_bbox_from_record(data)
                bbox_norm = normalize_bbox_no_clamp(bbox_raw)
                if bbox_norm is None:
                    continue

                existing = best_by_crop.get(crop_id)
                if existing is None or score < existing.score:
                    best_by_crop[crop_id] = MatchResult(
                        shelf_id=str(parent_id),
                        bbox=bbox_norm,
                        score=score,
                    )

    if not best_by_crop:
        raise HTTPException(status_code=404, detail="No matches found")

    all_matches = list(best_by_crop.values())

    # Group by shelf and apply NMS
    shelf_to_matches: Dict[str, List[MatchResult]] = defaultdict(list)
    for m in all_matches:
        shelf_to_matches[m.shelf_id].append(m)

    filtered_shelf_matches: Dict[str, List[MatchResult]] = {}
    for shelf_id, shelf_matches in shelf_to_matches.items():
        nms_matches = apply_nms(shelf_matches, iou_threshold=nms_iou_threshold)
        nms_matches.sort(key=lambda m: m.score)  # lower distance first
        filtered_shelf_matches[shelf_id] = nms_matches[:max_results_per_shelf]

    if not filtered_shelf_matches:
        raise HTTPException(status_code=404, detail="No matches remaining after filtering")

    # Fetch original bboxes per shelf (for drawing non-matches in red)
    all_shelf_ids = set(filtered_shelf_matches.keys())
    original_bboxes_by_shelf: Dict[str, List[Dict[str, Any]]] = {}

    for shelf_id in all_shelf_ids:
        original_boxes = []
        if getattr(datastore, "use_mongo", False) and getattr(datastore, "mongo_coll", None) is not None:
            docs2 = datastore.mongo_coll.find({"parent_image_id": shelf_id})
            for d in docs2:
                bbox = get_bbox_from_record(d)
                if bbox and len(bbox) == 4:
                    original_boxes.append({"bbox": bbox})
        else:
            for item in getattr(datastore, "local_meta", []):
                if item.get("parent_image_id") == shelf_id:
                    bbox = get_bbox_from_record(item)
                    if bbox and len(bbox) == 4:
                        original_boxes.append({"bbox": bbox})
        original_bboxes_by_shelf[shelf_id] = original_boxes

    # Load shelf images and draw bboxes
    shelves_dir = settings.SHELF_DIR
    annotated_images: Dict[str, np.ndarray] = {}

    for shelf_id, shelf_matches in filtered_shelf_matches.items():
        shelf_path = shelves_dir / f"{shelf_id}.png"
        if not shelf_path.exists():
            continue

        shelf_img_bgr = cv2.imread(str(shelf_path))
        if shelf_img_bgr is None:
            continue

        img_height, img_width = shelf_img_bgr.shape[:2]
        print(f"Loaded shelf image {shelf_id}: {img_width}x{img_height}")

        # OPTIONAL: ORB reranking on top candidates (label verification)
        if orb_rerank and query_label_refs_rgb:
            limit = min(len(shelf_matches), max(1, int(orb_max_per_shelf)))
            candidates = shelf_matches[:limit]
            rest = shelf_matches[limit:]

            shelf_rgb = cv2.cvtColor(shelf_img_bgr, cv2.COLOR_BGR2RGB)
            reranked: List[tuple[int, float, MatchResult]] = []

            for m in candidates:
                vb = validate_and_clamp_bbox(m.bbox, img_width, img_height)
                if vb is None:
                    continue
                x1, y1, x2, y2 = vb
                cand_rgb = shelf_rgb[y1:y2, x1:x2]

                best_orb = 0
                for qref in query_label_refs_rgb:
                    best_orb = max(best_orb, _orb_good_match_count(qref, cand_rgb))

                reranked.append((best_orb, m.score, m))

            if reranked:
                reranked.sort(key=lambda t: (-t[0], t[1]))  # ORB desc, distance asc
                shelf_matches = [t[2] for t in reranked] + rest
                filtered_shelf_matches[shelf_id] = shelf_matches

        # Build matching set AFTER clamp for this shelf (prevents false mismatches)
        matching_set = set()
        for m in shelf_matches:
            vb = validate_and_clamp_bbox(m.bbox, img_width, img_height)
            if vb is not None:
                matching_set.add(tuple(vb))

        # Draw original (non-matching) boxes in RED
        for orig in original_bboxes_by_shelf.get(shelf_id, []):
            raw_bbox = orig.get("bbox")
            vb = validate_and_clamp_bbox(raw_bbox, img_width, img_height)
            if vb is None:
                continue
            if tuple(vb) in matching_set:
                continue

            x1, y1, x2, y2 = vb
            cv2.rectangle(shelf_img_bgr, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 2)

        # Draw matching boxes in GREEN + distance label
        for m in shelf_matches:
            vb = validate_and_clamp_bbox(m.bbox, img_width, img_height)
            if vb is None:
                continue

            x1, y1, x2, y2 = vb
            cv2.rectangle(shelf_img_bgr, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 3)

            label_text = f"{m.score:.3f}"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            y_top = max(y1 - (th + baseline + 6), 0)
            cv2.rectangle(
                shelf_img_bgr,
                (x1, y_top),
                (x1 + tw + 6, y_top + th + baseline + 6),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                shelf_img_bgr,
                label_text,
                (x1 + 3, y_top + th + baseline + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        annotated_images[shelf_id] = shelf_img_bgr

    if not annotated_images:
        raise HTTPException(status_code=404, detail="No shelf images found on disk")

    # Mode A: ZIP
    if as_zip:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for sid, img in annotated_images.items():
                ok, buf = cv2.imencode(".png", img)
                if ok:
                    zipf.writestr(f"{sid}.png", buf.tobytes())
        zip_buffer.seek(0)
        headers = {"Content-Disposition": 'attachment; filename="search_results_by_available.zip"'}
        return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)

    # Mode B: one combined PNG stacked vertically
    imgs = list(annotated_images.values())
    widths = [img.shape[1] for img in imgs]
    heights = [img.shape[0] for img in imgs]
    max_width = max(widths) if widths else 0
    total_height = sum(heights) + 10 * (len(imgs) - 1) if heights else 0

    if max_width <= 0 or total_height <= 0:
        raise HTTPException(status_code=500, detail="Invalid image dimensions")

    combined = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    y_offset = 0
    for img in imgs:
        h, w, _ = img.shape
        combined[y_offset:y_offset + h, 0:w] = img
        y_offset += h + 10

    ok, buf = cv2.imencode(".png", combined)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode combined image")

    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


