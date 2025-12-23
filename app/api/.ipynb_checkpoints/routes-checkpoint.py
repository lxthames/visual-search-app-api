from __future__ import annotations

from typing import List, Dict, Optional, Any
from collections import defaultdict
import io
from io import BytesIO
import uuid
import base64
from datetime import datetime
import tempfile
from pathlib import Path
import asyncio
from functools import partial

from pymongo.errors import OperationFailure
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import kornia as K
from kornia.feature import LoFTR
from PIL import Image, ImageOps
import cv2
import torch
import numpy as np
import pymongo
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from ultralytics import YOLO

# Import your internal modules
from app.services.detection import run_detection
from app.core.config import settings
from app.services.vectorizer import get_vectorizer
from app.services.datastore import get_datastore

router = APIRouter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@router.get("/health")
async def health_check():
    return {"status": "ok"}

# ===========================
# Shape Classifier
# ===========================
# ===========================
# Shape Classifier (Updated: Force Prediction)
# ===========================
# ===========================
# Shape Classifier (Force Prediction)
# ===========================
class ShapeClassifier:
    def __init__(self, device_str="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device_str
        print(f"Loading Shape Classifier on {self.device}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.labels = ["a photo of a bottle", "a photo of a can"]
        self.clean_labels = ["bottle", "can"]
        
    def predict(self, image_rgb: np.ndarray) -> str:
        if image_rgb.size == 0: return "unknown"
        try:
            inputs = self.processor(
                text=self.labels, 
                images=image_rgb, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                
            argmax_idx = probs.argmax().item()
            # No threshold -> Force best guess
            return self.clean_labels[argmax_idx]
        except Exception as e:
            print(f"Shape classifier error: {e}")
            return "unknown"

shape_classifier = ShapeClassifier()
# ===========================
# BBox normalization / schema helpers
# ===========================

def get_bbox_from_record(d: Dict[str, Any]) -> Optional[List[int]]:
    if not isinstance(d, dict):
        return None
    bbox = d.get("bbox")
    if bbox is None:
        bbox = (d.get("metadata") or {}).get("bbox")
    if bbox is None:
        bbox = d.get("bbox_list")
    return bbox

def normalize_bbox_no_clamp(bbox: List[Any]) -> Optional[List[int]]:
    if not bbox or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(c))) for c in bbox]
    except (ValueError, TypeError):
        return None

    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1: return None

    return [x1, y1, x2, y2]

def validate_and_clamp_bbox(bbox: List[Any], img_width: int, img_height: int) -> Optional[List[int]]:
    norm = normalize_bbox_no_clamp(bbox)
    if norm is None: return None
    x1, y1, x2, y2 = norm
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(x1 + 1, min(x2, img_width))
    y2 = max(y1 + 1, min(y2, img_height))
    if x2 <= x1 or y2 <= y1: return None
    return [x1, y1, x2, y2]

def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    b1 = normalize_bbox_no_clamp(bbox1)
    b2 = normalize_bbox_no_clamp(bbox2)
    if b1 is None or b2 is None: return 0.0

    x1_1, y1_1, x2_1, y2_1 = b1
    x1_2, y1_2, x2_2, y2_2 = b2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    iw = max(0, x2_i - x1_i)
    ih = max(0, y2_i - y1_i)
    inter = iw * ih
    if inter <= 0: return 0.0

    area1 = max(0, x2_1 - x1_1) * max(0, y2_1 - y1_1)
    area2 = max(0, x2_2 - x1_2) * max(0, y2_2 - y1_2)
    union = area1 + area2 - inter
    if union <= 0: return 0.0

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
    bbox: List[int]
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
    shape_label: Optional[str] = None 

# ===========================
# Helper function for parallel processing
# ===========================

def extract_embedding_only(
    crop: Image.Image,
    bbox_list: List[int],
    label: str,
    conf: float,
    shape_label: str, 
    vectorizer,
) -> Optional[Dict[str, Any]]:
    """
    Extract embedding from a single crop.
    """
    try:
        w, h = crop.size
        if w < 5 or h < 5: return None

        if crop.mode != "RGB":
            crop = crop.convert("RGB")

        vector = vectorizer.get_image_embedding(crop)
        crop_id = str(uuid.uuid4())

        return {
            "crop_id": crop_id,
            "vector": vector,
            "bbox": bbox_list,
            # ==========================================
            # ✅ FIX: This line saves the label where Search can see it
            "shape_label": shape_label, 
            # ==========================================
            "metadata": {
                "bbox": bbox_list,
                "label": label,
                "confidence": conf,
                "shape_label": shape_label, 
            },
            "bbox_list": bbox_list,
        }
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

# ===========================
# Mongo & YOLO
# ===========================

def get_single_query_images_collection():
    if not settings.MONGO_URI:
        raise HTTPException(status_code=500, detail="MongoDB URI is not configured")
    try:
        mongo_client = pymongo.MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=2000)
        if settings.MONGO_DB_NAME:
            mongo_db = mongo_client[settings.MONGO_DB_NAME]
        else:
            try:
                mongo_db = mongo_client.get_default_database()
            except Exception:
                mongo_db = mongo_client["cstore-ai"]
        return mongo_db[settings.MONGO_SINGLE_QUERY_COLLECTION_NAME]
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")

@lru_cache(maxsize=1)
def get_yolo_v11_model() -> YOLO:
    # Ensure this path matches your project structure
    model_path = "models/yolo/yolo_v11_best.pt"
    return YOLO(model_path)

# ===========================
# Index shelf with YOLO (UPDATED)
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

    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    original_width, original_height = pil_img.size
    
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    pil_img.save(tmp_path)

    model = get_yolo_v11_model()
    results = model.predict(source=tmp_path, imgsz=640, conf=box_thresh, save=False, max_det=650)
    
    if not results:
        raise HTTPException(status_code=500, detail="YOLO v11 returned no results")

    result = results[0]
    boxes = result.boxes
    image_id = str(uuid.uuid4())

    if boxes is None or len(boxes) == 0:
        shelves_dir = settings.SHELF_DIR
        shelves_dir.mkdir(parents=True, exist_ok=True)
        out_path = shelves_dir / f"{image_id}.png"
        pil_img.save(out_path, format="PNG")
        return IndexShelfResponse(image_id=image_id, num_detections=0, num_indexed=0)

    vectorizer = get_vectorizer()
    datastore = get_datastore()
    names = result.names

    crop_tasks = []
    
    print(f"\n--- PROCESSING SHELF {image_id} ---")

    for i in range(len(boxes)):
        box = boxes[i]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = names.get(class_id, str(class_id))

        bbox_list = [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
        vb = validate_and_clamp_bbox(bbox_list, original_width, original_height)
        if vb is None: continue
        bbox_list = vb

        crop = pil_img.crop((bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3]))
        
        # --- PREDICT SHAPE ---
        crop_np = np.array(crop)
        predicted_shape = shape_classifier.predict(crop_np)
        # ---------------------

        # --- DEBUG PRINT ---
        print(f"DEBUG: Item {i} ({label}) | Conf: {conf:.2f} | Predicted Shape: {predicted_shape}")

        crop_tasks.append((crop, bbox_list, label, conf, predicted_shape))

    objects_to_save: List[Dict[str, Any]] = []
    max_workers = min(len(crop_tasks), 8) if crop_tasks else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract_embedding_only, crop, bbox_list, label, conf, shape_label, vectorizer)
            for crop, bbox_list, label, conf, shape_label in crop_tasks
        ]
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                objects_to_save.append(r)

    if objects_to_save:
        datastore.batch_save_objects(image_id, objects_to_save)

    shelves_dir = settings.SHELF_DIR
    shelves_dir.mkdir(parents=True, exist_ok=True)
    out_path = shelves_dir / f"{image_id}.png"
    pil_img.save(out_path, format="PNG")

    return IndexShelfResponse(
        image_id=image_id,
        num_detections=len(boxes),
        num_indexed=len(objects_to_save),
    )
# ===========================
# Upload Single Query (UPDATED)
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
    file_bytes = await image.read()
    
    # 1. Load the FULL Image
    img = Image.open(BytesIO(file_bytes)).convert("RGB")

    # =========================================================
    # FIX: Classify Shape using the FULL IMAGE
    # The full image has the "neck" of the bottle visible.
    # =========================================================
    full_img_np = np.array(img)
    predicted_shape = shape_classifier.predict(full_img_np)
    # =========================================================

    # 2. Run detection to find the logo
    label = "a logo."
    detections = run_detection(img, prompt=label, box_thresh=0.1)
    if len(detections) == 0:
        raise HTTPException(status_code=404, detail="No logo detected")

    idx = detections.confidence.argmax()
    box = detections.xyxy[idx]
    confidence = float(detections.confidence[idx])
    detected_label = detections.data["class_name"][idx]

    # 3. NOW Crop the image (for vector storage/matching)
    x1, y1, x2, y2 = [int(v) for v in box]
    cropped_img = img.crop((x1, y1, x2, y2))
    
    buffer = BytesIO()
    cropped_img.save(buffer, format="PNG")
    buffer.seek(0)
    cropped_image_b64 = base64.b64encode(buffer.read()).decode("utf-8")

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
        "image_base64": cropped_image_b64, 
        "detected_label": detected_label,
        "confidence": confidence,
        "shape_label": predicted_shape, # Saved correctly using the full image
        "created_at": datetime.utcnow(),
    }

    try:
        result = collection.insert_one(document)
    except OperationFailure as e:
        raise HTTPException(status_code=500, detail=f"MongoDB write error: {e}")

    return {
        "id": str(result.inserted_id),
        "status": "stored",
        "detected_label": detected_label,
        "confidence": confidence,
        "shape_label": predicted_shape, 
    }

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
    if skuId: query["skuId"] = skuId
    if tenantId: query["tenantId"] = tenantId
    if skuDescription: query["skuDescription"] = skuDescription
    if clientId: query["clientId"] = clientId
    if categoryId: query["categoryId"] = categoryId
    if brandId: query["brandId"] = brandId

    docs = list(collection.find(query))
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
                shape_label=doc.get("shape_label", None), # <--- RETURN SHAPE
            )
        )
    return results

# ===========================
# LoFTR & Visual Search
# ===========================

matcher = LoFTR(pretrained="indoor").to(device)

def get_feature_match_score(query_img_rgb: np.ndarray, candidate_img_rgb: np.ndarray, debug_id: str = "") -> float:
    """
    BALANCED LoFTR Scorer.
    """
    h_q, w_q = query_img_rgb.shape[:2]
    h_c, w_c = candidate_img_rgb.shape[:2]
    
    if h_q < 20 or w_q < 20 or h_c < 20 or w_c < 20:
        return 0.0

    t_query = K.image_to_tensor(query_img_rgb, False).float() / 255.0
    t_cand = K.image_to_tensor(candidate_img_rgb, False).float() / 255.0
    
    t_query = K.color.rgb_to_grayscale(t_query).to(device)
    t_cand = K.color.rgb_to_grayscale(t_cand).to(device)

    with torch.inference_mode():
        input_dict = {"image0": t_query, "image1": t_cand}
        results = matcher(input_dict)
    
    confidences = results['confidence']
    CONFIDENCE_THRESHOLD = 0.50 
    strong_matches = confidences > CONFIDENCE_THRESHOLD
    count = strong_matches.sum().item()
    
    MIN_MATCH_COUNT = 7
    normalized_score = min(count / 100.0, 1.0)
    
    if count < MIN_MATCH_COUNT:
        return 0.0
        
    return normalized_score

def apply_nms(matches: List[MatchResult], iou_threshold: float = 0.5) -> List[MatchResult]:
    if not matches: return []
    valid = []
    for m in matches:
        nb = normalize_bbox_no_clamp(m.bbox)
        if nb: valid.append(MatchResult(shelf_id=m.shelf_id, bbox=nb, score=m.score))
    if not valid: return []
    
    valid.sort(key=lambda m: m.score)
    kept = []
    while valid:
        best = valid.pop(0)
        kept.append(best)
        remaining = []
        for m in valid:
            if calculate_iou(best.bbox, m.bbox) < iou_threshold:
                remaining.append(m)
        valid = remaining
    return kept

def crop_to_content(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA"):
        alpha = np.array(img.split()[-1])
        ys, xs = np.where(alpha > 0)
        if len(xs) > 0:
            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1
            return img.crop((x1, y1, x2, y2)).convert("RGB")
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    mask = np.any(arr < 245, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0: return rgb
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return rgb.crop((x1, y1, x2, y2))

def label_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((int(w * 0.15), int(h * 0.35), int(w * 0.85), int(h * 0.90)))

def make_query_views(img: Image.Image) -> List[Image.Image]:
    base = crop_to_content(img).convert("RGB")
    views = [base, label_crop(base)]
    for angle in (90, 180, 270):
        r = base.rotate(angle, expand=True)
        views.append(r)
        views.append(label_crop(r))
    return views

# ===========================
# NON-BLOCKING VERIFICATION WRAPPER
# ===========================

def run_verification_sync(
    candidates: List[MatchResult],
    shelf_img_bgr: np.ndarray,
    query_refs: List[np.ndarray],
    verification_threshold: float,
    strict_mode: bool
) -> List[MatchResult]:
    """
    Synchronous function to be run in a separate thread.
    """
    img_height, img_width = shelf_img_bgr.shape[:2]
    shelf_rgb = cv2.cvtColor(shelf_img_bgr, cv2.COLOR_BGR2RGB)
    
    verified_list = []
    
    for i, m in enumerate(candidates):
        vb = validate_and_clamp_bbox(m.bbox, img_width, img_height)
        if vb is None: continue
        x1, y1, x2, y2 = vb
        cand_rgb = shelf_rgb[y1:y2, x1:x2]

        best_feature_score = 0.0
        for qref in query_refs:
            try:
                # Note: get_feature_match_score calls GPU. 
                # If running in thread, ensure CUDA context is handled or use CPU.
                score = get_feature_match_score(qref, cand_rgb, debug_id=f"c{i}")
                best_feature_score = max(best_feature_score, score)
            except Exception: pass
        
        if best_feature_score >= verification_threshold:
            verified_m = MatchResult(
                shelf_id=m.shelf_id,
                bbox=m.bbox,
                score=best_feature_score
            )
            verified_list.append((best_feature_score, m.score, verified_m))
        else:
            if not strict_mode:
                verified_list.append((0.0, m.score, m))
    
    verified_list.sort(key=lambda t: (-t[0], t[1]))
    return [t[2] for t in verified_list]

# ===========================
# Main Endpoint (Updated Limit: 100)
# ===========================

# @router.get("/search-visual-by-available")
# async def search_visual_by_available(
#     modelName: str,
#     skuId: str | None = None,
#     tenantId: str | None = None,
#     skuDescription: str | None = None,
#     clientId: str | None = None,
#     categoryId: str | None = None,
#     brandId: str | None = None,
#     as_zip: bool = False,
#     # UPDATE 1: Allow 100 items per shelf to pass the NMS/Vector stage
#     max_results_per_shelf: int = 100,
#     nms_iou_threshold: float = 0.5,
#     verify_matches: bool = Query(True, description="Enable LoFTR geometric verification"),
#     # UPDATE 2: Verify up to 100 items (The user's request)
#     max_verification_candidates: int = Query(100, description="Items per shelf to verify"),
#     verification_threshold: float = Query(0.05, description="Min LoFTR score"),
#     strict_mode: bool = Query(True, description="If True, discard unverified matches"),
#     filter_by_shape: bool = Query(True, description="If True, only match items with the same shape"),
# ):
#     print(f"\n--- NEW REQUEST (High Capacity | Check 100 | Shape Filter: {filter_by_shape}): {modelName} ---")
#     collection = get_single_query_images_collection()

#     query: Dict[str, str] = {"modelName": modelName}
#     if skuId: query["skuId"] = skuId
#     if tenantId: query["tenantId"] = tenantId
#     if skuDescription: query["skuDescription"] = skuDescription
#     if clientId: query["clientId"] = clientId
#     if categoryId: query["categoryId"] = categoryId
#     if brandId: query["brandId"] = brandId

#     docs = list(collection.find(query))
#     if not docs:
#         raise HTTPException(status_code=404, detail="No matching AvailableModel entries found")

#     vectorizer = get_vectorizer()
#     datastore = get_datastore()

#     query_label_refs_rgb: List[np.ndarray] = []
#     best_by_crop: Dict[str, MatchResult] = {}
    
#     # Ensure we fetch enough candidates from the DB to fill 100 slots
#     PER_VIEW_LIMIT = 250 

#     total_raw = 0
#     total_dropped_score = 0
#     total_dropped_meta = 0
#     total_dropped_shape = 0 

#     for doc in docs:
#         image_b64 = doc.get("image_base64")
#         if not image_b64: continue

#         # 1. Get Query Shape
#         query_shape = doc.get("shape_label", "unknown")

#         try:
#             image_bytes = base64.b64decode(image_b64)
#             img = Image.open(io.BytesIO(image_bytes))
#             img = ImageOps.exif_transpose(img).convert("RGB")
#         except Exception:
#             continue

#         views = make_query_views(img)

#         try:
#             ref = label_crop(crop_to_content(img)).convert("RGB")
#             query_label_refs_rgb.append(np.array(ref))
#         except Exception:
#             pass

#         for view in views:
#             try:
#                 query_vector = vectorizer.get_image_embedding(view)
#             except Exception:
#                 continue

#             results = datastore.query_similar(query_vector, n_results=PER_VIEW_LIMIT)
#             total_raw += len(results)

#             for res in results:
#                 data = res.get("data") or {}
#                 crop_id = data.get("_id")
                
#                 if not crop_id or data.get("parent_image_id") is None: 
#                     total_dropped_meta += 1
#                     continue

#                 score = res.get("score")
#                 if score is None: continue
#                 score = float(score)

#                 # 1. VECTOR LIMIT (0.8)
#                 if score > 1.18: 
#                     total_dropped_score += 1
#                     continue

#                 # 2. SHAPE FILTERING
#                 if filter_by_shape and query_shape != "unknown":
#                     candidate_metadata = data.get("metadata", {})
#                     # Handle flattened vs nested metadata
#                     candidate_shape = data.get("shape_label") or candidate_metadata.get("shape_label") or "unknown"
                    
#                     if candidate_shape != "unknown" and query_shape != candidate_shape:
#                         total_dropped_shape += 1
#                         continue

#                 bbox_raw = get_bbox_from_record(data)
#                 bbox_norm = normalize_bbox_no_clamp(bbox_raw)
#                 if bbox_norm is None: 
#                     total_dropped_meta += 1
#                     continue

#                 existing = best_by_crop.get(crop_id)
#                 if existing is None or score < existing.score:
#                     best_by_crop[crop_id] = MatchResult(
#                         shelf_id=str(data.get("parent_image_id")),
#                         bbox=bbox_norm,
#                         score=score, 
#                     )

#     print(f"DEBUG: Raw: {total_raw} | Drop(Meta): {total_dropped_meta} | Drop(Score>0.8): {total_dropped_score} | Drop(Shape): {total_dropped_shape}")
#     print(f"DEBUG: Unique Candidates Kept: {len(best_by_crop)}")

#     if not best_by_crop:
#         print("DEBUG: Zero matches found.")
#         raise HTTPException(status_code=404, detail="No matches found")

#     # 3. Group by Shelf & Apply NMS
#     all_matches = list(best_by_crop.values())
#     shelf_to_matches: Dict[str, List[MatchResult]] = defaultdict(list)
#     for m in all_matches:
#         shelf_to_matches[m.shelf_id].append(m)

#     filtered_shelf_matches: Dict[str, List[MatchResult]] = {}
#     for shelf_id, shelf_matches in shelf_to_matches.items():
#         nms_matches = apply_nms(shelf_matches, iou_threshold=nms_iou_threshold)
#         nms_matches.sort(key=lambda m: m.score)
#         # Keeps up to 100 items per shelf
#         filtered_shelf_matches[shelf_id] = nms_matches[:max_results_per_shelf]

#     # 4. Load Images & Verify (THREAD-SAFE)
#     shelves_dir = settings.SHELF_DIR
#     annotated_images: Dict[str, np.ndarray] = {}
    
#     # Get asyncio loop
#     loop = asyncio.get_running_loop()

#     for shelf_id, shelf_matches in filtered_shelf_matches.items():
#         shelf_path = shelves_dir / f"{shelf_id}.png"
#         if not shelf_path.exists(): continue

#         shelf_img_bgr = cv2.imread(str(shelf_path))
#         if shelf_img_bgr is None: continue
        
#         # Verify Matches (Heavy GPU/CPU) in a thread
#         if verify_matches and query_label_refs_rgb:
#             # Explicitly limiting to 100 max
#             limit = min(len(shelf_matches), max(1, int(max_verification_candidates)))
#             candidates = shelf_matches[:limit]
#             rest = shelf_matches[limit:] 
            
#             confirmed_matches = await loop.run_in_executor(
#                 None, 
#                 partial(
#                     run_verification_sync, 
#                     candidates, 
#                     shelf_img_bgr, 
#                     query_label_refs_rgb, 
#                     verification_threshold, 
#                     strict_mode
#                 )
#             )

#             if strict_mode:
#                 shelf_matches = confirmed_matches
#             else:
#                 shelf_matches = confirmed_matches + rest
            
#             filtered_shelf_matches[shelf_id] = shelf_matches

#         # Visualization
#         img_height, img_width = shelf_img_bgr.shape[:2]
#         matching_set = set()
#         for m in shelf_matches:
#             vb = validate_and_clamp_bbox(m.bbox, img_width, img_height)
#             if vb is not None: matching_set.add(tuple(vb))

#         for m in shelf_matches:
#             vb = validate_and_clamp_bbox(m.bbox, img_width, img_height)
#             if vb:
#                 cv2.rectangle(shelf_img_bgr, (vb[0], vb[1]), (vb[2]-1, vb[3]-1), (0, 255, 0), 3)
#                 label_text = f"{m.score:.3f}"
#                 (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#                 y_top = max(vb[1] - (th + baseline + 6), 0)
#                 cv2.rectangle(shelf_img_bgr, (vb[0], y_top), (vb[0] + tw + 6, y_top + th + baseline + 6), (0, 255, 0), -1)
#                 cv2.putText(shelf_img_bgr, label_text, (vb[0] + 3, y_top + th + baseline + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

#         annotated_images[shelf_id] = shelf_img_bgr

#     if not annotated_images:
#         raise HTTPException(status_code=404, detail="No shelf images found on disk")

#     imgs = list(annotated_images.values())
#     widths = [img.shape[1] for img in imgs]
#     heights = [img.shape[0] for img in imgs]
#     max_width = max(widths) if widths else 0
#     total_height = sum(heights) + 10 * (len(imgs) - 1) if heights else 0
#     combined = np.zeros((total_height, max_width, 3), dtype=np.uint8)
#     y_offset = 0
#     for img in imgs:
#         h, w, _ = img.shape
#         combined[y_offset:y_offset + h, 0:w] = img
#         y_offset += h + 10
#     ok, buf = cv2.imencode(".png", combined)
#     if not ok: raise HTTPException(status_code=500, detail="Encoding failed")
#     return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")

def l2_to_cosine(l2_distance: float) -> float:
    # Formula for normalized vectors: Cosine = 1 - (Distance^2 / 2)
    return max(0.0, min(1.0, 1.0 - (l2_distance ** 2) / 2))

@router.get("/search-visual-by-available")
async def search_visual_by_available(
    modelName: str,
    skuId: str | None = None,
    tenantId: str | None = None,
    
    # 1. HARD FLOOR: Basic sanity check (e.g., 0.40)
    similarity_threshold: float = Query(0.40, description="Hard Floor: Min Similarity (0.0-1.0)"),
    
    # 2. DYNAMIC THRESHOLD: The "Relative" check (e.g., 0.30)
    relative_drop_off: float = Query(0.30, description="Dynamic Threshold: Allowed deviation from best match"),
    
    filter_by_shape: bool = Query(True, description="Strictly enforce shape matching"),
):
    print(f"\n--- SEARCH: {modelName} | Floor: {similarity_threshold} | Drop-off: {relative_drop_off} ---")
    
    # --- 1. SETUP & QUERY ---
    collection = get_single_query_images_collection()
    query_doc = collection.find_one({"modelName": modelName})
    if not query_doc:
        raise HTTPException(status_code=404, detail="Model not found")

    query_shape = query_doc.get("shape_label", "unknown")
    image_bytes = base64.b64decode(query_doc.get("image_base64"))
    query_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # --- 2. VECTOR SEARCH ---
    vectorizer = get_vectorizer()
    datastore = get_datastore()
    
    query_vector = vectorizer.get_image_embedding(query_img)
    results = datastore.query_similar(query_vector, n_results=2000)

    # --- 3. FILTERING ---
    potential_matches = []
    
    for res in results:
        data = res.get("data") or {}
        
        # Shape Filter
        if filter_by_shape and query_shape != "unknown":
            candidate_shape = data.get("shape_label") or data.get("metadata", {}).get("shape_label") or "unknown"
            if candidate_shape != "unknown" and query_shape != candidate_shape:
                continue 

        # Calculate Score
        l2_distance = float(res.get("score", 2.0))
        similarity = l2_to_cosine(l2_distance)

        # Hard Floor
        if similarity < similarity_threshold:
            continue

        bbox = get_bbox_from_record(data)
        if not bbox: continue

        potential_matches.append({
            "crop_id": data.get("_id"),
            "shelf_id": str(data.get("parent_image_id")),
            "bbox": bbox,
            "score": similarity
        })

    if not potential_matches:
        print("DEBUG: No matches met the Hard Floor.")
        raise HTTPException(status_code=404, detail="No matches found")

    # Dynamic Threshold Calculation
    global_max_score = max(m["score"] for m in potential_matches)
    dynamic_cutoff = global_max_score - relative_drop_off
    final_cutoff = max(similarity_threshold, dynamic_cutoff)

    best_matches: Dict[str, MatchResult] = {}

    for m in potential_matches:
        if m["score"] < final_cutoff:
            continue # Dropped by dynamic threshold
            
        c_id = m["crop_id"]
        # Keep best score for unique crop ID
        if c_id not in best_matches or m["score"] > best_matches[c_id].score:
            best_matches[c_id] = MatchResult(
                shelf_id=m["shelf_id"],
                bbox=m["bbox"],
                score=m["score"]
            )

    if not best_matches:
        raise HTTPException(status_code=404, detail="No matches passed Dynamic Threshold")

    # Group matches by shelf
    shelf_matches = defaultdict(list)
    for m in best_matches.values():
        shelf_matches[m.shelf_id].append(m)

    # --- 4. VISUALIZATION (RED & GREEN BOXES) ---
    shelves_dir = settings.SHELF_DIR
    annotated_images = []

    for shelf_id, matches in shelf_matches.items():
        shelf_path = shelves_dir / f"{shelf_id}.png"
        if not shelf_path.exists(): continue
        
        img = cv2.imread(str(shelf_path))
        if img is None: continue
        h, w = img.shape[:2]

        # A. Draw RED boxes for everything on the shelf first
        # ---------------------------------------------------
        if datastore.mongo_coll is not None:
            # Fetch EVERY item detected on this shelf
            all_shelf_items = datastore.mongo_coll.find({"parent_image_id": shelf_id})
            
            for item in all_shelf_items:
                bbox = get_bbox_from_record(item)
                if not bbox: continue
                
                vb = validate_and_clamp_bbox(bbox, w, h)
                if vb:
                    # RED Box (BGR: 0, 0, 255)
                    # Use thickness=2 so it's slightly thinner than the Green box
                    cv2.rectangle(img, (vb[0], vb[1]), (vb[2], vb[3]), (0, 0, 255), 2)

        # B. Draw GREEN boxes for Matches (Overwriting Red)
        # ---------------------------------------------------
        matches.sort(key=lambda x: x.score, reverse=True)
        # Apply NMS to clean up Green boxes
        cleaned_matches = apply_nms(matches, iou_threshold=0.5)

        for m in cleaned_matches:
            vb = validate_and_clamp_bbox(m.bbox, w, h)
            if vb:
                # GREEN Box (BGR: 0, 255, 0)
                # Use thickness=3 to fully cover the Red box underneath
                cv2.rectangle(img, (vb[0], vb[1]), (vb[2], vb[3]), (0, 255, 0), 3)
                
                # Label
                text = f"{m.score:.2f}"
                cv2.putText(img, text, (vb[0], vb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        annotated_images.append(img)

    if not annotated_images:
        raise HTTPException(status_code=404, detail="Shelf image not found")

    # Combine images
    total_h = sum(img.shape[0] for img in annotated_images)
    max_w = max(img.shape[1] for img in annotated_images)
    combined = np.zeros((total_h, max_w, 3), dtype=np.uint8)
    
    y = 0
    for img in annotated_images:
        h, w = img.shape[:2]
        combined[y:y+h, 0:w] = img
        y += h
        
    ok, buf = cv2.imencode(".png", combined)
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")




@router.delete("/reset-database")
async def reset_database():
    # 1. Clear MongoDB Metadata
    datastore = get_datastore()
    if datastore.mongo_coll is not None:
        datastore.mongo_coll.delete_many({}) # Delete ALL products
    
    # 2. Clear Single Query Images (Models)
    collection = get_single_query_images_collection()
    collection.delete_many({})
    
    # 3. Clear Vector DB (Milvus/Chroma)
    if datastore.vector_backend == "milvus":
        if hasattr(datastore, "milvus_collection") and datastore.milvus_collection:
            datastore.milvus_collection.drop()
            datastore._init_milvus_vector_store()
    else:
        # For Chroma
        try:
            datastore.chroma_client.delete_collection("product_vectors")
        except:
            pass # Collection might not exist
        datastore._init_chroma_vector_store()

    return {"status": "Database completely wiped. Please re-upload everything."}



@router.get("/debug-db-counts")
async def debug_db_counts():
    datastore = get_datastore()
    
    # Check MongoDB Count (Should be small, e.g., 50)
    mongo_count = 0
    if datastore.mongo_coll is not None:
        mongo_count = datastore.mongo_coll.count_documents({})

    # Check Vector DB Count (If this is 2000+, you have Ghosts!)
    vector_count = "N/A"
    try:
        if datastore.vector_backend == "milvus":
            vector_count = datastore.milvus_collection.num_entities
        else:
            vector_count = datastore.chroma_collection.count()
    except:
        pass

    return {
        "MongoDB_Items": mongo_count,
        "VectorDB_Items": vector_count,
        "Status": "SYNCED" if mongo_count == vector_count else "GHOSTS DETECTED"
    }


@router.get("/shelf-stats/{image_id}")
async def get_shelf_shape_stats(image_id: str):
    """
    Returns the count of bottles, cans, and unknown items 
    for a specific shelf image.
    """
    datastore = get_datastore()
    if datastore.mongo_coll is None:
        raise HTTPException(status_code=500, detail="MongoDB not connected")

    # Count by Shape
    num_bottles = datastore.mongo_coll.count_documents({
        "parent_image_id": image_id,
        "shape_label": "bottle"
    })
    
    num_cans = datastore.mongo_coll.count_documents({
        "parent_image_id": image_id,
        "shape_label": "can"
    })
    
    # Check for "canister" too, just in case the code wasn't updated
    num_canisters = datastore.mongo_coll.count_documents({
        "parent_image_id": image_id,
        "shape_label": "canister"
    })

    num_unknown = datastore.mongo_coll.count_documents({
        "parent_image_id": image_id, 
        "shape_label": "unknown"
    })
    
    # Check for missing labels (Old data / Ghosts)
    num_missing = datastore.mongo_coll.count_documents({
        "parent_image_id": image_id, 
        "shape_label": {"$exists": False}
    })

    total = num_bottles + num_cans + num_canisters + num_unknown + num_missing

    return {
        "shelf_id": image_id,
        "total_items": total,
        "breakdown": {
            "bottle": num_bottles,
            "can": num_cans,
            "canister": num_canisters, # Should be 0 if you fixed the code
            "unknown": num_unknown,
            "missing_label": num_missing
        }
    }




@router.get("/shelf-stats/{image_id}")
async def get_shelf_shape_stats(image_id: str):
    """
    Returns the count of bottles, cans, and unknown items 
    for a specific shelf image.
    """
    datastore = get_datastore()
    if datastore.mongo_coll is None:
        raise HTTPException(status_code=500, detail="MongoDB not connected")

    # Count by Shape
    num_bottles = datastore.mongo_coll.count_documents({
        "parent_image_id": image_id,
        "shape_label": "bottle"
    })
    
    num_cans = datastore.mongo_coll.count_documents({
        "parent_image_id": image_id,
        "shape_label": "can"
    })
    
    # Check for "canister" too, just in case the code wasn't updated
    num_canisters = datastore.mongo_coll.count_documents({
        "parent_image_id": image_id,
        "shape_label": "canister"
    })

    num_unknown = datastore.mongo_coll.count_documents({
        "parent_image_id": image_id, 
        "shape_label": "unknown"
    })
    
    # Check for missing labels (Old data / Ghosts)
    num_missing = datastore.mongo_coll.count_documents({
        "parent_image_id": image_id, 
        "shape_label": {"$exists": False}
    })

    total = num_bottles + num_cans + num_canisters + num_unknown + num_missing

    return {
        "shelf_id": image_id,
        "total_items": total,
        "breakdown": {
            "bottle": num_bottles,
            "can": num_cans,
            "canister": num_canisters, # Should be 0 if you fixed the code
            "unknown": num_unknown,
            "missing_label": num_missing
        }
    }