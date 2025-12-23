from __future__ import annotations 

from typing import List, Dict, Optional
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
from PIL import Image
import cv2
import numpy as np
import pymongo
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.services.yolo_v11_detection import run_yolo_v11_detection
from functools import lru_cache
from ultralytics import YOLO
from app.core.config import settings
from app.services.vectorizer import get_vectorizer
from app.services.datastore import get_datastore
from app.services.detection import run_detection

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


# ===========================
# Helper function for parallel processing
# ===========================

def extract_embedding_only(
    crop: Image.Image,
    bbox_list: List[int],
    label: str,
    conf: float,
    vectorizer,
) -> Dict[str, Any]:
    """
    Extract embedding from a single crop (no database save).
    This function is designed to be run in parallel threads.
    Returns: dict with vector, bbox_list, label, conf, crop_id
    """
    try:
        # Extract embedding
        vector = vectorizer.get_image_embedding(crop)
        
        # Generate unique ID
        crop_id = str(uuid.uuid4())
        
        return {
            "crop_id": crop_id,
            "vector": vector,
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
# Models
# ===========================

class IndexShelfResponse(BaseModel):
    image_id: str
    num_detections: int
    num_indexed: int


class ShelfInfo(BaseModel):
    shelf_id: str
    num_objects: int


class MatchResult(BaseModel):
    # include shelf_id so we can search across all shelves
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


class YoloDetectionItem(BaseModel):
    bbox: List[int]          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class YoloDetectionResponse(BaseModel):
    detections: List[YoloDetectionItem]
    annotated_image_base64: str



# ===========================
# Shelf indexing & listing
# ===========================

@router.post("/index-shelf", response_model=IndexShelfResponse)
async def index_shelf(
    file: UploadFile = File(...),
    prompt: str = "Bottle",
    box_thresh: float = 0.3,
):
    """
    Upload a full shelf image, detect objects with GroundingDINO,
    and index them in the vector DB as a new shelf (identified by image_id).
    """
    if not (0.0 <= box_thresh <= 1.0):
        raise HTTPException(status_code=400, detail="box_thresh must be between 0.0 and 1.0")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    detections = run_detection(image, prompt=prompt, box_thresh=box_thresh)

    image_id = str(uuid.uuid4())
    vectorizer = get_vectorizer()
    datastore = get_datastore()

    # Prepare all crops and metadata for parallel processing
    crop_tasks = []
    for i in range(len(detections.xyxy)):
        box = detections.xyxy[i]
        label = detections.data["class_name"][i]
        conf = float(detections.confidence[i])

        # Convert bbox to [x1, y1, x2, y2]
        bbox_list = [int(coord) for coord in box]

        # Crop the object from the original image
        x1, y1, x2, y2 = bbox_list
        crop = image.crop((x1, y1, x2, y2))

        crop_tasks.append((crop, bbox_list, label, conf))

    # Step 1: Extract embeddings in parallel (no database operations)
    all_bboxes: List[List[int]] = []
    objects_to_save: List[Dict[str, Any]] = []
    
    # Use ThreadPoolExecutor for parallel embedding extraction
    max_workers = min(len(crop_tasks), 8)  # Limit to 8 threads
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all embedding extraction tasks
        futures = [
            executor.submit(
                extract_embedding_only,
                crop,
                bbox_list,
                label,
                conf,
                vectorizer,
            )
            for crop, bbox_list, label, conf in crop_tasks
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                objects_to_save.append(result)
                all_bboxes.append(result["bbox_list"])
    
    # Step 2: Batch save all objects to database at once (much faster!)
    if objects_to_save:
        datastore.batch_save_objects(image_id, objects_to_save)
    
    num_indexed = len(objects_to_save)

    # Save this shelf image to disk under its image_id
    shelves_dir = settings.SHELF_DIR
    shelves_dir.mkdir(parents=True, exist_ok=True)
    out_path = shelves_dir / f"{image_id}.png"
    image.save(out_path)

    return IndexShelfResponse(
        image_id=image_id,
        num_detections=len(detections),
        num_indexed=num_indexed,
    )


@router.get("/shelves", response_model=List[ShelfInfo])
async def list_shelves():
    """List all shelves that have indexed objects."""
    datastore = get_datastore()
    raw = datastore.list_shelves()
    return [ShelfInfo(**s) for s in raw]


# ===========================
# Search (Visual, per-shelf)
# ===========================

@router.post("/search-visual/{shelf_id}", response_model=SearchResponse)
async def search_visual(
    shelf_id: str,
    file: UploadFile = File(...),
    max_results: int = 10,
    match_threshold: float = 0.19,
    only_matches: bool = True,
):
    """
    Visual search within a single shelf (identified by shelf_id).
    - shelf_id is the ID of the shelf image previously indexed via /index-shelf.
    - file is the query image (or object).
    """

    contents = await file.read()
    try:
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    vectorizer = get_vectorizer()
    datastore = get_datastore()

    query_vector = vectorizer.get_image_embedding(query_image)
    # Ask for a larger pool of results so we have enough hits on the top shelf
    results = datastore.query_similar(query_vector, n_results=max_results * 10)

    if not results:
        raise HTTPException(status_code=404, detail="No indexed objects found in database")

    # Filter to only results matching the given shelf_id
    filtered: List[MatchResult] = []
    for res in results:
        data = res["data"]
        parent_id = data.get("parent_image_id")
        if parent_id != shelf_id:
            continue

        score = res["score"]
        if only_matches and score > match_threshold:
            continue

        bbox = data.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        filtered.append(
            MatchResult(
                shelf_id=parent_id,
                bbox=bbox,
                score=score,
            )
        )

        if len(filtered) >= max_results:
            break

    return SearchResponse(matches=filtered)


# ===========================
# Search (Visual, GLOBAL - BEST shelf only)
# ===========================

@router.post("/search-visual", response_model=SearchResponse)
async def search_visual_best_shelf(
    file: UploadFile = File(...),
    max_results: int = 10,
    match_threshold: float = 0.19,
    only_matches: bool = True,
):
    """
    Global visual search across all shelves, but only return matches
    from the single "best" shelf for this query.
    """

    contents = await file.read()
    try:
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    vectorizer = get_vectorizer()
    datastore = get_datastore()

    query_vector = vectorizer.get_image_embedding(query_image)
    # Ask for a larger pool of results so we have enough hits on the top shelf
    results = datastore.query_similar(query_vector, n_results=max_results * 10)

    if not results:
        raise HTTPException(status_code=404, detail="No indexed objects found in database")

    # Tally matches per shelf_id, gather them
    shelf_matches: Dict[str, List[MatchResult]] = defaultdict(list)
    for res in results:
        data = res["data"]
        parent_id = data.get("parent_image_id")
        if parent_id is None:
            continue

        score = res["score"]
        if only_matches and score > match_threshold:
            continue

        bbox = data.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        shelf_matches[parent_id].append(
            MatchResult(
                shelf_id=parent_id,
                bbox=bbox,
                score=score,
            )
        )

    if not shelf_matches:
        # Means no matches were under threshold
        raise HTTPException(status_code=404, detail="No sufficiently good matches found")

    # Decide which shelf is best: e.g., the one with the most matches
    best_shelf_id = max(shelf_matches.keys(), key=lambda sid: len(shelf_matches[sid]))
    best_matches = shelf_matches[best_shelf_id]

    # Sort matches by score (ascending, so best is first if score is distance)
    best_matches.sort(key=lambda m: m.score)

    # Limit to max_results
    best_matches = best_matches[:max_results]

    return SearchResponse(matches=best_matches)


# ===========================
# Search (Visual, GLOBAL - ALL matching shelves)
# ===========================

@router.post("/search-visual-multi")
async def search_visual_multi(
    file: UploadFile = File(...),
    max_results: int = 10,
    match_threshold: float = 0.19,
    only_matches: bool = True,
    as_zip: bool = False,  # False => combined PNG, True => ZIP of separate PNGs
):
    """
    GLOBAL visual search across ALL shelves.
    - file: query image
    - Returns:
      - if as_zip = False: one big PNG of all annotated shelves stacked vertically
      - if as_zip = True: a ZIP file, with one PNG per shelf
    """
    contents = await file.read()
    try:
        query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    vectorizer = get_vectorizer()
    datastore = get_datastore()

    query_vector = vectorizer.get_image_embedding(query_image)
    # Ask for extra results, then filter by threshold (and optional shelf)
    results = datastore.query_similar(query_vector, n_results=max_results * 3)

    matches: List[MatchResult] = []

    for res in results:
        data = res["data"]
        parent_id = data.get("parent_image_id")
        if parent_id is None:
            continue

        score = res["score"]
        if only_matches and score > match_threshold:
            continue

        bbox = data.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        matches.append(
            MatchResult(
                shelf_id=parent_id,
                bbox=bbox,
                score=score,
            )
        )

        if len(matches) >= max_results:
            break

    if not matches:
        raise HTTPException(status_code=404, detail="No sufficiently good matches found")

    # ------------------------------
    # We now have matches across shelves.
    # We want to draw them on shelf images and either:
    #  - return them as a single combined PNG, or
    #  - put each shelf’s annotated image in a ZIP file.
    # ------------------------------

    # Group matches by shelf_id
    shelf_to_matches: Dict[str, List[MatchResult]] = defaultdict(list)
    for m in matches:
        shelf_to_matches[m.shelf_id].append(m)

    # Sort each shelf's matches by score ascending
    for sid in shelf_to_matches:
        shelf_to_matches[sid].sort(key=lambda m: m.score)

    # Load each shelf image from disk
    shelves_dir = settings.SHELF_DIR
    annotated_images: Dict[str, np.ndarray] = {}

    for shelf_id, shelf_matches in shelf_to_matches.items():
        shelf_path = shelves_dir / f"{shelf_id}.png"
        if not shelf_path.exists():
            # If the shelf image is missing, skip it
            continue

        shelf_img = cv2.imread(str(shelf_path))
        if shelf_img is None:
            # If cv2 fails to read, skip it
            continue

        # Draw bounding boxes
        for m in shelf_matches:
            x1, y1, x2, y2 = m.bbox

            cv2.rectangle(
                shelf_img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),  # green
                2,
            )

            cv2.putText(
                shelf_img,
                f"{m.score:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        annotated_images[shelf_id] = shelf_img

    if not annotated_images:
        raise HTTPException(status_code=404, detail="No shelf images found on disk")

    # ------------------------------
    # Mode A: Return ZIP of per-shelf images
    # ------------------------------
    if as_zip:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for shelf_id, img in annotated_images.items():
                success, buffer = cv2.imencode(".png", img)
                if not success:
                    continue

                png_bytes = buffer.tobytes()
                filename = f"{shelf_id}.png"
                zipf.writestr(filename, png_bytes)

        zip_buffer.seek(0)
        headers = {
            "Content-Disposition": 'attachment; filename="search_results.zip"',
        }
        return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)

    # ----- Option 2: ONE combined PNG (Swagger-friendly) -----
    imgs = list(annotated_images.values())

    widths = [img.shape[1] for img in imgs]
    heights = [img.shape[0] for img in imgs]
    max_width = max(widths)
    total_height = sum(heights) + 10 * (len(imgs) - 1)

    combined = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    y_offset = 0
    for img in imgs:
        h, w, _ = img.shape
        combined[y_offset:y_offset + h, 0:w] = img
        y_offset += h + 10

    success, buffer = cv2.imencode(".png", combined)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode combined image")

    bytes_io = io.BytesIO(buffer.tobytes())
    return StreamingResponse(bytes_io, media_type="image/png")


# ===========================
# Single query image storage
# ===========================

def get_single_query_images_collection():
    """Get the MongoDB collection used to store single query images."""
    if not settings.MONGO_URI:
        raise HTTPException(status_code=500, detail="MongoDB URI is not configured")

    try:
        mongo_client = pymongo.MongoClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=2000,
        )
        mongo_client.server_info()  # quick connection check

        # Use DB name from .env if provided; else default from URI; else fallback
        if settings.MONGO_DB_NAME:
            mongo_db = mongo_client[settings.MONGO_DB_NAME]
        else:
            try:
                mongo_db = mongo_client.get_default_database()
            except Exception:
                mongo_db = mongo_client["cstore-ai"]

        collection = mongo_db["single_query_images"]
        return collection

    except Exception:
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")

@lru_cache(maxsize=1)
def get_yolo_v11_model() -> YOLO:
    # USE YOUR ACTUAL FILE PATH HERE
    model_path = "models/yolo/yolo_v11_best.pt"
    # or an absolute path if necessary, but keep it the same as Colab’s weights
    return YOLO(model_path)

@router.post("/index-shelf-yolo", response_model=IndexShelfResponse)
async def index_shelf_yolo(
    file: UploadFile = File(...),
    box_thresh: float = 0.2,
):
    """
    Upload a full shelf image, detect objects with YOLO v11,
    and index them in the vector DB as a new shelf (identified by image_id).
    """
    if not (0.0 <= box_thresh <= 1.0):
        raise HTTPException(status_code=400, detail="box_thresh must be between 0.0 and 1.0")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # Save uploaded file to a temp path (so YOLO uses 'source=path' like in Colab)
    try:
        suffix = Path(file.filename).suffix or ".jpg"
    except Exception:
        suffix = ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Also load as PIL for cropping and saving shelf image
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    model = get_yolo_v11_model()

    # --- YOLO PREDICT (Colab-style) ---
    results = model.predict(
        source=tmp_path,        # <--- path, same as Colab pattern
        imgsz=640,
        conf=box_thresh,        # equivalent of confidence threshold
        save=False,             # we handle saving ourselves
        show_labels=False,
        show_conf=False,
        verbose=False,
    )

    if not results:
        raise HTTPException(status_code=500, detail="YOLO v11 returned no results")

    result = results[0]
    boxes = result.boxes

    # DEBUG: you can temporarily print or log this to verify
    print("YOLO result:", result)
    print("Number of boxes:", 0 if boxes is None else len(boxes))

    image_id = str(uuid.uuid4())

    # If no detections, still save shelf image but index zero
    if boxes is None or len(boxes) == 0:
        shelves_dir = settings.SHELF_DIR
        shelves_dir.mkdir(parents=True, exist_ok=True)
        out_path = shelves_dir / f"{image_id}.png"
        image.save(out_path)

        return IndexShelfResponse(
            image_id=image_id,
            num_detections=0,
            num_indexed=0,
        )

    vectorizer = get_vectorizer()
    datastore = get_datastore()

    names = result.names  # {class_id: class_name}

    # Prepare all crops and metadata for parallel processing
    crop_tasks = []
    for i in range(len(boxes)):
        box = boxes[i]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = names.get(class_id, str(class_id))

        bbox_list = [int(x1), int(y1), int(x2), int(y2)]
        crop = image.crop((bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3]))
        
        crop_tasks.append((crop, bbox_list, label, conf))

    # Step 1: Extract embeddings in parallel (no database operations)
    objects_to_save: List[Dict[str, Any]] = []
    
    # Use ThreadPoolExecutor for parallel embedding extraction
    max_workers = min(len(crop_tasks), 8)  # Limit to 8 threads
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all embedding extraction tasks
        futures = [
            executor.submit(
                extract_embedding_only,
                crop,
                bbox_list,
                label,
                conf,
                vectorizer,
            )
            for crop, bbox_list, label, conf in crop_tasks
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                objects_to_save.append(result)
    
    # Step 2: Batch save all objects to database at once (much faster!)
    if objects_to_save:
        datastore.batch_save_objects(image_id, objects_to_save)
    
    num_indexed = len(objects_to_save)

    shelves_dir = settings.SHELF_DIR
    shelves_dir.mkdir(parents=True, exist_ok=True)
    out_path = shelves_dir / f"{image_id}.png"
    image.save(out_path)

    return IndexShelfResponse(
        image_id=image_id,
        num_detections=len(boxes),
        num_indexed=num_indexed,
    )


@router.post("/ModelTraining")
async def upload_single_query_image(
    modelName: str = Form(...),
    skuDescription: str = Form(...),
    skuId: str = Form(...),
    tenantId: str = Form(...),

    # NEW optional fields (only these three are optional)
    clientId: str | None = Form(None),
    categoryId: str | None = Form(None),
    brandId: str | None = Form(None),

    image: UploadFile = File(...),
):
    """Upload a single query image with SKU + tenant info and store in MongoDB (base64)."""
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
        try:
            errmsg = e.details.get("errmsg", "")
        except Exception:
            errmsg = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"MongoDB write error: {errmsg}",
        )

    return {
        "id": str(result.inserted_id),
        "status": "stored",
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
    """
    Return stored query images filtered by:
    - required: modelName
    - optional: skuId, tenantId, skuDescription, clientId, categoryId, brandId
    """
    collection = get_single_query_images_collection()

    # Build MongoDB query dynamically
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
                image_base64=doc.get("image_base64", "")
            )
        )

    return results




 

#return models coreesponding images


@router.get("/search-visual-model")
async def search_visual_by_model(
    modelName: str,
    max_results: int = 10,
    match_threshold: float = 0.19,
    only_matches: bool = True,
    as_zip: bool = False,
):
    """
    Visual search across ALL shelves using all stored images for the given modelName.

    Works similarly to /search-visual-multi, but:
    - No image upload in the request.
    - Loads all images from MongoDB where modelName matches.
    """
    # 1. Get all query images for this modelName from MongoDB
    collection = get_single_query_images_collection()
    try:
        docs = list(collection.find({"modelName": modelName}))
    except OperationFailure as e:
        try:
            errmsg = e.details.get("errmsg", "")
        except Exception:
            errmsg = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"MongoDB read error: {errmsg}",
        )

    if not docs:
        raise HTTPException(
            status_code=404,
            detail=f"No stored images found for modelName='{modelName}'",
        )

    vectorizer = get_vectorizer()
    datastore = get_datastore()

    matches: List[MatchResult] = []

    # 2. For each stored image, run visual search and accumulate matches
    for doc in docs:
        image_b64 = doc.get("image_base64")
        if not image_b64:
            continue
        try:
            image_bytes = base64.b64decode(image_b64)
            query_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            # Skip invalid or unreadable images
            continue

        query_vector = vectorizer.get_image_embedding(query_image)

        # Ask for extra results (similar to /search-visual-multi)
        results = datastore.query_similar(query_vector, n_results=max_results * 3)

        for res in results:
            data = res["data"]
            parent_id = data.get("parent_image_id")
            if parent_id is None:
                continue

            score = res["score"]
            if only_matches and score > match_threshold:
                continue

            bbox = data.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            matches.append(
                MatchResult(
                    shelf_id=parent_id,
                    bbox=bbox,
                    score=score,
                )
            )

            # Stop when we collected enough global matches
            if len(matches) >= max_results:
                break

        if len(matches) >= max_results:
            break

    if not matches:
        raise HTTPException(
            status_code=404,
            detail="No sufficiently good matches found for this modelName",
        )

    # 3. Same drawing/response logic as in /search-visual-multi

    # Group matches by shelf_id
    shelf_to_matches: Dict[str, List[MatchResult]] = defaultdict(list)
    for m in matches:
        shelf_to_matches[m.shelf_id].append(m)

    # Sort each shelf's matches by score ascending
    for sid in shelf_to_matches:
        shelf_to_matches[sid].sort(key=lambda m: m.score)

    # Load each shelf image from disk and draw boxes
    shelves_dir = settings.SHELF_DIR
    annotated_images: Dict[str, np.ndarray] = {}

    for shelf_id, shelf_matches in shelf_to_matches.items():
        shelf_path = shelves_dir / f"{shelf_id}.png"
        if not shelf_path.exists():
            continue

        shelf_img = cv2.imread(str(shelf_path))
        if shelf_img is None:
            continue

        for m in shelf_matches:
            x1, y1, x2, y2 = m.bbox

            cv2.rectangle(
                shelf_img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),  # green
                2,
            )

            cv2.putText(
                shelf_img,
                f"{m.score:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        annotated_images[shelf_id] = shelf_img

    if not annotated_images:
        raise HTTPException(status_code=404, detail="No shelf images found on disk")

    # 4. Mode A: Return ZIP of per-shelf images
    if as_zip:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for shelf_id, img in annotated_images.items():
                success, buffer = cv2.imencode(".png", img)
                if not success:
                    continue

                png_bytes = buffer.tobytes()
                filename = f"{shelf_id}.png"
                zipf.writestr(filename, png_bytes)

        zip_buffer.seek(0)
        headers = {
            "Content-Disposition": 'attachment; filename="search_results_model.zip"',
        }
        return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)

    # 5. Mode B: One combined PNG (stacked vertically)
    imgs = list(annotated_images.values())

    widths = [img.shape[1] for img in imgs]
    heights = [img.shape[0] for img in imgs]
    max_width = max(widths)
    total_height = sum(heights) + 10 * (len(imgs) - 1)

    combined = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    y_offset = 0
    for img in imgs:
        h, w, _ = img.shape
        combined[y_offset:y_offset + h, 0:w] = img
        y_offset += h + 10

    success, buffer = cv2.imencode(".png", combined)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode combined image")

    bytes_io = io.BytesIO(buffer.tobytes())
    return StreamingResponse(bytes_io, media_type="image/png")


# ===========================
# YOLO v11 Detection Endpoint
# ===========================

@router.post("/yolo-v11/detect", response_model=YoloDetectionResponse)
async def yolo_v11_detect(
    file: UploadFile = File(...),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
):
    """
    Use YOLO v11 (ultralytics) to detect objects in a single image.
    """
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if not (0.0 <= conf_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="conf_threshold must be between 0.0 and 1.0")
    if not (0.0 <= iou_threshold <= 1.0):
        raise HTTPException(status_code=400, detail="iou_threshold must be between 0.0 and 1.0")

    try:
        result = run_yolo_v11_detection(
            image=image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    detections = [
        YoloDetectionItem(
            bbox=d["bbox"],
            confidence=d["confidence"],
            class_id=d["class_id"],
            class_name=d["class_name"],
        )
        for d in result["detections"]
    ]

    return YoloDetectionResponse(
        detections=detections,
        annotated_image_base64=result["annotated_image_base64"],
    )





@router.get("/search-visual-by-available")
async def search_visual_by_available(
    modelName: str,
    skuId: str | None = None,
    tenantId: str | None = None,
    skuDescription: str | None = None,
    clientId: str | None = None,
    categoryId: str | None = None,
    brandId: str | None = None,
    match_threshold: float = 0.19,
    only_matches: bool = True,
    as_zip: bool = False,
):
    """
    Use stored single query images (AvailableModel) as queries to search
    across all shelves.

    - Required filter: modelName
    - Optional filters: skuId, tenantId, skuDescription, clientId, categoryId, brandId
    - Returns:
      - if as_zip = False: one big PNG with all matching shelves stacked vertically
      - if as_zip = True: ZIP file with one PNG per shelf
    """

    # 1) Load matching query images from Mongo
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

    # Collect all matches from all query images
    matches: List[MatchResult] = []

    # Internal per-query limit; no max_results parameter is exposed to user
    PER_QUERY_LIMIT = 1000  # adjust as needed for your data size

    for doc in docs:
        image_b64 = doc.get("image_base64")
        if not image_b64:
            continue

        try:
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            # Skip broken images
            continue

        query_vector = vectorizer.get_image_embedding(img)

        # Query vector store for this query image
        results = datastore.query_similar(
            query_vector,
            n_results=PER_QUERY_LIMIT,
        )

        for res in results:
            data = res["data"]
            parent_id = data.get("parent_image_id")
            if parent_id is None:
                continue

            score = res["score"]
            if only_matches and score > match_threshold:
                continue

            bbox = data.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            matches.append(
                MatchResult(
                    shelf_id=parent_id,
                    bbox=bbox,
                    score=score,
                )
            )

    if not matches:
        raise HTTPException(status_code=404, detail="No sufficiently good matches found")

    # ------------------------------
    # Same drawing logic as /search-visual-multi
    # ------------------------------
    shelf_to_matches: Dict[str, List[MatchResult]] = defaultdict(list)
    for m in matches:
        shelf_to_matches[m.shelf_id].append(m)

    for sid in shelf_to_matches:
        shelf_to_matches[sid].sort(key=lambda m: m.score)

    shelves_dir = settings.SHELF_DIR
    annotated_images: Dict[str, np.ndarray] = {}

    for shelf_id, shelf_matches in shelf_to_matches.items():
        shelf_path = shelves_dir / f"{shelf_id}.png"
        if not shelf_path.exists():
            continue

        shelf_img = cv2.imread(str(shelf_path))
        if shelf_img is None:
            continue

        for m in shelf_matches:
            x1, y1, x2, y2 = m.bbox

            cv2.rectangle(
                shelf_img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                shelf_img,
                f"{m.score:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        annotated_images[shelf_id] = shelf_img

    if not annotated_images:
        raise HTTPException(status_code=404, detail="No shelf images found on disk")

    # Mode A: ZIP
    if as_zip:
        import zipfile

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for shelf_id, img in annotated_images.items():
                success, buffer = cv2.imencode(".png", img)
                if not success:
                    continue
                png_bytes = buffer.tobytes()
                filename = f"{shelf_id}.png"
                zipf.writestr(filename, png_bytes)

        zip_buffer.seek(0)
        headers = {
            "Content-Disposition": 'attachment; filename="search_results_by_available.zip"',
        }
        return StreamingResponse(zip_buffer, media_type="application/zip", headers=headers)

    # Mode B: single combined PNG
    imgs = list(annotated_images.values())

    widths = [img.shape[1] for img in imgs]
    heights = [img.shape[0] for img in imgs]
    max_width = max(widths)
    total_height = sum(heights) + 10 * (len(imgs) - 1)

    combined = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    y_offset = 0
    for img in imgs:
        h, w, _ = img.shape
        combined[y_offset:y_offset + h, 0:w] = img
        y_offset += h + 10

    success, buffer = cv2.imencode(".png", combined)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode combined image")

    bytes_io = io.BytesIO(buffer.tobytes())
    return StreamingResponse(bytes_io, media_type="image/png")