from __future__ import annotations 

from typing import List, Dict
from collections import defaultdict
import io
import uuid
import zipfile
import base64
from datetime import datetime
from pymongo.errors import OperationFailure
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
import cv2
import numpy as np
import pymongo

from app.core.config import settings
from app.services.vectorizer import get_vectorizer
from app.services.datastore import get_datastore
from app.services.detection import run_detection

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok"}


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

    all_bboxes: List[List[int]] = []
    num_indexed = 0

    for i in range(len(detections.xyxy)):
        box = detections.xyxy[i]
        label = detections.data["class_name"][i]
        conf = float(detections.confidence[i])

        # Convert bbox to [x1, y1, x2, y2]
        bbox_list = [int(coord) for coord in box]

        # Crop the object from the original image
        x1, y1, x2, y2 = bbox_list
        crop = image.crop((x1, y1, x2, y2))

        # Get vector
        vector = vectorizer.get_image_embedding(crop)

        # Make a unique ID for this object
        crop_id = str(uuid.uuid4())

        # Save vector + metadata
        datastore.save_object(
            image_id=image_id,
            crop_id=crop_id,
            vector=vector,
            metadata={
                "bbox": bbox_list,
                "label": label,
                "confidence": conf,
            },
        )

        all_bboxes.append(bbox_list)
        num_indexed += 1

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
@router.post("/single-query-image")
async def single_query_image(
    modelName: str = Form(...),
    ClassificationItem: str = Form(...),
    image: UploadFile = File(...),
):
    """Store a single query image + metadata into MongoDB as base64."""
    if not settings.MONGO_URI:
        raise HTTPException(status_code=500, detail="MongoDB URI is not configured")

    # Connect to MongoDB
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

    except Exception:
        raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")

    file_bytes = await image.read()
    image_b64 = base64.b64encode(file_bytes).decode("utf-8")

    doc_id = str(uuid.uuid4())
    document = {
        "_id": doc_id,
        "modelName": modelName,
        "ClassificationItem": ClassificationItem,
        "filename": image.filename,
        "content_type": image.content_type,
        "image_base64": image_b64,
        "created_at": datetime.utcnow(),
    }

    # Insert into MongoDB and handle permission errors
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
