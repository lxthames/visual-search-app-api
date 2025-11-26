from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import pymongo

from app.core.config import settings


class DataStore:
    def __init__(self, mongo_uri: Optional[str] = None) -> None:
        # ---------- Vector DB (Chroma) ----------
        chroma_path = Path(settings.CHROMA_PATH)
        chroma_path.parent.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        self.vector_collection = self.chroma_client.get_or_create_collection(
            name="product_vectors"
        )

        # ---------- Metadata DB (Mongo or JSON) ----------
        self.use_mongo = False
        self.mongo_coll = None

        self.local_meta: List[Dict[str, Any]] = []
        self.local_meta_path = Path(settings.METADATA_PATH)
        self.local_meta_path.parent.mkdir(parents=True, exist_ok=True)

        if mongo_uri:
            try:
                mongo_client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
                mongo_client.server_info()
                mongo_db = mongo_client["visual_search_db"]
                self.mongo_coll = mongo_db["products"]
                self.use_mongo = True
            except Exception:
                self.use_mongo = False

        if not self.use_mongo:
            if self.local_meta_path.exists():
                with self.local_meta_path.open("r", encoding="utf-8") as f:
                    self.local_meta = json.load(f)




    def list_shelves(self) -> List[Dict[str, Any]]:
        """Return list of shelves with number of indexed objects."""
        shelves: Dict[str, int] = {}

        if self.use_mongo and self.mongo_coll is not None:
            pipeline = [
                {"$group": {"_id": "$parent_image_id", "count": {"$sum": 1}}},
            ]
            for doc in self.mongo_coll.aggregate(pipeline):
                sid = doc["_id"]
                if sid:
                    shelves[sid] = doc["count"]
        else:
            for item in self.local_meta:
                sid = item.get("parent_image_id")
                if not sid:
                    continue
                shelves[sid] = shelves.get(sid, 0) + 1

        return [
            {"shelf_id": sid, "num_objects": count}
            for sid, count in shelves.items()
        ]

    def has_shelf(self, shelf_id: str) -> bool:
        """Check if at least one object exists for this shelf_id."""
        if self.use_mongo and self.mongo_coll is not None:
            return (
                self.mongo_coll.count_documents({"parent_image_id": shelf_id}, limit=1)
                > 0
            )
        return any(item.get("parent_image_id") == shelf_id for item in self.local_meta)
 
    # ---------- Persistence helpers ----------
    def _save_local_meta(self) -> None:
        with self.local_meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.local_meta, f, ensure_ascii=False, indent=2)

    # ---------- Public API ----------
    def save_object(
        self,
        image_id: str,
        crop_id: str,
        vector: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        # 1. Save vector in Chroma
        self.vector_collection.add(
            embeddings=[vector],
            metadatas=[{"mongo_id": crop_id}],
            ids=[crop_id],
        )

        # 2. Prepare metadata record
        record = {
            "_id": crop_id,
            "parent_image_id": image_id,
            "label": str(metadata.get("label", "")),
            "confidence": metadata.get("confidence"),
            "bbox": metadata.get("bbox"),
            "timestamp": datetime.now().isoformat(),
        }

        # 3. Save metadata in Mongo or local JSON
        if self.use_mongo and self.mongo_coll is not None:
            self.mongo_coll.insert_one(record)
        else:
            self.local_meta.append(record)
            self._save_local_meta()
    
    def query_similar(self, query_vector: List[float], n_results: int = 10) -> List[Dict[str, Any]]:
        results = self.vector_collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
        )

        if not results.get("ids") or not results["ids"][0]:
            return []

        found_objects: List[Dict[str, Any]] = []

        for i, crop_id in enumerate(results["ids"][0]):
            score = results["distances"][0][i]

            meta_data: Optional[Dict[str, Any]] = None
            if self.use_mongo and self.mongo_coll is not None:
                meta_data = self.mongo_coll.find_one({"_id": crop_id})
            else:
                meta_data = next(
                    (item for item in self.local_meta if item["_id"] == crop_id), None
                )

            if meta_data:
                found_objects.append({"score": score, "data": meta_data})

        return found_objects


# Singleton-style accessor
_datastore_instance: DataStore | None = None

def get_datastore() -> DataStore:
    global _datastore_instance
    if _datastore_instance is None:
        _datastore_instance = DataStore(mongo_uri=settings.MONGO_URI)
    return _datastore_instance
