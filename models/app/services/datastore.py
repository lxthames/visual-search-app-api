from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import pymongo
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

from app.core.config import settings


class DataStore:
    def __init__(self, mongo_uri: Optional[str] = None) -> None:
        # Which vector backend to use: 'chroma' or 'milvus'
        # Use exactly what comes from .env via settings, no inline default
        self.vector_backend = settings.VECTOR_BACKEND.lower()

        # ---------- Vector DB ----------
        if self.vector_backend == "milvus":
            self._init_milvus_vector_store()
        else:
            self._init_chroma_vector_store()

        # ---------- Metadata DB (Mongo or JSON) ----------
        self.use_mongo = False
        self.mongo_coll = None

        self.local_meta: List[Dict[str, Any]] = []
        self.local_meta_path = Path(settings.METADATA_PATH)
        self.local_meta_path.parent.mkdir(parents=True, exist_ok=True)
        # Thread lock for thread-safe local metadata operations
        self._local_meta_lock = threading.Lock()

        if mongo_uri:
            try:
                mongo_client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
                mongo_client.server_info()

                # DB name and collection name come from .env via settings
                if settings.MONGO_DB_NAME:
                    mongo_db = mongo_client[settings.MONGO_DB_NAME]
                else:
                    # Try default DB from URI, then fall back to a generic name
                    try:
                        mongo_db = mongo_client.get_default_database()
                    except Exception:
                        mongo_db = mongo_client["cstore-ai"]

                coll_name = (
                    settings.MONGO_COLLECTION_NAME
                    if getattr(settings, "MONGO_COLLECTION_NAME", None)
                    else "products"
                )
                self.mongo_coll = mongo_db[coll_name]
                self.use_mongo = True
            except Exception:
                self.use_mongo = False

        if not self.use_mongo:
            if self.local_meta_path.exists():
                with self.local_meta_path.open("r", encoding="utf-8") as f:
                    self.local_meta = json.load(f)

    # ---------- Vector store init helpers ----------

    def _init_chroma_vector_store(self) -> None:
        """Initialize local Chroma-based vector store."""
        chroma_path = Path(settings.CHROMA_PATH)
        chroma_path.parent.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        self.vector_collection = self.chroma_client.get_or_create_collection(
            name="product_vectors"
        )

    def _init_milvus_vector_store(self) -> None:
        """Initialize Milvus-based vector store (creates collection if missing)."""
        host = settings.MILVUS_HOST
        port = settings.MILVUS_PORT
        username = settings.MILVUS_USERNAME
        password = settings.MILVUS_PASSWORD
        db_name = settings.MILVUS_DB_NAME
        collection_name = settings.MILVUS_COLLECTION_NAME

        # Connect (alias 'default' reused by whole app)
        connections.connect(
            alias="default",
            host=host,
            port=port,
            user=username,
            password=password,
            db_name=db_name,
            secure=False,
        )

        # CLIP ViT-B/32 typically => 512-dim; keep as a code-level constant
        dim = 512

        if not utility.has_collection(collection_name):
            crop_id_field = FieldSchema(
                name="crop_id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=64,
            )
            embedding_field = FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=dim,
            )
            schema = CollectionSchema(
                fields=[crop_id_field, embedding_field],
                description="Visual search embeddings",
            )
            self.milvus_collection = Collection(
                name=collection_name,
                schema=schema,
                using="default",
                shards_num=2,
            )
        else:
            self.milvus_collection = Collection(collection_name)

    # ---------- Shelf helpers ----------

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
        # 1. Save vector
        if self.vector_backend == "milvus":
            # Order of fields must match schema: [crop_id, embedding]
            data = [
                [crop_id],
                [vector],
            ]
            self.milvus_collection.insert(data)
            # Don't flush here - flush only in batch operations
        else:
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
            # Thread-safe append and save for local JSON storage
            with self._local_meta_lock:
                self.local_meta.append(record)
                self._save_local_meta()

    def batch_save_objects(
        self,
        image_id: str,
        objects: List[Dict[str, Any]],  # List of {crop_id, vector, metadata}
    ) -> None:
        """
        Batch save multiple objects to database. Much faster than individual saves.
        objects: List of dicts with keys: crop_id, vector, metadata
        """
        if not objects:
            return

        # 1. Batch save vectors
        if self.vector_backend == "milvus":
            # Prepare batch data for Milvus
            crop_ids = [obj["crop_id"] for obj in objects]
            vectors = [obj["vector"] for obj in objects]
            
            data = [
                crop_ids,
                vectors,
            ]
            self.milvus_collection.insert(data)
            # Flush once at the end for the entire batch
            self.milvus_collection.flush()
        else:
            # Batch add to Chroma
            crop_ids = [obj["crop_id"] for obj in objects]
            vectors = [obj["vector"] for obj in objects]
            metadatas = [{"mongo_id": crop_id} for crop_id in crop_ids]
            
            self.vector_collection.add(
                embeddings=vectors,
                metadatas=metadatas,
                ids=crop_ids,
            )

        # 2. Prepare metadata records
        records = []
        for obj in objects:
            record = {
                "_id": obj["crop_id"],
                "parent_image_id": image_id,
                "label": str(obj["metadata"].get("label", "")),
                "confidence": obj["metadata"].get("confidence"),
                "bbox": obj["metadata"].get("bbox"),
                "timestamp": datetime.now().isoformat(),
            }
            records.append(record)

        # 3. Batch save metadata
        if self.use_mongo and self.mongo_coll is not None:
            # Batch insert to MongoDB
            self.mongo_coll.insert_many(records)
        else:
            # Thread-safe batch append for local JSON storage
            with self._local_meta_lock:
                self.local_meta.extend(records)
                self._save_local_meta()

    def query_similar(self, query_vector: List[float], n_results: int = 10) -> List[Dict[str, Any]]:
        if self.vector_backend == "milvus":
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            search_results = self.milvus_collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=n_results,
                output_fields=["crop_id"],
            )

            if not search_results or not search_results[0]:
                return []

            hits = search_results[0]
            crop_ids = [hit.id for hit in hits]
            distances = [hit.distance for hit in hits]
        else:
            results = self.vector_collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
            )

            if not results.get("ids") or not results["ids"][0]:
                return []

            crop_ids = results["ids"][0]
            distances = results["distances"][0]

        found_objects: List[Dict[str, Any]] = []

        for i, crop_id in enumerate(crop_ids):
            score = distances[i]

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
