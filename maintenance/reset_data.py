from __future__ import annotations

import sys
from pathlib import Path

# --- Make project root importable when run as a script ---
# This makes `import app.*` work even when you run:
#   python app/maintenance/reset_data.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now these imports will work
from app.core.config import settings
from pymongo import MongoClient
from pymilvus import connections, utility
import shutil
from typing import Any, Dict, Optional# adjust import path as needed


# ---------- MongoDB (metadata) ----------

def _get_mongo_collection() -> Collection:
    """
    Return the configured MongoDB collection for metadata.

    Raises a RuntimeError if required settings are missing.
    """
    if not settings.MONGO_URI:
        raise RuntimeError("MONGO_URI is not configured in .env")

    if not settings.MONGO_DB_NAME:
        raise RuntimeError("MONGO_DB_NAME is not configured in .env")

    if not settings.MONGO_COLLECTION_NAME:
        raise RuntimeError("MONGO_COLLECTION_NAME is not configured in .env")

    client = MongoClient(settings.MONGO_URI)
    db = client[settings.MONGO_DB_NAME]
    collection = db[settings.MONGO_COLLECTION_NAME]
    return collection


def clear_mongo_metadata() -> int:
    """
    Delete all documents from the configured MongoDB collection.

    Returns:
        Number of deleted documents.
    """
    collection = _get_mongo_collection()
    result = collection.delete_many({})
    return result.deleted_count


# ---------- Chroma (local vector DB) ----------

def reset_chroma_storage() -> None:
    """
    Hard-reset the local Chroma storage directory by deleting CHROMA_PATH.

    This is safe only if you are OK losing all local vector data.
    """
    chroma_dir = Path(settings.CHROMA_PATH)

    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    # Recreate empty directory so that Chroma can start clean
    chroma_dir.mkdir(parents=True, exist_ok=True)


# ---------- Milvus (vector DB) ----------

def _connect_milvus() -> None:
    """
    Establish a connection to Milvus using your settings.
    """
    conn_kwargs: Dict[str, Any] = {
        "alias": "default",
        "host": settings.MILVUS_HOST,
        "port": settings.MILVUS_PORT,
    }

    if settings.MILVUS_USERNAME and settings.MILVUS_PASSWORD:
        conn_kwargs["user"] = settings.MILVUS_USERNAME
        conn_kwargs["password"] = settings.MILVUS_PASSWORD

    # For Milvus 2.x, db_name can be passed in connect; if not used in your
    # deployment, this parameter is simply ignored.
    if settings.MILVUS_DB_NAME:
        conn_kwargs["db_name"] = settings.MILVUS_DB_NAME

    connections.connect(**conn_kwargs)


def drop_milvus_collection(collection_name: Optional[str] = None) -> bool:
    """
    Drop the specified Milvus collection (or the default one from settings).

    Returns:
        True if a collection was found and dropped, False otherwise.
    """
    _connect_milvus()

    coll_name = collection_name or settings.MILVUS_COLLECTION_NAME

    if not coll_name:
        raise RuntimeError("MILVUS_COLLECTION_NAME is not configured in .env")

    if not utility.has_collection(coll_name):
        # Nothing to drop
        return False

    utility.drop_collection(coll_name)
    return True


# ---------- Filesystem "shelf" state ----------

def reset_local_shelves() -> None:
    """
    Delete all files under SHELF_DIR and remove the shelf_state.json file.
    """
    # Remove shelf directory (if exists)
    if settings.SHELF_DIR.exists():
        shutil.rmtree(settings.SHELF_DIR)
    settings.SHELF_DIR.mkdir(parents=True, exist_ok=True)

    # Remove shelf state file
    if settings.SHELF_STATE_PATH.exists():
        settings.SHELF_STATE_PATH.unlink()


# ---------- High-level orchestrator ----------

def reset_all_data(dev_only: bool = True) -> Dict[str, Any]:
    """
    High-level reset function that:
        - clears MongoDB metadata
        - resets vector backend (Chroma or Milvus)
        - resets local shelves state

    Args:
        dev_only: for safety, allows you to explicitly indicate this is being
                  used in a development/test context.

    Returns:
        A dictionary summarizing what was done.
    """
    if dev_only is not True:
        # You can remove this guard if you *really* intend to call this in prod,
        # but it's strongly recommended to keep some safety checks.
        raise RuntimeError("reset_all_data() is intended for dev/test use only")

    summary: Dict[str, Any] = {}

    # 1) MongoDB
    try:
        deleted_docs = clear_mongo_metadata()
        summary["mongo_metadata_deleted"] = deleted_docs
    except Exception as exc:
        summary["mongo_error"] = str(exc)

    # 2) Vector backend
    if settings.VECTOR_BACKEND.lower() == "chroma":
        try:
            reset_chroma_storage()
            summary["vector_backend"] = "chroma"
            summary["chroma_reset"] = True
        except Exception as exc:
            summary["chroma_error"] = str(exc)

    elif settings.VECTOR_BACKEND.lower() == "milvus":
        try:
            dropped = drop_milvus_collection()
            summary["vector_backend"] = "milvus"
            summary["milvus_collection_dropped"] = dropped
        except Exception as exc:
            summary["milvus_error"] = str(exc)
    else:
        summary["vector_backend"] = settings.VECTOR_BACKEND
        summary["vector_backend_warning"] = "Unknown VECTOR_BACKEND; no vector store reset performed"

    # 3) Local shelves
    try:
        reset_local_shelves()
        summary["shelves_reset"] = True
    except Exception as exc:
        summary["shelves_error"] = str(exc)

    return summary


if __name__ == "__main__":
    # Allow you to run: `python -m app.maintenance.reset_data`
    result = reset_all_data(dev_only=True)
    print("Reset summary:", result)
