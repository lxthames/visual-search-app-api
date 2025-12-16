from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# --- Make project root importable when run as a script ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings  # adjust import path as needed
from pymongo import MongoClient
from pymongo.database import Database
from pymilvus import connections, utility


# -------------------------
# MongoDB helpers
# -------------------------

def _get_mongo_db() -> Database:
    """
    Return the configured MongoDB database.

    Raises a RuntimeError if required settings are missing.
    """
    if not settings.MONGO_URI:
        raise RuntimeError("MONGO_URI is not configured in .env")

    if not settings.MONGO_DB_NAME:
        raise RuntimeError("MONGO_DB_NAME is not configured in .env")

    client = MongoClient(settings.MONGO_URI)
    return client[settings.MONGO_DB_NAME]


def drop_mongo_collection(collection_name: str) -> bool:
    """
    Drop a MongoDB collection by name.

    Returns:
        True if the collection existed and was dropped, False if it did not exist.
    """
    db = _get_mongo_db()
    try:
        existing = set(db.list_collection_names())
        if collection_name not in existing:
            return False
        db.drop_collection(collection_name)
        return True
    finally:
        # Close the underlying client connection
        db.client.close()


# -------------------------
# Milvus helpers
# -------------------------

def _connect_milvus() -> None:
    """
    Establish a connection to Milvus using your settings.
    """
    conn_kwargs: Dict[str, Any] = {
        "alias": "default",
        "host": settings.MILVUS_HOST,
        "port": settings.MILVUS_PORT,
    }

    if getattr(settings, "MILVUS_USERNAME", None) and getattr(settings, "MILVUS_PASSWORD", None):
        conn_kwargs["user"] = settings.MILVUS_USERNAME
        conn_kwargs["password"] = settings.MILVUS_PASSWORD

    if getattr(settings, "MILVUS_DB_NAME", None):
        conn_kwargs["db_name"] = settings.MILVUS_DB_NAME

    connections.connect(**conn_kwargs)


def drop_milvus_collection_by_name(collection_name: str) -> bool:
    """
    Drop a Milvus collection by name.

    Returns:
        True if a collection was found and dropped, False otherwise.
    """
    _connect_milvus()

    if not utility.has_collection(collection_name):
        return False

    utility.drop_collection(collection_name)
    return True


# -------------------------
# Orchestrator
# -------------------------

def reset_test_data(dev_only: bool = True) -> Dict[str, Any]:
    """
    Drops the configured *test* collections:
      - Mongo: MONGO_COLLECTION_NAME_TEST
      - Mongo: MONGO_SINGLE_QUERY_COLLECTION_NAME_TEST
      - Milvus: MILVUS_COLLECTION_NAME_TEST

    Uses environment variables directly so you don't need to modify settings.py.
    """
    if dev_only is not True:
        raise RuntimeError("reset_test_data() is intended for dev/test use only")

    summary: Dict[str, Any] = {}

    mongo_products_test = os.getenv("MONGO_COLLECTION_NAME_TEST", "products_test")
    mongo_single_query_test = os.getenv(
        "MONGO_SINGLE_QUERY_COLLECTION_NAME_TEST",
        "single_query_images_test",
    )
    milvus_visual_search_test = os.getenv("MILVUS_COLLECTION_NAME_TEST", "visual_search_test")

    # 1) MongoDB: drop test collections
    try:
        summary["mongo_collection_test"] = mongo_products_test
        summary["mongo_collection_test_dropped"] = drop_mongo_collection(mongo_products_test)
    except Exception as exc:
        summary["mongo_collection_test_error"] = str(exc)

    try:
        summary["mongo_single_query_collection_test"] = mongo_single_query_test
        summary["mongo_single_query_collection_test_dropped"] = drop_mongo_collection(mongo_single_query_test)
    except Exception as exc:
        summary["mongo_single_query_collection_test_error"] = str(exc)

    # 2) Milvus: drop test collection
    try:
        summary["milvus_collection_test"] = milvus_visual_search_test
        summary["milvus_collection_test_dropped"] = drop_milvus_collection_by_name(milvus_visual_search_test)
    except Exception as exc:
        summary["milvus_collection_test_error"] = str(exc)

    return summary


if __name__ == "__main__":
    # Run:
    #   python -m app.maintenance.reset_test_data
    result = reset_test_data(dev_only=True)
    print("Test reset summary:", result)
