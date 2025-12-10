from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    PROJECT_NAME: str = "Visual Search API"

    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", str(DATA_DIR / "visual_db"))
    METADATA_PATH: str = os.getenv("METADATA_PATH", str(DATA_DIR / "visual_metadata.json"))

    # --- MongoDB (metadata) ---
    MONGO_URI: str | None = os.getenv("MONGO_URI") or None
    # Let .env decide; if missing, we won't hardcode DB/collection in code.
    MONGO_DB_NAME: str | None = os.getenv("MONGO_DB_NAME") or None
    MONGO_COLLECTION_NAME: str | None = os.getenv("MONGO_COLLECTION_NAME") or None

    # --- Vector backend selection ---
    # "chroma" for local dev, "milvus" to use PyMilvus
    VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "chroma")

    # --- Milvus / "Malvius" config ---
    # Optional HTTP gateway info (not used by PyMilvus, but kept for completeness)
    MILVUS_URI: str | None = os.getenv("MILVUS_URI")      # e.g. "http://devenv.catalist-me.com:8000"
    MILVUS_TOKEN: str | None = os.getenv("MILVUS_TOKEN")  # e.g. "user:password" for HTTP gateway

    # gRPC connection details used by PyMilvus
    # These are read directly from .env; no app-specific hardcoding.
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: str = os.getenv("MILVUS_PORT", "19530")
    MILVUS_USERNAME: str | None = os.getenv("MILVUS_USERNAME") or None
    MILVUS_PASSWORD: str | None = os.getenv("MILVUS_PASSWORD") or None
    MILVUS_DB_NAME: str = os.getenv("MILVUS_DB_NAME", "default")
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "visual_search")

    DEVICE: str = os.getenv("DEVICE", "auto")

    SHELF_DIR: Path = DATA_DIR / "shelves"
    SHELF_STATE_PATH: Path = DATA_DIR / "shelf_state.json"


settings = Settings()
