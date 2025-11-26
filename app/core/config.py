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

    MONGO_URI: str | None = os.getenv("MONGO_URI") or None
    DEVICE: str = os.getenv("DEVICE", "auto")

    # Shelf persistence
    SHELF_DIR: Path = DATA_DIR / "shelves"
    SHELF_STATE_PATH: Path = DATA_DIR / "shelf_state.json"  # (not strictly needed anymore, but ok to keep)



settings = Settings()
