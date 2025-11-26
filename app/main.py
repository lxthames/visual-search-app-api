from fastapi import FastAPI

from .core.config import settings
from .api.routes import router as api_router

# Create FastAPI app
app = FastAPI(title=settings.PROJECT_NAME)

# API routes (your real functionality)
app.include_router(api_router, prefix="/api")


# Optional: simple root endpoint (no frontend, just JSON)
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Visual Semantic API is running. See /docs for API docs."}
