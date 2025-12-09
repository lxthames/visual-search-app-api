# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .core.config import settings
from .api.routes import router as api_router

app = FastAPI(title=settings.PROJECT_NAME)

# API routes
app.include_router(api_router, prefix="/api")

# Static files – NOTE: directory="static" (root-level folder)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Visual Semantic API is running. See /docs for API docs."}
