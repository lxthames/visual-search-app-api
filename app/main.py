# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse  # ← add this

from .core.config import settings
from .api.routes import router as api_router

app = FastAPI(title=settings.PROJECT_NAME)

# API routes – all your routes are under /api (e.g. /api/health)
app.include_router(api_router, prefix="/api")

# Static files – root-level "static" folder
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def root():
    # Serve your frontend HTML
    return FileResponse("static/index.html")
    # If you prefer keeping the JSON at `/`, you could instead expose
    # the HTML at `/ui` and keep the original root():
    # return {"message": "Visual Semantic API is running. See /docs for API docs."}
