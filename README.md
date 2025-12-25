Visual Semantic API

This project is a **backend-only FastAPI application** that exposes a set of API endpoints for interacting with a visual / semantic processing pipeline (e.g. indexing data and running searches).  
There is **no frontend UI** – you interact with the app through HTTP endpoints using tools like **Insomnia**, **Postman**, or your own client code.



Features

- ✅ Built with **FastAPI**
- ✅ Pure **API backend** (no HTML or static frontend)
- ✅ Automatic interactive docs via **OpenAPI** (`/docs`)
- ✅ Ready to be consumed by any frontend, script, or external service


Requirements

- Python `3.10+` (adjust if your project uses a different version)
- `pip` or `uv` / `poetry` / `pipenv` (depending on your setup)

Common Python dependencies (see `requirements.txt` in the repo for the full list):

- `fastapi`
- `uvicorn`
- Other libraries specific to your visual/semantic logic (embeddings, models, etc.)



 Installation

1. Clone the repository

   ```bash
   git clone repo.git
   cd project folder
   ````

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate     # on macOS / Linux
   # .venv\Scripts\activate      # on Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables** (if needed)

   If the app uses a `.env` file or config settings (e.g. API keys, model paths, DB URLs), create it:

   ```bash
   cp .env.example .env   # if an example file exists
   ```

   Then edit `.env` and fill in your values.

   ### Test Collections Configuration

   The application supports separate test collections for MongoDB and Milvus to allow testing without affecting production data.

   **To use test collections**, add these variables to your `.env` file:

   ```bash
   # MongoDB Test Collections
   MONGO_COLLECTION_NAME_TEST=products_test
   MONGO_SINGLE_QUERY_COLLECTION_NAME_TEST=single_query_images_test

   # Milvus Test Collection
   MILVUS_COLLECTION_NAME_TEST=visual_search_test
   ```

   **To use production collections**, simply don't set the `_TEST` variables (or comment them out).

   The application will automatically:
   - Use test collections if `_TEST` variables are set
   - Use production collections if `_TEST` variables are not set
   - Print which collections are being used at startup

---

## Running the Application

Run the FastAPI app with a simple Python command:

```bash
python run.py
```

This will start the server with:
* Auto-reload enabled for development
* Host: `0.0.0.0` (accessible from all network interfaces)
* Port: `8000`
* The API will be available at: `http://localhost:8000`

**Alternative:** You can also run it directly with uvicorn if needed:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API Documentation

FastAPI automatically provides interactive documentation:

* Swagger UI: `http://localhost:8000/docs`
* OpenAPI JSON: `http://localhost:8000/openapi.json`

Use these pages to:

* Explore available endpoints
* Inspect request/response models
* Try endpoints directly from the browser

> ⚠️ If the project disables docs (`docs_url=None` / `redoc_url=None`), only the raw endpoints will be available. In that case, inspect the code or OpenAPI JSON to see the schema.

---

## Basic Usage

The API is namespaced under `/api`. Typical pattern:

* Base URL: `http://localhost:8000/api/...`

Example (adjust to match your actual routes):

* `GET  /api/health` – health check
* `POST /api/search` – semantic/visual search with a JSON query
* `POST /api/search-visual` – visual search using an uploaded image

You can call these endpoints using:

* **Insomnia / Postman** – create requests and inspect responses

* **curl** – e.g.:

  ```bash
  curl -X GET http://localhost:8000/api/health
  ```

* Any custom frontend or script – using `fetch`, `axios`, `requests`, etc.

---

## Development Notes

* The main entrypoint is: `app/main.py`
* API routes are organized using FastAPI **routers** (e.g. under `app/api/`)
* There is **no bundled frontend**; this service is intended to be used as an API by other applications

---

## License

 MIT 
