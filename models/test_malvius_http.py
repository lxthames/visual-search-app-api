import os
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

load_dotenv()

# Prefer MALVIUS_URL, fallback to MILVUS_URI if that's what's set
base_url = os.getenv("MALVIUS_URL") or os.getenv("MILVUS_URI")

if not base_url:
    raise RuntimeError(
        "MALVIUS_URL or MILVUS_URI is not set.\n"
        "Add MALVIUS_URL=http://devenv.catalist-me.com:8000 to your .env file."
    )

if not base_url.startswith("http://") and not base_url.startswith("https://"):
    base_url = "http://" + base_url

base_url = base_url.rstrip("/") + "/"

username = os.getenv("MALVIUS_USERNAME")
password = os.getenv("MALVIUS_PASSWORD")
token = os.getenv("MALVIUS_TOKEN")

headers = {"User-Agent": "malvius-http-probe/0.1"}
auth = None

if token:
    # If your service expects a bearer-like token, adjust as needed
    headers["Authorization"] = f"Bearer {token}"
elif username and password:
    auth = HTTPBasicAuth(username, password)

paths = [
    "",                 # base /
    "/",                # base /
    "health",
    "api/health",
    "status",
    "api/status",
    "docs",
    "redoc",
    "openapi.json",
    "swagger.json",
    "v1/vector/collections",
    "v1/vector/collections/describe",
]

def probe(path: str) -> None:
    url = urljoin(base_url, path)
    print(f"\n=== GET {url} ===")
    try:
        resp = requests.get(url, headers=headers, auth=auth, timeout=5)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return

    print(f"Status: {resp.status_code}")
    ct = resp.headers.get("Content-Type", "<none>")
    print(f"Content-Type: {ct}")

    # Show small preview of the body (first 300 chars)
    text = resp.text
    preview = text[:300].replace("\n", "\\n")
    print(f"Body preview (first 300 chars): {preview!r}")

def main():
    print(f"Base URL: {base_url}")
    print(f"Using auth: {'token' if token else 'basic' if username and password else 'none'}")

    for p in paths:
        probe(p)

    print("\nDone. Look at which endpoints returned clean HTTP responses.")
    print("If you see normal HTML/JSON responses, this is an HTTP service, not Milvus gRPC.")


if __name__ == "__main__":
    main()
