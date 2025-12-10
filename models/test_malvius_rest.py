# file: ping_malvius.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Base URL, e.g. http://devenv.catalist-me.com:8000
BASE_URL = (os.getenv("MALVIUS_URL") or "http://devenv.catalist-me.com:8000").rstrip("/")

# Auth: username:password -> Bearer token
USER = os.getenv("MALVIUS_USERNAME", "cstoreadmin")
PWD = os.getenv("MALVIUS_PASSWORD", "admin_cstore123")
TOKEN = f"{USER}:{PWD}"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "accept": "application/json",
    "Content-Type": "application/json",
}

url = f"{BASE_URL}/v2/vectordb/collections/list"
body = {"dbName": "_default"}  # default Milvus DB name

print("Pinging:", url)

try:
    resp = requests.post(url, headers=headers, json=body, timeout=5)
    print("HTTP status:", resp.status_code)
    print("Raw body:", resp.text)

    if resp.ok:
        print("✅ REST endpoint reachable")
    else:
        print("⚠️ REST endpoint responded but not OK (auth / path / config issue?)")

except requests.RequestException as e:
    print("❌ Could not connect to REST endpoint:", e)
