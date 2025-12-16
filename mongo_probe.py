"""
Mongo connectivity + inventory probe (includes products collections).

Reads from env (preferred):
  MONGO_URI
  MONGO_DB_NAME
  MONGO_SINGLE_QUERY_COLLECTION_NAME
  MONGO_SINGLE_QUERY_COLLECTION_NAME_TEST
  MONGO_COLLECTION_NAME
  MONGO_COLLECTION_NAME_TEST

Fallback defaults (edit if needed):
  MONGO_URI=mongodb://cstore-ai:...@devenv.catalist-me.com:27017/?authSource=cstore-ai
  MONGO_DB_NAME=cstore-ai
  MONGO_COLLECTION_NAME=products
  MONGO_COLLECTION_NAME_TEST=products_test

Run:
  python mongo_probe.py
"""

from __future__ import annotations

import os
import sys
import pprint
from urllib.parse import urlsplit, urlunsplit

import pymongo
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError


DEFAULT_MONGO_URI = "mongodb://cstore-ai:xY2BmwXTXPNh87DK@devenv.catalist-me.com:27017/?authSource=cstore-ai"
DEFAULT_DB_NAME = "cstore-ai"

DEFAULT_PRODUCTS_COLL = "products"
DEFAULT_PRODUCTS_COLL_TEST = "products_test"


def redact_mongo_uri(uri: str) -> str:
    """Redact password in mongodb/mongodb+srv URI for safe printing."""
    try:
        parts = urlsplit(uri)
        netloc = parts.netloc
        if "@" in netloc and ":" in netloc.split("@", 1)[0]:
            userinfo, hostinfo = netloc.split("@", 1)
            user, _pwd = userinfo.split(":", 1)
            netloc = f"{user}:***@{hostinfo}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return "<unparseable uri>"


def env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def safe_print(title: str, value):
    print(f"\n=== {title} ===")
    if isinstance(value, (dict, list, tuple)):
        pprint.pprint(value, width=120, compact=True)
    else:
        print(value)


def probe_collection(db, name: str, known_collections: list[str] | None) -> dict:
    """
    Probe a collection:
      - whether it exists in collection list (if available)
      - estimated count
      - sample _id from find_one
    """
    out: dict = {"name": name}

    if known_collections is not None:
        out["exists_in_list"] = name in known_collections
    else:
        out["exists_in_list"] = None

    try:
        coll = db[name]
        out["estimated_document_count"] = coll.estimated_document_count()
    except PyMongoError as e:
        out["estimated_document_count_error"] = f"{type(e).__name__}: {e}"

    try:
        doc = db[name].find_one({}, projection={"_id": 1})
        out["sample_find_one"] = "OK" if doc is not None else "EMPTY_OR_NO_ACCESS"
    except PyMongoError as e:
        out["sample_find_one_error"] = f"{type(e).__name__}: {e}"

    return out


def main() -> int:
    mongo_uri = env("MONGO_URI", DEFAULT_MONGO_URI)
    db_name = env("MONGO_DB_NAME", DEFAULT_DB_NAME)

    # Optional envs from your app
    single_query_coll = env("MONGO_SINGLE_QUERY_COLLECTION_NAME", None)
    single_query_coll_test = env("MONGO_SINGLE_QUERY_COLLECTION_NAME_TEST", None)

    # Products collections requested by you
    products_coll = env("MONGO_COLLECTION_NAME", DEFAULT_PRODUCTS_COLL)
    products_coll_test = env("MONGO_COLLECTION_NAME_TEST", DEFAULT_PRODUCTS_COLL_TEST)

    safe_print("Environment (presence)", {
        "MONGO_URI_set": bool(os.environ.get("MONGO_URI")),
        "MONGO_DB_NAME_set": bool(os.environ.get("MONGO_DB_NAME")),
        "MONGO_SINGLE_QUERY_COLLECTION_NAME": single_query_coll,
        "MONGO_SINGLE_QUERY_COLLECTION_NAME_TEST": single_query_coll_test,
        "MONGO_COLLECTION_NAME": products_coll,
        "MONGO_COLLECTION_NAME_TEST": products_coll_test,
    })

    safe_print("MONGO_URI (redacted)", redact_mongo_uri(mongo_uri))
    safe_print("Target DB", db_name)

    # Client with tight timeouts
    try:
        client = pymongo.MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            socketTimeoutMS=5000,
            appname="mongo-probe",
        )
    except PyMongoError as e:
        safe_print("Client construction failed", f"{type(e).__name__}: {e}")
        return 2

    # Connectivity check
    try:
        pong = client.admin.command("ping")
        safe_print("Ping", pong)
    except ServerSelectionTimeoutError as e:
        safe_print("Ping failed (ServerSelectionTimeoutError)", str(e))
        safe_print(
            "Common causes",
            [
                "Host unreachable / DNS issue / firewall / VPN / container network",
                "Mongo not listening on that host:port",
                "Wrong host/port in URI",
                "Atlas: IP not allowlisted",
            ],
        )
        return 3
    except PyMongoError as e:
        safe_print("Ping failed", f"{type(e).__name__}: {e}")
        return 4

    # Server / topology basics
    try:
        safe_print("Server address (primary from client)", client.address)
        safe_print("Topology description", str(client.topology_description))
    except Exception as e:
        safe_print("Topology info error (non-fatal)", f"{type(e).__name__}: {e}")

    # Attempt hello/isMaster
    for cmd in ("hello", "isMaster"):
        try:
            res = client.admin.command(cmd)
            safe_print(
                f"{cmd}() (selected fields)",
                {k: res.get(k) for k in ("ok", "isWritablePrimary", "ismaster", "secondary", "setName", "hosts", "me", "primary", "msg", "version") if k in res},
            )
            break
        except PyMongoError:
            continue

    # Database handle
    try:
        db = client[db_name]
        safe_print("DB object created", f"client['{db_name}']")
    except Exception as e:
        safe_print("DB selection failed", f"{type(e).__name__}: {e}")
        return 5

    # List databases (may require privileges)
    try:
        dbs = client.list_database_names()
        safe_print("Databases visible to this user", dbs)
    except PyMongoError as e:
        safe_print("list_database_names() failed (permission?)", f"{type(e).__name__}: {e}")

    # List collections (may require privileges)
    colls: list[str] | None
    try:
        colls = db.list_collection_names()
        safe_print(f"Collections in '{db_name}'", colls)
    except PyMongoError as e:
        safe_print("list_collection_names() failed (permission?)", f"{type(e).__name__}: {e}")
        colls = None

    # Estimated counts for all collections (best-effort)
    if colls:
        stats = {}
        for c in colls:
            try:
                est = db[c].estimated_document_count()
                stats[c] = {"estimated_document_count": est}
            except PyMongoError as e:
                stats[c] = {"error": f"{type(e).__name__}: {e}"}
        safe_print("Collection counts (estimated)", stats)

    # Probe configured collections (single-query + products)
    candidates: list[str] = []
    if single_query_coll:
        candidates.append(single_query_coll)
    if single_query_coll_test:
        candidates.append(single_query_coll_test)
    if products_coll:
        candidates.append(products_coll)
    if products_coll_test:
        candidates.append(products_coll_test)

    # de-dupe while preserving order
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    probes = [probe_collection(db, name, colls) for name in candidates]
    safe_print("Configured collection probes", probes)

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
