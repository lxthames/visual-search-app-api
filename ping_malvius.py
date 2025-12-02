# file: ping_malvius.py
from pymilvus import connections, utility

# ======== CONFIG (from your message) ========
HOST = "devenv.catalist-me.com"
PORT = "19538"
USERNAME = "root"
PASSWORD = "cstore_admin123"
DB_NAME = "default"   # change if your Milvus DB name is different
SECURE = False        # set True only if server uses TLS
# ===========================================

print("Connecting to Milvus...")
print(f"  Host: {HOST}")
print(f"  Port: {PORT}")
print(f"  DB:   {DB_NAME}")
print(f"  Username: {USERNAME}")

try:
    connections.connect(
        alias="default",
        host=HOST,
        port=PORT,
        user=USERNAME,
        password=PASSWORD,
        db_name=DB_NAME,
        secure=SECURE,
    )

    print("✅ Connected to Milvus")

    collections = utility.list_collections()
    print("📌 Collections:", collections)

except Exception as e:
    msg = str(e)
    print("❌ Failed to connect:", e)

    if "UNAUTHENTICATED" in msg or "auth check failure" in msg:
        print("\n🔐 Authentication failed.")
        print("   → Double-check that USERNAME and PASSWORD match the Milvus server config.")
    else:
        print("\n⚠️ Non-auth error. Check host/port/network or Milvus status.")
