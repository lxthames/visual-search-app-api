from pymilvus import MilvusClient
import os
from dotenv import load_dotenv

# Load .env from current directory
load_dotenv()

uri = os.getenv("MILVUS_URI")
token = os.getenv("MILVUS_TOKEN")
db_name = os.getenv("MILVUS_DB_NAME", "default")
collection_name = os.getenv("MILVUS_COLLECTION_NAME", "visual_search")

print("MILVUS_URI =", uri)
print("MILVUS_TOKEN =", token)
print("MILVUS_DB_NAME =", db_name)
print("MILVUS_COLLECTION_NAME =", collection_name)

if not uri:
    raise RuntimeError(
        "MILVUS_URI is not set. "
        "Add MILVUS_URI=http://devenv.catalist-me.com:8000 to your .env file."
    )

client_args = {"uri": uri, "db_name": db_name}
if token:
    client_args["token"] = token

print("Connecting with:", client_args)

client = MilvusClient(**client_args)
print("✅ Connected to Milvus.")

print("Existing collections:", client.list_collections())

if not client.has_collection(collection_name=collection_name):
    print(f"Collection '{collection_name}' does NOT exist.")
else:
    desc = client.describe_collection(collection_name=collection_name)
    print("Collection description:", desc)
