from pymilvus import connections, Collection
from app.core.config import settings  # or manually set same envs

connections.connect(
    alias="default",
    host=settings.MILVUS_HOST,
    port=settings.MILVUS_PORT,
    user=settings.MILVUS_USERNAME,
    password=settings.MILVUS_PASSWORD,
    db_name=settings.MILVUS_DB_NAME,
)

c = Collection(settings.MILVUS_COLLECTION_NAME)
print("num_entities:", c.num_entities)
