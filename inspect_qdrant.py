from backend.retrieval.dense_retriever import get_qdrant_client
from backend.config import get_settings

client = get_qdrant_client()
settings = get_settings()

print("Collection:", settings.qdrant_collection_name)

points, _ = client.scroll(
    collection_name=settings.qdrant_collection_name,
    limit=20,
    with_payload=True
)

for i, p in enumerate(points, 1):
    print(f"\n--- {i} ---")
    print("ID:", p.id)
    print("TITLE:", p.payload.get("title"))
    print("SOURCE:", p.payload.get("source")) 