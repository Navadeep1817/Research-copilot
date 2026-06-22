from qdrant_client import QdrantClient

qdrant_client = QdrantClient(host="localhost", port=6333)

points = qdrant_client.scroll(
    collection_name="research_docs",
    limit=5,
    with_payload=True
)

print(points) 