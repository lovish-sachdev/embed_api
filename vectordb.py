
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from onnx_infer import get_embedding, cosine_similarity
from qdrant_client.http.models import VectorParams, Distance

# Qdrant setup
qdrant_client = QdrantClient(
    url="https://5831b7fa-b26b-4363-aa04-73a059239151.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7C5ShfjTKWCYMeN4NjtepkbBi4JJ_Q2oNZNjTFjncvs",
)

collection_name = "lavi"  # your existing collection name

def create_collection_if_not_exists(collection_name, vector_dim=384):
    try:
        qdrant_client.get_collection(collection_name)
        print("✅ Collection already exists.")
    except:
        print("⚠️ Collection does not exist. Creating it...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        print("✅ Collection created.")

create_collection_if_not_exists(collection_name)

def upload_text_embedding(audio_id: str, text: str):
    """
    Uploads embedding for the given text with the given ID to Qdrant.
    """
    embedding = get_embedding(text)  # <- Use your existing function
    point = PointStruct(
        id=audio_id,
        vector=embedding
    )
    qdrant_client.upsert(collection_name=collection_name, points=[point])
    print(f"✅ Uploaded embedding for ID: {audio_id}")

def search_similar_ids(query: str, top_k: int = 5) -> list:
    """
    Searches Qdrant for similar vectors and returns only their IDs.
    """
    query_vector = get_embedding(query)

    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    return [point.id for point in search_result]

