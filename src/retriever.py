from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid


class Retriever:
    def __init__(self, collection_name: str = "docs", embed_model: str = "BAAI/bge-small-en-v1.5"):
        self.client = QdrantClient(":memory:")
        self.model = SentenceTransformer(embed_model)
        self.collection_name = collection_name
        self.vector_size = self.model.get_sentence_embedding_dimension()

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def add_documents(self, documents: list[str]):
        embeddings = self.model.encode(documents, show_progress_bar=True)
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=emb.tolist(), payload={"text": doc})
            for doc, emb in zip(documents, embeddings)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5):
        query_vector = self.model.encode(query).tolist()
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return [{"text": r.payload["text"], "score": r.score} for r in results]


if __name__ == "__main__":
    retriever = Retriever()
    retriever.add_documents([
        "Paris is the capital of France.",
        "The Eiffel Tower was completed in 1889.",
        "France's population is about 68 million.",
    ])
    results = retriever.search("What is the capital of France?")
    for r in results:
        print(r)