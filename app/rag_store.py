import math
import ollama


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class EmbeddingRAGStore:
    def __init__(self, embedding_model="nomic-embed-text"):
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = []

    def embed_text(self, text):
        response = ollama.embeddings(
            model=self.embedding_model,
            prompt=text
        )
        return response["embedding"]

    def load_documents(self, docs):
        self.documents = docs
        self.embeddings = [self.embed_text(doc["content"]) for doc in docs]

    def search(self, query, top_k=3):
        query_embedding = self.embed_text(query)
        results = []

        for doc, emb in zip(self.documents, self.embeddings):
            score = cosine_similarity(query_embedding, emb)
            results.append({
                "doc_id": doc["id"],
                "content": doc["content"],
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]