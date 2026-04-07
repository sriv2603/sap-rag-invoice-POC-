from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class InMemoryVectorStore:
    def __init__(self):
        self.documents = []
        self.texts = []
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None

    def add_documents(self, documents):
        self.documents = documents
        self.texts = [doc["text"] for doc in documents]
        self.document_vectors = self.vectorizer.fit_transform(self.texts)

    def search(self, query, top_k=3):
        if not self.documents or self.document_vectors is None:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()

        ranked_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(similarities[idx])
            })

        return results