from fastapi import FastAPI
from pydantic import BaseModel

from app.embed_store import InMemoryVectorStore
from app.data import documents

app = FastAPI(title="SAP RAG Invoice POC")

vector_store = InMemoryVectorStore()
vector_store.add_documents(documents)


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "SAP RAG Invoice POC is running"}


@app.post("/search")
def search_docs(request: QueryRequest):
    results = vector_store.search(request.question, top_k=5)

    if not results:
        return {
            "question": request.question,
            "answer": "No relevant result found.",
            "results": []
        }

    filtered_results = results

    if "blocked" in request.question.lower():
        filtered_results = [
            r for r in results
            if "blocked" in r["document"]["text"].lower()
        ]

    if not filtered_results:
        filtered_results = results

    top_result = filtered_results[0]

    return {
        "question": request.question,
        "answer": top_result["document"]["text"],
        "score": top_result["score"],
        "results": filtered_results
    }
