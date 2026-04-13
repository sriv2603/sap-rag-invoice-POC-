from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.data import DOCUMENTS
from app.rag_store import EmbeddingRAGStore
from app.llm_client import LLMClient

app = FastAPI(
    title="SAP Invoice AI POC",
    description="Embedding-based retrieval + local LLM explanation over SAP-like invoice data",
    version="2.0.0"
)

rag_store = EmbeddingRAGStore("nomic-embed-text")
llm_client = LLMClient("llama3.2:1b")


class SearchRequest(BaseModel):
    question: str = Field(..., examples=["Why is invoice 1001 blocked?"])
    top_k: int = Field(default=3, ge=1, le=5)


@app.on_event("startup")
def startup_event():
    rag_store.load_documents(DOCUMENTS)


@app.get("/")
def root():
    return {
        "message": "SAP Invoice AI POC is running",
        "docs": "/docs"
    }


@app.post("/search")
def search(payload: SearchRequest):
    question = payload.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    results = rag_store.search(question, payload.top_k)
    answer = llm_client.answer_question(question, results)

    return {
        "question": question,
        "answer": answer,
        "retrieved_docs": results
    }