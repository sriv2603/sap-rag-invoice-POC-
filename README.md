# SAP RAG Invoice POC

This project demonstrates how SAP-like enterprise data can be enhanced using AI techniques such as semantic search and LLM-based explanations.

It started as a simple keyword-based retrieval system and has now evolved into a more intelligent, context-aware prototype.

---

## What it does

- Accepts natural language questions
- Searches invoice data using semantic similarity
- Retrieves relevant results
- Generates context-aware explanations using a local LLM

---

## Concept

This project explores how AI can be layered on top of SAP systems:

- Structured enterprise data (simulated SAP invoices)
- Context retrieval (semantic search)
- API-based interaction (FastAPI)
- AI interpretation layer (LLM)

The goal is to move from:

> "Show me the data"  
to  
> "Help me understand the data"

---

## Architecture
User Question
↓
Embedding Model (nomic-embed-text)
↓
Semantic Search (Cosine Similarity)
↓
Retrieve Relevant Invoice Data
↓
LLM (llama3.2:1b via Ollama)
↓
Context-Aware Answer

---

## Tech Stack

- Python
- FastAPI
- Ollama (local AI models)
  - `nomic-embed-text` → embeddings
  - `llama3.2:1b` → response generation
- Cosine similarity (semantic search)

---

## Project Structure
app/
├── data.py # Mock SAP invoice data
├── rag_store.py # Embedding + retrieval logic
├── llm_client.py # LLM interaction layer
└── main.py # API layer

requirements.txt
README.md

---

## Run the App

### 1. Install dependencies

```bash
pip install -r requirements.txt
