# SAP RAG Invoice POC

This project demonstrates a simple Retrieval-Augmented Generation (RAG) approach applied to SAP-like invoice data.

## What it does

- Accepts natural language questions
- Searches invoice data
- Returns relevant results

## Concept

This is a basic simulation of how AI can be layered on top of SAP systems using:

- OData (simulated)
- Context retrieval
- API-based interaction

## Note

This is a simple keyword-based retrieval system and does not yet use an LLM for reasoning.

## Future Scope

- Integrate with SAP OData services
- Add LLM (OpenAI / others) for intelligent responses
- Improve context understanding

## Run the app

```bash
uvicorn app.main:app --reload