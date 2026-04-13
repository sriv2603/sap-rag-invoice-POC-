import ollama


class LLMClient:
    def __init__(self, model="llama3.2:1b"):
        self.model = model

    def answer_question(self, question, results):
        if not results:
            return "I could not find any relevant invoice context."

        context = "\n\n".join(
            [f"{r['doc_id']}: {r['content']}" for r in results]
        )

        prompt = f"""
You are an SAP finance assistant.

Use ONLY the provided context.
Do NOT add facts that are not explicitly present in the context.
Do NOT assume business steps unless the context clearly says them.
If the context is limited, say "Based on the available context..."
Keep the answer concise and factual.
Mention invoice IDs when relevant.

Question:
{question}

Context:
{context}
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt.strip()}
            ]
        )

        return response["message"]["content"]