import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

load_dotenv()


def answer_agent(user_query: str, compressed_context: str, docs: List[Document]) -> str:
    """
    Generates final answer with citations.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )

    citations = []
    for i, d in enumerate(docs, start=1):
        handbook = d.metadata.get("handbook_name", "unknown")
        page = d.metadata.get("page", "N/A")
        chunk = d.metadata.get("chunk_id", "N/A")
        citations.append(f"[{i}] {handbook} (page {page}, chunk {chunk})")

    citations_text = "\n".join(citations)

    prompt = f"""
You are an enterprise handbook assistant.

Answer the question ONLY using the provided context.
If the context does not contain the answer, say: "Not found in handbook documents."

User question:
{user_query}

Compressed Context:
{compressed_context}

Citations available:
{citations_text}

Rules:
- Use bullet points if possible
- Be precise
- End your answer with a "Sources:" section listing citations you used.

Now write the final answer.
"""

    return llm.invoke(prompt).content.strip()