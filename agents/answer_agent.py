import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

load_dotenv()


def answer_agent(
    user_query: str,
    compressed_context: str,
    docs: List[Document],
    chat_history: Optional[list] = None
) -> str:

    # safety: if no docs or no context
    if not docs or not compressed_context.strip():
        return "Not found in handbook documents."

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )

    # primary handbook = handbook of first doc
    primary_handbook = docs[0].metadata.get("handbook_name", "unknown")

    # citations
    citations = []
    for i, d in enumerate(docs, start=1):
        handbook = d.metadata.get("handbook_name", "unknown")
        page = d.metadata.get("page", "N/A")
        chunk = d.metadata.get("chunk_id", "N/A")
        citations.append(f"[{i}] {handbook} (page {page}, chunk {chunk})")

    citations_text = "\n".join(citations)

    # conversation history (optional)
    history_text = ""
    if chat_history:
        for turn in chat_history[-6:]:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    prompt = f"""
You are an enterprise employee handbook assistant.

Selected Handbook (IMPORTANT):
{primary_handbook}

Conversation so far:
{history_text}

Task:
Answer the user's question ONLY using the provided context.

User question:
{user_query}

Compressed Context (ONLY from the selected handbook):
{compressed_context}

Citations available:
{citations_text}

Rules:
- Use bullet points if possible
- Be precise and policy-like
- Do not hallucinate or assume anything
- If the context does not contain the answer, reply exactly:
  Not found in handbook documents.
- End your answer with a "Sources:" section listing ONLY citations you actually used
- Do NOT mix policies from different handbooks

Now write the final answer.
"""

    return llm.invoke(prompt).content.strip()