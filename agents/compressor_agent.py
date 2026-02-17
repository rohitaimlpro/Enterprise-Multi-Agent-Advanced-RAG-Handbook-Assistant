import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

load_dotenv()


def compressor_agent(query: str, docs: List[Document]) -> str:
    """
    Compresses retrieved chunks into the most relevant lines only.
    Returns a single compressed context string.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0
    )

    joined = "\n\n".join(
        [f"[Source {i+1}] {d.page_content}" for i, d in enumerate(docs)]
    )

    prompt = f"""
You are a context compression agent.

User query: {query}

From the sources below, extract ONLY the sentences relevant to answering the query.
Do not add any new information.
Keep it concise.

Sources:
{joined}

Return the compressed context only.
"""

    return llm.invoke(prompt).content.strip()