import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def query_rewrite_agent(user_query: str, intent: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2
    )

    prompt = f"""
You rewrite employee handbook questions into optimized retrieval queries.

Rules:
- keep it short
- include keywords
- include synonyms
- remove unnecessary words

Intent: {intent}
User query: {user_query}

Return ONLY the rewritten query text.
"""

    return llm.invoke(prompt).content.strip()