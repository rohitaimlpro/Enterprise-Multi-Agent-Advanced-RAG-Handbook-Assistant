import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def action_agent(user_query: str, context: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3
    )

    prompt = f"""
You are an enterprise action agent.

Based ONLY on the handbook context, generate the requested deliverable.
Examples: email draft, checklist, summary.

User request:
{user_query}

Handbook Context:
{context}

Rules:
- If information is missing, say it clearly
- Do not invent policy numbers
- Keep output professional

Return the deliverable.
"""

    return llm.invoke(prompt).content.strip()