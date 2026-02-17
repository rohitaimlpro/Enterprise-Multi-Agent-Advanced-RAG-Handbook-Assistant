import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def verifier_agent(user_query: str, answer: str, context: str) -> dict:
    """
    Returns:
    {
      "is_grounded": true/false,
      "confidence": 0-100,
      "issues": ["..."]
    }
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.0
    )

    prompt = f"""
You are a hallucination verification agent.

Check whether the answer is fully supported by the context.
Return ONLY valid JSON.

User query:
{user_query}

Context:
{context}

Answer:
{answer}

Return JSON:
{{
  "is_grounded": true/false,
  "confidence": 0-100,
  "issues": ["missing citations", "unsupported claim", "too vague", ...]
}}
"""

    resp = llm.invoke(prompt).content.strip()

    try:
        return json.loads(resp)
    except Exception:
        return {
            "is_grounded": False,
            "confidence": 30,
            "issues": ["verifier_failed_to_parse_json"]
        }