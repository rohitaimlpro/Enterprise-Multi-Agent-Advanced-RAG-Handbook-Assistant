import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def query_understanding_agent(user_query: str) -> dict:
    """
    Returns a JSON dict:
    {
      "intent": "...",
      "entities": {...},
      "retrieval_strategy": "single_hop" | "multi_hop",
      "needs_action": true/false
    }
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1
    )

    prompt = f"""
You are a company handbook query understanding agent.

Extract intent and entities from the user query.
Return ONLY valid JSON.

Possible intents:
- leave_policy
- benefits
- payroll
- resignation
- notice_period
- probation
- wfh_policy
- code_of_conduct
- termination
- grievance
- travel_policy
- general_policy

retrieval_strategy:
- "single_hop" if answer likely in one section
- "multi_hop" if answer needs multiple sections

needs_action:
- true if user is asking for drafting something (email/checklist)
- false otherwise

User query: {user_query}
"""

    resp = llm.invoke(prompt).content.strip()

    try:
        return json.loads(resp)
    except Exception:
        # fallback safe response
        return {
            "intent": "general_policy",
            "entities": {},
            "retrieval_strategy": "single_hop",
            "needs_action": False
        }