import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found in .env")

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=1024,
    )
    return llm