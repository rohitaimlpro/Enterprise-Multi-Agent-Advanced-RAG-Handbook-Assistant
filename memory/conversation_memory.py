import json
import os
from typing import List, Dict

MEMORY_PATH = "memory/chat_memory.json"


def load_memory() -> List[Dict[str, str]]:
    if not os.path.exists(MEMORY_PATH):
        return []
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_memory(history: List[Dict[str, str]]) -> None:
    os.makedirs("memory", exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history[-30:], f, indent=2, ensure_ascii=False)


def append_turn(user_query: str, assistant_answer: str) -> None:
    history = load_memory()
    history.append({"user": user_query, "assistant": assistant_answer})
    save_memory(history)