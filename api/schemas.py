from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    thread_id: str = Field(default="default_thread")


class ChatResponse(BaseModel):
    answer: str
    confidence: int
    is_grounded: bool
    issues: List[str]

    action_output: Optional[str] = None

    intent: Optional[str] = None
    rewritten_query: Optional[str] = None

    stream_log: List[str] = []
    sources: List[Dict[str, Any]] = []