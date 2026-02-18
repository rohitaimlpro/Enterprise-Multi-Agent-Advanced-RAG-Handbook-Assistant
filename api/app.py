from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from typing import Dict, Any, List
import asyncio
import re

from api.schemas import ChatRequest, ChatResponse
from agents.langgraph_supervisor import build_graph


app = FastAPI(title="Enterprise Handbook RAG API", version="1.0")

# Allow UI / browser calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build graph once (important)
GRAPH_APP = build_graph()


def _extract_sources_from_answer(answer: str) -> List[Dict[str, Any]]:
    """
    Extracts sources from the answer text.
    Your answer_agent prints citations like:
    Sources:
    [1] handbook (page x, chunk y)
    ...
    """
    sources = []
    if "Sources:" not in answer:
        return sources

    after = answer.split("Sources:", 1)[-1].strip()
    lines = [l.strip() for l in after.splitlines() if l.strip()]

    for line in lines:
        # Example:
        # [1] ABC Handbook (page 10, chunk 5)
        m = re.match(r"^\[(\d+)\]\s+(.*)$", line)
        if m:
            sources.append({"id": int(m.group(1)), "text": m.group(2)})

    return sources


@app.get("/")
def root():
    return {"status": "ok", "message": "Enterprise Handbook RAG API is running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    initial_state = {
        "user_query": req.query,
        "retry_count": 0,
        "max_retries": 1
    }

    config = {
        "configurable": {
            "thread_id": req.thread_id,
        }
    }

    result: Dict[str, Any] = GRAPH_APP.invoke(initial_state, config=config)

    verification = result.get("verification", {})
    answer = result.get("answer", "")

    sources = _extract_sources_from_answer(answer)

    return ChatResponse(
        answer=answer,
        confidence=int(verification.get("confidence", 0)),
        is_grounded=bool(verification.get("is_grounded", False)),
        issues=verification.get("issues", []),
        action_output=result.get("action_output"),
        intent=result.get("intent"),
        rewritten_query=result.get("rewritten_query"),
        stream_log=result.get("stream_log", []),
        sources=sources
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming endpoint using SSE.
    We stream the node updates from LangGraph.
    """

    initial_state = {
        "user_query": req.query,
        "retry_count": 0,
        "max_retries": 1
    }

    config = {
        "configurable": {
            "thread_id": req.thread_id,
        }
    }

    async def event_generator():
        # LangGraph streaming yields events
        for event in GRAPH_APP.stream(initial_state, config=config):
            # event is usually dict: {"node_name": {...state...}}
            yield {
                "event": "message",
                "data": str(event)
            }
            await asyncio.sleep(0.01)

        yield {"event": "done", "data": "DONE"}

    return EventSourceResponse(event_generator())