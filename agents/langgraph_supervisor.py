import sqlite3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.state import RAGState
from agents.nodes import (
    node_query_understanding,
    node_query_rewrite,
    node_retrieval,
    node_multihop,
    node_rerank,
    node_compress,
    node_answer,
    node_verify,
    node_action,
    node_retry
)


def route_after_verify(state: RAGState) -> str:
    """
    Decide next step after verification.
    """

    verification = state.get("verification", {})
    confidence = verification.get("confidence", 0)

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)

    # Retry if weak confidence
    if confidence < 60 and retry_count < max_retries:
        return "retry"

    # If user asked for an action deliverable (email/checklist/summary)
    if state.get("needs_action", False):
        return "action"

    return "end"


def build_graph():
    graph = StateGraph(RAGState)

    # -----------------------------
    # Nodes
    # -----------------------------
    graph.add_node("understand", node_query_understanding)
    graph.add_node("rewrite", node_query_rewrite)
    graph.add_node("retrieve", node_retrieval)
    graph.add_node("multihop", node_multihop)
    graph.add_node("rerank", node_rerank)
    graph.add_node("compress", node_compress)
    graph.add_node("answer", node_answer)
    graph.add_node("verify", node_verify)
    graph.add_node("retry", node_retry)
    graph.add_node("action", node_action)

    # -----------------------------
    # Flow edges
    # -----------------------------
    graph.set_entry_point("understand")

    graph.add_edge("understand", "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "multihop")
    graph.add_edge("multihop", "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "answer")
    graph.add_edge("answer", "verify")

    # -----------------------------
    # Conditional routing after verify
    # -----------------------------
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "retry": "retry",
            "action": "action",
            "end": END
        }
    )

    # retry goes back to verify (it re-generates answer inside retry node)
    graph.add_edge("retry", "verify")

    # action ends
    graph.add_edge("action", END)

    # -----------------------------
    # SQLite Checkpointing (Fixed)
    # -----------------------------
    conn = sqlite3.connect("memory/checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)