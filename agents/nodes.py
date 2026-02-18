from agents.state import RAGState
from agents.streaming_agent import log_step
from memory.conversation_memory import load_memory, append_turn
from agents.handbook_filter import pick_primary_handbook, filter_docs_by_handbook
from agents.query_understanding_agent import query_understanding_agent
from agents.query_rewrite_agent import query_rewrite_agent
from agents.retrieval_agent import hybrid_retrieval_agent
from agents.multihop_agent import multihop_agent
from agents.reranker_agent import reranker_agent
from agents.compressor_agent import compressor_agent
from agents.answer_agent import answer_agent
from agents.verifier_agent import verifier_agent
from agents.action_agent import action_agent

def node_rerank(state: RAGState) -> RAGState:
    docs = reranker_agent(state["rewritten_query"], state["retrieved_docs"], top_n=8)

    # pick primary handbook
    primary, dist = pick_primary_handbook(docs)

    # filter docs to only primary handbook
    filtered_docs = filter_docs_by_handbook(docs, primary)

    state["reranked_docs"] = filtered_docs
    state["primary_handbook"] = primary
    state["handbook_distribution"] = dist

    return state

def node_query_understanding(state: RAGState) -> RAGState:
    log_step(state, "🧭 Understanding query (local classifier)...")

    user_query = state["user_query"]
    result = query_understanding_agent(user_query)

    state["intent"] = result.get("intent", "general_policy")
    state["entities"] = result.get("entities", {})
    state["retrieval_strategy"] = result.get("retrieval_strategy", "single_hop")
    state["needs_action"] = result.get("needs_action", False)

    return state


def node_query_rewrite(state: RAGState) -> RAGState:
    log_step(state, "✍️ Rewriting query (local FLAN-T5)...")

    rewritten = query_rewrite_agent(state["user_query"], state["intent"])
    state["rewritten_query"] = rewritten
    return state


def node_retrieval(state: RAGState) -> RAGState:
    log_step(state, "🔎 Retrieving relevant handbook sections (hybrid search)...")

    query = state["rewritten_query"]
    docs = hybrid_retrieval_agent(query, k_dense=10, k_bm25=10)
    state["retrieved_docs"] = docs
    return state


def node_multihop(state: RAGState) -> RAGState:
    if state.get("retrieval_strategy") == "multi_hop":
        log_step(state, "🧩 Multi-hop retrieval enabled...")
        docs = multihop_agent(state["rewritten_query"], state["retrieved_docs"])
        state["retrieved_docs"] = docs
    return state


def node_rerank(state: RAGState) -> RAGState:
    log_step(state, "📌 Reranking retrieved chunks (cross-encoder)...")

    docs = reranker_agent(state["rewritten_query"], state["retrieved_docs"], top_n=6)
    state["reranked_docs"] = docs
    return state


def node_compress(state: RAGState) -> RAGState:
    log_step(state, "🧽 Compressing context (local sentence selection)...")

    compressed = compressor_agent(state["user_query"], state["reranked_docs"])
    state["compressed_context"] = compressed
    return state


def node_answer(state: RAGState) -> RAGState:
    log_step(state, "🧠 Generating answer (Gemini)...")

    history = load_memory()
    state["chat_history"] = history

    ans = answer_agent(
        state["user_query"],
        state["compressed_context"],
        state["reranked_docs"],
        chat_history=history
    )
    state["answer"] = ans

    # store memory
    append_turn(state["user_query"], ans)

    return state


def node_verify(state: RAGState) -> RAGState:
    log_step(state, "✅ Verifying grounding (local verifier)...")

    verification = verifier_agent(
        state["user_query"],
        state["answer"],
        state["compressed_context"]
    )
    state["verification"] = verification
    return state


def node_action(state: RAGState) -> RAGState:
    log_step(state, "📝 Generating requested deliverable (Gemini Action Agent)...")

    output = action_agent(state["user_query"], state["compressed_context"])
    state["action_output"] = output
    return state


def node_retry(state: RAGState) -> RAGState:
    """
    Retry strategy:
    - Boost query
    - Re-run retrieval + rerank + compress + answer
    """
    log_step(state, "🔁 Retrying with boosted retrieval query...")

    state["retry_count"] = state.get("retry_count", 0) + 1

    boosted_query = state["user_query"] + " handbook policy rules eligibility process exceptions"
    docs = hybrid_retrieval_agent(boosted_query, k_dense=12, k_bm25=12)

    docs = reranker_agent(boosted_query, docs, top_n=6)
    state["reranked_docs"] = docs

    compressed = compressor_agent(state["user_query"], docs)
    state["compressed_context"] = compressed

    history = load_memory()
    ans = answer_agent(state["user_query"], compressed, docs, chat_history=history)
    state["answer"] = ans

    return state