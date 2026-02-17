from agents.state import RAGState
from agents.query_understanding_agent import query_understanding_agent
from agents.query_rewrite_agent import query_rewrite_agent
from agents.retrieval_agent import hybrid_retrieval_agent
from agents.multihop_agent import multihop_agent
from agents.reranker_agent import reranker_agent
from agents.compressor_agent import compressor_agent
from agents.answer_agent import answer_agent
from agents.verifier_agent import verifier_agent
from agents.action_agent import action_agent


def node_query_understanding(state: RAGState) -> RAGState:
    user_query = state["user_query"]
    result = query_understanding_agent(user_query)

    state["intent"] = result.get("intent", "general_policy")
    state["entities"] = result.get("entities", {})
    state["retrieval_strategy"] = result.get("retrieval_strategy", "single_hop")
    state["needs_action"] = result.get("needs_action", False)

    return state


def node_query_rewrite(state: RAGState) -> RAGState:
    rewritten = query_rewrite_agent(state["user_query"], state["intent"])
    state["rewritten_query"] = rewritten
    return state


def node_retrieval(state: RAGState) -> RAGState:
    query = state["rewritten_query"]
    docs = hybrid_retrieval_agent(query, k_dense=10, k_bm25=10)
    state["retrieved_docs"] = docs
    return state


def node_multihop(state: RAGState) -> RAGState:
    if state.get("retrieval_strategy") == "multi_hop":
        docs = multihop_agent(state["rewritten_query"], state["retrieved_docs"])
        state["retrieved_docs"] = docs
    return state


def node_rerank(state: RAGState) -> RAGState:
    docs = reranker_agent(state["rewritten_query"], state["retrieved_docs"], top_n=6)
    state["reranked_docs"] = docs
    return state


def node_compress(state: RAGState) -> RAGState:
    compressed = compressor_agent(state["user_query"], state["reranked_docs"])
    state["compressed_context"] = compressed
    return state


def node_answer(state: RAGState) -> RAGState:
    ans = answer_agent(state["user_query"], state["compressed_context"], state["reranked_docs"])
    state["answer"] = ans
    return state


def node_verify(state: RAGState) -> RAGState:
    verification = verifier_agent(state["user_query"], state["answer"], state["compressed_context"])
    state["verification"] = verification
    return state


def node_action(state: RAGState) -> RAGState:
    output = action_agent(state["user_query"], state["compressed_context"])
    state["action_output"] = output
    return state


def node_retry(state: RAGState) -> RAGState:
    """
    Retry strategy: do retrieval again with stronger query.
    """
    state["retry_count"] = state.get("retry_count", 0) + 1

    boosted_query = state["user_query"] + " handbook policy rules"
    docs = hybrid_retrieval_agent(boosted_query, k_dense=12, k_bm25=12)

    # rerank + compress + answer again
    docs = reranker_agent(boosted_query, docs, top_n=6)
    state["reranked_docs"] = docs

    compressed = compressor_agent(state["user_query"], docs)
    state["compressed_context"] = compressed

    ans = answer_agent(state["user_query"], compressed, docs)
    state["answer"] = ans

    return state