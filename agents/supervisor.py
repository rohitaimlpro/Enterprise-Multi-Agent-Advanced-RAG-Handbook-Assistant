from agents.query_understanding_agent import query_understanding_agent
from agents.query_rewrite_agent import query_rewrite_agent
from agents.retrieval_agent import hybrid_retrieval_agent
from agents.multihop_agent import multihop_agent
from agents.reranker_agent import reranker_agent
from agents.compressor_agent import compressor_agent
from agents.answer_agent import answer_agent
from agents.verifier_agent import verifier_agent
from agents.action_agent import action_agent


def run_supervisor(user_query: str):
    # 1) Understand query
    understanding = query_understanding_agent(user_query)
    intent = understanding.get("intent", "general_policy")
    strategy = understanding.get("retrieval_strategy", "single_hop")
    needs_action = understanding.get("needs_action", False)

    # 2) Rewrite query
    rewritten = query_rewrite_agent(user_query, intent)

    # 3) Retrieve
    docs = hybrid_retrieval_agent(rewritten, k_dense=10, k_bm25=10)

    # 4) Multi-hop if needed
    if strategy == "multi_hop":
        docs = multihop_agent(rewritten, docs)

    # 5) Rerank
    docs = reranker_agent(rewritten, docs, top_n=6)

    # 6) Compress context
    compressed = compressor_agent(user_query, docs)

    # 7) Answer
    answer = answer_agent(user_query, compressed, docs)

    # 8) Verify
    verification = verifier_agent(user_query, answer, compressed)

    # 9) Retry once if weak
    if verification["confidence"] < 60:
        docs = hybrid_retrieval_agent(user_query + " handbook policy", k_dense=12, k_bm25=12)
        docs = reranker_agent(user_query, docs, top_n=6)
        compressed = compressor_agent(user_query, docs)
        answer = answer_agent(user_query, compressed, docs)
        verification = verifier_agent(user_query, answer, compressed)

    # 10) Optional action
    action_output = None
    if needs_action:
        action_output = action_agent(user_query, compressed)

    return {
        "intent": intent,
        "rewritten_query": rewritten,
        "answer": answer,
        "verification": verification,
        "action_output": action_output
    }