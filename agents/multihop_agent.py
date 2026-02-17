from typing import List
from langchain_core.documents import Document
from agents.retrieval_agent import hybrid_retrieval_agent


def multihop_agent(original_query: str, first_pass_docs: List[Document]) -> List[Document]:
    """
    Multi-hop retrieval:
    - Extract keywords from first-pass docs
    - Re-run retrieval with expanded query
    """

    extra_terms = []

    for d in first_pass_docs[:3]:
        text = d.page_content.lower()
        if "probation" in text:
            extra_terms.append("probation")
        if "notice period" in text:
            extra_terms.append("notice period")
        if "termination" in text:
            extra_terms.append("termination")
        if "leave" in text:
            extra_terms.append("leave policy")

    expanded_query = original_query + " " + " ".join(set(extra_terms))

    second_pass_docs = hybrid_retrieval_agent(expanded_query, k_dense=8, k_bm25=8)

    # merge
    seen = set()
    merged = []
    for d in first_pass_docs + second_pass_docs:
        key = (d.metadata.get("handbook_name"), d.metadata.get("page"), d.metadata.get("chunk_id"))
        if key not in seen:
            seen.add(key)
            merged.append(d)

    return merged