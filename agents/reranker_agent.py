from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# lightweight reranker (works on CPU)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def reranker_agent(query: str, docs: List[Document], top_n: int = 6) -> List[Document]:
    """
    Reranks docs using cross-encoder and returns top_n.
    """

    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [d for d, s in scored_docs[:top_n]]