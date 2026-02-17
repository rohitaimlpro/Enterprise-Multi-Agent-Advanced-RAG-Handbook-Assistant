import os
from typing import List, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


CHROMA_DIR = "data/vectorstore"
COLLECTION_NAME = "company_handbooks"

# same embeddings used during ingestion
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _build_bm25_index(all_docs: List[Document]) -> BM25Okapi:
    tokenized_corpus = [_tokenize(d.page_content) for d in all_docs]
    return BM25Okapi(tokenized_corpus)


def hybrid_retrieval_agent(query: str, k_dense: int = 8, k_bm25: int = 8) -> List[Document]:
    """
    Returns merged docs from:
    - dense similarity search
    - BM25 keyword search
    """

    # Dense retrieval
    dense_docs = vectordb.similarity_search(query, k=k_dense)

    # BM25 retrieval (needs full corpus)
    all_docs = vectordb.get()["documents"]
    all_metas = vectordb.get()["metadatas"]

    corpus_docs = [
        Document(page_content=all_docs[i], metadata=all_metas[i])
        for i in range(len(all_docs))
    ]

    bm25 = _build_bm25_index(corpus_docs)

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k_bm25]
    bm25_docs = [corpus_docs[i] for i in top_indices]

    # Merge + deduplicate
    seen = set()
    merged = []

    for d in dense_docs + bm25_docs:
        key = (d.metadata.get("handbook_name"), d.metadata.get("page"), d.metadata.get("chunk_id"))
        if key not in seen:
            seen.add(key)
            merged.append(d)

    return merged