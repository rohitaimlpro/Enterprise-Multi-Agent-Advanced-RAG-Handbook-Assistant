from typing import List, Optional, TypedDict, Dict, Any
from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    # user input
    user_query: str

    # query understanding
    intent: str
    entities: Dict[str, Any]
    retrieval_strategy: str
    needs_action: bool

    # rewritten query
    rewritten_query: str

    # retrieval pipeline
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    compressed_context: str

    # answer pipeline
    answer: str
    verification: Dict[str, Any]

    # retry loop
    retry_count: int
    max_retries: int

    # action
    action_output: Optional[str]