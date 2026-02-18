from typing import List
import re
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util

_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 25]


def compressor_agent(query: str, docs: List[Document]) -> str:
    """
    Compress context locally:
    - Split docs into sentences
    - Rank sentences by similarity to query
    - Keep top sentences
    """

    if not docs:
        return ""

    sentences = []
    for d in docs[:6]:
        sents = _split_sentences(d.page_content)
        for s in sents:
            sentences.append(s)

    if not sentences:
        return "\n\n".join([d.page_content[:600] for d in docs[:3]])

    q_emb = _embedder.encode(query, convert_to_tensor=True)
    s_emb = _embedder.encode(sentences, convert_to_tensor=True)

    scores = util.cos_sim(q_emb, s_emb)[0].cpu().tolist()

    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

    top_sentences = []
    for sent, sc in ranked[:18]:
        top_sentences.append(sent)

    return "\n".join(top_sentences)