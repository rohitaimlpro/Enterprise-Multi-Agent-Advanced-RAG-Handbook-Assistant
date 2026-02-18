from typing import Dict, Any
from sentence_transformers import SentenceTransformer, util

_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def verifier_agent(user_query: str, answer: str, context: str) -> Dict[str, Any]:
    """
    Local verification:
    - checks similarity answer <-> context
    - checks if answer has sources
    - outputs confidence score
    """

    issues = []

    if not context.strip():
        return {
            "is_grounded": False,
            "confidence": 10,
            "issues": ["no_context_found"]
        }

    if "Sources:" not in answer:
        issues.append("missing_sources_section")

    # embedding similarity
    a_emb = _embedder.encode(answer, convert_to_tensor=True)
    c_emb = _embedder.encode(context, convert_to_tensor=True)

    sim = util.cos_sim(a_emb, c_emb)[0][0].item()

    # scale to 0-100
    confidence = int(max(0, min(100, sim * 100)))

    if confidence < 55:
        issues.append("weak_grounding_similarity")

    is_grounded = confidence >= 60 and len(issues) == 0

    return {
        "is_grounded": is_grounded,
        "confidence": confidence,
        "issues": issues
    }