from typing import List, Tuple, Dict
from collections import Counter
from langchain_core.documents import Document


def pick_primary_handbook(docs: List[Document]) -> Tuple[str, Dict[str, int]]:
    """
    Finds the most frequent handbook in the docs.
    Returns (primary_handbook, distribution)
    """
    names = []
    for d in docs:
        names.append(d.metadata.get("handbook_name", "unknown"))

    counts = Counter(names)

    if not counts:
        return "unknown", {}

    primary = counts.most_common(1)[0][0]
    return primary, dict(counts)


def filter_docs_by_handbook(docs: List[Document], handbook_name: str) -> List[Document]:
    """
    Keep only docs belonging to the chosen handbook.
    """
    if handbook_name == "unknown":
        return docs

    filtered = []
    for d in docs:
        if d.metadata.get("handbook_name", "unknown") == handbook_name:
            filtered.append(d)

    return filtered