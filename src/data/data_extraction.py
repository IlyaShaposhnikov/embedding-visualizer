"""
Data extraction module for embedding models.
Contains pure functions for data retrieval without side effects.
"""

from typing import List, Tuple, Optional

from gensim.models import KeyedVectors


def get_nearest_neighbors(
    word: str,
    model: KeyedVectors,
    topn: int = 5,
) -> Optional[List[Tuple[str, float]]]:
    """
    Retrieve nearest neighbors of a word from the embedding model.

    Args:
        word: Target word to find neighbors for
        model: Embedding model to query
        topn: Number of neighbors to return (default: 5)

    Returns:
        List of (neighbor, similarity) tuples or None if word not in vocabulary
    """
    if model is None or word not in model.key_to_index:
        return None

    try:
        return model.most_similar(positive=[word], topn=topn)
    except Exception:
        return None


def get_analogy_solution(
    w1: str,
    w2: str,
    w3: str,
    model: KeyedVectors,
    topn: int = 1,
) -> Optional[List[Tuple[str, float]]]:
    """
    Solve word analogy problem: w1 - w2 + w3 = ?

    Args:
        w1, w2, w3: Words forming the analogy w1 - w2 = ? - w3
        model: Embedding model to query
        topn: Number of solutions to return (default: 1)

    Returns:
        List of (candidate, similarity) tuples or None if words missing
    """
    if model is None:
        return None

    # Check that all input words exist in vocabulary
    missing = [word for word in (w1, w2, w3) if word not in model.key_to_index]
    if missing:
        return None

    try:
        # Vector arithmetic: w1 - w2 + w3
        return model.most_similar(
            positive=[w1, w3], negative=[w2], topn=topn
        )
    except Exception:
        return None
