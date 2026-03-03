"""
Query operations on embedding models: nearest neighbors and analogies.
"""

from pathlib import Path
from typing import List, Tuple, Optional

from gensim.models import KeyedVectors

from src.data.data_extraction import (
    get_nearest_neighbors, get_analogy_solution
)
from src.visualize import visualize_analogy


def nearest_neighbors(
    word: str,
    model: KeyedVectors,
    topn: int = 5,
    model_name: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Find nearest neighbors of a word in the embedding space.
    Returns list of (neighbor, similarity) tuples.
    Returns empty list if word not found.
    """
    if model is None:
        print("Model is None. Load a model first.")
        return []

    results = get_nearest_neighbors(word, model, topn)
    if results is None:
        print(f"Word '{word}' not in vocabulary.")
        sample = list(model.key_to_index.keys())[:10]
        print(f"Sample vocabulary: {', '.join(sample)}")
        return []

    # Pretty print with model name
    model_label = f"{model_name}" if model_name else ""
    print(f"\n{model_label} | NEAREST NEIGHBORS: '{word}'")
    print("─" * 60)
    for i, (neighbor, sim) in enumerate(results, 1):
        bar = "=" * int(sim * 20)
        print(f"{i:2d}. {neighbor:20s} | {sim:.4f} | {bar}")
    print("─" * 60)

    return results


def find_analogies(
    w1: str,
    w2: str,
    w3: str,
    model: KeyedVectors,
    topn: int = 3,
    model_name: Optional[str] = None,
    visualize: bool = False,
    method: str = "pca",
    save: Optional[Path] = None,
) -> List[Tuple[str, float]]:
    """Solve word analogy: w1 - w2 = ? - w3   (vector: w1 - w2 + w3)"""
    results = get_analogy_solution(w1, w2, w3, model, topn=topn)

    if results is None:
        if model is None:
            print("Load a model first.")
        else:
            missing = [
                word for word in (w1, w2, w3) if word not in model.key_to_index
            ]
            if missing:
                print(f"Words not in vocabulary: {', '.join(missing)}")
        return []

    # Pretty output
    model_label = f"{model_name}" if model_name else ""
    print(f"\n{model_label} | ANALOGY: {w1} - {w2} = ? - {w3}")
    print("─" * 60)
    print(f"{'#':>2s}  {'Solution':<20s}  {'Similarity':>10s}")
    print("─" * 60)
    for i, (candidate, sim) in enumerate(results, 1):
        print(f"{i:2d}. {candidate:<20s}  {sim:>10.4f}")
    print("─" * 60)

    # Visualization with auto-saving if requested
    if visualize and model is not None and results:
        try:
            visualize_analogy(
                w1, w2, w3, results,
                model,
                model_name=model_name or "Model",
                method=method,
                save=save
            )
        except Exception as e:
            print(f"Visualization failed: {e}")
    elif visualize and not results:
        print(
            "Cannot visualize: no valid results found "
            "(words may be missing from vocabulary)."
        )

    return results
