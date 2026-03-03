"""
Logic for visualizing word analogies.
Delegates data preparation and plotting.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

from gensim.models import KeyedVectors

from src.visualization.data_preparation import prepare_analogy_data
from src.visualization.projections import project_words
from src.visualization.plotting import plot_analogy

logger = logging.getLogger(__name__)


def visualize_analogy(
    w1: str,
    w2: str,
    w3: str,
    results: List[Tuple[str, float]],
    model: KeyedVectors,
    model_name: str = "Model",
    method: str = "pca",
    save: Optional[Path] = None,
) -> None:
    """
    Visualize word analogy with vector relationships.

    Color scheme:
      • w1 and predicted words: blue (left side of analogy)
      • w2: red (subtracted word)
      • w3: green (right side of analogy)

    Shows vector arrows: w2 → w1 and predicted → w3
    """
    valid_words, labels, word_to_idx = prepare_analogy_data(
        w1, w2, w3, results, model
    )

    if valid_words is None or labels is None or word_to_idx is None:
        print("No valid words for visualization.")
        return

    # Project
    coords = project_words(model, valid_words, method=method)
    if coords is None or len(coords) != len(valid_words):
        # Projection failed or returned wrong size
        return

    title = (
        f"{model_name} - {method.upper()} | Analogy: {w1} - {w2} = ? - {w3}"
    )
    plot_analogy(
        coords,
        valid_words,
        labels,
        w1_idx=word_to_idx.get(w1),
        w2_idx=word_to_idx.get(w2),
        w3_idx=word_to_idx.get(w3),
        result_indices=[
            word_to_idx.get(r[0]) for r in results if r[0] in word_to_idx
        ],
        title=title,
        save_path=save,
    )
