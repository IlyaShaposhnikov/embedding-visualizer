"""
Dimensionality reduction and 2D projection of word embeddings.
Supports PCA and t-SNE.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors

# Default visualization settings
DEFAULT_N_WORDS = 50
DEFAULT_METHOD = "pca"
RANDOM_SEED = 42
RANDOM_GENERATOR = np.random.default_rng(RANDOM_SEED)


def project_words(
    model: KeyedVectors,
    words: list,
    method: str = "pca",
    perplexity: int = 30,
    random_state: int = RANDOM_SEED,
) -> Optional[np.ndarray]:
    """Project word vectors into 2D space using PCA or t-SNE."""
    if not words:
        print("No words provided.")
        return None

    # Filter words present in vocabulary
    valid_words = [w for w in words if w in model.key_to_index]
    if not valid_words:
        print("None of the provided words are in the vocabulary.")
        return None

    if len(valid_words) < len(words):
        print(
            f"Skipped {len(words) - len(valid_words)} word(s) "
            "not in vocabulary."
        )

    # Get vectors
    vectors = model[valid_words]

    try:
        if method == "pca":
            if len(valid_words) < 2:
                print(
                    "PCA requires at least 2 points "
                    f"(got {len(valid_words)}). "
                    "Provide more words."
                )
                return None
            reducer = PCA(n_components=2, random_state=random_state)
        elif method == "tsne":
            if len(valid_words) < 3:
                print(
                    "t-SNE requires at least 3 points "
                    f"(got {len(valid_words)}). "
                    "Use PCA instead or provide more words."
                )
                return None
            reducer = TSNE(
                n_components=2,
                perplexity=min(perplexity, len(valid_words) - 2),
                random_state=random_state,
                init="pca",
                learning_rate="auto",
            )
        else:
            print(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
            return None

        coords = reducer.fit_transform(vectors)
        return coords

    except Exception as e:
        print(f"Projection failed: {e}")
        return None


def plot_embeddings(
    coords: np.ndarray,
    words: list,
    title: str = "Word Embeddings Visualization",
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
    show_labels: bool = True,
) -> None:
    """Plot 2D projections with optional word labels."""
    if coords is None or len(coords) == 0:
        print("No coordinates to plot.")
        return

    plt.figure(figsize=figsize)
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6, edgecolors="k")

    # Add labels (if requested)
    if show_labels:
        for i, word in enumerate(words):
            plt.annotate(
                word,
                (coords[i, 0], coords[i, 1]),
                fontsize=9,
                alpha=0.8,
                xytext=(5, 5),
                textcoords="offset points",
            )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def visualize_random_words(
    model: KeyedVectors,
    n_words: int = DEFAULT_N_WORDS,
    method: str = DEFAULT_METHOD,
    model_name: str = "Model",
    save: Optional[str] = None,
) -> None:
    """Select random words from model vocabulary and visualize them."""
    if model is None:
        print("Model is None. Load a model first.")
        return

    vocab_size = len(model)
    if vocab_size == 0:
        print("Vocabulary is empty.")
        return

    n_words = min(n_words, vocab_size)
    # Random sample
    words = RANDOM_GENERATOR.choice(
        list(model.key_to_index.keys()),
        size=n_words,
        replace=False
    ).tolist()

    print(f"Projecting {n_words} random words using {method.upper()}...")
    coords = project_words(model, words, method=method)

    if coords is not None:
        show_labels = n_words <= 30
        if not show_labels:
            print(
                f"Note: Labels hidden for clarity (n_words={n_words} > 30). "
                "Use fewer words or call plot_embeddings() directly "
                "with show_labels=True."
            )

        title = (
            f"{model_name} - {method.upper()} projection of {n_words} words"
        )
        plot_embeddings(
            coords, words, title=title, save_path=save, show_labels=show_labels
        )
