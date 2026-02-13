"""
Dimensionality reduction and 2D projection of word embeddings.
Visualizes semantic clusters formed by seed words and their nearest neighbors.
"""

from typing import Optional

from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.queries import get_nearest_neighbors

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
    labels: Optional[list] = None,
    seed_words: Optional[list] = None,
    title: str = "Word Embeddings Visualization",
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
) -> None:
    """Plot 2D projections with optional colored clusters and word labels."""
    if coords is None or len(coords) == 0:
        print("No coordinates to plot.")
        return

    plt.figure(figsize=figsize)

    if labels is not None and seed_words is not None:
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for cluster_id in unique_labels:
            mask = [i for i, label in enumerate(labels) if label == cluster_id]
            plt.scatter(
                coords[mask, 0],
                coords[mask, 1],
                color=colors[cluster_id],
                label=f"Cluster: {seed_words[cluster_id]}",
                alpha=0.7,
                edgecolors="k",
                s=100
            )
    else:
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6, edgecolors="k")

    for i, word in enumerate(words):
        plt.annotate(
            word,
            (coords[i, 0], coords[i, 1]),
            fontsize=9,
            alpha=0.9,
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, linestyle="--", alpha=0.5)

    if labels is not None:
        plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def visualize_word_clusters(
    seed_words: list,
    model: KeyedVectors,
    topn: int = 3,
    method: str = "pca",
    model_name: str = "Model",
    save: Optional[str] = None,
) -> None:
    """
    Visualize semantic clusters formed by seed words
    and their nearest neighbors.
    """
    if model is None:
        print("Model is None. Load a model first.")
        return

    # Validate seed words
    missing = [w for w in seed_words if w not in model.key_to_index]
    if missing:
        print(f"Seed words not in vocabulary: {', '.join(missing)}")
        return

    if len(seed_words) > 6:
        print(
            f"Warning: {len(seed_words)} seed words "
            "may produce a crowded plot. "
            "Consider using 3-4 words for better readability."
        )

    # Gather words and their cluster labels using nearest_neighbors logic
    word_to_cluster = {}
    cluster_words = []  # list of (word, cluster_id)

    for idx, seed in enumerate(seed_words):
        # Add seed word itself
        if seed not in word_to_cluster:
            word_to_cluster[seed] = idx
            cluster_words.append((seed, idx))

        # Fetch neighbors using ВАШУ функцию (единая точка контроля)
        neighbors = get_nearest_neighbors(seed, model, topn=topn)
        if neighbors:
            for neighbor, _ in neighbors:
                if neighbor not in word_to_cluster:
                    word_to_cluster[neighbor] = idx
                    cluster_words.append((neighbor, idx))
        else:
            print(f"Warning: no neighbors found for '{seed}'")

    total_words = len(cluster_words)
    print(
        f"Collected {total_words} words: {len(seed_words)} seeds "
        f"+ up to {topn} neighbors each."
    )

    if total_words < 2:
        print("Insufficient words for projection.")
        return

    # Extract words and labels
    words = [w for w, _ in cluster_words]
    labels = [c for _, c in cluster_words]

    # Project
    coords = project_words(model, words, method=method)
    if coords is None:
        return

    # Plot with colors
    title = (
        f"{model_name} - {method.upper()} clusters "
        f"(seeds: {', '.join(seed_words)})"
    )
    plot_embeddings(
        coords,
        words,
        labels,
        seed_words,
        title=title,
        save_path=save,
    )
