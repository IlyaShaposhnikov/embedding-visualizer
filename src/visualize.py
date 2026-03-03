"""
Dimensionality reduction and 2D projection of word embeddings.
Visualizes semantic clusters formed by seed words and their nearest neighbors.
"""

from pathlib import Path
from typing import List, Tuple, Optional

from gensim.models import KeyedVectors
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data.data_extraction import get_nearest_neighbors

# Default visualization settings
DEFAULT_N_WORDS = 50
DEFAULT_METHOD = "pca"
RANDOM_SEED = 42  # Fixed seed for reproducible PCA/t-SNE results


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
            # t-SNE perplexity should be less than number of points.
            # We cap it at (n_samples - 2) to avoid warnings/errors.
            reducer = TSNE(
                n_components=2,
                perplexity=min(perplexity, len(valid_words) - 2),
                random_state=random_state,
                init="pca",  # Initialise with PCA for faster convergence
                learning_rate="auto",  # Let sklearn set appropriate rate
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
        # Assign distinct colors to clusters using tab10 colormap
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

    # Collect words and assign cluster labels based on seed index
    word_to_cluster = {}
    cluster_words = []  # list of (word, cluster_id)

    for idx, seed in enumerate(seed_words):
        # Add seed word itself
        if seed not in word_to_cluster:
            word_to_cluster[seed] = idx
            cluster_words.append((seed, idx))

        # Fetch neighbors using the shared nearest-neighbor logic
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
        f"{model_name} - {method.upper()} | Semantic Clusters | "
        f"(Seeds: {', '.join(seed_words)})"
    )
    plot_embeddings(
        coords,
        words,
        labels,
        seed_words,
        title=title,
        save_path=save,
    )


def _plot_analogy(
    coords: np.ndarray,
    words: List[str],
    labels: List[int],
    w1_idx: Optional[int],
    w2_idx: Optional[int],
    w3_idx: Optional[int],
    result_indices: List[int],
    title: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot analogy with vector arrows and semantic coloring."""
    plt.figure(figsize=(14, 10))

    # Color mapping:
    # 0 (w1) and 3 (results) -> blue, 1 (w2) -> red, 2 (w3) -> green
    colors = {
        0: '#1f77b4',  # blue for w1
        1: '#d62728',  # red for w2
        2: '#2ca02c',  # green for w3
        3: '#1f77b4',  # blue for predicted results (same as w1 side)
    }

    marker_map = {
        0: 'o',  # w1 — circle
        1: 's',  # w2 — square
        2: '^',  # w3 — triangle
        3: 'D',  # results — diamond
    }

    for i, (word, label) in enumerate(zip(words, labels)):
        plt.scatter(
            coords[i, 0], coords[i, 1],
            color=colors.get(label, '#7f7f7f'),
            marker=marker_map.get(label, 'o'),
            s=300 if i in [w1_idx, w2_idx, w3_idx] + result_indices else 150,
            edgecolors='k',
            linewidths=1.5,
            alpha=0.9,
            # Higher zorder brings key points to front
            zorder=3 if i in [w1_idx, w2_idx, w3_idx] + result_indices else 2
        )
        plt.annotate(
            word,
            (coords[i, 0], coords[i, 1]),
            fontsize=(
                11
                if i in [w1_idx, w2_idx, w3_idx] + result_indices
                else 9
            ),
            alpha=0.95,
            xytext=(8, 8),
            textcoords='offset points',
            fontweight=(
                'bold'
                if i in [w1_idx, w2_idx, w3_idx] + result_indices
                else 'normal'
            ),
            zorder=4  # annotations above points
        )

    # Draw arrow from w2 to w1 representing (w1 - w2)
    if w1_idx is not None and w2_idx is not None:
        # w2 → w1
        arrow = FancyArrowPatch(
            (coords[w2_idx, 0], coords[w2_idx, 1]),
            (coords[w1_idx, 0], coords[w1_idx, 1]),
            arrowstyle='->,head_width=0.8,head_length=1.2',
            color='#d62728',
            linewidth=2.5,
            alpha=0.7,
            zorder=1  # arrows behind points
        )
        plt.gca().add_patch(arrow)
        # Place text at midpoint of arrow
        plt.text(
            (coords[w2_idx, 0] + coords[w1_idx, 0]) / 2,
            (coords[w2_idx, 1] + coords[w1_idx, 1]) / 2,
            'w1 - w2',
            fontsize=10,
            color='#d62728',
            fontweight='bold',
            ha='center',
            va='bottom'
        )

    # Draw arrow from the top result to w3 representing (? - w3)
    if w3_idx is not None and result_indices:
        # result → w3
        result_idx = result_indices[0]  # use the top predicted word
        arrow = FancyArrowPatch(
            (coords[result_idx, 0], coords[result_idx, 1]),
            (coords[w3_idx, 0], coords[w3_idx, 1]),
            arrowstyle='->,head_width=0.8,head_length=1.2',
            color='#2ca02c',
            linewidth=2.5,
            alpha=0.7,
            zorder=1
        )
        plt.gca().add_patch(arrow)
        plt.text(
            (coords[result_idx, 0] + coords[w3_idx, 0]) / 2,
            (coords[result_idx, 1] + coords[w3_idx, 1]) / 2,
            '? - w3',
            fontsize=10,
            color='#2ca02c',
            fontweight='bold',
            ha='center',
            va='bottom'
        )

    # Build custom legend
    legend_elements = [
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
            markersize=12, label=(
                'w1 / Results '
                f'({words[w1_idx] if w1_idx is not None else "?"})'
            )
        ),
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='#d62728',
            markersize=12, label=(
                'w2 '
                f'({words[w2_idx] if w2_idx is not None else "?"})'
            )
        ),
        plt.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
            markersize=12, label=(
                'w3 '
                f'({words[w3_idx] if w3_idx is not None else "?"})'
            )
        ),
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path.as_posix()}")

    plt.show()


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
    # Collect all words to be plotted
    words = [w1, w2, w3] + [word for word, _ in results]

    valid_words = [w for w in words if w in model.key_to_index]
    if not valid_words:
        print("No valid words for visualization.")
        return

    coords = project_words(model, valid_words, method=method)
    if coords is None or len(coords) != len(valid_words):
        return

    # Assign label: 0 for w1, 1 for w2, 2 for w3, 3 for predicted results
    word_to_idx = {word: i for i, word in enumerate(valid_words)}
    labels = []
    for word in valid_words:
        if word == w1:
            labels.append(0)
        elif word == w2:
            labels.append(1)
        elif word == w3:
            labels.append(2)
        else:
            labels.append(3)

    title = (
        f"{model_name} - {method.upper()} | Analogy: {w1} - {w2} = ? - {w3}"
    )
    _plot_analogy(
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
