"""
Pure logic for preparing data for visualization.
Does not perform any plotting or printing.
"""

from typing import Dict, List, Tuple, Optional

from gensim.models import KeyedVectors

from src.data.data_extraction import get_nearest_neighbors


def prepare_cluster_data(
    seed_words: list,
    model: KeyedVectors,
    topn: int = 3,
) -> Tuple[Optional[List[str]], Optional[List[int]], Optional[int]]:
    """
    Prepare data for cluster visualization.
    Returns:
        - List of words to visualize
        - List of cluster labels for each word
        - Total number of collected words
    """
    if model is None:
        return None, None, None

    # Validate seed words
    missing = [w for w in seed_words if w not in model.key_to_index]
    if missing:
        # Could return error information, for now just return None
        return None, None, None

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
        # else: # Optionally log that no neighbors were found for a seed

    total_words = len(cluster_words)

    if total_words < 2:
        # Insufficient words for projection
        return None, None, total_words

    # Extract words and labels
    words = [w for w, _ in cluster_words]
    labels = [c for _, c in cluster_words]

    return words, labels, total_words


def prepare_analogy_data(
    w1: str,
    w2: str,
    w3: str,
    results: List[Tuple[str, float]],
    model: KeyedVectors,
) -> Tuple[Optional[List[str]], Optional[List[int]], Optional[Dict[str, int]]]:
    """
    Prepare data for analogy visualization.
    Returns:
        - List of words to visualize (w1, w2, w3, top results)
        - List of labels for each word
        (0 for w1, 1 for w2, 2 for w3, 3 for results)
        - Dictionary mapping words to their index in the valid_words list
    """
    if model is None:
        return None, None, None

    # Collect all words that were intended to be plotted
    all_candidate_words = [w1, w2, w3] + [word for word, _ in results]

    # Filter for words present in the model's vocabulary
    # This ensures that only words with known vectors
    # are included in the visualization
    valid_words = [w for w in all_candidate_words if w in model.key_to_index]

    if not valid_words:
        # No words from the candidates are in the model vocabulary
        # Cannot proceed with visualization
        return None, None, None

    # Assign labels based on the role of the word in the analogy:
    # 0: First term (w1) - typically shown in blue
    # 1: Second term (w2) - typically shown in red
    # 2: Third term (w3) - typically shown in green
    # 3: Predicted result(s) - typically shown in blue (like w1)
    labels = []
    for word in valid_words:
        if word == w1:
            labels.append(0)
        elif word == w2:
            labels.append(1)
        elif word == w3:
            labels.append(2)
        # Must be a result word (since it's in valid_words but not w1, w2, w3)
        else:
            labels.append(3)

    # Create mapping for indices
    # (used for plotting arrows between specific points)
    # word_to_idx[word] gives the index of 'word'
    # in the 'valid_words' and 'labels' lists
    word_to_idx = {word: i for i, word in enumerate(valid_words)}

    # Return the prepared data structures
    return valid_words, labels, word_to_idx
