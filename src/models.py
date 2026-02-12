"""
Load pre-trained embedding models (Word2Vec, GloVe) into Gensim KeyedVectors.
Provides fast loading via cached binary format and nearest neighbors search.
"""

import os
import tempfile
from typing import Optional, List, Tuple

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

DEFAULT_DATA_DIR = "data"
W2V_BIN = "GoogleNews-vectors-negative300.bin"
DEFAULT_GLOVE_VERSION = "6B.100d"


def _get_cached_model_path(original_path: str) -> str:
    """Generate path for cached Gensim model."""
    base, _ = os.path.splitext(original_path)
    return base + ".model"


def _load_cached_model(
    cached_path: str,
    model_name: str,
) -> Optional[KeyedVectors]:
    """Load model from cache with integrity validation."""
    try:
        print(f"Loading cached {model_name} model: {cached_path}")
        model = KeyedVectors.load(cached_path)
        # Validate integrity: non-empty vocabulary and valid dimensionality
        if len(model) == 0 or model.vector_size == 0:
            raise ValueError("Cached model is empty/corrupted")
        print(
            f"Loaded from cache (vocab size: {len(model):,}, "
            f"dim: {model.vector_size})"
        )
        return model
    except (EOFError, ValueError, KeyError, OSError) as e:
        print(
            f"Corrupted {model_name} cache "
            f"({type(e).__name__}): {e}. Rebuilding from source..."
        )
        return None


def load_word2vec_model(
    bin_path: Optional[str] = None,
    data_dir: str = DEFAULT_DATA_DIR,
    use_cached: bool = True,
    force_reload: bool = False,
) -> Optional[KeyedVectors]:
    """
    Load GoogleNews Word2Vec model from binary file.
    If a cached .model file exists, loads it instantly (recommended).
    """
    if bin_path is None:
        bin_path = os.path.join(data_dir, W2V_BIN)

    if not os.path.exists(bin_path) or os.path.getsize(bin_path) == 0:
        print(f"Word2Vec binary not found: {bin_path}")
        print(
            "Please run download_word2vec_model() first "
            "or place the file manually."
        )
        return None

    cached_path = _get_cached_model_path(bin_path)

    # Try cached model
    if use_cached and not force_reload and os.path.exists(cached_path):
        model = _load_cached_model(cached_path, "Word2Vec")
        if model is not None:
            return model

    # Load from original .bin (slow, memory heavy)
    print(
        "Loading Word2Vec from binary "
        "(this may take several minutes and ~4GB RAM)..."
    )
    try:
        model = KeyedVectors.load_word2vec_format(bin_path, binary=True)
        print("Word2Vec loaded successfully.")
        print(f"Vocabulary: {len(model):,} words")
        print(f"Vector dimension: {model.vector_size}")

        # Save to cache for future fast loading
        if use_cached:
            try:
                print(f"Saving cached model to {cached_path} ...")
                model.save(cached_path)
                print("Cached model saved (next load will be instant).")
            except Exception as e:
                print(f"Could not save cache: {e}")

        return model

    except Exception as e:
        print(f"Failed to load Word2Vec model: {e}")
        return None


def load_glove_model(
    txt_path: Optional[str] = None,
    version: str = DEFAULT_GLOVE_VERSION,
    data_dir: str = DEFAULT_DATA_DIR,
    use_cached: bool = True,
    force_reload: bool = False,
) -> Optional[KeyedVectors]:
    """
    Load GloVe model from .txt file
    (converted to Word2Vec format internally).
    """
    if txt_path is None:
        txt_path = os.path.join(data_dir, f"glove.{version}.txt")

    if not os.path.exists(txt_path):
        print(f"GloVe file not found: {txt_path}")
        print(
            "Please run download_glove_model() first "
            "or place the file manually."
        )
        return None

    cached_path = _get_cached_model_path(txt_path)

    # Try cached model
    if use_cached and not force_reload and os.path.exists(cached_path):
        model = _load_cached_model(cached_path, "GloVe")
        if model is not None:
            return model

    # Convert GloVe .txt → Word2Vec .txt (temporary) and load
    print(f"Converting and loading GloVe from {txt_path} ...")
    try:
        # GloVe files are in word2vec format but without the header line.
        # Gensim's glove2word2vec adds the header and saves as a new file.
        with tempfile.TemporaryDirectory() as tmp_dir:
            w2v_tmp = os.path.join(tmp_dir, "glove_converted.w2v.txt")
            glove2word2vec(txt_path, w2v_tmp)
            model = KeyedVectors.load_word2vec_format(w2v_tmp, binary=False)

        print("GloVe loaded successfully.")
        print(f"Vocabulary: {len(model):,} words")
        print(f"Vector dimension: {model.vector_size}")

        # Save to cache
        if use_cached:
            try:
                print(f"Saving cached model to {cached_path} ...")
                model.save(cached_path)
                print("Cached model saved (next load will be instant).")
            except Exception as e:
                print(f"Could not save cache: {e}")

        return model

    except Exception as e:
        print(f"Failed to load GloVe model: {e}")
        return None


def nearest_neighbors(
    word: str,
    model: KeyedVectors,
    topn: int = 5,
) -> List[Tuple[str, float]]:
    """
    Find nearest neighbors of a word in the embedding space.
    Returns list of (neighbor, similarity) tuples.
    Returns empty list if word not found.
    """
    if model is None:
        print("Model is None. Load a model first.")
        return []

    if word not in model.key_to_index:
        print(f"Word '{word}' not in vocabulary.")
        # Show some random words from vocabulary as hint
        sample = list(model.key_to_index.keys())[:10]
        print(f"Sample vocabulary: {', '.join(sample)}")
        return []

    try:
        results = model.most_similar(positive=[word], topn=topn)

        # Pretty print
        print(f"\nNEAREST NEIGHBORS: '{word}'")
        print("─" * 60)
        for i, (neighbor, sim) in enumerate(results, 1):
            bar = "=" * int(sim * 20)  # visual similarity bar (max 20 chars)
            print(f"{i:2d}. {neighbor:20s} | {sim:.4f} | {bar}")
        print("─" * 60)

        return results

    except Exception as e:
        print(f"Error during nearest neighbor search: {e}")
        return []


def model_info(model: KeyedVectors, name: str = "Model") -> None:
    """Display basic information about the loaded model."""
    if model is None:
        print(f"{name} is not loaded.")
        return
    print(f"\n{name} INFO")
    print("─" * 60)
    print(f"Vocabulary size: {len(model):,} words")
    print(f"Vector dimension: {model.vector_size}")
    print(f"Vector dtype: {model.vectors.dtype}")
    # Estimate memory usage (approximate)
    mem_bytes = model.vectors.nbytes
    print(
        f"Vector memory: {mem_bytes / 1024**3:.2f} GB "
        "(total RAM usage may be higher)"
    )
    print("─" * 60)
