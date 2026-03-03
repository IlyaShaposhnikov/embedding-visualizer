"""
Load and interact with pre-trained embedding models (Word2Vec, GloVe).
Features:
  - Fast loading via cached binary format (.model)
  - Nearest neighbors search
  - Word analogy solving (king - man + woman = queen)
  - Model metadata inspection
"""

import os
import tempfile
import logging
from typing import Optional

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

logger = logging.getLogger(__name__)

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
        logger.info(f"Loading cached {model_name} model: {cached_path}")
        model = KeyedVectors.load(cached_path)
        # Validate integrity: non-empty vocabulary and valid dimensionality
        if len(model) == 0 or model.vector_size == 0:
            raise ValueError("Cached model is empty/corrupted")
        logger.info(
            f"Loaded from cache (vocab size: {len(model):,}, "
            f"dim: {model.vector_size})"
        )
        return model
    except (EOFError, ValueError, KeyError, OSError) as e:
        logger.warning(
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

    # Check if the original binary exists and is non-empty
    if not os.path.exists(bin_path) or os.path.getsize(bin_path) == 0:
        print(f"Word2Vec binary not found: {bin_path}")
        print(
            "Please run download_word2vec_model() first "
            "or place the file manually."
        )
        return None

    cached_path = _get_cached_model_path(bin_path)

    # Try cached model (fast path)
    if use_cached and not force_reload and os.path.exists(cached_path):
        model = _load_cached_model(cached_path, "Word2Vec")
        if model is not None:
            return model

    # Load from original .bin (slow, memory heavy)
    logger.info(
        "Loading Word2Vec from binary "
        "(this may take several minutes and ~4GB RAM)..."
    )
    try:
        # Binary format is specific to GoogleNews;
        # load_word2vec_format handles it
        model = KeyedVectors.load_word2vec_format(bin_path, binary=True)
        logger.info("Word2Vec loaded successfully.")
        logger.info(f"Vocabulary: {len(model):,} words")
        logger.info(f"Vector dimension: {model.vector_size}")

        # Save to cache for future fast loading
        if use_cached:
            try:
                logger.info(f"Saving cached model to {cached_path} ...")
                model.save(cached_path)
                logger.info("Cached model saved (next load will be instant).")
            except Exception as e:
                logger.warning(f"Could not save cache: {e}")

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

    # Check if the original text file exists and is non-empty
    if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
        print(f"GloVe file not found: {txt_path}")
        print(
            "Please run download_glove_model() first "
            "or place the file manually."
        )
        return None

    cached_path = _get_cached_model_path(txt_path)

    # Try cached model (fast path)
    if use_cached and not force_reload and os.path.exists(cached_path):
        model = _load_cached_model(cached_path, "GloVe")
        if model is not None:
            return model

    # Convert GloVe .txt → Word2Vec .txt (temporary) and load
    logger.info(f"Converting and loading GloVe from {txt_path} ...")
    try:
        # GloVe files are in word2vec format but without the header line.
        # Gensim's glove2word2vec adds the header and saves as a new file.
        # Use a temporary directory to avoid leaving intermediate files.
        with tempfile.TemporaryDirectory() as tmp_dir:
            w2v_tmp = os.path.join(tmp_dir, "glove_converted.w2v.txt")
            glove2word2vec(txt_path, w2v_tmp)
            model = KeyedVectors.load_word2vec_format(w2v_tmp, binary=False)

        logger.info("GloVe loaded successfully.")
        logger.info(f"Vocabulary: {len(model):,} words")
        logger.info(f"Vector dimension: {model.vector_size}")

        # Save to cache
        if use_cached:
            try:
                logger.info(f"Saving cached model to {cached_path} ...")
                model.save(cached_path)
                logger.info("Cached model saved (next load will be instant).")
            except Exception as e:
                logger.warning(f"Could not save cache: {e}")

        return model

    except Exception as e:
        print(f"Failed to load GloVe model: {e}")
        return None


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
    # Gensim stores vectors in a numpy array.
    mem_bytes = model.vectors.nbytes
    print(
        f"Vector memory: {mem_bytes / 1024**3:.2f} GB "
        "(total RAM usage may be higher)"
    )
    print("─" * 60)
