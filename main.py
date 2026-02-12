"""
Entry point for embedding-playground.
Demonstrates loading Word2Vec / GloVe and performing nearest neighbor search.
"""

from src.download import download_word2vec_model, download_glove_model
from src.models import (
    load_word2vec_model, load_glove_model, nearest_neighbors, model_info
)


def main():
    print("=" * 60)
    print("Embedding Playground — Model Loading & Nearest Neighbors")
    print("=" * 60)

    # Word2Vec
    print("\n[1] Word2Vec")
    print("-" * 60)

    # Ensure the binary file is available (download if missing)
    w2v_path = download_word2vec_model()
    if not w2v_path:
        print("Word2Vec binary not available. Skipping Word2Vec demo.")
        w2v_model = None
    else:
        w2v_model = load_word2vec_model(w2v_path, use_cached=True)
        if w2v_model:
            model_info(w2v_model, "Word2Vec (GoogleNews)")
            # Test nearest neighbors
            for word in ["king", "france", "computer"]:
                nearest_neighbors(word, w2v_model, topn=5)

    # GloVe
    print("\n[2] GloVe (6B.100d)")
    print("-" * 60)

    glove_path = download_glove_model(version="6B.100d")
    if not glove_path:
        print("GloVe file not available. Skipping GloVe demo.")
        glove_model = None
    else:
        glove_model = load_glove_model(glove_path, use_cached=True)
        if glove_model:
            model_info(glove_model, "GloVe (6B.100d)")
            for word in ["king", "france", "computer"]:
                nearest_neighbors(word, glove_model, topn=5)

    print("\n" + "=" * 60)
    if w2v_model is None and glove_model is None:
        print("Demo incomplete: both models failed to load.")
    else:
        print("Demo completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")
